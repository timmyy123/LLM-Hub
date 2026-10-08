package com.llmhub.llmhub.ui.components

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.llmhub.llmhub.data.SupertonicModel
import org.json.JSONArray
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.text.Normalizer
import java.util.Random
import java.util.concurrent.CancellationException
import kotlin.math.ceil

/** CPU inference adapted from Supertone's MIT Java/Python examples. See assets/supertonic-NOTICE.txt. */
class SupertonicEngine(private val directory: File) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val sessions = mutableListOf<OrtSession>()
    private val config = JSONObject(File(directory, "tts.json").readText())
    val sampleRate = config.getJSONObject("ae").getInt("sample_rate")
    private val chunkSize = config.getJSONObject("ae").getInt("base_chunk_size") *
        config.getJSONObject("ttl").getInt("chunk_compress_factor")
    private val latentDim = config.getJSONObject("ttl").getInt("latent_dim") *
        config.getJSONObject("ttl").getInt("chunk_compress_factor")
    private val indexer = JSONArray(File(directory, "unicode_indexer.json").readText()).let { a ->
        LongArray(a.length()) { a.getLong(it) }
    }
    private var closed = false

    init {
        try {
            OrtSession.SessionOptions().use { opts ->
                opts.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
                opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                for (name in listOf("duration_predictor", "text_encoder", "vector_estimator", "vocoder")) {
                    sessions.add(env.createSession(File(directory, "$name.onnx").absolutePath, opts))
                }
            }
        } catch (e: Exception) {
            sessions.forEach { it.close() }
            throw e
        }
    }

    private fun tensor(values: FloatArray, vararg shape: Long) =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(values), shape)
    private fun floats(tensor: OnnxTensor): FloatArray = tensor.floatBuffer.let { buffer ->
        FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    /** One bounded chunk; locking prevents session disposal while a native inference is in flight. */
    @Synchronized
    fun synthesize(text: String, language: String, voice: SupertonicModel.Voice, speed: Float,
                   cancelled: () -> Boolean): FloatArray {
        check(!closed)
        fun ensureActive() { if (cancelled()) throw CancellationException() }
        ensureActive()
        require(language in SupertonicModel.languages)
        val normalized = preprocess(text, language)
        val points = normalized.codePoints().toArray()
        val ids = LongArray(points.size) { indexer.getOrElse(points[it]) { -1L } }
        require(ids.all { it >= 0 }) { "Unsupported text characters" }
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong())).use { idsTensor ->
            tensor(FloatArray(ids.size) { 1f }, 1, 1, ids.size.toLong()).use { textMask ->
                tensor(voice.dp, 1, 8, 16).use { dp ->
                    tensor(voice.ttl, 1, 50, 256).use { ttl ->
                        val seconds = sessions[0].run(mapOf("text_ids" to idsTensor, "text_mask" to textMask, "style_dp" to dp)).use {
                            floats(it[0] as OnnxTensor)[0] / speed.coerceIn(0.5f, 2f)
                        }
                        require(seconds.isFinite() && seconds > 0 && seconds <= 60) { "Invalid speech duration" }
                        ensureActive()
                        sessions[1].run(mapOf("text_ids" to idsTensor, "text_mask" to textMask, "style_ttl" to ttl)).use { encoded ->
                            val textEmbedding = encoded[0] as OnnxTensor
                            val samples = (seconds * sampleRate).toInt()
                            val length = ceil(seconds * sampleRate / chunkSize).toInt().coerceAtLeast(1)
                            val maskLength = (samples + chunkSize - 1) / chunkSize
                            val random = Random()
                            var latent = FloatArray(latentDim * length) { i ->
                                if (i % length < maskLength) random.nextGaussian().toFloat() else 0f
                            }
                            val steps = 8
                            tensor(FloatArray(length) { if (it < maskLength) 1f else 0f }, 1, 1, length.toLong()).use { latentMask ->
                                tensor(floatArrayOf(steps.toFloat()), 1).use { totalStep ->
                                    repeat(steps) { step ->
                                        ensureActive()
                                        tensor(floatArrayOf(step.toFloat()), 1).use { currentStep ->
                                            tensor(latent, 1, latentDim.toLong(), length.toLong()).use { noise ->
                                                sessions[2].run(mapOf(
                                                    "noisy_latent" to noise, "text_emb" to textEmbedding,
                                                    "style_ttl" to ttl, "latent_mask" to latentMask,
                                                    "text_mask" to textMask, "current_step" to currentStep,
                                                    "total_step" to totalStep
                                                )).use { latent = floats(it[0] as OnnxTensor) }
                                            }
                                        }
                                    }
                                }
                            }
                            ensureActive()
                            tensor(latent, 1, latentDim.toLong(), length.toLong()).use { finalLatent ->
                                return sessions[3].run(mapOf("latent" to finalLatent)).use {
                                    floats(it[0] as OnnxTensor).let { wav -> wav.copyOf(minOf(samples, wav.size)) }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        sessions.forEach { it.close() }
        // OrtEnvironment is shared by all app ONNX engines and must stay alive.
    }

    companion object {
        /** Bound even unbroken CJK text and long tokens without splitting surrogate pairs. */
        internal fun chunks(text: String, maxCodePoints: Int): List<String> {
            require(maxCodePoints > 0)
            val result = mutableListOf<String>()
            var start = 0
            while (start < text.length) {
                val count = text.codePointCount(start, text.length).coerceAtMost(maxCodePoints)
                var end = text.offsetByCodePoints(start, count)
                if (end < text.length) {
                    val space = text.lastIndexOf(' ', end - 1)
                    if (space > start && space - start >= (end - start) / 2) end = space
                }
                text.substring(start, end).trim().takeIf { it.isNotEmpty() }?.let(result::add)
                start = end
            }
            return result
        }

        internal fun preprocess(input: String, language: String): String {
            var text = Normalizer.normalize(input, Normalizer.Form.NFKD)
            text = buildString {
                text.codePoints().forEach { cp ->
                    if (cp !in 0x1F300..0x1FAFF && cp !in 0x2600..0x27BF && cp !in 0x1F1E6..0x1F1FF) appendCodePoint(cp)
                }
            }
            for ((from, to) in mapOf("–" to "-", "‑" to "-", "—" to "-", "_" to " ",
                "“" to "\"", "”" to "\"", "‘" to "'", "’" to "'", "´" to "'", "`" to "'",
                "[" to " ", "]" to " ", "|" to " ", "/" to " ", "#" to " ", "→" to " ", "←" to " ",
                "@" to " at ", "e.g.," to "for example, ", "i.e.," to "that is, ")) text = text.replace(from, to)
            text = text.replace(Regex("[♥☆♡©\\\\]"), "")
                .replace(Regex("\\s+([,.!?;:'])"), "$1")
                .replace(Regex("([\"'])\\1+"), "$1")
                .replace(Regex("\\s+"), " ").trim()
            if (text.isEmpty() || text.last() !in ".!?;:,'\"“”‘’)\u005d}…。」』】〉》›»") text += "."
            return "<$language>$text</$language>"
        }
    }
}
