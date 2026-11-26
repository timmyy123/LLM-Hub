package com.llmhub.llmhub.data

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Random
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Stable Audio 3 (Small Music, Small SFX, Medium) on the CPU through LiteRT, following Stability AI's
 * official runtime in stable-audio-3/optimized/tflite (tflite_pipeline.py, rung_dit.py, rung_decoder.py).
 *
 * The files are "rung" models: one .tflite holds several fixed-length subgraphs (signatures s<N>) that share
 * one copy of the weights. A render of L latents runs on the smallest rung >= L with the extra keys masked
 * out of attention, so the result is exact and RAM stays flat whatever the length. The XNNPACK weight cache
 * is what lets the rungs share packed weights, so every model here is compiled with one.
 *
 *   prompt -> SentencePiece -> T5Gemma encoder -> DiT (8 pingpong steps, gcond per step) -> SAME-S decoder
 */
internal object StableAudio3 {
    private const val TAG = "StableAudio3"

    const val SAMPLE_RATE = 44_100
    private const val CHANNELS = 2
    private const val SAMPLES_PER_LATENT = 4_096
    private const val LATENT_CH = 256
    private const val COND_TOKENS = 256
    private const val COND_DIM = 768
    private const val MEM_TOKENS = 64
    private const val STEPS = 8
    private const val DECODER_TRIM = 16 // SAME-S tiling overlap per side (its receptive field)

    /**
     * XNNPACK threads. On the Tab S9+ (Snapdragon 8 Gen 2) 6 threads were no faster than 4 for a 30 s Small Music
     * render (11.9 s vs 11.6 s); more threads land on the little cores.
     */
    private val threads = min(4, Runtime.getRuntime().availableProcessors())

    /** DiT rung ladders of the files pinned in ModelData (signature names are "s" + size). */
    private val MEDIUM_RUNGS = intArrayOf(8, 48, 96, 192, 416, 704, 1056, 1496, 2040, 2824, 3536, 4096)
    private val SMALL_RUNGS = intArrayOf(8, 48, 96, 192, 416, 704, 1056, 1292)
    private val DECODER_RUNGS = intArrayOf(2, 4, 8, 12, 16, 32, 64, 128, 256)

    const val DIT_FILE = "dit_w8a8.tflite"
    const val DECODER_FILE = "dec_w8a8.tflite"
    const val TEXT_ENCODER_FILE = "encoder_fp16.tflite"
    const val TOKENIZER_FILE = "tokenizer.model"

    fun isModel(modelName: String): Boolean = modelName.startsWith("Stable Audio 3", ignoreCase = true)

    private fun isMedium(modelName: String) = modelName.contains("Medium", ignoreCase = true)

    /** Medium's ladder tops at 4096 latents (380 s); the small DiTs at 1292 (their 2 minute maximum). */
    fun maxSeconds(modelName: String): Float = if (isMedium(modelName)) 380f else 120f

    private fun latentsFor(seconds: Float): Int = max(1, ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT.toDouble()).toInt())

    /**
     * Renders [seconds] of stereo audio and writes it straight to a 16-bit WAV, one decoder window at a time,
     * so a 380 s render never holds the whole waveform in the Java heap. The decoder has the output limiter
     * baked in (peak <= 0.977), so no normalization is applied.
     */
    suspend fun generate(
        modelName: String,
        modelDir: File,
        tokenIds: IntArray,
        seconds: Float,
        seed: Long,
        output: File,
        onProgress: (Float) -> Unit
    ): File {
        val ditFile = File(modelDir, DIT_FILE)
        val decoderFile = File(modelDir, DECODER_FILE)
        val encoderFile = File(modelDir, TEXT_ENCODER_FILE)
        val latents = latentsFor(seconds)
        val rungs = if (isMedium(modelName)) MEDIUM_RUNGS else SMALL_RUNGS
        require(latents <= rungs.last()) { "$modelName renders at most ${maxSeconds(modelName).toInt()} s" }

        // Stage 1: T5Gemma text encoder, released before the DiT compiles.
        onProgress(0.01f)
        val ids = IntArray(COND_TOKENS)
        val mask = IntArray(COND_TOKENS)
        repeat(min(tokenIds.size, COND_TOKENS)) { ids[it] = tokenIds[it]; mask[it] = 1 }
        val t5Hidden = compile(encoderFile, "T5Gemma", weightCache = false).useModel { model ->
            model.buffers { inputs, outputs ->
                inputs[0].writeInt(ids)
                inputs[1].writeInt(mask)
                model.run(inputs, outputs)
                outputs[0].readFloat()
            }
        }
        require(t5Hidden.size == COND_TOKENS * COND_DIM) { "T5Gemma returned ${t5Hidden.size} values" }
        val t5Mask = FloatArray(COND_TOKENS) { mask[it].toFloat() }
        currentCoroutineContext().ensureActive()
        onProgress(0.05f)

        // Stage 2: DiT on the smallest rung >= latents, 8 rectified-flow pingpong steps.
        val rung = rungs.first { it >= latents }
        val latent = sampleLatents(ditFile, rung, latents, t5Hidden, t5Mask, seconds, seed, onProgress)
        onProgress(0.86f)

        // Stage 3: SAME-S decoder, streamed into the WAV.
        val frames = min(latents * SAMPLES_PER_LATENT, max(1, (seconds * SAMPLE_RATE).roundToInt()))
        decodeToWav(decoderFile, latent, latents, frames, output) { done -> onProgress(0.86f + 0.14f * done) }
        onProgress(1f)
        return output
    }

    private suspend fun sampleLatents(
        ditFile: File,
        rung: Int,
        length: Int,
        t5Hidden: FloatArray,
        t5Mask: FloatArray,
        seconds: Float,
        seed: Long,
        onProgress: (Float) -> Unit
    ): FloatArray {
        val signature = "s$rung"
        // Rectified-flow pingpong schedule: logSNR shift of linspace(1, 0, 9), first step exactly 1.
        val sigmas = FloatArray(STEPS + 1) { index ->
            val t = 1f - index.toFloat() / STEPS
            when {
                t >= 1f -> 1f
                t <= 0f -> 0f
                else -> (1.0 / (1.0 + exp(2.0 - t * 8.2))).toFloat()
            }
        }
        val size = LATENT_CH * length
        val random = Random(seed)
        val x = FloatArray(size) { random.nextGaussian().toFloat() }
        val padded = FloatArray(LATENT_CH * rung)
        val attentionMask = FloatArray(MEM_TOKENS + rung).also { mask ->
            for (i in MEM_TOKENS + length until mask.size) mask[i] = -1e9f
        }
        val secondsArray = floatArrayOf(seconds)
        val timeArray = FloatArray(1)

        compile(ditFile, "DiT $signature", weightCache = true).useModel { model ->
            val gcIn = listOf(model.createInputBuffer("args_0", "gcond"), model.createInputBuffer("args_1", "gcond"))
            val gcOut = model.createOutputBuffer("output_0", "gcond")
            val names = (0..6).map { "args_$it" }
            val inputs = names.associateWith { model.createInputBuffer(it, signature) }
            val outputs = mapOf("output_0" to model.createOutputBuffer("output_0", signature))
            try {
                // Constant across steps: T5 hidden + mask, seconds, local_add_cond (zeros) and the pad-key mask.
                inputs.getValue("args_2").writeFloat(t5Hidden)
                inputs.getValue("args_3").writeFloat(t5Mask)
                inputs.getValue("args_4").writeFloat(secondsArray)
                inputs.getValue("args_5").writeFloat(FloatArray(257 * rung))
                inputs.getValue("args_6").writeFloat(attentionMask)
                gcIn[0].writeFloat(secondsArray)
                onProgress(0.1f)

                for (step in 0 until STEPS) {
                    currentCoroutineContext().ensureActive()
                    val tc = sigmas[step]
                    val tn = sigmas[step + 1]
                    timeArray[0] = tc
                    gcIn[1].writeFloat(timeArray)
                    model.run(gcIn, listOf(gcOut), "gcond")
                    inputs.getValue("args_1").writeFloat(gcOut.readFloat())

                    for (c in 0 until LATENT_CH) System.arraycopy(x, c * length, padded, c * rung, length)
                    inputs.getValue("args_0").writeFloat(padded)
                    val started = System.nanoTime()
                    model.run(inputs, outputs, signature)
                    val velocity = outputs.getValue("output_0").readFloat()
                    Log.i(TAG, "DiT $signature step ${step + 1}/$STEPS: ${(System.nanoTime() - started) / 1_000_000} ms")

                    val renoise = step < STEPS - 1 && tn > 0f
                    for (c in 0 until LATENT_CH) {
                        val src = c * length
                        val dst = c * rung
                        for (i in 0 until length) {
                            val denoised = x[src + i] - tc * velocity[dst + i]
                            x[src + i] = if (renoise) (1f - tn) * denoised else denoised
                        }
                    }
                    if (renoise) for (i in 0 until size) x[i] += tn * random.nextGaussian().toFloat()
                    onProgress(0.1f + 0.75f * (step + 1) / STEPS)
                }
            } finally {
                (gcIn + gcOut + inputs.values + outputs.values).forEach { runCatching { it.close() } }
            }
        }
        return x
    }

    /**
     * RungDecoder.decode: SAME-S needs an even length (pad one latent), then tiles on the rung with the least
     * total work among the lossless choices (the exact length, or a tileable rung below it), with a 16-latent
     * overlap per side so window seams are bit-exact.
     */
    private suspend fun decodeToWav(
        decoderFile: File,
        latent: FloatArray,
        length: Int,
        frames: Int,
        output: File,
        onProgress: (Float) -> Unit
    ) {
        val decodeLength = if (length % 2 != 0) length + 1 else length
        val chunk = bestDecoderRung(decodeLength)
        val starts = if (decodeLength <= chunk) listOf(0) else {
            val step = chunk - 2 * DECODER_TRIM
            val list = (0 until decodeLength - chunk step step).toMutableList()
            if (list.lastOrNull() != decodeLength - chunk) list += decodeLength - chunk
            list
        }
        val signature = "s$chunk"
        val window = FloatArray(LATENT_CH * chunk)

        FileOutputStream(output).use { fileStream ->
            val stream = fileStream.buffered(1 shl 20)
            stream.write(wavHeader(frames))
            var written = 0
            compile(decoderFile, "SAME-S decoder $signature", weightCache = true).useModel { model ->
                val inputs = model.createInputBuffers(signature)
                val outputs = model.createOutputBuffers(signature)
                try {
                    starts.forEachIndexed { k, start ->
                        currentCoroutineContext().ensureActive()
                        // Window [start, start+chunk), repeating the last latent past the end (pad up, trim).
                        for (c in 0 until LATENT_CH) {
                            for (i in 0 until chunk) {
                                val t = min(start + i, length - 1)
                                window[c * chunk + i] = latent[c * length + t]
                            }
                        }
                        inputs[0].writeFloat(window)
                        model.run(inputs, outputs, signature)
                        val audio = outputs[0].readFloat() // [1, 2, chunk * 4096], planar
                        val channelSamples = chunk * SAMPLES_PER_LATENT
                        val lo = if (k == 0) 0 else DECODER_TRIM
                        val hi = if (start + chunk >= decodeLength) chunk else chunk - DECODER_TRIM
                        // The last window can reach back over audio already written (rung_decoder overwrites
                        // it); every sample at least TRIM latents from a window edge is exact in both
                        // windows, so append only what follows the write position.
                        val from = max((start + lo) * SAMPLES_PER_LATENT, written)
                        val until = min((start + hi) * SAMPLES_PER_LATENT, frames)
                        if (until > from) {
                            writePcm(stream, audio, from - start * SAMPLES_PER_LATENT, channelSamples, until - from)
                            written = until
                        }
                        onProgress((k + 1f) / starts.size)
                    }
                } finally {
                    (inputs + outputs).forEach { runCatching { it.close() } }
                }
            }
            stream.flush()
            check(written == frames) { "Decoded $written of $frames frames" }
        }
    }

    private fun bestDecoderRung(length: Int): Int {
        fun tileable(r: Int) = r - 2 * DECODER_TRIM > 0
        fun windows(r: Int): Int {
            if (length <= r) return 1
            val step = r - 2 * DECODER_TRIM
            var n = (length - r + step - 1) / step
            if ((n - 1) * step != length - r) n += 1
            return n
        }
        var best = -1
        var bestCost = Long.MAX_VALUE
        var bestWindows = Int.MAX_VALUE
        for (r in DECODER_RUNGS) {
            if (r != length && !(r < length && tileable(r))) continue
            val n = windows(r)
            val cost = n.toLong() * r
            if (cost < bestCost || (cost == bestCost && n < bestWindows)) {
                best = r; bestCost = cost; bestWindows = n
            }
        }
        return if (best > 0) best else DECODER_RUNGS.firstOrNull { it >= length } ?: DECODER_RUNGS.last()
    }

    private fun writePcm(stream: java.io.OutputStream, planar: FloatArray, offset: Int, channelSamples: Int, count: Int) {
        val buffer = ByteBuffer.allocate(4096 * CHANNELS * 2).order(ByteOrder.LITTLE_ENDIAN)
        var i = 0
        while (i < count) {
            buffer.clear()
            val n = min(4096, count - i)
            for (j in 0 until n) {
                val idx = offset + i + j
                buffer.putShort((planar[idx].coerceIn(-1f, 1f) * 32767f).roundToInt().toShort())
                buffer.putShort((planar[channelSamples + idx].coerceIn(-1f, 1f) * 32767f).roundToInt().toShort())
            }
            stream.write(buffer.array(), 0, n * CHANNELS * 2)
            i += n
        }
    }

    private fun wavHeader(frames: Int): ByteArray {
        val dataSize = frames * CHANNELS * 2
        return ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII)); putInt(36 + dataSize)
            put("WAVE".toByteArray(Charsets.US_ASCII)); put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16); putShort(1); putShort(CHANNELS.toShort()); putInt(SAMPLE_RATE)
            putInt(SAMPLE_RATE * CHANNELS * 2); putShort((CHANNELS * 2).toShort()); putShort(16)
            put("data".toByteArray(Charsets.US_ASCII)); putInt(dataSize)
        }.array()
    }

    /**
     * CPU (XNNPACK). For the rung files the weight cache sits next to the model, so the rungs share one packed
     * copy of the weights (file-backed, not anonymous memory) and later loads skip the packing. A cache left
     * half-written by a killed process is deleted and rebuilt once.
     */
    private fun compile(file: File, stage: String, weightCache: Boolean): CompiledModel {
        val cache = File(file.absolutePath + ".xnnwc")
        fun create(): CompiledModel {
            val options = CompiledModel.Options(Accelerator.CPU)
            options.cpuOptions = CompiledModel.CpuOptions(threads, null, if (weightCache) cache.absolutePath else null)
            val started = System.nanoTime()
            return CompiledModel.create(file.absolutePath, options).also {
                Log.i(TAG, "$stage compiled in ${(System.nanoTime() - started) / 1_000_000} ms with $threads threads (${file.name})")
            }
        }
        return try {
            create()
        } catch (t: Throwable) {
            if (!weightCache || !cache.exists()) throw t
            Log.w(TAG, "$stage failed with the existing weight cache; rebuilding it", t)
            cache.delete()
            create()
        }
    }

    private inline fun <T> CompiledModel.useModel(block: (CompiledModel) -> T): T =
        try { block(this) } finally { runCatching { close() } }

    private inline fun <T> CompiledModel.buffers(block: (List<TensorBuffer>, List<TensorBuffer>) -> T): T {
        val inputs = createInputBuffers()
        val outputs = createOutputBuffers()
        return try { block(inputs, outputs) } finally { (inputs + outputs).forEach { runCatching { it.close() } } }
    }
}
