package com.llmhub.llmhub.data

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.Normalizer
import java.util.Random
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Local SoundGen implementation ported from the inference pipeline shipped by Box.
 *
 * These models are LiteRT compiled graphs, not ordinary single-input TFLite models:
 * text conditioning, iterative flow sampling, and audio decoding are separate graphs.
 */
object MusicGeneratorBackend {
    private const val TAG = "MusicGeneratorBackend"
    private const val SAMPLE_RATE = 44_100
    private const val CHANNELS = 2
    private const val QUICK_CHANNEL_SAMPLES = 524_288
    private const val HD_SAMPLES_PER_LATENT_FRAME = 4_096
    private const val HD_SHORT_LATENT_FRAMES = 256
    private const val HD_LONG_LATENT_FRAMES = 2_048
    private val modelMutex = Mutex()
    @Volatile private var loadedBundle: LoadedBundle? = null

    private sealed class LoadedBundle(
        val modelName: String,
        val textModel: CompiledModel,
        val coreModel: CompiledModel,
        val decodeModel: CompiledModel
    ) : AutoCloseable {
        override fun close() {
            runCatching { decodeModel.close() }
            runCatching { coreModel.close() }
            runCatching { textModel.close() }
        }
    }

    private class QuickBundle(
        modelName: String,
        textModel: CompiledModel,
        coreModel: CompiledModel,
        decodeModel: CompiledModel,
        val tokenizer: SentencePieceTokenizer
    ) : LoadedBundle(modelName, textModel, coreModel, decodeModel)

    private class HdBundle(
        modelName: String,
        textModel: CompiledModel,
        coreModel: CompiledModel,
        decodeModel: CompiledModel,
        val tokenizer: BpeTokenizer,
        val longModel: Boolean
    ) : LoadedBundle(modelName, textModel, coreModel, decodeModel)

    fun isModelLoaded(modelName: String): Boolean = loadedBundle?.modelName == modelName

    suspend fun loadModel(context: Context, modelName: String): Boolean = withContext(Dispatchers.IO) {
        modelMutex.withLock {
            try {
                loadModelLocked(context, modelName)
                true
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to load local SoundGen model", t)
                false
            }
        }
    }

    suspend fun unloadModel() = withContext(Dispatchers.IO) {
        modelMutex.withLock {
            loadedBundle?.close()
            loadedBundle = null
            Log.i(TAG, "Local SoundGen model unloaded")
        }
    }

    /** Duration capacities imposed by each fixed-shape decoder export. */
    fun durationRange(modelName: String): ClosedFloatingPointRange<Float> {
        val sampleCapacity = when {
            modelName.contains("HD Long", ignoreCase = true) ->
                HD_LONG_LATENT_FRAMES * HD_SAMPLES_PER_LATENT_FRAME
            modelName.contains("HD", ignoreCase = true) ->
                HD_SHORT_LATENT_FRAMES * HD_SAMPLES_PER_LATENT_FRAME
            else -> QUICK_CHANNEL_SAMPLES
        }
        return 1f..(sampleCapacity.toFloat() / SAMPLE_RATE).roundToInt().toFloat()
    }

    suspend fun generateMusic(
        context: Context,
        modelName: String,
        prompt: String,
        durationSeconds: Double,
        onProgress: (Float) -> Unit
    ): File? = withContext(Dispatchers.IO) {
        try {
            require(prompt.isNotBlank()) { "Describe the music to generate" }
            modelMutex.withLock {
                val bundle = loadModelLocked(context, modelName)
                val seed = System.nanoTime()
                when (bundle) {
                    is HdBundle -> generateHd(
                        context = context,
                        bundle = bundle,
                        prompt = prompt,
                        durationSeconds = durationSeconds.toFloat(),
                        seed = seed,
                        onProgress = onProgress
                    )
                    is QuickBundle -> generateQuick(
                        context,
                        bundle,
                        prompt,
                        durationSeconds.toFloat(),
                        seed,
                        onProgress
                    )
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Local SoundGen generation failed", t)
            null
        }
    }

    private fun generateQuick(
        context: Context,
        bundle: QuickBundle,
        prompt: String,
        durationSeconds: Float,
        seed: Long,
        onProgress: (Float) -> Unit
    ): File {
        val tokenizer = bundle.tokenizer

        // This exported Quick bundle cannot be compiled by Samsung's current
        // LiteRT OpenCL delegate: the conditioner uses INT64 and DiT contains GPU
        // kernels that fail initialization after partitioning. Use one explicit
        // CPU execution plan for every graph; there is no retry or fallback path.
        onProgress(0.01f)
        bundle.textModel.let { textModel ->
            onProgress(0.03f)
            bundle.coreModel.let { coreModel ->
                onProgress(0.06f)
                bundle.decodeModel.let { decodeModel ->
                    onProgress(0.09f)
                    val tokenIds = tokenizer.encode(prompt)
                    val ids = LongArray(128)
                    val mask = LongArray(128)
                    repeat(min(tokenIds.size, 128)) { index ->
                        ids[index] = tokenIds[index].toLong()
                        mask[index] = 1L
                    }

                    val (conditioning, conditioningMask) = textModel.buffers { inputs, outputs ->
                        inputs[0].writeLong(ids)
                        inputs[1].writeLong(mask)
                        inputs[2].writeFloat(floatArrayOf(durationSeconds))
                        Log.i(TAG, "Running Quick conditioner")
                        textModel.run(inputs, outputs)
                        Log.i(TAG, "Quick conditioner complete")
                        outputs[0].readFloat() to outputs[2].readFloat()
                    }
                    onProgress(0.12f)

                    val schedule = FloatArray(9) { index ->
                        when (index) {
                            0 -> 1f
                            8 -> 0f
                            else -> 1f / (exp((-6f + index).toDouble()).toFloat() + 1f)
                        }
                    }
                    var latent = gaussian(seed, 16_384)
                    coreModel.buffers { inputs, outputs ->
                        inputs[0].writeFloat(conditioning)
                        inputs[1].writeFloat(conditioningMask)
                        repeat(8) { step ->
                            val time = schedule[step]
                            val nextTime = schedule[step + 1]
                            inputs[2].writeFloat(latent)
                            inputs[3].writeFloat(floatArrayOf(time))
                            Log.i(TAG, "Running Quick DiT step ${step + 1}/8")
                            coreModel.run(inputs, outputs)
                            val velocity = outputs[0].readFloat()
                            val noise = gaussian(seed + step + 4564L, latent.size)
                            latent = FloatArray(latent.size) { i ->
                                noise[i] * nextTime + (1f - nextTime) * (latent[i] - velocity[i] * time)
                            }
                            onProgress(0.12f + ((step + 1) * 0.78f / 8f))
                        }
                    }

                    val audio = decodeModel.buffers { inputs, outputs ->
                        inputs[0].writeFloat(latent)
                        Log.i(TAG, "Running Quick decoder")
                        decodeModel.run(inputs, outputs)
                        Log.i(TAG, "Quick decoder complete")
                        outputs[0].readFloat()
                    }
                    onProgress(0.95f)
                    val frames = min(QUICK_CHANNEL_SAMPLES, max(1, (durationSeconds * SAMPLE_RATE).roundToInt()))
                    return writeStereoWav(context, "soundgen", audio, QUICK_CHANNEL_SAMPLES, frames, normalize = false)
                        .also { onProgress(1f) }
                }
            }
        }
    }

    private fun generateHd(
        context: Context,
        bundle: HdBundle,
        prompt: String,
        durationSeconds: Float,
        seed: Long,
        onProgress: (Float) -> Unit
    ): File {
        val longModel = bundle.longModel
        val tokenizer = bundle.tokenizer
        val frameCount = if (longModel) HD_LONG_LATENT_FRAMES else HD_SHORT_LATENT_FRAMES
        val latentSize = frameCount * 256
        val channelSamples = frameCount * HD_SAMPLES_PER_LATENT_FRAME

        bundle.textModel.let { textModel ->
            bundle.coreModel.let { coreModel ->
                bundle.decodeModel.let { decodeModel ->
                    val tokenIds = tokenizer.encode(prompt)
                    val ids = LongArray(256)
                    val mask = LongArray(256)
                    repeat(min(tokenIds.size, 256)) { index ->
                        ids[index] = tokenIds[index].toLong()
                        mask[index] = 1L
                    }

                    val conditioning = textModel.buffers { inputs, outputs ->
                        inputs[0].writeLong(ids)
                        inputs[1].writeLong(mask)
                        textModel.run(inputs, outputs)
                        require(outputs.size == 1) {
                            "SoundGen HD encoder returned ${outputs.size} outputs; expected 1"
                        }
                        outputs[0].readFloat()
                    }
                    // The pinned T5Gemma encoder exports only the hidden states.
                    // The DiT's [1, 256] conditioning mask is the encoder attention
                    // mask converted from INT64 to FLOAT32, not a second encoder output.
                    val conditioningMask = FloatArray(mask.size) { mask[it].toFloat() }
                    onProgress(0.05f)

                    val shift = exp(-((((frameCount - 256) * 0.65f) / 3840f) + 0.5f).toDouble()).toFloat()
                    val schedule = FloatArray(9) { index ->
                        val raw = 1f - index / 8f
                        when {
                            raw >= 1f -> 1f
                            raw <= 0f -> 0f
                            else -> 1f - shift / ((1f / (1f - raw) - 1f) + shift)
                        }
                    }
                    schedule[0] = 1f
                    var latent = gaussian(seed, latentSize)
                    coreModel.buffers { inputs, outputs ->
                        inputs[2].writeFloat(conditioning)
                        inputs[3].writeFloat(conditioningMask)
                        inputs[4].writeFloat(floatArrayOf(durationSeconds))
                        inputs[5].writeFloat(FloatArray(frameCount * 257))
                        repeat(8) { step ->
                            val time = schedule[step]
                            val nextTime = schedule[step + 1]
                            inputs[0].writeFloat(latent)
                            inputs[1].writeFloat(floatArrayOf(time))
                            coreModel.run(inputs, outputs)
                            val velocity = outputs[0].readFloat()
                            val noise = if (step < 7) gaussian(seed + step + 1L, latentSize) else null
                            latent = FloatArray(latentSize) { i ->
                                val predicted = latent[i] - velocity[i] * time
                                if (noise == null) predicted else noise[i] * nextTime + (1f - nextTime) * predicted
                            }
                            onProgress(0.05f + ((step + 1) * 0.8f / 8f))
                        }
                    }

                    val audio = decodeModel.buffers { inputs, outputs ->
                        inputs[0].writeFloat(latent)
                        decodeModel.run(inputs, outputs)
                        outputs[0].readFloat()
                    }
                    onProgress(0.97f)
                    val frames = min(channelSamples, max(1, (durationSeconds * SAMPLE_RATE).roundToInt()))
                    return writeStereoWav(context, if (longModel) "soundgenhd_long" else "soundgenhd", audio, channelSamples, frames, normalize = true)
                        .also { onProgress(1f) }
                }
            }
        }
    }

    private fun createCpuModel(file: File, stage: String = file.name): CompiledModel {
        Log.i(TAG, "Compiling $stage with CPU: ${file.name}")
        return CompiledModel.create(file.absolutePath, CompiledModel.Options(Accelerator.CPU))
            .also { Log.i(TAG, "$stage compilation complete") }
    }

    private fun loadModelLocked(context: Context, modelName: String): LoadedBundle {
        loadedBundle?.takeIf { it.modelName == modelName }?.let { return it }
        loadedBundle?.close()
        loadedBundle = null

        val modelDir = modelDirectory(context, modelName)
        var textModel: CompiledModel? = null
        var coreModel: CompiledModel? = null
        var decodeModel: CompiledModel? = null
        try {
            val bundle = if (modelName.contains("HD", ignoreCase = true)) {
                val longModel = modelName.contains("Long", ignoreCase = true)
                val coreOriginal = if (longModel) "dit_L2048_int8.tflite" else "dit_L256_int8.tflite"
                val decodeOriginal = if (longModel) "ae_dec_L2048_int8.tflite" else "ae_dec_L256_int8.tflite"
                val textFile = requireFile(modelDir, "t5gemma_enc_int8.tflite", "sghd_text.litert")
                val coreFile = requireFile(
                    modelDir,
                    coreOriginal,
                    coreOriginal.removeSuffix(".tflite") + ".litert",
                    "sghd_core.litert"
                )
                val decodeFile = requireFile(modelDir, decodeOriginal, "sghd_decode.litert")
                val tokenizer = BpeTokenizer.load(requireFile(modelDir, "tokenizer.model", "sghd_vocab.spm"))
                textModel = createCpuModel(textFile, "SoundGen HD conditioner")
                coreModel = createCpuModel(coreFile, "SoundGen HD DiT")
                decodeModel = createCpuModel(decodeFile, "SoundGen HD decoder")
                HdBundle(
                    modelName,
                    requireNotNull(textModel),
                    requireNotNull(coreModel),
                    requireNotNull(decodeModel),
                    tokenizer,
                    longModel
                )
            } else {
                val textFile = requireFile(modelDir, "conditioners_float32.tflite", "sg_text.litert")
                val coreFile = requireFile(modelDir, "dit_model.tflite", "dit_model.litert", "sg_core.litert")
                val decodeFile = requireFile(modelDir, "autoencoder_model.tflite", "sg_decode.litert")
                val tokenizer = SentencePieceTokenizer.load(requireFile(modelDir, "spiece.model", "sg_vocab.spm"))
                textModel = createCpuModel(textFile, "Quick conditioner")
                coreModel = createCpuModel(coreFile, "Quick DiT")
                decodeModel = createCpuModel(decodeFile, "Quick decoder")
                QuickBundle(
                    modelName,
                    requireNotNull(textModel),
                    requireNotNull(coreModel),
                    requireNotNull(decodeModel),
                    tokenizer
                )
            }
            loadedBundle = bundle
            Log.i(TAG, "Local SoundGen model loaded: $modelName")
            return bundle
        } catch (t: Throwable) {
            runCatching { decodeModel?.close() }
            runCatching { coreModel?.close() }
            runCatching { textModel?.close() }
            throw t
        }
    }

    private inline fun <T> CompiledModel.buffers(block: (List<TensorBuffer>, List<TensorBuffer>) -> T): T {
        val inputs = createInputBuffers()
        val outputs = createOutputBuffers()
        return try {
            block(inputs, outputs)
        } finally {
            inputs.forEach { runCatching { it.close() } }
            outputs.forEach { runCatching { it.close() } }
        }
    }

    private fun modelDirectory(context: Context, modelName: String): File {
        val models = File(context.filesDir, "models")
        val clean = modelName.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
        return File(models, clean).takeIf { it.isDirectory } ?: models
    }

    private fun requireFile(directory: File, vararg names: String): File {
        for (name in names) {
            val inDirectory = File(directory, name)
            if (inDirectory.isFile && inDirectory.length() > 0) return inDirectory
            val inRoot = File(directory.parentFile ?: directory, name)
            if (inRoot.isFile && inRoot.length() > 0) return inRoot
        }
        error("Missing SoundGen component (${names.joinToString()}) in ${directory.absolutePath}; redownload this model")
    }

    private fun gaussian(seed: Long, size: Int): FloatArray {
        val random = Random(seed)
        return FloatArray(size) { random.nextGaussian().toFloat() }
    }

    private fun writeStereoWav(
        context: Context,
        prefix: String,
        planarAudio: FloatArray,
        rightOffset: Int,
        frames: Int,
        normalize: Boolean
    ): File {
        require(planarAudio.size >= rightOffset + frames) {
            "Decoder returned ${planarAudio.size} samples, expected at least ${rightOffset + frames}"
        }
        val scale = if (normalize) {
            val peak = planarAudio.maxOf { kotlin.math.abs(it) }
            if (peak > 1e-6f) 1f / peak else 1f
        } else 1f
        val pcm = ByteBuffer.allocate(frames * CHANNELS * 2).order(ByteOrder.LITTLE_ENDIAN)
        repeat(frames) { index ->
            pcm.putShort((planarAudio[index] * scale).coerceIn(-1f, 1f).times(32767f).roundToInt().toShort())
            pcm.putShort((planarAudio[rightOffset + index] * scale).coerceIn(-1f, 1f).times(32767f).roundToInt().toShort())
        }
        val outputDir = File(context.cacheDir, prefix).apply { mkdirs() }
        val output = File(outputDir, "${prefix}_${System.currentTimeMillis()}.wav")
        FileOutputStream(output).use { stream ->
            stream.write(createWavHeader(pcm.array().size, SAMPLE_RATE, CHANNELS))
            stream.write(pcm.array())
        }
        return output
    }

    private fun createWavHeader(dataSize: Int, sampleRate: Int, channels: Int): ByteArray =
        ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII)); putInt(36 + dataSize)
            put("WAVE".toByteArray(Charsets.US_ASCII)); put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16); putShort(1); putShort(channels.toShort()); putInt(sampleRate)
            putInt(sampleRate * channels * 2); putShort((channels * 2).toShort()); putShort(16)
            put("data".toByteArray(Charsets.US_ASCII)); putInt(dataSize)
        }.array()

    private data class Piece(val text: String, val score: Float, val type: Int)

    private class SentencePieceTokenizer(
        private val pieces: Map<String, Int>,
        private val scores: FloatArray,
        private val maxPieceLength: Int,
        private val fallbackScore: Float
    ) {
        fun encode(value: String): IntArray {
            val normalized = Normalizer.normalize(value, Normalizer.Form.NFKC)
                .trim().replace(Regex("\\s+"), " ")
            val text = "▁" + normalized.replace(' ', '▁')
            if (text.isEmpty()) return intArrayOf(1)
            val best = FloatArray(text.length + 1) { Float.NEGATIVE_INFINITY }.also { it[0] = 0f }
            val previous = IntArray(text.length + 1) { -1 }
            val token = IntArray(text.length + 1) { 2 }
            for (start in text.indices) {
                if (best[start] == Float.NEGATIVE_INFINITY) continue
                for (length in 1..min(maxPieceLength, text.length - start)) {
                    val id = pieces[text.substring(start, start + length)] ?: continue
                    val candidate = best[start] + scores[id]
                    if (candidate > best[start + length]) {
                        best[start + length] = candidate
                        previous[start + length] = start
                        token[start + length] = id
                    }
                }
                if (best[start] + fallbackScore > best[start + 1]) {
                    best[start + 1] = best[start] + fallbackScore
                    previous[start + 1] = start
                    token[start + 1] = 2
                }
            }
            val result = ArrayList<Int>()
            var cursor = text.length
            while (cursor > 0) {
                result += token[cursor]
                cursor = previous[cursor]
            }
            result.reverse()
            if (result.lastOrNull() != 1) result += 1
            return result.toIntArray()
        }

        companion object {
            fun load(file: File): SentencePieceTokenizer {
                val parsed = parseSentencePiece(file.readBytes())
                val scores = FloatArray(parsed.size) { Float.NEGATIVE_INFINITY }
                val map = HashMap<String, Int>(parsed.size * 2)
                var maxLength = 1
                var minimum = 0f
                parsed.forEachIndexed { index, piece ->
                    scores[index] = piece.score
                    if ((piece.type == 1 || piece.type == 4) && piece.text.isNotEmpty()) {
                        map[piece.text] = index
                        maxLength = max(maxLength, piece.text.length)
                        minimum = min(minimum, piece.score)
                    }
                }
                return SentencePieceTokenizer(map, scores, maxLength, minimum - 10f)
            }
        }
    }

    private class BpeTokenizer(
        private val pieces: Map<String, Int>,
        private val scores: FloatArray,
        private val byteIds: IntArray,
        private val unknownId: Int
    ) {
        fun encode(value: String): IntArray {
            val symbols = value.replace(' ', '▁').codePoints().toArray()
                .map { String(Character.toChars(it)) }.toMutableList()
            while (symbols.size > 1) {
                var bestIndex = -1
                var bestScore = Float.NEGATIVE_INFINITY
                for (index in 0 until symbols.lastIndex) {
                    val id = pieces[symbols[index] + symbols[index + 1]] ?: continue
                    if (scores[id] > bestScore) {
                        bestScore = scores[id]
                        bestIndex = index
                    }
                }
                if (bestIndex < 0) break
                symbols[bestIndex] = symbols[bestIndex] + symbols[bestIndex + 1]
                symbols.removeAt(bestIndex + 1)
            }
            val result = ArrayList<Int>()
            for (symbol in symbols) {
                val id = pieces[symbol]
                if (id != null) result += id else symbol.toByteArray(Charsets.UTF_8).forEach { byte ->
                    result += byteIds[byte.toInt() and 0xff].takeIf { it >= 0 } ?: unknownId
                }
            }
            return result.toIntArray()
        }

        companion object {
            fun load(file: File): BpeTokenizer {
                val parsed = parseSentencePiece(file.readBytes())
                val scores = FloatArray(parsed.size)
                val map = HashMap<String, Int>(parsed.size * 2)
                val byteIds = IntArray(256) { -1 }
                var unknown = 0
                parsed.forEachIndexed { index, piece ->
                    scores[index] = piece.score
                    when (piece.type) {
                        2 -> unknown = index
                        6 -> if (piece.text.matches(Regex("<0x[0-9A-Fa-f]{2}>"))) {
                            byteIds[piece.text.substring(3, 5).toInt(16)] = index
                        }
                        else -> if (piece.text.isNotEmpty()) map[piece.text] = index
                    }
                }
                return BpeTokenizer(map, scores, byteIds, unknown)
            }
        }
    }

    private fun parseSentencePiece(bytes: ByteArray): List<Piece> {
        val reader = ProtoReader(bytes)
        val result = ArrayList<Piece>()
        while (!reader.done) {
            val tag = reader.varint().toInt()
            if (tag ushr 3 == 1 && tag and 7 == 2) {
                // Read the length first. Reading reader.position on the left side of
                // this expression used to capture the position *before* the varint,
                // making the embedded message end a few bytes early and desynchronizing
                // all following protobuf fields.
                val messageLength = reader.varint().toInt()
                val end = reader.position + messageLength
                require(messageLength >= 0 && end in reader.position..bytes.size) {
                    "Invalid SentencePiece entry length $messageLength at ${reader.position}"
                }
                var text = ""
                var score = 0f
                var type = 1
                while (reader.position < end) {
                    val pieceTag = reader.varint().toInt()
                    when {
                        pieceTag ushr 3 == 1 && pieceTag and 7 == 2 -> text = reader.string()
                        pieceTag ushr 3 == 2 && pieceTag and 7 == 5 -> score = reader.float32()
                        pieceTag ushr 3 == 3 && pieceTag and 7 == 0 -> type = reader.varint().toInt()
                        else -> reader.skip(pieceTag and 7)
                    }
                }
                reader.position = end
                result += Piece(text, score, type)
            } else reader.skip(tag and 7)
        }
        require(result.isNotEmpty()) { "Invalid SentencePiece model" }
        return result
    }

    private class ProtoReader(private val data: ByteArray) {
        var position = 0
        val done: Boolean get() = position >= data.size

        fun varint(): Long {
            var result = 0L
            var shift = 0
            while (shift < 64) {
                require(position < data.size) { "Truncated protobuf varint" }
                val byte = data[position++].toInt() and 0xff
                result = result or ((byte and 0x7f).toLong() shl shift)
                if (byte and 0x80 == 0) return result
                shift += 7
            }
            error("Malformed protobuf varint")
        }

        fun string(): String {
            val length = varint().toInt()
            require(length >= 0 && position + length in position..data.size) {
                "Invalid protobuf string length $length at $position"
            }
            return data.decodeToString(position, position + length).also { position += length }
        }

        fun float32(): Float {
            require(position + 4 <= data.size) { "Truncated protobuf float at $position" }
            val value = ByteBuffer.wrap(data, position, 4).order(ByteOrder.LITTLE_ENDIAN).float
            position += 4
            return value
        }

        fun skip(wireType: Int) {
            when (wireType) {
                0 -> varint()
                1 -> advance(8)
                2 -> advance(varint().toInt())
                5 -> advance(4)
                else -> error("Unsupported protobuf wire type $wireType")
            }
        }

        private fun advance(count: Int) {
            require(count >= 0 && position + count in position..data.size) {
                "Invalid protobuf field length $count at $position"
            }
            position += count
        }
    }
}
