package com.llmhub.llmhub.inference

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorType
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.imagegen.LiteRtSdPipeline
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Parakeet TDT 0.6B v3 compiled ahead of time for one SoC's NPU
 * (`parakeet_tdt_0.6b_v3_5s_f32_stateful_<SoC>.tflite` in litert-community/parakeet-tdt-0.6b-v3).
 *
 * Unlike the `.litertlm` bundle these files hold only the network, so the log-mel front end,
 * greedy TDT decoding, detokenizing and the stitching of 5-second windows happen here. The
 * decoding follows Google's litert-samples speech_recognition app, which runs the same files.
 */
class ParakeetNpuEngine private constructor(
    private val environment: Environment,
    private val model: CompiledModel,
    private val vocabulary: List<String>,
    private val executor: ExecutorService
) {
    private val dispatcher = executor.asCoroutineDispatcher()
    private val encodeInputs = model.createInputBuffers(ENCODE)
    private val encodeOutputs = model.createOutputBuffers(ENCODE)
    private val decoder = TdtDecoder(model)

    /** Transcribes 16 kHz mono PCM16. Returns null when nothing was recognized. */
    suspend fun transcribe(pcm16Bytes: ByteArray): String? = withContext(dispatcher) {
        val shorts = ByteBuffer.wrap(pcm16Bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val samples = FloatArray(shorts.remaining()) { shorts.get(it) / 32768f }
        if (samples.size < ParakeetFeatures.HOP * 2) return@withContext null

        val words = mutableListOf<ParakeetText.Word>()
        var previousEnd = 0
        for (start in windowStarts(samples.size)) {
            val valid = min(ParakeetFeatures.WINDOW_SAMPLES, samples.size - start)
            val window = samples.copyOfRange(start, start + valid).copyOf(ParakeetFeatures.WINDOW_SAMPLES)
            encodeInputs[0].writeFloat(ParakeetFeatures.logMel(window, valid))
            model.run(encodeInputs, encodeOutputs, ENCODE)
            val tokens = decoder.decode(encodeOutputs[0].readFloat(), (valid + SAMPLES_PER_FRAME - 1) / SAMPLES_PER_FRAME)
            val heard = ParakeetText.words(tokens, vocabulary, start * 1000 / ParakeetFeatures.SAMPLE_RATE)
            // Both windows heard the overlap; each keeps the half that is further from its own edge.
            val cutMs = (previousEnd + start) / 2 * 1000 / ParakeetFeatures.SAMPLE_RATE
            ParakeetText.merge(words, heard, if (start == 0) 0 else cutMs)
            previousEnd = start + ParakeetFeatures.WINDOW_SAMPLES
        }
        words.joinToString(" ") { it.text }.ifEmpty { null }
    }

    fun close() {
        executor.execute {
            runCatching { model.close() }
            runCatching { environment.close() }
        }
        executor.shutdown()
    }

    /** Greedy Token-and-Duration Transducer decoding over the `decode` and `decode_1` signatures. */
    private class TdtDecoder(private val model: CompiledModel) {
        private val inputs = model.createInputBuffers(DECODE)
        private val outputs = model.createOutputBuffers(DECODE)
        private val frames = model.getInputTensorType("args_0", DECODE).elements / ENCODER_DIM
        private val prefixTokens = model.getInputTensorType("args_1", DECODE).elements
        private val logitsPerToken = model.getOutputTensorType("output_0", DECODE).elements / prefixTokens / frames
        private val stateSize = model.getInputTensorType("args_2", DECODE).elements
        private val blank = logitsPerToken - DURATIONS - 1

        // Single-token stateful decoding; absent from the stateless builds.
        private val stepInputs = runCatching { model.createInputBuffers(DECODE_STEP) }.getOrNull()
        private val stepOutputs = stepInputs?.let { model.createOutputBuffers(DECODE_STEP) }

        /**
         * Returns (token id, encoder frame) pairs for the first [validFrames] frames.
         *
         * Every signature keeps its own buffers and data is copied between them. Handing one
         * signature's buffers to another, or swapping the LSTM state buffers between runs, saves
         * the copies but makes the Qualcomm dispatch lose its buffer registrations mid-decode.
         */
        fun decode(encoded: FloatArray, validFrames: Int): List<Pair<Int, Int>> {
            val result = mutableListOf<Pair<Int, Int>>()
            inputs[0].writeFloat(encoded)
            inputs[2].writeFloat(FloatArray(stateSize))
            inputs[3].writeFloat(FloatArray(stateSize))

            // The first tokens go through stateless `decode`, which keeps their context better.
            var stateful = false
            val tokenIds = IntArray(prefixTokens).also { it[0] = blank }
            var tokenIndex = 0
            var frame = 0
            var emittedAtFrame = 0
            val lastFrame = min(frames, validFrames)
            while (frame < lastFrame) {
                val logits = if (stateful) {
                    model.run(stepInputs!!, stepOutputs!!, DECODE_STEP)
                    stepOutputs[0].readFloat()
                } else {
                    inputs[1].writeInt(tokenIds)
                    model.run(inputs, outputs, DECODE)
                    outputs[0].readFloat()
                }
                val tokensPerRun = if (stateful) 1 else prefixTokens
                val tokenStart = (frame * tokensPerRun + if (stateful) 0 else tokenIndex) * logitsPerToken
                val durationStart = tokenStart + logitsPerToken - DURATIONS
                val token = argmax(logits, tokenStart, durationStart) - tokenStart
                var duration = argmax(logits, durationStart, durationStart + DURATIONS) - durationStart

                if (token != blank) {
                    result += token to frame
                    emittedAtFrame++
                    if (stateful) {
                        // The prediction network only advances when a token is emitted.
                        stepInputs!![1].writeInt(intArrayOf(token))
                        stepInputs[2].writeFloat(stepOutputs!![1].readFloat())
                        stepInputs[3].writeFloat(stepOutputs[2].readFloat())
                    } else if (++tokenIndex < prefixTokens) {
                        tokenIds[tokenIndex] = token
                    } else if (stepInputs != null) {
                        // The prefix is full: carry on one token at a time from the state it left.
                        stateful = true
                        stepInputs[0].writeFloat(encoded)
                        stepInputs[1].writeInt(intArrayOf(token))
                        stepInputs[2].writeFloat(outputs[1].readFloat())
                        stepInputs[3].writeFloat(outputs[2].readFloat())
                    } else break
                }
                // A blank, or too many tokens on one frame, always moves on.
                if (duration == 0 && (token == blank || emittedAtFrame >= MAX_TOKENS_PER_FRAME)) duration = 1
                if (duration > 0) emittedAtFrame = 0
                frame += duration
            }
            return result
        }

        private fun argmax(values: FloatArray, from: Int, until: Int): Int {
            var best = from
            for (i in from + 1 until until) if (values[i] > values[best]) best = i
            return best
        }
    }

    companion object {
        private const val TAG = "ParakeetNpuEngine"
        private const val ENCODE = "encode"
        private const val DECODE = "decode"
        private const val DECODE_STEP = "decode_1"
        private const val ENCODER_DIM = 1024
        private const val DURATIONS = 5
        private const val MAX_TOKENS_PER_FRAME = 10
        /** One encoder frame: a 160-sample mel hop, subsampled 8 times. */
        private const val SAMPLES_PER_FRAME = ParakeetFeatures.HOP * 8
        /** Windows advance 3 s, so neighbours share 2 s. */
        private const val STRIDE_SAMPLES = 3 * ParakeetFeatures.SAMPLE_RATE
        private const val VOCABULARY_ASSET = "parakeet_tdt_v3_vocab.txt"

        private val TensorType.elements: Int get() = layout!!.dimensions.fold(1, Int::times)

        /**
         * Start samples of the 5-second windows covering [totalSamples]. The last window is moved
         * back to end with the audio, so only a recording shorter than one window is zero-padded.
         */
        internal fun windowStarts(totalSamples: Int): List<Int> {
            val starts = mutableListOf(0)
            var start = 0
            while (start + ParakeetFeatures.WINDOW_SAMPLES < totalSamples) {
                start = min(start + STRIDE_SAMPLES, totalSamples - ParakeetFeatures.WINDOW_SAMPLES)
                starts += start
            }
            return starts
        }

        /** Loads [modelFile] on the NPU. Null when the file or this device's NPU runtime is unusable. */
        suspend fun create(context: Context, modelFile: File): ParakeetNpuEngine? {
            if (!modelFile.isFile) return null
            val executor = Executors.newSingleThreadExecutor { Thread(it, "parakeet-npu").apply { isDaemon = true } }
            // Uncancellable: a cancelled withContext would drop the native model it returns.
            val engine = withContext(NonCancellable + executor.asCoroutineDispatcher()) {
                var environment: Environment? = null
                var model: CompiledModel? = null
                try {
                    val vocabulary = context.assets.open(VOCABULARY_ASSET).bufferedReader().use { it.readLines() }
                    environment = npuEnvironment(context)
                    val options = CompiledModel.Options(setOf(Accelerator.NPU, Accelerator.CPU))
                    if (DeviceInfo.isSnapdragon()) {
                        options.qualcommOptions = CompiledModel.QualcommOptions(
                            htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
                        )
                    }
                    val started = System.currentTimeMillis()
                    model = CompiledModel.create(modelFile.absolutePath, options, environment)
                    Log.i(TAG, "Loaded ${modelFile.name} in ${System.currentTimeMillis() - started} ms")
                    check(encoderHearsInput(model)) { "The NPU encoder gives no usable output on this device" }
                    ParakeetNpuEngine(environment, model, vocabulary, executor)
                } catch (t: Throwable) {
                    Log.e(TAG, "Failed to load ${modelFile.name} on the NPU", t)
                    runCatching { model?.close() }
                    runCatching { environment?.close() }
                    null
                }
            }
            if (engine == null) executor.shutdown()
            return engine
        }

        /**
         * Some chips load an ahead-of-time compiled encoder and then return NaN (Snapdragon 8 Elite)
         * or the same output for any audio (Dimensity 9400/9500), which decodes to nothing. Two
         * different inputs must give different, finite encodings, or the model is unusable here.
         */
        private fun encoderHearsInput(model: CompiledModel): Boolean {
            val inputs = model.createInputBuffers(ENCODE)
            val outputs = model.createOutputBuffers(ENCODE)
            val size = ParakeetFeatures.MELS * ParakeetFeatures.FRAMES
            inputs[0].writeFloat(FloatArray(size))
            model.run(inputs, outputs, ENCODE)
            val silence = outputs[0].readFloat()
            inputs[0].writeFloat(FloatArray(size) { sin(it * 0.37f) })
            model.run(inputs, outputs, ENCODE)
            val pattern = outputs[0].readFloat()
            return pattern.none { it.isNaN() } && !pattern.contentEquals(silence)
        }

        private fun npuEnvironment(context: Context): Environment {
            val dir = if (DeviceInfo.isSnapdragon()) qualcommLibraryDir(context) else LiteRtSdPipeline.vendorLibraryDir(context)
            return dir?.let { Environment.create(mapOf(Environment.Option.DispatchLibraryDir to it.path)) }
                ?: Environment.create(BuiltinNpuAcceleratorProvider(context))
        }

        /**
         * The Qualcomm dispatch library loads QNN from its own directory, and LiteRT 2.3 needs the
         * QAIRT 2.50 runtime shipped in assets/qnnlibs_litert (see LiteRtLmEmbeddingService).
         */
        private fun qualcommLibraryDir(context: Context): File? {
            val dispatch = File(context.applicationInfo.nativeLibraryDir, "libLiteRtDispatch_Qualcomm.so")
            val names = context.assets.list("qnnlibs_litert").orEmpty()
            if (!dispatch.isFile || names.isEmpty()) return null
            val dir = File(context.filesDir, "litert_npu_qnn_250").apply { mkdirs() }
            val copy = File(dir, dispatch.name)
            if (copy.length() != dispatch.length()) dispatch.copyTo(copy, overwrite = true)
            for (name in names) {
                val target = File(dir, name)
                if (target.length() > 0L) continue
                context.assets.open("qnnlibs_litert/$name").use { input -> target.outputStream().use { input.copyTo(it) } }
                target.setReadable(true, false)
                target.setExecutable(true, false)
            }
            return dir
        }
    }
}

/** The log-mel front end Parakeet was trained with (NeMo AudioToMelSpectrogramPreprocessor). */
internal object ParakeetFeatures {
    const val SAMPLE_RATE = 16000
    const val WINDOW_SAMPLES = 5 * SAMPLE_RATE
    const val HOP = 160
    const val MELS = 128
    const val FRAMES = WINDOW_SAMPLES / HOP
    private const val FFT = 512
    private const val WINDOW = 400
    private const val BINS = FFT / 2 + 1
    private const val PREEMPHASIS = 0.97f
    private const val LOG_GUARD = 5.9604645e-8f // 2^-24

    /** Symmetric Hann window of 400 samples, centred in the 512-point FFT frame. */
    private val window = FloatArray(FFT).also {
        val offset = (FFT - WINDOW) / 2
        for (i in 0 until WINDOW) it[offset + i] = (0.5 - 0.5 * cos(2.0 * PI * i / (WINDOW - 1))).toFloat()
    }
    private val cosTable = FloatArray(FFT / 2) { cos(2.0 * PI * it / FFT).toFloat() }
    private val sinTable = FloatArray(FFT / 2) { sin(2.0 * PI * it / FFT).toFloat() }
    private val reversed = IntArray(FFT) { Integer.reverse(it) ushr (32 - Integer.numberOfTrailingZeros(FFT)) }

    /** Slaney-scale triangular filters with Slaney area normalization, as librosa.filters.mel builds them. */
    private val filterStart = IntArray(MELS)
    private val filters: Array<FloatArray>

    init {
        val step = 200.0 / 3
        val logStep = ln(6.4) / 27
        fun toMel(hz: Double) = if (hz < 1000) hz / step else 15 + ln(hz / 1000) / logStep
        fun toHz(mel: Double) = if (mel < 15) mel * step else 1000 * exp((mel - 15) * logStep)
        val top = toMel(SAMPLE_RATE / 2.0)
        val edges = DoubleArray(MELS + 2) { toHz(top * it / (MELS + 1)) }
        filters = Array(MELS) { m ->
            val weights = DoubleArray(BINS) { bin ->
                val hz = bin * SAMPLE_RATE.toDouble() / FFT
                val rising = (hz - edges[m]) / (edges[m + 1] - edges[m])
                val falling = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1])
                max(0.0, min(rising, falling)) * 2 / (edges[m + 2] - edges[m])
            }
            val first = weights.indexOfFirst { it > 0 }.coerceAtLeast(0)
            val last = weights.indexOfLast { it > 0 }.coerceAtLeast(first)
            filterStart[m] = first
            FloatArray(last - first + 1) { weights[first + it].toFloat() }
        }
    }

    /**
     * Features of one [WINDOW_SAMPLES] window as [MELS] rows of [FRAMES] values. Each mel band is
     * normalized over the frames holding the first [validSamples] samples; the padding stays zero.
     */
    fun logMel(audio: FloatArray, validSamples: Int = audio.size): FloatArray {
        require(audio.size == WINDOW_SAMPLES) { "Expected $WINDOW_SAMPLES samples, got ${audio.size}" }
        val signal = FloatArray(WINDOW_SAMPLES)
        signal[0] = audio[0]
        for (i in 1 until WINDOW_SAMPLES) signal[i] = audio[i] - PREEMPHASIS * audio[i - 1]

        val validFrames = (validSamples / HOP).coerceIn(1, FRAMES)
        val features = FloatArray(MELS * FRAMES)
        val re = FloatArray(FFT)
        val im = FloatArray(FFT)
        val power = FloatArray(BINS)
        for (frame in 0 until validFrames) {
            // torch.stft(center=True): the frame is centred on its hop, reflecting at the edges.
            val origin = frame * HOP - FFT / 2
            for (i in 0 until FFT) {
                var index = origin + i
                if (index < 0) index = -index
                if (index >= WINDOW_SAMPLES) index = 2 * (WINDOW_SAMPLES - 1) - index
                re[reversed[i]] = signal[index] * window[i]
                im[reversed[i]] = 0f
            }
            fft(re, im)
            for (bin in 0 until BINS) power[bin] = re[bin] * re[bin] + im[bin] * im[bin]
            for (m in 0 until MELS) {
                val weights = filters[m]
                var energy = 0f
                for (i in weights.indices) energy += weights[i] * power[filterStart[m] + i]
                features[m * FRAMES + frame] = ln(energy + LOG_GUARD)
            }
        }

        for (m in 0 until MELS) {
            val row = m * FRAMES
            var mean = 0f
            for (frame in 0 until validFrames) mean += features[row + frame]
            mean /= validFrames
            var variance = 0f
            for (frame in 0 until validFrames) variance += (features[row + frame] - mean).let { it * it }
            val deviation = sqrt(variance / max(1, validFrames - 1)) + 1e-5f
            for (frame in 0 until validFrames) features[row + frame] = (features[row + frame] - mean) / deviation
        }
        return features
    }

    /** In-place radix-2 FFT of bit-reversed input. */
    private fun fft(re: FloatArray, im: FloatArray) {
        var size = 2
        while (size <= FFT) {
            val half = size / 2
            val step = FFT / size
            for (start in 0 until FFT step size) {
                for (k in 0 until half) {
                    val wr = cosTable[k * step]
                    val wi = -sinTable[k * step]
                    val a = start + k
                    val b = a + half
                    val tr = re[b] * wr - im[b] * wi
                    val ti = re[b] * wi + im[b] * wr
                    re[b] = re[a] - tr
                    im[b] = im[a] - ti
                    re[a] += tr
                    im[a] += ti
                }
            }
            size *= 2
        }
    }
}

/** Turns per-window tokens into words and stitches overlapping windows together. */
internal object ParakeetText {
    data class Word(val text: String, val ms: Int)

    private const val WORD_START = '▁' // SentencePiece's "▁"
    private const val FRAME_MS = 80
    /** Words this close to the cut are kept from both windows and then de-duplicated. */
    private const val CUT_SLACK_MS = 160
    private val punctuation = Regex("[\\p{Punct}\\s]")

    /** Words of one window, timed by their first token; [offsetMs] is where the window starts. */
    fun words(tokens: List<Pair<Int, Int>>, vocabulary: List<String>, offsetMs: Int): List<Word> {
        val words = mutableListOf<Word>()
        val current = StringBuilder()
        var startMs = 0
        fun flush() {
            val text = current.toString().replace(WORD_START, ' ').trim()
            if (text.isNotEmpty()) words += Word(text, startMs)
            current.clear()
        }
        for ((id, frame) in tokens) {
            val piece = vocabulary.getOrNull(id).orEmpty()
            // Control tokens such as <|nospeech|> are not text.
            if (piece.isEmpty() || (piece.startsWith("<|") && piece.endsWith("|>"))) continue
            if (piece[0] == WORD_START || current.isEmpty()) {
                flush()
                startMs = offsetMs + frame * FRAME_MS
            }
            current.append(piece)
        }
        flush()
        return words
    }

    /** Appends [next] to [words], taking words before [cutMs] from [words] and later ones from [next]. */
    fun merge(words: MutableList<Word>, next: List<Word>, cutMs: Int) {
        while (words.isNotEmpty() && words.last().ms >= cutMs + CUT_SLACK_MS) words.removeAt(words.lastIndex)
        var incoming = next.filter { it.ms >= cutMs - CUT_SLACK_MS }
        for (count in min(3, min(words.size, incoming.size)) downTo 1) {
            val tail = words.takeLast(count)
            if (tail.indices.all { sameWord(tail[it].text, incoming[it].text) }) {
                incoming = incoming.drop(count)
                break
            }
        }
        words += incoming
    }

    private fun sameWord(a: String, b: String): Boolean {
        val x = a.lowercase().replace(punctuation, "")
        val y = b.lowercase().replace(punctuation, "")
        return x == y || (min(x.length, y.length) > 3 && editDistance(x, y) <= 1)
    }

    private fun editDistance(a: String, b: String): Int {
        var previous = IntArray(b.length + 1) { it }
        for (i in 1..a.length) {
            val row = IntArray(b.length + 1)
            row[0] = i
            for (j in 1..b.length) {
                row[j] = min(min(previous[j] + 1, row[j - 1] + 1), previous[j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
            }
            previous = row
        }
        return previous[b.length]
    }
}
