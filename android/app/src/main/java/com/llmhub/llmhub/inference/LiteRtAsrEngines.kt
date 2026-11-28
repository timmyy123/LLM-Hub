package com.llmhub.llmhub.inference

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import org.json.JSONObject
import android.os.SystemClock
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorType
import com.llmhub.llmhub.data.ParakeetUltraModel
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * IBM Granite Speech 5.0 2B on the LiteRT-LM engine: one message per piece of audio (16 kHz mono WAV first, then
 * the source card's instruction), greedy decoding. The `_ekv2048` file holds about two minutes of audio per message,
 * so longer recordings go in 100-second pieces.
 */
class GraniteSpeechEngine private constructor(
    private val engine: Engine,
    private val executor: ExecutorService,
    val usesGpu: Boolean
) {
    private val dispatcher = executor.asCoroutineDispatcher()
    @Volatile private var current: com.google.ai.edge.litertlm.Conversation? = null
    @Volatile private var closed = false

    /** Transcribes 16 kHz mono PCM16; "" when nothing was said. */
    suspend fun transcribe(pcm16Bytes: ByteArray): String = withContext(dispatcher) {
        check(!closed) { "Granite Speech is closed" }
        val pieces = (pcm16Bytes.indices step PIECE_BYTES).map { pcm16Bytes.copyOfRange(it, minOf(it + PIECE_BYTES, pcm16Bytes.size)) }
        pieces.filter { it.size >= MIN_BYTES }.joinToString(" ") { piece ->
            engine.createConversation(ConversationConfig(samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0))).use { conversation ->
                current = conversation
                try {
                    val started = android.os.SystemClock.elapsedRealtime()
                    val reply = conversation.sendMessage(Contents.of(Content.AudioBytes(wav(piece)), Content.Text(PROMPT)))
                    logSpeed(conversation, piece.size / 32000.0, android.os.SystemClock.elapsedRealtime() - started)
                    reply.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }.trim()
                } finally {
                    current = null
                }
            }
        }.trim()
    }

    /** LiteRT-LM's own prefill (audio and prompt) and decode (transcript) token counts and speeds for one piece. */
    @OptIn(com.google.ai.edge.litertlm.ExperimentalApi::class)
    private fun logSpeed(conversation: com.google.ai.edge.litertlm.Conversation, audioSeconds: Double, ms: Long) {
        val info = runCatching { conversation.getBenchmarkInfo() }.getOrNull()
        Log.i(TAG, "Granite Speech: %.1f s of audio in %d ms".format(audioSeconds, ms) + (info?.let {
            ", prefill ${it.lastPrefillTokenCount} tokens at %.1f tok/s, decode ${it.lastDecodeTokenCount} tokens at %.1f tok/s"
                .format(it.lastPrefillTokensPerSecond, it.lastDecodeTokensPerSecond)
        } ?: ""))
    }

    /** Stops the transcription in progress, from any thread. */
    fun abort() { runCatching { current?.cancelProcess() } }

    fun close() {
        if (closed) return
        closed = true
        abort()
        executor.execute { runCatching { engine.close() }.onFailure { Log.w(TAG, "Failed to close Granite Speech", it) } }
        executor.shutdown()
    }

    companion object {
        private const val TAG = "GraniteSpeechEngine"
        /** The source card's prompt for a punctuated, capitalized transcript. */
        private const val PROMPT = "transcribe the speech with proper punctuation and capitalization."
        private const val PIECE_BYTES = 100 * 16000 * 2
        private const val MIN_BYTES = 16000 / 10 * 2

        /** Loads [file] on the GPU when [preferGpu] (falling back to the CPU), on the engine's own thread. */
        suspend fun create(context: Context, file: File, preferGpu: Boolean): GraniteSpeechEngine? {
            if (!file.isFile) return null
            val executor = Executors.newSingleThreadExecutor { Thread(it, "granite-speech").apply { isDaemon = true } }
            return try {
                withContext(NonCancellable + executor.asCoroutineDispatcher()) {
                    for (gpu in if (preferGpu) listOf(true, false) else listOf(false)) {
                        val engine = runCatching {
                            Engine(EngineConfig(
                                modelPath = file.absolutePath,
                                backend = if (gpu) Backend.GPU() else Backend.CPU(),
                                audioBackend = if (gpu) Backend.GPU() else Backend.CPU(),
                                cacheDir = context.cacheDir.path
                            )).also { it.initialize() }
                        }.onFailure { Log.w(TAG, "Granite Speech failed to load on ${if (gpu) "GPU" else "CPU"}", it) }.getOrNull()
                        if (engine != null) {
                            Log.i(TAG, "Granite Speech loaded on ${if (gpu) "GPU" else "CPU"}")
                            return@withContext GraniteSpeechEngine(engine, executor, gpu)
                        }
                    }
                    null
                }.also { if (it == null) executor.shutdown() }
            } catch (e: CancellationException) {
                executor.shutdown(); throw e
            }
        }

        /** 16 kHz mono PCM16 WAV around [pcm]. */
        internal fun wav(pcm: ByteArray): ByteArray = ByteBuffer.allocate(44 + pcm.size).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + pcm.size); put("WAVE".toByteArray())
            put("fmt ".toByteArray()); putInt(16); putShort(1); putShort(1); putInt(16000); putInt(32000); putShort(2); putShort(16)
            put("data".toByteArray()); putInt(pcm.size); put(pcm)
        }.array()
    }
}

/**
 * Parakeet Ultra (moondream's post-trained Parakeet TDT 0.6B v3) as litert-community's LiteRT graphs, run the way its
 * card's runner runs them: 30-second windows, each through the smallest mel + encoder pair that holds it (10 s or 30 s),
 * then greedy Token-and-Duration Transducer decoding over the decoder's `predict` and `joint` signatures. The mel graphs
 * and the decoder run on the CPU; the encoder runs on the CPU (int8 files), the GPU (float16) or the NPU (float16
 * compiled for one SoC). Every graph is a LiteRT CompiledModel with its tensor buffers made once.
 */
class ParakeetUltraEngine private constructor(
    private val environment: Environment?,
    private val mel10: Graph,
    private val mel30: Graph,
    /** Null on the GPU, which holds only the 30 s encoder: two float16 encoders do not fit in a 12 GB tablet's memory. */
    private val encoder10: Graph?,
    private val encoder30: Graph,
    private val decoder: CompiledModel,
    private val vocabulary: List<String>,
    /** "CPU", "GPU" or "NPU": where the encoder runs. */
    val accelerator: String,
    private val executor: ExecutorService,
) {
    private val dispatcher = executor.asCoroutineDispatcher()
    private val predictIn = mapOf("token" to decoder.createInputBuffer("token", PREDICT), "h" to decoder.createInputBuffer("h", PREDICT),
        "c" to decoder.createInputBuffer("c", PREDICT))
    private val predictOut = mapOf("dec_out" to decoder.createOutputBuffer("dec_out", PREDICT),
        "h_out" to decoder.createOutputBuffer("h_out", PREDICT), "c_out" to decoder.createOutputBuffer("c_out", PREDICT))
    private val jointIn = mapOf("enc_frame" to decoder.createInputBuffer("enc_frame", JOINT), "dec_out" to decoder.createInputBuffer("dec_out", JOINT))
    private val jointOut = mapOf("logits" to decoder.createOutputBuffer("logits", JOINT))
    @Volatile private var aborted = false
    @Volatile private var closed = false

    /** One mel or encoder graph (first signature) with its buffers, by tensor name. */
    internal class Graph(val model: CompiledModel, inputs: List<String>, outputs: List<String>) {
        val ins = inputs.associateWith { model.createInputBuffer(it) }
        val outs = outputs.associateWith { model.createOutputBuffer(it) }
        private val inTypes = inputs.associateWith { model.getInputTensorType(it).elementType }
        private val outTypes = outputs.associateWith { model.getOutputTensorType(it).elementType }
        fun run() = model.run(ins, outs)

        /** An output as floats, whatever its element type (the masks may be float, int or bool). */
        fun read(name: String): FloatArray {
            val buffer = outs.getValue(name)
            return when (outTypes.getValue(name)) {
                TensorType.ElementType.FLOAT -> buffer.readFloat()
                TensorType.ElementType.INT -> buffer.readInt().let { v -> FloatArray(v.size) { v[it].toFloat() } }
                TensorType.ElementType.BOOLEAN -> buffer.readBoolean().let { v -> FloatArray(v.size) { if (v[it]) 1f else 0f } }
                TensorType.ElementType.INT8 -> buffer.readInt8().let { v -> FloatArray(v.size) { v[it].toFloat() } }
                else -> error("Unexpected type of $name")
            }
        }

        /** Writes [values] to an input in its element type. */
        fun write(name: String, values: FloatArray) {
            val buffer = ins.getValue(name)
            when (inTypes.getValue(name)) {
                TensorType.ElementType.FLOAT -> buffer.writeFloat(values)
                TensorType.ElementType.INT -> buffer.writeInt(IntArray(values.size) { values[it].toInt() })
                TensorType.ElementType.BOOLEAN -> buffer.writeBoolean(BooleanArray(values.size) { values[it] > 0.5f })
                TensorType.ElementType.INT8 -> buffer.writeInt8(ByteArray(values.size) { values[it].toInt().toByte() })
                else -> error("Unexpected type of $name")
            }
        }
        fun close() { (ins.values + outs.values).forEach { runCatching { it.close() } }; runCatching { model.close() } }
    }

    /** Transcribes 16 kHz mono PCM16; "" when nothing was said. */
    suspend fun transcribe(pcm16Bytes: ByteArray): String = withContext(dispatcher) {
        check(!closed) { "Parakeet Ultra is closed" }
        aborted = false
        val shorts = ByteBuffer.wrap(pcm16Bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val samples = FloatArray(shorts.remaining()) { shorts.get(it) / 32768f }
        val texts = mutableListOf<String>()
        var start = 0
        while (start < samples.size && !aborted) {
            val valid = minOf(WINDOW, samples.size - start)
            if (valid >= MIN_SAMPLES) {
                val tokens = window(samples, start, valid)
                val text = ParakeetText.words(tokens.map { it to 0 }, vocabulary, 0).joinToString(" ") { it.text }
                if (text.isNotEmpty()) texts += text
            }
            start += WINDOW
        }
        texts.joinToString(" ")
    }

    fun abort() { aborted = true }

    private fun window(samples: FloatArray, start: Int, valid: Int): List<Int> {
        val short = valid <= 10 * 16000 && encoder10 != null
        val mel = if (short) mel10 else mel30
        val encoder = if (short) encoder10!! else encoder30
        val bucket = (if (short) 10 else 30) * 16000
        val audio = FloatArray(bucket).also { samples.copyInto(it, 0, start, start + valid) }

        val t0 = SystemClock.elapsedRealtime()
        mel.ins.getValue("audio").writeFloat(audio)
        mel.ins.getValue("n_valid").writeInt(intArrayOf(valid))
        mel.run()
        val t1 = SystemClock.elapsedRealtime()
        encoder.write("features", mel.read("features"))
        encoder.write("mel_mask", mel.read("mel_mask"))
        encoder.run()
        val enc = encoder.outs.getValue("enc").readFloat()
        val t2 = SystemClock.elapsedRealtime()
        val frames = minOf(enc.size / ENC_DIM, mel.read("frame_mask").count { it > 0.5f })

        var decOut = predict(BLANK, FloatArray(STATE), FloatArray(STATE))
        val ids = mutableListOf<Int>()
        var t = 0
        var steps = 0
        var run = 0
        while (t < frames && steps < MAX_STEPS && !aborted) {
            steps++
            val logits = joint(enc.copyOfRange(t * ENC_DIM, (t + 1) * ENC_DIM), decOut)
            val token = argmax(logits, 0, BLANK + 1)
            var d = argmax(logits, BLANK + 1, logits.size) - (BLANK + 1)
            if (token == BLANK && d == 0) d = 1
            if (token != BLANK) {
                if (token == EOS) break
                ids += token
                decOut = predict(token, null, null)
            }
            if (d == 0) { run++; if (run >= MAX_SYMBOLS_PER_FRAME) d = 1 }
            if (d > 0) run = 0
            t += d
        }
        Log.i(TAG, "${valid / 16000.0} s window ($accelerator, ${bucket / 16000} s graphs): mel ${t1 - t0} ms, encoder ${t2 - t1} ms, " +
            "decoding ${SystemClock.elapsedRealtime() - t2} ms ($steps steps, ${ids.size} tokens)")
        return ids
    }

    /** Feeds [token] to the prediction network; null [h]/[c] continue from its last state. Returns dec_out. */
    private fun predict(token: Int, h: FloatArray?, c: FloatArray?): FloatArray {
        predictIn.getValue("token").writeInt(intArrayOf(token))
        predictIn.getValue("h").writeFloat(h ?: predictOut.getValue("h_out").readFloat())
        predictIn.getValue("c").writeFloat(c ?: predictOut.getValue("c_out").readFloat())
        decoder.run(predictIn, predictOut, PREDICT)
        return predictOut.getValue("dec_out").readFloat()
    }

    private fun joint(frame: FloatArray, decOut: FloatArray): FloatArray {
        jointIn.getValue("enc_frame").writeFloat(frame)
        jointIn.getValue("dec_out").writeFloat(decOut)
        decoder.run(jointIn, jointOut, JOINT)
        return jointOut.getValue("logits").readFloat()
    }

    fun close() {
        if (closed) return
        closed = true
        aborted = true
        executor.execute {
            listOfNotNull(mel10, mel30, encoder10, encoder30).forEach { it.close() }
            (predictIn.values + predictOut.values + jointIn.values + jointOut.values).forEach { runCatching { it.close() } }
            runCatching { decoder.close() }
            runCatching { environment?.close() }
        }
        executor.shutdown()
    }

    companion object {
        private const val TAG = "ParakeetUltraEngine"
        private const val PREDICT = "predict"
        private const val JOINT = "joint"
        private const val WINDOW = 479840
        private const val MIN_SAMPLES = 320
        private const val ENC_DIM = 640
        private const val STATE = 2 * 640
        private const val BLANK = 8192
        private const val EOS = 3
        private const val MAX_STEPS = 3750
        private const val MAX_SYMBOLS_PER_FRAME = 10
        private val MEL_INPUTS = listOf("audio", "n_valid")
        private val MEL_OUTPUTS = listOf("features", "mel_mask", "frame_mask")
        private val ENCODER_INPUTS = listOf("features", "mel_mask")
        private val ENCODER_OUTPUTS = listOf("enc")

        /** Test knob: the GPU encoder's arithmetic precision. */
        @Volatile internal var gpuPrecision = CompiledModel.GpuOptions.Precision.FP16

        /** Test knob: XNNPACK threads of the CPU encoder (the card's runner uses 4; its S26 numbers 8). */
        @Volatile internal var cpuThreads = 4

        private fun cpuOptions(threads: Int, weightCache: File? = null) = CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(threads, null, weightCache?.path)
        }

        /**
         * Loads [name]'s graphs from [dir] on the engine's own thread. The GPU download's encoder runs on the GPU when
         * [preferGpu] (else, or when the GPU refuses it, on the CPU); null when a file is missing or fails to load.
         */
        suspend fun create(context: android.content.Context, dir: File, name: String, preferGpu: Boolean): ParakeetUltraEngine? {
            val executor = Executors.newSingleThreadExecutor { Thread(it, "parakeet-ultra").apply { isDaemon = true } }
            // Uncancellable: a cancelled withContext would drop the native models it returns.
            val engine = withContext(NonCancellable + executor.asCoroutineDispatcher()) {
                val opened = mutableListOf<AutoCloseable>()
                try {
                    val started = SystemClock.elapsedRealtime()
                    val vocabulary = readVocabulary(File(dir, "tokenizer.json"))
                    fun graph(file: String, options: CompiledModel.Options, inputs: List<String>, outputs: List<String>) =
                        Graph(CompiledModel.create(File(dir, file).path, options), inputs, outputs).also { g -> opened += AutoCloseable { g.close() } }
                    val mel10 = graph("parakeet_ultra_mel_10s.tflite", cpuOptions(2), MEL_INPUTS, MEL_OUTPUTS)
                    val mel30 = graph("parakeet_ultra_mel_30s.tflite", cpuOptions(2), MEL_INPUTS, MEL_OUTPUTS)
                    val decoder = CompiledModel.create(File(dir, "parakeet_ultra_decoder_i8.tflite").path, cpuOptions(1))
                    opened += AutoCloseable { decoder.close() }

                    if (ParakeetUltraModel.isNpu(name)) {
                        // This chip's compiled encoders; LiteRT runs any part the NPU cannot on the CPU.
                        val environment = ParakeetNpuEngine.npuEnvironment(context)
                        opened += AutoCloseable { environment.close() }
                        val pair = ParakeetUltraModel.encoderSeconds(name).map { seconds ->
                            val options = CompiledModel.Options(setOf(Accelerator.NPU, Accelerator.CPU))
                            if (com.llmhub.llmhub.data.DeviceInfo.isSnapdragon()) {
                                options.qualcommOptions = CompiledModel.QualcommOptions(
                                    htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
                                )
                            }
                            Graph(CompiledModel.create(File(dir, ParakeetUltraModel.encoder(name, seconds)).path, options, environment),
                                ENCODER_INPUTS, ENCODER_OUTPUTS).also { g -> opened += AutoCloseable { g.close() } }
                        }
                        Log.i(TAG, "Parakeet Ultra loaded in ${SystemClock.elapsedRealtime() - started} ms, encoder on NPU")
                        return@withContext ParakeetUltraEngine(environment, mel10, mel30, pair[0], pair[1], decoder, vocabulary, "NPU", executor)
                    }

                    val gpu = ParakeetUltraModel.isGpu(name)
                    fun encoders(options: () -> CompiledModel.Options) = ParakeetUltraModel.encoderSeconds(name).map { seconds ->
                        graph(ParakeetUltraModel.encoder(name, seconds), options(), ENCODER_INPUTS, ENCODER_OUTPUTS)
                    }
                    var accelerator = "CPU"
                    val pair = (if (gpu && preferGpu) runCatching {
                        encoders {
                            CompiledModel.Options(Accelerator.GPU).apply {
                                gpuOptions = CompiledModel.GpuOptions(
                                    precision = gpuPrecision,
                                    serializationDir = File(dir, "gpu_cache").apply { mkdirs() }.path,
                                    modelCacheKey = "parakeet_ultra_f16_$gpuPrecision",
                                    serializeProgramCache = true,
                                )
                            }
                        }.also { accelerator = "GPU" }
                    }.onFailure { Log.w(TAG, "Parakeet Ultra encoder failed on the GPU; using the CPU", it) }.getOrNull() else null)
                        ?: encoders { cpuOptions(cpuThreads, File(dir, "xnnpack_${if (gpu) "f16" else "i8"}.cache")) }
                    Log.i(TAG, "Parakeet Ultra loaded in ${SystemClock.elapsedRealtime() - started} ms, encoder on $accelerator")
                    ParakeetUltraEngine(null, mel10, mel30, pair.getOrNull(pair.size - 2), pair.last(), decoder, vocabulary, accelerator, executor)
                } catch (t: Throwable) {
                    Log.e(TAG, "Parakeet Ultra failed to load", t)
                    opened.asReversed().forEach { runCatching { it.close() } }
                    null
                }
            }
            if (engine == null) executor.shutdown()
            return engine
        }

        private fun argmax(values: FloatArray, from: Int, to: Int): Int {
            var best = from
            for (i in from + 1 until to) if (values[i] > values[best]) best = i
            return best
        }

        /** Id to piece from tokenizer.json (BPE, Metaspace "▁"); byte-fallback pieces become their character. */
        internal fun readVocabulary(file: File): List<String> {
            val model = JSONObject(file.readText()).getJSONObject("model")
            val vocab = model.getJSONObject("vocab")
            val pieces = arrayOfNulls<String>(vocab.length())
            for (key in vocab.keys()) {
                val id = vocab.getInt(key)
                if (id in pieces.indices) pieces[id] = key
            }
            val byte = Regex("<0x([0-9A-Fa-f]{2})>")
            return pieces.map { piece ->
                val p = piece.orEmpty()
                byte.matchEntire(p)?.let { m -> m.groupValues[1].toInt(16).takeIf { it in 0x20..0x7E }?.toChar()?.toString() ?: "" } ?: p
            }
        }
    }
}
