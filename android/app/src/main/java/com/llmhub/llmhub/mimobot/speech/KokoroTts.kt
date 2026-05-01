package com.llmhub.llmhub.mimobot.speech

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.util.Log
import com.llmhub.llmhub.mimobot.MimoBotIds
import com.llmhub.llmhub.mimobot.speech.kokoro.G2P
import com.llmhub.llmhub.mimobot.speech.kokoro.KokoroVocab
import com.llmhub.llmhub.mimobot.speech.kokoro.VoicePack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.EnumSet

/**
 * Kokoro-82M neural TTS via ONNX Runtime.
 *
 * Reuses the existing `com.microsoft.onnxruntime:onnxruntime-android:1.24.1`
 * dependency — no new native libs.
 *
 * Pipeline (per utterance):
 *   text ──► G2P (espeak-ng or dict) ──► IPA ──► KokoroVocab.tokenise ──► int64 ids
 *                                                                       │
 *                                                                       ▼
 *   pad with 0 on both ends → (1, L+2) tokens, style row L from VoicePack
 *                                                                       │
 *                                                                       ▼
 *                                                ONNX run → float32 24 kHz audio
 *                                                                       │
 *                                                                       ▼
 *                                                resample 24 k → 16 k Int16
 *                                                                       │
 *                                                                       ▼
 *                                                chunk into 320-sample frames
 *
 * Models (download on first use, see KokoroAssets):
 *   - kokoro-v1.0.fp16.onnx   (ONNX export from onnx-community/Kokoro-82M-ONNX)
 *   - voices/<id>.bin         (one voice file per voice, ~512 KB each)
 *
 * G2P selection: defaults to [G2P.best], which prefers espeak-ng when its
 * native libs are present and falls back to the bundled dictionary otherwise.
 */
class KokoroTts(
    private val context: Context,
    private val modelPath: String,
    private val voicePackPath: String,
    @Suppress("unused") private val voiceId: String = "af_heart",
    private val speed: Float = 1.0f,
    private val g2p: G2P = G2P.best(context),
) : Tts {

    /** The G2P engine actually being used at runtime — exposed so the UI can show it. */
    val g2pName: String get() = g2p.displayName

    private var ortEnv: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var voicePack: VoicePack? = null
    private val mutex = Mutex()

    // Discovered at load time — different exports name these differently.
    private var inputTokensName: String = "input_ids"
    private var inputStyleName: String = "style"
    private var inputSpeedName: String = "speed"

    override suspend fun init() {
        withContext(Dispatchers.IO) {
            mutex.withLock {
                if (session != null) return@withContext
                val modelFile = File(modelPath)
                val voiceFile = File(voicePackPath)
                require(modelFile.exists()) { "Kokoro model not found at $modelPath" }
                require(voiceFile.exists()) { "Kokoro voice pack not found at $voicePackPath" }

                val env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    try {
                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                    } catch (t: Throwable) {
                        Log.w(TAG, "NNAPI EP unavailable, falling back to CPU: ${t.message}")
                    }
                }

                val s = env.createSession(modelFile.absolutePath, opts)
                // Probe input names — handles both `input_ids/style/speed` and
                // `tokens/style/speed` exports.
                for (name in s.inputNames) {
                    when (name.lowercase()) {
                        "input_ids", "tokens" -> inputTokensName = name
                        "style", "ref_s" -> inputStyleName = name
                        "speed" -> inputSpeedName = name
                    }
                }

                ortEnv = env
                session = s
                voicePack = VoicePack.load(voiceFile)
                Log.i(TAG, "Kokoro loaded — inputs=${s.inputNames}, voice rows=${voicePack!!.rows}")
            }
        }
    }

    override fun speakToPcm(text: String, language: String): Flow<ShortArray> = flow {
        if (session == null) init()
        val s = session ?: throw IllegalStateException("Kokoro not loaded")
        val env = ortEnv ?: throw IllegalStateException("OrtEnvironment missing")
        val pack = voicePack ?: throw IllegalStateException("VoicePack missing")

        val ipa = g2p.phonemize(text, language)
        if (ipa.isBlank()) return@flow
        val ids = KokoroVocab.tokenise(ipa)
        if (ids.isEmpty()) return@flow

        // Pad with 0 (PAD token) on both ends per kokoro-onnx convention.
        val padded = LongArray(ids.size + 2)
        for (i in ids.indices) padded[i + 1] = ids[i].toLong()

        val tokensTensor = OnnxTensor.createTensor(
            env, LongBuffer.wrap(padded), longArrayOf(1L, padded.size.toLong())
        )
        val styleVec = pack.styleFor(ids.size)
        val styleTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(styleVec), longArrayOf(1L, styleVec.size.toLong())
        )
        val speedTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(floatArrayOf(speed)), longArrayOf(1L)
        )

        val audio: FloatArray = try {
            val outputs = s.run(
                mapOf(
                    inputTokensName to tokensTensor,
                    inputStyleName to styleTensor,
                    inputSpeedName to speedTensor,
                )
            )
            outputs.use {
                val first = it[0]
                val v = first.value
                when (v) {
                    is FloatArray -> v
                    is Array<*> -> {
                        // (1, N) — flatten the inner FloatArray.
                        @Suppress("UNCHECKED_CAST")
                        ((v[0] as FloatArray))
                    }
                    else -> throw IllegalStateException("unexpected ONNX output type: ${v::class}")
                }
            }
        } finally {
            tokensTensor.close(); styleTensor.close(); speedTensor.close()
        }

        // Kokoro outputs at 24 kHz; we want 16 kHz mono Int16 frames.
        val pcm16k = resampleLinear(audio, 24_000, MimoBotIds.SAMPLE_RATE_HZ)
        val frame = MimoBotIds.FRAME_SAMPLES
        var i = 0
        while (i + frame <= pcm16k.size) {
            emit(pcm16k.copyOfRange(i, i + frame))
            i += frame
        }
        if (i < pcm16k.size) {
            val tail = ShortArray(frame)
            System.arraycopy(pcm16k, i, tail, 0, pcm16k.size - i)
            emit(tail)
        }
    }.flowOn(Dispatchers.IO)

    override fun stop() { /* one-shot synth — nothing to interrupt mid-run */ }

    override fun close() {
        try { session?.close() } catch (_: Throwable) {}
        // Don't close OrtEnvironment — it's process-wide singleton and other
        // services use it.
        session = null
        ortEnv = null
        voicePack = null
    }

    /** Linear-interp resample float32 → Int16 at the target rate. */
    private fun resampleLinear(input: FloatArray, srcRate: Int, dstRate: Int): ShortArray {
        if (input.isEmpty()) return ShortArray(0)
        if (srcRate == dstRate) {
            val out = ShortArray(input.size)
            for (i in input.indices) out[i] = (input[i].coerceIn(-1f, 1f) * 32767f).toInt().toShort()
            return out
        }
        val ratio = srcRate.toDouble() / dstRate.toDouble()
        val outLen = (input.size / ratio).toInt()
        val out = ShortArray(outLen)
        var srcPos = 0.0
        for (i in 0 until outLen) {
            val idx = srcPos.toInt()
            val frac = (srcPos - idx).toFloat()
            val s0 = input[idx]
            val s1 = if (idx + 1 < input.size) input[idx + 1] else s0
            val sample = (s0 + (s1 - s0) * frac).coerceIn(-1f, 1f)
            out[i] = (sample * 32767f).toInt().toShort()
        }
        return out
    }

    companion object { private const val TAG = "MimoKokoroTts" }
}
