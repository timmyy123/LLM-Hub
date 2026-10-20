package com.llmhub.llmhub.embedding

import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.util.Log
import com.google.ai.edge.litertlm.InputData
import com.llmhub.llmhub.BuildConfig
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.mediasearch.AUDIO_SAMPLE_RATE
import com.llmhub.llmhub.mediasearch.pcm16Wav
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.concurrent.thread
import kotlin.coroutines.resume
import kotlin.math.PI
import kotlin.math.sin

/**
 * Checks EmbeddingGemma 2 NPU builds in the separate `:npu_probe` process before the app loads them.
 *
 * Vendor NPU libraries can crash the process instead of returning an error: on Samsung's MediaTek firmware the
 * NeuroPilot neuron service is not available to apps and NeuronAdapter dereferences null during LiteRT-LM engine
 * creation. A crash there only ends the probe process, and the model is then reported as unusable on this device.
 */
object EmbeddingNpuProbe {
    private const val TAG = "EmbeddingNpuProbe"
    private const val PREFS = "embedding_npu_probe"
    private const val TIMEOUT_MS = 180_000L
    internal const val KEY_MODEL = "model"
    internal const val KEY_BACKEND = "backend"
    internal const val KEY_MEDIA = "media"
    internal const val KEY_VISION_TOKENS = "visionTokens"
    internal const val KEY_MAX_INPUT = "maxInput"
    private val mutex = Mutex()

    /**
     * The first of [backends] that initializes and embeds in the probe process, or null when none does. Results
     * are cached per model file, configuration, system build and app version.
     */
    suspend fun verifiedBackend(
        context: Context,
        model: LLMModel,
        modelFile: File,
        backends: List<String>,
        enableMedia: Boolean,
        visionTokensPerImage: Int?,
        maxInputLength: Int?
    ): String? = mutex.withLock {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val key = "${deviceKey(modelFile)}|m$enableMedia|v${visionTokensPerImage ?: 0}|t${maxInputLength ?: 0}"
        prefs.getString(key, null)?.let { return@withLock it.ifEmpty { null } }

        val verified = backends.firstOrNull { backend ->
            val args = Bundle().apply {
                putString(KEY_MODEL, model.name)
                putString(KEY_BACKEND, backend)
                putBoolean(KEY_MEDIA, enableMedia)
                putInt(KEY_VISION_TOKENS, visionTokensPerImage ?: 0)
                putInt(KEY_MAX_INPUT, maxInputLength ?: 0)
            }
            (probe(context.applicationContext, args) == true).also { Log.i(TAG, "${model.name} with $backend: ${if (it) "works" else "failed"}") }
        }
        prefs.edit().putString(key, verified.orEmpty()).apply()
        verified
    }

    /** True when a probe already found that this device's NPU can't run the model in [modelFile]. */
    fun knownUnsupported(context: Context, modelFile: File): Boolean {
        val prefix = deviceKey(modelFile) + "|"
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).all.any { (key, value) -> key.startsWith(prefix) && value == "" }
    }

    private fun deviceKey(modelFile: File) =
        "${modelFile.name}|${modelFile.length()}|${Build.FINGERPRINT}|${BuildConfig.VERSION_CODE}"

    /** True or false from the probe process, or null when it crashed, hung or could not start. */
    private suspend fun probe(context: Context, args: Bundle): Boolean? = withTimeoutOrNull(TIMEOUT_MS) {
        suspendCancellableCoroutine { continuation ->
            var connection: ServiceConnection? = null
            fun finish(result: Boolean?) {
                if (!continuation.isActive) return
                continuation.resume(result)
                connection?.let { runCatching { context.unbindService(it) } }
            }
            val replies = Messenger(Handler(Looper.getMainLooper()) { message -> finish(message.arg1 == 1); true })
            connection = object : ServiceConnection {
                override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                    runCatching { Messenger(binder).send(Message.obtain().apply { data = args; replyTo = replies }) }
                        .onFailure { finish(null) }
                }
                override fun onServiceDisconnected(name: ComponentName) {
                    Log.w(TAG, "Probe process died while loading ${args.getString(KEY_MODEL)} with ${args.getString(KEY_BACKEND)}")
                    finish(null)
                }
                override fun onBindingDied(name: ComponentName) = finish(null)
                override fun onNullBinding(name: ComponentName) = finish(null)
            }
            continuation.invokeOnCancellation { runCatching { context.unbindService(connection) } }
            if (!context.bindService(Intent(context, EmbeddingNpuProbeService::class.java), connection, Context.BIND_AUTO_CREATE)) {
                finish(null)
            }
        }
    }
}

/** Runs one EmbeddingGemma 2 NPU configuration for [EmbeddingNpuProbe] in the `:npu_probe` process. */
class EmbeddingNpuProbeService : Service() {
    override fun onBind(intent: Intent): IBinder = Messenger(Handler(Looper.getMainLooper()) { message ->
        val replyTo = message.replyTo
        val args = Bundle(message.data)
        thread(name = "npu-probe") {
            val ok = runCatching { runBlocking { probe(args) } }.getOrDefault(false)
            runCatching { replyTo.send(Message.obtain().apply { arg1 = if (ok) 1 else 0 }) }
        }
        true
    }).binder

    override fun onDestroy() {
        super.onDestroy()
        // The process exists only for the probe; exiting releases the NPU runtime and model memory.
        Process.killProcess(Process.myPid())
    }

    private suspend fun probe(args: Bundle): Boolean {
        val model = ModelData.embeddingGemma2Models.firstOrNull { it.name == args.getString(EmbeddingNpuProbe.KEY_MODEL) } ?: return false
        val service = LiteRtLmEmbeddingService(
            this, model,
            enableMedia = args.getBoolean(EmbeddingNpuProbe.KEY_MEDIA),
            visionTokensPerImage = args.getInt(EmbeddingNpuProbe.KEY_VISION_TOKENS).takeIf { it > 0 },
            maxInputLength = args.getInt(EmbeddingNpuProbe.KEY_MAX_INPUT).takeIf { it > 0 },
            npuBackend = args.getString(EmbeddingNpuProbe.KEY_BACKEND)
        )
        return try {
            // initialize() embeds a text probe; media runs the vision and audio encoders the way Video Moment Finder does.
            service.initialize() && (!service.supportsImageEmbedding || service.generateMixedEmbedding(sampleWindow(service.supportsAudioEmbedding)) != null)
        } finally {
            service.close()
        }
    }

    private fun sampleWindow(withAudio: Boolean): List<InputData> {
        val bitmap = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        for (y in 0 until 64) for (x in 0 until 64) bitmap.setPixel(x, y, Color.rgb(x * 4, y * 4, 128))
        val jpeg = ByteArrayOutputStream().also { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }.toByteArray()
        bitmap.recycle()
        if (!withAudio) return listOf(InputData.Image(jpeg))
        val tone = FloatArray(AUDIO_SAMPLE_RATE) { 0.3f * sin(2 * PI * 440 * it / AUDIO_SAMPLE_RATE).toFloat() }
        return listOf(InputData.Text("00:00"), InputData.Audio(pcm16Wav(tone, 0, tone.size)), InputData.Text("00:00"), InputData.Image(jpeg))
    }
}
