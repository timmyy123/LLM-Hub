package com.llmhub.llmhub.imagegen

import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.os.RemoteException
import android.util.Log
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import java.util.concurrent.Executors
import kotlin.coroutines.resume

private const val TAG = "LiteRtSd"
private const val MSG_LOAD = 1
private const val MSG_GENERATE = 2
private const val MSG_PROGRESS = 3
private const val MSG_DONE = 4
private const val KEY_MODEL_DIR = "modelDir"
private const val KEY_PROMPT = "prompt"
private const val KEY_NEGATIVE = "negative"
private const val KEY_STEPS = "steps"
private const val KEY_GUIDANCE = "guidance"
private const val KEY_SEED = "seed"
private const val KEY_OUTPUT = "output"

/**
 * Runs [LiteRtSdPipeline] in the `:litert_sd` process. Vendor NPU drivers can abort instead of returning an error
 * (Google Tensor G5's TPU driver does on the first UNet step with current Pixel 10 firmware), and here that fails
 * one generation instead of closing the app.
 */
class LiteRtSdService : Service() {
    private val worker = Executors.newSingleThreadExecutor()
    private var pipeline: LiteRtSdPipeline? = null
    private var modelDir: String? = null

    private val messenger = Messenger(Handler(Looper.getMainLooper()) { message ->
        val what = message.what
        val data = Bundle(message.data)
        val replyTo = message.replyTo
        worker.execute { handle(what, data, replyTo) }
        true
    })

    override fun onBind(intent: Intent): IBinder = messenger.binder

    override fun onDestroy() {
        super.onDestroy()
        // The process exists only for this pipeline; exiting releases the NPU runtime and model memory.
        Process.killProcess(Process.myPid())
    }

    private fun handle(what: Int, data: Bundle, replyTo: Messenger) {
        val ok = try {
            when (what) {
                MSG_LOAD -> load(data.getString(KEY_MODEL_DIR).orEmpty())
                MSG_GENERATE -> generate(data, replyTo)
                else -> false
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Request $what failed", e)
            false
        }
        send(replyTo, MSG_DONE, if (ok) 1 else 0)
    }

    private fun load(dir: String): Boolean {
        if (dir == modelDir && pipeline != null) return true
        pipeline?.close()
        pipeline = null
        pipeline = LiteRtSdPipeline(this, File(dir))
        modelDir = dir
        return true
    }

    private fun generate(data: Bundle, replyTo: Messenger): Boolean {
        val pipeline = pipeline ?: return false
        val image = pipeline.generate(
            data.getString(KEY_PROMPT).orEmpty(),
            data.getString(KEY_NEGATIVE).orEmpty(),
            data.getInt(KEY_STEPS),
            data.getFloat(KEY_GUIDANCE),
            data.getLong(KEY_SEED)
        ) { step, total -> send(replyTo, MSG_PROGRESS, step, total) }
        File(data.getString(KEY_OUTPUT).orEmpty()).outputStream().use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
        return true
    }

    private fun send(target: Messenger, what: Int, arg1: Int = 0, arg2: Int = 0) {
        try {
            target.send(Message.obtain(null, what, arg1, arg2))
        } catch (e: RemoteException) {
            Log.w(TAG, "Client is gone: ${e.message}")
        }
    }
}

/** The app-process side of [LiteRtSdService]: one bound pipeline process, one request at a time. */
internal class LiteRtSdClient(context: Context) {
    private val app = context.applicationContext
    private val mutex = Mutex()
    @Volatile private var connection: ServiceConnection? = null
    @Volatile private var service: Messenger? = null
    @Volatile private var pending: CompletableDeferred<Boolean>? = null
    @Volatile private var onProgress: ((Int, Int) -> Unit)? = null
    private var modelDir: File? = null

    private val replies = Messenger(Handler(Looper.getMainLooper()) { message ->
        when (message.what) {
            MSG_PROGRESS -> onProgress?.invoke(message.arg1, message.arg2)
            MSG_DONE -> pending?.complete(message.arg1 == 1)
        }
        true
    })

    /** True when the pipeline process died or hung during the last request, e.g. in a vendor NPU driver. */
    @Volatile var crashed = false
        private set

    suspend fun load(dir: File): Boolean {
        modelDir = dir
        return request(MSG_LOAD, Bundle().apply { putString(KEY_MODEL_DIR, dir.path) }, LOAD_TIMEOUT_MS)
    }

    /** Writes a 512x512 PNG to [output]. A crashed pipeline process is started and loaded again first. */
    suspend fun generate(
        prompt: String,
        negativePrompt: String,
        steps: Int,
        guidance: Float,
        seed: Long,
        output: File,
        progress: (step: Int, total: Int) -> Unit = { _, _ -> }
    ): Boolean {
        if (service == null && !load(modelDir ?: return false)) return false
        val data = Bundle().apply {
            putString(KEY_PROMPT, prompt)
            putString(KEY_NEGATIVE, negativePrompt)
            putInt(KEY_STEPS, steps)
            putFloat(KEY_GUIDANCE, guidance)
            putLong(KEY_SEED, seed)
            putString(KEY_OUTPUT, output.path)
        }
        return request(MSG_GENERATE, data, GENERATE_TIMEOUT_MS, progress)
    }

    private suspend fun request(what: Int, data: Bundle, timeoutMs: Long, progress: ((Int, Int) -> Unit)? = null): Boolean = mutex.withLock {
        crashed = false
        val target = service ?: bind() ?: return@withLock false
        val result = CompletableDeferred<Boolean>()
        pending = result
        onProgress = progress
        try {
            target.send(Message.obtain(null, what).apply { this.data = data; replyTo = replies })
            withTimeoutOrNull(timeoutMs) { result.await() } ?: run {
                Log.e(TAG, "Pipeline process did not answer request $what within ${timeoutMs / 1000} s")
                crashed = true
                close()
                false
            }
        } catch (e: RemoteException) {
            false
        } finally {
            pending = null
            onProgress = null
        }
    }

    private suspend fun bind(): Messenger? = suspendCancellableCoroutine { continuation ->
        fun disconnected() {
            // Stay unbound, so Android does not restart a pipeline that keeps crashing.
            crashed = true
            close()
            pending?.complete(false)
            if (continuation.isActive) continuation.resume(null)
        }
        val newConnection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                val messenger = Messenger(binder)
                service = messenger
                if (continuation.isActive) continuation.resume(messenger)
            }
            override fun onServiceDisconnected(name: ComponentName) {
                Log.e(TAG, "Pipeline process died")
                disconnected()
            }
            override fun onBindingDied(name: ComponentName) = disconnected()
            override fun onNullBinding(name: ComponentName) = disconnected()
        }
        connection = newConnection
        continuation.invokeOnCancellation { close() }
        if (!app.bindService(Intent(app, LiteRtSdService::class.java), newConnection, Context.BIND_AUTO_CREATE)) {
            connection = null
            continuation.resume(null)
        }
    }

    /** Unbinds, which ends the pipeline process. */
    fun close() {
        connection?.let { runCatching { app.unbindService(it) } }
        connection = null
        service = null
    }

    private companion object {
        const val LOAD_TIMEOUT_MS = 3 * 60_000L
        const val GENERATE_TIMEOUT_MS = 15 * 60_000L
    }
}
