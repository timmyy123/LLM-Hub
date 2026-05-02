package com.llmhub.llmhub.mimobot.speech.kokoro

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Manages on-device Kokoro asset files. Downloads from the canonical
 * onnx-community Hugging Face mirror on first use and caches under
 * `filesDir/kokoro/`.
 *
 * Files (one-time download):
 *   - kokoro-v1.0.fp16.onnx  (≈ 165 MB)
 *   - voices/<voice>.bin     (≈ 512 KB each)
 *
 * Reuses OkHttp which is already a project dependency.
 */
object KokoroAssets {

    private const val TAG = "KokoroAssets"

    // The fp16 export is small enough for phones, fast enough on NNAPI EP.
    private const val MODEL_URL =
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_fp16.onnx"

    private fun voiceUrl(voiceId: String) =
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/$voiceId.bin"

    private val http: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(20, TimeUnit.MINUTES)   // big model download
            .writeTimeout(2, TimeUnit.MINUTES)
            .build()
    }

    fun root(context: Context): File =
        File(context.filesDir, "kokoro").also { it.mkdirs() }

    fun modelFile(context: Context): File = File(root(context), "model_fp16.onnx")
    fun voiceFile(context: Context, voiceId: String): File =
        File(File(root(context), "voices").also { it.mkdirs() }, "$voiceId.bin")

    fun isReady(context: Context, voiceId: String): Boolean =
        modelFile(context).exists() && voiceFile(context, voiceId).exists()

    /** Progress event emitted during download. [bytesDone] / [bytesTotal] in bytes; -1 unknown. */
    data class Progress(
        val stage: String,
        val bytesDone: Long,
        val bytesTotal: Long,
    )

    /**
     * Download model + voice if missing. Emits one [Progress] per ~1 MB of data
     * plus stage transitions. Caller can collect this Flow to drive a UI.
     */
    fun ensure(context: Context, voiceId: String = "af_heart"): Flow<Progress> = flow {
        val mf = modelFile(context)
        val vf = voiceFile(context, voiceId)

        if (!mf.exists()) {
            emit(Progress("Downloading Kokoro model", 0, -1))
            download(MODEL_URL, mf) { done, total ->
                emit(Progress("Downloading Kokoro model", done, total))
            }
        }
        if (!vf.exists()) {
            emit(Progress("Downloading voice $voiceId", 0, -1))
            download(voiceUrl(voiceId), vf) { done, total ->
                emit(Progress("Downloading voice $voiceId", done, total))
            }
        }
        emit(Progress("Ready", 1, 1))
    }.flowOn(Dispatchers.IO)

    private suspend fun download(
        url: String,
        target: File,
        onProgress: suspend (done: Long, total: Long) -> Unit,
    ) {
        val req = Request.Builder().url(url).build()
        val tmp = File(target.parentFile, target.name + ".part")
        try {
            http.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) throw IllegalStateException("HTTP ${resp.code} for $url")
                val body = resp.body ?: throw IllegalStateException("empty body for $url")
                val total = body.contentLength()
                var done = 0L
                var lastEmit = 0L
                tmp.outputStream().use { out ->
                    val src = body.source()
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = src.read(buf)
                        if (n <= 0) break
                        out.write(buf, 0, n)
                        done += n
                        if (done - lastEmit >= 1_000_000L) {
                            onProgress(done, total)
                            lastEmit = done
                        }
                    }
                }
            }
            if (!tmp.renameTo(target)) {
                tmp.copyTo(target, overwrite = true)
                tmp.delete()
            }
            Log.i(TAG, "downloaded ${target.absolutePath} (${target.length()} bytes)")
        } catch (t: Throwable) {
            tmp.delete()
            throw t
        }
    }
}
