package com.example.llmhub.data

import io.ktor.client.* // kept for potential future use but NOT used for large downloads
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.roundToLong
import android.util.Log

private const val TAG = "ModelDownloader"

data class DownloadStatus(
    val downloadedBytes: Long,
    val totalBytes: Long,
    val downloadSpeedBytesPerSec: Long
)

class ModelDownloader(
    private val client: HttpClient,
    private val context: android.content.Context,
    private val hfToken: String? = null // optional Hugging Face access token
) {

    fun downloadModel(model: LLMModel): Flow<DownloadStatus> = flow {
        Log.i(TAG, "Preparing to download model: ${model.name} from ${model.url}")
        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
        }
        val modelFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")
        // Remove any existing file so we don't mistakenly think a previous partial file is complete
        if (modelFile.exists()) {
            Log.w(TAG, "Existing file found for ${model.name}. Deleting before re-download.")
            modelFile.delete()
        }

        val url = URL(model.url)
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 15_000
            readTimeout = 60_000
            // Resume with system's default buffering; we stream manually.
            if (!hfToken.isNullOrBlank()) {
                setRequestProperty("Authorization", "Bearer $hfToken")
            }
        }

        val responseCode = connection.responseCode
        if (responseCode !in 200..299) {
            connection.disconnect()
            throw RuntimeException("Download failed with HTTP $responseCode")
        }

        Log.i(TAG, "Connected. HTTP $responseCode")

        val totalBytes = connection.contentLengthLong.takeIf { it > 0 } ?: model.sizeBytes
        Log.i(TAG, "Total bytes (Content-Length or fallback): $totalBytes")

        val safeTotal = if (totalBytes <= 0) Long.MAX_VALUE else totalBytes
        var downloadedBytes = 0L
        var lastEmitTime = System.currentTimeMillis()
        var bytesSinceLastEmit = 0L
        var lastSpeed = 0L

        emit(DownloadStatus(downloadedBytes, safeTotal, 0))
        Log.d(TAG, "Start downloading ${model.name}")

        connection.inputStream.use { input ->
            modelFile.outputStream().use { output ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val read = input.read(buffer)
                    if (read == -1) break
                    output.write(buffer, 0, read)

                    downloadedBytes += read
                    bytesSinceLastEmit += read

                    val currentTime = System.currentTimeMillis()
                    val elapsedTime = currentTime - lastEmitTime
                    if (elapsedTime > 250) {
                        val computed = if (elapsedTime > 0) (bytesSinceLastEmit * 1000 / elapsedTime) else 0L
                        if (computed > 0) lastSpeed = computed
                        val speed = if (lastSpeed > 0) lastSpeed else computed
                        emit(DownloadStatus(downloadedBytes, safeTotal, speed))
                        Log.d(TAG, "Progress ${downloadedBytes}/${safeTotal} bytes. Speed ${speed} B/s")
                        lastEmitTime = currentTime
                        bytesSinceLastEmit = 0L
                    }

                    if (safeTotal != Long.MAX_VALUE && downloadedBytes >= safeTotal) {
                        emit(DownloadStatus(downloadedBytes, safeTotal, lastSpeed))
                        Log.d(TAG, "Progress ${downloadedBytes}/${safeTotal} bytes. Speed ${lastSpeed} B/s")
                    }
                }
            }
        }

        Log.i(TAG, "Finished downloading ${model.name}. Total bytes written: $downloadedBytes")
    }.flowOn(Dispatchers.IO)
} 