package com.llmhub.llmhub.data

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
import com.llmhub.llmhub.data.localFileName

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
        val modelFile = File(modelsDir, model.localFileName())
        val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")
        if (!modelFile.exists() && legacyFile.exists()) {
            legacyFile.renameTo(modelFile)
        }

        // Check for partial file
        var downloadedBytes = if (modelFile.exists()) modelFile.length() else 0L
        val totalBytes = model.sizeBytes
        val safeTotal = if (totalBytes <= 0) Long.MAX_VALUE else totalBytes

        // If file exists and is complete, emit complete status
        if (modelFile.exists() && modelFile.length() == safeTotal) {
            emit(DownloadStatus(safeTotal, safeTotal, 0))
            Log.i(TAG, "Model already fully downloaded: ${model.name}")
            return@flow
        }

        emit(DownloadStatus(downloadedBytes, safeTotal, 0))
        Log.d(TAG, "Start downloading ${model.name} from byte $downloadedBytes")

        val url = URL(model.url)
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 15_000
            readTimeout = 60_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            if (!hfToken.isNullOrBlank()) {
                Log.d(TAG, "[downloadModel] Setting Authorization header: Bearer "+hfToken.take(8)+"...")
                setRequestProperty("Authorization", "Bearer $hfToken")
            } else {
                Log.d(TAG, "[downloadModel] No HF token provided, not setting Authorization header.")
            }
            // If resuming, set Range header
            if (downloadedBytes > 0) {
                setRequestProperty("Range", "bytes=$downloadedBytes-")
            }
        }

        val responseCode = connection.responseCode
        if (responseCode !in 200..299 && responseCode != 206) {
            connection.disconnect()
            throw RuntimeException("Download failed with HTTP $responseCode at final URL ${connection.url}")
        }

        Log.i(TAG, "Connected. HTTP $responseCode, final URL: ${connection.url}")

        var lastEmitTime = System.currentTimeMillis()
        var bytesSinceLastEmit = 0L
        var lastSpeed = 0L

        try {
            connection.inputStream.use { input ->
                // If resuming, open output in append mode
                modelFile.outputStream().use { output ->
                    if (downloadedBytes > 0) {
                        output.channel.position(downloadedBytes)
                    }
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        val read = input.read(buffer)
                        if (read == -1) break
                        output.write(buffer, 0, read)
                        downloadedBytes += read
                        bytesSinceLastEmit += read

                        val currentTime = System.currentTimeMillis()
                        val elapsedTime = currentTime - lastEmitTime
                        val shouldEmit = if (downloadedBytes < 10_000_000) {
                            elapsedTime > 250
                        } else {
                            elapsedTime > 1000
                        }
                        if (shouldEmit) {
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
                            Log.d(TAG, "Final progress ${downloadedBytes}/${safeTotal} bytes. Speed ${lastSpeed} B/s")
                            break
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Download error for ${model.name}: ${e.message}", e)
            throw e
        } finally {
            connection.disconnect()
        }

        Log.i(TAG, "Finished downloading ${model.name}. Total bytes written: $downloadedBytes")
    }.flowOn(Dispatchers.IO)
} 