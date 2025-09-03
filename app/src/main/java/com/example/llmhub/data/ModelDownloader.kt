package com.llmhub.llmhub.data

import io.ktor.client.* // kept for potential future use but NOT used for large downloads
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import android.util.Log
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.data.isModelFileValid

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
        var inferredTotalBytes = if (model.sizeBytes > 0) model.sizeBytes else -1L

        Log.d(TAG, "File check for ${model.name}: exists=${modelFile.exists()}, size=${if (modelFile.exists()) modelFile.length() else 0}, expectedSize=$inferredTotalBytes, path=${modelFile.absolutePath}")

        // If file exists and we know the exact size and it's complete, short-circuit
        if (modelFile.exists() && inferredTotalBytes > 0 && modelFile.length() >= inferredTotalBytes) {
            // Double-check that the file is actually valid by checking if it's not corrupted
            val fileIsValid = try {
                isModelFileValid(modelFile, model.modelFormat)
            } catch (e: Exception) {
                Log.w(TAG, "Error validating model file ${modelFile.absolutePath}: ${e.message}")
                false
            }
            
            if (fileIsValid) {
                emit(DownloadStatus(inferredTotalBytes, inferredTotalBytes, 0))
                Log.i(TAG, "Model already fully downloaded and validated: ${model.name}")
                return@flow
            } else {
                Log.w(TAG, "Model file exists but is invalid, redownloading: ${model.name}")
                // Delete the invalid file and continue with download
                modelFile.delete()
                downloadedBytes = 0L
            }
        }

        // For unknown totals, emit 0 so UI shows indeterminate
        emit(DownloadStatus(downloadedBytes, if (inferredTotalBytes > 0) inferredTotalBytes else 0L, 0))
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

        // Try to infer total size from headers
        try {
            val contentRange = connection.getHeaderField("Content-Range") // bytes start-end/total
            val contentLength = connection.getHeaderField("Content-Length")?.toLongOrNull()
            if (contentRange != null) {
                val rangePart = contentRange.substringAfter(' ').substringBefore('/') // e.g. bytes 100-999
                val start = rangePart.substringBefore('-').toLongOrNull()
                val total = contentRange.substringAfter('/').toLongOrNull()
                if (total != null && total > 0) inferredTotalBytes = total
                if (start != null && start >= 0 && start != downloadedBytes) {
                    Log.w(TAG, "Server resumed at $start but local file is $downloadedBytes. Truncating to $start.")
                    // Truncate local file to server start
                    if (modelFile.exists()) {
                        RandomAccessFile(modelFile, "rw").use { raf ->
                            raf.setLength(start)
                        }
                    }
                    downloadedBytes = start
                }
            } else if (contentLength != null && contentLength > 0) {
                inferredTotalBytes = if (downloadedBytes > 0 && responseCode == 206) downloadedBytes + contentLength else contentLength
            }
        } catch (_: Exception) { /* ignore */ }

        // If we attempted resume but got 200 (no Range support), restart from 0
        if (downloadedBytes > 0 && responseCode == 200) {
            Log.w(TAG, "Server ignored Range header. Restarting full download and overwriting partial file.")
            if (modelFile.exists()) modelFile.delete()
            modelFile.parentFile?.mkdirs()
            modelFile.createNewFile()
            downloadedBytes = 0L
            emit(DownloadStatus(0, if (inferredTotalBytes > 0) inferredTotalBytes else 0L, 0))
        }

        Log.i(TAG, "Connected. HTTP $responseCode, final URL: ${connection.url}, total=${inferredTotalBytes}")

        var lastEmitTime = System.currentTimeMillis()
        var bytesSinceLastEmit = 0L
        var lastSpeed = 0L

        try {
            connection.inputStream.use { input ->
                // Open output and position at downloadedBytes
                RandomAccessFile(modelFile, "rw").use { raf ->
                    raf.seek(downloadedBytes)
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        val read = input.read(buffer)
                        if (read == -1) break
                        raf.write(buffer, 0, read)
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
                            val speed = if (elapsedTime > 0) (bytesSinceLastEmit * 1000 / elapsedTime) else 0L
                            emit(DownloadStatus(downloadedBytes, if (inferredTotalBytes > 0) inferredTotalBytes else 0L, speed))
                            Log.d(TAG, "Progress ${downloadedBytes}/${inferredTotalBytes} bytes. Speed ${speed} B/s")
                            lastEmitTime = currentTime
                            bytesSinceLastEmit = 0L
                        }
                        if (inferredTotalBytes > 0 && downloadedBytes >= inferredTotalBytes) {
                            emit(DownloadStatus(downloadedBytes, inferredTotalBytes, (bytesSinceLastEmit)))
                            Log.d(TAG, "Final progress ${downloadedBytes}/${inferredTotalBytes} bytes.")
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