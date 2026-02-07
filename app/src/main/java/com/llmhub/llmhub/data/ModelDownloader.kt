package com.llmhub.llmhub.data

import io.ktor.client.* // kept for potential future use but NOT used for large downloads
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
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
import org.json.JSONObject

private const val TAG = "ModelDownloader"

data class DownloadStatus(
    val downloadedBytes: Long,
    val totalBytes: Long,
    val downloadSpeedBytesPerSec: Long,
    val isExtracting: Boolean = false
)

class ModelDownloader(
    private val client: HttpClient,
    private val context: android.content.Context,
    private val hfToken: String? = null // optional Hugging Face access token
) {

    fun downloadModel(model: LLMModel): Flow<DownloadStatus> = flow {
        Log.i(TAG, "Preparing to download model: ${model.name} from ${model.url}")
        
        // Handle ONNX models with additional files (tokenizer, data files)
        if (model.modelFormat == "onnx" && model.additionalFiles.isNotEmpty()) {
            downloadOnnxModel(model).collect { emit(it) }
            return@flow
        }
        
        // Handle image_generator models specially (multi-file format)
        if (model.modelFormat == "image_generator") {
            downloadImageGeneratorModel(model).collect { emit(it) }
            return@flow
        }
        
        // Handle stable_diffusion models (ZIP extraction) - check by category
        if (model.category == "image_generation") {
            Log.i(TAG, "Routing to Stable Diffusion downloader for: ${model.name}")
            downloadStableDiffusionModel(model).collect { emit(it) }
            return@flow
        }
        
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
        
        // HTTP 416 = "Range Not Satisfiable" - file is already complete
        if (responseCode == 416) {
            connection.disconnect()
            Log.i(TAG, "HTTP 416: File already complete at $downloadedBytes bytes")
            emit(DownloadStatus(downloadedBytes, downloadedBytes, 0))
            return@flow
        }
        
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
                    
                    // Emit final progress after download completes naturally
                    val finalTotalBytes = if (inferredTotalBytes > 0) maxOf(inferredTotalBytes, downloadedBytes) else downloadedBytes
                    emit(DownloadStatus(downloadedBytes, finalTotalBytes, 0))
                    Log.d(TAG, "Download completed naturally. Final progress ${downloadedBytes}/${finalTotalBytes} bytes.")
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
    
    /**
     * Downloads image generator models (multi-file format with manifest.json)
     */
    private fun downloadImageGeneratorModel(model: LLMModel): Flow<DownloadStatus> = flow {
        Log.i(TAG, "Downloading image generator model: ${model.name}")
        
        // Target directory: app's files dir + image_generator/bins (app has write access)
        val targetDir = File(context.filesDir, "image_generator/bins")
        if (!targetDir.exists()) {
            targetDir.mkdirs()
            Log.d(TAG, "Created directory: ${targetDir.absolutePath}")
        }
        
        // Step 1: Download manifest.json
        Log.d(TAG, "Downloading manifest from: ${model.url}")
        val manifestUrl = URL(model.url)
        val connection = (manifestUrl.openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 15_000
            readTimeout = 30_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "Mozilla/5.0")
            if (!hfToken.isNullOrBlank()) {
                setRequestProperty("Authorization", "Bearer $hfToken")
            }
        }
        
        val manifestJson = connection.inputStream.bufferedReader().use { it.readText() }
        connection.disconnect()
        
        val manifest = JSONObject(manifestJson)
        val filesArray = manifest.getJSONArray("files")
        val baseUrl = manifest.optString("base_url", model.url.substringBeforeLast("/") + "/")
        val totalFiles = filesArray.length()
        
        Log.i(TAG, "Manifest loaded: $totalFiles files to download")
        
        // Check how many files already exist
        var filesDownloaded = 0
        var totalDownloaded = 0L
        for (i in 0 until totalFiles) {
            val fileName = filesArray.getString(i)
            val targetFile = File(targetDir, fileName)
            if (targetFile.exists() && targetFile.length() > 0) {
                filesDownloaded++
                totalDownloaded += targetFile.length()
            }
        }
        
        // If all files exist, we're done
        if (filesDownloaded == totalFiles) {
            Log.i(TAG, "All $totalFiles files already downloaded! Total size: $totalDownloaded bytes")
            emit(DownloadStatus(totalDownloaded, totalDownloaded, 0))
            return@flow
        }
        
        Log.i(TAG, "Found $filesDownloaded/$totalFiles existing files, downloading remaining...")
        
        // Step 2: Download each missing file
        val totalSize = model.sizeBytes
        var lastEmitTime = System.currentTimeMillis()
        var bytesSinceLastEmit = 0L
        
        for (i in 0 until totalFiles) {
            val fileName = filesArray.getString(i)
            val targetFile = File(targetDir, fileName)
            
            // Skip if file already exists and is non-empty
            if (targetFile.exists() && targetFile.length() > 0) {
                Log.d(TAG, "Skipping existing file ($filesDownloaded/$totalFiles): $fileName")
                continue
            }
            
            // Download the file
            filesDownloaded++
            Log.d(TAG, "Downloading ($filesDownloaded/$totalFiles): $fileName")
            val fileUrl = URL(baseUrl + fileName)
            val fileConnection = (fileUrl.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 15_000
                readTimeout = 60_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "Mozilla/5.0")
                if (!hfToken.isNullOrBlank()) {
                    setRequestProperty("Authorization", "Bearer $hfToken")
                }
            }
            
            try {
                fileConnection.inputStream.use { input ->
                    targetFile.outputStream().use { output ->
                        val buffer = ByteArray(8192)
                        var bytesRead: Int
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                            totalDownloaded += bytesRead
                            bytesSinceLastEmit += bytesRead
                            
                            // Emit progress every 500ms
                            val currentTime = System.currentTimeMillis()
                            val elapsed = currentTime - lastEmitTime
                            if (elapsed > 500) {
                                val speed = if (elapsed > 0) (bytesSinceLastEmit * 1000 / elapsed) else 0L
                                emit(DownloadStatus(totalDownloaded, totalSize, speed))
                                lastEmitTime = currentTime
                                bytesSinceLastEmit = 0L
                            }
                        }
                    }
                }
                Log.d(TAG, "Downloaded ($filesDownloaded/$totalFiles): $fileName (${targetFile.length()} bytes)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to download $fileName: ${e.message}")
                throw e
            } finally {
                fileConnection.disconnect()
            }
        }
        
        // Final emit - use actual total downloaded size
        emit(DownloadStatus(totalDownloaded, totalDownloaded, 0))
        Log.i(TAG, "Completed downloading all $totalFiles files for ${model.name}")
    }.flowOn(Dispatchers.IO)
    
    /**
     * Downloads Stable Diffusion models (ZIP format from HuggingFace)
     * Downloads ZIP, then extracts to sd_models directory
     */
    private fun downloadStableDiffusionModel(model: LLMModel): Flow<DownloadStatus> = flow {
        Log.i(TAG, "Downloading Stable Diffusion model: ${model.name}")
        
        // Target directory for extracted model files
        val sdModelsDir = File(context.filesDir, "sd_models")
        if (!sdModelsDir.exists()) {
            sdModelsDir.mkdirs()
            Log.d(TAG, "Created directory: ${sdModelsDir.absolutePath}")
        }
        
        // Model-specific folder for this model
        val modelTargetDir = File(sdModelsDir, model.name.replace(" ", "_"))
        
        // Check if model already extracted by searching recursively for unet files
        val modelType = if (model.name.contains("NPU", ignoreCase = true) || 
                            model.modelFormat.contains("qnn", ignoreCase = true)) "qnn" else "mnn"
        
        fun findUnetFile(dir: File, depth: Int = 0): Boolean {
            if (depth > 6 || !dir.exists() || !dir.isDirectory) return false
            val files = dir.listFiles() ?: return false
            for (f in files) {
                if (f.isFile) {
                    val name = f.name.lowercase()
                    if ((modelType == "qnn" && name.contains("unet") && name.endsWith(".bin")) ||
                        (modelType == "mnn" && name.contains("unet") && name.endsWith(".mnn"))) {
                        return true
                    }
                } else if (f.isDirectory && findUnetFile(f, depth + 1)) {
                    return true
                }
            }
            return false
        }
        
        val modelAlreadyExtracted = modelTargetDir.exists() && findUnetFile(modelTargetDir)
        Log.i(TAG, "Model type detected: $modelType, checking in: ${modelTargetDir.absolutePath}, exists: $modelAlreadyExtracted")
        
        if (modelAlreadyExtracted) {
            Log.i(TAG, "Model already extracted at ${modelTargetDir.absolutePath}")
            emit(DownloadStatus(model.sizeBytes, model.sizeBytes, 0))
            return@flow
        }
        
        // Download ZIP to a persistent temp location so it survives process death and
        // the user can resume after killing/restarting the app.
        val tempDir = File(context.filesDir, "sd_downloads")
        if (!tempDir.exists()) {
            tempDir.mkdirs()
        }
        val zipFile = File(tempDir, "${model.name.replace(" ", "_")}.zip")
        
        // Download the ZIP if not already present
        if (!zipFile.exists() || zipFile.length() < model.sizeBytes * 0.9) {
            Log.d(TAG, "Downloading ZIP from: ${model.url}")
            
            val url = URL(model.url)
            // Support resume: if a partial zip exists, request Range header and append to file
            val existingBytes = if (zipFile.exists()) zipFile.length() else 0L

            val connection = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 15_000
                readTimeout = 60_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "Mozilla/5.0")
                if (!hfToken.isNullOrBlank()) {
                    setRequestProperty("Authorization", "Bearer $hfToken")
                }
                if (existingBytes > 0L) {
                    // Ask server to send remaining bytes
                    setRequestProperty("Range", "bytes=$existingBytes-")
                    Log.d(TAG, "Requesting Range: bytes=$existingBytes-")
                }
            }

            val responseCode = connection.responseCode
            val acceptRanges = connection.getHeaderField("Accept-Ranges")
            val contentRangeHeader = connection.getHeaderField("Content-Range")
            val contentLengthHeader = connection.getHeaderField("Content-Length")
            Log.d(TAG, "ZIP download HTTP response: $responseCode, Accept-Ranges=$acceptRanges, Content-Range=$contentRangeHeader, Content-Length=$contentLengthHeader, localZipSize=$existingBytes")
            if (responseCode !in 200..299 && responseCode != 206) {
                connection.disconnect()
                throw RuntimeException("Download failed with HTTP $responseCode")
            }

            // Compute total bytes: if server returned 206 with Content-Range/Content-Length, account for existing
            val totalBytes = when {
                responseCode == 206 && contentLengthHeader != null -> existingBytes + (contentLengthHeader.toLongOrNull() ?: 0L)
                connection.contentLengthLong > 0L -> connection.contentLengthLong
                else -> model.sizeBytes
            }

            var downloadedBytes = existingBytes
            var lastEmitTime = System.currentTimeMillis()
            var bytesSinceLastEmit = 0L

            try {
                connection.inputStream.use { input ->
                    // Use RandomAccessFile to append to existing partial file when resuming
                    RandomAccessFile(zipFile, "rw").use { raf ->
                        raf.seek(existingBytes)
                        val buffer = ByteArray(8192)
                        var bytesRead: Int
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            raf.write(buffer, 0, bytesRead)
                            downloadedBytes += bytesRead
                            bytesSinceLastEmit += bytesRead

                            val currentTime = System.currentTimeMillis()
                            val elapsed = currentTime - lastEmitTime
                            if (elapsed > 1000) {
                                val speed = if (elapsed > 0) (bytesSinceLastEmit * 1000 / elapsed) else 0L
                                // Report download as 90% of total progress (extraction is remaining 10%)
                                val progressBytes = (downloadedBytes * 0.9).toLong()
                                emit(DownloadStatus(progressBytes, totalBytes, speed))
                                lastEmitTime = currentTime
                                bytesSinceLastEmit = 0L
                            }
                        }
                    }
                }
                Log.i(TAG, "Downloaded ZIP: ${zipFile.length()} bytes")
                Log.d(TAG, "Post-download zip size: ${zipFile.length()} bytes (path=${zipFile.absolutePath})")
            } catch (e: Exception) {
                Log.e(TAG, "Download error: ${e.message}", e)
                throw e
            } finally {
                connection.disconnect()
            }
        } else {
            Log.i(TAG, "Using existing ZIP file: ${zipFile.absolutePath}")
        }
        
        // Clean up model folder if it exists (from a previous failed extraction)
        if (modelTargetDir.exists()) {
            modelTargetDir.deleteRecursively()
        }
        modelTargetDir.mkdirs()
        
        // Extract the ZIP with progress updates
        // Reset progress to 0 for extraction phase (separate from download progress)
        Log.i(TAG, "Extracting model files to ${modelTargetDir.absolutePath}...")
        emit(DownloadStatus(0, model.sizeBytes, 0, isExtracting = true))
        
        // Use atomic reference to track progress from callback
        val extractProgress = java.util.concurrent.atomic.AtomicReference(0f)
        var extractSuccess = false
        
        // Run extraction in a separate coroutine so we can poll progress
        coroutineScope {
            val extractJob = async(Dispatchers.IO) {
                com.llmhub.llmhub.utils.ZipExtractor.extractZip(
                    zipFile = zipFile,
                    targetDir = modelTargetDir,
                    onProgress = { progress ->
                        extractProgress.set(progress)
                    }
                )
            }
            
            // Poll and emit progress while extraction is running
            while (!extractJob.isCompleted) {
                val progress = extractProgress.get()
                val progressBytes = (progress * model.sizeBytes).toLong()
                emit(DownloadStatus(progressBytes, model.sizeBytes, 0, isExtracting = true))
                delay(100)
            }
            
            extractSuccess = extractJob.await()
        }
        
        if (!extractSuccess) {
            throw RuntimeException("Failed to extract model ZIP")
        }
        
        // Clean up ZIP file
        try {
            zipFile.delete()
            Log.d(TAG, "Cleaned up temporary ZIP file")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to delete temp ZIP: ${e.message}")
        }
        
        // Final emit
        emit(DownloadStatus(model.sizeBytes, model.sizeBytes, 0))
        Log.i(TAG, "Completed downloading and extracting ${model.name}")
    }.flowOn(Dispatchers.IO)
    
    /**
     * Downloads ONNX models with additional files (tokenizer.json, data files, etc.)
     * All files are downloaded to the same directory as the main model file.
     */
    private fun downloadOnnxModel(model: LLMModel): Flow<DownloadStatus> = flow {
        Log.i(TAG, "Downloading ONNX model: ${model.name} with ${model.additionalFiles.size} additional files")
        
        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
        }
        
        // Create model-specific subdirectory for ONNX models using sanitized model name
        val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
        val modelDir = File(modelsDir, modelDirName)
        if (!modelDir.exists()) {
            modelDir.mkdirs()
            Log.d(TAG, "Created ONNX model directory: ${modelDir.absolutePath}")
        }
        
        // Helper to extract clean filename from URL (strip query params)
        fun extractFileName(url: String): String {
            return url.substringAfterLast("/").substringBefore("?")
        }
        
        // Build list of all files to download: main model + additional files
        val baseUrl = model.url.substringBefore("?").substringBeforeLast("/") + "/"
        val allFiles = mutableListOf<Pair<String, String>>() // (url, localFileName)
        
        // Add main model file
        val mainFileName = extractFileName(model.url)
        allFiles.add(model.url to mainFileName)
        
        // Add additional files (they use the same base URL)
        for (additionalFile in model.additionalFiles) {
            val fileName = extractFileName(additionalFile)
            val fileUrl = if (additionalFile.startsWith("http")) additionalFile else baseUrl + additionalFile
            allFiles.add(fileUrl to fileName)
        }
        
        Log.i(TAG, "Total files to download: ${allFiles.size}")
        
        // Check which files already exist
        var totalDownloaded = 0L
        var filesComplete = 0
        for ((_, fileName) in allFiles) {
            val file = File(modelDir, fileName)
            if (file.exists() && file.length() > 0) {
                totalDownloaded += file.length()
                filesComplete++
            }
        }
        
        // If all files exist, we're done
        if (filesComplete == allFiles.size) {
            Log.i(TAG, "All ${allFiles.size} ONNX files already downloaded! Total size: $totalDownloaded bytes")
            emit(DownloadStatus(totalDownloaded, totalDownloaded, 0))
            return@flow
        }
        
        Log.i(TAG, "Found $filesComplete/${allFiles.size} existing files, downloading remaining...")
        
        val totalSize = model.sizeBytes
        var lastEmitTime = System.currentTimeMillis()
        var bytesSinceLastEmit = 0L
        var currentFileIndex = 0
        
        for ((fileUrl, fileName) in allFiles) {
            currentFileIndex++
            val targetFile = File(modelDir, fileName)
            
            // Skip if file already exists
            if (targetFile.exists() && targetFile.length() > 0) {
                Log.d(TAG, "Skipping existing file ($currentFileIndex/${allFiles.size}): $fileName")
                continue
            }
            
            Log.d(TAG, "Downloading ($currentFileIndex/${allFiles.size}): $fileName from $fileUrl")
            
            val url = URL(fileUrl)
            val connection = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 15_000
                readTimeout = 60_000
                instanceFollowRedirects = true
                setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                if (!hfToken.isNullOrBlank()) {
                    setRequestProperty("Authorization", "Bearer $hfToken")
                }
            }
            
            val responseCode = connection.responseCode
            if (responseCode !in 200..299) {
                connection.disconnect()
                throw RuntimeException("Download failed for $fileName with HTTP $responseCode")
            }
            
            try {
                connection.inputStream.use { input ->
                    targetFile.outputStream().use { output ->
                        val buffer = ByteArray(8192)
                        var bytesRead: Int
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                            totalDownloaded += bytesRead
                            bytesSinceLastEmit += bytesRead
                            
                            // Emit progress every 500ms
                            val currentTime = System.currentTimeMillis()
                            val elapsed = currentTime - lastEmitTime
                            if (elapsed > 500) {
                                val speed = if (elapsed > 0) (bytesSinceLastEmit * 1000 / elapsed) else 0L
                                emit(DownloadStatus(totalDownloaded, totalSize, speed))
                                lastEmitTime = currentTime
                                bytesSinceLastEmit = 0L
                            }
                        }
                    }
                }
                Log.d(TAG, "Downloaded ($currentFileIndex/${allFiles.size}): $fileName (${targetFile.length()} bytes)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to download $fileName: ${e.message}")
                throw e
            } finally {
                connection.disconnect()
            }
        }
        
        // Final emit
        emit(DownloadStatus(totalDownloaded, totalDownloaded, 0))
        Log.i(TAG, "Completed downloading all ${allFiles.size} files for ONNX model ${model.name}")
    }.flowOn(Dispatchers.IO)
}
