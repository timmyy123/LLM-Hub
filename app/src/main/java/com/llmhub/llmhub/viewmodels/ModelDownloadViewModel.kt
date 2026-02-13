package com.llmhub.llmhub.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import com.llmhub.llmhub.data.ModelDownloader
import io.ktor.client.HttpClient
import io.ktor.client.engine.android.Android
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.onCompletion
import android.app.Application
import androidx.lifecycle.AndroidViewModel
import java.io.File
import kotlinx.coroutines.Dispatchers
import com.llmhub.llmhub.data.localFileName
import android.content.Context
import com.llmhub.llmhub.BuildConfig
import com.llmhub.llmhub.data.isModelFileValid
import com.google.gson.Gson
import android.net.Uri

class ModelDownloadViewModel(application: Application) : AndroidViewModel(application) {
    private val _models = MutableStateFlow<List<LLMModel>>(emptyList())
    val models: StateFlow<List<LLMModel>> = _models.asStateFlow()
    
    private val _hfToken = MutableStateFlow<String?>(null)
    val hfToken: StateFlow<String?> = _hfToken.asStateFlow()

    private val ktorClient = HttpClient(Android)
    private val context = application.applicationContext
    private var modelDownloader: ModelDownloader

    private var lastProgressMap: MutableMap<String, Pair<Long, Float>> = mutableMapOf()

    private val downloadJobs = mutableMapOf<String, kotlinx.coroutines.Job>()

    init {
        // Load HF token from preferences, with your provided token as default
        val prefs = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
        val savedToken = prefs.getString("hf_token", BuildConfig.HF_TOKEN)
        android.util.Log.d("ModelDownloadViewModel", "[init] Loaded HF token: ${savedToken?.take(8)}... from prefs, BuildConfig.HF_TOKEN: ${BuildConfig.HF_TOKEN?.take(8)}...")
        _hfToken.value = savedToken
        
        // Initialize ModelDownloader with token
        modelDownloader = ModelDownloader(ktorClient, context, savedToken)
        
        loadModels()
        loadImportedModels()
    }

    override fun onCleared() {
        super.onCleared()
        ktorClient.close()
    }

    fun setHuggingFaceToken(token: String?) {
        // Save token to preferences
        val prefs = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
        prefs.edit().putString("hf_token", token).apply()
        android.util.Log.d("ModelDownloadViewModel", "[setHuggingFaceToken] Token set: ${token?.take(8)}...")
        _hfToken.value = token
        
        // Recreate ModelDownloader with new token
        modelDownloader = ModelDownloader(ktorClient, context, token)
    }

    private fun loadModels() {
        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists()) modelsDir.mkdirs()

        // Prepare list with real downloaded/partial state
        val baseModels = ModelData.models.map { model ->
            // Handle Stable Diffusion NPU models (QNN format)
            if (model.modelFormat == "qnn_npu") {
                val sdModelsDir = File(context.filesDir, "sd_models")
                // Check in model-specific folder
                val modelTargetDir = File(sdModelsDir, model.name.replace(" ", "_"))
                val isDownloaded = checkSdModelExists(modelTargetDir, "qnn")
                if (isDownloaded) {
                    model.copy(
                        isDownloaded = true,
                        isDownloading = false,
                        downloadProgress = 1f,
                        downloadedBytes = model.sizeBytes,
                        totalBytes = model.sizeBytes
                    )
                } else {
                    // Detect partial ZIPs stored in files/sd_downloads to allow resume after kill
                    val sdTempFiles = File(context.filesDir, "sd_downloads")
                    var partialBytes = 0L
                    if (sdTempFiles.exists() && sdTempFiles.isDirectory) {
                        val base = model.name.replace(" ", "_")
                        val found = sdTempFiles.listFiles()?.firstOrNull { it.name.startsWith(base) && it.name.endsWith(".zip") }
                        if (found != null) partialBytes = found.length()
                    }

                    val progress = if (model.sizeBytes > 0 && partialBytes > 0) {
                        (partialBytes.toFloat() / model.sizeBytes).coerceIn(0f, 0.999f)
                    } else 0f

                    model.copy(isDownloaded = false, isDownloading = false, downloadProgress = progress, downloadedBytes = partialBytes, totalBytes = model.sizeBytes)
                }
            }
            // Handle Stable Diffusion CPU models (MNN format)
            else if (model.modelFormat == "mnn_cpu") {
                val sdModelsDir = File(context.filesDir, "sd_models")
                // Check in model-specific folder
                val modelTargetDir = File(sdModelsDir, model.name.replace(" ", "_"))
                val isDownloaded = checkSdModelExists(modelTargetDir, "mnn")
                if (isDownloaded) {
                    model.copy(
                        isDownloaded = true,
                        isDownloading = false,
                        downloadProgress = 1f,
                        downloadedBytes = model.sizeBytes,
                        totalBytes = model.sizeBytes
                    )
                } else {
                    // Detect partial ZIPs stored in files/sd_downloads to allow resume after kill
                    val sdTempFiles = File(context.filesDir, "sd_downloads")
                    var partialBytes = 0L
                    if (sdTempFiles.exists() && sdTempFiles.isDirectory) {
                        val base = model.name.replace(" ", "_")
                        val found = sdTempFiles.listFiles()?.firstOrNull { it.name.startsWith(base) && it.name.endsWith(".zip") }
                        if (found != null) partialBytes = found.length()
                    }

                    val progress = if (model.sizeBytes > 0 && partialBytes > 0) {
                        (partialBytes.toFloat() / model.sizeBytes).coerceIn(0f, 0.999f)
                    } else 0f

                    model.copy(isDownloaded = false, isDownloading = false, downloadProgress = progress, downloadedBytes = partialBytes, totalBytes = model.sizeBytes)
                }
            }
            // Handle image_generator models specially (multi-file format)
            else if (model.modelFormat == "image_generator") {
                val imageGenDir = File(context.filesDir, "image_generator/bins")
                    if (imageGenDir.exists() && imageGenDir.isDirectory) {
                    val files = imageGenDir.listFiles() ?: emptyArray()
                    val fileCount = files.filter { it.length() > 0 }.size
                    val totalDownloaded = files.sumOf { it.length() }

                    // Try to get expected file count from manifest
                    val expectedFileCount = try {
                        val url = java.net.URL(model.url)
                        val conn = url.openConnection() as java.net.HttpURLConnection
                        conn.connectTimeout = 3000
                        conn.readTimeout = 3000
                        val json = conn.inputStream.bufferedReader().use { it.readText() }
                        conn.disconnect()
                        val manifestObj = org.json.JSONObject(json)
                        manifestObj.getJSONArray("files").length()
                    } catch (e: Exception) {
                        -1 // Unknown, fall back to size check
                    }

                    if (fileCount > 0) {
                        android.util.Log.d("ModelDownloadViewModel", "[loadModels:image_gen] dir=${imageGenDir.absolutePath} fileCount=$fileCount totalDownloaded=$totalDownloaded model.sizeBytes=${model.sizeBytes} expectedFileCount=$expectedFileCount")
                        val completeEnough = if (expectedFileCount > 0) {
                            fileCount >= expectedFileCount
                        } else {
                            model.sizeBytes > 0 && totalDownloaded >= (model.sizeBytes * 0.95).toLong()
                        }

                        if (completeEnough) {
                            model.copy(
                                isDownloaded = true,
                                isDownloading = false,
                                downloadProgress = 1f,
                                downloadedBytes = totalDownloaded,
                                totalBytes = totalDownloaded
                            )
                        } else {
                            // Partial download: if total size unknown, set a small positive
                            // progress so the UI treats this as a resumable partial instead
                            // of showing the full "Download" button.
                            val progress = if (model.sizeBytes > 0) {
                                (totalDownloaded.toFloat() / model.sizeBytes).coerceIn(0f, 0.999f)
                            } else 0.0001f
                            model.copy(
                                isDownloaded = false,
                                isDownloading = false,
                                downloadProgress = progress,
                                downloadedBytes = totalDownloaded,
                                totalBytes = if (model.sizeBytes > 0) model.sizeBytes else null
                            )
                        }
                    } else {
                        model.copy(isDownloaded = false, isDownloading = false, downloadProgress = 0f, downloadedBytes = 0, totalBytes = model.sizeBytes)
                    }
                } else {
                    model.copy(isDownloaded = false, isDownloading = false, downloadProgress = 0f, downloadedBytes = 0, totalBytes = model.sizeBytes)
                }
            } else if (model.modelFormat == "onnx" && model.additionalFiles.isNotEmpty()) {
                // ONNX models with additional files (tokenizer, data files, etc.)
                val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                val onnxModelDir = File(modelsDir, modelDirName)
                
                if (onnxModelDir.exists() && onnxModelDir.isDirectory) {
                    val files = onnxModelDir.listFiles() ?: emptyArray()
                    val fileCount = files.filter { it.length() > 0 }.size
                    val totalDownloaded = files.sumOf { it.length() }
                    
                    // Expected file count = 1 main file + additionalFiles count
                    val expectedFileCount = 1 + model.additionalFiles.size
                    
                    if (fileCount > 0) {
                        android.util.Log.d("ModelDownloadViewModel", "[loadModels:onnx] dir=${onnxModelDir.absolutePath} fileCount=$fileCount totalDownloaded=$totalDownloaded expectedFileCount=$expectedFileCount")
                        
                        // Check if we have all expected files
                        val completeEnough = fileCount >= expectedFileCount
                        
                        if (completeEnough) {
                            model.copy(
                                isDownloaded = true,
                                isDownloading = false,
                                downloadProgress = 1f,
                                downloadedBytes = totalDownloaded,
                                totalBytes = totalDownloaded,
                                sizeBytes = totalDownloaded
                            )
                        } else {
                            // Partial download
                            val progress = if (model.sizeBytes > 0) {
                                (totalDownloaded.toFloat() / model.sizeBytes).coerceIn(0f, 0.999f)
                            } else 0.0001f
                            model.copy(
                                isDownloaded = false,
                                isDownloading = false,
                                downloadProgress = progress,
                                downloadedBytes = totalDownloaded,
                                totalBytes = if (model.sizeBytes > 0) model.sizeBytes else null
                            )
                        }
                    } else {
                        model.copy(isDownloaded = false, isDownloading = false, downloadProgress = 0f, downloadedBytes = 0, totalBytes = model.sizeBytes)
                    }
                } else {
                    model.copy(isDownloaded = false, isDownloading = false, downloadProgress = 0f, downloadedBytes = 0, totalBytes = model.sizeBytes)
                }
            } else if (model.modelFormat == "gguf" && model.additionalFiles.isNotEmpty()) {
                // GGUF models with additional files (e.g., mmproj vision projector)
                val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                val modelDir = File(modelsDir, modelDirName)
                if (!modelDir.exists()) modelDir.mkdirs()
                val primaryFile = File(modelDir, model.localFileName())

                // Calculate total size of all files found on disk (within model dir)
                var totalFoundBytes = if (primaryFile.exists()) primaryFile.length() else 0L
                var allFilesFound = primaryFile.exists()

                // Check additional files
                val baseUrl = model.url.substringBefore("?").substringBeforeLast("/") + "/"
                for (fileUrlOrPath in model.additionalFiles) {
                    val fileUrl = if (fileUrlOrPath.startsWith("http")) fileUrlOrPath else baseUrl + fileUrlOrPath
                    val fileName = fileUrl.substringAfterLast("/").substringBefore("?")
                    val file = File(modelDir, fileName)
                    if (file.exists()) {
                        totalFoundBytes += file.length()
                        android.util.Log.d("ModelDownloadViewModel", "Found additional file in model dir: ${fileName} (${file.length()} bytes)")
                    } else {
                        android.util.Log.d("ModelDownloadViewModel", "Missing additional file in model dir: ${fileName}")
                        allFilesFound = false
                    }
                }

                val sizeKnown = model.sizeBytes > 0
                // 98% threshold to account for minor size differences or overhead
                val completeEnough = sizeKnown && totalFoundBytes >= (model.sizeBytes * 0.98).toLong()

                if (completeEnough) {
                    model.copy(
                        isDownloaded = true,
                        isDownloading = false,
                        sizeBytes = totalFoundBytes, // Use actual size
                        downloadProgress = 1f,
                        downloadedBytes = totalFoundBytes,
                        totalBytes = totalFoundBytes
                    )
                } else {
                    val progress = if (sizeKnown) (totalFoundBytes.toFloat() / model.sizeBytes).coerceIn(0f, 1f) else -1f
                    model.copy(
                        isDownloaded = false,
                        isDownloading = false,
                        downloadProgress = progress,
                        downloadedBytes = totalFoundBytes,
                        totalBytes = if (sizeKnown) model.sizeBytes else null
                    )
                }
            } else {
                // Regular single-file models
                val primaryFile = File(modelsDir, model.localFileName())
                val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")

                if (!primaryFile.exists() && legacyFile.exists()) {
                    legacyFile.renameTo(primaryFile)
                }

                // Prefer the canonical primary file, but also detect partial/temp files that
                // may have different extensions (e.g., .part, .tmp) by searching for files
                // that start with the base name. This helps detect partial downloads after
                // the app was killed and allow resume.
                var file: File = primaryFile
                if (!file.exists()) {
                    val baseName = model.localFileName().substringBeforeLast('.')
                    val found = modelsDir.listFiles()?.firstOrNull { it.name.startsWith(baseName) }
                    if (found != null) file = found
                }

                if (file.exists()) {
                    val sizeKnown = model.sizeBytes > 0
                    val completeEnough = sizeKnown && file.length() >= (model.sizeBytes * 0.98).toLong()
                    val valid = isModelFileValid(file, model.modelFormat)

                    if (completeEnough && valid) {
                        model.copy(
                            isDownloaded = true,
                            isDownloading = false,
                            sizeBytes = file.length(),
                            downloadProgress = 1f,
                            downloadedBytes = file.length(),
                            totalBytes = file.length()
                        )
                    } else {
                        val progress = if (sizeKnown) (file.length().toFloat() / model.sizeBytes).coerceIn(0f, 1f) else -1f
                        model.copy(
                            isDownloaded = false,
                            isDownloading = false,
                            downloadProgress = progress,
                            downloadedBytes = file.length(),
                            totalBytes = if (sizeKnown) model.sizeBytes else null
                        )
                    }
                } else {
                    model.copy(isDownloaded = false, isDownloading = false, downloadProgress = 0f, downloadedBytes = 0, totalBytes = model.sizeBytes)
                }
            }
        }.toMutableList()

        _models.value = baseModels

        // Re-validate partials that may actually be complete (e.g., size unknown in HEAD)
        baseModels.filter { !it.isDownloaded && it.downloadedBytes > 0 && (it.totalBytes == null || it.totalBytes == 0L) }
            .forEach { partial ->
                viewModelScope.launch(Dispatchers.IO) {
                    val modelsDir = File(context.filesDir, "models")
                    val file = File(modelsDir, partial.localFileName())
                    if (file.exists()) {
                        val valid = isModelFileValid(file, partial.modelFormat)
                        if (valid) {
                            updateModel(partial.name) {
                                it.copy(
                                    isDownloaded = true,
                                    isDownloading = false,
                                    downloadProgress = 1f,
                                    downloadedBytes = file.length(),
                                    totalBytes = file.length(),
                                    sizeBytes = file.length()
                                )
                            }
                        }
                    }
                }
            }

        // For models with unknown sizeBytes, fetch HEAD in background to populate size
        baseModels.filter { it.sizeBytes == 0L }.forEach { unknownModel ->
            viewModelScope.launch(Dispatchers.IO) {
                try {
                    val url = java.net.URL(unknownModel.url)
                    val conn = url.openConnection() as java.net.HttpURLConnection
                    conn.requestMethod = "HEAD"
                    conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
                    
                    // Add HF token if available for HEAD requests
                    _hfToken.value?.let { token ->
                        if (token.isNotBlank()) {
                            conn.setRequestProperty("Authorization", "Bearer $token")
                        }
                    }
                    
                    conn.connectTimeout = 10_000
                    conn.readTimeout = 10_000
                    val size = conn.contentLengthLong
                    conn.disconnect()
                    if (size > 0) {
                        updateModel(unknownModel.name) { existing ->
                            // If there is already a partial file, recompute progress
                            val modelsDirHead = File(context.filesDir, "models")
                            val file = File(modelsDirHead, existing.localFileName())
                            if (file.exists()) {
                                val progress = (file.length().toFloat() / size).coerceIn(0f, 1f)
                                existing.copy(sizeBytes = size, totalBytes = size, downloadProgress = progress, downloadedBytes = file.length())
                            } else {
                                existing.copy(sizeBytes = size)
                            }
                        }
                    }
                } catch (_: Exception) {
                    // ignore
                }
            }
        }
    }
    
    /**
     * Check if SD model exists in the given directory or subdirectories
     * @param baseDir The base directory to search (e.g., sd_models)
     * @param modelType "qnn" for QNN models (unet.bin) or "mnn" for MNN models (unet.mnn)
     */
    private fun checkSdModelExists(baseDir: File, modelType: String): Boolean {
        if (!baseDir.exists() || !baseDir.isDirectory) return false

        // Accept variants like "unet_qnn.bin" or "unet-qnn.bin" by searching
        // recursively for any file that starts with "unet" and ends with the
        // expected extension. This is more robust across ZIPs that contain a
        // top-level folder or slightly different filenames.
        val expectedExt = if (modelType == "qnn") ".bin" else ".mnn"

        fun search(dir: File, depth: Int): Boolean {
            if (depth > 4 || !dir.exists() || !dir.isDirectory) return false
            dir.listFiles()?.forEach { f ->
                try {
                    if (f.isFile) {
                        val name = f.name.lowercase()
                        if (name.startsWith("unet") && name.endsWith(expectedExt)) return true
                    } else if (f.isDirectory) {
                        if (search(f, depth + 1)) return true
                    }
                } catch (_: Exception) { /* ignore permission issues */ }
            }
            return false
        }

        return search(baseDir, 0)
    }

    fun downloadModel(model: LLMModel) {
        val modelsDir = File(context.filesDir, "models")
        val existingFile = File(modelsDir, model.localFileName())
        val startingBytes = if (existingFile.exists()) existingFile.length() else 0L
        val total = if (model.sizeBytes > 0) model.sizeBytes else model.totalBytes ?: 0L

        android.util.Log.d("ModelDownloadViewModel", "[downloadModel] Starting download for ${model.name}, file exists: ${existingFile.exists()}, size: $startingBytes, expected: $total")

        // Immediately flip UI into downloading state with clear progress indicators
        updateModel(model.name) { 
            val progress = if (total > 0L) {
                (startingBytes.toFloat() / total).coerceIn(0f, 0.99f)
            } else {
                -1f
            }
            it.copy(
                isDownloading = true,
                isPaused = false, // Reset paused state when resuming
                downloadProgress = if (startingBytes == 0L && total > 0L) 0.0001f else progress,
                downloadedBytes = startingBytes,
                totalBytes = if (total > 0L) total else null,
                downloadSpeedBytesPerSec = 0L
            ) 
        }

        android.util.Log.d("ModelDownloadViewModel", "[downloadModel] Using HF token: ${_hfToken.value?.take(8)}... for model: ${model.name}")

        val job = viewModelScope.launch {
            var latestStatus: com.llmhub.llmhub.data.DownloadStatus? = null
            modelDownloader.downloadModel(model)
                .catch { exception ->
                    // Handle exceptions with better error reporting
                    android.util.Log.e("ModelDownloadViewModel", "Download failed for ${model.name}: ${exception.message}", exception)
                    updateModel(model.name) { 
                        it.copy(
                            isDownloading = false,
                            isExtracting = false,
                            downloadProgress = 0f,
                            downloadedBytes = 0L,
                            totalBytes = null,
                            downloadSpeedBytesPerSec = null
                        ) 
                    }
                }
                .onCompletion { cause ->
                    // Handle Stable Diffusion models (qnn_npu or mnn_cpu format)
                    if (model.modelFormat == "qnn_npu" || model.modelFormat == "mnn_cpu") {
                        val sdModelsDir = File(context.filesDir, "sd_models")
                        // Check in model-specific folder
                        val modelTargetDir = File(sdModelsDir, model.name.replace(" ", "_"))
                        val modelType = if (model.modelFormat == "qnn_npu") "qnn" else "mnn"
                        val isComplete = checkSdModelExists(modelTargetDir, modelType)
                        
                        if (isComplete && cause == null) {
                            android.util.Log.i("ModelDownloadViewModel", "SD model download completed: ${model.name}")
                            updateModel(model.name) { 
                                it.copy(
                                    isDownloaded = true, 
                                    isDownloading = false,
                                    isExtracting = false,
                                    downloadProgress = 1f, 
                                    downloadedBytes = model.sizeBytes, 
                                    totalBytes = model.sizeBytes
                                ) 
                            }
                        } else {
                            android.util.Log.w("ModelDownloadViewModel", "SD model download incomplete: ${model.name}")
                            updateModel(model.name) {
                                it.copy(
                                    isDownloaded = false,
                                    isDownloading = false,
                                    isExtracting = false,
                                    downloadProgress = 0f
                                )
                            }
                        }
                        return@onCompletion
                    }
                    
                    // Handle image_generator models specially (multi-file format)
                    if (model.modelFormat == "image_generator") {
                        val imageGenDir = File(context.filesDir, "image_generator/bins")
                        
                        if (imageGenDir.exists() && imageGenDir.isDirectory) {
                            val files = imageGenDir.listFiles() ?: emptyArray()
                            val fileCount = files.filter { it.length() > 0 }.size
                            val totalDownloaded = files.sumOf { it.length() }
                            
                            // Check if download completed successfully by checking if downloadedBytes == totalBytes in latest status
                            // This means all files from manifest were processed
                            val completed = latestStatus?.let { 
                                it.downloadedBytes == it.totalBytes && it.totalBytes > 0 
                            } ?: false
                            
                            if (completed && cause == null) {
                                android.util.Log.i("ModelDownloadViewModel", "Image generator download completed: ${model.name}, files: $fileCount, size: $totalDownloaded")
                                updateModel(model.name) { 
                                    it.copy(
                                        isDownloaded = true, 
                                        isDownloading = false, 
                                        downloadProgress = 1f, 
                                        downloadedBytes = totalDownloaded, 
                                        totalBytes = totalDownloaded
                                    ) 
                                }
                            } else {
                                android.util.Log.w("ModelDownloadViewModel", "Image generator download incomplete: ${model.name}, files: $fileCount, downloaded: $totalDownloaded, latestStatus: ${latestStatus?.downloadedBytes}/${latestStatus?.totalBytes}")
                                updateModel(model.name) {
                                    val progress = if (model.sizeBytes > 0) (totalDownloaded.toFloat() / model.sizeBytes).coerceIn(0f, 0.999f) else 0f
                                    it.copy(
                                        isDownloaded = false,
                                        isDownloading = false,
                                        downloadProgress = progress,
                                        downloadedBytes = totalDownloaded,
                                        totalBytes = model.sizeBytes
                                    )
                                }
                            }
                        } else {
                            updateModel(model.name) { it.copy(isDownloaded = false, isDownloading = false, downloadProgress = 0f) }
                        }
                        return@onCompletion
                    }
                    
                    // ONNX models with additional files (decoder + embed + tokenizer etc.) - files live in a subdir
                    if (model.modelFormat == "onnx" && model.additionalFiles.isNotEmpty()) {
                        val modelsDir = File(context.filesDir, "models")
                        val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                        val onnxModelDir = File(modelsDir, modelDirName)
                        if (onnxModelDir.exists() && onnxModelDir.isDirectory) {
                            val files = onnxModelDir.listFiles()?.filter { it.length() > 0 } ?: emptyList()
                            val fileCount = files.size
                            val totalDownloaded = files.sumOf { it.length() }
                            val expectedFileCount = 1 + model.additionalFiles.size
                            val completeEnough = fileCount >= expectedFileCount
                            if (completeEnough && cause == null) {
                                android.util.Log.i("ModelDownloadViewModel", "ONNX model download completed: ${model.name}, files: $fileCount/$expectedFileCount, size: $totalDownloaded")
                                updateModel(model.name) {
                                    it.copy(
                                        isDownloaded = true,
                                        isDownloading = false,
                                        isExtracting = false,
                                        downloadProgress = 1f,
                                        downloadedBytes = totalDownloaded,
                                        totalBytes = totalDownloaded,
                                        sizeBytes = totalDownloaded
                                    )
                                }
                            } else {
                                android.util.Log.d("ModelDownloadViewModel", "ONNX model incomplete: ${model.name}, files: $fileCount/$expectedFileCount")
                                updateModel(model.name) {
                                    val progress = if (model.sizeBytes > 0) (totalDownloaded.toFloat() / model.sizeBytes).coerceIn(0f, 0.999f) else 0.0001f
                                    it.copy(
                                        isDownloaded = false,
                                        isDownloading = false,
                                        isExtracting = false,
                                        downloadProgress = progress,
                                        downloadedBytes = totalDownloaded,
                                        totalBytes = if (model.sizeBytes > 0) model.sizeBytes else null
                                    )
                                }
                            }
                        } else {
                            updateModel(model.name) { it.copy(isDownloaded = false, isDownloading = false, downloadProgress = 0f) }
                        }
                        return@onCompletion
                    }
                    
                    // GGUF models with additional files (like mmproj)
                    if (model.modelFormat == "gguf" && model.additionalFiles.isNotEmpty()) {
                        val modelsDir = File(context.filesDir, "models")
                        val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                        val modelDir = File(modelsDir, modelDirName)
                        val primaryFile = File(modelDir, model.localFileName())

                        var totalDownloadedOnDisk = if (primaryFile.exists()) primaryFile.length() else 0L
                        val baseUrl = model.url.substringBefore("?").substringBeforeLast("/") + "/"

                        for (fileUrlOrPath in model.additionalFiles) {
                            val fileUrl = if (fileUrlOrPath.startsWith("http")) fileUrlOrPath else baseUrl + fileUrlOrPath
                            val fileName = fileUrl.substringAfterLast("/").substringBefore("?")
                            val file = File(modelDir, fileName)
                            if (file.exists()) {
                                totalDownloadedOnDisk += file.length()
                                android.util.Log.d("ModelDownloadViewModel", "Found additional file in model dir on completion: ${fileName} (${file.length()} bytes)")
                            } else {
                                android.util.Log.d("ModelDownloadViewModel", "Missing additional file in model dir on completion: ${fileName}")
                            }
                        }

                        // Prefer authoritative total from latestStatus (downloader), otherwise use model.sizeBytes
                        val expectedTotal = if (latestStatus != null && latestStatus!!.totalBytes > 0) latestStatus!!.totalBytes else model.sizeBytes

                        // If expectedTotal is zero or unknown, fall back to comparing against actual bytes (exact equality)
                        val completeEnough = if (expectedTotal > 0) {
                            totalDownloadedOnDisk >= (expectedTotal * 0.995).toLong()
                        } else {
                            // If downloader reported nothing, require the files to be non-empty and at least 90% of previously known size
                            totalDownloadedOnDisk > 0 && (model.sizeBytes <= 0 || totalDownloadedOnDisk >= (model.sizeBytes * 0.95).toLong())
                        }

                        if (completeEnough && cause == null) {
                            android.util.Log.i("ModelDownloadViewModel", "GGUF model download completed for ${model.name}: size=$totalDownloadedOnDisk, expected=$expectedTotal")
                            updateModel(model.name) {
                                it.copy(
                                    isDownloaded = true,
                                    isDownloading = false,
                                    downloadProgress = 1f,
                                    sizeBytes = totalDownloadedOnDisk,
                                    downloadedBytes = totalDownloadedOnDisk,
                                    totalBytes = totalDownloadedOnDisk
                                )
                            }
                        } else {
                            android.util.Log.w("ModelDownloadViewModel", "GGUF model download incomplete for ${model.name}: found=$totalDownloadedOnDisk expected=$expectedTotal")
                            // Partial
                            updateModel(model.name) {
                                val progress = if (expectedTotal > 0) (totalDownloadedOnDisk.toFloat() / expectedTotal).coerceIn(0f, 1f) else -1f
                                it.copy(
                                    isDownloaded = false,
                                    isDownloading = false,
                                    downloadProgress = progress,
                                    downloadedBytes = totalDownloadedOnDisk,
                                    totalBytes = if (expectedTotal > 0) expectedTotal else null
                                )
                            }
                        }
                        return@onCompletion
                    }

                    // Regular single-file models
                    val modelsDir = File(context.filesDir, "models")
                    val primaryFile = File(modelsDir, model.localFileName())
                    val legacyFile  = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")
                    if (!primaryFile.exists() && legacyFile.exists()) {
                        legacyFile.renameTo(primaryFile)
                    }

                    val modelFile = primaryFile

                    val expectedBytes = latestStatus?.totalBytes ?: model.sizeBytes

                    val minReasonableSize = 10 * 1024 * 1024 // 10 MiB
                    val completed = modelFile.exists() && (
                        (expectedBytes > 0 && modelFile.length() >= (expectedBytes * 0.98).toLong()) ||
                        (expectedBytes <= 0 && modelFile.length() >= minReasonableSize)
                    )

                    val valid = if (completed) isModelFileValid(modelFile, model.modelFormat) else false

                    if (completed && valid && cause == null) {
                        updateModel(model.name) { it.copy(isDownloaded = true, isDownloading = false, downloadProgress = 1f, sizeBytes = modelFile.length(), downloadedBytes = modelFile.length(), totalBytes = modelFile.length()) }
                    } else {
                        // Keep partial file so user can resume, unless explicitly cancelled
                        updateModel(model.name) {
                            val sizeKnown = expectedBytes > 0
                            val progress = if (sizeKnown && modelFile.exists()) {
                                (modelFile.length().toFloat() / expectedBytes).coerceIn(0f, 1f)
                            } else if (modelFile.exists()) {
                                -1f
                            } else 0f
                            it.copy(
                                isDownloaded = false,
                                isDownloading = false,
                                downloadProgress = progress,
                                downloadedBytes = if (modelFile.exists()) modelFile.length() else 0L,
                                totalBytes = if (sizeKnown) expectedBytes else null
                            )
                        }
                    }
                }
                .collect { status ->
                    latestStatus = status
                    updateModel(model.name, rateLimit = true) {
                        val progress = if (status.totalBytes > 0) {
                            kotlin.math.min(0.999f, status.downloadedBytes.toFloat() / status.totalBytes)
                        } else {
                            // If total bytes unknown, show indeterminate progress
                            -1f
                        }
                        it.copy(
                            isDownloading = true,
                            isExtracting = status.isExtracting,
                            sizeBytes = if (status.totalBytes > it.sizeBytes) status.totalBytes else it.sizeBytes,
                            downloadProgress = progress,
                            downloadedBytes = status.downloadedBytes,
                            totalBytes = status.totalBytes,
                            downloadSpeedBytesPerSec = status.downloadSpeedBytesPerSec
                        )
                    }
                }
        }

        downloadJobs[model.name] = job
    }

    fun cancelDownload(model: LLMModel) {
        downloadJobs[model.name]?.cancel()
        downloadJobs.remove(model.name)
        // Also notify any foreground service handling downloads to cancel
        try {
            val cancelIntent = android.content.Intent(context, com.llmhub.llmhub.service.ModelDownloadService::class.java).apply {
                action = com.llmhub.llmhub.service.ModelDownloadService.ACTION_CANCEL_DOWNLOAD
                putExtra(com.llmhub.llmhub.service.ModelDownloadService.EXTRA_MODEL_NAME, model.name)
            }
            context.startService(cancelIntent)
        } catch (e: Exception) {
            android.util.Log.w("ModelDownloadViewModel", "Failed to notify ModelDownloadService to cancel: ${e.message}")
        }

        // Delete any partial files that may exist under models/ (supporting multiple filename variants)
        val modelsDir = File(context.filesDir, "models")
        if (modelsDir.exists() && modelsDir.isDirectory) {
            try {
                // Primary expected filename
                val primary = File(modelsDir, model.localFileName())
                if (primary.exists()) primary.delete()

                // Legacy .gguf name
                val legacy = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")
                if (legacy.exists()) legacy.delete()

                // Also remove any files that start with the URL-derived base name (handles .part, .tmp, etc.)
                val base = model.localFileName().substringBeforeLast('.')
                modelsDir.listFiles()?.forEach { f ->
                    if (f.name.startsWith(base) && f.name != model.localFileName()) {
                        try { f.delete() } catch (_: Exception) { }
                    }
                }
            } catch (e: Exception) {
                android.util.Log.w("ModelDownloadViewModel", "Error cleaning partial files: ${e.message}")
            }
        }

        // Also remove any temp SD zip downloads for Stable Diffusion models (check both cache and files dirs)
        try {
            val sdTempCache = File(context.cacheDir, "sd_downloads")
            if (sdTempCache.exists() && sdTempCache.isDirectory) {
                sdTempCache.listFiles()?.forEach { f ->
                    if (f.name.contains(model.name.replace(" ", "_")).or(f.name.endsWith(".zip"))) {
                        try { f.delete() } catch (_: Exception) { }
                    }
                }
            }

            val sdTempFiles = File(context.filesDir, "sd_downloads")
            if (sdTempFiles.exists() && sdTempFiles.isDirectory) {
                sdTempFiles.listFiles()?.forEach { f ->
                    if (f.name.contains(model.name.replace(" ", "_")).or(f.name.endsWith(".zip"))) {
                        try { f.delete() } catch (_: Exception) { }
                    }
                }
            }
        } catch (_: Exception) { }

        // Reset UI state (including isPaused so clear works from paused state)
        updateModel(model.name) {
            it.copy(
                isDownloading = false,
                isExtracting = false,
                isPaused = false,
                downloadProgress = 0f,
                downloadedBytes = 0L,
                totalBytes = null,
                downloadSpeedBytesPerSec = null
            )
        }
    }

    fun pauseDownload(model: LLMModel) {
        downloadJobs[model.name]?.cancel()
        downloadJobs.remove(model.name)

        // Update UI state to show paused (keep partial file)
        updateModel(model.name) {
            it.copy(
                isDownloading = false,
                isExtracting = false,
                isPaused = true,
                // Keep download progress and bytes for resume capability
                downloadSpeedBytesPerSec = null
            )
        }
    }

    fun resumeDownload(model: LLMModel) {
        // Resume is the same as starting a download - it will handle resuming from the correct position
        downloadModel(model)
    }

    fun deleteModel(model: LLMModel) {
        // First cancel any ongoing download
        cancelDownload(model)
        
        // Handle imported models differently - just remove from list, don't delete file
        if (model.source == "Custom") {
            val currentModels = _models.value.toMutableList()
            currentModels.removeAll { it.name == model.name }
            _models.value = currentModels
            saveImportedModels()
            android.util.Log.d("ModelDownloadViewModel", "[deleteModel] Removed imported model: ${model.name}")
            return
        }
        
        viewModelScope.launch(Dispatchers.IO) {
            // Handle Stable Diffusion models (stored in sd_models directory)
            if (model.modelFormat == "qnn_npu" || model.modelFormat == "mnn_cpu") {
                val sdModelsDir = File(context.filesDir, "sd_models")
                val modelTargetDir = File(sdModelsDir, model.name.replace(" ", "_"))
                
                if (modelTargetDir.exists() && modelTargetDir.isDirectory) {
                    // Delete all files and subdirectories recursively for this model only
                    modelTargetDir.deleteRecursively()
                    android.util.Log.d("ModelDownloadViewModel", "[deleteModel] Deleted SD model directory: ${modelTargetDir.absolutePath}")
                }
            } else {
                // Physically remove the file from the app's private storage so that
                // the model will no longer be detected as «downloaded» next time we
                // build the list (e.g. after an app restart).
                val modelsDir = File(context.filesDir, "models")
                val primaryFile = File(modelsDir, model.localFileName())
                val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")
                
                var deletedPrimary = false
                var deletedLegacy = false
                
                if (primaryFile.exists()) {
                    deletedPrimary = primaryFile.delete()
                    android.util.Log.d("ModelDownloadViewModel", "[deleteModel] Deleted primary file: $deletedPrimary, path: ${primaryFile.absolutePath}")
                }
                if (legacyFile.exists()) {
                    deletedLegacy = legacyFile.delete()
                    android.util.Log.d("ModelDownloadViewModel", "[deleteModel] Deleted legacy file: $deletedLegacy, path: ${legacyFile.absolutePath}")
                }
                
                // Verify files are actually gone
                val primaryExists = primaryFile.exists()
                val legacyExists = legacyFile.exists()
                android.util.Log.d("ModelDownloadViewModel", "[deleteModel] Post-deletion check - Primary exists: $primaryExists, Legacy exists: $legacyExists")
            }
        }

        updateModel(model.name) {
            it.copy(
                isDownloaded = false,
                isDownloading = false,
                isExtracting = false,
                isPaused = false, // Reset paused state when deleting
                // reset the other transient download fields so the UI shows the
                // correct state immediately after we press the button
                downloadProgress = 0f,
                downloadedBytes = 0L,
                totalBytes = null,
                downloadSpeedBytesPerSec = null
            )
        }
    }

    fun addExternalModel(externalModel: LLMModel): Boolean {
        val currentModels = _models.value.toMutableList()
        
        // Check if model with same name already exists
        if (currentModels.any { it.name == externalModel.name }) {
            android.util.Log.w("ModelDownloadViewModel", "Model with name '${externalModel.name}' already exists")
            return false
        }
        
        // Add the external model to the list
        currentModels.add(externalModel)
        _models.value = currentModels
        
        android.util.Log.d("ModelDownloadViewModel", "Added external model: ${externalModel.name}")
        
        // If it's an image model (ZIP file), extract it
        if ((externalModel.category == "qnn_npu" || externalModel.category == "mnn_cpu") && externalModel.isExtracting) {
            viewModelScope.launch(Dispatchers.IO) {
                try {
                    android.util.Log.d("ModelDownloadViewModel", "Extracting imported image model: ${externalModel.name}")

                    // Ensure UI shows this as an active operation
                    updateModel(externalModel.name) {
                        it.copy(isDownloading = true)
                    }
                    
                    // Get the ZIP file from URI
                    val uri = Uri.parse(externalModel.url)
                    val tempFile = File(context.cacheDir, "imported_${System.currentTimeMillis()}.zip")
                    
                    // Copy URI content to temp file
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        tempFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                    
                    // Extract to sd_models/<modelName> directory so the folder is named after user's chosen name
                    val sdModelsDir = File(context.filesDir, "sd_models")
                    if (!sdModelsDir.exists()) {
                        sdModelsDir.mkdirs()
                    }
                    
                    // Create a folder specifically for this model using the user-provided name
                    val modelTargetDir = File(sdModelsDir, externalModel.name.replace(" ", "_"))
                    if (modelTargetDir.exists()) {
                        // Clean up previous failed extraction
                        modelTargetDir.deleteRecursively()
                    }
                    modelTargetDir.mkdirs()
                    
                    // Extract the ZIP into the model-specific folder
                    val extractSuccess = com.llmhub.llmhub.utils.ZipExtractor.extractZip(
                        zipFile = tempFile,
                        targetDir = modelTargetDir,
                        onProgress = { progress ->
                            // Update progress
                            updateModel(externalModel.name) {
                                it.copy(downloadProgress = progress)
                            }
                        }
                    )
                    
                    // Clean up temp file
                    tempFile.delete()
                    
                    if (extractSuccess) {
                        // Mark as downloaded
                        updateModel(externalModel.name) {
                            it.copy(
                                isDownloaded = true,
                                isDownloading = false,
                                isExtracting = false,
                                downloadProgress = 1.0f
                            )
                        }
                        android.util.Log.d("ModelDownloadViewModel", "Successfully extracted imported image model")
                        // Save to SharedPreferences
                        saveImportedModels()
                    } else {
                        // Extraction failed - remove from list
                        val updatedModels = _models.value.toMutableList()
                        updatedModels.removeAll { it.name == externalModel.name }
                        _models.value = updatedModels
                        android.util.Log.e("ModelDownloadViewModel", "Failed to extract imported image model")
                    }
                } catch (e: Exception) {
                    android.util.Log.e("ModelDownloadViewModel", "Error extracting imported image model: ${e.message}")
                    // Remove from list on error
                    val updatedModels = _models.value.toMutableList()
                    updatedModels.removeAll { it.name == externalModel.name }
                    _models.value = updatedModels
                }
            }
        } else {
            saveImportedModels()
        }
        
        return true
    }

    /**
     * Import a Vision Projector file (mmproj/gguf projector) for an already-added custom model.
     * Runs on Dispatchers.IO and updates the imported model record (additionalFiles) when complete.
     */
    fun importVisionProjector(modelName: String, projectorUri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val resolver = context.contentResolver
                val projName = resolver.query(projectorUri, null, null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                        if (idx >= 0) cursor.getString(idx) else null
                    } else null
                } ?: projectorUri.lastPathSegment ?: "vision_projector.gguf"

                val modelsDir = File(context.filesDir, "models")
                if (!modelsDir.exists()) modelsDir.mkdirs()

                // Place projector inside a model-specific folder to keep things tidy
                val modelDirName = modelName.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                val modelDir = File(modelsDir, modelDirName)
                if (!modelDir.exists()) modelDir.mkdirs()

                val outFile = File(modelDir, projName)
                resolver.openInputStream(projectorUri)?.use { input ->
                    outFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }

                // Update the imported model entry to record the projector filename
                updateModel(modelName) { existing ->
                    val updatedFiles = existing.additionalFiles + projName
                    existing.copy(additionalFiles = updatedFiles)
                }

                saveImportedModels()
                android.util.Log.d("ModelDownloadViewModel", "Imported vision projector for $modelName -> ${outFile.absolutePath}")
            } catch (e: Exception) {
                android.util.Log.e("ModelDownloadViewModel", "Failed to import vision projector for $modelName: ${e.message}")
            }
        }
    }
    
    fun getImportedModels(): List<LLMModel> {
        return _models.value.filter { it.source == "Custom" && it.isDownloaded }
    }
    
    private fun loadImportedModels() {
        val prefs = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
        val importedModelsJson = prefs.getString("imported_models", null)
        
        if (importedModelsJson != null) {
            try {
                val importedModels = Gson().fromJson(importedModelsJson, Array<LLMModel>::class.java).toList()
                val currentModels = _models.value.toMutableList()
                
                // Add imported models that aren't already in the list
                importedModels.forEach { importedModel ->
                    if (!currentModels.any { it.name == importedModel.name }) {
                        currentModels.add(importedModel)
                    }
                }
                
                _models.value = currentModels
                android.util.Log.d("ModelDownloadViewModel", "Loaded ${importedModels.size} imported models from preferences")
            } catch (e: Exception) {
                android.util.Log.e("ModelDownloadViewModel", "Failed to load imported models: ${e.message}")
            }
        }
    }
    
    private fun saveImportedModels() {
        val importedModels = getImportedModels()
        val prefs = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
        
        try {
            val json = Gson().toJson(importedModels)
            prefs.edit().putString("imported_models", json).apply()
            android.util.Log.d("ModelDownloadViewModel", "Saved ${importedModels.size} imported models to preferences")
        } catch (e: Exception) {
            android.util.Log.e("ModelDownloadViewModel", "Failed to save imported models: ${e.message}")
        }
    }

    private fun updateModel(
        modelName: String,
        rateLimit: Boolean = false,
        updateAction: (LLMModel) -> LLMModel
    ) {
        val currentModels = _models.value.toMutableList()
        val modelIndex = currentModels.indexOfFirst { it.name == modelName }
        if (modelIndex != -1) {
            val updatedModel = updateAction(currentModels[modelIndex])

            // If rateLimit is enabled, only propagate when progress advanced ≥0.1% or ≥1 MB.
            if (rateLimit) {
                val prev = lastProgressMap[modelName]
                val deltaBytes = (updatedModel.downloadedBytes - (prev?.first ?: 0L))
                val deltaProgress = kotlin.math.abs(updatedModel.downloadProgress - (prev?.second ?: 0f))
                if (deltaBytes < 1_000_000 && deltaProgress < 0.001f) {
                    return
                }
                lastProgressMap[modelName] = Pair(updatedModel.downloadedBytes, updatedModel.downloadProgress)
            }

            currentModels[modelIndex] = updatedModel
            _models.value = currentModels
        }
    }
}
