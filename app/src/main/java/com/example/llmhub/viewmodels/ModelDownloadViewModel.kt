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
import android.content.Intent
import com.llmhub.llmhub.service.ModelDownloadService
import android.content.BroadcastReceiver
import android.content.IntentFilter
import androidx.localbroadcastmanager.content.LocalBroadcastManager

class ModelDownloadViewModel(application: Application) : AndroidViewModel(application) {
    private val _models = MutableStateFlow<List<LLMModel>>(emptyList())
    val models: StateFlow<List<LLMModel>> = _models.asStateFlow()
    
    private val _hfToken = MutableStateFlow<String?>(null)
    val hfToken: StateFlow<String?> = _hfToken.asStateFlow()

    private val ktorClient = HttpClient(Android)
    private val context = application.applicationContext
    private var modelDownloader: ModelDownloader

    private var lastProgressMap: MutableMap<String, Pair<Long, Float>> = mutableMapOf()

    private val downloadProgressReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                ModelDownloadService.ACTION_DOWNLOAD_PROGRESS -> {
                    val modelName = intent.getStringExtra(ModelDownloadService.EXTRA_MODEL_NAME) ?: return
                    val downloadedBytes = intent.getLongExtra(ModelDownloadService.EXTRA_DOWNLOADED_BYTES, 0)
                    val totalBytes = intent.getLongExtra(ModelDownloadService.EXTRA_TOTAL_BYTES, 0)
                    val downloadSpeed = intent.getLongExtra(ModelDownloadService.EXTRA_DOWNLOAD_SPEED, 0)
                    
                    updateModel(modelName, rateLimit = true) { model ->
                        val progress = if (totalBytes > 0) {
                            // Allow progress to reach 1.0f when download is complete
                            (downloadedBytes.toFloat() / totalBytes).coerceIn(0f, 1f)
                        } else {
                            -1f
                        }
                        
                        // Check if download is actually complete
                        val isComplete = downloadedBytes >= totalBytes && totalBytes > 0
                        
                        model.copy(
                            isDownloading = !isComplete,
                            isDownloaded = isComplete,
                            sizeBytes = if (totalBytes > model.sizeBytes) totalBytes else model.sizeBytes,
                            downloadProgress = progress,
                            downloadedBytes = downloadedBytes,
                            totalBytes = totalBytes,
                            downloadSpeedBytesPerSec = downloadSpeed
                        )
                    }
                }
                ModelDownloadService.ACTION_DOWNLOAD_COMPLETED -> {
                    val modelName = intent.getStringExtra(ModelDownloadService.EXTRA_MODEL_NAME) ?: return
                    // Re-validate the downloaded file
                    val modelsDir = File(this@ModelDownloadViewModel.context.filesDir, "models")
                    val model = _models.value.find { it.name == modelName }
                    if (model != null) {
                        val primaryFile = File(modelsDir, model.localFileName())
                        val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")
                        
                        if (!primaryFile.exists() && legacyFile.exists()) {
                            legacyFile.renameTo(primaryFile)
                        }
                        
                        val modelFile = primaryFile
                        val valid = if (modelFile.exists()) isModelFileValid(modelFile, model.modelFormat) else false
                        
                        if (valid) {
                            updateModel(modelName) {
                                it.copy(
                                    isDownloaded = true,
                                    isDownloading = false,
                                    downloadProgress = 1f,
                                    sizeBytes = modelFile.length(),
                                    downloadedBytes = modelFile.length(),
                                    totalBytes = modelFile.length()
                                )
                            }
                        }
                    }
                }
                ModelDownloadService.ACTION_DOWNLOAD_ERROR -> {
                    val modelName = intent.getStringExtra(ModelDownloadService.EXTRA_MODEL_NAME) ?: return
                    // Reset UI state on error
                    updateModel(modelName) {
                        it.copy(
                            isDownloading = false,
                            downloadProgress = 0f,
                            downloadedBytes = 0L,
                            totalBytes = null,
                            downloadSpeedBytesPerSec = null
                        )
                    }
                }
            }
        }
    }

    init {
        // Load HF token from preferences, with your provided token as default
        val prefs = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
        val savedToken = prefs.getString("hf_token", BuildConfig.HF_TOKEN)
        android.util.Log.d("ModelDownloadViewModel", "[init] Loaded HF token: ${savedToken?.take(8)}... from prefs, BuildConfig.HF_TOKEN: ${BuildConfig.HF_TOKEN?.take(8)}...")
        _hfToken.value = savedToken
        
        // Initialize ModelDownloader with token (still needed for size checks)
        modelDownloader = ModelDownloader(ktorClient, context, savedToken)
        
        // Register broadcast receiver for download updates
        val filter = IntentFilter().apply {
            addAction(ModelDownloadService.ACTION_DOWNLOAD_PROGRESS)
            addAction(ModelDownloadService.ACTION_DOWNLOAD_COMPLETED)
            addAction(ModelDownloadService.ACTION_DOWNLOAD_ERROR)
        }
        LocalBroadcastManager.getInstance(context).registerReceiver(downloadProgressReceiver, filter)
        
        loadModels()
    }

    override fun onCleared() {
        super.onCleared()
        LocalBroadcastManager.getInstance(context).unregisterReceiver(downloadProgressReceiver)
        ktorClient.close()
    }

    fun setHuggingFaceToken(token: String?) {
        // Save token to preferences
        val prefs = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
        prefs.edit().putString("hf_token", token).apply()
        android.util.Log.d("ModelDownloadViewModel", "[setHuggingFaceToken] Token set: ${token?.take(8)}...")
        _hfToken.value = token
        
        // Recreate ModelDownloader with new token (still needed for size checks)
        modelDownloader = ModelDownloader(ktorClient, context, token)
    }

    private fun loadModels() {
        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists()) modelsDir.mkdirs()

        // Prepare list with real downloaded/partial state
        val baseModels = ModelData.models.map { model ->
            val primaryFile = File(modelsDir, model.localFileName())
            val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")

            if (!primaryFile.exists() && legacyFile.exists()) {
                legacyFile.renameTo(primaryFile)
            }

            val file = primaryFile
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
            } else model
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

        // Start the background service for download instead of downloading in ViewModel
        val intent = Intent(context, ModelDownloadService::class.java).apply {
            putExtra("modelName", model.name)
            putExtra("modelDescription", model.description)
            putExtra("modelUrl", model.url)
            putExtra("modelSize", model.sizeBytes)
            putExtra("modelCategory", model.category)
            putExtra("modelSource", model.source)
            putExtra("supportsVision", model.supportsVision)
            putExtra("supportsGpu", model.supportsGpu)
            putExtra("minRamGB", model.requirements.minRamGB)
            putExtra("recommendedRamGB", model.requirements.recommendedRamGB)
            putExtra("hfToken", _hfToken.value)
        }
        
        try {
            context.startForegroundService(intent)
            android.util.Log.d("ModelDownloadViewModel", "Started background download service for model: ${model.name}")
        } catch (e: Exception) {
            android.util.Log.e("ModelDownloadViewModel", "Failed to start download service", e)
            // Revert UI state on failure
            updateModel(model.name) { 
                it.copy(
                    isDownloading = false,
                    downloadProgress = if (startingBytes > 0 && total > 0) (startingBytes.toFloat() / total).coerceIn(0f, 1f) else 0f,
                    downloadedBytes = startingBytes,
                    totalBytes = if (total > 0L) total else null,
                    downloadSpeedBytesPerSec = null
                ) 
            }
        }
    }

    fun cancelDownload(model: LLMModel) {
        // Send cancel action to the service
        val cancelIntent = Intent(context, ModelDownloadService::class.java).apply {
            action = ModelDownloadService.ACTION_CANCEL_DOWNLOAD
            putExtra(ModelDownloadService.EXTRA_MODEL_NAME, model.name)
        }
        context.startService(cancelIntent)

        // Reset UI state immediately
        updateModel(model.name) {
            it.copy(
                isDownloading = false,
                isPaused = false, // Reset paused state when canceling
                downloadProgress = 0f,
                downloadedBytes = 0L,
                totalBytes = null,
                downloadSpeedBytesPerSec = null
            )
        }
    }

    fun pauseDownload(model: LLMModel) {
        // Send pause action to the service
        val pauseIntent = Intent(context, ModelDownloadService::class.java).apply {
            action = ModelDownloadService.ACTION_PAUSE_DOWNLOAD
            putExtra(ModelDownloadService.EXTRA_MODEL_NAME, model.name)
        }
        context.startService(pauseIntent)

        // Update UI state to show paused
        updateModel(model.name) {
            it.copy(
                isDownloading = false,
                isPaused = true,
                // Keep download progress and bytes for resume capability
                downloadSpeedBytesPerSec = null
            )
        }
    }

    fun resumeDownload(model: LLMModel) {
        // Resume is the same as starting a download - the service will handle resuming from the correct position
        downloadModel(model)
    }

    fun deleteModel(model: LLMModel) {
        // First cancel any ongoing download
        cancelDownload(model)
        
        viewModelScope.launch(Dispatchers.IO) {
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

        updateModel(model.name) {
            it.copy(
                isDownloaded = false,
                isDownloading = false,
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