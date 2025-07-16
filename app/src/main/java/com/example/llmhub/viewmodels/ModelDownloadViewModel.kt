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

        // Prepare list with real downloaded state
        val baseModels = ModelData.models.map { model ->
            // Preferred filename (stable): derived from URL
            val primaryFile = File(modelsDir, model.localFileName())

            // Older builds used a filename derived from the human-readable model name.
            val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")

            // If legacy exists but primary doesn't, rename so we unify the scheme.
            if (!primaryFile.exists() && legacyFile.exists()) {
                legacyFile.renameTo(primaryFile)
            }

            val file = primaryFile
            if (file.exists()) {
                val minSize = 10 * 1024 * 1024 // 10 MiB safeguard against HTML error pages
                val completeEnough = file.length() >= minSize

                if (completeEnough) model.copy(isDownloaded = true, sizeBytes = file.length()) else model
            } else model
        }.toMutableList()

        _models.value = baseModels

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
                        updateModel(unknownModel.name) { it.copy(sizeBytes = size) }
                    }
                } catch (_: Exception) {
                    // ignore
                }
            }
        }
    }

    fun downloadModel(model: LLMModel) {
        // Immediately flip UI into downloading state with clear progress indicators
        updateModel(model.name) { 
            it.copy(
                downloadProgress = 0.0001f, 
                downloadedBytes = 0L,
                totalBytes = if (it.sizeBytes > 0) it.sizeBytes else null,
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
                            downloadProgress = 0f,
                            downloadedBytes = 0L,
                            totalBytes = null,
                            downloadSpeedBytesPerSec = null
                        ) 
                    }
                }
                .onCompletion { cause ->
                    val modelsDir = File(context.filesDir, "models")
                    val primaryFile = File(modelsDir, model.localFileName())
                    val legacyFile  = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")
                    if (!primaryFile.exists() && legacyFile.exists()) {
                        legacyFile.renameTo(primaryFile)
                    }

                    val modelFile = primaryFile

                    val expectedBytes = latestStatus?.totalBytes ?: 0L

                    /**
                     * Treat the download as successful when:
                     *   • we didn’t receive an exception (cause == null) AND
                     *   • the file exists AND is at least 10 MiB –OR– >= 90 % of the expected size.
                     *
                     * Rationale: some hosts omit the Content-Length header or our static
                     * size in ModelData may be off by a few percent.  The 10 MiB guard
                     * prevents zero-byte / HTML error files from being marked as complete
                     * while keeping the logic tolerant to minor size mismatches.
                     */
                    val minReasonableSize = 10 * 1024 * 1024 // 10 MiB
                    val completed = modelFile.exists() && modelFile.length() >= minReasonableSize

                    if (completed && cause == null) {
                        updateModel(model.name) { it.copy(isDownloaded = true, downloadProgress = 1f, sizeBytes = modelFile.length()) }
                    } else {
                        // Delete incomplete file and reset progress
                        if (modelFile.exists()) modelFile.delete()
                        updateModel(model.name) {
                            it.copy(
                                isDownloaded = false,
                                downloadProgress = 0f,
                                downloadedBytes = 0L,
                                totalBytes = null,
                                downloadSpeedBytesPerSec = null
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

        // Delete partial file if exists
        val modelsDir = File(context.filesDir, "models")
        val file = File(modelsDir, model.localFileName())
        if (file.exists()) file.delete()

        // Reset UI state
        updateModel(model.name) {
            it.copy(
                downloadProgress = 0f,
                downloadedBytes = 0L,
                totalBytes = null,
                downloadSpeedBytesPerSec = null
            )
        }
    }

    fun deleteModel(model: LLMModel) {
        // Physically remove the file from the app's private storage so that
        // the model will no longer be detected as «downloaded» next time we
        // build the list (e.g. after an app restart).
        val modelsDir = File(context.filesDir, "models")
        val primaryFile = File(modelsDir, model.localFileName())
        val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")
        if (primaryFile.exists()) primaryFile.delete()
        if (legacyFile.exists()) legacyFile.delete()

        updateModel(model.name) {
            it.copy(
                isDownloaded = false,
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