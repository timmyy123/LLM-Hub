package com.example.llmhub.viewmodels

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llmhub.data.LLMModel
import com.example.llmhub.data.ModelData
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import com.example.llmhub.data.ModelDownloader
import io.ktor.client.HttpClient
import io.ktor.client.engine.android.Android
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.onCompletion
import android.app.Application
import androidx.lifecycle.AndroidViewModel
import java.io.File

class ModelDownloadViewModel(application: Application) : AndroidViewModel(application) {
    private val _models = MutableStateFlow<List<LLMModel>>(emptyList())
    val models: StateFlow<List<LLMModel>> = _models.asStateFlow()

    private val ktorClient = HttpClient(Android)
    private val modelDownloader = ModelDownloader(ktorClient, application.applicationContext)

    init {
        loadModels()
    }

    private fun loadModels() {
        _models.value = ModelData.models
    }

    fun downloadModel(model: LLMModel) {
        // Immediately flip UI into downloading state
        updateModel(model.name) { it.copy(downloadProgress = 0.0001f, downloadedBytes = 0L) }

        viewModelScope.launch {
            var latestStatus: com.example.llmhub.data.DownloadStatus? = null
            modelDownloader.downloadModel(model)
                .catch {
                    // Handle exceptions
                    updateModel(model.name) { it.copy(downloadProgress = 0f) }
                }
                .onCompletion { cause ->
                    val modelsDir = File(getApplication<Application>().filesDir, "models")
                    val modelFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")

                    val expectedBytes = latestStatus?.totalBytes ?: 0L
                    val completed = modelFile.exists() && expectedBytes > 0 && modelFile.length() >= (expectedBytes * 9 / 10)

                    if (completed && cause == null) {
                        updateModel(model.name) { it.copy(isDownloaded = true, downloadProgress = 1f) }
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
                    updateModel(model.name) {
                        it.copy(
                            sizeBytes = if (status.totalBytes > it.sizeBytes) status.totalBytes else it.sizeBytes,
                            downloadProgress = kotlin.math.min(0.999f, status.downloadedBytes.toFloat() / status.totalBytes),
                            downloadedBytes = status.downloadedBytes,
                            totalBytes = status.totalBytes,
                            downloadSpeedBytesPerSec = status.downloadSpeedBytesPerSec
                        )
                    }
                }
        }
    }

    fun deleteModel(model: LLMModel) {
        // Placeholder for delete logic
        updateModel(model.name) {
            it.copy(isDownloaded = false)
        }
    }

    private fun updateModel(modelName: String, updateAction: (LLMModel) -> LLMModel) {
        val currentModels = _models.value.toMutableList()
        val modelIndex = currentModels.indexOfFirst { it.name == modelName }
        if (modelIndex != -1) {
            val updatedModel = updateAction(currentModels[modelIndex])
            currentModels[modelIndex] = updatedModel
            _models.value = currentModels
        }
    }
} 