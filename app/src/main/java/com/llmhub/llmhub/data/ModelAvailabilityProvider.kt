package com.llmhub.llmhub.data

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

object ModelAvailabilityProvider {
    private const val PREFS_NAME = "model_prefs"
    private const val IMPORTED_MODELS_KEY = "imported_models"
    private val gson = Gson()

    suspend fun loadAvailableModels(context: Context): List<LLMModel> = withContext(Dispatchers.IO) {
        val baseModels = ModelData.models
            .filter { it.category != "embedding" }
            .mapNotNull { model ->
                resolveModelFromStorage(context, model)
            }

        val importedModels = loadImportedModels(context)
        (baseModels + importedModels)
            .distinctBy { it.name }
            .sortedBy { it.name.lowercase() }
    }

    private fun resolveModelFromStorage(context: Context, model: LLMModel): LLMModel? {
        var isAvailable = false
        var actualSize = model.sizeBytes

        val assetPath = if (model.url.startsWith("file://models/")) {
            model.url.removePrefix("file://")
        } else {
            "models/${model.localFileName()}"
        }

        try {
            context.assets.open(assetPath).use { inputStream ->
                actualSize = inputStream.available().toLong()
                isAvailable = true
                Log.d("ModelAvailability", "Found asset model: $assetPath")
            }
        } catch (_: Exception) {
            val modelsDir = File(context.filesDir, "models")
            val primaryFile = File(modelsDir, model.localFileName())
            val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")

            if (!primaryFile.exists() && legacyFile.exists()) {
                legacyFile.renameTo(primaryFile)
            }

            if (primaryFile.exists()) {
                val sizeKnown = model.sizeBytes > 0
                val sizeOk = if (sizeKnown) {
                    primaryFile.length() >= (model.sizeBytes * 0.98).toLong()
                } else {
                    primaryFile.length() >= 10L * 1024 * 1024
                }
                val valid = isModelFileValid(primaryFile, model.modelFormat)
                if (sizeOk && valid) {
                    isAvailable = true
                    actualSize = primaryFile.length()
                }
            }
        }

        return if (isAvailable) {
            model.copy(isDownloaded = true, sizeBytes = actualSize)
        } else {
            null
        }
    }

    private fun loadImportedModels(context: Context): List<LLMModel> {
        return try {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val json = prefs.getString(IMPORTED_MODELS_KEY, null) ?: return emptyList()
            val imported = gson.fromJson(json, Array<LLMModel>::class.java)?.toList().orEmpty()
            imported.filter { it.isDownloaded }
        } catch (e: Exception) {
            Log.w("ModelAvailability", "Failed to load imported models: ${e.message}")
            emptyList()
        }
    }
}
