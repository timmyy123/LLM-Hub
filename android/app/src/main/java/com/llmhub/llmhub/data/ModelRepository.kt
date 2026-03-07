package com.llmhub.llmhub.data

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.google.gson.Gson
import java.io.File

import com.llmhub.llmhub.data.isModelFileValid
import com.llmhub.llmhub.data.localFileName

object ModelRepository {
    private const val TAG = "ModelRepository"
    private const val MODEL_PREFS = "model_prefs"
    private const val IMPORTED_MODELS_KEY = "imported_models"

    /**
     * Returns all models that are currently available on-device (bundled, downloaded, or imported)
     * and match the provided [predicate]. Only models whose files can be validated are returned.
     */
    fun getAvailableModels(
        context: Context,
        predicate: (LLMModel) -> Boolean = { true }
    ): List<LLMModel> {
        val downloaded = loadBundledAndDownloadedModels(context)
        val imported = loadImportedModels(context)

        return (downloaded + imported)
            .distinctBy { it.name }
            .filter(predicate)
    }

    private fun loadBundledAndDownloadedModels(context: Context): List<LLMModel> {
        val modelsDir = File(context.filesDir, "models")
        if (!modelsDir.exists()) {
            modelsDir.mkdirs()
        }

        return ModelData.models
            .filter { it.category != "embedding" }
            .mapNotNull { model -> resolveAvailableModel(context, modelsDir, model) }
    }

    private fun resolveAvailableModel(
        context: Context,
        modelsDir: File,
        model: LLMModel
    ): LLMModel? {
        var actualSize = model.sizeBytes

        val assetPath = if (model.url.startsWith("file://models/")) {
            model.url.removePrefix("file://")
        } else {
            "models/${model.localFileName()}"
        }

        // 1. Check bundled assets
        try {
            context.assets.open(assetPath).use { stream ->
                actualSize = stream.available().toLong()
                return model.copy(
                    isDownloaded = true,
                    sizeBytes = actualSize
                )
            }
        } catch (_: Exception) {
            // Not bundled, fall through to file checks
        }

        // 2. Check files directory
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
                return model.copy(
                    isDownloaded = true,
                    sizeBytes = primaryFile.length()
                )
            } else {
                Log.d(TAG, "Ignoring incomplete/invalid model file: ${primaryFile.absolutePath} sizeOk=$sizeOk valid=$valid")
            }
        }

        return null
    }

    private fun loadImportedModels(context: Context): List<LLMModel> {
        val prefs: SharedPreferences = context.getSharedPreferences(MODEL_PREFS, Context.MODE_PRIVATE)
        val importedJson = prefs.getString(IMPORTED_MODELS_KEY, null) ?: return emptyList()

        return try {
            val imported = Gson().fromJson(importedJson, Array<LLMModel>::class.java)?.toList().orEmpty()
            imported.filter { model ->
                // Filter out image generation models (qnn_npu, mnn_cpu) - those are for Image Generator only
                if (model.category == "qnn_npu" || model.category == "mnn_cpu") {
                    return@filter false
                }
                // Ensure the backing file still exists and is valid
                val modelsDir = File(context.filesDir, "models")
                val file = File(modelsDir, model.localFileName())
                val exists = file.exists()
                val valid = exists && isModelFileValid(file, model.modelFormat)
                if (!valid) {
                    Log.d(TAG, "Skipping stale imported model ${model.name}: exists=$exists valid=$valid")
                }
                exists && valid
            }.map { it.copy(isDownloaded = true) }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse imported models: ${e.message}")
            emptyList()
        }
    }
}
