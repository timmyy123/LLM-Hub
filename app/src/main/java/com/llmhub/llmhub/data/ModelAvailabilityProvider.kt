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

            // ONNX multi-file models: same logic as ChatViewModel.loadAvailableModelsSync (subdir, all files required)
            if (model.modelFormat == "onnx" && model.additionalFiles.isNotEmpty()) {
                val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                val onnxModelDir = File(modelsDir, modelDirName)
                Log.d("ModelAvailability", "Checking ONNX model: ${model.name}")
                Log.d("ModelAvailability", "  ONNX dir: ${onnxModelDir.absolutePath}, exists: ${onnxModelDir.exists()}")
                if (onnxModelDir.exists() && onnxModelDir.isDirectory) {
                    val files = onnxModelDir.listFiles() ?: emptyArray()
                    val fileCount = files.filter { it.length() > 0 }.size
                    val expectedFileCount = 1 + model.additionalFiles.size
                    val totalSize = files.sumOf { it.length() }
                    if (fileCount >= expectedFileCount) {
                        isAvailable = true
                        actualSize = totalSize
                        Log.d("ModelAvailability", "  ✓ ONNX model available: ${model.name} ($fileCount files)")
                    } else {
                        Log.d("ModelAvailability", "  ✗ ONNX incomplete: $fileCount/$expectedFileCount files")
                    }
                } else {
                    Log.d("ModelAvailability", "  ✗ ONNX dir does not exist: ${model.name}")
                }
            } else {
                val primaryFile = File(modelsDir, model.localFileName())
                val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")
                Log.d("ModelAvailability", "Checking for model: ${model.name} (format: ${model.modelFormat})")
                Log.d("ModelAvailability", "  Primary path: ${primaryFile.absolutePath}, exists: ${primaryFile.exists()}")
                Log.d("ModelAvailability", "  Legacy path: ${legacyFile.absolutePath}, exists: ${legacyFile.exists()}")

                if (!primaryFile.exists() && legacyFile.exists()) {
                    Log.d("ModelAvailability", "Renaming legacy file to primary name")
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
                    Log.d("ModelAvailability", "  File size: ${primaryFile.length()}, sizeOk: $sizeOk, valid: $valid")
                    if (sizeOk && valid) {
                        isAvailable = true
                        actualSize = primaryFile.length()
                        Log.d("ModelAvailability", "  ✓ Model available: ${model.name}")
                    } else {
                        Log.d("ModelAvailability", "  ✗ Model NOT available (size or validation failed): ${model.name}")
                    }
                } else {
                    Log.d("ModelAvailability", "  ✗ Model file does not exist: ${model.name}")
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
            // Filter out image generation models (qnn_npu, mnn_cpu) - those are for Image Generator only
            imported.filter { it.isDownloaded && it.category != "qnn_npu" && it.category != "mnn_cpu" }
        } catch (e: Exception) {
            Log.w("ModelAvailability", "Failed to load imported models: ${e.message}")
            emptyList()
        }
    }
}
