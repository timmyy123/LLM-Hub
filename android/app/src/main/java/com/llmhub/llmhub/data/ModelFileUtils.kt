package com.llmhub.llmhub.data

import android.content.Context
import java.io.File

/**
 * Returns a stable filename to store the downloaded model for this LLMModel.
 * Supports both GGUF and MediaPipe .task formats based on the modelFormat field.
 * Prefer the last path segment of the URL (without query parameters) so that
 * future renames of the model's user-visible name do **not** break the
 * detection logic.
 */
fun LLMModel.localFileName(): String {
    val candidate = url.substringAfterLast('/')
        .substringBefore('?')
        .substringBefore('#')
    
    // Determine the appropriate extension based on model format
    val extension = when (modelFormat.lowercase()) {
        "task" -> ".task"
        "litertlm" -> ".litertlm"
        "tflite" -> ".tflite"
        "model" -> ".model"
        "bin" -> ".bin"
        "upscaler_bin" -> ".bin"
        "gguf" -> ".gguf"
        "onnx" -> ".onnx"
        else -> ".gguf" // Default fallback
    }
    
    // Use URL-derived filename if available, otherwise create from model name
    return if (candidate.isNotBlank() && !candidate.endsWith("/")) {
        // If the URL already has the correct extension, use it as-is
        if (candidate.endsWith(extension)) {
            candidate
        } else {
            // Replace any existing extension with the correct one
            val nameWithoutExt = candidate.substringBeforeLast('.')
            "${nameWithoutExt}${extension}"
        }
    } else {
        // Fallback to sanitized model name with appropriate extension
        "${name.replace(" ", "_").replace("[^a-zA-Z0-9_-]".toRegex(), "")}${extension}"
    }
} 

/**
 * GGUF vision models require an external mmproj file for image input in GenieX VLM mode.
 */
fun LLMModel.requiresExternalVisionProjector(): Boolean {
    return modelFormat.equals("gguf", ignoreCase = true) && supportsVision
}

/**
 * Returns true only when every file declared by a model is present and non-empty.
 *
 * Multi-file models are stored in a model-specific directory by ModelDownloader.
 * Checking only the primary file is unsafe: older app versions may have downloaded
 * that file before the rest of the model bundle was added to the manifest.
 */
fun LLMModel.hasCompleteDownloadedBundle(context: Context): Boolean {
    val modelsDir = File(context.filesDir, "models")
    val targetDir = if (additionalFiles.isNotEmpty()) {
        val safeName = name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
        File(modelsDir, safeName)
    } else {
        modelsDir
    }

    val primaryFile = File(targetDir, localFileName())
    if (!primaryFile.isFile || !isModelFileValid(primaryFile, modelFormat)) return false

    val additional = additionalFiles.map { urlOrPath ->
        val fileName = urlOrPath.substringAfterLast('/').substringBefore('?').substringBefore('#')
        if (fileName.isBlank()) return false
        File(targetDir, fileName)
    }
    if (additional.any { !it.isFile || it.length() <= 0L }) return false

    val downloadedBytes = primaryFile.length() + additional.sumOf { it.length() }
    return sizeBytes <= 0L || downloadedBytes >= (sizeBytes * 0.98).toLong()
}

/**
 * Mirrors GeniexInferenceService mmproj lookup logic to determine if vision can be enabled.
 */
fun LLMModel.hasDownloadedVisionProjector(context: Context): Boolean {
    if (!requiresExternalVisionProjector()) return true

    val modelsDir = File(context.filesDir, "models")
    val modelDirName = name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
    val modelDir = File(modelsDir, modelDirName)

    // 1) If this is an imported model and additionalFiles are defined (e.g. HuggingFace search import)
    if (additionalFiles.isNotEmpty()) {
        for (urlOrPath in additionalFiles) {
            val fileName = urlOrPath.substringAfterLast('/').substringBefore('?').substringBefore('#')
            if (File(modelDir, fileName).exists() || File(modelsDir, fileName).exists() ||
                File(modelDir, urlOrPath).exists() || File(modelsDir, urlOrPath).exists()) {
                return true
            }
        }
    }

    // 2) Check if any mmproj / projector exists in the model-specific directory
    if (modelDir.exists() && modelDir.isDirectory) {
        val projInDir = modelDir.listFiles { _, name ->
            name.endsWith(".gguf", ignoreCase = true) &&
                (name.contains("mmproj", ignoreCase = true) || name.contains("projector", ignoreCase = true))
        }
        if (!projInDir.isNullOrEmpty()) return true
    }

    // 3) Fallback: look for known projector entries in ModelData (for bundled/catalog models)
    val compatibleProjectors = ModelData.models.filter { candidate ->
        candidate.modelFormat.equals("gguf", ignoreCase = true) &&
            candidate.name.contains("Projector", ignoreCase = true) &&
            normalizeVisionPairBaseName(candidate.name) == normalizeVisionPairBaseName(name)
    }

    if (compatibleProjectors.isNotEmpty()) {
        val found = compatibleProjectors.any { projector ->
            val projectorFileName = projector.localFileName()
            File(modelDir, projectorFileName).exists() || File(modelsDir, projectorFileName).exists()
        }
        if (found) return true
    }

    // 4) Check root models directory for any matching projector file
    val rootProj = modelsDir.listFiles { _, name ->
        name.endsWith(".gguf", ignoreCase = true) &&
            (name.contains("mmproj", ignoreCase = true) || name.contains("projector", ignoreCase = true))
    }
    if (!rootProj.isNullOrEmpty()) {
        val modelCore = name.lowercase().replace(" ", "").replace(Regex("[^a-z0-9]"), "")
        return rootProj.any { f ->
            val fCore = f.nameWithoutExtension.lowercase().replace("mmproj", "").replace("projector", "").replace("vision", "").replace(Regex("[^a-z0-9]"), "")
            fCore.isNotEmpty() && (modelCore.contains(fCore) || fCore.contains(modelCore))
        }
    }

    return false
}

private fun normalizeVisionPairBaseName(modelName: String): String {
    return modelName
        .substringBefore(" (")
        .replace("Vision Projector", "", ignoreCase = true)
        .replace("Projector", "", ignoreCase = true)
        .trim()
        .lowercase()
}
