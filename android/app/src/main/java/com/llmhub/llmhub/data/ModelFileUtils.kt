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
 * GGUF vision models require an external mmproj file for image input in Nexa VLM mode.
 */
fun LLMModel.requiresExternalVisionProjector(): Boolean {
    return modelFormat.equals("gguf", ignoreCase = true) && supportsVision
}

/**
 * Mirrors NexaInferenceService mmproj lookup logic to determine if vision can be enabled.
 */
fun LLMModel.hasDownloadedVisionProjector(context: Context): Boolean {
    if (!requiresExternalVisionProjector()) return true

    val modelsDir = File(context.filesDir, "models")
    val modelDirName = name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
    val modelDir = File(modelsDir, modelDirName)

    // 1) If this is an imported model and the importer recorded additionalFiles, check them first.
    if (additionalFiles.isNotEmpty()) {
        additionalFiles.forEach { fname ->
            if (File(modelDir, fname).exists() || File(modelsDir, fname).exists()) return true
        }
    }

    // 2) Fallback: look for known projector entries in ModelData (for bundled/downloaded models)
    val compatibleProjectors = ModelData.models.filter { candidate ->
        candidate.modelFormat.equals("gguf", ignoreCase = true) &&
            candidate.name.contains("Projector", ignoreCase = true) &&
            normalizeVisionPairBaseName(candidate.name) == normalizeVisionPairBaseName(name)
    }

    if (compatibleProjectors.isEmpty()) return false

    return compatibleProjectors.any { projector ->
        val projectorFileName = projector.localFileName()
        File(modelDir, projectorFileName).exists() || File(modelsDir, projectorFileName).exists()
    }
}

private fun normalizeVisionPairBaseName(modelName: String): String {
    return modelName
        .substringBefore(" (")
        .replace("Vision Projector", "", ignoreCase = true)
        .replace("Projector", "", ignoreCase = true)
        .trim()
        .lowercase()
}
