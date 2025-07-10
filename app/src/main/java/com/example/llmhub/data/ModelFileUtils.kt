package com.example.llmhub.data

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
        "gguf" -> ".gguf" 
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