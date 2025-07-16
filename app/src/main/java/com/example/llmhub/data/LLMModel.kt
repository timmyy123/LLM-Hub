package com.llmhub.llmhub.data

data class ModelRequirements(
    val minRamGB: Int,
    val recommendedRamGB: Int
)

data class LLMModel(
    val name: String,
    val description: String,
    val url: String,
    val category: String,
    val sizeBytes: Long,
    val source: String,
    val supportsVision: Boolean,
    val supportsGpu: Boolean = false,
    val requirements: ModelRequirements,
    val contextWindowSize: Int = 2048, // Default context window size in tokens
    val modelFormat: String = "gguf", // "gguf" or "task" (MediaPipe format)
    var isDownloaded: Boolean = false,
    var downloadProgress: Float = 0f,
    var downloadedBytes: Long = 0L,
    var totalBytes: Long? = null,
    var downloadSpeedBytesPerSec: Long? = null
) 