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
    val supportsAudio: Boolean = false, // New: whether model supports audio input
    val supportsGpu: Boolean = false,
    val requirements: ModelRequirements,
    val contextWindowSize: Int = 2048, // Default context window size in tokens
    val modelFormat: String = "task", // "task", or "litertlm" (MediaPipe formats)
    var isDownloaded: Boolean = false,
    var isDownloading: Boolean = false, // New: whether a download is currently in progress for this model
    var isPaused: Boolean = false, // Whether download is paused
    var downloadProgress: Float = 0f,
    var downloadedBytes: Long = 0L,
    var totalBytes: Long? = null,
    var downloadSpeedBytesPerSec: Long? = null
)