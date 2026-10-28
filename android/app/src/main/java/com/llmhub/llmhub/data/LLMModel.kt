package com.llmhub.llmhub.data

import android.content.SharedPreferences

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
    val supportsThinking: Boolean = false,
    val supportsGpu: Boolean = false,
    val supportsMtp: Boolean = true,
    val requirements: ModelRequirements,
    val contextWindowSize: Int = 2048, // Default context window size in tokens
    val modelFormat: String = "task", // "task", "litertlm" (MediaPipe formats), or "onnx" (ONNX Runtime)
    val additionalFiles: List<String> = emptyList(), // Additional files required (for ONNX models that have separate data files)
    var isDownloaded: Boolean = false,
    var isDownloading: Boolean = false, // New: whether a download is currently in progress for this model
    var isExtracting: Boolean = false, // Whether ZIP extraction is in progress
    var isPaused: Boolean = false, // Whether download is paused
    var downloadProgress: Float = 0f,
    var downloadedBytes: Long = 0L,
    var totalBytes: Long? = null,
    var downloadSpeedBytesPerSec: Long? = null
)

fun LLMModel.hasNativeVoiceSupport(): Boolean {
    return this.supportsAudio && 
           (this.name.contains("Gemma-3n", ignoreCase = true) ||
            this.name.contains("Gemma-4 E2", ignoreCase = true) ||
            this.name.contains("Gemma-4 E4", ignoreCase = true))
}

/** LiquidAI's LFM3 family: thinks in a <think> block by default and can be switched off per request. */
fun LLMModel.isLfm3(): Boolean = name.contains("lfm3", ignoreCase = true)

/**
 * A feature's thinking switch outside AI Chat. LFM3 starts with thinking off and remembers its own choice per
 * model; every other model keeps the feature's existing [key] and [default].
 */
fun SharedPreferences.featureThinking(model: LLMModel?, key: String, default: Boolean): Boolean =
    if (model?.isLfm3() == true) getBoolean("lfm3_thinking_${model.name}", false) else getBoolean(key, default)

fun SharedPreferences.Editor.putFeatureThinking(model: LLMModel?, key: String, enabled: Boolean): SharedPreferences.Editor =
    if (model?.isLfm3() == true) putBoolean("lfm3_thinking_${model.name}", enabled) else putBoolean(key, enabled)