package com.llmhub.llmhub.data

import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.llmhub.llmhub.inference.UnifiedInferenceService

fun defaultConfigForModel(model: LLMModel): ModelConfig {
    val modelMaxCap = MediaPipeInferenceService.getMaxTokensForModelStatic(model)
    val isGemma3n = model.name.contains("Gemma-3n", ignoreCase = true)
    val isGemma4_12B = model.modelFormat == "litertlm" &&
        (model.name.contains("Gemma-4 12B", ignoreCase = true) || model.name.contains("Gemma 4 12B", ignoreCase = true))
    val isGemma4Small = model.modelFormat == "litertlm" &&
        (model.name.contains("Gemma-4 E2", ignoreCase = true) ||
            model.name.contains("Gemma-4 E4", ignoreCase = true))
    val isPhi4Mini = model.name.contains("Phi-4 Mini", ignoreCase = true)
    val supportsVision = model.supportsVision
    val effCap = if (supportsVision) minOf(modelMaxCap, 8192) else modelMaxCap
    val defaultCtx = minOf(4096, effCap)
    val defaultMax = minOf(4096, defaultCtx)
    val defaultUseGpu = when {
        model.modelFormat == "gguf" -> false
        isGemma4_12B -> true
        isPhi4Mini -> false
        else -> model.supportsGpu
    }
    return ModelConfig(
        maxTokens = defaultMax,
        topK = 64,
        topP = 0.95f,
        temperature = 1.0f,
        backend = if (defaultUseGpu) "GPU" else "CPU",
        deviceId = null,
        // Gemma-4 (E2B, E4B and 12B) starts with vision, audio and agent tools off; users switch them on.
        disableVision = isGemma3n || isGemma4Small || isGemma4_12B || !supportsVision,
        disableAudio = isGemma3n || isGemma4Small || isGemma4_12B,
        nGpuLayers = 999,
        enableThinking = true,
        systemPrompt = "",
        contextWindow = defaultCtx,
        agentToolsEnabled = !(isGemma4Small || isGemma4_12B)
    )
}

suspend fun loadModelWithSavedConfig(
    model: LLMModel,
    modelPrefs: ModelPreferences,
    inferenceService: InferenceService,
    disableVisionOverride: Boolean? = null,
    disableAudioOverride: Boolean? = null,
    backendOverride: LlmInference.Backend? = null,
    deviceIdOverride: String? = DEVICE_ID_NO_OVERRIDE,
    enableThinkingOverride: Boolean? = null,
    disableAgentTools: Boolean = false,
    contextWindowOverride: Int? = null,
    maxTokensOverride: Int? = null,
    nGpuLayersOverride: Int? = null,
    onConfigApplied: ((ModelConfig) -> Unit)? = null
): Boolean {
    val saved = modelPrefs.getModelConfig(model.name)
    val cfg = saved ?: defaultConfigForModel(model)

    val disableVision = disableVisionOverride ?: cfg.disableVision
    val disableAudio = disableAudioOverride ?: cfg.disableAudio
    val effectiveEnableThinking = enableThinkingOverride ?: cfg.enableThinking

    val effectiveContextWindow = contextWindowOverride?.takeIf { it > 0 }
        ?: cfg.contextWindow.takeIf { it > 0 }
        ?: maxTokensOverride?.takeIf { it > 0 }
        ?: cfg.maxTokens.takeIf { it > 0 }
        ?: model.contextWindowSize
    val effectiveMaxTokens = maxTokensOverride?.takeIf { it > 0 }
        ?: cfg.maxTokens.takeIf { it > 0 }
        ?: effectiveContextWindow
    val effectiveGpuLayers = nGpuLayersOverride ?: cfg.nGpuLayers

    inferenceService.setGenerationParameters(
        maxTokens = effectiveMaxTokens,
        topK = cfg.topK,
        topP = cfg.topP,
        temperature = cfg.temperature,
        nGpuLayers = effectiveGpuLayers,
        enableThinking = effectiveEnableThinking,
        contextWindow = effectiveContextWindow
    )
    (inferenceService as? UnifiedInferenceService)?.setAgentToolsEnabled(if (disableAgentTools) false else (cfg.agentToolsEnabled ?: false))

    val backend = backendOverride ?: cfg.backend?.let {
        try { LlmInference.Backend.valueOf(it) } catch (_: Exception) { null }
    }
    val deviceId = if (deviceIdOverride != DEVICE_ID_NO_OVERRIDE) deviceIdOverride else cfg.deviceId

    val effectiveCfg = cfg.copy(
        maxTokens = effectiveMaxTokens,
        contextWindow = effectiveContextWindow,
        nGpuLayers = effectiveGpuLayers,
        disableVision = disableVision,
        disableAudio = disableAudio
    )
    onConfigApplied?.invoke(effectiveCfg)

    Log.d("ModelLoader", "Loading ${model.name} (saved=${saved != null}): ctx=$effectiveContextWindow max=$effectiveMaxTokens vision=$disableVision audio=$disableAudio backend=${backend} deviceId=${deviceId}")

    return inferenceService.loadModel(
        model = model,
        preferredBackend = backend,
        disableVision = disableVision,
        disableAudio = disableAudio,
        deviceId = deviceId
    )
}

private const val DEVICE_ID_NO_OVERRIDE = "_NO_OVERRIDE_"
