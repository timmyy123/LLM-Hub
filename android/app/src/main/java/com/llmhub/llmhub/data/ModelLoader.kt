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
    val isPhi4Mini = model.name.contains("Phi-4 Mini", ignoreCase = true)
    val supportsVision = model.supportsVision
    val effCap = if (supportsVision) minOf(modelMaxCap, 8192) else modelMaxCap
    val defaultCtx = minOf(4096, effCap)
    val defaultMax = minOf(4096, defaultCtx)
    val defaultUseGpu = when {
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
        disableVision = isGemma3n || !supportsVision,
        disableAudio = isGemma3n,
        nGpuLayers = 999,
        enableThinking = true,
        systemPrompt = "",
        contextWindow = defaultCtx,
        agentToolsEnabled = true
    )
}

suspend fun loadModelWithSavedConfig(
    model: LLMModel,
    modelPrefs: ModelPreferences,
    inferenceService: InferenceService,
    disableVisionOverride: Boolean? = null,
    disableAudioOverride: Boolean? = null,
    onConfigApplied: ((ModelConfig) -> Unit)? = null
): Boolean {
    val saved = modelPrefs.getModelConfig(model.name)
    val cfg = saved ?: defaultConfigForModel(model)

    val disableVision = disableVisionOverride ?: cfg.disableVision
    val disableAudio = disableAudioOverride ?: cfg.disableAudio

    inferenceService.setGenerationParameters(
        maxTokens = cfg.maxTokens,
        topK = cfg.topK,
        topP = cfg.topP,
        temperature = cfg.temperature,
        nGpuLayers = cfg.nGpuLayers,
        enableThinking = cfg.enableThinking,
        contextWindow = cfg.contextWindow
    )
    (inferenceService as? UnifiedInferenceService)?.setAgentToolsEnabled(cfg.agentToolsEnabled ?: false)

    val backend = cfg.backend?.let {
        try { LlmInference.Backend.valueOf(it) } catch (_: Exception) { null }
    }

    val effectiveCfg = cfg.copy(disableVision = disableVision, disableAudio = disableAudio)
    onConfigApplied?.invoke(effectiveCfg)

    Log.d("ModelLoader", "Loading ${model.name} (saved=${saved != null}): ctx=${cfg.contextWindow} max=${cfg.maxTokens} vision=$disableVision audio=$disableAudio backend=${cfg.backend}")

    return inferenceService.loadModel(
        model = model,
        preferredBackend = backend,
        disableVision = disableVision,
        disableAudio = disableAudio,
        deviceId = cfg.deviceId
    )
}
