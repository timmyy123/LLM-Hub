package com.llmhub.llmhub.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.llmhub.llmhub.data.LLMModel
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow

/**
 * Unified Inference Service that routes requests to the appropriate backend Service
 * (MediaPipe or ONNX) based on the model format.
 */
class UnifiedInferenceService(private val context: Context) : InferenceService {

    private val mediaPipeService by lazy { MediaPipeInferenceService(context) }
    private val liteRtLmService by lazy { LiteRtLmInferenceService(context) }
    private val onnxService by lazy { OnnxInferenceService(context) }
    private val geniexService by lazy { GeniexInferenceService(context) }
    private val llamaCppService by lazy { LlamaCppInferenceService(context) }
    
    private var currentService: InferenceService = mediaPipeService
    private var currentModel: LLMModel? = null
    private var isVisionDisabled: Boolean = false
    private var isAudioDisabled: Boolean = false

    override suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend?, deviceId: String?): Boolean =
        loadModel(model, preferredBackend, disableVision = false, disableAudio = false, deviceId = deviceId)

    override suspend fun loadModel(
        model: LLMModel, 
        preferredBackend: LlmInference.Backend?, 
        disableVision: Boolean, 
        disableAudio: Boolean,
        deviceId: String?
    ): Boolean {
        val modelPrefs = com.llmhub.llmhub.data.ModelPreferences(context)
        val cfg = modelPrefs.getModelConfig(model.name)
        val finalBackend = if (cfg?.backend != null) {
            try { LlmInference.Backend.valueOf(cfg.backend) } catch (_: Exception) { preferredBackend }
        } else {
            preferredBackend
        }
        val finalDeviceId = cfg?.deviceId ?: deviceId
        val finalDisableVision = disableVision
        val finalDisableAudio = disableAudio

        val isGemma4 = model.name.contains("Gemma-4", ignoreCase = true)
        val targetService = when (model.modelFormat) {
            "onnx" -> onnxService
            "gguf" -> if (currentService === llamaCppService) llamaCppService else geniexService
            "litertlm" -> if (isGemma4 || model.source == "Custom") liteRtLmService else mediaPipeService
            else -> mediaPipeService
        }

        // Same service and same model already loaded with same backend and same vision/audio modality settings: skip reload
        if (currentService == targetService) {
            val loaded = currentService.getCurrentlyLoadedModel()
            val currentBackend = currentService.getCurrentlyLoadedBackend()
            if (loaded?.name == model.name && 
                (finalBackend == null || finalBackend == currentBackend) &&
                finalDisableVision == isVisionDisabled &&
                finalDisableAudio == isAudioDisabled
            ) {
                currentModel = model
                updateAgentTools(model)
                return true
            }
        }

        // If switching services, or configuration changed, unload the old one
        if (currentModel != null) {
            currentService.unloadModel()
        }

        try {
            val loadedService = if (model.modelFormat == "gguf") {
                loadGgufWithCpuFallback(
                    model,
                    finalBackend,
                    finalDisableVision,
                    finalDisableAudio,
                    finalDeviceId,
                )
            } else {
                val success = targetService.loadModel(
                    model,
                    finalBackend,
                    finalDisableVision,
                    finalDisableAudio,
                    finalDeviceId,
                )
                if (success) targetService else null
            }
            val success = loadedService != null
            if (!success) {
                currentModel = null
                throw AllBackendsFailedException("All compatible backends failed to load model '${model.name}'")
            }
            currentService = loadedService
            currentModel = model
            isVisionDisabled = loadedService.isVisionCurrentlyDisabled()
            isAudioDisabled = loadedService.isAudioCurrentlyDisabled()
            updateAgentTools(model)
            return true
        } catch (e: AllBackendsFailedException) {
            currentModel = null
            throw e
        } catch (e: Exception) {
            Log.e("UnifiedInferenceService", "Failed to load model '${model.name}'", e)
            currentModel = null
            throw AllBackendsFailedException("Failed to load model '${model.name}': ${e.message}")
        }
    }

    private suspend fun loadGgufWithCpuFallback(
        model: LLMModel,
        backend: LlmInference.Backend?,
        disableVision: Boolean,
        disableAudio: Boolean,
        deviceId: String?,
    ): InferenceService? {
        if (geniexService.isAvailable()) {
            try {
                if (geniexService.loadModel(model, backend, disableVision, disableAudio, deviceId)) {
                    Log.i("UnifiedInferenceService", "Loaded '${model.name}' with GenieX")
                    return geniexService
                }
                Log.w("UnifiedInferenceService", "GenieX returned failure; trying llama.cpp CPU fallback")
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (fatal: VirtualMachineError) {
                throw fatal
            } catch (error: Throwable) {
                // Includes native linkage failures such as a missing vendor OpenCL library.
                Log.w("UnifiedInferenceService", "GenieX failed; trying llama.cpp CPU fallback", error)
            }
            runCatching { geniexService.unloadModel() }
        } else {
            Log.w("UnifiedInferenceService", "GenieX is unavailable; trying llama.cpp CPU fallback")
        }

        return if (llamaCppService.loadModel(
                model,
                LlmInference.Backend.CPU,
                disableVision = true,
                disableAudio = true,
                deviceId = null,
            )
        ) {
            Log.i("UnifiedInferenceService", "Loaded '${model.name}' with llama.cpp b10603 CPU fallback")
            llamaCppService
        } else {
            Log.e("UnifiedInferenceService", "Both GenieX and llama.cpp failed for '${model.name}'")
            null
        }
    }

    override suspend fun unloadModel() {
        currentService.unloadModel()
        currentModel = null
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        return currentService.generateResponse(prompt, model)
    }

    override suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String> {
        return currentService.generateResponseStream(prompt, model)
    }

    override suspend fun generateResponseStreamWithSession(
        prompt: String,
        model: LLMModel,
        chatId: String,
        images: List<Bitmap>,
        audioData: ByteArray?,
        webSearchEnabled: Boolean,
        imagePaths: List<String>
    ): Flow<String> {
        return currentService.generateResponseStreamWithSession(prompt, model, chatId, images, audioData, webSearchEnabled, imagePaths)
    }

    override suspend fun resetChatSession(chatId: String) {
        currentService.resetChatSession(chatId)
    }

    override suspend fun onCleared() {
        mediaPipeService.onCleared()
        liteRtLmService.onCleared()
        onnxService.onCleared()
        if ((geniexService as? com.llmhub.llmhub.inference.GeniexInferenceService)?.isAvailable() == true) {
            geniexService.onCleared()
        }
        llamaCppService.onCleared()
    }

    override fun getCurrentlyLoadedModel(): LLMModel? {
        return currentService.getCurrentlyLoadedModel()
    }

    override fun getCurrentlyLoadedBackend(): LlmInference.Backend? {
        return currentService.getCurrentlyLoadedBackend()
    }

    override fun getMemoryWarningForImages(images: List<Bitmap>): String? {
        return currentService.getMemoryWarningForImages(images)
    }

    override fun wasSessionRecentlyReset(chatId: String): Boolean {
        return currentService.wasSessionRecentlyReset(chatId)
    }

    override fun setGenerationParameters(maxTokens: Int?, topK: Int?, topP: Float?, temperature: Float?, nGpuLayers: Int?, enableThinking: Boolean?, contextWindow: Int?) {
        mediaPipeService.setGenerationParameters(maxTokens, topK, topP, temperature, nGpuLayers, enableThinking, contextWindow)
        liteRtLmService.setGenerationParameters(maxTokens, topK, topP, temperature, nGpuLayers, enableThinking, contextWindow)
        onnxService.setGenerationParameters(maxTokens, topK, topP, temperature, nGpuLayers, enableThinking, contextWindow)
        if ((geniexService as? com.llmhub.llmhub.inference.GeniexInferenceService)?.isAvailable() == true) {
            geniexService.setGenerationParameters(maxTokens, topK, topP, temperature, nGpuLayers, enableThinking, contextWindow)
        }
        llamaCppService.setGenerationParameters(maxTokens, topK, topP, temperature, nGpuLayers, enableThinking, contextWindow)
    }

    override fun isVisionCurrentlyDisabled(): Boolean {
        return currentService.isVisionCurrentlyDisabled()
    }

    override fun isAudioCurrentlyDisabled(): Boolean {
        return currentService.isAudioCurrentlyDisabled()
    }

    override fun isGpuBackendEnabled(): Boolean {
        return currentService.isGpuBackendEnabled()
    }

    override fun isNpuBackendEnabled(): Boolean {
        return currentService.isNpuBackendEnabled()
    }

    override fun getEffectiveMaxTokens(model: LLMModel): Int {
        return when (model.modelFormat) {
            "onnx" -> onnxService.getEffectiveMaxTokens(model)
            "gguf" -> if (currentService === llamaCppService) llamaCppService.getEffectiveMaxTokens(model) else geniexService.getEffectiveMaxTokens(model)
            "litertlm" -> liteRtLmService.getEffectiveMaxTokens(model)
            else -> mediaPipeService.getEffectiveMaxTokens(model)
        }
    }

    /**
     * Activate or deactivate the Gemma-4 agent skills toolset.
     * Tools are enabled only for Gemma-4 models because they are specifically trained
     * for function calling via the LiteRT-LM SDK. All other models get tools cleared.
     */
    private var agentToolsEnabled: Boolean = true

    fun setAgentToolsEnabled(enabled: Boolean) {
        agentToolsEnabled = enabled
        currentModel?.let { updateAgentTools(it) }
    }

    private fun updateAgentTools(model: LLMModel) {
        if (agentToolsEnabled && model.modelFormat == "litertlm") {
            liteRtLmService.setAgentTools(ChatAgentSkillsTools(context))
        } else {
            liteRtLmService.setAgentTools(null)
        }
    }
}
