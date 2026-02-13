package com.llmhub.llmhub.inference

import android.content.Context
import android.graphics.Bitmap
import com.llmhub.llmhub.data.LLMModel
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject

/**
 * Unified Inference Service that routes requests to the appropriate backend Service
 * (MediaPipe or ONNX) based on the model format.
 */
class UnifiedInferenceService(private val context: Context) : InferenceService {

    private val mediaPipeService = MediaPipeInferenceService(context)
    private val onnxService = OnnxInferenceService(context)
    private val nexaService = NexaInferenceService(context)
    
    private var currentService: InferenceService = mediaPipeService
    private var currentModel: LLMModel? = null

    override suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend?, deviceId: String?): Boolean {
        // Determine which service to use
        val newService = when (model.modelFormat) {
            "onnx" -> onnxService
            "gguf" -> if ((nexaService as? com.llmhub.llmhub.inference.NexaInferenceService)?.isAvailable() == true) nexaService else mediaPipeService
            else -> mediaPipeService
        }

        // Same service and same model already loaded with same backend: skip reload (honor user's CPU/GPU choice on next load)
        if (currentService == newService) {
            val loaded = currentService.getCurrentlyLoadedModel()
            val currentBackend = currentService.getCurrentlyLoadedBackend()
            if (loaded?.name == model.name && (preferredBackend == null || preferredBackend == currentBackend)) {
                currentModel = model
                return true
            }
        }

        // If switching services, unload the old one
        if (currentService != newService && currentModel != null) {
            currentService.unloadModel()
        }

        currentService = newService
        currentModel = model
        
        return currentService.loadModel(model, preferredBackend, deviceId)
    }

    override suspend fun loadModel(
        model: LLMModel, 
        preferredBackend: LlmInference.Backend?, 
        disableVision: Boolean, 
        disableAudio: Boolean,
        deviceId: String?
    ): Boolean {
         val newService = when (model.modelFormat) {
            "onnx" -> onnxService
            "gguf" -> if ((nexaService as? com.llmhub.llmhub.inference.NexaInferenceService)?.isAvailable() == true) nexaService else mediaPipeService
            else -> mediaPipeService
        }

        // Same service and same model already loaded with same backend: skip reload (honor user's CPU/GPU choice on next load)
        if (currentService == newService) {
            val loaded = currentService.getCurrentlyLoadedModel()
            val currentBackend = currentService.getCurrentlyLoadedBackend()
            if (loaded?.name == model.name && (preferredBackend == null || preferredBackend == currentBackend)) {
                currentModel = model
                return true
            }
        }

        // If switching services, unload the old one
        if (currentService != newService && currentModel != null) {
            currentService.unloadModel()
        }

        currentService = newService
        currentModel = model
        
        return currentService.loadModel(model, preferredBackend, disableVision, disableAudio, deviceId)
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
        onnxService.onCleared()
        if ((nexaService as? com.llmhub.llmhub.inference.NexaInferenceService)?.isAvailable() == true) {
            nexaService.onCleared()
        }
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

    override fun setGenerationParameters(maxTokens: Int?, topK: Int?, topP: Float?, temperature: Float?) {
        mediaPipeService.setGenerationParameters(maxTokens, topK, topP, temperature)
        onnxService.setGenerationParameters(maxTokens, topK, topP, temperature)
        if ((nexaService as? com.llmhub.llmhub.inference.NexaInferenceService)?.isAvailable() == true) {
            nexaService.setGenerationParameters(maxTokens, topK, topP, temperature)
        }
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

    override fun getEffectiveMaxTokens(model: LLMModel): Int {
        return when (model.modelFormat) {
            "onnx" -> onnxService.getEffectiveMaxTokens(model)
            "gguf" -> if ((nexaService as? com.llmhub.llmhub.inference.NexaInferenceService)?.isAvailable() == true) nexaService.getEffectiveMaxTokens(model) else mediaPipeService.getEffectiveMaxTokens(model)
            else -> mediaPipeService.getEffectiveMaxTokens(model)
        }
    }
}
