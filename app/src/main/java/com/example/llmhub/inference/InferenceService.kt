package com.example.llmhub.inference

import android.content.Context
import com.example.llmhub.data.LLMModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import androidx.annotation.Keep
import java.io.File
import com.example.llmhub.data.localFileName
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import android.util.Log

/**
 * Interface for a service that can run model inference.
 */
interface InferenceService {
    suspend fun generateResponse(prompt: String, model: LLMModel): String
    suspend fun generateResponseStream(prompt: String, model: LLMModel): SharedFlow<String>
    suspend fun onCleared()
}

/**
 * MediaPipe-based implementation of InferenceService that supports GPU acceleration
 * and real-time streaming inference.
 */
class MediaPipeInferenceService(private val context: Context) : InferenceService {
    
    private var llmInference: LlmInference? = null
    private var session: LlmInferenceSession? = null
    private var currentModel: LLMModel? = null
    
    private val responseFlow = MutableSharedFlow<String>(extraBufferCapacity = 100)
    
    private suspend fun ensureModelLoaded(model: LLMModel) {
        if (currentModel?.name != model.name) {
            // Release existing model
            session?.close()
            llmInference?.close()
            
            withContext(Dispatchers.IO) {
                try {
                    // Determine model path based on URL format
                    val modelAssetPath = if (model.url.startsWith("file://models/")) {
                        // Local asset reference: file://models/gemma3-1b-it-int4.task -> models/gemma3-1b-it-int4.task
                        model.url.removePrefix("file://")
                    } else {
                        // Try to find model in assets based on filename
                        "models/${model.localFileName()}"
                    }
                    
                    Log.d("MediaPipeInference", "Loading model from assets: $modelAssetPath")
                    
                    // Check if model exists in assets folder
                    try {
                        context.assets.open(modelAssetPath).use { 
                            // File exists in assets
                        }
                    } catch (e: Exception) {
                        // Try alternative path: check if model exists in files directory
                        val modelFile = File(context.filesDir, modelAssetPath)
                        if (modelFile.exists()) {
                            Log.d("MediaPipeInference", "Model found in files directory: ${modelFile.absolutePath}")
                            loadModelFromFile(modelFile, model)
                            return@withContext
                        } else {
                            throw IllegalStateException("Model not found in assets or files: $modelAssetPath")
                        }
                    }
                    
                    // Load model from assets using MediaPipe's direct asset loading
                    loadModelFromAssets(modelAssetPath, model)
                    
                } catch (e: Exception) {
                    Log.e("MediaPipeInference", "Failed to load model: ${model.name}", e)
                    throw RuntimeException("Failed to load model: ${e.message}", e)
                }
            }
        }
    }
    
    private suspend fun loadModelFromAssets(assetPath: String, model: LLMModel) {
        // MediaPipe LLM Inference doesn't support direct asset loading
        // Copy from assets to files directory first
        val modelFile = File(context.filesDir, "models/${model.localFileName()}")
        
        if (!modelFile.exists()) {
            // Copy from assets to files directory
            modelFile.parentFile?.mkdirs()
            context.assets.open(assetPath).use { inputStream ->
                modelFile.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        
        // Now load from file
        loadModelFromFile(modelFile, model)
    }
    
    private suspend fun loadModelFromFile(modelFile: File, model: LLMModel) {
        // Configure LLM inference options for file loading
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelFile.absolutePath)  // Use file path
            .setMaxTokens(2048)
            .setMaxTopK(40)
            .setPreferredBackend(LlmInference.Backend.GPU)  // Enable GPU acceleration
            .build()
        
        // Create LLM inference engine from file
        llmInference = LlmInference.createFromOptions(context, options)
        
        // Create session with proper configuration
        session = LlmInferenceSession.createFromOptions(
            llmInference!!,
            LlmInferenceSession.LlmInferenceSessionOptions.builder()
                .setTemperature(0.8f)
                .build()
        )
        
        currentModel = model
        Log.d("MediaPipeInference", "Successfully loaded model from file: ${model.name}")
    }
    
    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        ensureModelLoaded(model)
        
        return withContext(Dispatchers.IO) {
            try {
                val responseBuilder = StringBuilder()
                var isComplete = false
                
                // Add query to session
                session!!.addQueryChunk(prompt)
                
                // Generate response synchronously
                session!!.generateResponseAsync { partialResult, done ->
                    responseBuilder.append(partialResult)
                    if (done) {
                        isComplete = true
                    }
                }
                
                // Wait for completion (simple polling)
                while (!isComplete) {
                    Thread.sleep(10)
                }
                
                responseBuilder.toString()
                
            } catch (e: Exception) {
                Log.e("MediaPipeInference", "Inference failed", e)
                "Error: ${e.message}"
            }
        }
    }
    
    override suspend fun generateResponseStream(prompt: String, model: LLMModel): SharedFlow<String> {
        ensureModelLoaded(model)
        
        withContext(Dispatchers.IO) {
            try {
                // Add query to session
                session!!.addQueryChunk(prompt)
                
                // Generate streaming response
                session!!.generateResponseAsync { partialResult, done ->
                    responseFlow.tryEmit(partialResult)
                    if (done) {
                        // Signal completion
                        responseFlow.tryEmit("[DONE]")
                    }
                }
                
            } catch (e: Exception) {
                Log.e("MediaPipeInference", "Streaming inference failed", e)
                responseFlow.tryEmit("Error: ${e.message}")
                responseFlow.tryEmit("[DONE]")
            }
        }
        
        return responseFlow
    }
    
    override suspend fun onCleared() {
        withContext(Dispatchers.IO) {
            try {
                session?.close()
                llmInference?.close()
                session = null
                llmInference = null
                currentModel = null
                Log.d("MediaPipeInference", "Resources released")
            } catch (e: Exception) {
                Log.e("MediaPipeInference", "Error releasing resources", e)
            }
        }
    }
} 