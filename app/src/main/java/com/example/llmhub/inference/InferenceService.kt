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
            
            // Load new model
            val modelFile = File(context.filesDir, "models/${model.localFileName()}")
            if (!modelFile.exists()) {
                throw IllegalStateException("Model file not found: ${modelFile.absolutePath}")
            }
            
            withContext(Dispatchers.IO) {
                try {
                    // Configure LLM inference options with correct API
                    val options = LlmInference.LlmInferenceOptions.builder()
                        .setModelPath(modelFile.absolutePath)
                        .setMaxTokens(2048)
                        .setMaxTopK(40)  // Correct method name
                        .setPreferredBackend(LlmInference.Backend.GPU)  // Enable GPU acceleration
                        .build()
                    
                    // Create LLM inference engine
                    llmInference = LlmInference.createFromOptions(context, options)
                    
                    // Create session with proper configuration
                    session = LlmInferenceSession.createFromOptions(
                        llmInference!!,
                        LlmInferenceSession.LlmInferenceSessionOptions.builder()
                            .setTemperature(0.8f)
                            .build()
                    )
                    
                    currentModel = model
                    Log.d("MediaPipeInference", "Successfully loaded model: ${model.name}")
                    
                } catch (e: Exception) {
                    Log.e("MediaPipeInference", "Failed to load model: ${model.name}", e)
                    throw RuntimeException("Failed to load model: ${e.message}", e)
                }
            }
        }
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