package com.example.llmhub.inference

import android.content.Context
import com.example.llmhub.data.LLMModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import androidx.annotation.Keep
import java.io.File
import com.example.llmhub.data.localFileName
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import android.util.Log
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn

/**
 * Interface for a service that can run model inference.
 */
interface InferenceService {
    suspend fun loadModel(model: LLMModel): Boolean
    suspend fun generateResponse(prompt: String, model: LLMModel): String
    suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String>
    suspend fun generateResponseStreamWithSession(prompt: String, model: LLMModel, chatId: String): Flow<String>
    suspend fun resetChatSession(chatId: String)
    suspend fun onCleared()
    fun getCurrentlyLoadedModel(): LLMModel?
}

/**
 * Data class to hold MediaPipe LLM inference engine and session
 */
data class LlmModelInstance(val engine: LlmInference, var session: LlmInferenceSession)

/**
 * MediaPipe-based implementation of InferenceService following Google AI Edge Gallery pattern.
 * Uses a single session approach: one session per model that gets reset as needed.
 */
class MediaPipeInferenceService(private val context: Context) : InferenceService {
    
    private var modelInstance: LlmModelInstance? = null
    private var currentModel: LLMModel? = null
    
    // Mutex to prevent race conditions during session operations
    private val sessionMutex = Mutex()
    
    companion object {
        private const val TAG = "MediaPipeInference"
        private const val DEFAULT_MAX_TOKENS = 1024
        private const val DEFAULT_TOP_K = 40
        private const val DEFAULT_TOP_P = 0.8f
        private const val DEFAULT_TEMPERATURE = 0.8f
        
        /**
         * Determine the correct max tokens based on model's context window size.
         * The maxTokens must not exceed the model's internal cache size.
         * For models with "ekvXXXX" in the filename, XXXX is the cache limit.
         */
        private fun getMaxTokensForModel(model: LLMModel): Int {
            // Use the model's context window size, but respect cache limitations
            val contextWindowSize = model.contextWindowSize
            
            // Extract cache size from model URL if available (e.g., ekv2048 means 2048 token cache)
            val cacheSize = extractCacheSizeFromUrl(model.url) ?: contextWindowSize
            
            // Use the smaller of context window size or cache size
            val maxTokens = minOf(contextWindowSize, cacheSize)
            
            Log.d(TAG, "Model: ${model.name}")
            Log.d(TAG, "Context window: $contextWindowSize, Cache size: $cacheSize, Max tokens: $maxTokens")
            return maxTokens
        }
        
        /**
         * Extract cache size from model URL (e.g., ekv2048 -> 2048)
         */
        private fun extractCacheSizeFromUrl(url: String): Int? {
            val ekvPattern = Regex("ekv(\\d+)")
            val match = ekvPattern.find(url)
            return match?.groupValues?.get(1)?.toIntOrNull()
        }
    }

    override suspend fun loadModel(model: LLMModel): Boolean {
        return try {
            ensureModelLoaded(model)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${e.message}", e)
            false
        }
    }

    override suspend fun resetChatSession(chatId: String) {
        sessionMutex.withLock {
            Log.d(TAG, "Resetting session for chat $chatId")
            
            val instance = modelInstance ?: return@withLock
            
            // Close current session and create a new one (Gallery approach)
            try {
                // First try to cancel any ongoing generation
                try {
                    instance.session.cancelGenerateResponseAsync()
                    Log.d(TAG, "Cancelled ongoing generation")
                } catch (e: Exception) {
                    Log.d(TAG, "No ongoing generation to cancel or cancel failed: ${e.message}")
                }
                
                // Small delay to let cancellation complete
                delay(100)
                
                instance.session.close()
                Log.d(TAG, "Closed existing session")
                
                // Create new session with same options
                val newSession = LlmInferenceSession.createFromOptions(
                    instance.engine,
                    LlmInferenceSession.LlmInferenceSessionOptions.builder()
                        .setTopK(DEFAULT_TOP_K)
                        .setTopP(DEFAULT_TOP_P)
                        .setTemperature(DEFAULT_TEMPERATURE)
                        .build()
                )
                instance.session = newSession
                Log.d(TAG, "Created new session for chat $chatId")
                
                // Give MediaPipe time to clean up (Gallery uses 500ms)
                delay(500)
                
            } catch (e: Exception) {
                Log.e(TAG, "Error resetting session for chat $chatId: ${e.message}", e)
                // On error, try to recreate the entire model instance
                try {
                    currentModel?.let { model ->
                        loadModelFromPath(model)
                    }
                } catch (retryException: Exception) {
                    Log.e(TAG, "Failed to recreate model instance: ${retryException.message}", retryException)
                }
            }
            
            Log.d(TAG, "Session reset completed for chat $chatId")
        }
    }

    private suspend fun ensureModelLoaded(model: LLMModel) {
        if (currentModel?.name != model.name) {
            Log.d(TAG, "Model changed from ${currentModel?.name} to ${model.name}, reloading")
            
            // Release existing model instance when switching models
            modelInstance?.let { instance ->
                try {
                    instance.session.close()
                } catch (e: Exception) {
                    Log.w(TAG, "Error closing session: ${e.message}")
                }
                try {
                    instance.engine.close()
                } catch (e: Exception) {
                    Log.w(TAG, "Error closing engine: ${e.message}")
                }
            }
            
            withContext(Dispatchers.IO) {
                loadModelFromPath(model)
            }
        }
    }

    private suspend fun loadModelFromPath(model: LLMModel) {
        try {
            // Determine model path
            val modelAssetPath = if (model.url.startsWith("file://models/")) {
                model.url.removePrefix("file://")
            } else {
                "models/${model.localFileName()}"
            }
            
            Log.d(TAG, "Loading model from: $modelAssetPath")
            
            // Check if model exists in assets folder
            val modelFile = try {
                context.assets.open(modelAssetPath).use { 
                    // File exists in assets, copy to files directory
                    val targetFile = File(context.filesDir, "models/${model.localFileName()}")
                    targetFile.parentFile?.mkdirs()
                    
                    if (!targetFile.exists()) {
                        targetFile.outputStream().use { outputStream ->
                            context.assets.open(modelAssetPath).use { inputStream ->
                                inputStream.copyTo(outputStream)
                            }
                        }
                        Log.d(TAG, "Copied model to ${targetFile.absolutePath}")
                    }
                    targetFile
                }
            } catch (e: Exception) {
                // Try to find model in files directory
                val modelFile = File(context.filesDir, modelAssetPath)
                if (modelFile.exists()) {
                    Log.d(TAG, "Model found in files directory: ${modelFile.absolutePath}")
                    modelFile
                } else {
                    throw IllegalStateException("Model not found in assets or files: $modelAssetPath")
                }
            }
            
            // Determine backend - some models have GPU compatibility issues
            val useCpu = modelFile.name.contains("Phi-4", ignoreCase = true) ||
                         modelFile.name.contains("phi", ignoreCase = true) ||
                         modelFile.name.contains("Llama", ignoreCase = true)

            val backend = if (useCpu) {
                LlmInference.Backend.CPU
            } else {
                LlmInference.Backend.GPU
            }
            
            Log.d(TAG, "Using backend: $backend for model: ${modelFile.name}")
            
            // Determine max tokens based on model configuration
            val maxTokens = getMaxTokensForModel(model)
            
            Log.d(TAG, "Model configuration:")
            Log.d(TAG, "  - Name: ${model.name}")
            Log.d(TAG, "  - File: ${modelFile.name}")
            Log.d(TAG, "  - Path: ${modelFile.absolutePath}")
            Log.d(TAG, "  - Context window: ${model.contextWindowSize}")
            Log.d(TAG, "  - Max tokens: $maxTokens")
            Log.d(TAG, "  - Backend: $backend")
            
            // Create LLM inference options
            val optionsBuilder = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelFile.absolutePath)
                .setMaxTokens(maxTokens)
                .setPreferredBackend(backend)
            
            val options = optionsBuilder.build()
            
            // Create LLM inference engine
            val llmInference = LlmInference.createFromOptions(context, options)
            
            // Create initial session
            val session = createSession(llmInference)
            
            // Store model instance
            modelInstance = LlmModelInstance(engine = llmInference, session = session)
            currentModel = model
            
            Log.d(TAG, "Successfully loaded model: ${model.name}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${model.name}", e)
            throw RuntimeException("Failed to load model: ${e.message}", e)
        }
    }

    private fun createSession(engine: LlmInference): LlmInferenceSession {
        val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTopK(DEFAULT_TOP_K)
            .setTemperature(DEFAULT_TEMPERATURE)
            .setTopP(DEFAULT_TOP_P)
            .setRandomSeed(System.currentTimeMillis().toInt())
            .build()
        
        return LlmInferenceSession.createFromOptions(engine, sessionOptions)
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        ensureModelLoaded(model)
        
        return withContext(Dispatchers.IO) {
            val responseBuilder = StringBuilder()
            var localSession: LlmInferenceSession? = null
            try {
                // Create a new session for each single request to ensure clean state
                val engine = modelInstance?.engine
                    ?: throw IllegalStateException("No model loaded")
                
                localSession = createSession(engine)
                var isComplete = false
                
                // Add query to session
                localSession.addQueryChunk(prompt)
                
                // Generate response synchronously
                localSession.generateResponseAsync { partialResult, done ->
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
                Log.e(TAG, "Inference failed", e)
                "Error: ${e.message}"
            } finally {
                localSession?.close()
            }
        }
    }
    
    override suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String> = callbackFlow {
        ensureModelLoaded(model)
        
        val localSession = try {
            val engine = modelInstance?.engine
                ?: throw IllegalStateException("No model loaded")
            createSession(engine)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create new session", e)
            close(e)
            return@callbackFlow
        }
        
        var isGenerationComplete = false
        
        try {
            localSession.addQueryChunk(prompt)
            localSession.generateResponseAsync { partialResult, done ->
                if (!isClosedForSend) {
                    trySend(partialResult)
                }
                if (done) {
                    isGenerationComplete = true
                    close()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Streaming inference failed", e)
            isGenerationComplete = true
            close(e)
        }

        awaitClose {
            Log.d(TAG, "Closing session and resources.")
            try {
                // If generation is still in progress, we need to wait a bit before closing
                if (!isGenerationComplete) {
                    Log.d(TAG, "Waiting for generation to complete before cleanup...")
                    Thread.sleep(100)
                }
                localSession.close()
            } catch (e: Exception) {
                Log.w(TAG, "Error during session cleanup: ${e.message}")
            }
        }
    }.flowOn(Dispatchers.IO)

    override suspend fun generateResponseStreamWithSession(prompt: String, model: LLMModel, chatId: String): Flow<String> = callbackFlow {
        ensureModelLoaded(model)
        
        var isGenerationComplete = false
        
        try {
            // Use the single session from the model instance (Gallery approach)
            val instance = modelInstance ?: throw IllegalStateException("No model loaded")
            val session = instance.session
            
            // Check if previous generation is still processing and cancel it
            try {
                session.cancelGenerateResponseAsync()
                Log.d(TAG, "Cancelled any previous generation for chat $chatId")
                delay(100) // Wait for cancellation to complete
            } catch (e: Exception) {
                Log.d(TAG, "No previous generation to cancel: ${e.message}")
            }
            
            // Check token count and reset session if approaching limit (Gallery approach)
            val maxTokens = getMaxTokensForModel(model)
            val currentTokens = session.sizeInTokens(prompt)
            val promptTokens = session.sizeInTokens(prompt) - session.sizeInTokens("")
            
            Log.d(TAG, "Token usage for chat $chatId:")
            Log.d(TAG, "  - Current session tokens: $currentTokens")
            Log.d(TAG, "  - Prompt tokens: $promptTokens")
            Log.d(TAG, "  - Max tokens: $maxTokens")
            
            // If adding the prompt would exceed ~80% of max tokens, reset the session
            val tokenThreshold = (maxTokens * 0.8).toInt()
            if (currentTokens + promptTokens > tokenThreshold) {
                Log.w(TAG, "Token count ($currentTokens + $promptTokens = ${currentTokens + promptTokens}) approaching limit ($maxTokens)")
                Log.w(TAG, "Resetting session for chat $chatId to prevent OUT_OF_RANGE error")
                
                // Reset session before it gets full
                sessionMutex.withLock {
                    try {
                        session.close()
                        Log.d(TAG, "Closed session due to token limit approach")
                        
                        // Create new session
                        val newSession = LlmInferenceSession.createFromOptions(
                            instance.engine,
                            LlmInferenceSession.LlmInferenceSessionOptions.builder()
                                .setTopK(DEFAULT_TOP_K)
                                .setTopP(DEFAULT_TOP_P)
                                .setTemperature(DEFAULT_TEMPERATURE)
                                .build()
                        )
                        instance.session = newSession
                        Log.d(TAG, "Created fresh session for chat $chatId")
                        
                        // Give MediaPipe time to clean up
                        delay(500)
                        
                    } catch (resetException: Exception) {
                        Log.e(TAG, "Failed to reset session before token limit: ${resetException.message}")
                        throw resetException
                    }
                }
            }
            
            // Now use the session (either existing or freshly reset)
            val currentSession = instance.session
            currentSession.addQueryChunk(prompt)
            currentSession.generateResponseAsync { partialResult, done ->
                if (!isClosedForSend) {
                    trySend(partialResult)
                }
                if (done) {
                    isGenerationComplete = true
                    close()
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Streaming inference failed for chat $chatId: ${e.message}", e)
            
            // Check if this is a MediaPipe session error and try to recover
            if (isMediaPipeSessionError(e) || isTokenLimitError(e)) {
                Log.w(TAG, "Detected MediaPipe session/token error, attempting recovery for chat $chatId")
                
                try {
                    // First try normal reset
                    resetChatSession(chatId)
                    
                    // Use the new session
                    val instance = modelInstance ?: throw IllegalStateException("No model loaded after reset")
                    val session = instance.session
                    
                    Log.d(TAG, "Created new session for recovery, attempting generation retry")
                    
                    session.addQueryChunk(prompt)
                    session.generateResponseAsync { partialResult, done ->
                        if (!isClosedForSend) {
                            trySend(partialResult)
                        }
                        if (done) {
                            isGenerationComplete = true
                            close()
                        }
                    }
                    
                    Log.d(TAG, "Successfully recovered from MediaPipe session error for chat $chatId")
                    
                } catch (recoveryException: Exception) {
                    Log.e(TAG, "Normal recovery failed for chat $chatId, trying force recreate", recoveryException)
                    
                    // Last resort: force recreate everything
                    try {
                        if (forceRecreateSession()) {
                            val instance = modelInstance ?: throw IllegalStateException("No model loaded after force recreate")
                            val session = instance.session
                            
                            Log.d(TAG, "Force recreated session, attempting generation retry")
                            
                            session.addQueryChunk(prompt)
                            session.generateResponseAsync { partialResult, done ->
                                if (!isClosedForSend) {
                                    trySend(partialResult)
                                }
                                if (done) {
                                    isGenerationComplete = true
                                    close()
                                }
                            }
                            
                            Log.d(TAG, "Successfully recovered using force recreate for chat $chatId")
                        } else {
                            Log.e(TAG, "Force recreate failed for chat $chatId")
                            isGenerationComplete = true
                            close(Exception("Failed to recover session after multiple attempts"))
                        }
                    } catch (forceException: Exception) {
                        Log.e(TAG, "Force recreate attempt failed for chat $chatId", forceException)
                        isGenerationComplete = true
                        close(forceException)
                    }
                }
            } else {
                isGenerationComplete = true
                close(e)
            }
        }

        awaitClose {
            Log.d(TAG, "Generation complete for chat $chatId")
            // Don't close the session here - it's managed by the model instance
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Check if the exception is a known MediaPipe session error
     */
    private fun isMediaPipeSessionError(e: Exception): Boolean {
        val errorMessage = e.message?.lowercase() ?: ""
        return errorMessage.contains("detokenizercalculator") ||
                errorMessage.contains("id >= 0") ||
                errorMessage.contains("no id available to be decoded") ||
                errorMessage.contains("llmexecutorcalculator") ||
                errorMessage.contains("please create a new session") ||
                errorMessage.contains("invalid_argument") ||
                errorMessage.contains("failed to add query chunk") ||
                errorMessage.contains("graph has errors") ||
                errorMessage.contains("previous invocation still processing") ||
                errorMessage.contains("wait for done=true")
    }

    /**
     * Check if the exception is a token limit error
     */
    private fun isTokenLimitError(e: Exception): Boolean {
        val errorMessage = e.message?.lowercase() ?: ""
        return errorMessage.contains("max number of tokens") ||
                errorMessage.contains("maximum cache size") ||
                errorMessage.contains("out_of_range") ||
                errorMessage.contains("current_step") ||
                errorMessage.contains("input_size") ||
                errorMessage.contains("token limit") ||
                errorMessage.contains("exceeded") ||
                errorMessage.contains("larger than the maximum")
    }

    override suspend fun onCleared() {
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Clearing all resources and sessions")
                
                sessionMutex.withLock {
                    // Close model instance
                    modelInstance?.let { instance ->
                        try {
                            instance.session.close()
                        } catch (e: Exception) {
                            Log.w(TAG, "Error closing session during cleanup: ${e.message}")
                        }
                        
                        try {
                            instance.engine.close()
                        } catch (e: Exception) {
                            Log.w(TAG, "Error closing LLM inference during cleanup: ${e.message}")
                        }
                    }
                    
                    modelInstance = null
                    currentModel = null
                }
                
                Log.d(TAG, "Resources released")
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing resources", e)
            }
        }
    }

    override fun getCurrentlyLoadedModel(): LLMModel? {
        return currentModel
    }

    /**
     * Force recreate the entire session when reset fails (last resort recovery)
     */
    private suspend fun forceRecreateSession(): Boolean {
        return sessionMutex.withLock {
            try {
                Log.d(TAG, "Force recreating session as last resort")
                val currentModelBackup = currentModel
                
                // Close everything
                modelInstance?.let { instance ->
                    try {
                        instance.session.close()
                    } catch (e: Exception) {
                        Log.d(TAG, "Error closing session during force recreate: ${e.message}")
                    }
                    try {
                        instance.engine.close()
                    } catch (e: Exception) {
                        Log.d(TAG, "Error closing engine during force recreate: ${e.message}")
                    }
                }
                
                modelInstance = null
                currentModel = null
                
                // Wait longer for cleanup
                delay(1000)
                
                // Reload the model if we had one
                if (currentModelBackup != null) {
                    try {
                        loadModelFromPath(currentModelBackup)
                        Log.d(TAG, "Successfully force recreated session")
                        return@withLock true
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to reload model during force recreate: ${e.message}", e)
                    }
                }
                
                Log.e(TAG, "Force recreate failed")
                return@withLock false
            } catch (e: Exception) {
                Log.e(TAG, "Exception during force recreate: ${e.message}", e)
                return@withLock false
            }
        }
    }
}