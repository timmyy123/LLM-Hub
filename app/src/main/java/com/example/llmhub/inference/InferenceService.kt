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
 * MediaPipe-based implementation of InferenceService inspired by Google AI Edge Gallery.
 * Uses a simplified session management approach: one session per chat that gets reset as needed.
 */
class MediaPipeInferenceService(private val context: Context) : InferenceService {
    
    private var modelInstance: LlmModelInstance? = null
    private var currentModel: LLMModel? = null
    
    // Keep track of sessions per chat - simplified approach
    private val chatSessions = mutableMapOf<String, LlmInferenceSession>()
    
    // Mutex to prevent race conditions during session operations
    private val sessionMutex = Mutex()
    
    companion object {
        private const val TAG = "MediaPipeInference"
        private const val DEFAULT_MAX_TOKENS = 4096
        private const val DEFAULT_TOP_K = 40
        private const val DEFAULT_TOP_P = 0.8f
        private const val DEFAULT_TEMPERATURE = 0.8f
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
            
            // Close existing session if it exists
            chatSessions.remove(chatId)?.let { session ->
                try {
                    session.close()
                    Log.d(TAG, "Closed old session for chat $chatId")
                } catch (e: Exception) {
                    Log.w(TAG, "Error closing session for chat $chatId: ${e.message}")
                }
            }
            
            // Give MediaPipe time to clean up
            delay(200)
            Log.d(TAG, "Session reset completed for chat $chatId")
        }
    }

    private suspend fun ensureModelLoaded(model: LLMModel) {
        if (currentModel?.name != model.name) {
            Log.d(TAG, "Model changed from ${currentModel?.name} to ${model.name}, reloading")
            
            // Close all existing sessions when switching models
            sessionMutex.withLock {
                chatSessions.values.forEach { session ->
                    try {
                        session.close()
                    } catch (e: Exception) {
                        Log.w(TAG, "Error closing session during model switch: ${e.message}")
                    }
                }
                chatSessions.clear()
            }
            
            // Release existing model instance
            modelInstance?.let { instance ->
                try {
                    instance.session.close()
                } catch (e: Exception) {
                    Log.w(TAG, "Error closing main session: ${e.message}")
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
            
            // Create LLM inference options
            val optionsBuilder = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelFile.absolutePath)
                .setMaxTokens(DEFAULT_MAX_TOKENS)
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

    private suspend fun getOrCreateChatSession(chatId: String): LlmInferenceSession {
        return sessionMutex.withLock {
            chatSessions[chatId] ?: run {
                val engine = modelInstance?.engine 
                    ?: throw IllegalStateException("No model loaded - call loadModel first")
                
                Log.d(TAG, "Creating new session for chat $chatId")
                val newSession = createSession(engine)
                chatSessions[chatId] = newSession
                newSession
            }
        }
    }

    private suspend fun resetSession(chatId: String): LlmInferenceSession {
        sessionMutex.withLock {
            Log.d(TAG, "Resetting session for chat $chatId")
            
            // Close old session if exists
            chatSessions.remove(chatId)?.let { oldSession ->
                try {
                    oldSession.close()
                    delay(200) // Give MediaPipe time to clean up
                } catch (e: Exception) {
                    Log.w(TAG, "Error closing old session for chat $chatId: ${e.message}")
                }
            }
            
            // Create new session
            val engine = modelInstance?.engine 
                ?: throw IllegalStateException("No model loaded")
            
            val newSession = createSession(engine)
            chatSessions[chatId] = newSession
            Log.d(TAG, "Created new session for chat $chatId")
            
            return newSession
        }
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
        
        var localSession: LlmInferenceSession? = null
        var isGenerationComplete = false
        
        try {
            // Get or create session for this chat
            localSession = getOrCreateChatSession(chatId)
            
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
            Log.e(TAG, "Streaming inference failed for chat $chatId: ${e.message}", e)
            
            // Check if this is a MediaPipe session error and try to recover
            if (isMediaPipeSessionError(e)) {
                Log.w(TAG, "Detected MediaPipe session error, attempting recovery for chat $chatId")
                
                try {
                    // Reset the session and retry
                    localSession = resetSession(chatId)
                    
                    Log.d(TAG, "Created new session for recovery, attempting generation retry")
                    
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
                    
                    Log.d(TAG, "Successfully recovered from MediaPipe session error for chat $chatId")
                    
                } catch (recoveryException: Exception) {
                    Log.e(TAG, "Failed to recover from MediaPipe session error for chat $chatId", recoveryException)
                    isGenerationComplete = true
                    close(recoveryException)
                }
            } else {
                isGenerationComplete = true
                close(e)
            }
        }

        awaitClose {
            Log.d(TAG, "Generation complete for chat $chatId")
            // Don't close the session here - keep it for future use in this chat
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
                errorMessage.contains("graph has errors")
    }

    override suspend fun onCleared() {
        withContext(Dispatchers.IO) {
            try {
                Log.d(TAG, "Clearing all resources and sessions")
                
                // Close all chat sessions
                val sessionCount = chatSessions.size
                sessionMutex.withLock {
                    chatSessions.values.forEach { session ->
                        try {
                            session.close()
                        } catch (e: Exception) {
                            Log.w(TAG, "Error closing chat session during cleanup: ${e.message}")
                        }
                    }
                    chatSessions.clear()
                }
                
                // Close model instance
                modelInstance?.let { instance ->
                    try {
                        instance.session.close()
                    } catch (e: Exception) {
                        Log.w(TAG, "Error closing main session during cleanup: ${e.message}")
                    }
                    
                    try {
                        instance.engine.close()
                    } catch (e: Exception) {
                        Log.w(TAG, "Error closing LLM inference during cleanup: ${e.message}")
                    }
                }
                
                modelInstance = null
                currentModel = null
                
                Log.d(TAG, "Resources released (closed $sessionCount sessions)")
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing resources", e)
            }
        }
    }

    override fun getCurrentlyLoadedModel(): LLMModel? {
        return currentModel
    }
}