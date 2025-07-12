package com.example.llmhub.inference

import android.content.Context
import com.example.llmhub.data.LLMModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
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
    suspend fun getChatSession(chatId: String): LlmInferenceSession?
    suspend fun createChatSession(chatId: String): LlmInferenceSession
    suspend fun closeChatSession(chatId: String)
    suspend fun resetChatSession(chatId: String)
    suspend fun onCleared()
    fun getCurrentlyLoadedModel(): LLMModel?
}

/**
 * MediaPipe-based implementation of InferenceService that supports GPU acceleration
 * and real-time streaming inference.
 */
class MediaPipeInferenceService(private val context: Context) : InferenceService {
    
    private var llmInference: LlmInference? = null
    private var session: LlmInferenceSession? = null
    private var currentModel: LLMModel? = null
    private var currentChatId: String? = null
    
    // Keep track of conversation state per chat
    private val chatSessions = mutableMapOf<String, LlmInferenceSession>()
    
    override suspend fun loadModel(model: LLMModel): Boolean {
        return try {
            ensureModelLoaded(model)
            true
        } catch (e: Exception) {
            Log.e("MediaPipeInference", "Failed to load model: ${e.message}", e)
            false
        }
    }
    
    override suspend fun getChatSession(chatId: String): LlmInferenceSession? {
        return chatSessions[chatId]
    }
    
    override suspend fun createChatSession(chatId: String): LlmInferenceSession {
        val model = currentModel ?: throw IllegalStateException("No model loaded - call loadModel first")
        ensureModelLoaded(model)
        val newSession = createNewSession()
        chatSessions[chatId] = newSession
        return newSession
    }
    
    override suspend fun closeChatSession(chatId: String) {
        chatSessions.remove(chatId)?.let { session ->
            try {
                Log.d("MediaPipeInference", "Closing session for chat $chatId")
                session.close()
            } catch (e: Exception) {
                Log.w("MediaPipeInference", "Error closing session for chat $chatId: ${e.message}")
            }
        }
    }
    
    // Force close and recreate session for a chat (useful for error recovery)
    override suspend fun resetChatSession(chatId: String) {
        Log.d("MediaPipeInference", "Resetting session for chat $chatId")
        closeChatSession(chatId)
        // Session will be recreated on next use
    }
    
    private suspend fun ensureModelLoaded(model: LLMModel) {
        if (currentModel?.name != model.name) {
            Log.d("MediaPipeInference", "Model changed from ${currentModel?.name} to ${model.name}, clearing all sessions")
            
            // Close all existing chat sessions first
            chatSessions.values.forEach { session ->
                try {
                    session.close()
                } catch (e: Exception) {
                    Log.w("MediaPipeInference", "Error closing chat session during model switch: ${e.message}")
                }
            }
            chatSessions.clear()
            
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
        // Determine backend based on model type - some models have GPU compatibility issues
        val useCpu = modelFile.name.contains("Phi-4", ignoreCase = true) ||
                     modelFile.name.contains("phi", ignoreCase = true) ||
                     modelFile.name.contains("Llama", ignoreCase = true)

        val backend = if (useCpu) {
            LlmInference.Backend.CPU
        } else {
            LlmInference.Backend.GPU
        }
        
        // Dynamically detect the maximum context window from the filename pattern `_ekvXXXX`
        // For example: `..._ekv1280.task` -> 1280 tokens, `..._ekv4096.task` -> 4096 tokens.
        val contextWindowSize = Regex("_ekv(\\d+)")
            .find(modelFile.name)
            ?.groups?.get(1)
            ?.value
            ?.toIntOrNull()
            ?.coerceAtLeast(1280) // Ensure a sensible lower bound
            ?: 2048 // Default when pattern not present

        // Set maxTokens for OUTPUT generation (not context window)
        // Use a reasonable output limit that allows for longer responses
        val maxOutputTokens = 4096 // Allow up to 4096 tokens per generation (much higher than previous ~2000)

        val optionsBuilder = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelFile.absolutePath)  // Use file path
            .setMaxTokens(maxOutputTokens) // This is for output generation, not context window
            .setPreferredBackend(backend)

        Log.d("MediaPipeInference", "Model: ${modelFile.name}, Context Window: $contextWindowSize, Max Output Tokens: $maxOutputTokens")

        // When using CPU, we rely on the backend's default thread management
        if (backend == LlmInference.Backend.CPU) {
            Log.d("MediaPipeInference", "CPU backend selected. Using default thread management.")
        }

        val options = optionsBuilder.build()
        
        // Create LLM inference engine from file
        llmInference = LlmInference.createFromOptions(context, options)
        
        // DO NOT create session here. Create it per-request.
        currentModel = model
        Log.d("MediaPipeInference", "Successfully loaded model from file: ${model.name}")
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        ensureModelLoaded(model)
        
        return withContext(Dispatchers.IO) {
            val responseBuilder = StringBuilder()
            var localSession: LlmInferenceSession? = null
            try {
                // Create a new session for each request to ensure a clean state
                localSession = createNewSession()

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
                Log.e("MediaPipeInference", "Inference failed", e)
                "Error: ${e.message}"
            } finally {
                localSession?.close()
            }
        }
    }
    
    override suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String> = callbackFlow {
        ensureModelLoaded(model)
        
        val localSession = try {
            createNewSession()
        } catch (e: Exception) {
            Log.e("MediaPipeInference", "Failed to create new session", e)
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
            Log.e("MediaPipeInference", "Streaming inference failed", e)
            isGenerationComplete = true
            close(e)
        }

        awaitClose {
            Log.d("MediaPipeInference", "Closing session and resources.")
            try {
                // If generation is still in progress, we need to wait a bit before closing
                if (!isGenerationComplete) {
                    Log.d("MediaPipeInference", "Waiting for generation to complete before cleanup...")
                    // Give it a short time to finish naturally
                    Thread.sleep(100)
                }
                localSession.close()
            } catch (e: Exception) {
                Log.w("MediaPipeInference", "Error during session cleanup: ${e.message}")
                // Continue cleanup even if session close fails
            }
        }
    }.flowOn(Dispatchers.IO)

    override suspend fun generateResponseStreamWithSession(prompt: String, model: LLMModel, chatId: String): Flow<String> = callbackFlow {
        ensureModelLoaded(model)
        
        // Get or create persistent session for this chat
        var localSession = try {
            getChatSession(chatId) ?: createChatSession(chatId)
        } catch (e: Exception) {
            Log.e("MediaPipeInference", "Failed to get/create session for chat $chatId", e)
            close(e)
            return@callbackFlow
        }
        
        var isGenerationComplete = false
        var sessionRecreated = false
        
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
            Log.e("MediaPipeInference", "Streaming inference failed for chat $chatId: ${e.message}", e)
            
            // If we get a MediaPipe session error, try to recover by creating a new session
            if (e.message?.contains("DetokenizerCalculator") == true || 
                e.message?.contains("id >= 0") == true ||
                e.message?.contains("Please create a new Session") == true) {
                
                Log.w("MediaPipeInference", "Detected MediaPipe session error, attempting recovery for chat $chatId")
                
                try {
                    // Close the old session and create a new one
                    closeChatSession(chatId)
                    localSession = createChatSession(chatId)
                    sessionRecreated = true
                    
                    // Retry the generation with the new session
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
                    
                    Log.d("MediaPipeInference", "Successfully recovered from MediaPipe session error for chat $chatId")
                    
                } catch (recoveryException: Exception) {
                    Log.e("MediaPipeInference", "Failed to recover from MediaPipe session error for chat $chatId", recoveryException)
                    isGenerationComplete = true
                    close(recoveryException)
                }
            } else {
                isGenerationComplete = true
                close(e)
            }
        }

        awaitClose {
            Log.d("MediaPipeInference", "Generation complete for chat $chatId (session recreated: $sessionRecreated)")
            // NOTE: We don't close the session here because we want to keep it for future use
            // The session will be closed when the chat is explicitly closed or app shuts down
        }
    }.flowOn(Dispatchers.IO)

    private fun createNewSession(): LlmInferenceSession {
        val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTopK(40)
            .setTemperature(0.8f)
            .setRandomSeed(System.currentTimeMillis().toInt()) // Use different seed
            .build()
        return LlmInferenceSession.createFromOptions(llmInference!!, sessionOptions)
    }
    
    override suspend fun onCleared() {
        withContext(Dispatchers.IO) {
            try {
                // Close all chat sessions
                chatSessions.values.forEach { session ->
                    try {
                        session.close()
                    } catch (e: Exception) {
                        Log.w("MediaPipeInference", "Error closing chat session: ${e.message}")
                    }
                }
                chatSessions.clear()
                
                // Close main session and inference engine
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

    override fun getCurrentlyLoadedModel(): LLMModel? {
        return currentModel
    }
}