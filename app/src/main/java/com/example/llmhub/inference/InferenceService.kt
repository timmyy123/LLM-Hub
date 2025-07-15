package com.example.llmhub.inference

import android.content.Context
import android.graphics.Bitmap
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
import com.google.mediapipe.framework.image.BitmapImageBuilder
import android.util.Log
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import android.app.ActivityManager

/**
 * Interface for a service that can run model inference.
 */
interface InferenceService {
    suspend fun loadModel(model: LLMModel): Boolean
    suspend fun generateResponse(prompt: String, model: LLMModel): String
    suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String>
    suspend fun generateResponseStreamWithSession(prompt: String, model: LLMModel, chatId: String, images: List<Bitmap> = emptyList()): Flow<String>
    suspend fun resetChatSession(chatId: String)
    suspend fun onCleared()
    fun getCurrentlyLoadedModel(): LLMModel?
    fun getMemoryWarningForImages(images: List<Bitmap>): String?
}

/**
 * Data class to hold MediaPipe LLM inference engine and session
 */
data class LlmModelInstance(val engine: LlmInference, var session: LlmInferenceSession)

/**
 * MediaPipe-based implementation of InferenceService following Google AI Edge Gallery pattern.
 * Uses a single session approach: one session per model that gets reset as needed.
 * 
 * Memory Management Strategy:
 * - Device RAM <= 6GB: Force CPU backend for vision models (GPU + Vision causes crashes)
 * - Device RAM 6-8GB: Use GPU for text-only, CPU for vision (warnings shown)
 * - Device RAM > 8GB: GPU + Vision allowed without restrictions
 * 
 * Backend Selection Logic:
 * - Models with known GPU issues (Phi, Llama): Always use CPU
 * - Vision models on low memory devices: Use CPU to prevent crashes
 * - Text-only or high memory devices: Use GPU for performance
 */
class MediaPipeInferenceService(private val context: Context) : InferenceService {
    
    private var modelInstance: LlmModelInstance? = null
    private var currentModel: LLMModel? = null
    private var currentBackend: LlmInference.Backend? = null
    
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
        
        /**
         * Get device total memory in GB
         */
        private fun getDeviceMemoryGB(context: Context): Double {
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)
            return memoryInfo.totalMem / (1024.0 * 1024.0 * 1024.0) // Convert bytes to GB
        }
        
        /**
         * Determine optimal backend based on device memory and vision usage
         * Memory-aware strategy:
         * - <= 6GB RAM: Force CPU for vision models (GPU + Vision causes OOM)
         * - 6-8GB RAM: GPU for text-only, CPU for vision (hybrid approach)
         * - > 8GB RAM: GPU with vision allowed (sufficient memory)
         */
        private fun determineOptimalBackend(
            context: Context,
            model: LLMModel,
            willUseImages: Boolean
        ): LlmInference.Backend {
            val deviceMemoryGB = getDeviceMemoryGB(context)
            
            // Check for models with known GPU compatibility issues
            val hasGpuCompatibilityIssues = model.localFileName().contains("Phi-4", ignoreCase = true) ||
                                           model.localFileName().contains("phi", ignoreCase = true) ||
                                           model.localFileName().contains("Llama", ignoreCase = true)
            
            Log.d(TAG, "Memory-aware backend selection:")
            Log.d(TAG, "  - Device memory: ${String.format("%.1f", deviceMemoryGB)}GB")
            Log.d(TAG, "  - Model supports vision: ${model.supportsVision}")
            Log.d(TAG, "  - Expected to use images: $willUseImages")
            Log.d(TAG, "  - Has GPU compatibility issues: $hasGpuCompatibilityIssues")
            
            return when {
                // Force CPU for models with known GPU issues
                hasGpuCompatibilityIssues -> {
                    Log.d(TAG, "  → CPU backend (GPU compatibility issues)")
                    LlmInference.Backend.CPU
                }
                
                // Very low memory devices: CPU for vision, GPU for text
                deviceMemoryGB <= 6.0 -> {
                    if (model.supportsVision) {
                        Log.d(TAG, "  → CPU backend (low memory + vision model)")
                        LlmInference.Backend.CPU
                    } else {
                        Log.d(TAG, "  → GPU backend (low memory + text-only model)")
                        LlmInference.Backend.GPU
                    }
                }
                
                // Medium memory devices: CPU for vision usage, GPU for text
                deviceMemoryGB <= 8.0 -> {
                    if (model.supportsVision && willUseImages) {
                        Log.d(TAG, "  → CPU backend (medium memory + vision with images)")
                        LlmInference.Backend.CPU
                    } else {
                        Log.d(TAG, "  → GPU backend (medium memory + text or vision without images)")
                        LlmInference.Backend.GPU
                    }
                }
                
                // High memory devices: GPU is safe for everything
                else -> {
                    Log.d(TAG, "  → GPU backend (high memory device)")
                    LlmInference.Backend.GPU
                }
            }
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
                val newSession = createSession(instance.engine)
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
            
            // Determine backend using memory-aware selection
            // For loading, assume images might be used for vision models
            val willUseImages = model.supportsVision
            val backend = determineOptimalBackend(context, model, willUseImages)
            
            Log.d(TAG, "Selected backend: $backend for model: ${modelFile.name}")
            
            // Determine max tokens based on model configuration
            val maxTokens = getMaxTokensForModel(model)
            
            Log.d(TAG, "Model configuration:")
            Log.d(TAG, "  - Name: ${model.name}")
            Log.d(TAG, "  - File: ${modelFile.name}")
            Log.d(TAG, "  - Path: ${modelFile.absolutePath}")
            Log.d(TAG, "  - Context window: ${model.contextWindowSize}")
            Log.d(TAG, "  - Max tokens: $maxTokens")
            Log.d(TAG, "  - Backend: $backend")
            Log.d(TAG, "  - Supports vision: ${model.supportsVision}")
            
            // Create LLM inference options
            val optionsBuilder = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelFile.absolutePath)
                .setMaxTokens(maxTokens)
                .setPreferredBackend(backend)
                
            // Enable vision modality for multimodal models (following Google AI Edge Gallery pattern)
            if (model.supportsVision) {
                optionsBuilder.setMaxNumImages(10) // Allow up to 10 images per session
                Log.d(TAG, "  - Enabled vision modality with max 10 images")
            } else {
                optionsBuilder.setMaxNumImages(0) // Explicitly disable for non-vision models
                Log.d(TAG, "  - Vision modality disabled (maxNumImages=0)")
            }
            
            val options = optionsBuilder.build()
            
            // Create LLM inference engine
            val llmInference = LlmInference.createFromOptions(context, options)
            
            // Set current model BEFORE creating session (critical for vision modality)
            currentModel = model
            
            // Create initial session
            val session = createSession(llmInference)
            
            // Store model instance and backend
            modelInstance = LlmModelInstance(engine = llmInference, session = session)
            currentBackend = backend
            
            Log.d(TAG, "Successfully loaded model: ${model.name} with backend: $backend")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${model.name}", e)
            throw RuntimeException("Failed to load model: ${e.message}", e)
        }
    }

    private fun createSession(engine: LlmInference): LlmInferenceSession {
        val model = currentModel
        
        Log.d(TAG, "Creating session for model: ${model?.name}")
        Log.d(TAG, "Model supports vision: ${model?.supportsVision}")
        
        val sessionOptionsBuilder = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTopK(DEFAULT_TOP_K)
            .setTemperature(DEFAULT_TEMPERATURE)
            .setTopP(DEFAULT_TOP_P)
            .setRandomSeed(System.currentTimeMillis().toInt())
            
        // Enable vision modality for multimodal models (following Google AI Edge Gallery pattern)
        if (model?.supportsVision == true) {
            try {
                val graphOptions = com.google.mediapipe.tasks.genai.llminference.GraphOptions.builder()
                    .setEnableVisionModality(true)
                    .build()
                sessionOptionsBuilder.setGraphOptions(graphOptions)
                Log.d(TAG, "Session created with vision modality ENABLED for model ${model.name}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to enable vision modality for model ${model.name}: ${e.message}", e)
                // Continue without vision modality - this might be the issue
                throw e  // Don't continue if vision fails - this is critical for vision models
            }
        } else {
            // Explicitly disable vision modality for non-vision models
            try {
                val graphOptions = com.google.mediapipe.tasks.genai.llminference.GraphOptions.builder()
                    .setEnableVisionModality(false)
                    .build()
                sessionOptionsBuilder.setGraphOptions(graphOptions)
                Log.d(TAG, "Session created with vision modality DISABLED for model ${model?.name}")
            } catch (e: Exception) {
                Log.w(TAG, "Failed to disable vision modality for model ${model?.name}: ${e.message}", e)
                // Continue without setting graph options for non-vision models
            }
        }
        
        val sessionOptions = sessionOptionsBuilder.build()
        
        try {
            val session = LlmInferenceSession.createFromOptions(engine, sessionOptions)
            Log.d(TAG, "Successfully created session for model ${model?.name}")
            return session
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create session for model ${model?.name}: ${e.message}", e)
            throw e
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

    override suspend fun generateResponseStreamWithSession(prompt: String, model: LLMModel, chatId: String, images: List<Bitmap>): Flow<String> = callbackFlow {
        ensureModelLoaded(model)
        
        // Check memory constraints for vision usage
        if (images.isNotEmpty() && model.supportsVision) {
            val memoryWarning = checkMemoryConstraintsForVision(images)
            if (memoryWarning != null) {
                Log.w(TAG, memoryWarning)
                // In a real app, you might want to show this warning to the user
                // For now, we'll continue but the user should be aware of potential issues
            }
        }
        
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
                        val newSession = createSession(instance.engine)
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
            
            // CRITICAL: For vision models, text query MUST be added before images
            // This is required by MediaPipe's vision implementation
            if (prompt.trim().isNotEmpty()) {
                Log.d(TAG, "Adding text query to session for chat $chatId: '${prompt.take(100)}...'")
                currentSession.addQueryChunk(prompt)
            } else if (images.isNotEmpty() && model.supportsVision) {
                // If we have images but no text, add a default query for vision models
                Log.d(TAG, "Adding default vision query for images in chat $chatId")
                currentSession.addQueryChunk("What do you see in this image?")
            }
            
            // Add images AFTER text query (MediaPipe requirement for vision models)
            if (images.isNotEmpty() && model.supportsVision) {
                Log.d(TAG, "Adding ${images.size} images to session for chat $chatId")
                for ((index, image) in images.withIndex()) {
                    try {
                        Log.d(TAG, "Adding image $index (${image.width}x${image.height}) to session")
                        
                        // Validate image dimensions
                        if (image.width <= 0 || image.height <= 0) {
                            Log.e(TAG, "Invalid image dimensions: ${image.width}x${image.height}")
                            continue
                        }
                        
                        // Create MediaPipe image and add to session
                        val mpImage = BitmapImageBuilder(image).build()
                        currentSession.addImage(mpImage)
                        Log.d(TAG, "Successfully added image $index to session")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to add image $index to session: ${e.message}", e)
                    }
                }
            } else if (images.isNotEmpty() && !model.supportsVision) {
                Log.w(TAG, "Model ${model.name} does not support vision, ignoring ${images.size} images")
            } else if (images.isEmpty() && model.supportsVision) {
                Log.d(TAG, "No images provided for vision-capable model ${model.name}")
            }
            
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
                    
                    // Re-add text query first (CRITICAL for vision models)
                    if (prompt.trim().isNotEmpty()) {
                        Log.d(TAG, "Re-adding text query to recovery session for chat $chatId")
                        session.addQueryChunk(prompt)
                    } else if (images.isNotEmpty() && model.supportsVision) {
                        Log.d(TAG, "Adding default vision query for recovery session")
                        session.addQueryChunk("What do you see in this image?")
                    }
                    
                    // Re-add images if provided and model supports vision
                    if (images.isNotEmpty() && model.supportsVision) {
                        Log.d(TAG, "Re-adding ${images.size} images to recovery session for chat $chatId")
                        for ((index, image) in images.withIndex()) {
                            try {
                                Log.d(TAG, "Re-adding image $index (${image.width}x${image.height}) to recovery session")
                                session.addImage(BitmapImageBuilder(image).build())
                                Log.d(TAG, "Successfully re-added image $index to recovery session")
                            } catch (e: Exception) {
                                Log.e(TAG, "Failed to add image $index to recovery session: ${e.message}")
                            }
                        }
                    }
                    
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
                            
                            // Re-add text query first (CRITICAL for vision models)
                            if (prompt.trim().isNotEmpty()) {
                                Log.d(TAG, "Re-adding text query to force recreated session for chat $chatId")
                                session.addQueryChunk(prompt)
                            } else if (images.isNotEmpty() && model.supportsVision) {
                                Log.d(TAG, "Adding default vision query for force recreated session")
                                session.addQueryChunk("What do you see in this image?")
                            }
                            
                            // Re-add images if provided and model supports vision
                            if (images.isNotEmpty() && model.supportsVision) {
                                Log.d(TAG, "Re-adding ${images.size} images to force recreated session for chat $chatId")
                                for ((index, image) in images.withIndex()) {
                                    try {
                                        Log.d(TAG, "Re-adding image $index (${image.width}x${image.height}) to force recreated session")
                                        session.addImage(BitmapImageBuilder(image).build())
                                        Log.d(TAG, "Successfully re-added image $index to force recreated session")
                                    } catch (e: Exception) {
                                        Log.e(TAG, "Failed to add image $index to force recreated session: ${e.message}")
                                    }
                                }
                            }
                            
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
    
    /**
     * Provide user-friendly memory guidance for vision usage
     */
    private fun checkMemoryConstraintsForVision(images: List<Bitmap>): String? {
        if (images.isEmpty()) return null
        
        val model = currentModel ?: return null
        if (!model.supportsVision) return null
        
        val deviceMemoryGB = getDeviceMemoryGB(context)
        val backend = currentBackend
        
        return when {
            deviceMemoryGB <= 6.0 -> {
                "⚠️ Low memory device (${String.format("%.1f", deviceMemoryGB)}GB RAM): Vision models may cause crashes. Consider using text-only queries."
            }
            deviceMemoryGB <= 8.0 && backend == LlmInference.Backend.GPU -> {
                "⚠️ Medium memory device (${String.format("%.1f", deviceMemoryGB)}GB RAM): GPU + Vision may be unstable. Reduce image size or use text-only if crashes occur."
            }
            deviceMemoryGB <= 8.0 && backend == LlmInference.Backend.CPU -> {
                "ℹ️ Using CPU backend for vision on ${String.format("%.1f", deviceMemoryGB)}GB RAM device for stability."
            }
            else -> null // No warning needed for high memory devices
        }
    }

    override fun getMemoryWarningForImages(images: List<Bitmap>): String? {
        return checkMemoryConstraintsForVision(images)
    }
}