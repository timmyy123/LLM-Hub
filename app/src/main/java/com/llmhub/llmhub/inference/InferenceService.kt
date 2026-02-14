package com.llmhub.llmhub.inference

import android.content.Context
import android.content.res.Configuration
import android.graphics.Bitmap
import android.net.Uri
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.websearch.WebSearchService
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.SearchIntentDetector
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
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.R
import com.llmhub.llmhub.utils.LocaleHelper
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import com.google.mediapipe.tasks.genai.llminference.AudioModelOptions
import com.google.mediapipe.framework.image.BitmapImageBuilder
import android.util.Log
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import android.app.ActivityManager
import kotlinx.coroutines.CancellationException

/**
 * Interface for a service that can run model inference.
 */
interface InferenceService {
    /**
     * Load a model. For GGUF models an optional deviceId (e.g. "HTP0") may be supplied to request
     * Hexagon/NPU usage; implementations that don't support deviceId should ignore it.
     */
    suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend? = null, deviceId: String? = null): Boolean
    suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend? = null, disableVision: Boolean = false, disableAudio: Boolean = false, deviceId: String? = null): Boolean
    suspend fun unloadModel()
    suspend fun generateResponse(prompt: String, model: LLMModel): String
    suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String>
    suspend fun generateResponseStreamWithSession(
        prompt: String, 
        model: LLMModel, 
        chatId: String, 
        images: List<Bitmap> = emptyList(), 
        audioData: ByteArray? = null,
        webSearchEnabled: Boolean = true,
        imagePaths: List<String> = emptyList()
    ): Flow<String>
    suspend fun resetChatSession(chatId: String)
    suspend fun onCleared()
    fun getCurrentlyLoadedModel(): LLMModel?
    /** Backend the current session was created with (GPU/CPU); null if unknown or not applicable. */
    fun getCurrentlyLoadedBackend(): LlmInference.Backend?
    fun getMemoryWarningForImages(images: List<Bitmap>): String?
    fun wasSessionRecentlyReset(chatId: String): Boolean
    // Allow runtime update of generation parameters (from UI dialog)
    fun setGenerationParameters(maxTokens: Int?, topK: Int?, topP: Float?, temperature: Float?)
    // Get current modality disabled states
    fun isVisionCurrentlyDisabled(): Boolean
    fun isAudioCurrentlyDisabled(): Boolean
    fun isGpuBackendEnabled(): Boolean
    // Return the applied/effective max tokens for a model (honors overrides)
    fun getEffectiveMaxTokens(model: LLMModel): Int
}

/**
 * Data class to hold MediaPipe LLM inference engine and session
 */
data class LlmModelInstance(val engine: LlmInference, var session: LlmInferenceSession)

/**
 * MediaPipe-based implementation of InferenceService following Google AI Edge Gallery pattern.
 * Uses a single session approach: one session per model that gets reset as needed.
 * 
 * Backend Selection Logic:
 * - Uses model.supportsGpu flag to determine GPU availability
 * - User preferences override automatic selection
 * - Simplified approach for better stability
 */
class MediaPipeInferenceService(private val applicationContext: Context) : InferenceService {
    
    // Get current locale-aware context
    private fun getCurrentContext(): Context {
        // Get the current locale from LocaleHelper
        val currentLocale = LocaleHelper.getCurrentLocale(applicationContext)
        Log.d(TAG, "getCurrentContext - current locale: $currentLocale")
        
        // Create a fresh context with the current locale
        val configuration = Configuration(applicationContext.resources.configuration)
        configuration.setLocale(currentLocale)
        
        val newContext = applicationContext.createConfigurationContext(configuration)
        Log.d(TAG, "getCurrentContext - new context locale: ${newContext.resources.configuration.locales[0]}")
        return newContext
    }
    
    private var modelInstance: LlmModelInstance? = null
    private var currentModel: LLMModel? = null
    private var currentBackend: LlmInference.Backend? = null
    // Track if vision/audio is disabled for current model loading
    private var isVisionDisabled: Boolean = false
    private var isAudioDisabled: Boolean = false
    // Optional overrides provided by UI (null means use defaults)
    private var overrideMaxTokens: Int? = null
    private var overrideTopK: Int? = null
    private var overrideTopP: Float? = null
    private var overrideTemperature: Float? = null
    // Estimated tokens accumulated in current session (prompt + responses); heuristic
    private var estimatedSessionTokens: Int = 0
    private var isGenerating: Boolean = false
    // Track chats where we aborted due to repetition so we can reset session after close
    private val repetitionAbortFlags = mutableSetOf<String>()
    
    // Track when sessions are reset to help ChatViewModel use minimal context
    private val sessionResetTimes = mutableMapOf<String, Long>()
    
    // Web search service for enhanced responses
    private val webSearchService: WebSearchService = DuckDuckGoSearchService()
    
    // Mutex to prevent race conditions during session operations
    private val sessionMutex = Mutex()
    
    companion object {
        private const val TAG = "MediaPipeInference"
        private const val DEFAULT_MAX_TOKENS = 1024
        private const val DEFAULT_TOP_K = 40
        private const val DEFAULT_TOP_P = 0.8f
        private const val DEFAULT_TEMPERATURE = 0.8f
        
        /**
         * Get max tokens for model - simplified approach
         */
        private fun getMaxTokensForModel(model: LLMModel): Int {
            return model.contextWindowSize
        }

    // Public accessor for UI code to fetch the cap without instantiating the service
    @JvmStatic
    fun getMaxTokensForModelStatic(model: LLMModel): Int = getMaxTokensForModel(model)
    }

    override fun getEffectiveMaxTokens(model: LLMModel): Int {
        val defaultMaxTokens = getMaxTokensForModel(model)
        return overrideMaxTokens?.coerceAtMost(defaultMaxTokens) ?: defaultMaxTokens
    }

    override suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend?, deviceId: String?): Boolean {
        // deviceId is irrelevant for MediaPipe implementation; ignore it
        return try {
            ensureModelLoaded(model, preferredBackend, disableVision = false, disableAudio = false)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${e.message}", e)
            false
        }
    }
    
    override suspend fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend?, disableVision: Boolean, disableAudio: Boolean, deviceId: String?): Boolean {
        // deviceId is ignored for MediaPipe; preserved in signature to match interface
        return try {
            ensureModelLoaded(model, preferredBackend, disableVision, disableAudio)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model with modality settings: ${e.message}", e)
            false
        }
    }

    override fun setGenerationParameters(maxTokens: Int?, topK: Int?, topP: Float?, temperature: Float?) {
        overrideMaxTokens = maxTokens
        overrideTopK = topK
        overrideTopP = topP
        overrideTemperature = temperature
        Log.d(TAG, "Set generation parameters: maxTokens=$maxTokens topK=$topK topP=$topP temperature=$temperature")
    }

    override suspend fun unloadModel() {
        sessionMutex.withLock {
            Log.d(TAG, "Unloading current model: ${currentModel?.name}")
            
            // Release model instance
            modelInstance?.let { instance ->
                try {
                    // Cancel any ongoing generation
                    try {
                        instance.session.cancelGenerateResponseAsync()
                        Log.d(TAG, "Cancelled ongoing generation during unload")
                    } catch (e: Exception) {
                        Log.d(TAG, "No ongoing generation to cancel during unload: ${e.message}")
                    }
                    
                    // Close session
                    instance.session.close()
                    Log.d(TAG, "Closed session during unload")
                    
                    // Close engine
                    instance.engine.close()
                    Log.d(TAG, "Closed engine during unload")
                    
                } catch (e: Exception) {
                    Log.w(TAG, "Error during model unload: ${e.message}")
                }
            }
            
            // Clear references
            modelInstance = null
            currentModel = null
            currentBackend = null
            
            Log.d(TAG, "Model unloaded successfully")
        }
    }

    override suspend fun resetChatSession(chatId: String) {
        sessionMutex.withLock {
            Log.d(TAG, "Resetting session for chat $chatId")
            
            // Record the session reset
            recordSessionReset(chatId)
            
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
                
                // Longer delay to let cancellation complete, especially for "Previous invocation still processing" errors
                delay(1000) // Increased delay to ensure cancellation completes
                
                try {
                    instance.session.close()
                    Log.d(TAG, "Closed existing session")
                } catch (closeException: Exception) {
                    if (closeException.message?.contains("Previous invocation still processing") == true) {
                        Log.w(TAG, "Session still processing, forcing close after delay")
                        delay(2000) // Wait even longer
                        try {
                            instance.session.close()
                            Log.d(TAG, "Forced close of session")
                        } catch (forceCloseException: Exception) {
                            Log.e(TAG, "Failed to force close session: ${forceCloseException.message}")
                            // Continue with session creation anyway
                        }
                    } else {
                        throw closeException
                    }
                }
                
//                // Try to close the session with retry logic
//                var closeAttempts = 0
//                val maxCloseAttempts = 3
//                while (closeAttempts < maxCloseAttempts) {
//                    try {
//                        instance.session.close()
//                        Log.d(TAG, "Closed existing session")
//                        break
//                    } catch (e: Exception) {
//                        closeAttempts++
//                        if (e.message?.contains("Previous invocation still processing") == true) {
//                            Log.w(TAG, "Session still processing, waiting longer (attempt $closeAttempts/$maxCloseAttempts)")
//                            delay(1000) // Wait longer for processing to complete
//                        } else {
//                            Log.w(TAG, "Failed to close session (attempt $closeAttempts/$maxCloseAttempts): ${e.message}")
//                            if (closeAttempts >= maxCloseAttempts) {
//                                Log.e(TAG, "Failed to close session after $maxCloseAttempts attempts")
//                                throw e
//                            }
//                            delay(500)
//                        }
//                    }
//                }
                
                // Create new session with same options
                val newSession = createSession(instance.engine)
                instance.session = newSession
                Log.d(TAG, "Created new session for chat $chatId")
                estimatedSessionTokens = 0
                
                // Give MediaPipe time to clean up (Gallery uses 500ms)
                delay(500)
                
            } catch (e: Exception) {
                Log.e(TAG, "Error resetting session for chat $chatId: ${e.message}", e)
                // Avoid full model reload for benign cancellations (e.g., repetition abort path)
                val isBenignCancellation = (e is CancellationException) ||
                        (e.message?.contains("ProducerCoroutine was cancelled", ignoreCase = true) == true)

                try {
                    val instance = modelInstance
                    if (instance != null) {
                        try {
                            instance.session.close()
                        } catch (closeEx: Exception) {
                            Log.w(TAG, "Error closing session during recovery: ${closeEx.message}")
                        }
                        instance.session = createSession(instance.engine)
                        estimatedSessionTokens = 0
                        Log.d(TAG, "Recreated session on existing engine after reset error (benign=${isBenignCancellation})")
                        delay(300)
                    } else if (!isBenignCancellation) {
                        // If we truly have no instance and this wasn't a benign cancellation, last resort reload
                        currentModel?.let { model ->
                            loadModelFromPath(model, currentBackend)
                            Log.d(TAG, "Reloaded model as last resort due to missing instance")
                        }
                    } else {
                        Log.w(TAG, "No model instance present; skipping model reload after benign cancellation")
                    }
                } catch (retryException: Exception) {
                    Log.e(TAG, "Recovery after reset error failed: ${retryException.message}", retryException)
                }
            }
            
            Log.d(TAG, "Session reset completed for chat $chatId")
        }
    }

    private suspend fun ensureModelLoaded(model: LLMModel, preferredBackend: LlmInference.Backend? = null) {
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
                loadModelFromPath(model, preferredBackend)
            }
        }
    }

    private suspend fun ensureModelLoaded(model: LLMModel, preferredBackend: LlmInference.Backend? = null, disableVision: Boolean = false, disableAudio: Boolean = false) {
        if (currentModel?.name != model.name) {
            val modalityInfo = buildList {
                if (disableVision) add("vision disabled")
                if (disableAudio) add("audio disabled")
                if (isEmpty()) add("all modalities enabled")
            }.joinToString(", ")
            
            Log.d(TAG, "Model changed from ${currentModel?.name} to ${model.name}, reloading with $modalityInfo")
            
            // Store the modality disabled states
            isVisionDisabled = disableVision
            isAudioDisabled = disableAudio
            
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
                loadModelFromPath(model, preferredBackend, disableVision, disableAudio)
            }
        }
    }

    private suspend fun loadModelFromPath(model: LLMModel, preferredBackend: LlmInference.Backend? = null, disableVision: Boolean = false, disableAudio: Boolean = false) {
        try {
            val modelFile: File
            
            // Handle imported models (Custom source with URI)
            if (model.source == "Custom" && model.url.startsWith("content://")) {
                Log.d(TAG, "Loading imported model from URI: ${model.url}")
                
                // For imported models, we need to copy to local storage for MediaPipe
                // MediaPipe requires a local file path, not a content URI
                val targetFile = File(applicationContext.filesDir, "models/${model.localFileName()}")
                targetFile.parentFile?.mkdirs()
                
                // Only copy if file doesn't exist or is outdated
                if (!targetFile.exists()) {
                    try {
                        applicationContext.contentResolver.openInputStream(Uri.parse(model.url))?.use { inputStream ->
                            targetFile.outputStream().use { outputStream ->
                                inputStream.copyTo(outputStream)
                            }
                        }
                        Log.d(TAG, "Copied imported model to: ${targetFile.absolutePath}")
                    } catch (e: SecurityException) {
                        Log.e(TAG, "Permission denied for URI: ${model.url}")
                        throw IllegalStateException("Permission denied for imported model. Please re-import the model.", e)
                    }
                } else {
                    Log.d(TAG, "Using existing copied model: ${targetFile.absolutePath}")
                }
                
                if (targetFile.exists()) {
                    modelFile = targetFile
                } else {
                    throw IllegalStateException("Failed to access imported model from URI: ${model.url}")
                }
            } else {
                // Handle regular models (assets or files directory)
                val modelAssetPath = if (model.url.startsWith("file://models/")) {
                    model.url.removePrefix("file://")
                } else {
                    "models/${model.localFileName()}"
                }
                
                Log.d(TAG, "Loading model from: $modelAssetPath")
                
                // Check if model exists in assets folder
                modelFile = try {
                    applicationContext.assets.open(modelAssetPath).use { 
                        // File exists in assets, copy to files directory
                        val targetFile = File(applicationContext.filesDir, "models/${model.localFileName()}")
                        targetFile.parentFile?.mkdirs()
                        
                        if (!targetFile.exists()) {
                            targetFile.outputStream().use { outputStream ->
                                applicationContext.assets.open(modelAssetPath).use { inputStream ->
                                    inputStream.copyTo(outputStream)
                                }
                            }
                            Log.d(TAG, "Copied model to ${targetFile.absolutePath}")
                        }
                        targetFile
                    }
                } catch (e: Exception) {
                    // Try to find model in files directory
                    val modelFile = File(applicationContext.filesDir, modelAssetPath)
                    if (modelFile.exists()) {
                        Log.d(TAG, "Model found in files directory: ${modelFile.absolutePath}")
                        modelFile
                    } else {
                        throw IllegalStateException("Model not found in assets or files: $modelAssetPath")
                    }
                }
            }
            
            // Determine backend - use preferred backend if provided, otherwise use model's GPU support
            val backend = preferredBackend ?: if (model.supportsGpu) LlmInference.Backend.GPU else LlmInference.Backend.CPU
            
            Log.d(TAG, "Selected backend: $backend for model: ${modelFile.name} ${if (preferredBackend != null) "(user preference)" else "(auto-selected)"}")
            
            // Determine max tokens based on model configuration (allow override)
            val defaultMaxTokens = getMaxTokensForModel(model)
            val maxTokens = overrideMaxTokens?.coerceAtMost(defaultMaxTokens) ?: defaultMaxTokens

            Log.d(TAG, "Model configuration:")
            Log.d(TAG, "  - Name: ${model.name}")
            Log.d(TAG, "  - File: ${modelFile.name}")
            Log.d(TAG, "  - Path: ${modelFile.absolutePath}")
            Log.d(TAG, "  - Context window: ${model.contextWindowSize}")
            Log.d(TAG, "  - Max tokens: $maxTokens (default was $defaultMaxTokens)")
            Log.d(TAG, "  - Backend: $backend")
            Log.d(TAG, "  - Supports vision: ${model.supportsVision}")
            Log.d(TAG, "  - Supports audio: ${model.supportsAudio}")
            
            // Create LLM inference options
            val optionsBuilder = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelFile.absolutePath)
                .setMaxTokens(maxTokens)
                .setPreferredBackend(backend)
                
            // Enable vision modality for multimodal models (following Google AI Edge Gallery pattern)
            // In version 0.10.29+, setting maxNumImages > 0 triggers vision model pre-loading
            // which significantly increases initialization time. Only enable when actually needed.
            if (model.supportsVision && !disableVision) {
                optionsBuilder.setMaxNumImages(10) // Allow up to 10 images per session
                Log.d(TAG, "  - Enabled vision modality with max 10 images (WARNING: increases load time)")
            } else {
                optionsBuilder.setMaxNumImages(0) // Explicitly disable for non-vision models or when vision is disabled
                if (disableVision) {
                    Log.d(TAG, "  - Vision modality disabled by user (maxNumImages=0) - faster loading")
                } else {
                Log.d(TAG, "  - Vision modality disabled (maxNumImages=0) - faster loading")
                }
            }
            
            // Try to create inference engine with audio support if claimed by model and not disabled
            // === Simplified engine creation (align with Google AI Edge Gallery) ===
            var llmInference: LlmInference? = null
            var actualAudioSupport = false

            // If the model claims audio support and the user did not disable it, add AudioModelOptions directly.
            if (model.supportsAudio && !disableAudio) {
                optionsBuilder.setAudioModelOptions(AudioModelOptions.builder().build())
                actualAudioSupport = true
                Log.d(TAG, "  - Audio modality enabled")
            } else {
                if (disableAudio && model.supportsAudio) {
                    Log.d(TAG, "  - Audio modality disabled by user")
                } else if (!model.supportsAudio) {
                    Log.d(TAG, "  - Model does not support audio, skipping audio options")
                }
            }

            // Build final options and create a single inference engine
            val options = optionsBuilder.build()
            llmInference = LlmInference.createFromOptions(applicationContext, options)
            // === End simplified engine creation ===
            
            // Preserve the model's base capabilities. Use the original model object so supportsAudio
            // stays TRUE even if the user disabled audio for this particular load.
            // We rely on isAudioDisabled flag to know whether the session should include audio.
            currentModel = model
            
            // Create initial session
            val session = createSession(llmInference)
            
            // Store model instance and backend
            modelInstance = LlmModelInstance(engine = llmInference, session = session)
            currentBackend = backend
            estimatedSessionTokens = 0
            
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
        Log.d(TAG, "Model supports audio: ${model?.supportsAudio}")
        
        // Configure modality support based on model capabilities and user settings
        val needsVisionModality = model?.supportsVision == true && !isVisionDisabled
        val needsAudioModality = model?.supportsAudio == true && !isAudioDisabled
        
        Log.d(TAG, "Session creation - needsVisionModality: $needsVisionModality, needsAudioModality: $needsAudioModality")
        
        try {
            val session = LlmInferenceSession.createFromOptions(
                engine,
                LlmInferenceSession.LlmInferenceSessionOptions.builder()
                    .setTopK(overrideTopK ?: DEFAULT_TOP_K)
                    .setTemperature(overrideTemperature ?: DEFAULT_TEMPERATURE)
                    .setTopP(overrideTopP ?: DEFAULT_TOP_P)
                    .setRandomSeed(System.currentTimeMillis().toInt())
                    .setGraphOptions(
                        com.google.mediapipe.tasks.genai.llminference.GraphOptions.builder()
                            .setEnableVisionModality(needsVisionModality)
                            .setEnableAudioModality(needsAudioModality)
                            .build()
                    )
                    .build()
            )
            
            val modalityTypes = buildList {
                if (needsVisionModality) add("vision")
                if (needsAudioModality) add("audio")
            }.joinToString(" + ")
            
            if (modalityTypes.isNotEmpty()) {
                Log.d(TAG, "Session created with $modalityTypes modality ENABLED for model ${model?.name}")
            } else {
                Log.d(TAG, "Session created with text-only modality for model ${model?.name}")
            }
            
            Log.d(TAG, "Successfully created session for model ${model?.name}")
            estimatedSessionTokens = 0
            return session
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create session for model ${model?.name}: ${e.message}", e)
            throw e
        }
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        ensureModelLoaded(model)
        
        // Inject Kid Mode Instruction if enabled (Global Injection)
        val kidModeManager = com.llmhub.llmhub.utils.KidModeManager(applicationContext)
        var finalPrompt = prompt
        if (kidModeManager.isKidModeEnabled.value) {
            Log.d(TAG, "Global Injection: Injecting Kid Mode Instruction")
            val kidInstruction = com.llmhub.llmhub.utils.KidModeManager.SYSTEM_INSTRUCTION
            if (finalPrompt.startsWith("system:")) {
                finalPrompt = finalPrompt.replaceFirst("system:", "system: $kidInstruction\n\n")
            } else {
                finalPrompt = "system: $kidInstruction\n\n$finalPrompt"
            }
        }
        
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
                localSession.addQueryChunk(finalPrompt)
                
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

        // Inject Kid Mode Instruction if enabled (Global Injection)
        val kidModeManager = com.llmhub.llmhub.utils.KidModeManager(applicationContext)
        var finalPrompt = prompt
        if (kidModeManager.isKidModeEnabled.value) {
            Log.d(TAG, "Global Injection: Injecting Kid Mode Instruction")
            val kidInstruction = com.llmhub.llmhub.utils.KidModeManager.SYSTEM_INSTRUCTION
            if (finalPrompt.startsWith("system:")) {
                finalPrompt = finalPrompt.replaceFirst("system:", "system: $kidInstruction\n\n")
            } else {
                finalPrompt = "system: $kidInstruction\n\n$finalPrompt"
            }
        }
        
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
            localSession.addQueryChunk(finalPrompt)
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

    override suspend fun generateResponseStreamWithSession(
        prompt: String, 
        model: LLMModel, 
        chatId: String, 
        images: List<Bitmap>, 
        audioData: ByteArray?,
        webSearchEnabled: Boolean,
        imagePaths: List<String>
    ): Flow<String> = callbackFlow {
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
        
        // Extract the current user message from the prompt for web search detection
        val currentUserMessage = extractCurrentUserMessage(prompt)
        val needsWebSearch = webSearchEnabled && SearchIntentDetector.needsWebSearch(currentUserMessage)
        var enhancedPrompt = prompt

        // Inject Kid Mode Instruction if enabled (Global Injection) - MOVED DOWN
        // to prevent overwrite by Web Search logic
        
        try {
            if (needsWebSearch) {
                Log.d(TAG, "Web search detected for chat $chatId. Current message: '$currentUserMessage'")
                trySend(getCurrentContext().getString(R.string.web_searching))
                
                try {
                    val searchQuery = SearchIntentDetector.extractSearchQuery(currentUserMessage)
                    Log.d(TAG, "Extracted search query: '$searchQuery'")
                    
                    val searchResults = webSearchService.search(searchQuery, maxResults = 5)
                    
                    if (searchResults.isNotEmpty()) {
                        Log.d(TAG, "Found ${searchResults.size} search results")
                        trySend(getCurrentContext().getString(R.string.web_search_found_results, searchResults.size))
                        
                        // Create enhanced prompt with search results
                        val resultsText = searchResults.joinToString("\n\n") { result ->
                            "SOURCE: ${result.source}\nTITLE: ${result.title}\nCONTENT: ${result.snippet}\n---"
                        }
                        
                        // Extract just the current user question for better clarity
                        enhancedPrompt = """
                            CURRENT WEB SEARCH RESULTS:
                            $resultsText
                            
                            Based on the above current web search results, please answer the user's question: "$currentUserMessage"
                            
                            IMPORTANT INSTRUCTIONS:
                            - Use ONLY the information from the web search results above
                            - If the search results contain the answer, provide a clear and specific response
                            - If the search results don't contain enough information, say so clearly
                            - For dates and events, be specific based on what you find in the results
                            - Do not make up information not found in the search results
                            
                            Answer the question directly and clearly:
                        """.trimIndent()
                        
                        Log.d(TAG, "Enhanced prompt created with ${searchResults.size} search results")
                        Log.d(TAG, "User question: '$currentUserMessage'")
                        Log.d(TAG, "Search results preview: ${resultsText.take(200)}...")
                    } else {
                        Log.w(TAG, "No search results found for query: '$searchQuery'")
                        trySend(getCurrentContext().getString(R.string.web_search_no_results) + "\n\n")
                        // Continue with original prompt
                    }
                } catch (searchException: Exception) {
                    Log.e(TAG, "Web search failed for chat $chatId", searchException)
                    trySend(getCurrentContext().getString(R.string.web_search_failed, searchException.message ?: "Unknown error") + "\n\n")
                    // Continue with original prompt
                }
            }
            
            // For web search queries, reset session to ensure clean context
            if (needsWebSearch) {
                try {
                    Log.d(TAG, "Resetting session for web search to ensure clean context")
                    resetChatSession(chatId)
                    delay(50) // Brief delay after reset
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to reset session for web search: ${e.message}")
                }
            }
            
            // Use the single session from the model instance (Gallery approach)
            val instance = modelInstance ?: throw IllegalStateException("No model loaded")
            val session = instance.session
            
            // Check if we're already generating and wait for completion
            if (isGenerating) {
                Log.w(TAG, "Generation already in progress for chat $chatId, waiting...")
                var waitAttempts = 0
                val maxWaitAttempts = 50 // 5 seconds total
                while (isGenerating && waitAttempts < maxWaitAttempts) {
                    delay(100)
                    waitAttempts++
                }
                if (isGenerating) {
                    Log.e(TAG, "Generation still in progress after waiting, forcing reset")
                    resetChatSession(chatId)
                    return@callbackFlow
                }
            }
            
            // Check if previous generation is still processing and cancel it
            try {
                session.cancelGenerateResponseAsync()
                Log.d(TAG, "Cancelled any previous generation for chat $chatId")
                delay(500) // Wait longer for cancellation to complete
            } catch (e: Exception) {
                Log.d(TAG, "No previous generation to cancel: ${e.message}")
                // If cancellation fails, the session might be in an invalid state
                // Try to reset the session immediately
                if (e.message?.contains("Previous invocation still processing") == true) {
                    Log.w(TAG, "Session still processing, resetting immediately for chat $chatId")
                    try {
                        session.close()
                        val newSession = createSession(instance.engine)
                        instance.session = newSession
                        estimatedSessionTokens = 0
                        Log.d(TAG, "Created fresh session due to processing conflict")
                        delay(200) // Brief delay after reset
                    } catch (resetException: Exception) {
                        Log.e(TAG, "Failed to reset session after processing conflict: ${resetException.message}")
                        throw resetException
                    }
                }
            }
            
            // Check token count and reset session if approaching limit (Gallery approach)
            val defaultMaxTokens = getMaxTokensForModel(model)
            val maxTokens = overrideMaxTokens?.coerceAtMost(defaultMaxTokens) ?: defaultMaxTokens
            Log.d(TAG, "Using token limits - defaultMaxTokens=$defaultMaxTokens overrideMaxTokens=${overrideMaxTokens ?: "null"} effectiveMaxTokens=$maxTokens")
            // Reserve ~1/3 for model response; prevent sending input when it eats into reserve
            val currentUserInput = extractCurrentUserMessage(prompt)
            val promptTokens = (currentUserInput.length / 4).coerceAtLeast(1)
            val outputReserve = (maxTokens * 0.33).toInt().coerceAtLeast(128)
            var currentTokens = estimatedSessionTokens
            // If our estimate undercounts (e.g., after recovery) fall back to session.sizeInTokens(prompt) heuristic not available; keep estimate
            
            Log.d(TAG, "Token usage for chat $chatId:")
            Log.d(TAG, "  - Current session tokens: $currentTokens")
            Log.d(TAG, "  - Prompt tokens: $promptTokens")
            Log.d(TAG, "  - Max tokens: $maxTokens")
            
            // If adding the prompt would exceed ~80% of max tokens, reset the session
            // Use a lower threshold since we're using conservative estimation
            val tokenThreshold = (maxTokens - outputReserve).coerceAtLeast(1)
            if (currentTokens + promptTokens >= tokenThreshold) {
                Log.w(TAG, "Token count ($currentTokens + $promptTokens = ${currentTokens + promptTokens}) approaching limit ($maxTokens)")
                Log.w(TAG, "Resetting session for chat $chatId to prevent OUT_OF_RANGE error")
                
                // Record the session reset
                recordSessionReset(chatId)
                
                // Reset session before it gets full
                sessionMutex.withLock {
                    try {
                        session.close()
                        Log.d(TAG, "Closed session due to token limit approach")
                        
                        // Create new session
                        val newSession = createSession(instance.engine)
                        instance.session = newSession
                        Log.d(TAG, "Created fresh session for chat $chatId")
                        estimatedSessionTokens = 0
                        
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
            // Update estimation after any reset
            currentTokens = estimatedSessionTokens

            // Inject Kid Mode Instruction if enabled (Global Injection) - Check AGAIN
            // This ensures we inject even if Web Search modified the prompt.
            val kidModeManager = com.llmhub.llmhub.utils.KidModeManager(applicationContext)
            if (kidModeManager.isKidModeEnabled.value) {
                Log.d(TAG, "Global Injection: Injecting Kid Mode Instruction (Final Stage)")
                val kidInstruction = com.llmhub.llmhub.utils.KidModeManager.SYSTEM_INSTRUCTION
                if (enhancedPrompt.startsWith("system:")) {
                    enhancedPrompt = enhancedPrompt.replaceFirst("system:", "system: $kidInstruction\n\n")
                } else {
                    // If prompt was enhanced by web search, it might not have "system:" prefix anymore but is just text.
                    // We just prepend the instruction.
                    enhancedPrompt = "system: $kidInstruction\n\n$enhancedPrompt"
                }
            }
            
            // CRITICAL: For vision models, text query MUST be added before images
            // This is required by MediaPipe's vision implementation
            if (enhancedPrompt.trim().isNotEmpty()) {
                Log.d(TAG, "Adding text query to session for chat $chatId: '${enhancedPrompt.take(100)}...'")
                try {
                    currentSession.addQueryChunk(enhancedPrompt)
                    estimatedSessionTokens += promptTokens
                } catch (e: Exception) {
                    val msg = e.message ?: ""
                    if (msg.contains("Previous invocation still processing", ignoreCase = true)) {
                        Log.w(TAG, "Session busy on addQueryChunk; doing one session-only recreate and retry once")
                        // One-shot session-only recreate, no model reload
                        sessionMutex.withLock {
                            modelInstance?.let { inst ->
                                try { inst.session.close() } catch (_: Exception) {}
                                inst.session = createSession(inst.engine)
                            }
                        }
                        // single retry
                        currentSession.addQueryChunk(enhancedPrompt)
                        estimatedSessionTokens += promptTokens
                    } else {
                        throw e
                    }
                }
            } else if (images.isNotEmpty() && model.supportsVision) {
                // If we have images but no text, add a default query for vision models
                Log.d(TAG, "Adding default vision query for images in chat $chatId")
                try {
                    val defaultQuery = "What do you see in this image?"
                    currentSession.addQueryChunk(defaultQuery)
                    estimatedSessionTokens += kotlin.math.ceil(defaultQuery.length / 4.0).toInt().coerceAtLeast(1)
                } catch (e: Exception) {
                    val msg = e.message ?: ""
                    if (msg.contains("Previous invocation still processing", ignoreCase = true)) {
                        Log.w(TAG, "Session busy on vision default query; session-only recreate then single retry")
                        sessionMutex.withLock {
                            modelInstance?.let { inst ->
                                try { inst.session.close() } catch (_: Exception) {}
                                inst.session = createSession(inst.engine)
                            }
                        }
                        val defaultQuery = "What do you see in this image?"
                        currentSession.addQueryChunk(defaultQuery)
                        estimatedSessionTokens += kotlin.math.ceil(defaultQuery.length / 4.0).toInt().coerceAtLeast(1)
                    } else {
                        throw e
                    }
                }
            }
            // Note: Audio handling - ChatViewModel adds "Respond to the audio:" instruction
            // when user sends audio without text, so we don't need a separate check here
            
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
            
            // Add audio AFTER text query and images (MediaPipe requirement for multimodal models)
            if (audioData != null && model.supportsAudio) {
                Log.d(TAG, "Adding audio data to session for chat $chatId (${audioData.size} bytes)")
                try {
                    // Validate audio data
                    if (audioData.isEmpty()) {
                        Log.e(TAG, "Audio data is empty")
                    } else {
                        // Check if session was properly created with audio modality
                        Log.d(TAG, "Attempting to add audio to session...")
                        
                        // Add audio data to session (MediaPipe expects mono WAV format)
                        currentSession.addAudio(audioData)
                        Log.d(TAG, "Successfully added audio data to session")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to add audio data to session: ${e.message}", e)

                    // If audio modality isn't enabled in the current engine, don't retry
                    if (e.message?.contains("Audio modality is not enabled") == true) {
                        Log.w(TAG, "Engine/session has audio disabled. Skipping audio for this request (but keeping model's audio capability intact)")
                        // Don't modify currentModel.supportsAudio - it should reflect the model's base capability
                    }
                    // Continue with text/image processing even if audio fails
                }
            } else if (audioData != null && !model.supportsAudio) {
                Log.w(TAG, "Model ${model.name} does not support audio, ignoring audio input")
            } else if (audioData == null && model.supportsAudio) {
                Log.d(TAG, "No audio provided for audio-capable model ${model.name}")
            }
            
            val responseBuilder = StringBuilder()
            isGenerating = true
            val isLlama = isLlamaModel(model)
            
            currentSession.generateResponseAsync { partialResult, done ->
                var processedResult = partialResult
                var forceDone = done
                
                // For Llama models, check for stop tokens
                if (isLlama && processedResult.isNotEmpty()) {
                    val (cleaned, shouldStop) = processLlamaStopTokens(processedResult)
                    processedResult = cleaned
                    if (shouldStop) {
                        forceDone = true
                        Log.d(TAG, "Llama stop token detected for chat $chatId - ending generation")
                    }
                }
                
                if (!isClosedForSend && processedResult.isNotEmpty()) {
                    trySend(processedResult)
                    responseBuilder.append(processedResult)
                }
                
                // Simple repetition detection: check recent window for repeated n-grams (only for Gemma-3 1B models)
                if (isGemma31BModel(model)) {
                    val recent = responseBuilder.takeLast(600).toString()
                    if (recent.isNotEmpty() && hasRepetitionPattern(recent)) {
                        Log.w(TAG, "Detected repetition pattern in output for chat $chatId - aborting stream and scheduling session reset")
                        repetitionAbortFlags.add(chatId)
                        isGenerating = false
                        isGenerationComplete = true
                        close()
                        return@generateResponseAsync
                    }
                }
                
                if (forceDone) {
                    isGenerationComplete = true
                    isGenerating = false
                    // Update session token count with response tokens (using standardized 4 chars/token estimation)
                    try {
                        val fullResponse = responseBuilder.toString()
                        val responseTokens = kotlin.math.ceil(fullResponse.length / 4.0).toInt().coerceAtLeast(1)
                        estimatedSessionTokens += responseTokens
                        Log.d(TAG, "Updated session tokens: +$responseTokens = $estimatedSessionTokens")
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to update token count: ${e.message}")
                    }
                    close()
                }
            }
            
        } catch (e: Exception) {
            isGenerating = false
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
                    val isLlama = isLlamaModel(model)
                    
                    Log.d(TAG, "Created new session for recovery, attempting generation retry")
                    
                    // Re-add text query first (CRITICAL for vision models)
                    if (enhancedPrompt.trim().isNotEmpty()) {
                        Log.d(TAG, "Re-adding text query to recovery session for chat $chatId")
                        session.addQueryChunk(enhancedPrompt)
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
                        var processedResult = partialResult
                        var forceDone = done
                        
                        // For Llama models, check for stop tokens
                        if (isLlama && processedResult.isNotEmpty()) {
                            val (cleaned, shouldStop) = processLlamaStopTokens(processedResult)
                            processedResult = cleaned
                            if (shouldStop) {
                                forceDone = true
                                Log.d(TAG, "Llama stop token detected in recovery for chat $chatId")
                            }
                        }
                        
                        if (!isClosedForSend && processedResult.isNotEmpty()) {
                            trySend(processedResult)
                        }
                        if (forceDone) {
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
                            val isLlama = isLlamaModel(model)
                            
                            Log.d(TAG, "Force recreated session, attempting generation retry")
                            
                            // Re-add text query first (CRITICAL for vision models)
                            if (enhancedPrompt.trim().isNotEmpty()) {
                                Log.d(TAG, "Re-adding text query to force recreated session for chat $chatId")
                                session.addQueryChunk(enhancedPrompt)
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
                                var processedResult = partialResult
                                var forceDone = done
                                
                                // For Llama models, check for stop tokens
                                if (isLlama && processedResult.isNotEmpty()) {
                                    val (cleaned, shouldStop) = processLlamaStopTokens(processedResult)
                                    processedResult = cleaned
                                    if (shouldStop) {
                                        forceDone = true
                                        Log.d(TAG, "Llama stop token detected in force recovery for chat $chatId")
                                    }
                                }
                                
                                if (!isClosedForSend && processedResult.isNotEmpty()) {
                                    trySend(processedResult)
                                }
                                if (forceDone) {
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
            isGenerating = false
            Log.d(TAG, "Generation complete for chat $chatId")
            // Don't close the session here - it's managed by the model instance
            // If we aborted due to repetition, reset the session to clear cached state
            if (repetitionAbortFlags.remove(chatId)) {
                Log.w(TAG, "Resetting session after repetition abort for chat $chatId")
                // Launch a coroutine since awaitClose is not a suspend context
                launch(Dispatchers.IO) {
                    try {
                        resetChatSession(chatId)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to reset session after repetition abort: ${e.message}")
                    }
                }
            }
        }
    }.flowOn(Dispatchers.IO)

    // Detect simple repetition patterns in a small window of text.
    // Looks for any n-gram (n=4) that occurs 3+ times in the recent window.
    private fun hasRepetitionPattern(textWindow: String, n: Int = 4, repeats: Int = 3): Boolean {
        if (textWindow.length < n * repeats) return false
        val tokens = textWindow.split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (tokens.size < n * repeats) return false
        val counts = HashMap<String, Int>()
        for (i in 0..tokens.size - n) {
            val key = tokens.subList(i, i + n).joinToString(" ")
            val c = (counts[key] ?: 0) + 1
            if (c >= repeats) return true
            counts[key] = c
        }
        return false
    }
    
    /**
     * Check if model is a Llama model based on name or source
     */
    private fun isLlamaModel(model: LLMModel): Boolean {
        return model.name.contains("Llama", ignoreCase = true) || 
               model.source.contains("Llama", ignoreCase = true)
    }
    
    /**
     * Check if model is a Gemma-3 1B model based on name
     */
    private fun isGemma31BModel(model: LLMModel): Boolean {
        return model.name.contains("Gemma-3 1B", ignoreCase = true)
    }
    
    /**
     * Clean Llama stop tokens from output and check if generation should stop
     * Returns: Pair(cleanedText, shouldStop)
     */
    private fun processLlamaStopTokens(text: String): Pair<String, Boolean> {
        // Llama 3.2 stop tokens
        val stopTokens = listOf(
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|end|>",
            "</s>"
        )
        
        var cleaned = text
        var shouldStop = false
        
        for (stopToken in stopTokens) {
            if (cleaned.contains(stopToken)) {
                shouldStop = true
                // Remove the stop token and everything after it
                val index = cleaned.indexOf(stopToken)
                cleaned = cleaned.substring(0, index)
                break
            }
        }
        
        // Also clean up any exposed role tokens
        cleaned = cleaned.replace("<|start_header_id|>", "")
                        .replace("<|end_header_id|>", "")
                        .replace("assistant:", "")
                        .replace("user:", "")
        
        return Pair(cleaned, shouldStop)
    }

    /**
     * Check if the exception is a known MediaPipe session error
     */
    private fun isMediaPipeSessionError(e: Exception): Boolean {
        val errorMessage = e.message?.lowercase() ?: ""
        return errorMessage.contains("detokenizercalculator") ||
                errorMessage.contains("id >= 0") ||
                errorMessage.contains("no id available to be decoded") ||
                errorMessage.contains("previous invocation still processing") ||
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

    override fun getCurrentlyLoadedBackend(): LlmInference.Backend? {
        return currentBackend
    }
    
    override fun isVisionCurrentlyDisabled(): Boolean {
        return isVisionDisabled
    }
    
    override fun isAudioCurrentlyDisabled(): Boolean {
        return isAudioDisabled
    }
    
    override fun isGpuBackendEnabled(): Boolean {
        return currentBackend == LlmInference.Backend.GPU
    }

    /**
     * Force recreate the entire session when reset fails (last resort recovery)
     */
    private suspend fun forceRecreateSession(): Boolean {
        return sessionMutex.withLock {
            try {
                Log.d(TAG, "Force recreating session (session-only)")
                val instance = modelInstance
                if (instance != null) {
                    try {
                        instance.session.close()
                    } catch (e: Exception) {
                        Log.d(TAG, "Error closing session during force recreate: ${e.message}")
                    }
                    // Do not close engine; rebuild session only
                    instance.session = createSession(instance.engine)
                    estimatedSessionTokens = 0
                    delay(300)
                    Log.d(TAG, "Successfully recreated session without model reload")
                    true
                } else {
                    // Only if there is truly no instance, reload as a last last resort
                    val currentModelBackup = currentModel
                    if (currentModelBackup != null) {
                        try {
                            loadModelFromPath(currentModelBackup, currentBackend)
                            Log.d(TAG, "Recreated engine+session due to missing instance")
                            true
                        } catch (e: Exception) {
                            Log.e(TAG, "Failed to reload model during force recreate: ${e.message}", e)
                            false
                        }
                    } else {
                        Log.e(TAG, "No model to reload during force recreate")
                        false
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception during session-only force recreate: ${e.message}", e)
                false
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
        
        // Simplified approach - just provide general guidance
        return "ℹ️ Vision processing may use significant memory. If you experience crashes, try reducing image size or using text-only queries."
    }

    override fun getMemoryWarningForImages(images: List<Bitmap>): String? {
        return checkMemoryConstraintsForVision(images)
    }
    
    /**
     * Extract the current user message from a conversation prompt
     * This handles various prompt formats and extracts just the latest user input
     */
    private fun extractCurrentUserMessage(prompt: String): String {
        val lines = prompt.trim().split('\n')
        
        // Look for the last user message in the conversation
        for (i in lines.lastIndex downTo 0) {
            val line = lines[i].trim()
            if (line.startsWith("user:")) {
                return line.removePrefix("user:").trim()
            }
        }
        
        // If no "user:" prefix found, check if the entire prompt is just a user message
        // This handles cases where the prompt is minimal (like "1+1")
        if (!prompt.contains("assistant:") && !prompt.contains("user:")) {
            return prompt.trim()
        }
        
        // Fallback: return the last non-empty line that doesn't start with "assistant:"
        for (i in lines.lastIndex downTo 0) {
            val line = lines[i].trim()
            if (line.isNotEmpty() && !line.startsWith("assistant:")) {
                return line
            }
        }
        
        return prompt.trim()
    }
    
    /**
     * Check if a session was recently reset (within the last 2 seconds)
     * This helps ChatViewModel determine if it should use minimal context
     */
    override fun wasSessionRecentlyReset(chatId: String): Boolean {
        val resetTime = sessionResetTimes[chatId] ?: return false
        val timeSinceReset = System.currentTimeMillis() - resetTime
        // Extend window so downstream callers reliably detect a fresh reset
        return timeSinceReset < 10_000 // 10 seconds
    }
    
    /**
     * Record that a session was reset for a specific chat
     */
    private fun recordSessionReset(chatId: String) {
        sessionResetTimes[chatId] = System.currentTimeMillis()
        Log.d(TAG, "Recorded session reset for chat $chatId")
    }
    
    // Removed calculateTokenCount from interface; rely on chars/4 approximation at call sites
}