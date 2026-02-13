package com.llmhub.llmhub.viewmodels

import android.content.Context
import android.net.Uri
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.SavedStateHandle
import com.llmhub.llmhub.data.*
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.repository.ChatRepository
import com.llmhub.llmhub.utils.FileUtils
import com.llmhub.llmhub.R
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import java.io.File
import kotlinx.coroutines.Job
import android.os.Environment
import androidx.core.content.FileProvider
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.data.isModelFileValid
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.embedding.RagServiceManager
import com.llmhub.llmhub.embedding.ContextChunk
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import androidx.lifecycle.ViewModelProvider
import androidx.activity.ComponentActivity
import com.llmhub.llmhub.ui.components.TtsService

class ChatViewModel(
    private val inferenceService: InferenceService,
    private val repository: ChatRepository,
    private val context: Context,
    private val savedStateHandle: SavedStateHandle = SavedStateHandle()
) : ViewModel() {

    private val themePreferences = ThemePreferences(context)
    private val ragServiceManager = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(context)
    // Expose TTS service so ChatScreen can observe its isSpeaking state
    val ttsService = TtsService(context)
    
    // SharedPreferences for persisting chat settings
    private val prefs = context.getSharedPreferences("chat_prefs", Context.MODE_PRIVATE)

    companion object {
        private const val KEY_CURRENT_CHAT_ID = "current_chat_id"
        private const val KEY_CURRENT_MODEL_NAME = "current_model_name"
        private const val KEY_IS_GENERATING = "is_generating"
    }

    private val _messages = MutableStateFlow<List<MessageEntity>>(emptyList())
    val messages: StateFlow<List<MessageEntity>> = _messages.asStateFlow()

    private val _currentChat = MutableStateFlow<ChatEntity?>(null)
    val currentChat: StateFlow<ChatEntity?> = _currentChat.asStateFlow()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()

    // Streaming response state (per message)
    private val _streamingContents = MutableStateFlow<Map<String, String>>(emptyMap())
    val streamingContents: StateFlow<Map<String, String>> = _streamingContents.asStateFlow()

    private val _isLoadingModel = MutableStateFlow(false)
    val isLoadingModel: StateFlow<Boolean> = _isLoadingModel.asStateFlow()

    private val _currentlyLoadedModel = MutableStateFlow<LLMModel?>(null)
    val currentlyLoadedModel: StateFlow<LLMModel?> = _currentlyLoadedModel.asStateFlow()
    // Model the user has selected (may be in the process of loading). Use this for immediate UI feedback.
    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()
    
    // Selected backend for model loading
    private val _selectedBackend = MutableStateFlow<LlmInference.Backend?>(null)
    val selectedBackend: StateFlow<LlmInference.Backend?> = _selectedBackend.asStateFlow()

    // RAG status state
    private val _isRagReady = MutableStateFlow(false)
    val isRagReady: StateFlow<Boolean> = _isRagReady.asStateFlow()

    private val _ragStatus = MutableStateFlow("Initializing document chat...")
    val ragStatus: StateFlow<String> = _ragStatus.asStateFlow()

    private val _documentCount = MutableStateFlow(0)
    val documentCount: StateFlow<Int> = _documentCount.asStateFlow()

    // Embedding enabled state
    private val _isEmbeddingEnabled = MutableStateFlow(false)
    val isEmbeddingEnabled: StateFlow<Boolean> = _isEmbeddingEnabled.asStateFlow()
    // Re-embedding in progress (when embedding model changes)
    private val _isReembedding = MutableStateFlow(false)
    val isReembedding: StateFlow<Boolean> = _isReembedding.asStateFlow()

    // TTS state - tracks which message is currently being read aloud
    private val _currentTtsMessageId = MutableStateFlow<String?>(null)
    val currentTtsMessageId: StateFlow<String?> = _currentTtsMessageId.asStateFlow()

    // Vision disabled state (for GPU with vision disabled on low RAM)
    private var isVisionDisabled: Boolean = false
    // Audio disabled state (for GPU with audio disabled on low RAM)
    private var isAudioDisabled: Boolean = false

    // NOTE: intent heuristics removed — global memory will be queried whenever the
    // memory preference is enabled. Localization-specific intent checks were removed
    // to avoid hardcoding English phrases.

    init {
        // Load saved settings first
        loadSavedSettings()
        
        // Initialize RAG service in the background and track status
        viewModelScope.launch {
            _ragStatus.value = "Initializing document chat..."
            try {
                ragServiceManager.initializeAsync().join()
                _isRagReady.value = ragServiceManager.isReady()
                _ragStatus.value = if (_isRagReady.value) {
                    "Document chat ready"
                } else {
                    "Document chat unavailable"
                }
            } catch (e: Exception) {
                Log.e("ChatViewModel", "Failed to initialize RAG service", e)
                _ragStatus.value = "Document chat failed to initialize"
                _isRagReady.value = false
            }
        }

        // Observe embedding enabled changes - need both enabled preference AND selected model
        var isFirstEmbeddingUpdate = true
        viewModelScope.launch {
            combine(
                themePreferences.embeddingEnabled,
                themePreferences.selectedEmbeddingModel
            ) { enabled, selectedModel ->
                enabled && !selectedModel.isNullOrBlank()
            }.collect { isEnabled ->
                val previousState = _isEmbeddingEnabled.value
                _isEmbeddingEnabled.value = isEnabled
                
                // Reinitialize RAG service when embedding settings change (but not on first initialization)
                if (!isFirstEmbeddingUpdate && previousState != isEnabled) {
                    Log.d("ChatViewModel", "Embedding settings changed from $previousState to $isEnabled, reinitializing RAG service")
                    try {
                        ragServiceManager.cleanup()
                        ragServiceManager.initializeAsync().join()
                        _isRagReady.value = ragServiceManager.isReady()
                        _ragStatus.value = if (_isRagReady.value) {
                            "Document chat ready"
                        } else {
                            "Document chat unavailable"
                        }
                        Log.d("ChatViewModel", "RAG service reinitialized successfully")

                        // After RAG has been reinitialized due to embedding toggle, rehydrate
                        // persisted global memory chunks so memory isn't lost when the user
                        // disabled both embeddings (RAG) and memory and then re-enabled them.
                        if (_isRagReady.value && isEnabled) {
                            try {
                                val memoryEnabledPref = themePreferences.memoryEnabled.first()
                                if (memoryEnabledPref) {
                                    val chunkList = withContext(kotlinx.coroutines.Dispatchers.IO) {
                                        com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context).memoryDao().getAllChunks()
                                    }
                                    if (chunkList.isNotEmpty()) {
                                        Log.d("ChatViewModel", "Restoring ${chunkList.size} global memory chunks after RAG reinit")
                                        ragServiceManager.restoreGlobalDocumentsFromChunks(chunkList)
                                    }
                                }
                            } catch (e: Exception) {
                                Log.w("ChatViewModel", "Failed to restore global memory chunks after RAG reinit: ${e.message}")
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("ChatViewModel", "Failed to reinitialize RAG service", e)
                        _ragStatus.value = "Document chat failed to initialize"
                        _isRagReady.value = false
                    }
                }
                isFirstEmbeddingUpdate = false
            }
        }

        // Observe embedding model changes specifically to trigger re-embedding of existing memory
        viewModelScope.launch {
            var prevModel: String? = try { themePreferences.selectedEmbeddingModel.first() } catch (_: Exception) { null }
            themePreferences.selectedEmbeddingModel.collect { currentModelName ->
                try {
                    val embeddingsOn = themePreferences.embeddingEnabled.first()
                    val memoryOn = themePreferences.memoryEnabled.first()
                    val modelChanged = prevModel != null && currentModelName != null && prevModel != currentModelName
                    prevModel = currentModelName

                    if (!embeddingsOn || !memoryOn) return@collect
                    if (!modelChanged) return@collect

                    // Check if there is any memory to re-embed
                    val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                    val hasAnyMemory = withContext(kotlinx.coroutines.Dispatchers.IO) {
                        db.memoryDao().getAllMemory().first().isNotEmpty()
                    }
                    if (!hasAnyMemory) return@collect

                    // Start re-embedding flow
                    _isReembedding.value = true
                    _ragStatus.value = context.getString(R.string.reembedding_memory_in_progress)

                    viewModelScope.launch(Dispatchers.IO) {
                        try {
                            // Clear in-memory global documents first to avoid mixing vector spaces
                            ragServiceManager.clearGlobalDocuments()

                            // Reset all chunk embeddings and mark docs pending
                            val docs = db.memoryDao().getAllMemory().first()
                            docs.forEach { doc ->
                                try {
                                    db.memoryDao().update(doc.copy(status = "PENDING", chunkCount = 0))
                                } catch (_: Exception) { }
                            }
                            try {
                                db.memoryDao().deleteAllChunks()
                            } catch (_: Exception) { }

                            // Kick off (re)processing with new embedding model
                            val processor = com.llmhub.llmhub.data.MemoryProcessor(context, db)
                            processor.processPending()

                            // Wait for processing to start (become true), then to finish (become false)
                            try {
                                withTimeoutOrNull(5_000) {
                                    com.llmhub.llmhub.data.MemoryProcessor.processing
                                        .first { it }
                                }
                            } catch (_: Exception) { }

                            com.llmhub.llmhub.data.MemoryProcessor.processing
                                .first { running -> !running }
                        } catch (_: CancellationException) {
                            // Used to break collection when processing completes
                        } catch (e: Exception) {
                            Log.w("ChatViewModel", "Re-embedding failed: ${e.message}")
                        } finally {
                            withContext(Dispatchers.Main) {
                                _isReembedding.value = false
                                _ragStatus.value = if (_isRagReady.value) {
                                    context.getString(R.string.reembedding_memory_done)
                                } else {
                                    "Document chat unavailable"
                                }
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Error reacting to embedding model change: ${e.message}")
                }
            }
        }
    }

    var currentModel: LLMModel? = null
        private set
    
    // Load saved settings from SharedPreferences
    private fun loadSavedSettings() {
        // Restore backend
        val savedBackendName = prefs.getString("selected_backend", LlmInference.Backend.GPU.name)
        _selectedBackend.value = try {
            LlmInference.Backend.valueOf(savedBackendName ?: LlmInference.Backend.GPU.name)
        } catch (_: IllegalArgumentException) {
            LlmInference.Backend.GPU
        }
        
        // Restore modality settings
        isVisionDisabled = prefs.getBoolean("vision_disabled", false)
        isAudioDisabled = prefs.getBoolean("audio_disabled", false)
        
        // Restore selected model by name (after available models are loaded)
        val savedModelName = prefs.getString("selected_model_name", null)
        if (savedModelName != null) {
            viewModelScope.launch {
                // Wait a bit for available models to be loaded
                delay(200)
                val model = _availableModels.value.find { it.name == savedModelName }
                if (model != null) {
                    _selectedModel.value = model
                }
            }
        }
    }
    
    // Save settings to SharedPreferences
    private fun saveChatSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            putString("selected_backend", _selectedBackend.value?.name)
            putBoolean("vision_disabled", isVisionDisabled)
            putBoolean("audio_disabled", isAudioDisabled)
            apply()
        }
    }
    
    // Public method to select model and persist
    fun selectModel(model: LLMModel) {
        _selectedModel.value = model
        // Auto-select backend if not set
        if (_selectedBackend.value == null) {
            _selectedBackend.value = if (model.supportsGpu) {
                LlmInference.Backend.GPU
            } else {
                LlmInference.Backend.CPU
            }
        }
        saveChatSettings()
    }
    
    // Public method to select backend and persist
    fun selectBackend(backend: LlmInference.Backend) {
        _selectedBackend.value = backend
        saveChatSettings()
    }

    private var isGenerating = false
        get() = savedStateHandle.get<Boolean>(KEY_IS_GENERATING) ?: false
        set(value) {
            field = value
            savedStateHandle.set(KEY_IS_GENERATING, value)
        }
    
    // Track whether we've already retried after a context reset to prevent infinite loops
    private var isRetryAfterContextReset = false

    // Track when session was reset due to repetitive content to ensure clean next generation
    private var lastSessionResetAt = 0L
    // Force drop of history on the next prompt after any session reset
    private var dropHistoryOnce = false
    // When returning to an existing chat, prime the next prompt with only the last
    // user/assistant pair. Consumed once inside buildContextAwareHistory.
    private var primeWithLastPairOnce = false
    
    private var currentChatId: String?
        get() = savedStateHandle.get<String>(KEY_CURRENT_CHAT_ID)
        set(value) = savedStateHandle.set(KEY_CURRENT_CHAT_ID, value)
    
    private var currentModelName: String?
        get() = savedStateHandle.get<String>(KEY_CURRENT_MODEL_NAME)
        set(value) = savedStateHandle.set(KEY_CURRENT_MODEL_NAME, value)
    
    // Keep reference to the running generation so the UI can interrupt it
    private var generationJob: Job? = null
    
    // Keep reference to message collection to prevent multiple collectors
    private var messageCollectionJob: Job? = null

    // Avoid aggressive continuation for the very first generation after model load
    private var firstGenerationSinceLoad: Boolean = false
    // Guard to avoid infinite retry loops when a response comes back empty
    private val oneShotRetriedMessageIds = mutableSetOf<String>()

    fun hasDownloadedModels(): Boolean {
        return _availableModels.value.isNotEmpty()
    }

    /**
     * Check if we need to restore generation state after configuration change
     */
    private fun checkAndRestoreGenerationState(context: Context) {
        if (isGenerating && generationJob?.isActive != true) {
            // Generation was interrupted by configuration change, try to restore
            Log.d("ChatViewModel", "Detected interrupted generation, checking for incomplete messages")
            
            viewModelScope.launch {
                val chatId = currentChatId ?: return@launch
                val messages = repository.getMessagesForChatSync(chatId)
                
                // Find the last message that might be incomplete
                val lastBotMessage = messages.lastOrNull { !it.isFromUser }
                if (lastBotMessage != null && lastBotMessage.content.trim() == "…") {
                    Log.d("ChatViewModel", "Found incomplete message, cleaning up generation state")
                    // Clean up the incomplete message placeholder
                    val updatedStreaming = _streamingContents.value.toMutableMap()
                    updatedStreaming.remove(lastBotMessage.id)
                    _streamingContents.value = updatedStreaming
                    
                    // Reset generation state
                    isGenerating = false
                    _isLoading.value = false
                    _isLoadingModel.value = false
                }
            }
        }
    }

    fun syncCurrentlyLoadedModel() {
        viewModelScope.launch {
            try {
                val loadedModel = inferenceService.getCurrentlyLoadedModel()
                _currentlyLoadedModel.value = loadedModel
                // Also update the local currentModel to keep them in sync
                currentModel = loadedModel
                
                // Sync the modality disabled states from the inference service
                isVisionDisabled = inferenceService.isVisionCurrentlyDisabled()
                isAudioDisabled = inferenceService.isAudioCurrentlyDisabled()
                
                Log.d("ChatViewModel", "Synced currently loaded model: ${loadedModel?.name ?: "None"}, vision disabled: $isVisionDisabled, audio disabled: $isAudioDisabled")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Error syncing currently loaded model: ${e.message}")
                // Don't update the state if there's an error
            }
        }
    }

    private val _currentCreator = MutableStateFlow<CreatorEntity?>(null)
    val currentCreator: StateFlow<CreatorEntity?> = _currentCreator.asStateFlow()

    // ... (existing code)

    fun initializeChat(chatId: String, context: Context, creatorId: String? = null) {
        // Sync the currently loaded model from inference service
        syncCurrentlyLoadedModel()
        
        // Stop collecting from any previous chat's message flow
        generationJob?.cancel()
        messageCollectionJob?.cancel()

        // Close previous chat session if switching chats
        if (currentChatId != null && currentChatId != chatId) {
             // ... (existing cleanup logic)
        }

        // Remember the previously loaded model to preserve it across chat switches
        val previousModel = inferenceService.getCurrentlyLoadedModel()
        Log.d("ChatViewModel", "Current model before switch: ${previousModel?.name ?: "None"}")

        viewModelScope.launch {
            // Load the models synchronously so we know what's available before creating/attaching a chat
            loadAvailableModelsSync(context)

            if (chatId == "new") {
                initializeNewChat(context, creatorId)
            } else {
                // Check if the chat still exists
                val chat = repository.getChatById(chatId)
                if (chat == null) {
                    Log.e("ChatViewModel", "Chat $chatId does not exist, creating new chat instead")
                    initializeNewChat(context, creatorId)
                    return@launch
                }
                
                // Load creator if associated with this chat
                if (chat.creatorId != null) {
                    _currentCreator.value = repository.getCreatorById(chat.creatorId)
                } else {
                    _currentCreator.value = null
                }
                
                // ... (rest of existing logic)

                
                // Check if chat contains images and current model supports vision
                val chatMessages = repository.getMessagesForChatSync(chatId)
                val chatHasImages = chatMessages.any { it.attachmentType == "image" }
                val currentlyLoadedModel = inferenceService.getCurrentlyLoadedModel()
                
                if (chatHasImages && currentlyLoadedModel != null && !currentlyLoadedModel.supportsVision) {
                    Log.w("ChatViewModel", "Cannot open chat with images when text-only model is loaded")
                    
                    // Show an error message to the user explaining why the chat can't be opened
                    _messages.value = listOf(
                        MessageEntity(
                            id = "error-${System.currentTimeMillis()}",
                            chatId = chatId,
                            content = context.getString(R.string.vision_chat_cannot_open, currentlyLoadedModel.name),
                            isFromUser = false,
                            timestamp = System.currentTimeMillis()
                        )
                    )
                    
                    // Set basic chat info without loading images
                    currentChatId = chatId
                    _currentChat.value = chat
                    currentModel = currentlyLoadedModel
                    
                    // Don't reset session or load messages - just show the error
                    return@launch
                }
                
                currentChatId = chatId
                _currentChat.value = chat
                // Ensure global memories are replicated into this chat so they act like attached docs
                try {
                    Log.d("ChatViewModel", "Replicating global memory into chat $chatId")
                    ragServiceManager.replicateGlobalChunksToChat(chatId)
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Failed to replicate global chunks into chat $chatId: ${e.message}")
                }
                // We are returning to an existing chat: on the next prompt construction,
                // include only the last prompt/response pair for quick context priming.
                primeWithLastPairOnce = true
                val foundModel = _availableModels.value.find { it.name == chat.modelName }
                    ?: ModelData.models.find { it.name == chat.modelName }
                
                if (foundModel != null && foundModel.isDownloaded) {
                    // Use the model associated with this chat
                    Log.d("ChatViewModel", "Chat requires model: ${foundModel.name}")
                    currentModel = foundModel
                    // Don't auto-load the model - let user manually load it if needed
                    // Users can select the model from the dropdown if they want to load it
                    Log.d("ChatViewModel", "Model not auto-loaded, user can select it manually")
                } else if (previousModel != null && previousModel.isDownloaded) {
                    // Fallback to the previously loaded model and assign it to this chat
                    Log.d("ChatViewModel", "Using previous model for chat: ${previousModel.name}")
                    currentModel = previousModel
                    repository.updateChatModel(chatId, previousModel.name)
                    _currentChat.value = repository.getChatById(chatId)
                    // Don't auto-load previous model either - keep current state
                    Log.d("ChatViewModel", "Previous model reference set but not loaded")
                } else {
                    // No valid model available
                    Log.d("ChatViewModel", "No valid model available for chat")
                    currentModel = null
                }
                
                // Always ensure we have a fresh session for the chat when switching
                if (currentChatId != null && currentChatId != chatId) {
                    // Reset session when switching to ensure clean state
                    resetChatSession(chatId)
                } else {
                    // For the same chat, still do a reset to ensure clean state
                    // This helps prevent session corruption issues when switching between chats
                    viewModelScope.launch {
                        try {
                            resetChatSession(chatId)
                            Log.d("ChatViewModel", "Reset session for current chat $chatId to ensure clean state")
                        } catch (e: Exception) {
                            Log.w("ChatViewModel", "Error resetting session for chat $chatId: ${e.message}")
                        }
                    }
                }
                
                messageCollectionJob = launch {
                    repository.getMessagesForChat(chatId).collectLatest { messageList ->
                        _messages.value = messageList
                    }
                }
            }
        }
    }

    /**
     * Load downloaded models synchronously so callers can rely on the result immediately.
     */
    private suspend fun loadAvailableModelsSync(context: Context) {
        val downloadedModels = ModelData.models
            .filter { it.category != "embedding" && !it.name.contains("Projector", ignoreCase = true) }
            .mapNotNull { model ->
            var isAvailable = false
            var actualSize = model.sizeBytes

            // Check if model is available in assets (priority)
            val assetPath = if (model.url.startsWith("file://models/")) {
                model.url.removePrefix("file://")
            } else {
                "models/${model.localFileName()}"
            }

            try {
                context.assets.open(assetPath).use { inputStream ->
                    actualSize = inputStream.available().toLong()
                    isAvailable = true
                    Log.d("ChatViewModel", "Found model in assets: $assetPath (${actualSize / (1024*1024)} MB)")
                }
            } catch (e: Exception) {
                // Model not in assets, check files directory
                val modelsDir = File(context.filesDir, "models")
                
                // Check for ONNX models with additional files first
                if (model.modelFormat == "onnx" && model.additionalFiles.isNotEmpty()) {
                    val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                    val onnxModelDir = File(modelsDir, modelDirName)
                    
                    if (onnxModelDir.exists() && onnxModelDir.isDirectory) {
                        val files = onnxModelDir.listFiles() ?: emptyArray()
                        val fileCount = files.filter { it.length() > 0 }.size
                        val expectedFileCount = 1 + model.additionalFiles.size
                        val totalSize = files.sumOf { it.length() }
                        
                        if (fileCount >= expectedFileCount) {
                            isAvailable = true
                            actualSize = totalSize
                            Log.d("ChatViewModel", "Found ONNX model in ${onnxModelDir.absolutePath} with $fileCount files (${totalSize / (1024*1024)} MB)")
                        } else {
                            Log.d("ChatViewModel", "ONNX model incomplete: only $fileCount/$expectedFileCount files in ${onnxModelDir.absolutePath}")
                        }
                    }
                } else {
                    // Regular single-file models
                    val primaryFile = File(modelsDir, model.localFileName())
                    val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")

                    // Migrate legacy file if needed
                    if (!primaryFile.exists() && legacyFile.exists()) {
                        legacyFile.renameTo(primaryFile)
                    }

                    if (primaryFile.exists()) {
                        // Only treat as available if file is fully downloaded (at least 99% of expected size)
                        val sizeKnown = model.sizeBytes > 0
                        val sizeOk = if (sizeKnown) {
                            primaryFile.length() >= (model.sizeBytes * 0.99).toLong()
                        } else {
                            primaryFile.length() >= 10L * 1024 * 1024 // Fallback for unknown size: at least 10MB
                        }
                        val valid = isModelFileValid(primaryFile, model.modelFormat)
                        if (sizeOk && valid) {
                            isAvailable = true
                            actualSize = primaryFile.length()
                            Log.d("ChatViewModel", "Found VALID model in files: ${primaryFile.absolutePath} (${actualSize / (1024*1024)} MB)")
                        } else {
                            Log.d("ChatViewModel", "Ignoring incomplete/invalid model file: ${primaryFile.absolutePath} sizeOk=$sizeOk valid=$valid size=${primaryFile.length()}/${model.sizeBytes}")
                        }
                        }
                    }
                }


            if (isAvailable) {
                model.copy(isDownloaded = true, sizeBytes = actualSize)
            } else {
                null
            }
        }

        // Get imported models from ModelDownloadViewModel
        val importedModels = try {
            // Get ModelDownloadViewModel instance to access imported models
            val modelDownloadViewModel = androidx.lifecycle.ViewModelProvider(
                context as androidx.activity.ComponentActivity,
                androidx.lifecycle.ViewModelProvider.AndroidViewModelFactory.getInstance(context.application)
            )[ModelDownloadViewModel::class.java]
            // Filter out image generation models (qnn_npu, mnn_cpu) - those are for Image Generator, not AI Chat
            modelDownloadViewModel.getImportedModels().filter { 
                it.category != "qnn_npu" && it.category != "mnn_cpu" 
            }
        } catch (e: Exception) {
            Log.w("ChatViewModel", "Could not get imported models: ${e.message}")
            emptyList()
        }
        
        // Combine downloaded and imported models
        val allAvailableModels = downloadedModels + importedModels
        
        _availableModels.value = allAvailableModels
        
        // If we have a current model but it's not in the available models, clear it
        if (currentModel != null && !allAvailableModels.any { it.name == currentModel?.name }) {
            Log.d("ChatViewModel", "Current model ${currentModel?.name} is no longer available")
            currentModel = null
        }
    }

    fun sendMessage(context: Context, text: String, attachmentUri: Uri?, audioData: ByteArray? = null) {
        val chatId = currentChatId
        if (chatId == null) {
            Log.e("ChatViewModel", "No current chat ID available, creating new chat")
            // If no current chat, create a new one
            viewModelScope.launch {
                initializeNewChat(context)
                // Retry sending the message after creating a new chat
                kotlinx.coroutines.delay(100) // Small delay to ensure chat is created
                sendMessage(context, text, attachmentUri, audioData)
            }
            return
        }
        val messageText = text.trim()

        if (messageText.isEmpty() && attachmentUri == null && audioData == null) return

        // Set loading state immediately to provide responsive UI feedback
        _isLoading.value = true
        isGenerating = true

    viewModelScope.launch {
            // Small delay to allow keyboard dismissal animation to complete
            // This prevents heavy processing from interfering with the keyboard animation
            kotlinx.coroutines.delay(150)
            
            // Check if the current chat still exists
            val currentChat = repository.getChatById(chatId)
            if (currentChat == null) {
                Log.e("ChatViewModel", "Current chat $chatId does not exist, creating new chat and retrying")
                // If current chat doesn't exist, create a new one and retry
                initializeNewChat(context)
                kotlinx.coroutines.delay(100) // Small delay to ensure chat is created
                sendMessage(context, text, attachmentUri)
                return@launch
            }

            // Verify we have a working model
            if (currentModel == null || !currentModel!!.isDownloaded) {
                Log.e("ChatViewModel", "No valid model available for chat $chatId. CurrentModel: ${currentModel?.name}, isDownloaded: ${currentModel?.isDownloaded}, availableModels: ${_availableModels.value.size}")
                val errorMessage = if (_availableModels.value.isEmpty()) {
                    context.getString(R.string.please_download_model)
                } else {
                    context.getString(R.string.model_not_loaded)
                }
                repository.addMessage(chatId, errorMessage, isFromUser = false)
                _isLoading.value = false
                isGenerating = false
                return@launch
            }

            // Process attachment if present
            var processedAttachmentUri: Uri? = null
            var attachmentFileInfo: FileUtils.FileInfo? = null
            var fileTextContent: String? = null
            
            // Process audio data if present
            if (audioData != null && audioData.isNotEmpty()) {
                try {
                    // Save audio data to a file
                    val audioFileName = "audio_${System.currentTimeMillis()}.wav"
                    val audioFile = File(context.getExternalFilesDir(Environment.DIRECTORY_MUSIC), audioFileName)
                    audioFile.parentFile?.mkdirs()
                    
                    audioFile.writeBytes(audioData)
                    
                    processedAttachmentUri = FileProvider.getUriForFile(
                        context,
                        "${context.packageName}.fileprovider",
                        audioFile
                    )
                    
                    attachmentFileInfo = FileUtils.FileInfo(
                        name = audioFileName,
                        size = audioData.size.toLong(),
                        mimeType = "audio/wav",
                        type = FileUtils.SupportedFileType.AUDIO,
                        uri = processedAttachmentUri
                    )
                    
                    Log.d("ChatViewModel", "Saved audio data to file: ${audioFile.absolutePath} (${audioData.size} bytes)")
                    
                } catch (e: Exception) {
                    Log.e("ChatViewModel", "Failed to save audio data: ${e.message}", e)
                }
            } else if (attachmentUri != null) {
                try {
                    // Get file information
                    attachmentFileInfo = FileUtils.getFileInfo(context, attachmentUri)
                    
                    if (attachmentFileInfo == null) {
                        Log.e("ChatViewModel", "Failed to get file info for URI: $attachmentUri")
                        val errorMessage = "📄 **File Processing Error**\n\nCould not process the selected file. This might be due to:\n\n❌ **File access issues**\n❌ **Corrupted file**\n❌ **System permissions**\n\n---\n\n💡 **Try:** Selecting a different file or restarting the app."
                        repository.addMessage(chatId, errorMessage, isFromUser = false)
                        _isLoading.value = false
                        isGenerating = false
                        return@launch
                    }
                    
                    // Check if file type is unknown/unsupported
                    if (attachmentFileInfo.type == FileUtils.SupportedFileType.UNKNOWN) {
                        Log.w("ChatViewModel", "Unsupported file type for: ${attachmentFileInfo.name}")
                        val errorMessage = "📄 **Unsupported File Type**\n\nThe file **${attachmentFileInfo.name}** is not supported. Please try one of these formats:\n\n🖼️ **Images:** JPG, PNG, GIF, WebP\n📝 **Text:** TXT, MD, CSV, JSON, XML\n📄 **Documents:** PDF, DOC, DOCX, XLS, XLSX, PPT, PPTX\n\n---\n\n💡 **Tip:** Make sure the file isn't corrupted and try again!"
                        repository.addMessage(chatId, errorMessage, isFromUser = false)
                        _isLoading.value = false
                        isGenerating = false
                        return@launch
                    }
                    
                    when (attachmentFileInfo.type) {
                        FileUtils.SupportedFileType.IMAGE -> {
                            // Check if current model supports vision before processing image
                            if (currentModel != null && !currentModel!!.supportsVision) {
                                Log.w("ChatViewModel", "Cannot send image with text-only model: ${currentModel!!.name}")
                                val errorMessage = context.getString(R.string.vision_image_cannot_send, currentModel!!.name)
                                repository.addMessage(chatId, errorMessage, isFromUser = false)
                                _isLoading.value = false
                                isGenerating = false
                                return@launch
                            }
                            
                            // Copy the image to internal storage to ensure it persists
                            processedAttachmentUri = copyImageToInternalStorage(context, attachmentUri)
                            Log.d("ChatViewModel", "Copied image to internal storage: $processedAttachmentUri")
                        }
                        
                        FileUtils.SupportedFileType.TEXT,
                        FileUtils.SupportedFileType.JSON,
                        FileUtils.SupportedFileType.XML -> {
                            // Extract text content for text-based files
                            fileTextContent = FileUtils.extractTextContent(context, attachmentUri, attachmentFileInfo.type)
                            if (fileTextContent != null) {
                                Log.d("ChatViewModel", "Extracted ${fileTextContent.length} characters from ${attachmentFileInfo.type.displayName}")
                                // Copy file to internal storage for persistence
                                processedAttachmentUri = FileUtils.copyFileToInternalStorage(
                                    context, 
                                    attachmentUri, 
                                    attachmentFileInfo.name
                                )
                            } else {
                                Log.w("ChatViewModel", "Failed to extract text content from file")
                                val errorMessage = "📄 **Could Not Read File**\n\nI wasn't able to read the content of **${attachmentFileInfo.name}**.\n\n---\n\n## 🎯 Possible Solutions:\n\n1. **Check file format** - Make sure it's a supported text file\n2. **Try a different file** - The file might be corrupted\n3. **Copy and paste** - You can copy the text content directly into the chat\n\n---\n\n💡 **Supported text formats:** TXT, MD, CSV, JSON, XML"
                                repository.addMessage(chatId, errorMessage, isFromUser = false)
                                _isLoading.value = false
                                isGenerating = false
                                return@launch
                            }
                        }
                        
                        else -> {
                            // For other file types (PDF, Word, etc.)
                            Log.d("ChatViewModel", "Processing ${attachmentFileInfo.type.displayName}: ${attachmentFileInfo.name}")
                            processedAttachmentUri = FileUtils.copyFileToInternalStorage(
                                context, 
                                attachmentUri, 
                                attachmentFileInfo.name
                            )
                            
                            // Extract content from all supported document types
                            fileTextContent = when (attachmentFileInfo.type) {
                                FileUtils.SupportedFileType.TEXT,
                                FileUtils.SupportedFileType.JSON,
                                FileUtils.SupportedFileType.XML,
                                FileUtils.SupportedFileType.PDF,
                                FileUtils.SupportedFileType.WORD,
                                FileUtils.SupportedFileType.EXCEL,
                                FileUtils.SupportedFileType.POWERPOINT -> {
                                    FileUtils.extractTextContent(context, attachmentUri, attachmentFileInfo.type)
                                }
                                else -> null // For unsupported types (like images), don't extract text
                            }
                        }
                    }
                    
                } catch (e: Exception) {
                    Log.e("ChatViewModel", "Failed to process attachment", e)
                    val errorMessage = "📄 **File Processing Error**\n\nThere was an error processing your file:\n\n**Error:** ${e.message ?: "Unknown error"}\n\n---\n\n## 🎯 Try This:\n\n1. **Check the file** - Make sure it's not corrupted\n2. **Try a different format** - Convert to a supported format\n3. **Restart the app** - Sometimes this resolves temporary issues\n\n---\n\n💡 **Need help?** Try copying and pasting the content as text instead!"
                    repository.addMessage(chatId, errorMessage, isFromUser = false)
                    _isLoading.value = false
                    isGenerating = false
                    return@launch
                }
            }

            // Quick safety screening before enqueueing user message (basic client-side filter)
            if (isDisallowedPrompt(messageText)) {
                repository.addMessage(
                    chatId = chatId,
                    content = "I can't assist with that request. Please rephrase with a different, safe topic.",
                    isFromUser = false
                )
                _isLoading.value = false
                isGenerating = false
                return@launch
            }

            // Prepare the final message content (what the user sees)
            var finalMessageContent = messageText
            var modelPromptContent = messageText // What gets sent to the model
            
            // For documents with extracted content, add content to model prompt but keep user message clean
            if (fileTextContent != null && attachmentFileInfo?.type in listOf(
                FileUtils.SupportedFileType.TEXT,
                FileUtils.SupportedFileType.JSON,
                FileUtils.SupportedFileType.XML,
                FileUtils.SupportedFileType.PDF,
                FileUtils.SupportedFileType.WORD,
                FileUtils.SupportedFileType.EXCEL,
                FileUtils.SupportedFileType.POWERPOINT
            )) {
                // Add document to RAG system for future semantic search
                try {
                    val fileName = attachmentFileInfo?.name ?: "document"
                    // Store a non-localized metadata token so the database doesn't contain language-specific text
                    val metadata = "uploaded"
                    val success = ragServiceManager.addDocument(chatId, fileTextContent, fileName, metadata)
                    if (success) {
                        // Update document count
                        val count = ragServiceManager.getDocumentCount(chatId)
                        _documentCount.value = count
                        _ragStatus.value = context.getString(R.string.documents_available_format, count)
                        Log.d("ChatViewModel", "✅ Added document '$fileName' to RAG system for chat $chatId")
                        
                        // For chat-specific documents, always use semantic search to find relevant content
                        // Don't inject the entire document - let RAG find the relevant parts
                        modelPromptContent = messageText
                    } else {
                        Log.w("ChatViewModel", "❌ Failed to add document '$fileName' to RAG system (embeddings disabled or service unavailable)")
                        // Show localized failure
                        _ragStatus.value = context.getString(R.string.memory_upload_failed)
                        
                        // When RAG is disabled, inject the full document content directly
                        Log.d("ChatViewModel", "📄 RAG disabled - injecting full document content directly")
                        modelPromptContent = if (messageText.isNotEmpty()) {
                            "$messageText\n\n---\n\n$fileTextContent"
                        } else {
                            fileTextContent
                        }
                    }
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Failed to add document to RAG: ${e.message}")
                    // When RAG fails, inject the full document content directly
                    Log.d("ChatViewModel", "📄 RAG failed - injecting full document content directly")
                    modelPromptContent = if (messageText.isNotEmpty()) {
                        "$messageText\n\n---\n\n$fileTextContent"
                    } else {
                        fileTextContent
                    }
                }
                
                // Keep user message clean - just show that a file was attached
                if (messageText.isEmpty()) {
                    finalMessageContent = "📄 ${attachmentFileInfo?.name}"
                }
            } else if (fileTextContent != null) {
                // For other file types, add a note about the attachment
                modelPromptContent = if (messageText.isNotEmpty()) {
                    "$messageText\n\n---\n\n$fileTextContent"
                } else {
                    fileTextContent
                }
                
                if (messageText.isEmpty()) {
                    finalMessageContent = "📄 ${attachmentFileInfo?.name}"
                }
            } else if (messageText.isEmpty() && processedAttachmentUri != null) {
                // For audio files, don't show filename or any text - just let the audio player handle it
                if (attachmentFileInfo?.type == FileUtils.SupportedFileType.AUDIO) {
                    finalMessageContent = "" // Empty content, audio player will be shown via attachment
                    modelPromptContent = "" // Audio token will be added later in prompt building
                } else {
                    finalMessageContent = "📄 ${attachmentFileInfo?.name}"
                    modelPromptContent = context.getString(R.string.shared_file)
                }
            }

            repository.addMessage(
                chatId = chatId,
                content = finalMessageContent,
                isFromUser = true,
                attachmentPath = processedAttachmentUri?.toString(),
                attachmentType = attachmentFileInfo?.type?.name,
                attachmentFileName = attachmentFileInfo?.name,
                attachmentFileSize = attachmentFileInfo?.size
            )
            val userMessageId = _messages.value.lastOrNull { it.isFromUser && it.chatId == chatId }?.id
            
            // Debug logging for file size
            if (attachmentFileInfo != null) {
                Log.d("ChatViewModel", "Adding message with attachment:")
                Log.d("ChatViewModel", "  File name: ${attachmentFileInfo.name}")
                Log.d("ChatViewModel", "  File size: ${attachmentFileInfo.size} bytes")
                Log.d("ChatViewModel", "  File size formatted: ${FileUtils.formatFileSize(attachmentFileInfo.size)}")
            }

            // The first message sets the title
            if (_messages.value.size == 1) {
                val chatTitle = when {
                    messageText.isNotEmpty() -> messageText.take(50)
                    attachmentFileInfo?.type == FileUtils.SupportedFileType.AUDIO -> context.getString(R.string.audio_message)
                    attachmentFileInfo?.type == FileUtils.SupportedFileType.IMAGE -> context.getString(R.string.image_message)
                    attachmentFileInfo != null -> "📄 ${attachmentFileInfo.name.take(30)}"
                    else -> context.getString(R.string.drawer_new_chat)
                }
                repository.updateChatTitle(chatId, chatTitle)
                _currentChat.value = repository.getChatById(chatId)
            }

            if (currentModel != null && currentModel!!.isDownloaded) {
                // Ensure the model is loaded in the inference service before generating
                try {
                    _isLoadingModel.value = true
                    // Perform model load on IO dispatcher to avoid UI blocking / ANR
                    val loaded = withContext(Dispatchers.IO) {
                        inferenceService.loadModel(currentModel!!, _selectedBackend.value)
                    }
                    // Sync the currently loaded model state
                    syncCurrentlyLoadedModel()
                    _isLoadingModel.value = false
                    Log.d("ChatViewModel", "Successfully ensured model ${currentModel!!.name} is loaded for generation (loaded=$loaded)")
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Failed to ensure model is loaded: ${e.message}")
                    repository.addMessage(chatId, "Failed to load model. Please try again.", isFromUser = false)
                    _isLoading.value = false
                    _isLoadingModel.value = false
                    isGenerating = false
                    return@launch
                }
                
                // Session management is now handled internally by the inference service
                // No need to manually ensure or create sessions
                Log.d("ChatViewModel", "Starting generation for chat $chatId")
                
                // Check if we should reset the session before generating to prevent token overflow
                val shouldResetSession = shouldResetSessionBeforeMessage(messageText, chatId)
                if (shouldResetSession) {
                    Log.d("ChatViewModel", "Proactively resetting session for chat $chatId before generation")
                    try {
                        delay(100) // Small delay to ensure model is ready
                        inferenceService.resetChatSession(chatId)
                        Log.d("ChatViewModel", "Successfully reset session for chat $chatId")
                    } catch (e: Exception) {
                        Log.w("ChatViewModel", "Error resetting session for chat $chatId: ${e.message}")
                    }
                }
                
                // Pass the conversation history to the model with context window management
                // Build history including the current user message
                val currentUserMessage = MessageEntity(
                    id = "current-${System.currentTimeMillis()}",
                    chatId = chatId,
                    content = modelPromptContent, // Use the full content including document text for the model
                    isFromUser = true,
                    timestamp = System.currentTimeMillis(),
                    attachmentPath = processedAttachmentUri?.toString(),
                    attachmentType = determineAttachmentType(processedAttachmentUri)
                )
                val allMessages = _messages.value.toMutableList().apply { add(currentUserMessage) }
                val history = buildContextAwareHistory(allMessages)

                // Insert a visible placeholder so the bubble stays rendered while tokens stream
                val placeholderId = repository.addMessage(chatId, "…", isFromUser = false)
                _streamingContents.value = mapOf(placeholderId to "") // Initialize with empty string                        // Run generation with streaming tokens
                        generationJob = launch {
                            val generationStartTime = System.currentTimeMillis()
                            // Track time spent performing RAG/document searches so we can exclude it
                            // from the model generation timing used for tok/sec reporting.
                            var ragSearchTimeMs = 0L
                            var totalContent = ""
                            var firstChunkAt = 0L
                            var lastChunkAt = 0L
                            var handedOffToAutoRegenerate = false
                            var triggeredRetry = false // Track if this attempt triggered a retry
                            
                            try {
                                // Pre-load model separately from generation timing
                                _isLoadingModel.value = true
                                
                                // Ensure model is loaded first
                                inferenceService.loadModel(currentModel!!, _selectedBackend.value)
                                
                                _isLoadingModel.value = false
                                
                                // Continue generation with context window management
                                var continuationCount = 0
                                val maxContinuations = 2 // Further reduced since we have better context management
                                
                                while (continuationCount < maxContinuations) {
                                    var currentSegment = ""
                                    val segmentStartTime = System.currentTimeMillis()
                                    
                                    // Build the prompt for this segment with context window management
                                    val currentPrompt = if (continuationCount == 0) {
                                        // First generation of this reply
                                        val baseUserContent = currentUserMessage.content.trim()
                                        // Add audio token if audio data is present (MediaPipe requirement)
                        Log.d("ChatViewModel", "Audio token check: audioData=${audioData?.size ?: 0} bytes, supportsAudio=${currentModel!!.supportsAudio}, isAudioDisabled=$isAudioDisabled, baseUserContent='$baseUserContent'")
                        val lastUserContent = baseUserContent // Do not inject <audio_soft_token>; follow Google AI Edge Gallery pattern
                        Log.d("ChatViewModel", "Final lastUserContent (no audio soft token): '$lastUserContent'")
                                        
                                        // Search for relevant document context using RAG (per-chat and optional global memory)
                                        var ragContext = ""
                                        // Precompute the query embedding once per prompt to avoid duplicate embed calls
                                        var precomputedQueryEmbedding: FloatArray? = null
                                        try {
                                            precomputedQueryEmbedding = ragServiceManager.generateEmbedding(lastUserContent)
                                            if (precomputedQueryEmbedding != null) {
                                                Log.d("ChatViewModel", "Precomputed query embedding (${precomputedQueryEmbedding.size} dims)")
                                            }
                                        } catch (e: Exception) {
                                            Log.w("ChatViewModel", "Failed to precompute query embedding: ${e.message}")
                                            precomputedQueryEmbedding = null
                                        }
                                        try {
                                            val contextParts = mutableListOf<String>()

                                            val memoryEnabledPref = themePreferences.memoryEnabled.first()
                                            val recentManualReset = System.currentTimeMillis() - lastSessionResetAt < 10000 // 10s
                                            val recentAutoReset = inferenceService.wasSessionRecentlyReset(chatId)
                                            val recentSessionReset = recentManualReset || recentAutoReset

                                            // If we perform a forced injection of global memories (after a recent reset)
                                            // we should NOT also include per-chat results later, otherwise the same
                                            // global chunks can be added twice (direct global fetch + merged chat+global).
                                            var forcedInjectedGlobal = false

                                            if (memoryEnabledPref && !_isReembedding.value) {
                                                if (recentSessionReset) {
                                                    Log.d("ChatViewModel", "🔔 Recent session reset detected - forcibly injecting global memories into prompt")
                                                    // Try semantic search first
                                                    var globalChunks = ragServiceManager.searchGlobalContext(lastUserContent, 10, relaxedLexicalFallback = true, queryEmbedding = precomputedQueryEmbedding)
                                                    // Fallback to persisted DB chunks if embedder returned nothing
                                                    if (globalChunks.isEmpty()) {
                                                        try {
                                                            val persisted = withContext(Dispatchers.IO) {
                                                                com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context).memoryDao().getAllChunks()
                                                            }
                                                            if (persisted.isNotEmpty()) {
                                                                globalChunks = persisted.map { pc ->
                                                                    ContextChunk(pc.content, pc.docId ?: "", pc.fileName, 1.0f, pc.chunkIndex)
                                                                }
                                                            }
                                                        } catch (e: Exception) {
                                                            Log.w("ChatViewModel", "Failed to read persisted memory chunks for forced injection: ${e.message}")
                                                        }
                                                    }

                                                    if (globalChunks.isNotEmpty()) {
                                                        // Present memory chunks as direct factual context (no filename prompt)
                                                        contextParts.addAll(globalChunks.take(3).map { chunk ->
                                                            chunk.content.trim()
                                                        })
                                                        forcedInjectedGlobal = true
                                                    } else {
                                                        Log.d("ChatViewModel", "🔍 Forced injection: no global chunks available to inject")
                                                    }
                                                } else {
                                                    // Normal memory-enabled path: do not separately call searchGlobalContext here.
                                                    // We'll rely on the later call to `searchRelevantContext(chatId,...)`
                                                    // which already merges chat-specific and global results when memory is enabled.
                                                    Log.d("ChatViewModel", "🔍 Memory enabled - deferring global inclusion to combined chat search to avoid duplication")
                                                }
                                            } else if (!memoryEnabledPref) {
                                                Log.d("ChatViewModel", "ℹ️ Global memory disabled by user preference; not querying global memory")
                                            } else if (_isReembedding.value) {
                                                Log.d("ChatViewModel", "ℹ️ Skipping global memory during re-embedding")
                                            }

                                            // Always search per-chat documents for relevant content
                                            // This works even when memory is disabled, using semantic search
                                            if (ragServiceManager.hasDocuments(chatId)) {
                                                Log.d("ChatViewModel", "🔍 Searching chat-specific documents for relevant context")
                                                val ragStartChat = System.currentTimeMillis()
                                                val relevantChunks = ragServiceManager.searchRelevantContext(
                                                    chatId = chatId,
                                                    query = lastUserContent,
                                                    maxResults = 3,
                                                    queryEmbedding = precomputedQueryEmbedding
                                                )
                                                ragSearchTimeMs += (System.currentTimeMillis() - ragStartChat)

                                                if (relevantChunks.isNotEmpty()) {
                                                    // For chat-specific documents, always use semantic search to find relevant content
                                                    // This works regardless of memory settings - it's about the attached documents
                                                    val topSimilarity = relevantChunks.map { chunk -> chunk.similarity }.maxOrNull() ?: 0.0f
                                                    
                                                    // Calculate lexical overlap for additional validation
                                                    fun wordJaccardLocal(a: String, b: String): Double {
                                                        val wa = a.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }.toSet()
                                                        val wb = b.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }.toSet()
                                                        if (wa.isEmpty() || wb.isEmpty()) return 0.0
                                                        val inter = wa.intersect(wb).size.toDouble()
                                                        val union = wa.union(wb).size.toDouble()
                                                        return if (union == 0.0) 0.0 else inter / union
                                                    }
                                                    
                                                    val topOverlap = relevantChunks.map { chunk -> wordJaccardLocal(lastUserContent, chunk.content) }.maxOrNull() ?: 0.0
                                                    
                                                    // Use model-specific thresholds for injection decision
                                                    // EmbeddingGemma produces better semantic embeddings and needs lower thresholds
                                                    // These thresholds MUST match RagService.kt for consistency
                                                    val modelName = ragServiceManager.getCurrentEmbeddingModelName() ?: "Gecko"
                                                    val isEmbeddingGemma = modelName.contains("EmbeddingGemma", ignoreCase = true)
                                                    
                                                    val shouldInject = if (isEmbeddingGemma) {
                                                        // EmbeddingGemma: trust semantic similarity more, need less lexical overlap
                                                        // Match RagService primary=0.65, fallback=0.30, lexical=0.02
                                                        (topSimilarity > 0.65f ) ||  // High semantic alone
                                                        (topSimilarity > 0.50f && topOverlap > 0.035)  // Moderate semantic + minimal lexical
                                                    } else {
                                                        // Gecko: need higher thresholds and more lexical validation
                                                        // Keep conservative thresholds: primary=0.80, lexical=0.05
                                                        (topSimilarity > 0.80f && topOverlap > 0.05) ||  // High semantic + some lexical overlap
                                                        (topSimilarity > 0.95f && topOverlap > 0.005)  // Very high semantic + minimal lexical overlap
                                                    }

                                                    if (shouldInject) {
                                                        Log.d("ChatViewModel", "✅ Found ${relevantChunks.size} relevant document chunks for query (similarity=${"%.3f".format(topSimilarity)}, overlap=${"%.3f".format(topOverlap)}, model=$modelName) - injecting into prompt")
                                                        contextParts.addAll(relevantChunks.map { chunk ->
                                                            "📄 **${chunk.fileName}**:\n${chunk.content}"
                                                        })
                                                    } else {
                                                        Log.d("ChatViewModel", "ℹ️ Skipping document injection: similarity=${"%.3f".format(topSimilarity)}, overlap=${"%.3f".format(topOverlap)}, model=$modelName (below thresholds)")
                                                    }
                                                }
                                            } else {
                                                Log.d("ChatViewModel", "ℹ️ No documents available in chat $chatId for context search")
                                            }

                                            if (contextParts.isNotEmpty()) {
                                                // Strong instruction: treat the following lines as confirmed user facts.
                                                // Do NOT ask the user to paste file contents or reference file names.
                                                val memoryInstruction = "IMPORTANT: The following lines are USER MEMORY facts or relevant content in attatched documents. Incorporate them directly into your answer if relevant. Do NOT ask for file contents or reference filenames. If a fact is irrelevant, ignore it."
                                                    ragContext = "\n\n---\n\nUSER MEMORY FACTS:\n\n" +
                                                        memoryInstruction + "\n\n" +
                                                        contextParts.joinToString("\n\n") +
                                                        "\n\n---\n\n"
                                            } else {
                                                Log.d("ChatViewModel", "❌ No relevant document chunks found for query (no matches or embeddings disabled)")
                                            }
                                        } catch (e: Exception) {
                                            Log.w("ChatViewModel", "❌ RAG context search failed: ${e.message}")
                                        }
                                        
                                        val tinyArithmetic = lastUserContent.matches(Regex("^[0-9+*/().=\\s-]{1,12}$")) && lastUserContent.any { it.isDigit() }
                                        val veryShort = lastUserContent.length <= 8
                                        val historyIsLarge = history.length > 3500 // heuristic char threshold
                                        // Detect explicit user intent to shift topics
                                        val explicitShift = lastUserContent.lowercase().startsWith("new topic") ||
                                                lastUserContent.lowercase().startsWith("fresh start") ||
                                                lastUserContent.lowercase().startsWith("unrelated:")
                                        // Heuristic semantic/topic shift detection (low lexical overlap with recent user turns)
                                        val topicShift = historyIsLarge && isTopicShift(lastUserContent, _messages.value)
                                        // Check if session was recently reset (either manually or automatically)
                                        val recentManualReset = System.currentTimeMillis() - lastSessionResetAt < 10000 // 10 seconds
                                        val recentAutoReset = inferenceService.wasSessionRecentlyReset(chatId)
                                        val recentSessionReset = recentManualReset || recentAutoReset
                                        val forceMinimal = (tinyArithmetic || veryShort || explicitShift || topicShift || recentSessionReset) && historyIsLarge
                                        if (topicShift) {
                                            Log.d("ChatViewModel", "Topic shift detected for prompt '${lastUserContent.take(60)}' (history=${history.length} chars) -> minimal context path")
                                        }
                                        if (recentSessionReset) {
                                            Log.d("ChatViewModel", "Recent session reset detected (manual=$recentManualReset, auto=$recentAutoReset) -> forcing minimal context path")
                                        }
                                        if (forceMinimal) {
                                            // Proactively reset session to ensure totally clean small-context answer
                                            try {
                                                Log.d("ChatViewModel", "Force minimal prompt path for tiny query '${lastUserContent}' (history ${history.length} chars) - resetting session & dropping history")
                                                inferenceService.resetChatSession(chatId)
                                                lastSessionResetAt = System.currentTimeMillis()
                                            } catch (e: Exception) {
                                                Log.w("ChatViewModel", "Failed to reset session for minimal prompt: ${e.message}")
                                            }
                                            // Minimal path after reset: keep a small sanitized tail of history (last 2 pairs)
                                            val tail = _messages.value.takeLast(4)
                                                .map { it.copy(content = sanitizeForPrompt(it.content)) }
                                                .filter { it.content.isNotBlank() && it.content.length >= 20 && it.content != "…" }
                                            val tailHistory = if (tail.isNotEmpty()) {
                                                tail.joinToString("\n\n") { msg ->
                                                    if (msg.isFromUser) "user: ${msg.content}" else "assistant: ${msg.content}"
                                                }
                                            } else ""
                                            
                                            // For audio messages, include instruction in the user message
                                            val userMessage = if (audioData != null) {
                                                if (lastUserContent.trim().isEmpty()) {
                                                    // Audio-only: provide clear instruction to transcribe and respond
                                                    "Listen to the audio and respond to it's content like normal text prompt" // More explicit instruction
                                                } else {
                                                    lastUserContent // Text+audio: use user's text as-is
                                                }
                                            } else {
                                                sanitizeForPrompt(lastUserContent)
                                            }
                                            
                                            var basePrompt = if (tailHistory.isNotEmpty()) {
                                                "$tailHistory\n\nuser: $userMessage\nassistant:"
                                            } else {
                                                "user: $userMessage\nassistant:"
                                            }
                                            if (ragContext.isNotEmpty()) {
                                                basePrompt = basePrompt.replace("\nassistant:", "${ragContext}\nassistant:")
                                            }
                                            basePrompt
                                        } else {
                                            // Normal path: include trimmed history and explicit assistant cue
                                            // For audio messages, we add the instruction as part of the user message
                                            var basePrompt = if (audioData != null && lastUserContent.trim().isEmpty()) {
                                                // Audio-only: replace the last user message with clear audio instruction
                                                val historyLines = history.split("\n")
                                                val modifiedHistory = historyLines.dropLast(1).joinToString("\n") + 
                                                    "\nuser: Listen to the audio and respond to what was said.\nassistant:"
                                                modifiedHistory
                                            } else {
                                                // Text-only or text+audio: use history as-is
                                                if (!history.endsWith("assistant:")) {
                                                    history + "\nassistant:"
                                                } else history
                                            }

                                            // Insert RAG context before the assistant response (assign the result)
                                            if (ragContext.isNotEmpty()) {
                                                basePrompt = basePrompt.replace("\nassistant:", "${ragContext}\nassistant:")
                                            }
                                            
                                            // Inject Creator Persona if active
                                            val creator = _currentCreator.value
                                            if (creator != null && creator.pctfPrompt.isNotBlank()) {
                                                Log.d("ChatViewModel", "Injecting Creator Persona: ${creator.name}")
                                                // Prepend the system prompt. We use a clear separation.
                                                // Some models handle "system:" tag, others just take the first segment.
                                                // We'll use a "System:" prefix to be consistent with the chat format.
                                                basePrompt = "system: ${creator.pctfPrompt}\n\n$basePrompt"
                                            }
                                            
                                            basePrompt
                                        }
                                    } else {
                                        // Continuation: build a new context-aware history including the current response
                                        val allMessages = _messages.value.toMutableList()
                                        // Add the current partial response as a temporary message for context
                                        allMessages.add(MessageEntity(
                                            id = "temp-${System.currentTimeMillis()}", // Temporary ID as String
                                            chatId = currentChatId ?: "",
                                            content = totalContent.trimEnd(),
                                            isFromUser = false,
                                            timestamp = System.currentTimeMillis()
                                        ))
                                        val continuationHistory = buildContextAwareHistory(allMessages)
                                        if (!continuationHistory.endsWith("assistant:")) continuationHistory + "\nassistant:" else continuationHistory
                                    }
                                    
                                    // Extract images and documents for multimodal models
                    // Note: We include both images and document attachments for vision models
                    val images = if (currentModel!!.supportsVision) {
                        Log.d("ChatViewModel", "Current model supports vision: ${currentModel!!.name}")
                        val recentMessages = _messages.value.takeLast(10)
                        Log.d("ChatViewModel", "Checking ${recentMessages.size} recent messages for images")
                        
                        // Get images from the current user message first, then from recent messages
                        val currentImages = mutableListOf<Bitmap>()
                        
                        // Check if the current message has an image attachment
                        if (processedAttachmentUri != null && attachmentFileInfo?.type == FileUtils.SupportedFileType.IMAGE) {
                            try {
                                Log.d("ChatViewModel", "Loading current message image from URI: $processedAttachmentUri")
                                val bitmap = loadImageFromUri(context, processedAttachmentUri)
                                if (bitmap != null) {
                                    currentImages.add(bitmap)
                                    Log.d("ChatViewModel", "Added current message image: ${bitmap.width}x${bitmap.height}")
                                } else {
                                    Log.w("ChatViewModel", "Failed to load bitmap from current message URI")
                                }
                            } catch (e: Exception) {
                                Log.e("ChatViewModel", "Failed to load current message image", e)
                            }
                        } else if (processedAttachmentUri != null && attachmentFileInfo?.type != FileUtils.SupportedFileType.IMAGE) {
                            Log.d("ChatViewModel", "Current message has non-image attachment (${attachmentFileInfo?.type?.displayName}): ${attachmentFileInfo?.name}")
                            // For vision models with document attachments, we note this in the context
                        } else {
                            Log.d("ChatViewModel", "No current message image attachment")
                        }
                        
                        // Add images from recent messages (for context)
                        // For now, we'll only include the current message's image to avoid confusion
                        // In the future, you might want to include context images for multi-turn vision conversations
                        // val contextImages = extractImagesFromAttachments(context, recentMessages)
                        // currentImages.addAll(contextImages)
                        
                        // Note: Context images are disabled to prevent accumulation across messages
                        // Each message should focus on its own image content
                        val contextImages = emptyList<Bitmap>() // Disabled for now
                        
                        Log.d("ChatViewModel", "Total images for generation: ${currentImages.size} (current: ${if (processedAttachmentUri != null && attachmentFileInfo?.type == FileUtils.SupportedFileType.IMAGE) 1 else 0}, context: ${contextImages.size} - disabled)")
                        if (processedAttachmentUri != null && attachmentFileInfo?.type != FileUtils.SupportedFileType.IMAGE) {
                            Log.d("ChatViewModel", "Vision model also processing document: ${attachmentFileInfo?.type?.displayName} - ${attachmentFileInfo?.name}")
                        }
                        
                        // Log details of each image
                        currentImages.forEachIndexed { index, bitmap ->
                            Log.d("ChatViewModel", "Image $index: ${bitmap.width}x${bitmap.height}, config: ${bitmap.config}")
                        }
                        
                        currentImages
                    } else {
                        Log.d("ChatViewModel", "Current model does not support vision: ${currentModel!!.name}")
                        emptyList()
                    }
                    
                    // Get web search preference
                    val webSearchEnabled = runBlocking { themePreferences.webSearchEnabled.first() }
                    
                    // Log a preview of the final prompt passed to the model (trimmed) to
                    // help debug why memory facts may not be used. This logs only the
                    // first 1000 characters to avoid huge logs.
                    try {
                        val promptPreview = if (currentPrompt.length > 1000) currentPrompt.substring(0, 1000) + "..." else currentPrompt
                        Log.d("ChatViewModel", "Final prompt preview:\n$promptPreview")
                    } catch (e: Exception) {
                        Log.w("ChatViewModel", "Failed to log prompt preview: ${e.message}")
                    }

                    // Safety guard: ensure the latest user message is present in the final prompt
                    val finalPrompt = try {
                        val latestUser = _messages.value.lastOrNull { it.isFromUser }?.content ?: ""
                        var p = currentPrompt
                        if (latestUser.isNotBlank()) {
                            val mustContain = "user: ${sanitizeForPrompt(latestUser)}"
                            if (!p.contains(mustContain)) {
                                Log.w("ChatViewModel", "Latest user line missing from prompt. Appending minimal user line before generation.")
                                p = p.trimEnd() + "\n\n" + mustContain + "\nassistant:"
                            }
                        }
                        p
                    } catch (_: Exception) { currentPrompt }

                    val responseStream = inferenceService.generateResponseStreamWithSession(
                        finalPrompt,
                        currentModel!!, 
                        chatId, 
                        images, 
                        audioData, // Pass audio data for Gemma-3n models
                        webSearchEnabled
                    )
                    // Track precise generation window based on first and last streamed chunks
                                    var lastUpdateTime = 0L
                                    val updateIntervalMs = 50L // Update UI every 50ms instead of every token
                                    var segmentEnded = false
                                    var segmentHasContent = false
                                    
                                    // Check if auto-readout is enabled before starting generation
                                    val autoReadoutEnabled = try {
                                        themePreferences.autoReadoutEnabled.first()
                                    } catch (e: Exception) {
                                        false
                                    }
                                    
                                    // Initialize TTS for streaming if auto-readout is enabled or if the
                                    // ViewModel's current TTS message ID was set (manual read trigger).
                                    if (autoReadoutEnabled || _currentTtsMessageId.value == placeholderId) {
                                        Log.d("ChatViewModel", "Preparing TTS for streaming message: $placeholderId (autoReadout=$autoReadoutEnabled, currentTts=${_currentTtsMessageId.value})")
                                        // Stop any previous TTS before starting new generation
                                        ttsService.stop()
                                        // Set language to app's current locale
                                        val appLocale = com.llmhub.llmhub.utils.LocaleHelper.getCurrentLocale(context)
                                        ttsService.setLanguage(appLocale)
                                        // Ensure the current TTS message ID is set so streaming picks it up
                                        _currentTtsMessageId.value = placeholderId
                                    }
                                    
                                    responseStream.collect { piece ->
                        val nowTs = System.currentTimeMillis()
                        if (piece.isNotEmpty() && firstChunkAt == 0L) firstChunkAt = nowTs
                        lastChunkAt = nowTs
                                        currentSegment += piece
                                        totalContent += piece
                                        
                                        // Check if we're getting meaningful content
                                        if (piece.trim().isNotEmpty()) {
                                            segmentHasContent = true
                                        }
                                        
                                        // Stream text to TTS as it arrives when this message is the
                                        // active TTS target (either auto-readout or manual trigger).
                                        if (_currentTtsMessageId.value == placeholderId && piece.isNotEmpty()) {
                                            try {
                                                ttsService.addStreamingText(piece)
                                            } catch (e: Exception) {
                                                Log.w("ChatViewModel", "Failed to add text to TTS stream: ${e.message}")
                                            }
                                        }
                                        
                                        // Note: Repetition detection is handled at the InferenceService layer
                                        // Note: No artificial length limit - let the model's natural token limit handle it
                                        // The model will stop when it reaches its context window or generates EOS token
                                        
                                        // Update UI with the complete content so far
                                        val updated = _streamingContents.value.toMutableMap()
                                        updated[placeholderId] = totalContent
                                        _streamingContents.value = updated
                                        // Debounced database updates to reduce blinking
                                        val currentTime = System.currentTimeMillis()
                                        if (currentTime - lastUpdateTime > updateIntervalMs) {
                                            // Trim trailing whitespace/newlines before persisting to avoid large empty gaps
                                            repository.updateMessageContent(placeholderId, totalContent.trimEnd())
                                            lastUpdateTime = currentTime
                                        }
                                    }
                                    
                                    // Check if this segment looks like it was truncated
                                    val segmentTime = System.currentTimeMillis() - segmentStartTime
                                    var isLikelyTruncated = isResponseTruncated(currentSegment, segmentTime)

                                    // Guard: avoid continuing on the first segment after a fresh model load
                                    if (firstGenerationSinceLoad && continuationCount == 0) {
                                        Log.d("ChatViewModel", "First generation after model load - disabling continuation to avoid flicker/loops")
                                        isLikelyTruncated = false
                                        firstGenerationSinceLoad = false
                                    }
                                    
                                    Log.d("ChatViewModel", "Segment ${continuationCount + 1}: length=${currentSegment.length}, time=${segmentTime}ms, hasContent=$segmentHasContent, truncated=$isLikelyTruncated")
                                    
                                    // If the segment has no meaningful content, stop continuing
                                    if (!segmentHasContent && continuationCount > 0) {
                                        Log.d("ChatViewModel", "Segment has no meaningful content, stopping continuation")
                                        break
                                    }
                                    
                                    if (!isLikelyTruncated) {
                                        // Response appears complete
                                        Log.d("ChatViewModel", "Response appears complete, stopping continuation")
                                        break
                                    }
                                    
                                    continuationCount++
                                    Log.d("ChatViewModel", "Detected truncation, continuing generation (attempt ${continuationCount})")
                                    
                                    // Small delay to prevent overwhelming the model
                                    delay(100)
                                }                                // SUCCESS: This section now executes only after the stream is fully collected
                                val finalContent = sanitizeModelOutput(totalContent)
                                Log.d("ChatViewModel", "Generation completed successfully with ${continuationCount} continuations")
                                Log.d("ChatViewModel", "Final content length: ${finalContent.length}")
                                if (finalContent.isBlank()) {
                                    // Check if this is a GGUF model - if so, auto-reset session and retry
                                    val isGgufModel = currentModel?.modelFormat?.equals("gguf", ignoreCase = true) == true
                                    if (isGgufModel && !isRetryAfterContextReset) {
                                        Log.w("ChatViewModel", "No response from GGUF model - context window likely full. Auto-resetting session and retrying...")
                                        triggeredRetry = true
                                        isRetryAfterContextReset = true
                                        // Delete the placeholder and user message (sendMessage will re-add user message)
                                        repository.deleteMessageById(placeholderId)
                                        if (userMessageId != null) repository.deleteMessageById(userMessageId)
                                        // Destroy and reload the model to clear KV cache
                                        try {
                                            inferenceService.resetChatSession(chatId)
                                            Log.d("ChatViewModel", "Successfully reset GGUF model for chat $chatId")
                                        } catch (resetEx: Exception) {
                                            Log.e("ChatViewModel", "Failed to reset GGUF model: ${resetEx.message}")
                                        }
                                        // Drop history so the retry only sends the current prompt
                                        dropHistoryOnce = true
                                        sendMessage(context, messageText, attachmentUri, audioData)
                                        return@launch
                                    } else {
                                        Log.w("ChatViewModel", "No response produced – showing error to user")
                                        val fallback = context.getString(R.string.no_response_produced)
                                        repository.updateMessageContent(placeholderId, fallback.trimEnd())
                                        val streamDurationMs = ((if (lastChunkAt > 0) lastChunkAt else System.currentTimeMillis()) - (if (firstChunkAt > 0) firstChunkAt else generationStartTime)).coerceAtLeast(1L)
                                        finalizeMessage(placeholderId, fallback, streamDurationMs)
                                    }
                                } else {
                                    val safeFinal = finalContent
                                    repository.updateMessageContent(placeholderId, safeFinal.trimEnd())
                                    val time = System.currentTimeMillis() - generationStartTime
                                    val netTime = (time - ragSearchTimeMs).coerceAtLeast(1L)
                                    val streamDurationMs = ((if (lastChunkAt > 0) lastChunkAt else System.currentTimeMillis()) - (if (firstChunkAt > 0) firstChunkAt else generationStartTime)).coerceAtLeast(1L)
                                    Log.d("ChatViewModel", "About to call finalizeMessage for success (raw=${time}ms, rag=${ragSearchTimeMs}ms, net=${netTime}ms, stream=${streamDurationMs}ms)")
                                    finalizeMessage(placeholderId, safeFinal, streamDurationMs)
                                }

                    } catch (e: Exception) {
                        val finalContent = totalContent
                        val time = System.currentTimeMillis() - generationStartTime
                        val netTime = (time - ragSearchTimeMs).coerceAtLeast(1L)
                        
                        Log.d("ChatViewModel", "Exception caught: ${e.javaClass.simpleName}: ${e.message}")
                        Log.d("ChatViewModel", "Final content length: ${finalContent.length}")
                        Log.d("ChatViewModel", "Generation time: ${time}ms")
                        
                        // Handle both CancellationException and JobCancellationException (which extends CancellationException)
                        if (e is kotlinx.coroutines.CancellationException || e.javaClass.simpleName.contains("Cancellation")) {
                            // Check if this was due to repetitive content
                            if (e.message?.contains("Repetitive content detected") == true) {
                                Log.w("ChatViewModel", "Generation cancelled due to repetitive content. Ensuring session is clean.")
                                // Additional session reset to ensure clean state for next interaction
                                viewModelScope.launch {
                                    try {
                                        delay(1000) // Wait for current cleanup to complete
                                        inferenceService.resetChatSession(chatId)
                                        Log.d("ChatViewModel", "Performed additional session reset after repetitive content")
                                    } catch (resetException: Exception) {
                                        Log.w("ChatViewModel", "Failed additional session reset: ${resetException.message}")
                                    }
                                }
                            }
                            // CANCEL: Save partial progress
                            Log.d("ChatViewModel", "Generation was cancelled by user.")
                            Log.d("ChatViewModel", "About to call finalizeMessage for cancellation (raw=${time}ms, rag=${ragSearchTimeMs}ms, net=${netTime}ms)")
                        } else {
                            // ERROR: Model generation error (MediaPipe, ONNX, etc.)
                            Log.e("ChatViewModel", "Model generation error: ${e.message}", e)
                            Log.d("ChatViewModel", "About to call finalizeMessage for error")
                        }
                        
                        // ALWAYS save final content and call finalizeMessage (for both cancel and error) even if the parent Job is cancelled
                        withContext(kotlinx.coroutines.NonCancellable) {
                            if (finalContent.isBlank()) {
                                // Check if this is a GGUF model - if so, auto-reset session and retry
                                val isGgufModel = currentModel?.modelFormat?.equals("gguf", ignoreCase = true) == true
                                if (isGgufModel && !isRetryAfterContextReset) {
                                    Log.w("ChatViewModel", "No response from GGUF model (error path) - context window likely full. Auto-resetting session and retrying...")
                                    triggeredRetry = true
                                    isRetryAfterContextReset = true
                                    // Delete the placeholder and user message (sendMessage will re-add user message)
                                    repository.deleteMessageById(placeholderId)
                                    if (userMessageId != null) repository.deleteMessageById(userMessageId)
                                    // Destroy and reload the model to clear KV cache
                                    try {
                                        inferenceService.resetChatSession(chatId)
                                        Log.d("ChatViewModel", "Successfully reset GGUF model for chat $chatId (error path)")
                                    } catch (resetEx: Exception) {
                                        Log.e("ChatViewModel", "Failed to reset GGUF model (error path): ${resetEx.message}")
                                    }
                                    // Drop history so the retry only sends the current prompt
                                    dropHistoryOnce = true
                                    withContext(Dispatchers.Main) {
                                        sendMessage(context, messageText, attachmentUri, audioData)
                                    }
                                } else {
                                    Log.w("ChatViewModel", "No response produced in error path – showing error to user")
                                    val fallback = context.getString(R.string.no_response_produced)
                                    repository.updateMessageContent(placeholderId, fallback.trimEnd())
                                    val streamDurationMs = ((if (lastChunkAt > 0) lastChunkAt else System.currentTimeMillis()) - (if (firstChunkAt > 0) firstChunkAt else generationStartTime)).coerceAtLeast(1L)
                                    finalizeMessage(placeholderId, fallback, streamDurationMs)
                                }
                            } else {
                                val safeFinal = sanitizeModelOutput(finalContent)
                                repository.updateMessageContent(placeholderId, safeFinal.trimEnd())
                                Log.d("ChatViewModel", "About to call finalizeMessage (NonCancellable)")
                                val streamDurationMs = ((if (lastChunkAt > 0) lastChunkAt else System.currentTimeMillis()) - (if (firstChunkAt > 0) firstChunkAt else generationStartTime)).coerceAtLeast(1L)
                                finalizeMessage(placeholderId, safeFinal, streamDurationMs)
                            }
                        }
                    } finally {
                        if (!handedOffToAutoRegenerate && !triggeredRetry) {
                            _isLoading.value = false
                            _isLoadingModel.value = false
                            isGenerating = false
                            // Clear streaming state for this message, regardless of outcome
                            val updatedStreaming = _streamingContents.value.toMutableMap()
                            updatedStreaming.remove(placeholderId)
                            _streamingContents.value = updatedStreaming
                        }
                        // Reset the retry flag only if we didn't just trigger a retry
                        // This ensures the flag persists into the retry attempt
                        if (isRetryAfterContextReset && !triggeredRetry) {
                            isRetryAfterContextReset = false
                        }
                    }
                }
            } else {
                repository.addMessage(chatId, context.getString(R.string.please_download_model), isFromUser = false)
                _isLoading.value = false
                isGenerating = false
            }
        }
    }

    /**
     * Helper to save the final message content and its token stats.
     * This is called on successful completion or on cancellation.
     */
    private suspend fun finalizeMessage(placeholderId: String, finalContent: String, generationTimeMs: Long) {
        // Ensure we reset the first-generation guard after any completion
        firstGenerationSinceLoad = false

        // Only compute token statistics if we have any content to analyse.
        if (finalContent.isNotBlank()) {

            val actualTokens = kotlin.math.ceil(finalContent.length / 4.0).toInt()
            val tokensPerSecond = if (generationTimeMs > 0) {
                (actualTokens * 1000.0) / generationTimeMs
            } else {
                0.0
            }

            Log.d("ChatViewModel", "Saving stats for message $placeholderId: $actualTokens tokens, ${String.format("%.1f", tokensPerSecond)} tok/sec")

            // Auto-readout: Flush any remaining buffered text when generation completes
            try {
                val autoReadoutEnabled = themePreferences.autoReadoutEnabled.first()
                if (autoReadoutEnabled && _currentTtsMessageId.value == placeholderId) {
                    Log.d("ChatViewModel", "Auto-readout enabled, flushing TTS buffer for message: $placeholderId")
                    // Flush any remaining text in the buffer
                    ttsService.flushStreamingBuffer()
                    
                    // Observe TTS speaking state and clear message ID when it finishes
                    viewModelScope.launch {
                        // Wait for TTS to actually finish speaking
                        ttsService.isSpeaking.collect { isSpeaking ->
                            if (!isSpeaking && _currentTtsMessageId.value == placeholderId) {
                                // TTS has stopped speaking for this message
                                _currentTtsMessageId.value = null
                                Log.d("ChatViewModel", "Cleared TTS message ID after speaking completed")
                                // Cancel this observer job after clearing
                                this.coroutineContext[Job]?.cancel()
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Failed to flush TTS buffer: ${e.message}")
            }
            repository.updateMessageStats(placeholderId, actualTokens, tokensPerSecond)
        } else {
            Log.d("ChatViewModel", "No stats to save for message $placeholderId - content is blank")
        }
    }

    /**
     * Auto-regenerate flow used when the model produces no output.
     * Creates a fresh session and resends only the latest user prompt, clearing prior history for this send.
     * Also rejects prompts that exceed the applied context window using characters/4 estimation.
     */
    private suspend fun autoRegenerateAfterNoResponse(context: Context, chatId: String, placeholderId: String) {
        // Ensure we have a model and latest user message
        val model = currentModel ?: throw IllegalStateException("No current model for auto-regenerate")
        val currentMessages = _messages.value
        val lastUser = currentMessages.lastOrNull { it.isFromUser }
            ?: throw IllegalStateException("No user message to regenerate")

        // Prompt-length validation using applied context window (not model max): use chars/4 approximation
        val promptText = lastUser.content.trim()
        // Use applied/effective context window (honors overrideMaxTokens from inference service)
        val effectiveMax = inferenceService.getEffectiveMaxTokens(model)
        val estimatedPromptTokens = kotlin.math.ceil(promptText.length / 4.0).toInt().coerceAtLeast(1)
        val outputReserve = (effectiveMax * 0.33).toInt().coerceAtLeast(128)
        val threshold = (effectiveMax - outputReserve).coerceAtLeast(1)
        if (estimatedPromptTokens >= threshold) {
            // Replace placeholder with localized too-long message
            repository.updateMessageContent(placeholderId, context.getString(R.string.prompt_too_long))
            finalizeMessage(placeholderId, context.getString(R.string.prompt_too_long), 1L)
            return
        }

        // Reset session and drop history
        try {
            inferenceService.resetChatSession(chatId)
        } catch (_: Exception) { /* best effort */ }
        lastSessionResetAt = System.currentTimeMillis()
        dropHistoryOnce = true

        // Build prompt with only the latest user message
        val fullHistory = "user: ${promptText}\nassistant:"

        // Mark UI as generating so stop button stays visible for the entire auto-regenerate flow
        _isLoading.value = true
        isGenerating = true
        // Ensure model is loaded
        withContext(Dispatchers.IO) { inferenceService.loadModel(model, _selectedBackend.value) }

        // Start a new generation streaming into the same placeholder with proper cancellation and TTS support
        val webSearchEnabled = try { themePreferences.webSearchEnabled.first() } catch (_: Exception) { false }
        val images = if (model.supportsVision && lastUser.attachmentPath != null && lastUser.attachmentType == "image") {
            try {
                val bmp = loadImageFromUri(context, Uri.parse(lastUser.attachmentPath))
                if (bmp != null) listOf(bmp) else emptyList()
            } catch (_: Exception) { emptyList() }
        } else emptyList()

        // Check if auto-readout is enabled before starting generation
        val autoReadoutEnabled = try {
            themePreferences.autoReadoutEnabled.first()
        } catch (e: Exception) {
            false
        }

        // Initialize TTS for streaming if auto-readout is enabled
        if (autoReadoutEnabled) {
            Log.d("ChatViewModel", "Auto-readout enabled for auto-regenerate, preparing TTS for streaming message: $placeholderId")
            // Stop any previous TTS before starting new generation
            ttsService.stop()
            // Set language to app's current locale
            val appLocale = com.llmhub.llmhub.utils.LocaleHelper.getCurrentLocale(context)
            ttsService.setLanguage(appLocale)
            // Set the current TTS message ID so the UI can show stop icon
            _currentTtsMessageId.value = placeholderId
        }

        var totalContent = ""
        var firstChunkAt = 0L
        var lastChunkAt = 0L
        var lastUpdateTime = 0L
        val updateIntervalMs = 50L

        // Create a new generation job that can be cancelled
        generationJob = viewModelScope.launch {
            try {
                val responseStream = inferenceService.generateResponseStreamWithSession(
                    fullHistory, model, chatId, images, null, webSearchEnabled
                )

                responseStream.collect { piece ->
                    val nowTs = System.currentTimeMillis()
                    if (piece.isNotEmpty() && firstChunkAt == 0L) firstChunkAt = nowTs
                    lastChunkAt = nowTs
                    totalContent += piece

                    // Auto-readout: Stream text to TTS as it arrives
                    if (autoReadoutEnabled && _currentTtsMessageId.value == placeholderId && piece.isNotEmpty()) {
                        try {
                            ttsService.addStreamingText(piece)
                        } catch (e: Exception) {
                            Log.w("ChatViewModel", "Failed to add text to TTS stream during auto-regenerate: ${e.message}")
                        }
                    }

                    // Update UI
                    val updated = _streamingContents.value.toMutableMap()
                    updated[placeholderId] = totalContent
                    _streamingContents.value = updated

                    // Debounced database updates
                    val currentTime = System.currentTimeMillis()
                    if (currentTime - lastUpdateTime > updateIntervalMs) {
                        repository.updateMessageContent(placeholderId, totalContent.trimEnd())
                        lastUpdateTime = currentTime
                    }
                }

                // Success - finalize the message
                repository.updateMessageContent(placeholderId, totalContent.trimEnd())
                val streamDurationMs = ((if (lastChunkAt > 0) lastChunkAt else System.currentTimeMillis()) - (if (firstChunkAt > 0) firstChunkAt else System.currentTimeMillis())).coerceAtLeast(1L)
                finalizeMessage(placeholderId, totalContent, streamDurationMs)

            } catch (e: Exception) {
                Log.d("ChatViewModel", "Auto-regenerate exception: ${e.javaClass.simpleName}: ${e.message}")
                
                // Save partial content and finalize
                withContext(kotlinx.coroutines.NonCancellable) {
                    repository.updateMessageContent(placeholderId, totalContent.trimEnd())
                    val streamDurationMs = ((if (lastChunkAt > 0) lastChunkAt else System.currentTimeMillis()) - (if (firstChunkAt > 0) firstChunkAt else System.currentTimeMillis())).coerceAtLeast(1L)
                    finalizeMessage(placeholderId, totalContent, streamDurationMs)
                }
            } finally {
                _isLoading.value = false
                isGenerating = false
                
                // Clear streaming state
                val updatedStreaming = _streamingContents.value.toMutableMap()
                updatedStreaming.remove(placeholderId)
                _streamingContents.value = updatedStreaming
            }
        }
    }

    // Basic disallowed content filter (client-side heuristic; not exhaustive)
    private fun isDisallowedPrompt(prompt: String): Boolean {
        // User requested to always allow responses, so client-side disallow list is disabled.
        // Retain method for potential future policy reinstatement; always return false now.
        return false
    }

    /**
     * Sanitize raw model output:
     * - Trim surrounding whitespace
     * - Drop leading role tags like "assistant:" / "user:" that some models echo
     * - Collapse excessive blank lines
     */
    private fun sanitizeModelOutput(raw: String): String {
        if (raw.isBlank()) return ""
        var cleaned = raw
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            // Convert visible escape sequences like "\n" into real newlines if the model emitted them
            .replace("\\n", "\n")
            .trim()

        // Remove any leading role tags on first line
        cleaned = cleaned.replaceFirst(Regex("^(assistant|system|user)\\s*:\\s*", RegexOption.IGNORE_CASE), "")

        // If the whole thing is just a role word, blank it out
        if (cleaned.equals("assistant", ignoreCase = true) || cleaned.equals("user", ignoreCase = true) || cleaned.equals("system", ignoreCase = true)) {
            return ""
        }

        // Collapse runs of 3+ newlines to at most 2
        cleaned = cleaned.replace(Regex("\n{3,}"), "\n\n")

        return cleaned.trimEnd()
    }
    
    /**
     * Heuristic to detect if a response was truncated due to token limits.
     * This looks for patterns that suggest the model stopped mid-sentence.
     */
    private fun isResponseTruncated(content: String, generationTimeMs: Long): Boolean {
        if (content.isBlank()) return false
        
        val trimmed = content.trim()
        Log.d("ChatViewModel", "Checking truncation for content ending with: '${trimmed.takeLast(20)}'")
        
        // First check: If the content is just whitespace or newlines, the model has reached its limit
        val lastChars = content.takeLast(10)
        val isOnlyWhitespace = lastChars.all { it.isWhitespace() }
        if (isOnlyWhitespace && content.length > 10) {
            Log.d("ChatViewModel", "Model output is only whitespace - stopping continuation")
            return false // Don't continue if model is only outputting whitespace
        }
        
        // Second check: If the content is very short with minimal generation time, it might be truncated
        val isVeryShort = generationTimeMs < 1000 && trimmed.length < 50
        
        // Third check: Look for clear signs of completion
        val endsWithProperPunctuation = trimmed.matches(Regex(".*[.!?][)\\]\"']*\\s*$"))
        
        // Fourth check: Look for clear signs of incompleteness
        val endsWithIncompletePattern = trimmed.matches(Regex(".*[,;:]\\s*$")) || // Ends with comma, semicolon, or colon
                trimmed.matches(Regex(".*\\b(and|or|but|the|a|an|to|for|with|in|on|at|by|from|of|as|if|when|where|while|until|because|since|although|though|however|therefore|thus|hence|moreover|furthermore|nevertheless|nonetheless|meanwhile|otherwise|instead|besides|additionally|finally|consequently|specifically|particularly|especially|importantly|significantly|unfortunately|surprisingly|interestingly|obviously|clearly|essentially|basically|generally|typically|usually|normally|commonly|frequently|occasionally|rarely|sometimes|often|always|never|perhaps|possibly|probably|likely|certainly|definitely|absolutely|completely|entirely|totally|quite|rather|very|extremely|incredibly|remarkably|surprisingly|unfortunately|hopefully|thankfully|luckily|fortunately|regrettably|sadly|happily|proudly|confidently|eagerly|patiently|carefully|quickly|slowly|quietly|loudly|gently|firmly|softly|harshly|kindly|warmly|coolly|coldly|hotly|angrily|calmly|peacefully|violently|suddenly|gradually|immediately|eventually|ultimately|initially|originally|previously|recently|currently|presently|temporarily|permanently|briefly|extensively|thoroughly|partially|completely|entirely|fully|barely|hardly|scarcely|almost|nearly|approximately|roughly|exactly|precisely|specifically|generally|particularly|especially|mainly|primarily|chiefly|largely|mostly|partly|somewhat|quite|rather|fairly|pretty|really|truly|actually|literally|virtually|practically|essentially|basically|fundamentally|inherently|naturally|obviously|clearly|apparently|evidently|presumably|supposedly|allegedly|reportedly|seemingly)\\s*$", RegexOption.IGNORE_CASE)) || // Ends with common incomplete words
                trimmed.matches(Regex(".*\\b\\w+[-']\\s*$")) // Ends with hyphenated/apostrophe word (incomplete)
        
        // Fifth check: Look for unclosed markdown code blocks
        val unclosedCodeBlocks = content.count { it == '`' } % 2 != 0
        
        // Sixth check: If content is getting repetitive (same phrases repeating), don't continue
        val words = trimmed.split(Regex("\\s+"))
        var isRepetitive = false
        
        if (words.size > 20) {
            // Check for repeating 10-word patterns
            val lastTenWords = words.takeLast(10).joinToString(" ")
            val beforeTenWords = words.dropLast(10).takeLast(10).joinToString(" ")
            if (lastTenWords == beforeTenWords) {
                isRepetitive = true
            }
        }
        
        // Additional check for character-level repetition (like long URLs with repeated characters)
        if (!isRepetitive && trimmed.length > 100) {
            val last50Chars = trimmed.takeLast(50)
            val before50Chars = trimmed.dropLast(50).takeLast(50)
            if (last50Chars == before50Chars) {
                isRepetitive = true
                Log.d("ChatViewModel", "Detected character-level repetition")
            }
        }
        
        // Check for excessive repetition of the same character (like 7777777...)
        if (!isRepetitive && trimmed.length > 50) {
            val last30Chars = trimmed.takeLast(30)
            val mostCommonChar = last30Chars.groupingBy { it }.eachCount().maxByOrNull { it.value }
            if (mostCommonChar != null && mostCommonChar.value > 20) {
                isRepetitive = true
                Log.d("ChatViewModel", "Detected excessive character repetition: '${mostCommonChar.key}' repeated ${mostCommonChar.value} times")
            }
        }
        
        // Check for URL-like patterns that are getting too long
        if (!isRepetitive && trimmed.contains("http") && trimmed.length > 200) {
            val urlPattern = Regex("https?://[^\\s]+")
            val urls = urlPattern.findAll(trimmed).toList()
            for (url in urls) {
                if (url.value.length > 150) {
                    isRepetitive = true
                    Log.d("ChatViewModel", "Detected excessively long URL pattern: ${url.value.take(50)}...")
                    break
                }
            }
        }
        
        if (isRepetitive) {
            Log.d("ChatViewModel", "Content is repetitive - stopping continuation")
            return false
        }
        
        // Only continue if we have clear signs of incompleteness AND the content is meaningful
        val shouldContinue = !endsWithProperPunctuation && (endsWithIncompletePattern || isVeryShort || unclosedCodeBlocks)
        
        Log.d("ChatViewModel", "Truncation check: endsWithProperPunctuation=$endsWithProperPunctuation, endsWithIncompletePattern=$endsWithIncompletePattern, isVeryShort=$isVeryShort, unclosedCodeBlocks=$unclosedCodeBlocks, isOnlyWhitespace=$isOnlyWhitespace, isRepetitive=$isRepetitive, shouldContinue=$shouldContinue")
        
        return shouldContinue
    }

    /** Quick check for repetitive patterns during streaming */
    /** Interrupt the current response generation if one is running */
    fun stopGeneration() {
        generationJob?.cancel()
        generationJob = null
        _isLoading.value = false
        _isLoadingModel.value = false
        isGenerating = false
        // Clear TTS message ID when generation is stopped
        _currentTtsMessageId.value = null
        ttsService.stop()
        // Note: Don't clear streaming contents here to preserve partial responses
        // _tokenStats.value = null // No longer needed
    }
    
    /** Clear the current TTS message ID (called when manually stopping TTS) */
    fun clearCurrentTtsMessage() {
        _currentTtsMessageId.value = null
    }

    /**
     * Enable manual TTS for a specific message id. This will prepare the TTS engine,
     * set the current TTS message id so streaming will be routed to the TTS service,
     * and enqueue the already-generated content without flushing the streaming queue.
     */
    fun enableManualTtsForMessage(messageId: String, content: String) {
        try {
            // Stop any non-stream playback and prepare language
            ttsService.stop()
            val appLocale = com.llmhub.llmhub.utils.LocaleHelper.getCurrentLocale(context)
            ttsService.setLanguage(appLocale)

            // Mark this message as the active TTS target so generation will stream into TTS
            _currentTtsMessageId.value = messageId

            // Enqueue already-generated content without flushing queued streaming utterances
            if (content.isNotBlank()) {
                ttsService.speakAppend(content)
            }
        } catch (e: Exception) {
            Log.w("ChatViewModel", "enableManualTtsForMessage failed: ${e.message}")
        }
    }

    fun switchModel(newModel: LLMModel) {
        viewModelScope.launch {
            // Reset vision disabled flag for regular model switching
            isVisionDisabled = false
            // Immediately reflect the user's choice in UI
            _selectedModel.value = newModel
            // Ensure ongoing generation is fully cancelled before switching models to avoid
            // MediaPipe graph errors (e.g., DetokenizerCalculator id >= 0) that can happen
            // when a new session starts while the previous one is still cleaning up.
            generationJob?.let { job ->
                job.cancel()
                // Wait for the coroutine (and its awaitClose cleanup) to finish
                try {
                    job.join()
                } catch (ignored: CancellationException) {
                    // Expected when job is already cancelled
                }
                generationJob = null
            }

            _isLoading.value = true
            _isLoadingModel.value = true
            isGenerating = true
            
            // Clear all existing sessions before switching models
            try {
                inferenceService.onCleared()
                delay(1000) // Give more time for complete cleanup when switching models
                Log.d("ChatViewModel", "Cleared all sessions before model switch")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Error clearing sessions before model switch: ${e.message}")
            }
            
            currentModel = newModel
            val updatedChat = _currentChat.value?.copy(modelName = newModel.name)
            if (updatedChat != null) {
                currentChatId?.let { repository.updateChatModel(it, newModel.name) }
                _currentChat.value = updatedChat
            }
            
            // Pre-load the model when switching
            try {
                // Trigger model loading without generating content
                inferenceService.loadModel(newModel, _selectedBackend.value)
                // Sync the currently loaded model state
                syncCurrentlyLoadedModel()
                // Set the first-generation guard
                firstGenerationSinceLoad = true
                Log.d("ChatViewModel", "Successfully loaded new model: ${newModel.name}")
                
                // Clear the first generation guard after a reasonable timeout
                viewModelScope.launch {
                    delay(3000) // Clear guard after 3 seconds
                    firstGenerationSinceLoad = false
                    Log.d("ChatViewModel", "Cleared first generation guard for model: ${newModel.name}")
                }
                
                // Only reset session if we're switching to a different model
                currentChatId?.let { chatId ->
                    val currentModel = inferenceService.getCurrentlyLoadedModel()
                    if (currentModel?.name != newModel.name) {
                        try {
                            delay(500) // Give more time for model loading to complete
                            inferenceService.resetChatSession(chatId)
                            delay(200) // Additional time for session to stabilize
                            Log.d("ChatViewModel", "Reset chat session $chatId after model switch")
                        } catch (e: Exception) {
                            Log.w("ChatViewModel", "Failed to reset chat session after model switch: ${e.message}")
                            // If reset fails, try to clear and recreate the entire session
                            try {
                                delay(300)
                                inferenceService.onCleared()
                                delay(300)
                                withContext(Dispatchers.IO) { inferenceService.loadModel(newModel, _selectedBackend.value) }
                                Log.d("ChatViewModel", "Force recreated session after reset failure")
                            } catch (recreateException: Exception) {
                                Log.w("ChatViewModel", "Force recreation also failed: ${recreateException.message}")
                            }
                        }
                    } else {
                        Log.d("ChatViewModel", "Skipping session reset - same model already loaded")
                    }
                }
                
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Failed to load new model: ${e.message}")
                // Even if loading defers to first use, still guard the first generation
                firstGenerationSinceLoad = true
            }
            
            _isLoadingModel.value = false
            _isLoading.value = false
            isGenerating = false
        }
    }
    
    fun switchModelWithBackend(newModel: LLMModel, backend: LlmInference.Backend) {
        viewModelScope.launch {
            // Reset vision disabled flag for regular model switching
            isVisionDisabled = false
            // Immediately reflect the user's choice in UI
            _selectedModel.value = newModel
            // Ensure ongoing generation is fully cancelled before switching models to avoid
            // MediaPipe graph errors (e.g., DetokenizerCalculator id >= 0) that can happen
            // when a new session starts while the previous one is still cleaning up.
            generationJob?.let { job ->
                job.cancel()
                // Wait for the coroutine (and its awaitClose cleanup) to finish
                try {
                    job.join()
                } catch (ignored: CancellationException) {
                    // Expected when job is already cancelled
                }
                generationJob = null
            }

            _isLoading.value = true
            _isLoadingModel.value = true
            isGenerating = true
            
            // Clear all existing sessions before switching models
            try {
                inferenceService.onCleared()
                delay(1000) // Give more time for complete cleanup when switching models
                Log.d("ChatViewModel", "Cleared all sessions before model switch")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Error clearing sessions before model switch: ${e.message}")
            }
            
            currentModel = newModel
            val updatedChat = _currentChat.value?.copy(modelName = newModel.name)
            if (updatedChat != null) {
                currentChatId?.let { repository.updateChatModel(it, newModel.name) }
                _currentChat.value = updatedChat
            }
            
            // Pre-load the model when switching with specified backend
            try {
                // Trigger model loading without generating content
                inferenceService.loadModel(newModel, backend)
                // Sync the currently loaded model state
                syncCurrentlyLoadedModel()
                // Set the first-generation guard
                firstGenerationSinceLoad = true
                Log.d("ChatViewModel", "Successfully loaded new model: ${newModel.name} with backend: $backend")
                
                // Clear the first generation guard after a reasonable timeout
                viewModelScope.launch {
                    delay(3000) // Clear guard after 3 seconds
                    firstGenerationSinceLoad = false
                    Log.d("ChatViewModel", "Cleared first generation guard for model: ${newModel.name}")
                }
                
                // Only reset session if we're switching to a different model
                currentChatId?.let { chatId ->
                    val currentModel = inferenceService.getCurrentlyLoadedModel()
                    if (currentModel?.name != newModel.name) {
                        try {
                            delay(500) // Give more time for model loading to complete
                            inferenceService.resetChatSession(chatId)
                            delay(200) // Additional time for session to stabilize
                            Log.d("ChatViewModel", "Reset chat session $chatId after model switch")
                        } catch (e: Exception) {
                            Log.w("ChatViewModel", "Failed to reset chat session after model switch: ${e.message}")
                            // If reset fails, try to clear and recreate the entire session
                            try {
                                delay(300)
                                inferenceService.onCleared()
                                delay(300)
                                inferenceService.loadModel(newModel, backend)
                                Log.d("ChatViewModel", "Force recreated session after reset failure")
                            } catch (recreateException: Exception) {
                                Log.w("ChatViewModel", "Force recreation also failed: ${recreateException.message}")
                            }
                        }
                    } else {
                        Log.d("ChatViewModel", "Skipping session reset - same model already loaded")
                    }
                }
                
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Failed to load new model: ${e.message}")
                // Even if loading defers to first use, still guard the first generation
                firstGenerationSinceLoad = true
            }
            
            _isLoadingModel.value = false
            _isLoading.value = false
            isGenerating = false
        }
    }

    // Remove stray artifacts that confuse tiny models (ellipsis blocks, zero-width spaces, markdown fences)
    private fun sanitizeForPrompt(text: String): String {
        if (text.isBlank()) return text
        var cleaned = text
            .replace("\u200B", "")
            .replace(Regex("[\u2028\u2029]"), "\n")
            .replace(Regex("`{3,}"), "")
            .replace(Regex("\n{3,}"), "\n\n")
            .trim()
        cleaned = cleaned.lines()
            .map { it.trimEnd() }
            .filter { line -> line.length >= 3 && !Regex("^[`*._-]+$").matches(line) }
            .joinToString("\n")
        return cleaned
    }
    
    fun switchModelWithBackend(newModel: LLMModel, backend: LlmInference.Backend, disableVision: Boolean) {
        // Call the new overloaded method with disableAudio = false for backward compatibility
        switchModelWithBackend(newModel, backend, disableVision, disableAudio = false)
    }
    
    fun switchModelWithBackend(newModel: LLMModel, backend: LlmInference.Backend, disableVision: Boolean, disableAudio: Boolean) {
        viewModelScope.launch {
            // Store the modality disabled states
            isVisionDisabled = disableVision
            isAudioDisabled = disableAudio
            // Immediately reflect the user's choice in UI
            _selectedModel.value = newModel
            _selectedBackend.value = backend
            
            // Persist settings
            saveChatSettings()
            
            // Ensure ongoing generation is fully cancelled before switching models to avoid
            // MediaPipe graph errors (e.g., DetokenizerCalculator id >= 0) that can happen
            // when a new session starts while the previous one is still cleaning up.
            generationJob?.let { job ->
                job.cancel()
                // Wait for the coroutine (and its awaitClose cleanup) to finish
                try {
                    job.join()
                } catch (ignored: CancellationException) {
                    // Expected when job is already cancelled
                }
                generationJob = null
            }

            _isLoading.value = true
            _isLoadingModel.value = true
            isGenerating = true
            
            // Clear all existing sessions before switching models
            try {
                inferenceService.onCleared()
                delay(1000) // Give more time for complete cleanup when switching models
                Log.d("ChatViewModel", "Cleared all sessions before model switch")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Error clearing sessions before model switch: ${e.message}")
            }
            
            val currentModel = inferenceService.getCurrentlyLoadedModel()
            val updatedChat = _currentChat.value?.copy(modelName = newModel.name)
            if (updatedChat != null) {
                currentChatId?.let { repository.updateChatModel(it, newModel.name) }
                _currentChat.value = updatedChat
            }
            
            // Pre-load the model when switching with specified backend and modality settings
            try {
                // Trigger model loading with modality options
                inferenceService.loadModel(newModel, backend, disableVision, disableAudio)
                // Sync the currently loaded model state
                syncCurrentlyLoadedModel()
                // Set the first-generation guard
                firstGenerationSinceLoad = true
                Log.d("ChatViewModel", "Successfully loaded new model: ${newModel.name} with backend: $backend, vision disabled: $disableVision, audio disabled: $disableAudio")
                
                // Clear the first generation guard after a reasonable timeout
                viewModelScope.launch {
                    delay(3000) // Clear guard after 3 seconds
                    firstGenerationSinceLoad = false
                    Log.d("ChatViewModel", "Cleared first generation guard for model: ${newModel.name}")
                }
                
                // Only reset session if we're switching to a different model
                currentChatId?.let { chatId ->
                    val currentModel = inferenceService.getCurrentlyLoadedModel()
                    if (currentModel?.name != newModel.name) {
                        try {
                            delay(500) // Give more time for model loading to complete
                            inferenceService.resetChatSession(chatId)
                            delay(200) // Additional time for session to stabilize
                            Log.d("ChatViewModel", "Reset chat session $chatId after model switch")
                        } catch (e: Exception) {
                            Log.w("ChatViewModel", "Failed to reset chat session after model switch: ${e.message}")
                            // If reset fails, try to clear and recreate the entire session
                            try {
                                delay(300)
                                inferenceService.onCleared()
                                delay(300)
                                inferenceService.loadModel(newModel, backend, disableVision)
                                Log.d("ChatViewModel", "Force recreated session after reset failure")
                            } catch (recreateException: Exception) {
                                Log.w("ChatViewModel", "Force recreation also failed: ${recreateException.message}")
                            }
                        }
                    } else {
                        Log.d("ChatViewModel", "Skipping session reset - same model already loaded")
                    }
                }
                
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Failed to load new model: ${e.message}")
                // Even if loading defers to first use, still guard the first generation
                firstGenerationSinceLoad = true
            }
            
            _isLoadingModel.value = false
            _isLoading.value = false
            isGenerating = false
        }
    }
    
    fun isGemmaModel(model: LLMModel): Boolean {
        return model.name.contains("Gemma", ignoreCase = true)
    }

    fun unloadModel() {
        viewModelScope.launch {
            try {
                _isLoadingModel.value = true
                Log.d("ChatViewModel", "Unloading current model")
                
                // Cancel any ongoing generation
                generationJob?.let { job ->
                    job.cancel()
                    try {
                        job.join()
                    } catch (ignored: CancellationException) {
                        // Expected when job is already cancelled
                    }
                    generationJob = null
                }
                
                // Unload the model from inference service
                inferenceService.unloadModel()
                
                // Clear current model reference
                currentModel = null
                _selectedModel.value = null
                
                // Update the currently loaded model state
                syncCurrentlyLoadedModel()
                
                // Update current chat to show no model selected
                _currentChat.value?.let { chat ->
                    currentChatId?.let { chatId ->
                        repository.updateChatModel(chatId, context.getString(R.string.no_model_selected))
                        _currentChat.value = repository.getChatById(chatId)
                    }
                }
                
                Log.d("ChatViewModel", "Model unloaded successfully")
                
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Failed to unload model: ${e.message}")
            } finally {
                _isLoadingModel.value = false
            }
        }
    }

    /**
     * Unloads the model from the inference engine only, without clearing the selected model.
     * Used when leaving the chat screen so memory is freed but the user's model selection is kept.
     * When the user returns and sends a message, the model will be loaded again.
     */
    fun unloadModelFromInferenceOnly() {
        viewModelScope.launch {
            try {
                generationJob?.let { job ->
                    job.cancel()
                    try {
                        job.join()
                    } catch (_: CancellationException) { }
                    generationJob = null
                }
                inferenceService.unloadModel()
                _currentlyLoadedModel.value = null
                Log.d("ChatViewModel", "Model unloaded from inference (selection kept)")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Failed to unload model from inference: ${e.message}")
            }
        }
    }

    override fun onCleared() {
        viewModelScope.launch {
            // Clean up RAG and TTS only. Do NOT call inferenceService.onCleared() here:
            // the inference service is app-scoped and the ViewModel is keyed by chatId,
            // so onCleared() runs when switching chats (new ViewModel for new chat).
            // Unloading here would force ONNX (and other backends) to reload on every new chat.
            ragServiceManager.cleanup() // Clean up RAG service
            ttsService.shutdown() // Clean up TTS service
        }
        super.onCleared()
    }
    
    fun closeCurrentChatSession() {
        currentChatId?.let { chatId ->
            viewModelScope.launch {
                try {
                    // Reset the session instead of closing it
                    inferenceService.resetChatSession(chatId)
                    Log.d("ChatViewModel", "Reset session for chat $chatId")
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Error resetting chat session: ${e.message}")
                }
            }
        }
    }
    
    fun currentModelSupportsVision(): Boolean {
        val currentModel = inferenceService.getCurrentlyLoadedModel()
        return currentModel?.supportsVision == true && !isVisionDisabled
    }
    
    fun currentModelSupportsAudio(): Boolean {
        val currentModel = inferenceService.getCurrentlyLoadedModel()
        return currentModel?.supportsAudio == true && !isAudioDisabled
    }
    
    fun isGpuBackendEnabled(): Boolean {
        return inferenceService.isGpuBackendEnabled()
    }
    
    fun isVisionCurrentlyDisabled(): Boolean {
        return isVisionDisabled
    }
    
    fun isAudioCurrentlyDisabled(): Boolean {
        return isAudioDisabled
    }

    fun setGenerationParameters(maxTokens: Int?, topK: Int?, topP: Float?, temperature: Float?) {
        try {
            inferenceService.setGenerationParameters(maxTokens, topK, topP, temperature)
            Log.d("ChatViewModel", "Forwarded generation parameters to inference service: maxTokens=$maxTokens topK=$topK topP=$topP temperature=$temperature")
        } catch (e: Exception) {
            Log.w("ChatViewModel", "Failed to set generation parameters: ${e.message}")
        }
    }
    
    /**
     * Check if a chat contains images by examining its messages
     */
    suspend fun chatContainsImages(chatId: String): Boolean {
        return try {
            val messages = repository.getMessagesForChatSync(chatId)
            messages.any { it.attachmentType == "image" }
        } catch (e: Exception) {
            Log.w("ChatViewModel", "Error checking if chat contains images: ${e.message}")
            false
        }
    }

    private fun determineAttachmentType(uri: Uri?): String? {
        if (uri == null) return null
        val uriString = uri.toString().lowercase()
        return when {
            uriString.contains("image") || uriString.endsWith(".jpg") ||
            uriString.endsWith(".jpeg") || uriString.endsWith(".png") ||
            uriString.endsWith(".gif") || uriString.endsWith(".webp") -> "IMAGE"
            uriString.endsWith(".pdf") -> "PDF"
            uriString.endsWith(".doc") || uriString.endsWith(".docx") -> "WORD"
            uriString.endsWith(".xls") || uriString.endsWith(".xlsx") -> "EXCEL"
            uriString.endsWith(".ppt") || uriString.endsWith(".pptx") -> "POWERPOINT"
            uriString.endsWith(".txt") || uriString.endsWith(".md") || uriString.endsWith(".csv") -> "TEXT"
            uriString.endsWith(".json") -> "JSON"
            uriString.endsWith(".xml") -> "XML"
            else -> "UNKNOWN"
        }
    }

    /**
     * Build context-aware history that respects the model's context window limits.
     * This implements conversation-pair-aware truncation to maintain context flow.
     */
    private fun buildContextAwareHistory(messages: List<MessageEntity>): String {
        val model = currentModel ?: return ""
        val currentChatId = currentChatId ?: return ""
        
        // One-time priming: when returning to a chat from history, include ONLY
        // the last user/assistant pair to quickly re-establish immediate context.
        if (primeWithLastPairOnce) {
            val chatMessages = messages.filter { it.chatId == currentChatId }
            // Find the last assistant message and its preceding user message
            var lastAssistant: MessageEntity? = null
            var precedingUser: MessageEntity? = null
            for (i in chatMessages.indices.reversed()) {
                val msg = chatMessages[i]
                if (!msg.isFromUser) {
                    lastAssistant = msg
                    // Walk backwards to find the nearest prior user message
                    var j = i - 1
                    while (j >= 0) {
                        val prev = chatMessages[j]
                        if (prev.isFromUser) {
                            precedingUser = prev
                            break
                        }
                        j--
                    }
                    break
                }
            }
            primeWithLastPairOnce = false
            if (lastAssistant != null && precedingUser != null) {
                val userText = if (precedingUser!!.attachmentPath != null && precedingUser!!.attachmentType == "image") {
                    val base = precedingUser!!.content
                    if (base.isNotEmpty()) "$base [Image attached]" else "[Image attached]"
                } else {
                    precedingUser!!.content
                }
                return listOf(
                    "user: $userText",
                    if (lastAssistant!!.content.isNotEmpty()) "assistant: ${lastAssistant!!.content}" else "assistant:"
                ).joinToString("\n\n")
            } else {
                // If no complete pair, fall through to normal handling
            }
        }
        
    // If we very recently reset the underlying model session, avoid immediately stuffing
    // the entire prior history back in. That would just refill the fresh session and can
    // cause the next tiny user prompt (e.g. "1+1") to appear unanswerable if the model
    // internally rejects overlong initial contexts. We keep only a small tail plus a
    // synthetic summary note in that case.
    val recentResetWindowMs = 5_000L
    val now = System.currentTimeMillis()
    // Consider both local and service-level reset markers
    val recentlyResetLocal = (now - lastSessionResetAt) < recentResetWindowMs
    val recentlyResetService = try { inferenceService.wasSessionRecentlyReset(currentChatId) } catch (_: Exception) { false }
    val recentlyReset = recentlyResetLocal || recentlyResetService
    if (dropHistoryOnce) {
        Log.d("ChatViewModel", "dropHistoryOnce flag set; returning empty history and clearing flag")
        dropHistoryOnce = false
        return ""
    }
        
        // Filter messages to only include current chat messages
        val chatMessages = messages.filter { it.chatId == currentChatId }
        
    // Sliding window policy: reserve ~1/3 of the model window for the next response and
    // system overhead. Use the remaining ~2/3 for history + current user prompt.
    // If we just reset, be stricter to guarantee fast recovery.
    val historyFraction = if (recentlyReset) 0.30 else 0.66
        
        // For GGUF/ONNX, respect user slider. For Task models, use model defaults.
        val limitBase = if (model.modelFormat.equals("onnx", ignoreCase = true) || model.modelFormat.equals("gguf", ignoreCase = true)) {
            inferenceService.getEffectiveMaxTokens(model)
        } else {
            model.contextWindowSize
        }
        
        // For local GGUF/Nexa, we can use most of the window for history. 
        // We'll leave only a 2048 token buffer for the response, using the rest for history.
        val maxContextTokens = if (model.modelFormat.equals("gguf", ignoreCase = true)) {
            (limitBase - 2048).coerceAtLeast(limitBase / 2).coerceAtLeast(512)
        } else {
            (limitBase * historyFraction).toInt().coerceAtLeast(256)
        }
        val maxContextChars = maxContextTokens * 4 
        
        Log.d("ChatViewModel", "Context window: Model ${model.name} (${model.modelFormat}) has ${model.contextWindowSize} tokens, effectively using $limitBase. Allocating $maxContextTokens tokens for history.")
        
        // Convert messages to conversation pairs for better context management
        val conversationPairs = mutableListOf<Pair<MessageEntity?, MessageEntity>>()
        var currentUserMessage: MessageEntity? = null
        
        for (message in chatMessages) {
            if (message.isFromUser) {
                currentUserMessage = message
            } else {
                // Assistant message - pair with previous user message if available
                conversationPairs.add(Pair(currentUserMessage, message))
                currentUserMessage = null
            }
        }
        
        // If there's a pending user message without response, add it
        if (currentUserMessage != null) {
            conversationPairs.add(Pair(currentUserMessage, MessageEntity(
                id = "pending",
                chatId = currentUserMessage.chatId,
                content = "",
                isFromUser = false,
                timestamp = System.currentTimeMillis()
            )))
        }
        
        Log.d("ChatViewModel", "Built ${conversationPairs.size} conversation pairs from ${chatMessages.size} chat messages")
        
        // Convert pairs to formatted strings
        val pairStrings = conversationPairs.map { (userMsg, assistantMsg) ->
            val userPart = if (userMsg != null) {
                var userContent = userMsg.content
                // For vision models, indicate when images are present
                if (userMsg.attachmentPath != null && userMsg.attachmentType == "image") {
                    userContent = if (userContent.isNotEmpty()) "$userContent [Image attached]" else "[Image attached]"
                }
                "user: $userContent"
            } else ""
            val assistantPart = if (assistantMsg.content.isNotEmpty()) "assistant: ${assistantMsg.content}" else ""
            
            listOf(userPart, assistantPart).filter { it.isNotEmpty() }.joinToString("\n")
        }
        
        // Calculate total length
        val fullHistory = pairStrings.joinToString(separator = "\n\n")
        
        // QUICK EXIT: If recent reset, drop all prior history. Start truly fresh.
        if (recentlyReset) {
            Log.d("ChatViewModel", "Recent reset detected; returning empty history (fully fresh context)")
            return ""
        }
        
        // If full history fits under relaxed (non-reset) fraction, return it.
        if (fullHistory.length <= maxContextChars) {
            Log.d("ChatViewModel", "Full conversation history fits in allotted history window (${fullHistory.length} chars)")
            return fullHistory
        }
        
        // Otherwise, implement smart truncation
        val recentPairs = mutableListOf<String>()
        var currentLength = 0
        
    // Always keep at least a small recent window (minimum viable context)
    val minimumPairs = 1
        
    // Hard upper bound safeguard: never include more than maxPairsHistory pairs to avoid pathological large messages.
    val maxPairsHistory = 18
    val cappedPairStrings = if (pairStrings.size > maxPairsHistory) pairStrings.takeLast(maxPairsHistory) else pairStrings
        
        // Add pairs from most recent backwards until we hit the limit
        for (i in cappedPairStrings.indices.reversed()) {
            val pairString = cappedPairStrings[i]
            val pairLength = pairString.length + 2 // +2 for double newline separator
            
            if (currentLength + pairLength <= maxContextChars) {
                recentPairs.add(0, pairString) // Add to beginning
                currentLength += pairLength
            } else if (recentPairs.size < minimumPairs) {
                // Force include minimum pairs even if they exceed limit slightly
                recentPairs.add(0, pairString)
                currentLength += pairLength
                Log.d("ChatViewModel", "Forced inclusion of essential conversation pair (exceeds limit)")
            } else {
                break
            }
        }
        
        // Return truncated history without meta-commentary (which confuses models)
        val result = recentPairs.joinToString("\n\n")
        val truncatedCount = cappedPairStrings.size - recentPairs.size
        
        Log.d("ChatViewModel", "Context window management: Original ${fullHistory.length} chars (${pairStrings.size} pairs), trimmed to ${result.length} chars (${recentPairs.size} pairs, ${truncatedCount} truncated)")
        
        // Debug: Log first 200 chars of context to verify it's correct
        val preview = result.take(200) + if (result.length > 200) "..." else ""
        Log.d("ChatViewModel", "Context preview for chat $currentChatId: $preview")
        
        return result
    }

    /**
     * Lightweight topic-shift heuristic. We avoid any heavy NLP to keep on-device cost near-zero.
     * Strategy:
     *  - Look at the last few user messages (excluding the current one) and build a bag-of-words.
     *  - If lexical overlap (unique stem-ish tokens) between current prompt and that bag is very low (<15%) AND
     *    the current prompt introduces new high-signal nouns/adjectives, treat as topic shift.
     *  - Ignore extremely short prompts (handled elsewhere) and prompts that are obviously follow-ups (start with pronouns like "it", "that", etc.).
     */
    private fun isTopicShift(currentPrompt: String, messages: List<MessageEntity>): Boolean {
        val cleaned = currentPrompt.lowercase().trim()
        if (cleaned.length < 5) return false // tiny handled separately
        // If starts with anaphoric reference, likely continuation
        if (cleaned.startsWith("it ") || cleaned.startsWith("that ") || cleaned.startsWith("they ") || cleaned.startsWith("those ")) return false

        // Collect last 5 prior user messages (excluding current construction temp message)
        val priorUserTexts = messages.asReversed()
            .filter { it.isFromUser }
            .map { it.content }
            .filter { it.isNotBlank() }
            .drop(1) // drop the just-added current user message if present
            .take(5)

        if (priorUserTexts.isEmpty()) return false

        val tokenize: (String) -> Set<String> = { text ->
            text.lowercase()
                .replace(Regex("[^a-z0-9\\s]"), " ")
                .split(Regex("\\s+"))
                .map { it.trim() }
                .filter { it.length >= 3 }
                .map { it.trimEnd('s') } // crude plural singularization
                .filter { it.isNotBlank() }
                .toSet()
        }

        val priorTokens = tokenize(priorUserTexts.joinToString(" "))
        if (priorTokens.isEmpty()) return false
        val currentTokens = tokenize(cleaned)
        if (currentTokens.isEmpty()) return false

        val overlap = currentTokens.intersect(priorTokens).size
        val overlapRatio = overlap.toDouble() / currentTokens.size.toDouble().coerceAtLeast(1.0)

        // New high-signal tokens = tokens not in prior AND length>=5
        val newSignal = currentTokens.count { it !in priorTokens && it.length >= 5 }

        // Trigger if overlap very low AND there is at least one new high-signal token
        return overlapRatio < 0.15 && newSignal > 0
    }

    /**
     * Check if we should reset the session before sending a message to prevent token overflow
     */
    private suspend fun shouldResetSessionBeforeMessage(message: String, chatId: String): Boolean {
        val model = currentModel ?: return false
        
        try {
            // Get the current conversation history that would be sent
            val history = buildContextAwareHistory(_messages.value)
            val fullPrompt = if (history.isNotEmpty()) {
                "$history\n\nuser: $message\nassistant:"
            } else {
                "user: $message\nassistant:"
            }
            
            // Rough estimate of token count (1 token ≈ 4 characters)
            val estimatedTokens = fullPrompt.length / 4
            val maxTokens = minOf(model.contextWindowSize, extractCacheSizeFromUrl(model.url) ?: model.contextWindowSize)
            // Reserve ~1/3 of the context window for model response. Reset when input >= (max - reserve)
            val reserveForResponse = (maxTokens * 0.33).toInt().coerceAtLeast(128)
            val tokenThreshold = (maxTokens - reserveForResponse).coerceAtLeast(1)
            
            Log.d("ChatViewModel", "Token check for chat $chatId: ~$estimatedTokens tokens, threshold=$tokenThreshold (reserve=$reserveForResponse), max=$maxTokens")
            
            if (estimatedTokens > tokenThreshold) {
                Log.w("ChatViewModel", "Token usage approaching limit ($estimatedTokens > $tokenThreshold), recommending session reset")
                return true
            }
            
        } catch (e: Exception) {
            Log.w("ChatViewModel", "Error checking token usage for chat $chatId: ${e.message}")
        }
        
        return false
    }

    /**
     * Extract cache size from model URL (e.g., ekv2048 -> 2048)
     */
    private fun extractCacheSizeFromUrl(url: String): Int? {
        val ekvPattern = Regex("ekv(\\d+)")
        val match = ekvPattern.find(url)
        return match?.groupValues?.get(1)?.toIntOrNull()
    }

    fun clearAllChatsAndCreateNew(context: Context) {
        viewModelScope.launch {
            // Cancel any ongoing operations first
            generationJob?.cancel()
            generationJob = null
            messageCollectionJob?.cancel()
            messageCollectionJob = null
            
            // Store the current chat ID before clearing state
            val oldChatId = currentChatId
            
            // Clear current state IMMEDIATELY to prevent UI freezing
            currentChatId = null
            _currentChat.value = null
            _messages.value = emptyList()
            _streamingContents.value = emptyMap()
            _isLoading.value = false
            isGenerating = false
            
            // For any existing chat session, attempt a background cleanup
            // This is fire-and-forget - don't let it block the UI
            if (oldChatId != null) {
                // Launch session cleanup in background with no UI dependency
                launch(Dispatchers.IO) {
                    try {
                        // Brief wait to let any ongoing operations wind down
                        kotlinx.coroutines.delay(500)
                        
                        // Attempt session reset with a reasonable timeout
                        // If it fails, that's okay - session will be recreated as needed
                        withTimeoutOrNull(2000) { // Shorter timeout to avoid hanging
                            inferenceService.resetChatSession(oldChatId)
                            Log.d("ChatViewModel", "Successfully reset session for chat $oldChatId during clear all")
                        } ?: run {
                            Log.d("ChatViewModel", "Session reset timed out for chat $oldChatId - will recreate as needed")
                        }
                    } catch (e: Exception) {
                        Log.d("ChatViewModel", "Session reset failed for chat $oldChatId: ${e.message} - will recreate as needed")
                        // This is expected for some models/states - just log and continue
                    }
                }
            }
            
            // Clear all chats from database
            repository.deleteAllChats()
            
            // Wait a moment to ensure database operations complete
            kotlinx.coroutines.delay(100)
            
            // Create a new empty chat immediately (don't wait for session reset)
            initializeNewChat(context)
        }
    }
    
    private suspend fun initializeNewChat(context: Context, creatorId: String? = null) {
        // Store the current model before clearing state
        val previousModel = currentModel
        
        // Load available models first
        loadAvailableModelsSync(context)
        
        // Determine which model to use - prefer the previously loaded model
        val modelToUse = when {
            // First priority: use the previously loaded model if it's still available
            previousModel != null && _availableModels.value.any { it.name == previousModel.name } -> {
                Log.d("ChatViewModel", "Reusing previously loaded model: ${previousModel.name}")
                previousModel
            }
            // Second priority: try to get currently loaded model from inference service
            else -> {
                try {
                    val currentModel = inferenceService.getCurrentlyLoadedModel()
                    if (currentModel != null && _availableModels.value.any { it.name == currentModel.name }) {
                        Log.d("ChatViewModel", "Using currently loaded model from service: ${currentModel.name}")
                        currentModel
                    } else {
                        Log.d("ChatViewModel", "No valid loaded model, using first available")
                        _availableModels.value.firstOrNull()
                    }
                } catch (e: Exception) {
                    Log.d("ChatViewModel", "Error getting loaded model, using first available: ${e.message}")
                    _availableModels.value.firstOrNull()
                }
            }
        }
        
        // Handle creator logic
        var chatTitle = context.getString(R.string.drawer_new_chat)
        if (creatorId != null) {
            val creator = repository.getCreatorById(creatorId)
            if (creator != null) {
                _currentCreator.value = creator
                chatTitle = "${creator.icon} ${creator.name}"
            } else {
                _currentCreator.value = null
            }
        } else {
            _currentCreator.value = null
        }
        
        // Create new chat with appropriate model
        val newChatId = repository.createNewChat(
            chatTitle,
            if (_availableModels.value.isEmpty()) context.getString(R.string.no_model_downloaded) else 
            (modelToUse?.name ?: context.getString(R.string.no_model_selected)),
            creatorId
        )
        
        // Set as current chat
        currentChatId = newChatId
        _currentChat.value = repository.getChatById(newChatId)
        
        Log.d("ChatViewModel", "Set new chat ID: $newChatId, chat exists: ${_currentChat.value != null}")
        
        // Clear any transient streaming state and proactively reset the session so no old context leaks
        _streamingContents.value = emptyMap()
        try {
            inferenceService.resetChatSession(newChatId)
            Log.d("ChatViewModel", "Proactively reset session for lazily created new chat $newChatId")
        } catch (e: Exception) {
            Log.w("ChatViewModel", "Unable to reset session for lazily created new chat: ${e.message}")
        }
        
        // Set the current model in the ViewModel
        if (modelToUse != null) {
            this.currentModel = modelToUse
            repository.updateChatModel(newChatId, modelToUse.name)
            _currentChat.value = repository.getChatById(newChatId)
            
            // Sync the currently loaded model state - this should be quick
            syncCurrentlyLoadedModel()
            
            // Don't auto-load the model for new chats - let user decide when to load it
            Log.d("ChatViewModel", "Set model ${modelToUse.name} for new chat but didn't auto-load it")
        }
        
        // Start collecting messages for the new chat
        messageCollectionJob?.cancel()
        messageCollectionJob = viewModelScope.launch {
            repository.getMessagesForChat(newChatId).collectLatest { messageList ->
                _messages.value = messageList
            }
        }
        
        Log.d("ChatViewModel", "Created new chat $newChatId after clearing all chats with model: ${modelToUse?.name}")
    }

    private fun resetChatSession(chatId: String) {
        viewModelScope.launch {
            try {
                Log.d("ChatViewModel", "Resetting chat session for chat $chatId")
                
                // Cancel any ongoing generation first
                generationJob?.cancel()
                
                // Check if the current model supports vision and the chat has images
                // If there's a model mismatch, skip session reset to prevent JNI errors
                val currentlyLoadedModel = inferenceService.getCurrentlyLoadedModel()
                val chatHasImages = chatContainsImages(chatId)
                
                if (chatHasImages && currentlyLoadedModel?.supportsVision != true) {
                    Log.w("ChatViewModel", "Skipping session reset for chat with images when text-only model is loaded to prevent crashes")
                    return@launch
                }
                
                // Use the reset method which handles MediaPipe session errors
                inferenceService.resetChatSession(chatId)
                lastSessionResetAt = System.currentTimeMillis()
                
                // Give MediaPipe significantly more time to clean up properly
                delay(600)
                
                Log.d("ChatViewModel", "Successfully reset chat session for chat $chatId")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Error during session reset for chat $chatId: ${e.message}")
                // Don't rethrow - session errors shouldn't crash the app
            }
        }
    }

    /**
     * Extract images from message attachments for multimodal model support
     */
    private suspend fun extractImagesFromAttachments(context: Context, messages: List<MessageEntity>): List<Bitmap> {
        val images = mutableListOf<Bitmap>()
        
        Log.d("ChatViewModel", "Extracting images from ${messages.size} messages")
        
        for (message in messages.reversed()) { // Process from newest to oldest
            Log.d("ChatViewModel", "Message ${message.id}: attachmentPath=${message.attachmentPath}, attachmentType=${message.attachmentType}")
            
            if (message.attachmentPath != null && message.attachmentType == "image") {
                try {
                    Log.d("ChatViewModel", "Attempting to load image from: ${message.attachmentPath}")
                    val uri = Uri.parse(message.attachmentPath)
                    val bitmap = loadImageFromUri(context, uri)
                    if (bitmap != null) {
                        images.add(bitmap)
                        Log.d("ChatViewModel", "Successfully loaded image from attachment: ${message.attachmentPath} (${bitmap.width}x${bitmap.height})")
                    } else {
                        Log.w("ChatViewModel", "Failed to load bitmap from URI: ${message.attachmentPath}")
                    }
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Failed to load image from attachment: ${message.attachmentPath}", e)
                }
            }
        }
        
        Log.d("ChatViewModel", "Total images extracted: ${images.size}")
        return images
    }

    /**
     * Copy an image from external URI to internal storage to ensure persistence
     */
    private suspend fun copyImageToInternalStorage(context: Context, sourceUri: Uri): Uri {
        return withContext(kotlinx.coroutines.Dispatchers.IO) {
            try {
                val contentResolver = context.contentResolver
                val inputStream = contentResolver.openInputStream(sourceUri)
                
                if (inputStream != null) {
                    // Create a unique filename
                    val timestamp = System.currentTimeMillis()
                    val fileName = "image_${timestamp}.jpg"
                    
                    // Create internal storage directory for images
                    val imagesDir = File(context.filesDir, "images")
                    if (!imagesDir.exists()) {
                        imagesDir.mkdirs()
                    }
                    
                    val outputFile = File(imagesDir, fileName)
                    
                    inputStream.use { input ->
                        outputFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                    
                    Log.d("ChatViewModel", "Copied image to: ${outputFile.absolutePath}")
                    Uri.fromFile(outputFile)
                } else {
                    Log.w("ChatViewModel", "Failed to open input stream for URI: $sourceUri")
                    sourceUri // Return original URI as fallback
                }
            } catch (e: Exception) {
                Log.e("ChatViewModel", "Failed to copy image to internal storage", e)
                sourceUri // Return original URI as fallback
            }
        }
    }

    /**
     * Load a bitmap from a URI with better error handling
     */
    private suspend fun loadImageFromUri(context: Context, uri: Uri): Bitmap? {
        return withContext(kotlinx.coroutines.Dispatchers.IO) {
            try {
                Log.d("ChatViewModel", "Opening input stream for URI: $uri")
                
                // Get content resolver and open input stream
                val contentResolver = context.contentResolver
                val inputStream = contentResolver.openInputStream(uri)
                
                if (inputStream != null) {
                    Log.d("ChatViewModel", "Input stream opened successfully")
                    inputStream.use { stream ->
                        val bitmap = BitmapFactory.decodeStream(stream)
                        if (bitmap != null) {
                            Log.d("ChatViewModel", "Bitmap decoded successfully: ${bitmap.width}x${bitmap.height}")
                            bitmap
                        } else {
                            Log.w("ChatViewModel", "BitmapFactory.decodeStream returned null")
                            null
                        }
                    }
                } else {
                    Log.w("ChatViewModel", "Failed to open input stream for URI: $uri")
                    null
                }
            } catch (e: SecurityException) {
                Log.e("ChatViewModel", "Security exception accessing URI: $uri", e)
                null
            } catch (e: Exception) {
                Log.e("ChatViewModel", "Failed to load image from URI: $uri", e)
                null
            }
        }
    }

    fun setCurrentModel(model: LLMModel) {
        currentModel = model
        savedStateHandle[KEY_CURRENT_MODEL_NAME] = model.name
    }

    fun clearMessagesForChat(chatId: String) {
        viewModelScope.launch {
            repository.clearMessagesForChat(chatId)
            ragServiceManager.clearChatDocuments(chatId) // Clear RAG documents too
            if (chatId == currentChatId) {
                _messages.value = emptyList()
            }
        }
    }

    fun clearAllChats() {
        viewModelScope.launch {
            repository.deleteAllChats()
            // Clear all RAG documents across all chats
            try {
                // Since we don't have a clearAll method, we'd need to track chatIds
                // For now, just cleanup and reinitialize RAG service
                ragServiceManager.cleanup()
                ragServiceManager.initializeAsync()
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Failed to clear RAG documents: ${e.message}")
            }
            _messages.value = emptyList()
            _currentChat.value = null
            currentChatId = null
        }
    }

    /**
     * Regenerate a specific AI response by finding the previous user message and re-generating
     */
    fun regenerateResponse(context: Context, messageId: String) {
        val chatId = currentChatId ?: return
        
        viewModelScope.launch {
            try {
                // Find the message to regenerate
                val currentMessages = _messages.value
                val messageToRegenerate = currentMessages.find { it.id == messageId }
                
                if (messageToRegenerate == null || messageToRegenerate.isFromUser) {
                    Log.w("ChatViewModel", "Cannot regenerate user message or message not found: $messageId")
                    return@launch
                }
                
                // If the message was a "no response" sentinel, force a fresh context (forget history for this regen)
                val forceFreshContext = messageToRegenerate.content.contains("No response produced.")

                // Find the user message that prompted this response
                val messageIndex = currentMessages.indexOf(messageToRegenerate)
                var userMessage = if (messageIndex > 0) {
                    // Look for the previous user message
                    currentMessages.subList(0, messageIndex).lastOrNull { it.isFromUser }
                } else null
                
                if (userMessage == null) {
                    Log.w("ChatViewModel", "Cannot find user message that prompted response: $messageId")
                    return@launch
                }
                
                // Check if we have a valid model
                if (currentModel == null || !currentModel!!.isDownloaded) {
                    Log.e("ChatViewModel", "No valid model available for regeneration")
                    return@launch
                }
                
                // Check if we're still loading or switching models
                if (_isLoadingModel.value || firstGenerationSinceLoad) {
                    Log.w("ChatViewModel", "Cannot regenerate while model is loading or switching")
                    repository.addMessage(chatId, "Please wait for the model to finish loading before regenerating.", isFromUser = false)
                    return@launch
                }
                
                // If there is a newer user message (typed after this assistant reply), regenerate for the latest user turn
                val latestUser = currentMessages.lastOrNull { it.isFromUser }
                if (latestUser != null && userMessage != null && latestUser.timestamp > userMessage.timestamp) {
                    Log.d("ChatViewModel", "Regenerate targets newer user turn; switching from ${userMessage.id} -> ${latestUser.id}")
                    userMessage = latestUser
                }

                // Delete the current AI response and any messages after the target user message
                val messagesToKeep = currentMessages.takeWhile { it.id != (userMessage?.id ?: messageId) }
                
                // Clear messages after the user message from database
                repository.deleteMessagesAfter(chatId, userMessage.timestamp)
                
                // Update the messages state immediately
                _messages.value = messagesToKeep
                
                // Mark as loading
                _isLoading.value = true
                isGenerating = true
                
                // Ensure the model is loaded
                try {
                    inferenceService.loadModel(currentModel!!, _selectedBackend.value)
                    // Sync the currently loaded model state
                    syncCurrentlyLoadedModel()
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Failed to ensure model is loaded for regeneration: ${e.message}")
                    repository.addMessage(chatId, "Failed to load model. Please try again.", isFromUser = false)
                    _isLoading.value = false
                    isGenerating = false
                    return@launch
                }
                
                // Reset the session to provide clean context for regeneration
                try {
                    // Give more time to ensure any previous operations have completed
                    delay(300)
                    inferenceService.resetChatSession(chatId)
                    // Record the reset and force-drop history for the next prompt
                    lastSessionResetAt = System.currentTimeMillis()
                    dropHistoryOnce = true
                    delay(200) // Additional time for session to stabilize
                    Log.d("ChatViewModel", "Reset session for regeneration")
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Error resetting session for regeneration: ${e.message}")
                    // If session reset fails, try to force recreate the session
                    try {
                        delay(200)
                        inferenceService.onCleared()
                        delay(500)
                        inferenceService.loadModel(currentModel!!, _selectedBackend.value)
                        delay(200)
                        Log.d("ChatViewModel", "Force recreated session for regeneration after reset failure")
                    } catch (recreateException: Exception) {
                        Log.e("ChatViewModel", "Force recreation for regeneration also failed: ${recreateException.message}")
                        repository.addMessage(chatId, "Failed to prepare session for regeneration. Please try switching models or restarting the app.", isFromUser = false)
                        _isLoading.value = false
                        isGenerating = false
                        return@launch
                    }
                }
                
                // Build conversation history up to the user message (excluding the user message itself)
                val history = if (forceFreshContext) {
                    Log.d("ChatViewModel", "Force fresh context for regeneration (dropping all prior history)")
                    ""
                } else {
                    val historyMessages = messagesToKeep.filter { it.id != userMessage.id }
                    buildContextAwareHistory(historyMessages)
                }
                
                // Extract images and handle documents if the user message has attachments
                val images = if (currentModel!!.supportsVision && userMessage.attachmentPath != null) {
                    when (userMessage.attachmentType) {
                        "image" -> {
                            try {
                                val bitmap = loadImageFromUri(context, Uri.parse(userMessage.attachmentPath))
                                if (bitmap != null) {
                                    Log.d("ChatViewModel", "Loaded image for regeneration: ${bitmap.width}x${bitmap.height}")
                                    listOf(bitmap)
                                } else {
                                    emptyList()
                                }
                            } catch (e: Exception) {
                                Log.w("ChatViewModel", "Failed to load image for regeneration: ${e.message}")
                                emptyList()
                            }
                        }
                        else -> {
                            // For documents and other attachments with vision models
                            Log.d("ChatViewModel", "Vision model regenerating with document attachment: ${userMessage.attachmentType}")
                            emptyList()  // No bitmap needed, but vision model can still process document context
                        }
                    }
                } else {
                    emptyList()
                }
                
                // Create a new response placeholder
                val placeholderId = repository.addMessage(chatId, "…", isFromUser = false)
                _streamingContents.value = mapOf(placeholderId to "")
                
                // Generate new response
                generationJob = launch {
                    val generationStartTime = System.currentTimeMillis()
                    var ragSearchTimeMs = 0L
                    var totalContent = ""
                    var firstChunkAt = 0L
                    var lastChunkAt = 0L
                    
                    try {
                        // Build the full history including the user message at the end
                        val fullHistory = if (history.isNotEmpty()) {
                            "$history\n\nuser: ${userMessage.content}\nassistant:"
                        } else {
                            "user: ${userMessage.content}\nassistant:"
                        }
                        
                        Log.d("ChatViewModel", "Regeneration prompt for chat $chatId: '${fullHistory.take(100)}${if (fullHistory.length > 100) "..." else ""}'")
                        
                        // Ensure the model is freshly loaded before regeneration
                        _isLoadingModel.value = true
                        inferenceService.loadModel(currentModel!!, _selectedBackend.value)
                        _isLoadingModel.value = false
                        
                        // Get web search preference
                        val webSearchEnabled = runBlocking { themePreferences.webSearchEnabled.first() }
                        
                        val responseStream = inferenceService.generateResponseStreamWithSession(
                            fullHistory, 
                            currentModel!!, 
                            chatId, 
                            images,
                            null, // No audio for regeneration - only used for new messages
                            webSearchEnabled
                        )
                        
                        var lastUpdateTime = 0L
                        val updateIntervalMs = 50L
                        
                        // Check if auto-readout is enabled before starting generation
                        val autoReadoutEnabled = try {
                            themePreferences.autoReadoutEnabled.first()
                        } catch (e: Exception) {
                            false
                        }
                        
                        // Initialize TTS for streaming if auto-readout is enabled
                        if (autoReadoutEnabled) {
                            Log.d("ChatViewModel", "Auto-readout enabled for regeneration, preparing TTS for streaming message: $placeholderId")
                            // Stop any previous TTS before starting new generation
                            ttsService.stop()
                            // Set language to app's current locale
                            val appLocale = com.llmhub.llmhub.utils.LocaleHelper.getCurrentLocale(context)
                            ttsService.setLanguage(appLocale)
                            // Set the current TTS message ID so the UI can show stop icon
                            _currentTtsMessageId.value = placeholderId
                        }
                        
                        responseStream.collect { piece ->
                            val nowTs = System.currentTimeMillis()
                            if (piece.isNotEmpty() && firstChunkAt == 0L) firstChunkAt = nowTs
                            lastChunkAt = nowTs
                            totalContent += piece
                            
                            // Auto-readout: Stream text to TTS as it arrives (for regeneration)
                            if (autoReadoutEnabled && _currentTtsMessageId.value == placeholderId && piece.isNotEmpty()) {
                                try {
                                    ttsService.addStreamingText(piece)
                                } catch (e: Exception) {
                                    Log.w("ChatViewModel", "Failed to add text to TTS stream during regeneration: ${e.message}")
                                }
                            }
                            
                            // Note: Repetition detection is handled at the InferenceService layer
                            // Note: No artificial length limit - let the model's natural token limit handle it
                            
                            // Update UI
                            val updated = _streamingContents.value.toMutableMap()
                            updated[placeholderId] = totalContent
                            _streamingContents.value = updated
                            
                            // Debounced database updates
                            val currentTime = System.currentTimeMillis()
                            if (currentTime - lastUpdateTime > updateIntervalMs) {
                                repository.updateMessageContent(placeholderId, totalContent.trimEnd())
                                lastUpdateTime = currentTime
                            }
                        }
                        
                        // Success - finalize the message
                        val finalContent = totalContent
                        repository.updateMessageContent(placeholderId, finalContent.trimEnd())
                        val time = System.currentTimeMillis() - generationStartTime
                        val netTime = (time - ragSearchTimeMs).coerceAtLeast(1L)
                        val streamDurationMs = ((if (lastChunkAt > 0) lastChunkAt else System.currentTimeMillis()) - (if (firstChunkAt > 0) firstChunkAt else generationStartTime)).coerceAtLeast(1L)
                        Log.d("ChatViewModel", "About to call finalizeMessage for regeneration (raw=${time}ms, rag=${ragSearchTimeMs}ms, net=${netTime}ms, stream=${streamDurationMs}ms)")
                        finalizeMessage(placeholderId, finalContent, streamDurationMs)
                        
                        Log.d("ChatViewModel", "Regeneration completed successfully")
                        
                    } catch (e: Exception) {
                        val finalContent = totalContent
                        val time = System.currentTimeMillis() - generationStartTime
                        val netTime = (time - ragSearchTimeMs).coerceAtLeast(1L)

                        Log.d("ChatViewModel", "Regeneration exception: ${e.javaClass.simpleName}: ${e.message}")

                        // Save partial content and finalize
                        withContext(kotlinx.coroutines.NonCancellable) {
                            repository.updateMessageContent(placeholderId, finalContent.trimEnd())
                            Log.d("ChatViewModel", "About to call finalizeMessage for regeneration (exception) (raw=${time}ms, rag=${ragSearchTimeMs}ms, net=${netTime}ms)")
                            val streamDurationMs = ((if (lastChunkAt > 0) lastChunkAt else System.currentTimeMillis()) - (if (firstChunkAt > 0) firstChunkAt else generationStartTime)).coerceAtLeast(1L)
                            finalizeMessage(placeholderId, finalContent, streamDurationMs)
                        }
                        
                    } finally {
                        _isLoading.value = false
                        isGenerating = false
                        
                        // Clear streaming state
                        val updatedStreaming = _streamingContents.value.toMutableMap()
                        updatedStreaming.remove(placeholderId)
                        _streamingContents.value = updatedStreaming
                    }
                }
                
            } catch (e: Exception) {
                Log.e("ChatViewModel", "Error during regeneration setup: ${e.message}", e)
                _isLoading.value = false
                isGenerating = false
            }
        }
    }

    /**
     * Edit the last user message content and resend from that point, replacing subsequent messages.
     */
    fun editLastUserMessageAndResend(context: Context, newUserText: String) {
        val chatId = currentChatId ?: return
        val trimmed = newUserText.trim()
        if (trimmed.isEmpty()) return

        viewModelScope.launch {
            val currentMessages = _messages.value
            val lastUser = currentMessages.lastOrNull { it.isFromUser } ?: return@launch

            // Prevent while loading/switching
            if (_isLoadingModel.value || firstGenerationSinceLoad) {
                repository.addMessage(chatId, "Please wait for the model to finish loading before editing.", isFromUser = false)
                return@launch
            }

            // Update message content in DB
            repository.updateMessageContent(lastUser.id, trimmed)

            // Delete all messages after the edited user message
            repository.deleteMessagesAfter(chatId, lastUser.timestamp)

            // Update local state to reflect truncation
            val kept = currentMessages.takeWhile { it.id != lastUser.id } + lastUser.copy(content = trimmed)
            _messages.value = kept

            // Ensure model is ready
            try {
                inferenceService.loadModel(currentModel!!, _selectedBackend.value)
                syncCurrentlyLoadedModel()
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Failed to ensure model is loaded for edit+resend: ${e.message}")
                repository.addMessage(chatId, "Failed to load model. Please try again.", isFromUser = false)
                return@launch
            }

            // Reset session for clean context
            try {
                delay(300)
                inferenceService.resetChatSession(chatId)
                delay(200)
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Error resetting session for edit+resend: ${e.message}")
                try {
                    delay(200)
                    inferenceService.onCleared()
                    delay(500)
                    inferenceService.loadModel(currentModel!!, _selectedBackend.value)
                    delay(200)
                } catch (re: Exception) {
                    repository.addMessage(chatId, "Failed to prepare session. Try switching models.", isFromUser = false)
                    return@launch
                }
            }

            _isLoading.value = true
            isGenerating = true

            // Build history up to but excluding the edited user message
            val historyMessages = kept.filter { it.id != lastUser.id }
            val history = buildContextAwareHistory(historyMessages)

            // Create a new response placeholder
            val placeholderId = repository.addMessage(chatId, "…", isFromUser = false)
            _streamingContents.value = mapOf(placeholderId to "")

            // Start generation with edited prompt
            generationJob = launch {
                val generationStartTime = System.currentTimeMillis()
                var totalContent = ""
                var firstChunkAt = 0L
                var lastChunkAt = 0L
                try {
                    val fullHistory = if (history.isNotEmpty()) {
                        "$history\n\nuser: ${trimmed}\nassistant:"
                    } else {
                        "user: ${trimmed}\nassistant:"
                    }

                    _isLoadingModel.value = true
                    inferenceService.loadModel(currentModel!!, _selectedBackend.value)
                    _isLoadingModel.value = false

                    val webSearchEnabled = runBlocking { themePreferences.webSearchEnabled.first() }

                    // For now, do not include images/audio during edit-resend; extend if needed
                    val images: List<android.graphics.Bitmap> = emptyList()
                    val responseStream = inferenceService.generateResponseStreamWithSession(
                        fullHistory,
                        currentModel!!,
                        chatId,
                        images,
                        null,
                        webSearchEnabled
                    )

                    responseStream.collect { chunk ->
                        val nowTs = System.currentTimeMillis()
                        if (chunk.isNotEmpty() && firstChunkAt == 0L) firstChunkAt = nowTs
                        lastChunkAt = nowTs
                        totalContent += chunk
                        _streamingContents.value = _streamingContents.value + (placeholderId to totalContent)
                    }

                    // Finalize message
                    repository.updateMessageContent(placeholderId, totalContent)
                    _streamingContents.value = emptyMap()

                    // Stats based on streaming window
                    val streamDurationMs = ((if (lastChunkAt > 0) lastChunkAt else System.currentTimeMillis()) - (if (firstChunkAt > 0) firstChunkAt else generationStartTime)).coerceAtLeast(1L)
                    finalizeMessage(placeholderId, totalContent, streamDurationMs)
                } catch (e: CancellationException) {
                    Log.i("ChatViewModel", "Generation cancelled: ${e.message}")
                } catch (e: Exception) {
                    Log.e("ChatViewModel", "Error during generation (edit+resend): ${e.message}", e)
                    repository.updateMessageContent(placeholderId, "Generation failed: ${e.message ?: "Unknown error"}")
                } finally {
                    _isLoading.value = false
                    isGenerating = false
                }
            }
        }
    }
}