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
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.data.isModelFileValid

class ChatViewModel(
    private val inferenceService: InferenceService,
    private val repository: ChatRepository,
    private val savedStateHandle: SavedStateHandle = SavedStateHandle()
) : ViewModel() {

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

    var currentModel: LLMModel? = null
        private set

    private var isGenerating = false
        get() = savedStateHandle.get<Boolean>(KEY_IS_GENERATING) ?: false
        set(value) {
            field = value
            savedStateHandle.set(KEY_IS_GENERATING, value)
        }

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

    private fun syncCurrentlyLoadedModel() {
        viewModelScope.launch {
            try {
                val loadedModel = inferenceService.getCurrentlyLoadedModel()
                _currentlyLoadedModel.value = loadedModel
                Log.d("ChatViewModel", "Synced currently loaded model: ${loadedModel?.name ?: "None"}")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Error syncing currently loaded model: ${e.message}")
                // Don't update the state if there's an error
            }
        }
    }

    fun initializeChat(chatId: String, context: Context) {
        // Sync the currently loaded model from inference service
        syncCurrentlyLoadedModel()
        
        // Stop collecting from any previous chat's message flow
        generationJob?.cancel()
        messageCollectionJob?.cancel()

        // Close previous chat session if switching chats
        if (currentChatId != null && currentChatId != chatId) {
            Log.d("ChatViewModel", "Switching from chat ${currentChatId} to chat $chatId")
            val previousChatId = currentChatId!!
            
            // Cancel any ongoing generation before switching
            generationJob?.cancel()
            generationJob = null
            
            // Clear any streaming state
            _streamingContents.value = emptyMap()
            
            // Reset chat session synchronously to prevent session conflicts
            try {
                runBlocking {
                    inferenceService.resetChatSession(previousChatId)
                }
                Log.d("ChatViewModel", "Completed session cleanup for chat switch")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Error during session cleanup: ${e.message}")
            }
        }

        // Remember the previously loaded model to preserve it across chat switches
        // Use the inference service's currently loaded model as the source of truth
        val previousModel = inferenceService.getCurrentlyLoadedModel()
        Log.d("ChatViewModel", "Current model before switch: ${previousModel?.name ?: "None"}")

        viewModelScope.launch {
            // Load the models synchronously so we know what's available before creating/attaching a chat
            loadAvailableModelsSync(context)

            if (chatId == "new") {
                // For new chats, preserve the current model if one is loaded
                val newChatId = repository.createNewChat(
                    "New Chat",
                    if (_availableModels.value.isEmpty()) "No model downloaded" else 
                    (previousModel?.name ?: "No model selected")
                )
                currentChatId = newChatId
                _currentChat.value = repository.getChatById(newChatId)
                
                // Ensure a fresh inference session for the brand new chat to avoid stale context
                try {
                    inferenceService.resetChatSession(newChatId)
                    Log.d("ChatViewModel", "Proactively reset session for new chat $newChatId")
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Unable to reset session for new chat: ${e.message}")
                }
                
                // Preserve the current model for new chats
                currentModel = previousModel
                
                // If we have a model, update the chat to use it
                if (previousModel != null) {
                    repository.updateChatModel(newChatId, previousModel.name)
                    _currentChat.value = repository.getChatById(newChatId)
                }

                // Begin collecting messages for the newly created chat
                messageCollectionJob = launch {
                    repository.getMessagesForChat(newChatId).collectLatest { messageList ->
                        _messages.value = messageList
                    }
                }
            } else {
                // Check if the chat still exists
                val chat = repository.getChatById(chatId)
                if (chat == null) {
                    Log.e("ChatViewModel", "Chat $chatId does not exist, creating new chat instead")
                    initializeChat("new", context)
                    return@launch
                }
                
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
                            content = "🚫 **Cannot Open This Chat**\n\nThis conversation contains images, but you currently have a **text-only model** loaded:\n\n📱 **Current Model:** ${currentlyLoadedModel.name}\n🖼️ **This Chat:** Contains images\n\n---\n\n## 🎯 How to Fix This:\n\n1. **📥 Download a Vision Model**\n   - Go to \"Download Models\"\n   - Look for models marked with 🖼️ or \"Vision+Text\"\n   - Try: **Gemma-3n E2B (Vision+Text)**\n\n2. **🔄 Switch to Vision Model**\n   - Load the vision model you downloaded\n   - Vision models can handle both text AND images\n\n3. **✅ Return to This Chat**\n   - Once a vision model is loaded, this chat will work perfectly!\n\n---\n\n💡 **Pro Tip:** Vision models can do everything text models can do, plus handle images!",
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
                val foundModel = _availableModels.value.find { it.name == chat.modelName }
                    ?: ModelData.models.find { it.name == chat.modelName }
                
                if (foundModel != null && foundModel.isDownloaded) {
                    // Use the model associated with this chat
                    Log.d("ChatViewModel", "Using chat-specific model: ${foundModel.name}")
                    currentModel = foundModel
                    // Ensure the model is loaded in the inference service
                    viewModelScope.launch {
                        try {
                            inferenceService.loadModel(foundModel)
                            // Sync the currently loaded model state
                            syncCurrentlyLoadedModel()
                        } catch (e: Exception) {
                            Log.w("ChatViewModel", "Failed to load model for chat: ${e.message}")
                        }
                    }
                } else if (previousModel != null && previousModel.isDownloaded) {
                    // Fallback to the previously loaded model and assign it to this chat
                    Log.d("ChatViewModel", "Using previous model for chat: ${previousModel.name}")
                    currentModel = previousModel
                    repository.updateChatModel(chatId, previousModel.name)
                    _currentChat.value = repository.getChatById(chatId)
                    // Ensure the model is loaded in the inference service
                    viewModelScope.launch {
                        try {
                            inferenceService.loadModel(previousModel)
                            // Sync the currently loaded model state
                            syncCurrentlyLoadedModel()
                        } catch (e: Exception) {
                            Log.w("ChatViewModel", "Failed to load previous model for chat: ${e.message}")
                        }
                    }
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
        val downloadedModels = ModelData.models.mapNotNull { model ->
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
                val primaryFile = File(modelsDir, model.localFileName())
                val legacyFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")

                // Migrate legacy file if needed
                if (!primaryFile.exists() && legacyFile.exists()) {
                    legacyFile.renameTo(primaryFile)
                }

                if (primaryFile.exists()) {
                    // Only treat as available if passes integrity checks and size is close to expected when known
                    val sizeKnown = model.sizeBytes > 0
                    val sizeOk = if (sizeKnown) primaryFile.length() >= (model.sizeBytes * 0.98).toLong() else primaryFile.length() >= 10L * 1024 * 1024
                    val valid = isModelFileValid(primaryFile, model.modelFormat)
                    if (sizeOk && valid) {
                        isAvailable = true
                        actualSize = primaryFile.length()
                        Log.d("ChatViewModel", "Found VALID model in files: ${primaryFile.absolutePath} (${actualSize / (1024*1024)} MB)")
                    } else {
                        Log.d("ChatViewModel", "Ignoring incomplete/invalid model file: ${primaryFile.absolutePath} sizeOk=$sizeOk valid=$valid")
                    }
                }
            }

            if (isAvailable) {
                model.copy(isDownloaded = true, sizeBytes = actualSize)
            } else {
                null
            }
        }

        _availableModels.value = downloadedModels
        
        // If we have a current model but it's not in the available models, clear it
        if (currentModel != null && !downloadedModels.any { it.name == currentModel?.name }) {
            Log.d("ChatViewModel", "Current model ${currentModel?.name} is no longer available")
            currentModel = null
        }
    }

    fun sendMessage(context: Context, text: String, attachmentUri: Uri?) {
        val chatId = currentChatId
        if (chatId == null) {
            Log.e("ChatViewModel", "No current chat ID available, creating new chat")
            // If no current chat, create a new one
            viewModelScope.launch {
                initializeNewChat(context)
                // Retry sending the message after creating a new chat
                kotlinx.coroutines.delay(100) // Small delay to ensure chat is created
                sendMessage(context, text, attachmentUri)
            }
            return
        }
        val messageText = text.trim()

        if (messageText.isEmpty() && attachmentUri == null) return

        viewModelScope.launch {
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
                    "Please download a model to start chatting."
                } else {
                    "Model not properly loaded. Please try switching to a different model or restart the app."
                }
                repository.addMessage(chatId, errorMessage, isFromUser = false)
                return@launch
            }

            // Process attachment if present
            var processedAttachmentUri: Uri? = null
            if (attachmentUri != null) {
                // Check if current model supports vision before processing image
                if (currentModel != null && !currentModel!!.supportsVision) {
                    Log.w("ChatViewModel", "Cannot send image with text-only model: ${currentModel!!.name}")
                    val errorMessage = "🖼️ **Cannot Send Images**\n\nYou're trying to send an image, but the current model is **text-only**:\n\n📱 **Current Model:** ${currentModel!!.name}\n\n---\n\n## 🎯 To Send Images:\n\n1. **📥 Download a Vision Model**\n   - Tap the menu → \"Download Models\"\n   - Look for 🖼️ **Vision+Text** models\n   - Recommended: **Gemma-3n E2B (Vision+Text)**\n\n2. **🔄 Switch Models**\n   - Load the vision model you downloaded\n   - Vision models handle both text AND images\n\n3. **✅ Try Again**\n   - Once loaded, you can send images freely!\n\n---\n\n💡 **Pro Tip:** Vision models can do everything text models can do, plus understand images!"
                    repository.addMessage(chatId, errorMessage, isFromUser = false)
                    _isLoading.value = false
                    isGenerating = false
                    return@launch
                }
                
                try {
                    // Copy the image to internal storage to ensure it persists
                    processedAttachmentUri = copyImageToInternalStorage(context, attachmentUri)
                    Log.d("ChatViewModel", "Copied image to internal storage: $processedAttachmentUri")
                } catch (e: Exception) {
                    Log.e("ChatViewModel", "Failed to copy image to internal storage", e)
                    // Fall back to original URI
                    processedAttachmentUri = attachmentUri
                }
            }

            repository.addMessage(
                chatId = chatId,
                content = if (messageText.isNotEmpty()) messageText else "Shared a file",
                isFromUser = true,
                attachmentPath = processedAttachmentUri?.toString(),
                attachmentType = determineAttachmentType(processedAttachmentUri)
            )

            // The first message sets the title
            if (_messages.value.size == 1) {
                repository.updateChatTitle(chatId, messageText.take(50))
                _currentChat.value = repository.getChatById(chatId)
            }

            _isLoading.value = true
            isGenerating = true

            if (currentModel != null && currentModel!!.isDownloaded) {
                // Ensure the model is loaded in the inference service before generating
                try {
                    _isLoadingModel.value = true
                    inferenceService.loadModel(currentModel!!)
                    // Sync the currently loaded model state
                    syncCurrentlyLoadedModel()
                    _isLoadingModel.value = false
                    Log.d("ChatViewModel", "Successfully ensured model ${currentModel!!.name} is loaded for generation")
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
                    content = if (messageText.isNotEmpty()) messageText else "Shared a file",
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
                            var totalContent = ""
                            
                            try {
                                // Pre-load model separately from generation timing
                                _isLoadingModel.value = true
                                
                                // Ensure model is loaded first
                                inferenceService.loadModel(currentModel!!)
                                
                                _isLoadingModel.value = false
                                
                                // Continue generation with context window management
                                var continuationCount = 0
                                val maxContinuations = 2 // Further reduced since we have better context management
                                
                                while (continuationCount < maxContinuations) {
                                    var currentSegment = ""
                                    val segmentStartTime = System.currentTimeMillis()
                                    
                                    // Build the prompt for this segment with context window management
                                    val currentPrompt = if (continuationCount == 0) {
                                        // First generation: use context-aware history
                                        history
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
                                        continuationHistory
                                    }
                                    
                                    // Extract images from recent messages for multimodal models
                    // Note: We only include the current message's image to avoid confusion
                    // where the model would reference previous images in new responses
                    val images = if (currentModel!!.supportsVision) {
                        Log.d("ChatViewModel", "Current model supports vision: ${currentModel!!.name}")
                        val recentMessages = _messages.value.takeLast(10)
                        Log.d("ChatViewModel", "Checking ${recentMessages.size} recent messages for images")
                        
                        // Get images from the current user message first, then from recent messages
                        val currentImages = mutableListOf<Bitmap>()
                        
                        // Check if the current message has an image attachment
                        if (processedAttachmentUri != null) {
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
                        
                        Log.d("ChatViewModel", "Total images for generation: ${currentImages.size} (current: ${if (processedAttachmentUri != null) 1 else 0}, context: ${contextImages.size} - disabled)")
                        
                        // Log details of each image
                        currentImages.forEachIndexed { index, bitmap ->
                            Log.d("ChatViewModel", "Image $index: ${bitmap.width}x${bitmap.height}, config: ${bitmap.config}")
                        }
                        
                        currentImages
                    } else {
                        Log.d("ChatViewModel", "Current model does not support vision: ${currentModel!!.name}")
                        emptyList()
                    }
                    
                    val responseStream = inferenceService.generateResponseStreamWithSession(currentPrompt, currentModel!!, chatId, images)
                                    var lastUpdateTime = 0L
                                    val updateIntervalMs = 50L // Update UI every 50ms instead of every token
                                    var segmentEnded = false
                                    var segmentHasContent = false
                                    
                                    responseStream.collect { piece ->
                                        currentSegment += piece
                                        totalContent += piece
                                        
                                        // Check if we're getting meaningful content
                                        if (piece.trim().isNotEmpty()) {
                                            segmentHasContent = true
                                        }
                                        
                                        // Real-time repetition check to stop runaway generation immediately
                                        if (totalContent.length > 200) {
                                            val isRepetitive = checkForRepetition(totalContent)
                                            if (isRepetitive) {
                                                Log.w("ChatViewModel", "Stopping generation due to repetitive content detected")
                                                // Reset the session to prevent future issues
                                                try {
                                                    inferenceService.resetChatSession(chatId)
                                                    Log.d("ChatViewModel", "Reset MediaPipe session due to repetitive content")
                                                } catch (e: Exception) {
                                                    Log.w("ChatViewModel", "Failed to reset session: ${e.message}")
                                                }
                                                // Actively cancel the running generation coroutine so the stream stops immediately
                                                throw kotlinx.coroutines.CancellationException("Repetitive content detected")
                                            }
                                        }
                                        
                                        // Maximum response length check to prevent runaway generation
                                        if (totalContent.length > 5000) {
                                            Log.w("ChatViewModel", "Stopping generation due to maximum length exceeded (${totalContent.length} chars)")
                                            // Reset the session to prevent future issues
                                            try {
                                                inferenceService.resetChatSession(chatId)
                                                Log.d("ChatViewModel", "Reset MediaPipe session due to length limit")
                                            } catch (e: Exception) {
                                                Log.w("ChatViewModel", "Failed to reset session: ${e.message}")
                                            }
                                            // Actively cancel to terminate collection immediately
                                            throw kotlinx.coroutines.CancellationException("Length limit reached")
                                        }
                                        
                                        // Update UI with the complete content so far
                                        val updated = _streamingContents.value.toMutableMap()
                                        updated[placeholderId] = totalContent
                                        _streamingContents.value = updated
                                        
                                        // Debounced database updates to reduce blinking
                                        val currentTime = System.currentTimeMillis()
                                        if (currentTime - lastUpdateTime > updateIntervalMs) {
                                            repository.updateMessageContent(placeholderId, totalContent)
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
                                val finalContent = totalContent
                                Log.d("ChatViewModel", "Generation completed successfully with ${continuationCount} continuations")
                                Log.d("ChatViewModel", "Final content length: ${finalContent.length}")
                                // Ensure the final content is saved to database before computing stats
                                repository.updateMessageContent(placeholderId, finalContent)
                                val time = System.currentTimeMillis() - generationStartTime
                                Log.d("ChatViewModel", "About to call finalizeMessage for success")
                                finalizeMessage(placeholderId, finalContent, time)

                    } catch (e: Exception) {
                        val finalContent = totalContent
                        val time = System.currentTimeMillis() - generationStartTime
                        
                        Log.d("ChatViewModel", "Exception caught: ${e.javaClass.simpleName}: ${e.message}")
                        Log.d("ChatViewModel", "Final content length: ${finalContent.length}")
                        Log.d("ChatViewModel", "Generation time: ${time}ms")
                        
                        // Handle both CancellationException and JobCancellationException (which extends CancellationException)
                        if (e is kotlinx.coroutines.CancellationException || e.javaClass.simpleName.contains("Cancellation")) {
                            // CANCEL: Save partial progress
                            Log.d("ChatViewModel", "Generation was cancelled by user.")
                            Log.d("ChatViewModel", "About to call finalizeMessage for cancellation")
                        } else {
                            // ERROR: MediaPipe or other error
                            Log.e("ChatViewModel", "MediaPipe generation error: ${e.message}", e)
                            Log.d("ChatViewModel", "About to call finalizeMessage for error")
                        }
                        
                        // ALWAYS save final content and call finalizeMessage (for both cancel and error) even if the parent Job is cancelled
                        withContext(kotlinx.coroutines.NonCancellable) {
                            repository.updateMessageContent(placeholderId, finalContent)
                            Log.d("ChatViewModel", "About to call finalizeMessage (NonCancellable)")
                            finalizeMessage(placeholderId, finalContent, time)
                        }
                    } finally {
                        _isLoading.value = false
                        _isLoadingModel.value = false
                        isGenerating = false
                        
                        // Clear streaming state for this message, regardless of outcome
                        val updatedStreaming = _streamingContents.value.toMutableMap()
                        updatedStreaming.remove(placeholderId)
                        _streamingContents.value = updatedStreaming
                    }
                }
            } else {
                repository.addMessage(chatId, "Please download a model to start chatting.", isFromUser = false)
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
            // Approximate tokenisation: empirical average ~4 characters per token across English text.
            // This gives results closer to what web AI UIs display (OpenAI, Claude, etc.).
            val estimatedTokens = kotlin.math.ceil(finalContent.length / 4.0).toInt().coerceAtLeast(1)
            val tokensPerSecond = if (generationTimeMs > 0) {
                (estimatedTokens * 1000.0) / generationTimeMs
            } else {
                0.0
            }

            Log.d("ChatViewModel", "Saving stats for message $placeholderId: $estimatedTokens tokens, ${String.format("%.1f", tokensPerSecond)} tok/sec")
            repository.updateMessageStats(placeholderId, estimatedTokens, tokensPerSecond)
        } else {
            Log.d("ChatViewModel", "No stats to save for message $placeholderId - content is blank")
        }
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
    private fun checkForRepetition(content: String): Boolean {
        if (content.length < 100) return false
        
        val trimmed = content.trim()
        
        // Check for excessive repetition of the same character (like 7777777...)
        val last50Chars = trimmed.takeLast(50)
        val charCounts = last50Chars.groupingBy { it }.eachCount()
        val mostCommonChar = charCounts.maxByOrNull { it.value }
        if (mostCommonChar != null && mostCommonChar.value > 30) {
            Log.d("ChatViewModel", "Real-time: Detected excessive character repetition: '${mostCommonChar.key}' repeated ${mostCommonChar.value} times")
            return true
        }
        
        // Check for sentence/phrase repetition (n-gram at word level)
        if (trimmed.length > 160) {
            val words = trimmed.split(Regex("\\s+")).map { it.lowercase() }
            if (words.size > 30) {
                // compare last 8-15 word shingles with the previous window
                val windowSizes = listOf(8, 10, 12, 15)
                for (n in windowSizes) {
                    if (words.size > n * 2) {
                        val tail = words.takeLast(n)
                        val before = words.dropLast(n).takeLast(n)
                        if (tail == before) {
                            Log.d("ChatViewModel", "Real-time: Detected ${n}-word repetition block")
                            return true
                        }
                    }
                }
            }
        }

        // Check for character-level repetition patterns
        if (trimmed.length > 100) {
            val last30Chars = trimmed.takeLast(30)
            val before30Chars = trimmed.dropLast(30).takeLast(30)
            if (last30Chars == before30Chars) {
                Log.d("ChatViewModel", "Real-time: Detected character-level repetition")
                return true
            }
        }
        
        // Check for URLs getting too long
        if (trimmed.contains("http") || trimmed.contains("www")) {
            val urlPattern = Regex("https?://[^\\s]+|www\\.[^\\s]+")
            val urls = urlPattern.findAll(trimmed).toList()
            for (url in urls) {
                if (url.value.length > 200) {
                    Log.d("ChatViewModel", "Real-time: Detected excessively long URL pattern")
                    return true
                }
            }
        }
        
        return false
    }

    /** Interrupt the current response generation if one is running */
    fun stopGeneration() {
        generationJob?.cancel()
        generationJob = null
        _isLoading.value = false
        _isLoadingModel.value = false
        isGenerating = false
        // Note: Don't clear streaming contents here to preserve partial responses
        // _tokenStats.value = null // No longer needed
    }

    fun switchModel(newModel: LLMModel) {
        viewModelScope.launch {
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
                inferenceService.loadModel(newModel)
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
                
                // Reset the current chat session to ensure it works with the new model
                currentChatId?.let { chatId ->
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
                            inferenceService.loadModel(newModel)
                            Log.d("ChatViewModel", "Force recreated session after reset failure")
                        } catch (recreateException: Exception) {
                            Log.w("ChatViewModel", "Force recreation also failed: ${recreateException.message}")
                        }
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

    override fun onCleared() {
        viewModelScope.launch {
            // Clean up resources
            currentChatId?.let { chatId ->
                // Session cleanup is now handled by the inference service
                Log.d("ChatViewModel", "Chat session cleanup delegated to inference service")
            }
            inferenceService.onCleared()
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
        return currentModel?.supportsVision == true
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
            uriString.endsWith(".gif") || uriString.endsWith(".webp") -> "image"
            else -> "file"
        }
    }

    /**
     * Build context-aware history that respects the model's context window limits.
     * This implements conversation-pair-aware truncation to maintain context flow.
     */
    private fun buildContextAwareHistory(messages: List<MessageEntity>): String {
        val model = currentModel ?: return ""
        val currentChatId = currentChatId ?: return ""
        
        // Filter messages to only include current chat messages
        val chatMessages = messages.filter { it.chatId == currentChatId }
        
        // Use the model's actual context window size with some safety margin
        val maxContextTokens = (model.contextWindowSize * 0.75).toInt() // Use 75% of max to leave room for response
        val maxContextChars = maxContextTokens * 4 // Rough character limit (1 token ≈ 4 characters)
        
        Log.d("ChatViewModel", "Context window: Model ${model.name} has ${model.contextWindowSize} tokens, using ${maxContextTokens} tokens (${maxContextChars} chars) for chat $currentChatId with ${chatMessages.size} messages")
        
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
        
        // If full history fits, return it
        if (fullHistory.length <= maxContextChars) {
            Log.d("ChatViewModel", "Full conversation history fits in context window (${fullHistory.length} chars)")
            return fullHistory
        }
        
        // Otherwise, implement smart truncation
        val recentPairs = mutableListOf<String>()
        var currentLength = 0
        
        // Always keep the last few conversation pairs (minimum viable context)
        val minimumPairs = 3
        
        // Add pairs from most recent backwards until we hit the limit
        for (i in pairStrings.indices.reversed()) {
            val pairString = pairStrings[i]
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
        
        // If we had to truncate, add context summary
        val truncatedCount = pairStrings.size - recentPairs.size
        val result = if (truncatedCount > 0) {
            val contextSummary = "[Previous conversation context: ${truncatedCount} earlier exchanges were truncated to fit context window]"
            listOf(contextSummary, recentPairs.joinToString("\n\n")).joinToString("\n\n")
        } else {
            recentPairs.joinToString("\n\n")
        }
        
        Log.d("ChatViewModel", "Context window management: Original ${fullHistory.length} chars (${pairStrings.size} pairs), trimmed to ${result.length} chars (${recentPairs.size} pairs, ${truncatedCount} truncated)")
        
        // Debug: Log first 200 chars of context to verify it's correct
        val preview = result.take(200) + if (result.length > 200) "..." else ""
        Log.d("ChatViewModel", "Context preview for chat $currentChatId: $preview")
        
        return result
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
            val tokenThreshold = (maxTokens * 0.7).toInt() // Reset at 70% to be safe
            
            Log.d("ChatViewModel", "Token check for chat $chatId: ~$estimatedTokens tokens, threshold: $tokenThreshold, max: $maxTokens")
            
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
    
    private suspend fun initializeNewChat(context: Context) {
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
        
        // Create new chat with appropriate model
        val newChatId = repository.createNewChat(
            "New Chat",
            if (_availableModels.value.isEmpty()) "No model downloaded" else 
            (modelToUse?.name ?: "No model selected")
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
            
            // Ensure the model is loaded in inference service in background
            // Don't wait for this to complete - it can happen asynchronously
            viewModelScope.launch {
                try {
                    inferenceService.loadModel(modelToUse)
                    Log.d("ChatViewModel", "Successfully loaded model ${modelToUse.name} for new chat")
                    // Update the state after successful load
                    syncCurrentlyLoadedModel()
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Failed to load model ${modelToUse.name} for new chat: ${e.message}")
                    // Model loading failed, but don't block the UI
                }
            }
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
            if (chatId == currentChatId) {
                _messages.value = emptyList()
            }
        }
    }

    fun clearAllChats() {
        viewModelScope.launch {
            repository.deleteAllChats()
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
                
                // Find the user message that prompted this response
                val messageIndex = currentMessages.indexOf(messageToRegenerate)
                val userMessage = if (messageIndex > 0) {
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
                
                // Delete the current AI response and any messages after it
                val messagesToKeep = currentMessages.takeWhile { it.id != messageId }
                
                // Clear messages after the user message from database
                repository.deleteMessagesAfter(chatId, userMessage.timestamp)
                
                // Update the messages state immediately
                _messages.value = messagesToKeep
                
                // Mark as loading
                _isLoading.value = true
                isGenerating = true
                
                // Ensure the model is loaded
                try {
                    inferenceService.loadModel(currentModel!!)
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
                    delay(200) // Additional time for session to stabilize
                    Log.d("ChatViewModel", "Reset session for regeneration")
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Error resetting session for regeneration: ${e.message}")
                    // If session reset fails, try to force recreate the session
                    try {
                        delay(200)
                        inferenceService.onCleared()
                        delay(500)
                        inferenceService.loadModel(currentModel!!)
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
                val historyMessages = messagesToKeep.filter { it.id != userMessage.id }
                val history = buildContextAwareHistory(historyMessages)
                
                // Extract images if the user message has attachments
                val images = if (currentModel!!.supportsVision && userMessage.attachmentPath != null && userMessage.attachmentType == "image") {
                    try {
                        val bitmap = loadImageFromUri(context, Uri.parse(userMessage.attachmentPath))
                        if (bitmap != null) listOf(bitmap) else emptyList()
                    } catch (e: Exception) {
                        Log.w("ChatViewModel", "Failed to load image for regeneration: ${e.message}")
                        emptyList()
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
                    var totalContent = ""
                    
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
                        inferenceService.loadModel(currentModel!!)
                        _isLoadingModel.value = false
                        
                        val responseStream = inferenceService.generateResponseStreamWithSession(
                            fullHistory, 
                            currentModel!!, 
                            chatId, 
                            images
                        )
                        
                        var lastUpdateTime = 0L
                        val updateIntervalMs = 50L
                        
                        responseStream.collect { piece ->
                            totalContent += piece
                            
                            // Real-time checks for repetition and length
                            if (totalContent.length > 200) {
                                val isRepetitive = checkForRepetition(totalContent)
                                if (isRepetitive) {
                                    Log.w("ChatViewModel", "Stopping regeneration due to repetitive content")
                                    try {
                                        inferenceService.resetChatSession(chatId)
                                    } catch (e: Exception) {
                                        Log.w("ChatViewModel", "Failed to reset session: ${e.message}")
                                    }
                                    throw kotlinx.coroutines.CancellationException("Repetitive content detected")
                                }
                            }
                            
                            if (totalContent.length > 5000) {
                                Log.w("ChatViewModel", "Stopping regeneration due to maximum length exceeded")
                                try {
                                    inferenceService.resetChatSession(chatId)
                                } catch (e: Exception) {
                                    Log.w("ChatViewModel", "Failed to reset session: ${e.message}")
                                }
                                throw kotlinx.coroutines.CancellationException("Length limit reached")
                            }
                            
                            // Update UI
                            val updated = _streamingContents.value.toMutableMap()
                            updated[placeholderId] = totalContent
                            _streamingContents.value = updated
                            
                            // Debounced database updates
                            val currentTime = System.currentTimeMillis()
                            if (currentTime - lastUpdateTime > updateIntervalMs) {
                                repository.updateMessageContent(placeholderId, totalContent)
                                lastUpdateTime = currentTime
                            }
                        }
                        
                        // Success - finalize the message
                        val finalContent = totalContent
                        repository.updateMessageContent(placeholderId, finalContent)
                        val time = System.currentTimeMillis() - generationStartTime
                        finalizeMessage(placeholderId, finalContent, time)
                        
                        Log.d("ChatViewModel", "Regeneration completed successfully")
                        
                    } catch (e: Exception) {
                        val finalContent = totalContent
                        val time = System.currentTimeMillis() - generationStartTime
                        
                        Log.d("ChatViewModel", "Regeneration exception: ${e.javaClass.simpleName}: ${e.message}")
                        
                        // Save partial content and finalize
                        withContext(kotlinx.coroutines.NonCancellable) {
                            repository.updateMessageContent(placeholderId, finalContent)
                            finalizeMessage(placeholderId, finalContent, time)
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
}