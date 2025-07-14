package com.example.llmhub.viewmodels

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.SavedStateHandle
import com.example.llmhub.data.*
import com.example.llmhub.inference.InferenceService
import com.example.llmhub.repository.ChatRepository
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import java.io.File
import kotlinx.coroutines.Job
import com.example.llmhub.data.localFileName
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.withContext

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
    
    // Model loading state
    private val _isLoadingModel = MutableStateFlow(false)
    val isLoadingModel: StateFlow<Boolean> = _isLoadingModel.asStateFlow()
    
    // Currently loaded model (global state) - now reflects the inference service state
    private val _currentlyLoadedModel = MutableStateFlow<LLMModel?>(null)
    val currentlyLoadedModel: StateFlow<LLMModel?> = _currentlyLoadedModel.asStateFlow()

    private var currentChatId: String?
        get() = savedStateHandle.get<String>(KEY_CURRENT_CHAT_ID)
        set(value) = savedStateHandle.set(KEY_CURRENT_CHAT_ID, value)
    
    private var currentModelName: String?
        get() = savedStateHandle.get<String>(KEY_CURRENT_MODEL_NAME)
        set(value) = savedStateHandle.set(KEY_CURRENT_MODEL_NAME, value)
    
    private var isGenerating: Boolean
        get() = savedStateHandle.get<Boolean>(KEY_IS_GENERATING) ?: false
        set(value) = savedStateHandle.set(KEY_IS_GENERATING, value)
    
    private var currentModel: LLMModel? = null
        get() = field ?: currentModelName?.let { modelName ->
            _availableModels.value.find { it.name == modelName }
        }
        set(value) {
            field = value
            currentModelName = value?.name
            // Sync with the inference service's currently loaded model
            syncCurrentlyLoadedModel()
        }

    // Keep reference to the running generation so the UI can interrupt it
    private var generationJob: Job? = null

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
            val loadedModel = inferenceService.getCurrentlyLoadedModel()
            _currentlyLoadedModel.value = loadedModel
        }
    }

    fun initializeChat(chatId: String, context: Context) {
        // Sync the currently loaded model from inference service
        syncCurrentlyLoadedModel()
        
        // Stop collecting from any previous chat's message flow
        generationJob?.cancel()

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
                
                // Preserve the current model for new chats
                currentModel = previousModel
                
                // If we have a model, update the chat to use it
                if (previousModel != null) {
                    repository.updateChatModel(newChatId, previousModel.name)
                    _currentChat.value = repository.getChatById(newChatId)
                }

                // Begin collecting messages for the newly created chat
                repository.getMessagesForChat(newChatId).collectLatest { messageList ->
                    _messages.value = messageList
                }
            } else {
                // Check if the chat still exists
                val chat = repository.getChatById(chatId)
                if (chat == null) {
                    Log.e("ChatViewModel", "Chat $chatId does not exist, creating new chat instead")
                    initializeChat("new", context)
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
                
                repository.getMessagesForChat(chatId).collectLatest { messageList ->
                    _messages.value = messageList
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
                    val minSize = 10 * 1024 * 1024
                    if (primaryFile.length() >= minSize) {
                        isAvailable = true
                        actualSize = primaryFile.length()
                        Log.d("ChatViewModel", "Found model in files: ${primaryFile.absolutePath} (${actualSize / (1024*1024)} MB)")
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

    fun sendMessage(text: String, attachmentUri: Uri?) {
        val chatId = currentChatId
        if (chatId == null) {
            Log.e("ChatViewModel", "No current chat ID available")
            return
        }
        val messageText = text.trim()

        if (messageText.isEmpty() && attachmentUri == null) return

        viewModelScope.launch {
            // Check if the current chat still exists
            val currentChat = repository.getChatById(chatId)
            if (currentChat == null) {
                Log.e("ChatViewModel", "Current chat $chatId does not exist, cannot send message")
                return@launch
            }

            // Verify we have a working model
            if (currentModel == null || !currentModel!!.isDownloaded) {
                Log.e("ChatViewModel", "No valid model available for chat $chatId")
                repository.addMessage(chatId, "Please download a model to start chatting.", isFromUser = false)
                return@launch
            }

            repository.addMessage(
                chatId = chatId,
                content = if (messageText.isNotEmpty()) messageText else "Shared a file",
                isFromUser = true,
                attachmentPath = attachmentUri?.toString(),
                attachmentType = determineAttachmentType(attachmentUri)
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
                    inferenceService.loadModel(currentModel!!)
                    // Sync the currently loaded model state
                    syncCurrentlyLoadedModel()
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Failed to ensure model is loaded: ${e.message}")
                    repository.addMessage(chatId, "Failed to load model. Please try again.", isFromUser = false)
                    _isLoading.value = false
                    isGenerating = false
                    return@launch
                }
                
                // Session management is now handled internally by the inference service
                // No need to manually ensure or create sessions
                Log.d("ChatViewModel", "Starting generation for chat $chatId")
                
                // Pass the conversation history to the model with context window management
                val history = buildContextAwareHistory(_messages.value)

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
                                    
                                    val responseStream = inferenceService.generateResponseStreamWithSession(currentPrompt, currentModel!!, chatId)
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
                                    val isLikelyTruncated = isResponseTruncated(currentSegment, segmentTime)
                                    
                                    Log.d("ChatViewModel", "Segment ${continuationCount + 1}: length=${currentSegment.length}, time=${segmentTime}ms, hasContent=$segmentHasContent, truncated=$isLikelyTruncated")
                                    Log.d("ChatViewModel", "Segment ends with: '${currentSegment.takeLast(20)}'")
                                    
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
        val isRepetitive = if (words.size > 20) {
            val lastTenWords = words.takeLast(10).joinToString(" ")
            val beforeTenWords = words.dropLast(10).takeLast(10).joinToString(" ")
            lastTenWords == beforeTenWords
        } else false
        
        if (isRepetitive) {
            Log.d("ChatViewModel", "Content is repetitive - stopping continuation")
            return false
        }
        
        // Only continue if we have clear signs of incompleteness AND the content is meaningful
        val shouldContinue = !endsWithProperPunctuation && (endsWithIncompletePattern || isVeryShort || unclosedCodeBlocks)
        
        Log.d("ChatViewModel", "Truncation check: endsWithProperPunctuation=$endsWithProperPunctuation, endsWithIncompletePattern=$endsWithIncompletePattern, isVeryShort=$isVeryShort, unclosedCodeBlocks=$unclosedCodeBlocks, isOnlyWhitespace=$isOnlyWhitespace, isRepetitive=$isRepetitive, shouldContinue=$shouldContinue")
        
        return shouldContinue
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
                delay(500) // Give extra time for cleanup when switching models
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
                Log.d("ChatViewModel", "Successfully loaded new model: ${newModel.name}")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Failed to load new model: ${e.message}")
                // Model loading will happen on first actual use
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
        return currentModel?.supportsVision ?: false
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
            val userPart = if (userMsg != null) "user: ${userMsg.content}" else ""
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

    fun clearAllChatsAndCreateNew(context: Context) {
        viewModelScope.launch {
            // Reset current chat session
            currentChatId?.let { chatId ->
                try {
                    inferenceService.resetChatSession(chatId)
                    Log.d("ChatViewModel", "Reset session for chat $chatId during clear all")
                } catch (e: Exception) {
                    Log.w("ChatViewModel", "Error resetting chat session during clear all: ${e.message}")
                }
            }
            
            // Clear all chats from database
            repository.deleteAllChats()
            
            // Clear current state
            currentChatId = null
            _currentChat.value = null
            _messages.value = emptyList()
            _streamingContents.value = emptyMap()
            
            // Create a new empty chat
            initializeNewChat(context)
        }
    }
    
    private suspend fun initializeNewChat(context: Context) {
        // Load available models first
        loadAvailableModelsSync(context)
        
        // Get the currently loaded model from inference service
        val currentModel = inferenceService.getCurrentlyLoadedModel()
        
        // Create new chat with appropriate model
        val newChatId = repository.createNewChat(
            "New Chat",
            if (_availableModels.value.isEmpty()) "No model downloaded" else 
            (currentModel?.name ?: "No model selected")
        )
        
        // Set as current chat
        currentChatId = newChatId
        _currentChat.value = repository.getChatById(newChatId)
        
        // If we have a model, update the chat to use it
        if (currentModel != null) {
            repository.updateChatModel(newChatId, currentModel.name)
            _currentChat.value = repository.getChatById(newChatId)
            this.currentModel = currentModel
        }
        
        // Start collecting messages for the new chat in a separate coroutine
        viewModelScope.launch {
            repository.getMessagesForChat(newChatId).collectLatest { messageList ->
                _messages.value = messageList
            }
        }
        
        Log.d("ChatViewModel", "Created new chat $newChatId after clearing all chats")
    }

    private fun resetChatSession(chatId: String) {
        viewModelScope.launch {
            try {
                Log.d("ChatViewModel", "Resetting chat session for chat $chatId")
                
                // Cancel any ongoing generation first
                generationJob?.cancel()
                
                // Use the reset method which handles MediaPipe session errors
                inferenceService.resetChatSession(chatId)
                
                // Give MediaPipe significantly more time to clean up properly
                delay(600)
                
                Log.d("ChatViewModel", "Successfully reset chat session for chat $chatId")
            } catch (e: Exception) {
                Log.w("ChatViewModel", "Error during session reset for chat $chatId: ${e.message}")
            }
        }
    }
}