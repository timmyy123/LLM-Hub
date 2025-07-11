package com.example.llmhub.viewmodels

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llmhub.data.*
import com.example.llmhub.inference.MediaPipeInferenceService
import com.example.llmhub.repository.ChatRepository
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import java.io.File
import kotlinx.coroutines.Job
import com.example.llmhub.data.localFileName
import android.util.Log

class ChatViewModel : ViewModel() {

    // keep a single inference engine instance across all ChatViewModel objects
    companion object {
        private var sharedInferenceService: MediaPipeInferenceService? = null
    }

    private lateinit var repository: ChatRepository
    private lateinit var inferenceService: MediaPipeInferenceService

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

    private var currentChatId: String? = null
    private var currentModel: LLMModel? = null

    // Keep reference to the running generation so the UI can interrupt it
    private var generationJob: Job? = null

    fun hasDownloadedModels(): Boolean {
        return _availableModels.value.isNotEmpty()
    }

    fun initializeChat(chatId: String, context: Context) {
        val database = LlmHubDatabase.getDatabase(context)
        repository = ChatRepository(database.chatDao(), database.messageDao())
        // reuse a single inference engine instance to avoid reloading the model every time
        inferenceService = sharedInferenceService ?: MediaPipeInferenceService(context).also {
            sharedInferenceService = it
        }

        // Stop collecting from any previous chat's message flow
        generationJob?.cancel()

        viewModelScope.launch {
            // Load the models synchronously so we know what's available before creating/attaching a chat
            loadAvailableModelsSync(context)

            if (chatId == "new") {
                // Do NOT auto-select a model; let user choose if any are downloaded
                val newChatId = repository.createNewChat(
                    "New Chat",
                    if (_availableModels.value.isEmpty()) "No model downloaded" else "No model selected"
                )
                currentChatId = newChatId
                _currentChat.value = repository.getChatById(newChatId)
                currentModel = null

                // Begin collecting messages for the newly created chat
                repository.getMessagesForChat(newChatId).collectLatest { messageList ->
                    _messages.value = messageList
                }
            } else {
                currentChatId = chatId
                val chat = repository.getChatById(chatId)
                _currentChat.value = chat
                currentModel = _availableModels.value.find { it.name == chat?.modelName }
                    ?: ModelData.models.find { it.name == chat?.modelName }
                
                if (currentModel != null && !currentModel!!.isDownloaded) {
                    currentModel = null // Model associated with chat is not downloaded
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
    }

    fun sendMessage(text: String, attachmentUri: Uri?) {
        val chatId = currentChatId ?: return
        val messageText = text.trim()

        if (messageText.isEmpty() && attachmentUri == null) return

        viewModelScope.launch {
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
            // _tokenStats.value = null // No longer need separate token stats state

            if (currentModel != null && currentModel!!.isDownloaded) {
                // Pass the conversation history to the model
                val history = _messages.value.map {
                    (if (it.isFromUser) "user: " else "model: ") + it.content
                }.joinToString(separator = "\n")

                // Insert a visible placeholder so the bubble stays rendered while tokens stream
                val placeholderId = repository.addMessage(chatId, "…", isFromUser = false)
                _streamingContents.value = mapOf(placeholderId to "") // Initialize with empty string

                // Run generation with streaming tokens
                generationJob = launch {
                    var finalContent: String? = null
                    var generationTimeMs: Long = 0
                    val generationStartTime = System.currentTimeMillis()

                    try {
                        // Pre-load model separately from generation timing
                        _isLoadingModel.value = true
                        
                        // Ensure model is loaded first
                        inferenceService.loadModel(currentModel!!)
                        
                        _isLoadingModel.value = false
                        
                        val responseStream = inferenceService.generateResponseStream(history, currentModel!!)
                        var lastUpdateTime = 0L
                        val updateIntervalMs = 50L // Update UI every 50ms instead of every token
                        
                        responseStream.collect { piece ->
                            // Append piece to the content for the specific message
                            val updated = _streamingContents.value.toMutableMap()
                            updated[placeholderId] = (updated[placeholderId] ?: "") + piece
                            _streamingContents.value = updated
                            
                            // Debounced database updates to reduce blinking
                            val currentTime = System.currentTimeMillis()
                            if (currentTime - lastUpdateTime > updateIntervalMs) {
                                repository.updateMessageContent(placeholderId, updated[placeholderId] ?: "")
                                lastUpdateTime = currentTime
                            }
                        }

                        // This section now executes after the stream is fully collected
                        finalContent = _streamingContents.value[placeholderId] ?: ""
                        generationTimeMs = System.currentTimeMillis() - generationStartTime

                    } catch (e: Exception) {
                        if (e is kotlinx.coroutines.CancellationException) {
                            Log.d("ChatViewModel", "Generation was cancelled by user.")
                            // Save whatever was partially generated
                            finalContent = _streamingContents.value[placeholderId] ?: ""
                            generationTimeMs = System.currentTimeMillis() - generationStartTime
                        } else {
                            Log.e("ChatViewModel", "Error during generation", e)
                            finalContent = _streamingContents.value[placeholderId]
                            repository.updateMessageContent(placeholderId, "Error: ${e.message}")
                        }
                    } finally {
                        _isLoading.value = false
                        _isLoadingModel.value = false
                        
                        // Finalize the message content and stats, even if cancelled
                        if (finalContent != null) {
                            val estimatedTokens = if (finalContent!!.isNotBlank()) {
                                finalContent!!.split(Regex("\\s+|[.,!?;:]")).filter { it.isNotBlank() }.size
                            } else { 1 }
                            val tokensPerSecond = if (generationTimeMs > 0) (estimatedTokens * 1000.0) / generationTimeMs else 0.0

                            repository.updateMessageContent(placeholderId, finalContent!!)
                            repository.updateMessageStats(placeholderId, estimatedTokens, tokensPerSecond)
                        }

                        // Clear streaming state for this message
                        val updatedStreaming = _streamingContents.value.toMutableMap()
                        updatedStreaming.remove(placeholderId)
                        _streamingContents.value = updatedStreaming
                    }
                }
            } else {
                repository.addMessage(chatId, "Please download a model to start chatting.", isFromUser = false)
                _isLoading.value = false
            }
        }
    }

    /** Interrupt the current response generation if one is running */
    fun stopGeneration() {
        generationJob?.cancel()
        generationJob = null
        _isLoading.value = false
        _isLoadingModel.value = false
        // Note: Don't clear streaming contents here to preserve partial responses
        // _tokenStats.value = null // No longer needed
    }

    fun switchModel(newModel: LLMModel) {
        viewModelScope.launch {
            _isLoading.value = true
            _isLoadingModel.value = true
            
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
            } catch (e: Exception) {
                // Model loading will happen on first actual use
            }
            
            _isLoadingModel.value = false
            _isLoading.value = false
        }
    }

    override fun onCleared() {
        viewModelScope.launch {
            inferenceService.onCleared()
        }
        super.onCleared()
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
} 