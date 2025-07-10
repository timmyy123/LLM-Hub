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

    private var currentChatId: String? = null
    private var currentModel: LLMModel? = null

    // Keep reference to the running generation so the UI can interrupt it
    private var generationJob: Job? = null

    fun initializeChat(chatId: String, context: Context) {
        val database = LlmHubDatabase.getDatabase(context)
        repository = ChatRepository(database.chatDao(), database.messageDao())
        // reuse a single inference engine instance to avoid reloading the model every time
        inferenceService = sharedInferenceService ?: MediaPipeInferenceService(context).also {
            sharedInferenceService = it
        }

        viewModelScope.launch {
            loadAvailableModels(context)

            if (chatId == "new") {
                val defaultModel = _availableModels.value.firstOrNull()
                val newChatId = repository.createNewChat("New Chat", defaultModel?.name ?: "No model downloaded")
                currentChatId = newChatId
                _currentChat.value = repository.getChatById(newChatId)
                currentModel = defaultModel

                // Begin collecting messages for the newly created chat
                repository.getMessagesForChat(newChatId).collect { messageList ->
                    _messages.value = messageList
                }
            } else {
                currentChatId = chatId
                val chat = repository.getChatById(chatId)
                _currentChat.value = chat
                currentModel = _availableModels.value.find { it.name == chat?.modelName }
                    ?: ModelData.models.find { it.name == chat?.modelName }
                repository.getMessagesForChat(chatId).collect { messageList ->
                    _messages.value = messageList
                }
            }
        }
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

            if (_messages.value.isEmpty()) {
                repository.updateChatTitle(chatId, messageText.take(50))
                _currentChat.value = repository.getChatById(chatId)
            }

            _isLoading.value = true
            delay(500) // Short delay just for UX (optional)

            if (currentModel != null && currentModel!!.isDownloaded) {
                val prompt = if (attachmentUri != null) "Image attached: $messageText" else messageText

                // Insert a visible placeholder so the bubble stays rendered while tokens stream
                val placeholderId = repository.addMessage(chatId, "…", isFromUser = false)

                // Run generation with streaming tokens
                generationJob = launch {
                    try {
                        // Use the new MediaPipe streaming API
                        val responseStream = inferenceService.generateResponseStream(prompt, currentModel!!)
                        val sb = StringBuilder()
                        
                        responseStream.collect { piece ->
                            if (piece == "[DONE]") {
                                // Generation completed
                                _isLoading.value = false
                                return@collect
                            }
                            sb.append(piece)
                            repository.updateMessageContent(placeholderId, sb.toString())
                        }
                    } catch (e: Exception) {
                        // Handle errors
                        repository.updateMessageContent(placeholderId, "Error: ${e.message}")
                        _isLoading.value = false
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
    }

    fun switchModel(newModel: LLMModel) {
        viewModelScope.launch {
            _isLoading.value = true
            
            // MediaPipe automatically loads models when needed, no separate loadModel() call
            currentModel = newModel
            val updatedChat = _currentChat.value?.copy(modelName = newModel.name)
            if (updatedChat != null) {
                currentChatId?.let { repository.updateChatModel(it, newModel.name) }
                _currentChat.value = updatedChat
            }
            
            _isLoading.value = false
        }
    }

    override fun onCleared() {
        viewModelScope.launch {
            inferenceService.onCleared()
        }
        super.onCleared()
    }

    private fun loadAvailableModels(context: Context) {
        viewModelScope.launch {
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
            
            Log.d("ChatViewModel", "Available models: ${downloadedModels.map { it.name }}")
            _availableModels.value = downloadedModels
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
} 