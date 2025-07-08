package com.example.llmhub.viewmodels

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llmhub.data.*
import com.example.llmhub.inference.LlamaInferenceService
import com.example.llmhub.repository.ChatRepository
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay
import java.io.File

class ChatViewModel : ViewModel() {

    private lateinit var repository: ChatRepository
    private lateinit var inferenceService: LlamaInferenceService

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

    fun initializeChat(chatId: String, context: Context) {
        val database = LlmHubDatabase.getDatabase(context)
        repository = ChatRepository(database.chatDao(), database.messageDao())
        inferenceService = LlamaInferenceService(context)

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
            delay(500) // Shorter delay

            if (currentModel != null && currentModel!!.isDownloaded) {
                val prompt = if (attachmentUri != null) "Image attached: $messageText" else messageText

                // Insert placeholder assistant message; we'll update it with tokens
                val placeholderId = repository.addMessage(chatId, "", isFromUser = false)

                // Collect tokens
                val job = launch {
                    val sb = StringBuilder()
                    inferenceService.tokenStream().collect { piece ->
                        sb.append(piece)
                        repository.updateMessageContent(placeholderId, sb.toString())
                    }
                }

                // Run generation (blocking IO)
                inferenceService.generateResponse(prompt, currentModel!!)

                job.cancel() // Stop collecting
            } else {
                repository.addMessage(chatId, "Please download a model to start chatting.", isFromUser = false)
            }

            _isLoading.value = false
        }
    }

    fun switchModel(newModel: LLMModel) {
        viewModelScope.launch {
            _isLoading.value = true
            val success = inferenceService.loadModel(newModel)
            if (success) {
                currentModel = newModel
                val updatedChat = _currentChat.value?.copy(modelName = newModel.name)
                if (updatedChat != null) {
                    currentChatId?.let { repository.updateChatModel(it, newModel.name) }
                    _currentChat.value = updatedChat
                }
            } else {
                // TODO: Show an error message to the user
            }
            _isLoading.value = false
        }
    }

    override fun onCleared() {
        viewModelScope.launch {
            inferenceService.releaseModel()
        }
        super.onCleared()
    }

    private fun loadAvailableModels(context: Context) {
        viewModelScope.launch {
            val modelsDir = File(context.filesDir, "models")
            val downloadedModels = ModelData.models.mapNotNull { model ->
                val modelFile = File(modelsDir, "${model.name.replace(" ", "_")}.gguf")

                if (modelFile.exists()) {
                    model.copy(isDownloaded = true)
                } else {
                    null
                }
            }
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