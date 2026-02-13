package com.llmhub.llmhub.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.ChatEntity
import com.llmhub.llmhub.data.LlmHubDatabase
import com.llmhub.llmhub.repository.ChatRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class ChatDrawerViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: ChatRepository

    val allChats: StateFlow<List<ChatEntity>>
    val allCreators: StateFlow<List<com.llmhub.llmhub.data.CreatorEntity>>

    init {
        val database = LlmHubDatabase.getDatabase(application)
        repository = ChatRepository(database.chatDao(), database.messageDao(), database.creatorDao())

        allChats = repository.getActiveChats()
            .stateIn(
                viewModelScope,
                SharingStarted.WhileSubscribed(5000),
                emptyList()
            )
            
        allCreators = repository.getAllCreators()
            .stateIn(
                viewModelScope,
                SharingStarted.WhileSubscribed(5000),
                emptyList()
            )
    }

    fun deleteChat(chatId: String) = viewModelScope.launch {
        repository.deleteChat(chatId)
    }

    fun deleteAllChats() = viewModelScope.launch {
        repository.deleteAllChats()
    }
    
    fun deleteCreator(creator: com.llmhub.llmhub.data.CreatorEntity) = viewModelScope.launch {
        repository.deleteCreator(creator)
    }
} 