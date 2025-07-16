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

    init {
        val database = LlmHubDatabase.getDatabase(application)
        repository = ChatRepository(database.chatDao(), database.messageDao())

        allChats = repository.getActiveChats()
            .stateIn(
                scope = viewModelScope,
                started = SharingStarted.WhileSubscribed(5000),
                initialValue = emptyList()
            )
    }

    fun deleteChat(chatId: String) = viewModelScope.launch {
        repository.deleteChat(chatId)
    }

    fun deleteAllChats() = viewModelScope.launch {
        repository.deleteAllChats()
    }
} 