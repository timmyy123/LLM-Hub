package com.example.llmhub.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.llmhub.data.ChatEntity
import com.example.llmhub.data.LlmHubDatabase
import com.example.llmhub.repository.ChatRepository
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