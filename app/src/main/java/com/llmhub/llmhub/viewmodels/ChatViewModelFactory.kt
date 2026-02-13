package com.llmhub.llmhub.viewmodels

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.repository.ChatRepository
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.createSavedStateHandle
import androidx.lifecycle.viewmodel.CreationExtras

class ChatViewModelFactory(
    private val application: android.app.Application,
    private val repository: ChatRepository,
    private val context: Context
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>, extras: CreationExtras): T {
        val inferenceService = (application as com.llmhub.llmhub.LlmHubApplication).inferenceService
        
        if (modelClass.isAssignableFrom(ChatViewModel::class.java)) {
            val savedStateHandle = extras.createSavedStateHandle()
            @Suppress("UNCHECKED_CAST")
            return ChatViewModel(inferenceService, repository, application, savedStateHandle) as T
        }
        if (modelClass.isAssignableFrom(CreatorViewModel::class.java)) {
            @Suppress("UNCHECKED_CAST")
            return CreatorViewModel(repository, inferenceService, application) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
