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
        if (modelClass.isAssignableFrom(ChatViewModel::class.java)) {
            val savedStateHandle = extras.createSavedStateHandle()
            // Use application-scoped InferenceService so model state persists across ViewModels
            // (avoids creating a new MediaPipeInferenceService per ViewModel which would cause
            // models to be unloaded when a ViewModel is cleared)
            val inferenceService = (application as com.llmhub.llmhub.LlmHubApplication).inferenceService
            @Suppress("UNCHECKED_CAST")
            return ChatViewModel(inferenceService, repository, context, savedStateHandle) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
