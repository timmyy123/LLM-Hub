package com.example.llmhub

import android.app.Application
import com.example.llmhub.data.LlmHubDatabase
import com.example.llmhub.inference.InferenceService
import com.example.llmhub.inference.MediaPipeInferenceService
import com.example.llmhub.repository.ChatRepository

class LlmHubApplication : Application() {
    val inferenceService: InferenceService by lazy {
        MediaPipeInferenceService(applicationContext)
    }
    
    val database by lazy { LlmHubDatabase.getDatabase(this) }
    val chatRepository by lazy { ChatRepository(database.chatDao(), database.messageDao()) }
}
