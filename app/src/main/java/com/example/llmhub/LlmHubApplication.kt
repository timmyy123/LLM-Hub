package com.llmhub.llmhub

import android.app.Application
import com.llmhub.llmhub.data.LlmHubDatabase
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.llmhub.llmhub.repository.ChatRepository

class LlmHubApplication : Application() {
    val inferenceService: InferenceService by lazy {
        MediaPipeInferenceService(applicationContext)
    }
    
    val database by lazy { LlmHubDatabase.getDatabase(this) }
    val chatRepository by lazy { ChatRepository(database.chatDao(), database.messageDao()) }
}
