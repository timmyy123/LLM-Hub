package com.llmhub.llmhub

import android.app.Application
import android.content.Context
import android.content.res.Configuration
import com.llmhub.llmhub.data.LlmHubDatabase
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.llmhub.llmhub.repository.ChatRepository
import com.llmhub.llmhub.utils.LocaleHelper

class LlmHubApplication : Application() {
    val inferenceService: InferenceService by lazy {
        MediaPipeInferenceService(applicationContext)
    }
    
    val database by lazy { LlmHubDatabase.getDatabase(this) }
    val chatRepository by lazy { ChatRepository(database.chatDao(), database.messageDao()) }
    
    override fun onCreate() {
        super.onCreate()
        // Apply system locale or default to English if not supported
        LocaleHelper.setLocale(this)
    }
    
    override fun attachBaseContext(base: Context) {
        // Apply locale configuration before attaching base context
        super.attachBaseContext(LocaleHelper.setLocale(base))
    }
    
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // Handle configuration changes (like language changes)
        LocaleHelper.setLocale(this)
    }
}
