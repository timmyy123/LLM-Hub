package com.llmhub.llmhub

import android.app.Application
import android.content.Context
import android.content.res.Configuration
import androidx.lifecycle.lifecycleScope
import com.llmhub.llmhub.data.LlmHubDatabase
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.llmhub.llmhub.repository.ChatRepository
import com.llmhub.llmhub.utils.LocaleHelper
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking

class LlmHubApplication : Application() {
    val inferenceService: InferenceService by lazy {
        MediaPipeInferenceService(applicationContext)
    }
    
    val database by lazy { LlmHubDatabase.getDatabase(this) }
    val chatRepository by lazy { ChatRepository(database.chatDao(), database.messageDao()) }
    
    override fun onCreate() {
        super.onCreate()
        // Apply saved language preference or system locale
        applySavedLanguage()
    }
    
    override fun attachBaseContext(base: Context) {
        // Try to apply saved language preference first
        val context = try {
            val themePreferences = ThemePreferences(base)
            val savedLanguage = runBlocking { themePreferences.appLanguage.first() }
            LocaleHelper.setLocale(base, savedLanguage)
        } catch (e: Exception) {
            // Fallback to system locale if unable to read preferences
            LocaleHelper.setLocale(base)
        }
        super.attachBaseContext(context)
    }
    
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // Handle configuration changes (like language changes)
        applySavedLanguage()
    }
    
    private fun applySavedLanguage() {
        try {
            val themePreferences = ThemePreferences(this)
            runBlocking {
                val savedLanguage = themePreferences.appLanguage.first()
                LocaleHelper.applyLocale(this@LlmHubApplication, savedLanguage)
            }
        } catch (e: Exception) {
            // Fallback to system locale if error occurs
            LocaleHelper.setLocale(this)
        }
    }
}
