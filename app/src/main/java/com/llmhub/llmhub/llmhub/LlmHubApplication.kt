package com.llmhub.llmhub

import android.app.Application
import android.content.Context
import android.content.res.Configuration
import androidx.lifecycle.lifecycleScope
import com.llmhub.llmhub.data.LlmHubDatabase
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.inference.InferenceService

import com.llmhub.llmhub.inference.UnifiedInferenceService
import com.llmhub.llmhub.repository.ChatRepository
import com.llmhub.llmhub.utils.LocaleHelper
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import android.util.Log

class LlmHubApplication : Application() {
    private var _inferenceService: InferenceService? = null
    
    val inferenceService: InferenceService
        get() {
            if (_inferenceService == null) {
                _inferenceService = UnifiedInferenceService(this)
            }
            return _inferenceService!!
        }
    
    val database by lazy { LlmHubDatabase.getDatabase(this) }
    val chatRepository by lazy { ChatRepository(database.chatDao(), database.messageDao(), database.creatorDao()) }
    
    override fun onCreate() {
        super.onCreate()
        // Apply saved language preference or system locale
        applySavedLanguage()
        // Initialize RAG and restore any embedded global memories so they are available
        // immediately after app restart. Also kick off processing for any pending uploads.
        try {
            val ragManager = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(this)
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val job = ragManager.initializeAsync()
                    job.join()

                    // Restore persisted per-chunk embeddings into the in-memory RAG index
                    // to avoid re-running the embedder on startup.
                    try {
                        val chunkList = database.memoryDao().getAllChunks()
                        if (chunkList.isNotEmpty()) {
                            Log.d("LlmHubApplication", "Restoring ${chunkList.size} persisted memory chunks into RAG")
                            ragManager.restoreGlobalDocumentsFromChunks(chunkList)
                        } else {
                            Log.d("LlmHubApplication", "No persisted memory chunks found for restore")
                        }
                    } catch (e: Exception) {
                        Log.w("LlmHubApplication", "Failed to read persisted memory chunks for restore: ${e.message}")
                    }

                    // Also attempt to process any pending/pending-failed documents
                    try {
                        val processor = com.llmhub.llmhub.data.MemoryProcessor(this@LlmHubApplication, database)
                        processor.processPending()
                    } catch (e: Exception) {
                        Log.w("LlmHubApplication", "Failed to start MemoryProcessor: ${e.message}")
                    }
                } catch (e: Exception) {
                    Log.w("LlmHubApplication", "RAG initialization/restore failed at startup: ${e.message}")
                }
            }
        } catch (e: Exception) {
            Log.w("LlmHubApplication", "Failed to schedule RAG startup restore: ${e.message}")
        }
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
        // Note: Do NOT call applySavedLanguage() here as it destroys the InferenceService
        // and unloads any loaded model. The InferenceService already handles locale changes
        // internally, and destroying it on every configuration change (rotation, keyboard, etc.)
        // causes the model to be unnecessarily unloaded.
        // Language changes are applied in onCreate() and attachBaseContext() which is sufficient.
    }
    
    private fun applySavedLanguage() {
        try {
            val themePreferences = ThemePreferences(this)
            runBlocking {
                val savedLanguage = themePreferences.appLanguage.first()
                LocaleHelper.applyLocale(this@LlmHubApplication, savedLanguage)
                
                // Note: Do NOT destroy InferenceService here. The MediaPipeInferenceService
                // has its own getCurrentContext() method that properly handles locale changes.
                // Destroying it here causes loaded models to be unloaded on config changes.
            }
        } catch (e: Exception) {
            // Fallback to system locale if error occurs
            LocaleHelper.setLocale(this)
        }
    }
}
