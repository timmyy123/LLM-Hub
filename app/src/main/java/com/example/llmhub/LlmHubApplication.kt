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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import android.util.Log

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
        // Initialize RAG and restore any embedded global memories so they are available
        // immediately after app restart. Also kick off processing for any pending uploads.
        try {
            val ragManager = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(this)
            CoroutineScope(Dispatchers.IO).launch {
                try {
                    val job = ragManager.initializeAsync()
                    job.join()

                    // Re-add any previously embedded global memories (they were stored in DB but
                    // embeddings live in-memory and must be recreated at startup).
                    try {
                        val list = database.memoryDao().getAllMemory().first()
                        val embedded = list.filter { it.status == "EMBEDDED" }
                        for (doc in embedded) {
                            try {
                                val added = ragManager.addGlobalDocument(doc.content, doc.fileName, doc.metadata)
                                Log.d("LlmHubApplication", "Restored embedded memory id=${doc.id} added=$added chunkCount=${ragManager.getDocumentCount("__global_memory__")}")
                            } catch (e: Exception) {
                                Log.w("LlmHubApplication", "Failed to restore memory ${doc.id}: ${e.message}")
                            }
                        }
                    } catch (e: Exception) {
                        Log.w("LlmHubApplication", "Failed to read memory DB for restore: ${e.message}")
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
