package com.llmhub.llmhub

import android.app.Application
import android.content.Context
import android.content.res.Configuration
import com.llmhub.llmhub.ads.AdManager
import com.llmhub.llmhub.ads.InterstitialAdManager
import com.llmhub.llmhub.ads.RewardedAdManager
import com.llmhub.llmhub.billing.BillingManager
import com.llmhub.llmhub.data.LlmHubDatabase
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.UnifiedInferenceService
import com.llmhub.llmhub.repository.ChatRepository
import com.llmhub.llmhub.utils.LocaleHelper
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking

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

    /** Billing manager — lazily constructed, persists for App lifetime. */
    val billingManager by lazy { BillingManager(this) }

    /** Interstitial ad manager — lazily constructed, persists for App lifetime. */
    val interstitialAdManager by lazy { InterstitialAdManager(this) }

    /** Rewarded ad manager — lazily constructed, persists for App lifetime. */
    val rewardedAdManager by lazy { RewardedAdManager(this) }

    override fun onCreate() {
        super.onCreate()
        // Apply saved language preference or system locale
        applySavedLanguage()
        // Initialise AdMob SDK
        AdManager.initialize(this)
        // Eagerly construct billing & ads managers so they start connecting immediately
        billingManager
        interstitialAdManager
        rewardedAdManager
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
