package com.llmhub.llmhub.viewmodels

import android.app.Activity
import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.ThemeMode
import com.llmhub.llmhub.data.ThemePreferences
import com.llmhub.llmhub.utils.LocaleHelper
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class ThemeViewModel(private val context: Context) : ViewModel() {
    private val themePreferences = ThemePreferences(context)
    
    private val _themeMode = MutableStateFlow(ThemeMode.SYSTEM)
    val themeMode: StateFlow<ThemeMode> = _themeMode.asStateFlow()
    
    private val _webSearchEnabled = MutableStateFlow(true)
    val webSearchEnabled: StateFlow<Boolean> = _webSearchEnabled.asStateFlow()
    
    private val _appLanguage = MutableStateFlow<String?>(null)
    val appLanguage: StateFlow<String?> = _appLanguage.asStateFlow()
    
    private val _selectedEmbeddingModel = MutableStateFlow<String?>(null)
    val selectedEmbeddingModel: StateFlow<String?> = _selectedEmbeddingModel.asStateFlow()
    
    init {
        // Load the saved theme preference
        viewModelScope.launch {
            themePreferences.themeMode.collect { mode ->
                _themeMode.value = mode
            }
        }
        
        // Load the saved web search preference
        viewModelScope.launch {
            themePreferences.webSearchEnabled.collect { enabled ->
                _webSearchEnabled.value = enabled
            }
        }
        
        // Load the saved language preference
        viewModelScope.launch {
            themePreferences.appLanguage.collect { language ->
                _appLanguage.value = language
            }
        }
        
        // Load the saved embedding model preference
        viewModelScope.launch {
            themePreferences.selectedEmbeddingModel.collect { model ->
                _selectedEmbeddingModel.value = model
            }
        }
    }
    
    fun setThemeMode(mode: ThemeMode) {
        viewModelScope.launch {
            themePreferences.setThemeMode(mode)
            _themeMode.value = mode
        }
    }
    
    fun setWebSearchEnabled(enabled: Boolean) {
        viewModelScope.launch {
            themePreferences.setWebSearchEnabled(enabled)
            _webSearchEnabled.value = enabled
        }
    }
    
    fun setSelectedEmbeddingModel(modelName: String?) {
        viewModelScope.launch {
            themePreferences.setSelectedEmbeddingModel(modelName)
            _selectedEmbeddingModel.value = modelName
        }
    }
    
    private val _languageChangeCounter = MutableStateFlow(0)
    val languageChangeCounter: StateFlow<Int> = _languageChangeCounter.asStateFlow()
    
    fun setAppLanguage(languageCode: String?) {
        viewModelScope.launch {
            val oldLanguage = _appLanguage.value
            themePreferences.setAppLanguage(languageCode)
            _appLanguage.value = languageCode
            
            // Apply the new locale
            LocaleHelper.setLocale(context, languageCode)
            
            // Only recreate if language actually changed
            if (oldLanguage != languageCode) {
                // Trigger a counter change to force recomposition
                _languageChangeCounter.value += 1
                
                // Recreate the activity if it's an Activity context
                if (context is Activity) {
                    context.recreate()
                }
            }
        }
    }
    
    fun getSupportedLanguages(): List<Pair<String, String>> {
        return LocaleHelper.getSupportedLanguages(context)
    }
    
    fun getCurrentLanguageDisplayName(): String {
        val currentLanguage = _appLanguage.value
        return if (currentLanguage != null) {
            getSupportedLanguages().find { it.first == currentLanguage }?.second ?: "System Default"
        } else {
            "System Default"
        }
    }
    
    /**
     * Get the effective language code currently being used
     */
    fun getEffectiveLanguageCode(): String {
        val userPreference = _appLanguage.value
        return if (userPreference != null && isLanguageSupported(userPreference)) {
            userPreference
        } else {
            // Fall back to system locale
            val systemLocale = LocaleHelper.getCurrentLocale(context)
            val systemLanguage = systemLocale.language
            if (isLanguageSupported(systemLanguage)) systemLanguage else "en"
        }
    }
    
    private fun isLanguageSupported(languageCode: String): Boolean {
        return LocaleHelper.isLanguageSupported(languageCode)
    }
}
