package com.llmhub.llmhub.viewmodels

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.ThemeMode
import com.llmhub.llmhub.data.ThemePreferences
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
}
