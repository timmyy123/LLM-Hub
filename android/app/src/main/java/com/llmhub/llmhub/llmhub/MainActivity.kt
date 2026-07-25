package com.llmhub.llmhub

import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.LayoutDirection
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.compose.rememberNavController
import kotlinx.coroutines.flow.first
import com.llmhub.llmhub.navigation.LlmHubNavigation
import com.llmhub.llmhub.ui.theme.LlmHubTheme
import com.llmhub.llmhub.viewmodels.ChatViewModelFactory
import com.llmhub.llmhub.viewmodels.ThemeViewModel
import com.llmhub.llmhub.utils.LocaleHelper

class MainActivity : ComponentActivity() {
    private lateinit var themeViewModel: ThemeViewModel
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val app = application as LlmHubApplication
        val chatRepository = app.chatRepository
        val chatViewModelFactory = ChatViewModelFactory(app, chatRepository, this)

        // Initialize ThemeViewModel
        themeViewModel = ThemeViewModel(this)

        enableEdgeToEdge()
        setContent {
            val currentThemeMode by themeViewModel.themeMode.collectAsState()
            val currentLanguage by themeViewModel.appLanguage.collectAsState()

            // Apply locale (updates Context resources + AppCompat locales)
            LocaleHelper.setLocale(this@MainActivity, currentLanguage)

            // RTL locales: Persian (fa), Arabic (ar), Hebrew (he/iw)
            val isRtl = currentLanguage in setOf("fa", "ar", "he", "iw")
            val layoutDirection = if (isRtl) LayoutDirection.Rtl else LayoutDirection.Ltr

            LlmHubTheme(themeMode = currentThemeMode) {
                CompositionLocalProvider(LocalLayoutDirection provides layoutDirection) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val navController = rememberNavController()
                    LlmHubNavigation(
                        navController = navController,
                        chatViewModelFactory = chatViewModelFactory,
                        themeViewModel = themeViewModel
                    )
                }
                } // CompositionLocalProvider
            }
        }
    }
    
    override fun attachBaseContext(newBase: Context) {
        // Get the saved language preference and apply locale configuration
        val themePrefs = com.llmhub.llmhub.data.ThemePreferences(newBase)
        val savedLanguage = try {
            // Try to get the saved language synchronously
            kotlinx.coroutines.runBlocking {
                themePrefs.appLanguage.first()
            }
        } catch (e: Exception) {
            null // Fall back to system default if we can't read preferences
        }
        
        super.attachBaseContext(LocaleHelper.setLocale(newBase, savedLanguage))
    }
}