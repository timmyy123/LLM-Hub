package com.llmhub.llmhub

import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.compose.rememberNavController
import com.llmhub.llmhub.navigation.LlmHubNavigation
import com.llmhub.llmhub.ui.theme.LlmHubTheme
import com.llmhub.llmhub.viewmodels.ChatViewModelFactory
import com.llmhub.llmhub.viewmodels.ThemeViewModel
import com.llmhub.llmhub.utils.LocaleHelper

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val app = application as LlmHubApplication
        val inferenceService = app.inferenceService
        val chatRepository = app.chatRepository
        val chatViewModelFactory = ChatViewModelFactory(inferenceService, chatRepository)

        enableEdgeToEdge()
        setContent {
            val themeViewModel = ThemeViewModel(this)
            val currentThemeMode by themeViewModel.themeMode.collectAsState()
            
            LlmHubTheme(themeMode = currentThemeMode) {
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
            }
        }
    }
    
    override fun attachBaseContext(newBase: Context) {
        // Apply locale configuration before attaching base context
        super.attachBaseContext(LocaleHelper.setLocale(newBase))
    }
}