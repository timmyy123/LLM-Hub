package com.llmhub.llmhub

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.compose.rememberNavController
import com.llmhub.llmhub.navigation.LlmHubNavigation
import com.llmhub.llmhub.ui.theme.LlmHubTheme
import com.llmhub.llmhub.viewmodels.ChatViewModelFactory

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val app = application as LlmHubApplication
        val inferenceService = app.inferenceService
        val chatRepository = app.chatRepository
        val chatViewModelFactory = ChatViewModelFactory(inferenceService, chatRepository)

        enableEdgeToEdge()
        setContent {
            LlmHubTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    val navController = rememberNavController()
                    LlmHubNavigation(
                        navController = navController,
                        chatViewModelFactory = chatViewModelFactory
                    )
                }
            }
        }
    }
}