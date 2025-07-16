package com.example.llmhub.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.example.llmhub.screens.ChatScreen
import com.example.llmhub.screens.SettingsScreen
import com.example.llmhub.screens.ModelDownloadScreen
import com.example.llmhub.screens.AboutScreen
import com.example.llmhub.screens.TermsOfServiceScreen
import com.example.llmhub.viewmodels.ChatViewModelFactory

sealed class Screen(val route: String) {
    object Chat : Screen("chat/{chatId}") {
        fun createRoute(chatId: String = "new") = "chat/$chatId"
    }
    object Settings : Screen("settings")
    object Models : Screen("models")
    object About : Screen("about")
    object Terms : Screen("terms")
}

@Composable
fun LlmHubNavigation(
    navController: NavHostController,
    chatViewModelFactory: ChatViewModelFactory,
    startDestination: String = Screen.Chat.createRoute()
) {
    NavHost(
        navController = navController,
        startDestination = startDestination
    ) {
        composable(
            route = Screen.Chat.route
        ) { backStackEntry ->
            val chatId = backStackEntry.arguments?.getString("chatId") ?: "new"
            ChatScreen(
                chatId = chatId,
                viewModelFactory = chatViewModelFactory,
                onNavigateToSettings = {
                    navController.navigate(Screen.Settings.route)
                },
                onNavigateToModels = {
                    navController.navigate(Screen.Models.route)
                },
                onNavigateToChat = { newChatId ->
                    navController.navigate(Screen.Chat.createRoute(newChatId)) {
                        popUpTo(Screen.Chat.route) { inclusive = true }
                    }
                }
            )
        }
        
        composable(Screen.Settings.route) {
            SettingsScreen(
                onNavigateBack = {
                    navController.popBackStack()
                },
                onNavigateToModels = {
                    navController.navigate(Screen.Models.route)
                },
                onNavigateToAbout = {
                    navController.navigate(Screen.About.route)
                },
                onNavigateToTerms = {
                    navController.navigate(Screen.Terms.route)
                }
            )
        }
        
        composable(Screen.Models.route) {
            ModelDownloadScreen(
                onNavigateBack = {
                    navController.popBackStack()
                }
            )
        }
        
        composable(Screen.About.route) {
            AboutScreen(
                onNavigateBack = {
                    navController.popBackStack()
                }
            )
        }
        
        composable(Screen.Terms.route) {
            TermsOfServiceScreen(
                onNavigateBack = {
                    navController.popBackStack()
                }
            )
        }
    }
}