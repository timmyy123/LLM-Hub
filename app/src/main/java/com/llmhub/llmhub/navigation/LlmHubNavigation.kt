package com.llmhub.llmhub.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.material3.DrawerState
import androidx.compose.material3.DrawerValue
import androidx.navigation.compose.currentBackStackEntryAsState
import kotlinx.coroutines.launch
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import com.llmhub.llmhub.LlmHubApplication
import com.llmhub.llmhub.screens.*
import com.llmhub.llmhub.viewmodels.ChatViewModelFactory
import com.llmhub.llmhub.viewmodels.ThemeViewModel

sealed class Screen(val route: String) {
    object Home : Screen("home")
    object Chat : Screen("chat/{chatId}") {
        fun createRoute(chatId: String = "new") = "chat/$chatId"
    }
    object ChatHistory : Screen("chat_history")
    object WritingAid : Screen("writing_aid")
    object Translator : Screen("translator")
    object Transcriber : Screen("transcriber")
    object ScamDetector : Screen("scam_detector")
    object ImageGenerator : Screen("image_generator")
    object Settings : Screen("settings")
    object Models : Screen("models")
    object About : Screen("about")
    object Terms : Screen("terms")
}

@Composable
fun LlmHubNavigation(
    navController: NavHostController,
    chatViewModelFactory: ChatViewModelFactory,
    themeViewModel: ThemeViewModel,
    startDestination: String = Screen.Home.route
) {
    val drawerState = rememberSaveable(saver = DrawerState.Saver(confirmStateChange = { true })) {
        DrawerState(DrawerValue.Closed)
    }

    // Unload chat model only when leaving Chat route (e.g. to Home). Don't unload when switching chat/123 -> chat/new.
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route
    val isOnChatRoute = currentRoute == Screen.Chat.route
    var wasOnChatRoute by remember { mutableStateOf(isOnChatRoute) }
    val context = LocalContext.current
    LaunchedEffect(isOnChatRoute) {
        if (wasOnChatRoute && !isOnChatRoute) {
            (context.applicationContext as? LlmHubApplication)?.inferenceService?.unloadModel()
        }
        wasOnChatRoute = isOnChatRoute
    }

    NavHost(
        navController = navController,
        startDestination = startDestination
    ) {
        // Home/Landing Screen
        composable(Screen.Home.route) {
            HomeScreen(
                onNavigateToFeature = { route ->
                    when (route) {
                        "chat" -> navController.navigate(Screen.Chat.createRoute("new"))
                        "writing_aid" -> navController.navigate(Screen.WritingAid.route)
                        "translator" -> navController.navigate(Screen.Translator.route)
                        "transcriber" -> navController.navigate(Screen.Transcriber.route)
                        "scam_detector" -> navController.navigate(Screen.ScamDetector.route)
                        "image_generator" -> navController.navigate(Screen.ImageGenerator.route)
                    }
                },
                onNavigateToSettings = {
                    navController.navigate(Screen.Settings.route)
                },
                onNavigateToModels = {
                    navController.navigate(Screen.Models.route)
                },
                onNavigateToChatHistory = {
                    navController.navigate(Screen.ChatHistory.route)
                }
            )
        }
        
        // Chat History Screen
        composable(Screen.ChatHistory.route) {
            ChatHistoryScreen(
                onNavigateBack = {
                    navController.popBackStack()
                },
                onNavigateToChat = { chatId ->
                    navController.navigate(Screen.Chat.createRoute(chatId))
                },
                onCreateNewChat = {
                    navController.navigate(Screen.Chat.createRoute("new"))
                }
            )
        }
        
        // Chat Screen (existing functionality preserved)
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
                },
                onNavigateBack = {
                    navController.popBackStack(Screen.Home.route, inclusive = false)
                },
                drawerState = drawerState
            )
        }
        
        // Feature Screens
        composable(Screen.WritingAid.route) {
            WritingAidScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToModels = { navController.navigate(Screen.Models.route) }
            )
        }
        
        composable(Screen.Translator.route) {
            TranslatorScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToModels = { navController.navigate(Screen.Models.route) }
            )
        }
        
        composable(Screen.Transcriber.route) {
            TranscriberScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToModels = { navController.navigate(Screen.Models.route) }
            )
        }
        
        composable(Screen.ScamDetector.route) {
            ScamDetectorScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToModels = { navController.navigate(Screen.Models.route) }
            )
        }
        
        composable(Screen.ImageGenerator.route) {
            ImageGeneratorScreen(
                onNavigateBack = { navController.popBackStack() },
                onNavigateToModels = { navController.navigate(Screen.Models.route) }
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
                },
                themeViewModel = themeViewModel
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