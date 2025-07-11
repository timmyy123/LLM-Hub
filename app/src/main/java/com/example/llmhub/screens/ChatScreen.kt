package com.example.llmhub.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.llmhub.components.ChatDrawer
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.foundation.clickable
import com.example.llmhub.components.MessageBubble
import com.example.llmhub.components.MessageInput
import com.example.llmhub.data.MessageEntity
import com.example.llmhub.viewmodels.ChatViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    chatId: String,
    onNavigateToSettings: () -> Unit,
    onNavigateToModels: () -> Unit,
    onNavigateToChat: (String) -> Unit,
    viewModel: ChatViewModel = viewModel()
) {
    val context = LocalContext.current
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    
    val messages by viewModel.messages.collectAsState()
    val currentChat by viewModel.currentChat.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val availableModels by viewModel.availableModels.collectAsState()
    val streamingContents by viewModel.streamingContents.collectAsState()
    val tokenStats by viewModel.tokenStats.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    var modelMenuExpanded by remember { mutableStateOf(false) }
    
    val listState = rememberLazyListState()
    
    // Auto-scroll behaviour
    LaunchedEffect(messages.size) {
        // When a message finishes (size changes), animate to bottom once
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(maxOf(0, messages.size - 1))
        }
    }

    // During streaming, keep list pinned to bottom without animation (smoother)
    LaunchedEffect(streamingContents) {
        if (streamingContents.isNotEmpty()) {
            listState.scrollToItem(maxOf(0, messages.size - 1))
        }
    }
    
    // Initialize chat
    LaunchedEffect(chatId) {
        viewModel.initializeChat(chatId, context)
    }
    
    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ChatDrawer(
                onNavigateToChat = { newChatId ->
                    scope.launch {
                        drawerState.close()
                    }
                    onNavigateToChat(newChatId)
                },
                onCreateNewChat = {
                    scope.launch {
                        drawerState.close()
                    }
                    onNavigateToChat("new")
                },
                onNavigateToSettings = {
                    scope.launch {
                        drawerState.close()
                    }
                    onNavigateToSettings()
                },
                onNavigateToModels = {
                    scope.launch {
                        drawerState.close()
                    }
                    onNavigateToModels()
                }
            )
        }
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text(currentChat?.title ?: "New Chat") },
                    navigationIcon = {
                        IconButton(onClick = {
                            scope.launch {
                                drawerState.open()
                            }
                        }) {
                            Icon(
                                imageVector = Icons.Default.Menu,
                                contentDescription = "Menu"
                            )
                        }
                    },
                    actions = {
                        // Model selector
                        Box {
                            TextButton(
                                onClick = { modelMenuExpanded = !modelMenuExpanded },
                                enabled = availableModels.isNotEmpty()
                            ) {
                                Text(
                                    text = currentChat?.modelName?.take(20) ?: "No model",
                                    maxLines = 1
                                )
                                Icon(
                                    imageVector = Icons.Default.ArrowDropDown,
                                    contentDescription = "Select model"
                                )
                            }
                            
                            DropdownMenu(
                                expanded = modelMenuExpanded,
                                onDismissRequest = { modelMenuExpanded = false }
                            ) {
                                availableModels.forEach { model ->
                                    DropdownMenuItem(
                                        text = { Text(model.name) },
                                        onClick = {
                                            viewModel.switchModel(model)
                                            modelMenuExpanded = false
                                        }
                                    )
                                }
                            }
                        }
                        
                        // Stop generation button (only show when loading)
                        if (isLoading) {
                            IconButton(onClick = { viewModel.stopGeneration() }) {
                                Icon(
                                    imageVector = Icons.Default.Stop,
                                    contentDescription = "Stop generation"
                                )
                            }
                        }
                    }
                )
            }
        ) { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
            ) {
                // Messages list
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    state = listState,
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (messages.isEmpty() && !isLoading) {
                        item {
                            WelcomeMessage(
                                currentModel = currentChat?.modelName ?: "No model selected",
                                onNavigateToModels = onNavigateToModels,
                                hasDownloadedModels = viewModel.hasDownloadedModels()
                            )
                        }
                    }
                    
                    items(messages) { message ->
                        val streamingText = streamingContents[message.id] ?: ""
                        val isFinished = streamingText.isEmpty()
                        MessageBubble(
                            message = message,
                            tokenStats = if (!message.isFromUser && isFinished && message == messages.lastOrNull { !it.isFromUser }) tokenStats else null,
                            streamingContent = streamingText
                        )
                    }
                    
                    if (isLoadingModel) {
                        item {
                            Row(
                                modifier = Modifier.padding(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(16.dp),
                                    strokeWidth = 2.dp
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = "Loading model...",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                    
                    if (isLoading && streamingContents.isEmpty() && !isLoadingModel) {
                        item {
                            TypingIndicator()
                        }
                    }
                }
                
                // Message input
                MessageInput(
                    onSendMessage = { text, attachmentUri ->
                        viewModel.sendMessage(text, attachmentUri)
                    },
                    enabled = !isLoading && !isLoadingModel && currentChat != null,
                    supportsAttachments = viewModel.currentModelSupportsVision()
                )
            }
        }
    }
}

@Composable
private fun WelcomeMessage(
    currentModel: String,
    onNavigateToModels: () -> Unit,
    hasDownloadedModels: Boolean
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                Icons.Default.SmartToy,
                contentDescription = null,
                modifier = Modifier.size(48.dp),
                tint = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Welcome to LLM Hub!",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onPrimaryContainer
            )
            Spacer(modifier = Modifier.height(4.dp))
            
            if (!hasDownloadedModels) {
                Text(
                    text = "No models downloaded",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
                Spacer(modifier = Modifier.height(8.dp))
                FilledTonalButton(onClick = onNavigateToModels) {
                    Icon(Icons.Default.GetApp, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Download a Model")
                }
            } else if (currentModel == "No model selected" || currentModel == "No model downloaded") {
                Text(
                    text = "Ready to chat! Please select a model above.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
            } else {
                Text(
                    text = "Current model: $currentModel",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Start chatting by typing a message below!",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }
        }
    }
}

@Composable
private fun TypingIndicator() {
    Row(
        modifier = Modifier.padding(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        CircularProgressIndicator(
            modifier = Modifier.size(16.dp),
            strokeWidth = 2.dp
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "AI is thinking...",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
} 