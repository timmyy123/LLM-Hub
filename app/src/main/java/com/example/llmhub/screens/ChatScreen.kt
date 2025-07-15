package com.example.llmhub.screens

import androidx.compose.foundation.gestures.detectTapGestures
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
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.llmhub.components.ChatDrawer
import com.example.llmhub.components.MessageBubble
import com.example.llmhub.components.MessageInput
import com.example.llmhub.viewmodels.ChatViewModel
import com.example.llmhub.viewmodels.ChatViewModelFactory
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    chatId: String,
    viewModelFactory: ChatViewModelFactory,
    onNavigateToSettings: () -> Unit,
    onNavigateToModels: () -> Unit,
    onNavigateToChat: (String) -> Unit
) {
    val viewModel: ChatViewModel = viewModel(
        key = "chat_$chatId",
        factory = viewModelFactory
    )
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    
    val messages by viewModel.messages.collectAsState()
    val currentChat by viewModel.currentChat.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val availableModels by viewModel.availableModels.collectAsState()
    val streamingContents by viewModel.streamingContents.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val currentlyLoadedModel by viewModel.currentlyLoadedModel.collectAsState()
    var modelMenuExpanded by remember { mutableStateOf(false) }
    
    val listState = rememberLazyListState()
    val focusManager = LocalFocusManager.current
    
    // Auto-scroll to bottom when a new message finishes
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(0)
        }
    }

    // Initialize chat - only run once per chatId or when context changes
    LaunchedEffect(chatId) {
        viewModel.initializeChat(chatId, context)
    }
    
    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ChatDrawer(
                onNavigateToChat = { newChatId ->
                    coroutineScope.launch {
                        drawerState.close()
                    }
                    onNavigateToChat(newChatId)
                },
                onCreateNewChat = {
                    coroutineScope.launch {
                        drawerState.close()
                    }
                    onNavigateToChat("new")
                },
                onNavigateToSettings = {
                    coroutineScope.launch {
                        drawerState.close()
                    }
                    onNavigateToSettings()
                },
                onNavigateToModels = {
                    coroutineScope.launch {
                        drawerState.close()
                    }
                    onNavigateToModels()
                },
                onClearAllChats = {
                    coroutineScope.launch {
                        drawerState.close()
                    }
                    viewModel.clearAllChatsAndCreateNew(context)
                    onNavigateToChat("new")
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
                            coroutineScope.launch {
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
                                Row(
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = currentlyLoadedModel?.name?.take(25) ?: "Select model",
                                        maxLines = 1
                                    )
                                    // Show vision indicator
                                    if (currentlyLoadedModel?.supportsVision == true) {
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Icon(
                                            Icons.Default.RemoveRedEye,
                                            contentDescription = "Vision model",
                                            modifier = Modifier.size(16.dp),
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                }
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
                                        text = { 
                                            Row(
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Text(model.name)
                                                
                                                // Show vision indicator
                                                if (model.supportsVision) {
                                                    Spacer(modifier = Modifier.width(4.dp))
                                                    Icon(
                                                        Icons.Default.RemoveRedEye,
                                                        contentDescription = "Vision model",
                                                        modifier = Modifier.size(14.dp),
                                                        tint = MaterialTheme.colorScheme.primary
                                                    )
                                                }
                                                
                                                if (model.name == currentlyLoadedModel?.name) {
                                                    Spacer(modifier = Modifier.width(8.dp))
                                                    Icon(
                                                        imageVector = Icons.Default.Check,
                                                        contentDescription = "Currently loaded",
                                                        tint = MaterialTheme.colorScheme.primary,
                                                        modifier = Modifier.size(16.dp)
                                                    )
                                                }
                                            }
                                        },
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
                    .pointerInput(Unit) {
                        // Dismiss keyboard when tapping anywhere in the chat window
                        detectTapGestures(onTap = { focusManager.clearFocus() })
                    }
                    // REMOVED imePadding() from here
            ) {
                // Messages list
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    state = listState,
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    reverseLayout = true
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
                    
                    items(messages.reversed(), key = { it.id }) { message ->
                        val streamingText = streamingContents[message.id] ?: ""
                        MessageBubble(
                            message = message,
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
                Box(modifier = Modifier.imePadding()) {
                MessageInput(
                    onSendMessage = { text, attachmentUri ->
                        viewModel.sendMessage(context, text, attachmentUri)
                    },
                    enabled = !isLoading && !isLoadingModel && currentChat != null,
                    supportsAttachments = viewModel.currentModelSupportsVision()
                )
                }
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