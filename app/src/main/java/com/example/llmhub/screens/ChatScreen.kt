package com.llmhub.llmhub.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.components.ChatDrawer
import com.llmhub.llmhub.components.MessageBubble
import com.llmhub.llmhub.components.MessageInput
import com.llmhub.llmhub.ui.components.ModernCard
import com.llmhub.llmhub.ui.components.StatusChip
import com.llmhub.llmhub.ui.components.SectionHeader
import com.llmhub.llmhub.viewmodels.ChatViewModel
import com.llmhub.llmhub.viewmodels.ChatViewModelFactory
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
    val selectedModel by viewModel.selectedModel.collectAsState()
    var modelMenuExpanded by remember { mutableStateOf(false) }
    
    val listState = rememberLazyListState()
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current
    
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
                    title = { 
                        Column(
                            modifier = Modifier.height(72.dp),
                            verticalArrangement = Arrangement.Center
                        ) {
                            Text(
                                text = currentChat?.title ?: "New Chat",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                            (selectedModel ?: currentlyLoadedModel)?.let { model ->
                                Row(
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = model.name,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                    if (model.supportsVision) {
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Icon(
                                            Icons.Default.RemoveRedEye,
                                            contentDescription = "Vision model",
                                            modifier = Modifier.size(12.dp),
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                }
                            }
                        }
                    },
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
                            IconButton(
                                onClick = { modelMenuExpanded = !modelMenuExpanded },
                                enabled = availableModels.isNotEmpty()
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Tune,
                                    contentDescription = "Select model",
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                            
                            DropdownMenu(
                                expanded = modelMenuExpanded,
                                onDismissRequest = { modelMenuExpanded = false },
                                modifier = Modifier
                                    .widthIn(min = 250.dp, max = 320.dp)
                                    .background(
                                        color = MaterialTheme.colorScheme.surfaceContainer,
                                        shape = MaterialTheme.shapes.medium
                                    )
                            ) {
                                // Add unload model option if a model is currently loaded
                                if (currentlyLoadedModel != null) {
                                    DropdownMenuItem(
                                        text = { 
                                            Row(
                                                verticalAlignment = Alignment.CenterVertically
                                            ) {
                                                Icon(
                                                    imageVector = Icons.Default.PowerOff,
                                                    contentDescription = "Unload model",
                                                    tint = MaterialTheme.colorScheme.error,
                                                    modifier = Modifier.size(18.dp)
                                                )
                                                Spacer(modifier = Modifier.width(12.dp))
                                                Text(
                                                    text = "Unload Model",
                                                    style = MaterialTheme.typography.bodyMedium,
                                                    fontWeight = FontWeight.Medium,
                                                    color = MaterialTheme.colorScheme.error
                                                )
                                            }
                                        },
                                        onClick = {
                                            viewModel.unloadModel()
                                            modelMenuExpanded = false
                                        }
                                    )
                                    
                                    HorizontalDivider(
                                        modifier = Modifier.padding(vertical = 4.dp),
                                        color = MaterialTheme.colorScheme.outlineVariant
                                    )
                                }
                                
                                availableModels.forEach { model ->
                                    DropdownMenuItem(
                                        text = { 
                                            Column(
                                                modifier = Modifier.fillMaxWidth()
                                            ) {
                                                Row(
                                                    verticalAlignment = Alignment.CenterVertically,
                                                    horizontalArrangement = Arrangement.SpaceBetween,
                                                    modifier = Modifier.fillMaxWidth()
                                                ) {
                                                    Text(
                                                        text = model.name,
                                                        style = MaterialTheme.typography.bodyMedium,
                                                        fontWeight = FontWeight.Medium,
                                                        color = MaterialTheme.colorScheme.onSurface,
                                                        modifier = Modifier.weight(1f)
                                                    )
                                                    
                                                    // Show vision indicator with better styling
                                                    if (model.supportsVision) {
                                                        Surface(
                                                            shape = RoundedCornerShape(12.dp),
                                                            color = MaterialTheme.colorScheme.tertiary,
                                                            modifier = Modifier.padding(start = 8.dp)
                                                        ) {
                                                            Row(
                                                                verticalAlignment = Alignment.CenterVertically,
                                                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                                            ) {
                                                                Icon(
                                                                    Icons.Default.RemoveRedEye,
                                                                    contentDescription = "Vision enabled",
                                                                    modifier = Modifier.size(14.dp),
                                                                    tint = MaterialTheme.colorScheme.onTertiary
                                                                )
                                                                Spacer(modifier = Modifier.width(4.dp))
                                                                Text(
                                                                    text = "Vision",
                                                                    style = MaterialTheme.typography.labelSmall,
                                                                    fontWeight = FontWeight.SemiBold,
                                                                    color = MaterialTheme.colorScheme.onTertiary
                                                                )
                                                            }
                                                        }
                                                    }
                                                }
                                                
                                                // Add model details subtitle
                                                if (model.contextWindowSize > 0) {
                                                    Text(
                                                        text = "${model.contextWindowSize / 1024}k context • ${if (model.supportsVision) "Multimodal" else "Text only"}",
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                        modifier = Modifier.padding(top = 2.dp)
                                                    )
                                                }
                                            }
                                        },
                                        trailingIcon = if (model.name == currentlyLoadedModel?.name) {
                                            {
                                                Icon(
                                                    imageVector = Icons.Default.Check,
                                                    contentDescription = "Currently loaded",
                                                    tint = MaterialTheme.colorScheme.primary,
                                                    modifier = Modifier.size(18.dp)
                                                )
                                            }
                                        } else null,
                                        onClick = {
                                            viewModel.switchModel(model)
                                            modelMenuExpanded = false
                                        }
                                    )
                                }
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
                    contentPadding = PaddingValues(8.dp),
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
                    
                    // Show model loading indicator after the latest message
                    if (isLoadingModel) {
                        item {
                            val name = (selectedModel ?: currentlyLoadedModel)?.name
                                ?: currentChat?.modelName ?: "AI Model"
                            ModelLoadingIndicator(modelName = name)
                        }
                    }
                    
                    // Show typing indicator for regular generation
                    if (isLoading && streamingContents.isEmpty() && !isLoadingModel) {
                        item {
                            TypingIndicator()
                        }
                    }
                    
                    items(messages.reversed(), key = { it.id }) { message ->
                        val streamingText = streamingContents[message.id] ?: ""
                        val isLatestAiMessage = !message.isFromUser && 
                                                message.content != "…" && 
                                                message == messages.lastOrNull { !it.isFromUser }
                        val canRegenerate = isLatestAiMessage && !isLoading && !isLoadingModel
                        MessageBubble(
                            message = message,
                            streamingContent = streamingText,
                            onRegenerateResponse = if (canRegenerate) {
                                { viewModel.regenerateResponse(context, message.id) }
                            } else null
                        )
                    }
                }

                // Message input
                Box(modifier = Modifier.imePadding()) {
                MessageInput(
                    onSendMessage = { text, attachmentUri ->
                        // Triple-layer keyboard dismissal for maximum reliability
                        keyboardController?.hide()
                        focusManager.clearFocus()
                        viewModel.sendMessage(context, text, attachmentUri)
                    },
                    enabled = !isLoading && !isLoadingModel && currentChat != null,
                    supportsAttachments = viewModel.currentModelSupportsVision(),
                    isLoading = isLoading,
                    onCancelGeneration = if (isLoading) {
                        { viewModel.stopGeneration() }
                    } else null
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
    ModernCard(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Surface(
                modifier = Modifier.size(64.dp),
                shape = MaterialTheme.shapes.medium,
                color = MaterialTheme.colorScheme.primaryContainer,
                tonalElevation = 8.dp
            ) {
                Box(contentAlignment = Alignment.Center) {
                    Icon(
                        Icons.Default.SmartToy,
                        contentDescription = null,
                        modifier = Modifier.size(32.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = "Welcome to LLM Hub!",
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            if (!hasDownloadedModels) {
                Text(
                    text = "No models downloaded yet",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(16.dp))
                FilledTonalButton(
                    onClick = onNavigateToModels,
                    modifier = Modifier.height(48.dp)
                ) {
                    Icon(
                        Icons.Default.GetApp, 
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        "Download a Model",
                        style = MaterialTheme.typography.labelLarge
                    )
                }
            } else if (currentModel == "No model selected" || currentModel == "No model downloaded") {
                Text(
                    text = "Ready to chat! Please select a model above.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else {
                StatusChip(
                    text = currentModel,
                    icon = Icons.Default.Psychology,
                    containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                    contentColor = MaterialTheme.colorScheme.onTertiaryContainer
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Start chatting by typing a message below!",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun TypingIndicator() {
    Surface(
        modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp),
        shape = MaterialTheme.shapes.large,
        color = MaterialTheme.colorScheme.surfaceVariant,
        tonalElevation = 2.dp
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(18.dp),
                strokeWidth = 2.dp,
                color = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = "AI is thinking...",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun ModelLoadingIndicator(modelName: String) {
    // Pulsing animation for the loading indicator
    val infiniteTransition = rememberInfiniteTransition(label = "ModelLoadingPulse")
    val scale by infiniteTransition.animateFloat(
        initialValue = 0.95f,
        targetValue = 1.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "ModelLoadingScale"
    )
    
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .graphicsLayer {
                    scaleX = scale
                    scaleY = scale
                },
            shape = MaterialTheme.shapes.large,
            color = MaterialTheme.colorScheme.primaryContainer,
            tonalElevation = 4.dp
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Model icon with background
                Surface(
                    modifier = Modifier.size(48.dp),
                    shape = MaterialTheme.shapes.medium,
                    color = MaterialTheme.colorScheme.primary,
                    tonalElevation = 8.dp
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            Icons.Default.SmartToy,
                            contentDescription = null,
                            modifier = Modifier.size(24.dp),
                            tint = MaterialTheme.colorScheme.onPrimary
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Loading text with model name
                Text(
                    text = "Loading ${modelName.take(30)}${if (modelName.length > 30) "..." else ""}",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
                
                Spacer(modifier = Modifier.height(4.dp))
                
                Text(
                    text = "Please wait while the model initializes...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Animated progress indicator
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.5.dp,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(
                        text = "Initializing neural network...",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                    )
                }
            }
        }
    }
}