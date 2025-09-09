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
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.components.ChatDrawer
import com.llmhub.llmhub.components.MessageBubble
import com.llmhub.llmhub.components.MessageInput
import com.llmhub.llmhub.components.BackendSelectionDialog
import com.llmhub.llmhub.components.CompactRagIndicator
import com.llmhub.llmhub.ui.components.ModernCard
import com.llmhub.llmhub.ui.components.StatusChip
import com.llmhub.llmhub.ui.components.SectionHeader
import com.llmhub.llmhub.viewmodels.ChatViewModel
import com.llmhub.llmhub.viewmodels.ChatViewModelFactory
import com.google.mediapipe.tasks.genai.llminference.LlmInference
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
    
    // RAG state
    val isRagReady by viewModel.isRagReady.collectAsState()
    val ragStatus by viewModel.ragStatus.collectAsState()
    val documentCount by viewModel.documentCount.collectAsState()
    
    var modelMenuExpanded by remember { mutableStateOf(false) }
    
    // Backend selection state
    var showBackendDialog by remember { mutableStateOf(false) }
    var pendingModelForBackendSelection by remember { mutableStateOf<LLMModel?>(null) }
    
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
                                text = currentChat?.title ?: stringResource(R.string.chat),
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
                                            contentDescription = stringResource(R.string.vision_enabled),
                                            modifier = Modifier.size(12.dp),
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                    // Show compact RAG indicator
                                    Spacer(modifier = Modifier.width(4.dp))
                                    CompactRagIndicator(
                                        hasDocuments = documentCount > 0,
                                        documentCount = documentCount
                                    )
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
                                    contentDescription = stringResource(R.string.select_model),
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
                                                    contentDescription = stringResource(R.string.unload_model),
                                                    tint = MaterialTheme.colorScheme.error,
                                                    modifier = Modifier.size(18.dp)
                                                )
                                                Spacer(modifier = Modifier.width(12.dp))
                                                Text(
                                                    text = stringResource(R.string.unload_model),
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
                                                    // Vision support badge
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
                                                                    contentDescription = stringResource(R.string.vision_enabled),
                                                                    modifier = Modifier.size(14.dp),
                                                                    tint = MaterialTheme.colorScheme.onTertiary
                                                                )
                                                                Spacer(modifier = Modifier.width(4.dp))
                                                                Text(
                                                                    text = stringResource(R.string.vision),
                                                                    style = MaterialTheme.typography.labelSmall,
                                                                    fontWeight = FontWeight.SemiBold,
                                                                    color = MaterialTheme.colorScheme.onTertiary
                                                                )
                                                            }
                                                        }
                                                    }
                                                    
                                                    // Audio support badge
                                                    if (model.supportsAudio) {
                                                        Surface(
                                                            shape = RoundedCornerShape(12.dp),
                                                            color = MaterialTheme.colorScheme.secondary,
                                                            modifier = Modifier.padding(start = 4.dp)
                                                        ) {
                                                            Row(
                                                                verticalAlignment = Alignment.CenterVertically,
                                                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                                            ) {
                                                                Icon(
                                                                    Icons.Default.Mic,
                                                                    contentDescription = stringResource(R.string.audio_enabled),
                                                                    modifier = Modifier.size(14.dp),
                                                                    tint = MaterialTheme.colorScheme.onSecondary
                                                                )
                                                                Spacer(modifier = Modifier.width(4.dp))
                                                                Text(
                                                                    text = stringResource(R.string.audio),
                                                                    style = MaterialTheme.typography.labelSmall,
                                                                    fontWeight = FontWeight.SemiBold,
                                                                    color = MaterialTheme.colorScheme.onSecondary
                                                                )
                                                            }
                                                        }
                                                    }
                                                }
                                                
                                                // Add model details subtitle
                                                if (model.contextWindowSize > 0) {
                                                    Text(
                                                        text = stringResource(
                                                            R.string.context_multimodal_format,
                                                            model.contextWindowSize / 1024,
                                                            when {
                                                                model.supportsVision && model.supportsAudio -> stringResource(R.string.vision_audio_text)
                                                                model.supportsVision -> stringResource(R.string.multimodal)
                                                                model.supportsAudio -> stringResource(R.string.audio_text)
                                                                else -> stringResource(R.string.text_only)
                                                            }
                                                        ),
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
                                                    contentDescription = stringResource(R.string.currently_loaded),
                                                    tint = MaterialTheme.colorScheme.primary,
                                                    modifier = Modifier.size(18.dp)
                                                )
                                            }
                                        } else null,
                                        onClick = {
                                            if (viewModel.isGemmaModel(model)) {
                                                pendingModelForBackendSelection = model
                                                showBackendDialog = true
                                                modelMenuExpanded = false
                                            } else {
                                                viewModel.switchModel(model)
                                                modelMenuExpanded = false
                                            }
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
                                currentModel = currentChat?.modelName ?: stringResource(R.string.no_model_selected),
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
                    onSendMessage = { text, attachmentUri, audioData ->
                        // Triple-layer keyboard dismissal for maximum reliability
                        keyboardController?.hide()
                        focusManager.clearFocus()
                        viewModel.sendMessage(context, text, attachmentUri, audioData)
                    },
                    enabled = !isLoading && !isLoadingModel && currentChat != null,
                    supportsAttachments = true, // Enable attachments for all models
                    supportsVision = viewModel.currentModelSupportsVision(), // Only show images for vision models
                    supportsAudio = viewModel.currentModelSupportsAudio(), // Only show audio for audio models
                    isLoading = isLoading,
                    onCancelGeneration = if (isLoading) {
                        { viewModel.stopGeneration() }
                    } else null
                )
                }
            }
        }
    }
    
    // Backend Selection Dialog
    if (showBackendDialog) {
        pendingModelForBackendSelection?.let { model ->
            BackendSelectionDialog(
                modelName = model.name,
                onBackendSelected = { backend ->
                    viewModel.switchModelWithBackend(model, backend)
                },
                onDismiss = {
                    showBackendDialog = false
                    pendingModelForBackendSelection = null
                }
            )
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
                text = stringResource(R.string.welcome_to_llm_hub),
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            if (!hasDownloadedModels) {
                Text(
                    text = stringResource(R.string.no_models_downloaded),
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
                        stringResource(R.string.download_a_model),
                        style = MaterialTheme.typography.labelLarge
                    )
                }
            } else if (currentModel == stringResource(R.string.no_model_selected) || currentModel == stringResource(R.string.no_model_downloaded)) {
                Text(
                    text = stringResource(R.string.ready_to_chat),
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
                    text = stringResource(R.string.start_chatting),
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
                text = stringResource(R.string.ai_thinking),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun ModelLoadingIndicator(modelName: String) {
    // Multiple smooth animations for the loading indicator
    val infiniteTransition = rememberInfiniteTransition(label = "ModelLoadingAnimations")
    
    // Gentle pulsing scale animation
    val scale by infiniteTransition.animateFloat(
        initialValue = 0.98f,
        targetValue = 1.02f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = EaseInOutCubic),
            repeatMode = RepeatMode.Reverse
        ),
        label = "ModelLoadingScale"
    )
    
    // Subtle glow effect through alpha animation
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.8f,
        animationSpec = infiniteRepeatable(
            animation = tween(1800, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "ModelLoadingGlow"
    )
    
    // Gentle rotation for the icon
    val iconRotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "ModelLoadingIconRotation"
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
                    alpha = glowAlpha
                },
            shape = MaterialTheme.shapes.large,
            color = MaterialTheme.colorScheme.primaryContainer,
            tonalElevation = 4.dp
        ) {
            Column(
                modifier = Modifier.padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Model icon with background and rotation animation
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
                            modifier = Modifier
                                .size(24.dp)
                                .graphicsLayer {
                                    rotationZ = iconRotation
                                },
                            tint = MaterialTheme.colorScheme.onPrimary
                        )
                    }
                }
                
                Spacer(modifier = Modifier.height(12.dp))
                
                // Loading text with model name
                Text(
                    text = stringResource(R.string.loading_model_format, modelName.take(30) + if (modelName.length > 30) "..." else ""),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer
                )
                
                Spacer(modifier = Modifier.height(4.dp))
                
                Text(
                    text = stringResource(R.string.please_wait_model_initialize),
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
                        text = stringResource(R.string.initializing_neural_network),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                    )
                }
            }
        }
    }
}