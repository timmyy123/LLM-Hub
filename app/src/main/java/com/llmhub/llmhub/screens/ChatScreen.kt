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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.components.ChatDrawer
import com.llmhub.llmhub.components.MessageBubble
import com.llmhub.llmhub.components.MessageInput
import com.llmhub.llmhub.components.ChatSettingsSheet
import com.llmhub.llmhub.ui.components.ModernCard
import com.llmhub.llmhub.ui.components.StatusChip
import com.llmhub.llmhub.ui.components.SectionHeader
import com.llmhub.llmhub.viewmodels.ChatViewModel
import com.llmhub.llmhub.viewmodels.ChatViewModelFactory
import kotlinx.coroutines.launch
import android.util.Log

@Composable
fun getLocalizedModelName(model: LLMModel): String {
    val baseName = model.name.substringBefore(" (")
    val capabilities = model.name.substringAfter("(").substringBefore(")")
    
    if (!model.name.contains("(")) {
        return model.name
    }
    
    val localizedCapabilities = when {
        capabilities.equals("Vision+Audio+Text", ignoreCase = true) -> {
            stringResource(R.string.vision_audio_text)
        }
        capabilities.equals("Vision+Text", ignoreCase = true) -> {
            stringResource(R.string.vision_text)
        }
        capabilities.equals("Audio+Text", ignoreCase = true) -> {
            stringResource(R.string.audio_text)
        }
        else -> capabilities
    }
    
    return "$baseName ($localizedCapabilities)"
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    chatId: String,
    creatorId: String? = null,
    viewModelFactory: ChatViewModelFactory,
    onNavigateToSettings: () -> Unit,
    onNavigateToModels: () -> Unit,
    onNavigateToChat: (String) -> Unit,
    onNavigateToCreatorChat: (String) -> Unit,
    onNavigateBack: () -> Unit,
    drawerState: DrawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
) {
    val viewModel: ChatViewModel = viewModel(
        key = "chat_$chatId",
        factory = viewModelFactory
    )
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    val messages by viewModel.messages.collectAsState()
    val currentChat by viewModel.currentChat.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val availableModels by viewModel.availableModels.collectAsState()
    val streamingContents by viewModel.streamingContents.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val currentlyLoadedModel by viewModel.currentlyLoadedModel.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    
    // Creator state for header
    val currentCreator by viewModel.currentCreator.collectAsState()
    
    // RAG state
    val isRagReady by viewModel.isRagReady.collectAsState()
    val ragStatus by viewModel.ragStatus.collectAsState()
    val documentCount by viewModel.documentCount.collectAsState()
    
    // Embedding state
    val isEmbeddingEnabled by viewModel.isEmbeddingEnabled.collectAsState()
    
    // TTS Service - use ViewModel's TTS service (same instance for auto-readout and manual)
    val ttsService = viewModel.ttsService
    val isTtsSpeaking by ttsService.isSpeaking.collectAsState()
    // Track TTS message from ViewModel (for auto-readout) and local manual TTS
    val viewModelTtsMessageId by viewModel.currentTtsMessageId.collectAsState()
    var manualTtsMessageId by remember { mutableStateOf<String?>(null) }
    // Combined: use ViewModel's ID if set (auto-readout), otherwise use manual ID
    val currentTtsMessageId = viewModelTtsMessageId ?: manualTtsMessageId
    
    // Settings bottom sheet state
    var showSettingsSheet by remember { mutableStateOf(false) }
    
    val listState = rememberLazyListState()
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current
    
    // Edit-last-prompt state
    var isEditingLastPrompt by remember { mutableStateOf(false) }
    var editedPromptText by remember { mutableStateOf("") }
    val latestUserMessage = messages.lastOrNull { it.isFromUser }

    // Auto-scroll to bottom when a new message finishes
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(0)
        }
    }

    // Initialize chat - only run once per chatId or when context changes
    // Initialize chat - only run once per chatId/creatorId or when context changes
    LaunchedEffect(chatId, creatorId) {
        viewModel.initializeChat(chatId, context, creatorId)
    }
    
    // Sync model state immediately to show icons
    LaunchedEffect(Unit) {
        // Force immediate sync
        viewModel.syncCurrentlyLoadedModel()
    }
    
    // Also sync when the currently loaded model changes
    LaunchedEffect(viewModel.currentlyLoadedModel) {
        viewModel.syncCurrentlyLoadedModel()
    }
    
    // Unload model when leaving the chat screen to free memory. Selection is kept so
    // when the user returns, the same model can be loaded again on next send.
    // Unload when leaving Chat is handled in LlmHubNavigation (route observer); no unload on dispose so new chat doesn't reload.

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
                onNavigateToCreatorChat = { creatorId ->
                    coroutineScope.launch {
                        drawerState.close()
                    }
                    onNavigateToCreatorChat(creatorId)
                },
                onNavigateBack = {
                    coroutineScope.launch {
                        drawerState.close()
                    }
                    onNavigateBack()
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
                    // Avoid a fixed title height so the title / chips stay vertically centered
                    // across portrait/landscape and larger tablet screens.
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.Center
                    ) {
                        val headerTitle = when {
                            currentChat != null -> currentChat!!.title
                            currentCreator != null -> "${currentCreator!!.icon} ${currentCreator!!.name}"
                            else -> stringResource(R.string.chat) // "New Chat" or similar
                        }
                        
                        Text(
                            text = headerTitle,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                },
                    navigationIcon = {
                        IconButton(onClick = {
                            coroutineScope.launch { drawerState.open() }
                        }) {
                            Icon(
                                imageVector = Icons.Default.Menu,
                                contentDescription = "Menu"
                            )
                        }
                    },
                    actions = {
                        // Settings button - opens bottom sheet for model selection and config
                        IconButton(
                            onClick = { showSettingsSheet = true }
                        ) {
                            Icon(
                                imageVector = Icons.Default.Tune,
                                contentDescription = stringResource(R.string.feature_settings_title),
                                tint = MaterialTheme.colorScheme.primary
                            )
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
                            Box(
                                modifier = Modifier.fillMaxWidth(),
                                contentAlignment = Alignment.Center
                            ) {
                                WelcomeMessage(
                                    currentModel = currentChat?.modelName ?: stringResource(R.string.no_model_selected),
                                    onNavigateToModels = onNavigateToModels,
                                    hasDownloadedModels = viewModel.hasDownloadedModels()
                                )
                            }
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
                        val canEditThisUser = message.isFromUser && message.id == latestUserMessage?.id && !isLoading && !isLoadingModel
                        MessageBubble(
                            message = message,
                            streamingContent = streamingText,
                            onRegenerateResponse = if (canRegenerate) {
                                { viewModel.regenerateResponse(context, message.id) }
                            } else null,
                            onEditUserMessage = if (canEditThisUser) {
                                {
                                    isEditingLastPrompt = true
                                    editedPromptText = message.content
                                    // Do not clear focus; MessageInput will request focus and open keyboard
                                }
                            } else null,
                            onTtsSpeak = if (!message.isFromUser && message.content.isNotBlank()) {
                                { text ->
                                    // Manual TTS button - use local manual ID and ask ViewModel
                                    // to enable manual TTS streaming for this message so that
                                    // future generated tokens are also routed to TTS.
                                    manualTtsMessageId = message.id
                                    val appLocale = com.llmhub.llmhub.utils.LocaleHelper.getCurrentLocale(context)
                                    viewModel.enableManualTtsForMessage(message.id, text)
                                }
                            } else null,
                            onTtsStop = if (!message.isFromUser && message.content.isNotBlank()) {
                                {
                                    ttsService.stop()
                                    // Clear both manual and auto-readout IDs
                                    manualTtsMessageId = null
                                    viewModel.clearCurrentTtsMessage()
                                }
                            } else null,
                            isTtsSpeaking = isTtsSpeaking && currentTtsMessageId == message.id
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
                    } else null,
                    isEditing = isEditingLastPrompt,
                    editText = editedPromptText,
                    onEditTextChange = { editedPromptText = it },
                    onConfirmEdit = {
                        val text = editedPromptText.trim()
                        if (text.isNotEmpty()) {
                            isEditingLastPrompt = false
                            editedPromptText = ""
                            // Dispatch after clearing local UI state so input doesn't linger
                            viewModel.editLastUserMessageAndResend(context, text)
                        }
                    },
                    onCancelEdit = {
                        isEditingLastPrompt = false
                        editedPromptText = ""
                    }
                )
                }
            }
        }
    }
    
    // Settings Bottom Sheet for model selection and configuration
    if (showSettingsSheet) {
        ChatSettingsSheet(
            availableModels = availableModels,
            initialSelectedModel = selectedModel,
            initialSelectedBackend = selectedBackend,
            currentlyLoadedModel = currentlyLoadedModel,
            isLoadingModel = isLoadingModel,
            onModelSelected = { model ->
                viewModel.selectModel(model)
            },
            onBackendSelected = { backend ->
                viewModel.selectBackend(backend)
            },
            onLoadModel = { model, maxTokens, topK, topP, temperature, backend, disableVision, disableAudio ->
                Log.d("ChatScreen", "Model configs confirmed: maxTokens=$maxTokens topK=$topK topP=$topP temperature=$temperature backend=$backend disableVision=$disableVision disableAudio=$disableAudio for model ${model.name}")
                
                // Push generation parameters to inference service via ViewModel
                viewModel.setGenerationParameters(maxTokens, topK, topP, temperature)
                
                if (backend != null) {
                    viewModel.switchModelWithBackend(model, backend, disableVision, disableAudio)
                } else {
                    viewModel.switchModel(model)
                }
                
                showSettingsSheet = false
            },
            onUnloadModel = {
                viewModel.unloadModel()
            },
            onDismiss = {
                showSettingsSheet = false
            }
        )
    }
}

@Composable
private fun WelcomeMessage(
    currentModel: String,
    onNavigateToModels: () -> Unit,
    hasDownloadedModels: Boolean
) {
    ModernCard(
        modifier = Modifier
            .widthIn(max = 640.dp) // limit width on large screens/tablets
            .wrapContentWidth(Alignment.CenterHorizontally) // center the card horizontally
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
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier.clip(MaterialTheme.shapes.medium)
                ) {
                    Icon(
                        painter = painterResource(id = R.mipmap.ic_launcher_foreground),
                        contentDescription = null,
                        modifier = Modifier
                            .size(54.dp)
                            .scale(2.0f),
                        tint = Color.Unspecified
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
                    // Ensure the chip is horizontally centered even in landscape/tablet widths
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                        StatusChip(
                            text = currentModel,
                            icon = Icons.Default.Link,
                            containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                            contentColor = MaterialTheme.colorScheme.onTertiaryContainer
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = stringResource(R.string.start_chatting),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.fillMaxWidth(),
                        textAlign = TextAlign.Center
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
    // Smooth breathing effect (scale only). No transparency or icon rotation.
    val infiniteTransition = rememberInfiniteTransition(label = "ModelLoadingAnimations")

    val scale by infiniteTransition.animateFloat(
        initialValue = 0.96f,
        targetValue = 1.04f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1600, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "ModelLoadingBreathScale"
    )
    
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 12.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Surface(
            modifier = Modifier
                .widthIn(max = 600.dp)
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
                // Model icon with background and rotation animation
                Surface(
                    modifier = Modifier.size(48.dp),
                    shape = MaterialTheme.shapes.medium,
                    color = MaterialTheme.colorScheme.primary,
                    tonalElevation = 8.dp
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(MaterialTheme.shapes.medium),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            painter = painterResource(R.mipmap.ic_launcher_foreground),
                            contentDescription = null,
                            modifier = Modifier
                                .size(52.dp)
                                .scale(1.6f),
                            tint = Color.Unspecified
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