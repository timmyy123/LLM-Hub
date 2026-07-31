package com.llmhub.llmhub.screens

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.R
import com.llmhub.llmhub.agent.AgentMessage
import com.llmhub.llmhub.agent.AgentViewModel
import com.llmhub.llmhub.agent.VoiceMode
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.components.FeatureModelSettingsSheet
import com.llmhub.llmhub.components.MessageInput
import com.llmhub.llmhub.components.ThinkingAwareResultContent
import com.llmhub.llmhub.components.SelectableMarkdownText
import com.llmhub.llmhub.components.MarkdownTableView
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AgentScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    viewModel: AgentViewModel = viewModel()
) {
    val context = LocalContext.current
    val messages by viewModel.messages.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val isGemmaAudioEnabled by viewModel.isGemmaAudioEnabled.collectAsState()
    var showSettingsSheet by remember { mutableStateOf(false) }
    // LiteRT-LM models only for AI Agent feature
    val availableModelsState = produceState<List<LLMModel>>(initialValue = emptyList(), context) {
        value = ModelAvailabilityProvider.loadAvailableModels(context).filter { it.modelFormat == "litertlm" }
    }
    val availableModels = availableModelsState.value
    val loadingModelName by viewModel.loadingModelName.collectAsState()
    val activeModelName by viewModel.activeModelName.collectAsState()

    val listState = rememberLazyListState()

    val lastMessageContent = remember(messages) {
        val last = messages.lastOrNull()
        when (last) {
            is AgentMessage.Text -> last.text
            is AgentMessage.ToolCall -> "${last.status}_${last.args}_${last.result}"
            else -> last?.id ?: ""
        }
    }

    LaunchedEffect(messages.size, lastMessageContent) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    val contactsPermissionLauncher = androidx.activity.compose.rememberLauncherForActivityResult(
        contract = androidx.activity.result.contract.ActivityResultContracts.RequestPermission()
    ) { _ -> }

    LaunchedEffect(availableModels) {
        Configuration.getInstance().userAgentValue = context.packageName
        viewModel.initializeWelcomeMessage(context, availableModels.isNotEmpty())
        if (androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.READ_CONTACTS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            contactsPermissionLauncher.launch(android.Manifest.permission.READ_CONTACTS)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        Text(
                            text = stringResource(R.string.agent_title),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        if (loadingModelName != null) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.padding(top = 2.dp)
                            ) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(12.dp),
                                    strokeWidth = 1.5.dp,
                                    color = MaterialTheme.colorScheme.primary
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = stringResource(R.string.loading_model_format, loadingModelName!!),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.primary,
                                    fontWeight = FontWeight.Medium,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        } else {
                            Text(
                                text = activeModelName ?: stringResource(R.string.agent_subtitle),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = "Settings")
                    }
                }
            )
        }
    ) { paddingValues ->

        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {

                // Messages list or Empty State View
                if (messages.isEmpty()) {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(
                            modifier = Modifier
                                .widthIn(max = 640.dp)
                                .padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = stringResource(R.string.welcome_to_llm_hub),
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSurface,
                                textAlign = TextAlign.Center
                            )

                            Spacer(modifier = Modifier.height(12.dp))

                            Text(
                                text = stringResource(R.string.agent_no_model_android),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                textAlign = TextAlign.Center
                            )

                            Spacer(modifier = Modifier.height(20.dp))

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
                        }
                    }
                } else {
                    val isStreamingAIResponse = remember(messages) {
                        val last = messages.lastOrNull()
                        last is AgentMessage.Text && last.sender == AgentMessage.Sender.AGENT && last.text.isNotEmpty()
                    }
                    val hasActiveToolCard = remember(messages) {
                        messages.any { it is AgentMessage.ToolCall && (it.status == AgentMessage.ToolCall.Status.PENDING_APPROVAL || it.status == AgentMessage.ToolCall.Status.RUNNING) }
                    }

                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .weight(1f)
                            .padding(horizontal = 16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                        contentPadding = PaddingValues(top = 16.dp, bottom = 120.dp)
                    ) {
                        itemsIndexed(messages, key = { index, msg -> "${msg.id}_$index" }) { _, msg ->
                            when (msg) {
                                is AgentMessage.Text -> AgentTextMessageBubble(msg)
                                is AgentMessage.ToolCall -> AgentToolCallBubble(
                                    msg = msg,
                                    onExecuteCommand = { callId, cmd -> viewModel.approveAndExecuteTermuxCommand(callId, cmd) },
                                    onCancelCommand = { callId -> viewModel.cancelTermuxCommand(callId) }
                                )
                                is AgentMessage.MapLocation -> AgentMapCardBubble(msg, context)
                                is AgentMessage.Audio -> com.llmhub.llmhub.components.AudioMessageCard(
                                    audioPath = msg.audioPath,
                                    fileName = "Voice message",
                                    fileSize = runCatching { java.io.File(msg.audioPath).length() }.getOrNull(),
                                    isFromUser = msg.sender == AgentMessage.Sender.USER
                                )
                            }
                        }

                        if (isGenerating && !isStreamingAIResponse && !hasActiveToolCard) {
                            item {
                                AgentProcessingBubble()
                            }
                        }
                    }
                }

                val isWebSearchEnabled by viewModel.isWebSearchEnabled.collectAsState()

                val isGemmaAudioEnabled by viewModel.isGemmaAudioEnabled.collectAsState()
                val loadedModel = availableModels.find { it.name == activeModelName }
                val modelSupportsAudio = loadedModel?.let { model ->
                    model.supportsAudio || model.name.contains("Gemma-4", ignoreCase = true) || model.name.contains("Gemma 4", ignoreCase = true)
                } ?: true
                val isAudioSupportedInAgent = if (loadedModel != null) {
                    modelSupportsAudio && isGemmaAudioEnabled
                } else {
                    isGemmaAudioEnabled
                }

                // Input bar using standard AI Chat MessageInput component
                Box(modifier = Modifier.imePadding()) {
                    MessageInput(
                        onSendMessage = { text, _, audioData ->
                            if (text.isNotBlank()) {
                                viewModel.sendMessage(text)
                            } else if (audioData != null) {
                                viewModel.sendAudioMessage(audioData, context)
                            }
                        },
                        enabled = true,
                        supportsAttachments = true,
                        supportsVision = false,
                        supportsAudio = isAudioSupportedInAgent,
                        isLoading = isGenerating,
                        onCancelGeneration = if (isGenerating) { { viewModel.stopGeneration() } } else null,
                        isWebSearchEnabled = isWebSearchEnabled,
                        onToggleWebSearch = { viewModel.toggleWebSearch() }
                    )
                }
            }
        }

        // Settings Bottom Sheet (Model Selection + Voice Mode Options + Termux Commands)
        if (showSettingsSheet) {
            val agentPrefs = remember(context) { context.getSharedPreferences("agent_prefs", Context.MODE_PRIVATE) }
            var selectedModelName by remember { mutableStateOf(agentPrefs.getString("selected_model_name", "") ?: "") }
            var selectedBackendName by remember { mutableStateOf(agentPrefs.getString("selected_backend", null)) }
            var selectedNpuDeviceId by remember { mutableStateOf(agentPrefs.getString("selected_npu_device_id", null)) }
            var selectedMaxTokens by remember { mutableStateOf(agentPrefs.getInt("selected_max_tokens", 2048)) }

            val selectedModel = remember(availableModels, selectedModelName) {
                availableModels.find { it.name == selectedModelName } ?: availableModels.firstOrNull()
            }
            val selectedBackend = remember(selectedBackendName) {
                when (selectedBackendName) {
                    "GPU" -> LlmInference.Backend.GPU
                    "CPU" -> LlmInference.Backend.CPU
                    else -> null
                }
            }

            val activeModelName by viewModel.activeModelName.collectAsState()
            val currentlyLoadedModel = remember(availableModels, activeModelName) {
                availableModels.find { it.name == activeModelName }
            }

            ModalBottomSheet(
                onDismissRequest = { showSettingsSheet = false }
            ) {
                FeatureModelSettingsSheet(
                    availableModels = availableModels,
                    initialSelectedModel = selectedModel,
                    initialSelectedBackend = selectedBackend,
                    initialSelectedNpuDeviceId = selectedNpuDeviceId,
                    initialMaxTokens = selectedMaxTokens,
                    currentlyLoadedModel = currentlyLoadedModel,
                    isLoadingModel = isGenerating,
                    onModelSelected = { model ->
                        selectedModelName = model.name
                        agentPrefs.edit().putString("selected_model_name", model.name).apply()
                    },
                    onBackendSelected = { backend, deviceId ->
                        selectedBackendName = backend.name
                        selectedNpuDeviceId = deviceId
                        agentPrefs.edit()
                            .putString("selected_backend", backend.name)
                            .putString("selected_npu_device_id", deviceId)
                            .apply()
                    },
                    onMaxTokensChanged = { tokens ->
                        selectedMaxTokens = tokens
                        agentPrefs.edit().putInt("selected_max_tokens", tokens).apply()
                    },
                    onLoadModel = { model, maxTokens, backend, deviceId, _, enableThinking ->
                        selectedModelName = model.name
                        selectedMaxTokens = maxTokens
                        selectedBackendName = backend?.name
                        selectedNpuDeviceId = deviceId
                        agentPrefs.edit()
                            .putString("selected_model_name", model.name)
                            .putInt("selected_max_tokens", maxTokens)
                            .putBoolean("agent_enable_thinking", enableThinking)
                            .putString("selected_backend", backend?.name)
                            .putString("selected_npu_device_id", deviceId)
                            .apply()
                        viewModel.setGenerationParameters(maxTokens = maxTokens, enableThinking = enableThinking, contextWindow = maxTokens)
                        viewModel.loadModel(model, preferredBackend = backend, deviceId = deviceId)
                        showSettingsSheet = false
                    },
                    onUnloadModel = {
                        viewModel.unloadModel()
                    },
                    onDismiss = { showSettingsSheet = false },
                    extraModelConfigsContent = {
                        val isTermuxEnabled by viewModel.isTermuxEnabled.collectAsState()
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = stringResource(R.string.enable_audio),
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.weight(1f)
                            )
                            Switch(
                                checked = isGemmaAudioEnabled,
                                onCheckedChange = { viewModel.setGemmaAudioEnabled(it) }
                            )
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = stringResource(R.string.agent_termux_toggle),
                                style = MaterialTheme.typography.bodyMedium,
                                modifier = Modifier.weight(1f)
                            )
                            Switch(
                                checked = isTermuxEnabled,
                                onCheckedChange = { viewModel.toggleTermux(it) }
                            )
                        }

                        Spacer(modifier = Modifier.height(12.dp))
                        val clipboardManager = LocalClipboardManager.current
                        val context = LocalContext.current
                        Surface(
                            shape = RoundedCornerShape(12.dp),
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Terminal,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(18.dp)
                                    )
                                    Text(
                                        text = stringResource(R.string.agent_termux_guide_title),
                                        style = MaterialTheme.typography.titleSmall,
                                        fontWeight = FontWeight.Bold,
                                        color = MaterialTheme.colorScheme.onSurface
                                    )
                                }
                                Spacer(modifier = Modifier.height(6.dp))
                                Text(
                                    text = stringResource(R.string.agent_termux_guide_desc),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                val termuxCmd = "mkdir -p ~/.termux/ && echo \"allow-external-apps = true\" >> ~/.termux/termux.properties"
                                Surface(
                                    shape = RoundedCornerShape(8.dp),
                                    color = MaterialTheme.colorScheme.surface,
                                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)),
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable {
                                            clipboardManager.setText(AnnotatedString(termuxCmd))
                                            Toast.makeText(context, context.getString(R.string.agent_termux_cmd_copied), Toast.LENGTH_SHORT).show()
                                        }
                                ) {
                                    Row(
                                        modifier = Modifier.padding(8.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = termuxCmd,
                                            style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                                            color = MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.weight(1f)
                                        )
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Icon(
                                            imageVector = Icons.Default.ContentCopy,
                                            contentDescription = stringResource(R.string.copy),
                                            tint = MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.size(16.dp)
                                        )
                                    }
                                }
                            }
                        }
                    }
                )
            }
        }
    }
}

@Composable
fun AgentTextMessageBubble(msg: AgentMessage.Text) {
    val isUser = msg.sender == AgentMessage.Sender.USER
    if (isUser) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End
        ) {
            Surface(
                shape = RoundedCornerShape(
                    topStart = 18.dp, topEnd = 18.dp,
                    bottomStart = 18.dp, bottomEnd = 4.dp
                ),
                color = MaterialTheme.colorScheme.primary,
                contentColor = MaterialTheme.colorScheme.onPrimary,
                modifier = Modifier.widthIn(max = 300.dp)
            ) {
                Text(
                    text = msg.text,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                    fontSize = 15.sp
                )
            }
        }
    } else {
        // AI Messages (Greetings & AI Responses): Full Markdown, Tables, LaTeX 1:1 matching AI Chat!
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp)
        ) {
            if (msg.text.contains("|") && msg.text.contains("-|-")) {
                MarkdownTableView(
                    rawTable = msg.text,
                    baseColor = MaterialTheme.colorScheme.onBackground
                )
            } else {
                ThinkingAwareResultContent(
                    content = msg.text,
                    useMarkdownForAnswer = true
                )
            }
        }
    }
}

@Composable
fun AgentToolCallBubble(
    msg: AgentMessage.ToolCall,
    onExecuteCommand: ((String, String) -> Unit)? = null,
    onCancelCommand: ((String) -> Unit)? = null
) {
    if (msg.toolName.contains("termux", ignoreCase = true) || msg.status == AgentMessage.ToolCall.Status.PENDING_APPROVAL) {
        val clipboardManager = LocalClipboardManager.current
        val context = LocalContext.current
        var editedCommand by remember(msg.callId, msg.args) { mutableStateOf(msg.args) }

        Surface(
            shape = RoundedCornerShape(16.dp),
            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f),
            border = BorderStroke(
                1.dp,
                when (msg.status) {
                    AgentMessage.ToolCall.Status.PENDING_APPROVAL -> Color(0xFFF59E0B)
                    AgentMessage.ToolCall.Status.RUNNING -> Color(0xFF3B82F6)
                    AgentMessage.ToolCall.Status.SUCCESS -> Color(0xFF10B981)
                    AgentMessage.ToolCall.Status.FAILED -> Color(0xFFEF4444)
                    AgentMessage.ToolCall.Status.CANCELLED -> MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                }
            ),
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 6.dp)
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Terminal,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(20.dp)
                        )
                        Text(
                            text = "Termux Shell Command",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }

                    val statusText = when (msg.status) {
                        AgentMessage.ToolCall.Status.PENDING_APPROVAL -> stringResource(R.string.agent_termux_pending_approval)
                        AgentMessage.ToolCall.Status.RUNNING -> stringResource(R.string.agent_tool_running, "")
                        AgentMessage.ToolCall.Status.SUCCESS -> stringResource(R.string.agent_tool_success)
                        AgentMessage.ToolCall.Status.FAILED -> stringResource(R.string.agent_tool_failed)
                        AgentMessage.ToolCall.Status.CANCELLED -> stringResource(R.string.agent_termux_cancelled)
                    }

                    val statusColor = when (msg.status) {
                        AgentMessage.ToolCall.Status.PENDING_APPROVAL -> Color(0xFFF59E0B)
                        AgentMessage.ToolCall.Status.RUNNING -> Color(0xFF3B82F6)
                        AgentMessage.ToolCall.Status.SUCCESS -> Color(0xFF10B981)
                        AgentMessage.ToolCall.Status.FAILED -> Color(0xFFEF4444)
                        AgentMessage.ToolCall.Status.CANCELLED -> MaterialTheme.colorScheme.onSurfaceVariant
                    }

                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = statusColor.copy(alpha = 0.15f)
                    ) {
                        Text(
                            text = statusText,
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold,
                            color = statusColor,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.surface,
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(10.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        if (msg.status == AgentMessage.ToolCall.Status.PENDING_APPROVAL) {
                            androidx.compose.foundation.text.BasicTextField(
                                value = editedCommand,
                                onValueChange = { editedCommand = it },
                                textStyle = MaterialTheme.typography.bodyMedium.copy(
                                    fontFamily = FontFamily.Monospace,
                                    color = MaterialTheme.colorScheme.primary
                                ),
                                modifier = Modifier
                                    .weight(1f)
                                    .padding(vertical = 4.dp),
                                singleLine = false
                            )
                        } else {
                            Text(
                                text = msg.args,
                                style = MaterialTheme.typography.bodyMedium.copy(fontFamily = FontFamily.Monospace),
                                color = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.weight(1f)
                            )
                        }

                        IconButton(
                            onClick = {
                                val textToCopy = if (msg.status == AgentMessage.ToolCall.Status.PENDING_APPROVAL) editedCommand else msg.args
                                clipboardManager.setText(AnnotatedString(textToCopy))
                                Toast.makeText(context, context.getString(R.string.agent_termux_cmd_copied), Toast.LENGTH_SHORT).show()
                            },
                            modifier = Modifier.size(28.dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.ContentCopy,
                                contentDescription = stringResource(R.string.copy),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.size(16.dp)
                            )
                        }
                    }
                }

                if (msg.result != null) {
                    Spacer(modifier = Modifier.height(10.dp))
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = Color(0xFF0F172A),
                        border = BorderStroke(1.dp, Color(0xFF334155)),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(10.dp)) {
                            Text(
                                text = "STDOUT / OUTPUT",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                color = Color(0xFF64748B)
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = msg.result,
                                style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                                color = Color(0xFF38BDF8)
                            )
                        }
                    }
                }

                if (msg.status == AgentMessage.ToolCall.Status.PENDING_APPROVAL) {
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = { onExecuteCommand?.invoke(msg.callId, editedCommand) },
                            shape = RoundedCornerShape(10.dp),
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF10B981))
                        ) {
                            Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(stringResource(R.string.agent_termux_run_cmd), fontWeight = FontWeight.Bold)
                        }
                        OutlinedButton(
                            onClick = { onCancelCommand?.invoke(msg.callId) },
                            shape = RoundedCornerShape(10.dp),
                            modifier = Modifier.weight(1f)
                        ) {
                            Icon(Icons.Default.Close, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(stringResource(R.string.agent_termux_cancel_cmd))
                        }
                    }
                }
            }
        }
    } else {
        Surface(
            shape = RoundedCornerShape(14.dp),
            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
            border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)),
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp)
        ) {
            Row(
                modifier = Modifier.padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = when (msg.status) {
                        AgentMessage.ToolCall.Status.PENDING_APPROVAL, AgentMessage.ToolCall.Status.RUNNING -> Icons.Default.Build
                        AgentMessage.ToolCall.Status.SUCCESS -> Icons.Default.CheckCircle
                        AgentMessage.ToolCall.Status.FAILED, AgentMessage.ToolCall.Status.CANCELLED -> Icons.Default.Error
                    },
                    contentDescription = null,
                    tint = when (msg.status) {
                        AgentMessage.ToolCall.Status.PENDING_APPROVAL, AgentMessage.ToolCall.Status.RUNNING -> Color(0xFFFBBF24)
                        AgentMessage.ToolCall.Status.SUCCESS -> Color(0xFF34D399)
                        AgentMessage.ToolCall.Status.FAILED, AgentMessage.ToolCall.Status.CANCELLED -> Color(0xFFF87171)
                    },
                    modifier = Modifier.size(20.dp)
                )

                Spacer(modifier = Modifier.width(10.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "${msg.toolName}(${msg.args})",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    if (msg.result != null) {
                        Text(
                            text = msg.result,
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 2
                        )
                    }
                }

                Text(
                    text = when (msg.status) {
                        AgentMessage.ToolCall.Status.PENDING_APPROVAL -> stringResource(R.string.agent_termux_pending_approval)
                        AgentMessage.ToolCall.Status.RUNNING -> stringResource(R.string.agent_tool_running, "")
                        AgentMessage.ToolCall.Status.SUCCESS -> stringResource(R.string.agent_tool_success)
                        AgentMessage.ToolCall.Status.FAILED -> stringResource(R.string.agent_tool_failed)
                        AgentMessage.ToolCall.Status.CANCELLED -> stringResource(R.string.agent_termux_cancelled)
                    },
                    fontSize = 11.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
fun AgentMapCardBubble(msg: AgentMessage.MapLocation, context: Context) {
    val openMapAction = {
        val uri = Uri.parse("geo:${msg.latitude},${msg.longitude}?q=${Uri.encode(msg.label)}")
        val mapIntent = Intent(Intent.ACTION_VIEW, uri)
        mapIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        runCatching { context.startActivity(mapIntent) }
    }

    Surface(
        shape = RoundedCornerShape(16.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier
            .fillMaxWidth()
            .height(220.dp)
            .padding(vertical = 6.dp)
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { ctx ->
                    MapView(ctx).apply {
                        setTileSource(TileSourceFactory.MAPNIK)
                        setMultiTouchControls(true)
                        controller.setZoom(15.0)
                        val point = GeoPoint(msg.latitude, msg.longitude)
                        controller.setCenter(point)

                        val marker = Marker(this).apply {
                            position = point
                            title = msg.label
                            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
                            setOnMarkerClickListener { _, _ ->
                                openMapAction()
                                true
                            }
                        }
                        overlays.add(marker)
                    }
                },
                modifier = Modifier.fillMaxSize()
            )

            Surface(
                color = Color.Black.copy(alpha = 0.7f),
                shape = RoundedCornerShape(bottomEnd = 12.dp),
                modifier = Modifier.align(Alignment.TopStart)
            ) {
                Text(
                    text = msg.label,
                    color = Color.White,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                )
            }

            Surface(
                color = MaterialTheme.colorScheme.primary,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(8.dp)
            ) {
                Row(
                    modifier = Modifier
                        .clickable { openMapAction() }
                        .padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "${stringResource(R.string.open_maps)} ↗",
                        color = MaterialTheme.colorScheme.onPrimary,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

@Composable
fun AgentProcessingBubble() {
    Surface(
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
        modifier = Modifier
            .fillMaxWidth(0.85f)
            .padding(vertical = 4.dp)
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(18.dp),
                strokeWidth = 2.dp,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = stringResource(R.string.agent_tool_running, "..."),
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
