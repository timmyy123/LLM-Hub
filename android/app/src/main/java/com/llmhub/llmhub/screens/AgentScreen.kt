package com.llmhub.llmhub.screens

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
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
    // LLM models only (no mmproj/ASR/embedding) — used for empty-state check
    val availableModelsState = produceState<List<LLMModel>>(initialValue = emptyList(), context) {
        value = ModelAvailabilityProvider.loadAvailableModels(context)
    }
    val availableModels = availableModelsState.value

    val listState = rememberLazyListState()

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    LaunchedEffect(availableModels) {
        Configuration.getInstance().userAgentValue = context.packageName
        viewModel.initializeWelcomeMessage(context, availableModels.isNotEmpty())
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    val loadingModelName by viewModel.loadingModelName.collectAsState()
                    val activeModelName by viewModel.activeModelName.collectAsState()
                    val subtitleText = when {
                        loadingModelName != null -> stringResource(R.string.loading_model_format, loadingModelName!!)
                        activeModelName != null -> activeModelName!!
                        else -> stringResource(R.string.agent_subtitle)
                    }

                    Column(modifier = Modifier.fillMaxWidth()) {
                        Text(
                            text = stringResource(R.string.agent_title),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(
                            text = subtitleText,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
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
                    LazyColumn(
                        state = listState,
                        modifier = Modifier
                            .weight(1f)
                            .padding(horizontal = 16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                        contentPadding = PaddingValues(vertical = 16.dp)
                    ) {
                        items(messages, key = { it.id }) { msg ->
                            when (msg) {
                                is AgentMessage.Text -> AgentTextMessageBubble(msg)
                                is AgentMessage.ToolCall -> AgentToolCallBubble(msg)
                                is AgentMessage.MapLocation -> AgentMapCardBubble(msg, context)
                                is AgentMessage.Audio -> com.llmhub.llmhub.components.AudioMessageCard(
                                    audioPath = msg.audioPath,
                                    fileName = "Voice message",
                                    fileSize = runCatching { java.io.File(msg.audioPath).length() }.getOrNull(),
                                    isFromUser = msg.sender == AgentMessage.Sender.USER
                                )
                            }
                        }

                        if (isGenerating) {
                            item {
                                AgentProcessingBubble()
                            }
                        }
                    }
                }

                val isWebSearchEnabled by viewModel.isWebSearchEnabled.collectAsState()

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
                        enabled = !isGenerating,
                        supportsAttachments = true,
                        supportsVision = false,
                        supportsAudio = true,
                        isLoading = isGenerating,
                        onCancelGeneration = if (isGenerating) { { } } else null,
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
                    onLoadModel = { model, maxTokens, backend, deviceId, _, _ ->
                        selectedModelName = model.name
                        selectedMaxTokens = maxTokens
                        selectedBackendName = backend?.name
                        selectedNpuDeviceId = deviceId
                        agentPrefs.edit()
                            .putString("selected_model_name", model.name)
                            .putInt("selected_max_tokens", maxTokens)
                            .putString("selected_backend", backend?.name)
                            .putString("selected_npu_device_id", deviceId)
                            .apply()
                        viewModel.loadModel(model, preferredBackend = backend, deviceId = deviceId)
                        showSettingsSheet = false
                    },
                    onUnloadModel = {
                        viewModel.unloadModel()
                    },
                    onDismiss = { showSettingsSheet = false },
                    extraModelConfigsContent = {
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

                        Spacer(modifier = Modifier.height(12.dp))

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
                                checked = viewModel.toolSet.isTermuxEnabled,
                                onCheckedChange = { viewModel.toggleTermux(it) }
                            )
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
        // AI Messages (Greetings & AI Responses): FULL WIDTH WITH NO BUBBLE/SURFACE CONTAINER!
        Text(
            text = msg.text,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp),
            fontSize = 15.sp,
            color = MaterialTheme.colorScheme.onBackground
        )
    }
}

@Composable
fun AgentToolCallBubble(msg: AgentMessage.ToolCall) {
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        border = androidx.compose.foundation.BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)),
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
                    AgentMessage.ToolCall.Status.RUNNING -> Icons.Default.Build
                    AgentMessage.ToolCall.Status.SUCCESS -> Icons.Default.CheckCircle
                    AgentMessage.ToolCall.Status.FAILED -> Icons.Default.Error
                },
                contentDescription = null,
                tint = when (msg.status) {
                    AgentMessage.ToolCall.Status.RUNNING -> Color(0xFFFBBF24)
                    AgentMessage.ToolCall.Status.SUCCESS -> Color(0xFF34D399)
                    AgentMessage.ToolCall.Status.FAILED -> Color(0xFFF87171)
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
                    AgentMessage.ToolCall.Status.RUNNING -> stringResource(R.string.agent_tool_running, "")
                    AgentMessage.ToolCall.Status.SUCCESS -> stringResource(R.string.agent_tool_success)
                    AgentMessage.ToolCall.Status.FAILED -> stringResource(R.string.agent_tool_failed)
                },
                fontSize = 11.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
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
