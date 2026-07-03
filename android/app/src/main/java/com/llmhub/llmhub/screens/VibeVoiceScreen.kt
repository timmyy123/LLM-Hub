package com.llmhub.llmhub.screens

import android.Manifest
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.GetApp
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.StopCircle
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.LlmHubApplication
import com.llmhub.llmhub.R
import com.llmhub.llmhub.ads.BannerAd
import com.llmhub.llmhub.components.ModelSelectorCard
import com.llmhub.llmhub.ui.components.AudioInputService
import com.llmhub.llmhub.ui.components.TtsService
import com.llmhub.llmhub.viewmodels.VibeVoiceViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VibeVoiceScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    viewModel: VibeVoiceViewModel = viewModel()
) {
    val availableLlmModels by viewModel.availableLlmModels.collectAsState()
    val availableAsrModels by viewModel.availableAsrModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedVoiceModel by viewModel.selectedVoiceModel.collectAsState()
    val hasRequiredModels by viewModel.hasRequiredModels.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    val selectedNpuDeviceId by viewModel.selectedNpuDeviceId.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val isRecording by viewModel.isRecording.collectAsState()
    val isResponding by viewModel.isResponding.collectAsState()
    val loadError by viewModel.loadError.collectAsState()
    val conversationTurns by viewModel.conversationTurns.collectAsState()
    val transcription by viewModel.transcription.collectAsState()

    val context = LocalContext.current
    val isPremium by (context.applicationContext as LlmHubApplication).billingManager.isPremium.collectAsState(initial = false)

    val snackbarHostState = remember { SnackbarHostState() }
    val coroutineScope = rememberCoroutineScope()

    val audioService = remember { AudioInputService(context) }
    val ttsService = remember { TtsService(context, isTranslationFeature = false) }
    val isTtsSpeaking by ttsService.isSpeaking.collectAsState()

    var recordedAudioData by remember { mutableStateOf<ByteArray?>(null) }
    var isAutoStopping by remember { mutableStateOf(false) }
    var isManualStopping by remember { mutableStateOf(false) }
    var isChatActive by remember { mutableStateOf(false) }
    var hasAudioPermission by remember { mutableStateOf(false) }
    var showSettingsSheet by remember { mutableStateOf(false) }
    val settingsSheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    val micLevelRaw by audioService.audioLevel.collectAsState()
    val latestAssistantText = conversationTurns.lastOrNull()?.assistantText.orEmpty()

    val micLevel by animateFloatAsState(
        targetValue = micLevelRaw.coerceIn(0f, 1f),
        animationSpec = tween(durationMillis = 90),
        label = "mic_level_smooth"
    )

    val pulseTransition = rememberInfiniteTransition(label = "vibevoice_pulse")
    val globeScale by pulseTransition.animateFloat(
        initialValue = 1f,
        targetValue = if (isRecording) 1.08f else if (isResponding) 1.04f else 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = if (isRecording) 520 else if (isResponding) 900 else 1800,
                easing = FastOutSlowInEasing
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "globe_scale"
    )
    val glowAlpha by pulseTransition.animateFloat(
        initialValue = 0.28f,
        targetValue = if (isRecording) 0.75f else if (isResponding) 0.48f else 0.28f,
        animationSpec = infiniteRepeatable(
            animation = tween(
                durationMillis = if (isRecording) 520 else if (isResponding) 900 else 1800,
                easing = FastOutSlowInEasing
            ),
            repeatMode = RepeatMode.Reverse
        ),
        label = "globe_glow"
    )

    val reactiveScale = globeScale + if (isRecording) (0.08f + micLevel * 0.55f) else 0f
    val reactiveGlowAlpha = (glowAlpha + if (isRecording) (0.12f + micLevel * 0.65f) else 0f).coerceIn(0f, 1f)

    val audioPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasAudioPermission = granted
        if (granted && isChatActive) {
            viewModel.setRecording(true)
        } else if (!granted) {
            isChatActive = false
        }
    }

    LaunchedEffect(Unit) {
        hasAudioPermission = audioService.hasAudioPermission()
        audioService.onRecordingAutoStopped = {
            isAutoStopping = true
            viewModel.setRecording(false)
        }
    }

    LaunchedEffect(isRecording) {
        if (isRecording && hasAudioPermission) {
            val ok = audioService.startRecording()
            if (!ok) viewModel.setRecording(false)
        } else if (!isRecording) {
            if (audioService.isRecording() || isAutoStopping) {
                val data = audioService.stopRecording()
                isAutoStopping = false

                if (isManualStopping) {
                    isManualStopping = false
                    recordedAudioData = null
                    viewModel.setAudioData(null)
                    return@LaunchedEffect
                }

                // Ignore empty/too-short clips to avoid loop churn.
                if (data != null && data.size > 640) {
                    recordedAudioData = data
                    viewModel.setAudioData(data)
                    if (isModelLoaded && !isResponding) {
                        viewModel.sendVoiceTurn()
                    }
                } else {
                    recordedAudioData = null
                    viewModel.setAudioData(null)
                }
            }
        }
    }

    // Hands-free loop: when model is loaded and system is idle, auto-start listening.
    LaunchedEffect(isModelLoaded, isChatActive, isRecording, isResponding, isTtsSpeaking, hasAudioPermission, isAutoStopping, showSettingsSheet) {
        if (
            isModelLoaded &&
            isChatActive &&
            hasAudioPermission &&
            !isRecording &&
            !isResponding &&
            !isTtsSpeaking &&
            !isAutoStopping &&
            !showSettingsSheet
        ) {
            viewModel.setRecording(true)
        }
    }

    LaunchedEffect(isTtsSpeaking) {
        if (isTtsSpeaking && isRecording) {
            viewModel.setRecording(false)
        }
    }

    LaunchedEffect(isModelLoaded, isChatActive) {
        if (!isModelLoaded) {
            isChatActive = false
            viewModel.setRecording(false)
        }
        if (!isChatActive) {
            viewModel.setRecording(false)
        }
    }

    LaunchedEffect(loadError) {
        loadError?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }

    LaunchedEffect(conversationTurns.size) {
        // Keep existing effect trigger in place for first turn insertion.
    }

    var streamedTtsIndex by remember { mutableStateOf(0) }
    var lastCompletedReply by remember { mutableStateOf("") }
    LaunchedEffect(latestAssistantText, isResponding, isChatActive) {
        if (!isChatActive) {
            // Hard stop when chat is manually inactive to block delayed TTS callbacks.
            ttsService.stop()
            streamedTtsIndex = latestAssistantText.length
            return@LaunchedEffect
        }

        if (isResponding) {
            if (latestAssistantText.length < streamedTtsIndex) {
                streamedTtsIndex = 0
            }
            if (latestAssistantText.length > streamedTtsIndex) {
                val delta = latestAssistantText.substring(streamedTtsIndex)
                streamedTtsIndex = latestAssistantText.length
                ttsService.addStreamingText(delta)
            }
        } else {
            ttsService.flushStreamingBuffer()
            if (latestAssistantText.isNotBlank()) {
                lastCompletedReply = latestAssistantText
            }
            streamedTtsIndex = latestAssistantText.length
        }
    }

    // Safety valve: if TTS gets stuck after generation, stop it so hands-free listening can continue.
    LaunchedEffect(isTtsSpeaking, isResponding, isChatActive) {
        if (isChatActive && isTtsSpeaking && !isResponding) {
            delay(90000)
            if (isChatActive && isTtsSpeaking && !isResponding) {
                ttsService.stop()
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            if (audioService.isRecording()) {
                coroutineScope.launch { audioService.stopRecording() }
            }
            if (isTtsSpeaking) {
                ttsService.stop()
            }
            ttsService.shutdown()
            viewModel.unloadModel()
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = stringResource(R.string.feature_vibevoice),
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = stringResource(R.string.feature_settings_title))
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        if (!isModelLoaded) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(paddingValues)
                    .padding(32.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Default.ModelTraining,
                    contentDescription = null,
                    modifier = Modifier.size(80.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = stringResource(
                        if (!hasRequiredModels) R.string.download_models
                        else R.string.vibevoice_load_model
                    ),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(
                        if (!hasRequiredModels) R.string.vibevoice_requires_models
                        else R.string.vibevoice_load_model_desc
                    ),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(32.dp))
                FilledTonalButton(
                    onClick = {
                        if (!hasRequiredModels) onNavigateToModels()
                        else showSettingsSheet = true
                    },
                    modifier = Modifier.fillMaxWidth(0.6f)
                ) {
                    Icon(
                        imageVector = if (!hasRequiredModels) Icons.Default.GetApp else Icons.Default.Tune,
                        contentDescription = null
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        stringResource(
                            if (!hasRequiredModels) R.string.download_models
                            else R.string.feature_settings_title
                        )
                    )
                }
            }
        } else {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(horizontal = 24.dp)
                        .padding(bottom = 90.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Box(
                        modifier = Modifier
                            .size(248.dp)
                            .scale(reactiveScale),
                        contentAlignment = Alignment.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .clip(CircleShape)
                                .background(
                                    Brush.radialGradient(
                                        colors = listOf(
                                            Color(0xFFF7EFE1).copy(alpha = reactiveGlowAlpha),
                                            Color(0xFF69C6FF).copy(alpha = reactiveGlowAlpha),
                                            Color(0xFF1478F4).copy(alpha = reactiveGlowAlpha)
                                        )
                                    )
                                )
                        )
                        Box(
                            modifier = Modifier
                                .size(218.dp)
                                .clip(CircleShape)
                                .background(
                                    Brush.radialGradient(
                                        colors = listOf(
                                            Color(0xFFE5F8FF),
                                            Color(0xFF4FAAF8),
                                            Color(0xFF0E67E8)
                                        )
                                    )
                                )
                                .clickable {
                                    if (!isChatActive) {
                                        isChatActive = true
                                        if (hasAudioPermission) {
                                            viewModel.setRecording(true)
                                        } else {
                                            audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                        }
                                    } else {
                                        isChatActive = false
                                        if (isRecording) {
                                            isManualStopping = true
                                            viewModel.setRecording(false)
                                        }
                                        viewModel.cancelResponse()
                                        ttsService.stop()
                                        recordedAudioData = null
                                        viewModel.clearCurrentAudioInput()
                                    }
                                },
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = if (isChatActive) {
                                    if (isResponding) Icons.Default.StopCircle else if (isRecording) Icons.Default.Mic else Icons.Default.GraphicEq
                                } else {
                                    Icons.Default.Mic
                                },
                                contentDescription = null,
                                modifier = Modifier.size(72.dp),
                                tint = Color.White.copy(alpha = 0.92f)
                            )
                        }
                    }
                    
                    if (transcription.isNotBlank()) {
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = transcription,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.55f),
                            textAlign = TextAlign.Center,
                            modifier = Modifier
                                .padding(horizontal = 32.dp)
                                .fillMaxWidth()
                        )
                    }

                    if (latestAssistantText.isNotBlank()) {
                        Spacer(modifier = Modifier.height(24.dp))
                        Card(
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                            shape = RoundedCornerShape(18.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(
                                modifier = Modifier.padding(14.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Text(
                                    text = latestAssistantText,
                                    style = MaterialTheme.typography.bodyLarge
                                )
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.End
                                ) {
                                    IconButton(onClick = {
                                        if (isTtsSpeaking) ttsService.stop() else ttsService.speak(lastCompletedReply.ifBlank { latestAssistantText })
                                    }) {
                                        Icon(Icons.Default.VolumeUp, contentDescription = null)
                                    }
                                }
                            }
                        }
                    }
                }

                if (!isPremium) {
                    BannerAd(
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .fillMaxWidth()
                    )
                }
            }
        }
    }

    if (showSettingsSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSettingsSheet = false },
            sheetState = settingsSheetState
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
                    .padding(bottom = 32.dp)
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = stringResource(R.string.feature_settings_title),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                ModelSelectorCard(
                    models = availableLlmModels,
                    selectedModel = selectedModel,
                    selectedBackend = selectedBackend,
                    selectedNpuDeviceId = selectedNpuDeviceId,
                    isLoading = isLoadingModel,
                    isModelLoaded = isModelLoaded,
                    onModelSelected = { viewModel.selectModel(it) },
                    onBackendSelected = { backend, deviceId -> viewModel.selectBackend(backend, deviceId) },
                    onLoadModel = {
                        isChatActive = false
                        viewModel.loadModel()
                    },
                    onUnloadModel = {
                        if (isTtsSpeaking) {
                            ttsService.stop()
                        }
                        viewModel.unloadModel()
                    },
                    filterMultimodalOnly = false,
                    voiceModels = availableAsrModels,
                    selectedVoiceModel = selectedVoiceModel,
                    onVoiceModelSelected = { viewModel.selectVoiceModel(it) },
                    llmModelLabel = stringResource(R.string.llm_model),
                    voiceModelLabel = stringResource(R.string.voice_model),
                    backendLabel = stringResource(R.string.llm_model_backend)
                )
            }
        }
    }
}