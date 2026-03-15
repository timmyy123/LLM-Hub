package com.llmhub.llmhub.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.activity.ComponentActivity
import com.llmhub.llmhub.LlmHubApplication
import com.llmhub.llmhub.ads.BannerAd
import com.llmhub.llmhub.R
import com.llmhub.llmhub.components.ModelSelectorCard
import com.llmhub.llmhub.components.ThinkingAwareResultContent
import com.llmhub.llmhub.components.getDisplayContentWithoutThinking
import com.llmhub.llmhub.viewmodels.WritingAidViewModel
import com.llmhub.llmhub.viewmodels.WritingMode
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WritingAidScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    viewModel: WritingAidViewModel = viewModel()
) {
    val context = LocalContext.current
    val activity = context as ComponentActivity
    val isPremium by (context.applicationContext as LlmHubApplication).billingManager.isPremium.collectAsState(initial = false)
    val rewardedAdManager = remember { (context.applicationContext as LlmHubApplication).rewardedAdManager }
    val keyboard = LocalSoftwareKeyboardController.current
    val clipboardManager = LocalClipboardManager.current
    
    // UI State
    var inputText by remember { mutableStateOf("") }
    var showModeMenu by remember { mutableStateOf(false) }
    var showSettingsSheet by remember { mutableStateOf(false) }
    
    // ViewModel states
    val availableModels by viewModel.availableModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    val selectedNpuDeviceId by viewModel.selectedNpuDeviceId.collectAsState()
    val selectedMode by viewModel.selectedMode.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val isProcessing by viewModel.isProcessing.collectAsState()
    val outputText by viewModel.outputText.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    val enableThinking by viewModel.enableThinking.collectAsState()
    val selectedMaxTokens by viewModel.selectedMaxTokens.collectAsState()
    val selectedNGpuLayers by viewModel.selectedNGpuLayers.collectAsState()

    // Slider local state (synced from viewmodel)
    val baseMaxTokensCap by remember(selectedModel) {
        derivedStateOf { selectedModel?.contextWindowSize?.coerceAtLeast(1) ?: 4096 }
    }
    var maxTokensValue by remember(selectedMaxTokens, baseMaxTokensCap) {
        mutableStateOf(selectedMaxTokens.coerceIn(1, baseMaxTokensCap))
    }
    var maxTokensText by remember(maxTokensValue) { mutableStateOf(maxTokensValue.toString()) }
    var gpuLayers by remember(selectedNGpuLayers) { mutableStateOf(selectedNGpuLayers ?: 999) }
    val isGguf by remember(selectedModel) { derivedStateOf { selectedModel?.modelFormat == "gguf" } }
    val isThinkingOrHarmonyModel by remember(selectedModel) {
        derivedStateOf {
            val name = selectedModel?.name?.lowercase() ?: ""
            name.contains("thinking") || name.contains("reasoning") ||
                name.contains("gpt-oss") || name.contains("gpt_oss")
        }
    }
    
    // TTS Service
    val ttsService = remember { com.llmhub.llmhub.ui.components.TtsService(context) }
    val isTtsSpeaking by ttsService.isSpeaking.collectAsState()
    
    // Scroll state for auto-scrolling
    val scrollState = rememberScrollState()
    val coroutineScope = rememberCoroutineScope()
    
    // Show error snackbar
    val snackbarHostState = remember { SnackbarHostState() }
    LaunchedEffect(errorMessage) {
        errorMessage?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }
    
    // Auto-scroll to bottom when output text changes (during generation)
    LaunchedEffect(outputText) {
        if (outputText.isNotEmpty() && isProcessing) {
            coroutineScope.launch {
                scrollState.animateScrollTo(scrollState.maxValue)
            }
        }
    }
    
    // Cleanup on dispose - unload model to free memory and shutdown TTS
    DisposableEffect(Unit) {
        onDispose {
            viewModel.unloadModel()
            ttsService.shutdown()
        }
    }
    
    // Settings Bottom Sheet
    if (showSettingsSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSettingsSheet = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = stringResource(R.string.feature_settings_title),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
                
                ModelSelectorCard(
                    models = availableModels,
                    selectedModel = selectedModel,
                    selectedBackend = selectedBackend,
                    selectedNpuDeviceId = selectedNpuDeviceId,
                    isLoading = isLoading,
                    isModelLoaded = isModelLoaded,
                    onModelSelected = { viewModel.selectModel(it) },
                    onBackendSelected = { backend, deviceId -> viewModel.selectBackend(backend, deviceId) },
                    onLoadModel = {
                        if (isPremium) {
                            viewModel.loadModel()
                        } else {
                            rewardedAdManager.showAdOrGrant(activity) { viewModel.loadModel() }
                        }
                    },
                    onUnloadModel = { viewModel.unloadModel() },
                    filterMultimodalOnly = false,
                    modifier = Modifier.fillMaxWidth()
                )

                // Max Tokens slider
                Card(
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            text = stringResource(R.string.model_configs_title),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                            Text(text = stringResource(R.string.max_tokens), style = MaterialTheme.typography.bodyMedium)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "$baseMaxTokensCap ${stringResource(R.string.max)}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                            Slider(
                                value = maxTokensValue.toFloat(),
                                onValueChange = {
                                    val v = it.toInt().coerceIn(1, baseMaxTokensCap)
                                    maxTokensValue = v
                                    maxTokensText = v.toString()
                                    viewModel.setMaxTokens(v)
                                },
                                valueRange = 1f..baseMaxTokensCap.toFloat(),
                                modifier = Modifier.weight(1f).height(36.dp),
                                thumb = {
                                    SliderDefaults.Thumb(
                                        interactionSource = remember { MutableInteractionSource() },
                                        thumbSize = DpSize(24.dp, 24.dp)
                                    )
                                }
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            OutlinedTextField(
                                value = maxTokensText,
                                onValueChange = { input ->
                                    val numeric = input.filter { it.isDigit() }
                                    val v = (numeric.toIntOrNull() ?: 1).coerceIn(1, baseMaxTokensCap)
                                    maxTokensText = v.toString()
                                    maxTokensValue = v
                                    viewModel.setMaxTokens(v)
                                },
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                singleLine = true,
                                modifier = Modifier.width(72.dp)
                            )
                        }

                        // GPU Layers (GGUF only)
                        if (isGguf) {
                            Text(
                                text = stringResource(R.string.gpu_layers_label, gpuLayers),
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                                Slider(
                                    value = gpuLayers.toFloat(),
                                    onValueChange = {
                                        gpuLayers = it.toInt()
                                        viewModel.setNGpuLayers(gpuLayers)
                                    },
                                    valueRange = 0f..999f,
                                    modifier = Modifier.weight(1f).height(28.dp),
                                    thumb = {
                                        SliderDefaults.Thumb(
                                            interactionSource = remember { MutableInteractionSource() },
                                            thumbSize = DpSize(24.dp, 24.dp)
                                        )
                                    }
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                OutlinedTextField(
                                    value = gpuLayers.toString(),
                                    onValueChange = { v ->
                                        val n = v.filter { it.isDigit() }.toIntOrNull()?.coerceIn(0, 999) ?: gpuLayers
                                        gpuLayers = n
                                        viewModel.setNGpuLayers(n)
                                    },
                                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                    singleLine = true,
                                    modifier = Modifier.width(72.dp)
                                )
                            }
                        }
                    }
                }

                // Tone Selector
                Text(
                    text = stringResource(R.string.writing_aid_select_mode),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                ExposedDropdownMenuBox(
                    expanded = showModeMenu,
                    onExpandedChange = { showModeMenu = !showModeMenu }
                ) {
                    OutlinedTextField(
                        value = getModeString(selectedMode),
                        onValueChange = {},
                        readOnly = true,
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = showModeMenu) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .menuAnchor(),
                        colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(),
                        shape = RoundedCornerShape(16.dp)
                    )
                    ExposedDropdownMenu(
                        expanded = showModeMenu,
                        onDismissRequest = { showModeMenu = false }
                    ) {
                        WritingMode.values().forEach { mode ->
                            DropdownMenuItem(
                                text = { Text(getModeString(mode)) },
                                onClick = {
                                    viewModel.selectMode(mode)
                                    showModeMenu = false
                                }
                            )
                        }
                    }
                }
                                // Thinking toggle (shown only for thinking/reasoning models)
                if (isThinkingOrHarmonyModel) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = stringResource(R.string.enable_thinking),
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = FontWeight.Medium
                        )
                        Switch(
                            checked = enableThinking,
                            onCheckedChange = { viewModel.setEnableThinking(it) }
                        )
                    }
                }
                                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = stringResource(R.string.writing_aid_title),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = stringResource(R.string.back)
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = "Settings")
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        // Show "Load Model First" screen if model not loaded
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
                        if (availableModels.isEmpty()) R.string.download_models_first
                        else R.string.scam_detector_load_model
                    ),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(R.string.scam_detector_load_model_desc),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )
                Spacer(modifier = Modifier.height(32.dp))
                FilledTonalButton(
                    onClick = { 
                        if (availableModels.isEmpty()) onNavigateToModels()
                        else showSettingsSheet = true
                    },
                    modifier = Modifier.fillMaxWidth(0.6f)
                ) {
                    Icon(
                        imageVector = if (availableModels.isEmpty()) Icons.Default.GetApp else Icons.Default.Tune,
                        contentDescription = null
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(
                        if (availableModels.isEmpty()) R.string.download_models
                        else R.string.feature_settings_title
                    ))
                }
            }
        } else {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .imePadding()
        ) {
            // Scrollable content (input + output)
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scrollState)
                    .padding(bottom = 80.dp) // Space for fixed button
            ) {
                // Input Area (no background card, just text field)
                Column(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        placeholder = { 
                            Text(
                                stringResource(
                                    if (!isModelLoaded) R.string.load_model_to_start 
                                    else R.string.writing_aid_input_hint
                                )
                            ) 
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        minLines = 3,
                        maxLines = 10,
                        shape = RoundedCornerShape(12.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Color.Transparent,
                            unfocusedBorderColor = Color.Transparent
                        )
                    )
                    
                    // Input action bar with paste button
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 4.dp),
                        horizontalArrangement = Arrangement.Start,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Paste button
                        IconButton(
                            onClick = { 
                                val clipText = clipboardManager.getText()?.text
                                if (!clipText.isNullOrBlank()) {
                                    inputText += clipText
                                }
                            }
                        ) {
                            Icon(
                                Icons.Default.ContentPaste,
                                contentDescription = "Paste",
                                tint = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                }
                
                // Output Area: expandable thinking + answer (same as chat).
                if (outputText.isNotEmpty()) {
                    val displayForActions = getDisplayContentWithoutThinking(outputText)
                    Divider()
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        ThinkingAwareResultContent(content = outputText)
                        if (displayForActions.isNotEmpty()) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.End
                            ) {
                                IconButton(
                                    onClick = {
                                        if (isTtsSpeaking) ttsService.stop()
                                        else {
                                            val appLocale = com.llmhub.llmhub.utils.LocaleHelper.getCurrentLocale(context)
                                            ttsService.setLanguage(appLocale)
                                            ttsService.speak(displayForActions)
                                        }
                                    }
                                ) {
                                    Icon(
                                        if (isTtsSpeaking) Icons.Default.Stop else Icons.Default.VolumeUp,
                                        contentDescription = if (isTtsSpeaking) "Stop reading" else "Read aloud",
                                        tint = if (isTtsSpeaking) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                                IconButton(
                                    onClick = { clipboardManager.setText(AnnotatedString(displayForActions)) }
                                ) {
                                    Icon(
                                        Icons.Default.ContentCopy,
                                        contentDescription = stringResource(R.string.copy),
                                        tint = MaterialTheme.colorScheme.primary
                                    )
                                }
                            }
                        }
                    }
                }
            }
            
            // Fixed Process Button at bottom
            Surface(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth(),
                shadowElevation = 8.dp,
                color = MaterialTheme.colorScheme.surface
            ) {
                Column {
                if (!isPremium) {
                    BannerAd(modifier = Modifier.fillMaxWidth())
                }
                if (isProcessing) {
                    // Show Cancel button while processing
                    OutlinedButton(
                        onClick = { viewModel.cancelProcessing() },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = MaterialTheme.colorScheme.error
                        ),
                        border = ButtonDefaults.outlinedButtonBorder.copy(
                            brush = Brush.linearGradient(
                                colors = listOf(MaterialTheme.colorScheme.error, MaterialTheme.colorScheme.error)
                            )
                        )
                    ) {
                        Icon(Icons.Default.StopCircle, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = stringResource(R.string.processing_tap_to_cancel),
                            fontWeight = FontWeight.Bold
                        )
                    }
                } else {
                    Button(
                        onClick = {
                            keyboard?.hide()
                            viewModel.processText(inputText, selectedMode)
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        enabled = inputText.isNotBlank() && !isProcessing && isModelLoaded,
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Default.PlayArrow, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = stringResource(R.string.writing_aid_process),
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
                } // Column
            }
        }
        }
    }
}

@Composable
private fun getModeString(mode: WritingMode): String {
    return when (mode) {
        WritingMode.FRIENDLY -> stringResource(R.string.writing_aid_tone_friendly)
        WritingMode.PROFESSIONAL -> stringResource(R.string.writing_aid_tone_professional)
        WritingMode.CONCISE -> stringResource(R.string.writing_aid_tone_concise)
    }
}
