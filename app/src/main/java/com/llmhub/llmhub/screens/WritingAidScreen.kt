package com.llmhub.llmhub.screens

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
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
                    onLoadModel = { viewModel.loadModel() },
                    onUnloadModel = { viewModel.unloadModel() },
                    filterMultimodalOnly = false,
                    modifier = Modifier.fillMaxWidth()
                )

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
