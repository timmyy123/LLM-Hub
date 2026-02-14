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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.R
import com.llmhub.llmhub.components.FeatureModelSettingsSheet
import com.llmhub.llmhub.viewmodels.VibeCoderViewModel
import com.llmhub.llmhub.viewmodels.CodeLanguage
import kotlinx.coroutines.launch
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.platform.LocalView
import androidx.compose.runtime.DisposableEffect
import android.graphics.Rect
import android.view.ViewTreeObserver

@OptIn(ExperimentalMaterial3Api::class, androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
fun VibeCoderScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    onNavigateToCanvas: ((String, String) -> Unit)? = null,
    viewModel: VibeCoderViewModel = viewModel()
) {
    val keyboard = LocalSoftwareKeyboardController.current
    val clipboardManager = LocalClipboardManager.current
    val promptBringRequester = remember { BringIntoViewRequester() }
    var promptFocused by remember { mutableStateOf(false) }
    // Detect keyboard (IME) visibility via root view global layout and trigger bringIntoView
    val view = LocalView.current
    val imeVisible = remember { mutableStateOf(false) }
    DisposableEffect(view) {
        val listener = ViewTreeObserver.OnGlobalLayoutListener {
            val r = Rect()
            view.getWindowVisibleDisplayFrame(r)
            val screenHeight = view.rootView.height
            val keypadHeight = screenHeight - r.bottom
            val visible = keypadHeight > screenHeight * 0.15
            if (imeVisible.value != visible) imeVisible.value = visible
        }
        view.viewTreeObserver.addOnGlobalLayoutListener(listener)
        onDispose { view.viewTreeObserver.removeOnGlobalLayoutListener(listener) }
    }

    LaunchedEffect(imeVisible.value) {
        if (imeVisible.value && promptFocused) {
            promptBringRequester.bringIntoView()
        }
    }
    
    // UI State
    var promptText by remember { mutableStateOf("") }
    var showSettingsSheet by remember { mutableStateOf(false) }
    
    // ViewModel states
    val availableModels by viewModel.availableModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    val selectedNpuDeviceId by viewModel.selectedNpuDeviceId.collectAsState()
    val selectedMaxTokens by viewModel.selectedMaxTokens.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val isProcessing by viewModel.isProcessing.collectAsState()
    val isPlanning by viewModel.isPlanning.collectAsState()
    val generatedCode by viewModel.generatedCode.collectAsState()
    val codeLanguage by viewModel.codeLanguage.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    
    // Scroll state for auto-scrolling
    val scrollState = rememberScrollState()
    val coroutineScope = rememberCoroutineScope()
    
    // Snackbar
    val snackbarHostState = remember { SnackbarHostState() }
    LaunchedEffect(errorMessage) {
        errorMessage?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }
    // Auto-scroll to bottom when code is being generated
    LaunchedEffect(generatedCode) {
        if (generatedCode.isNotEmpty() && isProcessing) {
            coroutineScope.launch {
                scrollState.animateScrollTo(scrollState.maxValue)
            }
        }
    }
    
    // Note: We intentionally DO NOT unload the model on screen dispose.
    // This preserves the model state when navigating to preview and back.
    // The model will be unloaded when user explicitly selects a different model
    // or closes the app.
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.vibe_coder_title)) },
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
    ) { innerPadding ->
        if (!isModelLoaded) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(innerPadding)
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
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(R.string.scam_detector_load_model_desc),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center
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
                    Text(
                        stringResource(
                            if (availableModels.isEmpty()) R.string.download_models
                            else R.string.feature_settings_title
                        )
                    )
                }
            }
        } else {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .imePadding()
                    .verticalScroll(scrollState)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = stringResource(R.string.vibe_coder_prompt_label),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold
                            )
                            Spacer(Modifier.weight(1f))
                            if (generatedCode.isNotBlank()) {
                                Surface(
                                    color = MaterialTheme.colorScheme.tertiaryContainer,
                                    shape = RoundedCornerShape(4.dp)
                                ) {
                                    Text(
                                        text = "Modification Mode",
                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onTertiaryContainer
                                    )
                                }
                            }
                        }
                        
                        OutlinedTextField(
                            value = promptText,
                            onValueChange = { promptText = it },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(120.dp)
                                .bringIntoViewRequester(promptBringRequester)
                                .onFocusChanged { promptFocused = it.isFocused },
                            placeholder = {
                                Text(stringResource(R.string.vibe_coder_prompt_hint))
                            },
                            shape = RoundedCornerShape(8.dp),
                            enabled = !isProcessing,
                            maxLines = 6
                        )
                        
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Button(
                                onClick = {
                                    keyboard?.hide()
                                    viewModel.generateCode(promptText)
                                },
                                modifier = Modifier
                                    .weight(1f)
                                    .height(48.dp),
                                enabled = promptText.isNotBlank() && !isProcessing,
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.primary
                                )
                            ) {
                                if (isProcessing) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(20.dp),
                                        color = MaterialTheme.colorScheme.onPrimary,
                                        strokeWidth = 2.dp
                                    )
                                } else {
                                    Icon(
                                        Icons.Default.Lightbulb,
                                        contentDescription = null,
                                        modifier = Modifier.size(20.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        if (generatedCode.isNotBlank()) "Refine Code" 
                                        else stringResource(R.string.vibe_coder_generate)
                                    )
                                }
                            }
                            
                            Button(
                                onClick = { 
                                    promptText = ""
                                    viewModel.clearCode()
                                },
                                modifier = Modifier
                                    .height(48.dp)
                                    .padding(end = 0.dp),
                                colors = ButtonDefaults.outlinedButtonColors()
                            ) {
                                Icon(Icons.Default.Clear, contentDescription = "New Project / Clear Context")
                            }
                        }
                    }
                }
                // Code Output Section
                if (generatedCode.isNotEmpty() || isProcessing) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Header
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = stringResource(R.string.vibe_coder_generated_code),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold
                            )
                            
                            if (codeLanguage != CodeLanguage.UNKNOWN) {
                                Surface(
                                    modifier = Modifier
                                        .padding(start = 8.dp),
                                    shape = RoundedCornerShape(16.dp),
                                    color = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.2f)
                                ) {
                                    Text(
                                        text = codeLanguage.name,
                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.tertiary
                                    )
                                }
                            }
                        }
                        
                        // Loading indicator
                        if (isProcessing && generatedCode.isEmpty()) {
                            LinearProgressIndicator(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(4.dp),
                                color = MaterialTheme.colorScheme.primary
                            )
                            Text(
                                text = if (isPlanning) "Planning architecture..." else stringResource(R.string.vibe_coder_generating),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        } else if (generatedCode.isNotEmpty()) {
                            // Code display
                            Surface(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(min = 100.dp, max = 300.dp),
                                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                                shape = RoundedCornerShape(8.dp)
                            ) {
                                Text(
                                    text = generatedCode,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(12.dp)
                                        .verticalScroll(rememberScrollState()),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurface,
                                    fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                                )
                            }
                            
                            // Action buttons for generated code
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Button(
                                    onClick = {
                                        clipboardManager.setText(AnnotatedString(generatedCode))
                                    },
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(40.dp),
                                    colors = ButtonDefaults.outlinedButtonColors()
                                ) {
                                    Icon(Icons.Default.ContentCopy, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(4.dp))
                                    Text(stringResource(R.string.vibe_coder_copy), style = MaterialTheme.typography.labelSmall)
                                }
                                
                                if (codeLanguage == CodeLanguage.HTML) {
                                    Button(
                                        onClick = {
                                            onNavigateToCanvas?.invoke(generatedCode, "html")
                                        },
                                        modifier = Modifier
                                            .weight(1f)
                                            .height(40.dp),
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = MaterialTheme.colorScheme.primary
                                        )
                                    ) {
                                        Icon(Icons.Default.Preview, contentDescription = null, modifier = Modifier.size(16.dp))
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text(stringResource(R.string.vibe_coder_preview), style = MaterialTheme.typography.labelSmall)
                                    }
                                }
                            }
                        }
                    }
                }
                }
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
    
    // Settings Bottom Sheet
    if (showSettingsSheet) {
        FeatureModelSettingsSheet(
            availableModels = availableModels,
            initialSelectedModel = selectedModel,
            initialSelectedBackend = selectedBackend,
            initialSelectedNpuDeviceId = selectedNpuDeviceId,
            initialMaxTokens = selectedMaxTokens,
            currentlyLoadedModel = if (isModelLoaded) selectedModel else null,
            isLoadingModel = isLoading,
            onModelSelected = { viewModel.selectModel(it) },
            onBackendSelected = { backend, deviceId -> viewModel.selectBackend(backend, deviceId) },
            onMaxTokensChanged = { viewModel.setMaxTokens(it) },
            onLoadModel = { model, maxTokens, backend, deviceId ->
                viewModel.selectModel(model)
                viewModel.setMaxTokens(maxTokens)
                if (backend != null) viewModel.selectBackend(backend, deviceId)
                viewModel.loadModel()
            },
            onUnloadModel = { viewModel.unloadModel() },
            onDismiss = { showSettingsSheet = false }
        )
    }
}
