package com.llmhub.llmhub.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.GetApp
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.ui.res.stringResource
import com.llmhub.llmhub.R
import com.llmhub.llmhub.viewmodels.ChatViewModelFactory
import com.llmhub.llmhub.viewmodels.CreatorViewModel
import com.llmhub.llmhub.components.FeatureModelSettingsSheet
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.activity.ComponentActivity
import android.graphics.Rect
import android.view.ViewTreeObserver
import com.llmhub.llmhub.LlmHubApplication

@OptIn(ExperimentalMaterial3Api::class, androidx.compose.foundation.ExperimentalFoundationApi::class)
@Composable
fun CreatorGenerationScreen(
    onNavigateBack: () -> Unit,
    onNavigateToChat: (String) -> Unit, // Navigate to new chat with creator ID
    onNavigateToModels: (() -> Unit)? = null,
    viewModelFactory: ChatViewModelFactory
) {
    val viewModel: CreatorViewModel = viewModel(factory = viewModelFactory)
    val isGenerating by viewModel.isGenerating.collectAsState()
    val generatedCreator by viewModel.generatedCreator.collectAsState()
    val error by viewModel.error.collectAsState()
    
    // Model States
    val availableModels by viewModel.availableModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    val selectedNpuDeviceId by viewModel.selectedNpuDeviceId.collectAsState()
    val selectedMaxTokens by viewModel.selectedMaxTokens.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    
    var userPrompt by remember { mutableStateOf("") }
    var showSettingsSheet by remember { mutableStateOf(false) }
    var navigatingBack by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()

    // Keyboard handling
    val promptBringRequester = remember { BringIntoViewRequester() }
    var promptFocused by remember { mutableStateOf(false) }
    
    // Ad gating for free users
    val context = LocalContext.current
    val activity = context as ComponentActivity
    val isPremium by (context.applicationContext as LlmHubApplication).billingManager.isPremium.collectAsState(initial = false)
    val rewardedAdManager = remember { (context.applicationContext as LlmHubApplication).rewardedAdManager }

    // Detect keyboard (IME) visibility
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

    // Unload model when leaving this screen
    DisposableEffect(Unit) {
        onDispose { if (!navigatingBack) viewModel.stopAndUnloadOnExit() }
    }

    LaunchedEffect(imeVisible.value) {
        if (imeVisible.value && promptFocused) {
            promptBringRequester.bringIntoView()
        }
    }

    LaunchedEffect(generatedCreator) {
        if (generatedCreator != null) {
            scrollState.animateScrollTo(scrollState.maxValue)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.creator_screen_title), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = {
                        navigatingBack = true
                        viewModel.stopAndUnloadOnExit()
                        onNavigateBack()
                    }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = stringResource(R.string.feature_settings_title))
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Transparent,
                    titleContentColor = MaterialTheme.colorScheme.onBackground
                )
            )
        }
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
                        if (availableModels.isEmpty()) onNavigateToModels?.invoke() ?: run { showSettingsSheet = true }
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
                .padding(paddingValues)
                .imePadding() // Fix for keyboard overlay
                .padding(16.dp)
                .verticalScroll(scrollState),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Header Card
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = stringResource(R.string.creator_bring_to_life),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = stringResource(R.string.creator_description),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // Input Area
            OutlinedTextField(
                value = userPrompt,
                onValueChange = { userPrompt = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(150.dp)
                    .bringIntoViewRequester(promptBringRequester)
                    .onFocusChanged { promptFocused = it.isFocused },
                minLines = 3,
                label = { Text(stringResource(R.string.creator_prompt_label)) },
                placeholder = { Text(stringResource(R.string.creator_prompt_placeholder)) },
                shape = RoundedCornerShape(12.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = MaterialTheme.colorScheme.outline
                ),
                enabled = !isGenerating // Disable input while generating
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Generate Button
            Button(
                onClick = { viewModel.generateCreator(userPrompt) },
                enabled = userPrompt.isNotBlank() && !isGenerating && isModelLoaded,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.primary
                ),
                shape = RoundedCornerShape(12.dp)
            ) {
                if (isGenerating) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        color = MaterialTheme.colorScheme.onPrimary,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.creator_brewing))
                } else {
                    Icon(Icons.Default.AutoAwesome, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.creator_generate_persona))
                }
            }
            
            if (!isModelLoaded && userPrompt.isNotBlank()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = stringResource(R.string.creator_load_model_first),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.error
                )
            }

            // Error Display
            if (error != null) {
                Spacer(modifier = Modifier.height(16.dp))
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = error ?: "",
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        modifier = Modifier.padding(16.dp),
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // Result Display
            if (generatedCreator != null) {
                Spacer(modifier = Modifier.height(24.dp))
                
                Text(
                    text = stringResource(R.string.creator_result_heading),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.align(Alignment.Start)
                )
                
                Spacer(modifier = Modifier.height(8.dp))

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = generatedCreator!!.icon,
                                style = MaterialTheme.typography.displayMedium
                            )
                            Spacer(modifier = Modifier.width(16.dp))
                            Column {
                                Text(
                                    text = generatedCreator!!.name,
                                    style = MaterialTheme.typography.titleLarge,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = generatedCreator!!.description,
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        
                        HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
                        
                        Text(
                            text = stringResource(R.string.creator_system_prompt_label),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Text(
                            text = generatedCreator!!.pctfPrompt,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier
                                .background(
                                    MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
                                    RoundedCornerShape(8.dp)
                                )
                                .padding(8.dp)
                                .fillMaxWidth()
                        )
                        
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        Button(
                            onClick = {
                                viewModel.saveCreator(generatedCreator!!) {
                                    // Navigate to a new chat with this creator
                                    onNavigateToChat(generatedCreator!!.id)
                                }
                            },
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.primary,
                                contentColor = MaterialTheme.colorScheme.onPrimary
                            )
                        ) {
                            Icon(Icons.Default.Save, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(stringResource(R.string.creator_save_and_chat))
                        }
                    }
                }
            }
        }
        } // end else
    }

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
            onLoadModel = { model, maxTokens, backend, deviceId, nGpuLayers, enableThinking ->
                viewModel.selectModel(model)
                viewModel.setMaxTokens(maxTokens)
                if (backend != null) viewModel.selectBackend(backend, deviceId)
                viewModel.setNGpuLayers(nGpuLayers)
                viewModel.setEnableThinking(enableThinking)
                if (isPremium) {
                    viewModel.loadModel()
                } else {
                    rewardedAdManager.showAdOrGrant(activity) { viewModel.loadModel() }
                }
            },
            onUnloadModel = { viewModel.unloadModel() },
            onDismiss = { showSettingsSheet = false }
        )
    }
}
