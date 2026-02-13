package com.llmhub.llmhub.components

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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelConfig
import com.llmhub.llmhub.data.ModelPreferences
import com.llmhub.llmhub.data.hasDownloadedVisionProjector
import com.llmhub.llmhub.data.requiresExternalVisionProjector
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Bottom sheet for AI Chat settings - combines model selection with model configs
 * Similar to other feature screens (Writing Aid, Scam Detector, etc.)
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatSettingsSheet(
    availableModels: List<LLMModel>,
    initialSelectedModel: LLMModel?,
    initialSelectedBackend: LlmInference.Backend?,
    initialSelectedNpuDeviceId: String?,
    currentlyLoadedModel: LLMModel?,
    isLoadingModel: Boolean,
    onModelSelected: (LLMModel) -> Unit,
    onBackendSelected: (LlmInference.Backend, String?) -> Unit,
    onLoadModel: (model: LLMModel, maxTokens: Int, topK: Int, topP: Float, temperature: Float, backend: LlmInference.Backend?, disableVision: Boolean, disableAudio: Boolean) -> Unit,
    onUnloadModel: () -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val focusManager = LocalFocusManager.current
    val scope = rememberCoroutineScope()
    val modelPrefs = ModelPreferences(context)
    
    var showModelMenu by remember { mutableStateOf(false) }
    var showBackendMenu by remember { mutableStateOf(false) }
    
    // Selected model state (use initial value from ViewModel)
    var selectedModel by remember { 
        mutableStateOf(initialSelectedModel ?: currentlyLoadedModel ?: availableModels.firstOrNull()) 
    }
    
    // Model-specific configurations
    val baseMaxTokensCap = remember(selectedModel) { 
        selectedModel?.let { MediaPipeInferenceService.getMaxTokensForModelStatic(it) } ?: 2048 
    }


    
    // Gemma-3n detection (used for modality toggles)
    val isGemma3nModel = remember(selectedModel) {
        selectedModel?.name?.contains("Gemma-3n", ignoreCase = true) == true
    }
    
    // Phi-4 Mini detection
    val isPhi4Mini = remember(selectedModel) {
        selectedModel?.name?.contains("Phi-4 Mini", ignoreCase = true) == true
    }
    
    // Allow accelerator selection for any model that declares GPU support
    val canSelectAccelerator = remember(selectedModel) {
        selectedModel?.supportsGpu == true
    }
    
    val defaultUseGpu = remember(selectedModel) { 
        if (isPhi4Mini) false else selectedModel?.supportsGpu == true
    }
    
    // Config state
    var maxTokensValue by remember { mutableStateOf(minOf(4096, baseMaxTokensCap)) }
    var maxTokensText by remember { mutableStateOf(minOf(4096, baseMaxTokensCap).toString()) }
    var topK by remember { mutableStateOf(64) }
    var topP by remember { mutableStateOf(0.95f) }
    var temperature by remember { mutableStateOf(1.0f) }
    // Initialize useGpu from initial backend passed in
    var useGpu by remember { 
        mutableStateOf(
            when (initialSelectedBackend) {
                LlmInference.Backend.GPU -> true
                LlmInference.Backend.CPU -> false
                else -> defaultUseGpu
            }
        )
    }

    // Track whether NPU (Hexagon GGUF) is selected — initial value comes from the caller
    var useNpu by remember(initialSelectedNpuDeviceId) { mutableStateOf(initialSelectedNpuDeviceId != null) }

    var disableVision by remember { mutableStateOf(isGemma3nModel) }
    var disableAudio by remember { mutableStateOf(isGemma3nModel) }

    val selectedModelSupportsVisionInput by remember(selectedModel, context) {
        derivedStateOf {
            selectedModel?.let { model ->
                model.supportsVision &&
                    (!model.requiresExternalVisionProjector() || model.hasDownloadedVisionProjector(context))
            } == true
        }
    }

    // Effective cap depends on whether vision is enabled for the selected model
    val effectiveMaxTokensCap by remember(selectedModel, baseMaxTokensCap, disableVision) {
        derivedStateOf {
            if (selectedModelSupportsVisionInput && !disableVision) minOf(baseMaxTokensCap, 8192) else baseMaxTokensCap
        }
    }
    
    // Load saved config when model changes
    LaunchedEffect(selectedModel?.name) {
        selectedModel?.let { model ->
            val newBaseCap = MediaPipeInferenceService.getMaxTokensForModelStatic(model)
            val newIsGemma3n = model.name.contains("Gemma-3n", ignoreCase = true)
            val newIsPhi4Mini = model.name.contains("Phi-4 Mini", ignoreCase = true)
            val newDefaultUseGpu = if (newIsPhi4Mini) false else model.supportsGpu
            
            try {
                val saved = modelPrefs.getModelConfig(model.name)
                if (saved != null) {
                    // Clamp saved value to the effective cap
                    maxTokensValue = saved.maxTokens.coerceIn(1, minOf(newBaseCap, if (selectedModelSupportsVisionInput && !saved.disableVision) 8192 else newBaseCap))
                    maxTokensText = maxTokensValue.toString()
                    topK = saved.topK
                    topP = saved.topP
                    temperature = saved.temperature
                    useGpu = when (saved.backend) {
                        "GPU" -> true
                        "CPU" -> false
                        else -> newDefaultUseGpu
                    }
                    disableVision = saved.disableVision || !selectedModelSupportsVisionInput
                    disableAudio = saved.disableAudio
                } else {
                    // Reset to defaults for new model
                    val effCap = if (selectedModelSupportsVisionInput && !newIsGemma3n) minOf(newBaseCap, 8192) else newBaseCap
                    val defaultMax = minOf(4096, effCap)
                    maxTokensValue = defaultMax
                    maxTokensText = defaultMax.toString()
                    topK = 64
                    topP = 0.95f
                    temperature = 1.0f
                    useGpu = newDefaultUseGpu
                    disableVision = newIsGemma3n || !selectedModelSupportsVisionInput
                    disableAudio = newIsGemma3n
                }
            } catch (e: Exception) {
                // Reset to defaults on error
                val effCap = if (selectedModelSupportsVisionInput) minOf(newBaseCap, 8192) else newBaseCap
                val defaultMax = minOf(4096, effCap)
                maxTokensValue = defaultMax
                maxTokensText = defaultMax.toString()
                topK = 64
                topP = 0.95f
                temperature = 1.0f
                useGpu = newDefaultUseGpu
                disableVision = newIsGemma3n || !selectedModelSupportsVisionInput
                disableAudio = newIsGemma3n
            }
        }
    }

    LaunchedEffect(selectedModelSupportsVisionInput) {
        if (!selectedModelSupportsVisionInput) {
            disableVision = true
        }
    }
    
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp)
                .padding(bottom = 32.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Title
            Text(
                text = stringResource(R.string.feature_settings_title),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            
            // Model Selector Card
            Card(
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Select Model Title
                    Text(
                        text = stringResource(R.string.select_model_title),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    
                    if (availableModels.isEmpty()) {
                        // No models available
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Default.Warning,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.error
                            )
                            Column {
                                Text(
                                    text = stringResource(R.string.no_models_available),
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.Medium
                                )
                                Text(
                                    text = stringResource(R.string.download_models_first),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    } else {
                        // Model Dropdown
                        ExposedDropdownMenuBox(
                            expanded = showModelMenu,
                            onExpandedChange = { showModelMenu = !showModelMenu }
                        ) {
                            OutlinedTextField(
                                value = selectedModel?.name ?: stringResource(R.string.select_model),
                                onValueChange = {},
                                readOnly = true,
                                label = { Text(stringResource(R.string.select_model)) },
                                trailingIcon = { 
                                    ExposedDropdownMenuDefaults.TrailingIcon(expanded = showModelMenu) 
                                },
                                leadingIcon = {
                                    Icon(Icons.Default.Memory, contentDescription = null)
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .menuAnchor(),
                                colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(),
                                shape = RoundedCornerShape(16.dp)
                            )
                            ExposedDropdownMenu(
                                expanded = showModelMenu,
                                onDismissRequest = { showModelMenu = false }
                            ) {
                                availableModels.forEach { model ->
                                    DropdownMenuItem(
                                        text = { 
                                            Column {
                                                Row(
                                                    verticalAlignment = Alignment.CenterVertically
                                                ) {
                                                    Text(
                                                        text = model.name,
                                                        style = MaterialTheme.typography.bodyMedium,
                                                        fontWeight = FontWeight.Medium,
                                                        modifier = Modifier.weight(1f)
                                                    )
                                                    // Vision badge
                                                    if (model.supportsVision) {
                                                        Spacer(modifier = Modifier.width(4.dp))
                                                        Icon(
                                                            Icons.Default.RemoveRedEye,
                                                            contentDescription = null,
                                                            modifier = Modifier.size(14.dp),
                                                            tint = MaterialTheme.colorScheme.tertiary
                                                        )
                                                    }
                                                    // Audio badge
                                                    if (model.supportsAudio) {
                                                        Spacer(modifier = Modifier.width(4.dp))
                                                        Icon(
                                                            Icons.Default.Mic,
                                                            contentDescription = null,
                                                            modifier = Modifier.size(14.dp),
                                                            tint = MaterialTheme.colorScheme.secondary
                                                        )
                                                    }
                                                }
                                                // Context window info
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
                                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                                    )
                                                }
                                            }
                                        },
                                        onClick = {
                                            selectedModel = model
                                            onModelSelected(model)
                                            showModelMenu = false
                                        },
                                        leadingIcon = {
                                            if (selectedModel?.name == model.name) {
                                                Icon(
                                                    Icons.Default.CheckCircle,
                                                    contentDescription = null,
                                                    tint = MaterialTheme.colorScheme.primary
                                                )
                                            }
                                        },
                                        trailingIcon = {
                                            if (currentlyLoadedModel?.name == model.name) {
                                                Icon(
                                                    Icons.Default.Check,
                                                    contentDescription = stringResource(R.string.currently_loaded),
                                                    tint = MaterialTheme.colorScheme.primary,
                                                    modifier = Modifier.size(18.dp)
                                                )
                                            }
                                        }
                                    )
                                }
                            }
                        }
                        
                        // Backend Selection
                        if (canSelectAccelerator) {
                            ExposedDropdownMenuBox(
                                expanded = showBackendMenu,
                                onExpandedChange = { showBackendMenu = !showBackendMenu }
                            ) {
                                OutlinedTextField(
                                    value = when {
                                        useNpu -> stringResource(R.string.backend_npu)
                                        useGpu -> stringResource(R.string.backend_gpu)
                                        else -> stringResource(R.string.backend_cpu)
                                    },
                                    onValueChange = {},
                                    readOnly = true,
                                    label = { Text(stringResource(R.string.select_backend)) },
                                    trailingIcon = { 
                                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = showBackendMenu) 
                                    },
                                    leadingIcon = {
                                        Icon(
                                            if (useGpu) Icons.Default.Bolt else Icons.Default.Computer,
                                            contentDescription = null
                                        )
                                    },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .menuAnchor(),
                                    colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(),
                                    shape = RoundedCornerShape(16.dp)
                                )
                                ExposedDropdownMenu(
                                    expanded = showBackendMenu,
                                    onDismissRequest = { showBackendMenu = false }
                                ) {
                                    DropdownMenuItem(
                                        text = { Text(stringResource(R.string.backend_cpu)) },
                                        onClick = {
                                            useGpu = false
                                            useNpu = false
                                            onBackendSelected(LlmInference.Backend.CPU, null)
                                            showBackendMenu = false
                                        },
                                        leadingIcon = { Icon(Icons.Default.Computer, contentDescription = null) },
                                        trailingIcon = {
                                            if (!useGpu && !useNpu) {
                                                Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                                            }
                                        }
                                    )

                                    // GPU option (clears any NPU device)
                                    DropdownMenuItem(
                                        text = { Text(stringResource(R.string.backend_gpu)) },
                                        onClick = {
                                            useGpu = true
                                            useNpu = false
                                            onBackendSelected(LlmInference.Backend.GPU, null)
                                            showBackendMenu = false
                                        },
                                        leadingIcon = { Icon(Icons.Default.Bolt, contentDescription = null) },
                                        trailingIcon = {
                                            if (useGpu && !useNpu) {
                                                Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                                            }
                                        }
                                    )

                                    // NPU option for GGUF on Qualcomm Hexagon
                                    val showNpuOption = selectedModel?.modelFormat == "gguf" && com.llmhub.llmhub.data.DeviceInfo.isQualcommNpuSupported()
                                    if (showNpuOption) {
                                        DropdownMenuItem(
                                            text = { Text(stringResource(R.string.backend_npu)) },
                                            onClick = {
                                                useGpu = true
                                                useNpu = true
                                                onBackendSelected(LlmInference.Backend.GPU, "HTP0")
                                                showBackendMenu = false
                                            },
                                            leadingIcon = { Icon(Icons.Default.Memory, contentDescription = null) },
                                            trailingIcon = {
                                                if (useGpu && useNpu) {
                                                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                                                }
                                            }
                                        )
                                    }
                                }
                            }
                        }
                        
                        // Load/Unload button
                        if (currentlyLoadedModel?.name == selectedModel?.name) {
                            // Unload button
                            OutlinedButton(
                                onClick = onUnloadModel,
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(16.dp),
                                enabled = !isLoadingModel
                            ) {
                                Icon(Icons.Default.PowerOff, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.unload_model))
                            }
                        } else {
                            // Load button
                            Button(
                                onClick = {
                                    selectedModel?.let { model ->
                                        // Ensure finalMax respects the effective cap when vision is enabled
                                        val finalMax = maxTokensValue.coerceIn(1, if (selectedModelSupportsVisionInput && !disableVision) minOf(baseMaxTokensCap, 8192) else baseMaxTokensCap)
                                        val backend = if (canSelectAccelerator) {
                                            if (useGpu) LlmInference.Backend.GPU else LlmInference.Backend.CPU
                                        } else null
                                        
                                        // Save config
                                        scope.launch(Dispatchers.IO) {
                                            try {
                                                val cfg = ModelConfig(
                                                    maxTokens = finalMax,
                                                    topK = topK,
                                                    topP = topP,
                                                    temperature = temperature,
                                                    backend = backend?.name,
                                                    disableVision = disableVision,
                                                    disableAudio = disableAudio
                                                )
                                                modelPrefs.setModelConfig(model.name, cfg)
                                            } catch (_: Exception) {}
                                        }
                                        
                                        onLoadModel(model, finalMax, topK, topP, temperature, backend, disableVision, disableAudio)
                                    }
                                },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(16.dp),
                                enabled = selectedModel != null && !isLoadingModel
                            ) {
                                if (isLoadingModel) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(20.dp),
                                        color = MaterialTheme.colorScheme.onPrimary,
                                        strokeWidth = 2.dp
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                }
                                Text(stringResource(R.string.load_model))
                            }
                        }
                    }
                }
            }
            
            // Model Configs Section (only show when a model is selected)
            if (selectedModel != null) {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Model Configs Title
                        Text(
                            text = stringResource(R.string.model_configs_title),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                        
                        // Max tokens
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = stringResource(R.string.max_tokens),
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "${effectiveMaxTokensCap} ${stringResource(R.string.max)}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Slider(
                                value = maxTokensValue.toFloat(),
                                onValueChange = {
                                    val cap = if (selectedModelSupportsVisionInput && !disableVision) minOf(baseMaxTokensCap, 8192) else baseMaxTokensCap
                                    val intVal = it.toInt().coerceIn(1, cap)
                                    maxTokensValue = intVal
                                    maxTokensText = intVal.toString()
                                },
                                valueRange = 1f..effectiveMaxTokensCap.toFloat(),
                                modifier = Modifier.weight(1f).height(36.dp),
                                thumb = {
                                    SliderDefaults.Thumb(
                                        interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                                        thumbSize = androidx.compose.ui.unit.DpSize(24.dp, 24.dp)
                                    )
                                }
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            OutlinedTextField(
                                value = maxTokensText,
                                onValueChange = { input ->
                                    val numeric = input.filter { it.isDigit() }
                                    val intVal = numeric.toIntOrNull() ?: 0
                                    val cap = if (selectedModelSupportsVisionInput && !disableVision) minOf(baseMaxTokensCap, 8192) else baseMaxTokensCap
                                    val clamped = intVal.coerceIn(1, cap)
                                    maxTokensText = clamped.toString()
                                    maxTokensValue = clamped
                                },
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                singleLine = true,
                                modifier = Modifier.width(72.dp)
                            )
                        }
                        
                        // TopK
                        Text(
                            text = stringResource(R.string.top_k),
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Slider(
                                value = topK.toFloat(),
                                onValueChange = { topK = it.toInt() },
                                valueRange = 1f..256f,
                                modifier = Modifier.weight(1f).height(28.dp),
                                thumb = {
                                    SliderDefaults.Thumb(
                                        interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                                        thumbSize = androidx.compose.ui.unit.DpSize(24.dp, 24.dp)
                                    )
                                }
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            OutlinedTextField(
                                value = topK.toString(),
                                onValueChange = { v -> topK = v.filter { it.isDigit() }.toIntOrNull() ?: topK },
                                modifier = Modifier.width(72.dp),
                                singleLine = true
                            )
                        }
                        
                        // TopP
                        Text(
                            text = stringResource(R.string.top_p),
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Slider(
                                value = topP,
                                onValueChange = { topP = it },
                                valueRange = 0.0f..1.0f,
                                modifier = Modifier.weight(1f).height(28.dp),
                                thumb = {
                                    SliderDefaults.Thumb(
                                        interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                                        thumbSize = androidx.compose.ui.unit.DpSize(24.dp, 24.dp)
                                    )
                                }
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            OutlinedTextField(
                                value = String.format("%.2f", topP),
                                onValueChange = { v -> topP = v.toFloatOrNull() ?: topP },
                                modifier = Modifier.width(72.dp),
                                singleLine = true
                            )
                        }
                        
                        // Temperature
                        Text(
                            text = stringResource(R.string.temperature),
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Slider(
                                value = temperature,
                                onValueChange = { temperature = it },
                                valueRange = 0.0f..2.0f,
                                modifier = Modifier.weight(1f).height(28.dp),
                                thumb = {
                                    SliderDefaults.Thumb(
                                        interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                                        thumbSize = androidx.compose.ui.unit.DpSize(24.dp, 24.dp)
                                    )
                                }
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            OutlinedTextField(
                                value = String.format("%.2f", temperature),
                                onValueChange = { v -> temperature = v.toFloatOrNull() ?: temperature },
                                modifier = Modifier.width(72.dp),
                                singleLine = true
                            )
                        }
                        
                        // Modality options (Vision/Audio toggles)
                        if (selectedModelSupportsVisionInput || selectedModel?.supportsAudio == true) {
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = stringResource(R.string.modality_options),
                                style = MaterialTheme.typography.bodyMedium
                            )
                            
                            // Vision toggle
                            if (selectedModelSupportsVisionInput) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = stringResource(R.string.enable_vision),
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Switch(
                                        checked = !disableVision,
                                        onCheckedChange = { disableVision = !it }
                                    )
                                }
                            }
                            
                            // Audio toggle
                            if (selectedModel?.supportsAudio == true) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = stringResource(R.string.enable_audio),
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                    Switch(
                                        checked = !disableAudio,
                                        onCheckedChange = { disableAudio = !it }
                                    )
                                }
                            }
                            
                            // Performance tip
                            Text(
                                text = stringResource(R.string.gemma3n_performance_tip),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                        
                        // Reset to Defaults button
                        TextButton(
                            onClick = {
                                selectedModel?.let { model ->
                                    // Remove saved config
                                    scope.launch(Dispatchers.IO) {
                                        try {
                                            modelPrefs.removeModelConfig(model.name)
                                        } catch (_: Exception) {}
                                    }
                                    
                                    // Reset to defaults
                                    val newMaxTokensCap = MediaPipeInferenceService.getMaxTokensForModelStatic(model)
                                    val newIsGemma3n = model.name.contains("Gemma-3n", ignoreCase = true)
                                    val newIsPhi4Mini = model.name.contains("Phi-4 Mini", ignoreCase = true)
                                    val newDefaultUseGpu = if (newIsPhi4Mini) false else model.supportsGpu
                                    val defaultMax = minOf(4096, newMaxTokensCap)
                                    
                                    maxTokensValue = defaultMax
                                    maxTokensText = defaultMax.toString()
                                    topK = 64
                                    topP = 0.95f
                                    temperature = 1.0f
                                    useGpu = newDefaultUseGpu
                                    disableVision = newIsGemma3n || !selectedModelSupportsVisionInput
                                    disableAudio = newIsGemma3n
                                }
                            },
                            modifier = Modifier.align(Alignment.End)
                        ) {
                            Text(stringResource(R.string.reset_to_defaults))
                        }
                    }
                }
            }
        }
    }
}
