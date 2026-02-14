package com.llmhub.llmhub.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Computer
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PowerOff
import androidx.compose.material.icons.filled.RemoveRedEye
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.*
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.hasDownloadedVisionProjector
import com.llmhub.llmhub.data.requiresExternalVisionProjector
import com.llmhub.llmhub.inference.MediaPipeInferenceService

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FeatureModelSettingsSheet(
    availableModels: List<LLMModel>,
    initialSelectedModel: LLMModel?,
    initialSelectedBackend: LlmInference.Backend?,
    initialSelectedNpuDeviceId: String?,
    initialMaxTokens: Int,
    currentlyLoadedModel: LLMModel?,
    isLoadingModel: Boolean,
    onModelSelected: (LLMModel) -> Unit,
    onBackendSelected: (LlmInference.Backend, String?) -> Unit,
    onMaxTokensChanged: (Int) -> Unit,
    onLoadModel: (model: LLMModel, maxTokens: Int, backend: LlmInference.Backend?, deviceId: String?) -> Unit,
    onUnloadModel: () -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current

    var showModelMenu by remember { mutableStateOf(false) }
    var showBackendMenu by remember { mutableStateOf(false) }
    var selectedModel by remember { mutableStateOf(initialSelectedModel ?: currentlyLoadedModel ?: availableModels.firstOrNull()) }

    val baseMaxTokensCap = remember(selectedModel) {
        selectedModel?.let { MediaPipeInferenceService.getMaxTokensForModelStatic(it) } ?: 2048
    }
    val isPhi4Mini = remember(selectedModel) {
        selectedModel?.name?.contains("Phi-4 Mini", ignoreCase = true) == true
    }
    val canSelectAccelerator = remember(selectedModel) {
        selectedModel?.supportsGpu == true
    }
    val defaultUseGpu = remember(selectedModel) {
        if (isPhi4Mini) false else selectedModel?.supportsGpu == true
    }

    var maxTokensValue by remember { mutableStateOf(initialMaxTokens.coerceIn(1, baseMaxTokensCap.coerceAtLeast(1))) }
    var maxTokensText by remember { mutableStateOf(maxTokensValue.toString()) }
    var useGpu by remember {
        mutableStateOf(
            when (initialSelectedBackend) {
                LlmInference.Backend.GPU -> true
                LlmInference.Backend.CPU -> false
                else -> defaultUseGpu
            }
        )
    }
    var useNpu by remember(initialSelectedNpuDeviceId) { mutableStateOf(initialSelectedNpuDeviceId != null) }

    val selectedModelSupportsVisionInput by remember(selectedModel, context) {
        derivedStateOf {
            selectedModel?.let { model ->
                model.supportsVision &&
                    (!model.requiresExternalVisionProjector() || model.hasDownloadedVisionProjector(context))
            } == true
        }
    }

    LaunchedEffect(selectedModel?.name, baseMaxTokensCap) {
        maxTokensValue = maxTokensValue.coerceIn(1, baseMaxTokensCap.coerceAtLeast(1))
        maxTokensText = maxTokensValue.toString()
        onMaxTokensChanged(maxTokensValue)
        if (selectedModel != null) {
            useGpu = if (isPhi4Mini) false else useGpu
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
            Text(
                text = stringResource(R.string.feature_settings_title),
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )

            Card(
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = stringResource(R.string.select_model_title),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    if (availableModels.isEmpty()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(Icons.Default.Warning, contentDescription = null, tint = MaterialTheme.colorScheme.error)
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
                        ExposedDropdownMenuBox(
                            expanded = showModelMenu,
                            onExpandedChange = { showModelMenu = !showModelMenu }
                        ) {
                            OutlinedTextField(
                                value = selectedModel?.name ?: stringResource(R.string.select_model),
                                onValueChange = {},
                                readOnly = true,
                                label = { Text(stringResource(R.string.select_model)) },
                                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = showModelMenu) },
                                leadingIcon = { Icon(Icons.Default.Memory, contentDescription = null) },
                                modifier = Modifier.fillMaxWidth().menuAnchor(),
                                colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(),
                                shape = RoundedCornerShape(16.dp)
                            )
                            ExposedDropdownMenu(expanded = showModelMenu, onDismissRequest = { showModelMenu = false }) {
                                availableModels.forEach { model ->
                                    DropdownMenuItem(
                                        text = {
                                            Column {
                                                Row(verticalAlignment = Alignment.CenterVertically) {
                                                    Text(
                                                        text = model.name,
                                                        style = MaterialTheme.typography.bodyMedium,
                                                        fontWeight = FontWeight.Medium,
                                                        modifier = Modifier.weight(1f)
                                                    )
                                                    if (model.supportsVision) {
                                                        Spacer(modifier = Modifier.width(4.dp))
                                                        Icon(Icons.Default.RemoveRedEye, contentDescription = null, modifier = Modifier.size(14.dp), tint = MaterialTheme.colorScheme.tertiary)
                                                    }
                                                    if (model.supportsAudio) {
                                                        Spacer(modifier = Modifier.width(4.dp))
                                                        Icon(Icons.Default.Mic, contentDescription = null, modifier = Modifier.size(14.dp), tint = MaterialTheme.colorScheme.secondary)
                                                    }
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
                                                Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                                            }
                                        },
                                        trailingIcon = {
                                            if (currentlyLoadedModel?.name == model.name) {
                                                Icon(Icons.Default.Check, contentDescription = stringResource(R.string.currently_loaded), tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp))
                                            }
                                        }
                                    )
                                }
                            }
                        }

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
                                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = showBackendMenu) },
                                    leadingIcon = {
                                        Icon(
                                            when {
                                                useNpu -> Icons.Default.Bolt
                                                useGpu -> Icons.Default.Speed
                                                else -> Icons.Default.Computer
                                            },
                                            contentDescription = null
                                        )
                                    },
                                    modifier = Modifier.fillMaxWidth().menuAnchor(),
                                    colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(),
                                    shape = RoundedCornerShape(16.dp)
                                )
                                ExposedDropdownMenu(expanded = showBackendMenu, onDismissRequest = { showBackendMenu = false }) {
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
                                            if (!useGpu) {
                                                Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                                            }
                                        }
                                    )

                                    val gpuSupported = selectedModel?.supportsGpu == true
                                    DropdownMenuItem(
                                        text = {
                                            Column {
                                                Text(stringResource(R.string.backend_gpu))
                                                if (!gpuSupported) {
                                                    Text(
                                                        text = stringResource(R.string.gpu_not_supported),
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.error
                                                    )
                                                }
                                            }
                                        },
                                        onClick = {
                                            if (gpuSupported) {
                                                useGpu = true
                                                useNpu = false
                                                onBackendSelected(LlmInference.Backend.GPU, null)
                                                showBackendMenu = false
                                            }
                                        },
                                        enabled = gpuSupported,
                                        leadingIcon = { Icon(Icons.Default.Speed, contentDescription = null) },
                                        trailingIcon = {
                                            if (useGpu && !useNpu && gpuSupported) {
                                                Icon(Icons.Default.CheckCircle, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                                            }
                                        }
                                    )

                                    val showNpuOption = selectedModel?.modelFormat == "gguf" && DeviceInfo.isQualcommNpuSupported()
                                    if (showNpuOption) {
                                        DropdownMenuItem(
                                            text = { Text(stringResource(R.string.backend_npu)) },
                                            onClick = {
                                                useGpu = true
                                                useNpu = true
                                                onBackendSelected(LlmInference.Backend.GPU, "HTP0")
                                                showBackendMenu = false
                                            },
                                            leadingIcon = { Icon(Icons.Default.Bolt, contentDescription = null) },
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

                        if (currentlyLoadedModel?.name == selectedModel?.name) {
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
                            Button(
                                onClick = {
                                    selectedModel?.let { model ->
                                        val finalMax = maxTokensValue.coerceIn(1, baseMaxTokensCap.coerceAtLeast(1))
                                        val backend = if (canSelectAccelerator) {
                                            if (useGpu) LlmInference.Backend.GPU else LlmInference.Backend.CPU
                                        } else null
                                        val deviceId = if (useNpu) "HTP0" else null
                                        onLoadModel(model, finalMax, backend, deviceId)
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

            if (selectedModel != null) {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
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
                                text = "${baseMaxTokensCap} ${stringResource(R.string.max)}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                            Slider(
                                value = maxTokensValue.toFloat(),
                                onValueChange = {
                                    val intVal = it.toInt().coerceIn(1, baseMaxTokensCap.coerceAtLeast(1))
                                    maxTokensValue = intVal
                                    maxTokensText = intVal.toString()
                                    onMaxTokensChanged(intVal)
                                },
                                valueRange = 1f..baseMaxTokensCap.toFloat(),
                                modifier = Modifier.weight(1f).height(36.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            OutlinedTextField(
                                value = maxTokensText,
                                onValueChange = { input ->
                                    val numeric = input.filter { it.isDigit() }
                                    val intVal = (numeric.toIntOrNull() ?: 1).coerceIn(1, baseMaxTokensCap.coerceAtLeast(1))
                                    maxTokensText = intVal.toString()
                                    maxTokensValue = intVal
                                    onMaxTokensChanged(intVal)
                                },
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                singleLine = true,
                                modifier = Modifier.width(72.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}
