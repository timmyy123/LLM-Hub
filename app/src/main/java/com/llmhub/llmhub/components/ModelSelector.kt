package com.llmhub.llmhub.components

import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel
import com.google.mediapipe.tasks.genai.llminference.LlmInference

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelSelectorCard(
    models: List<LLMModel>,
    selectedModel: LLMModel?,
    onModelSelected: (LLMModel) -> Unit,
    selectedBackend: LlmInference.Backend?,
    onBackendSelected: (LlmInference.Backend, String?) -> Unit,
    onLoadModel: () -> Unit,
    isLoading: Boolean,
    isModelLoaded: Boolean = false,
    onUnloadModel: (() -> Unit)? = null,
    filterMultimodalOnly: Boolean = false,
    modifier: Modifier = Modifier,
    extraContent: (@Composable ColumnScope.() -> Unit)? = null
) {
    val filteredModels = if (filterMultimodalOnly) {
        models.filter { it.supportsAudio || it.supportsVision }
    } else {
        models
    }
    
    var showModelMenu by remember { mutableStateOf(false) }
    var showBackendMenu by remember { mutableStateOf(false) }
    
    Card(
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
        modifier = modifier
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Title
            Text(
                text = stringResource(R.string.select_model_title),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
            
            if (filteredModels.isEmpty()) {
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
                            text = stringResource(
                                if (filterMultimodalOnly) R.string.multimodal_models_only 
                                else R.string.download_models_first
                            ),
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
                        filteredModels.forEach { model ->
                            DropdownMenuItem(
                                text = { 
                                    Text(
                                        text = model.name,
                                        style = MaterialTheme.typography.bodyMedium,
                                        fontWeight = FontWeight.Medium
                                    )
                                },
                                onClick = {
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
                                }
                            )
                        }
                    }
                }
                
                // Backend Selection
                AnimatedVisibility(
                    visible = selectedModel != null,
                    enter = fadeIn() + expandVertically()
                ) {
                    ExposedDropdownMenuBox(
                        expanded = showBackendMenu,
                        onExpandedChange = { showBackendMenu = !showBackendMenu }
                    ) {
                        OutlinedTextField(
                            value = when(selectedBackend) {
                                LlmInference.Backend.CPU -> stringResource(R.string.backend_cpu)
                                LlmInference.Backend.GPU -> stringResource(R.string.backend_gpu)
                                else -> stringResource(R.string.select_backend)
                            },
                            onValueChange = {},
                            readOnly = true,
                            label = { Text(stringResource(R.string.select_backend)) },
                            trailingIcon = { 
                                ExposedDropdownMenuDefaults.TrailingIcon(expanded = showBackendMenu) 
                            },
                            leadingIcon = {
                                Icon(
                                    if (selectedBackend == LlmInference.Backend.GPU) Icons.Default.Bolt 
                                    else Icons.Default.Computer,
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
                            // CPU Option
                            DropdownMenuItem(
                                text = { Text(stringResource(R.string.backend_cpu)) },
                                onClick = {
                                    onBackendSelected(LlmInference.Backend.CPU, null)
                                    showBackendMenu = false
                                },
                                leadingIcon = {
                                    Icon(Icons.Default.Computer, contentDescription = null)
                                },
                                trailingIcon = {
                                    if (selectedBackend == LlmInference.Backend.CPU) {
                                        Icon(
                                            Icons.Default.CheckCircle,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                }
                            )
                            
                            // GPU Option
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
                                        // GPU selection clears any previously selected NPU device
                                        onBackendSelected(LlmInference.Backend.GPU, null)
                                        showBackendMenu = false
                                    }
                                },
                                enabled = gpuSupported,
                                leadingIcon = {
                                    Icon(Icons.Default.Bolt, contentDescription = null)
                                },
                                trailingIcon = {
                                    if (selectedBackend == LlmInference.Backend.GPU && gpuSupported) {
                                        Icon(
                                            Icons.Default.CheckCircle,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                }
                            )

                            // NPU Option (GGUF-on-Hexagon). Show only for GGUF models on Qualcomm devices.
                            val showNpuOption = selectedModel?.modelFormat == "gguf" && com.llmhub.llmhub.data.DeviceInfo.isQualcommNpuSupported()
                            if (showNpuOption) {
                                DropdownMenuItem(
                                    text = { Text(stringResource(R.string.backend_npu)) },
                                    onClick = {
                                        // Represent NPU by selecting GPU backend + deviceId="HTP0"
                                        onBackendSelected(LlmInference.Backend.GPU, "HTP0")
                                        showBackendMenu = false
                                    },
                                    leadingIcon = { Icon(Icons.Default.Memory, contentDescription = null) },
                                    trailingIcon = {
                                        // Consider NPU selected when backend==GPU and the ViewModel stored deviceId==HTP0 — UI reflects that via selectedBackend only; the exact device badge is persisted in ViewModel
                                        // We don't have direct access to the selected deviceId here, so visual check will be handled by callers if needed.
                                    }
                                )
                            }
                        }
                    }
                }
                
                // Load/Reload/Unload Model Buttons
                AnimatedVisibility(
                    visible = selectedModel != null && selectedBackend != null,
                    enter = fadeIn() + expandVertically()
                ) {
                    Column(
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        // Load or Reload Model Button
                        Button(
                            onClick = onLoadModel,
                            modifier = Modifier.fillMaxWidth(),
                            enabled = !isLoading,
                            shape = RoundedCornerShape(16.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.primary
                            )
                        ) {
                            if (isLoading) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    color = MaterialTheme.colorScheme.onPrimary
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.model_loading), fontWeight = FontWeight.Bold)
                            } else {
                                Text(
                                    text = stringResource(
                                        if (isModelLoaded) R.string.reload_model 
                                        else R.string.load_model
                                    ), 
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                        
                        // Unload Model Button (only show when model is loaded)
                        if (isModelLoaded && onUnloadModel != null) {
                            OutlinedButton(
                                onClick = onUnloadModel,
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(16.dp),
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = MaterialTheme.colorScheme.error
                                ),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.error)
                            ) {
                                Icon(
                                    Icons.Default.PowerOff,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.unload_model), fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }
            }

            extraContent?.let { content ->
                Spacer(modifier = Modifier.height(8.dp))
                content()
            }
        }
    }
}
