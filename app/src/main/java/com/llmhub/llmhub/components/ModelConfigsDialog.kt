package com.llmhub.llmhub.components

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.*
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.window.Dialog
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelConfigsDialog(
    model: LLMModel,
    initialMaxTokens: Int,
    onConfirm: (maxTokens: Int, topK: Int, topP: Float, temperature: Float, backend: LlmInference.Backend?, disableVision: Boolean, disableAudio: Boolean) -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val configuration = LocalConfiguration.current
    val screenWidthDp = configuration.screenWidthDp.dp
    val screenHeightDp = configuration.screenHeightDp.dp
    val isLandscape = configuration.screenWidthDp > configuration.screenHeightDp

    // Make dialog responsive: slimmer in landscape, allow scroll if height is short
    val dialogWidth = if (isLandscape) {
        (screenWidthDp * 0.62f).coerceAtMost(720.dp)
    } else {
        (screenWidthDp * 0.92f).coerceAtMost(720.dp)
    }

    val dialogMaxHeight = if (isLandscape) {
        (screenHeightDp * 0.86f).coerceAtMost(520.dp)
    } else {
        (screenHeightDp * 0.72f).coerceAtMost(640.dp)
    }

    // Always allow GPU for Gemma-3n models - let users have full control
    val isGemma3nE2B = model.name.contains("Gemma-3n E2B", ignoreCase = true)
    val isGemma3nE4B = model.name.contains("Gemma-3n E4B", ignoreCase = true)
    val isGemma3nModel = isGemma3nE2B || isGemma3nE4B
    val canUseGpu = true // Always allow GPU selection for all models

    // Cap for max tokens
    val maxTokensCap = remember(model) { MediaPipeInferenceService.getMaxTokensForModelStatic(model) }

    // State for fields
    var maxTokensText by remember { mutableStateOf(initialMaxTokens.toString()) }
    // Keep numeric and slider in sync using an Int state for tokens
    var maxTokensValue by remember { mutableStateOf(initialMaxTokens.coerceIn(1, maxTokensCap)) }
    var topK by remember { mutableStateOf(64) }
    var topP by remember { mutableStateOf(0.95f) }
    var temperature by remember { mutableStateOf(1.0f) }
    var useGpu by remember { mutableStateOf(true) } // Default to GPU
    var disableVision by remember { mutableStateOf(false) }
    var disableAudio by remember { mutableStateOf(false) }

    val focusManager = androidx.compose.ui.platform.LocalFocusManager.current

    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .width(dialogWidth)
                .padding(12.dp)
                .heightIn(max = dialogMaxHeight),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            val contentPadding = if (isLandscape) 8.dp else 12.dp
            // chip width will be handled by weight so they stay equal

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(contentPadding)
                    .heightIn(max = dialogMaxHeight)
                    .verticalScroll(rememberScrollState())
                    .clickable(indication = null, interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() }) {
                        // Click outside inputs to clear focus (dismiss keyboard)
                        focusManager.clearFocus()
                    },
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = stringResource(R.string.model_configs_title),
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = androidx.compose.ui.text.font.FontWeight.Bold
                )

                // Max tokens (label row with cap next to label, slider row below takes full width)
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                    Text(text = stringResource(R.string.max_tokens), style = MaterialTheme.typography.bodyMedium)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(text = "${maxTokensCap} ${stringResource(R.string.max)}", style = MaterialTheme.typography.bodySmall)
                }

                 Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                     // Slider takes the available width, matching other sliders' appearance
                     Slider(
                         value = maxTokensValue.toFloat(),
                         onValueChange = {
                             val intVal = it.toInt().coerceIn(1, maxTokensCap)
                             maxTokensValue = intVal
                             maxTokensText = intVal.toString()
                         },
                         valueRange = 1f..maxTokensCap.toFloat(),
                         modifier = Modifier.weight(1f).height(36.dp),
                         // Add more padding around the thumb for easier interaction
                         thumb = {
                             SliderDefaults.Thumb(
                                 interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                                 thumbSize = androidx.compose.ui.unit.DpSize(24.dp, 24.dp)
                             )
                         }
                     )

                    Spacer(modifier = Modifier.width(6.dp))

                    // Numeric field on the right - use same small width as other numeric inputs
                    val smallNumWidth = if (isLandscape) 64.dp else 72.dp
                    OutlinedTextField(
                        value = maxTokensText,
                        onValueChange = { input ->
                            // Only allow numbers and clamp
                            val numeric = input.filter { it.isDigit() }
                            val intVal = numeric.toIntOrNull() ?: 0
                            val clamped = intVal.coerceIn(1, maxTokensCap)
                            maxTokensText = clamped.toString()
                            maxTokensValue = clamped
                        },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        singleLine = true,
                        modifier = Modifier.width(smallNumWidth)
                    )
                }

                // TopK
                Text(text = stringResource(R.string.top_k), style = MaterialTheme.typography.bodyMedium)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    val sliderModifier = Modifier.weight(1f).height(28.dp)
                    val smallNumWidth = if (isLandscape) 64.dp else 72.dp
                    Slider(
                        value = topK.toFloat(), 
                        onValueChange = { topK = it.toInt() }, 
                        valueRange = 1f..256f, 
                        modifier = sliderModifier,
                        // Add larger thumb for easier interaction
                        thumb = {
                            SliderDefaults.Thumb(
                                interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                                thumbSize = androidx.compose.ui.unit.DpSize(24.dp, 24.dp)
                            )
                        }
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    OutlinedTextField(value = topK.toString(), onValueChange = { v -> topK = v.filter { it.isDigit() }.toIntOrNull() ?: topK }, modifier = Modifier.width(smallNumWidth))
                }

                 // TopP
                 Text(text = stringResource(R.string.top_p), style = MaterialTheme.typography.bodyMedium)
                 Row(verticalAlignment = Alignment.CenterVertically) {
                     val sliderModifier = Modifier.weight(1f).height(28.dp)
                     val smallNumWidth = if (isLandscape) 64.dp else 72.dp
                     Slider(
                         value = topP, 
                         onValueChange = { topP = it }, 
                         valueRange = 0.0f..1.0f, 
                         modifier = sliderModifier,
                         // Add larger thumb for easier interaction at edges
                         thumb = {
                             SliderDefaults.Thumb(
                                 interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                                 thumbSize = androidx.compose.ui.unit.DpSize(24.dp, 24.dp)
                             )
                         }
                     )
                     Spacer(modifier = Modifier.width(8.dp))
                     OutlinedTextField(value = String.format("%.2f", topP), onValueChange = { v -> topP = v.toFloatOrNull() ?: topP }, modifier = Modifier.width(smallNumWidth))
                 }

                // Temperature
                Text(text = stringResource(R.string.temperature), style = MaterialTheme.typography.bodyMedium)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    val sliderModifier = Modifier.weight(1f).height(28.dp)
                    val smallNumWidth = if (isLandscape) 64.dp else 72.dp
                    Slider(
                        value = temperature, 
                        onValueChange = { temperature = it }, 
                        valueRange = 0.0f..2.0f, 
                        modifier = sliderModifier,
                        // Add larger thumb for easier interaction
                        thumb = {
                            SliderDefaults.Thumb(
                                interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() },
                                thumbSize = androidx.compose.ui.unit.DpSize(24.dp, 24.dp)
                            )
                        }
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    OutlinedTextField(value = String.format("%.2f", temperature), onValueChange = { v -> temperature = v.toFloatOrNull() ?: temperature }, modifier = Modifier.width(smallNumWidth))
                }

                // Accelerator toggle only for Gemma models
                if (model.name.contains("Gemma", ignoreCase = true)) {
                    Text(text = stringResource(R.string.choose_accelerator), style = MaterialTheme.typography.bodyMedium)
                    
                    // CPU/GPU Selection - always available
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilterChip(
                            selected = !useGpu,
                            onClick = { useGpu = false },
                            label = { Text("CPU") },
                            leadingIcon = { if (!useGpu) Icon(Icons.Filled.Check, contentDescription = null) },
                            modifier = Modifier.weight(1f).height(44.dp)
                        )
                        FilterChip(
                            selected = useGpu,
                            onClick = { useGpu = true },
                            label = { Text("GPU") },
                            leadingIcon = { if (useGpu) Icon(Icons.Filled.Check, contentDescription = null) },
                            modifier = Modifier.weight(1f).height(44.dp)
                        )
                    }
                    
                    // Always show toggles for multimodal models (E2B and E4B)
                    if (isGemma3nModel && (model.supportsVision || model.supportsAudio)) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = stringResource(R.string.modality_options), 
                            style = MaterialTheme.typography.bodyMedium
                        )
                        
                        // Vision toggle
                        if (model.supportsVision) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Switch(
                                    checked = !disableVision,
                                    onCheckedChange = { disableVision = !it }
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = stringResource(R.string.enable_vision),
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        }
                        
                        // Audio toggle
                        if (model.supportsAudio) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Switch(
                                    checked = !disableAudio,
                                    onCheckedChange = { disableAudio = !it }
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    text = stringResource(R.string.enable_audio),
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        }
                        
                        // General performance tip for all devices
                        Text(
                            text = stringResource(R.string.gemma3n_performance_tip),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                // Actions
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    TextButton(onClick = onDismiss) { Text(stringResource(R.string.cancel)) }
                    Spacer(modifier = Modifier.width(8.dp))
                    Button(onClick = {
                        val finalMax = maxTokensValue.coerceIn(1, maxTokensCap)
                        val backend = if (model.name.contains("Gemma", ignoreCase = true)) {
                            if (useGpu) LlmInference.Backend.GPU else LlmInference.Backend.CPU
                        } else {
                            null
                        }
                        onConfirm(finalMax.coerceIn(1, maxTokensCap), topK, topP, temperature, backend, disableVision, disableAudio)
                        onDismiss()
                    }) { Text(stringResource(R.string.ok)) }
                }
            }
        }
    }
}

