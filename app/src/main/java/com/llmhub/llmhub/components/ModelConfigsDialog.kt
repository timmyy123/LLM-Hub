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
import com.llmhub.llmhub.data.ModelPreferences
import com.llmhub.llmhub.data.ModelConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

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

    // Gemma-3n detection (used for modality toggles)
    val isGemma3nE2B = model.name.contains("Gemma-3n E2B", ignoreCase = true)
    val isGemma3nE4B = model.name.contains("Gemma-3n E4B", ignoreCase = true)
    val isGemma3nModel = isGemma3nE2B || isGemma3nE4B

    // Phi-4 Mini detection: supports GPU on devices with sufficient memory
    val isPhi4Mini = model.name.contains("Phi-4 Mini", ignoreCase = true)

    // Allow accelerator selection for any model that declares GPU support
    val canSelectAccelerator = model.supportsGpu

    // Cap for max tokens
    val maxTokensCap = remember(model) { MediaPipeInferenceService.getMaxTokensForModelStatic(model) }

    // State for fields
    var maxTokensText by remember { mutableStateOf(initialMaxTokens.toString()) }
    // Keep numeric and slider in sync using an Int state for tokens
    var maxTokensValue by remember { mutableStateOf(initialMaxTokens.coerceIn(1, maxTokensCap)) }
    var topK by remember { mutableStateOf(64) }
    var topP by remember { mutableStateOf(0.95f) }
    var temperature by remember { mutableStateOf(1.0f) }
    val defaultUseGpu = remember(model) { model.supportsGpu }
    var useGpu by remember { mutableStateOf(defaultUseGpu) } // Default accelerator based on model support
    var disableVision by remember { mutableStateOf(false) }
    // Default audio disabled for Gemma-3n models to conserve resources on mobile
    var disableAudio by remember { mutableStateOf(isGemma3nModel) }

    // Load/save preferences for this model
    val modelPrefs = ModelPreferences(context)
    val scope = rememberCoroutineScope()

    // Load saved config when dialog opens
    LaunchedEffect(model.name) {
        try {
            val saved = modelPrefs.getModelConfig(model.name)
                if (saved != null) {
                maxTokensValue = saved.maxTokens.coerceIn(1, maxTokensCap)
                maxTokensText = maxTokensValue.toString()
                topK = saved.topK
                topP = saved.topP
                temperature = saved.temperature
                useGpu = when (saved.backend) {
                    "GPU" -> true
                    "CPU" -> false
                    else -> defaultUseGpu
                }
                disableVision = saved.disableVision
                disableAudio = saved.disableAudio
            }
        } catch (e: Exception) {
            // ignore failures and keep defaults
        }
    }

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

                // Accelerator toggle when model declares GPU support (Gemma, Llama, Phi, etc.)
                if (canSelectAccelerator) {
                    Text(text = stringResource(R.string.choose_accelerator), style = MaterialTheme.typography.bodyMedium)
                    
                    // CPU/GPU Selection - available when model.supportsGpu is true
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
                    
                    // Show toggles for any model that supports vision or audio
                    if (model.supportsVision || model.supportsAudio) {
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
                        
                        // General performance tip for Gemma-3n devices
                        Text(
                            text = stringResource(R.string.gemma3n_performance_tip),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }

                // Actions (even spacing, consistent styles)
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) {
                    // Reset to defaults button
                    TextButton(
                        onClick = {
                        // Remove saved config and reset UI fields
                        scope.launch(Dispatchers.IO) {
                            try {
                                modelPrefs.removeModelConfig(model.name)
                            } catch (_: Exception) {
                            }
                        }

                        // Reset fields to defaults (do this on UI thread)
                        maxTokensValue = initialMaxTokens.coerceIn(1, maxTokensCap)
                        maxTokensText = maxTokensValue.toString()
                        topK = 64
                        topP = 0.95f
                        temperature = 1.0f
                        useGpu = defaultUseGpu
                        disableVision = false
                        disableAudio = isGemma3nModel
                    }, modifier = Modifier
                        .height(48.dp)
                        .defaultMinSize(minWidth = 88.dp)) { Text(stringResource(R.string.reset_to_defaults)) }

                    TextButton(onClick = onDismiss, modifier = Modifier.height(48.dp).defaultMinSize(minWidth = 88.dp)) { Text(stringResource(R.string.cancel)) }

                    // OK uses the same TextButton style so visuals match Reset/Cancel; min size kept
                    Button(
                        onClick = {
                            val finalMax = maxTokensValue.coerceIn(1, maxTokensCap)
                            val backend = if (canSelectAccelerator) {
                                if (useGpu) LlmInference.Backend.GPU else LlmInference.Backend.CPU
                            } else {
                                null
                            }

                            // Persist model-specific config asynchronously
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
                                } catch (e: Exception) {
                                    // ignore persistence errors
                                }
                            }

                            onConfirm(finalMax.coerceIn(1, maxTokensCap), topK, topP, temperature, backend, disableVision, disableAudio)
                            onDismiss()
                        }, modifier = Modifier.height(48.dp).defaultMinSize(minWidth = 96.dp)
                    ) { Text(stringResource(R.string.ok)) }
                }
            }
        }
    }
}

