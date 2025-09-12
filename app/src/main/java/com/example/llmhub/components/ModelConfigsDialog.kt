package com.llmhub.llmhub.components

import android.app.ActivityManager
import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.clickable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.*
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

@Composable
fun ModelConfigsDialog(
    model: LLMModel,
    initialMaxTokens: Int,
    onConfirm: (maxTokens: Int, topK: Int, topP: Float, temperature: Float, backend: LlmInference.Backend?) -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val configuration = LocalConfiguration.current
    val screenWidthDp = configuration.screenWidthDp.dp
    val screenHeightDp = configuration.screenHeightDp.dp
    // Make dialog wider and less tall than before
    val dialogWidth = (screenWidthDp * 0.92f).coerceAtMost(720.dp)
    val dialogMaxHeight = (screenHeightDp * 0.72f).coerceAtMost(640.dp)

    // Determine GPU availability for Gemma models using same logic as BackendSelectionDialog
    val deviceMemoryGB = getDeviceMemoryGB(context)
    val isGemma3nE2B = model.name.contains("Gemma-3n E2B", ignoreCase = true)
    val isGemma3nE4B = model.name.contains("Gemma-3n E4B", ignoreCase = true)
    val gpuDisabled = when {
        isGemma3nE2B -> deviceMemoryGB <= 5.0f
        isGemma3nE4B -> deviceMemoryGB <= 8.0f
        else -> false
    }

    // State for fields
    var maxTokensText by remember { mutableStateOf(initialMaxTokens.toString()) }
    var topK by remember { mutableStateOf(64) }
    var topP by remember { mutableStateOf(0.95f) }
    var temperature by remember { mutableStateOf(1.0f) }
    var useGpu by remember { mutableStateOf(false) }

    // Cap for max tokens
    val maxTokensCap = remember(model) { MediaPipeInferenceService.getMaxTokensForModelStatic(model) }

    val focusManager = androidx.compose.ui.platform.LocalFocusManager.current

    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .width(dialogWidth)
                .padding(12.dp)
                .heightIn(max = dialogMaxHeight),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
                    .clickable(indication = null, interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() }) {
                        // Click outside inputs to clear focus (dismiss keyboard)
                        focusManager.clearFocus()
                    },
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = stringResource(R.string.model_configs_title),
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = androidx.compose.ui.text.font.FontWeight.Bold
                )

                // Max tokens
                Text(text = stringResource(R.string.max_tokens), style = MaterialTheme.typography.bodyMedium)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    OutlinedTextField(
                        value = maxTokensText,
                        onValueChange = { input ->
                            // Only allow numbers and clamp
                            val numeric = input.filter { it.isDigit() }
                            val intVal = numeric.toIntOrNull() ?: 0
                            val clamped = intVal.coerceIn(1, maxTokensCap)
                            maxTokensText = clamped.toString()
                        },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.width(160.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(text = "${maxTokensCap} ${stringResource(R.string.max)}", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(start = 4.dp))
                }

                // TopK
                Text(text = stringResource(R.string.top_k), style = MaterialTheme.typography.bodyMedium)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Slider(value = topK.toFloat(), onValueChange = { topK = it.toInt() }, valueRange = 1f..256f, modifier = Modifier.weight(1f))
                    Spacer(modifier = Modifier.width(8.dp))
                    OutlinedTextField(value = topK.toString(), onValueChange = { v -> topK = v.filter { it.isDigit() }.toIntOrNull() ?: topK }, modifier = Modifier.width(100.dp))
                }

                // TopP
                Text(text = stringResource(R.string.top_p), style = MaterialTheme.typography.bodyMedium)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Slider(value = topP, onValueChange = { topP = it }, valueRange = 0.0f..1.0f, modifier = Modifier.weight(1f))
                    Spacer(modifier = Modifier.width(8.dp))
                    OutlinedTextField(value = String.format("%.2f", topP), onValueChange = { v -> topP = v.toFloatOrNull() ?: topP }, modifier = Modifier.width(100.dp))
                }

                // Temperature
                Text(text = stringResource(R.string.temperature), style = MaterialTheme.typography.bodyMedium)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Slider(value = temperature, onValueChange = { temperature = it }, valueRange = 0.0f..2.0f, modifier = Modifier.weight(1f))
                    Spacer(modifier = Modifier.width(8.dp))
                    OutlinedTextField(value = String.format("%.2f", temperature), onValueChange = { v -> temperature = v.toFloatOrNull() ?: temperature }, modifier = Modifier.width(100.dp))
                }

                // Accelerator toggle only for Gemma models
                if (model.name.contains("Gemma", ignoreCase = true)) {
                    Text(text = stringResource(R.string.choose_accelerator), style = MaterialTheme.typography.bodyMedium)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        // Make accelerator buttons same size
                        val chipModifier = Modifier.width(160.dp).height(48.dp)
                        FilterChip(
                            selected = !useGpu,
                            onClick = { useGpu = false },
                            label = { Text(stringResource(R.string.cpu_backend)) },
                            leadingIcon = { if (!useGpu) Icon(Icons.Filled.Check, contentDescription = null) },
                            modifier = chipModifier
                        )
                        Spacer(modifier = Modifier.width(12.dp))
                        FilterChip(
                            selected = useGpu,
                            onClick = { if (!gpuDisabled) useGpu = true },
                            label = { Text(stringResource(R.string.gpu_backend)) },
                            leadingIcon = { if (useGpu) Icon(Icons.Filled.Check, contentDescription = null) },
                            modifier = chipModifier
                        )
                    }
                    if (gpuDisabled) {
                        Text(text = stringResource(R.string.gpu_disabled_low_memory, 8), style = MaterialTheme.typography.bodySmall)
                    }
                }

                // Actions
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                    TextButton(onClick = onDismiss) { Text(stringResource(R.string.cancel)) }
                    Spacer(modifier = Modifier.width(8.dp))
                    Button(onClick = {
                        val finalMax = maxTokensText.toIntOrNull() ?: initialMaxTokens
                        val backend = if (model.name.contains("Gemma", ignoreCase = true)) {
                            if (useGpu && !gpuDisabled) LlmInference.Backend.GPU else LlmInference.Backend.CPU
                        } else {
                            null
                        }
                        onConfirm(finalMax.coerceIn(1, maxTokensCap), topK, topP, temperature, backend)
                        onDismiss()
                    }) { Text(stringResource(R.string.ok)) }
                }
            }
        }
    }
}

private fun getDeviceMemoryGB(context: Context): Float {
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val memInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memInfo)
    return memInfo.totalMem / (1024f * 1024f * 1024f)
}
