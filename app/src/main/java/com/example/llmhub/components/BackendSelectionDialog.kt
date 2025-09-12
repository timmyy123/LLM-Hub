package com.llmhub.llmhub.components

import android.app.ActivityManager
import android.content.Context
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.R

data class BackendSelection(
    val backend: LlmInference.Backend,
    val disableVision: Boolean = false
)

@Composable
fun BackendSelectionDialog(
    modelName: String,
    onBackendSelected: (BackendSelection) -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    
    // Check specific Gemma-3n model requirements and get device memory
    val deviceMemoryGB = getDeviceMemoryGB(context)
    val isGemma3nE2B = modelName.contains("Gemma-3n E2B", ignoreCase = true)
    val isGemma3nE4B = modelName.contains("Gemma-3n E4B", ignoreCase = true)
    
    // GPU requirements for Gemma-3n models:
    // E2B: requires > 5GB RAM (disable if <= 5GB)
    // E4B: requires > 7GB RAM (disable if <= 7GB, but allow GPU without vision for 8GB)
    val gpuDisabled = when {
        isGemma3nE2B -> deviceMemoryGB <= 5.0
        isGemma3nE4B -> deviceMemoryGB <= 7.0
        else -> false
    }
    
    // Special case: E4B can use GPU without vision for devices with 8GB RAM
    val canUseGpuWithoutVision = isGemma3nE4B && deviceMemoryGB > 7.0 && deviceMemoryGB <= 8.0
    // Get screen configuration for responsive sizing
    val configuration = LocalConfiguration.current
    val screenHeightDp = configuration.screenHeightDp.dp
    val dialogMaxHeight = if (screenHeightDp * 0.85f > 600.dp) 600.dp else screenHeightDp * 0.85f

    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
                .heightIn(max = dialogMaxHeight),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Title
                Text(
                    text = stringResource(R.string.select_backend),
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                
                // Description
                Text(
                    text = stringResource(R.string.choose_backend_for_gemma),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // GPU Option (Full Vision)
                if (!gpuDisabled) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.primaryContainer
                        ),
                        onClick = {
                            onBackendSelected(BackendSelection(LlmInference.Backend.GPU, disableVision = false))
                            onDismiss()
                        }
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp)
                        ) {
                            Text(
                                text = stringResource(R.string.gpu_backend),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                            Text(
                                text = stringResource(R.string.gpu_backend_description),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                            )
                        }
                    }
                }
                
                // GPU Option (Vision Disabled) - Special case for 8GB RAM with E4B
                if (canUseGpuWithoutVision) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.tertiaryContainer
                        ),
                        onClick = {
                            // TODO: Show warning dialog and then proceed with GPU + Vision disabled
                            onBackendSelected(BackendSelection(LlmInference.Backend.GPU, disableVision = true))
                            onDismiss()
                        }
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp)
                        ) {
                            Text(
                                text = stringResource(R.string.gpu_with_vision_disabled),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = MaterialTheme.colorScheme.onTertiaryContainer
                            )
                            Text(
                                text = stringResource(R.string.ram_limit_gpu_vision_disabled, deviceMemoryGB),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onTertiaryContainer.copy(alpha = 0.8f)
                            )
                        }
                    }
                }
                
                // GPU Disabled Option
                if (gpuDisabled && !canUseGpuWithoutVision) {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        ),
                        onClick = { /* No action when disabled */ }
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp)
                        ) {
                            Text(
                                text = "${stringResource(R.string.gpu_backend)} (${stringResource(R.string.not_available)})",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                            )
                            Text(
                                text = when {
                                    isGemma3nE2B -> stringResource(R.string.gemma3n_e2b_gpu_requires_6gb)
                                    isGemma3nE4B -> stringResource(R.string.gemma3n_e4b_gpu_requires_8gb)
                                    else -> stringResource(R.string.gpu_disabled_low_memory, 8)
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                            )
                        }
                    }
                }
                
                // CPU Option
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    ),
                    onClick = {
                        onBackendSelected(BackendSelection(LlmInference.Backend.CPU, disableVision = false))
                        onDismiss()
                    }
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp)
                    ) {
                        Text(
                            text = stringResource(R.string.cpu_backend),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = stringResource(R.string.cpu_backend_description),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }
                }
                
                // Hint
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.tertiaryContainer
                    )
                ) {
                    Text(
                        text = stringResource(R.string.backend_selection_hint),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onTertiaryContainer,
                        modifier = Modifier.padding(12.dp)
                    )
                }
                
                // Cancel button
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    TextButton(onClick = onDismiss) {
                        Text(stringResource(R.string.cancel))
                    }
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
