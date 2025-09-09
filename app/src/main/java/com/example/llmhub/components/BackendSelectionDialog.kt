package com.llmhub.llmhub.components

import android.app.ActivityManager
import android.content.Context
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.R

@Composable
fun BackendSelectionDialog(
    modelName: String,
    onBackendSelected: (LlmInference.Backend) -> Unit,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    
    // Check specific Gemma-3n model requirements and get device memory
    val deviceMemoryGB = getDeviceMemoryGB(context)
    val isGemma3nE2B = modelName.contains("Gemma-3n E2B", ignoreCase = true)
    val isGemma3nE4B = modelName.contains("Gemma-3n E4B", ignoreCase = true)
    
    // GPU requirements for Gemma-3n models:
    // E2B: requires > 5GB RAM (disable if <= 5GB)
    // E4B: requires > 8GB RAM (disable if <= 8GB)
    val gpuDisabled = when {
        isGemma3nE2B -> deviceMemoryGB <= 5.0
        isGemma3nE4B -> deviceMemoryGB <= 8.0
        else -> false
    }
    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(24.dp),
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
                
                // GPU Option
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = if (gpuDisabled) 
                            MaterialTheme.colorScheme.surfaceVariant 
                        else 
                            MaterialTheme.colorScheme.primaryContainer
                    ),
                    onClick = if (gpuDisabled) {
                        { /* No action when disabled */ }
                    } else {
                        {
                            onBackendSelected(LlmInference.Backend.GPU)
                            onDismiss()
                        }
                    }
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp)
                    ) {
                        Text(
                            text = if (gpuDisabled) 
                                "${stringResource(R.string.gpu_backend)} (${stringResource(R.string.not_available)})"
                            else 
                                stringResource(R.string.gpu_backend),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = if (gpuDisabled)
                                MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                            else
                                MaterialTheme.colorScheme.onPrimaryContainer
                        )
                        Text(
                            text = when {
                                gpuDisabled && isGemma3nE2B -> stringResource(R.string.gemma3n_e2b_gpu_requires_6gb)
                                gpuDisabled && isGemma3nE4B -> stringResource(R.string.gemma3n_e4b_gpu_requires_8gb)
                                gpuDisabled -> stringResource(R.string.gpu_disabled_low_memory, 8)
                                else -> stringResource(R.string.gpu_backend_description)
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = if (gpuDisabled)
                                MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                            else
                                MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                        )
                    }
                }
                
                // CPU Option
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    ),
                    onClick = {
                        onBackendSelected(LlmInference.Backend.CPU)
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
