package com.example.llmhub.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.llmhub.data.LLMModel
import com.example.llmhub.data.ModelData
import com.example.llmhub.viewmodels.ModelDownloadViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelDownloadScreen(
    onNavigateBack: () -> Unit = {},
    viewModel: ModelDownloadViewModel = viewModel()
) {
    val models by viewModel.models.collectAsState()
    val textModels = models.filter { it.category == "text" }
    val visionModels = models.filter { it.category == "vision" }
    var showInstructions by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Download Models") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { showInstructions = !showInstructions }) {
                        Icon(
                            imageVector = Icons.Default.Info,
                            contentDescription = "Instructions"
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
        ) {
            // Instructions Card
            item {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Warning,
                                contentDescription = "Warning",
                                tint = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "MediaPipe Model Conversion Required",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "MediaPipe requires .task format models. The download button will download safetensors files that need manual conversion.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                        
                        if (showInstructions) {
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                text = ModelData.getStatusMessage(),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        TextButton(
                            onClick = { showInstructions = !showInstructions }
                        ) {
                            Text(
                                text = if (showInstructions) "Hide Instructions" else "Show Conversion Instructions",
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                        }
                    }
                }
            }

            item {
                Text(
                    text = "Text Models",
                    style = MaterialTheme.typography.headlineSmall,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }
            items(textModels) { model ->
                ModelItem(
                    model = model,
                    onDownload = { viewModel.downloadModel(it) },
                    onDelete = { viewModel.deleteModel(it) }
                )
            }

            item {
                Divider(modifier = Modifier.padding(vertical = 16.dp))
            }

            item {
                Text(
                    text = "Vision Models", 
                    style = MaterialTheme.typography.headlineSmall,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }
            items(visionModels) { model ->
                ModelItem(
                    model = model,
                    onDownload = { viewModel.downloadModel(it) },
                    onDelete = { viewModel.deleteModel(it) }
                )
            }
        }
    }
}

@Composable
fun ModelItem(
    model: LLMModel,
    onDownload: (LLMModel) -> Unit,
    onDelete: (LLMModel) -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(text = model.name, style = MaterialTheme.typography.titleLarge)
            Text(text = "Source: ${model.source}", style = MaterialTheme.typography.bodySmall)
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = model.description, style = MaterialTheme.typography.bodyMedium)
            Spacer(modifier = Modifier.height(16.dp))

            // Vision Support and RAM Requirements
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = if (model.supportsVision) Icons.Default.Visibility else Icons.Default.TextFields,
                        contentDescription = "Vision Support",
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(text = if (model.supportsVision) "Vision" else "Text")
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Memory,
                        contentDescription = "RAM",
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(text = "${model.requirements.minRamGB}GB - ${model.requirements.recommendedRamGB}GB RAM")
                }
            }
            Spacer(modifier = Modifier.height(16.dp))

            // Download Button and Progress
            if (model.isDownloaded) {
                Button(onClick = { onDelete(model) }) {
                    Text("Delete")
                }
            } else if (model.downloadProgress != 1f && (model.downloadedBytes > 0 || model.downloadProgress < 0)) {
                // Downloading (either determinate or indeterminate)
                Column {
                    if (model.downloadProgress < 0f) {
                        // Indeterminate progress when total size is unknown
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    } else {
                    LinearProgressIndicator(
                        progress = { model.downloadProgress },
                        modifier = Modifier.fillMaxWidth()
                    )
                    }

                    Spacer(modifier = Modifier.height(4.dp))

                    // Show downloaded bytes and speed when we have at least some information
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        val totalDisplayBytes = model.totalBytes ?: model.sizeBytes
                        if (totalDisplayBytes > 0) {
                        Text(
                            text = String.format(
                                "%.1fMB / %.1fMB",
                                model.downloadedBytes / 1_000_000f,
                                    totalDisplayBytes / 1_000_000f
                            ),
                            style = MaterialTheme.typography.bodySmall
                        )
                        } else {
                            Text(
                                text = String.format("%.1fMB", model.downloadedBytes / 1_000_000f),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }

                        Text(
                            text = formatSpeed(model.downloadSpeedBytesPerSec ?: 0),
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            } else {
                // Not downloaded
                Button(onClick = { onDownload(model) }) {
                    if (model.sizeBytes > 0) {
                    Text("Download (${formatBytes(model.sizeBytes)})")
                    } else {
                        Text("Download")
                    }
                }
            }
        }
    }
}

private fun formatBytes(bytes: Long): String {
    val gb = bytes / (1024.0 * 1024.0 * 1024.0)
    if (gb >= 1) return String.format("%.2f GB", gb)

    val mb = bytes / (1024.0 * 1024.0)
    if (mb >= 1) return String.format("%.2f MB", mb)

    val kb = bytes / 1024.0
    if (kb >= 1) return String.format("%.0f KB", kb)

    return String.format("%d Bytes", bytes)
}

private fun formatSpeed(bytesPerSec: Long): String {
    if (bytesPerSec <= 0) return "0 KB/s"
    val mb = bytesPerSec / (1024.0 * 1024.0)
    return if (mb >= 1) {
        String.format("%.2f MB/s", mb)
    } else {
        val kb = bytesPerSec / 1024.0
        String.format("%.0f KB/s", kb)
    }
} 