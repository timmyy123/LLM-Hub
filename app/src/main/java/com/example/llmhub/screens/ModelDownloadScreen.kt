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
import androidx.compose.foundation.clickable
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import android.content.Context
import android.app.ActivityManager

/**
 * Get device total memory in GB
 */
private fun getDeviceMemoryGB(context: Context): Double {
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val memoryInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memoryInfo)
    return memoryInfo.totalMem / (1024.0 * 1024.0 * 1024.0) // Convert bytes to GB
}

/**
 * Check if GPU is actually supported for this model on this device
 */
private fun isGpuSupportedForModel(model: LLMModel, context: Context): Boolean {
    if (!model.supportsGpu) return false
    
    // For Gemma-3n models, require >8GB RAM
    if (model.supportsVision && model.name.contains("Gemma-3n", ignoreCase = true)) {
        return getDeviceMemoryGB(context) > 8.0
    }
    
    // For other models that support GPU, return true
    return true
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelDownloadScreen(
    onNavigateBack: () -> Unit = {},
    viewModel: ModelDownloadViewModel = viewModel()
) {
    val context = LocalContext.current
    val models by viewModel.models.collectAsState()
    val textModels = models.filter { it.category == "text" }
    val multimodalModels = models.filter { it.category == "multimodal" }
    val textGrouped = textModels.groupBy { it.name.substringBefore("(").trim() }
    val multimodalGrouped = multimodalModels.groupBy { it.name.substringBefore("(").trim() }

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
            item {
                Text(
                    text = "Available Models",
                    style = MaterialTheme.typography.headlineSmall,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }
            
            // Text Models Section
            if (textGrouped.isNotEmpty()) {
                item {
                    Text(
                        text = "Text Models",
                        style = MaterialTheme.typography.titleLarge,
                        modifier = Modifier.padding(vertical = 8.dp),
                        color = MaterialTheme.colorScheme.primary
                    )
                }
                
                textGrouped.forEach { (family, variants) ->
                    // Each family gets one expandable card
                    item {
                        var expanded by remember { mutableStateOf(false) }

                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 4.dp)
                                .clickable { expanded = !expanded },
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp)
                            ) {
                                Text(
                                    family,
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.weight(1f)
                                )
                                
                                // Show GPU support if any variant supports GPU on this device
                                if (variants.any { isGpuSupportedForModel(it, context) }) {
                                    Icon(
                                        Icons.Default.Speed,
                                        contentDescription = "GPU acceleration support",
                                        modifier = Modifier.size(20.dp),
                                        tint = MaterialTheme.colorScheme.secondary
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                }
                                
                                Icon(
                                    imageVector = if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                                    contentDescription = if (expanded) "Collapse" else "Expand"
                                )
                            }

                            if (expanded) {
                                Column(modifier = Modifier.padding(bottom = 8.dp)) {
                                    variants.forEach { model ->
                                        ModelItem(
                                            model = model,
                                            context = context,
                                            onDownload = { viewModel.downloadModel(it) },
                                            onDelete = { viewModel.deleteModel(it) },
                                            onCancel = { viewModel.cancelDownload(it) }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Multimodal Models Section
            if (multimodalGrouped.isNotEmpty()) {
                item {
                    Text(
                        text = "Multimodal Models (Vision + Text)",
                        style = MaterialTheme.typography.titleLarge,
                        modifier = Modifier.padding(vertical = 8.dp),
                        color = MaterialTheme.colorScheme.primary
                    )
                }
                
                multimodalGrouped.forEach { (family, variants) ->
                    // Each family gets one expandable card
                    item {
                        var expanded by remember { mutableStateOf(false) }

                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 4.dp)
                                .clickable { expanded = !expanded },
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp)
                            ) {
                                Text(
                                    family,
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.weight(1f)
                                )
                                
                                // Show GPU support if any variant supports GPU on this device
                                if (variants.any { isGpuSupportedForModel(it, context) }) {
                                    Icon(
                                        Icons.Default.Speed,
                                        contentDescription = "GPU acceleration support",
                                        modifier = Modifier.size(20.dp),
                                        tint = MaterialTheme.colorScheme.secondary
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                }
                                
                                // Vision indicator
                                Icon(
                                    Icons.Default.RemoveRedEye,
                                    contentDescription = "Vision support",
                                    modifier = Modifier.size(20.dp),
                                    tint = MaterialTheme.colorScheme.primary
                                )
                                
                                Spacer(modifier = Modifier.width(8.dp))
                                
                                Icon(
                                    imageVector = if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                                    contentDescription = if (expanded) "Collapse" else "Expand"
                                )
                            }

                            if (expanded) {
                                Column(modifier = Modifier.padding(bottom = 8.dp)) {
                                    variants.forEach { model ->
                                        ModelItem(
                                            model = model,
                                            context = context,
                                            onDownload = { viewModel.downloadModel(it) },
                                            onDelete = { viewModel.deleteModel(it) },
                                            onCancel = { viewModel.cancelDownload(it) }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ModelItem(
    model: LLMModel,
    context: Context,
    onDownload: (LLMModel) -> Unit,
    onDelete: (LLMModel) -> Unit,
    onCancel: (LLMModel) -> Unit = {}
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = model.name,
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.weight(1f)
                )
                
                // GPU support indicator
                if (isGpuSupportedForModel(model, context)) {
                    Icon(
                        Icons.Default.Speed,
                        contentDescription = "GPU acceleration support",
                        modifier = Modifier.size(20.dp),
                        tint = MaterialTheme.colorScheme.secondary
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                }
                
                // Vision indicator
                if (model.supportsVision) {
                    Icon(
                        Icons.Default.RemoveRedEye,
                        contentDescription = "Vision support",
                        modifier = Modifier.size(20.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }
            
            Text(text = "Source: ${model.source}", style = MaterialTheme.typography.bodySmall)
            
            // GPU support indicator
            if (isGpuSupportedForModel(model, context)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(top = 4.dp)
                ) {
                    Icon(
                        Icons.Default.Speed,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint = MaterialTheme.colorScheme.secondary
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = if (model.supportsVision) "GPU acceleration supported" else "GPU acceleration supported",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.secondary
                    )
                }
            } else if (model.supportsGpu && model.supportsVision) {
                // Show why GPU is not supported for vision models
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(top = 4.dp)
                ) {
                    Icon(
                        Icons.Default.Speed,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "GPU acceleration (requires >8GB RAM)",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            
            if (model.supportsVision) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(top = 4.dp)
                ) {
                    Icon(
                        Icons.Default.Image,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Supports image input",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = model.description, style = MaterialTheme.typography.bodyMedium)
            Spacer(modifier = Modifier.height(16.dp))

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

                    Spacer(modifier = Modifier.height(8.dp))

                    // Cancel button
                    Spacer(modifier = Modifier.height(4.dp))
                    Button(onClick = { onCancel(model) }) {
                        Text("Cancel")
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