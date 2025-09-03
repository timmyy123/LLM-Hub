package com.llmhub.llmhub.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
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
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.viewmodels.ModelDownloadViewModel
import androidx.compose.ui.platform.LocalContext
import android.content.Context
import android.app.ActivityManager
import com.llmhub.llmhub.ui.components.*
import androidx.compose.ui.res.stringResource
import com.llmhub.llmhub.R

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
                title = { 
                    Text(
                        stringResource(R.string.ai_models),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold
                    ) 
                },
                navigationIcon = {
                    IconButton(
                        onClick = onNavigateBack,
                        modifier = Modifier.size(48.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = stringResource(R.string.back),
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            )
        },
        containerColor = MaterialTheme.colorScheme.surface
    ) { paddingValues ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues),
            contentPadding = PaddingValues(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    ),
                    shape = MaterialTheme.shapes.large
                ) {
                    Column(
                        modifier = Modifier.padding(20.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Icon(
                                Icons.Default.CloudDownload,
                                contentDescription = null,
                                modifier = Modifier.size(32.dp),
                                tint = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                            Column {
                                Text(
                                    text = stringResource(R.string.download_ai_models),
                                    style = MaterialTheme.typography.titleLarge,
                                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = stringResource(R.string.choose_text_multimodal_models),
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                                )
                            }
                        }
                    }
                }
            }
            
            // Text Models Section
            if (textGrouped.isNotEmpty()) {
                item {
                    SectionHeader(
                        title = stringResource(R.string.text_models),
                        subtitle = stringResource(R.string.text_models_description)
                    )
                }
                
                textGrouped.forEach { (family, variants) ->
                    item {
                        ModelFamilyCard(
                            family = family,
                            variants = variants,
                            context = context,
                            viewModel = viewModel,
                            isMultimodal = false
                        )
                    }
                }
            }
            
            // Multimodal Models Section
            if (multimodalGrouped.isNotEmpty()) {
                item {
                    SectionHeader(
                        title = stringResource(R.string.vision_models),
                        subtitle = stringResource(R.string.vision_models_description)
                    )
                }
                
                multimodalGrouped.forEach { (family, variants) ->
                    item {
                        ModelFamilyCard(
                            family = family,
                            variants = variants,
                            context = context,
                            viewModel = viewModel,
                            isMultimodal = true
                        )
                    }
                }
            }
            
            item {
                Spacer(modifier = Modifier.height(80.dp))
            }
        }
    }
}

@Composable
private fun ModelFamilyCard(
    family: String,
    variants: List<LLMModel>,
    context: Context,
    viewModel: ModelDownloadViewModel,
    isMultimodal: Boolean
) {
    var expanded by remember { mutableStateOf(false) }
    
    ModernCard(
        onClick = { expanded = !expanded }
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = family,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface
                )
                
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.padding(top = 8.dp)
                ) {
                    if (isMultimodal) {
                        IconWithLabel(
                            icon = Icons.Default.RemoveRedEye,
                            label = stringResource(R.string.vision),
                            tint = MaterialTheme.colorScheme.tertiary
                        )
                    }
                    
                    if (variants.any { isGpuSupportedForModel(it, context) }) {
                        IconWithLabel(
                            icon = Icons.Default.Speed,
                            label = stringResource(R.string.gpu),
                            tint = MaterialTheme.colorScheme.secondary
                        )
                    }
                    
                    IconWithLabel(
                        icon = Icons.Default.Storage,
                        label = "${variants.size} ${if (variants.size > 1) stringResource(R.string.variants) else stringResource(R.string.variant)}",
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }
            
            Icon(
                imageVector = if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                contentDescription = if (expanded) stringResource(R.string.collapse) else stringResource(R.string.expand),
                modifier = Modifier.size(24.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
        
        AnimatedVisibility(
            visible = expanded,
            enter = expandVertically(animationSpec = tween(300)) + fadeIn(),
            exit = shrinkVertically(animationSpec = tween(300)) + fadeOut()
        ) {
            Column(
                modifier = Modifier
                    .padding(top = 16.dp)
                    .animateContentSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                variants.forEach { model ->
                    ModelVariantItem(
                        model = model,
                        context = context,
                        onDownload = { viewModel.downloadModel(it) },
                        onCancel = { viewModel.cancelDownload(it) },
                        onPause = { viewModel.pauseDownload(it) },
                        onResume = { viewModel.resumeDownload(it) },
                        onDelete = { viewModel.deleteModel(it) }
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelVariantItem(
    model: LLMModel,
    context: Context,
    onDownload: (LLMModel) -> Unit,
    onCancel: (LLMModel) -> Unit,
    onPause: (LLMModel) -> Unit,
    onResume: (LLMModel) -> Unit,
    onDelete: (LLMModel) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        ),
        shape = MaterialTheme.shapes.medium
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Model name and status
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = model.name.substringAfter("(").substringBefore(")"),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.weight(1f)
                )
                
                when {
                    model.isDownloaded -> StatusChip(
                        text = stringResource(R.string.downloaded),
                        containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                        contentColor = MaterialTheme.colorScheme.onTertiaryContainer
                    )
                    model.isDownloading -> StatusChip(
                        text = stringResource(R.string.downloading),
                        containerColor = MaterialTheme.colorScheme.secondaryContainer,
                        contentColor = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                    model.isPaused -> StatusChip(
                        text = stringResource(R.string.paused),
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                        contentColor = MaterialTheme.colorScheme.onErrorContainer
                    )
                    model.downloadProgress > 0f && !model.isDownloaded -> StatusChip(
                        text = stringResource(R.string.partial),
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                        contentColor = MaterialTheme.colorScheme.onErrorContainer
                    )
                    else -> StatusChip(
                        text = stringResource(R.string.not_downloaded),
                        containerColor = MaterialTheme.colorScheme.surfaceVariant,
                        contentColor = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            
            // Model info
            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                IconWithLabel(
                    icon = Icons.Default.Storage,
                    label = formatFileSize(model.sizeBytes),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
                
                IconWithLabel(
                    icon = Icons.Default.Memory,
                    label = stringResource(R.string.ram_requirement_format, model.requirements.minRamGB),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
                
                if (isGpuSupportedForModel(model, context)) {
                                            IconWithLabel(
                            icon = Icons.Default.Speed,
                            label = stringResource(R.string.gpu),
                            tint = MaterialTheme.colorScheme.secondary
                        )
                }
            }
            
            // Progress indicator for downloading models
            if (model.isDownloading && model.downloadProgress > 0f) {
                ModernProgressIndicator(
                    progress = model.downloadProgress,
                    modifier = Modifier.padding(top = 8.dp)
                )
                
                // Download speed and size info
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    val totalDisplayBytes = model.totalBytes ?: model.sizeBytes
                    if (totalDisplayBytes > 0) {
                        Text(
                            text = "${formatFileSize(model.downloadedBytes)} / ${formatFileSize(totalDisplayBytes)}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    } else {
                        Text(
                            text = formatFileSize(model.downloadedBytes),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Text(
                        text = formatSpeed(model.downloadSpeedBytesPerSec ?: 0),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            } else if (model.downloadProgress > 0f && !model.isDownloading && !model.isDownloaded) {
                // Paused download - only show for incomplete models
                ModernProgressIndicator(
                    progress = model.downloadProgress.coerceIn(0f, 0.99f),
                    modifier = Modifier.padding(top = 8.dp)
                )
                
                Text(
                    text = stringResource(R.string.paused_downloaded_format, formatFileSize(model.downloadedBytes)),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            // Action button
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                when {
                    model.isDownloaded -> {
                        OutlinedButton(
                            onClick = { onDelete(model) },
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.error
                            ),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.error)
                        ) {
                            Icon(
                                Icons.Default.Delete,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(stringResource(R.string.delete))
                        }
                    }
                    
                    model.isDownloading -> {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(
                                onClick = { onCancel(model) },
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = MaterialTheme.colorScheme.error
                                ),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.error)
                            ) {
                                Icon(
                                    Icons.Default.Cancel,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.cancel))
                            }
                            
                            Button(
                                onClick = { onPause(model) },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.secondary
                                )
                            ) {
                                Icon(
                                    Icons.Default.Pause,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.pause_download))
                            }
                        }
                    }
                    
                    model.isPaused || (model.downloadProgress > 0f && !model.isDownloading && !model.isDownloaded) -> {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(
                                onClick = { onCancel(model) },
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = MaterialTheme.colorScheme.error
                                ),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.error)
                            ) {
                                Icon(
                                    Icons.Default.Clear,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.clear))
                            }
                            
                            Button(
                                onClick = { onResume(model) },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.primary
                                )
                            ) {
                                Icon(
                                    Icons.Default.PlayArrow,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(
                                    if (model.isPaused) {
                                        stringResource(R.string.continue_download)
                                    } else {
                                        stringResource(R.string.resume_download)
                                    }
                                )
                            }
                        }
                    }
                    
                    else -> {
                        Button(
                            onClick = { onDownload(model) },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.primary
                            )
                        ) {
                            Icon(
                                Icons.Default.CloudDownload,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(stringResource(R.string.download))
                        }
                    }
                }
            }
        }
    }
}

private fun formatFileSize(bytes: Long): String {
    return when {
        bytes >= 1024 * 1024 * 1024 -> String.format("%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0))
        bytes >= 1024 * 1024 -> String.format("%.0f MB", bytes / (1024.0 * 1024.0))
        else -> String.format("%.0f KB", bytes / 1024.0)
    }
}

private fun formatSpeed(bytesPerSec: Long): String {
    if (bytesPerSec <= 0) return "0 KB/s"
    val mb = bytesPerSec / (1024.0 * 1024.0)
    return if (mb >= 1) {
        String.format("%.1f MB/s", mb)
    } else {
        val kb = bytesPerSec / 1024.0
        String.format("%.0f KB/s", kb)
    }
}