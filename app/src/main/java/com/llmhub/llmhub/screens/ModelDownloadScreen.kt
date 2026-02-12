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
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.foundation.text.ClickableText
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import java.io.File
import kotlinx.coroutines.launch
import androidx.activity.ComponentActivity
import android.content.Intent
import android.util.Log
import androidx.lifecycle.ViewModelProvider
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.viewmodels.ModelDownloadViewModel
import androidx.compose.ui.platform.LocalContext
import android.content.Context
import android.app.ActivityManager
import com.llmhub.llmhub.ui.components.*
import androidx.compose.ui.res.stringResource
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.ModelRequirements
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import android.net.Uri

enum class ModelFormat {
    TASK, LITERTLM, GGUF, QNN_NPU, MNN_CPU
}

/**
 * Check if GPU is supported for this model
 * Simplified approach - just use the model's supportsGpu flag
 */
private fun isGpuSupportedForModel(model: LLMModel, context: Context): Boolean {
    return model.supportsGpu
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelDownloadScreen(
    onNavigateBack: () -> Unit = {},
    viewModel: ModelDownloadViewModel? = null
) {
    val context = LocalContext.current
    val activity = context as ComponentActivity
    
    // Use activity-scoped ViewModel to ensure downloads persist across navigation
    val downloadViewModel = viewModel ?: ViewModelProvider(
        activity,
        ViewModelProvider.AndroidViewModelFactory.getInstance(activity.application)
    )[ModelDownloadViewModel::class.java]
    
    val models by downloadViewModel.models.collectAsState()
    val textModels = models.filter { it.category == "text" }
    val multimodalModels = models.filter { it.category == "multimodal" }
    val embeddingModels = models.filter { it.category == "embedding" }
    val imageGenerationModels = models.filter {
        it.category == "image_generation" || it.category == "qnn_npu" || it.category == "mnn_cpu"
    }
    val textGrouped = textModels.groupBy { it.name.substringBefore("(").trim() }
    val multimodalGrouped = multimodalModels.groupBy { it.name.substringBefore("(").trim() }
    val embeddingGrouped = embeddingModels.groupBy { it.name.substringBefore("(").trim() }
    val imageGenGrouped = imageGenerationModels.groupBy { it.name.substringBefore("(").trim() }

    var showImportDialog by remember { mutableStateOf(false) }

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
        floatingActionButton = {
            FloatingActionButton(
                onClick = { showImportDialog = true },
                containerColor = MaterialTheme.colorScheme.primary,
                contentColor = MaterialTheme.colorScheme.onPrimary
            ) {
                Icon(
                    imageVector = Icons.Default.Add,
                    contentDescription = stringResource(R.string.import_external_model)
                )
            }
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
                            viewModel = downloadViewModel,
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
                            viewModel = downloadViewModel,
                            isMultimodal = true
                        )
                    }
                }
            }
            
            // Embedding Models Section
            if (embeddingGrouped.isNotEmpty()) {
                item {
                    SectionHeader(
                        title = stringResource(R.string.embedding_models),
                        subtitle = stringResource(R.string.embedding_models_description)
                    )
                }
                
                embeddingGrouped.forEach { (family, variants) ->
                    item {
                        ModelFamilyCard(
                            family = family,
                            variants = variants,
                            context = context,
                            viewModel = downloadViewModel,
                            isMultimodal = false
                        )
                    }
                }
            }
            
            // Image Generation Models Section
            if (imageGenGrouped.isNotEmpty()) {
                item {
                    SectionHeader(
                        title = stringResource(R.string.image_generation_models),
                        subtitle = stringResource(R.string.image_generation_models_description)
                    )
                }
                
                imageGenGrouped.forEach { (family, variants) ->
                    item {
                        ModelFamilyCard(
                            family = family,
                            variants = variants,
                            context = context,
                            viewModel = downloadViewModel,
                            isMultimodal = false
                        )
                    }
                }
            }
            
            item {
                Spacer(modifier = Modifier.height(80.dp))
            }
        }
        
        // Import External Model Dialog
        if (showImportDialog) {
            ImportExternalModelDialog(
                onDismiss = { showImportDialog = false },
                onImport = { externalModel, projectorUri ->
                    val success = downloadViewModel.addExternalModel(externalModel)
                    if (success) {
                        // If a projector URI was provided, copy it asynchronously in the ViewModel
                        if (projectorUri != null && externalModel.supportsVision && externalModel.modelFormat.equals("gguf", ignoreCase = true)) {
                            downloadViewModel.importVisionProjector(externalModel.name, projectorUri)
                        }
                        showImportDialog = false
                    }
                    success
                }
            )
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
                
                // First row: Capabilities (vision, audio, thinking)
                val hasThinking = variants.any { it.name.contains("Thinking", ignoreCase = true) }
                if (isMultimodal || hasThinking) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        modifier = Modifier.padding(top = 8.dp)
                    ) {
                        if (variants.any { it.supportsVision }) {
                            IconWithLabel(
                                icon = Icons.Default.RemoveRedEye,
                                label = stringResource(R.string.vision),
                                tint = MaterialTheme.colorScheme.tertiary
                            )
                        }
                        if (variants.any { it.supportsAudio }) {
                            IconWithLabel(
                                icon = Icons.Default.Mic,
                                label = stringResource(R.string.audio_support),
                                tint = MaterialTheme.colorScheme.tertiary
                            )
                        }
                        if (hasThinking) {
                            IconWithLabel(
                                icon = Icons.Default.Psychology,
                                label = stringResource(R.string.thinking_label),
                                tint = MaterialTheme.colorScheme.tertiary
                            )
                        }
                    }
                }
                
                // Second row: GPU support and variant count
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.padding(top = if (isMultimodal || hasThinking) 4.dp else 8.dp)
                ) {
                    // Show a GPU badge for the family header when any variant declares GPU support.
                    // Per-variant GPU availability (device checks) is still handled in the variant row.
                    if (variants.any { it.supportsGpu }) {
                        IconWithLabel(
                            icon = Icons.Default.Speed,
                            label = stringResource(R.string.gpu),
                            tint = MaterialTheme.colorScheme.secondary
                        )
                    }
                    
                    IconWithLabel(
                        icon = Icons.Default.Storage,
                        label = variants.size.toString() + " " + if (variants.size > 1) stringResource(R.string.variants) else stringResource(R.string.variant),
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
                    text = getModelDisplayName(model, context),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.weight(1f)
                )
                
                when {
                    model.isDownloaded && model.source == "Custom" -> StatusChip(
                        text = stringResource(R.string.ready_to_use),
                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                        contentColor = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                    model.isDownloaded -> StatusChip(
                        text = stringResource(R.string.downloaded),
                        containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                        contentColor = MaterialTheme.colorScheme.onTertiaryContainer
                    )
                    model.isExtracting -> StatusChip(
                        text = stringResource(R.string.extracting),
                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                        contentColor = MaterialTheme.colorScheme.onPrimaryContainer
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
                
                // Only show RAM requirement for non-imported models
                if (model.source != "Custom") {
                    IconWithLabel(
                        icon = Icons.Default.Memory,
                        label = context.getString(R.string.ram_requirement_format, model.requirements.minRamGB),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                
                // Show the GPU icon when the model declares GPU support. If the device
                // does not actually support GPU for this model (e.g., Gemma-3n on low-RAM
                // devices), render the icon dimmed to indicate limited availability.
                if (model.supportsGpu) {
                    val gpuAvailable = isGpuSupportedForModel(model, context)
                    IconWithLabel(
                        icon = Icons.Default.Speed,
                        label = stringResource(R.string.gpu),
                        tint = MaterialTheme.colorScheme.secondary
                    )
                }
            }
            
            // Progress indicator for downloading models
            if ((model.isDownloading || model.isExtracting) && model.downloadProgress > 0f) {
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
                    if (model.isExtracting) {
                        // Show extracting status
                        Text(
                            text = stringResource(R.string.extracting_model),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    } else if (totalDisplayBytes > 0) {
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

                    if (!model.isExtracting) {
                        Text(
                            text = formatSpeed(model.downloadSpeedBytesPerSec ?: 0),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
            } else if ((model.isPaused || model.downloadProgress > 0f || model.downloadedBytes > 0L) && !model.isDownloading && !model.isDownloaded) {
                // Paused or partial download - show progress even when paused or when only downloadedBytes is set
                ModernProgressIndicator(
                    progress = (if (model.downloadProgress > 0f) model.downloadProgress else if (model.totalBytes != null && model.totalBytes!! > 0) {
                        (model.downloadedBytes.toFloat() / model.totalBytes!!.toFloat()).coerceIn(0f, 0.99f)
                    } else 0.0001f).coerceIn(0f, 0.99f),
                    modifier = Modifier.padding(top = 8.dp)
                )

                Text(
                    text = if (model.isPaused) {
                        context.getString(R.string.paused_downloaded_format, formatFileSize(model.downloadedBytes))
                    } else if (model.downloadedBytes > 0L) {
                        context.getString(R.string.partial_downloaded_format, formatFileSize(model.downloadedBytes))
                    } else {
                        context.getString(R.string.partial)
                    },
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
                    model.isDownloaded && model.source == "Custom" -> {
                        // Imported models - show Remove button
                        OutlinedButton(
                            onClick = { onDelete(model) },
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.error
                            ),
                            border = BorderStroke(1.dp, MaterialTheme.colorScheme.error)
                        ) {
                            Icon(
                                Icons.Default.Remove,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(stringResource(R.string.remove))
                        }
                    }
                    
                    model.isDownloaded -> {
                        // Downloaded models - show Delete button
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
                            
                            // Disable pause button during extraction (can't pause/resume extraction)
                            Button(
                                onClick = { onPause(model) },
                                enabled = !model.isExtracting,
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
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            OutlinedButton(
                                onClick = { onCancel(model) },
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = MaterialTheme.colorScheme.error
                                ),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.error),
                                modifier = Modifier
                                    .weight(1f)
                                    .height(40.dp),
                                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
                            ) {
                                Icon(
                                    Icons.Default.Clear,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = stringResource(R.string.clear),
                                    style = MaterialTheme.typography.labelLarge
                                )
                            }
                            
                            Button(
                                onClick = { onResume(model) },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.primary
                                ),
                                modifier = Modifier
                                    .weight(1f)
                                    .height(40.dp),
                                contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp)
                            ) {
                                Icon(
                                    Icons.Default.PlayArrow,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = if (model.isPaused) stringResource(R.string.continue_download) else stringResource(R.string.resume_download),
                                    style = MaterialTheme.typography.labelLarge
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

private fun getModelDisplayName(model: LLMModel, context: Context): String {
    // Extract the part in parentheses from the model name
    val nameInParentheses = model.name.substringAfter("(").substringBefore(")")
    
    // Check if it's a capabilities description we need to translate
    when {
        nameInParentheses.equals("Vision+Audio+Text", ignoreCase = true) -> {
            return context.getString(R.string.vision_audio_text)
        }
        nameInParentheses.equals("Vision+Text", ignoreCase = true) -> {
            return context.getString(R.string.vision_text)
        }
        nameInParentheses.equals("Audio+Text", ignoreCase = true) -> {
            return context.getString(R.string.audio_text)
        }
        else -> {
            // For other formats like "INT4, 2k", return as-is
            return nameInParentheses
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ImportExternalModelDialog(
    onDismiss: () -> Unit,
    onImport: (LLMModel, Uri?) -> Boolean
) {
    val context = LocalContext.current
    var modelName by remember { mutableStateOf("") }
    var selectedFileUri by remember { mutableStateOf<Uri?>(null) }
    var selectedFileName by remember { mutableStateOf("") }
    var selectedVisionProjectorUri by remember { mutableStateOf<Uri?>(null) }
    var selectedVisionProjectorName by remember { mutableStateOf("") }
    var supportsVision by remember { mutableStateOf(false) }
    var supportsAudio by remember { mutableStateOf(false) }
    var supportsGpu by remember { mutableStateOf(false) }
    var modelFormat by remember { mutableStateOf(ModelFormat.TASK) }
    var contextWindowSize by remember { mutableStateOf("2048") }
    
    var showError by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }
    
    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let {
            // Try to get filename from content resolver first, fallback to lastPathSegment
            val fileName = try {
                context.contentResolver.query(it, null, null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                        if (nameIndex >= 0) cursor.getString(nameIndex) else null
                    } else null
                } ?: it.lastPathSegment ?: "Unknown file"
            } catch (e: Exception) {
                it.lastPathSegment ?: "Unknown file"
            }
            
            val fileExtension = fileName.substringAfterLast(".", "").lowercase()
            
            // Debug logging
            Log.d("ModelImport", "Selected URI: $it")
            Log.d("ModelImport", "Extracted fileName: $fileName")
            Log.d("ModelImport", "Derived fileExtension: $fileExtension")
            
            // Validate file format
            if (fileExtension != "task" && fileExtension != "litertlm" && fileExtension != "gguf" && fileExtension != "zip") {
                showError = true
                errorMessage = context.getString(R.string.unsupported_file_format)
                return@let
            }
            
            selectedFileUri = it
            // Show first few words of filename, max 20 characters
            selectedFileName = if (fileName.length > 20) {
                fileName.take(20) + "..."
            } else {
                fileName
            }
            
            // Request persistent permission for the URI
            try {
                context.contentResolver.takePersistableUriPermission(
                    it,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
                Log.d("ModelDownloadScreen", "Successfully took persistent permission for URI: $it")
            } catch (e: Exception) {
                Log.w("ModelDownloadScreen", "Could not take persistent permission for URI: ${e.message}")
            }
        }
    }

    // Separate launcher for selecting a Vision Projector file (mmproj / gguf projector)
    val visionProjectorLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let {
            val fileName = try {
                context.contentResolver.query(it, null, null, null, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                        if (nameIndex >= 0) cursor.getString(nameIndex) else null
                    } else null
                } ?: it.lastPathSegment ?: "Unknown file"
            } catch (e: Exception) {
                it.lastPathSegment ?: "Unknown file"
            }
            selectedVisionProjectorUri = it
            selectedVisionProjectorName = if (fileName.length > 20) fileName.take(20) + "..." else fileName
            try {
                context.contentResolver.takePersistableUriPermission(
                    it,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            } catch (e: Exception) {
                Log.w("ModelDownloadScreen", "Could not take persistent permission for projector URI: ${e.message}")
            }
        }
    }

        val keyboardController = LocalSoftwareKeyboardController.current
        
        AlertDialog(
            onDismissRequest = {
                keyboardController?.hide()
                onDismiss()
            },
            title = {
                Text(
                    text = stringResource(R.string.import_external_model),
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                LazyColumn(
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                    modifier = Modifier
                        .heightIn(max = 400.dp)
                        .clickable { keyboardController?.hide() }
                ) {
                item {
                    OutlinedTextField(
                        value = modelName,
                        onValueChange = { modelName = it },
                        label = { Text(stringResource(R.string.model_name)) },
                        isError = modelName.isBlank(),
                        supportingText = if (modelName.isBlank()) {
                            { Text(stringResource(R.string.model_name_required)) }
                        } else null,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
                
                    item {
                        Button(
                            onClick = { filePickerLauncher.launch(arrayOf("*/*")) },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(
                                imageVector = Icons.Default.AttachFile,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                if (selectedFileName.isNotBlank()) selectedFileName
                                else stringResource(R.string.select_model_file)
                            )
                        }
                    }

                    // If user enabled Supports Vision, show a file selector for the Vision Projector
                    if (supportsVision) {
                        item {
                            Spacer(modifier = Modifier.height(6.dp))
                            Button(
                                onClick = { visionProjectorLauncher.launch(arrayOf("*/*")) },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Icon(
                                    imageVector = Icons.Default.PhotoCamera,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                // Show selected projector name when available (same UX as model file button)
                                Text(
                                    if (selectedVisionProjectorName.isNotBlank()) selectedVisionProjectorName
                                    else stringResource(R.string.select) + " Vision Projector"
                                )
                            }


                        }
                    }
                    
                    item {
                        Text(
                            text = stringResource(R.string.download_models_from_litert_community),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    
                    item {
                        val context = LocalContext.current
                        ClickableText(
                            text = AnnotatedString(stringResource(R.string.litert_community_link)),
                            style = TextStyle(
                                color = MaterialTheme.colorScheme.primary,
                                textDecoration = TextDecoration.Underline
                            ),
                            onClick = { offset ->
                                val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://huggingface.co/litert-community"))
                                context.startActivity(intent)
                            },
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    
                    item {
                        Text(
                            text = stringResource(R.string.download_image_models_from),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    
                    item {
                        val context = LocalContext.current
                        ClickableText(
                            text = AnnotatedString(stringResource(R.string.sd_models_link)),
                            style = TextStyle(
                                color = MaterialTheme.colorScheme.primary,
                                textDecoration = TextDecoration.Underline
                            ),
                            onClick = { offset ->
                                val intent = Intent(Intent.ACTION_VIEW, Uri.parse("https://huggingface.co/xororz/sd-qnn/tree/main"))
                                context.startActivity(intent)
                            },
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                    
                    item {
                        Text(
                            text = stringResource(R.string.import_model_warning),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                
                
                // Show Vision option for GGUF and other text formats.
                // For GGUF: show only the "Supports Vision" toggle (no GPU/audio toggles).
                if (modelFormat == ModelFormat.GGUF) {
                    item {
                        Row(
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = stringResource(R.string.supports_vision),
                                modifier = Modifier.clickable { supportsVision = !supportsVision }
                            )
                            RadioButton(
                                selected = supportsVision,
                                onClick = { supportsVision = !supportsVision }
                            )
                        }
                    }
                } else if (modelFormat != ModelFormat.QNN_NPU && modelFormat != ModelFormat.MNN_CPU) {
                    // Existing behavior for TASK/LITERTLM and other text formats
                    item {
                        Row(
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = stringResource(R.string.supports_vision),
                                modifier = Modifier.clickable { supportsVision = !supportsVision }
                            )
                            RadioButton(
                                selected = supportsVision,
                                onClick = { supportsVision = !supportsVision }
                            )
                        }
                    }

                    item {
                        Row(
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = stringResource(R.string.supports_audio),
                                modifier = Modifier.clickable { supportsAudio = !supportsAudio }
                            )
                            RadioButton(
                                selected = supportsAudio,
                                onClick = { supportsAudio = !supportsAudio }
                            )
                        }
                    }

                    item {
                        Row(
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = stringResource(R.string.supports_gpu),
                                modifier = Modifier.clickable { supportsGpu = !supportsGpu }
                            )
                            RadioButton(
                                selected = supportsGpu,
                                onClick = { supportsGpu = !supportsGpu }
                            )
                        }
                    }
                }
                
                item {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(16.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        var showFormatMenu by remember { mutableStateOf(false) }
                        
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .clickable { showFormatMenu = true }
                        ) {
                            OutlinedTextField(
                                value = modelFormat.name.lowercase(),
                                onValueChange = { },
                                label = { Text(stringResource(R.string.model_format)) },
                                readOnly = true,
                                trailingIcon = {
                                    Icon(Icons.Default.ArrowDropDown, contentDescription = null)
                                },
                                modifier = Modifier.fillMaxWidth(),
                                enabled = false,
                                colors = OutlinedTextFieldDefaults.colors(
                                    disabledTextColor = MaterialTheme.colorScheme.onSurface,
                                    disabledBorderColor = MaterialTheme.colorScheme.outline,
                                    disabledLabelColor = MaterialTheme.colorScheme.onSurfaceVariant,
                                    disabledTrailingIconColor = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            )
                            
                            DropdownMenu(
                                expanded = showFormatMenu,
                                onDismissRequest = { showFormatMenu = false }
                            ) {
                                ModelFormat.values().forEach { format ->
                                    DropdownMenuItem(
                                        text = { Text(format.name.lowercase()) },
                                        onClick = {
                                            modelFormat = format
                                            showFormatMenu = false
                                        }
                                    )
                                }
                            }
                        }
                        
                        // Only show context window for text models (TASK, LITERTLM, GGUF)
                        if (modelFormat == ModelFormat.TASK || modelFormat == ModelFormat.LITERTLM || modelFormat == ModelFormat.GGUF) {
                            val contextWindowError = contextWindowSize.toIntOrNull() == null || contextWindowSize.toIntOrNull()!! <= 0
                            val contextWindowErrorText = stringResource(R.string.context_window_size_invalid)
                            
                            OutlinedTextField(
                                value = contextWindowSize,
                                onValueChange = { contextWindowSize = it },
                                label = { Text(stringResource(R.string.context_window_size)) },
                                isError = contextWindowError,
                                supportingText = if (contextWindowError) {
                                    { Text(contextWindowErrorText) }
                                } else null,
                                modifier = Modifier.weight(1f)
                            )
                        }
                    }
                }
                
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    // Validate inputs
                    val nameValid = modelName.isNotBlank()
                    val fileValid = selectedFileUri != null
                    // Context window only required for text models (TASK, LITERTLM, GGUF)
                    val contextValid = if (modelFormat == ModelFormat.TASK || modelFormat == ModelFormat.LITERTLM || modelFormat == ModelFormat.GGUF) {
                        contextWindowSize.toIntOrNull() != null && contextWindowSize.toIntOrNull()!! > 0
                    } else {
                        true // Image models don't need context window
                    }
                    
                    if (!nameValid || !fileValid || !contextValid) {
                        showError = true
                        errorMessage = "Please fix the errors above"
                        return@Button
                    }
                    
                    // Get file size from URI
                    val fileSize = selectedFileUri?.let { uri ->
                        try {
                            context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                                if (cursor.moveToFirst()) {
                                    val sizeIndex = cursor.getColumnIndex(android.provider.OpenableColumns.SIZE)
                                    if (sizeIndex >= 0) cursor.getLong(sizeIndex) else 0L
                                } else 0L
                            } ?: 0L
                        } catch (e: Exception) {
                            Log.w("ModelDownloadScreen", "Could not get file size: ${e.message}")
                            0L
                        }
                    } ?: 0L
                    
                    // For image models (QNN_NPU or MNN_CPU), we need to extract the ZIP first
                    if (modelFormat == ModelFormat.QNN_NPU || modelFormat == ModelFormat.MNN_CPU) {
                        // Mark as extracting
                        val externalModel = LLMModel(
                            name = modelName,
                            description = "Custom image generation model: $modelName",
                            url = selectedFileUri.toString(),
                            category = modelFormat.name.lowercase(),
                            sizeBytes = fileSize,
                            source = "Custom",
                            supportsVision = false,
                            supportsAudio = false,
                            supportsGpu = false,
                            requirements = ModelRequirements(
                                minRamGB = 2,
                                recommendedRamGB = 4
                            ),
                            contextWindowSize = 0,
                            modelFormat = modelFormat.name.lowercase(),
                            isDownloaded = false,
                            isDownloading = false,
                            isExtracting = true,
                            downloadProgress = 0.0f
                        )
                        val success = onImport(externalModel, null)
                        if (!success) {
                            showError = true
                            errorMessage = context.getString(R.string.model_name_already_exists, modelName)
                            return@Button
                        }
                    } else {
                        // For text models (TASK, LITERTLM, GGUF)
                        // For GGUF: always GPU=true, Vision=false, Audio=false
                        // GGUF can be vision-capable now — allow user to toggle 'Supports Vision' for GGUF as well
                        val actualSupportsVision = supportsVision
                        val actualSupportsAudio = if (modelFormat == ModelFormat.GGUF) false else supportsAudio
                        val actualSupportsGpu = if (modelFormat == ModelFormat.GGUF) true else supportsGpu
                        
                        // Determine category based on capabilities - use "multimodal" if vision/audio, else "text"
                        val modelCategory = if (actualSupportsVision || actualSupportsAudio) "multimodal" else "text"
                        val externalModel = LLMModel(
                            name = modelName,
                            description = "Custom LLM model: $modelName",
                            url = selectedFileUri.toString(),
                            category = modelCategory,
                            sizeBytes = fileSize,
                            source = "Custom",
                            supportsVision = actualSupportsVision,
                            supportsAudio = actualSupportsAudio,
                            supportsGpu = actualSupportsGpu,
                            requirements = ModelRequirements(
                                minRamGB = 4,
                                recommendedRamGB = 8
                            ),
                            contextWindowSize = contextWindowSize.toInt(),
                            modelFormat = modelFormat.name.lowercase(),
                            // projector file will be copied asynchronously by ViewModel after import
                            additionalFiles = emptyList(),
                            isDownloaded = true,
                            isDownloading = false,
                            downloadProgress = 1.0f
                        )
                        val success = onImport(externalModel, selectedVisionProjectorUri)
                        if (!success) {
                            showError = true
                            errorMessage = context.getString(R.string.model_name_already_exists, modelName)
                            return@Button
                        }
                    }
                },
                enabled = modelName.isNotBlank() && selectedFileUri != null
            ) {
                Text(stringResource(R.string.import_model))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.cancel))
            }
        }
    )
    
    // Error dialog
    if (showError) {
        AlertDialog(
            onDismissRequest = { showError = false },
            title = { Text("Error") },
            text = { Text(errorMessage) },
            confirmButton = {
                TextButton(onClick = { showError = false }) {
                    Text("OK")
                }
            }
        )
    }
}