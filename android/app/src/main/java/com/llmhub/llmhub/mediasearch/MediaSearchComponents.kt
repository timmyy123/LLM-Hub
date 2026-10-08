package com.llmhub.llmhub.mediasearch

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.llmhub.llmhub.R
import com.llmhub.llmhub.components.ModelSelectorCard
import com.llmhub.llmhub.data.LLMModel

/** Same layout as the other features' "download a model first" state. */
@Composable
internal fun MediaSearchDownloadGate(icon: ImageVector, onNavigateToModelDownload: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.6f), modifier = Modifier.size(64.dp))
        Spacer(Modifier.height(24.dp))
        Text(
            stringResource(R.string.media_search_download_model),
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(12.dp))
        Text(
            stringResource(R.string.media_search_download_model_desc),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(32.dp))
        FilledTonalButton(onClick = onNavigateToModelDownload, modifier = Modifier.fillMaxWidth(0.6f)) {
            Icon(Icons.Default.GetApp, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text(stringResource(R.string.download_models_title))
        }
    }
}

/** First-run card: what the feature does plus the ways to add content. */
@Composable
internal fun MediaSearchOnboarding(
    icon: ImageVector,
    title: String,
    description: String,
    primaryLabel: String,
    onPrimary: () -> Unit,
    secondaryLabel: String? = null,
    onSecondary: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(56.dp))
                Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
                Text(
                    description,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center
                )
                Spacer(Modifier.height(8.dp))
                Button(onClick = onPrimary, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                    Text(primaryLabel)
                }
                if (secondaryLabel != null && onSecondary != null) {
                    OutlinedButton(onClick = onSecondary, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                        Text(secondaryLabel)
                    }
                }
            }
        }
    }
}

@Composable
internal fun MediaSearchField(query: String, onQueryChange: (String) -> Unit, hint: String, isSearching: Boolean) {
    OutlinedTextField(
        value = query,
        onValueChange = onQueryChange,
        modifier = Modifier.fillMaxWidth(),
        placeholder = { Text(hint, color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)) },
        leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
        trailingIcon = {
            when {
                isSearching -> CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                query.isNotEmpty() -> IconButton(onClick = { onQueryChange("") }) {
                    Icon(Icons.Default.Close, contentDescription = stringResource(R.string.close))
                }
            }
        },
        singleLine = true,
        shape = RoundedCornerShape(12.dp)
    )
}

/** Model loading / failure / analysis progress, matching Gallery's analysis progress card. */
@Composable
internal fun MediaSearchStatusCard(
    isLoadingModel: Boolean,
    modelError: Boolean,
    progress: IndexingProgress,
    isPaused: Boolean,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onRetryModel: () -> Unit
) {
    if (!isLoadingModel && !modelError && progress.isComplete) return
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            when {
                isLoadingModel -> {
                    Text(stringResource(R.string.media_search_loading_model), style = MaterialTheme.typography.bodyMedium)
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                }
                modelError -> Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        stringResource(R.string.media_search_model_failed),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.weight(1f)
                    )
                    TextButton(onClick = onRetryModel) { Text(stringResource(R.string.retry)) }
                }
                else -> {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                stringResource(if (isPaused) R.string.media_search_paused else R.string.media_search_analyzing),
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.SemiBold
                            )
                            Text(
                                stringResource(R.string.media_search_progress, progress.processed, progress.total, progress.percent),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        TextButton(onClick = if (isPaused) onResume else onPause) {
                            Icon(if (isPaused) Icons.Default.PlayArrow else Icons.Default.Pause, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(4.dp))
                            Text(stringResource(if (isPaused) R.string.media_search_resume else R.string.media_search_pause))
                        }
                    }
                    LinearProgressIndicator(
                        progress = { progress.percent / 100f },
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        }
    }
}

/** Bottom sheet opened from the top-right settings icon: model picker + library management. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun MediaSearchSettingsSheet(
    models: List<LLMModel>,
    selectedModel: LLMModel?,
    isLoadingModel: Boolean,
    isModelLoaded: Boolean,
    onModelSelected: (LLMModel) -> Unit,
    onLoadModel: () -> Unit,
    onUnloadModel: () -> Unit,
    libraryCountText: String,
    libraryActions: @Composable ColumnScope.() -> Unit,
    onClearAll: () -> Unit,
    onDismiss: () -> Unit
) {
    ModalBottomSheet(onDismissRequest = onDismiss, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(16.dp).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(stringResource(R.string.feature_settings_title), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            ModelSelectorCard(
                models = models,
                selectedModel = selectedModel,
                selectedBackend = com.google.mediapipe.tasks.genai.llminference.LlmInference.Backend.CPU,
                selectedNpuDeviceId = null,
                isLoading = isLoadingModel,
                isModelLoaded = isModelLoaded,
                onModelSelected = onModelSelected,
                onBackendSelected = null,
                onLoadModel = onLoadModel,
                onUnloadModel = onUnloadModel,
                filterMultimodalOnly = false
            )
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            ) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(stringResource(R.string.media_search_library), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(libraryCountText, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    libraryActions()
                    OutlinedButton(
                        onClick = onClearAll,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = MaterialTheme.colorScheme.error),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.error)
                    ) {
                        Icon(Icons.Default.DeleteSweep, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.media_search_clear))
                    }
                }
            }
            Spacer(Modifier.height(24.dp))
        }
    }
}
