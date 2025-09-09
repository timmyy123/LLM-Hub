package com.llmhub.llmhub.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.R
import java.io.File
import com.llmhub.llmhub.data.ThemeMode
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.viewmodels.ThemeViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    onNavigateToAbout: () -> Unit,
    onNavigateToTerms: () -> Unit,
    themeViewModel: ThemeViewModel = viewModel()
) {
    val uriHandler = LocalUriHandler.current
    var showThemeDialog by remember { mutableStateOf(false) }
    var showLanguageDialog by remember { mutableStateOf(false) }
    val currentThemeMode by themeViewModel.themeMode.collectAsState()
    val webSearchEnabled by themeViewModel.webSearchEnabled.collectAsState()
    val currentLanguage by themeViewModel.appLanguage.collectAsState()
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.settings)) },
                navigationIcon = {
                    IconButton(
                        onClick = onNavigateBack,
                        modifier = Modifier.size(48.dp)
                    ) {
                        Icon(
                            Icons.Default.ArrowBack, 
                            contentDescription = stringResource(R.string.content_description_back),
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            item {
                SettingsSection(title = stringResource(R.string.models)) {
                    SettingsItem(
                        icon = Icons.Default.GetApp,
                        title = stringResource(R.string.download_models),
                        subtitle = stringResource(R.string.browse_download_models),
                        onClick = onNavigateToModels
                    )
                    
                    // Embedding Model Selection
                    EmbeddingModelSelector(themeViewModel = themeViewModel)
                }
            }
            
            item {
                SettingsSection(title = stringResource(R.string.features)) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.Search,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(24.dp)
                        )
                        
                        Spacer(modifier = Modifier.width(16.dp))
                        
                        Column(
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(
                                text = stringResource(R.string.web_search),
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                text = if (webSearchEnabled) stringResource(R.string.web_search_description_enabled) else stringResource(R.string.web_search_description_disabled),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        
                        Switch(
                            checked = webSearchEnabled,
                            onCheckedChange = { enabled ->
                                themeViewModel.setWebSearchEnabled(enabled)
                            }
                        )
                    }
                }
            }
            
            item {
                SettingsSection(title = stringResource(R.string.appearance)) {
                    SettingsItem(
                        icon = Icons.Outlined.Palette,
                        title = stringResource(R.string.theme),
                        subtitle = when (currentThemeMode) {
                            ThemeMode.LIGHT -> stringResource(R.string.theme_light)
                            ThemeMode.DARK -> stringResource(R.string.theme_dark)
                            ThemeMode.SYSTEM -> stringResource(R.string.theme_system)
                        },
                        onClick = {
                            showThemeDialog = true
                        }
                    )
                    
                    SettingsItem(
                        icon = Icons.Default.Language,
                        title = stringResource(R.string.language),
                        subtitle = themeViewModel.getCurrentLanguageDisplayName(),
                        onClick = {
                            showLanguageDialog = true
                        }
                    )
                }
            }
            
            item {
                SettingsSection(title = stringResource(R.string.information)) {
                    SettingsItem(
                        icon = Icons.Default.Info,
                        title = stringResource(R.string.about),
                        subtitle = stringResource(R.string.app_info_contact),
                        onClick = onNavigateToAbout
                    )
                    
                    SettingsItem(
                        icon = Icons.Default.Description,
                        title = stringResource(R.string.terms_of_service),
                        subtitle = stringResource(R.string.legal_terms_conditions),
                        onClick = onNavigateToTerms
                    )
                }
            }
            
            item {
                SettingsSection(title = stringResource(R.string.source_code_section)) {
                    SettingsItem(
                        icon = Icons.Outlined.Code,
                        title = stringResource(R.string.github_repository),
                        subtitle = stringResource(R.string.view_source_contribute),
                        onClick = {
                            uriHandler.openUri("https://github.com/timmyy123/LLM-Hub")
                        }
                    )
                }
            }
        }
    }
    
    // Theme Selection Dialog
    if (showThemeDialog) {
        AlertDialog(
            onDismissRequest = { showThemeDialog = false },
            title = { Text(stringResource(R.string.choose_theme)) },
            text = {
                Column {
                    val themeOptions = listOf(
                        ThemeMode.LIGHT to stringResource(R.string.theme_light),
                        ThemeMode.DARK to stringResource(R.string.theme_dark),
                        ThemeMode.SYSTEM to stringResource(R.string.theme_system)
                    )
                    themeOptions.forEach { (mode, label) ->
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 4.dp)
                        ) {
                            RadioButton(
                                selected = currentThemeMode == mode,
                                onClick = {
                                    themeViewModel.setThemeMode(mode)
                                    showThemeDialog = false
                                }
                            )
                            
                            Spacer(modifier = Modifier.width(8.dp))
                            
                            Text(
                                text = label,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = { showThemeDialog = false }
                ) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }
    
    // Language Selection Dialog
    if (showLanguageDialog) {
        AlertDialog(
            onDismissRequest = { showLanguageDialog = false },
            title = { Text(stringResource(R.string.select_language)) },
            text = {
                Column {
                    // System Default option
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp)
                    ) {
                        RadioButton(
                            selected = currentLanguage == null,
                            onClick = {
                                themeViewModel.setAppLanguage(null)
                                showLanguageDialog = false
                            }
                        )
                        
                        Spacer(modifier = Modifier.width(8.dp))
                        
                        Text(
                            text = stringResource(R.string.system_default),
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                    
                    // Language options
                    themeViewModel.getSupportedLanguages().forEach { (code, displayName) ->
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 4.dp)
                        ) {
                            RadioButton(
                                selected = currentLanguage == code,
                                onClick = {
                                    themeViewModel.setAppLanguage(code)
                                    showLanguageDialog = false
                                }
                            )
                            
                            Spacer(modifier = Modifier.width(8.dp))
                            
                            Text(
                                text = displayName,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = { showLanguageDialog = false }
                ) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }
}

@Composable
private fun SettingsSection(
    title: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(bottom = 8.dp)
            )
            content()
        }
    }
}

@Composable
private fun SettingsItem(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    title: String,
    subtitle: String,
    onClick: () -> Unit
) {
    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier = Modifier.size(24.dp),
                tint = MaterialTheme.colorScheme.onSurface
            )
            
            Spacer(modifier = Modifier.width(16.dp))
            
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            Icon(
                imageVector = Icons.Default.KeyboardArrowRight,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun EmbeddingModelSelector(themeViewModel: ThemeViewModel) {
    val selectedEmbeddingModel by themeViewModel.selectedEmbeddingModel.collectAsState()
    var showEmbeddingModelDialog by remember { mutableStateOf(false) }
    val context = LocalContext.current
    
    // Get downloaded embedding models only
    val downloadedEmbeddingModels = remember(context) {
        com.llmhub.llmhub.data.ModelData.models
            .filter { it.category == "embedding" }
            .filter { model ->
                val modelsDir = File(context.filesDir, "models")
                val modelFile = File(modelsDir, model.localFileName())
                modelFile.exists() && modelFile.length() > 0
            }
            .map { it.name }
    }
    
    SettingsItem(
        icon = Icons.Default.Memory,
        title = stringResource(R.string.embedding_model),
        subtitle = selectedEmbeddingModel ?: stringResource(R.string.no_embedding_model_selected),
        onClick = { showEmbeddingModelDialog = true }
    )
    
    if (showEmbeddingModelDialog) {
        AlertDialog(
            onDismissRequest = { showEmbeddingModelDialog = false },
            title = { Text(stringResource(R.string.select_embedding_model)) },
            text = {
                LazyColumn {
                    items(downloadedEmbeddingModels.size) { index ->
                        val model = downloadedEmbeddingModels[index]
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    themeViewModel.setSelectedEmbeddingModel(model)
                                    showEmbeddingModelDialog = false
                                }
                                .padding(vertical = 12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = selectedEmbeddingModel == model,
                                onClick = {
                                    themeViewModel.setSelectedEmbeddingModel(model)
                                    showEmbeddingModelDialog = false
                                }
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                            Text(
                                text = model,
                                style = MaterialTheme.typography.bodyLarge
                            )
                        }
                    }
                    
                    // Option to disable embedding
                    item {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    themeViewModel.setSelectedEmbeddingModel(null)
                                    showEmbeddingModelDialog = false
                                }
                                .padding(vertical = 12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = selectedEmbeddingModel == null,
                                onClick = {
                                    themeViewModel.setSelectedEmbeddingModel(null)
                                    showEmbeddingModelDialog = false
                                }
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                            Text(
                                text = stringResource(R.string.disable_embeddings),
                                style = MaterialTheme.typography.bodyLarge
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showEmbeddingModelDialog = false }) {
                    Text(stringResource(R.string.cancel))
                }
            }
        )
    }
} 