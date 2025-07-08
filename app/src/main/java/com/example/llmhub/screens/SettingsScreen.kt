package com.example.llmhub.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit
) {
    val uriHandler = LocalUriHandler.current
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
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
                SettingsSection(title = "Models") {
                    SettingsItem(
                        icon = Icons.Default.GetApp,
                        title = "Download Models",
                        subtitle = "Browse and download LLM models",
                        onClick = onNavigateToModels
                    )
                    
                }
            }
            
            item {
                SettingsSection(title = "Chat") {
                    SettingsItem(
                        icon = Icons.Outlined.History,
                        title = "Clear Chat History",
                        subtitle = "Delete all conversations",
                        onClick = {
                            // TODO: Implement clear history
                        }
                    )
                    
                    SettingsItem(
                        icon = Icons.Default.CloudUpload,
                        title = "Export Chats",
                        subtitle = "Export your conversations",
                        onClick = {
                            // TODO: Implement export
                        }
                    )
                }
            }
            
            item {
                SettingsSection(title = "App") {
                    SettingsItem(
                        icon = Icons.Outlined.Palette,
                        title = "Theme",
                        subtitle = "Appearance settings",
                        onClick = {
                            // TODO: Implement theme settings
                        }
                    )
                    
                    SettingsItem(
                        icon = Icons.Default.Info,
                        title = "About",
                        subtitle = "App information and licenses",
                        onClick = {
                            // TODO: Implement about screen
                        }
                    )
                }
            }
            
            item {
                SettingsSection(title = "Resources") {
                    SettingsItem(
                        icon = Icons.Default.Language,
                        title = "Hugging Face Hub",
                        subtitle = "Browse models on Hugging Face",
                        onClick = {
                            uriHandler.openUri("https://huggingface.co/models")
                        }
                    )
                    
                    SettingsItem(
                        icon = Icons.Outlined.Code,
                        title = "GitHub Repository",
                        subtitle = "View source code and contribute",
                        onClick = {
                            uriHandler.openUri("https://github.com/")
                        }
                    )
                }
            }
        }
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