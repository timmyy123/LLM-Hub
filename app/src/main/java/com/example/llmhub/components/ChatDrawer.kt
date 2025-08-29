package com.llmhub.llmhub.components

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.data.ChatEntity
import com.llmhub.llmhub.viewmodels.ChatDrawerViewModel
import java.text.SimpleDateFormat
import java.util.*
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import com.llmhub.llmhub.R

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatDrawer(
    onNavigateToChat: (String) -> Unit,
    onCreateNewChat: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateToModels: () -> Unit,
    onClearAllChats: (() -> Unit)? = null,
    viewModel: ChatDrawerViewModel = viewModel()
) {
    // Now gets chats from the view model
    val chats by viewModel.allChats.collectAsState()
    var showDeleteAllDialog by remember { mutableStateOf(false) }

    if (showDeleteAllDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteAllDialog = false },
            title = { Text(stringResource(R.string.dialog_delete_all_chats_title)) },
            text = { Text(stringResource(R.string.dialog_delete_all_chats_message)) },
            confirmButton = {
                TextButton(
                    onClick = {
                        if (onClearAllChats != null) {
                            onClearAllChats()
                        } else {
                            viewModel.deleteAllChats()
                        }
                        showDeleteAllDialog = false
                    }
                ) {
                    Text(stringResource(R.string.action_delete_all))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteAllDialog = false }) {
                    Text(stringResource(R.string.action_cancel))
                }
            }
        )
    }
    
    ModalDrawerSheet(
        modifier = Modifier.width(300.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            // Header
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(bottom = 16.dp)
            ) {
                Icon(
                    Icons.Default.PhoneAndroid,
                    contentDescription = null,
                    modifier = Modifier.size(32.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = stringResource(R.string.drawer_title),
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            
            // New Chat Button
            FilledTonalButton(
                onClick = onCreateNewChat,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Default.Add, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text(stringResource(R.string.drawer_new_chat))
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Chat History
            Text(
                text = stringResource(R.string.drawer_recent_chats),
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(bottom = 8.dp)
            )
            
            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                items(chats) { chat ->
                    ChatHistoryItem(
                        chat = chat,
                        onClick = { onNavigateToChat(chat.id) },
                        onDelete = { viewModel.deleteChat(chat.id) }
                    )
                }
                
                if (chats.isEmpty()) {
                    item {
                        Text(
                            text = stringResource(R.string.drawer_no_chats),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(16.dp)
                        )
                    }
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            Divider()
            Spacer(modifier = Modifier.height(16.dp))
            
            // Navigation Options
            DrawerNavigationItem(
                icon = Icons.Default.GetApp,
                text = stringResource(R.string.drawer_download_models),
                onClick = onNavigateToModels
            )
            
            DrawerNavigationItem(
                icon = Icons.Outlined.DeleteSweep,
                text = stringResource(R.string.drawer_clear_all_chats),
                onClick = { showDeleteAllDialog = true }
            )

            DrawerNavigationItem(
                icon = Icons.Default.Settings,
                text = stringResource(R.string.drawer_settings),
                onClick = onNavigateToSettings
            )
        }
    }
}

@Composable
private fun ChatHistoryItem(
    chat: ChatEntity,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    var showDeleteDialog by remember { mutableStateOf(false) }

    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text(stringResource(R.string.dialog_delete_chat_title)) },
            text = { Text(stringResource(R.string.dialog_delete_chat_message, chat.title)) },
            confirmButton = {
                TextButton(
                    onClick = {
                        onDelete()
                        showDeleteDialog = false
                    }
                ) {
                    Text(stringResource(R.string.action_delete))
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text(stringResource(R.string.action_cancel))
                }
            }
        )
    }

    Surface(
        onClick = onClick,
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surface,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp, horizontal = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = chat.title,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = formatTimestamp(chat.updatedAt),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            IconButton(onClick = { showDeleteDialog = true }) {
                Icon(
                    Icons.Outlined.Delete,
                    contentDescription = stringResource(R.string.content_description_delete_chat),
                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun DrawerNavigationItem(
    icon: ImageVector,
    text: String,
    onClick: () -> Unit
) {
    Surface(
        onClick = onClick,
        shape = MaterialTheme.shapes.medium,
        color = Color.Transparent,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(icon, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
            Spacer(modifier = Modifier.width(16.dp))
            Text(text = text, style = MaterialTheme.typography.bodyLarge)
        }
    }
}

@Composable
private fun formatTimestamp(timestamp: Long): String {
    val now = System.currentTimeMillis()
    val diff = now - timestamp

    val seconds = diff / 1000
    val minutes = seconds / 60
    val hours = minutes / 60
    val days = hours / 24

    return when {
        days > 1 -> SimpleDateFormat("MMM dd, yyyy", Locale.getDefault()).format(Date(timestamp))
        days == 1L -> stringResource(R.string.time_yesterday)
        hours > 0 -> stringResource(R.string.time_hours_ago, hours.toInt())
        minutes > 0 -> stringResource(R.string.time_minutes_ago, minutes.toInt())
        else -> stringResource(R.string.time_just_now)
    }
} 