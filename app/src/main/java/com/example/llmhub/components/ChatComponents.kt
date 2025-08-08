package com.llmhub.llmhub.components

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.foundation.clickable
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.llmhub.llmhub.data.MessageEntity
import dev.jeziellago.compose.markdowntext.MarkdownText

/**
 * Enhanced chat bubble that shows user/assistant messages with modern Material Design 3 styling.
 * Features rounded corners, proper elevation, and adaptive colors.
 */
@Composable
fun MessageBubble(
    message: MessageEntity,
    streamingContent: String = ""
) {
    val context = LocalContext.current
    val isUser = message.isFromUser
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        if (!isUser) {
            // Assistant avatar
            Surface(
                modifier = Modifier.size(32.dp),
                shape = RoundedCornerShape(16.dp),
                color = MaterialTheme.colorScheme.primaryContainer
            ) {
                Icon(
                    Icons.Default.SmartToy,
                    contentDescription = "AI Assistant",
                    modifier = Modifier.padding(6.dp),
                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }
            Spacer(modifier = Modifier.width(8.dp))
        }
        
        Column(
            modifier = Modifier.weight(1f, fill = false)
        ) {
            Surface(
                modifier = Modifier.widthIn(max = 280.dp),
                shape = RoundedCornerShape(
                    topStart = if (isUser) 20.dp else 4.dp,
                    topEnd = if (isUser) 4.dp else 20.dp,
                    bottomStart = 20.dp,
                    bottomEnd = 20.dp
                ),
                color = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceContainerHigh,
                shadowElevation = 1.dp
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    // Display image if attachment exists
                    if (message.attachmentPath != null && message.attachmentType == "image") {
                        AsyncImage(
                            model = ImageRequest.Builder(context)
                                .data(Uri.parse(message.attachmentPath))
                                .crossfade(true)
                                .build(),
                            contentDescription = "Attached image",
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(max = 200.dp)
                                .clip(RoundedCornerShape(12.dp)),
                            contentScale = ContentScale.Crop,
                            onError = { 
                                android.util.Log.w("MessageBubble", "Failed to load image: ${message.attachmentPath}")
                            }
                        )
                        
                        if (message.content.isNotEmpty() && message.content != "Shared a file") {
                            Spacer(modifier = Modifier.height(8.dp))
                        }
                    }
                    
                    // Display text content
                    if (message.content.isNotEmpty() && message.content != "Shared a file") {
                        val displayContent = if (!isUser && streamingContent.isNotEmpty()) streamingContent else message.content
                        
                        MarkdownText(
                            markdown = displayContent,
                            color = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface,
                            fontSize = MaterialTheme.typography.bodyLarge.fontSize,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }
            
            // Show token statistics for assistant messages
            val hasStats = message.tokenCount != null && message.tokensPerSecond != null
            val showStats = !isUser && hasStats && message.content != "…"
            if (showStats) {
                Row(
                    modifier = Modifier.padding(start = 8.dp, top = 4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        Icons.Outlined.Speed,
                        contentDescription = null,
                        modifier = Modifier.size(12.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "${message.tokenCount} tokens • ${String.format("%.1f", message.tokensPerSecond!!)} tok/sec",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
        
        if (isUser) {
            Spacer(modifier = Modifier.width(8.dp))
            // User avatar
            Surface(
                modifier = Modifier.size(32.dp),
                shape = RoundedCornerShape(16.dp),
                color = MaterialTheme.colorScheme.tertiaryContainer
            ) {
                Icon(
                    Icons.Default.Person,
                    contentDescription = "You",
                    modifier = Modifier.padding(6.dp),
                    tint = MaterialTheme.colorScheme.onTertiaryContainer
                )
            }
        }
    }
}

/**
 * Modern input bar with Material Design 3 styling, attachment support, and smooth animations.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MessageInput(
    onSendMessage: (String, Uri?) -> Unit,
    enabled: Boolean,
    supportsAttachments: Boolean
) {
    val context = LocalContext.current
    var text by remember { mutableStateOf("") }
    var attachmentUri by remember { mutableStateOf<Uri?>(null) }
    var showImagePreview by remember { mutableStateOf(false) }
    
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            attachmentUri = it
            showImagePreview = true
        }
    }

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surface,
        shadowElevation = 8.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Image attachment preview
            if (attachmentUri != null) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    ),
                    shape = MaterialTheme.shapes.medium
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        AsyncImage(
                            model = ImageRequest.Builder(context)
                                .data(attachmentUri)
                                .crossfade(true)
                                .build(),
                            contentDescription = "Selected image",
                            modifier = Modifier
                                .size(48.dp)
                                .clip(MaterialTheme.shapes.small)
                                .clickable { showImagePreview = true },
                            contentScale = ContentScale.Crop
                        )
                        
                        Spacer(modifier = Modifier.width(12.dp))
                        
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Image attached",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                            Text(
                                text = "Tap to preview",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                            )
                        }
                        
                        IconButton(onClick = { attachmentUri = null }) {
                            Icon(
                                Icons.Default.Close,
                                contentDescription = "Remove attachment",
                                tint = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        }
                    }
                }
            }

            // Input field with modern styling
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                modifier = Modifier.fillMaxWidth(),
                placeholder = { 
                    Text(
                        if (supportsAttachments) "Ask me anything or add an image..." 
                        else "Ask me anything...",
                        style = MaterialTheme.typography.bodyLarge
                    )
                },
                enabled = enabled,
                singleLine = false,
                maxLines = 4,
                shape = RoundedCornerShape(24.dp),
                leadingIcon = if (supportsAttachments) {
                    {
                        IconButton(
                            onClick = { imagePickerLauncher.launch("image/*") },
                            enabled = enabled
                        ) {
                            Icon(
                                Icons.Outlined.Image,
                                contentDescription = "Add image",
                                tint = if (enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                            )
                        }
                    }
                } else null,
                trailingIcon = {
                    IconButton(
                        onClick = {
                            if (text.isNotBlank() || attachmentUri != null) {
                                onSendMessage(text, attachmentUri)
                                text = ""
                                attachmentUri = null
                            }
                        },
                        enabled = enabled && (text.isNotBlank() || attachmentUri != null)
                    ) {
                        Surface(
                            modifier = Modifier.size(32.dp),
                            shape = RoundedCornerShape(16.dp),
                            color = if (enabled && (text.isNotBlank() || attachmentUri != null)) 
                                MaterialTheme.colorScheme.primary 
                            else MaterialTheme.colorScheme.surfaceVariant
                        ) {
                            Icon(
                                Icons.Default.Send,
                                contentDescription = "Send",
                                modifier = Modifier.padding(6.dp),
                                tint = if (enabled && (text.isNotBlank() || attachmentUri != null)) 
                                    MaterialTheme.colorScheme.onPrimary 
                                else MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                },
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = MaterialTheme.colorScheme.primary,
                    unfocusedBorderColor = MaterialTheme.colorScheme.outline
                ),
                textStyle = MaterialTheme.typography.bodyLarge
            )
        }
    }
    
    // Full-screen image preview dialog
    if (showImagePreview && attachmentUri != null) {
        Dialog(onDismissRequest = { showImagePreview = false }) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(0.9f),
                shape = MaterialTheme.shapes.extraLarge,
                color = MaterialTheme.colorScheme.surface
            ) {
                Column {
                    // Header
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceContainer
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Image Preview",
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold
                            )
                            IconButton(onClick = { showImagePreview = false }) {
                                Icon(Icons.Default.Close, contentDescription = "Close")
                            }
                        }
                    }
                    
                    // Image
                    AsyncImage(
                        model = ImageRequest.Builder(context)
                            .data(attachmentUri)
                            .crossfade(true)
                            .build(),
                        contentDescription = "Image preview",
                        modifier = Modifier
                            .fillMaxWidth()
                            .weight(1f)
                            .padding(16.dp)
                            .clip(MaterialTheme.shapes.large),
                        contentScale = ContentScale.Fit
                    )
                    
                    // Actions
                    Surface(
                        color = MaterialTheme.colorScheme.surfaceContainer
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End)
                        ) {
                            OutlinedButton(
                                onClick = { 
                                    attachmentUri = null
                                    showImagePreview = false
                                }
                            ) {
                                Icon(
                                    Icons.Default.Delete,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Remove")
                            }
                            
                            Button(
                                onClick = { showImagePreview = false }
                            ) {
                                Icon(
                                    Icons.Default.Check,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text("Keep")
                            }
                        }
                    }
                }
            }
        }
    }
} 