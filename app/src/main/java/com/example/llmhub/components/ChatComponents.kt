package com.example.llmhub.components

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.example.llmhub.data.MessageEntity
import com.example.llmhub.viewmodels.ChatViewModel
import dev.jeziellago.compose.markdowntext.MarkdownText

/**
 * Enhanced chat bubble that shows user/assistant messages with optional token statistics.
 * Aligns right for user and left for assistant.
 */
@Composable
fun MessageBubble(
    message: MessageEntity,
    streamingContent: String = ""
) {
    val isUser = message.isFromUser
    val bubbleColor = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
    val textColor = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Column {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(12.dp))
                    .background(bubbleColor)
                    .padding(12.dp)
                    .run { 
                        if (isUser) {
                            // User messages: fit content with max width
                            this.wrapContentWidth().widthIn(max = 280.dp)
                        } else {
                            // Assistant messages: use wider width
                            this.widthIn(max = 280.dp)
                        }
                    }
            ) {
                // Show streaming content for assistant messages during generation, otherwise show message content
                val displayContent = if (!isUser && streamingContent.isNotEmpty()) streamingContent else message.content
                
                MarkdownText(
                    markdown = displayContent,
                    color = textColor,
                    fontSize = MaterialTheme.typography.bodyMedium.fontSize,
                    modifier = if (isUser) Modifier.wrapContentWidth() else Modifier.fillMaxWidth()
                )
            }
            
            // Show token statistics for assistant messages after completion
            val hasStats = message.tokenCount != null && message.tokensPerSecond != null
            // Show stats if we have them and the message has real content (not just the placeholder "…")
            val showStats = !isUser && hasStats && message.content != "…"
            if (showStats) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${message.tokenCount} tokens • ${String.format("%.1f", message.tokensPerSecond!!)} tok/sec",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(start = 12.dp)
                )
            }
        }
    }
}

/**
 * Input bar with text field, optional attachment button, and send button.
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
    
    // Image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            attachmentUri = it
            showImagePreview = true
        }
    }

    Column {
        // Image attachment preview
        if (attachmentUri != null) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Image preview
                    AsyncImage(
                        model = ImageRequest.Builder(context)
                            .data(attachmentUri)
                            .crossfade(true)
                            .build(),
                        contentDescription = "Selected image",
                        modifier = Modifier
                            .size(48.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { showImagePreview = true },
                        contentScale = ContentScale.Crop
                    )
                    
                    Spacer(modifier = Modifier.width(12.dp))
                    
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Image attached",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = "Tap to preview",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                        )
                    }
                    
                    IconButton(onClick = { attachmentUri = null }) {
                        Icon(
                            Icons.Default.Close,
                            contentDescription = "Remove attachment",
                            tint = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    }
                }
            }
        }

        // Message input row
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 4.dp)
        ) {
            // Image attachment button (for vision models)
            if (supportsAttachments) {
                IconButton(
                    onClick = {
                        imagePickerLauncher.launch("image/*")
                    },
                    enabled = enabled
                ) {
                    Icon(
                        Icons.Default.Image,
                        contentDescription = "Add image",
                        tint = if (enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                    )
                }
            }

            TextField(
                value = text,
                onValueChange = { text = it },
                modifier = Modifier.weight(1f),
                placeholder = { 
                    Text(
                        if (supportsAttachments) "Type a message or add an image..." 
                        else "Type a message"
                    )
                },
                enabled = enabled,
                singleLine = false,
                maxLines = 4
            )
            
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
                Icon(Icons.Default.Send, contentDescription = "Send")
            }
        }
    }
    
    // Full-screen image preview dialog
    if (showImagePreview && attachmentUri != null) {
        Dialog(onDismissRequest = { showImagePreview = false }) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(0.8f),
                shape = RoundedCornerShape(16.dp),
                color = MaterialTheme.colorScheme.surface
            ) {
                Column {
                    // Header
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Image Preview",
                            style = MaterialTheme.typography.headlineSmall
                        )
                        IconButton(onClick = { showImagePreview = false }) {
                            Icon(Icons.Default.Close, contentDescription = "Close")
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
                            .clip(RoundedCornerShape(8.dp)),
                        contentScale = ContentScale.Fit
                    )
                    
                    // Actions
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.SpaceEvenly
                    ) {
                        OutlinedButton(
                            onClick = { 
                                attachmentUri = null
                                showImagePreview = false
                            }
                        ) {
                            Text("Remove")
                        }
                        
                        Button(
                            onClick = { showImagePreview = false }
                        ) {
                            Text("Keep")
                        }
                    }
                }
            }
        }
    }
} 