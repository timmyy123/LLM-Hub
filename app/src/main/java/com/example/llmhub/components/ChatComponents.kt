package com.example.llmhub.components

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Send
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
    val uriHandler = LocalUriHandler.current // For future attachment picker stub
    var text by remember { mutableStateOf("") }
    var attachmentUri by remember { mutableStateOf<Uri?>(null) }

    Column {
        if (attachmentUri != null) {
            // Simple chip to show attached file path
            AssistChip(
                onClick = { /* Could preview attachment */ },
                label = { Text("Attachment added") },
                trailingIcon = {
                    IconButton(onClick = { attachmentUri = null }) {
                        Icon(Icons.Default.AttachFile, contentDescription = "Remove attachment")
                    }
                }
            )
            Spacer(Modifier.height(4.dp))
        }

        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 4.dp)
        ) {
            if (supportsAttachments) {
                IconButton(onClick = {
                    // Placeholder: open external picker link (would use ActivityResult in real app)
                    // For now we'll just simulate attaching a dummy Uri
                    attachmentUri = Uri.parse("dummy://attachment")
                }, enabled = enabled) {
                    Icon(Icons.Default.AttachFile, contentDescription = "Attach")
                }
            }

            TextField(
                value = text,
                onValueChange = { text = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Type a message") },
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
} 