package com.llmhub.llmhub.components

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.ContentResolver
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import com.llmhub.llmhub.R
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.compose.foundation.background
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.text.ClickableText
import android.content.Intent
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.platform.LocalClipboardManager
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import coil.ImageLoader
import coil.request.SuccessResult
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.llmhub.llmhub.data.MessageEntity
import dev.jeziellago.compose.markdowntext.MarkdownText
import androidx.compose.ui.platform.LocalUriHandler
import com.llmhub.llmhub.utils.FileUtils
import com.example.llmhub.utils.CodeBlockParser
import com.example.llmhub.utils.CodeBlockParser.ParsedSegment
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.TextUnit
import com.llmhub.llmhub.ui.components.AudioInputService
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Delete
import kotlinx.coroutines.flow.collectLatest
import androidx.compose.runtime.rememberCoroutineScope
import android.Manifest
import android.content.pm.PackageManager
import kotlinx.coroutines.launch
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material.icons.filled.Pause
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.TextRange

/**
 * Custom selectable markdown text component that supports both markdown rendering and text selection.
 * This addresses the issue where MarkdownText doesn't properly support text selection.
 */
@Composable
fun SelectableMarkdownText(
    markdown: String,
    color: Color,
    fontSize: androidx.compose.ui.unit.TextUnit,
    modifier: Modifier = Modifier,
    textAlign: TextAlign = TextAlign.Start,
    linkColorOverride: Color? = null
) {
    val context = LocalContext.current
    val linkColor = linkColorOverride ?: MaterialTheme.colorScheme.primary
    
    // First parse the markdown (bold/italic/lists etc.)
    val baseAnnotated = remember(markdown, color, linkColor) {
        parseMarkdownToAnnotatedString(markdown, color, linkColor)
    }

    // Then detect links & phone numbers and add annotations/styles
    val finalAnnotated = remember(baseAnnotated, linkColor) {
        annotateLinksAndPhones(baseAnnotated, linkColor)
    }

    SelectionContainer {
        ClickableText(
            text = finalAnnotated,
            modifier = modifier,
            style = LocalTextStyle.current.copy(
                fontSize = fontSize, 
                lineHeight = fontSize * 1.4,
                textAlign = textAlign
            ),
            onClick = { offset ->
                // Handle URL clicks
                finalAnnotated.getStringAnnotations("URL", offset, offset)
                    .firstOrNull()?.let { ann ->
                        val intent = Intent(Intent.ACTION_VIEW, Uri.parse(ann.item))
                        context.startActivity(intent)
                        return@ClickableText
                    }

                // Handle phone number clicks
                finalAnnotated.getStringAnnotations("PHONE", offset, offset)
                    .firstOrNull()?.let { ann ->
                        val intent = Intent(Intent.ACTION_DIAL, Uri.parse("tel:${ann.item}"))
                        context.startActivity(intent)
                    }
            }
        )
    }
}

/**
 * Post-process an AnnotatedString to detect web URLs and phone numbers, add underline/primary-color
 * styling, and attach StringAnnotations so we can handle clicks in ClickableText.
 * Excludes URLs that are already part of markdown link syntax to avoid duplicates.
 */
fun annotateLinksAndPhones(source: AnnotatedString, linkColor: Color): AnnotatedString {
    val text = source.text
    val builder = AnnotatedString.Builder()
    builder.append(source)

    // Regex that excludes URLs inside markdown link syntax [text](url)
    // More precise regex to avoid matching text like "analyzing...okay"
    // Only matches proper domain patterns and full URLs
    val urlRegex = Regex("""(?<!\]\()(https?://[^\s\)]+|(?<!\]\()www\.(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}[^\s\)]*|(?<!\]\()(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.){2,}[a-zA-Z]{2,}(?:/[^\s\)]*)?)(?!\))""")
    val phoneRegex = Regex("""\+?[0-9][0-9\-\s]{6,}[0-9]""")

    fun addAnnotation(range: IntRange, annotationTag: String, annotationValue: String) {
        builder.addStyle(
            SpanStyle(
                color = linkColor,
                textDecoration = TextDecoration.Underline
            ),
            range.first,
            range.last + 1
        )
        builder.addStringAnnotation(annotationTag, annotationValue, range.first, range.last + 1)
    }

    for (match in urlRegex.findAll(text)) {
        val originalUrl = match.value
        var url = originalUrl.trim()
        
        // Clean up common trailing characters that shouldn't be part of URL
        val trimmedChars = url.length
        url = url.trimEnd('.', ',', ')', ']', '}', '!', '?', ';', ':')
        val charsRemoved = trimmedChars - url.length
        
        // Add protocol if missing
        if (!url.startsWith("http://") && !url.startsWith("https://")) {
            url = "https://$url" // use HTTPS by default for security
        }
        
        // Adjust range for trimmed characters
        val adjustedRange = match.range.first until (match.range.last + 1 - charsRemoved)
        addAnnotation(adjustedRange, "URL", url)
    }

    for (match in phoneRegex.findAll(text)) {
        val numberDigits = match.value.filter { it.isDigit() || it == '+' }
        addAnnotation(match.range, "PHONE", numberDigits)
    }

    return builder.toAnnotatedString()
}

/**
 * Enhanced markdown parser that converts markdown syntax to AnnotatedString
 * Supports: **bold**, *italic*, `code`, ### headers, - lists, [links](url), and preserves line breaks
 */
fun parseMarkdownToAnnotatedString(markdown: String, baseColor: Color, linkColor: Color): AnnotatedString {
    // Normalize common escaped newline/tab sequences if the text came in JSON-escaped form
    var normalized = markdown
    if ('\r' in normalized) {
        normalized = normalized.replace("\r\n", "\n").replace('\r', '\n')
    }
    if ("\\n" in normalized) {
        val hasActualNewlines = normalized.count { it == '\n' } > 0
        if (!hasActualNewlines || normalized.contains("\\n\\n")) {
            normalized = normalized.replace("\\n", "\n")
        }
    }
    if ("\\t" in normalized) {
        normalized = normalized.replace("\\t", "    ")
    }
    return buildAnnotatedString {
        val lines = normalized.split('\n')
        
        for (lineIndex in lines.indices) {
            val line = lines[lineIndex]
            
            when {
                // Headers (### Header)
                line.startsWith("### ") -> {
                    withStyle(SpanStyle(
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                        color = baseColor
                    )) {
                        append(line.substring(4))
                    }
                }
                line.startsWith("## ") -> {
                    withStyle(SpanStyle(
                        fontWeight = FontWeight.Bold,
                        fontSize = 20.sp,
                        color = baseColor
                    )) {
                        append(line.substring(3))
                    }
                }
                line.startsWith("# ") -> {
                    withStyle(SpanStyle(
                        fontWeight = FontWeight.Bold,
                        fontSize = 22.sp,
                        color = baseColor
                    )) {
                        append(line.substring(2))
                    }
                }
                // List items (- item or * item)
                line.trimStart().startsWith("- ") || line.trimStart().startsWith("* ") -> {
                    val indent = line.length - line.trimStart().length
                    append("  ".repeat(indent / 2)) // Convert spaces to proper indent
                    append("• ") // Bullet point
                    parseInlineMarkdown(line.trimStart().substring(2), baseColor, this, linkColor)
                }
                // Regular text with inline formatting
                else -> {
                    parseInlineMarkdown(line, baseColor, this, linkColor)
                }
            }
            
            // Add line break except for the last line
            if (lineIndex < lines.size - 1) {
                append('\n')
            }
        }
    }
}

/**
 * Parse inline markdown formatting (bold, italic, code, links) within a line
 */
fun parseInlineMarkdown(text: String, baseColor: Color, builder: AnnotatedString.Builder, linkColor: Color = baseColor) {
    var i = 0
    
    while (i < text.length) {
        when {
            // Markdown links [text](url)
            text[i] == '[' -> {
                val textEndIndex = text.indexOf(']', i + 1)
                if (textEndIndex != -1 && textEndIndex + 1 < text.length && text[textEndIndex + 1] == '(') {
                    val urlEndIndex = text.indexOf(')', textEndIndex + 2)
                    if (urlEndIndex != -1) {
                        val linkText = text.substring(i + 1, textEndIndex)
                        val linkUrl = text.substring(textEndIndex + 2, urlEndIndex)
                        
                        // Add the link with URL annotation and styling
                        val startIndex = builder.length
                        builder.withStyle(SpanStyle(
                            color = linkColor,
                            textDecoration = TextDecoration.Underline
                        )) {
                            append(linkText)
                        }
                        val endIndex = builder.length
                        
                        // Add URL annotation for click handling
                        var processedUrl = linkUrl.trim()
                        if (!processedUrl.startsWith("http://") && !processedUrl.startsWith("https://")) {
                            processedUrl = "https://$processedUrl"
                        }
                        builder.addStringAnnotation("URL", processedUrl, startIndex, endIndex)
                        
                        i = urlEndIndex + 1
                        continue
                    }
                }
                // If not a valid markdown link, treat as regular character
                builder.withStyle(SpanStyle(color = baseColor)) { append(text[i]) }
                i++
            }
            // Bold text **text**
            i < text.length - 1 && text[i] == '*' && text[i + 1] == '*' -> {
                val endIndex = text.indexOf("**", i + 2)
                if (endIndex != -1) {
                    builder.withStyle(SpanStyle(fontWeight = FontWeight.Bold, color = baseColor)) {
                        append(text.substring(i + 2, endIndex))
                    }
                    i = endIndex + 2
                } else {
                    builder.withStyle(SpanStyle(color = baseColor)) { append(text[i]) }
                    i++
                }
            }
            // Italic text *text*
            text[i] == '*' -> {
                val endIndex = text.indexOf('*', i + 1)
                if (endIndex != -1) {
                    builder.withStyle(SpanStyle(fontStyle = FontStyle.Italic, color = baseColor)) {
                        append(text.substring(i + 1, endIndex))
                    }
                    i = endIndex + 1
                } else {
                    builder.withStyle(SpanStyle(color = baseColor)) { append(text[i]) }
                    i++
                }
            }
            // Code text `text`
            text[i] == '`' -> {
                val endIndex = text.indexOf('`', i + 1)
                if (endIndex != -1) {
                    builder.withStyle(SpanStyle(
                        fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                        background = baseColor.copy(alpha = 0.15f),
                        color = baseColor
                    )) {
                        append(text.substring(i + 1, endIndex))
                    }
                    i = endIndex + 1
                } else {
                    builder.withStyle(SpanStyle(color = baseColor)) { append(text[i]) }
                    i++
                }
            }
            else -> {
                builder.withStyle(SpanStyle(color = baseColor)) {
                    append(text[i])
                }
                i++
            }
        }
    }
}

/**
 * Enhanced chat bubble that shows user/assistant messages in ChatGPT mobile style.
 * User messages have bubbles, AI responses are plain text without background.
 */
@Composable
fun MessageBubble(
    message: MessageEntity,
    streamingContent: String = "",
    onRegenerateResponse: (() -> Unit)? = null,
    onEditUserMessage: (() -> Unit)? = null,
    onTtsSpeak: ((String) -> Unit)? = null,
    onTtsStop: (() -> Unit)? = null,
    isTtsSpeaking: Boolean = false
) {
    var showFullScreenImage by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val isUser = message.isFromUser
    
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp, vertical = 8.dp),
        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
    ) {
        if (isUser) {
            // User messages - bubble design
            Surface(
                modifier = Modifier
                    .wrapContentWidth()
                    .widthIn(max = 300.dp),
                shape = RoundedCornerShape(
                        topStart = 20.dp,
                        topEnd = 4.dp,
                    bottomStart = 20.dp,
                    bottomEnd = 20.dp
                ),
                    color = MaterialTheme.colorScheme.primary,
                shadowElevation = 1.dp
            ) {
                Column(
                    modifier = Modifier
                        .wrapContentWidth()
                        .padding(16.dp)
                ) {
                    // Display attachment if exists
                    if (message.attachmentPath != null && message.attachmentType != null) {
                        when (message.attachmentType.lowercase()) {
                            "image" -> {
                                AsyncImage(
                                    model = ImageRequest.Builder(context)
                                        .data(Uri.parse(message.attachmentPath))
                                        .crossfade(true)
                                        .build(),
                                    contentDescription = "Attached image",
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .heightIn(max = 200.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .clickable { showFullScreenImage = true },
                                    contentScale = ContentScale.Crop,
                                    onError = { 
                                        android.util.Log.w("MessageBubble", "Failed to load image: ${message.attachmentPath}")
                                    }
                                )
                            }
                            "audio" -> {
                                // Display audio playback controls
                                AudioMessageCard(
                                    audioPath = message.attachmentPath,
                                    fileName = message.attachmentFileName ?: "Audio",
                                    fileSize = message.attachmentFileSize,
                                    isFromUser = message.isFromUser
                                )
                            }
                            else -> {
                                // Display file attachment card for non-images
                                FileAttachmentCard(
                                    attachmentPath = message.attachmentPath,
                                    attachmentType = message.attachmentType,
                                    attachmentFileName = message.attachmentFileName,
                                    attachmentFileSize = message.attachmentFileSize,
                                    isFromUser = message.isFromUser
                                )
                            }
                        }
                        
                        if (message.content.isNotEmpty() && 
                            message.content != "Shared a file" && 
                            !message.content.startsWith("📄 **File Content**") &&
                            !message.content.contains("---\n\n📄 **File Content**")) {
                            Spacer(modifier = Modifier.height(8.dp))
                        }
                    }
                    
                    // Display text content
                    if (message.content.isNotEmpty() && message.content != "Shared a file") {
                                        RenderMessageSegments(
                                            displayContent = message.content,
                                            isUser = true,
                                            baseColor = MaterialTheme.colorScheme.onPrimary,
                                            linkColor = MaterialTheme.colorScheme.onPrimary,
                                            fontSize = MaterialTheme.typography.bodyLarge.fontSize,
                                            modifier = Modifier.wrapContentWidth(),
                                            context = context
                                        )
                                    }
                }
            }
            
            // Show actions row for user messages
            if (message.content.isNotEmpty() && message.content != "Shared a file") {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 4.dp),
                    horizontalArrangement = Arrangement.End
                ) {
                    IconButton(
                        onClick = {
                            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                            val clip = ClipData.newPlainText("Message", message.content)
                            clipboard.setPrimaryClip(clip)
                        },
                        modifier = Modifier.size(24.dp)
                    ) {
                        Icon(
                            Icons.Outlined.ContentCopy,
                            contentDescription = "Copy message",
                            modifier = Modifier.size(12.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    if (onEditUserMessage != null) {
                        Spacer(modifier = Modifier.width(4.dp))
                        IconButton(
                            onClick = onEditUserMessage,
                            modifier = Modifier.size(24.dp)
                        ) {
                            Icon(
                                Icons.Outlined.Edit,
                                contentDescription = "Edit message",
                                modifier = Modifier.size(12.dp),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        } else {
            // AI responses - plain text without background bubble, like ChatGPT mobile
            Column(
                modifier = Modifier.fillMaxWidth()
            ) {
                // Display attachment if exists (for AI messages with attachments)
                if (message.attachmentPath != null && message.attachmentType != null) {
                    when (message.attachmentType.lowercase()) {
                        "image" -> {
                            AsyncImage(
                                model = ImageRequest.Builder(context)
                                    .data(Uri.parse(message.attachmentPath))
                                    .crossfade(true)
                                    .build(),
                                contentDescription = "Attached image",
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 200.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .clickable { showFullScreenImage = true },
                                contentScale = ContentScale.Crop,
                                onError = { 
                                    android.util.Log.w("MessageBubble", "Failed to load image: ${message.attachmentPath}")
                                }
                            )
                        }
                        "audio" -> {
                            // Display audio playback controls
                            AudioMessageCard(
                                audioPath = message.attachmentPath,
                                fileName = message.attachmentFileName ?: "Audio",
                                fileSize = message.attachmentFileSize,
                                isFromUser = message.isFromUser
                            )
                        }
                        else -> {
                            // Display file attachment card for non-images
                            FileAttachmentCard(
                                attachmentPath = message.attachmentPath,
                                attachmentType = message.attachmentType,
                                attachmentFileName = message.attachmentFileName,
                                attachmentFileSize = message.attachmentFileSize,
                                isFromUser = message.isFromUser
                            )
                        }
                    }
                    
                    if (message.content.isNotEmpty() && 
                        message.content != "Shared a file" && 
                        message.content != "…" &&
                        !message.content.startsWith("📄 **File Content**") &&
                        !message.content.contains("---\n\n📄 **File Content**")) {
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                }
                
                // Display text content - plain text without background
                if (message.content.isNotEmpty() && message.content != "Shared a file") {
                    val displayContent = if (streamingContent.isNotEmpty()) streamingContent else message.content

                    RenderMessageSegments(
                        displayContent = displayContent,
                        isUser = false,
                        baseColor = MaterialTheme.colorScheme.onSurface,
                        linkColor = MaterialTheme.colorScheme.primary,
                        fontSize = MaterialTheme.typography.bodyLarge.fontSize,
                        modifier = Modifier.fillMaxWidth(),
                        context = context
                    )
                }
                
                // Action buttons row for AI messages
                if (message.content != "…" && message.content.isNotEmpty()) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // TTS button (Read aloud)
                        if (onTtsSpeak != null && onTtsStop != null) {
                            IconButton(
                                onClick = {
                                    if (isTtsSpeaking) {
                                        onTtsStop()
                                    } else {
                                        val displayContent = if (streamingContent.isNotEmpty()) streamingContent else message.content
                                        onTtsSpeak(displayContent)
                                    }
                                },
                                modifier = Modifier.size(32.dp)
                            ) {
                                Icon(
                                    if (isTtsSpeaking) Icons.Filled.Stop else Icons.Filled.VolumeUp,
                                    contentDescription = if (isTtsSpeaking) "Stop reading" else "Read aloud",
                                    modifier = Modifier.size(16.dp),
                                    tint = if (isTtsSpeaking) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        
                        // Copy button
                        IconButton(
                            onClick = {
                                val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                val displayContent = if (streamingContent.isNotEmpty()) streamingContent else message.content
                                val clip = ClipData.newPlainText("Message", displayContent)
                                clipboard.setPrimaryClip(clip)
                            },
                            modifier = Modifier.size(32.dp)
                        ) {
                            Icon(
                                Icons.Outlined.ContentCopy,
                                contentDescription = "Copy message",
                                modifier = Modifier.size(16.dp),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        
                        // Regenerate button
                        if (onRegenerateResponse != null) {
                            IconButton(
                                onClick = onRegenerateResponse,
                                modifier = Modifier.size(32.dp)
                            ) {
                                Icon(
                                    Icons.Default.Refresh,
                                    contentDescription = "Regenerate response",
                                    modifier = Modifier.size(16.dp),
                                    tint = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        
                        // Token statistics for assistant messages
                        val hasStats = message.tokenCount != null && message.tokensPerSecond != null
                        if (hasStats) {
                            Spacer(modifier = Modifier.weight(1f))
                            Row(
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
                }
            }
        }
    }
    
    // Full-screen image viewer
    if (showFullScreenImage && message.attachmentPath != null && message.attachmentType?.lowercase() == "image") {
        FullScreenImageViewer(
            imageUri = Uri.parse(message.attachmentPath),
            onDismiss = { showFullScreenImage = false }
        )
    }
}

/**
 * Full-screen image viewer for chat attachments with enhanced features
 */
@Composable
fun FullScreenImageViewer(
    imageUri: Uri,
    onDismiss: () -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var showControls by remember { mutableStateOf(true) }
    
    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            usePlatformDefaultWidth = false,
            decorFitsSystemWindows = false
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black)
                .systemBarsPadding()
                .clickable { showControls = !showControls }
        ) {
            // Full-screen image
            AsyncImage(
                model = ImageRequest.Builder(LocalContext.current)
                    .data(imageUri)
                    .crossfade(true)
                    .build(),
                contentDescription = "Full screen image",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
                onError = { 
                    android.util.Log.w("FullScreenImageViewer", "Failed to load image: $imageUri")
                }
            )
            
            // Controls overlay (shows/hides on tap)
            if (showControls) {
                // Close button
                Surface(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(16.dp)
                        .size(48.dp),
                    shape = CircleShape,
                    color = Color.Black.copy(alpha = 0.6f)
                ) {
                    IconButton(
                        onClick = onDismiss,
                        modifier = Modifier.fillMaxSize()
                    ) {
                        Icon(
                            Icons.Default.Close,
                            contentDescription = stringResource(R.string.close),
                            tint = Color.White,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                }
                
                // Image title/info overlay at top
                Surface(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(16.dp),
                    shape = RoundedCornerShape(12.dp),
                    color = Color.Black.copy(alpha = 0.6f)
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.Image,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(20.dp)
                        )
            Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = stringResource(R.string.image_attachment),
                            color = Color.White,
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
                
                // Action buttons at bottom
            Surface(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(16.dp),
                shape = RoundedCornerShape(16.dp),
                    color = Color.Black.copy(alpha = 0.6f)
                ) {
                    Row(
                        modifier = Modifier
                            .padding(16.dp)
                            .horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Copy button
                        OutlinedButton(
                            onClick = {
                                coroutineScope.launch {
                                    copyImageToClipboard(context, imageUri)
                                }
                            },
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = Color.White,
                                containerColor = Color.Transparent
                            ),
                            border = androidx.compose.foundation.BorderStroke(
                                1.dp, 
                                Color.White.copy(alpha = 0.6f)
                            )
                        ) {
                            Icon(
                                Icons.Default.ContentCopy,
                                contentDescription = stringResource(R.string.copy),
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(stringResource(R.string.copy))
                        }
                        
                        // Save button
                        OutlinedButton(
                            onClick = {
                                coroutineScope.launch {
                                    saveImageToGallery(context, imageUri)
                                }
                            },
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = Color.White,
                                containerColor = Color.Transparent
                            ),
                            border = androidx.compose.foundation.BorderStroke(
                                1.dp, 
                                Color.White.copy(alpha = 0.6f)
                            )
            ) {
                Icon(
                                Icons.Default.Download,
                                contentDescription = stringResource(R.string.save_image),
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(stringResource(R.string.save_image))
                        }
                    }
                }
                
                // Instructions overlay at top of action buttons
                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(horizontal = 16.dp, vertical = 80.dp),
                    shape = RoundedCornerShape(8.dp),
                    color = Color.Black.copy(alpha = 0.4f)
                ) {
                    Text(
                        text = stringResource(R.string.tap_image_toggle_controls),
                        color = Color.White.copy(alpha = 0.8f),
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun RenderMessageSegments(
    displayContent: String,
    isUser: Boolean,
    baseColor: Color,
    linkColor: Color,
    fontSize: TextUnit,
    modifier: Modifier = Modifier,
    context: Context? = null
) {
    val segments = remember(displayContent) { CodeBlockParser.parseSegments(displayContent) }

    Column(modifier = modifier) {
        for (seg in segments) {
            when (seg) {
                is ParsedSegment.Text -> {
                    val textModifier = if (isUser) {
                        Modifier
                            .wrapContentWidth()
                            .widthIn(max = 280.dp)
                    } else {
                        Modifier.fillMaxWidth()
                    }

                    SelectableMarkdownText(
                        markdown = seg.text,
                        color = baseColor,
                        fontSize = fontSize,
                        modifier = textModifier,
                        textAlign = if (isUser) TextAlign.End else TextAlign.Start,
                        linkColorOverride = linkColor
                    )
                }
                is ParsedSegment.Code -> {
                    // Render code block with monospace and a subtle background
                    Box {
                        val codeModifier = if (isUser) {
                            Modifier
                                .wrapContentWidth()
                                .widthIn(max = 280.dp)
                                .background(baseColor.copy(alpha = 0.06f))
                                .padding(12.dp)
                        } else {
                            Modifier
                                .fillMaxWidth()
                                .background(baseColor.copy(alpha = 0.06f))
                                .padding(12.dp)
                        }

                        SelectionContainer {
                            Text(
                                text = seg.content.trimEnd('\n'),
                                fontFamily = FontFamily.Monospace,
                                fontSize = (fontSize.value - 0).sp,
                                color = baseColor,
                                modifier = codeModifier
                            )
                        }
                        
                        // Copy icon for code blocks (only for assistant messages and when context is available)
                        if (!isUser && context != null) {
                            IconButton(
                                onClick = {
                                    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                    val clip = ClipData.newPlainText("Code", seg.content.trimEnd('\n'))
                                    clipboard.setPrimaryClip(clip)
                                },
                                modifier = Modifier
                                    .align(Alignment.TopEnd)
                                    .padding(8.dp)
                                    .size(32.dp)
                                    .background(
                                        MaterialTheme.colorScheme.surface.copy(alpha = 0.9f),
                                        CircleShape
                                    )
                            ) {
                                Icon(
                                    Icons.Outlined.ContentCopy,
                                    contentDescription = "Copy code",
                                    modifier = Modifier.size(16.dp),
                                    tint = MaterialTheme.colorScheme.onSurface
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * Copy image to clipboard
 */
private suspend fun copyImageToClipboard(context: Context, imageUri: Uri) {
    withContext(Dispatchers.IO) {
        try {
            val imageLoader = ImageLoader(context)
            val request = ImageRequest.Builder(context)
                .data(imageUri)
                .build()
            
            val result = imageLoader.execute(request)
            if (result is SuccessResult) {
                val bitmap = (result.drawable as? BitmapDrawable)?.bitmap
                if (bitmap != null) {
                    withContext(Dispatchers.Main) {
                        val clipboardManager = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        // For now, we'll copy the URI as text since bitmap clipboard requires API 28+
            val clip = ClipData.newPlainText("Image", context.getString(R.string.image_copied_clipboard))
            clipboardManager.setPrimaryClip(clip)
            Toast.makeText(context, context.getString(R.string.image_copied_to_clipboard), Toast.LENGTH_SHORT).show()
                    }
                } else {
                    withContext(Dispatchers.Main) {
            Toast.makeText(context, context.getString(R.string.failed_to_copy_image), Toast.LENGTH_SHORT).show()
                    }
                }
            }
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
        Toast.makeText(context, context.getString(R.string.failed_to_copy_image_with_error, e.message ?: ""), Toast.LENGTH_SHORT).show()
            }
        }
    }
}

/**
 * Save image to gallery
 */
private suspend fun saveImageToGallery(context: Context, imageUri: Uri) {
    withContext(Dispatchers.IO) {
        try {
            val imageLoader = ImageLoader(context)
            val request = ImageRequest.Builder(context)
                .data(imageUri)
                .build()
            
            val result = imageLoader.execute(request)
            if (result is SuccessResult) {
                val bitmap = (result.drawable as? BitmapDrawable)?.bitmap
                if (bitmap != null) {
                    val saved = saveBitmapToGallery(context, bitmap)
                    withContext(Dispatchers.Main) {
                        if (saved) {
                            Toast.makeText(context, context.getString(R.string.image_saved_to_gallery), Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(context, context.getString(R.string.failed_to_save_image), Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(context, context.getString(R.string.failed_to_load_image), Toast.LENGTH_SHORT).show()
                    }
                }
            }
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                Toast.makeText(context, context.getString(R.string.failed_to_save_image_with_error, e.message ?: ""), Toast.LENGTH_SHORT).show()
            }
        }
    }
}

/**
 * Save bitmap to gallery using MediaStore
 */
private fun saveBitmapToGallery(context: Context, bitmap: Bitmap): Boolean {
    return try {
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "LLMHub_Image_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/LLMHub")
            }
        }
        
        val contentResolver = context.contentResolver
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
        
        uri?.let { imageUri ->
            contentResolver.openOutputStream(imageUri)?.use { outputStream ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, outputStream)
            }
            true
        } ?: false
    } catch (e: Exception) {
        android.util.Log.e("ImageSave", "Failed to save image: ${e.message}")
        false
    }
}



/**
 * Modern input bar with Material Design 3 styling, attachment support, audio recording, and smooth animations.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MessageInput(
    onSendMessage: (String, Uri?, ByteArray?) -> Unit, // Updated to include audio data
    enabled: Boolean,
    supportsAttachments: Boolean,
    supportsVision: Boolean = false,
    supportsAudio: Boolean = false, // New parameter for audio support
    isLoading: Boolean = false,
    onCancelGeneration: (() -> Unit)? = null,
    // Edit mode support
    isEditing: Boolean = false,
    editText: String? = null,
    onEditTextChange: ((String) -> Unit)? = null,
    onConfirmEdit: (() -> Unit)? = null,
    onCancelEdit: (() -> Unit)? = null
) {
    val context = LocalContext.current
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current
    var textState by remember { mutableStateOf(TextFieldValue("")) }
    var attachmentUri by remember { mutableStateOf<Uri?>(null) }
    var attachmentInfo by remember { mutableStateOf<FileUtils.FileInfo?>(null) }
    var showFilePreview by remember { mutableStateOf(false) }
    var showAttachmentOptions by remember { mutableStateOf(false) }
    
    // Audio recording states
    var isRecording by remember { mutableStateOf(false) }
    var audioLevel by remember { mutableStateOf(0f) }
    var recordedAudioData by remember { mutableStateOf<ByteArray?>(null) }
    var hasAudioPermission by remember { mutableStateOf(false) }
    
    // Audio service
    val audioService = remember { AudioInputService(context) }
    val coroutineScope = rememberCoroutineScope()
    
    // Observe elapsed time from audio service
    val elapsedTimeMs by audioService.elapsedTimeMs.collectAsState()
    val remainingSeconds = ((29500L - elapsedTimeMs) / 1000).coerceAtLeast(0)
    // Sync internal text with external edit text when provided
    LaunchedEffect(isEditing, editText) {
        if (isEditing && editText != null) {
            textState = TextFieldValue(
                text = editText,
                selection = TextRange(editText.length)
            )
        }
    }

    // Focus management for edit mode
    val inputFocusRequester = remember { FocusRequester() }
    LaunchedEffect(isEditing) {
        if (isEditing) {
            // Small delay to ensure composable is in the tree before requesting focus
            kotlinx.coroutines.delay(50)
            runCatching { inputFocusRequester.requestFocus() }
            keyboardController?.show()
        }
    }

    
    // Audio permission launcher
    val audioPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasAudioPermission = isGranted
        if (isGranted) {
            isRecording = true
        }
    }
    
    // Check audio permission on first composition
    LaunchedEffect(Unit) {
        hasAudioPermission = audioService.hasAudioPermission()
        
        // Set up callback for auto-stop
        audioService.onRecordingAutoStopped = {
            isRecording = false
        }
    }
    
    // Audio recording effect
    LaunchedEffect(isRecording) {
        if (isRecording && hasAudioPermission) {
            val success = audioService.startRecording()
            if (!success) {
                isRecording = false
            }
        } else if (!isRecording) {
            // Stop recording when isRecording becomes false (either manual or auto-stop)
            if (audioService.isRecording() || recordedAudioData == null) {
                val audioData = audioService.stopRecording()
                if (audioData != null) {
                    recordedAudioData = audioData
                }
            }
        }
    }
    
    // Image picker for vision models
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            attachmentUri = it
            // Get file info
            kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.Main).launch {
                attachmentInfo = FileUtils.getFileInfo(context, it)
                showFilePreview = true
            }
        }
    }
    
    // General file picker for documents
    val documentPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            attachmentUri = it
            // Get file info
            kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.Main).launch {
                attachmentInfo = FileUtils.getFileInfo(context, it)
                showFilePreview = true
            }
        }
    }

    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surface,
        shadowElevation = 8.dp
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            // File attachment preview
            if (attachmentUri != null && attachmentInfo != null) {
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
                        // File type icon or image preview
                        if (attachmentInfo!!.type == FileUtils.SupportedFileType.IMAGE) {
                            AsyncImage(
                                model = ImageRequest.Builder(context)
                                    .data(attachmentUri)
                                    .crossfade(true)
                                    .build(),
                                contentDescription = stringResource(R.string.attached_image),
                                modifier = Modifier
                                    .size(48.dp)
                                    .clip(MaterialTheme.shapes.small)
                                    .clickable { showFilePreview = true },
                                contentScale = ContentScale.Crop
                            )
                        } else {
                            // Show file type icon for non-images
                            Surface(
                                modifier = Modifier.size(48.dp),
                                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
                                shape = MaterialTheme.shapes.small
                            ) {
                                Column(
                                    modifier = Modifier.fillMaxSize(),
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    verticalArrangement = Arrangement.Center
                                ) {
                                    Text(
                                        text = attachmentInfo!!.type.icon,
                                        style = MaterialTheme.typography.headlineSmall
                                    )
                                }
                            }
                        }
                        
                        Spacer(modifier = Modifier.width(12.dp))
                        
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = attachmentInfo!!.name,
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium,
                                color = MaterialTheme.colorScheme.onSecondaryContainer,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Text(
                                text = FileUtils.getLocalizedDisplayName(LocalContext.current, attachmentInfo!!.type),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                            )
                        }
                        
                        IconButton(onClick = { 
                            attachmentUri = null
                            attachmentInfo = null
                        }) {
                            Icon(
                                Icons.Default.Close,
                                contentDescription = "Remove attachment",
                                tint = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        }
                    }
                }
            }
            
            // Audio recording indicator
            if (isRecording || recordedAudioData != null) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isRecording) 
                            MaterialTheme.colorScheme.errorContainer 
                        else MaterialTheme.colorScheme.tertiaryContainer
                    ),
                    shape = MaterialTheme.shapes.medium
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Recording icon with animation
                        Surface(
                            modifier = Modifier.size(48.dp),
                            color = if (isRecording) 
                                MaterialTheme.colorScheme.error.copy(alpha = 0.2f)
                            else MaterialTheme.colorScheme.tertiary.copy(alpha = 0.2f),
                            shape = CircleShape
                        ) {
                            Icon(
                                if (isRecording) Icons.Default.Mic else Icons.Default.MicOff,
                                contentDescription = if (isRecording) stringResource(R.string.recording) else stringResource(R.string.audio_recorded),
                                modifier = Modifier.padding(12.dp),
                                tint = if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.tertiary
                            )
                        }
                        
                        Spacer(modifier = Modifier.width(12.dp))
                        
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = if (isRecording) stringResource(R.string.recording) else stringResource(R.string.audio_recorded),
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium,
                                color = if (isRecording) 
                                    MaterialTheme.colorScheme.onErrorContainer
                                else MaterialTheme.colorScheme.onTertiaryContainer
                            )
                            if (isRecording) {
                                Text(
                                    text = "${remainingSeconds}s ${stringResource(R.string.remaining)}",
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = if (remainingSeconds <= 5) FontWeight.Bold else FontWeight.Normal,
                                    color = if (remainingSeconds <= 5) 
                                        MaterialTheme.colorScheme.error
                                    else MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.7f)
                                )
                            } else {
                                Text(
                                    text = "${recordedAudioData?.size?.let { "${it / 1000}KB" } ?: "0KB"} • ${stringResource(R.string.ready_to_send)}",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onTertiaryContainer.copy(alpha = 0.7f)
                                )
                            }
                        }
                        
                        // Audio controls
                        // Capture context in composition scope (used by both branches)
                        val previewContext = LocalContext.current
                        if (isRecording) {
                            // Stop recording button
                            IconButton(onClick = {
                                coroutineScope.launch {
                                    recordedAudioData = audioService.stopRecording()
                                    isRecording = false
                                }
                            }) {
                                Icon(
                                    Icons.Default.Stop,
                                    contentDescription = stringResource(R.string.tap_to_stop),
                                    tint = MaterialTheme.colorScheme.error
                                )
                            }
                        } else if (recordedAudioData != null) {
                            // Play button for previewing recorded audio before send
                            var previewPlayer by remember { mutableStateOf<android.media.MediaPlayer?>(null) }
                            var isPreviewPlaying by remember { mutableStateOf(false) }
                            DisposableEffect(Unit) {
                                onDispose {
                                    previewPlayer?.release()
                                    previewPlayer = null
                                }
                            }
                            IconButton(onClick = {
                                try {
                                    if (isPreviewPlaying) {
                                        previewPlayer?.pause()
                                        isPreviewPlaying = false
                                    } else {
                                        if (previewPlayer == null) {
                                            // Write bytes to a temp file so MediaPlayer can read
                                            val tmp = java.io.File(previewContext.cacheDir, "preview_${System.currentTimeMillis()}.wav")
                                            tmp.writeBytes(recordedAudioData!!)
                                            previewPlayer = android.media.MediaPlayer().apply {
                                                setDataSource(tmp.absolutePath)
                                                setOnCompletionListener {
                                                    isPreviewPlaying = false
                                                    // best-effort delete
                                                    runCatching { tmp.delete() }
                                                }
                                                prepare()
                                                start()
                                            }
                                        } else {
                                            previewPlayer?.start()
                                        }
                                        isPreviewPlaying = true
                                    }
                                } catch (e: Exception) {
                                    android.util.Log.e("ChatInput", "Preview playback failed: ${e.message}", e)
                                    isPreviewPlaying = false
                                }
                            }) {
                                Icon(
                                    if (isPreviewPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                                    contentDescription = if (isPreviewPlaying) "Pause" else "Play",
                                    tint = MaterialTheme.colorScheme.tertiary
                                )
                            }
                            
                            // Delete audio button
                            IconButton(onClick = { 
                                recordedAudioData = null
                            }) {
                                Icon(
                                    Icons.Default.Delete,
                                    contentDescription = stringResource(R.string.remove_audio),
                                    tint = MaterialTheme.colorScheme.error
                                )
                            }
                        }
                    }
                }
            }

            // Input field with modern styling
            OutlinedTextField(
                value = textState,
                onValueChange = {
                    textState = it
                    onEditTextChange?.invoke(it.text)
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .focusRequester(inputFocusRequester),
                placeholder = { 
                    Text(
                        stringResource(R.string.type_a_message),
                        style = MaterialTheme.typography.bodyLarge
                    )
                },
                enabled = enabled,
                singleLine = false,
                maxLines = 4,
                shape = RoundedCornerShape(24.dp),
                leadingIcon = if (supportsAttachments) {
                    {
                        Box {
                            IconButton(
                                onClick = { showAttachmentOptions = !showAttachmentOptions },
                                enabled = enabled
                            ) {
                                Icon(
                                    Icons.Outlined.AttachFile,
                                    contentDescription = "Attach file",
                                    tint = if (enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                                )
                            }
                            
                            // Attachment options dropdown
                            DropdownMenu(
                                expanded = showAttachmentOptions,
                                onDismissRequest = { showAttachmentOptions = false }
                            ) {
                                // Only show images option for vision models
                                if (supportsVision) {
                                    DropdownMenuItem(
                                        text = { 
                                            Row(verticalAlignment = Alignment.CenterVertically) {
                                                Text("🖼️", style = MaterialTheme.typography.headlineSmall)
                                                Spacer(modifier = Modifier.width(12.dp))
                                                Column {
                                                    Text(stringResource(R.string.images))
                                                    Text(
                                                        stringResource(R.string.images_description),
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                                    )
                                                }
                                            }
                                        },
                                        onClick = {
                                            showAttachmentOptions = false
                                            imagePickerLauncher.launch("image/*")
                                        }
                                    )
                                }
                                
                                DropdownMenuItem(
                                    text = { 
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Text("📄", style = MaterialTheme.typography.headlineSmall)
                                            Spacer(modifier = Modifier.width(12.dp))
                                            Column {
                                                Text(stringResource(R.string.documents))
                                                Text(
                                                    stringResource(R.string.documents_description),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                                )
                                            }
                                        }
                                    },
                                    onClick = {
                                        showAttachmentOptions = false
                                        documentPickerLauncher.launch("*/*")
                                    }
                                )
                                DropdownMenuItem(
                                    text = { 
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Text("📝", style = MaterialTheme.typography.headlineSmall)
                                            Spacer(modifier = Modifier.width(12.dp))
                                            Column {
                                                Text(stringResource(R.string.text_files))
                                                Text(
                                                    stringResource(R.string.text_files_description),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                                )
                                            }
                                        }
                                    },
                                    onClick = {
                                        showAttachmentOptions = false
                                        documentPickerLauncher.launch("text/*")
                                    }
                                )
                                
                                // Audio recording option for audio-capable models
                                if (supportsAudio) {
                                    DropdownMenuItem(
                                        text = { 
                                            Row(verticalAlignment = Alignment.CenterVertically) {
                                                Text("🎙️", style = MaterialTheme.typography.headlineSmall)
                                                Spacer(modifier = Modifier.width(12.dp))
                                                Column {
                                                    Text(stringResource(R.string.audio_recording))
                                                    Text(
                                                        stringResource(R.string.audio_description),
                                                        style = MaterialTheme.typography.bodySmall,
                                                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                                                    )
                                                }
                                            }
                                        },
                                        onClick = {
                                            showAttachmentOptions = false
                                            if (hasAudioPermission) {
                                                isRecording = true
                                            } else {
                                                audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                            }
                                        }
                                    )
                                }
                            }
                        }
                    }
                } else null,
                trailingIcon = {
                    if (isLoading && onCancelGeneration != null) {
                        // Show cancel button when loading
                        IconButton(
                            onClick = onCancelGeneration,
                            enabled = true
                        ) {
                            Surface(
                                modifier = Modifier.size(32.dp),
                                shape = RoundedCornerShape(16.dp),
                                color = MaterialTheme.colorScheme.error
                            ) {
                                Icon(
                                    Icons.Default.Stop,
                                contentDescription = stringResource(R.string.cancel_generation),
                                    modifier = Modifier.padding(6.dp),
                                    tint = MaterialTheme.colorScheme.onError
                                )
                            }
                        }
                    } else if (isEditing && (onConfirmEdit != null || onCancelEdit != null)) {
                        // Show confirm/cancel when editing
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            if (onCancelEdit != null || onEditTextChange != null) {
                                IconButton(onClick = {
                                    if (textState.text.isNotEmpty()) {
                                        textState = TextFieldValue("")
                                        onEditTextChange?.invoke("")
                                    } else {
                                        onCancelEdit?.invoke()
                                    }
                                }) {
                                    Surface(
                                        modifier = Modifier.size(32.dp),
                                        shape = RoundedCornerShape(16.dp),
                                        color = MaterialTheme.colorScheme.surfaceVariant
                                    ) {
                                        Icon(
                                            Icons.Default.Close,
                                            contentDescription = stringResource(R.string.close),
                                            modifier = Modifier.padding(6.dp),
                                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            }
                            if (onConfirmEdit != null) {
                                IconButton(
                                    onClick = {
                                        if (textState.text.isNotBlank() || attachmentUri != null || recordedAudioData != null) {
                                            keyboardController?.hide()
                                            focusManager.clearFocus()
                                            onConfirmEdit()
                                            // Immediately clear local input state after edit-send
                                            textState = TextFieldValue("")
                                            onEditTextChange?.invoke("")
                                            attachmentUri = null
                                            attachmentInfo = null
                                            recordedAudioData = null
                                        }
                                    },
                                    enabled = enabled && textState.text.isNotBlank()
                                ) {
                                    Surface(
                                        modifier = Modifier.size(32.dp),
                                        shape = RoundedCornerShape(16.dp),
                                        color = if (enabled && textState.text.isNotBlank())
                                            MaterialTheme.colorScheme.primary
                                        else MaterialTheme.colorScheme.surfaceVariant
                                    ) {
                                        Icon(
                                            Icons.Default.Send,
                                            contentDescription = stringResource(R.string.send),
                                            modifier = Modifier.padding(6.dp),
                                            tint = if (enabled && textState.text.isNotBlank())
                                                MaterialTheme.colorScheme.onPrimary
                                            else MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            }
                        }
                    } else {
                        // Show send button when not loading
                        IconButton(
                            onClick = {
                                if (textState.text.isNotBlank() || attachmentUri != null || recordedAudioData != null) {
                                    // Aggressively hide keyboard using multiple methods
                                    keyboardController?.hide()
                                    focusManager.clearFocus()
                                    onSendMessage(textState.text, attachmentUri, recordedAudioData)
                                    textState = TextFieldValue("")
                                    attachmentUri = null
                                    attachmentInfo = null
                                    recordedAudioData = null
                                }
                            },
                            enabled = enabled && (textState.text.isNotBlank() || attachmentUri != null || recordedAudioData != null)
                        ) {
                            Surface(
                                modifier = Modifier.size(32.dp),
                                shape = RoundedCornerShape(16.dp),
                                color = if (enabled && (textState.text.isNotBlank() || attachmentUri != null || recordedAudioData != null)) 
                                    MaterialTheme.colorScheme.primary 
                                else MaterialTheme.colorScheme.surfaceVariant
                            ) {
                                Icon(
                                    Icons.Default.Send,
                                    contentDescription = stringResource(R.string.send),
                                    modifier = Modifier.padding(6.dp),
                                    tint = if (enabled && (textState.text.isNotBlank() || attachmentUri != null || recordedAudioData != null)) 
                                        MaterialTheme.colorScheme.onPrimary 
                                    else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
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
    
    // File preview dialog
    if (showFilePreview && attachmentUri != null && attachmentInfo != null) {
        FilePreviewDialog(
            fileInfo = attachmentInfo!!,
            onDismiss = { showFilePreview = false },
            onRemove = {
                attachmentUri = null
                attachmentInfo = null
                showFilePreview = false
            }
        )
    }
}

/**
 * File preview dialog that handles different file types
 */
@Composable
fun FilePreviewDialog(
    fileInfo: FileUtils.FileInfo,
    onDismiss: () -> Unit,
    onRemove: (() -> Unit)? = null, // Make remove optional
    confirmAction: (() -> Unit)? = null,
    confirmText: String? = null
) {
    FilePreviewDialog(
        uri = fileInfo.uri,
        fileName = fileInfo.name,
        fileType = fileInfo.type,
        fileSize = fileInfo.size,
        onDismiss = onDismiss,
        onRemove = onRemove,
        confirmAction = confirmAction,
        confirmText = confirmText
    )
}

@Composable
fun FilePreviewDialog(
    uri: Uri,
    fileName: String,
    fileType: FileUtils.SupportedFileType,
    fileSize: Long,
    onDismiss: () -> Unit,
    onRemove: (() -> Unit)? = null, // Make remove optional
    confirmAction: (() -> Unit)? = null,
    confirmText: String? = null
) {
    val context = LocalContext.current
    var fileContent by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(false) }
    
    // Load file content for all supported document types
    LaunchedEffect(uri) {
        if (fileType in listOf(
            FileUtils.SupportedFileType.TEXT,
            FileUtils.SupportedFileType.JSON,
            FileUtils.SupportedFileType.XML,
            FileUtils.SupportedFileType.PDF,
            FileUtils.SupportedFileType.WORD,
            FileUtils.SupportedFileType.EXCEL,
            FileUtils.SupportedFileType.POWERPOINT
        )) {
            isLoading = true
            fileContent = FileUtils.extractTextContent(context, uri, fileType)
            isLoading = false
        }
    }
    
    Dialog(onDismissRequest = onDismiss) {
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
                        Column {
                            Text(
                                text = fileName,
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Text(
                                text = FileUtils.getLocalizedDisplayName(context, fileType),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        IconButton(onClick = onDismiss) {
                            Icon(Icons.Default.Close, contentDescription = stringResource(R.string.close))
                        }
                    }
                }
                
                // Content preview
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .padding(16.dp)
                ) {
                    when (fileType) {
                        FileUtils.SupportedFileType.IMAGE -> {
                            AsyncImage(
                                model = ImageRequest.Builder(context)
                                    .data(uri)
                                    .crossfade(true)
                                    .build(),
                                contentDescription = "Image preview",
                                modifier = Modifier
                                    .fillMaxSize()
                                    .clip(MaterialTheme.shapes.large),
                                contentScale = ContentScale.Fit
                            )
                        }
                        FileUtils.SupportedFileType.TEXT,
                        FileUtils.SupportedFileType.JSON,
                        FileUtils.SupportedFileType.XML,
                        FileUtils.SupportedFileType.PDF,
                        FileUtils.SupportedFileType.WORD,
                        FileUtils.SupportedFileType.EXCEL,
                        FileUtils.SupportedFileType.POWERPOINT -> {
                            if (isLoading) {
                                Box(
                                    modifier = Modifier.fillMaxSize(),
                                    contentAlignment = Alignment.Center
                                ) {
                                    CircularProgressIndicator()
                                    Text(
                                        text = "Extracting content...",
                                        modifier = Modifier.padding(top = 64.dp),
                                        style = MaterialTheme.typography.bodyMedium
                                    )
                                }
                            } else {
                                Card(
                                    modifier = Modifier.fillMaxSize(),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                                    )
                                ) {
                                    LazyColumn(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .padding(16.dp)
                                    ) {
                                        item {
                                            SelectionContainer {
                                                Text(
                                                    text = fileContent ?: "Could not load file content",
                                                    style = MaterialTheme.typography.bodySmall,
                                                    fontFamily = if (fileType in listOf(
                                                        FileUtils.SupportedFileType.TEXT,
                                                        FileUtils.SupportedFileType.JSON,
                                                        FileUtils.SupportedFileType.XML
                                                    )) {
                                                        androidx.compose.ui.text.font.FontFamily.Monospace
                                                    } else {
                                                        androidx.compose.ui.text.font.FontFamily.Default
                                                    },
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else -> {
                            // For other file types (PDF, Word, etc.)
                            Card(
                                modifier = Modifier.fillMaxSize(),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                                )
                            ) {
                                Column(
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .padding(32.dp),
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    verticalArrangement = Arrangement.Center
                                ) {
                                    Text(
                                        text = fileType.icon,
                                        style = MaterialTheme.typography.displayMedium
                                    )
                                    Spacer(modifier = Modifier.height(16.dp))
                                    Text(
                                        text = fileName,
                                        style = MaterialTheme.typography.headlineSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        textAlign = TextAlign.Center
                                    )
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        text = FileUtils.getLocalizedDisplayName(context, fileType),
                                        style = MaterialTheme.typography.bodyMedium,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        textAlign = TextAlign.Center
                                    )
                                }
                            }
                        }
                    }
                }
                
                // Actions
                Surface(
                    color = MaterialTheme.colorScheme.surfaceContainer
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Only show remove button if onRemove is provided
                        if (onRemove != null) {
                            OutlinedButton(
                                onClick = onRemove,
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(
                                    Icons.Default.Delete,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.remove_attachment))
                            }
                        }

                        // Optional confirm action (e.g. "Add to memory")
                        if (confirmAction != null) {
                            Button(
                                onClick = confirmAction,
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(
                                    Icons.Default.Check,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(confirmText ?: stringResource(R.string.keep_attachment))
                            }
                        } else {
                            Button(
                                onClick = onDismiss,
                                modifier = Modifier.weight(1f)
                            ) {
                                if (onRemove != null) {
                                    Icon(
                                        Icons.Default.Check,
                                        contentDescription = null,
                                        modifier = Modifier.size(18.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(stringResource(R.string.keep_attachment))
                                } else {
                                    Icon(
                                        Icons.Default.Close,
                                        contentDescription = null,
                                        modifier = Modifier.size(18.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(stringResource(R.string.close))
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * File attachment card for displaying non-image attachments in messages
 */
@Composable
fun FileAttachmentCard(
    attachmentPath: String,
    attachmentType: String,
    attachmentFileName: String? = null,
    attachmentFileSize: Long? = null,
    isFromUser: Boolean
) {
    val context = LocalContext.current
    var showPreview by remember { mutableStateOf(false) }
    var fileInfo by remember { mutableStateOf<FileUtils.FileInfo?>(null) }
    
    val fileTypeInfo = FileUtils.SupportedFileType.values().find { 
        it.name.equals(attachmentType, ignoreCase = true) 
    } ?: FileUtils.SupportedFileType.TEXT
    
    // Load file info if not provided (for backward compatibility with existing messages)
    LaunchedEffect(attachmentPath) {
        if (attachmentFileName == null || attachmentFileSize == null) {
            try {
                val uri = Uri.parse(attachmentPath)
                fileInfo = FileUtils.getFileInfo(context, uri)
            } catch (e: Exception) {
                android.util.Log.w("FileAttachmentCard", "Failed to get file info: ${e.message}")
            }
        }
    }
    
    // Use stored file info if available, otherwise fallback to loaded file info
    val displayName = attachmentFileName ?: fileInfo?.name ?: "File attachment"
    
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .clickable { 
                showPreview = true 
            },
        colors = CardDefaults.cardColors(
            containerColor = if (isFromUser) {
                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.7f)
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            }
        ),
        shape = MaterialTheme.shapes.medium
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // File type icon
            Surface(
                modifier = Modifier.size(40.dp),
                color = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
                shape = MaterialTheme.shapes.small
            ) {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = fileTypeInfo.icon,
                        style = MaterialTheme.typography.titleMedium
                    )
                }
            }
            
            Spacer(modifier = Modifier.width(12.dp))
            
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = displayName,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    color = if (isFromUser) {
                        MaterialTheme.colorScheme.onPrimaryContainer
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = "${FileUtils.getLocalizedDisplayName(context, fileTypeInfo)} • ${stringResource(R.string.tap_to_preview)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (isFromUser) {
                        MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                    }
                )
            }
        }
    }
    
    // Show preview dialog when requested
    if (showPreview) {
        val uri = Uri.parse(attachmentPath)
        val finalFileName = attachmentFileName ?: fileInfo?.name ?: "File attachment"
        val finalFileSize = attachmentFileSize ?: fileInfo?.size ?: 0L
        
        FilePreviewDialog(
            uri = uri,
            fileName = finalFileName,
            fileType = fileTypeInfo,
            fileSize = finalFileSize,
            onDismiss = { showPreview = false },
            onRemove = null // No remove/keep buttons for message attachments
        )
    }
}

/**
 * Audio message card with waveform playback controls
 */
@Composable
fun AudioMessageCard(
    audioPath: String,
    fileName: String,
    fileSize: Long?,
    isFromUser: Boolean
) {
    val context = LocalContext.current
    var isPlaying by remember { mutableStateOf(false) }
    var mediaPlayer by remember { mutableStateOf<android.media.MediaPlayer?>(null) }
    var currentPosition by remember { mutableStateOf(0) }
    var duration by remember { mutableStateOf(0) }
    
    // Clean up media player when composable is disposed
    DisposableEffect(audioPath) {
        onDispose {
            mediaPlayer?.release()
            mediaPlayer = null
        }
    }
    
    // Update playback position
    LaunchedEffect(isPlaying) {
        while (isPlaying && mediaPlayer != null) {
            currentPosition = mediaPlayer?.currentPosition ?: 0
            kotlinx.coroutines.delay(100)
        }
    }
    
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        shape = RoundedCornerShape(12.dp),
        color = if (isFromUser) {
            MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
        } else {
            MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        },
        border = BorderStroke(
            1.dp,
            if (isFromUser) {
                MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)
            } else {
                MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
            }
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            // Playback controls with waveform
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Play/Pause button
                IconButton(
                    onClick = {
                        try {
                            if (isPlaying) {
                                // Pause
                                mediaPlayer?.pause()
                                isPlaying = false
                            } else {
                                // Play
                                if (mediaPlayer == null) {
                                    // Initialize media player
                                    mediaPlayer = android.media.MediaPlayer().apply {
                                        setDataSource(context, Uri.parse(audioPath))
                                        prepareAsync()
                                        setOnPreparedListener { player ->
                                            duration = player.duration
                                            player.start()
                                            isPlaying = true
                                        }
                                        setOnCompletionListener {
                                            isPlaying = false
                                            currentPosition = 0
                                        }
                                        setOnErrorListener { _, what, extra ->
                                            android.util.Log.e("AudioMessageCard", "MediaPlayer error: what=$what, extra=$extra")
                                            isPlaying = false
                                            false
                                        }
                                    }
                                } else {
                                    // Resume playback
                                    mediaPlayer?.start()
                                    isPlaying = true
                                }
                            }
                        } catch (e: Exception) {
                            android.util.Log.e("AudioMessageCard", "Audio playback error: ${e.message}", e)
                            isPlaying = false
                        }
                    },
                    modifier = Modifier.size(40.dp)
                ) {
                    Icon(
                        if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                        contentDescription = if (isPlaying) "Pause" else "Play",
                        tint = if (isFromUser) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                }
                
                // Waveform progress visualization
                val progress = if (duration > 0) currentPosition.toFloat() / duration.toFloat() else 0f
                // Use high-contrast colors so the waveform is visible on both themes
                val baseActiveColor = if (isFromUser) {
                    MaterialTheme.colorScheme.onPrimaryContainer
                } else {
                    MaterialTheme.colorScheme.onSurface
                }
                val activeColor = baseActiveColor
                val inactiveColor = baseActiveColor.copy(alpha = 0.35f)
                val barHeights = remember(audioPath) {
                    val random = java.util.Random(audioPath.hashCode().toLong())
                    List(64) { 0.35f + random.nextFloat() * 0.65f }
                }
                Canvas(
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp)
                ) {
                    val barCount = barHeights.size
                    val spacingPx = 3f
                    val totalSpacing = spacingPx * (barCount - 1)
                    val barWidth = (size.width - totalSpacing) / barCount
                    val centerY = size.height / 2f
                    val maxBarHeight = size.height
                    val progressBars = (progress * barCount).coerceIn(0f, barCount.toFloat())
                    for (i in 0 until barCount) {
                        val height = (barHeights[i] * maxBarHeight).coerceAtMost(maxBarHeight)
                        val left = i * (barWidth + spacingPx)
                        val top = centerY - height / 2f
                        val color = if (i < progressBars) activeColor else inactiveColor
                        drawRoundRect(
                            color = color,
                            topLeft = androidx.compose.ui.geometry.Offset(left, top),
                            size = androidx.compose.ui.geometry.Size(barWidth, height),
                            cornerRadius = androidx.compose.ui.geometry.CornerRadius(x = barWidth / 2f, y = barWidth / 2f)
                        )
                    }
                }
                
                // Time text or hint
                if (duration > 0) {
                    Text(
                        text = "${formatTime(currentPosition)} / ${formatTime(duration)}",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (isFromUser) {
                            MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                        }
                    )
                } else {
                    Text(
                        text = stringResource(R.string.tap_to_play),
                        style = MaterialTheme.typography.bodySmall,
                        color = if (isFromUser) {
                            MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                        }
                    )
                }
            }
        }
    }
}

/**
 * Format time in milliseconds to MM:SS format
 */
private fun formatTime(timeMs: Int): String {
    val seconds = timeMs / 1000
    val minutes = seconds / 60
    val remainingSeconds = seconds % 60
    return "%d:%02d".format(minutes, remainingSeconds)
} 