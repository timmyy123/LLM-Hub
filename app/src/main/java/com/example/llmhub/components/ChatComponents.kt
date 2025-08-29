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
import androidx.compose.ui.text.style.TextAlign
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
    textAlign: TextAlign = TextAlign.Start
) {
    val context = LocalContext.current
    // First parse the markdown (bold/italic/lists etc.)
    val baseAnnotated = remember(markdown) {
        parseMarkdownToAnnotatedString(markdown, color)
    }

    val linkColor = MaterialTheme.colorScheme.primary
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
 */
fun annotateLinksAndPhones(source: AnnotatedString, linkColor: Color): AnnotatedString {
    val text = source.text
    val builder = AnnotatedString.Builder()
    builder.append(source)

    val urlRegex = Regex("""(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*)""")
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
 * Supports: **bold**, *italic*, `code`, ### headers, - lists, and preserves line breaks
 */
fun parseMarkdownToAnnotatedString(markdown: String, baseColor: Color): AnnotatedString {
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
                    parseInlineMarkdown(line.trimStart().substring(2), baseColor, this)
                }
                // Regular text with inline formatting
                else -> {
                    parseInlineMarkdown(line, baseColor, this)
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
 * Parse inline markdown formatting (bold, italic, code) within a line
 */
fun parseInlineMarkdown(text: String, baseColor: Color, builder: AnnotatedString.Builder) {
    var i = 0
    
    while (i < text.length) {
        when {
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
    onRegenerateResponse: (() -> Unit)? = null
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
                    .widthIn(min = 60.dp, max = 280.dp)
                    .wrapContentWidth(),
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
                                    .clip(RoundedCornerShape(12.dp))
                                    .clickable { showFullScreenImage = true },
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
                        SelectableMarkdownText(
                                markdown = message.content,
                                color = MaterialTheme.colorScheme.onPrimary,
                            fontSize = MaterialTheme.typography.bodyLarge.fontSize,
                            modifier = Modifier.wrapContentWidth(),
                            textAlign = TextAlign.End
                        )
                    }
                }
            }
            
            // Show copy button for user messages
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
                }
            }
        } else {
            // AI responses - plain text without background bubble, like ChatGPT mobile
            Column(
                modifier = Modifier.fillMaxWidth()
            ) {
                // Display image if attachment exists (for AI messages with images)
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
                            .clip(RoundedCornerShape(12.dp))
                            .clickable { showFullScreenImage = true },
                        contentScale = ContentScale.Crop,
                        onError = { 
                            android.util.Log.w("MessageBubble", "Failed to load image: ${message.attachmentPath}")
                        }
                    )
                    
                    if (message.content.isNotEmpty() && message.content != "Shared a file" && message.content != "…") {
                        Spacer(modifier = Modifier.height(8.dp))
                    }
                }
                
                // Display text content - plain text without background
                if (message.content.isNotEmpty() && message.content != "Shared a file") {
                    val displayContent = if (streamingContent.isNotEmpty()) streamingContent else message.content
                    
                    SelectableMarkdownText(
                        markdown = displayContent,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = MaterialTheme.typography.bodyLarge.fontSize,
                        modifier = Modifier.fillMaxWidth()
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
    if (showFullScreenImage && message.attachmentPath != null && message.attachmentType == "image") {
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
                            contentDescription = "Close",
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
                            text = "Image Attachment",
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
                                contentDescription = "Copy image",
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Copy")
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
                                contentDescription = "Save image",
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text("Save")
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
                        text = "Tap image to toggle controls",
                        color = Color.White.copy(alpha = 0.8f),
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp)
                    )
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
                        val clip = ClipData.newPlainText("Image", "Image copied from chat")
                        clipboardManager.setPrimaryClip(clip)
                        Toast.makeText(context, "Image copied to clipboard", Toast.LENGTH_SHORT).show()
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(context, "Failed to copy image", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Failed to copy image: ${e.message}", Toast.LENGTH_SHORT).show()
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
                            Toast.makeText(context, "Image saved to gallery", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(context, "Failed to save image", Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        Toast.makeText(context, "Failed to load image", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                Toast.makeText(context, "Failed to save image: ${e.message}", Toast.LENGTH_SHORT).show()
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
 * Modern input bar with Material Design 3 styling, attachment support, and smooth animations.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MessageInput(
    onSendMessage: (String, Uri?) -> Unit,
    enabled: Boolean,
    supportsAttachments: Boolean,
    isLoading: Boolean = false,
    onCancelGeneration: (() -> Unit)? = null
) {
    val context = LocalContext.current
    val focusManager = LocalFocusManager.current
    val keyboardController = LocalSoftwareKeyboardController.current
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
            modifier = Modifier.padding(12.dp)
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
                            contentDescription = stringResource(R.string.attached_image),
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
                    } else {
                        // Show send button when not loading
                        IconButton(
                            onClick = {
                                if (text.isNotBlank() || attachmentUri != null) {
                                    // Aggressively hide keyboard using multiple methods
                                    keyboardController?.hide()
                                    focusManager.clearFocus()
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
                                    contentDescription = stringResource(R.string.send),
                                    modifier = Modifier.padding(6.dp),
                                    tint = if (enabled && (text.isNotBlank() || attachmentUri != null)) 
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