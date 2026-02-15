package com.llmhub.llmhub.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.Dp
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.first
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.window.Dialog
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.runtime.rememberCoroutineScope
import android.net.Uri
import com.llmhub.llmhub.embedding.RagServiceManager
import com.llmhub.llmhub.utils.FileUtils
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
    val coroutineScope = rememberCoroutineScope()
    var showReembedDialogGlobal by remember { mutableStateOf(false) }
    val context = LocalContext.current
    var showThemeDialog by remember { mutableStateOf(false) }
    var showLanguageDialog by remember { mutableStateOf(false) }
    val currentThemeMode by themeViewModel.themeMode.collectAsState()
    val webSearchEnabled by themeViewModel.webSearchEnabled.collectAsState()
    val embeddingEnabled by themeViewModel.embeddingEnabled.collectAsState()
    val memoryEnabled by themeViewModel.memoryEnabled.collectAsState()
    val selectedEmbeddingModel by themeViewModel.selectedEmbeddingModel.collectAsState()
    val currentLanguage by themeViewModel.appLanguage.collectAsState()
    val autoReadoutEnabled by themeViewModel.autoReadoutEnabled.collectAsState()
    
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
                SettingsSection(title = "Safety & Guardrails") {
                    val context = androidx.compose.ui.platform.LocalContext.current
                    val kidModeManager = remember { com.llmhub.llmhub.utils.KidModeManager(context) }
                    val isKidModeEnabled by kidModeManager.isKidModeEnabled.collectAsState()
                    var showKidModeSetPin by remember { mutableStateOf(false) }
                    var showKidModeVerifyPin by remember { mutableStateOf(false) }

                    if (showKidModeSetPin) {
                        com.llmhub.llmhub.ui.components.KidModeSetPinDialog(
                            onConfirm = { pin ->
                                kidModeManager.enableKidMode(pin)
                                showKidModeSetPin = false
                            },
                            onDismiss = { showKidModeSetPin = false }
                        )
                    }

                    if (showKidModeVerifyPin) {
                        com.llmhub.llmhub.ui.components.KidModeVerifyPinDialog(
                            onConfirm = { pin ->
                                if (kidModeManager.disableKidMode(pin)) {
                                    showKidModeVerifyPin = false
                                } else {
                                    android.widget.Toast.makeText(context, "Incorrect PIN", android.widget.Toast.LENGTH_SHORT).show()
                                }
                            },
                            onDismiss = { showKidModeVerifyPin = false }
                        )
                    }

                    SettingsItem(
                        icon = if (isKidModeEnabled) Icons.Default.ChildCare else Icons.Outlined.ChildCare,
                        title = "Kid Mode",
                        subtitle = if (isKidModeEnabled) "Enabled - Safety guardrails active" else "Disabled",
                        onClick = {
                            if (isKidModeEnabled) {
                                showKidModeVerifyPin = true
                            } else {
                                showKidModeSetPin = true
                            }
                        }
                    )
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

                    // Memory (global context) toggle and manager
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.Storage,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(24.dp)
                        )

                        Spacer(modifier = Modifier.width(16.dp))

                        Column(
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(
                                text = stringResource(R.string.memory),
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                // Show memory description based on the memoryEnabled flag (was mistakenly using embeddingEnabled)
                                text = if (memoryEnabled) stringResource(R.string.memory_description_enabled) else stringResource(R.string.memory_requires_rag),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        Switch(
                            checked = memoryEnabled,
                            onCheckedChange = { enabled ->
                                // Only allow enabling if embeddings/RAG are enabled AND an embedding model is selected
                                if (enabled && (!embeddingEnabled || selectedEmbeddingModel.isNullOrBlank())) {
                                    // Ignore — UI indicates why toggle is disabled
                                    return@Switch
                                }
                                themeViewModel.setMemoryEnabled(enabled)

                                if (enabled) {
                                    // If user re-enables memory after changing embedding model, show re-embedding
                                    coroutineScope.launch {
                                        try {
                                            val prefs = com.llmhub.llmhub.data.ThemePreferences(context)
                                            val embeddingsOn = prefs.embeddingEnabled.first()
                                            val currentModel = prefs.selectedEmbeddingModel.first()
                                            if (!embeddingsOn || currentModel.isNullOrBlank()) return@launch

                                            val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                            val hasAnyMemory = withContext(kotlinx.coroutines.Dispatchers.IO) {
                                                db.memoryDao().getAllMemory().first().isNotEmpty()
                                            }
                                            if (!hasAnyMemory) return@launch

                                            val needsReembed = withContext(kotlinx.coroutines.Dispatchers.IO) {
                                                val chunks = db.memoryDao().getAllChunks()
                                                // No chunks (docs exist but not embedded) or any chunk model differs
                                                chunks.isEmpty() || chunks.any { it.embeddingModel != currentModel }
                                            }
                                            if (!needsReembed) return@launch

                                            showReembedDialogGlobal = true

                                            withContext(kotlinx.coroutines.Dispatchers.IO) {
                                                val ragManager = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(context)
                                                try {
                                                    ragManager.clearGlobalDocuments()
                                                    val docs = db.memoryDao().getAllMemory().first()
                                                    docs.forEach { doc ->
                                                        try { db.memoryDao().update(doc.copy(status = "PENDING", chunkCount = 0)) } catch (_: Exception) { }
                                                    }
                                                    try { db.memoryDao().deleteAllChunks() } catch (_: Exception) { }

                                                    val processor = com.llmhub.llmhub.data.MemoryProcessor(context, db)
                                                    processor.processPending()

                                                    try { withTimeoutOrNull(5_000) { com.llmhub.llmhub.data.MemoryProcessor.processing.first { it } } } catch (_: Exception) { }
                                                    com.llmhub.llmhub.data.MemoryProcessor.processing.first { running -> !running }
                                                } catch (e: Exception) {
                                                    android.util.Log.w("SettingsScreen", "Re-embedding after memory enable failed: ${e.message}")
                                                } finally {
                                                    withContext(kotlinx.coroutines.Dispatchers.Main) {
                                                        showReembedDialogGlobal = false
                                                        android.widget.Toast.makeText(context, context.getString(com.llmhub.llmhub.R.string.reembedding_memory_done), android.widget.Toast.LENGTH_SHORT).show()
                                                    }
                                                }
                                            }
                                        } catch (e: Exception) {
                                            android.util.Log.w("SettingsScreen", "Failed to check/trigger re-embedding on memory enable: ${e.message}")
                                        }
                                    }
                                }
                            },
                            enabled = embeddingEnabled && !selectedEmbeddingModel.isNullOrBlank()
                        )
                    }

                    // Auto Readout toggle
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.VolumeUp,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(24.dp)
                        )

                        Spacer(modifier = Modifier.width(16.dp))

                        Column(
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(
                                text = stringResource(R.string.auto_readout),
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Text(
                                text = stringResource(R.string.auto_readout_description),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        Switch(
                            checked = autoReadoutEnabled,
                            onCheckedChange = { enabled ->
                                themeViewModel.setAutoReadoutEnabled(enabled)
                            }
                        )
                    }

                    // Memory manager: allow paste/upload when memory is enabled
                    var showMemoryDialog by remember { mutableStateOf(false) }
                    if (memoryEnabled) {
                        SettingsItem(
                            icon = Icons.Default.UploadFile,
                            title = stringResource(R.string.manage_memory),
                            subtitle = stringResource(R.string.manage_memory_subtitle),
                            onClick = { showMemoryDialog = true }
                        )
                        if (showMemoryDialog) {
                            val context = LocalContext.current
                            // capture keyboard controller in composable scope so coroutine code can call hide()
                            val keyboardController = LocalSoftwareKeyboardController.current
                            val ragServiceManager = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(context)
                            val coroutineScope = rememberCoroutineScope()
                            var pasteText by remember { mutableStateOf(TextFieldValue("")) }
                            // Normalize old localized metadata values to tokens (uploaded/pasted) so rendering is locale-aware.
                            LaunchedEffect(Unit) {
                                val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                val list = db.memoryDao().getAllMemory().first()
                                list.forEach { doc ->
                                    val lower = doc.metadata.lowercase()
                                    when {
                                        // direct resource matches (already localized for this locale)
                                        doc.metadata == context.getString(R.string.global_memory_uploaded_by) || lower == context.getString(R.string.global_memory_uploaded_by).lowercase() -> {
                                            db.memoryDao().update(doc.copy(metadata = "uploaded"))
                                        }
                                        doc.metadata == context.getString(R.string.global_memory_pasted_by) || lower == context.getString(R.string.global_memory_pasted_by).lowercase() -> {
                                            db.memoryDao().update(doc.copy(metadata = "pasted"))
                                        }
                                        // English / other-language heuristics: look for common substrings used in translations
                                        lower.contains("uploaded") || lower.contains("uploaded to") || lower.contains("file saved") || lower.contains("subido") || lower.contains("enviado") || lower.contains("télévers") || lower.contains("enregistr") || lower.contains("hochgeladen") || lower.contains("загруз") -> {
                                            db.memoryDao().update(doc.copy(metadata = "uploaded"))
                                        }
                                        lower.contains("pasted") || lower.contains("pegad") || lower.contains("colad") || lower.contains("coll") || lower.contains("eingef") || lower.contains("встав") -> {
                                            db.memoryDao().update(doc.copy(metadata = "pasted"))
                                        }
                                        // otherwise leave as-is (could be custom text)
                                    }
                                }
                            }
                            // State to hold picked file info for preview-before-insert
                            var pickedFileInfo by remember { mutableStateOf<FileUtils.FileInfo?>(null) }
                            var showPickedFilePreview by remember { mutableStateOf(false) }

                            val documentPickerLauncher = rememberLauncherForActivityResult(
                                contract = ActivityResultContracts.GetContent()
                            ) { uri: Uri? ->
                                uri?.let { selectedUri ->
                                    coroutineScope.launch {
                                        val info = FileUtils.getFileInfo(context, selectedUri)
                                        if (info == null) {
                                            android.widget.Toast.makeText(context, context.getString(R.string.memory_upload_failed), android.widget.Toast.LENGTH_SHORT).show()
                                            return@launch
                                        }
                                        // Save file info and show preview dialog; actual insertion happens when user confirms
                                        pickedFileInfo = info.copy(uri = selectedUri)
                                        showPickedFilePreview = true
                                    }
                                }
                            }

                            // Show file preview dialog when a file was picked
                            if (showPickedFilePreview && pickedFileInfo != null) {
                                com.llmhub.llmhub.components.FilePreviewDialog(
                                    fileInfo = pickedFileInfo!!,
                                    onDismiss = {
                                        showPickedFilePreview = false
                                        pickedFileInfo = null
                                    },
                                    onRemove = null,
                                    confirmAction = {
                                        // Insert the picked file into memory and process embeddings
                                        coroutineScope.launch {
                                            try {
                                                val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                                val info = pickedFileInfo!!
                                                
                                                // Check if file type is supported for memory
                                                if (!FileUtils.isFileTypeSupportedForMemory(info.type)) {
                                                    android.widget.Toast.makeText(context, context.getString(R.string.memory_upload_unsupported_type), android.widget.Toast.LENGTH_SHORT).show()
                                                    return@launch
                                                }
                                                
                                                val content = FileUtils.extractTextContent(context, info.uri, info.type)
                                                if (content != null) {
                                                    val id = "mem_${System.currentTimeMillis()}"
                                                    val doc = com.llmhub.llmhub.data.MemoryDocument(
                                                        id = id,
                                                        fileName = info.name,
                                                        content = content,
                                                        metadata = "uploaded",
                                                        createdAt = System.currentTimeMillis(),
                                                        status = "PENDING",
                                                        chunkCount = 0
                                                    )
                                                    db.memoryDao().insert(doc)
                                                    val processor = com.llmhub.llmhub.data.MemoryProcessor(context, db)
                                                    processor.processPending()
                                                    keyboardController?.hide()
                                                    android.widget.Toast.makeText(context, context.getString(R.string.memory_upload_success), android.widget.Toast.LENGTH_SHORT).show()
                                                } else {
                                                    android.widget.Toast.makeText(context, context.getString(R.string.memory_upload_failed), android.widget.Toast.LENGTH_SHORT).show()
                                                }
                                            } catch (e: Exception) {
                                                android.util.Log.w("SettingsScreen", "Failed inserting uploaded memory: ${e.message}")
                                                android.widget.Toast.makeText(context, context.getString(R.string.memory_upload_failed), android.widget.Toast.LENGTH_SHORT).show()
                                            } finally {
                                                // close preview
                                                showPickedFilePreview = false
                                                pickedFileInfo = null
                                            }
                                        }
                                    },
                                    confirmText = context.getString(R.string.save_to_memory)
                                )
                            }


                            // collect saved memories from DB (used below for list + processing state)
                            val memoryFlow = remember { com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context).memoryDao().getAllMemory() }
                            val memoryList by memoryFlow.collectAsState(initial = emptyList())

                            // Use a Dialog with a Surface so we can fully control sizing and make
                            // the list scroll when vertical space is constrained (landscape/tablet)
                            val configuration = LocalConfiguration.current
                            val screenHeightDp = configuration.screenHeightDp.dp
                            val isLandscape = configuration.screenWidthDp > configuration.screenHeightDp
                            val dialogMaxHeight = if (isLandscape) {
                                (screenHeightDp * 0.75f).coerceAtMost(500.dp)
                            } else {
                                if (screenHeightDp * 0.8f > 640.dp) 640.dp else screenHeightDp * 0.8f
                            }

                            Dialog(onDismissRequest = { showMemoryDialog = false }) {
                                Surface(
                                    shape = MaterialTheme.shapes.large,
                                    tonalElevation = 8.dp,
                                    color = MaterialTheme.colorScheme.surfaceContainer,
                                    modifier = Modifier
                                        .fillMaxWidth(0.92f)
                                        .heightIn(max = dialogMaxHeight)
                                ) {
                                    Column(
                                                modifier = Modifier
                                                    .padding(16.dp)
                                                    // Always allow vertical scrolling when space is constrained.
                                                    .verticalScroll(rememberScrollState())
                                            ) {
                                        Text(stringResource(R.string.manage_memory), style = MaterialTheme.typography.headlineSmall)
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Text(stringResource(R.string.paste_or_upload_to_memory))
                                        Spacer(modifier = Modifier.height(8.dp))
                                        OutlinedTextField(
                                            value = pasteText,
                                            onValueChange = { pasteText = it },
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .height(120.dp),
                                            placeholder = { Text(stringResource(R.string.paste_memory_placeholder)) }
                                        )
                                        Spacer(modifier = Modifier.height(8.dp))

                                        // Buttons: Upload / Save
                                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                                            val buttonShape = MaterialTheme.shapes.large
                                            val buttonModifier = Modifier
                                                .weight(1f)
                                                .height(48.dp)

                                            Button(
                                                onClick = {
                                                    coroutineScope.launch {
                                                        documentPickerLauncher.launch("*/*")
                                                    }
                                                },
                                                modifier = buttonModifier,
                                                enabled = embeddingEnabled && !selectedEmbeddingModel.isNullOrBlank(),
                                                shape = buttonShape
                                            ) {
                                                Text(stringResource(R.string.upload_file))
                                            }

                                            Spacer(modifier = Modifier.width(8.dp))

                                            Button(
                                                onClick = {
                                                    if (pasteText.text.isNotBlank()) {
                                                        coroutineScope.launch {
                                                            val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                                            val id = "mem_${System.currentTimeMillis()}"
                                                            val doc = com.llmhub.llmhub.data.MemoryDocument(
                                                                id = id,
                                                                fileName = "pasted_memory_${System.currentTimeMillis()}.txt",
                                                                content = pasteText.text,
                                                                metadata = "pasted",
                                                                createdAt = System.currentTimeMillis(),
                                                                status = "PENDING",
                                                                chunkCount = 0
                                                            )
                                                            db.memoryDao().insert(doc)
                                                            val processor = com.llmhub.llmhub.data.MemoryProcessor(context, db)
                                                            processor.processPending()
                                                            pasteText = TextFieldValue("")
                                                            keyboardController?.hide()
                                                            android.widget.Toast.makeText(context, context.getString(R.string.memory_save_success), android.widget.Toast.LENGTH_SHORT).show()
                                                        }
                                                    } else {
                                                        android.widget.Toast.makeText(context, context.getString(R.string.paste_memory_placeholder), android.widget.Toast.LENGTH_SHORT).show()
                                                    }
                                                },
                                                modifier = buttonModifier,
                                                enabled = embeddingEnabled && !selectedEmbeddingModel.isNullOrBlank(),
                                                shape = buttonShape
                                            ) {
                                                Text(stringResource(R.string.save_to_memory))
                                            }
                                        }

                                        Spacer(modifier = Modifier.height(8.dp))
                                        
                                        // Import Chat History button
                                        var showChatImportDialog by remember { mutableStateOf(false) }
                                        Button(
                                            onClick = { showChatImportDialog = true },
                                            modifier = Modifier.fillMaxWidth().height(48.dp),
                                            enabled = embeddingEnabled && !selectedEmbeddingModel.isNullOrBlank(),
                                            shape = MaterialTheme.shapes.large
                                        ) {
                                            Text(stringResource(R.string.import_chat_history))
                                        }
                                        
                                        // Chat Import Dialog
                                        if (showChatImportDialog) {
                                            ChatImportDialog(
                                                onDismiss = { showChatImportDialog = false },
                                                onImport = { selectedChats ->
                                                    coroutineScope.launch {
                                                        val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                                        selectedChats.forEach { chat ->
                                                            val messages = db.messageDao().getMessagesForChatSync(chat.id)
                                                            val chatText = messages.joinToString("\n\n") { msg ->
                                                                val role = if (msg.isFromUser) "User" else "Assistant"
                                                                "$role: ${msg.content}"
                                                            }
                                                            
                                                            if (chatText.isNotBlank()) {
                                                                val id = "mem_chat_${chat.id}_${System.currentTimeMillis()}"
                                                                val doc = com.llmhub.llmhub.data.MemoryDocument(
                                                                    id = id,
                                                                    fileName = "Chat: ${chat.title}",
                                                                    content = chatText,
                                                                    metadata = "chat_import",
                                                                    createdAt = System.currentTimeMillis(),
                                                                    status = "PENDING",
                                                                    chunkCount = 0
                                                                )
                                                                db.memoryDao().insert(doc)
                                                            }
                                                        }
                                                        val processor = com.llmhub.llmhub.data.MemoryProcessor(context, db)
                                                        processor.processPending()
                                                        android.widget.Toast.makeText(
                                                            context,
                                                            context.getString(R.string.chat_imported_to_memory),
                                                            android.widget.Toast.LENGTH_SHORT
                                                        ).show()
                                                        showChatImportDialog = false
                                                    }
                                                }
                                            )
                                        }

                                        Spacer(modifier = Modifier.height(12.dp))

                                        // processing indicator
                                        val processorRunning = remember { com.llmhub.llmhub.data.MemoryProcessor.processing }
                                        val processorRunningState by processorRunning.collectAsState(initial = false)
                                        val hasPendingDocs = memoryList.any { it.status == "PENDING" || it.status == "EMBEDDING_IN_PROGRESS" }
                                        if (processorRunningState || hasPendingDocs) {
                                            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 8.dp)) {
                                                CircularProgressIndicator(modifier = Modifier.size(20.dp))
                                                Spacer(modifier = Modifier.width(8.dp))
                                                Text(stringResource(R.string.memory_processing), style = MaterialTheme.typography.bodySmall)
                                            }
                                        }

                                        Spacer(modifier = Modifier.height(8.dp))

                                        Text(stringResource(R.string.saved_memories), style = MaterialTheme.typography.titleSmall)
                                        Spacer(modifier = Modifier.height(8.dp))

                                        if (memoryList.isEmpty()) {
                                            Text(stringResource(R.string.no_memories), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                        } else {
                                            // Clear All button
                                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                                                TextButton(onClick = {
                                                    coroutineScope.launch {
                                                        val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                                        val ragManager = RagServiceManager.getInstance(context)
                                                        try {
                                                            db.memoryDao().deleteAll()
                                                            db.memoryDao().deleteAllChunks()
                                                            ragManager.clearGlobalDocuments()
                                                            android.widget.Toast.makeText(context, context.getString(R.string.memory_cleared), android.widget.Toast.LENGTH_SHORT).show()
                                                        } catch (e: Exception) {
                                                            android.util.Log.w("SettingsScreen", "Failed to clear memories: ${e.message}")
                                                        }
                                                    }
                                                }) {
                                                    Text(stringResource(R.string.clear_all))
                                                }
                                            }

                                            // Memory items list: allow it to size naturally inside the
                                            // dialog's scrollable column so the entire dialog can scroll
                                            // on small heights (e.g., landscape). Use a max height
                                            // to keep the list usable.
                                            LazyColumn(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .heightIn(max = 220.dp),
                                                verticalArrangement = Arrangement.spacedBy(6.dp)
                                            ) {
                                                items(memoryList) { mem ->
                                                    var showEditDialog by remember { mutableStateOf(false) }
                                                    
                                                    Row(
                                                        verticalAlignment = Alignment.CenterVertically,
                                                        modifier = Modifier
                                                            .fillMaxWidth()
                                                            .padding(vertical = 6.dp)
                                                    ) {
                                                        Column(modifier = Modifier.weight(1f)) {
                                                            val displayTitle = if (mem.metadata == "pasted") {
                                                                mem.content.trim().replace("\n", " ").take(120)
                                                            } else mem.fileName
                                                            Text(displayTitle, style = MaterialTheme.typography.bodyMedium)
                                                            val metaLabel = when (mem.metadata) {
                                                                "uploaded" -> context.getString(R.string.global_memory_uploaded_by)
                                                                "pasted" -> context.getString(R.string.global_memory_pasted_by)
                                                                "chat_import" -> context.getString(R.string.chat_imported_to_memory)
                                                                else -> mem.metadata
                                                            }
                                                            Text(metaLabel + " • " + android.text.format.DateFormat.getDateFormat(context).format(mem.createdAt), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                                        }
                                                        
                                                        // Edit button (only for pasted memories)
                                                        if (mem.metadata == "pasted") {
                                                            IconButton(onClick = { showEditDialog = true }) {
                                                                Icon(Icons.Default.Edit, contentDescription = "Edit")
                                                            }
                                                        }
                                                        
                                                        // Delete button
                                                        IconButton(onClick = {
                                                            coroutineScope.launch {
                                                                val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                                                val ragManager = RagServiceManager.getInstance(context)
                                                                try {
                                                                    db.memoryDao().delete(mem)
                                                                    db.memoryDao().deleteChunksForDoc(mem.id)
                                                                    ragManager.removeGlobalDocumentChunks(mem.id)
                                                                    val remaining = db.memoryDao().getAllChunks()
                                                                    if (remaining.isNotEmpty()) {
                                                                        ragManager.restoreGlobalDocumentsFromChunks(remaining)
                                                                    }
                                                                } catch (e: Exception) {
                                                                    android.util.Log.w("SettingsScreen", "Failed to delete memory ${mem.id}: ${e.message}")
                                                                }
                                                            }
                                                        }) {
                                                            Icon(Icons.Default.Delete, contentDescription = stringResource(R.string.delete))
                                                        }
                                                    }
                                                    
                                                    // Edit Dialog
                                                    if (showEditDialog) {
                                                        EditMemoryDialog(
                                                            memory = mem,
                                                            onDismiss = { showEditDialog = false },
                                                            onSave = { updatedContent ->
                                                                coroutineScope.launch {
                                                                    val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                                                    val ragManager = RagServiceManager.getInstance(context)
                                                                    try {
                                                                        // Update content
                                                                        val updatedMem = mem.copy(
                                                                            content = updatedContent,
                                                                            status = "PENDING"
                                                                        )
                                                                        db.memoryDao().update(updatedMem)
                                                                        
                                                                        // Delete old chunks and re-embed
                                                                        db.memoryDao().deleteChunksForDoc(mem.id)
                                                                        ragManager.removeGlobalDocumentChunks(mem.id)
                                                                        
                                                                        // Trigger re-embedding
                                                                        val processor = com.llmhub.llmhub.data.MemoryProcessor(context, db)
                                                                        processor.processPending()
                                                                        
                                                                        android.widget.Toast.makeText(
                                                                            context,
                                                                            "Memory updated and re-embedded",
                                                                            android.widget.Toast.LENGTH_SHORT
                                                                        ).show()
                                                                        showEditDialog = false
                                                                    } catch (e: Exception) {
                                                                        android.util.Log.e("SettingsScreen", "Failed to update memory: ${e.message}")
                                                                        android.widget.Toast.makeText(
                                                                            context,
                                                                            "Failed to update memory",
                                                                            android.widget.Toast.LENGTH_SHORT
                                                                        ).show()
                                                                    }
                                                                }
                                                            }
                                                        )
                                                    }
                                                }
                                            }
                                        }

                                        Spacer(modifier = Modifier.height(12.dp))

                                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                                            TextButton(onClick = { showMemoryDialog = false }) {
                                                Text(stringResource(R.string.done))
                                            }
                                        }
                                    }
                                }
                            }

                            // (replace-confirmation removed to allow multiple memory entries)
                        }
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
        val configuration = LocalConfiguration.current
        val isLandscapeDialog = configuration.screenWidthDp > configuration.screenHeightDp
        AlertDialog(
            onDismissRequest = { showThemeDialog = false },
            title = { Text(stringResource(R.string.choose_theme)) },
            text = {
                // Constrain the height and allow scrolling in landscape
                Column(
                    modifier = Modifier
                        .verticalScroll(rememberScrollState())
                        .heightIn(max = if (isLandscapeDialog) (configuration.screenHeightDp.dp * 0.6f) else Dp.Unspecified)
                ) {
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
                Column(
                    modifier = Modifier.verticalScroll(rememberScrollState())
                ) {
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

    // Global re-embedding modal dialog when triggered by memory re-enable path
    if (showReembedDialogGlobal) {
        Dialog(onDismissRequest = { /* block dismiss while running */ }) {
            Surface(
                shape = MaterialTheme.shapes.large,
                tonalElevation = 8.dp,
                color = MaterialTheme.colorScheme.surfaceContainer,
                modifier = Modifier
                    .fillMaxWidth(0.92f)
                    .wrapContentHeight()
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(text = stringResource(id = com.llmhub.llmhub.R.string.reembedding_memory_in_progress), style = MaterialTheme.typography.bodyLarge)
                    CircularProgressIndicator()
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

@Composable
private fun EmbeddingModelSelector(themeViewModel: ThemeViewModel) {
    val selectedEmbeddingModel by themeViewModel.selectedEmbeddingModel.collectAsState()
    var showEmbeddingModelDialog by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    var showReembedDialog by remember { mutableStateOf(false) }
    
    // Function to handle embedding model change
    fun handleEmbeddingModelChange(newModel: String?) {
        themeViewModel.setSelectedEmbeddingModel(newModel)
        showEmbeddingModelDialog = false

        // Kick off re-embedding immediately when switching between non-null models and memory is enabled
        coroutineScope.launch {
            try {
                val prefs = com.llmhub.llmhub.data.ThemePreferences(context)
                val embeddingsOn = prefs.embeddingEnabled.first()
                val memoryOn = prefs.memoryEnabled.first()
                if (!embeddingsOn || !memoryOn || newModel.isNullOrBlank()) return@launch

                val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                val hasAnyMemory = withContext(kotlinx.coroutines.Dispatchers.IO) {
                    db.memoryDao().getAllMemory().first().isNotEmpty()
                }
                if (!hasAnyMemory) return@launch

                showReembedDialog = true

                withContext(kotlinx.coroutines.Dispatchers.IO) {
                    val ragManager = com.llmhub.llmhub.embedding.RagServiceManager.getInstance(context)
                    try {
                        // Clear in-memory global docs
                        ragManager.clearGlobalDocuments()

                        // Mark docs pending and clear chunks
                        val docs = db.memoryDao().getAllMemory().first()
                        docs.forEach { doc ->
                            try { db.memoryDao().update(doc.copy(status = "PENDING", chunkCount = 0)) } catch (_: Exception) { }
                        }
                        try { db.memoryDao().deleteAllChunks() } catch (_: Exception) { }

                        // Start processing
                        val processor = com.llmhub.llmhub.data.MemoryProcessor(context, db)
                        processor.processPending()

                        // Wait until processing starts then completes
                        try {
                            withTimeoutOrNull(5_000) { com.llmhub.llmhub.data.MemoryProcessor.processing.first { it } }
                        } catch (_: Exception) { }
                        com.llmhub.llmhub.data.MemoryProcessor.processing.first { running -> !running }
                    } catch (e: Exception) {
                        android.util.Log.w("SettingsScreen", "Re-embedding from settings failed: ${e.message}")
                    } finally {
                        // Switch back to main for UI updates
                        showReembedDialog = false
                        android.os.Handler(android.os.Looper.getMainLooper()).post {
                            android.widget.Toast.makeText(context, context.getString(com.llmhub.llmhub.R.string.reembedding_memory_done), android.widget.Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            } catch (e: Exception) {
                android.util.Log.w("SettingsScreen", "Failed to trigger re-embedding: ${e.message}")
            }
        }
    }
    
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
                                    handleEmbeddingModelChange(model)
                                }
                                .padding(vertical = 12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = selectedEmbeddingModel == model,
                                onClick = {
                                    handleEmbeddingModelChange(model)
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
                                    handleEmbeddingModelChange(null)
                                }
                                .padding(vertical = 12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = selectedEmbeddingModel == null,
                                onClick = {
                                    handleEmbeddingModelChange(null)
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

    // Re-embedding modal dialog (blocking dismiss while running)
    if (showReembedDialog) {
        androidx.compose.ui.window.Dialog(onDismissRequest = { /* block dismiss while running */ }) {
            Surface(
                shape = MaterialTheme.shapes.large,
                tonalElevation = 6.dp,
                color = MaterialTheme.colorScheme.surfaceContainer,
                modifier = Modifier
                    .fillMaxWidth(0.9f)
                    .wrapContentHeight()
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(text = stringResource(id = com.llmhub.llmhub.R.string.reembedding_memory_in_progress), style = MaterialTheme.typography.bodyLarge)
                    CircularProgressIndicator()
                }
            }
        }
    }
}

@Composable
fun ChatImportDialog(
    onDismiss: () -> Unit,
    onImport: (List<com.llmhub.llmhub.data.ChatEntity>) -> Unit
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val db = remember { com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context) }
    val allChats by db.chatDao().getNonEmptyChats().collectAsState(initial = emptyList())
    val selectedChats = remember { mutableStateListOf<com.llmhub.llmhub.data.ChatEntity>() }
    
    androidx.compose.ui.window.Dialog(onDismissRequest = onDismiss) {
        Surface(
            shape = MaterialTheme.shapes.large,
            tonalElevation = 6.dp,
            color = MaterialTheme.colorScheme.surfaceContainer,
            modifier = Modifier
                .fillMaxWidth(0.9f)
                .fillMaxHeight(0.7f)
        ) {
            Column(
                modifier = Modifier.padding(20.dp)
            ) {
                Text(
                    text = stringResource(R.string.import_chat_history),
                    style = MaterialTheme.typography.headlineSmall
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = stringResource(R.string.select_chats_to_import),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(16.dp))
                
                // Chat list
                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(allChats) { chat ->
                        val isSelected = selectedChats.contains(chat)
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    if (isSelected) {
                                        selectedChats.remove(chat)
                                    } else {
                                        selectedChats.add(chat)
                                    }
                                }
                                .padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Checkbox(
                                checked = isSelected,
                                onCheckedChange = {
                                    if (it) {
                                        selectedChats.add(chat)
                                    } else {
                                        selectedChats.remove(chat)
                                    }
                                }
                            )
                            Spacer(modifier = Modifier.width(12.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = chat.title,
                                    style = MaterialTheme.typography.bodyLarge
                                )
                                Text(
                                    text = android.text.format.DateFormat.getDateFormat(context).format(chat.updatedAt),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Action buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    TextButton(
                        onClick = onDismiss,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(stringResource(android.R.string.cancel))
                    }
                    Button(
                        onClick = {
                            if (selectedChats.isNotEmpty()) {
                                onImport(selectedChats.toList())
                            } else {
                                android.widget.Toast.makeText(
                                    context,
                                    context.getString(R.string.no_chats_selected),
                                    android.widget.Toast.LENGTH_SHORT
                                ).show()
                            }
                        },
                        modifier = Modifier.weight(1f),
                        enabled = selectedChats.isNotEmpty()
                    ) {
                        Text(stringResource(R.string.import_selected_chats))
                    }
                }
            }
        }
    }
}

@Composable
fun EditMemoryDialog(
    memory: com.llmhub.llmhub.data.MemoryDocument,
    onDismiss: () -> Unit,
    onSave: (String) -> Unit
) {
    val context = LocalContext.current
    var editedContent by remember { mutableStateOf(memory.content) }
    
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.edit_memory)) },
        text = {
            Column {
                OutlinedTextField(
                    value = editedContent,
                    onValueChange = { editedContent = it },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 200.dp, max = 400.dp),
                    placeholder = { Text(stringResource(R.string.memory_content)) },
                    maxLines = 20
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (editedContent.isNotBlank()) {
                        onSave(editedContent.trim())
                    } else {
                        android.widget.Toast.makeText(
                            context,
                            "Memory content cannot be empty",
                            android.widget.Toast.LENGTH_SHORT
                        ).show()
                    }
                },
                enabled = editedContent.isNotBlank()
            ) {
                Text(stringResource(R.string.save_changes))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(android.R.string.cancel))
            }
        }
    )
} 