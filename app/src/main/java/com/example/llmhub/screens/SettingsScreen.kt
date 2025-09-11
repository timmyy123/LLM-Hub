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
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.first
import androidx.compose.ui.platform.LocalLifecycleOwner
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
    var showThemeDialog by remember { mutableStateOf(false) }
    var showLanguageDialog by remember { mutableStateOf(false) }
    val currentThemeMode by themeViewModel.themeMode.collectAsState()
    val webSearchEnabled by themeViewModel.webSearchEnabled.collectAsState()
    val embeddingEnabled by themeViewModel.embeddingEnabled.collectAsState()
    val memoryEnabled by themeViewModel.memoryEnabled.collectAsState()
    val selectedEmbeddingModel by themeViewModel.selectedEmbeddingModel.collectAsState()
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
                            },
                            enabled = embeddingEnabled && !selectedEmbeddingModel.isNullOrBlank()
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
                            val documentPickerLauncher = rememberLauncherForActivityResult(
                                contract = ActivityResultContracts.GetContent()
                            ) { uri: Uri? ->
                                uri?.let { selectedUri ->
                                    coroutineScope.launch {
                                        // Enforce single memory: clear existing memory rows/chunks and in-memory RAG before adding new
                                        val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                        val ragManager = RagServiceManager.getInstance(context)
                                        try {
                                            val existing = db.memoryDao().getAllMemory().first()
                                            existing.forEach { ex ->
                                                db.memoryDao().delete(ex)
                                                db.memoryDao().deleteChunksForDoc(ex.id)
                                            }
                                            ragManager.clearGlobalDocuments()
                                        } catch (e: Exception) {
                                            android.util.Log.w("SettingsScreen", "Failed clearing existing memories: ${e.message}")
                                        }
                                        // Persist the raw content immediately and mark PENDING; processing will run in background
                                        val info = FileUtils.getFileInfo(context, selectedUri)
                                        val content = info?.let { FileUtils.extractTextContent(context, selectedUri, it.type) }
                                        if (content != null && info != null) {
                                            val id = "mem_${System.currentTimeMillis()}"
                                            val doc = com.llmhub.llmhub.data.MemoryDocument(
                                                id = id,
                                                fileName = info.name,
                                                content = content,
                                                // store a non-localized token so UI can render the current locale at display time
                                                metadata = "uploaded",
                                                createdAt = System.currentTimeMillis(),
                                                status = "PENDING",
                                                chunkCount = 0
                                            )
                                            db.memoryDao().insert(doc)
                                            // Kick off processing (will re-embed and repopulate RAG)
                                            val processor = com.llmhub.llmhub.data.MemoryProcessor(context, db)
                                            processor.processPending()
                                            // hide keyboard if visible
                                            keyboardController?.hide()
                                            android.widget.Toast.makeText(context, context.getString(R.string.memory_upload_success), android.widget.Toast.LENGTH_SHORT).show()
                                        } else {
                                            android.widget.Toast.makeText(context, context.getString(R.string.memory_upload_failed), android.widget.Toast.LENGTH_SHORT).show()
                                        }
                                    }
                                }
                            }

                            // UI state for replace confirmation and processing
                            var showReplaceConfirmDialog by remember { mutableStateOf(false) }
                            var pendingUploadAction by remember { mutableStateOf(false) }
                            var isProcessing by remember { mutableStateOf(false) }

                            AlertDialog(
                                onDismissRequest = { showMemoryDialog = false },
                                title = { Text(stringResource(R.string.manage_memory)) },
                                text = {
                                    Column {
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
                                                            // Check for existing memory and confirm replace if present
                                                            coroutineScope.launch {
                                                                val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                                                val existing = db.memoryDao().getAllMemory().first()
                                                                if (existing.isNotEmpty()) {
                                                                    pendingUploadAction = true
                                                                    showReplaceConfirmDialog = true
                                                                } else {
                                                                    documentPickerLauncher.launch("*/*")
                                                                }
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
                                                                    val existing = db.memoryDao().getAllMemory().first()
                                                                    if (existing.isNotEmpty()) {
                                                                        // Ask for confirmation before replacing
                                                                        pendingUploadAction = false
                                                                        showReplaceConfirmDialog = true
                                                                    } else {
                                                                        // No existing memory, insert directly
                                                                        val id = "mem_${System.currentTimeMillis()}"
                                                                        val doc = com.llmhub.llmhub.data.MemoryDocument(
                                                                            id = id,
                                                                            fileName = "pasted_memory_${System.currentTimeMillis()}.txt",
                                                                            content = pasteText.text,
                                                                            // store a non-localized token so UI can render the current locale at display time
                                                                            metadata = "pasted",
                                                                            createdAt = System.currentTimeMillis(),
                                                                            status = "PENDING",
                                                                            chunkCount = 0
                                                                        )
                                                                        db.memoryDao().insert(doc)
                                                                        val processor = com.llmhub.llmhub.data.MemoryProcessor(context, db)
                                                                        isProcessing = true
                                                                        processor.processPending()
                                                                        pasteText = TextFieldValue("")
                                                                        // hide keyboard after save
                                                                        keyboardController?.hide()
                                                                        android.widget.Toast.makeText(context, context.getString(R.string.memory_save_success), android.widget.Toast.LENGTH_SHORT).show()
                                                                    }
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

                                                Spacer(modifier = Modifier.height(12.dp))

                                                // Show processing indicator while MemoryProcessor is running
                                                if (isProcessing) {
                                                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 8.dp)) {
                                                        CircularProgressIndicator(modifier = Modifier.size(20.dp))
                                                        Spacer(modifier = Modifier.width(8.dp))
                                                        Text(stringResource(R.string.memory_processing), style = MaterialTheme.typography.bodySmall)
                                                    }
                                                }

                                                // List saved memories
                                                Text(stringResource(R.string.saved_memories), style = MaterialTheme.typography.titleSmall)
                                                Spacer(modifier = Modifier.height(8.dp))
                                                // collect saved memories from DB
                                                val memoryFlow = remember { com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context).memoryDao().getAllMemory() }
                                                val memoryList by memoryFlow.collectAsState(initial = emptyList())

                                                if (memoryList.isEmpty()) {
                                                    Text(stringResource(R.string.no_memories), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                                } else {
                                                    // Only allow one memory entry: show the most recent (first) one
                                                    val mem = memoryList.first()
                                                    Column(modifier = Modifier.fillMaxWidth()) {
                                                        Row(
                                                            verticalAlignment = Alignment.CenterVertically,
                                                            modifier = Modifier
                                                                .fillMaxWidth()
                                                                .padding(vertical = 6.dp)
                                                        ) {
                                                            Column(modifier = Modifier.weight(1f)) {
                                                                // For pasted memories, show a short preview of the content instead of the autogenerated filename
                                                                val displayTitle = if (mem.metadata == "pasted") {
                                                                    mem.content.trim().replace("\n", " ").take(60)
                                                                } else mem.fileName
                                                                Text(displayTitle, style = MaterialTheme.typography.bodyMedium)
                                                                // Map stored metadata tokens to localized labels. If metadata is already localized text, show it as-is.
                                                                val metaLabel = when (mem.metadata) {
                                                                    "uploaded" -> context.getString(R.string.global_memory_uploaded_by)
                                                                    "pasted" -> context.getString(R.string.global_memory_pasted_by)
                                                                    else -> {
                                                                        val lowerMeta = mem.metadata.lowercase()
                                                                        when {
                                                                            lowerMeta.contains("uploaded") || lowerMeta.contains("subido") || lowerMeta.contains("télévers") || lowerMeta.contains("enviado") || lowerMeta.contains("hochgeladen") || lowerMeta.contains("загруз") -> context.getString(R.string.global_memory_uploaded_by)
                                                                            lowerMeta.contains("pasted") || lowerMeta.contains("pegad") || lowerMeta.contains("colad") || lowerMeta.contains("coll") || lowerMeta.contains("eingef") || lowerMeta.contains("встав") -> context.getString(R.string.global_memory_pasted_by)
                                                                            mem.metadata.isNotBlank() -> mem.metadata
                                                                            else -> ""
                                                                        }
                                                                    }
                                                                }
                                                                Text(metaLabel + " • " + android.text.format.DateFormat.getDateFormat(context).format(mem.createdAt), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                                            }
                                                            IconButton(onClick = {
                                                                // delete: remove memory row, its persisted chunk embeddings, and refresh the in-memory RAG
                                                                coroutineScope.launch {
                                                                    val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                                                    val ragManager = RagServiceManager.getInstance(context)
                                                                    try {
                                                                        // Delete DB row(s)
                                                                        db.memoryDao().delete(mem)
                                                                        db.memoryDao().deleteChunksForDoc(mem.id)

                                                                        // Remove the in-memory chunks for the deleted doc and reload remaining
                                                                        // chunks from DB to ensure the in-memory index matches persisted state.
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
                                                    }
                                                }
                                            }
                                },
                                confirmButton = {
                                    TextButton(onClick = { showMemoryDialog = false }) {
                                        Text(stringResource(R.string.done))
                                    }
                                }
                            )

                            // Confirmation dialog: replace existing memory
                            if (showReplaceConfirmDialog) {
                                AlertDialog(
                                    onDismissRequest = { showReplaceConfirmDialog = false },
                                    title = { Text(stringResource(R.string.confirm_replace_memory_title)) },
                                    text = { Text(stringResource(R.string.confirm_replace_memory_message)) },
                                    confirmButton = {
                                        TextButton(onClick = {
                                            coroutineScope.launch {
                                                val db = com.llmhub.llmhub.data.LlmHubDatabase.getDatabase(context)
                                                val ragManager = RagServiceManager.getInstance(context)
                                                try {
                                                    // delete existing memory rows/chunks
                                                    val existing = db.memoryDao().getAllMemory().first()
                                                    existing.forEach { ex ->
                                                        db.memoryDao().delete(ex)
                                                        db.memoryDao().deleteChunksForDoc(ex.id)
                                                    }
                                                    ragManager.clearGlobalDocuments()
                                                } catch (e: Exception) {
                                                    android.util.Log.w("SettingsScreen", "Failed clearing existing memories: ${e.message}")
                                                }

                                                // If this was triggered by upload, launch document picker; if paste, insert pasted text
                                                if (pendingUploadAction) {
                                                    // reset flag and open picker
                                                    pendingUploadAction = false
                                                    showReplaceConfirmDialog = false
                                                    documentPickerLauncher.launch("*/*")
                                                } else {
                                                    // perform paste insert: create doc and process
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
                                                    showReplaceConfirmDialog = false
                                                    android.widget.Toast.makeText(context, context.getString(R.string.memory_save_success), android.widget.Toast.LENGTH_SHORT).show()
                                                }
                                            }
                                        }) {
                                            Text(stringResource(R.string.confirm_replace_action))
                                        }
                                    },
                                    dismissButton = {
                                        TextButton(onClick = { showReplaceConfirmDialog = false }) {
                                            Text(stringResource(R.string.cancel))
                                        }
                                    }
                                )
                            }
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
    
    // Function to handle embedding model change
    fun handleEmbeddingModelChange(newModel: String?) {
        themeViewModel.setSelectedEmbeddingModel(newModel)
        showEmbeddingModelDialog = false
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
} 