package com.llmhub.llmhub.screens

import android.provider.OpenableColumns
import android.provider.DocumentsContract
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.GetApp
import androidx.compose.material.icons.filled.ModelTraining
import androidx.compose.material.icons.filled.OpenInBrowser
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Code
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.layout.onSizeChanged
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.R
import com.llmhub.llmhub.components.FeatureModelSettingsSheet
import com.llmhub.llmhub.components.ThinkingAwareResultContent
import com.llmhub.llmhub.viewmodels.CodeLanguage
import com.llmhub.llmhub.viewmodels.VibeChatSessionSummary
import com.llmhub.llmhub.viewmodels.VibeCoderViewModel
import com.llmhub.llmhub.viewmodels.VibeChatMessage
import kotlinx.coroutines.delay

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VibeCoderScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    onNavigateToCanvas: ((String, String) -> Unit)? = null,
    viewModel: VibeCoderViewModel = viewModel()
) {
    val context = LocalContext.current
    val clipboardManager = LocalClipboardManager.current
    var showSettingsSheet by remember { mutableStateOf(false) }
    var chatInput by rememberSaveable { mutableStateOf("") }
    var chatPaneVisible by rememberSaveable { mutableStateOf(true) }
    var chatPaneRatio by rememberSaveable { mutableStateOf(0.36f) }
    var showNewFileDialog by remember { mutableStateOf(false) }
    var newFileNameInput by rememberSaveable { mutableStateOf("") }

    val availableModels by viewModel.availableModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    val selectedNpuDeviceId by viewModel.selectedNpuDeviceId.collectAsState()
    val selectedMaxTokens by viewModel.selectedMaxTokens.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val isProcessing by viewModel.isProcessing.collectAsState()
    val generatedCode by viewModel.generatedCode.collectAsState()
    val chatMessages by viewModel.chatMessages.collectAsState()
    val chatSessions by viewModel.chatSessions.collectAsState()
    val activeChatSessionId by viewModel.activeChatSessionId.collectAsState()
    val editCheckpoints by viewModel.editCheckpoints.collectAsState()
    val codeLanguage by viewModel.codeLanguage.collectAsState()
    val currentFileUri by viewModel.currentFileUri.collectAsState()
    val currentFileName by viewModel.currentFileName.collectAsState()
    val currentFolderUri by viewModel.currentFolderUri.collectAsState()
    val isDirty by viewModel.isDirty.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()

    val snackbarHostState = remember { SnackbarHostState() }
    var folderFiles by remember { mutableStateOf<List<Pair<String, String>>>(emptyList()) }
    var didAutoRestore by remember { mutableStateOf(false) }
    val contextUsage by remember(chatMessages, generatedCode, chatInput, selectedMaxTokens) {
        derivedStateOf {
            val usedChars = chatMessages.sumOf { it.text.length } + generatedCode.length + chatInput.length
            val usedTokens = (usedChars / 4).coerceAtLeast(0)
            (usedTokens.toFloat() / selectedMaxTokens.coerceAtLeast(1).toFloat()).coerceIn(0f, 1f)
        }
    }

    LaunchedEffect(errorMessage) {
        errorMessage?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }

    val openFileLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            try {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )
            } catch (_: Exception) {}

            val content = runCatching {
                context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() } ?: ""
            }.getOrDefault("")

            val fileName = runCatching {
                context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (cursor.moveToFirst() && nameIndex >= 0) cursor.getString(nameIndex) else null
                }
            }.getOrNull() ?: "untitled.txt"

            viewModel.openEditorFile(uri.toString(), fileName, content)
        }
    }

    val openFolderLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { treeUri ->
        if (treeUri != null) {
            try {
                context.contentResolver.takePersistableUriPermission(
                    treeUri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )
            } catch (_: Exception) {}
            viewModel.openFolder(treeUri.toString())
        }
    }

    fun saveToUri(uriString: String?, fileNameFallback: String = "untitled.txt") {
        if (uriString.isNullOrBlank()) {
            viewModel.setError("No file selected. Create/open a file from the workspace folder first.")
            return
        }
        val uri = android.net.Uri.parse(uriString)
        val success = runCatching {
            context.contentResolver.openOutputStream(uri, "wt")?.use { os ->
                os.write(generatedCode.toByteArray())
                os.flush()
            }
        }.isSuccess
        if (success) {
            viewModel.markSaved(uriString, currentFileName ?: fileNameFallback)
        }
    }

    fun loadFileFromUri(uriString: String) {
        val uri = android.net.Uri.parse(uriString)
        val content = runCatching {
            context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() } ?: ""
        }.getOrDefault("")
        val fileName = runCatching {
            context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (cursor.moveToFirst() && nameIndex >= 0) cursor.getString(nameIndex) else null
            }
        }.getOrNull() ?: "untitled.txt"
        viewModel.openEditorFile(uriString, fileName, content)
    }

    fun refreshFolderFiles(folderUriString: String?): List<Pair<String, String>> {
        if (folderUriString.isNullOrBlank()) {
            folderFiles = emptyList()
            return emptyList()
        }
        val treeUri = android.net.Uri.parse(folderUriString)
        val docId = runCatching { DocumentsContract.getTreeDocumentId(treeUri) }.getOrNull()
        if (docId == null) {
            folderFiles = emptyList()
            return emptyList()
        }
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, docId)
        val proj = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE
        )
        val files = mutableListOf<Pair<String, String>>()
        runCatching {
            context.contentResolver.query(childrenUri, proj, null, null, null)?.use { cursor ->
                val idIdx = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
                val nameIdx = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
                val mimeIdx = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_MIME_TYPE)
                while (cursor.moveToNext()) {
                    if (idIdx < 0 || nameIdx < 0 || mimeIdx < 0) continue
                    val childDocId = cursor.getString(idIdx) ?: continue
                    val name = cursor.getString(nameIdx) ?: continue
                    val mime = cursor.getString(mimeIdx) ?: continue
                    if (mime == DocumentsContract.Document.MIME_TYPE_DIR) continue
                    val n = name.lowercase()
                    val isCode = n.endsWith(".py") || n.endsWith(".js") || n.endsWith(".ts") || n.endsWith(".java") ||
                        n.endsWith(".kt") || n.endsWith(".cs") || n.endsWith(".cpp") || n.endsWith(".cc") ||
                        n.endsWith(".cxx") || n.endsWith(".go") || n.endsWith(".rs") || n.endsWith(".html") ||
                        n.endsWith(".htm") || n.endsWith(".css")
                    if (!isCode) continue
                    val childUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, childDocId)
                    files.add(Pair(childUri.toString(), name))
                }
            }
        }
        files.sortBy { it.second.lowercase() }
        folderFiles = files
        return files
    }

    fun createFileInCurrentFolder(fileName: String) {
        val folder = currentFolderUri ?: return
        val treeUri = android.net.Uri.parse(folder)
        val treeDocId = runCatching { DocumentsContract.getTreeDocumentId(treeUri) }.getOrNull() ?: return
        val parentDocUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, treeDocId)
        val mime = when {
            fileName.endsWith(".py", true) -> "text/x-python"
            fileName.endsWith(".js", true) -> "application/javascript"
            fileName.endsWith(".ts", true) -> "application/typescript"
            fileName.endsWith(".html", true) || fileName.endsWith(".htm", true) -> "text/html"
            fileName.endsWith(".css", true) -> "text/css"
            else -> "text/plain"
        }
        val newUri = runCatching {
            DocumentsContract.createDocument(context.contentResolver, parentDocUri, mime, fileName)
        }.getOrNull()
        if (newUri != null) {
            runCatching {
                context.contentResolver.openOutputStream(newUri, "wt")?.use { os ->
                    os.write(ByteArray(0))
                    os.flush()
                }
            }
            loadFileFromUri(newUri.toString())
            refreshFolderFiles(currentFolderUri)
        }
    }

    LaunchedEffect(currentFolderUri) {
        refreshFolderFiles(currentFolderUri)
    }

    LaunchedEffect(currentFileUri, currentFolderUri, generatedCode, isDirty) {
        if (didAutoRestore) return@LaunchedEffect
        didAutoRestore = true
        when {
            !currentFileUri.isNullOrBlank() -> {
                val hasUnsavedDraft = isDirty || generatedCode.isNotBlank()
                if (!hasUnsavedDraft) loadFileFromUri(currentFileUri!!)
            }
            !currentFolderUri.isNullOrBlank() -> refreshFolderFiles(currentFolderUri)
        }
    }

    LaunchedEffect(generatedCode, isDirty, currentFileUri, currentFileName) {
        if (!isDirty) return@LaunchedEffect
        val uri = currentFileUri ?: return@LaunchedEffect
        delay(500)
        val success = runCatching {
            context.contentResolver.openOutputStream(android.net.Uri.parse(uri), "wt")?.use { os ->
                os.write(generatedCode.toByteArray())
                os.flush()
            }
        }.isSuccess
        if (success) {
            viewModel.markSaved(uri, currentFileName ?: "untitled.txt")
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.vibe_coder_title)) },
                navigationIcon = {
                    IconButton(onClick = {
                        viewModel.stopAndUnloadOnExit()
                        onNavigateBack()
                    }) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = stringResource(R.string.feature_settings_title))
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { innerPadding ->
        if (!isModelLoaded) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(innerPadding)
                    .padding(32.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Default.ModelTraining,
                    contentDescription = null,
                    modifier = Modifier.size(80.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = stringResource(
                        if (availableModels.isEmpty()) R.string.download_models_first
                        else R.string.scam_detector_load_model
                    ),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(R.string.scam_detector_load_model_desc),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(32.dp))
                FilledTonalButton(
                    onClick = {
                        if (availableModels.isEmpty()) onNavigateToModels()
                        else showSettingsSheet = true
                    },
                    modifier = Modifier.fillMaxWidth(0.7f)
                ) {
                    Icon(
                        imageVector = if (availableModels.isEmpty()) Icons.Default.GetApp else Icons.Default.Tune,
                        contentDescription = null
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        stringResource(
                            if (availableModels.isEmpty()) R.string.download_models
                            else R.string.feature_settings_title
                        )
                    )
                }
            }
        } else if (currentFolderUri.isNullOrBlank()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .padding(24.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(Icons.Default.FolderOpen, contentDescription = null, modifier = Modifier.size(72.dp))
                Spacer(modifier = Modifier.height(14.dp))
                Text(
                    text = "Select a project folder first",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = "Open a folder to enable multi-file scaffold and in-folder file creation.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(16.dp))
                Button(onClick = { openFolderLauncher.launch(null) }) {
                    Icon(Icons.Default.FolderOpen, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Open Folder")
                }
            }
        } else {
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .padding(12.dp)
            ) {
                val configuration = LocalConfiguration.current
                val isLandscape = configuration.orientation == android.content.res.Configuration.ORIENTATION_LANDSCAPE
                val isWideLayout = maxWidth >= 840.dp || (isLandscape && maxWidth >= 700.dp)

                if (isWideLayout) {
                    var wideRowWidthPx by remember { mutableStateOf(1) }
                    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(
                            modifier = Modifier
                                .fillMaxSize()
                                .onSizeChanged { wideRowWidthPx = it.width.coerceAtLeast(1) },
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            EditorPane(
                                modifier = Modifier.weight(if (chatPaneVisible) 1f - chatPaneRatio else 1f),
                                code = generatedCode,
                                onCodeChange = { viewModel.updateGeneratedCode(it) },
                                isProcessing = isProcessing,
                                codeLanguage = codeLanguage,
                                canDiscardLastAiEdit = editCheckpoints.isNotEmpty(),
                                onDiscardLastAiEdit = { viewModel.revertLastCheckpoint() },
                                currentFileName = currentFileName,
                                hasFileSession = currentFileName != null,
                                isDirty = isDirty,
                                onOpenFile = {},
                                onOpenFolder = { openFolderLauncher.launch(null) },
                                folderFiles = folderFiles,
                                currentFileUri = currentFileUri,
                                onSelectFolderFile = { uri -> loadFileFromUri(uri) },
                                onNewFile = {
                                    showNewFileDialog = true
                                },
                                onSaveFile = { saveToUri(currentFileUri, currentFileName ?: "untitled.py") },
                                onCopy = { clipboardManager.setText(AnnotatedString(generatedCode)) },
                                onPreview = {
                                    onNavigateToCanvas?.invoke(generatedCode, "html")
                                },
                                canPreview = generatedCode.isNotBlank() && codeLanguage == CodeLanguage.HTML
                            )
                            if (chatPaneVisible) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxHeight()
                                        .width(10.dp)
                                        .background(MaterialTheme.colorScheme.outline.copy(alpha = 0.35f), RoundedCornerShape(8.dp))
                                        .pointerInput(wideRowWidthPx) {
                                            detectDragGestures { change, dragAmount ->
                                                change.consume()
                                                val deltaRatio = dragAmount.x / wideRowWidthPx.toFloat()
                                                chatPaneRatio = (chatPaneRatio - deltaRatio).coerceIn(0.20f, 0.60f)
                                            }
                                        }
                                )
                                ChatPane(
                                    modifier = Modifier.weight(chatPaneRatio),
                                    messages = chatMessages,
                                    chatSessions = chatSessions,
                                    activeChatSessionId = activeChatSessionId,
                                    input = chatInput,
                                    onInputChange = { chatInput = it },
                                    isProcessing = isProcessing,
                                    hasFileSession = currentFileName != null,
                                    contextUsage = contextUsage,
                                    contextLabel = "${(contextUsage * 100).toInt()}%",
                                    showContextPercent = true,
                                    showHideButton = true,
                                    onHidePanel = { chatPaneVisible = false },
                                    onNewChat = { viewModel.createNewChatSession() },
                                    onSelectChat = { viewModel.selectChatSession(it) },
                                    onEditPrompt = { id, text -> viewModel.editAndResendFromPrompt(id, text) },
                                    onSend = {
                                        val p = chatInput.trim()
                                        if (p.isNotEmpty() && currentFileName != null) {
                                            chatInput = ""
                                            viewModel.generateCode(p)
                                        }
                                    },
                                    onClearChat = { viewModel.clearChatSession() }
                                )
                            } else {
                                IconButton(onClick = { chatPaneVisible = true }) {
                                    Icon(Icons.Default.Chat, contentDescription = "Show chat")
                                }
                            }
                        }
                    }
                } else {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .imePadding(),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        ChatPane(
                            modifier = Modifier.weight(0.45f),
                            messages = chatMessages,
                            chatSessions = chatSessions,
                            activeChatSessionId = activeChatSessionId,
                            input = chatInput,
                            onInputChange = { chatInput = it },
                            isProcessing = isProcessing,
                            hasFileSession = currentFileName != null,
                            contextUsage = contextUsage,
                            contextLabel = "${(contextUsage * 100).toInt()}%",
                            showContextPercent = true,
                            showHideButton = false,
                            onHidePanel = null,
                            onNewChat = { viewModel.createNewChatSession() },
                            onSelectChat = { viewModel.selectChatSession(it) },
                            onEditPrompt = { id, text -> viewModel.editAndResendFromPrompt(id, text) },
                            onSend = {
                                val p = chatInput.trim()
                                if (p.isNotEmpty() && currentFileName != null) {
                                    chatInput = ""
                                    viewModel.generateCode(p)
                                }
                            },
                            onClearChat = { viewModel.clearChatSession() }
                        )
                        EditorPane(
                            modifier = Modifier.weight(0.55f),
                            code = generatedCode,
                            onCodeChange = { viewModel.updateGeneratedCode(it) },
                            isProcessing = isProcessing,
                            codeLanguage = codeLanguage,
                            canDiscardLastAiEdit = editCheckpoints.isNotEmpty(),
                            onDiscardLastAiEdit = { viewModel.revertLastCheckpoint() },
                            currentFileName = currentFileName,
                            hasFileSession = currentFileName != null,
                            isDirty = isDirty,
                            onOpenFile = {},
                            onOpenFolder = { openFolderLauncher.launch(null) },
                            folderFiles = folderFiles,
                            currentFileUri = currentFileUri,
                            onSelectFolderFile = { uri -> loadFileFromUri(uri) },
                            onNewFile = {
                                showNewFileDialog = true
                            },
                            onSaveFile = { saveToUri(currentFileUri, currentFileName ?: "untitled.py") },
                            onCopy = { clipboardManager.setText(AnnotatedString(generatedCode)) },
                            onPreview = {
                                onNavigateToCanvas?.invoke(generatedCode, "html")
                            },
                            canPreview = generatedCode.isNotBlank() && codeLanguage == CodeLanguage.HTML
                        )
                    }
                }
            }
        }
    }

    if (showSettingsSheet) {
        FeatureModelSettingsSheet(
            availableModels = availableModels,
            initialSelectedModel = selectedModel,
            initialSelectedBackend = selectedBackend,
            initialSelectedNpuDeviceId = selectedNpuDeviceId,
            initialMaxTokens = selectedMaxTokens,
            currentlyLoadedModel = if (isModelLoaded) selectedModel else null,
            isLoadingModel = isLoading,
            onModelSelected = { viewModel.selectModel(it) },
            onBackendSelected = { backend, deviceId -> viewModel.selectBackend(backend, deviceId) },
            onMaxTokensChanged = { viewModel.setMaxTokens(it) },
            onLoadModel = { model, maxTokens, backend, deviceId, nGpuLayers, isThinkingEnabled ->
                viewModel.selectModel(model)
                viewModel.setMaxTokens(maxTokens)
                if (backend != null) viewModel.selectBackend(backend, deviceId)
                viewModel.setNGpuLayers(nGpuLayers)
                viewModel.setEnableThinking(isThinkingEnabled)
                viewModel.loadModel()
            },
            onUnloadModel = { viewModel.unloadModel() },
            onDismiss = { showSettingsSheet = false }
        )
    }

    if (showNewFileDialog) {
        AlertDialog(
            onDismissRequest = { showNewFileDialog = false },
            title = { Text("Create file in workspace") },
            text = {
                OutlinedTextField(
                    value = newFileNameInput,
                    onValueChange = { newFileNameInput = it },
                    singleLine = true,
                    placeholder = { Text("example.py or app.js") },
                    label = { Text("File name with extension") }
                )
            },
            confirmButton = {
                Button(onClick = {
                    val name = newFileNameInput.trim()
                    if (name.isBlank() || !name.contains(".")) {
                        viewModel.setError("Enter a valid file name with extension, e.g. main.py")
                        return@Button
                    }
                    createFileInCurrentFolder(name)
                    newFileNameInput = ""
                    showNewFileDialog = false
                }) {
                    Text("Create")
                }
            },
            dismissButton = {
                Button(onClick = {
                    showNewFileDialog = false
                    newFileNameInput = ""
                }) {
                    Text("Cancel")
                }
            }
        )
    }

}

@Composable
private fun ChatPane(
    modifier: Modifier,
    messages: List<VibeChatMessage>,
    chatSessions: List<VibeChatSessionSummary>,
    activeChatSessionId: String?,
    input: String,
    onInputChange: (String) -> Unit,
    isProcessing: Boolean,
    hasFileSession: Boolean,
    contextUsage: Float,
    contextLabel: String,
    showContextPercent: Boolean,
    showHideButton: Boolean,
    onHidePanel: (() -> Unit)?,
    onNewChat: () -> Unit,
    onSelectChat: (String) -> Unit,
    onEditPrompt: (String, String) -> Unit,
    onSend: () -> Unit,
    onClearChat: () -> Unit
) {
    val chatListState = rememberLazyListState()
    var editingPromptId by remember { mutableStateOf<String?>(null) }
    var editingPromptText by rememberSaveable { mutableStateOf("") }

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            chatListState.animateScrollToItem(messages.lastIndex)
        }
    }

    LaunchedEffect(messages.lastOrNull()?.text, isProcessing) {
        if (messages.isNotEmpty() && isProcessing) {
            chatListState.animateScrollToItem(messages.lastIndex)
        }
    }

    Card(
        modifier = modifier.fillMaxSize(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .imePadding()
                .padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "AI Chat",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.weight(1f))
                Box(contentAlignment = Alignment.Center, modifier = Modifier.size(28.dp)) {
                    CircularProgressIndicator(
                        progress = contextUsage,
                        modifier = Modifier.size(24.dp),
                        strokeWidth = 2.dp
                    )
                    if (showContextPercent) {
                        Text(
                            text = if (contextUsage < 0.995f) contextLabel else "!",
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                }
                Spacer(modifier = Modifier.width(6.dp))
                if (showHideButton && onHidePanel != null) {
                    IconButton(onClick = onHidePanel) {
                        Icon(Icons.Default.Code, contentDescription = "Hide chat")
                    }
                }
                IconButton(onClick = onClearChat, enabled = messages.isNotEmpty() && !isProcessing) {
                    Icon(Icons.Default.Clear, contentDescription = "Clear chat")
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                IconButton(onClick = onNewChat, enabled = !isProcessing) {
                    Icon(Icons.Default.Add, contentDescription = "New chat")
                }
                LazyRow(
                    modifier = Modifier.weight(1f),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    items(chatSessions) { s ->
                        val selected = s.id == activeChatSessionId
                        Button(
                            onClick = { onSelectChat(s.id) },
                            modifier = Modifier.height(30.dp),
                            shape = RoundedCornerShape(16.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                            )
                        ) {
                            Text(s.title, style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                state = chatListState,
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(messages) { msg ->
                    val isUser = msg.role == "user"
                    Surface(
                        shape = RoundedCornerShape(10.dp),
                        color = if (isUser) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.secondaryContainer,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(10.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = if (isUser) "You" else "AI",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Spacer(modifier = Modifier.weight(1f))
                                if (isUser) {
                                    IconButton(
                                        onClick = {
                                            editingPromptId = msg.id
                                            editingPromptText = msg.text
                                        },
                                        modifier = Modifier.size(20.dp)
                                    ) {
                                        Icon(Icons.Default.Edit, contentDescription = "Edit and resend prompt", modifier = Modifier.size(14.dp))
                                    }
                                }
                            }
                            Spacer(modifier = Modifier.height(4.dp))
                            if (isUser && editingPromptId == msg.id) {
                                OutlinedTextField(
                                    value = editingPromptText,
                                    onValueChange = { editingPromptText = it },
                                    modifier = Modifier.fillMaxWidth(),
                                    maxLines = 5,
                                    singleLine = false
                                )
                                Spacer(modifier = Modifier.height(6.dp))
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Button(
                                        onClick = {
                                            editingPromptId = null
                                            editingPromptText = ""
                                        },
                                        modifier = Modifier.weight(1f),
                                        colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
                                    ) { Text("Cancel") }
                                    Button(
                                        onClick = {
                                            onEditPrompt(msg.id, editingPromptText)
                                            editingPromptId = null
                                            editingPromptText = ""
                                        },
                                        enabled = editingPromptText.isNotBlank() && !isProcessing,
                                        modifier = Modifier.weight(1f)
                                    ) { Text("Resend") }
                                }
                            } else if (isUser) {
                                Text(text = msg.text, style = MaterialTheme.typography.bodyMedium)
                            } else {
                                ThinkingAwareResultContent(
                                    content = msg.text,
                                    modifier = Modifier.fillMaxWidth(),
                                    useMarkdownForAnswer = false
                                )
                            }
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = input,
                onValueChange = onInputChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = {
                    Text(
                        if (hasFileSession) "Ask AI to edit this file..."
                        else "Create/Open a file first (.py, .js, .ts, ...)"
                    )
                },
                maxLines = 5,
                enabled = !isProcessing && hasFileSession,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { onSend() }),
                trailingIcon = {
                    IconButton(onClick = onSend, enabled = input.isNotBlank() && !isProcessing && hasFileSession) {
                        if (isProcessing) {
                            CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                        } else {
                            Icon(Icons.Default.Send, contentDescription = "Send")
                        }
                    }
                }
            )
        }
    }
}

@Composable
private fun EditorPane(
    modifier: Modifier,
    code: String,
    onCodeChange: (String) -> Unit,
    isProcessing: Boolean,
    codeLanguage: CodeLanguage,
    canDiscardLastAiEdit: Boolean,
    onDiscardLastAiEdit: () -> Unit,
    currentFileName: String?,
    hasFileSession: Boolean,
    isDirty: Boolean,
    onOpenFile: () -> Unit,
    onOpenFolder: () -> Unit,
    folderFiles: List<Pair<String, String>>,
    currentFileUri: String?,
    onSelectFolderFile: (String) -> Unit,
    onNewFile: () -> Unit,
    onSaveFile: () -> Unit,
    onCopy: () -> Unit,
    onPreview: () -> Unit,
    canPreview: Boolean
) {
    val scrollState = rememberScrollState()
    val shownCode = code
    Card(
        modifier = modifier.fillMaxSize(),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = currentFileName ?: "Code Editor",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                if (isDirty) {
                    Text(
                        text = " *",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
                Spacer(modifier = Modifier.width(8.dp))
                if (codeLanguage != CodeLanguage.UNKNOWN) {
                    Surface(
                        shape = RoundedCornerShape(16.dp),
                        color = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.2f)
                    ) {
                        Text(
                            text = codeLanguage.name,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.tertiary
                        )
                    }
                }
                Spacer(modifier = Modifier.weight(1f))
                IconButton(onClick = onOpenFolder) {
                    Icon(Icons.Default.FolderOpen, contentDescription = "Open folder")
                }
                IconButton(onClick = onNewFile) {
                    Icon(Icons.Default.Add, contentDescription = "New file")
                }
                IconButton(onClick = onSaveFile) {
                    Icon(Icons.Default.Save, contentDescription = "Save file")
                }
                IconButton(onClick = onCopy, enabled = shownCode.isNotBlank()) {
                    Icon(Icons.Default.ContentCopy, contentDescription = "Copy code")
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            if (folderFiles.isNotEmpty()) {
                LazyRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    items(folderFiles) { item ->
                        val selected = item.first == currentFileUri
                        Button(
                            onClick = { onSelectFolderFile(item.first) },
                            modifier = Modifier.height(30.dp),
                            shape = RoundedCornerShape(16.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
                            )
                        ) {
                            Text(item.second, style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
                Spacer(modifier = Modifier.height(8.dp))
            }
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .background(
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                        shape = RoundedCornerShape(10.dp)
                    )
                    .verticalScroll(scrollState)
                    .padding(12.dp)
            ) {
                BasicTextField(
                    value = shownCode,
                    onValueChange = { if (hasFileSession) onCodeChange(it) },
                    modifier = Modifier.fillMaxWidth(),
                    readOnly = !hasFileSession,
                    textStyle = MaterialTheme.typography.bodySmall.copy(
                        fontFamily = FontFamily.Monospace,
                        color = MaterialTheme.colorScheme.onSurface
                    ),
                    decorationBox = { innerTextField ->
                        if (shownCode.isBlank()) {
                            Text(
                                text = when {
                                    !hasFileSession -> "Open or create a file to start editing."
                                    isProcessing -> "AI is editing code..."
                                    else -> "Write code here or ask AI in chat..."
                                },
                                style = MaterialTheme.typography.bodySmall.copy(
                                    fontFamily = FontFamily.Monospace,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            )
                        }
                        innerTextField()
                    }
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            if (canDiscardLastAiEdit) {
                Button(
                    onClick = onDiscardLastAiEdit,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
                ) {
                    Text("Discard")
                }
                Spacer(modifier = Modifier.height(8.dp))
            }
            Button(
                onClick = onPreview,
                enabled = canPreview,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(44.dp),
                shape = RoundedCornerShape(50),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Icon(Icons.Default.OpenInBrowser, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text(stringResource(R.string.vibe_coder_preview))
            }
        }
    }
}
