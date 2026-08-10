package com.llmhub.llmhub.screens

import android.content.Context
import android.content.ContentValues
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.llmhub.llmhub.R
import com.llmhub.llmhub.components.ModelSelectorCard
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.MusicGeneratorBackend
import com.llmhub.llmhub.data.hasCompleteDownloadedBundle
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class GeneratedMusicTrack(
    val id: Long,
    val prompt: String,
    val requestedDurationSeconds: Int,
    val file: java.io.File
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MusicGeneratorScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModelDownload: () -> Unit = {}
) {
    val context = LocalContext.current
    val keyboardController = LocalSoftwareKeyboardController.current
    val coroutineScope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    var promptText by remember { mutableStateOf("") }
    var durationSeconds by remember { mutableFloatStateOf(10f) }
    var isGenerating by remember { mutableStateOf(false) }
    val generatedTracks = remember { mutableStateListOf<GeneratedMusicTrack>() }
    var playingTrackId by remember { mutableStateOf<Long?>(null) }
    var loadedTrackId by remember { mutableStateOf<Long?>(null) }
    var mediaPlayer by remember { mutableStateOf<android.media.MediaPlayer?>(null) }
    val scrollState = rememberScrollState()

    LaunchedEffect(generatedTracks.size) {
        if (generatedTracks.isNotEmpty()) {
            scrollState.animateScrollTo(scrollState.maxValue)
        }
    }

    // A SoundGen install is usable only when its inference, conditioner, decoder,
    // and tokenizer files are all present. Older builds downloaded only the primary
    // file, so deliberately do not treat that stale install as complete.
    val downloadedModels = ModelData.musicGenerationModels.filter {
        it.hasCompleteDownloadedBundle(context)
    }
    val downloadedModelNames = downloadedModels.map { it.name }

    // Model settings state
    var showSettingsSheet by remember { mutableStateOf(false) }
    var selectedModel by remember(downloadedModelNames) {
        mutableStateOf<LLMModel?>(downloadedModels.firstOrNull())
    }
    var isModelLoaded by remember { mutableStateOf(false) }
    var isLoadingModel by remember { mutableStateOf(false) }

    val durationRange = remember(selectedModel?.name) {
        MusicGeneratorBackend.durationRange(selectedModel?.name.orEmpty())
    }
    LaunchedEffect(durationRange.start, durationRange.endInclusive) {
        durationSeconds = durationSeconds.coerceIn(durationRange.start, durationRange.endInclusive)
    }

    val presetPrompts = listOf(
        "Upbeat 80s Synthwave synth bass & drums",
        "Ambient relaxing acoustic piano & warm pads",
        "Epic cinematic trailer orchestral battle motif",
        "Chill Lo-Fi hip hop beat with rain sounds",
        "Energetic rock guitar riff with upbeat rhythm",
        "Smooth jazz saxophone melody with acoustic bass"
    )

    // Unload player on exit
    DisposableEffect(Unit) {
        onDispose {
            mediaPlayer?.stop()
            mediaPlayer?.release()
            mediaPlayer = null
            CoroutineScope(Dispatchers.IO).launch {
                MusicGeneratorBackend.unloadModel()
            }
        }
    }

    fun generateMusic() {
        val requestedPrompt = promptText.trim()
        if (requestedPrompt.isEmpty()) {
            coroutineScope.launch {
                snackbarHostState.showSnackbar(context.getString(R.string.music_prompt_empty_warning))
            }
            return
        }
        val model = selectedModel
        if (model == null || !model.hasCompleteDownloadedBundle(context)) {
            coroutineScope.launch {
                snackbarHostState.showSnackbar("SoundGen model is incomplete. Download or resume all model files.")
            }
            onNavigateToModelDownload()
            return
        }
        val requestedDuration = durationSeconds.toInt()
        keyboardController?.hide()
        isGenerating = true
        coroutineScope.launch {
            mediaPlayer?.stop()
            mediaPlayer?.release()
            mediaPlayer = null
            playingTrackId = null
            loadedTrackId = null

            if (!MusicGeneratorBackend.isModelLoaded(model.name)) {
                isLoadingModel = true
                isModelLoaded = MusicGeneratorBackend.loadModel(context, model.name)
                isLoadingModel = false
            }
            val file = if (isModelLoaded) MusicGeneratorBackend.generateMusic(
                context = context,
                modelName = model.name,
                prompt = requestedPrompt,
                durationSeconds = requestedDuration.toDouble(),
                onProgress = { }
            ) else null

            isGenerating = false
            if (file != null && file.exists()) {
                generatedTracks.add(
                    GeneratedMusicTrack(
                        id = System.nanoTime(),
                        prompt = requestedPrompt,
                        requestedDurationSeconds = requestedDuration,
                        file = file
                    )
                )
            } else {
                snackbarHostState.showSnackbar("Failed to generate music audio")
            }
        }
    }

    // Settings Bottom Sheet
    if (showSettingsSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSettingsSheet = false },
            sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = stringResource(R.string.feature_settings_title),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 16.dp)
                )

                ModelSelectorCard(
                    models = downloadedModels,
                    selectedModel = selectedModel,
                    selectedBackend = com.google.mediapipe.tasks.genai.llminference.LlmInference.Backend.CPU,
                    selectedNpuDeviceId = null,
                    isLoading = isLoadingModel,
                    isModelLoaded = isModelLoaded,
                    onModelSelected = {
                        if (selectedModel?.name != it.name) {
                            selectedModel = it
                            isModelLoaded = false
                            coroutineScope.launch { MusicGeneratorBackend.unloadModel() }
                        }
                    },
                    onBackendSelected = null,
                    onLoadModel = {
                        isLoadingModel = true
                        coroutineScope.launch {
                            val model = selectedModel
                            isModelLoaded = model != null && MusicGeneratorBackend.loadModel(context, model.name)
                            isLoadingModel = false
                            if (isModelLoaded) {
                                snackbarHostState.showSnackbar(context.getString(R.string.model_loaded))
                            } else {
                                snackbarHostState.showSnackbar("Failed to load SoundGen model")
                            }
                        }
                    },
                    onUnloadModel = {
                        isModelLoaded = false
                        isLoadingModel = true
                        coroutineScope.launch {
                            MusicGeneratorBackend.unloadModel()
                            isLoadingModel = false
                        }
                    },
                    filterMultimodalOnly = false
                )

                Spacer(modifier = Modifier.height(24.dp))
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.MusicNote,
                            contentDescription = null,
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(end = 8.dp)
                        )
                        Text(
                            text = stringResource(R.string.feature_music_generator),
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold
                        )
                    }
                },
                navigationIcon = {
                    IconButton(onClick = {
                        isModelLoaded = false
                        onNavigateBack()
                    }) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = stringResource(R.string.back)
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(
                            imageVector = Icons.Default.Tune,
                            contentDescription = stringResource(R.string.feature_settings_title)
                        )
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        if (downloadedModels.isEmpty()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Icon(
                    imageVector = Icons.Default.MusicNote,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.6f),
                    modifier = Modifier.size(64.dp)
                )
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = stringResource(R.string.music_generator_download_model),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(R.string.music_generator_download_model_desc),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )
                Spacer(modifier = Modifier.height(32.dp))
                FilledTonalButton(
                    onClick = onNavigateToModelDownload,
                    modifier = Modifier.fillMaxWidth(0.6f)
                ) {
                    Icon(
                        imageVector = Icons.Default.GetApp,
                        contentDescription = null
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.download_models_title))
                }
            }
        } else {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .imePadding()
            ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scrollState)
                    .padding(16.dp)
                    .padding(bottom = 84.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Prompt Input Card
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            text = stringResource(R.string.music_prompt_label),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )

                        OutlinedTextField(
                            value = promptText,
                            onValueChange = { promptText = it },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(110.dp),
                            placeholder = {
                                Text(
                                    text = stringResource(R.string.prompt_hint_music),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                                )
                            },
                            shape = RoundedCornerShape(12.dp)
                        )

                        // Presets
                        Text(
                            text = stringResource(R.string.music_style_presets),
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary
                        )
                        LazyRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            items(presetPrompts) { preset ->
                                SuggestionChip(
                                    onClick = { promptText = preset },
                                    label = { Text(preset) },
                                    icon = {
                                        Icon(
                                            Icons.Default.GraphicEq,
                                            contentDescription = null,
                                            modifier = Modifier.size(16.dp)
                                        )
                                    }
                                )
                            }
                        }
                    }
                }

                // Duration Controls
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = stringResource(R.string.music_duration_label),
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Text(
                                text = "${durationSeconds.toInt()}s",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                        Slider(
                            value = durationSeconds,
                            onValueChange = { durationSeconds = it },
                            valueRange = durationRange,
                            steps = (durationRange.endInclusive - durationRange.start).toInt().minus(1).coerceAtLeast(0)
                        )
                    }
                }

                generatedTracks.forEach { track ->
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.4f)
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(
                                    modifier = Modifier.weight(1f),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.GraphicEq,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Column {
                                        Text(
                                            text = track.prompt,
                                            style = MaterialTheme.typography.titleSmall,
                                            fontWeight = FontWeight.Bold,
                                            maxLines = 2
                                        )
                                        Text(
                                            text = "${track.requestedDurationSeconds}s Audio Clip",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                                IconButton(
                                    onClick = {
                                        coroutineScope.launch {
                                            val saved = saveAudioToMusicLibrary(context, track.file)
                                            Toast.makeText(
                                                context,
                                                if (saved) context.getString(R.string.music_saved_toast)
                                                else "Failed to save audio clip",
                                                Toast.LENGTH_SHORT
                                            ).show()
                                        }
                                    }
                                ) {
                                    Icon(Icons.Default.Download, contentDescription = "Save audio")
                                }
                            }

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.Center,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                FilledIconButton(
                                    onClick = {
                                        if (!track.file.exists()) return@FilledIconButton

                                        if (playingTrackId == track.id) {
                                            mediaPlayer?.pause()
                                            playingTrackId = null
                                        } else {
                                            try {
                                                if (mediaPlayer == null || loadedTrackId != track.id) {
                                                    mediaPlayer?.release()
                                                    mediaPlayer = android.media.MediaPlayer().apply {
                                                        setDataSource(track.file.absolutePath)
                                                        prepare()
                                                        setOnCompletionListener {
                                                            playingTrackId = null
                                                        }
                                                    }
                                                    loadedTrackId = track.id
                                                }
                                                mediaPlayer?.start()
                                                playingTrackId = track.id
                                            } catch (e: Exception) {
                                                android.util.Log.e("MusicGeneratorScreen", "Error playing audio: ${e.message}", e)
                                                playingTrackId = null
                                            }
                                        }
                                    },
                                    modifier = Modifier.size(56.dp)
                                ) {
                                    Icon(
                                        imageVector = if (playingTrackId == track.id) Icons.Default.Pause else Icons.Default.PlayArrow,
                                        contentDescription = null,
                                        modifier = Modifier.size(32.dp)
                                    )
                                }
                            }
                        }
                    }
                }
            }

            Surface(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth(),
                shadowElevation = 8.dp,
                color = MaterialTheme.colorScheme.surface
            ) {
                Column {
                    FilledTonalButton(
                        onClick = ::generateMusic,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        shape = RoundedCornerShape(12.dp),
                        enabled = promptText.isNotBlank() && !isGenerating && !isLoadingModel
                    ) {
                        if (isGenerating || isLoadingModel) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(22.dp),
                                strokeWidth = 2.5.dp
                            )
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(
                                if (isLoadingModel) stringResource(R.string.model_loading)
                                else stringResource(R.string.generating_music)
                            )
                        } else {
                            Icon(Icons.Default.AutoAwesome, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(stringResource(R.string.generate_music))
                        }
                    }
                }
            }
            }
        }
    }
}

private suspend fun saveAudioToMusicLibrary(context: Context, source: java.io.File): Boolean =
    withContext(Dispatchers.IO) {
        if (!source.isFile) return@withContext false
        val values = ContentValues().apply {
            put(MediaStore.Audio.Media.DISPLAY_NAME, "LLMHub_Music_${System.currentTimeMillis()}.wav")
            put(MediaStore.Audio.Media.MIME_TYPE, "audio/wav")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Audio.Media.RELATIVE_PATH, Environment.DIRECTORY_MUSIC + "/LLMHub")
                put(MediaStore.Audio.Media.IS_PENDING, 1)
            }
        }
        val resolver = context.contentResolver
        val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        } else {
            MediaStore.Audio.Media.EXTERNAL_CONTENT_URI
        }
        val uri = resolver.insert(collection, values) ?: return@withContext false
        try {
            resolver.openOutputStream(uri)?.use { output -> source.inputStream().use { it.copyTo(output) } }
                ?: error("Could not open MediaStore output")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.clear()
                values.put(MediaStore.Audio.Media.IS_PENDING, 0)
                resolver.update(uri, values, null, null)
            }
            true
        } catch (error: Throwable) {
            resolver.delete(uri, null, null)
            android.util.Log.e("MusicGeneratorScreen", "Failed to save audio", error)
            false
        }
    }
