package com.llmhub.llmhub.screens

import android.content.Context
import android.content.ContentValues
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
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
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.input.pointer.changedToUp
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
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
    val musicPreferences = remember(context) {
        context.getSharedPreferences("music_generator_prefs", Context.MODE_PRIVATE)
    }

    var promptText by remember { mutableStateOf("") }
    var durationSeconds by remember { mutableFloatStateOf(10f) }
    var isGenerating by remember { mutableStateOf(false) }
    val generatedTracks = remember { mutableStateListOf<GeneratedMusicTrack>() }
    var playingTrackId by remember { mutableStateOf<Long?>(null) }
    var loadedTrackId by remember { mutableStateOf<Long?>(null) }
    var currentPositionMs by remember { mutableIntStateOf(0) }
    var currentDurationMs by remember { mutableIntStateOf(0) }
    var isDraggingTimeline by remember { mutableStateOf(false) }
    var mediaPlayer by remember { mutableStateOf<android.media.MediaPlayer?>(null) }
    val scrollState = rememberScrollState()

    LaunchedEffect(generatedTracks.size) {
        if (generatedTracks.isNotEmpty()) {
            scrollState.animateScrollTo(scrollState.maxValue)
        }
    }

    LaunchedEffect(playingTrackId, isDraggingTimeline) {
        while (playingTrackId != null && !isDraggingTimeline) {
            val player = mediaPlayer
            if (player != null) {
                try {
                    if (player.isPlaying) {
                        currentPositionMs = player.currentPosition
                        val dur = player.duration
                        if (dur > 0) currentDurationMs = dur
                    }
                } catch (_: Exception) {}
            }
            kotlinx.coroutines.delay(40)
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
        val savedModelName = musicPreferences.getString("selected_model_name", null)
        mutableStateOf(
            downloadedModels.firstOrNull { it.name == savedModelName }
                ?: downloadedModels.firstOrNull()
        )
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
            playingTrackId = null
            loadedTrackId = null
            currentPositionMs = 0
            currentDurationMs = 0
            CoroutineScope(Dispatchers.IO).launch {
                MusicGeneratorBackend.unloadModel()
            }
        }
    }

    fun seekToFraction(track: GeneratedMusicTrack, fraction: Float) {
        try {
            if (loadedTrackId != track.id || mediaPlayer == null) {
                mediaPlayer?.release()
                mediaPlayer = android.media.MediaPlayer().apply {
                    setDataSource(track.file.absolutePath)
                    setOnCompletionListener {
                        playingTrackId = null
                        currentPositionMs = 0
                    }
                    prepare()
                }
                loadedTrackId = track.id
                currentDurationMs = mediaPlayer?.duration?.takeIf { it > 0 } ?: (track.requestedDurationSeconds * 1000)
            }
            val dur = if (currentDurationMs > 0) currentDurationMs else (track.requestedDurationSeconds * 1000)
            val targetMs = (dur * fraction).toInt().coerceIn(0, dur)
            currentPositionMs = targetMs
            mediaPlayer?.seekTo(targetMs)
        } catch (e: Exception) {
            android.util.Log.e("MusicGeneratorScreen", "Error seeking: ${e.message}", e)
        }
    }

    fun togglePlayPause(track: GeneratedMusicTrack) {
        if (!track.file.exists()) return
        try {
            if (loadedTrackId != track.id || mediaPlayer == null) {
                mediaPlayer?.release()
                mediaPlayer = android.media.MediaPlayer().apply {
                    setDataSource(track.file.absolutePath)
                    setOnCompletionListener {
                        playingTrackId = null
                        currentPositionMs = 0
                    }
                    prepare()
                }
                loadedTrackId = track.id
                currentDurationMs = mediaPlayer?.duration?.takeIf { it > 0 } ?: (track.requestedDurationSeconds * 1000)
                currentPositionMs = 0
            }

            if (playingTrackId == track.id) {
                mediaPlayer?.pause()
                playingTrackId = null
            } else {
                if (currentDurationMs > 0 && currentPositionMs >= currentDurationMs - 100) {
                    mediaPlayer?.seekTo(0)
                    currentPositionMs = 0
                }
                mediaPlayer?.start()
                playingTrackId = track.id
            }
        } catch (e: Exception) {
            android.util.Log.e("MusicGeneratorScreen", "Error playing audio: ${e.message}", e)
            playingTrackId = null
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
                            musicPreferences.edit()
                                .putString("selected_model_name", it.name)
                                .apply()
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
                            if (!isModelLoaded) {
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
                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                        )
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(12.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Row(
                                    modifier = Modifier.weight(1f),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.GraphicEq,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(20.dp)
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = track.prompt,
                                        style = MaterialTheme.typography.titleSmall,
                                        fontWeight = FontWeight.SemiBold,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
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
                                    },
                                    modifier = Modifier.size(36.dp)
                                ) {
                                    Icon(
                                        Icons.Default.Download,
                                        contentDescription = "Save audio",
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(20.dp)
                                    )
                                }
                                IconButton(
                                    onClick = {
                                        if (loadedTrackId == track.id) {
                                            mediaPlayer?.stop()
                                            mediaPlayer?.release()
                                            mediaPlayer = null
                                            loadedTrackId = null
                                            playingTrackId = null
                                            currentPositionMs = 0
                                            currentDurationMs = 0
                                        }
                                        track.file.delete()
                                        generatedTracks.remove(track)
                                    },
                                    modifier = Modifier.size(36.dp)
                                ) {
                                    Icon(
                                        Icons.Default.Delete,
                                        contentDescription = "Delete audio",
                                        tint = MaterialTheme.colorScheme.error,
                                        modifier = Modifier.size(20.dp)
                                    )
                                }
                            }

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                val isThisTrackLoaded = loadedTrackId == track.id
                                val isThisTrackPlaying = playingTrackId == track.id

                                IconButton(
                                    onClick = { togglePlayPause(track) },
                                    modifier = Modifier.size(40.dp)
                                ) {
                                    Icon(
                                        imageVector = if (isThisTrackPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                                        contentDescription = if (isThisTrackPlaying) "Pause" else "Play",
                                        tint = MaterialTheme.colorScheme.primary,
                                        modifier = Modifier.size(28.dp)
                                    )
                                }

                                val barHeights = remember(track.file.absolutePath, track.file.lastModified()) {
                                    extractRealWaveform(track.file, 36)
                                }
                                val waveformColor = MaterialTheme.colorScheme.primary
                                val totalTrackMs = if (isThisTrackLoaded && currentDurationMs > 0) {
                                    currentDurationMs
                                } else {
                                    track.requestedDurationSeconds * 1000
                                }
                                val posMs = if (isThisTrackLoaded) currentPositionMs else 0
                                val progress = if (totalTrackMs > 0) (posMs.toFloat() / totalTrackMs.toFloat()).coerceIn(0f, 1f) else 0f

                                Canvas(
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(40.dp)
                                        .pointerInput(track.id) {
                                            awaitEachGesture {
                                                val down = awaitFirstDown(requireUnconsumed = false)
                                                try {
                                                    isDraggingTimeline = true
                                                    val newFrac = (down.position.x / size.width.toFloat()).coerceIn(0f, 1f)
                                                    seekToFraction(track, newFrac)

                                                    while (true) {
                                                        val event = awaitPointerEvent()
                                                        val change = event.changes.firstOrNull() ?: break
                                                        if (change.changedToUp()) {
                                                            change.consume()
                                                            break
                                                        }
                                                        if (change.pressed) {
                                                            change.consume()
                                                            val dragFrac = (change.position.x / size.width.toFloat()).coerceIn(0f, 1f)
                                                            seekToFraction(track, dragFrac)
                                                        }
                                                    }
                                                } finally {
                                                    isDraggingTimeline = false
                                                }
                                            }
                                        }
                                ) {
                                    val barCount = barHeights.size
                                    val spacingPx = 3f
                                    val totalSpacing = spacingPx * (barCount - 1)
                                    val barWidth = ((size.width - totalSpacing) / barCount).coerceAtLeast(1f)
                                    val centerY = size.height / 2f
                                    val maxBarHeight = size.height
                                    val activeColor = waveformColor
                                    val inactiveColor = waveformColor.copy(alpha = 0.35f)
                                    val progressBars = (progress * barCount).coerceIn(0f, barCount.toFloat())

                                    for (i in 0 until barCount) {
                                        val height = (barHeights[i] * maxBarHeight).coerceAtLeast(4f).coerceAtMost(maxBarHeight)
                                        val left = i * (barWidth + spacingPx)
                                        val top = centerY - height / 2f
                                        val color = if (i < progressBars) activeColor else inactiveColor
                                        drawRoundRect(
                                            color = color,
                                            topLeft = Offset(left, top),
                                            size = Size(barWidth, height),
                                            cornerRadius = CornerRadius(barWidth / 2f)
                                        )
                                    }

                                    if (progress > 0f) {
                                        val playheadX = (progress * size.width).coerceIn(0f, size.width)
                                        drawLine(
                                            color = activeColor,
                                            start = Offset(playheadX, 0f),
                                            end = Offset(playheadX, size.height),
                                            strokeWidth = 2.dp.toPx()
                                        )
                                    }
                                }

                                Text(
                                    text = "${formatAudioTime(posMs)} / ${formatAudioTime(totalTrackMs)}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
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

private fun formatAudioTime(ms: Int): String {
    val totalSeconds = (ms / 1000).coerceAtLeast(0)
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return String.format(java.util.Locale.US, "%d:%02d", minutes, seconds)
}

private fun extractRealWaveform(file: java.io.File, barCount: Int = 36): List<Float> {
    if (!file.exists() || file.length() <= 44L) {
        return List(barCount) { 0.25f }
    }
    return try {
        val totalBytes = file.length() - 44L
        val totalFrames = totalBytes / 4L
        if (totalFrames <= 0L) return List(barCount) { 0.25f }

        val bars = FloatArray(barCount)
        val framesPerBar = (totalFrames / barCount).coerceAtLeast(1L)
        val buffer = ByteArray(8192)

        java.io.BufferedInputStream(java.io.FileInputStream(file)).use { input ->
            var skipped = 0L
            while (skipped < 44L) {
                val s = input.skip(44L - skipped)
                if (s <= 0) break
                skipped += s
            }

            var currentFrame = 0L
            var readBytes: Int
            while (input.read(buffer).also { readBytes = it } > 0) {
                var i = 0
                while (i + 3 < readBytes) {
                    val barIndex = ((currentFrame / framesPerBar).toInt()).coerceIn(0, barCount - 1)
                    val left = (buffer[i].toInt() and 0xFF) or (buffer[i + 1].toInt() shl 8)
                    val right = (buffer[i + 2].toInt() and 0xFF) or (buffer[i + 3].toInt() shl 8)
                    val absLeft = kotlin.math.abs(left.toShort().toInt())
                    val absRight = kotlin.math.abs(right.toShort().toInt())
                    val peak = maxOf(absLeft, absRight)
                    if (peak > bars[barIndex]) {
                        bars[barIndex] = peak.toFloat()
                    }
                    currentFrame++
                    i += 4
                }
            }
        }

        val maxVal = bars.maxOrNull() ?: 1f
        val norm = if (maxVal > 100f) 1f / maxVal else 1f / 32768f
        bars.map { ((it * norm) * 0.88f + 0.12f).coerceIn(0.12f, 1f) }
    } catch (e: Exception) {
        android.util.Log.e("MusicGeneratorScreen", "Failed to extract waveform: ${e.message}", e)
        List(barCount) { 0.25f }
    }
}
