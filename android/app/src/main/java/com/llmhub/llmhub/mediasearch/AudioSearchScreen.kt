package com.llmhub.llmhub.mediasearch

import android.Manifest
import android.media.MediaPlayer
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.R
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private fun audioPermissions(): Array<String> =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) arrayOf(Manifest.permission.READ_MEDIA_AUDIO)
    else arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)

private fun formatMs(ms: Long): String {
    val s = (ms / 1000).coerceAtLeast(0)
    return "%d:%02d".format(s / 60, s % 60)
}

/** One MediaPlayer for the whole screen, so only one file plays at a time. */
private class AudioPlayback(private val context: android.content.Context) {
    var playingId by mutableStateOf<String?>(null)
    var isPlaying by mutableStateOf(false)
    var positionMs by mutableLongStateOf(0L)
    var durationMs by mutableLongStateOf(0L)
    private var player: MediaPlayer? = null

    fun playFrom(asset: AudioAsset, startMs: Long) {
        if (playingId != asset.id || player == null) {
            release()
            player = try {
                MediaPlayer().apply {
                    setDataSource(context, asset.uri)
                    prepare()
                    setOnCompletionListener { this@AudioPlayback.isPlaying = false }
                }
            } catch (e: Exception) {
                android.util.Log.w("AudioSearch", "Playback failed: ${e.message}")
                null
            } ?: return
            playingId = asset.id
            durationMs = player?.duration?.toLong()?.takeIf { it > 0 } ?: asset.durationMs
        }
        player?.seekTo(startMs.toInt())
        positionMs = startMs
        player?.start()
        isPlaying = true
    }

    fun toggle(asset: AudioAsset) {
        val p = player
        if (playingId == asset.id && p != null) {
            if (p.isPlaying) { p.pause(); isPlaying = false } else { p.start(); isPlaying = true }
        } else {
            playFrom(asset, 0)
        }
    }

    fun seek(asset: AudioAsset, fraction: Float) {
        val total = if (playingId == asset.id && durationMs > 0) durationMs else asset.durationMs
        val target = (fraction.coerceIn(0f, 1f) * total).toLong()
        if (playingId == asset.id && player != null) {
            player?.seekTo(target.toInt())
            positionMs = target
        } else {
            playFrom(asset, target)
        }
    }

    fun pause() {
        player?.takeIf { it.isPlaying }?.pause()
        isPlaying = false
    }

    fun tick() {
        player?.let { if (it.isPlaying) positionMs = it.currentPosition.toLong() }
    }

    fun release() {
        player?.release()
        player = null
        playingId = null
        isPlaying = false
        positionMs = 0
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AudioSearchScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModelDownload: () -> Unit,
    viewModel: AudioSearchViewModel = viewModel()
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    val models by viewModel.downloadedModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val modelError by viewModel.modelError.collectAsState()
    val source by viewModel.source.collectAsState()
    val assets by viewModel.assets.collectAsState()
    val progress by viewModel.progress.collectAsState()
    val isPaused by viewModel.isPaused.collectAsState()
    val query by viewModel.query.collectAsState()
    val results by viewModel.results.collectAsState()
    val isSearching by viewModel.isSearching.collectAsState()

    var showSettings by remember { mutableStateOf(false) }
    val playback = remember { AudioPlayback(context.applicationContext) }
    DisposableEffect(Unit) { onDispose { playback.release() } }
    LaunchedEffect(playback.isPlaying) {
        while (playback.isPlaying) {
            playback.tick()
            delay(50)
        }
    }

    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) { viewModel.refresh() }
    LifecycleEventEffect(Lifecycle.Event.ON_PAUSE) { playback.pause() }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        if (grants.values.any { it }) {
            viewModel.useDeviceAudio()
        } else {
            scope.launch { snackbarHostState.showSnackbar(context.getString(R.string.audio_search_permission_denied)) }
        }
    }
    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        viewModel.addImportedAudio(uris)
    }
    val requestDeviceAudio = { permissionLauncher.launch(audioPermissions()) }
    val importAudio = { importLauncher.launch(arrayOf("audio/*")) }

    if (showSettings) {
        MediaSearchSettingsSheet(
            models = models,
            selectedModel = selectedModel,
            isLoadingModel = isLoadingModel,
            isModelLoaded = isModelLoaded,
            onModelSelected = viewModel::selectModel,
            onLoadModel = viewModel::loadModel,
            onUnloadModel = viewModel::unloadModel,
            libraryCountText = stringResource(R.string.audio_search_count, progress.processed, assets.size),
            libraryActions = {
                OutlinedButton(onClick = importAudio, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                    Icon(Icons.Default.AudioFile, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(stringResource(R.string.audio_search_import))
                }
                if (source != AudioSource.DEVICE) {
                    OutlinedButton(onClick = requestDeviceAudio, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                        Icon(Icons.Default.LibraryMusic, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.audio_search_scan_device))
                    }
                }
            },
            onClearAll = {
                playback.release()
                viewModel.clearAll()
                showSettings = false
            },
            onDismiss = { showSettings = false }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.GraphicEq, contentDescription = null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(end = 8.dp))
                        Text(stringResource(R.string.feature_audio_search), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    if (models.isNotEmpty()) {
                        IconButton(onClick = { showSettings = true }) {
                            Icon(Icons.Default.Tune, contentDescription = stringResource(R.string.feature_settings_title))
                        }
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { padding ->
        val contentModifier = Modifier.fillMaxSize().padding(padding)
        when {
            models.isEmpty() -> MediaSearchDownloadGate(Icons.Default.GraphicEq, onNavigateToModelDownload, contentModifier)
            source == AudioSource.NONE -> MediaSearchOnboarding(
                icon = Icons.Default.GraphicEq,
                title = stringResource(R.string.audio_search_onboarding_title),
                description = stringResource(R.string.audio_search_onboarding_desc),
                primaryLabel = stringResource(R.string.audio_search_scan_device),
                onPrimary = requestDeviceAudio,
                secondaryLabel = stringResource(R.string.audio_search_import),
                onSecondary = importAudio,
                modifier = contentModifier
            )
            else -> LazyColumn(
                modifier = contentModifier.imePadding(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                item {
                    MediaSearchField(query, viewModel::onQueryChange, stringResource(R.string.audio_search_hint), isSearching)
                }
                item {
                    MediaSearchStatusCard(
                        isLoadingModel = isLoadingModel,
                        modelError = modelError,
                        progress = progress,
                        isPaused = isPaused,
                        onPause = viewModel::pauseIndexing,
                        onResume = viewModel::resumeIndexing,
                        onRetryModel = viewModel::loadModel
                    )
                }
                if (results != null && !progress.isComplete) {
                    item {
                        Text(stringResource(R.string.media_search_incomplete_warning), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                if (source == AudioSource.DEVICE && assets.isEmpty()) {
                    item {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text(stringResource(R.string.audio_search_permission_denied), style = MaterialTheme.typography.bodyMedium)
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                Button(onClick = requestDeviceAudio) { Text(stringResource(R.string.media_search_grant_access)) }
                                OutlinedButton(onClick = importAudio) { Text(stringResource(R.string.audio_search_import)) }
                            }
                        }
                    }
                }
                val matches = results
                if (matches != null) {
                    if (matches.isEmpty() && !isSearching) {
                        item { Text(stringResource(R.string.media_search_no_results), style = MaterialTheme.typography.bodyMedium) }
                    }
                    items(matches, key = { it.asset.id }) { match ->
                        AudioResultCard(asset = match.asset, match = match, playback = playback)
                    }
                } else {
                    items(assets, key = { it.id }) { asset ->
                        AudioResultCard(asset = asset, match = null, playback = playback)
                    }
                }
            }
        }
    }
}

@Composable
private fun AudioResultCard(asset: AudioAsset, match: AudioMatch?, playback: AudioPlayback) {
    val isCurrent = playback.playingId == asset.id
    val totalMs = if (isCurrent && playback.durationMs > 0) playback.durationMs else asset.durationMs
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
    ) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                IconButton(onClick = { playback.toggle(asset) }, modifier = Modifier.size(40.dp)) {
                    Icon(
                        if (isCurrent && playback.isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(28.dp)
                    )
                }
                Column(Modifier.weight(1f)) {
                    Text(asset.name, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(
                        if (isCurrent) "${formatMs(playback.positionMs)} / ${formatMs(totalMs)}" else formatMs(totalMs),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            MomentTimeline(
                timeline = match?.timeline.orEmpty(),
                totalMs = totalMs,
                progress = if (isCurrent && totalMs > 0) playback.positionMs.toFloat() / totalMs else 0f,
                onSeek = { playback.seek(asset, it) }
            )
            if (match != null && match.moments.isNotEmpty()) {
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(match.moments) { moment ->
                        AssistChip(
                            onClick = { playback.playFrom(asset, moment.startMs.toLong()) },
                            label = { Text(stringResource(R.string.audio_search_moment, formatMs(moment.startMs.toLong()))) },
                            leadingIcon = { Icon(Icons.Default.PlayCircle, contentDescription = null, modifier = Modifier.size(16.dp)) }
                        )
                    }
                }
            }
        }
    }
}

/**
 * Waveform-style strip: one bar per analyzed moment, height = how well it matches the query.
 * Tap or drag to seek, like the music generator's waveform.
 */
@Composable
private fun MomentTimeline(timeline: List<AudioMoment>, totalMs: Long, progress: Float, onSeek: (Float) -> Unit) {
    val color = MaterialTheme.colorScheme.primary
    val bars = remember(timeline, totalMs) {
        val count = 48
        if (timeline.isEmpty() || totalMs <= 0) return@remember List(count) { 0.3f }
        val min = timeline.minOf { it.score }
        val max = timeline.maxOf { it.score }
        val range = (max - min).takeIf { it > 1e-4f } ?: 1f
        List(count) { i ->
            val t = (i + 0.5f) / count * totalMs
            val moment = timeline.firstOrNull { t >= it.startMs && t < it.endMs }
            if (moment == null) 0.12f else 0.15f + 0.85f * ((moment.score - min) / range)
        }
    }
    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(40.dp)
            .pointerInput(totalMs) {
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    onSeek(down.position.x / size.width)
                    while (true) {
                        val change = awaitPointerEvent().changes.firstOrNull() ?: break
                        if (change.changedToUp()) { change.consume(); break }
                        if (change.pressed) { change.consume(); onSeek(change.position.x / size.width) }
                    }
                }
            }
    ) {
        val spacing = 3f
        val barWidth = ((size.width - spacing * (bars.size - 1)) / bars.size).coerceAtLeast(1f)
        val played = progress * bars.size
        bars.forEachIndexed { i, value ->
            val h = (value * size.height).coerceAtLeast(4f)
            drawRoundRect(
                color = if (i < played) color else color.copy(alpha = 0.35f),
                topLeft = Offset(i * (barWidth + spacing), (size.height - h) / 2f),
                size = Size(barWidth, h),
                cornerRadius = CornerRadius(barWidth / 2f)
            )
        }
        if (progress > 0f) {
            val x = (progress * size.width).coerceIn(0f, size.width)
            drawLine(color, Offset(x, 0f), Offset(x, size.height), strokeWidth = 2.dp.toPx())
        }
    }
}
