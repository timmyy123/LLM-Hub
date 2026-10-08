package com.llmhub.llmhub.mediasearch

import android.media.MediaPlayer
import android.net.Uri
import android.widget.VideoView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.llmhub.llmhub.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VideoMomentScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModelDownload: () -> Unit,
    viewModel: VideoMomentViewModel = viewModel()
) {
    val models by viewModel.downloadedModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val modelError by viewModel.modelError.collectAsState()
    val videos by viewModel.videos.collectAsState()
    val open by viewModel.open.collectAsState()
    val progress by viewModel.progress.collectAsState()
    val isPaused by viewModel.isPaused.collectAsState()
    val query by viewModel.query.collectAsState()
    val results by viewModel.results.collectAsState()
    val isSearching by viewModel.isSearching.collectAsState()
    var showSettings by remember { mutableStateOf(false) }
    var selected by remember { mutableStateOf<VideoMoment?>(null) }
    var editing by remember { mutableStateOf<VideoMoment?>(null) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) viewModel.addVideos(listOf(uri))
    }
    val pickVideo = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) }
    BackHandler(enabled = open != null) { viewModel.closeVideo() }

    if (showSettings) {
        MediaSearchSettingsSheet(
            models = models,
            selectedModel = selectedModel,
            isLoadingModel = isLoadingModel,
            isModelLoaded = isModelLoaded,
            onModelSelected = viewModel::selectModel,
            onLoadModel = viewModel::loadModel,
            onUnloadModel = viewModel::unloadModel,
            libraryCountText = stringResource(R.string.media_search_progress, progress.processed, progress.total, progress.percent),
            libraryActions = {
                OutlinedButton(onClick = pickVideo, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                    Text(stringResource(R.string.video_moment_pick))
                }
            },
            onClearAll = { viewModel.clearAll(); showSettings = false },
            onDismiss = { showSettings = false }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(open?.name ?: stringResource(R.string.feature_video_moment), fontWeight = FontWeight.Bold, maxLines = 1) },
                navigationIcon = {
                    IconButton(onClick = { if (open != null) viewModel.closeVideo() else onNavigateBack() }) {
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
        }
    ) { padding ->
        when {
            models.isEmpty() -> MediaSearchDownloadGate(Icons.Default.VideoLibrary, onNavigateToModelDownload, Modifier.padding(padding))
            videos.isEmpty() -> MediaSearchOnboarding(
                icon = Icons.Default.VideoLibrary,
                title = stringResource(R.string.video_moment_onboarding_title),
                description = stringResource(R.string.video_moment_onboarding_desc),
                primaryLabel = stringResource(R.string.video_moment_pick),
                onPrimary = pickVideo,
                modifier = Modifier.padding(padding)
            )
            open == null -> VideoProjectGrid(videos, pickVideo, Modifier.padding(padding)) { viewModel.open(it) }
            else -> MomentStage(
                video = open!!,
                query = query,
                onQueryChange = {
                    selected = null
                    viewModel.onQueryChange(it)
                },
                onSearch = viewModel::submitSearch,
                onClear = {
                    selected = null
                    viewModel.onQueryChange("")
                },
                results = results,
                isSearching = isSearching,
                selected = selected,
                onSelect = { moment ->
                    selected = moment
                },
                onEdit = { editing = it },
                editorOpen = editing != null,
                isLoadingModel = isLoadingModel,
                modelError = modelError,
                progress = progress,
                isPaused = isPaused,
                onPause = viewModel::pause,
                onResume = viewModel::resume,
                onRetryModel = viewModel::loadModel,
                modifier = Modifier.padding(padding)
            )
        }
    }

    editing?.let { moment ->
        ClipEditDialog(moment, onDismiss = { editing = null })
    }
}

@Composable
private fun VideoProjectGrid(videos: List<MomentVideo>, onAdd: () -> Unit, modifier: Modifier = Modifier, onOpen: (MomentVideo) -> Unit) {
    LazyVerticalGrid(
        columns = GridCells.Fixed(2),
        modifier = modifier,
        contentPadding = PaddingValues(16.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Box(
                    modifier = Modifier.fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(16.dp)).clickable(onClick = onAdd),
                    contentAlignment = Alignment.Center
                ) {
                    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.primaryContainer, shape = RoundedCornerShape(16.dp)) {}
                    Icon(Icons.Default.Add, contentDescription = stringResource(R.string.video_moment_pick), modifier = Modifier.size(40.dp))
                }
                Text(stringResource(R.string.video_moment_pick), style = MaterialTheme.typography.bodyMedium)
            }
        }
        items(videos, key = { it.id }) { video ->
            VideoTile(video) { onOpen(video) }
        }
    }
}

@Composable
private fun VideoTile(video: MomentVideo, onClick: () -> Unit) {
    val context = LocalContext.current
    var frame by remember(video.id) { mutableStateOf<ByteArray?>(null) }
    LaunchedEffect(video.id) {
        frame = withContext(Dispatchers.IO) { frameJpegAt(context, video.uri, 500L) }
    }
    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.clickable(onClick = onClick)) {
        Box(Modifier.fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(16.dp))) {
            Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surfaceVariant) {}
            AsyncImage(model = frame, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
            Text(
                formatClock(video.durationMs.toInt()),
                color = Color.White,
                modifier = Modifier.align(Alignment.BottomStart).padding(8.dp)
            )
        }
        Text(video.name, maxLines = 1, style = MaterialTheme.typography.bodyMedium)
    }
}

private data class MergedInterval(val startMs: Int, val endMs: Int, val ids: Set<String>)

private fun mergedIntervals(results: List<VideoMoment>): List<MergedInterval> {
    if (results.isEmpty()) return emptyList()
    val merged = ArrayList<MergedInterval>()
    for (result in results.sortedBy { it.startMs }) {
        val last = merged.lastOrNull()
        val id = "${result.video.id}:${result.startMs}"
        if (last != null && result.startMs <= last.endMs + 500) {
            merged[merged.lastIndex] = MergedInterval(
                last.startMs,
                maxOf(last.endMs, result.endMs),
                last.ids + id
            )
        } else {
            merged.add(MergedInterval(result.startMs, result.endMs, setOf(id)))
        }
    }
    return merged
}

/** Inline player: a clip plays from its start until its end, then stops. The scrubber stays at the bottom. */
@Composable
private fun MomentStage(
    video: MomentVideo,
    query: String,
    onQueryChange: (String) -> Unit,
    onSearch: () -> Unit,
    onClear: () -> Unit,
    results: List<VideoMoment>?,
    isSearching: Boolean,
    selected: VideoMoment?,
    onSelect: (VideoMoment) -> Unit,
    onEdit: (VideoMoment) -> Unit,
    editorOpen: Boolean,
    isLoadingModel: Boolean,
    modelError: Boolean,
    progress: IndexingProgress,
    isPaused: Boolean,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onRetryModel: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val focus = LocalFocusManager.current
    val playback = remember(video.id) { MomentPlayback(context, video.uri) }
    DisposableEffect(playback) { onDispose { playback.release() } }
    var position by remember(video.id) { mutableIntStateOf(0) }
    var duration by remember(video.id) { mutableIntStateOf(video.durationMs.toInt().coerceAtLeast(1)) }
    var playing by remember(video.id) { mutableStateOf(false) }
    var muted by remember(video.id) { mutableStateOf(false) }
    var revealed by remember(video.id) { mutableStateOf(false) }
    var poster by remember(video.id) { mutableStateOf<ByteArray?>(null) }
    val hits = results.orEmpty()
    val intervals = remember(hits) { mergedIntervals(hits) }
    val selectedId = selected?.let { "${it.video.id}:${it.startMs}" }

    LaunchedEffect(playback) {
        while (true) {
            playback.stopIfClipEnded()
            position = playback.view.currentPosition
            val length = playback.view.duration
            if (length > 0) duration = length
            playing = playback.view.isPlaying
            delay(80)
        }
    }
    LaunchedEffect(playing) {
        if (playing) revealed = true
    }
    LaunchedEffect(video.id, position, revealed) {
        if (revealed) return@LaunchedEffect
        val at = if (position <= 0) 500L else position.toLong()
        val jpeg = withContext(Dispatchers.IO) { frameJpegAt(context, video.uri, at) } ?: return@LaunchedEffect
        poster = jpeg
    }
    LaunchedEffect(editorOpen) {
        if (editorOpen) {
            playback.userTransport()
            playback.view.pause()
        }
    }

    Box(modifier.fillMaxSize().background(Color.Black)) {
        AndroidView(factory = { playback.view }, modifier = Modifier.fillMaxSize())
        if (!revealed) {
            AsyncImage(
                model = poster,
                contentDescription = null,
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize()
            )
        }
        Column(
            Modifier.align(Alignment.TopCenter).fillMaxWidth().background(
                Brush.verticalGradient(listOf(Color.Black.copy(alpha = 0.9f), Color.Transparent))
            )
        ) {
            val primary = MaterialTheme.colorScheme.primaryContainer
            Row(
                Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, top = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = query,
                    onValueChange = onQueryChange,
                    enabled = !isSearching,
                    modifier = Modifier.weight(1f),
                    placeholder = { Text(stringResource(R.string.video_moment_hint)) },
                    leadingIcon = {
                        if (isSearching) CircularProgressIndicator(Modifier.size(24.dp), strokeWidth = 2.dp)
                        else Icon(Icons.Default.Search, contentDescription = null)
                    },
                    trailingIcon = {
                        if (query.isNotBlank()) {
                            IconButton(onClick = onClear, enabled = !isSearching) {
                                Icon(Icons.Default.Close, contentDescription = stringResource(R.string.close))
                            }
                        }
                    },
                    singleLine = true,
                    shape = CircleShape,
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                    keyboardActions = KeyboardActions(onSearch = {
                        focus.clearFocus()
                        onSearch()
                    }),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedContainerColor = primary.copy(alpha = 0.85f),
                        unfocusedContainerColor = MaterialTheme.colorScheme.surface,
                        disabledContainerColor = MaterialTheme.colorScheme.surface,
                        unfocusedBorderColor = Color.Transparent,
                        disabledBorderColor = Color.Transparent
                    )
                )
            }
            MediaSearchStatusCard(isLoadingModel, modelError, progress, isPaused, onPause, onResume, onRetryModel)
            if (hits.isNotEmpty()) {
                Text(
                    stringResource(R.string.video_moment_top),
                    color = Color.White,
                    style = MaterialTheme.typography.labelLarge,
                    modifier = Modifier.padding(start = 16.dp, top = 12.dp, bottom = 8.dp)
                )
                LazyRow(contentPadding = PaddingValues(horizontal = 12.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    items(hits, key = { "${it.video.id}:${it.startMs}" }) { moment ->
                        val id = "${moment.video.id}:${moment.startMs}"
                        MomentThumb(moment, selected = id == selectedId) {
                            onSelect(moment)
                            val end = intervals.firstOrNull { id in it.ids }?.endMs ?: moment.endMs
                            playback.playClip(moment.startMs, end)
                        }
                    }
                }
                if (selected != null) {
                    FilledTonalButton(
                        onClick = { onEdit(selected) },
                        modifier = Modifier.align(Alignment.CenterHorizontally).padding(top = 12.dp).shadow(6.dp, CircleShape)
                    ) {
                        Icon(Icons.Default.Edit, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(R.string.video_moment_save_edit))
                    }
                }
            } else if (results != null && !isSearching) {
                Text(
                    stringResource(R.string.media_search_no_results),
                    color = Color.White,
                    modifier = Modifier.padding(16.dp)
                )
            }
        }
        MomentScrubber(
            positionMs = position,
            durationMs = duration,
            playing = playing,
            muted = muted,
            intervals = intervals,
            selectedId = selectedId,
            onToggle = { playback.toggle() },
            onMute = {
                muted = !muted
                playback.setMuted(muted)
            },
            onSeek = { fraction ->
                playback.seekToMs((fraction * duration).toInt())
            },
            modifier = Modifier.align(Alignment.BottomCenter)
        )
    }
}

private class MomentPlayback(context: android.content.Context, uri: Uri) {
    val view = VideoView(context)
    private var pendingEndMs: Int? = null
    var clipEndMs: Int? = null
    private var muted = false
    private var mediaPlayer: MediaPlayer? = null

    init {
        view.setVideoURI(uri)
        view.setOnPreparedListener { player ->
            mediaPlayer = player
            applyMute()
            player.setOnSeekCompleteListener {
                val end = pendingEndMs ?: return@setOnSeekCompleteListener
                pendingEndMs = null
                clipEndMs = end
                view.start()
            }
        }
    }

    fun playClip(startMs: Int, endMs: Int) {
        clipEndMs = null
        if (endMs <= startMs) {
            pendingEndMs = null
            view.seekTo(startMs)
            view.pause()
            return
        }
        pendingEndMs = endMs
        view.seekTo(startMs.coerceAtLeast(0))
    }

    fun userTransport() {
        clipEndMs = null
        pendingEndMs = null
    }

    fun toggle() {
        userTransport()
        if (view.isPlaying) view.pause() else view.start()
    }

    fun seekToMs(ms: Int) {
        userTransport()
        view.seekTo(ms.coerceAtLeast(0))
    }

    fun setMuted(value: Boolean) {
        muted = value
        applyMute()
    }

    fun stopIfClipEnded() {
        val end = clipEndMs ?: return
        if (view.currentPosition >= end - 40) {
            clipEndMs = null
            view.pause()
            view.seekTo(end)
        }
    }

    fun release() {
        view.stopPlayback()
    }

    private fun applyMute() {
        val level = if (muted) 0f else 1f
        mediaPlayer?.setVolume(level, level)
    }
}

@Composable
private fun MomentScrubber(
    positionMs: Int,
    durationMs: Int,
    playing: Boolean,
    muted: Boolean,
    intervals: List<MergedInterval>,
    selectedId: String?,
    onToggle: () -> Unit,
    onMute: () -> Unit,
    onSeek: (Float) -> Unit,
    modifier: Modifier = Modifier
) {
    val fraction = if (durationMs <= 0) 0f else (positionMs.toFloat() / durationMs).coerceIn(0f, 1f)
    val primary = MaterialTheme.colorScheme.primary
    val marker = MaterialTheme.colorScheme.secondaryContainer
    Column(
        modifier.fillMaxWidth().background(
            Brush.verticalGradient(listOf(Color.Transparent, Color.Black.copy(alpha = 0.9f)))
        ).padding(top = 8.dp, bottom = 8.dp)
    ) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            IconButton(onClick = onToggle) {
                Icon(if (playing) Icons.Default.Pause else Icons.Default.PlayArrow, contentDescription = null, tint = Color.White)
            }
            Text("${timecode(positionMs)} / ${timecode(durationMs)}", color = Color.White, style = MaterialTheme.typography.labelMedium)
            IconButton(onClick = onMute) {
                Icon(if (muted) Icons.Default.VolumeOff else Icons.Default.VolumeUp, contentDescription = null, tint = Color.White)
            }
        }
        Canvas(
            Modifier.fillMaxWidth().padding(horizontal = 24.dp).height(48.dp).pointerInput(durationMs) {
                detectTapGestures { offset -> onSeek(offset.x / size.width) }
            }.pointerInput(durationMs) {
                detectDragGestures { change, _ -> onSeek(change.position.x / size.width) }
            }
        ) {
            val trackHeight = 8.dp.toPx()
            val y = (size.height - trackHeight) / 2f
            val radius = CornerRadius(trackHeight / 2f, trackHeight / 2f)
            drawRoundRect(Color.LightGray, Offset(0f, y), Size(size.width, trackHeight), radius)
            val head = size.width * fraction
            if (head > 0f) drawRoundRect(Color.White, Offset(0f, y), Size(head, trackHeight), radius)
            if (durationMs > 0) {
                for (interval in intervals) {
                    val start = (interval.startMs.toFloat() / durationMs).coerceIn(0f, 1f)
                    val end = (interval.endMs.toFloat() / durationMs).coerceIn(0f, 1f)
                    val selected = selectedId != null && selectedId in interval.ids
                    val minWidth = 6.dp.toPx()
                    val width = maxOf(minWidth, size.width * (end - start))
                    val x = (size.width * start).coerceIn(0f, size.width - width)
                    val height = if (selected) trackHeight + 10.dp.toPx() else trackHeight + 6.dp.toPx()
                    val top = (size.height - height) / 2f
                    val color = if (selected) primary else marker
                    val markerRadius = CornerRadius(3.dp.toPx(), 3.dp.toPx())
                    drawRoundRect(color, Offset(x, top), Size(width, height), markerRadius)
                    if (selected) {
                        drawRoundRect(Color.White, Offset(x, top), Size(width, height), markerRadius, style = Stroke(1.5.dp.toPx()))
                    }
                }
            }
            val handleWidth = 4.dp.toPx()
            val handleHeight = trackHeight + 16.dp.toPx()
            val handleX = (head - handleWidth / 2f).coerceIn(0f, size.width - handleWidth)
            val handleRadius = CornerRadius(handleWidth / 2f, handleWidth / 2f)
            drawRoundRect(Color.White, Offset(handleX, (size.height - handleHeight) / 2f), Size(handleWidth, handleHeight), handleRadius)
        }
    }
}

@Composable
private fun MomentThumb(moment: VideoMoment, selected: Boolean, onClick: () -> Unit) {
    val context = LocalContext.current
    var frame by remember(moment.video.id, moment.startMs) { mutableStateOf<ByteArray?>(null) }
    LaunchedEffect(moment.video.id, moment.startMs) {
        frame = withContext(Dispatchers.IO) { frameJpegAt(context, moment.video.uri, moment.startMs.toLong()) }
    }
    Box(
        modifier = Modifier
            .height(120.dp)
            .aspectRatio(0.72f)
            .clip(RoundedCornerShape(8.dp))
            .border(if (selected) 4.dp else 2.dp, if (selected) MaterialTheme.colorScheme.primary else Color.White, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
    ) {
        AsyncImage(model = frame, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
        Text("%.2f".format(moment.score), color = Color.White, style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.TopCenter).padding(top = 4.dp))
        Text(
            "${timecode(moment.startMs)} - ${timecode(moment.endMs)}",
            color = Color.White,
            style = MaterialTheme.typography.labelSmall,
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 4.dp)
        )
    }
}

@Composable
private fun ClipEditDialog(moment: VideoMoment, onDismiss: () -> Unit) {
    var start by remember(moment.startMs) { mutableFloatStateOf(moment.startMs.toFloat()) }
    var end by remember(moment.endMs) { mutableFloatStateOf(moment.endMs.toFloat()) }
    val duration = moment.video.durationMs.toFloat().coerceAtLeast(1f)
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.video_moment_save_edit)) },
        text = {
            Column {
                Text(formatClock(start.toInt()))
                Slider(value = start, onValueChange = { start = it.coerceAtMost(end) }, valueRange = 0f..duration)
                Text(formatClock(end.toInt()))
                Slider(value = end, onValueChange = { end = it.coerceAtLeast(start) }, valueRange = 0f..duration)
            }
        },
        confirmButton = {
            Button(onClick = onDismiss) { Text(stringResource(R.string.video_moment_save)) }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text(stringResource(R.string.cancel)) }
        }
    )
}

@Composable
private fun MomentRow(moment: VideoMoment, onPlay: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp), modifier = Modifier.fillMaxWidth().clickable(onClick = onPlay)) {
        AsyncImage(model = moment.video.uri, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.size(72.dp).clip(RoundedCornerShape(8.dp)))
        Column(Modifier.weight(1f)) {
            Text(moment.video.name, maxLines = 1, fontWeight = FontWeight.SemiBold)
            Text(stringResource(R.string.video_moment_play, formatClock(moment.startMs)))
        }
        Icon(Icons.Default.PlayCircle, contentDescription = null)
    }
}

private fun formatClock(ms: Int): String = timecode(ms)

private fun timecode(ms: Int): String {
    val seconds = (ms / 1000).coerceAtLeast(0)
    return "%02d:%02d".format(seconds / 60, seconds % 60)
}
