package com.llmhub.llmhub.mediasearch

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.llmhub.llmhub.R
import kotlinx.coroutines.Dispatchers
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
    var playing by remember { mutableStateOf<VideoMoment?>(null) }
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
            else -> {
                LazyColumn(
                    modifier = Modifier.padding(padding).imePadding(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    item { MediaSearchField(query, viewModel::onQueryChange, stringResource(R.string.video_moment_hint), isSearching) }
                    item {
                        MediaSearchStatusCard(
                            isLoadingModel = isLoadingModel,
                            modelError = modelError,
                            progress = progress,
                            isPaused = isPaused,
                            onPause = viewModel::pause,
                            onResume = viewModel::resume,
                            onRetryModel = viewModel::loadModel
                        )
                    }
                    if (results != null && results!!.isEmpty() && !isSearching) {
                        item { Text(stringResource(R.string.media_search_no_results)) }
                    }
                    if (!results.isNullOrEmpty()) {
                        item { Text(stringResource(R.string.video_moment_top), color = MaterialTheme.colorScheme.primary) }
                        item {
                            LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                items(results.orEmpty(), key = { "${it.video.id}:${it.startMs}" }) { moment ->
                                    MomentThumb(moment, selected = selected?.startMs == moment.startMs && selected?.video?.id == moment.video.id) {
                                        selected = moment
                                        playing = moment
                                    }
                                }
                            }
                        }
                    }
                    if (selected != null) {
                        item {
                            Button(onClick = { editing = selected }, modifier = Modifier.fillMaxWidth()) {
                                Icon(Icons.Default.Edit, contentDescription = null)
                                Spacer(Modifier.width(8.dp))
                                Text(stringResource(R.string.video_moment_save_edit))
                            }
                        }
                    }
                }
            }
        }
    }

    editing?.let { moment ->
        ClipEditDialog(moment, onDismiss = { editing = null })
    }

    playing?.let { moment ->
        Dialog(onDismissRequest = { playing = null }, properties = DialogProperties(usePlatformDefaultWidth = false)) {
            Surface(Modifier.fillMaxSize()) {
                Box {
                    AndroidView(
                        factory = { ctx ->
                            android.widget.VideoView(ctx).apply {
                                setVideoURI(moment.video.uri)
                                setOnPreparedListener {
                                    seekTo(moment.startMs)
                                    start()
                                }
                            }
                        },
                        modifier = Modifier.fillMaxSize()
                    )
                    IconButton(onClick = { playing = null }, modifier = Modifier.align(Alignment.TopEnd).padding(16.dp)) {
                        Icon(Icons.Default.Close, contentDescription = stringResource(R.string.close))
                    }
                }
            }
        }
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
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.clickable { onOpen(video) }) {
                Box(Modifier.fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(16.dp))) {
                    AsyncImage(model = video.uri, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                    Text(formatClock(video.durationMs.toInt()), color = androidx.compose.ui.graphics.Color.White, modifier = Modifier.align(Alignment.BottomStart).padding(8.dp))
                }
                Text(video.name, maxLines = 1, style = MaterialTheme.typography.bodyMedium)
            }
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
            .height(110.dp)
            .aspectRatio(0.72f)
            .clip(RoundedCornerShape(8.dp))
            .border(if (selected) 3.dp else 1.dp, if (selected) MaterialTheme.colorScheme.primary else Color.White, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
    ) {
        AsyncImage(model = frame, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
        Text("%.2f".format(moment.score), color = Color.White, style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.TopCenter).padding(top = 4.dp))
        Text(formatClock(moment.startMs), color = Color.White, style = MaterialTheme.typography.labelSmall, modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 4.dp))
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

private fun formatClock(ms: Int): String {
    val seconds = (ms / 1000).coerceAtLeast(0)
    return "%d:%02d".format(seconds / 60, seconds % 60)
}
