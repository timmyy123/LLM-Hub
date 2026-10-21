package com.llmhub.llmhub.screens

import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.view.ViewGroup
import android.widget.FrameLayout
import android.graphics.SurfaceTexture
import android.media.MediaPlayer
import android.view.Surface
import android.view.TextureView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.R
import com.llmhub.llmhub.videogen.VideoGenBackend
import com.llmhub.llmhub.videogen.VideoGenEngine
import com.llmhub.llmhub.videogen.VideoGenProgress
import com.llmhub.llmhub.viewmodels.VideoGeneratorViewModel
import kotlinx.coroutines.launch
import java.io.File
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VideoGeneratorScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    viewModel: VideoGeneratorViewModel = viewModel()
) {
    val context = LocalContext.current
    val snackbarHostState = remember { SnackbarHostState() }
    val coroutineScope = rememberCoroutineScope()

    val availableModels by viewModel.availableModels.collectAsState()
    val loadedModel by viewModel.loadedModel.collectAsState()
    val loadedBackend by viewModel.loadedBackend.collectAsState()
    val progress by viewModel.progress.collectAsState()
    var showModelSheet by remember { mutableStateOf(false) }
    var showInputImageFullscreen by remember { mutableStateOf(false) }

    // Start image picker (image-to-video)
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            try {
                viewModel.initImage = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val source = android.graphics.ImageDecoder.createSource(context.contentResolver, it)
                    android.graphics.ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                        // Software ARGB bitmap so the native engine can read the pixels
                        decoder.allocator = android.graphics.ImageDecoder.ALLOCATOR_SOFTWARE
                    }
                } else {
                    @Suppress("DEPRECATION")
                    MediaStore.Images.Media.getBitmap(context.contentResolver, it)
                }
            } catch (e: Exception) {
                viewModel.initImage = null
                coroutineScope.launch {
                    snackbarHostState.showSnackbar(context.getString(R.string.video_generator_image_load_failed))
                }
            }
        }
    }

    // Refresh model list when screen becomes visible (e.g., after navigating back from downloads)
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
                viewModel.refreshModels()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    // Errors and status messages
    LaunchedEffect(viewModel) {
        viewModel.messages.collect { snackbarHostState.showSnackbar(it) }
    }

    if (showModelSheet) {
        VideoGeneratorSettingsSheet(
            viewModel = viewModel,
            isModelLoaded = loadedModel != null,
            loadedModelName = loadedModel?.name,
            loadedBackend = loadedBackend,
            onDismiss = { showModelSheet = false }
        )
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        stringResource(R.string.feature_video_generator),
                        fontWeight = FontWeight.Bold
                    )
                },
                actions = {
                    IconButton(onClick = { showModelSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = stringResource(R.string.feature_settings_title))
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription = stringResource(R.string.back)
                        )
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        val models = availableModels
        when {
            models == null -> Box(modifier = Modifier.fillMaxSize().padding(paddingValues))
            models.isEmpty() -> {
                // "Download a video model" gate
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
                        .padding(paddingValues)
                        .padding(32.dp),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.Movie,
                        contentDescription = null,
                        modifier = Modifier.size(80.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    Text(
                        text = stringResource(R.string.video_generator_download_model),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = stringResource(R.string.video_generator_download_model_desc),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(32.dp))
                    FilledTonalButton(
                        onClick = onNavigateToModels,
                        modifier = Modifier.fillMaxWidth(0.6f)
                    ) {
                        Icon(
                            imageVector = Icons.Default.GetApp,
                            contentDescription = null
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(R.string.download_models))
                    }
                }
            }
            else -> {
                // Main content - use BoxWithConstraints for landscape detection
                BoxWithConstraints(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues)
                ) {
                    val isLandscape = maxWidth > maxHeight
                    val hasResult = viewModel.result != null

                    val inputs: @Composable ColumnScope.(Int, Int) -> Unit = { minLines, maxLines ->
                        VideoPromptCard(viewModel, minLines, maxLines)
                        VideoImageToVideoCard(
                            viewModel = viewModel,
                            onPickImage = { imagePickerLauncher.launch("image/*") },
                            onShowImage = { showInputImageFullscreen = true }
                        )
                        VideoGenerateButton(viewModel)
                        // Only for a generation (which may load the model first). A load started
                        // from the settings sheet shows its progress in the sheet only.
                        if (viewModel.isGenerating) {
                            VideoProgressCard(
                                progress = progress,
                                onCancel = { viewModel.cancel() },
                                modifier = Modifier.fillMaxWidth()
                            )
                        }
                    }

                    if (isLandscape) {
                        if (!hasResult) {
                            // Horizontally center the prompt column when no video generated yet
                            Box(
                                modifier = Modifier.fillMaxSize().padding(16.dp),
                                contentAlignment = Alignment.TopCenter
                            ) {
                                Column(
                                    modifier = Modifier.fillMaxWidth(0.5f).verticalScroll(rememberScrollState()),
                                    verticalArrangement = Arrangement.spacedBy(16.dp)
                                ) {
                                    inputs(5, 10)
                                }
                            }
                        } else {
                            Row(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(16.dp),
                                horizontalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                Column(
                                    modifier = Modifier
                                        .weight(0.5f)
                                        .fillMaxHeight()
                                        .verticalScroll(rememberScrollState()),
                                    verticalArrangement = Arrangement.spacedBy(16.dp)
                                ) {
                                    inputs(5, 10)
                                }
                                Column(
                                    modifier = Modifier
                                        .weight(0.5f)
                                        .fillMaxHeight()
                                        .verticalScroll(rememberScrollState()),
                                    verticalArrangement = Arrangement.spacedBy(16.dp)
                                ) {
                                    VideoResultSection(viewModel)
                                }
                            }
                        }
                    } else {
                        // Portrait layout
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .verticalScroll(rememberScrollState())
                                .padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            inputs(3, 5)
                            if (hasResult) {
                                VideoResultSection(viewModel)
                            }
                            Spacer(modifier = Modifier.height(32.dp))
                        }
                    }
                }
            }
        }
    }

    // Fullscreen start image viewer
    val fullscreenImage = viewModel.initImage
    if (showInputImageFullscreen && fullscreenImage != null) {
        androidx.compose.ui.window.Dialog(
            onDismissRequest = { showInputImageFullscreen = false },
            properties = androidx.compose.ui.window.DialogProperties(usePlatformDefaultWidth = false)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(androidx.compose.ui.graphics.Color.Black)
                    .clickable { showInputImageFullscreen = false },
                contentAlignment = Alignment.Center
            ) {
                Image(
                    bitmap = fullscreenImage.asImageBitmap(),
                    contentDescription = stringResource(R.string.video_generator_input_image),
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
                Surface(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .statusBarsPadding()
                        .padding(16.dp)
                        .size(48.dp),
                    shape = CircleShape,
                    color = androidx.compose.ui.graphics.Color.Black.copy(alpha = 0.6f)
                ) {
                    IconButton(
                        onClick = { showInputImageFullscreen = false },
                        modifier = Modifier.fillMaxSize()
                    ) {
                        Icon(
                            Icons.Default.Close,
                            contentDescription = stringResource(R.string.close),
                            tint = androidx.compose.ui.graphics.Color.White,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun VideoPromptCard(viewModel: VideoGeneratorViewModel, minLines: Int, maxLines: Int) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = stringResource(R.string.video_generator_prompt_label),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                value = viewModel.prompt,
                onValueChange = { viewModel.prompt = it },
                placeholder = {
                    Text(stringResource(R.string.video_generator_prompt_hint))
                },
                modifier = Modifier.fillMaxWidth(),
                minLines = minLines,
                maxLines = maxLines,
                enabled = !viewModel.isBusy
            )
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = stringResource(R.string.video_generator_prompt_tip),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun VideoImageToVideoCard(
    viewModel: VideoGeneratorViewModel,
    onPickImage: () -> Unit,
    onShowImage: () -> Unit
) {
    val enabled = !viewModel.isBusy
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Same as Image Generator's Image-to-Image: picking a start image switches to image to video.
            Text(
                text = stringResource(R.string.video_generator_image_to_video),
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedButton(
                    onClick = onPickImage,
                    enabled = enabled,
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.Image, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        stringResource(
                            if (viewModel.initImage != null) R.string.video_generator_change_image
                            else R.string.video_generator_select_image
                        )
                    )
                }

                viewModel.initImage?.let { bitmap ->
                    StartImageThumbnail(
                        bitmap = bitmap,
                        enabled = enabled,
                        onClick = onShowImage,
                        onRemoveClick = { viewModel.initImage = null }
                    )
                }
            }
        }
    }
}

@Composable
private fun StartImageThumbnail(
    bitmap: Bitmap,
    enabled: Boolean,
    onClick: () -> Unit,
    onRemoveClick: () -> Unit
) {
    Box(modifier = Modifier.size(64.dp)) {
        Card(
            modifier = Modifier
                .fillMaxSize()
                .clickable { onClick() },
            shape = RoundedCornerShape(8.dp)
        ) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = stringResource(R.string.video_generator_input_image),
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        }
        Surface(
            shape = CircleShape,
            color = if (enabled) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.surfaceVariant,
            contentColor = MaterialTheme.colorScheme.onErrorContainer,
            modifier = Modifier
                .size(24.dp)
                .align(Alignment.TopEnd)
                .offset(x = 6.dp, y = (-6).dp)
                .clickable(enabled = enabled) { onRemoveClick() },
            shadowElevation = 2.dp
        ) {
            Icon(
                imageVector = Icons.Default.Close,
                contentDescription = stringResource(R.string.close),
                modifier = Modifier.padding(4.dp)
            )
        }
    }
}

@Composable
private fun VideoGenerateButton(viewModel: VideoGeneratorViewModel) {
    Button(
        onClick = { viewModel.generate() },
        modifier = Modifier.fillMaxWidth(),
        enabled = viewModel.canGenerate()
    ) {
        if (viewModel.isGenerating) {
            CircularProgressIndicator(
                modifier = Modifier.size(20.dp),
                strokeWidth = 2.dp,
                color = MaterialTheme.colorScheme.onPrimary
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(stringResource(R.string.video_generator_generating))
        } else {
            Icon(Icons.Default.AutoAwesome, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text(stringResource(R.string.video_generator_generate))
        }
    }
}

@Composable
private fun VideoProgressCard(
    progress: VideoGenProgress?,
    onCancel: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            val label = when (progress) {
                is VideoGenProgress.LoadingModel -> stringResource(R.string.video_generator_stage_loading)
                VideoGenProgress.EncodingPrompt -> stringResource(R.string.video_generator_stage_encoding)
                is VideoGenProgress.Denoising -> stringResource(
                    R.string.video_generator_stage_denoising,
                    progress.step,
                    progress.totalSteps
                )
                VideoGenProgress.DecodingFrames -> stringResource(R.string.video_generator_stage_decoding)
                VideoGenProgress.WritingVideo -> stringResource(R.string.video_generator_stage_writing)
                null -> stringResource(R.string.video_generator_generating)
            }
            val fraction: Float? = when (progress) {
                is VideoGenProgress.LoadingModel -> progress.fraction.takeIf { it > 0f }?.coerceIn(0f, 1f)
                is VideoGenProgress.Denoising ->
                    if (progress.totalSteps > 0) (progress.step.toFloat() / progress.totalSteps).coerceIn(0f, 1f) else null
                else -> null
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = label,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.weight(1f)
                )
                when (progress) {
                    is VideoGenProgress.Denoising -> Text(
                        text = "${progress.step}/${progress.totalSteps}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    is VideoGenProgress.LoadingModel -> if (fraction != null) {
                        Text(
                            text = "${(fraction * 100).toInt()}%",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    else -> Unit
                }
            }
            if (fraction != null) {
                LinearProgressIndicator(
                    progress = { fraction },
                    modifier = Modifier.fillMaxWidth()
                )
            } else {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }
            OutlinedButton(
                onClick = onCancel,
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.outlinedButtonColors(
                    contentColor = MaterialTheme.colorScheme.error
                ),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.error)
            ) {
                Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text(stringResource(R.string.cancel), fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun VideoResultSection(viewModel: VideoGeneratorViewModel) {
    val result = viewModel.result ?: return
    val (resultWidth, resultHeight) = viewModel.resultSize
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerLow
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = stringResource(R.string.video_generator_result_title),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            LoopingVideoPlayer(
                file = result.videoFile,
                thumbnail = result.thumbnail,
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(resultWidth.toFloat() / resultHeight.coerceAtLeast(1))
                    .clip(RoundedCornerShape(12.dp))
            )
            Text(
                text = stringResource(
                    R.string.video_generator_result_info,
                    result.frameCount,
                    result.fps,
                    String.format(Locale.getDefault(), "%.1f", result.elapsedMs / 1000f)
                ),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }

    // Save action (outside Card for equal width with Generate button)
    Button(
        onClick = { viewModel.saveToGallery() },
        modifier = Modifier.fillMaxWidth()
    ) {
        Icon(Icons.Default.Save, contentDescription = null)
        Spacer(modifier = Modifier.width(8.dp))
        Text(stringResource(R.string.video_generator_save))
    }
}

/** Muted, looping inline player for the generated mp4. Tap toggles pause/play. */
@Composable
private fun LoopingVideoPlayer(
    file: File,
    thumbnail: Bitmap?,
    modifier: Modifier = Modifier
) {
    var isPrepared by remember(file.absolutePath) { mutableStateOf(false) }
    // TextureView, not VideoView: a SurfaceView sits outside the Compose layer, so it stayed on
    // screen during the exit animation instead of fading with the rest of the screen.
    val player = remember(file.absolutePath, file.lastModified()) { MediaPlayer() }

    Box(
        modifier = modifier.background(androidx.compose.ui.graphics.Color.Black),
        contentAlignment = Alignment.Center
    ) {
        key(file.absolutePath, file.lastModified()) {
            AndroidView(
                factory = { ctx ->
                    TextureView(ctx).apply {
                        layoutParams = FrameLayout.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT,
                            ViewGroup.LayoutParams.MATCH_PARENT
                        )
                        surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                            override fun onSurfaceTextureAvailable(texture: SurfaceTexture, width: Int, height: Int) {
                                runCatching {
                                    player.setSurface(Surface(texture))
                                    player.setDataSource(file.absolutePath)
                                    player.isLooping = true
                                    player.setVolume(0f, 0f)
                                    player.setOnVideoSizeChangedListener { _, w, h -> fitVideo(this@apply, w, h) }
                                    player.setOnPreparedListener {
                                        fitVideo(this@apply, it.videoWidth, it.videoHeight)
                                        isPrepared = true
                                        it.start()
                                    }
                                    player.setOnErrorListener { _, _, _ -> true }
                                    player.prepareAsync()
                                }
                            }
                            override fun onSurfaceTextureSizeChanged(texture: SurfaceTexture, width: Int, height: Int) {
                                runCatching { fitVideo(this@apply, player.videoWidth, player.videoHeight) }
                            }
                            override fun onSurfaceTextureDestroyed(texture: SurfaceTexture): Boolean = true
                            override fun onSurfaceTextureUpdated(texture: SurfaceTexture) {}
                        }
                        setOnClickListener {
                            if (isPrepared) { if (player.isPlaying) player.pause() else player.start() }
                        }
                    }
                },
                modifier = Modifier.fillMaxSize()
            )
        }
        if (!isPrepared && thumbnail != null) {
            Image(
                bitmap = thumbnail.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit
            )
        }
    }

    DisposableEffect(player) {
        onDispose { player.release() }
    }
}

/** Letterboxes the video inside the TextureView (VideoView did this itself). */
private fun fitVideo(view: TextureView, videoWidth: Int, videoHeight: Int) {
    if (videoWidth <= 0 || videoHeight <= 0 || view.width <= 0 || view.height <= 0) return
    val viewWidth = view.width.toFloat()
    val viewHeight = view.height.toFloat()
    val scale = minOf(viewWidth / videoWidth, viewHeight / videoHeight)
    val matrix = android.graphics.Matrix()
    matrix.setScale(videoWidth * scale / viewWidth, videoHeight * scale / viewHeight, viewWidth / 2, viewHeight / 2)
    view.setTransform(matrix)
}

private fun backendLabelRes(backend: VideoGenBackend): Int = when (backend) {
    VideoGenBackend.NPU -> R.string.video_generator_backend_npu
    VideoGenBackend.GPU -> R.string.video_generator_backend_gpu
    VideoGenBackend.CPU -> R.string.backend_cpu
}

private fun backendIcon(backend: VideoGenBackend) = when (backend) {
    VideoGenBackend.NPU -> Icons.Default.Bolt
    VideoGenBackend.GPU -> Icons.Default.Speed
    VideoGenBackend.CPU -> Icons.Default.Computer
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun VideoGeneratorSettingsSheet(
    viewModel: VideoGeneratorViewModel,
    isModelLoaded: Boolean,
    loadedModelName: String?,
    loadedBackend: VideoGenBackend?,
    onDismiss: () -> Unit
) {
    val availableModels = viewModel.availableModels.collectAsState().value.orEmpty()
    val selectedModel = viewModel.selectedModel
    val busy = viewModel.isBusy

    ModalBottomSheet(
        onDismissRequest = onDismiss,
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

            // Model Selector Card - matching the Image Generator sheet
            Card(
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        text = stringResource(R.string.select_model_title),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )

                    if (availableModels.isEmpty()) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Default.Warning,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.error
                            )
                            Column {
                                Text(
                                    text = stringResource(R.string.no_models_available),
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.Medium
                                )
                                Text(
                                    text = stringResource(R.string.download_models_first),
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    } else {
                        // Model dropdown
                        var modelExpanded by remember { mutableStateOf(false) }
                        ExposedDropdownMenuBox(
                            expanded = modelExpanded,
                            onExpandedChange = { if (!busy) modelExpanded = !modelExpanded }
                        ) {
                            OutlinedTextField(
                                value = selectedModel?.name ?: stringResource(R.string.select_model),
                                onValueChange = {},
                                readOnly = true,
                                enabled = !busy,
                                label = { Text(stringResource(R.string.select_model)) },
                                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = modelExpanded) },
                                leadingIcon = {
                                    Icon(Icons.Default.Memory, contentDescription = null)
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .menuAnchor(),
                                colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(),
                                shape = RoundedCornerShape(16.dp)
                            )
                            ExposedDropdownMenu(
                                expanded = modelExpanded,
                                onDismissRequest = { modelExpanded = false }
                            ) {
                                availableModels.forEach { model ->
                                    DropdownMenuItem(
                                        text = {
                                            Text(
                                                text = model.name,
                                                style = MaterialTheme.typography.bodyMedium,
                                                fontWeight = FontWeight.Medium
                                            )
                                        },
                                        onClick = {
                                            viewModel.selectModel(model)
                                            modelExpanded = false
                                        },
                                        leadingIcon = {
                                            if (selectedModel?.name == model.name) {
                                                Icon(
                                                    Icons.Default.CheckCircle,
                                                    contentDescription = null,
                                                    tint = MaterialTheme.colorScheme.primary
                                                )
                                            }
                                        }
                                    )
                                }
                            }
                        }

                        // Backend dropdown (limited to what the selected model supports)
                        val supportedBackends = selectedModel?.let { VideoGenEngine.supportedBackends(it) }.orEmpty()
                        val selectedBackend = viewModel.selectedBackend
                        if (supportedBackends.isNotEmpty()) {
                            var backendExpanded by remember { mutableStateOf(false) }
                            ExposedDropdownMenuBox(
                                expanded = backendExpanded,
                                onExpandedChange = { if (!busy) backendExpanded = !backendExpanded }
                            ) {
                                OutlinedTextField(
                                    value = selectedBackend?.let { stringResource(backendLabelRes(it)) } ?: "",
                                    onValueChange = {},
                                    readOnly = true,
                                    enabled = !busy,
                                    label = { Text(stringResource(R.string.select_backend)) },
                                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = backendExpanded) },
                                    leadingIcon = {
                                        Icon(
                                            backendIcon(selectedBackend ?: supportedBackends.first()),
                                            contentDescription = null
                                        )
                                    },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .menuAnchor(),
                                    colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(),
                                    shape = RoundedCornerShape(16.dp)
                                )
                                ExposedDropdownMenu(
                                    expanded = backendExpanded,
                                    onDismissRequest = { backendExpanded = false }
                                ) {
                                    supportedBackends.forEach { backend ->
                                        DropdownMenuItem(
                                            text = { Text(stringResource(backendLabelRes(backend))) },
                                            onClick = {
                                                viewModel.selectBackend(backend)
                                                backendExpanded = false
                                            },
                                            leadingIcon = {
                                                Icon(
                                                    if (backend == selectedBackend) Icons.Default.CheckCircle else backendIcon(backend),
                                                    contentDescription = null,
                                                    tint = if (backend == selectedBackend) MaterialTheme.colorScheme.primary
                                                    else MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                        )
                                    }
                                }
                            }
                        }

                        if (isModelLoaded && loadedModelName != null && loadedBackend != null) {
                            Text(
                                text = stringResource(
                                    R.string.video_generator_loaded_status,
                                    loadedModelName,
                                    stringResource(backendLabelRes(loadedBackend))
                                ),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        // Load/Reload Model Button
                        Button(
                            onClick = { viewModel.loadModel() },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = selectedModel != null && !busy,
                            shape = RoundedCornerShape(16.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.primary
                            )
                        ) {
                            if (viewModel.isLoading) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    color = MaterialTheme.colorScheme.onPrimary
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.model_loading), fontWeight = FontWeight.Bold)
                            } else {
                                Text(
                                    text = stringResource(
                                        if (isModelLoaded) R.string.reload_model
                                        else R.string.load_model
                                    ),
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }

                        viewModel.loadError?.let { error ->
                            Text(
                                text = error,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.error
                            )
                        }

                        // Unload Model Button (only show when model is loaded)
                        if (isModelLoaded) {
                            OutlinedButton(
                                onClick = { viewModel.unloadModel() },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(16.dp),
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = MaterialTheme.colorScheme.error
                                ),
                                border = BorderStroke(1.dp, MaterialTheme.colorScheme.error)
                            ) {
                                Icon(
                                    Icons.Default.PowerOff,
                                    contentDescription = null,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.unload_model), fontWeight = FontWeight.Bold)
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Generation settings card
            Card(
                shape = RoundedCornerShape(24.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Steps slider (4-8)
                    Column {
                        Text(
                            text = "${stringResource(R.string.video_generator_steps)}: ${viewModel.steps}",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                        Slider(
                            value = viewModel.steps.toFloat(),
                            onValueChange = { viewModel.updateSteps(Math.round(it)) },
                            valueRange = VideoGeneratorViewModel.MIN_STEPS.toFloat()..VideoGeneratorViewModel.MAX_STEPS.toFloat(),
                            steps = VideoGeneratorViewModel.MAX_STEPS - VideoGeneratorViewModel.MIN_STEPS - 1,
                            enabled = !busy
                        )
                    }

                    val dimensionRange = VideoGeneratorViewModel.MIN_DIMENSION.toFloat()..VideoGeneratorViewModel.MAX_DIMENSION.toFloat()
                    val dimensionSteps = (VideoGeneratorViewModel.MAX_DIMENSION - VideoGeneratorViewModel.MIN_DIMENSION) /
                        VideoGeneratorViewModel.DIMENSION_STEP - 1

                    // Width slider (multiples of 32)
                    Column {
                        Text(
                            text = "${stringResource(R.string.video_generator_width)}: ${viewModel.width}",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                        Slider(
                            value = viewModel.width.toFloat(),
                            onValueChange = { viewModel.updateWidth(Math.round(it)) },
                            valueRange = dimensionRange,
                            steps = dimensionSteps,
                            enabled = !busy && !viewModel.sizeLocked
                        )
                    }

                    // Height slider (multiples of 32)
                    Column {
                        Text(
                            text = "${stringResource(R.string.video_generator_height)}: ${viewModel.height}",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                        Slider(
                            value = viewModel.height.toFloat(),
                            onValueChange = { viewModel.updateHeight(Math.round(it)) },
                            valueRange = dimensionRange,
                            steps = dimensionSteps,
                            enabled = !busy && !viewModel.sizeLocked
                        )
                    }

                    // Frames slider (8n+1)
                    Column {
                        Text(
                            text = "${stringResource(R.string.video_generator_frames)}: ${viewModel.numFrames}",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = stringResource(
                                R.string.video_generator_duration,
                                viewModel.numFrames.toFloat() / viewModel.fps.coerceAtLeast(1)
                            ),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Slider(
                            value = viewModel.numFrames.toFloat(),
                            onValueChange = { viewModel.updateFrames(Math.round(it)) },
                            valueRange = VideoGeneratorViewModel.MIN_FRAMES.toFloat()..VideoGeneratorViewModel.MAX_FRAMES.toFloat(),
                            steps = (VideoGeneratorViewModel.MAX_FRAMES - VideoGeneratorViewModel.MIN_FRAMES) / 8 - 1,
                            enabled = !busy && !viewModel.sizeLocked
                        )
                        if (viewModel.sizeLocked) {
                            Text(
                                text = stringResource(
                                    R.string.video_generator_npu_fixed_size,
                                    viewModel.width,
                                    viewModel.height,
                                    viewModel.numFrames
                                ),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }

                    // FPS slider
                    Column {
                        Text(
                            text = "${stringResource(R.string.video_generator_fps)}: ${viewModel.fps}",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium
                        )
                        Slider(
                            value = viewModel.fps.toFloat(),
                            onValueChange = { viewModel.updateFps(Math.round(it)) },
                            valueRange = VideoGeneratorViewModel.MIN_FPS.toFloat()..VideoGeneratorViewModel.MAX_FPS.toFloat(),
                            steps = VideoGeneratorViewModel.MAX_FPS - VideoGeneratorViewModel.MIN_FPS - 1,
                            enabled = !busy
                        )
                    }

                    // Seed input with random button
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        OutlinedTextField(
                            value = viewModel.seed.toString(),
                            onValueChange = { text ->
                                if (text.isEmpty()) viewModel.updateSeed(0L)
                                else text.toLongOrNull()?.let { viewModel.updateSeed(it) }
                            },
                            label = { Text(stringResource(R.string.video_generator_seed)) },
                            modifier = Modifier.weight(1f),
                            enabled = !busy,
                            singleLine = true,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            shape = RoundedCornerShape(16.dp)
                        )
                        IconButton(
                            onClick = { viewModel.randomizeSeed() },
                            enabled = !busy
                        ) {
                            Icon(
                                Icons.Default.Refresh,
                                contentDescription = stringResource(R.string.video_generator_random_seed)
                            )
                        }
                    }

                    // Optional negative prompt
                    OutlinedTextField(
                        value = viewModel.negativePrompt,
                        onValueChange = { viewModel.updateNegativePrompt(it) },
                        label = { Text(stringResource(R.string.video_generator_negative_prompt)) },
                        placeholder = { Text(stringResource(R.string.video_generator_negative_prompt_hint)) },
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !busy,
                        minLines = 2,
                        maxLines = 4,
                        shape = RoundedCornerShape(16.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}
