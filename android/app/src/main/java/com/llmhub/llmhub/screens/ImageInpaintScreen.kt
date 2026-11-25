package com.llmhub.llmhub.screens

import androidx.compose.foundation.layout.Row
import androidx.compose.material3.Surface
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.material3.ModalBottomSheet
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.components.ModelSelectorCard
import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.InpaintKind
import com.llmhub.llmhub.viewmodels.BrushStroke
import com.llmhub.llmhub.viewmodels.ImageInpaintViewModel
import com.llmhub.llmhub.viewmodels.InpaintStatus
import kotlinx.coroutines.launch
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Object Eraser: pick a photo, paint over what to remove, erase it on the device with MI-GAN (Quick) or
 * LaMa (HD), compare, erase again, save or share. Leaving the screen stops a running erase and closes the model.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImageInpaintScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    viewModel: ImageInpaintViewModel = viewModel()
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }
    val status by viewModel.status.collectAsState()
    val onCpu by viewModel.onCpu.collectAsState()
    val busy = status != InpaintStatus.Idle
    val photo = viewModel.photo
    var showSettingsSheet by remember { mutableStateOf(false) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { viewModel.loadPhoto(it) }
    }
    fun pickPhoto() {
        if (!busy) picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }

    // Re-check the downloaded models when the screen comes back (e.g. from the download screen).
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) viewModel.refreshInstalled()
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }
    // Leaving the screen stops a running erase and closes the model.
    DisposableEffect(Unit) {
        onDispose { viewModel.release() }
    }
    LaunchedEffect(viewModel) {
        viewModel.messages.collect { snackbarHostState.showSnackbar(it) }
    }

    // Settings sheet: the same model and backend selector as the other features (GPU or CPU, no NPU).
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
                    models = viewModel.installedModels,
                    selectedModel = viewModel.selectedModel,
                    selectedBackend = if (viewModel.useGpu) LlmInference.Backend.GPU else LlmInference.Backend.CPU,
                    selectedNpuDeviceId = null,
                    onModelSelected = viewModel::selectModel,
                    onBackendSelected = { backend, _ -> viewModel.selectBackend(backend == LlmInference.Backend.GPU) },
                    onLoadModel = viewModel::loadModel,
                    isLoading = viewModel.isLoadingModel,
                    isModelLoaded = viewModel.isModelLoaded,
                    onUnloadModel = viewModel::unloadModel,
                    filterMultimodalOnly = false
                )
                Spacer(modifier = Modifier.height(24.dp))
            }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.feature_image_inpainting), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = stringResource(R.string.feature_settings_title))
                    }
                }
            )
        },
        bottomBar = {
            if (viewModel.installedChecked && viewModel.installed.isNotEmpty()) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shadowElevation = 8.dp,
                    color = MaterialTheme.colorScheme.surface
                ) {
                    EraseButton(
                        status = status,
                        isLoadingModel = viewModel.isLoadingModel,
                        enabled = photo != null && viewModel.strokes.isNotEmpty() && !busy && !viewModel.isLoadingModel,
                        onClick = viewModel::erase
                    )
                }
            }
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        when {
            !viewModel.installedChecked -> Box(Modifier.fillMaxSize().padding(paddingValues))

            // Gate: no eraser model downloaded yet.
            viewModel.installed.isEmpty() -> {
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
                        imageVector = Icons.Default.Healing,
                        contentDescription = null,
                        modifier = Modifier.size(80.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    Text(
                        text = stringResource(R.string.image_inpainting_download_model),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = stringResource(R.string.image_inpainting_download_model_desc),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(32.dp))
                    FilledTonalButton(
                        onClick = onNavigateToModels,
                        modifier = Modifier.fillMaxWidth(0.6f)
                    ) {
                        Icon(Icons.Default.GetApp, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(R.string.download_models_title))
                    }
                }
            }

            else -> {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues)
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    var showBefore by remember { mutableStateOf(false) }
                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        if (photo == null) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .clickable(enabled = !viewModel.isLoadingPhoto) { pickPhoto() },
                                contentAlignment = Alignment.Center
                            ) {
                                if (viewModel.isLoadingPhoto) {
                                    CircularProgressIndicator()
                                } else {
                                    Column(
                                        horizontalAlignment = Alignment.CenterHorizontally,
                                        verticalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        Icon(
                                            Icons.Default.AddPhotoAlternate,
                                            contentDescription = null,
                                            modifier = Modifier.size(40.dp),
                                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                        Text(
                                            text = stringResource(R.string.image_inpainting_pick_photo),
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            }
                        } else {
                            Box(Modifier.fillMaxSize()) {
                                val shown = if (showBefore) viewModel.previous ?: photo else photo
                                PaintCanvas(
                                    bitmap = shown,
                                    strokes = if (showBefore) emptyList() else viewModel.strokes,
                                    brushSizeDp = viewModel.brushSizeDp,
                                    enabled = !busy && !showBefore,
                                    onStroke = viewModel::addStroke,
                                    modifier = Modifier.fillMaxSize()
                                )
                                if (showBefore) {
                                    Surface(
                                        modifier = Modifier.align(Alignment.TopStart).padding(8.dp),
                                        shape = RoundedCornerShape(8.dp),
                                        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f)
                                    ) {
                                        Text(
                                            text = stringResource(R.string.image_inpainting_before),
                                            style = MaterialTheme.typography.labelMedium,
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                        )
                                    }
                                }
                                if (viewModel.previous != null && !busy) {
                                    Surface(
                                        modifier = Modifier
                                            .align(Alignment.TopEnd)
                                            .padding(8.dp)
                                            .size(44.dp)
                                            .clip(CircleShape)
                                            .pointerInput(Unit) {
                                                detectTapGestures(onPress = {
                                                    showBefore = true
                                                    try {
                                                        tryAwaitRelease()
                                                    } finally {
                                                        showBefore = false
                                                    }
                                                })
                                            },
                                        shape = CircleShape,
                                        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.85f)
                                    ) {
                                        Box(contentAlignment = Alignment.Center) {
                                            Icon(
                                                Icons.Default.Compare,
                                                contentDescription = stringResource(R.string.image_inpainting_compare),
                                                tint = MaterialTheme.colorScheme.onSurface
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (photo != null) {
                        val hint = if (busy && onCpu) {
                            stringResource(R.string.image_inpainting_cpu_note)
                        } else {
                            stringResource(R.string.image_inpainting_paint_hint)
                        }
                        Text(
                            text = hint,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        BrushControls(viewModel = viewModel, enabled = !busy, onChangePhoto = { pickPhoto() })
                    }

                    if (photo != null && viewModel.hasEdits) {
                        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                            OutlinedButton(
                                onClick = viewModel::saveToGallery,
                                enabled = !busy,
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(Icons.Default.SaveAlt, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.image_inpainting_save))
                            }
                            val shareFailed = stringResource(R.string.image_inpainting_share_failed)
                            OutlinedButton(
                                onClick = {
                                    scope.launch {
                                        val intent = viewModel.shareIntent()
                                        if (intent != null) {
                                            runCatching { context.startActivity(intent) }
                                                .onFailure { snackbarHostState.showSnackbar(shareFailed) }
                                        } else {
                                            snackbarHostState.showSnackbar(shareFailed)
                                        }
                                    }
                                },
                                enabled = !busy,
                                modifier = Modifier.weight(1f)
                            ) {
                                Icon(Icons.Default.Share, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.image_inpainting_share))
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun BrushControls(viewModel: ImageInpaintViewModel, enabled: Boolean, onChangePhoto: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            Icons.Default.Brush,
            contentDescription = stringResource(R.string.image_inpainting_brush_size),
            tint = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Slider(
            value = viewModel.brushSizeDp,
            onValueChange = viewModel::setBrushSize,
            valueRange = ImageInpaintViewModel.MIN_BRUSH_DP..ImageInpaintViewModel.MAX_BRUSH_DP,
            enabled = enabled,
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 8.dp)
        )
        IconButton(onClick = viewModel::undo, enabled = enabled && viewModel.canUndo) {
            Icon(Icons.AutoMirrored.Filled.Undo, contentDescription = stringResource(R.string.image_inpainting_undo))
        }
        IconButton(onClick = viewModel::clearStrokes, enabled = enabled && viewModel.strokes.isNotEmpty()) {
            Icon(Icons.Default.LayersClear, contentDescription = stringResource(R.string.image_inpainting_clear))
        }
        IconButton(onClick = onChangePhoto, enabled = enabled) {
            Icon(Icons.Default.PhotoLibrary, contentDescription = stringResource(R.string.image_inpainting_change_photo))
        }
    }
}

@Composable
private fun EraseButton(status: InpaintStatus, isLoadingModel: Boolean, enabled: Boolean, onClick: () -> Unit) {
    Column {
        if (status is InpaintStatus.Erasing && status.total > 0) {
            LinearProgressIndicator(
                progress = { status.done.toFloat() / status.total },
                modifier = Modifier.fillMaxWidth()
            )
        }
        FilledTonalButton(
            onClick = onClick,
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            shape = RoundedCornerShape(12.dp),
            enabled = enabled
        ) {
            when {
                // No spinner while a model loads: animating then can cause an ANR on the GPU (as in the other features).
                isLoadingModel || status == InpaintStatus.Loading -> Text(stringResource(R.string.model_loading))
                status is InpaintStatus.Erasing -> {
                    CircularProgressIndicator(modifier = Modifier.size(22.dp), strokeWidth = 2.5.dp)
                    Spacer(modifier = Modifier.width(10.dp))
                    Text(
                        stringResource(
                            R.string.image_inpainting_erasing,
                            min(status.done + 1, status.total.coerceAtLeast(1)),
                            status.total.coerceAtLeast(1)
                        )
                    )
                }
                else -> {
                    Icon(Icons.Default.AutoAwesome, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.image_inpainting_erase))
                }
            }
        }
    }
}

/**
 * The photo with the brush strokes over it. One finger paints (in photo pixels, so the brush keeps its on-screen
 * size at any zoom); two fingers zoom and pan, and cancel the stroke they interrupt.
 */
@Composable
private fun PaintCanvas(
    bitmap: Bitmap,
    strokes: List<BrushStroke>,
    brushSizeDp: Float,
    enabled: Boolean,
    onStroke: (BrushStroke) -> Unit,
    modifier: Modifier = Modifier
) {
    val image = remember(bitmap) { bitmap.asImageBitmap() }
    val imageWidth = bitmap.width
    val imageHeight = bitmap.height
    var scale by remember(imageWidth, imageHeight) { mutableFloatStateOf(1f) }
    var offset by remember(imageWidth, imageHeight) { mutableStateOf(Offset.Zero) }
    var boxSize by remember { mutableStateOf(IntSize.Zero) }
    val current = remember { mutableStateListOf<Offset>() }
    var currentRadius by remember { mutableFloatStateOf(0f) }
    val brushPx = with(LocalDensity.current) { brushSizeDp.dp.toPx() }
    val strokeColor = Color(0xFFFF3B5C)

    // Fit: scale from photo pixels to the (unzoomed) box, and the photo's top-left corner in the box.
    val fitScale = if (boxSize.width > 0 && boxSize.height > 0) {
        min(boxSize.width.toFloat() / imageWidth, boxSize.height.toFloat() / imageHeight)
    } else 1f
    val left = (boxSize.width - imageWidth * fitScale) / 2f
    val top = (boxSize.height - imageHeight * fitScale) / 2f
    val currentFit by rememberUpdatedState(Triple(fitScale, left, top))
    val currentOnStroke by rememberUpdatedState(onStroke)

    Box(
        modifier = modifier
            .clipToBounds()
            .onSizeChanged { boxSize = it }
            .pointerInput(enabled, brushPx, imageWidth, imageHeight) {
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    val (fs, l, t) = currentFit
                    fun toImage(p: Offset): Offset {
                        val cx = size.width / 2f
                        val cy = size.height / 2f
                        val qx = cx + (p.x - cx - offset.x) / scale
                        val qy = cy + (p.y - cy - offset.y) / scale
                        return Offset(
                            ((qx - l) / fs).coerceIn(0f, imageWidth - 1f),
                            ((qy - t) / fs).coerceIn(0f, imageHeight - 1f)
                        )
                    }
                    var multiTouch = false
                    current.clear()
                    if (enabled) {
                        currentRadius = brushPx / 2f / (fs * scale)
                        current.add(toImage(down.position))
                        down.consume()
                    }
                    while (true) {
                        val event = awaitPointerEvent()
                        val pressed = event.changes.filter { it.pressed }
                        if (pressed.isEmpty()) break
                        if (pressed.size >= 2) {
                            if (!multiTouch) {
                                multiTouch = true
                                current.clear()
                            }
                            val zoom = event.calculateZoom()
                            val pan = event.calculatePan()
                            val newScale = (scale * zoom).coerceIn(1f, 8f)
                            val maxX = size.width * (newScale - 1f) / 2f
                            val maxY = size.height * (newScale - 1f) / 2f
                            offset = Offset(
                                (offset.x + pan.x).coerceIn(-maxX, maxX),
                                (offset.y + pan.y).coerceIn(-maxY, maxY)
                            )
                            scale = newScale
                            event.changes.forEach { it.consume() }
                        } else if (!multiTouch && enabled) {
                            val change = pressed.first()
                            current.add(toImage(change.position))
                            change.consume()
                        }
                    }
                    if (!multiTouch && enabled && current.isNotEmpty()) {
                        currentOnStroke(BrushStroke(current.toList(), currentRadius))
                    }
                    current.clear()
                }
            }
    ) {
        val zoomed = Modifier
            .fillMaxSize()
            .graphicsLayer(scaleX = scale, scaleY = scale, translationX = offset.x, translationY = offset.y)
        Canvas(zoomed) {
            drawImage(
                image = image,
                dstOffset = IntOffset(left.roundToInt(), top.roundToInt()),
                dstSize = IntSize((imageWidth * fitScale).roundToInt(), (imageHeight * fitScale).roundToInt()),
                filterQuality = FilterQuality.Medium
            )
        }
        // Strokes drawn opaque in their own layer and faded together, so overlaps do not darken.
        Canvas(zoomed.graphicsLayer(alpha = 0.5f, compositingStrategy = CompositingStrategy.Offscreen)) {
            fun drawStroke(points: List<Offset>, radius: Float) {
                if (points.isEmpty()) return
                val first = Offset(left + points[0].x * fitScale, top + points[0].y * fitScale)
                if (points.size == 1) {
                    drawCircle(strokeColor, radius = radius * fitScale, center = first)
                    return
                }
                val path = Path().apply {
                    moveTo(first.x, first.y)
                    for (i in 1 until points.size) lineTo(left + points[i].x * fitScale, top + points[i].y * fitScale)
                }
                drawPath(
                    path,
                    strokeColor,
                    style = Stroke(width = radius * 2f * fitScale, cap = StrokeCap.Round, join = StrokeJoin.Round)
                )
            }
            strokes.forEach { drawStroke(it.points, it.radius) }
            drawStroke(current, currentRadius)
        }
    }
}
