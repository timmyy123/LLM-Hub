package com.llmhub.llmhub.screens

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.provider.MediaStore
import android.os.Build
import android.content.ContentValues
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.llmhub.llmhub.R
import com.llmhub.llmhub.service.SDBackendService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.TimeUnit

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImageUpscaleScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    val prefs = remember { context.getSharedPreferences("image_upscale_prefs", android.content.Context.MODE_PRIVATE) }

    var inputBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var outputBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isUpscaling by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var showModelSheet by remember { mutableStateOf(false) }

    // Scan for downloaded upscaler models
    val upscalerModelsDir = remember { File(context.filesDir, "upscaler_models") }
    val availableModels = remember(upscalerModelsDir) {
        if (!upscalerModelsDir.exists()) return@remember emptyList<Pair<String, File>>()
        upscalerModelsDir.listFiles()
            ?.filter { it.isDirectory && File(it, "upscaler.bin").exists() }
            ?.map { Pair(it.name.replace("_", " "), File(it, "upscaler.bin")) }
            ?: emptyList()
    }

    // Restore last selected model from prefs
    val lastModelPath = remember { prefs.getString("last_model_path", null) }
    var selectedModel by remember {
        mutableStateOf<Pair<String, File>?>(
            availableModels.find { it.second.absolutePath == lastModelPath }
        )
    }

    // Persist selection whenever it changes
    LaunchedEffect(selectedModel) {
        prefs.edit().putString("last_model_path", selectedModel?.second?.absolutePath).apply()
    }

    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            try {
                inputBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val source = android.graphics.ImageDecoder.createSource(context.contentResolver, it)
                    android.graphics.ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                        decoder.allocator = android.graphics.ImageDecoder.ALLOCATOR_SOFTWARE
                    }
                } else {
                    @Suppress("DEPRECATION")
                    MediaStore.Images.Media.getBitmap(context.contentResolver, it)
                }
                outputBitmap = null
            } catch (e: Exception) {
                errorMessage = "Failed to load image: ${e.message}"
            }
        }
    }

    LaunchedEffect(errorMessage) {
        errorMessage?.let {
            snackbarHostState.showSnackbar(it)
            errorMessage = null
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.image_upscale_title), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    if (availableModels.isNotEmpty()) {
                        IconButton(onClick = { showModelSheet = true }) {
                            Icon(Icons.Default.Tune, contentDescription = stringResource(R.string.feature_settings_title))
                        }
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->

        when {
            // State 1: No models downloaded
            availableModels.isEmpty() -> {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues)
                        .padding(32.dp),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.AutoFixHigh,
                        contentDescription = null,
                        modifier = Modifier.size(80.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    Text(
                        text = stringResource(R.string.image_upscale_download_model),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = stringResource(R.string.image_upscale_download_model_desc),
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
                        Text(stringResource(R.string.download_models))
                    }
                }
            }

            // State 2: Models available but none selected yet
            selectedModel == null -> {
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
                        imageVector = Icons.Default.ModelTraining,
                        contentDescription = null,
                        modifier = Modifier.size(80.dp),
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Spacer(modifier = Modifier.height(24.dp))
                    Text(
                        text = stringResource(R.string.image_upscale_load_model_title),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = stringResource(R.string.image_upscale_load_model_desc),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(32.dp))
                    FilledTonalButton(
                        onClick = { showModelSheet = true },
                        modifier = Modifier.fillMaxWidth(0.6f)
                    ) {
                        Icon(imageVector = Icons.Default.Tune, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(R.string.feature_settings_title))
                    }
                }
            }

            // State 3: Model selected — main upscale UI
            else -> {
                // Shared zoom/pan state — both images mirror each other
                var scale by remember { mutableStateOf(1f) }
                var offsetX by remember { mutableStateOf(0f) }
                var offsetY by remember { mutableStateOf(0f) }

                LaunchedEffect(outputBitmap) {
                    scale = 1f; offsetX = 0f; offsetY = 0f
                }

                // Zoomable image card — fills its parent's size via the modifier passed in
                @Composable
                fun ZoomableImageCard(
                    bitmap: Bitmap?,
                    label: String,
                    showClose: Boolean = false,
                    onClose: () -> Unit = {},
                    onTap: (() -> Unit)? = null,
                    modifier: Modifier = Modifier
                ) {
                    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Card(
                            modifier = Modifier.fillMaxSize(),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .pointerInput(bitmap) {
                                        detectTransformGestures { _, pan, zoom, _ ->
                                            scale = (scale * zoom).coerceIn(1f, 8f)
                                            val maxX = (size.width * (scale - 1)) / 2f
                                            val maxY = (size.height * (scale - 1)) / 2f
                                            offsetX = (offsetX + pan.x).coerceIn(-maxX, maxX)
                                            offsetY = (offsetY + pan.y).coerceIn(-maxY, maxY)
                                        }
                                    }
                                    .then(if (onTap != null && bitmap == null) Modifier.clickable { onTap() } else Modifier),
                                contentAlignment = Alignment.Center
                            ) {
                                if (bitmap != null) {
                                    Image(
                                        bitmap = bitmap.asImageBitmap(),
                                        contentDescription = null,
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .graphicsLayer(
                                                scaleX = scale,
                                                scaleY = scale,
                                                translationX = offsetX,
                                                translationY = offsetY
                                            ),
                                        contentScale = ContentScale.Fit
                                    )
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
                                            text = stringResource(R.string.image_upscale_select_image),
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                                // Close button on top-right when image is loaded
                                if (showClose && bitmap != null) {
                                    IconButton(
                                        onClick = onClose,
                                        modifier = Modifier
                                            .align(Alignment.TopEnd)
                                            .padding(4.dp)
                                            .size(32.dp)
                                    ) {
                                        Icon(
                                            Icons.Default.Close,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.onSurface
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                BoxWithConstraints(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(paddingValues)
                ) {
                    val isLandscape = maxWidth > maxHeight

                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        // Selected model chip
                        Text(
                            text = selectedModel!!.first,
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.primary
                        )

                        if (isLandscape) {
                            // Landscape: 2 columns side by side filling remaining height
                            Row(
                                modifier = Modifier
                                    .weight(1f)
                                    .fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                ZoomableImageCard(
                                    bitmap = inputBitmap,
                                    label = if (inputBitmap != null) "${inputBitmap!!.width}×${inputBitmap!!.height}" else "",
                                    showClose = !isUpscaling,
                                    onClose = { inputBitmap = null; outputBitmap = null },
                                    onTap = { if (!isUpscaling) imagePickerLauncher.launch("image/*") },
                                    modifier = Modifier.weight(1f).fillMaxHeight()
                                )
                                if (outputBitmap != null) {
                                    ZoomableImageCard(
                                        bitmap = outputBitmap,
                                        label = "${outputBitmap!!.width}×${outputBitmap!!.height}",
                                        modifier = Modifier.weight(1f).fillMaxHeight()
                                    )
                                } else {
                                    // Placeholder right column when no output yet
                                    Box(modifier = Modifier.weight(1f).fillMaxHeight())
                                }
                            }
                        } else {
                            // Portrait: input always occupies top half; output appears below when ready
                            ZoomableImageCard(
                                bitmap = inputBitmap,
                                label = if (inputBitmap != null) "${inputBitmap!!.width}×${inputBitmap!!.height}" else "",
                                showClose = !isUpscaling,
                                onClose = { inputBitmap = null; outputBitmap = null },
                                onTap = { if (!isUpscaling) imagePickerLauncher.launch("image/*") },
                                modifier = Modifier.weight(1f).fillMaxWidth()
                            )
                            if (outputBitmap != null) {
                                ZoomableImageCard(
                                    bitmap = outputBitmap,
                                    label = "${outputBitmap!!.width}×${outputBitmap!!.height}",
                                    modifier = Modifier.weight(1f).fillMaxWidth()
                                )
                            } else {
                                Spacer(modifier = Modifier.weight(1f))
                            }
                        }

                        // Upscale button
                        Button(
                            onClick = {
                                val model = selectedModel ?: return@Button
                                val bitmap = inputBitmap ?: return@Button
                                coroutineScope.launch {
                                    isUpscaling = true
                                    outputBitmap = null
                                    try {
                                        SDBackendService.startUpscaler(context)

                                        val ready = withContext(Dispatchers.IO) {
                                            repeat(20) {
                                                try {
                                                    Socket().use { sock ->
                                                        sock.connect(InetSocketAddress("127.0.0.1", 8081), 500)
                                                    }
                                                    return@withContext true
                                                } catch (_: Exception) {}
                                                delay(300)
                                            }
                                            false
                                        }

                                        if (!ready) {
                                            errorMessage = "Failed to start upscaler backend"
                                            return@launch
                                        }

                                        val result = withContext(Dispatchers.IO) {
                                            val width = bitmap.width
                                            val height = bitmap.height
                                            val pixels = IntArray(width * height)
                                            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
                                            val rgbBytes = ByteArray(width * height * 3)
                                            for (i in pixels.indices) {
                                                val pixel = pixels[i]
                                                rgbBytes[i * 3] = ((pixel shr 16) and 0xFF).toByte()
                                                rgbBytes[i * 3 + 1] = ((pixel shr 8) and 0xFF).toByte()
                                                rgbBytes[i * 3 + 2] = (pixel and 0xFF).toByte()
                                            }

                                            val client = OkHttpClient.Builder()
                                                .connectTimeout(5, TimeUnit.SECONDS)
                                                .readTimeout(300, TimeUnit.SECONDS)
                                                .writeTimeout(60, TimeUnit.SECONDS)
                                                .build()

                                            val requestBody = rgbBytes.toRequestBody("application/octet-stream".toMediaType())
                                            val request = Request.Builder()
                                                .url("http://127.0.0.1:8081/upscale")
                                                .post(requestBody)
                                                .addHeader("X-Image-Width", width.toString())
                                                .addHeader("X-Image-Height", height.toString())
                                                .addHeader("X-Upscaler-Path", model.second.absolutePath)
                                                .build()

                                            val response = client.newCall(request).execute()
                                            if (!response.isSuccessful) {
                                                Log.e("ImageUpscaleScreen", "Upscale HTTP ${response.code}")
                                                return@withContext null
                                            }

                                            val jpegBytes = response.body?.bytes() ?: return@withContext null
                                            BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
                                        }

                                        if (result != null) {
                                            outputBitmap = result
                                        } else {
                                            errorMessage = "Upscaling failed"
                                        }
                                    } catch (e: Exception) {
                                        errorMessage = "Error: ${e.message}"
                                        Log.e("ImageUpscaleScreen", "Upscale error", e)
                                    } finally {
                                        isUpscaling = false
                                        SDBackendService.stop(context)
                                    }
                                }
                            },
                            enabled = inputBitmap != null && !isUpscaling,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            if (isUpscaling) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(18.dp),
                                    strokeWidth = 2.dp,
                                    color = MaterialTheme.colorScheme.onPrimary
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.image_upscale_upscaling))
                            } else {
                                Icon(Icons.Default.AutoFixHigh, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.image_upscale_button))
                            }
                        }

                        // Save button
                        if (outputBitmap != null) {
                            OutlinedButton(
                                onClick = {
                                    coroutineScope.launch {
                                        try {
                                            val values = ContentValues().apply {
                                                put(MediaStore.Images.Media.DISPLAY_NAME, "upscaled_${System.currentTimeMillis()}.jpg")
                                                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                                                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/LLM Hub")
                                                }
                                            }
                                            val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                                            uri?.let {
                                                context.contentResolver.openOutputStream(it)?.use { out ->
                                                    outputBitmap!!.compress(Bitmap.CompressFormat.JPEG, 95, out)
                                                }
                                                snackbarHostState.showSnackbar("Image saved to gallery")
                                            }
                                        } catch (e: Exception) {
                                            errorMessage = "Failed to save: ${e.message}"
                                        }
                                    }
                                },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Icon(Icons.Default.SaveAlt, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.image_upscale_save))
                            }
                        }
                    }
                }
            }
        }
    }

    // Model selection bottom sheet — same structure as ImageGeneratorScreen
    if (showModelSheet) {
        ModalBottomSheet(
            onDismissRequest = { showModelSheet = false },
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
                            text = stringResource(R.string.image_upscale_load_model_title),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )

                        var dropdownExpanded by remember { mutableStateOf(false) }
                        ExposedDropdownMenuBox(
                            expanded = dropdownExpanded,
                            onExpandedChange = { dropdownExpanded = !dropdownExpanded }
                        ) {
                            OutlinedTextField(
                                value = selectedModel?.first ?: stringResource(R.string.select_model),
                                onValueChange = {},
                                readOnly = true,
                                label = { Text(stringResource(R.string.select_model)) },
                                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = dropdownExpanded) },
                                leadingIcon = { Icon(Icons.Default.AutoFixHigh, contentDescription = null) },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .menuAnchor(),
                                colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(),
                                shape = RoundedCornerShape(16.dp)
                            )
                            ExposedDropdownMenu(
                                expanded = dropdownExpanded,
                                onDismissRequest = { dropdownExpanded = false }
                            ) {
                                availableModels.forEach { model ->
                                    DropdownMenuItem(
                                        text = {
                                            Text(
                                                text = model.first,
                                                style = MaterialTheme.typography.bodyMedium,
                                                fontWeight = FontWeight.Medium
                                            )
                                        },
                                        onClick = {
                                            selectedModel = model
                                            dropdownExpanded = false
                                        },
                                        leadingIcon = {
                                            if (selectedModel == model) {
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

                        Button(
                            onClick = { showModelSheet = false },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = selectedModel != null,
                            shape = RoundedCornerShape(16.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.primary
                            )
                        ) {
                            Text(stringResource(R.string.load_model), fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }
    }
}
