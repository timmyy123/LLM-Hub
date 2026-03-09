package com.llmhub.llmhub.screens

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.llmhub.llmhub.R
import com.llmhub.llmhub.imagegen.ImageGeneratorHelper
import com.llmhub.llmhub.imagegen.ModelInfo
import com.llmhub.llmhub.imagegen.ModelType
import com.llmhub.llmhub.imagegen.StableDiffusionHelper
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.view.ViewGroup
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.window.DialogWindowProvider

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ImageGeneratorScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val scrollState = rememberScrollState()
    val snackbarHostState = remember { SnackbarHostState() }
    
    // SharedPreferences for remembering settings
    val prefs = remember { context.getSharedPreferences("image_generator_prefs", android.content.Context.MODE_PRIVATE) }
    
    // UI State - load from preferences
    var promptText by remember { mutableStateOf("") }
    var iterations by remember { mutableIntStateOf(prefs.getInt("iterations", 20)) }
    var seed by remember { mutableIntStateOf(prefs.getInt("seed", (0..999999).random())) }
    val generatedImages = remember { mutableStateListOf<Bitmap>() }
    var isGenerating by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var currentGenerationIndex by remember { mutableIntStateOf(0) }
    
    // Img2img state
    var denoiseStrength by remember { mutableStateOf(prefs.getFloat("denoise_strength", 0.7f)) }
    var inputImageUri by remember { mutableStateOf<Uri?>(null) }
    var inputImageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var showInputImageFullscreen by remember { mutableStateOf(false) }
    
    // Image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        inputImageUri = uri
        uri?.let {
            try {
                inputImageBitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val source = android.graphics.ImageDecoder.createSource(context.contentResolver, it)
                    android.graphics.ImageDecoder.decodeBitmap(source)
                } else {
                    @Suppress("DEPRECATION")
                    MediaStore.Images.Media.getBitmap(context.contentResolver, it)
                }
            } catch (e: Exception) {
                errorMessage = "Failed to load image: ${e.message}"
                inputImageUri = null
                inputImageBitmap = null
            }
        }
    }
    
    // Model availability state
    var availableModels by remember { mutableStateOf<List<ModelInfo>>(emptyList()) }
    var selectedModel by remember { mutableStateOf<ModelInfo?>(null) }
    var isModelLoaded by remember { mutableStateOf(false) }
    var isModelLoading by remember { mutableStateOf(false) }
    var showModelSheet by remember { mutableStateOf(false) }
    var useGpu by remember { mutableStateOf(false) }
    
    // Remember last used model and settings
    val lastModelPath = remember { prefs.getString("last_model_path", null) }
    LaunchedEffect(Unit) {
        useGpu = prefs.getBoolean("use_gpu", false)
    }
    
    // Image generator helper
    val imageGeneratorHelper = remember { ImageGeneratorHelper(context) }
    val sdHelper = remember { StableDiffusionHelper(context) }
    
    // Refresh trigger - incremented when we need to refresh model list
    var refreshTrigger by remember { mutableIntStateOf(0) }
    
    // Check if image generation model is available - refresh on trigger change
    LaunchedEffect(refreshTrigger) {
        val models = sdHelper.listModels()
        availableModels = models
        if (models.isNotEmpty()) {
            // Try to select last used model, otherwise first
            selectedModel = models.find { it.path == lastModelPath } ?: models.first()
        } else {
            selectedModel = null
            isModelLoaded = false
        }
    }
    
    // Refresh model list when screen becomes visible (e.g., after navigating back from downloads)
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
                refreshTrigger++
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }
    
    // Save settings when they change
    LaunchedEffect(iterations, seed, selectedModel, denoiseStrength, useGpu) {
        prefs.edit()
            .putInt("iterations", iterations)
            .putInt("seed", seed)
            .putString("last_model_path", selectedModel?.path)
            .putFloat("denoise_strength", denoiseStrength)
            .putBoolean("use_gpu", useGpu)
            .apply()
    }
    
    // Clean up on dispose
    DisposableEffect(Unit) {
        onDispose {
            imageGeneratorHelper.close()
        }
    }
    
    // Show error snackbar
    LaunchedEffect(errorMessage) {
        errorMessage?.let {
            snackbarHostState.showSnackbar(it)
            errorMessage = null
        }
    }
    
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
                
                // Model Selector Card - matching ModelSelectorCard style
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
                        // Title
                        Text(
                            text = stringResource(R.string.select_model_title),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                        
                        if (availableModels.isEmpty()) {
                            // No models available
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
                            // Model Dropdown - matching ModelSelectorCard style
                            var expanded by remember { mutableStateOf(false) }
                            ExposedDropdownMenuBox(
                                expanded = expanded,
                                onExpandedChange = { expanded = !expanded }
                            ) {
                                OutlinedTextField(
                                    value = selectedModel?.name ?: stringResource(R.string.select_model),
                                    onValueChange = {},
                                    readOnly = true,
                                    label = { Text(stringResource(R.string.select_model)) },
                                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
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
                                    expanded = expanded,
                                    onDismissRequest = { expanded = false }
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
                                                selectedModel = model
                                                expanded = false
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
                            
                            // GPU Toggle (MNN models only)
                            if (selectedModel?.type == ModelType.MNN_CPU) {
                                Row(
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable { useGpu = !useGpu }
                                        .padding(vertical = 8.dp)
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Speed,
                                            contentDescription = null,
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                        Column {
                                            Text(
                                                text = stringResource(R.string.image_generator_use_gpu),
                                                style = MaterialTheme.typography.bodyMedium,
                                                fontWeight = FontWeight.Medium
                                            )
                                            Text(
                                                text = stringResource(R.string.image_generator_use_gpu_desc),
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant
                                            )
                                        }
                                    }
                                    Switch(
                                        checked = useGpu,
                                        onCheckedChange = { useGpu = it }
                                    )
                                }
                            }
                            
                            // Load/Reload Model Button
                            Button(
                                onClick = {
                                    selectedModel?.let { model ->
                                        isModelLoading = true
                                        coroutineScope.launch {
                                            val success = imageGeneratorHelper.initialize(model.path, useGpu)
                                            isModelLoaded = success
                                            isModelLoading = false
                                            if (!success) {
                                                errorMessage = context.getString(R.string.image_generator_failed_load)
                                            }
                                        }
                                    }
                                },
                                modifier = Modifier.fillMaxWidth(),
                                enabled = selectedModel != null && !isModelLoading,
                                shape = RoundedCornerShape(16.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = MaterialTheme.colorScheme.primary
                                )
                            ) {
                                if (isModelLoading) {
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
                            
                            // Unload Model Button (only show when model is loaded)
                            if (isModelLoaded) {
                                OutlinedButton(
                                    onClick = {
                                        imageGeneratorHelper.close()
                                        isModelLoaded = false
                                    },
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
                        // Iterations slider
                        Column {
                            Text(
                                text = "${stringResource(R.string.image_generator_iterations)}: $iterations",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium
                            )
                            Slider(
                                value = iterations.toFloat(),
                                onValueChange = { iterations = it.toInt() },
                                valueRange = 10f..50f,
                                steps = 7,
                                enabled = !isGenerating
                            )
                        }
                        
                        // Seed input with random button
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            OutlinedTextField(
                                value = seed.toString(),
                                onValueChange = { 
                                    it.toIntOrNull()?.let { newSeed -> 
                                        seed = newSeed 
                                    }
                                },
                                label = { Text(stringResource(R.string.image_generator_seed)) },
                                modifier = Modifier.weight(1f),
                                enabled = !isGenerating,
                                singleLine = true,
                                shape = RoundedCornerShape(16.dp)
                            )
                            IconButton(
                                onClick = { seed = (0..999999).random() },
                                enabled = !isGenerating
                            ) {
                                Icon(
                                    Icons.Default.Refresh,
                                    contentDescription = stringResource(R.string.image_generator_random_seed)
                                )
                            }
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Text(
                        stringResource(R.string.image_generator_title),
                        fontWeight = FontWeight.Bold
                    )
                },
                actions = {
                    IconButton(onClick = { showModelSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = "Settings")
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
        
        // Show "Download Model First" screen if model not available
        if (availableModels.isEmpty()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .padding(32.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Default.Palette,
                    contentDescription = null,
                    modifier = Modifier.size(80.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = stringResource(R.string.image_generator_download_model),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(R.string.image_generator_download_model_desc),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
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
        } else if (!isModelLoaded) {
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
                    text = stringResource(R.string.image_generator_load_model_title),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(R.string.image_generator_load_model_desc),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(32.dp))
                FilledTonalButton(
                    onClick = { showModelSheet = true },
                    modifier = Modifier.fillMaxWidth(0.6f)
                ) {
                    Icon(
                        imageVector = Icons.Default.Tune,
                        contentDescription = null
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.feature_settings_title))
                }
            }
        } else {
            // Main content - use BoxWithConstraints for landscape detection
            BoxWithConstraints(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
            ) {
                val isLandscape = maxWidth > maxHeight
                val configuration = LocalConfiguration.current
                
                if (isLandscape) {
                    // Landscape layout: Prompt on left, Images on right
                    if (generatedImages.isEmpty()) {
                        // Horizontally center the prompt column when no images generated yet
                        Box(
                            modifier = Modifier.fillMaxSize().padding(16.dp),
                            contentAlignment = Alignment.TopCenter
                        ) {
                            Column(
                                modifier = Modifier.fillMaxWidth(0.5f).verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                // Prompt input
                                Card(
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                                    )
                                ) {
                                    Column(modifier = Modifier.padding(16.dp)) {
                                        Text(
                                            text = stringResource(R.string.image_generator_prompt_label),
                                            style = MaterialTheme.typography.titleMedium,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Spacer(modifier = Modifier.height(8.dp))
                                        OutlinedTextField(
                                            value = promptText,
                                            onValueChange = { promptText = it },
                                            placeholder = { 
                                                Text(stringResource(R.string.image_generator_prompt_hint))
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                            minLines = 5,
                                            maxLines = 10,
                                            enabled = !isGenerating && isModelLoaded
                                        )
                                    }
                                }
                                
                                // Img2img section
                                Card(
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                                    )
                                ) {
                                    Column(
                                        modifier = Modifier.padding(16.dp),
                                        verticalArrangement = Arrangement.spacedBy(12.dp)
                                    ) {
                                        Text(
                                            text = stringResource(R.string.image_generator_img2img),
                                            style = MaterialTheme.typography.bodyMedium,
                                            fontWeight = FontWeight.Bold
                                        )
                                        
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            OutlinedButton(
                                                onClick = { imagePickerLauncher.launch("image/*") },
                                                enabled = !isGenerating,
                                                shape = RoundedCornerShape(12.dp),
                                                modifier = Modifier.weight(1f)
                                            ) {
                                                Icon(Icons.Default.Image, contentDescription = null)
                                                Spacer(modifier = Modifier.width(8.dp))
                                                Text(
                                                    stringResource(
                                                        if (inputImageBitmap != null) R.string.image_generator_change_image
                                                        else R.string.image_generator_select_image
                                                    )
                                                )
                                            }
                                            
                                            inputImageBitmap?.let { bitmap ->
                                                Card(
                                                    modifier = Modifier
                                                        .size(64.dp)
                                                        .clickable { showInputImageFullscreen = true },
                                                    shape = RoundedCornerShape(8.dp)
                                                ) {
                                                    Image(
                                                        bitmap = bitmap.asImageBitmap(),
                                                        contentDescription = stringResource(R.string.image_generator_input_image),
                                                        modifier = Modifier.fillMaxSize()
                                                    )
                                                }
                                            }
                                        }
                                        
                                        if (inputImageBitmap != null) {
                                            Column {
                                                Text(
                                                    text = stringResource(R.string.image_generator_denoise_strength, denoiseStrength),
                                                    style = MaterialTheme.typography.bodyMedium,
                                                    fontWeight = FontWeight.Medium
                                                )
                                                Text(
                                                    text = stringResource(R.string.image_generator_denoise_strength_desc),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                                Slider(
                                                    value = denoiseStrength,
                                                    onValueChange = { denoiseStrength = it },
                                                    valueRange = 0.1f..1.0f,
                                                    enabled = !isGenerating
                                                )
                                            }
                                        }
                                    }
                                }
                                
                                // Generate button
                                Button(
                    onClick = {
                        if (promptText.isNotBlank()) {
                            isGenerating = true
                            // Clear previous images when starting new generation
                            generatedImages.clear()
                            currentGenerationIndex = 0
                            coroutineScope.launch {
                                try {
                                    // Generate first image
                                    val bitmap = imageGeneratorHelper.generateImage(
                                        prompt = promptText,
                                        iterations = iterations,
                                        seed = seed,
                                        inputImage = inputImageBitmap,
                                        denoiseStrength = denoiseStrength,
                                        useGpu = useGpu
                                    )
                                    if (bitmap != null) {
                                        generatedImages.add(bitmap)
                                    } else {
                                        errorMessage = context.getString(R.string.image_generator_error)
                                    }
                                } catch (e: Exception) {
                                    errorMessage = e.message ?: context.getString(R.string.image_generator_error)
                                } finally {
                                    isGenerating = false
                                }
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isGenerating && isModelLoaded && promptText.isNotBlank()
                ) {
                    if (isGenerating) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(R.string.image_generator_generating))
                    } else {
                        Icon(Icons.Default.AutoAwesome, contentDescription = null)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(R.string.image_generator_generate))
                    }
                }
                            }
                        }
                    } else {
                        // Show side-by-side when images exist
                        // Declare pagerState outside so both columns can access it
                        val pagerState = rememberPagerState(
                            initialPage = 0,
                            pageCount = { generatedImages.size + 1 } // +1 for "generate more" placeholder
                        )
                        
                        Row(
                            modifier = Modifier
                                .fillMaxSize()
                                .padding(16.dp),
                            horizontalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            // Left side: Prompt and Generate button
                            Column(
                                modifier = Modifier
                                    .weight(0.5f)
                                    .fillMaxHeight()
                                    .verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                // Prompt input
                                Card(
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                                    )
                                ) {
                                    Column(modifier = Modifier.padding(16.dp)) {
                                        Text(
                                            text = stringResource(R.string.image_generator_prompt_label),
                                            style = MaterialTheme.typography.titleMedium,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Spacer(modifier = Modifier.height(8.dp))
                                        OutlinedTextField(
                                            value = promptText,
                                            onValueChange = { promptText = it },
                                            placeholder = { 
                                                Text(stringResource(R.string.image_generator_prompt_hint))
                                            },
                                            modifier = Modifier.fillMaxWidth(),
                                            minLines = 5,
                                            maxLines = 10,
                                            enabled = !isGenerating && isModelLoaded
                                        )
                                    }
                                }
                                
                                // Img2img section
                                Card(
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                                    )
                                ) {
                                    Column(
                                        modifier = Modifier.padding(16.dp),
                                        verticalArrangement = Arrangement.spacedBy(12.dp)
                                    ) {
                                        Text(
                                            text = stringResource(R.string.image_generator_img2img),
                                            style = MaterialTheme.typography.bodyMedium,
                                            fontWeight = FontWeight.Bold
                                        )
                                        
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            OutlinedButton(
                                                onClick = { imagePickerLauncher.launch("image/*") },
                                                enabled = !isGenerating,
                                                shape = RoundedCornerShape(12.dp),
                                                modifier = Modifier.weight(1f)
                                            ) {
                                                Icon(Icons.Default.Image, contentDescription = null)
                                                Spacer(modifier = Modifier.width(8.dp))
                                                Text(
                                                    stringResource(
                                                        if (inputImageBitmap != null) R.string.image_generator_change_image
                                                        else R.string.image_generator_select_image
                                                    )
                                                )
                                            }
                                            
                                            inputImageBitmap?.let { bitmap ->
                                                Card(
                                                    modifier = Modifier
                                                        .size(64.dp)
                                                        .clickable { showInputImageFullscreen = true },
                                                    shape = RoundedCornerShape(8.dp)
                                                ) {
                                                    Image(
                                                        bitmap = bitmap.asImageBitmap(),
                                                        contentDescription = stringResource(R.string.image_generator_input_image),
                                                        modifier = Modifier.fillMaxSize()
                                                    )
                                                }
                                            }
                                        }
                                        
                                        if (inputImageBitmap != null) {
                                            Column {
                                                Text(
                                                    text = stringResource(R.string.image_generator_denoise_strength, denoiseStrength),
                                                    style = MaterialTheme.typography.bodyMedium,
                                                    fontWeight = FontWeight.Medium
                                                )
                                                Text(
                                                    text = stringResource(R.string.image_generator_denoise_strength_desc),
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                                Slider(
                                                    value = denoiseStrength,
                                                    onValueChange = { denoiseStrength = it },
                                                    valueRange = 0.1f..1.0f,
                                                    enabled = !isGenerating
                                                )
                                            }
                                        }
                                    }
                                }
                                
                                // Generate button
                                Button(
                                    onClick = {
                                        if (promptText.isNotBlank()) {
                                            isGenerating = true
                                            // Clear previous images when starting new generation
                                            generatedImages.clear()
                                            currentGenerationIndex = 0
                                            coroutineScope.launch {
                                                try {
                                                    // Generate first image
                                                    val bitmap = imageGeneratorHelper.generateImage(
                                                        prompt = promptText,
                                                        iterations = iterations,
                                                        seed = seed,
                                                        inputImage = inputImageBitmap,
                                                        denoiseStrength = denoiseStrength,
                                                        useGpu = useGpu
                                                    )
                                                    if (bitmap != null) {
                                                        generatedImages.add(bitmap)
                                                    } else {
                                                        errorMessage = context.getString(R.string.image_generator_error)
                                                    }
                                                } catch (e: Exception) {
                                                    errorMessage = e.message ?: context.getString(R.string.image_generator_error)
                                                } finally {
                                                    isGenerating = false
                                                }
                                            }
                                        }
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                    enabled = !isGenerating && isModelLoaded && promptText.isNotBlank()
                                ) {
                                    if (isGenerating) {
                                        CircularProgressIndicator(
                                            modifier = Modifier.size(20.dp),
                                            strokeWidth = 2.dp,
                                            color = MaterialTheme.colorScheme.onPrimary
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(stringResource(R.string.image_generator_generating))
                                    } else {
                                        Icon(Icons.Default.AutoAwesome, contentDescription = null)
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(stringResource(R.string.image_generator_generate))
                                    }
                                }
                                
                                // Save current image button (in landscape mode)
                                if (pagerState.currentPage < generatedImages.size) {
                                    Button(
                                        onClick = {
                                            coroutineScope.launch {
                                                val currentBitmap = generatedImages[pagerState.currentPage]
                                                val uri = imageGeneratorHelper.saveImageToGallery(currentBitmap, "Generated Image")
                                                if (uri != null) {
                                                    snackbarHostState.showSnackbar(context.getString(R.string.image_generator_saved))
                                                } else {
                                                    errorMessage = context.getString(R.string.image_generator_error)
                                                }
                                            }
                                        },
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Icon(Icons.Default.Save, contentDescription = null)
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(stringResource(R.string.image_generator_save))
                                    }
                                }
                            }
                            
                            // Right side: Generated Images
                            Column(
                                modifier = Modifier
                                    .weight(0.5f)
                                    .fillMaxHeight()
                                    .verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                    // Prefetch next image: always generate one ahead while viewing current
                    // Use collect instead of collectLatest to avoid cancellation when swiping back
                    LaunchedEffect(pagerState, generatedImages.size) {
                        snapshotFlow { pagerState.currentPage }
                            .collect { currentPage ->
                                // Start generating next image when viewing any image (not the placeholder)
                                // and we don't already have a next image ready
                                val shouldPrefetch = currentPage < generatedImages.size && 
                                    currentPage == generatedImages.size - 1 && 
                                    !isGenerating
                                val swipedToPlaceholder = currentPage == generatedImages.size && !isGenerating
                                
                                if (shouldPrefetch || swipedToPlaceholder) {
                                    // Launch in separate scope so it won't be cancelled when user swipes away
                                    coroutineScope.launch {
                                        if (isGenerating) return@launch // Double check to prevent race
                                        isGenerating = true
                                        currentGenerationIndex = generatedImages.size
                                        try {
                                            val newSeed = (0..999999).random()
                                            val bitmap = imageGeneratorHelper.generateImage(
                                                prompt = promptText,
                                                iterations = iterations,
                                                seed = newSeed,
                                                useGpu = useGpu
                                            )
                                            if (bitmap != null) {
                                                generatedImages.add(bitmap)
                                            }
                                        } catch (e: Exception) {
                                            errorMessage = e.message
                                        } finally {
                                            isGenerating = false
                                        }
                                    }
                                }
                            }
                    }
                    
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
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = stringResource(R.string.image_generator_success),
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "${minOf(pagerState.currentPage + 1, generatedImages.size)}/${generatedImages.size}",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            
                            // Horizontal pager for images
                            HorizontalPager(
                                state = pagerState,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(1f)
                            ) { page ->
                                if (page < generatedImages.size) {
                                    // Show generated image
                                    Image(
                                        bitmap = generatedImages[page].asImageBitmap(),
                                        contentDescription = "$promptText - variation ${page + 1}",
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .clip(RoundedCornerShape(12.dp))
                                    )
                                } else {
                                    // "Generate more" placeholder
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .clip(RoundedCornerShape(12.dp))
                                            .background(MaterialTheme.colorScheme.surfaceContainerHigh),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        if (isGenerating) {
                                            Column(
                                                horizontalAlignment = Alignment.CenterHorizontally,
                                                verticalArrangement = Arrangement.spacedBy(16.dp)
                                            ) {
                                                CircularProgressIndicator()
                                                Text(
                                                    text = stringResource(R.string.image_generator_variation, generatedImages.size + 1),
                                                    style = MaterialTheme.typography.bodyMedium,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                        } else {
                                            Column(
                                                horizontalAlignment = Alignment.CenterHorizontally,
                                                verticalArrangement = Arrangement.spacedBy(8.dp)
                                            ) {
                                                Icon(
                                                    Icons.Default.Add,
                                                    contentDescription = null,
                                                    modifier = Modifier.size(48.dp),
                                                    tint = MaterialTheme.colorScheme.primary
                                                )
                                                Text(
                                                    text = stringResource(R.string.image_generator_swipe_more),
                                                    style = MaterialTheme.typography.bodyMedium,
                                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                    textAlign = TextAlign.Center
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Page indicators
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.Center
                            ) {
                                repeat(generatedImages.size) { index ->
                                    Box(
                                        modifier = Modifier
                                            .padding(horizontal = 4.dp)
                                            .size(8.dp)
                                            .clip(CircleShape)
                                            .background(
                                                if (pagerState.currentPage == index)
                                                    MaterialTheme.colorScheme.primary
                                                else
                                                    MaterialTheme.colorScheme.outlineVariant
                                            )
                                    )
                                }
                                // Plus indicator for "generate more"
                                Box(
                                    modifier = Modifier
                                        .padding(horizontal = 4.dp)
                                        .size(8.dp)
                                        .clip(CircleShape)
                                        .background(
                                            if (pagerState.currentPage == generatedImages.size)
                                                MaterialTheme.colorScheme.primary
                                            else
                                                MaterialTheme.colorScheme.outlineVariant
                                        )
                                )
                            }
                        }
                    }
                            }
                        }
                    }
                } else {
                    // Portrait layout: Original vertical layout
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(scrollState)
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        // Prompt input
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                            )
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Text(
                                    text = stringResource(R.string.image_generator_prompt_label),
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                OutlinedTextField(
                                    value = promptText,
                                    onValueChange = { promptText = it },
                                    placeholder = { 
                                        Text(stringResource(R.string.image_generator_prompt_hint))
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                    minLines = 3,
                                    maxLines = 5,
                                    enabled = !isGenerating && isModelLoaded
                                )
                            }
                        }
                        
                        // Img2img section
                        Card(
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceContainerLow
                            )
                        ) {
                            Column(
                                modifier = Modifier.padding(16.dp),
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                Text(
                                    text = stringResource(R.string.image_generator_img2img),
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.Bold
                                )
                                
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    OutlinedButton(
                                        onClick = { imagePickerLauncher.launch("image/*") },
                                        enabled = !isGenerating,
                                        shape = RoundedCornerShape(12.dp),
                                        modifier = Modifier.weight(1f)
                                    ) {
                                        Icon(Icons.Default.Image, contentDescription = null)
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            stringResource(
                                                if (inputImageBitmap != null) R.string.image_generator_change_image
                                                else R.string.image_generator_select_image
                                            )
                                        )
                                    }
                                    
                                    inputImageBitmap?.let { bitmap ->
                                        Card(
                                            modifier = Modifier
                                                .size(64.dp)
                                                .clickable { showInputImageFullscreen = true },
                                            shape = RoundedCornerShape(8.dp)
                                        ) {
                                            Image(
                                                bitmap = bitmap.asImageBitmap(),
                                                contentDescription = stringResource(R.string.image_generator_input_image),
                                                modifier = Modifier.fillMaxSize()
                                            )
                                        }
                                    }
                                }
                                
                                if (inputImageBitmap != null) {
                                    Column {
                                        Text(
                                            text = stringResource(R.string.image_generator_denoise_strength, denoiseStrength),
                                            style = MaterialTheme.typography.bodyMedium,
                                            fontWeight = FontWeight.Medium
                                        )
                                        Text(
                                            text = stringResource(R.string.image_generator_denoise_strength_desc),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                        Slider(
                                            value = denoiseStrength,
                                            onValueChange = { denoiseStrength = it },
                                            valueRange = 0.1f..1.0f,
                                            enabled = !isGenerating
                                        )
                                    }
                                }
                            }
                        }
                        
                        // Generate button
                        Button(
                            onClick = {
                                if (promptText.isNotBlank()) {
                                    isGenerating = true
                                    // Clear previous images when starting new generation
                                    generatedImages.clear()
                                    currentGenerationIndex = 0
                                    coroutineScope.launch {
                                        try {
                                            // Generate first image
                                            val bitmap = imageGeneratorHelper.generateImage(
                                                prompt = promptText,
                                                iterations = iterations,
                                                seed = seed,
                                                inputImage = inputImageBitmap,
                                                denoiseStrength = denoiseStrength,
                                                useGpu = useGpu
                                            )
                                            if (bitmap != null) {
                                                generatedImages.add(bitmap)
                                            } else {
                                                errorMessage = context.getString(R.string.image_generator_error)
                                            }
                                        } catch (e: Exception) {
                                            errorMessage = e.message ?: context.getString(R.string.image_generator_error)
                                        } finally {
                                            isGenerating = false
                                        }
                                    }
                                }
                            },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = !isGenerating && isModelLoaded && promptText.isNotBlank()
                        ) {
                            if (isGenerating) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    strokeWidth = 2.dp,
                                    color = MaterialTheme.colorScheme.onPrimary
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.image_generator_generating))
                            } else {
                                Icon(Icons.Default.AutoAwesome, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Text(stringResource(R.string.image_generator_generate))
                            }
                        }
                        
                        // Generated images carousel (Apple Intelligence style)
                        if (generatedImages.isNotEmpty()) {
                            val pagerState = rememberPagerState(
                                initialPage = 0,
                                pageCount = { generatedImages.size + 1 } // +1 for "generate more" placeholder
                            )
                            
                            // Prefetch next image: always generate one ahead while viewing current
                            // Use collect instead of collectLatest to avoid cancellation when swiping back
                            LaunchedEffect(pagerState, generatedImages.size) {
                                snapshotFlow { pagerState.currentPage }
                                    .collect { currentPage ->
                                        // Start generating next image when viewing any image (not the placeholder)
                                        // and we don't already have a next image ready
                                        val shouldPrefetch = currentPage < generatedImages.size && 
                                            currentPage == generatedImages.size - 1 && 
                                            !isGenerating
                                        val swipedToPlaceholder = currentPage == generatedImages.size && !isGenerating
                                        
                                        if (shouldPrefetch || swipedToPlaceholder) {
                                            // Launch in separate scope so it won't be cancelled when user swipes away
                                            coroutineScope.launch {
                                                if (isGenerating) return@launch // Double check to prevent race
                                                isGenerating = true
                                                currentGenerationIndex = generatedImages.size
                                                try {
                                                    val newSeed = (0..999999).random()
                                                    val bitmap = imageGeneratorHelper.generateImage(
                                                        prompt = promptText,
                                                        iterations = iterations,
                                                        seed = newSeed,
                                                        inputImage = inputImageBitmap,
                                                        denoiseStrength = denoiseStrength,
                                                        useGpu = useGpu
                                                    )
                                                    if (bitmap != null) {
                                                        generatedImages.add(bitmap)
                                                    }
                                                } catch (e: Exception) {
                                                    errorMessage = e.message
                                                } finally {
                                                    isGenerating = false
                                                }
                                            }
                                        }
                                    }
                            }
                            
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
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = stringResource(R.string.image_generator_success),
                                            style = MaterialTheme.typography.titleMedium,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Text(
                                            text = "${minOf(pagerState.currentPage + 1, generatedImages.size)}/${generatedImages.size}",
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                    
                                    // Horizontal pager for images
                                    HorizontalPager(
                                        state = pagerState,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .aspectRatio(1f)
                                    ) { page ->
                                        if (page < generatedImages.size) {
                                            // Show generated image
                                            Image(
                                                bitmap = generatedImages[page].asImageBitmap(),
                                                contentDescription = "$promptText - variation ${page + 1}",
                                                modifier = Modifier
                                                    .fillMaxSize()
                                                    .clip(RoundedCornerShape(12.dp))
                                            )
                                        } else {
                                            // "Generate more" placeholder
                                            Box(
                                                modifier = Modifier
                                                    .fillMaxSize()
                                                    .clip(RoundedCornerShape(12.dp))
                                                    .background(MaterialTheme.colorScheme.surfaceContainerHigh),
                                                contentAlignment = Alignment.Center
                                            ) {
                                                if (isGenerating) {
                                                    Column(
                                                        horizontalAlignment = Alignment.CenterHorizontally,
                                                        verticalArrangement = Arrangement.spacedBy(16.dp)
                                                    ) {
                                                        CircularProgressIndicator()
                                                        Text(
                                                            text = stringResource(R.string.image_generator_variation, generatedImages.size + 1),
                                                            style = MaterialTheme.typography.bodyMedium,
                                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                                        )
                                                    }
                                                } else {
                                                    Column(
                                                        horizontalAlignment = Alignment.CenterHorizontally,
                                                        verticalArrangement = Arrangement.spacedBy(8.dp)
                                                    ) {
                                                        Icon(
                                                            Icons.Default.Add,
                                                            contentDescription = null,
                                                            modifier = Modifier.size(48.dp),
                                                            tint = MaterialTheme.colorScheme.primary
                                                        )
                                                        Text(
                                                            text = stringResource(R.string.image_generator_swipe_more),
                                                            style = MaterialTheme.typography.bodyMedium,
                                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                                            textAlign = TextAlign.Center
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Page indicators
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.Center
                                    ) {
                                        repeat(generatedImages.size) { index ->
                                            Box(
                                                modifier = Modifier
                                                    .padding(horizontal = 4.dp)
                                                    .size(8.dp)
                                                    .clip(CircleShape)
                                                    .background(
                                                        if (pagerState.currentPage == index)
                                                            MaterialTheme.colorScheme.primary
                                                        else
                                                            MaterialTheme.colorScheme.outlineVariant
                                                    )
                                            )
                                        }
                                        // Plus indicator for "generate more"
                                        Box(
                                            modifier = Modifier
                                                .padding(horizontal = 4.dp)
                                                .size(8.dp)
                                                .clip(CircleShape)
                                                .background(
                                                    if (pagerState.currentPage == generatedImages.size)
                                                        MaterialTheme.colorScheme.primary
                                                    else
                                                        MaterialTheme.colorScheme.outlineVariant
                                                )
                                        )
                                    }
                                }
                            }
                            
                            // Save current image button (outside Card for equal width with Generate button)
                            if (pagerState.currentPage < generatedImages.size) {
                                Button(
                                    onClick = {
                                        coroutineScope.launch {
                                            val currentBitmap = generatedImages[pagerState.currentPage]
                                            val uri = imageGeneratorHelper.saveImageToGallery(currentBitmap, "Generated Image")
                                            if (uri != null) {
                                                snackbarHostState.showSnackbar(context.getString(R.string.image_generator_saved))
                                            } else {
                                                errorMessage = context.getString(R.string.image_generator_error)
                                            }
                                        }
                                    },
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Icon(Icons.Default.Save, contentDescription = null)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(stringResource(R.string.image_generator_save))
                                }
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(32.dp))
                    }
                }
            }
        }
    }
    
    // Fullscreen input image viewer
    if (showInputImageFullscreen && inputImageBitmap != null) {
        val configuration = LocalConfiguration.current
        // Key forces full recomposition on orientation change
        key(configuration.orientation) {
            androidx.compose.ui.window.Dialog(
                onDismissRequest = { showInputImageFullscreen = false },
                properties = androidx.compose.ui.window.DialogProperties(
                    usePlatformDefaultWidth = false,
                    decorFitsSystemWindows = false
                )
            ) {
                val view = LocalView.current
                val window = (view.parent as? DialogWindowProvider)?.window
                
                LaunchedEffect(configuration.orientation) {
                    window?.let { win ->
                        win.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                        win.setFlags(
                            android.view.WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                            android.view.WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
                        )
                    }
                }
                
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(androidx.compose.ui.graphics.Color.Black)
                        .clickable { showInputImageFullscreen = false },
                    contentAlignment = Alignment.Center
                ) {
                    Image(
                        bitmap = inputImageBitmap!!.asImageBitmap(),
                        contentDescription = stringResource(R.string.image_generator_input_image),
                        modifier = Modifier.fillMaxSize(),
                        contentScale = androidx.compose.ui.layout.ContentScale.Fit
                    )
                    
                    // Close button
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
}
