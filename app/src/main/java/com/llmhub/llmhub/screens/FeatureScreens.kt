package com.llmhub.llmhub.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.llmhub.llmhub.R
import com.llmhub.llmhub.components.ModelSelectorCard
import com.llmhub.llmhub.viewmodels.TranslatorViewModel
import com.llmhub.llmhub.viewmodels.TranscriberViewModel

// Language data class
data class Language(val code: String, val nameResId: Int)

// All supported languages in alphabetical order
val supportedLanguages = listOf(
    Language("en", R.string.lang_english),
    Language("af", R.string.lang_afrikaans),
    Language("am", R.string.lang_amharic),
    Language("ar", R.string.lang_arabic),
    Language("hy", R.string.lang_armenian),
    Language("az", R.string.lang_azerbaijani),
    Language("eu", R.string.lang_basque),
    Language("bn", R.string.lang_bengali),
    Language("bg", R.string.lang_bulgarian),
    Language("my", R.string.lang_burmese),
    Language("ca", R.string.lang_catalan),
    Language("zh-CN", R.string.lang_chinese),
    Language("zh-TW", R.string.lang_chinese_traditional),
    Language("hr", R.string.lang_croatian),
    Language("cs", R.string.lang_czech),
    Language("da", R.string.lang_danish),
    Language("nl", R.string.lang_dutch),
    Language("et", R.string.lang_estonian),
    Language("tl", R.string.lang_filipino),
    Language("fi", R.string.lang_finnish),
    Language("fr", R.string.lang_french),
    Language("gl", R.string.lang_galician),
    Language("ka", R.string.lang_georgian),
    Language("de", R.string.lang_german),
    Language("el", R.string.lang_greek),
    Language("gu", R.string.lang_gujarati),
    Language("ha", R.string.lang_hausa),
    Language("he", R.string.lang_hebrew),
    Language("hi", R.string.lang_hindi),
    Language("hu", R.string.lang_hungarian),
    Language("is", R.string.lang_icelandic),
    Language("ig", R.string.lang_igbo),
    Language("id", R.string.lang_indonesian),
    Language("it", R.string.lang_italian),
    Language("ja", R.string.lang_japanese),
    Language("kn", R.string.lang_kannada),
    Language("kk", R.string.lang_kazakh),
    Language("km", R.string.lang_khmer),
    Language("ko", R.string.lang_korean),
    Language("lo", R.string.lang_lao),
    Language("lv", R.string.lang_latvian),
    Language("lt", R.string.lang_lithuanian),
    Language("ms", R.string.lang_malay),
    Language("ml", R.string.lang_malayalam),
    Language("mr", R.string.lang_marathi),
    Language("ne", R.string.lang_nepali),
    Language("no", R.string.lang_norwegian),
    Language("ps", R.string.lang_pashto),
    Language("fa", R.string.lang_persian),
    Language("pl", R.string.lang_polish),
    Language("pt", R.string.lang_portuguese),
    Language("pa", R.string.lang_punjabi),
    Language("ro", R.string.lang_romanian),
    Language("ru", R.string.lang_russian),
    Language("sr", R.string.lang_serbian),
    Language("sd", R.string.lang_sindhi),
    Language("si", R.string.lang_sinhala),
    Language("sk", R.string.lang_slovak),
    Language("sl", R.string.lang_slovenian),
    Language("so", R.string.lang_somali),
    Language("es", R.string.lang_spanish),
    Language("sw", R.string.lang_swahili),
    Language("sv", R.string.lang_swedish),
    Language("ta", R.string.lang_tamil),
    Language("te", R.string.lang_telugu),
    Language("th", R.string.lang_thai),
    Language("tr", R.string.lang_turkish),
    Language("uk", R.string.lang_ukrainian),
    Language("ur", R.string.lang_urdu),
    Language("uz", R.string.lang_uzbek),
    Language("vi", R.string.lang_vietnamese),
    Language("yo", R.string.lang_yoruba),
    Language("zu", R.string.lang_zulu)
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranslatorScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    viewModel: TranslatorViewModel = viewModel()
) {
    // ViewModel states
    val availableModels by viewModel.availableModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val isTranslating by viewModel.isTranslating.collectAsState()
    val loadError by viewModel.loadError.collectAsState()
    val visionEnabled by viewModel.visionEnabled.collectAsState()
    val audioEnabled by viewModel.audioEnabled.collectAsState()
    val autoDetectSource by viewModel.autoDetectSource.collectAsState()
    val detectedLanguage by viewModel.detectedLanguage.collectAsState()
    val inputText by viewModel.inputText.collectAsState()
    val inputImageUri by viewModel.inputImageUri.collectAsState()
    val outputText by viewModel.outputText.collectAsState()
    
    // UI state
    var sourceLanguageIndex by remember { mutableStateOf(0) } // English
    var targetLanguageIndex by remember { mutableStateOf(supportedLanguages.indexOfFirst { it.code == "es" }) } // Spanish
    var sourceExpanded by remember { mutableStateOf(false) }
    var targetExpanded by remember { mutableStateOf(false) }
    
    // Image picker
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        viewModel.setInputImage(uri)
    }
    
    // Snackbar
    val snackbarHostState = remember { SnackbarHostState() }
    
    LaunchedEffect(loadError) {
        loadError?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.translator_title), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Model Selection Card
            ModelSelectorCard(
                models = availableModels,
                selectedModel = selectedModel,
                selectedBackend = selectedBackend,
                isLoading = isLoadingModel,
                onModelSelected = { viewModel.selectModel(it) },
                onBackendSelected = { viewModel.selectBackend(it) },
                onLoadModel = { viewModel.loadModel() },
                filterMultimodalOnly = true
            )
            
            // Modality Toggles Card
            AnimatedVisibility(
                visible = selectedModel != null,
                enter = fadeIn() + slideInVertically()
            ) {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = stringResource(R.string.input_options),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        
                        // Vision toggle
                        if (selectedModel?.supportsVision == true) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(Icons.Default.Image, contentDescription = null)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Column {
                                        Text(
                                            text = stringResource(R.string.enable_vision_input),
                                            style = MaterialTheme.typography.bodyLarge
                                        )
                                        Text(
                                            text = stringResource(R.string.translate_from_image),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                                Switch(
                                    checked = visionEnabled,
                                    onCheckedChange = { viewModel.toggleVision(it) }
                                )
                            }
                        }
                        
                        // Audio toggle
                        if (selectedModel?.supportsAudio == true) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(Icons.Default.Mic, contentDescription = null)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Column {
                                        Text(
                                            text = stringResource(R.string.enable_audio_input),
                                            style = MaterialTheme.typography.bodyLarge
                                        )
                                        Text(
                                            text = stringResource(R.string.speak_to_translate),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                                Switch(
                                    checked = audioEnabled,
                                    onCheckedChange = { viewModel.toggleAudio(it) }
                                )
                            }
                        }
                        
                        // Auto-detect language toggle
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(Icons.Default.Translate, contentDescription = null)
                                Spacer(modifier = Modifier.width(8.dp))
                                Column {
                                    Text(
                                        text = stringResource(R.string.lang_auto_detect),
                                        style = MaterialTheme.typography.bodyLarge
                                    )
                                    detectedLanguage?.let { lang ->
                                        Text(
                                            text = stringResource(R.string.language_detected, lang),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.primary,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }
                            Switch(
                                checked = autoDetectSource,
                                onCheckedChange = { viewModel.toggleAutoDetect(it) }
                            )
                        }
                    }
                }
            }
            
            // Language Selection Card
            AnimatedVisibility(
                visible = true,
                enter = fadeIn() + slideInVertically()
            ) {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            // Source Language Dropdown
                            ExposedDropdownMenuBox(
                                expanded = sourceExpanded,
                                onExpandedChange = { sourceExpanded = it },
                                modifier = Modifier.weight(1f)
                            ) {
                                OutlinedTextField(
                                    value = if (autoDetectSource) stringResource(R.string.lang_auto_detect) else stringResource(supportedLanguages[sourceLanguageIndex].nameResId),
                                    onValueChange = {},
                                    readOnly = true,
                                    enabled = !autoDetectSource,
                                    label = { Text(stringResource(R.string.translator_source_lang)) },
                                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = sourceExpanded) },
                                    modifier = Modifier
                                        .menuAnchor()
                                        .fillMaxWidth(),
                                    shape = RoundedCornerShape(16.dp)
                                )
                                ExposedDropdownMenu(
                                    expanded = sourceExpanded,
                                    onDismissRequest = { sourceExpanded = false }
                                ) {
                                    supportedLanguages.forEachIndexed { index, language ->
                                        DropdownMenuItem(
                                            text = { Text(stringResource(language.nameResId)) },
                                            onClick = {
                                                sourceLanguageIndex = index
                                                sourceExpanded = false
                                            }
                                        )
                                    }
                                }
                            }

                            // Swap Languages Button
                            IconButton(
                                onClick = {
                                    val temp = sourceLanguageIndex
                                    sourceLanguageIndex = targetLanguageIndex
                                    targetLanguageIndex = temp
                                },
                                enabled = !autoDetectSource
                            ) {
                                Icon(Icons.Default.SwapHoriz, contentDescription = "Swap languages")
                            }

                            // Target Language Dropdown
                            ExposedDropdownMenuBox(
                                expanded = targetExpanded,
                                onExpandedChange = { targetExpanded = it },
                                modifier = Modifier.weight(1f)
                            ) {
                                OutlinedTextField(
                                    value = stringResource(supportedLanguages[targetLanguageIndex].nameResId),
                                    onValueChange = {},
                                    readOnly = true,
                                    label = { Text(stringResource(R.string.translator_target_lang)) },
                                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = targetExpanded) },
                                    modifier = Modifier
                                        .menuAnchor()
                                        .fillMaxWidth(),
                                    shape = RoundedCornerShape(16.dp)
                                )
                                ExposedDropdownMenu(
                                    expanded = targetExpanded,
                                    onDismissRequest = { targetExpanded = false }
                                ) {
                                    supportedLanguages.forEachIndexed { index, language ->
                                        DropdownMenuItem(
                                            text = { Text(stringResource(language.nameResId)) },
                                            onClick = {
                                                targetLanguageIndex = index
                                                targetExpanded = false
                                            }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Image Picker (if vision enabled)
            AnimatedVisibility(
                visible = visionEnabled,
                enter = fadeIn() + slideInVertically()
            ) {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = { imagePickerLauncher.launch("image/*") },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Icon(Icons.Default.AddPhotoAlternate, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                if (inputImageUri != null) stringResource(R.string.change_image)
                                else stringResource(R.string.select_image),
                                fontWeight = FontWeight.Bold
                            )
                        }
                        
                        inputImageUri?.let { uri ->
                            AsyncImage(
                                model = uri,
                                contentDescription = "Selected image",
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(200.dp)
                                    .background(
                                        MaterialTheme.colorScheme.surface,
                                        RoundedCornerShape(16.dp)
                                    )
                            )
                        }
                    }
                }
            }
            
            // Input Text Field
            AnimatedVisibility(
                visible = true,
                enter = fadeIn() + slideInVertically()
            ) {
                Card(
                    shape = RoundedCornerShape(24.dp)
                ) {
                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { viewModel.setInputText(it) },
                        label = { Text(stringResource(R.string.translator_input_label), fontWeight = FontWeight.Bold) },
                        placeholder = { Text(stringResource(R.string.translator_input_hint)) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                            .padding(12.dp),
                        maxLines = 10,
                        shape = RoundedCornerShape(16.dp)
                    )
                }
            }
            
            // Translate Button
            Button(
                onClick = {
                    viewModel.translate(
                        sourceLanguage = supportedLanguages[sourceLanguageIndex],
                        targetLanguage = supportedLanguages[targetLanguageIndex]
                    )
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = (inputText.isNotBlank() || inputImageUri != null) && !isTranslating && selectedModel != null,
                shape = RoundedCornerShape(16.dp)
            ) {
                if (isTranslating) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.translating), fontWeight = FontWeight.Bold)
                } else {
                    Icon(Icons.Default.Translate, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.translator_translate), fontWeight = FontWeight.Bold)
                }
            }
            
            // Output Text Field
            AnimatedVisibility(
                visible = outputText.isNotEmpty(),
                enter = fadeIn() + slideInVertically()
            ) {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    OutlinedTextField(
                        value = outputText,
                        onValueChange = {},
                        label = { Text(stringResource(R.string.translator_result), fontWeight = FontWeight.Bold) },
                        readOnly = true,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                            .padding(12.dp),
                        maxLines = 10,
                        shape = RoundedCornerShape(16.dp)
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TranscriberScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    viewModel: TranscriberViewModel = viewModel()
) {
    // ViewModel states
    val availableModels by viewModel.availableModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val isTranscribing by viewModel.isTranscribing.collectAsState()
    val isRecording by viewModel.isRecording.collectAsState()
    val loadError by viewModel.loadError.collectAsState()
    val transcriptionText by viewModel.transcriptionText.collectAsState()
    
    // Snackbar
    val snackbarHostState = remember { SnackbarHostState() }
    
    LaunchedEffect(loadError) {
        loadError?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.transcriber_title), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState())
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Model Selection Card
            ModelSelectorCard(
                models = availableModels,
                selectedModel = selectedModel,
                selectedBackend = selectedBackend,
                isLoading = isLoadingModel,
                onModelSelected = { viewModel.selectModel(it) },
                onBackendSelected = { viewModel.selectBackend(it) },
                onLoadModel = { viewModel.loadModel() },
                filterMultimodalOnly = true
            )
            
            // Recording Card
            AnimatedVisibility(
                visible = selectedModel != null,
                enter = fadeIn() + slideInVertically()
            ) {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isRecording) MaterialTheme.colorScheme.errorContainer
                        else MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(120.dp)
                                .background(
                                    brush = Brush.linearGradient(
                                        colors = if (isRecording) listOf(
                                            MaterialTheme.colorScheme.error,
                                            MaterialTheme.colorScheme.error.copy(alpha = 0.7f)
                                        ) else listOf(
                                            MaterialTheme.colorScheme.primary,
                                            MaterialTheme.colorScheme.tertiary
                                        )
                                    ),
                                    shape = CircleShape
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                                contentDescription = null,
                                modifier = Modifier.size(60.dp),
                                tint = MaterialTheme.colorScheme.onPrimary
                            )
                        }
                        
                        Text(
                            text = if (isRecording) stringResource(R.string.recording)
                            else stringResource(R.string.record_voice_message),
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            textAlign = TextAlign.Center
                        )
                        
                        if (isRecording) {
                            Text(
                                text = stringResource(R.string.tap_to_stop),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        
                        Button(
                            onClick = {
                                // TODO: Implement actual audio recording
                                viewModel.setRecording(!isRecording)
                            },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = !isTranscribing,
                            shape = RoundedCornerShape(16.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = if (isRecording) MaterialTheme.colorScheme.error
                                else MaterialTheme.colorScheme.primary
                            )
                        ) {
                            Icon(
                                if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                                contentDescription = null
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                if (isRecording) stringResource(R.string.stop_recording)
                                else stringResource(R.string.record_audio),
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }
            }
            
            // Transcription Output
            AnimatedVisibility(
                visible = transcriptionText.isNotEmpty(),
                enter = fadeIn() + slideInVertically()
            ) {
                Card(
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = stringResource(R.string.transcription_result),
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.Bold
                            )
                            
                            if (isTranscribing) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp)
                                )
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        OutlinedTextField(
                            value = transcriptionText,
                            onValueChange = {},
                            readOnly = true,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(200.dp),
                            maxLines = 10,
                            shape = RoundedCornerShape(16.dp)
                        )
                        
                        Spacer(modifier = Modifier.height(8.dp))
                        
                        Button(
                            onClick = { viewModel.clearTranscription() },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Icon(Icons.Default.Delete, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(stringResource(R.string.clear), fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CodeAssistantScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.code_assistant_title), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                }
            )
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues),
            contentAlignment = Alignment.Center
        ) {
            AnimatedVisibility(
                visible = true,
                enter = fadeIn() + slideInVertically()
            ) {
                Card(
                    modifier = Modifier.padding(24.dp),
                    shape = RoundedCornerShape(24.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(16.dp),
                        modifier = Modifier.padding(32.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(80.dp)
                                .background(
                                    brush = Brush.linearGradient(
                                        colors = listOf(
                                            MaterialTheme.colorScheme.primary,
                                            MaterialTheme.colorScheme.tertiary
                                        )
                                    ),
                                    shape = CircleShape
                                ),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Default.Code,
                                contentDescription = null,
                                modifier = Modifier.size(40.dp),
                                tint = MaterialTheme.colorScheme.onPrimary
                            )
                        }
                        Text(
                            text = stringResource(R.string.code_assistant_title),
                            style = MaterialTheme.typography.headlineMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = stringResource(R.string.code_assistant_no_model),
                            style = MaterialTheme.typography.bodyLarge,
                            textAlign = TextAlign.Center,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Button(
                            onClick = onNavigateToModels,
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Icon(Icons.Default.Download, contentDescription = null)
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(stringResource(R.string.download_models), fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }
    }
}
