package com.llmhub.llmhub.screens

import android.Manifest
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.llmhub.llmhub.R
import com.llmhub.llmhub.components.ModelSelectorCard
import com.llmhub.llmhub.components.SelectableMarkdownText
import com.llmhub.llmhub.ui.components.AudioInputService
import com.llmhub.llmhub.viewmodels.TranslatorViewModel
import com.llmhub.llmhub.viewmodels.TranscriberViewModel
import kotlinx.coroutines.launch
import java.util.Locale

// Language data class
data class Language(val code: String, val nameResId: Int)

// Map of language codes to English names for AI prompts
val languageCodeToEnglishName = mapOf(
    "en" to "English",
    "af" to "Afrikaans",
    "am" to "Amharic",
    "ar" to "Arabic",
    "hy" to "Armenian",
    "az" to "Azerbaijani",
    "eu" to "Basque",
    "bn" to "Bengali",
    "bg" to "Bulgarian",
    "my" to "Burmese",
    "ca" to "Catalan",
    "zh-CN" to "Chinese (Simplified)",
    "zh-TW" to "Chinese (Traditional)",
    "hr" to "Croatian",
    "cs" to "Czech",
    "da" to "Danish",
    "nl" to "Dutch",
    "et" to "Estonian",
    "tl" to "Filipino",
    "fi" to "Finnish",
    "fr" to "French",
    "gl" to "Galician",
    "ka" to "Georgian",
    "de" to "German",
    "el" to "Greek",
    "gu" to "Gujarati",
    "ha" to "Hausa",
    "he" to "Hebrew",
    "hi" to "Hindi",
    "hu" to "Hungarian",
    "is" to "Icelandic",
    "ig" to "Igbo",
    "id" to "Indonesian",
    "it" to "Italian",
    "ja" to "Japanese",
    "kn" to "Kannada",
    "kk" to "Kazakh",
    "km" to "Khmer",
    "ko" to "Korean",
    "lo" to "Lao",
    "lv" to "Latvian",
    "lt" to "Lithuanian",
    "ms" to "Malay",
    "ml" to "Malayalam",
    "mr" to "Marathi",
    "ne" to "Nepali",
    "no" to "Norwegian",
    "ps" to "Pashto",
    "fa" to "Persian",
    "pl" to "Polish",
    "pt" to "Portuguese",
    "pa" to "Punjabi",
    "ro" to "Romanian",
    "ru" to "Russian",
    "sr" to "Serbian",
    "sd" to "Sindhi",
    "si" to "Sinhala",
    "sk" to "Slovak",
    "sl" to "Slovenian",
    "so" to "Somali",
    "es" to "Spanish",
    "sw" to "Swahili",
    "sv" to "Swedish",
    "ta" to "Tamil",
    "te" to "Telugu",
    "th" to "Thai",
    "tr" to "Turkish",
    "uk" to "Ukrainian",
    "ur" to "Urdu",
    "uz" to "Uzbek",
    "vi" to "Vietnamese",
    "yo" to "Yoruba",
    "zu" to "Zulu"
)

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
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val keyboard = LocalSoftwareKeyboardController.current
    
    // ViewModel states
    val availableModels by viewModel.availableModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val isTranslating by viewModel.isTranslating.collectAsState()
    val loadError by viewModel.loadError.collectAsState()
    val autoDetectSource by viewModel.autoDetectSource.collectAsState()
    val detectedLanguage by viewModel.detectedLanguage.collectAsState()
    val inputText by viewModel.inputText.collectAsState()
    val inputImageUri by viewModel.inputImageUri.collectAsState()
    val inputAudioUri by viewModel.inputAudioUri.collectAsState()
    val inputMode by viewModel.inputMode.collectAsState()
    val outputText by viewModel.outputText.collectAsState()
    val visionEnabled by viewModel.visionEnabled.collectAsState()
    val audioEnabled by viewModel.audioEnabled.collectAsState()
    
    // Clipboard manager
    val clipboardManager = LocalClipboardManager.current
    
    // TTS Service
    val ttsService = remember { com.llmhub.llmhub.ui.components.TtsService(context) }
    val isTtsSpeaking by ttsService.isSpeaking.collectAsState()
    
    // Parsed transcription and translation for audio mode
    var transcriptionText by remember { mutableStateOf("") }
    var translationText by remember { mutableStateOf("") }
    
    // Parse output when it changes
    LaunchedEffect(outputText, inputMode) {
        if (inputMode == TranslatorViewModel.InputMode.AUDIO && 
            outputText.contains("Transcription:") && 
            outputText.contains("Translation:")) {
            
            transcriptionText = outputText
                .substringAfter("Transcription:", "")
                .substringBefore("Translation:", "")
                .trim()
            
            translationText = outputText
                .substringAfter("Translation:", "")
                .trim()
        } else {
            transcriptionText = ""
            translationText = ""
        }
    }
    
    // Audio recording states
    var isRecording by remember { mutableStateOf(false) }
    var recordedAudioData by remember { mutableStateOf<ByteArray?>(null) }
    var hasAudioPermission by remember { mutableStateOf(false) }
    
    // Audio playback states
    var audioPlayer by remember { mutableStateOf<android.media.MediaPlayer?>(null) }
    var isAudioPlaying by remember { mutableStateOf(false) }
    var audioCurrentPosition by remember { mutableStateOf(0) }
    var audioDuration by remember { mutableStateOf(0) }
    
    // Audio service
    val audioService = remember { AudioInputService(context) }

    // Observe elapsed time from audio service for countdown and auto-stop
    val elapsedTimeMs by audioService.elapsedTimeMs.collectAsState()
    val remainingSeconds = ((29500L - elapsedTimeMs) / 1000).coerceAtLeast(0)

    // UI state (initialize from persisted codes)
    val persistedSourceCode by viewModel.sourceLanguageCode.collectAsState()
    val persistedTargetCode by viewModel.targetLanguageCode.collectAsState()
    var sourceLanguageIndex by remember(persistedSourceCode) { mutableStateOf(supportedLanguages.indexOfFirst { it.code == persistedSourceCode }.coerceAtLeast(0)) }
    var targetLanguageIndex by remember(persistedTargetCode) { mutableStateOf(supportedLanguages.indexOfFirst { it.code == persistedTargetCode }.coerceAtLeast(0)) }
    var sourceExpanded by remember { mutableStateOf(false) }
    var targetExpanded by remember { mutableStateOf(false) }
    var showSettingsSheet by remember { mutableStateOf(false) }

    // If user changes languages or auto-detect setting, make sure any TTS state resets so icons don't stick
    LaunchedEffect(sourceLanguageIndex, targetLanguageIndex, autoDetectSource) {
        if (isTtsSpeaking) {
            ttsService.stop()
        }
    }

    // Helpers to choose appropriate TTS locale for source/target
    fun codeToLocale(code: String): Locale =
        Locale.forLanguageTag(code).let { tagLocale ->
            if (tagLocale.language.isNullOrBlank()) Locale(code) else tagLocale
        }

    fun localeForTarget(): Locale = codeToLocale(persistedTargetCode.ifBlank { Locale.getDefault().toLanguageTag() })

    fun localeForSource(): Locale {
        val code = if (!autoDetectSource) {
            persistedSourceCode
        } else {
            // Try detected language first when auto-detect is on; fallback to persisted code or device default
            val detected = detectedLanguage?.trim().orEmpty()
            val supportedCodes = supportedLanguages.map { it.code }.toSet()
            when {
                detected in supportedCodes -> detected
                persistedSourceCode.isNotBlank() -> persistedSourceCode
                else -> Locale.getDefault().toLanguageTag()
            }
        }
        return codeToLocale(code)
    }

    // Pickers
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.setInputImage(it) }
    }
    
    // Audio permission launcher
    val audioPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasAudioPermission = isGranted
        if (isGranted) {
            isRecording = true
        }
    }
    
    // Check audio permission on first composition and register auto-stop callback
    LaunchedEffect(Unit) {
        hasAudioPermission = audioService.hasAudioPermission()

        // When the service auto-stops (time limit reached), update UI state as if user pressed stop
        audioService.onRecordingAutoStopped = {
            isRecording = false
        }
    }
    
    // Audio recording effect
    LaunchedEffect(isRecording) {
        if (isRecording && hasAudioPermission) {
            val success = audioService.startRecording()
            if (!success) {
                isRecording = false
            }
        } else if (!isRecording) {
            // Stop recording when isRecording becomes false (either manual or auto-stop)
            if (audioService.isRecording() || recordedAudioData == null) {
                val audioData = audioService.stopRecording()
                if (audioData != null) {
                    recordedAudioData = audioData
                    // Set the audio data in the viewmodel
                    viewModel.setInputAudioData(audioData)
                }
            }
        }
    }
    
    // Audio playback progress tracking
    LaunchedEffect(isAudioPlaying) {
        while (isAudioPlaying && audioPlayer != null) {
            audioCurrentPosition = audioPlayer?.currentPosition ?: 0
            audioDuration = audioPlayer?.duration ?: 0
            kotlinx.coroutines.delay(50) // Update every 50ms
        }
    }
    
    // Cleanup on dispose
    DisposableEffect(Unit) {
        onDispose {
            if (audioService.isRecording()) {
                coroutineScope.launch {
                    audioService.stopRecording()
                }
            }
            // Release audio player
            audioPlayer?.release()
            audioPlayer = null
            // Unload model to free memory and shutdown TTS
            viewModel.unloadModel()
            ttsService.shutdown()
        }
    }

    // Scroll state for auto-scrolling
    val scrollState = rememberScrollState()
    
    // Snackbar
    val snackbarHostState = remember { SnackbarHostState() }
    
    LaunchedEffect(loadError) {
        loadError?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }
    
    // Auto-scroll to bottom when output text changes (during generation)
    LaunchedEffect(outputText) {
        if (outputText.isNotEmpty() && isTranslating) {
            coroutineScope.launch {
                scrollState.animateScrollTo(scrollState.maxValue)
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
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = stringResource(R.string.feature_settings_title),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
                
                ModelSelectorCard(
                    models = availableModels,
                    selectedModel = selectedModel,
                    selectedBackend = selectedBackend,
                    isLoading = isLoadingModel,
                    isModelLoaded = isModelLoaded,
                    onModelSelected = { viewModel.selectModel(it) },
                    onBackendSelected = { viewModel.selectBackend(it) },
                    onLoadModel = { viewModel.loadModel() },
                    onUnloadModel = { viewModel.unloadModel() },
                    filterMultimodalOnly = true,
                    modifier = Modifier.fillMaxWidth()
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Vision toggle
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = stringResource(R.string.translator_enable_vision),
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = stringResource(R.string.translator_vision_description),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    Switch(
                        checked = visionEnabled,
                        onCheckedChange = { viewModel.toggleVision(it) },
                        enabled = selectedModel?.supportsVision == true
                    )
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                // Audio toggle
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = stringResource(R.string.translator_enable_audio),
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = stringResource(R.string.translator_audio_description),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    Switch(
                        checked = audioEnabled,
                        onCheckedChange = { viewModel.toggleAudio(it) },
                        enabled = selectedModel?.supportsAudio == true
                    )
                }
                
                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.translator_title), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = "Settings")
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
        ) {
            // Language Selection Bar
            Card(
                shape = RoundedCornerShape(0.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    // Source Language
                    Box(modifier = Modifier.weight(1f)) {
                        TextButton(
                            onClick = { sourceExpanded = true },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = if (autoDetectSource) stringResource(R.string.lang_auto_detect) 
                                       else stringResource(supportedLanguages[sourceLanguageIndex].nameResId),
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        DropdownMenu(
                            expanded = sourceExpanded,
                            onDismissRequest = { sourceExpanded = false },
                            modifier = Modifier.heightIn(max = 500.dp)
                        ) {
                            // Auto detect option on top
                            DropdownMenuItem(
                                text = { Text(stringResource(R.string.lang_auto_detect)) },
                                onClick = {
                                    sourceExpanded = false
                                    // Enable auto-detect and disable source selection
                                    viewModel.toggleAutoDetect(true)
                                }
                            )
                            Divider()
                            supportedLanguages.forEachIndexed { index, language ->
                                DropdownMenuItem(
                                    text = { Text(stringResource(language.nameResId)) },
                                    onClick = {
                                        sourceLanguageIndex = index
                                        sourceExpanded = false
                                        // Disable auto-detect when user selects specific source
                                        if (autoDetectSource) viewModel.toggleAutoDetect(false)
                                        viewModel.setSourceLanguageCode(supportedLanguages[index].code)
                                    }
                                )
                            }
                        }
                    }
                    
                    // Swap Languages
                    IconButton(
                        onClick = {
                            val temp = sourceLanguageIndex
                            sourceLanguageIndex = targetLanguageIndex
                            targetLanguageIndex = temp
                        },
                        enabled = !autoDetectSource
                    ) {
                        Icon(Icons.Default.SwapHoriz, contentDescription = "Swap")
                    }
                    
                    // Target Language
                    Box(modifier = Modifier.weight(1f)) {
                        TextButton(
                            onClick = { targetExpanded = true },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = stringResource(supportedLanguages[targetLanguageIndex].nameResId),
                                style = MaterialTheme.typography.bodyLarge,
                                fontWeight = FontWeight.Bold
                            )
                        }
                        DropdownMenu(
                            expanded = targetExpanded,
                            onDismissRequest = { targetExpanded = false },
                            modifier = Modifier.heightIn(max = 500.dp)
                        ) {
                            supportedLanguages.forEachIndexed { index, language ->
                                DropdownMenuItem(
                                    text = { Text(stringResource(language.nameResId)) },
                                    onClick = {
                                        targetLanguageIndex = index
                                        targetExpanded = false
                                        viewModel.setTargetLanguageCode(supportedLanguages[index].code)
                                    }
                                )
                            }
                        }
                    }
                }
            }
            
            Divider()
            
            // Show "Load Model First" screen if model not loaded
            if (!isModelLoaded) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
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
                        text = stringResource(
                            if (availableModels.isEmpty()) R.string.download_models_first
                            else R.string.scam_detector_load_model
                        ),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = stringResource(R.string.scam_detector_load_model_desc),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(32.dp))
                    FilledTonalButton(
                        onClick = { 
                            if (availableModels.isEmpty()) onNavigateToModels()
                            else showSettingsSheet = true
                        },
                        modifier = Modifier.fillMaxWidth(0.6f)
                    ) {
                        Icon(
                            imageVector = if (availableModels.isEmpty()) Icons.Default.GetApp else Icons.Default.Tune,
                            contentDescription = null
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(
                            if (availableModels.isEmpty()) R.string.download_models
                            else R.string.feature_settings_title
                        ))
                    }
                }
            } else {
            
            // Box with scrollable content and fixed button at bottom
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .imePadding()
            ) {
                // Scrollable content (input + output)
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(scrollState)
                        .padding(bottom = 80.dp) // Space for fixed button
                ) {
                    // Input Area
                    Column(
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        // Text/Image/Audio Input based on mode
                        when (inputMode) {
                            TranslatorViewModel.InputMode.TEXT -> {
                                Column {
                                    OutlinedTextField(
                                        value = inputText,
                                        onValueChange = { viewModel.setInputText(it) },
                                        placeholder = { 
                                            Text(
                                                stringResource(
                                                    if (!isModelLoaded) R.string.load_model_to_start 
                                                    else R.string.translator_input_hint
                                                )
                                            ) 
                                        },
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(16.dp),
                                        minLines = 3,
                                        maxLines = 8,
                                        shape = RoundedCornerShape(12.dp),
                                        colors = OutlinedTextFieldDefaults.colors(
                                            focusedBorderColor = Color.Transparent,
                                            unfocusedBorderColor = Color.Transparent
                                        )
                                    )
                                    
                                    // Input action bar with paste and attachment buttons
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(horizontal = 16.dp, vertical = 4.dp),
                                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        // Paste button
                                        IconButton(
                                            onClick = { 
                                                val clipText = clipboardManager.getText()?.text
                                                if (!clipText.isNullOrBlank()) {
                                                    viewModel.setInputText(inputText + clipText)
                                                }
                                            }
                                        ) {
                                            Icon(
                                                Icons.Default.ContentPaste,
                                                contentDescription = "Paste",
                                                tint = MaterialTheme.colorScheme.primary
                                            )
                                        }
                                        
                                        // TTS button for input text (source language)
                                        if (inputText.isNotBlank()) {
                                            IconButton(
                                                onClick = { 
                                                    if (isTtsSpeaking) {
                                                        ttsService.stop()
                                                    } else {
                                                        // Set TTS to source language before speaking
                                                        ttsService.setLanguage(localeForSource())
                                                        ttsService.speak(inputText)
                                                    }
                                                }
                                            ) {
                                                Icon(
                                                    if (isTtsSpeaking) Icons.Default.Stop else Icons.Default.VolumeUp,
                                                    contentDescription = if (isTtsSpeaking) "Stop reading" else "Read aloud",
                                                    tint = if (isTtsSpeaking) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                                                )
                                            }
                                        }
                                        
                                        Spacer(modifier = Modifier.weight(1f))
                                        
                                        // Image attachment (only show if vision is enabled)
                                        if (visionEnabled && selectedModel?.supportsVision == true) {
                                            IconButton(
                                                onClick = { imagePickerLauncher.launch("image/*") }
                                            ) {
                                                Icon(
                                                    Icons.Default.Image,
                                                    contentDescription = "Attach image",
                                                    tint = MaterialTheme.colorScheme.primary
                                                )
                                            }
                                        }
                                        
                                        // Audio recording button (only show if audio is enabled)
                                        if (audioEnabled && selectedModel?.supportsAudio == true) {
                                            Row(
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                                            ) {
                                                if (isRecording) {
                                                    Text(
                                                        text = "${remainingSeconds}s ${stringResource(R.string.remaining)}",
                                                        style = MaterialTheme.typography.bodyMedium,
                                                        fontWeight = if (remainingSeconds <= 5) FontWeight.Bold else FontWeight.Normal,
                                                        color = if (remainingSeconds <= 5) 
                                                            MaterialTheme.colorScheme.error 
                                                        else 
                                                            MaterialTheme.colorScheme.onSurface
                                                    )
                                                }
                                                IconButton(
                                                    onClick = { 
                                                        if (isRecording) {
                                                            isRecording = false
                                                        } else {
                                                            if (hasAudioPermission) {
                                                                isRecording = true
                                                            } else {
                                                                audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                                            }
                                                        }
                                                    }
                                                ) {
                                                    Icon(
                                                        if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                                                        contentDescription = if (isRecording) "Stop recording" else "Record audio",
                                                        tint = if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                                                    )
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            TranslatorViewModel.InputMode.IMAGE -> {
                                inputImageUri?.let { uri ->
                                    Column(
                                        modifier = Modifier.padding(16.dp),
                                        verticalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        AsyncImage(
                                            model = uri,
                                            contentDescription = "Selected image",
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .height(250.dp)
                                                .background(
                                                    MaterialTheme.colorScheme.surfaceVariant,
                                                    RoundedCornerShape(12.dp)
                                                )
                                        )
                                        Row(
                                            modifier = Modifier.fillMaxWidth(),
                                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                                        ) {
                                            OutlinedButton(
                                                onClick = { imagePickerLauncher.launch("image/*") },
                                                modifier = Modifier.weight(1f)
                                            ) {
                                                Text("Change")
                                            }
                                            OutlinedButton(
                                                onClick = { viewModel.clearInput() },
                                                modifier = Modifier.weight(1f)
                                            ) {
                                                Text("Remove")
                                            }
                                        }
                                    }
                                }
                            }
                            TranslatorViewModel.InputMode.AUDIO -> {
                                Column(
                                    modifier = Modifier.padding(16.dp),
                                    verticalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    if (isRecording) {
                                        // Recording in progress
                                        Card(
                                            modifier = Modifier.fillMaxWidth(),
                                            colors = CardDefaults.cardColors(
                                                containerColor = MaterialTheme.colorScheme.errorContainer
                                            )
                                        ) {
                                            Row(
                                                modifier = Modifier.padding(16.dp),
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                                            ) {
                                                Icon(
                                                    Icons.Default.FiberManualRecord,
                                                    contentDescription = null,
                                                    modifier = Modifier.size(24.dp),
                                                    tint = MaterialTheme.colorScheme.error
                                                )
                                                Column {
                                                    Text(
                                                        text = "Recording...",
                                                        style = MaterialTheme.typography.bodyLarge,
                                                        color = MaterialTheme.colorScheme.onErrorContainer
                                                    )
                                                    Text(
                                                        text = "${remainingSeconds}s ${stringResource(R.string.remaining)}",
                                                        style = MaterialTheme.typography.bodyMedium,
                                                        fontWeight = if (remainingSeconds <= 5) FontWeight.Bold else FontWeight.Normal,
                                                        color = if (remainingSeconds <= 5) 
                                                            MaterialTheme.colorScheme.error 
                                                        else 
                                                            MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.7f)
                                                    )
                                                }
                                                Spacer(Modifier.weight(1f))
                                                IconButton(
                                                    onClick = { isRecording = false }
                                                ) {
                                                    Icon(
                                                        Icons.Default.Stop,
                                                        contentDescription = "Stop recording",
                                                        tint = MaterialTheme.colorScheme.error
                                                    )
                                                }
                                            }
                                        }
                                    } else if (recordedAudioData != null) {
                                        // Audio player card
                                        Card(
                                            modifier = Modifier.fillMaxWidth(),
                                            colors = CardDefaults.cardColors(
                                                containerColor = MaterialTheme.colorScheme.surfaceVariant
                                            )
                                        ) {
                                            Row(
                                                modifier = Modifier
                                                    .padding(12.dp)
                                                    .fillMaxWidth(),
                                                verticalAlignment = Alignment.CenterVertically,
                                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                                            ) {
                                                // Play/Pause button
                                                IconButton(
                                                    onClick = {
                                                        try {
                                                            if (isAudioPlaying) {
                                                                audioPlayer?.pause()
                                                                isAudioPlaying = false
                                                            } else {
                                                                if (audioPlayer == null) {
                                                                    // Write bytes to a temp file so MediaPlayer can read
                                                                    val tmp = java.io.File(context.cacheDir, "audio_${System.currentTimeMillis()}.wav")
                                                                    tmp.writeBytes(recordedAudioData!!)
                                                                    audioPlayer = android.media.MediaPlayer().apply {
                                                                        setDataSource(tmp.absolutePath)
                                                                        setOnCompletionListener {
                                                                            isAudioPlaying = false
                                                                            audioCurrentPosition = 0
                                                                            runCatching { tmp.delete() }
                                                                        }
                                                                        setOnPreparedListener { player ->
                                                                            audioDuration = player.duration
                                                                            player.start()
                                                                            isAudioPlaying = true
                                                                        }
                                                                        prepareAsync()
                                                                    }
                                                                } else {
                                                                    audioPlayer?.start()
                                                                    isAudioPlaying = true
                                                                }
                                                            }
                                                        } catch (e: Exception) {
                                                            android.util.Log.e("TranslatorScreen", "Audio playback failed: ${e.message}", e)
                                                            isAudioPlaying = false
                                                        }
                                                    },
                                                    modifier = Modifier.size(40.dp)
                                                ) {
                                                    Icon(
                                                        if (isAudioPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                                                        contentDescription = if (isAudioPlaying) "Pause" else "Play",
                                                        tint = MaterialTheme.colorScheme.primary
                                                    )
                                                }
                                                
                                                // Waveform visualization with progress
                                                val barHeights = remember {
                                                    val random = java.util.Random(recordedAudioData.hashCode().toLong())
                                                    List(32) { 0.35f + random.nextFloat() * 0.65f }
                                                }
                                                val waveformColor = MaterialTheme.colorScheme.primary
                                                val progress = if (audioDuration > 0) {
                                                    audioCurrentPosition.toFloat() / audioDuration.toFloat()
                                                } else 0f
                                                
                                                Canvas(
                                                    modifier = Modifier
                                                        .weight(1f)
                                                        .height(40.dp)
                                                ) {
                                                    val barCount = barHeights.size
                                                    val spacingPx = 3f
                                                    val totalSpacing = spacingPx * (barCount - 1)
                                                    val barWidth = (size.width - totalSpacing) / barCount
                                                    val centerY = size.height / 2f
                                                    val maxBarHeight = size.height
                                                    val activeColor = waveformColor
                                                    val inactiveColor = waveformColor.copy(alpha = 0.35f)
                                                    val progressBars = (progress * barCount).coerceIn(0f, barCount.toFloat())
                                                    
                                                    for (i in 0 until barCount) {
                                                        val height = (barHeights[i] * maxBarHeight).coerceAtMost(maxBarHeight)
                                                        val left = i * (barWidth + spacingPx)
                                                        val top = centerY - height / 2f
                                                        val color = if (i < progressBars) activeColor else inactiveColor
                                                        drawRoundRect(
                                                            color = color,
                                                            topLeft = androidx.compose.ui.geometry.Offset(left, top),
                                                            size = androidx.compose.ui.geometry.Size(barWidth, height),
                                                            cornerRadius = androidx.compose.ui.geometry.CornerRadius(barWidth / 2f)
                                                        )
                                                    }
                                                }
                                                
                                                // Delete button
                                                IconButton(
                                                    onClick = { 
                                                        audioPlayer?.release()
                                                        audioPlayer = null
                                                        isAudioPlaying = false
                                                        audioCurrentPosition = 0
                                                        audioDuration = 0
                                                        recordedAudioData = null
                                                        viewModel.clearInput()
                                                    }
                                                ) {
                                                    Icon(
                                                        Icons.Default.Delete,
                                                        contentDescription = "Delete",
                                                        tint = MaterialTheme.colorScheme.error
                                                    )
                                                }
                                            }
                                        }
                                        
                                        // Show transcription if available (after translation) - plain style (no Card)
                                        if (transcriptionText.isNotEmpty()) {
                                            Row(
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .padding(top = 8.dp),
                                                horizontalArrangement = Arrangement.SpaceBetween,
                                                verticalAlignment = Alignment.Top
                                            ) {
                                                Text(
                                                    text = transcriptionText,
                                                    style = MaterialTheme.typography.bodyLarge,
                                                    modifier = Modifier.weight(1f),
                                                    color = MaterialTheme.colorScheme.onSurface
                                                )
                                                // TTS button for source language (transcription)
                                                IconButton(
                                                    onClick = { 
                                                        if (isTtsSpeaking) {
                                                            ttsService.stop()
                                                        } else {
                                                            // Set TTS to source language before speaking
                                                            ttsService.setLanguage(localeForSource())
                                                            ttsService.speak(transcriptionText)
                                                        }
                                                    }
                                                ) {
                                                    Icon(
                                                        if (isTtsSpeaking) Icons.Default.Stop else Icons.Default.VolumeUp,
                                                        contentDescription = if (isTtsSpeaking) "Stop reading" else "Read aloud",
                                                        tint = if (isTtsSpeaking) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                                                        modifier = Modifier.size(20.dp)
                                                    )
                                                }
                                                // Copy button
                                                IconButton(
                                                    onClick = { 
                                                        clipboardManager.setText(AnnotatedString(transcriptionText))
                                                    }
                                                ) {
                                                    Icon(
                                                        Icons.Default.ContentCopy,
                                                        contentDescription = "Copy",
                                                        tint = MaterialTheme.colorScheme.primary,
                                                        modifier = Modifier.size(20.dp)
                                                    )
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    // Output Area - directly below input
                    if (outputText.isNotEmpty()) {
                        Divider()
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            // For audio mode, only show translation (transcription is above divider) - plain style (no Card)
                            if (inputMode == TranslatorViewModel.InputMode.AUDIO && translationText.isNotEmpty()) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.Top
                                ) {
                                    Text(
                                        text = translationText,
                                        style = MaterialTheme.typography.bodyLarge,
                                        modifier = Modifier.weight(1f)
                                    )
                                    // TTS button (target language)
                                    IconButton(
                                        onClick = { 
                                            if (isTtsSpeaking) {
                                                ttsService.stop()
                                            } else {
                                                // Set TTS to target language before speaking
                                                ttsService.setLanguage(localeForTarget())
                                                ttsService.speak(translationText)
                                            }
                                        }
                                    ) {
                                        Icon(
                                            if (isTtsSpeaking) Icons.Default.Stop else Icons.Default.VolumeUp,
                                            contentDescription = if (isTtsSpeaking) "Stop reading" else "Read aloud",
                                            tint = if (isTtsSpeaking) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier.size(20.dp)
                                        )
                                    }
                                    // Copy button
                                    IconButton(
                                        onClick = { 
                                            clipboardManager.setText(AnnotatedString(translationText))
                                        }
                                    ) {
                                        Icon(
                                            Icons.Default.ContentCopy,
                                            contentDescription = "Copy",
                                            tint = MaterialTheme.colorScheme.primary,
                                            modifier = Modifier.size(20.dp)
                                        )
                                    }
                                }
                            } else if (inputMode != TranslatorViewModel.InputMode.AUDIO) {
                                // Regular output for text/image modes
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.Top
                                ) {
                                    Text(
                                        text = outputText,
                                        style = MaterialTheme.typography.bodyLarge,
                                        modifier = Modifier.weight(1f)
                                    )
                                    // TTS button (target language)
                                    IconButton(
                                        onClick = { 
                                            if (isTtsSpeaking) {
                                                ttsService.stop()
                                            } else {
                                                // Set TTS to target language before speaking
                                                ttsService.setLanguage(localeForTarget())
                                                ttsService.speak(outputText)
                                            }
                                        }
                                    ) {
                                        Icon(
                                            if (isTtsSpeaking) Icons.Default.Stop else Icons.Default.VolumeUp,
                                            contentDescription = if (isTtsSpeaking) "Stop reading" else "Read aloud",
                                            tint = if (isTtsSpeaking) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                    // Copy button
                                    IconButton(
                                        onClick = { 
                                            clipboardManager.setText(AnnotatedString(outputText))
                                        }
                                    ) {
                                        Icon(
                                            Icons.Default.ContentCopy,
                                            contentDescription = "Copy",
                                            tint = MaterialTheme.colorScheme.primary
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Fixed Translate Button at bottom
                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth(),
                    shadowElevation = 8.dp,
                    color = MaterialTheme.colorScheme.surface
                ) {
                    if (isTranslating) {
                        // Show Cancel button while translating
                        OutlinedButton(
                            onClick = { viewModel.cancelTranslation() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.error
                            ),
                            border = ButtonDefaults.outlinedButtonBorder.copy(
                                brush = Brush.linearGradient(
                                    colors = listOf(MaterialTheme.colorScheme.error, MaterialTheme.colorScheme.error)
                                )
                            )
                        ) {
                            Icon(Icons.Default.StopCircle, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text(stringResource(R.string.translating_tap_to_cancel))
                        }
                    } else {
                        FilledTonalButton(
                            onClick = {
                                keyboard?.hide()
                                viewModel.translate(
                                    sourceLanguage = if (autoDetectSource) Language("", R.string.lang_auto_detect) else supportedLanguages[sourceLanguageIndex],
                                    targetLanguage = supportedLanguages[targetLanguageIndex]
                                )
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            enabled = (inputText.isNotBlank() || inputImageUri != null || inputAudioUri != null || recordedAudioData != null) && !isTranslating && isModelLoaded,
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text(stringResource(R.string.translator_translate))
                        }
                    }
                }
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
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
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
    
    // Audio recording components (same as Translator)
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val audioService = remember { AudioInputService(context) }
    var recordedAudioData by remember { mutableStateOf<ByteArray?>(null) }
    var hasAudioPermission by remember { mutableStateOf(false) }
    var audioPlayer by remember { mutableStateOf<android.media.MediaPlayer?>(null) }
    var isAudioPlaying by remember { mutableStateOf(false) }
    var audioCurrentPosition by remember { mutableStateOf(0) }
    var audioDuration by remember { mutableStateOf(0) }
    
    // Observe elapsed time from audio service
    val elapsedTimeMs by audioService.elapsedTimeMs.collectAsState()
    val remainingSeconds = ((29500L - elapsedTimeMs) / 1000).coerceAtLeast(0)

    val audioPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasAudioPermission = granted
        if (granted) viewModel.setRecording(true)
    }

    LaunchedEffect(Unit) { 
        hasAudioPermission = audioService.hasAudioPermission()
        
        // Set up callback for auto-stop
        audioService.onRecordingAutoStopped = {
            viewModel.setRecording(false)
        }
    }

    LaunchedEffect(isRecording) {
        if (isRecording && hasAudioPermission) {
            val ok = audioService.startRecording()
            if (!ok) viewModel.setRecording(false)
        } else if (!isRecording) {
            // Stop recording when isRecording becomes false (either manual or auto-stop)
            if (audioService.isRecording() || recordedAudioData == null) {
                val data = audioService.stopRecording()
                if (data != null) {
                    recordedAudioData = data
                    viewModel.setAudioData(data)
                }
            }
        }
    }

    LaunchedEffect(isAudioPlaying) {
        while (isAudioPlaying && audioPlayer != null) {
            audioCurrentPosition = audioPlayer?.currentPosition ?: 0
            audioDuration = audioPlayer?.duration ?: 0
            kotlinx.coroutines.delay(50)
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            if (audioService.isRecording()) {
                coroutineScope.launch { audioService.stopRecording() }
            }
            audioPlayer?.release(); audioPlayer = null
            // Unload model to free memory
            viewModel.unloadModel()
        }
    }

    // Scroll state for auto-scrolling
    val scrollState = rememberScrollState()
    
    var showSettingsSheet by remember { mutableStateOf(false) }
    val settingsSheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    
    // Auto-scroll to bottom when transcription text changes (during generation)
    LaunchedEffect(transcriptionText) {
        if (transcriptionText.isNotEmpty() && isTranscribing) {
            coroutineScope.launch {
                scrollState.animateScrollTo(scrollState.maxValue)
            }
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
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = stringResource(R.string.feature_settings_title))
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { paddingValues ->
        // Show "Load Model First" screen if model not loaded
        if (!isModelLoaded) {
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
                    text = stringResource(
                        if (availableModels.isEmpty()) R.string.download_models_first
                        else R.string.scam_detector_load_model
                    ),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(12.dp))
                Text(
                    text = stringResource(R.string.scam_detector_load_model_desc),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(32.dp))
                FilledTonalButton(
                    onClick = { 
                        if (availableModels.isEmpty()) onNavigateToModels()
                        else showSettingsSheet = true
                    },
                    modifier = Modifier.fillMaxWidth(0.6f)
                ) {
                    Icon(
                        imageVector = if (availableModels.isEmpty()) Icons.Default.GetApp else Icons.Default.Tune,
                        contentDescription = null
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(
                        if (availableModels.isEmpty()) R.string.download_models
                        else R.string.feature_settings_title
                    ))
                }
            }
        } else {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .imePadding()
            ) {
                // Scrollable content
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(scrollState)
                        .padding(bottom = 80.dp) // Space for fixed button
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Recording + Replay (same pattern as Translator)
                    if (selectedModel != null) {
                        Column(modifier = Modifier.fillMaxWidth()) {
                            if (recordedAudioData != null) {
                                Card(
                                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Row(
                                        modifier = Modifier.padding(12.dp).fillMaxWidth(),
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        IconButton(onClick = {
                                            try {
                                                if (isAudioPlaying) {
                                                    audioPlayer?.pause(); isAudioPlaying = false
                                                } else {
                                                    if (audioPlayer == null) {
                                                        val tmp = java.io.File(context.cacheDir, "audio_${System.currentTimeMillis()}.wav")
                                                        tmp.writeBytes(recordedAudioData!!)
                                                        audioPlayer = android.media.MediaPlayer().apply {
                                                            setDataSource(tmp.absolutePath)
                                                            setOnCompletionListener { isAudioPlaying = false; audioCurrentPosition = 0; runCatching { tmp.delete() } }
                                                            setOnPreparedListener { player -> audioDuration = player.duration; player.start(); isAudioPlaying = true }
                                                            prepareAsync()
                                                        }
                                                    } else {
                                                        audioPlayer?.start(); isAudioPlaying = true
                                                    }
                                                }
                                            } catch (_: Exception) { isAudioPlaying = false }
                                        }) { Icon(if (isAudioPlaying) Icons.Default.Pause else Icons.Default.PlayArrow, contentDescription = null, tint = MaterialTheme.colorScheme.primary) }

                                        val barHeights = remember { val rnd = java.util.Random(recordedAudioData.hashCode().toLong()); List(32) { 0.35f + rnd.nextFloat() * 0.65f } }
                                        val waveformColor = MaterialTheme.colorScheme.primary
                                        val progress = if (audioDuration > 0) audioCurrentPosition.toFloat()/audioDuration.toFloat() else 0f
                                        Canvas(modifier = Modifier.weight(1f).height(40.dp)) {
                                            val barCount = barHeights.size
                                            val spacingPx = 3f
                                            val totalSpacing = spacingPx * (barCount - 1)
                                            val barWidth = (size.width - totalSpacing) / barCount
                                            val centerY = size.height / 2f
                                            val maxBarHeight = size.height
                                            val activeColor = waveformColor
                                            val inactiveColor = waveformColor.copy(alpha = 0.35f)
                                            val progressBars = (progress * barCount).coerceIn(0f, barCount.toFloat())
                                            for (i in 0 until barCount) {
                                                val height = (barHeights[i] * maxBarHeight).coerceAtMost(maxBarHeight)
                                                val left = i * (barWidth + spacingPx)
                                                val top = centerY - height / 2f
                                                val color = if (i < progressBars) activeColor else inactiveColor
                                                drawRoundRect(color = color, topLeft = androidx.compose.ui.geometry.Offset(left, top), size = androidx.compose.ui.geometry.Size(barWidth, height), cornerRadius = androidx.compose.ui.geometry.CornerRadius(barWidth/2f))
                                            }
                                        }
                                        IconButton(onClick = {
                                            audioPlayer?.release(); audioPlayer = null; isAudioPlaying = false; audioCurrentPosition = 0; audioDuration = 0; recordedAudioData = null; viewModel.setAudioData(null)
                                        }) { Icon(Icons.Default.Delete, contentDescription = null, tint = MaterialTheme.colorScheme.error) }
                                    }
                                }
                            } else {
                                // Hero-style recording card UI (also used while recording)
                                Card(
                                    shape = RoundedCornerShape(24.dp),
                                    colors = CardDefaults.cardColors(
                                        containerColor = MaterialTheme.colorScheme.primaryContainer
                                    ),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Column(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(24.dp),
                                        horizontalAlignment = Alignment.CenterHorizontally,
                                        verticalArrangement = Arrangement.spacedBy(16.dp)
                                    ) {
                                        Box(
                                            modifier = Modifier
                                                .align(Alignment.CenterHorizontally)
                                                .size(120.dp)
                                                .background(
                                                    brush = Brush.linearGradient(
                                                        colors = listOf(
                                                            MaterialTheme.colorScheme.primary,
                                                            MaterialTheme.colorScheme.tertiary
                                                        )
                                                    ),
                                                    shape = CircleShape
                                                )
                                                .clickable {
                                                    if (isRecording) {
                                                        viewModel.setRecording(false)
                                                    } else {
                                                        if (hasAudioPermission) {
                                                            viewModel.setRecording(true)
                                                        } else {
                                                            audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                                        }
                                                    }
                                                },
                                            contentAlignment = Alignment.Center
                                        ) {
                                            Icon(
                                                Icons.Default.Mic,
                                                contentDescription = null,
                                                modifier = Modifier.size(60.dp),
                                                tint = if (isRecording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onPrimary
                                            )
                                        }

                                        Text(
                                            text = if (isRecording) stringResource(R.string.recording) else stringResource(R.string.record_voice_message),
                                            style = MaterialTheme.typography.titleLarge,
                                            fontWeight = FontWeight.Bold,
                                            textAlign = TextAlign.Center,
                                            modifier = Modifier.align(Alignment.CenterHorizontally)
                                        )
                                        
                                        // Show countdown timer when recording
                                        if (isRecording) {
                                            Text(
                                                text = "${remainingSeconds}s",
                                                style = MaterialTheme.typography.headlineMedium,
                                                fontWeight = FontWeight.Bold,
                                                color = if (remainingSeconds <= 5) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onPrimaryContainer,
                                                textAlign = TextAlign.Center,
                                                modifier = Modifier.align(Alignment.CenterHorizontally)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    if (transcriptionText.isNotEmpty()) {
                        Divider()
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Text(
                                text = transcriptionText,
                                style = MaterialTheme.typography.bodyLarge,
                                modifier = Modifier.fillMaxWidth()
                            )
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.End
                            ) {
                                val clipboard = LocalClipboardManager.current
                                IconButton(
                                    onClick = { 
                                        clipboard.setText(AnnotatedString(transcriptionText))
                                    }
                                ) {
                                    Icon(
                                        Icons.Default.ContentCopy,
                                        contentDescription = stringResource(R.string.copy),
                                        tint = MaterialTheme.colorScheme.primary
                                    )
                                }
                            }
                        }
                    }
                }
                
                // Fixed Transcribe button at bottom
                Surface(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth(),
                    shadowElevation = 8.dp,
                    color = MaterialTheme.colorScheme.surface
                ) {
                    if (isTranscribing) {
                        // Show Cancel button while transcribing
                        OutlinedButton(
                            onClick = { viewModel.cancelTranscription() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.error
                            ),
                            border = ButtonDefaults.outlinedButtonBorder.copy(
                                brush = Brush.linearGradient(
                                    colors = listOf(MaterialTheme.colorScheme.error, MaterialTheme.colorScheme.error)
                                )
                            )
                        ) {
                            Icon(Icons.Default.StopCircle, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text(stringResource(R.string.transcribing_tap_to_cancel))
                        }
                    } else {
                        FilledTonalButton(
                            onClick = { viewModel.transcribe() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            enabled = recordedAudioData != null && !isTranscribing && isModelLoaded,
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text(stringResource(R.string.transcriber_transcribe))
                        }
                    }
                }
            }
        }
    }
    
    // Settings Bottom Sheet
    if (showSettingsSheet) {
        ModalBottomSheet(
            onDismissRequest = { showSettingsSheet = false },
            sheetState = settingsSheetState
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp)
                    .padding(bottom = 32.dp)
                    .verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = stringResource(R.string.feature_settings_title),
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                
                // Model Selector (always enable audio, vision disabled)
                ModelSelectorCard(
                    models = availableModels,
                    selectedModel = selectedModel,
                    selectedBackend = selectedBackend,
                    isLoading = isLoadingModel,
                    isModelLoaded = isModelLoaded,
                    onModelSelected = { viewModel.selectModel(it) },
                    onBackendSelected = { viewModel.selectBackend(it) },
                    onLoadModel = { viewModel.loadModel() },
                    onUnloadModel = { viewModel.unloadModel() },
                    filterMultimodalOnly = true
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ScamDetectorScreen(
    onNavigateBack: () -> Unit,
    onNavigateToModels: () -> Unit,
    viewModel: com.llmhub.llmhub.viewmodels.ScamDetectorViewModel = viewModel()
) {
    val context = LocalContext.current
    val keyboard = LocalSoftwareKeyboardController.current
    val clipboardManager = LocalClipboardManager.current
    val coroutineScope = rememberCoroutineScope()
    
    // ViewModel states
    val availableModels by viewModel.availableModels.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val selectedBackend by viewModel.selectedBackend.collectAsState()
    val isLoadingModel by viewModel.isLoadingModel.collectAsState()
    val isModelLoaded by viewModel.isModelLoaded.collectAsState()
    val visionEnabled by viewModel.visionEnabled.collectAsState()
    val isAnalyzing by viewModel.isAnalyzing.collectAsState()
    val isFetchingUrl by viewModel.isFetchingUrl.collectAsState()
    val inputText by viewModel.inputText.collectAsState()
    val inputImageUri by viewModel.inputImageUri.collectAsState()
    val outputText by viewModel.outputText.collectAsState()
    val loadError by viewModel.loadError.collectAsState()
    
    // Settings sheet state
    var showSettingsSheet by remember { mutableStateOf(false) }
    
    // TTS Service
    val ttsService = remember { com.llmhub.llmhub.ui.components.TtsService(context) }
    val isTtsSpeaking by ttsService.isSpeaking.collectAsState()
    
    // Scroll state for auto-scrolling
    val scrollState = rememberScrollState()
    
    // Snackbar
    val snackbarHostState = remember { SnackbarHostState() }
    
    LaunchedEffect(loadError) {
        loadError?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }
    
    // Auto-scroll to bottom when output text changes (during generation)
    LaunchedEffect(outputText) {
        if (outputText.isNotEmpty() && isAnalyzing) {
            coroutineScope.launch {
                scrollState.animateScrollTo(scrollState.maxValue)
            }
        }
    }
    
    // Cleanup on dispose - unload model to free memory and shutdown TTS
    DisposableEffect(Unit) {
        onDispose {
            viewModel.unloadModel()
            ttsService.shutdown()
        }
    }
    
    // Image picker
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        viewModel.setInputImageUri(uri)
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
                
                // Model Selector
                ModelSelectorCard(
                    models = availableModels,
                    selectedModel = selectedModel,
                    selectedBackend = selectedBackend,
                    isLoading = isLoadingModel,
                    isModelLoaded = isModelLoaded,
                    onModelSelected = { viewModel.selectModel(it) },
                    onBackendSelected = { viewModel.selectBackend(it) },
                    onLoadModel = { viewModel.loadModel() },
                    onUnloadModel = { viewModel.unloadModel() }
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Vision toggle
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = stringResource(R.string.scam_detector_enable_vision),
                            style = MaterialTheme.typography.bodyLarge,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = stringResource(R.string.scam_detector_vision_desc),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    Switch(
                        checked = visionEnabled,
                        onCheckedChange = { viewModel.toggleVision(it) },
                        enabled = selectedModel?.supportsVision == true
                    )
                }
                
                Spacer(modifier = Modifier.height(32.dp))
            }
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.scam_detector_title), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = stringResource(R.string.back))
                    }
                },
                actions = {
                    IconButton(onClick = { showSettingsSheet = true }) {
                        Icon(Icons.Default.Tune, contentDescription = "Settings")
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
        ) {
            // Show "Load Model First" message if no model is loaded
            if (!isModelLoaded) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
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
                        text = stringResource(
                            if (availableModels.isEmpty()) R.string.download_models_first
                            else R.string.scam_detector_load_model
                        ),
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = stringResource(R.string.scam_detector_load_model_desc),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(32.dp))
                    FilledTonalButton(
                        onClick = { 
                            if (availableModels.isEmpty()) onNavigateToModels()
                            else showSettingsSheet = true
                        },
                        modifier = Modifier.fillMaxWidth(0.6f)
                    ) {
                        Icon(
                            imageVector = if (availableModels.isEmpty()) Icons.Default.GetApp else Icons.Default.Tune,
                            contentDescription = null
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(stringResource(
                            if (availableModels.isEmpty()) R.string.download_models
                            else R.string.feature_settings_title
                        ))
                    }
                }
            } else {
                // Show input/output interface when model is loaded
            // Box with scrollable content and fixed button at bottom
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .imePadding()
            ) {
                // Scrollable content (input + output)
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(scrollState)
                        .padding(bottom = 80.dp) // Space for fixed button
                ) {
                    // Input Area
                    Column(
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = stringResource(R.string.scam_detector_input_label),
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(16.dp)
                        )
                        
                        OutlinedTextField(
                            value = inputText,
                            onValueChange = { viewModel.setInputText(it) },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp)
                                .heightIn(min = 120.dp),
                            placeholder = { Text(stringResource(R.string.scam_detector_input_hint)) },
                            enabled = !isAnalyzing,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color.Transparent,
                                unfocusedBorderColor = Color.Transparent,
                                disabledBorderColor = Color.Transparent
                            )
                        )
                        
                        // Input action bar with paste button
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp, vertical = 4.dp),
                            horizontalArrangement = Arrangement.Start,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            // Paste button
                            IconButton(
                                onClick = { 
                                    val clipText = clipboardManager.getText()?.text
                                    if (!clipText.isNullOrBlank()) {
                                        viewModel.setInputText(inputText + clipText)
                                    }
                                }
                            ) {
                                Icon(
                                    Icons.Default.ContentPaste,
                                    contentDescription = "Paste",
                                    tint = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                        
                        // Image input section (only show if vision is enabled)
                        if (visionEnabled && selectedModel?.supportsVision == true) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(16.dp),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                if (inputImageUri != null) {
                                    Card(
                                        modifier = Modifier.weight(1f),
                                        colors = CardDefaults.cardColors(
                                            containerColor = MaterialTheme.colorScheme.surfaceVariant
                                        )
                                    ) {
                                        Box {
                                            AsyncImage(
                                                model = inputImageUri,
                                                contentDescription = "Selected image",
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .heightIn(max = 200.dp)
                                            )
                                            IconButton(
                                                onClick = { viewModel.setInputImageUri(null) },
                                                modifier = Modifier.align(Alignment.TopEnd)
                                            ) {
                                                Icon(Icons.Default.Close, "Remove image")
                                            }
                                        }
                                    }
                                }
                                
                                if (inputImageUri == null) {
                                    OutlinedButton(
                                        onClick = { imagePickerLauncher.launch("image/*") },
                                        modifier = Modifier.weight(1f)
                                    ) {
                                        Icon(Icons.Default.Image, contentDescription = null)
                                        Spacer(Modifier.width(8.dp))
                                        Text("Add Image")
                                    }
                                }
                            }
                        }
                    }
                    
                    // URL fetching indicator
                    if (isFetchingUrl) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(16.dp))
                            Text(
                                text = stringResource(R.string.scam_detector_fetching_url),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                    
                    // Output Area
                    if (outputText.isNotEmpty()) {
                        Divider()
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp)
                        ) {
                            SelectableMarkdownText(
                                markdown = outputText,
                                color = MaterialTheme.colorScheme.onSurface,
                                fontSize = MaterialTheme.typography.bodyLarge.fontSize,
                                modifier = Modifier.fillMaxWidth()
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.End
                            ) {
                                // TTS button
                                IconButton(
                                    onClick = {
                                        if (isTtsSpeaking) {
                                            ttsService.stop()
                                        } else {
                                            // Set language to app locale before speaking
                                            val appLocale = com.llmhub.llmhub.utils.LocaleHelper.getCurrentLocale(context)
                                            ttsService.setLanguage(appLocale)
                                            ttsService.speak(outputText)
                                        }
                                    }
                                ) {
                                    Icon(
                                        if (isTtsSpeaking) Icons.Default.Stop else Icons.Default.VolumeUp,
                                        contentDescription = if (isTtsSpeaking) "Stop reading" else "Read aloud",
                                        tint = if (isTtsSpeaking) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                                // Copy button
                                IconButton(
                                    onClick = {
                                        clipboardManager.setText(AnnotatedString(outputText))
                                    }
                                ) {
                                    Icon(
                                        Icons.Default.ContentCopy,
                                        contentDescription = "Copy",
                                        tint = MaterialTheme.colorScheme.primary
                                    )
                                }
                            }
                        }
                    }
                }
                
                // Fixed Analyze button at bottom
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .align(Alignment.BottomCenter),
                    shadowElevation = 8.dp,
                    color = MaterialTheme.colorScheme.surface
                ) {
                    if (isAnalyzing) {
                        FilledTonalButton(
                            onClick = { viewModel.cancelAnalysis() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.filledTonalButtonColors(
                                containerColor = MaterialTheme.colorScheme.errorContainer,
                                contentColor = MaterialTheme.colorScheme.onErrorContainer
                            )
                        ) {
                            Icon(Icons.Default.StopCircle, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text(stringResource(R.string.scam_detector_analyzing))
                        }
                    } else {
                        FilledTonalButton(
                            onClick = {
                                keyboard?.hide()
                                viewModel.analyze()
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            enabled = (inputText.isNotBlank() || inputImageUri != null) && !isAnalyzing && isModelLoaded,
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text(stringResource(R.string.scam_detector_analyze))
                        }
                    }
                }
            }
            } // end else (model loaded)
        }
    }
}


