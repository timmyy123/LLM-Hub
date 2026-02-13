package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.screens.Language
import com.llmhub.llmhub.screens.languageCodeToEnglishName
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import java.util.UUID
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.withContext

class TranslatorViewModel(application: Application) : AndroidViewModel(application) {
    private val inferenceService = MediaPipeInferenceService(application)
    private val prefs = application.getSharedPreferences("translator_prefs", android.content.Context.MODE_PRIVATE)
    
    // Model selection state
    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()
    
    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()
    
    private val _selectedBackend = MutableStateFlow(LlmInference.Backend.GPU)
    val selectedBackend: StateFlow<LlmInference.Backend> = _selectedBackend.asStateFlow()

    // Optional selected NPU device id when user chooses NPU for GGUF
    private val _selectedNpuDeviceId = MutableStateFlow<String?>(null)
    val selectedNpuDeviceId: StateFlow<String?> = _selectedNpuDeviceId.asStateFlow()
    
    // Loading states
    private val _isLoadingModel = MutableStateFlow(false)
    val isLoadingModel: StateFlow<Boolean> = _isLoadingModel.asStateFlow()
    
    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()
    
    private val _isTranslating = MutableStateFlow(false)
    val isTranslating: StateFlow<Boolean> = _isTranslating.asStateFlow()
    // Track running translation to allow cancellation
    private var translationJob: Job? = null
    private var currentChatId: String? = null
    
    private val _loadError = MutableStateFlow<String?>(null)
    val loadError: StateFlow<String?> = _loadError.asStateFlow()
    
    // Modality toggles
    private val _visionEnabled = MutableStateFlow(false)
    val visionEnabled: StateFlow<Boolean> = _visionEnabled.asStateFlow()
    
    private val _audioEnabled = MutableStateFlow(false)
    val audioEnabled: StateFlow<Boolean> = _audioEnabled.asStateFlow()
    
    // Auto-detect language
    private val _autoDetectSource = MutableStateFlow(false)
    val autoDetectSource: StateFlow<Boolean> = _autoDetectSource.asStateFlow()
    
    private val _detectedLanguage = MutableStateFlow<String?>(null)
    val detectedLanguage: StateFlow<String?> = _detectedLanguage.asStateFlow()
    
    // Persisted language selections (ISO codes)
    private val _sourceLanguageCode = MutableStateFlow("en")
    val sourceLanguageCode: StateFlow<String> = _sourceLanguageCode.asStateFlow()
    
    private val _targetLanguageCode = MutableStateFlow("es")
    val targetLanguageCode: StateFlow<String> = _targetLanguageCode.asStateFlow()
    
    // Translation input/output
    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()
    
    private val _inputImageUri = MutableStateFlow<Uri?>(null)
    val inputImageUri: StateFlow<Uri?> = _inputImageUri.asStateFlow()
    
    private val _inputAudioUri = MutableStateFlow<Uri?>(null)
    val inputAudioUri: StateFlow<Uri?> = _inputAudioUri.asStateFlow()
    
    private val _inputAudioData = MutableStateFlow<ByteArray?>(null)
    val inputAudioData: StateFlow<ByteArray?> = _inputAudioData.asStateFlow()
    
    private val _outputText = MutableStateFlow("")
    val outputText: StateFlow<String> = _outputText.asStateFlow()
    
    // Input mode (exclusive: TEXT, IMAGE, or AUDIO)
    enum class InputMode { TEXT, IMAGE, AUDIO }
    private val _inputMode = MutableStateFlow(InputMode.TEXT)
    val inputMode: StateFlow<InputMode> = _inputMode.asStateFlow()
    
    init {
        loadAvailableModels()
        loadSavedSettings()
    }
    
    private fun loadAvailableModels() {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val allModels = ModelAvailabilityProvider.loadAvailableModels(context)
            val multimodalModels = allModels.filter { it.supportsVision || it.supportsAudio }
            _availableModels.value = multimodalModels

            // Restore saved model or use first as default
            val savedModelName = prefs.getString("selected_model_name", null)
            if (savedModelName != null) {
                val savedModel = multimodalModels.find { it.name == savedModelName }
                if (savedModel != null) {
                    _selectedModel.value = savedModel
                }
            }
            
            if (multimodalModels.isNotEmpty() && _selectedModel.value == null) {
                _selectedModel.value = multimodalModels.first()
            }
        }
    }
    
    private fun loadSavedSettings() {
        // Restore backend (store enum name, fallback to GPU)
        val savedBackendName = prefs.getString("selected_backend", LlmInference.Backend.GPU.name)
        _selectedBackend.value = try {
            LlmInference.Backend.valueOf(savedBackendName ?: LlmInference.Backend.GPU.name)
        } catch (_: IllegalArgumentException) {
            LlmInference.Backend.GPU
        }
        
        // Restore modality settings
        _visionEnabled.value = prefs.getBoolean("vision_enabled", false)
        _audioEnabled.value = prefs.getBoolean("audio_enabled", false)
        _autoDetectSource.value = prefs.getBoolean("auto_detect", false)

        // Restore language selections
        _sourceLanguageCode.value = prefs.getString("source_lang", "en") ?: "en"
        _targetLanguageCode.value = prefs.getString("target_lang", "es") ?: "es"
    }
    
    private fun saveSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            // Save backend using enum name to avoid non-exhaustive mappings
            putString("selected_backend", _selectedBackend.value.name)
            putBoolean("vision_enabled", _visionEnabled.value)
            putBoolean("audio_enabled", _audioEnabled.value)
            putBoolean("auto_detect", _autoDetectSource.value)
            putString("source_lang", _sourceLanguageCode.value)
            putString("target_lang", _targetLanguageCode.value)
            apply()
        }
    }

    fun setSourceLanguageCode(code: String) {
        _sourceLanguageCode.value = code
        saveSettings()
    }

    fun setTargetLanguageCode(code: String) {
        _targetLanguageCode.value = code
        saveSettings()
    }
    
    fun selectModel(model: LLMModel) {
        // Unload current model before switching
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedModel.value = model
        // Reset modality toggles based on model capabilities
        if (!model.supportsVision) {
            _visionEnabled.value = false
        }
        if (!model.supportsAudio) {
            _audioEnabled.value = false
        }
        saveSettings()
    }
    
    fun selectBackend(backend: LlmInference.Backend, deviceId: String? = null) {
        // Unload current model before switching backend
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedBackend.value = backend
        _selectedNpuDeviceId.value = deviceId
        saveSettings()
    }
    
    fun toggleVision(enabled: Boolean) {
        // Unload current model before changing vision setting
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _visionEnabled.value = enabled
        if (!enabled) {
            _inputImageUri.value = null
        }
        saveSettings()
    }
    
    fun toggleAudio(enabled: Boolean) {
        // Unload current model before changing audio setting
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _audioEnabled.value = enabled
        saveSettings()
    }
    
    fun toggleAutoDetect(enabled: Boolean) {
        _autoDetectSource.value = enabled
        if (!enabled) {
            _detectedLanguage.value = null
        }
        saveSettings()
    }
    
    fun setInputText(text: String) {
        _inputText.value = text
        if (text.isNotEmpty()) {
            _inputMode.value = InputMode.TEXT
            _inputImageUri.value = null
            _inputAudioUri.value = null
        }
    }
    
    fun setInputImage(uri: Uri?) {
        _inputImageUri.value = uri
        if (uri != null) {
            _inputMode.value = InputMode.IMAGE
            _inputText.value = ""
            _inputAudioUri.value = null
        }
    }
    
    fun setInputAudio(uri: Uri?) {
        _inputAudioUri.value = uri
        _inputAudioData.value = null
        if (uri != null) {
            _inputMode.value = InputMode.AUDIO
            _inputText.value = ""
            _inputImageUri.value = null
        }
    }
    
    fun setInputAudioData(data: ByteArray?) {
        _inputAudioData.value = data
        _inputAudioUri.value = null
        if (data != null) {
            _inputMode.value = InputMode.AUDIO
            _inputText.value = ""
            _inputImageUri.value = null
        }
    }
    
    fun clearInput() {
        _inputText.value = ""
        _inputImageUri.value = null
        _inputAudioUri.value = null
        _inputAudioData.value = null
        _inputMode.value = InputMode.TEXT
        _outputText.value = ""
    }
    
    fun clearError() {
        _loadError.value = null
    }
    
    fun loadModel() {
        val model = _selectedModel.value ?: return
        
        // Prevent concurrent loads
        if (_isLoadingModel.value || _isModelLoaded.value) {
            return
        }
        
        viewModelScope.launch {
            _isLoadingModel.value = true
            _loadError.value = null
            
            try {
                // Load model with appropriate modality settings
                val disableVision = !_visionEnabled.value
                val disableAudio = !_audioEnabled.value
                
                inferenceService.loadModel(
                    model = model,
                    preferredBackend = _selectedBackend.value,
                    disableVision = disableVision,
                    disableAudio = disableAudio,
                    deviceId = _selectedNpuDeviceId.value
                )
                _isModelLoaded.value = true
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Failed to load model"
                _isModelLoaded.value = false
            } finally {
                _isLoadingModel.value = false
            }
        }
    }
    
    fun unloadModel() {
        viewModelScope.launch {
            try {
                // Cancel any in-flight translation before unloading
                translationJob?.cancel()
                inferenceService.unloadModel()
                _isModelLoaded.value = false
                _outputText.value = ""
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Failed to unload model"
            }
        }
    }
    
    fun translate(
        sourceLanguage: Language,
        targetLanguage: Language
    ) {
        val model = _selectedModel.value ?: return
        // Cancel any previous translation before starting a new one
        translationJob?.cancel()
        
        translationJob = viewModelScope.launch {
            _isTranslating.value = true
            _outputText.value = ""
            _detectedLanguage.value = null
            
            try {
                val prompt = buildTranslationPrompt(
                    sourceLanguage = if (_autoDetectSource.value) null else sourceLanguage,
                    targetLanguage = targetLanguage,
                    inputText = _inputText.value,
                    hasImage = _inputImageUri.value != null,
                    hasAudio = _inputAudioUri.value != null || _inputAudioData.value != null
                )
                
                val images = if (_inputMode.value == InputMode.IMAGE) {
                    _inputImageUri.value?.let { uri ->
                        loadBitmapFromUri(uri)?.let { listOf(it) } ?: emptyList()
                    } ?: emptyList()
                } else {
                    emptyList()
                }
                
                val audioData = if (_inputMode.value == InputMode.AUDIO) {
                    // Prefer recorded audio data over URI
                    _inputAudioData.value ?: _inputAudioUri.value?.let { uri ->
                        loadAudioFromUri(uri)
                    }
                } else {
                    null
                }
                
                val chatId = "translator-${UUID.randomUUID()}"

                val responseFlow = inferenceService.generateResponseStreamWithSession(
                    prompt = prompt,
                    model = model,
                    chatId = chatId,
                    images = images,
                    audioData = audioData,
                    webSearchEnabled = false
                )

                responseFlow.collect { token ->
                    if (token.isNotEmpty()) {
                        _outputText.value += token
                    }
                }
            } catch (_: CancellationException) {
                // Swallow cancellation - user-initiated cancel
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Translation failed"
            } finally {
                _isTranslating.value = false
            }
        }
    }

    fun cancelTranslation() {
        translationJob?.cancel()
    }
    
    private fun buildTranslationPrompt(
        sourceLanguage: Language?,
        targetLanguage: Language,
        inputText: String,
        hasImage: Boolean,
        hasAudio: Boolean
    ): String {
        // Get full English names for languages
        val targetLangName = languageCodeToEnglishName[targetLanguage.code] ?: targetLanguage.code
        val sourceLangName = sourceLanguage?.let { languageCodeToEnglishName[it.code] ?: it.code }
        
        return when {
            hasImage && sourceLanguage == null -> {
                // Auto-detect from image
                """You are a professional translator. 
Detect the language in the image and translate any text you see to $targetLangName.
Provide only the translation without explaining the detected language.
If there's also text input: $inputText, translate that as well.""".trimIndent()
            }
            hasImage && sourceLanguage != null -> {
                // Image with known source language
                """You are a professional translator.
Translate the text in the image from $sourceLangName to $targetLangName.
Provide only the translation.
${if (inputText.isNotBlank()) "Also translate this text: $inputText" else ""}""".trimIndent()
            }
            hasAudio && sourceLanguage == null -> {
                // Auto-detect from audio
                """You are a professional translator. 
Listen to the audio, transcribe what was said, detect the language, and translate the speech to $targetLangName.
Format your response exactly as follows:
Transcription: [the transcribed text in the original language]
Translation: [the translation in $targetLangName]""".trimIndent()
            }
            hasAudio && sourceLanguage != null -> {
                // Audio with known source language
                """You are a professional translator.
Listen to the audio, transcribe what was said in $sourceLangName, and translate it to $targetLangName.
Format your response exactly as follows:
Transcription: [the transcribed text in $sourceLangName]
Translation: [the translation in $targetLangName]""".trimIndent()
            }
            sourceLanguage == null -> {
                // Auto-detect from text
                """You are a professional translator.
Detect the language of the following text and translate it to $targetLangName.
Provide only the translation without explaining the detected language.

Text to translate:
$inputText""".trimIndent()
            }
            else -> {
                // Normal text translation
                """You are a professional translator.
Translate the following text from $sourceLangName to $targetLangName.
Provide only the translation.

Text to translate:
$inputText""".trimIndent()
            }
        }
    }

    private suspend fun loadBitmapFromUri(uri: Uri): Bitmap? {
        val app = getApplication<Application>()
        return withContext(Dispatchers.IO) {
            try {
                app.contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
            } catch (_: Exception) {
                null
            }
        }
    }
    
    private suspend fun loadAudioFromUri(uri: Uri): ByteArray? {
        val app = getApplication<Application>()
        return withContext(Dispatchers.IO) {
            try {
                app.contentResolver.openInputStream(uri)?.use { stream ->
                    stream.readBytes()
                }
            } catch (_: Exception) {
                null
            }
        }
    }
}
