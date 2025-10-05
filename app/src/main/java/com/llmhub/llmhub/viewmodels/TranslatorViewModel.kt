package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.screens.Language
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import java.util.UUID
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class TranslatorViewModel(application: Application) : AndroidViewModel(application) {
    private val inferenceService = MediaPipeInferenceService(application)
    
    // Model selection state
    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()
    
    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()
    
    private val _selectedBackend = MutableStateFlow(LlmInference.Backend.GPU)
    val selectedBackend: StateFlow<LlmInference.Backend> = _selectedBackend.asStateFlow()
    
    // Loading states
    private val _isLoadingModel = MutableStateFlow(false)
    val isLoadingModel: StateFlow<Boolean> = _isLoadingModel.asStateFlow()
    
    private val _isTranslating = MutableStateFlow(false)
    val isTranslating: StateFlow<Boolean> = _isTranslating.asStateFlow()
    
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
    
    // Translation input/output
    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()
    
    private val _inputImageUri = MutableStateFlow<Uri?>(null)
    val inputImageUri: StateFlow<Uri?> = _inputImageUri.asStateFlow()
    
    private val _outputText = MutableStateFlow("")
    val outputText: StateFlow<String> = _outputText.asStateFlow()
    
    init {
        loadAvailableModels()
    }
    
    private fun loadAvailableModels() {
        viewModelScope.launch {
            // Filter to multimodal Gemma-3n models that support vision or audio
            val models = ModelData.models
                .filter { it.category != "embedding" }
                .filter { it.supportsVision || it.supportsAudio }
                .filter { it.isDownloaded }
            _availableModels.value = models
            
            // Auto-select first model if available
            if (models.isNotEmpty() && _selectedModel.value == null) {
                _selectedModel.value = models.first()
            }
        }
    }
    
    fun selectModel(model: LLMModel) {
        _selectedModel.value = model
        // Reset modality toggles based on model capabilities
        if (!model.supportsVision) {
            _visionEnabled.value = false
        }
        if (!model.supportsAudio) {
            _audioEnabled.value = false
        }
    }
    
    fun selectBackend(backend: LlmInference.Backend) {
        _selectedBackend.value = backend
    }
    
    fun toggleVision(enabled: Boolean) {
        _visionEnabled.value = enabled
        if (!enabled) {
            _inputImageUri.value = null
        }
    }
    
    fun toggleAudio(enabled: Boolean) {
        _audioEnabled.value = enabled
    }
    
    fun toggleAutoDetect(enabled: Boolean) {
        _autoDetectSource.value = enabled
        if (!enabled) {
            _detectedLanguage.value = null
        }
    }
    
    fun setInputText(text: String) {
        _inputText.value = text
    }
    
    fun setInputImage(uri: Uri?) {
        _inputImageUri.value = uri
    }
    
    fun clearError() {
        _loadError.value = null
    }
    
    fun loadModel() {
        val model = _selectedModel.value ?: return
        
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
                    disableAudio = disableAudio
                )
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Failed to load model"
            } finally {
                _isLoadingModel.value = false
            }
        }
    }
    
    fun translate(
        sourceLanguage: Language,
        targetLanguage: Language
    ) {
        val model = _selectedModel.value ?: return
        
        viewModelScope.launch {
            _isTranslating.value = true
            _outputText.value = ""
            _detectedLanguage.value = null
            
            try {
                val prompt = buildTranslationPrompt(
                    sourceLanguage = if (_autoDetectSource.value) null else sourceLanguage,
                    targetLanguage = targetLanguage,
                    inputText = _inputText.value,
                    hasImage = _inputImageUri.value != null
                )
                
                val images = if (_visionEnabled.value) {
                    _inputImageUri.value?.let { uri ->
                        loadBitmapFromUri(uri)?.let { listOf(it) } ?: emptyList()
                    } ?: emptyList()
                } else {
                    emptyList()
                }
                val chatId = "translator-${UUID.randomUUID()}"

                val responseFlow = inferenceService.generateResponseStreamWithSession(
                    prompt = prompt,
                    model = model,
                    chatId = chatId,
                    images = images,
                    audioData = null,
                    webSearchEnabled = false
                )

                responseFlow.collect { token ->
                    if (token.isNotEmpty()) {
                        _outputText.value += token

                        if (_autoDetectSource.value && _detectedLanguage.value == null) {
                            detectLanguageFromResponse(_outputText.value)
                        }
                    }
                }
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Translation failed"
            } finally {
                _isTranslating.value = false
            }
        }
    }
    
    private fun buildTranslationPrompt(
        sourceLanguage: Language?,
        targetLanguage: Language,
        inputText: String,
        hasImage: Boolean
    ): String {
        return when {
            hasImage && sourceLanguage == null -> {
                // Auto-detect from image
                """You are a professional translator. 
Detect the language in the image and translate any text you see to ${targetLanguage.code}.
Provide only the translation without explaining the detected language.
If there's also text input: $inputText, translate that as well.""".trimIndent()
            }
            hasImage && sourceLanguage != null -> {
                // Image with known source language
                """You are a professional translator.
Translate the text in the image from ${sourceLanguage.code} to ${targetLanguage.code}.
Provide only the translation.
${if (inputText.isNotBlank()) "Also translate this text: $inputText" else ""}""".trimIndent()
            }
            sourceLanguage == null -> {
                // Auto-detect from text
                """You are a professional translator.
Detect the language of the following text and translate it to ${targetLanguage.code}.
Provide only the translation without explaining the detected language.

Text to translate:
$inputText""".trimIndent()
            }
            else -> {
                // Normal text translation
                """You are a professional translator.
Translate the following text from ${sourceLanguage.code} to ${targetLanguage.code}.
Provide only the translation.

Text to translate:
$inputText""".trimIndent()
            }
        }
    }
    
    private fun detectLanguageFromResponse(response: String) {
        // Simple heuristic to detect language mentions in response
        // This is a basic implementation - could be enhanced with better detection
        val languageMentions = listOf(
            "English", "Spanish", "French", "German", "Chinese", "Japanese",
            "Korean", "Arabic", "Russian", "Portuguese", "Italian", "Hindi"
        )
        
        for (lang in languageMentions) {
            if (response.contains(lang, ignoreCase = true)) {
                _detectedLanguage.value = lang
                break
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
}
