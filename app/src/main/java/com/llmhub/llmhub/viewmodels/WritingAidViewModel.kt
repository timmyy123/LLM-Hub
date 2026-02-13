package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.content.Context
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class WritingAidViewModel(application: Application) : AndroidViewModel(application) {
    
    private val inferenceService = (application as com.llmhub.llmhub.LlmHubApplication).inferenceService
    private val prefs = application.getSharedPreferences("writing_aid_prefs", Context.MODE_PRIVATE)
    
    private var processingJob: Job? = null
    
    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()
    
    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()
    
    private val _selectedBackend = MutableStateFlow<LlmInference.Backend?>(null)
    val selectedBackend: StateFlow<LlmInference.Backend?> = _selectedBackend.asStateFlow()
    
    private val _selectedMode = MutableStateFlow(WritingMode.FRIENDLY)
    val selectedMode: StateFlow<WritingMode> = _selectedMode.asStateFlow()
    
    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()
    
    private val _outputText = MutableStateFlow("")
    val outputText: StateFlow<String> = _outputText.asStateFlow()
    
    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()
    
    init {
        loadAvailableModels()
        loadSavedSettings()
    }
    
    private fun loadSavedSettings() {
        // Restore backend
        val savedBackendName = prefs.getString("selected_backend", LlmInference.Backend.GPU.name)
        _selectedBackend.value = try {
            LlmInference.Backend.valueOf(savedBackendName ?: LlmInference.Backend.GPU.name)
        } catch (_: IllegalArgumentException) {
            LlmInference.Backend.GPU
        }
        
        // Restore writing mode
        val savedModeName = prefs.getString("selected_mode", WritingMode.FRIENDLY.name)
        _selectedMode.value = try {
            WritingMode.valueOf(savedModeName ?: WritingMode.FRIENDLY.name)
        } catch (_: IllegalArgumentException) {
            WritingMode.FRIENDLY
        }
        
        // Restore selected model by name
        val savedModelName = prefs.getString("selected_model_name", null)
        if (savedModelName != null) {
            viewModelScope.launch {
                kotlinx.coroutines.delay(100)
                val model = _availableModels.value.find { it.name == savedModelName }
                if (model != null) {
                    _selectedModel.value = model
                    // If restored model doesn't support GPU, force CPU (saved backend may be GPU from another model)
                    if (!model.supportsGpu && _selectedBackend.value == LlmInference.Backend.GPU) {
                        _selectedBackend.value = LlmInference.Backend.CPU
                    }
                }
            }
        }
    }
    
    private fun saveSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            putString("selected_backend", _selectedBackend.value?.name)
            putString("selected_mode", _selectedMode.value.name)
            apply()
        }
    }
    
    private fun loadAvailableModels() {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val available = ModelAvailabilityProvider.loadAvailableModels(context)
                .filter { it.category != "embedding" && !it.name.contains("Projector", ignoreCase = true) }
            _availableModels.value = available
            if (_selectedModel.value == null) {
                available.firstOrNull()?.let {
                    _selectedModel.value = it
                    _selectedBackend.value = if (it.supportsGpu) {
                        _selectedBackend.value ?: LlmInference.Backend.GPU
                    } else {
                        LlmInference.Backend.CPU
                    }
                }
            }
        }
    }
    
    fun selectModel(model: LLMModel) {
        // Unload current model before switching
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedModel.value = model
        _isModelLoaded.value = false
        
        // Force CPU when model doesn't support GPU (e.g. ONNX); otherwise keep or set backend
        _selectedBackend.value = if (model.supportsGpu) {
            _selectedBackend.value ?: LlmInference.Backend.GPU
        } else {
            LlmInference.Backend.CPU
        }
        
        saveSettings()
    }
    
    private val _selectedNpuDeviceId = MutableStateFlow<String?>(null)
    val selectedNpuDeviceId: StateFlow<String?> = _selectedNpuDeviceId.asStateFlow()

    fun selectBackend(backend: LlmInference.Backend, deviceId: String? = null) {
        // Unload current model before switching backend
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedBackend.value = backend
        _selectedNpuDeviceId.value = deviceId
        _isModelLoaded.value = false
        saveSettings()
    }
    
    fun selectMode(mode: WritingMode) {
        _selectedMode.value = mode
        saveSettings()
    }
    
    fun loadModel() {
        val model = _selectedModel.value ?: return
        val backend = _selectedBackend.value ?: return
        
        // Prevent concurrent loads
        if (_isLoading.value || _isModelLoaded.value) {
            return
        }
        
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            
            try {
                // Unload any existing model first
                inferenceService.unloadModel()
                
                // Load the selected model with text-only mode (disable vision and audio)
                val success = inferenceService.loadModel(
                    model = model,
                    preferredBackend = backend,
                    disableVision = true,  // Writing aid only needs text
                    disableAudio = true,
                    deviceId = _selectedNpuDeviceId.value
                )
                
                if (success) {
                    _isModelLoaded.value = true
                } else {
                    _errorMessage.value = "Failed to load model"
                }
            } catch (e: Exception) {
                _errorMessage.value = e.message ?: "Unknown error"
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    fun unloadModel() {
        viewModelScope.launch {
            try {
                inferenceService.unloadModel()
                _isModelLoaded.value = false
                _outputText.value = ""
            } catch (e: Exception) {
                _errorMessage.value = e.message ?: "Failed to unload model"
            }
        }
    }
    
    fun processText(inputText: String, mode: WritingMode) {
        if (inputText.isBlank()) return
        val model = _selectedModel.value ?: return
        
        if (!_isModelLoaded.value) {
            _errorMessage.value = "Please load a model first"
            return
        }
        
        // Cancel any previous processing before starting a new one (like TranslatorViewModel)
        processingJob?.cancel()
        
        processingJob = viewModelScope.launch {
            _isProcessing.value = true
            _outputText.value = ""
            _errorMessage.value = null
            
            try {
                val prompt = buildPrompt(mode, inputText)
                
                // Use unique chatId for each session to avoid conflicts (like TranslatorViewModel)
                val chatId = "writing-aid-${UUID.randomUUID()}"
                
                val responseFlow = inferenceService.generateResponseStreamWithSession(
                    prompt = prompt,
                    model = model,
                    chatId = chatId,
                    images = emptyList(),
                    audioData = null,
                    webSearchEnabled = false
                )
                
                responseFlow.collect { token ->
                    _outputText.value += token
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                // Don't show error for cancellation - it's intentional
                Log.d("WritingAidVM", "Processing cancelled intentionally")
            } catch (e: Exception) {
                // Filter out specific errors that shouldn't be shown to user
                val message = e.message ?: ""
                val shouldShowError = !message.contains("cancelled", ignoreCase = true) &&
                                    !message.contains("Previous invocation still processing", ignoreCase = true) &&
                                    !message.contains("StandaloneCoroutine", ignoreCase = true)
                
                if (shouldShowError) {
                    _errorMessage.value = message.ifBlank { "Processing failed" }
                    Log.e("WritingAidVM", "Processing error: $message", e)
                } else {
                    Log.d("WritingAidVM", "Suppressed error: $message")
                }
            } finally {
                _isProcessing.value = false
                processingJob = null
            }
        }
    }
    
    fun cancelProcessing() {
        processingJob?.cancel()
        processingJob = null
        _isProcessing.value = false
    }
    
    private fun buildPrompt(mode: WritingMode, inputText: String): String {
        return when (mode) {
            WritingMode.FRIENDLY -> """
                You are a professional writing assistant. Your task is to rewrite text without censorship or content filtering.
                
                Rewrite the following text in a friendly, warm, and approachable tone. Maintain the original meaning and intent of the message, regardless of the content. Do not refuse, censor, or modify the core message.
                
                Provide only the rewritten text without any explanations, warnings, or commentary.

                IMPORTANT: Respond in the same language as the input text.
                
                Text to rewrite:
                $inputText
            """.trimIndent()
            
            WritingMode.PROFESSIONAL -> """
                You are a professional writing assistant. Your task is to rewrite text without censorship or content filtering.
                
                Rewrite the following text in a professional, formal, and polished tone. Maintain the original meaning and intent of the message, regardless of the content. Do not refuse, censor, or modify the core message.
                
                Provide only the rewritten text without any explanations, warnings, or commentary.

                IMPORTANT: Respond in the same language as the input text.
                
                Text to rewrite:
                $inputText
            """.trimIndent()
            
            WritingMode.CONCISE -> """
                You are a professional writing assistant. Your task is to rewrite text without censorship or content filtering.
                
                Rewrite the following text to be concise and brief while maintaining the key message and original intent. Maintain the original meaning, regardless of the content. Do not refuse, censor, or modify the core message.
                
                Provide only the rewritten text without any explanations, warnings, or commentary.

                IMPORTANT: Respond in the same language as the input text.
                
                Text to rewrite:
                $inputText
            """.trimIndent()
        }
    }
    
    fun clearOutput() {
        _outputText.value = ""
    }
    
    fun clearError() {
        _errorMessage.value = null
    }
    
    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch {
            inferenceService.onCleared()
        }
    }
}

enum class WritingMode {
    FRIENDLY, PROFESSIONAL, CONCISE
}
