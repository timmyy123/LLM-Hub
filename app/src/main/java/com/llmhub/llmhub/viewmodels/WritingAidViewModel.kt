package com.llmhub.llmhub.viewmodels

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class WritingAidViewModel(application: Application) : AndroidViewModel(application) {
    
    private val inferenceService = MediaPipeInferenceService(application)
    
    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()
    
    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()
    
    private val _selectedBackend = MutableStateFlow<LlmInference.Backend?>(null)
    val selectedBackend: StateFlow<LlmInference.Backend?> = _selectedBackend.asStateFlow()
    
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
    }
    
    private fun loadAvailableModels() {
        viewModelScope.launch {
            ModelData.models.filter { it.category != "embedding" }
                .also { models ->
                    _availableModels.value = models.filter { it.isDownloaded }
                    if (_selectedModel.value == null) {
                        _availableModels.value.firstOrNull()?.let { selectModel(it) }
                    }
                }
        }
    }
    
    fun selectModel(model: LLMModel) {
        _selectedModel.value = model
        // Auto-select backend based on GPU support
        if (_selectedBackend.value == null) {
            _selectedBackend.value = if (model.supportsGpu) {
                LlmInference.Backend.GPU
            } else {
                LlmInference.Backend.CPU
            }
        }
    }
    
    fun selectBackend(backend: LlmInference.Backend) {
        _selectedBackend.value = backend
    }
    
    fun loadModel() {
        val model = _selectedModel.value ?: return
        val backend = _selectedBackend.value ?: return
        
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
                    disableAudio = true
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
    
    fun processText(inputText: String, mode: WritingMode) {
        if (inputText.isBlank()) return
        val model = _selectedModel.value ?: return
        
        if (!_isModelLoaded.value) {
            _errorMessage.value = "Please load a model first"
            return
        }
        
        viewModelScope.launch {
            _isProcessing.value = true
            _outputText.value = ""
            _errorMessage.value = null
            
            try {
                val prompt = buildPrompt(mode, inputText)
                
                // Use streaming for better UX
                inferenceService.generateResponseStream(prompt, model).collect { token ->
                    _outputText.value += token
                }
            } catch (e: Exception) {
                _errorMessage.value = e.message ?: "Processing failed"
            } finally {
                _isProcessing.value = false
            }
        }
    }
    
    private fun buildPrompt(mode: WritingMode, inputText: String): String {
        return when (mode) {
            WritingMode.GRAMMAR -> """
                Fix the grammar and spelling in the following text. Provide only the corrected version:
                
                $inputText
            """.trimIndent()
            
            WritingMode.PARAPHRASE -> """
                Rewrite the following text while maintaining the same meaning:
                
                $inputText
            """.trimIndent()
            
            WritingMode.TONE -> """
                Adjust the tone of the following text to be more professional and polished:
                
                $inputText
            """.trimIndent()
            
            WritingMode.EMAIL -> """
                Convert the following text into a well-formatted professional email:
                
                $inputText
            """.trimIndent()
            
            WritingMode.SMS -> """
                Convert the following text into a concise SMS message:
                
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
    GRAMMAR, PARAPHRASE, TONE, EMAIL, SMS
}
