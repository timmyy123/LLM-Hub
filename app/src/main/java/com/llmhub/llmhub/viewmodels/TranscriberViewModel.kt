package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.inference.MediaPipeInferenceService
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import java.util.UUID
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class TranscriberViewModel(application: Application) : AndroidViewModel(application) {
    private val inferenceService = MediaPipeInferenceService(application)
    private val prefs = application.getSharedPreferences("transcriber_prefs", android.content.Context.MODE_PRIVATE)
    
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
    
    private val _isTranscribing = MutableStateFlow(false)
    val isTranscribing: StateFlow<Boolean> = _isTranscribing.asStateFlow()
    private var transcribeJob: Job? = null
    
    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()
    
    private val _loadError = MutableStateFlow<String?>(null)
    val loadError: StateFlow<String?> = _loadError.asStateFlow()
    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()
    
    // Audio input/output
    private val _audioUri = MutableStateFlow<Uri?>(null)
    val audioUri: StateFlow<Uri?> = _audioUri.asStateFlow()
    private val _audioData = MutableStateFlow<ByteArray?>(null)
    val audioData: StateFlow<ByteArray?> = _audioData.asStateFlow()
    
    private val _transcriptionText = MutableStateFlow("")
    val transcriptionText: StateFlow<String> = _transcriptionText.asStateFlow()
    
    init {
        loadAvailableModels()
        loadSavedSettings()
    }
    
    private fun loadAvailableModels() {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val allModels = ModelAvailabilityProvider.loadAvailableModels(context)
            val audioModels = allModels.filter { it.supportsAudio }
            _availableModels.value = audioModels

            // restore selected model by name
            val savedModelName = prefs.getString("selected_model_name", null)
            if (savedModelName != null) {
                val saved = audioModels.find { it.name == savedModelName }
                if (saved != null) _selectedModel.value = saved
            }
            if (audioModels.isNotEmpty() && _selectedModel.value == null) {
                _selectedModel.value = audioModels.first()
            }
        }
    }
    private fun loadSavedSettings() {
        val savedBackendName = prefs.getString("selected_backend", LlmInference.Backend.GPU.name)
        _selectedBackend.value = try {
            LlmInference.Backend.valueOf(savedBackendName ?: LlmInference.Backend.GPU.name)
        } catch (_: IllegalArgumentException) {
            LlmInference.Backend.GPU
        }
    }
    private fun saveSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            putString("selected_backend", _selectedBackend.value.name)
            apply()
        }
    }
    
    fun selectModel(model: LLMModel) {
        // Unload current model before switching
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedModel.value = model
        saveSettings()
    }
    
    fun selectBackend(backend: LlmInference.Backend) {
        // Unload current model before switching backend
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedBackend.value = backend
        saveSettings()
    }
    
    fun setAudioUri(uri: Uri?) {
        _audioUri.value = uri
        _audioData.value = null
    }
    
    fun setRecording(recording: Boolean) {
        _isRecording.value = recording
    }
    fun setAudioData(data: ByteArray?) {
        _audioData.value = data
        if (data != null) _audioUri.value = null
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
                // Load model with audio modality enabled
                inferenceService.loadModel(
                    model = model,
                    preferredBackend = _selectedBackend.value,
                    disableVision = true,  // Only audio, no vision
                    disableAudio = false   // Enable audio
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
                transcribeJob?.cancel()
                inferenceService.unloadModel()
                _isModelLoaded.value = false
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Failed to unload model"
            }
        }
    }
    
    fun cancelTranscription() {
        transcribeJob?.cancel()
        transcribeJob = null
        _isTranscribing.value = false
    }
    
    fun transcribe(audioUri: Uri? = null) {
        val model = _selectedModel.value ?: return
        transcribeJob?.cancel()
        
        transcribeJob = viewModelScope.launch {
            _isTranscribing.value = true
            _transcriptionText.value = ""
            
            try {
                val audioBytes = _audioData.value ?: (audioUri?.let { readAudioBytes(it) })
                    ?: throw IllegalStateException("Unable to read audio input")
                
                val prompt = """You are a professional transcriber.
Transcribe the audio content exactly as spoken.
Include proper punctuation and formatting.
Do not add any commentary or explanations.""".trimIndent()
                
                val chatId = "transcriber-${UUID.randomUUID()}"

                val responseFlow = inferenceService.generateResponseStreamWithSession(
                    prompt = prompt,
                    model = model,
                    chatId = chatId,
                    images = emptyList(),
                    audioData = audioBytes,
                    webSearchEnabled = false
                )

                responseFlow.collect { token ->
                    if (token.isNotEmpty()) {
                        _transcriptionText.value += token
                    }
                }
            } catch (_: CancellationException) {
                // ignore cancel
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Transcription failed"
            } finally {
                _isTranscribing.value = false
            }
        }
    }
    
    fun clearTranscription() {
        _transcriptionText.value = ""
        _audioUri.value = null
    }

    private suspend fun readAudioBytes(uri: Uri): ByteArray? {
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
