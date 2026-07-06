package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.UnifiedInferenceService
import com.llmhub.llmhub.utils.AudioConversionUtils
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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import com.llmhub.llmhub.data.DeviceInfo
import com.llmhub.llmhub.inference.WhisperBackend
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TranscriberViewModel(application: Application) : AndroidViewModel(application) {
    private val inferenceService = (application as com.llmhub.llmhub.LlmHubApplication).inferenceService
    private val prefs = application.getSharedPreferences("transcriber_prefs", android.content.Context.MODE_PRIVATE)
    private val whisperKitService = com.llmhub.llmhub.inference.WhisperKitService(application)
    private val asrMutex = Mutex()
    
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

    private val _selectedAsrBackend = MutableStateFlow(
        if (DeviceInfo.isQualcommNpuSupported()) WhisperBackend.NPU else WhisperBackend.CPU
    )
    val selectedAsrBackend: StateFlow<WhisperBackend> = _selectedAsrBackend.asStateFlow()
    
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
    
    private val _transcriptionHistory = MutableStateFlow<List<TranscriptionSession>>(emptyList())
    val transcriptionHistory: StateFlow<List<TranscriptionSession>> = _transcriptionHistory.asStateFlow()
    
    init {
        loadAvailableModels()
        loadSavedSettings()
    }
    
    private fun loadAvailableModels() {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val allModels = ModelAvailabilityProvider.loadAvailableModels(context, includeAsr = true)
            val audioModels = allModels.filter { it.supportsAudio && it.category != "tts" }
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
        val defaultAsrBackend = if (DeviceInfo.isQualcommNpuSupported()) WhisperBackend.NPU else WhisperBackend.CPU
        val savedAsrBackendName = prefs.getString("selected_asr_backend", defaultAsrBackend.name)
        _selectedAsrBackend.value = try {
            WhisperBackend.valueOf(savedAsrBackendName ?: defaultAsrBackend.name)
        } catch (_: IllegalArgumentException) {
            defaultAsrBackend
        }
    }
    private fun saveSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            putString("selected_backend", _selectedBackend.value.name)
            putString("selected_asr_backend", _selectedAsrBackend.value.name)
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
    
    fun selectBackend(backend: LlmInference.Backend, deviceId: String? = null) {
        if (_isModelLoaded.value) unloadModel()
        _selectedBackend.value = backend
        _selectedNpuDeviceId.value = deviceId
        saveSettings()
    }

    fun selectAsrBackend(backend: WhisperBackend) {
        if (_isModelLoaded.value) unloadModel()
        _selectedAsrBackend.value = backend
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
                if (model.category == "asr") {
                    val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                    val modelDir = java.io.File(getApplication<Application>().filesDir, "models/$modelDirName")
                    // For English-only models force "en"; for multilingual set empty = auto-detect.
                    val isEnglishOnly = model.name.contains("english", ignoreCase = true)
                    whisperKitService.transcribeLanguage = if (isEnglishOnly) "en" else ""
                    val loaded = whisperKitService.loadModel(modelDir.absolutePath, _selectedAsrBackend.value)
                    if (loaded) {
                        _isModelLoaded.value = true
                    } else {
                        throw Exception("Failed to load WhisperKit model from: ${modelDir.absolutePath}")
                    }
                } else {
                    // Load model with audio modality enabled
                    (inferenceService as? UnifiedInferenceService)?.setAgentToolsEnabled(false)
                    inferenceService.setGenerationParameters(null, null, null, null, enableThinking = if (model.name.contains("Gemma-4", ignoreCase = true)) false else null)
                    inferenceService.loadModel(
                        model = model,
                        preferredBackend = _selectedBackend.value,
                        disableVision = true,  // Only audio, no vision
                        disableAudio = false,  // Enable audio
                        deviceId = _selectedNpuDeviceId.value
                    )
                    _isModelLoaded.value = true
                }
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
                asrMutex.withLock {
                    whisperKitService.release()
                }
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
                val uriToUse = audioUri ?: _audioUri.value
                val audioBytes = _audioData.value ?: (uriToUse?.let { readAudioBytes(it) })
                    ?: throw IllegalStateException("Unable to read audio input")
                
                if (model.category == "asr") {
                    val pcm16Wav = AudioConversionUtils.float32WavToPcm16Wav(audioBytes)
                    android.util.Log.d("TranscriberViewModel", "ASR batch: input=${audioBytes.size}B float32WAV → pcm16WAV=${pcm16Wav.size}B")
                    // Strip WAV header (44 bytes) to get raw PCM16 bytes for WhisperKit
                    val pcm16Raw = if (pcm16Wav.size > 44) pcm16Wav.copyOfRange(44, pcm16Wav.size) else pcm16Wav
                    android.util.Log.d("TranscriberViewModel", "ASR batch: lang=${whisperKitService.transcribeLanguage.ifEmpty { "auto" }}, pcmBytes=${pcm16Raw.size}B")

                    val transcript = withContext(Dispatchers.IO) {
                        asrMutex.withLock {
                            whisperKitService.transcribe(pcm16Raw, whisperKitService.transcribeLanguage)
                        }
                    }
                    android.util.Log.d("TranscriberViewModel", "ASR batch result: transcript='$transcript'")
                    if (!transcript.isNullOrEmpty()) {
                        _transcriptionText.value = transcript
                    } else {
                        throw Exception("ASR Transcription returned empty result")
                    }
                } else {
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
                }
                val finalResult = _transcriptionText.value.trim()
                if (finalResult.isNotEmpty()) {
                    _transcriptionHistory.value = _transcriptionHistory.value + TranscriptionSession(text = finalResult)
                }
                _transcriptionText.value = ""
            } catch (_: CancellationException) {
                // ignore cancel
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Transcription failed"
            } finally {
                _isTranscribing.value = asrMutex.isLocked
            }
        }
    }
    
    fun clearTranscription() {
        _transcriptionText.value = ""
        _audioUri.value = null
        _audioData.value = null
        _transcriptionHistory.value = emptyList()
    }

    private suspend fun readAudioBytes(uri: Uri): ByteArray? {
        val app = getApplication<Application>()
        return AudioConversionUtils.convertUriToFloat32Wav(app, uri)
    }

}

data class TranscriptionSession(
    val id: String = java.util.UUID.randomUUID().toString(),
    val text: String,
    val timestamp: Long = System.currentTimeMillis()
)
