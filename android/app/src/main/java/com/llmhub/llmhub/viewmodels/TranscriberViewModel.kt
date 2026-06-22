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
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.Job
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TranscriberViewModel(application: Application) : AndroidViewModel(application) {
    private val inferenceService = (application as com.llmhub.llmhub.LlmHubApplication).inferenceService
    private val prefs = application.getSharedPreferences("transcriber_prefs", android.content.Context.MODE_PRIVATE)
    private val themePrefs = com.llmhub.llmhub.data.ThemePreferences(application)
    private var asrWrapper: com.nexa.sdk.AsrWrapper? = null
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
    
    fun selectBackend(backend: LlmInference.Backend, deviceId: String? = null) {
        // Unload current model before switching backend
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedBackend.value = backend
        _selectedNpuDeviceId.value = deviceId
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
                    val modelDir = getModelDirectory(model)
                    val modelFile = findModelFile(modelDir, model)
                    if (!modelFile.exists()) {
                        throw java.io.FileNotFoundException("Model file not found at ${modelFile.absolutePath}")
                    }
                    val config = com.nexa.sdk.bean.ModelConfig()
                    val deviceId = "cpu"
                    val createInput = com.nexa.sdk.bean.AsrCreateInput(
                        model_name = model.name,
                        model_path = modelFile.absolutePath,
                        config = config,
                        plugin_id = "whisper_cpp",
                        device_id = deviceId
                    )
                    val buildResult = com.nexa.sdk.AsrWrapper.builder()
                        .asrCreateInput(createInput)
                        .build()
                    
                    if (buildResult.isSuccess) {
                        asrWrapper = buildResult.getOrNull()
                        _isModelLoaded.value = true
                    } else {
                        throw buildResult.exceptionOrNull() ?: Exception("Failed to build AsrWrapper")
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
                    if (asrWrapper != null) {
                        try { asrWrapper?.close() } catch (_: Exception) {}
                        try { asrWrapper?.destroy() } catch (_: Exception) {}
                        asrWrapper = null
                    }
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
                    // Whisper expects PCM16 WAV; convert from float32 WAV if needed
                    val pcm16Wav = float32WavToPcm16Wav(audioBytes)
                    android.util.Log.d("TranscriberViewModel", "ASR batch: input=${audioBytes.size}B float32WAV → pcm16WAV=${pcm16Wav.size}B")
                    val tempWavFile = withContext(Dispatchers.IO) {
                        val file = java.io.File.createTempFile("asr_input_", ".wav", getApplication<Application>().cacheDir)
                        file.writeBytes(pcm16Wav)
                        file
                    }
                    try {
                        val isEnglishModel = model.name.contains("english", ignoreCase = true) || model.url.contains(".en.", ignoreCase = true)
                        val lang = if (isEnglishModel) "english" else "auto"
                        android.util.Log.d("TranscriberViewModel", "ASR batch: lang=$lang, wavFile=${tempWavFile.absolutePath} size=${tempWavFile.length()}B")
                        
                        val transcribeInput = com.nexa.sdk.bean.AsrTranscribeInput(
                            audioPath = tempWavFile.absolutePath,
                            language = lang,
                            config = com.nexa.sdk.bean.AsrConfig()
                        )
                        val result = withContext(Dispatchers.IO) {
                            asrMutex.withLock {
                                asrWrapper?.transcribe(transcribeInput)
                            }
                        }
                        android.util.Log.d("TranscriberViewModel", "ASR batch result: success=${result?.isSuccess}, transcript='${result?.getOrNull()?.result?.transcript}'")
                        if (result != null && result.isSuccess) {
                            val transcript = result.getOrNull()?.result?.transcript ?: ""
                            _transcriptionText.value = transcript
                        } else {
                            throw result?.exceptionOrNull() ?: Exception("ASR Transcription failed")
                        }
                    } finally {
                        tempWavFile.delete()
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

    fun transcribeLive(audioBytes: ByteArray) {
        val model = _selectedModel.value ?: return
        val asr = asrWrapper ?: return
        if (_isTranscribing.value || asrMutex.isLocked) return // avoid concurrent transcription runs
        
        viewModelScope.launch {
            _isTranscribing.value = true
            try {
                // Whisper expects PCM16 WAV; convert from float32 WAV
                val pcm16Wav = float32WavToPcm16Wav(audioBytes)
                android.util.Log.d("TranscriberViewModel", "ASR live: input=${audioBytes.size}B → pcm16WAV=${pcm16Wav.size}B")
                val tempWavFile = withContext(Dispatchers.IO) {
                    val file = java.io.File.createTempFile("asr_live_", ".wav", getApplication<Application>().cacheDir)
                    file.writeBytes(pcm16Wav)
                    file
                }
                
                try {
                    val isEnglishModel = model.name.contains("english", ignoreCase = true) || model.url.contains(".en.", ignoreCase = true)
                    val lang = if (isEnglishModel) "english" else "auto"
                    android.util.Log.d("TranscriberViewModel", "ASR live: lang=$lang, wavFile=${tempWavFile.absolutePath} size=${tempWavFile.length()}B")
                    
                    val transcribeInput = com.nexa.sdk.bean.AsrTranscribeInput(
                        audioPath = tempWavFile.absolutePath,
                        language = lang,
                        config = com.nexa.sdk.bean.AsrConfig()
                    )
                    val result = withContext(Dispatchers.IO) {
                        asrMutex.withLock {
                            asr.transcribe(transcribeInput)
                        }
                    }
                    android.util.Log.d("TranscriberViewModel", "ASR live result: success=${result?.isSuccess}, transcript='${result?.getOrNull()?.result?.transcript}'")
                    if (result != null && result.isSuccess) {
                        val transcript = result.getOrNull()?.result?.transcript ?: ""
                        if (transcript.isNotEmpty()) {
                            _transcriptionText.value = transcript
                        }
                    }
                } finally {
                    tempWavFile.delete()
                }
            } catch (e: Exception) {
                android.util.Log.e("TranscriberViewModel", "Live transcription error", e)
            } finally {
                _isTranscribing.value = asrMutex.isLocked
            }
        }
    }

    /**
     * Convert a float32 WAV (AudioFormat=3, 32-bit IEEE float) to PCM16 WAV (AudioFormat=1, 16-bit signed int).
     * Whisper/AsrWrapper requires standard PCM16 WAV input.
     * If the input is already PCM16 or not a recognized WAV, it is returned unchanged.
     */
    private fun float32WavToPcm16Wav(float32Wav: ByteArray): ByteArray {
        if (float32Wav.size < 44) return float32Wav
        val bb = ByteBuffer.wrap(float32Wav).order(ByteOrder.LITTLE_ENDIAN)
        // Check RIFF/WAVE magic
        val riff = String(float32Wav, 0, 4)
        val wave = String(float32Wav, 8, 4)
        if (riff != "RIFF" || wave != "WAVE") return float32Wav // not a WAV
        val audioFormat = bb.getShort(20).toInt() and 0xFFFF
        if (audioFormat != 3) return float32Wav // already PCM or other; pass through
        val channels = bb.getShort(22).toInt() and 0xFFFF
        val sampleRate = bb.getInt(24)
        val dataSize = bb.getInt(40)
        val numSamples = dataSize / 4
        // Convert float32 samples → int16 samples
        val pcm16Bytes = ByteArray(numSamples * 2)
        val pcm16Buf = ByteBuffer.wrap(pcm16Bytes).order(ByteOrder.LITTLE_ENDIAN)
        bb.position(44)
        for (i in 0 until numSamples) {
            val f = bb.getFloat().coerceIn(-1f, 1f)
            pcm16Buf.putShort((f * 32767f).toInt().toShort())
        }
        // Build PCM16 WAV header
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(36 + pcm16Bytes.size)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16)
        header.putShort(1)                              // PCM
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(sampleRate * channels * 2)        // ByteRate
        header.putShort((channels * 2).toShort())       // BlockAlign
        header.putShort(16)                             // BitsPerSample
        header.put("data".toByteArray())
        header.putInt(pcm16Bytes.size)
        return header.array() + pcm16Bytes
    }

    private fun getModelDirectory(model: LLMModel): java.io.File {
        val context = getApplication<Application>()
        val modelsDir = java.io.File(context.filesDir, "models")
        val modelDirName = model.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
        val modelDir = java.io.File(modelsDir, modelDirName)
        return if (modelDir.exists()) modelDir else modelsDir
    }
    
    private fun findModelFile(modelDir: java.io.File, model: LLMModel): java.io.File {
        val localName = model.url.substringAfterLast("/").substringBefore("?")
        var modelFile = java.io.File(modelDir, localName)
        if (modelFile.exists()) return modelFile
        
        val context = getApplication<Application>()
        val modelsDir = java.io.File(context.filesDir, "models")
        modelFile = java.io.File(modelsDir, localName)
        if (modelFile.exists()) return modelFile
        
        val files = modelDir.listFiles { _, name -> 
            name.endsWith(".bin") || name.endsWith(".gguf")
        }
        if (files?.isNotEmpty() == true) return files.first()
        
        return java.io.File(modelDir, localName)
    }
}

data class TranscriptionSession(
    val id: String = java.util.UUID.randomUUID().toString(),
    val text: String,
    val timestamp: Long = System.currentTimeMillis()
)
