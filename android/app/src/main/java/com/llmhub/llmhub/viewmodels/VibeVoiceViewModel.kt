package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.data.hasNativeVoiceSupport
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.UnifiedInferenceService
import com.llmhub.llmhub.utils.AudioConversionUtils
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class VibeVoiceViewModel(application: Application) : AndroidViewModel(application) {
    companion object {
        private const val VIBEVOICE_SYSTEM_PROMPT = """
            You are VibeVoice, a natural real-time conversation assistant.
            Keep responses conversational and useful.
            Match response length to the user's request.
            For simple requests, keep it brief.
            For "why/how/explain/compare/steps" requests, give a fuller multi-sentence answer.
            Respond to the meaning of what the user said in audio.
            Do not transcribe or repeat the user's words verbatim.
            Do not output role labels like 'assistant:' or 'user:'.
            If audio is unclear, ask one brief clarification question.
        """

        private const val VOICE_TURN_PROMPT = """
            The user just spoke in audio.
            Reply naturally and match the detail level the user asked for.
            Do not transcribe or quote the user's exact words.
        """

        private const val VOICE_FALLBACK_REPLY = "I heard you, but I missed part of that. Could you repeat it briefly?"
    }

    private val inferenceService = (application as com.llmhub.llmhub.LlmHubApplication).inferenceService
    private val prefs = application.getSharedPreferences("vibevoice_prefs", android.content.Context.MODE_PRIVATE)

    data class VoiceTurn(
        val id: String,
        val userText: String,
        val assistantText: String
    )

    private val _availableLlmModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableLlmModels: StateFlow<List<LLMModel>> = _availableLlmModels.asStateFlow()

    // Compatibility field so VibeVoiceScreen doesn't break
    val availableModels: StateFlow<List<LLMModel>> = _availableLlmModels.asStateFlow()

    private val _availableAsrModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableAsrModels: StateFlow<List<LLMModel>> = _availableAsrModels.asStateFlow()

    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()

    private val _selectedVoiceModel = MutableStateFlow<LLMModel?>(null)
    val selectedVoiceModel: StateFlow<LLMModel?> = _selectedVoiceModel.asStateFlow()

    private val _hasRequiredModels = MutableStateFlow(false)
    val hasRequiredModels: StateFlow<Boolean> = _hasRequiredModels.asStateFlow()

    private val _selectedBackend = MutableStateFlow(LlmInference.Backend.GPU)
    val selectedBackend: StateFlow<LlmInference.Backend> = _selectedBackend.asStateFlow()

    private val _selectedNpuDeviceId = MutableStateFlow<String?>(null)
    val selectedNpuDeviceId: StateFlow<String?> = _selectedNpuDeviceId.asStateFlow()

    private val _transcription = MutableStateFlow("")
    val transcription: StateFlow<String> = _transcription.asStateFlow()

    private val _isLoadingModel = MutableStateFlow(false)
    val isLoadingModel: StateFlow<Boolean> = _isLoadingModel.asStateFlow()

    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private val _isResponding = MutableStateFlow(false)
    val isResponding: StateFlow<Boolean> = _isResponding.asStateFlow()

    private val _loadError = MutableStateFlow<String?>(null)
    val loadError: StateFlow<String?> = _loadError.asStateFlow()

    private val _audioUri = MutableStateFlow<Uri?>(null)
    val audioUri: StateFlow<Uri?> = _audioUri.asStateFlow()

    private val _audioData = MutableStateFlow<ByteArray?>(null)
    val audioData: StateFlow<ByteArray?> = _audioData.asStateFlow()

    private val _liveResponseText = MutableStateFlow("")
    val liveResponseText: StateFlow<String> = _liveResponseText.asStateFlow()

    private val _conversationTurns = MutableStateFlow<List<VoiceTurn>>(emptyList())
    val conversationTurns: StateFlow<List<VoiceTurn>> = _conversationTurns.asStateFlow()

    private var respondJob: Job? = null
    private var sessionChatId: String = "vibevoice-${UUID.randomUUID()}"
    private var isSessionPrimed: Boolean = false

    private var asrWrapper: com.nexa.sdk.AsrWrapper? = null
    private val asrMutex = Mutex()

    init {
        loadAvailableModels()
        loadSavedSettings()
    }

    private fun loadAvailableModels() {
        viewModelScope.launch {
            val context = getApplication<Application>()
            
            // Standard LLM models (not ASR, not embedding)
            val allModels = ModelAvailabilityProvider.loadAvailableModels(context, includeAsr = false)
            val llmModels = allModels.filter { it.category == "text" || it.category == "multimodal" }
            _availableLlmModels.value = llmModels

            // ASR models
            val allWithAsr = ModelAvailabilityProvider.loadAvailableModels(context, includeAsr = true)
            val asrModels = allWithAsr.filter { it.category == "asr" }
            _availableAsrModels.value = asrModels

            // Check if required models are present
            val hasGemmaVoice = llmModels.any { it.hasNativeVoiceSupport() }
            val hasLlmAndAsr = llmModels.isNotEmpty() && asrModels.isNotEmpty()
            _hasRequiredModels.value = hasGemmaVoice || hasLlmAndAsr

            val savedModelName = prefs.getString("selected_model_name", null)
            if (savedModelName != null) {
                val saved = llmModels.find { it.name == savedModelName }
                if (saved != null) _selectedModel.value = saved
            }
            if (llmModels.isNotEmpty() && _selectedModel.value == null) {
                _selectedModel.value = llmModels.first()
            }

            val savedVoiceName = prefs.getString("selected_voice_model_name", null)
            if (savedVoiceName != null) {
                if (savedVoiceName == "gemma") {
                    _selectedVoiceModel.value = null
                } else {
                    val savedVoice = asrModels.find { it.name == savedVoiceName }
                    _selectedVoiceModel.value = savedVoice
                }
            } else {
                val currentLlm = _selectedModel.value
                if (currentLlm != null && currentLlm.hasNativeVoiceSupport()) {
                    _selectedVoiceModel.value = null
                } else {
                    _selectedVoiceModel.value = asrModels.firstOrNull()
                }
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
            putString("selected_voice_model_name", _selectedVoiceModel.value?.name ?: "gemma")
            putString("selected_backend", _selectedBackend.value.name)
            apply()
        }
    }

    fun selectModel(model: LLMModel) {
        if (_isModelLoaded.value) unloadModel()
        _selectedModel.value = model
        
        if (!model.hasNativeVoiceSupport() && _selectedVoiceModel.value == null) {
            _selectedVoiceModel.value = _availableAsrModels.value.firstOrNull()
        }
        saveSettings()
        clearConversation()
    }

    fun selectVoiceModel(voiceModel: LLMModel?) {
        if (_isModelLoaded.value) unloadModel()
        _selectedVoiceModel.value = voiceModel
        saveSettings()
        clearConversation()
    }

    fun selectBackend(backend: LlmInference.Backend, deviceId: String? = null) {
        if (_isModelLoaded.value) unloadModel()
        _selectedBackend.value = backend
        _selectedNpuDeviceId.value = deviceId
        saveSettings()
        clearConversation()
    }

    fun setAudioUri(uri: Uri?) {
        _audioUri.value = uri
        _audioData.value = null
    }

    fun setAudioData(data: ByteArray?) {
        _audioData.value = data
        if (data != null) _audioUri.value = null
    }

    fun setRecording(recording: Boolean) {
        _isRecording.value = recording
    }

    fun clearError() {
        _loadError.value = null
    }

    fun loadModel() {
        val model = _selectedModel.value ?: return
        if (_isLoadingModel.value || _isModelLoaded.value) return

        viewModelScope.launch {
            _isLoadingModel.value = true
            _loadError.value = null
            try {
                val voiceModel = _selectedVoiceModel.value
                val isUsingAsr = voiceModel != null
                val disableAudio = isUsingAsr
                
                (inferenceService as? UnifiedInferenceService)?.setAgentToolsEnabled(false)
                inferenceService.setGenerationParameters(null, null, null, null, enableThinking = if (model.name.contains("Gemma-4", ignoreCase = true)) false else null)
                inferenceService.loadModel(
                    model = model,
                    preferredBackend = _selectedBackend.value,
                    disableVision = true,
                    disableAudio = disableAudio,
                    deviceId = _selectedNpuDeviceId.value
                )
                
                if (isUsingAsr && voiceModel != null) {
                    val modelDir = getModelDirectory(voiceModel)
                    val modelFile = findModelFile(modelDir, voiceModel)
                    if (!modelFile.exists()) {
                        throw java.io.FileNotFoundException("Model file not found at ${modelFile.absolutePath}")
                    }
                    val config = com.nexa.sdk.bean.ModelConfig()
                    val createInput = com.nexa.sdk.bean.AsrCreateInput(
                        model_name = voiceModel.name,
                        model_path = modelFile.absolutePath,
                        config = config,
                        plugin_id = "whisper_cpp",
                        device_id = "cpu"
                    )
                    val buildResult = com.nexa.sdk.AsrWrapper.builder()
                        .asrCreateInput(createInput)
                        .build()
                    
                    if (buildResult.isSuccess) {
                        asrWrapper = buildResult.getOrNull()
                    } else {
                        throw buildResult.exceptionOrNull() ?: Exception("Failed to build AsrWrapper")
                    }
                }
                
                _isModelLoaded.value = true
            } catch (e: Exception) {
                _loadError.value = e.message ?: "Failed to load model"
                _isModelLoaded.value = false
                // Clean up ASR if initialized
                try {
                    asrMutex.withLock {
                        if (asrWrapper != null) {
                            try { asrWrapper?.close() } catch (_: Exception) {}
                            try { asrWrapper?.destroy() } catch (_: Exception) {}
                            asrWrapper = null
                        }
                    }
                } catch (_: Exception) {}
            } finally {
                _isLoadingModel.value = false
            }
        }
    }

    fun unloadModel() {
        viewModelScope.launch {
            try {
                respondJob?.cancel()
                _transcription.value = ""
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

    fun cancelResponse() {
        respondJob?.cancel()
        respondJob = null
        _isResponding.value = false
        _liveResponseText.value = ""
        _transcription.value = ""

        // Ensure the underlying MediaPipe generation is also cancelled so it cannot resume later.
        viewModelScope.launch {
            try {
                inferenceService.resetChatSession(sessionChatId)
                isSessionPrimed = false
            } catch (_: Exception) {
                // Best-effort cancel.
            }
        }
    }

    fun sendVoiceTurn(audioUri: Uri? = null) {
        val model = _selectedModel.value ?: return
        respondJob?.cancel()

        respondJob = viewModelScope.launch {
            _isResponding.value = true
            _liveResponseText.value = ""
            _transcription.value = ""

            try {
                val uriToUse = audioUri ?: _audioUri.value
                val audioBytes = _audioData.value ?: (uriToUse?.let { readAudioBytes(it) })
                    ?: throw IllegalStateException("Unable to read audio input")

                val turnId = UUID.randomUUID().toString()
                _conversationTurns.value = _conversationTurns.value + VoiceTurn(
                    id = turnId,
                    userText = "",
                    assistantText = ""
                )

                val asr = asrWrapper
                if (asr != null) {
                    // Translate voice input to text using ASR model first
                    val pcm16Wav = AudioConversionUtils.float32WavToPcm16Wav(audioBytes)
                    android.util.Log.d("VibeVoiceViewModel", "ASR transcribe: input=${audioBytes.size}B → pcm16WAV=${pcm16Wav.size}B")
                    val tempWavFile = withContext(Dispatchers.IO) {
                        val file = java.io.File.createTempFile("asr_vibevoice_", ".wav", getApplication<Application>().cacheDir)
                        file.writeBytes(pcm16Wav)
                        file
                    }
                    val transcription = try {
                        val asrModel = _selectedVoiceModel.value ?: throw IllegalStateException("ASR model not selected")
                        val isEnglishModel = asrModel.name.contains("english", ignoreCase = true) || asrModel.url.contains(".en.", ignoreCase = true)
                        val lang = if (isEnglishModel) "english" else "auto"
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
                        if (result != null && result.isSuccess) {
                            result.getOrNull()?.result?.transcript ?: ""
                        } else {
                            throw result?.exceptionOrNull() ?: Exception("ASR Transcription failed")
                        }
                    } finally {
                        tempWavFile.delete()
                    }

                    _transcription.value = transcription

                    if (transcription.isBlank()) {
                        _liveResponseText.value = VOICE_FALLBACK_REPLY
                        _conversationTurns.value = _conversationTurns.value.map {
                            if (it.id == turnId) it.copy(userText = "", assistantText = VOICE_FALLBACK_REPLY) else it
                        }
                        return@launch
                    }

                    // Update userText for the current turn
                    _conversationTurns.value = _conversationTurns.value.map {
                        if (it.id == turnId) it.copy(userText = transcription) else it
                    }

                    // Build multi-turn prompt from history
                    val sb = StringBuilder("system: $VIBEVOICE_SYSTEM_PROMPT")
                    for (turn in _conversationTurns.value) {
                        val uText = turn.userText
                        if (uText.isNotBlank()) {
                            sb.append("\n\nuser: $uText")
                        }
                        val aText = turn.assistantText
                        if (aText.isNotBlank() && turn.id != turnId) {
                            sb.append("\n\nassistant: $aText")
                        }
                    }
                    val turnPrompt = sb.toString()

                    android.util.Log.d("VibeVoiceViewModel", "ASR Transcription: $transcription -> LLM Prompt: $turnPrompt")

                    val responseFlow = inferenceService.generateResponseStreamWithSession(
                        prompt = turnPrompt,
                        model = model,
                        chatId = sessionChatId,
                        images = emptyList(),
                        audioData = null,
                        webSearchEnabled = false
                    )

                    responseFlow.collect { token ->
                        if (token.isNotEmpty()) {
                            _liveResponseText.value += token
                            _conversationTurns.value = _conversationTurns.value.map {
                                if (it.id == turnId) it.copy(assistantText = _liveResponseText.value) else it
                            }
                        }
                    }
                } else {
                    // Native Gemma voice input
                    _conversationTurns.value = _conversationTurns.value.map {
                        if (it.id == turnId) it.copy(userText = VOICE_TURN_PROMPT) else it
                    }

                    // Build multi-turn prompt from history
                    val sb = StringBuilder("system: $VIBEVOICE_SYSTEM_PROMPT")
                    for (turn in _conversationTurns.value) {
                        val uText = turn.userText
                        if (uText.isNotBlank()) {
                            sb.append("\n\nuser: $uText")
                        }
                        val aText = turn.assistantText
                        if (aText.isNotBlank() && turn.id != turnId) {
                            sb.append("\n\nassistant: $aText")
                        }
                    }
                    val turnPrompt = sb.toString()

                    val responseFlow = inferenceService.generateResponseStreamWithSession(
                        prompt = turnPrompt,
                        model = model,
                        chatId = sessionChatId,
                        images = emptyList(),
                        audioData = audioBytes,
                        webSearchEnabled = false
                    )

                    responseFlow.collect { token ->
                        if (token.isNotEmpty()) {
                            _liveResponseText.value += token
                            _conversationTurns.value = _conversationTurns.value.map {
                                if (it.id == turnId) it.copy(assistantText = _liveResponseText.value) else it
                            }
                        }
                    }
                }

                val cleaned = sanitizeVoiceResponse(_liveResponseText.value)
                val finalReply = if (cleaned.isBlank() || isPromptEcho(cleaned)) {
                    VOICE_FALLBACK_REPLY
                } else {
                    cleaned
                }

                _liveResponseText.value = finalReply
                _conversationTurns.value = _conversationTurns.value.map {
                    if (it.id == turnId) it.copy(assistantText = finalReply) else it
                }
            } catch (_: CancellationException) {
                // Ignore cancellations.
            } catch (e: Exception) {
                // Keep one ongoing session for normal turns, but recover automatically if context is full.
                if (isContextLimitError(e)) {
                    try {
                        inferenceService.resetChatSession(sessionChatId)
                    } catch (_: Exception) {
                        // Best-effort reset.
                    }
                    sessionChatId = "vibevoice-${UUID.randomUUID()}"
                    isSessionPrimed = false
                    _loadError.value = "Session context reached limit. Started a fresh voice session."
                } else {
                    _loadError.value = e.message ?: "Voice chat failed"
                }
            } finally {
                _isResponding.value = false
            }
        }
    }

    fun clearCurrentAudioInput() {
        _audioData.value = null
        _audioUri.value = null
    }

    fun clearConversation() {
        _conversationTurns.value = emptyList()
        _liveResponseText.value = ""
        sessionChatId = "vibevoice-${UUID.randomUUID()}"
        isSessionPrimed = false
    }

    private fun isContextLimitError(error: Throwable): Boolean {
        val msg = (error.message ?: "").lowercase()
        return msg.contains("out_of_range") ||
            msg.contains("context") ||
            msg.contains("token") ||
            msg.contains("exceed") ||
            msg.contains("capacity") ||
            msg.contains("limit")
    }

    private fun sanitizeVoiceResponse(raw: String): String {
        if (raw.isBlank()) return ""
        var cleaned = raw
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\\n", "\n")
            .trim()

        cleaned = cleaned.replaceFirst(Regex("^(assistant|system|user)\\s*:\\s*", RegexOption.IGNORE_CASE), "")
        cleaned = cleaned.replace(Regex("\n{3,}"), "\n\n").trim()

        return cleaned
    }

    private fun isPromptEcho(text: String): Boolean {
        val t = text.lowercase().trim()
        if (t.isBlank()) return true
        if (t == "assistant" || t == "user" || t == "system") return true
        if (t.startsWith("listen to the audio")) return true
        if (t.startsWith("the user just spoke in audio")) return true
        return false
    }

    private suspend fun readAudioBytes(uri: Uri): ByteArray? {
        val app = getApplication<Application>()
        return AudioConversionUtils.convertUriToFloat32Wav(app, uri)
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