package com.llmhub.llmhub.agent

import android.app.Application
import android.content.Context
import android.speech.tts.TextToSpeech
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.R
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.UnifiedInferenceService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.Locale
import java.util.UUID

sealed class AgentMessage(val id: String) {
    enum class Sender { USER, AGENT, SYSTEM }

    data class Text(
        val messageId: String = UUID.randomUUID().toString(),
        val sender: Sender,
        val text: String
    ) : AgentMessage(messageId)

    data class ToolCall(
        val callId: String = UUID.randomUUID().toString(),
        val toolName: String,
        val args: String,
        val status: Status,
        val result: String? = null
    ) : AgentMessage(callId) {
        enum class Status { RUNNING, SUCCESS, FAILED }
    }

    data class MapLocation(
        val locationId: String = UUID.randomUUID().toString(),
        val label: String,
        val latitude: Double,
        val longitude: Double
    ) : AgentMessage(locationId)
}

enum class VoiceMode {
    SYSTEM_STT,
    WHISPER_KIT,
    GEMMA_AUDIO
}

class AgentViewModel(application: Application) : AndroidViewModel(application) {

    private val _messages = MutableStateFlow<List<AgentMessage>>(emptyList())
    val messages: StateFlow<List<AgentMessage>> = _messages.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _voiceMode = MutableStateFlow(VoiceMode.GEMMA_AUDIO)
    val voiceMode: StateFlow<VoiceMode> = _voiceMode.asStateFlow()

    private val _isGemmaAudioEnabled = MutableStateFlow(true)
    val isGemmaAudioEnabled: StateFlow<Boolean> = _isGemmaAudioEnabled.asStateFlow()

    private val _isWebSearchEnabled = MutableStateFlow(false)
    val isWebSearchEnabled: StateFlow<Boolean> = _isWebSearchEnabled.asStateFlow()

    private val _activeModelName = MutableStateFlow<String?>(null)
    val activeModelName: StateFlow<String?> = _activeModelName.asStateFlow()

    val toolSet = AgentToolSet(application.applicationContext)
    private val inferenceService: InferenceService = UnifiedInferenceService(application.applicationContext)

    private var tts: TextToSpeech? = null
    private var isTtsReady = false

    private val _pendingConfirmation = MutableStateFlow<Pair<String, () -> Unit>?>(null)
    val pendingConfirmation: StateFlow<Pair<String, () -> Unit>?> = _pendingConfirmation.asStateFlow()

    init {
        _activeModelName.value = inferenceService.getCurrentlyLoadedModel()?.name
    }

    fun initializeWelcomeMessage(context: Context, hasDownloadedModels: Boolean) {
        if (hasDownloadedModels) {
            _messages.value = listOf(
                AgentMessage.Text(
                    messageId = "welcome",
                    sender = AgentMessage.Sender.AGENT,
                    text = context.getString(R.string.agent_welcome_message)
                )
            )
        } else {
            _messages.value = emptyList()
        }
    }

    fun setVoiceMode(mode: VoiceMode) {
        _voiceMode.value = mode
    }

    fun setGemmaAudioEnabled(enabled: Boolean) {
        _isGemmaAudioEnabled.value = enabled
        _voiceMode.value = if (enabled) VoiceMode.GEMMA_AUDIO else VoiceMode.SYSTEM_STT
    }

    fun toggleTermux(enabled: Boolean) {
        toolSet.isTermuxEnabled = enabled
    }

    fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend? = null, deviceId: String? = null) {
        viewModelScope.launch {
            _isGenerating.value = true
            inferenceService.loadModel(model, preferredBackend = preferredBackend, deviceId = deviceId)
            _activeModelName.value = inferenceService.getCurrentlyLoadedModel()?.name
            _isGenerating.value = false
        }
    }

    fun unloadModel() {
        viewModelScope.launch {
            inferenceService.unloadModel()
            _activeModelName.value = null
        }
    }

    fun initTts(context: Context) {
        if (tts == null) {
            tts = TextToSpeech(context) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    tts?.language = Locale.getDefault()
                    isTtsReady = true
                }
            }
        }
    }

    fun speak(text: String) {
        if (isTtsReady) {
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "AgentTTS")
        }
    }

    fun stopTts() {
        tts?.stop()
    }

    fun confirmAction() {
        _pendingConfirmation.value?.second?.invoke()
        _pendingConfirmation.value = null
    }

    fun cancelAction() {
        _pendingConfirmation.value = null
    }

    fun sendMessage(userText: String) {
        if (userText.isBlank() || _isGenerating.value) return

        val userMsg = AgentMessage.Text(sender = AgentMessage.Sender.USER, text = userText)
        _messages.value = _messages.value + userMsg

        _isGenerating.value = true

        viewModelScope.launch {
            if (inferenceService.getCurrentlyLoadedModel() == null && !_isWebSearchEnabled.value) {
                val agentPrefs = getApplication<Application>().getSharedPreferences("agent_prefs", Context.MODE_PRIVATE)
                val savedName = agentPrefs.getString("selected_model_name", "") ?: ""
                val availableModels = ModelAvailabilityProvider.loadAvailableModels(getApplication())
                val modelToLoad = availableModels.find { it.name == savedName } ?: availableModels.firstOrNull {
                    !it.name.lowercase().contains("vision projector") &&
                    !it.name.lowercase().contains("mmproj") &&
                    !it.name.lowercase().contains("projector")
                }
                if (modelToLoad != null) {
                    loadModel(modelToLoad)
                }
            }

            processPromptWithTools(userText)
            _isGenerating.value = false
        }
    }

    fun toggleWebSearch() {
        _isWebSearchEnabled.value = !_isWebSearchEnabled.value
    }

    private suspend fun processPromptWithTools(prompt: String) {
        val lower = prompt.lowercase()

        // Flashlight tool handling
        if (lower.contains("flashlight") || lower.contains("torch")) {
            val turnOn = lower.contains("on") || !lower.contains("off")
            val toolId = UUID.randomUUID().toString()
            addMessage(AgentMessage.ToolCall(callId = toolId, toolName = "toggle_flashlight", args = if (turnOn) "true" else "false", status = AgentMessage.ToolCall.Status.RUNNING))

            val resMap = toolSet.toggleFlashlight(if (turnOn) "true" else "false")
            val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Flashlight toggled."
            val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED

            updateToolCall(toolId, status, result)
            addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = result))
            return
        }

        if (_isWebSearchEnabled.value) {
            val toolId = UUID.randomUUID().toString()
            addMessage(AgentMessage.ToolCall(callId = toolId, toolName = "web_search", args = prompt, status = AgentMessage.ToolCall.Status.RUNNING))

            val resMap = toolSet.webSearch(prompt)
            val searchResult = resMap["result"] ?: resMap["error"] ?: "No results found"
            val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED

            updateToolCall(toolId, status, searchResult)

            val loadedModel = inferenceService.getCurrentlyLoadedModel()
            if (loadedModel != null) {
                val fullPrompt = "Web search results:\n$searchResult\n\nUser Question: $prompt"
                val response = inferenceService.generateResponse(fullPrompt, loadedModel)
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = response))
            } else {
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = "Search Results:\n\n$searchResult"))
            }
        } else {
            val loadedModel = inferenceService.getCurrentlyLoadedModel()
            if (loadedModel != null) {
                val response = inferenceService.generateResponse(prompt, loadedModel)
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = response))
            } else {
                addMessage(AgentMessage.Text(
                    sender = AgentMessage.Sender.SYSTEM,
                    text = getApplication<Application>().getString(R.string.agent_no_model_android)
                ))
            }
        }
    }

    private fun addMessage(msg: AgentMessage) {
        _messages.value = _messages.value + msg
    }

    private fun updateToolCall(callId: String, status: AgentMessage.ToolCall.Status, result: String) {
        _messages.value = _messages.value.map { msg ->
            if (msg is AgentMessage.ToolCall && msg.callId == callId) {
                msg.copy(status = status, result = result)
            } else {
                msg
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        tts?.shutdown()
    }
}
