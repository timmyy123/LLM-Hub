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
            if (inferenceService.getCurrentlyLoadedModel() == null) {
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
            return
        }

        val loadedModel = inferenceService.getCurrentlyLoadedModel()
        if (loadedModel != null) {
            val systemPrompt = """
                You are an AI Agent equipped with device tools:
                - show_map(location: "place/venue query")
                - send_sms(recipient: "contact name/phone", body: "message text")
                - toggle_flashlight(enabled: "true" or "false")

                To execute a tool call, output formatted exactly as:
                [TOOL: tool_name(arguments)]

                User Request: $prompt
            """.trimIndent()

            val response = inferenceService.generateResponse(systemPrompt, loadedModel)
            val toolMatch = parseToolCall(response)
            if (toolMatch != null) {
                handleParsedToolCall(toolMatch, prompt)
            } else {
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = response))
            }
        } else {
            // Model not loaded fallback: execute direct tool requests semantically
            executeToolOrFallback(prompt)
        }
    }

    private data class ParsedTool(val name: String, val args: String)

    private fun parseToolCall(text: String): ParsedTool? {
        val clean = text.trim()
        val regex = Regex("""(?:\[|\b)(TOOL:|SHOW_MAP|SEND_SMS|TOGGLE_FLASHLIGHT)[:(]([^\])]+)[\])]?""", RegexOption.IGNORE_CASE)
        val match = regex.find(clean) ?: return null
        val fullMatch = match.groupValues[0].trim('[', ']', ' ')

        val name: String
        val argsStr: String

        if (fullMatch.contains(":")) {
            val parts = fullMatch.split(":", limit = 2)
            name = parts[0].trim()
            argsStr = parts.getOrNull(1)?.trim() ?: ""
        } else if (fullMatch.contains("(")) {
            val parts = fullMatch.split("(", limit = 2)
            name = parts[0].trim()
            argsStr = parts.getOrNull(1)?.trim(')', ' ') ?: ""
        } else {
            return null
        }

        if (name.equals("TOOL", ignoreCase = true)) {
            return parseToolCall(argsStr)
        }

        return ParsedTool(name.lowercase(), argsStr)
    }

    private suspend fun handleParsedToolCall(tool: ParsedTool, originalPrompt: String) {
        val toolId = UUID.randomUUID().toString()
        addMessage(AgentMessage.ToolCall(callId = toolId, toolName = tool.name, args = tool.args, status = AgentMessage.ToolCall.Status.RUNNING))

        when (tool.name.lowercase()) {
            "show_map" -> {
                val cleanLoc = extractArgValue(tool.args, "location") ?: tool.args.trim('"', '\'', ' ')
                val resMap = toolSet.showMap(cleanLoc)
                val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
                val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Map location for '$cleanLoc'"
                updateToolCall(toolId, status, result)

                if (resMap["status"] == "succeeded") {
                    val lat = (resMap["lat"] as? Double) ?: 37.422
                    val lon = (resMap["lon"] as? Double) ?: -122.084
                    val label = (resMap["label"] as? String) ?: cleanLoc
                    addMessage(AgentMessage.MapLocation(label = label, latitude = lat, longitude = lon))
                } else {
                    addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = result))
                }
            }
            "send_sms" -> {
                val recipient = extractArgValue(tool.args, "recipient") ?: extractArgValue(tool.args, "to") ?: tool.args.split(",").firstOrNull()?.trim('"', '\'', ' ') ?: "contact"
                val body = extractArgValue(tool.args, "body") ?: extractArgValue(tool.args, "message") ?: originalPrompt
                val resMap = toolSet.sendSms(recipient, body)
                val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "SMS app opened."
                val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
                updateToolCall(toolId, status, result)
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = result))
            }
            "toggle_flashlight" -> {
                val enabled = tool.args.contains("true", ignoreCase = true) || tool.args.contains("on", ignoreCase = true)
                val resMap = toolSet.toggleFlashlight(if (enabled) "true" else "false")
                val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Flashlight toggled."
                val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
                updateToolCall(toolId, status, result)
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = result))
            }
            else -> {
                updateToolCall(toolId, AgentMessage.ToolCall.Status.FAILED, "Unknown tool")
            }
        }
    }

    private fun extractArgValue(args: String, key: String): String? {
        val regex = Regex("""$key\s*=\s*"([^"]+)"|$key\s*=\s*'([^']+)'|$key\s*=\s*([^,\s)]+)""", RegexOption.IGNORE_CASE)
        val match = regex.find(args) ?: return null
        val raw = match.value
        val parts = raw.split("=", limit = 2)
        return parts.getOrNull(1)?.trim('"', '\'', ' ')
    }

    private suspend fun executeToolOrFallback(prompt: String) {
        val toolId = UUID.randomUUID().toString()
        addMessage(AgentMessage.ToolCall(callId = toolId, toolName = "show_map", args = prompt, status = AgentMessage.ToolCall.Status.RUNNING))

        val resMap = toolSet.showMap(prompt)
        val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
        val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Map location for '$prompt'"
        updateToolCall(toolId, status, result)

        if (resMap["status"] == "succeeded") {
            val lat = (resMap["lat"] as? Double) ?: 37.422
            val lon = (resMap["lon"] as? Double) ?: -122.084
            val label = (resMap["label"] as? String) ?: prompt
            addMessage(AgentMessage.MapLocation(label = label, latitude = lat, longitude = lon))
        } else {
            addMessage(AgentMessage.Text(
                sender = AgentMessage.Sender.SYSTEM,
                text = getApplication<Application>().getString(R.string.agent_no_model_android)
            ))
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
