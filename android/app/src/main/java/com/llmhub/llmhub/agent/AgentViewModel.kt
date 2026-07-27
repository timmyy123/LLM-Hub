package com.llmhub.llmhub.agent

import android.app.Application
import android.content.Context
import android.util.Log
import com.llmhub.llmhub.utils.AudioConversionUtils
import android.speech.tts.TextToSpeech
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.R
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.inference.UnifiedInferenceService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.WebSearchService
import java.text.SimpleDateFormat
import java.util.Date
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
        enum class Status { PENDING_APPROVAL, RUNNING, SUCCESS, FAILED, CANCELLED }
    }

    data class MapLocation(
        val locationId: String = UUID.randomUUID().toString(),
        val label: String,
        val latitude: Double,
        val longitude: Double
    ) : AgentMessage(locationId)

    data class Audio(
        val audioId: String = UUID.randomUUID().toString(),
        val sender: Sender,
        val audioPath: String
    ) : AgentMessage(audioId)
}

enum class VoiceMode {
    SYSTEM_STT,
    WHISPER_KIT,
    GEMMA_AUDIO
}

class AgentViewModel(application: Application) : AndroidViewModel(application) {

    private val _messages = MutableStateFlow<List<AgentMessage>>(emptyList())
    val messages: StateFlow<List<AgentMessage>> = _messages.asStateFlow()

    private val agentPrefs by lazy {
        getApplication<Application>().getSharedPreferences("agent_prefs", Context.MODE_PRIVATE)
    }

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _voiceMode = MutableStateFlow(
        if (getApplication<Application>().getSharedPreferences("agent_prefs", Context.MODE_PRIVATE).getBoolean("is_gemma_audio_enabled", false)) VoiceMode.GEMMA_AUDIO else VoiceMode.SYSTEM_STT
    )
    val voiceMode: StateFlow<VoiceMode> = _voiceMode.asStateFlow()

    private val _isGemmaAudioEnabled = MutableStateFlow(
        getApplication<Application>().getSharedPreferences("agent_prefs", Context.MODE_PRIVATE).getBoolean("is_gemma_audio_enabled", false)
    )
    val isGemmaAudioEnabled: StateFlow<Boolean> = _isGemmaAudioEnabled.asStateFlow()

    private val _isWebSearchEnabled = MutableStateFlow(false)
    val isWebSearchEnabled: StateFlow<Boolean> = _isWebSearchEnabled.asStateFlow()

    private val _activeModelName = MutableStateFlow<String?>(null)
    val activeModelName: StateFlow<String?> = _activeModelName.asStateFlow()

    private val _loadingModelName = MutableStateFlow<String?>(null)
    val loadingModelName: StateFlow<String?> = _loadingModelName.asStateFlow()

    private val _isTermuxEnabled = MutableStateFlow(
        getApplication<Application>().getSharedPreferences("agent_prefs", Context.MODE_PRIVATE).getBoolean("is_termux_enabled", false)
    )
    val isTermuxEnabled: StateFlow<Boolean> = _isTermuxEnabled.asStateFlow()

    val toolSet = AgentToolSet(application.applicationContext)
    private val inferenceService: InferenceService = UnifiedInferenceService(application.applicationContext)

    private var tts: TextToSpeech? = null
    private var isTtsReady = false

    private val _pendingConfirmation = MutableStateFlow<Pair<String, () -> Unit>?>(null)
    val pendingConfirmation: StateFlow<Pair<String, () -> Unit>?> = _pendingConfirmation.asStateFlow()

    init {
        val termuxSaved = agentPrefs.getBoolean("is_termux_enabled", false)
        toolSet.isTermuxEnabled = termuxSaved
        _isTermuxEnabled.value = termuxSaved
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
        agentPrefs.edit().putBoolean("is_gemma_audio_enabled", enabled).commit()
    }

    fun toggleTermux(enabled: Boolean) {
        toolSet.isTermuxEnabled = enabled
        _isTermuxEnabled.value = enabled
        agentPrefs.edit().putBoolean("is_termux_enabled", enabled).apply()
    }

    suspend fun loadModelSuspend(model: LLMModel, preferredBackend: LlmInference.Backend? = null, deviceId: String? = null) {
        _loadingModelName.value = model.name
        try {
            inferenceService.loadModel(model, preferredBackend = preferredBackend, deviceId = deviceId)
            _activeModelName.value = inferenceService.getCurrentlyLoadedModel()?.name
        } finally {
            _loadingModelName.value = null
        }
    }

    fun loadModel(model: LLMModel, preferredBackend: LlmInference.Backend? = null, deviceId: String? = null) {
        viewModelScope.launch {
            loadModelSuspend(model, preferredBackend = preferredBackend, deviceId = deviceId)
        }
    }

    fun setGenerationParameters(maxTokens: Int? = null, enableThinking: Boolean? = null, contextWindow: Int? = null) {
        inferenceService.setGenerationParameters(
            maxTokens = maxTokens,
            topK = null,
            topP = null,
            temperature = null,
            nGpuLayers = null,
            enableThinking = enableThinking,
            contextWindow = contextWindow
        )
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
                    loadModelSuspend(modelToLoad)
                }
            }

            processPromptWithTools(userText)
            _isGenerating.value = false
        }
    }

    fun sendAudioMessage(audioBytes: ByteArray, context: Context) {
        if (_isGenerating.value) return
        _isGenerating.value = true

        val tmpAudioFile = java.io.File(context.cacheDir, "agent_voice_${System.currentTimeMillis()}.wav")
        try {
            tmpAudioFile.writeBytes(audioBytes)
        } catch (e: Exception) {
            Log.e("AgentViewModel", "Failed to write temp audio file: ${e.message}")
        }
        addMessage(AgentMessage.Audio(sender = AgentMessage.Sender.USER, audioPath = tmpAudioFile.absolutePath))

        viewModelScope.launch {
            try {
                var loadedModel = inferenceService.getCurrentlyLoadedModel()
                if (loadedModel == null) {
                    val agentPrefs = getApplication<Application>().getSharedPreferences("agent_prefs", Context.MODE_PRIVATE)
                    val savedName = agentPrefs.getString("selected_model_name", "") ?: ""
                    val availableModels = ModelAvailabilityProvider.loadAvailableModels(getApplication())
                    val modelToLoad = availableModels.find { it.name == savedName } ?: availableModels.firstOrNull {
                        !it.name.lowercase().contains("vision projector") &&
                        !it.name.lowercase().contains("mmproj") &&
                        !it.name.lowercase().contains("projector")
                    }
                    if (modelToLoad != null) {
                        loadModelSuspend(modelToLoad)
                        loadedModel = inferenceService.getCurrentlyLoadedModel()
                    }
                }

                if (loadedModel != null && (loadedModel.supportsAudio || loadedModel.name.contains("Gemma-4", ignoreCase = true))) {
                    // Native Gemma Audio ingestion
                    processPromptWithTools(prompt = "User spoken request", audioBytes = audioBytes)
                } else {
                    // Whisper ASR fallback
                    val whisperService = com.llmhub.llmhub.inference.WhisperKitService(context)
                    if (!whisperService.isLoaded) {
                        val asrModels = ModelAvailabilityProvider.loadAvailableModels(context, includeAsr = true)
                            .filter { it.modelFormat.lowercase() == "whisperkit" || it.name.lowercase().contains("whisper") }
                        val asrModel = asrModels.firstOrNull()
                        if (asrModel != null) {
                            val modelDirName = asrModel.name.replace(" ", "_").replace(Regex("[^a-zA-Z0-9_.-]"), "")
                            val modelDir = java.io.File(context.filesDir, "models/$modelDirName")
                            if (modelDir.exists()) {
                                whisperService.loadModel(modelDir.absolutePath)
                            }
                        }
                    }

                    if (whisperService.isLoaded) {
                        val pcm16Wav = AudioConversionUtils.float32WavToPcm16Wav(audioBytes)
                        val pcm16Raw = if (pcm16Wav.size > 44) pcm16Wav.copyOfRange(44, pcm16Wav.size) else pcm16Wav
                        val transcribed = whisperService.transcribe(pcm16Raw)
                        if (!transcribed.isNullOrBlank()) {
                            processPromptWithTools(transcribed)
                        } else {
                            addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = "Could not transcribe recorded audio."))
                        }
                    } else {
                        addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = "Native audio input requires Gemma 4 Audio model or a downloaded Whisper ASR model."))
                    }
                }
            } catch (e: Exception) {
                Log.e("AgentViewModel", "Failed to process audio input: ${e.message}", e)
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = "Audio processing error: ${e.message}"))
            } finally {
                _isGenerating.value = false
            }
        }
    }

    fun toggleWebSearch() {
        _isWebSearchEnabled.value = !_isWebSearchEnabled.value
    }

    private fun updateAgentTextMessage(id: String, newText: String) {
        _messages.value = _messages.value.map { msg ->
            if (msg is AgentMessage.Text && msg.id == id) {
                msg.copy(text = newText)
            } else {
                msg
            }
        }
    }

    private suspend fun processPromptWithTools(prompt: String, audioBytes: ByteArray? = null) {
        val todayStr = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())
        val agentPrefs = getApplication<Application>().getSharedPreferences("agent_prefs", Context.MODE_PRIVATE)
        val enableThinking = agentPrefs.getBoolean("agent_enable_thinking", true)
        val maxTokens = agentPrefs.getInt("selected_max_tokens", 4096)
        setGenerationParameters(maxTokens = maxTokens, enableThinking = enableThinking, contextWindow = maxTokens)

        if (_isWebSearchEnabled.value) {
            val loadedModel = inferenceService.getCurrentlyLoadedModel()
            val aiMsgId = UUID.randomUUID().toString()
            addMessage(AgentMessage.Text(messageId = aiMsgId, sender = AgentMessage.Sender.AGENT, text = ""))

            if (loadedModel != null) {
                var responseText = ""
                inferenceService.generateResponseStreamWithSession(
                    prompt = prompt,
                    model = loadedModel,
                    chatId = "agent_session",
                    images = emptyList(),
                    audioData = audioBytes,
                    webSearchEnabled = true,
                    imagePaths = emptyList()
                ).collect { chunk ->
                    responseText += chunk
                    updateAgentTextMessage(aiMsgId, responseText)
                }
            }
            return
        }

        val loadedModel = inferenceService.getCurrentlyLoadedModel()
        if (loadedModel != null) {
            val termuxToolDef = if (_isTermuxEnabled.value) {
                "\n- run_termux_command(command: \"shell command\"): Run terminal or shell commands in Termux (e.g. \"ls\", \"pkg install git\", \"uname -a\", \"python script.py\")."
            } else "\n- run_termux_command(command: \"shell command\"): Run terminal or shell commands in Termux (e.g. \"ls\", \"pkg install git\")."

            val systemPrompt = """
                You are an AI Agent equipped with device tools. Today's date is $todayStr.
                Available Tools:
                - show_map(location: "query"): Search for and display any place, venue, business category, address, or points of interest near the user (e.g. "bar near me", "coffee", "gas station", "Eiffel Tower").
                - send_email(recipient: "email address or contact name", subject: "subject line", body: "email body text"): Open email app to compose an email.
                - send_sms(recipient: "contact name or phone number", body: "SMS text content"): Open SMS app to send a text message.
                - add_calendar_event(title: "event title", date: "event date/time"): Add an event to device calendar.
                - check_weather(location: "city or 'my location'"): Check weather forecast for a specified city or the user's current location (e.g. "my location", "Tokyo", "London").
                - set_alarm(time: "time e.g. 7:00 AM", label: "alarm label"): Set an alarm on device.
                - toggle_flashlight(enabled: "true" or "false"): Turn flashlight ON or OFF.
                - calculate_hash(text: "text string", algorithm: "SHA-256 or MD5 or SHA-512"): Calculate cryptographic hash of input text.$termuxToolDef

                Instructions:
                - ALWAYS call a tool when the user asks to perform an action supported by the tools.
                - For any request to find, show, search for, or locate a place, business, venue, address, or directions (e.g. "find bar near me"), YOU MUST call show_map.
                - For any request about weather (e.g. "How's the weather", "Is it cold outside?"), YOU MUST call check_weather(location: "my location").
                - For any request to calculate a hash or hash a text (e.g. "Calculate SHA-256 hash for 'Hello World'"), YOU MUST call calculate_hash(text: "Hello World", algorithm: "SHA-256").
                - For any request to run, execute, or perform terminal/shell commands, list files (ls), install packages, or run scripts, YOU MUST call run_termux_command(command: "command string").
                - Output tool calls in this format ONLY:
                [TOOL: tool_name(arguments)]

                User Request: $prompt
            """.trimIndent()

            val aiMsgId = UUID.randomUUID().toString()
            addMessage(AgentMessage.Text(messageId = aiMsgId, sender = AgentMessage.Sender.AGENT, text = ""))

            var responseText = ""
            var executedTool = false
            try {
                inferenceService.generateResponseStreamWithSession(
                    prompt = systemPrompt,
                    model = loadedModel,
                    chatId = "agent_session",
                    images = emptyList(),
                    audioData = audioBytes,
                    webSearchEnabled = false,
                    imagePaths = emptyList()
                ).collect { chunk ->
                    if (executedTool) return@collect
                    responseText += chunk
                    val toolMatch = parseToolCall(responseText)
                    if (toolMatch != null) {
                        executedTool = true
                        _messages.value = _messages.value.filterNot { it.id == aiMsgId }
                        handleParsedToolCall(toolMatch, prompt)
                    } else {
                        val cleanText = responseText
                            .replace(Regex("""\[TOOL:[^\]]+\]""", RegexOption.IGNORE_CASE), "")
                            .replace(Regex("""(?:SHOW_MAP|SEND_SMS|ADD_CALENDAR_EVENT|CREATE_CALENDAR_EVENT|CHECK_WEATHER|GET_CURRENT_WEATHER|SET_ALARM|TOGGLE_FLASHLIGHT|CALCULATE_HASH|SEND_EMAIL|RUN_TERMUX_COMMAND|EXECUTE_TERMUX_COMMAND)\([^)]*\)""", RegexOption.IGNORE_CASE), "")
                            .trim()
                        updateAgentTextMessage(aiMsgId, cleanText)
                    }
                }
            } catch (e: Exception) {
                Log.e("AgentViewModel", "Stream error: ${e.message}")
            }

            if (_messages.value.any { it.id == aiMsgId && (it as? AgentMessage.Text)?.text?.isBlank() == true }) {
                _messages.value = _messages.value.filterNot { it.id == aiMsgId }
            }
        } else {
            // Model not loaded fallback: execute direct tool requests semantically
            executeToolOrFallback(prompt)
        }
    }

    private data class ParsedTool(val name: String, val args: String)

    private fun parseAlarmTime(text: String): Pair<Int, Int> {
        val lower = text.lowercase()
        val timeMatch = Regex("""(\d{1,2})(?::(\d{2}))?\s*(am|pm)?""").find(lower)
        if (timeMatch != null) {
            var hour = timeMatch.groupValues[1].toIntOrNull() ?: 8
            val min = timeMatch.groupValues[2].toIntOrNull() ?: 0
            val ampm = timeMatch.groupValues[3].ifEmpty { null }
            if (ampm == "pm" && hour < 12) hour += 12
            if (ampm == "am" && hour == 12) hour = 0
            return Pair(hour.coerceIn(0, 23), min.coerceIn(0, 59))
        }
        return Pair(8, 0)
    }

    private fun parseToolCall(text: String): ParsedTool? {
        // Match [TOOL: any_tool_name(args)] or standalone known tool patterns
        val regex = Regex("""\[?TOOL:\s*([a-zA-Z0-9._]+)\(([^)]*)\)\]?|\b([a-zA-Z_]+(?:_map|_sms|_email|_weather|_alarm|_flashlight|_event|_hash|_termux|_command))\(([^)]*)\)""", RegexOption.IGNORE_CASE)
        val match = regex.find(text) ?: return null
        val groups = match.groupValues
        val name = if (groups[1].isNotEmpty()) groups[1] else groups[3]
        val args = if (groups[2].isNotEmpty()) groups[2] else groups[4]
        val cleanName = name.replace(".", "_").replace("/", "_").lowercase()
        val cleanArgs = args.trim('"', '\'', ' ', ')', ']')
        return ParsedTool(cleanName, cleanArgs)
    }

    private suspend fun handleParsedToolCall(tool: ParsedTool, originalPrompt: String) {
        val toolId = UUID.randomUUID().toString()
        addMessage(AgentMessage.ToolCall(callId = toolId, toolName = tool.name, args = tool.args, status = AgentMessage.ToolCall.Status.RUNNING))

        // Reroute calendar tool calls to set_alarm if prompt or title specifies alarm
        val isAlarmIntent = originalPrompt.contains("alarm", ignoreCase = true) || tool.args.contains("alarm", ignoreCase = true)
        var effectiveToolName = if (isAlarmIntent && (tool.name.contains("calendar") || tool.name.contains("event"))) "set_alarm" else tool.name.lowercase()

        // Normalize tool name aliases — LiteRT-LM Gemma-4 may output open_map, find_map, display_map etc.
        if (effectiveToolName.contains("map")) effectiveToolName = "show_map"
        if (effectiveToolName.contains("weather")) effectiveToolName = "check_weather"
        if (effectiveToolName.contains("email") || effectiveToolName.contains("mail")) effectiveToolName = "send_email"
        if (effectiveToolName.contains("sms") || effectiveToolName == "send_message") effectiveToolName = "send_sms"
        if (effectiveToolName.contains("flashlight") || effectiveToolName.contains("torch")) effectiveToolName = "toggle_flashlight"
        if (effectiveToolName.contains("alarm") && effectiveToolName != "set_alarm") effectiveToolName = "set_alarm"
        if (effectiveToolName.contains("calendar") || effectiveToolName.contains("event")) effectiveToolName = "add_calendar_event"

        when (effectiveToolName) {
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
            "send_email" -> {
                val recipient = extractArgValue(tool.args, "recipient") ?: extractArgValue(tool.args, "to") ?: tool.args.split(",").firstOrNull()?.trim('"', '\'', ' ') ?: "contact"
                val body = extractArgValue(tool.args, "body") ?: extractArgValue(tool.args, "message") ?: originalPrompt
                val subject = extractArgValue(tool.args, "subject") ?: "Hello"
                val resMap = toolSet.sendEmail(recipient, subject, body)
                val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Email app opened."
                val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
                updateToolCall(toolId, status, result)
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = result))
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
            "add_calendar_event", "create_calendar_event" -> {
                val title = extractArgValue(tool.args, "title") ?: tool.args.split(",").firstOrNull()?.trim('"', '\'', ' ') ?: "New Event"
                val loc = extractArgValue(tool.args, "location") ?: ""
                val resMap = toolSet.createCalendarEvent(title, loc, "")
                val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Calendar event created."
                val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
                updateToolCall(toolId, status, result)
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = result))
            }
            "check_weather", "get_current_weather" -> {
                val loc = extractArgValue(tool.args, "location") ?: tool.args.trim('"', '\'', ' ')
                val resMap = toolSet.getCurrentWeather(loc)
                val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Weather information."
                val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
                updateToolCall(toolId, status, result)
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = result))
            }
            "set_alarm" -> {
                val label = extractArgValue(tool.args, "label") ?: extractArgValue(tool.args, "title") ?: "Alarm"
                val timeStr = extractArgValue(tool.args, "time") ?: extractArgValue(tool.args, "date") ?: tool.args
                val (hour, min) = parseAlarmTime("$timeStr $originalPrompt")
                val resMap = toolSet.setAlarm(hour, min, label)
                val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Alarm set."
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
            "calculate_hash" -> {
                val text = extractArgValue(tool.args, "text") ?: tool.args.split(",").firstOrNull()?.trim('"', '\'', ' ') ?: originalPrompt
                val algo = extractArgValue(tool.args, "algorithm") ?: "SHA-256"
                val resMap = toolSet.calculateHash(text, algo)
                val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Hash calculated."
                val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
                val displayResult = if (status == AgentMessage.ToolCall.Status.SUCCESS) "Hash (${resMap["algorithm"] ?: algo}) for '$text':\n$result" else result
                updateToolCall(toolId, status, displayResult)
                addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = displayResult))
            }
            "run_termux_command", "execute_termux_command" -> {
                val cmd = extractArgValue(tool.args, "command") ?: tool.args.trim('"', '\'', ' ')
                addMessage(AgentMessage.ToolCall(callId = toolId, toolName = "run_termux_command", args = cmd, status = AgentMessage.ToolCall.Status.PENDING_APPROVAL))
            }
            else -> {
                updateToolCall(toolId, AgentMessage.ToolCall.Status.FAILED, "Unknown tool")
            }
        }
    }

    private fun extractArgValue(args: String, key: String): String? {
        val regex = Regex("""$key\s*(?:=\s*|:\s*|\(\s*)"([^"]+)"|$key\s*(?:=\s*|:\s*|\(\s*)'([^']+)'|$key\s*(?:=\s*|:\s*)\s*([^,\s)]+)""", RegexOption.IGNORE_CASE)
        val match = regex.find(args)
        val raw = match?.groupValues?.get(1)?.ifEmpty { null }
            ?: match?.groupValues?.get(2)?.ifEmpty { null }
            ?: match?.groupValues?.get(3)?.ifEmpty { null }

        if (raw != null) {
            var cleanVal = raw.trim('"', '\'', ' ')
            val lowerKey = key.lowercase()
            if (cleanVal.lowercase().startsWith("$lowerKey:")) {
                cleanVal = cleanVal.substring(lowerKey.length + 1).trim('"', '\'', ' ')
            }
            val lower = cleanVal.lowercase()
            if (key == "location" && (lower == "query" || lower.contains("weather") || lower.contains("current location") || lower.contains("here") || lower.contains("my location"))) {
                return "my location"
            }
            return cleanVal
        }

        return null
    }

    private suspend fun executeToolOrFallback(prompt: String) {
        val lower = prompt.lowercase()
        val toolId = UUID.randomUUID().toString()

        if (lower.contains("weather") || lower.contains("forecast") || lower.contains("temperature")) {
            addMessage(AgentMessage.ToolCall(callId = toolId, toolName = "check_weather", args = "my location", status = AgentMessage.ToolCall.Status.RUNNING))
            val resMap = toolSet.getCurrentWeather("my location")
            val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Weather information."
            val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
            updateToolCall(toolId, status, result)
            addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = result))
        } else if (lower.contains("alarm") || lower.contains("remind") || lower.contains("wake")) {
            val (h, m) = parseAlarmTime(prompt)
            addMessage(AgentMessage.ToolCall(callId = toolId, toolName = "set_alarm", args = prompt, status = AgentMessage.ToolCall.Status.RUNNING))
            val resMap = toolSet.setAlarm(h, m, "Alarm")
            val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Alarm set."
            val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
            updateToolCall(toolId, status, result)
            addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = result))
        } else if (lower.contains("hash") || lower.contains("sha") || lower.contains("md5")) {
            var algo = "SHA-256"
            if (lower.contains("md5")) algo = "MD5"
            else if (lower.contains("sha-1") || lower.contains("sha1")) algo = "SHA-1"
            else if (lower.contains("sha-512") || lower.contains("sha512")) algo = "SHA-512"

            var textToHash = prompt
            val firstQuote = prompt.indexOf("'")
            val lastQuote = prompt.lastIndexOf("'")
            if (firstQuote != -1 && lastQuote > firstQuote) {
                textToHash = prompt.substring(firstQuote + 1, lastQuote)
            }

            addMessage(AgentMessage.ToolCall(callId = toolId, toolName = "calculate_hash", args = "text: \"$textToHash\", algorithm: \"$algo\"", status = AgentMessage.ToolCall.Status.RUNNING))
            val resMap = toolSet.calculateHash(textToHash, algo)
            val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Hash calculated."
            val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
            val displayResult = if (status == AgentMessage.ToolCall.Status.SUCCESS) "Hash (${resMap["algorithm"] ?: algo}) for '$textToHash':\n$result" else result
            updateToolCall(toolId, status, displayResult)
            addMessage(AgentMessage.Text(sender = AgentMessage.Sender.AGENT, text = displayResult))
        } else if (lower.contains("termux") || lower.contains("shell") || lower.contains("ls") || lower.contains("pkg ")) {
            var cmd = prompt
            if (lower.contains("termux")) {
                cmd = prompt.substringAfter("termux", prompt).trim()
            } else if (lower.startsWith("run ") || lower.startsWith("execute ")) {
                cmd = prompt.substringAfter(" ").trim()
                if (cmd.lowercase().startsWith("a ") || cmd.lowercase().startsWith("an ")) {
                    cmd = cmd.substringAfter(" ").trim()
                }
                if (cmd.lowercase().endsWith(" command")) {
                    cmd = cmd.substring(0, cmd.length - 8).trim()
                }
            }
            if (cmd.isBlank()) cmd = "ls"
            addMessage(AgentMessage.ToolCall(callId = toolId, toolName = "run_termux_command", args = cmd, status = AgentMessage.ToolCall.Status.PENDING_APPROVAL))
        } else if (lower.contains("map") || lower.contains("where is") || lower.contains("find") || lower.contains("direction")) {
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
        } else {
            addMessage(AgentMessage.Text(
                sender = AgentMessage.Sender.SYSTEM,
                text = getApplication<Application>().getString(R.string.agent_no_model_android)
            ))
        }
    }

    private fun addMessage(msg: AgentMessage) {
        _messages.value = _messages.value.filterNot { it.id == msg.id } + msg
    }

    fun approveAndExecuteTermuxCommand(callId: String, command: String) {
        viewModelScope.launch(Dispatchers.IO) {
            updateToolCall(callId, AgentMessage.ToolCall.Status.RUNNING, "Executing command in Termux...")
            val resMap = toolSet.runTermuxCommand(command)
            val result = resMap["result"] as? String ?: resMap["error"] as? String ?: "Execution error."
            val status = if (resMap["status"] == "succeeded") AgentMessage.ToolCall.Status.SUCCESS else AgentMessage.ToolCall.Status.FAILED
            updateToolCall(callId, status, result)
        }
    }

    fun cancelTermuxCommand(callId: String) {
        val cancelledText = getApplication<Application>().getString(R.string.agent_termux_cancelled)
        updateToolCall(callId, AgentMessage.ToolCall.Status.CANCELLED, cancelledText)
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
