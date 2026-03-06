package com.llmhub.llmhub.viewmodels

import android.app.Application
import android.content.Context
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelAvailabilityProvider
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class CodeLanguage {
    HTML, PYTHON, JAVASCRIPT, UNKNOWN
}

enum class ProgrammingLanguage {
    WEB,
    PYTHON,
    JAVASCRIPT,
    TYPESCRIPT,
    JAVA,
    KOTLIN,
    CSHARP,
    CPP,
    GO,
    RUST
}

data class VibeChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: String,
    val text: String
)

data class CodeProposal(
    val id: String = UUID.randomUUID().toString(),
    val prompt: String,
    val promptMessageId: String?,
    val code: String,
    val language: CodeLanguage
)

data class EditCheckpoint(
    val id: String = UUID.randomUUID().toString(),
    val prompt: String,
    val promptMessageId: String?,
    val beforeCode: String,
    val afterCode: String,
    val changedLines: Int
)

data class VibeChatSessionSummary(
    val id: String,
    val title: String
)

/**
 * VibeCoderViewModel handles code generation using LLM inference.
 * Users provide a prompt, and the model generates HTML/Python/JavaScript code.
 */
class VibeCoderViewModel(application: Application) : AndroidViewModel(application) {
    
    private val inferenceService = (application as com.llmhub.llmhub.LlmHubApplication).inferenceService
    private val prefs = application.getSharedPreferences("vibe_coder_prefs", Context.MODE_PRIVATE)
    
    private var processingJob: Job? = null
    private var streamingAssistantMessageId: String? = null
    private var currentPromptMessageId: String? = null
    private val chatSessionStore = mutableMapOf<String, SessionPayload>()
    
    // Available models
    private val _availableModels = MutableStateFlow<List<LLMModel>>(emptyList())
    val availableModels: StateFlow<List<LLMModel>> = _availableModels.asStateFlow()
    
    // Model selection & backend
    private val _selectedModel = MutableStateFlow<LLMModel?>(null)
    val selectedModel: StateFlow<LLMModel?> = _selectedModel.asStateFlow()
    
    private val _selectedBackend = MutableStateFlow<LlmInference.Backend?>(null)
    val selectedBackend: StateFlow<LlmInference.Backend?> = _selectedBackend.asStateFlow()
    
    // Optional selected NPU device id when user chooses NPU for GGUF
    private val _selectedNpuDeviceId = MutableStateFlow<String?>(null)
    val selectedNpuDeviceId: StateFlow<String?> = _selectedNpuDeviceId.asStateFlow()

    private val _selectedNGpuLayers = MutableStateFlow<Int?>(null)

    private val _selectedMaxTokens = MutableStateFlow(4096)
    val selectedMaxTokens: StateFlow<Int> = _selectedMaxTokens.asStateFlow()

    // Loading states
    private val _isModelLoaded = MutableStateFlow(false)
    val isModelLoaded: StateFlow<Boolean> = _isModelLoaded.asStateFlow()
    
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()
    
    private val _isPlanning = MutableStateFlow(false)
    val isPlanning: StateFlow<Boolean> = _isPlanning.asStateFlow()
    
    private var currentSpec: String = ""
    
    // Generated code & metadata
    private val _generatedCode = MutableStateFlow("")
    val generatedCode: StateFlow<String> = _generatedCode.asStateFlow()
    private val _currentFileUri = MutableStateFlow<String?>(null)
    val currentFileUri: StateFlow<String?> = _currentFileUri.asStateFlow()
    private val _currentFileName = MutableStateFlow<String?>(null)
    val currentFileName: StateFlow<String?> = _currentFileName.asStateFlow()
    private val _currentFolderUri = MutableStateFlow<String?>(null)
    val currentFolderUri: StateFlow<String?> = _currentFolderUri.asStateFlow()
    private val _isDirty = MutableStateFlow(false)
    val isDirty: StateFlow<Boolean> = _isDirty.asStateFlow()
    private val _chatMessages = MutableStateFlow<List<VibeChatMessage>>(emptyList())
    val chatMessages: StateFlow<List<VibeChatMessage>> = _chatMessages.asStateFlow()
    private val _pendingProposal = MutableStateFlow<CodeProposal?>(null)
    val pendingProposal: StateFlow<CodeProposal?> = _pendingProposal.asStateFlow()
    private val _editCheckpoints = MutableStateFlow<List<EditCheckpoint>>(emptyList())
    val editCheckpoints: StateFlow<List<EditCheckpoint>> = _editCheckpoints.asStateFlow()
    private val _lastUserPrompt = MutableStateFlow<String?>(null)
    val lastUserPrompt: StateFlow<String?> = _lastUserPrompt.asStateFlow()
    private val _chatSessions = MutableStateFlow<List<VibeChatSessionSummary>>(emptyList())
    val chatSessions: StateFlow<List<VibeChatSessionSummary>> = _chatSessions.asStateFlow()
    private val _activeChatSessionId = MutableStateFlow<String?>(null)
    val activeChatSessionId: StateFlow<String?> = _activeChatSessionId.asStateFlow()
    
    private val _codeLanguage = MutableStateFlow(CodeLanguage.UNKNOWN)
    val codeLanguage: StateFlow<CodeLanguage> = _codeLanguage.asStateFlow()
    
    private val _promptInput = MutableStateFlow("")
    val promptInput: StateFlow<String> = _promptInput.asStateFlow()
    
    // Error handling
    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()
    
    private val _enableThinking = MutableStateFlow(true)
    val enableThinking: StateFlow<Boolean> = _enableThinking.asStateFlow()
    private val _preferredLanguage = MutableStateFlow(ProgrammingLanguage.WEB)
    val preferredLanguage: StateFlow<ProgrammingLanguage> = _preferredLanguage.asStateFlow()

    private var pendingSavedModelName: String? = null

    private data class SessionPayload(
        val id: String,
        var title: String,
        var messages: MutableList<VibeChatMessage>,
        var checkpoints: MutableList<EditCheckpoint>,
        var lastPrompt: String?
    )
    
    init {
        loadSavedSettings()
        loadAvailableModels()
    }
    
    /**
     * Load previously saved settings (model, backend)
     */
    private fun loadSavedSettings() {
        pendingSavedModelName = prefs.getString("selected_model_name", null)
        _currentFileUri.value = prefs.getString("last_opened_file_uri", null)
        _currentFileName.value = prefs.getString("last_opened_file_name", null)
        _currentFolderUri.value = prefs.getString("last_opened_folder_uri", null)
        _generatedCode.value = prefs.getString("last_draft_code", "") ?: ""
        _isDirty.value = prefs.getBoolean("last_draft_dirty", false)
        if (_generatedCode.value.isNotBlank() && _currentFileName.value != null) {
            _codeLanguage.value = languageFromFileName(_currentFileName.value)
        }
        loadPersistedChatSessions()
    }

    private fun restoreSettingsForModel(model: LLMModel) {
        val savedTokens = prefs.getInt("max_tokens_${model.name}", minOf(4096, model.contextWindowSize.coerceAtLeast(1)))
        _selectedMaxTokens.value = savedTokens.coerceIn(1, model.contextWindowSize.coerceAtLeast(1))

        val savedBackendName = prefs.getString("selected_backend_${model.name}", prefs.getString("selected_backend", LlmInference.Backend.GPU.name))
        val restoredBackend = try {
            LlmInference.Backend.valueOf(savedBackendName ?: LlmInference.Backend.GPU.name)
        } catch (_: IllegalArgumentException) {
            LlmInference.Backend.GPU
        }
        _selectedBackend.value = if (model.supportsGpu) restoredBackend else LlmInference.Backend.CPU

        _selectedNpuDeviceId.value = if (_selectedBackend.value == LlmInference.Backend.GPU) {
            prefs.getString("selected_npu_device_id_${model.name}", prefs.getString("selected_npu_device_id", null))
        } else {
            null
        }

        _enableThinking.value = prefs.getBoolean("enable_thinking_${model.name}", prefs.getBoolean("enable_thinking", true))
        _selectedNGpuLayers.value = prefs.getInt("n_gpu_layers_${model.name}", 999).let { if (it == 999) null else it }
        val savedLangName = prefs.getString("code_language_${model.name}", prefs.getString("code_language", ProgrammingLanguage.WEB.name))
        _preferredLanguage.value = try {
            ProgrammingLanguage.valueOf(savedLangName ?: ProgrammingLanguage.WEB.name)
        } catch (_: Exception) {
            ProgrammingLanguage.WEB
        }
    }
    
    /**
     * Save current model and backend preferences
     */
    private fun saveSettings() {
        prefs.edit().apply {
            putString("selected_model_name", _selectedModel.value?.name)
            _selectedModel.value?.let { model ->
                putString("selected_backend_${model.name}", _selectedBackend.value?.name)
                putString("selected_npu_device_id_${model.name}", _selectedNpuDeviceId.value)
                putInt("max_tokens_${model.name}", _selectedMaxTokens.value)
                putBoolean("enable_thinking_${model.name}", _enableThinking.value)
                putInt("n_gpu_layers_${model.name}", _selectedNGpuLayers.value ?: 999)
                putString("code_language_${model.name}", _preferredLanguage.value.name)
            }
            putString("selected_backend", _selectedBackend.value?.name)
            putString("selected_npu_device_id", _selectedNpuDeviceId.value)
            putBoolean("enable_thinking", _enableThinking.value)
            putString("code_language", _preferredLanguage.value.name)
            putString("last_opened_file_uri", _currentFileUri.value)
            putString("last_opened_file_name", _currentFileName.value)
            putString("last_opened_folder_uri", _currentFolderUri.value)
            putString("last_draft_code", _generatedCode.value)
            putBoolean("last_draft_dirty", _isDirty.value)
            putString("active_chat_session_id", _activeChatSessionId.value)
            putString("chat_sessions_json", serializeChatSessions())
            apply()
        }
    }

    private fun serializeChatSessions(): String {
        val arr = JSONArray()
        chatSessionStore.values.forEach { s ->
            val obj = JSONObject()
            obj.put("id", s.id)
            obj.put("title", s.title)
            obj.put("lastPrompt", s.lastPrompt ?: JSONObject.NULL)
            val mArr = JSONArray()
            s.messages.forEach { m ->
                val mo = JSONObject()
                mo.put("id", m.id)
                mo.put("role", m.role)
                mo.put("text", m.text)
                mArr.put(mo)
            }
            obj.put("messages", mArr)
            val cArr = JSONArray()
            s.checkpoints.forEach { c ->
                val co = JSONObject()
                co.put("id", c.id)
                co.put("prompt", c.prompt)
                co.put("promptMessageId", c.promptMessageId ?: JSONObject.NULL)
                co.put("beforeCode", c.beforeCode)
                co.put("afterCode", c.afterCode)
                co.put("changedLines", c.changedLines)
                cArr.put(co)
            }
            obj.put("checkpoints", cArr)
            arr.put(obj)
        }
        return arr.toString()
    }

    private fun loadPersistedChatSessions() {
        chatSessionStore.clear()
        val raw = prefs.getString("chat_sessions_json", null)
        if (!raw.isNullOrBlank()) {
            runCatching {
                val arr = JSONArray(raw)
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    val id = o.optString("id")
                    if (id.isBlank()) continue
                    val title = o.optString("title", "Chat")
                    val lastPrompt = if (o.isNull("lastPrompt")) null else o.optString("lastPrompt")
                    val messages = mutableListOf<VibeChatMessage>()
                    val mArr = o.optJSONArray("messages") ?: JSONArray()
                    for (j in 0 until mArr.length()) {
                        val mo = mArr.getJSONObject(j)
                        messages.add(
                            VibeChatMessage(
                                id = mo.optString("id", UUID.randomUUID().toString()),
                                role = mo.optString("role", "assistant"),
                                text = mo.optString("text", "")
                            )
                        )
                    }
                    val checkpoints = mutableListOf<EditCheckpoint>()
                    val cArr = o.optJSONArray("checkpoints") ?: JSONArray()
                    for (j in 0 until cArr.length()) {
                        val co = cArr.getJSONObject(j)
                        checkpoints.add(
                            EditCheckpoint(
                                id = co.optString("id", UUID.randomUUID().toString()),
                                prompt = co.optString("prompt", ""),
                                promptMessageId = if (co.isNull("promptMessageId")) null else co.optString("promptMessageId"),
                                beforeCode = co.optString("beforeCode", ""),
                                afterCode = co.optString("afterCode", ""),
                                changedLines = co.optInt("changedLines", 0)
                            )
                        )
                    }
                    chatSessionStore[id] = SessionPayload(id, title, messages, checkpoints, lastPrompt)
                }
            }
        }
        if (chatSessionStore.isEmpty()) {
            val id = UUID.randomUUID().toString()
            chatSessionStore[id] = SessionPayload(id, "Chat 1", mutableListOf(), mutableListOf(), null)
        }
        _chatSessions.value = chatSessionStore.values.map { VibeChatSessionSummary(it.id, it.title) }
        val savedActive = prefs.getString("active_chat_session_id", null)
        val active = if (savedActive != null && chatSessionStore.containsKey(savedActive)) savedActive else chatSessionStore.keys.first()
        selectChatSession(active)
    }

    private fun persistActiveSession() {
        val id = _activeChatSessionId.value ?: return
        val s = chatSessionStore[id] ?: return
        s.messages = _chatMessages.value.toMutableList()
        s.checkpoints = _editCheckpoints.value.toMutableList()
        s.lastPrompt = _lastUserPrompt.value
        _chatSessions.value = chatSessionStore.values.map { VibeChatSessionSummary(it.id, it.title) }
        saveSettings()
    }

    fun createNewChatSession() {
        val index = chatSessionStore.size + 1
        val id = UUID.randomUUID().toString()
        chatSessionStore[id] = SessionPayload(id, "Chat $index", mutableListOf(), mutableListOf(), null)
        _chatSessions.value = chatSessionStore.values.map { VibeChatSessionSummary(it.id, it.title) }
        selectChatSession(id)
        saveSettings()
    }

    fun selectChatSession(sessionId: String) {
        val s = chatSessionStore[sessionId] ?: return
        _activeChatSessionId.value = sessionId
        _chatMessages.value = s.messages.toList()
        _editCheckpoints.value = s.checkpoints.toList()
        _lastUserPrompt.value = s.lastPrompt
        _pendingProposal.value = null
        streamingAssistantMessageId = null
        currentPromptMessageId = null
        saveSettings()
    }
    
    /**
     * Load all available models from device
     */
    private fun loadAvailableModels() {
        viewModelScope.launch {
            val context = getApplication<Application>()
            val available = ModelAvailabilityProvider.loadAvailableModels(context)
                .filter { it.category != "embedding" && !it.name.contains("Projector", ignoreCase = true) }
            _availableModels.value = available
            if (_selectedModel.value == null) {
                val modelToSelect = pendingSavedModelName?.let { savedName ->
                    available.find { it.name == savedName }
                } ?: available.firstOrNull()
                modelToSelect?.let {
                    _selectedModel.value = it
                    restoreSettingsForModel(it)
                }
                pendingSavedModelName = null
            }
        }
    }
    
    /**
     * Select a different model for code generation
     */
    fun selectModel(model: LLMModel) {
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedModel.value = model
        _isModelLoaded.value = false
        restoreSettingsForModel(model)

        saveSettings()
    }

    fun setMaxTokens(maxTokens: Int) {
        val cap = _selectedModel.value?.contextWindowSize?.coerceAtLeast(1) ?: 4096
        _selectedMaxTokens.value = maxTokens.coerceIn(1, cap)
        saveSettings()
        applyGenerationParametersToService()
    }

    /**
     * Select inference backend (GPU, CPU, etc.)
     */
    fun selectBackend(backend: LlmInference.Backend, deviceId: String? = null) {
        if (_isModelLoaded.value) {
            unloadModel()
        }
        
        _selectedBackend.value = backend
        _selectedNpuDeviceId.value = deviceId
        _isModelLoaded.value = false
        saveSettings()
    }

    fun setNGpuLayers(n: Int) {
        _selectedNGpuLayers.value = n
        saveSettings()
        applyGenerationParametersToService()
    }
    
    /**
     * Load the selected model into memory
     */
    fun setEnableThinking(enabled: Boolean) {
        _enableThinking.value = enabled
        saveSettings()
        applyGenerationParametersToService()
    }

    fun setPreferredLanguage(language: ProgrammingLanguage) {
        _preferredLanguage.value = language
        saveSettings()
    }

    fun openEditorFile(fileUri: String, fileName: String, content: String) {
        _currentFileUri.value = fileUri
        _currentFileName.value = fileName
        _generatedCode.value = content
        _codeLanguage.value = languageFromFileName(fileName)
        _isDirty.value = false
        _pendingProposal.value = null
        saveSettings()
        appendChat("assistant", "Opened $fileName")
    }

    fun openFolder(folderUri: String) {
        _currentFolderUri.value = folderUri
        saveSettings()
        appendChat("assistant", "Opened folder workspace")
    }

    fun createNewEditorFile(fileName: String) {
        _currentFileUri.value = null
        _currentFileName.value = fileName
        _generatedCode.value = ""
        _codeLanguage.value = languageFromFileName(fileName)
        _isDirty.value = false
        _pendingProposal.value = null
        saveSettings()
        appendChat("assistant", "Started new file: $fileName")
    }

    fun markSaved(fileUri: String, fileName: String) {
        _currentFileUri.value = fileUri
        _currentFileName.value = fileName
        _isDirty.value = false
        saveSettings()
    }

    private fun appendChat(role: String, text: String) {
        _chatMessages.value = _chatMessages.value + VibeChatMessage(role = role, text = text)
        persistActiveSession()
    }

    private fun beginStreamingAssistant() {
        val msg = VibeChatMessage(role = "assistant", text = "")
        streamingAssistantMessageId = msg.id
        _chatMessages.value = _chatMessages.value + msg
    }

    private fun updateStreamingAssistant(text: String) {
        val id = streamingAssistantMessageId ?: return
        _chatMessages.value = _chatMessages.value.map { msg ->
            if (msg.id == id) msg.copy(text = text) else msg
        }
    }

    private fun endStreamingAssistant(finalText: String) {
        val id = streamingAssistantMessageId
        if (id == null) {
            appendChat("assistant", finalText)
            return
        }
        _chatMessages.value = _chatMessages.value.map { msg ->
            if (msg.id == id) msg.copy(text = finalText) else msg
        }
        streamingAssistantMessageId = null
        persistActiveSession()
    }

    fun clearChatSession() {
        _chatMessages.value = emptyList()
        _pendingProposal.value = null
        streamingAssistantMessageId = null
        _editCheckpoints.value = emptyList()
        _lastUserPrompt.value = null
        persistActiveSession()
    }

    fun applyPendingProposal() {
        val proposal = _pendingProposal.value ?: return
        val before = _generatedCode.value
        if (!isSafeFullFileUpdate(before, proposal.code)) {
            appendChat(
                "assistant",
                "Blocked apply: model output looks partial and would overwrite file. Please ask AI to return the complete file."
            )
            return
        }
        _generatedCode.value = proposal.code
        _codeLanguage.value = proposal.language
        _isDirty.value = true
        val changed = countChangedLines(before, proposal.code)
        _editCheckpoints.value = (listOf(
            EditCheckpoint(
                prompt = proposal.prompt,
                promptMessageId = proposal.promptMessageId,
                beforeCode = before,
                afterCode = proposal.code,
                changedLines = changed
            )
        ) + _editCheckpoints.value).take(30)
        _pendingProposal.value = null
        saveSettings()
        appendChat("assistant", "Applied proposed changes to editor.")
        persistActiveSession()
    }

    private fun applyAutoProposal(
        prompt: String,
        promptMessageId: String?,
        proposedCode: String,
        proposedLanguage: CodeLanguage
    ) {
        val before = _generatedCode.value
        if (!isSafeFullFileUpdate(before, proposedCode)) {
            appendChat(
                "assistant",
                "Blocked apply: model output looks partial and would overwrite file. Please ask AI to return the complete file."
            )
            return
        }
        _generatedCode.value = proposedCode
        val extLanguage = languageFromFileName(_currentFileName.value)
        _codeLanguage.value = if (extLanguage != CodeLanguage.UNKNOWN) extLanguage else proposedLanguage
        _isDirty.value = true
        val changed = countChangedLines(before, proposedCode)
        _editCheckpoints.value = (listOf(
            EditCheckpoint(
                prompt = prompt,
                promptMessageId = promptMessageId,
                beforeCode = before,
                afterCode = proposedCode,
                changedLines = changed
            )
        ) + _editCheckpoints.value).take(30)
        _pendingProposal.value = null
        saveSettings()
        appendChat("assistant", "Applied AI edit automatically. Use Discard to revert.")
        persistActiveSession()
    }

    fun discardPendingProposal() {
        if (_pendingProposal.value != null) {
            _pendingProposal.value = null
            appendChat("assistant", "Discarded proposed changes.")
        }
    }

    fun revertLastCheckpoint() {
        val cp = _editCheckpoints.value.firstOrNull() ?: return
        _generatedCode.value = cp.beforeCode
        _isDirty.value = true
        _pendingProposal.value = null
        _editCheckpoints.value = _editCheckpoints.value.drop(1)
        saveSettings()
        appendChat("assistant", "Reverted last edit (${cp.changedLines} changed lines).")
        persistActiveSession()
    }

    fun resendLastPrompt() {
        val last = _lastUserPrompt.value ?: return
        generateCode(last)
    }

    fun editAndResendFromPrompt(promptMessageId: String, newPrompt: String) {
        val edited = newPrompt.trim()
        if (edited.isBlank()) return

        val currentMessages = _chatMessages.value
        val promptIndex = currentMessages.indexOfFirst { it.id == promptMessageId && it.role == "user" }
        if (promptIndex < 0) return

        // Remove the selected prompt and everything after it.
        // The edited prompt will be re-added as a fresh user message by generateCode().
        _chatMessages.value = currentMessages.take(promptIndex)

        _pendingProposal.value = null
        streamingAssistantMessageId = null

        val idx = _editCheckpoints.value.indexOfFirst { it.promptMessageId == promptMessageId }
        if (idx >= 0) {
            val checkpoint = _editCheckpoints.value[idx]
            _generatedCode.value = checkpoint.beforeCode
            _isDirty.value = true
            _pendingProposal.value = null
            _editCheckpoints.value = _editCheckpoints.value.drop(idx + 1)
            appendChat("assistant", "Branched from selected prompt checkpoint and regenerated.")
            saveSettings()
        }

        persistActiveSession()
        generateCode(edited)
    }

    private fun languageFromFileName(fileName: String?): CodeLanguage {
        val n = fileName?.lowercase() ?: return CodeLanguage.UNKNOWN
        return when {
            n.contains(".py") -> CodeLanguage.PYTHON
            n.contains(".js") || n.contains(".ts") -> CodeLanguage.JAVASCRIPT
            n.contains(".html") || n.contains(".htm") || n.contains(".css") -> CodeLanguage.HTML
            else -> CodeLanguage.UNKNOWN
        }
    }

    private fun languagePromptConfig(): Triple<String, String, String>? {
        val n = _currentFileName.value?.lowercase().orEmpty()
        val selectedLanguage: ProgrammingLanguage? = when {
            n.contains(".py") -> ProgrammingLanguage.PYTHON
            n.contains(".js") -> ProgrammingLanguage.JAVASCRIPT
            n.contains(".ts") -> ProgrammingLanguage.TYPESCRIPT
            n.contains(".java") -> ProgrammingLanguage.JAVA
            n.contains(".kt") -> ProgrammingLanguage.KOTLIN
            n.contains(".cs") -> ProgrammingLanguage.CSHARP
            n.contains(".cpp") || n.contains(".cc") || n.contains(".cxx") -> ProgrammingLanguage.CPP
            n.contains(".go") -> ProgrammingLanguage.GO
            n.contains(".rs") -> ProgrammingLanguage.RUST
            n.contains(".html") || n.contains(".htm") || n.contains(".css") -> ProgrammingLanguage.WEB
            else -> null
        }
        if (selectedLanguage == null) return null
        return when (selectedLanguage) {
            ProgrammingLanguage.WEB -> Triple(
                "Web App (HTML/CSS/JS)",
                "Build a single self-contained HTML file with embedded CSS and JavaScript.",
                "html"
            )
            ProgrammingLanguage.PYTHON -> Triple("Python", "Build a runnable Python script using only standard library.", "python")
            ProgrammingLanguage.JAVASCRIPT -> Triple("JavaScript", "Build a runnable JavaScript program (no TypeScript).", "javascript")
            ProgrammingLanguage.TYPESCRIPT -> Triple("TypeScript", "Build a runnable TypeScript program with clear types.", "typescript")
            ProgrammingLanguage.JAVA -> Triple("Java", "Build a runnable Java program with a main method.", "java")
            ProgrammingLanguage.KOTLIN -> Triple("Kotlin", "Build a runnable Kotlin console program with a main function.", "kotlin")
            ProgrammingLanguage.CSHARP -> Triple("C#", "Build a runnable C# console app entry point.", "csharp")
            ProgrammingLanguage.CPP -> Triple("C++", "Build a runnable modern C++ program (C++17 style).", "cpp")
            ProgrammingLanguage.GO -> Triple("Go", "Build a runnable Go program with package main and func main().", "go")
            ProgrammingLanguage.RUST -> Triple("Rust", "Build a runnable Rust program with fn main().", "rust")
        }
    }

    private fun applyGenerationParametersToService(
        maxTokens: Int? = null,
        topK: Int? = null,
        topP: Float? = null,
        temperature: Float? = null
    ) {
        val model = _selectedModel.value
        val effectiveMaxTokens = when {
            maxTokens != null && model != null -> maxTokens.coerceIn(1, model.contextWindowSize.coerceAtLeast(1))
            maxTokens != null -> maxTokens
            model != null -> _selectedMaxTokens.value.coerceIn(1, model.contextWindowSize.coerceAtLeast(1))
            else -> _selectedMaxTokens.value
        }

        inferenceService.setGenerationParameters(
            effectiveMaxTokens,
            topK,
            topP,
            temperature,
            _selectedNGpuLayers.value,
            _enableThinking.value
        )
    }
    
    fun loadModel() {
        val model = _selectedModel.value ?: return
        val backend = _selectedBackend.value ?: return
        
        if (_isLoading.value || _isModelLoaded.value) {
            return
        }
        
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            
            try {
                inferenceService.unloadModel()
                applyGenerationParametersToService()

                // Load model with text-only mode (vibe coder generates code as text)
                val success = inferenceService.loadModel(
                    model = model,
                    preferredBackend = backend,
                    disableVision = true,
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
    
    /**
     * Unload the current model from memory
     */
    fun unloadModel() {
        viewModelScope.launch {
            try {
                cancelGenerationInternal()
                inferenceService.unloadModel()
                _isModelLoaded.value = false
                _generatedCode.value = ""
            } catch (e: Exception) {
                _errorMessage.value = e.message ?: "Failed to unload model"
            }
        }
    }
    
    /**
     * Update the prompt input text
     */
    fun updatePromptInput(text: String) {
        _promptInput.value = text
    }
    
    /**
     * Generate code based on the user's prompt
     */
    fun generateCode(prompt: String) {
        if (prompt.isBlank()) return
        val model = _selectedModel.value ?: return
        
        if (!_isModelLoaded.value) {
            _errorMessage.value = "Please load a model first"
            return
        }
        if (_currentFileName.value.isNullOrBlank()) {
            _errorMessage.value = "Create or open a file first (e.g. .py, .js, .ts)"
            return
        }
        if (languagePromptConfig() == null) {
            _errorMessage.value = "Unsupported or unknown file extension. Use a code file like .py, .js, .ts, .java, .kt, .go, .rs, .cpp, .cs, .html"
            return
        }
        val normalizedPrompt = prompt.trim()
        val userMsg = VibeChatMessage(role = "user", text = normalizedPrompt)
        _chatMessages.value = _chatMessages.value + userMsg
        currentPromptMessageId = userMsg.id
        _lastUserPrompt.value = normalizedPrompt
        persistActiveSession()
        
        processingJob?.cancel()
        
        processingJob = viewModelScope.launch {
            _isProcessing.value = true
            _errorMessage.value = null
            val currentCode = _generatedCode.value
            
            try {
                val coderMaxTokens = _selectedMaxTokens.value.coerceAtLeast(512)
                applyGenerationParametersToService(
                    maxTokens = coderMaxTokens,
                    topK = 40,
                    topP = 0.95f,
                    temperature = 0.2f
                )
                val implementationPrompt = buildFileAwareEditPrompt(prompt, currentCode)
                
                val codeChatId = "vibe-coder-${UUID.randomUUID()}"
                beginStreamingAssistant()
                
                val responseFlow = inferenceService.generateResponseStreamWithSession(
                    prompt = implementationPrompt,
                    model = model,
                    chatId = codeChatId,
                    images = emptyList(),
                    audioData = null,
                    webSearchEnabled = false
                )
                
                var responseText = ""
                responseFlow.collect { token ->
                    responseText += token
                    updateStreamingAssistant(responseText)
                }
                
                // Parse generated code and auto-apply immediately.
                val (proposedCode, proposedLanguage) = extractCodeAndLanguage(responseText)
                if (proposedCode.isNotBlank()) {
                    val changedLines = countChangedLines(currentCode, proposedCode)
                    applyAutoProposal(prompt, currentPromptMessageId, proposedCode, proposedLanguage)
                    val chatLang = when (proposedLanguage) {
                        CodeLanguage.HTML -> "html"
                        CodeLanguage.PYTHON -> "python"
                        CodeLanguage.JAVASCRIPT -> "javascript"
                        CodeLanguage.UNKNOWN -> ""
                    }
                    val fencedCode = if (chatLang.isNotBlank()) {
                        "```$chatLang\n$proposedCode\n```"
                    } else {
                        "```\n$proposedCode\n```"
                    }
                    endStreamingAssistant(
                        "Updated `${_currentFileName.value}`.\nChanged lines: $changedLines\n\nModel output:\n$fencedCode"
                    )
                } else {
                    endStreamingAssistant("No usable code was produced. Try refining your prompt.")
                }
                
            } catch (e: kotlinx.coroutines.CancellationException) {
                Log.d("VibeCoderVM", "Generation cancelled")
                endStreamingAssistant("Generation cancelled.")
            } catch (e: Exception) {
                val message = e.message ?: ""
                val shouldShowError = !message.contains("cancelled", ignoreCase = true) &&
                                    !message.contains("Previous invocation still processing", ignoreCase = true) &&
                                    !message.contains("StandaloneCoroutine", ignoreCase = true)
                
                if (shouldShowError) {
                    _errorMessage.value = message.ifBlank { "Generation failed" }
                    endStreamingAssistant("Generation failed: ${_errorMessage.value}")
                    Log.e("VibeCoderVM", "Generation error: $message", e)
                } else {
                    Log.d("VibeCoderVM", "Suppressed error: $message")
                }
            } finally {
                // Reset parameters to defaults (null)
                inferenceService.setGenerationParameters(null, null, null, null)
                _isProcessing.value = false
                _isPlanning.value = false
                processingJob = null
            }
        }
    }

    private fun countChangedLines(oldCode: String, newCode: String): Int {
        val a = oldCode.lines()
        val b = newCode.lines()
        val max = maxOf(a.size, b.size)
        var changed = 0
        for (i in 0 until max) {
            val av = a.getOrNull(i).orEmpty()
            val bv = b.getOrNull(i).orEmpty()
            if (av != bv) changed++
        }
        return changed
    }

    private fun isSafeFullFileUpdate(currentCode: String, proposedCode: String): Boolean {
        if (currentCode.isBlank()) return proposedCode.isNotBlank()
        if (proposedCode.isBlank()) return false
        val currLen = currentCode.trim().length
        val nextLen = proposedCode.trim().length
        if (currLen < 200) return true
        val ratio = nextLen.toDouble() / currLen.toDouble()
        if (ratio >= 0.60) return true

        val n = (_currentFileName.value ?: "").lowercase()
        return when {
            n.endsWith(".html") || n.endsWith(".htm") ->
                proposedCode.contains("<html", true) || proposedCode.contains("<!doctype", true)
            n.endsWith(".py") ->
                proposedCode.contains("def ") || proposedCode.contains("class ") || proposedCode.contains("import ")
            n.endsWith(".js") || n.endsWith(".ts") ->
                proposedCode.contains("function ") || proposedCode.contains("const ") || proposedCode.contains("let ") || proposedCode.contains("class ")
            else -> false
        }
    }
    
    /**
     * Detect code language and extract clean code from the response.
     * Supports HTML, Python, JavaScript wrapped in markdown code blocks or XML tags.
     * Handles edge cases where code block markers aren't perfectly formatted.
     */
    private fun extractCodeAndLanguage(response: String): Pair<String, CodeLanguage> {
        val markerMatch = Regex("(?s)<<<FULL_FILE_START>>>\\s*(.*?)\\s*<<<FULL_FILE_END>>>").find(response)
        if (markerMatch != null) {
            val extracted = sanitizeExtractedCode(markerMatch.groupValues[1].trim())
            val lang = detectLanguageFromContent(extracted)
            return Pair(extracted, lang)
        }

        // Try to extract from markdown code blocks with language hints (```html, ```python, etc.)
        // Relaxed regex to allow immediate content after language tag (no newline required)
        val htmlMatch = Regex("```(?:html|htm)\\s*([\\s\\S]*?)```", RegexOption.IGNORE_CASE)
            .findAll(response).maxByOrNull { it.groupValues[1].length }
        if (htmlMatch != null) {
            return Pair(sanitizeExtractedCode(htmlMatch.groupValues[1].trim()), CodeLanguage.HTML)
        }
        
        val pythonMatch = Regex("```(?:python|py)\\s*([\\s\\S]*?)```", RegexOption.IGNORE_CASE)
            .findAll(response).maxByOrNull { it.groupValues[1].length }
        if (pythonMatch != null) {
            return Pair(sanitizeExtractedCode(pythonMatch.groupValues[1].trim()), CodeLanguage.PYTHON)
        }
        
        val jsMatch = Regex("```(?:javascript|js)\\s*([\\s\\S]*?)```", RegexOption.IGNORE_CASE)
            .findAll(response).maxByOrNull { it.groupValues[1].length }
        if (jsMatch != null) {
            return Pair(sanitizeExtractedCode(jsMatch.groupValues[1].trim()), CodeLanguage.JAVASCRIPT)
        }
        
        // Fallback: Extract any content between ``` markers (handles malformed responses)
        val genericMatch = Regex("```\\s*([\\s\\S]*?)```").find(response)
        if (genericMatch != null) {
            val extracted = sanitizeExtractedCode(genericMatch.groupValues[1].trim())
            // Detect language from content
            return Pair(extracted, detectLanguageFromContent(extracted))
        }
        
        // If no code block is found, assume the entire response is code if it loosely fits a pattern
        val isLikelyCode = response.contains("<!DOCTYPE", ignoreCase = true) || 
                           response.contains("<html", ignoreCase = true) || 
                           response.contains("def ") || 
                           response.contains("function ")
        
        if (isLikelyCode && !response.contains("```")) {
            val extracted = sanitizeExtractedCode(response.trim())
            return Pair(extracted, detectLanguageFromContent(extracted))
        }
        
        // Try to extract from XML-like tags (fallback)
        val xmlHtmlMatch = Regex("<code[^>]*>([\\s\\S]*?)</code>", RegexOption.IGNORE_CASE).find(response)
        if (xmlHtmlMatch != null) {
            val extracted = sanitizeExtractedCode(xmlHtmlMatch.groupValues[1].trim())
            return Pair(extracted, CodeLanguage.HTML)
        }

        val cleaned = sanitizeExtractedCode(response.trim())
        return Pair(cleaned, detectLanguageFromContent(cleaned))
    }

    private fun sanitizeExtractedCode(raw: String): String {
        val lines = raw.lines()
        val startIndex = lines.indexOfFirst { line ->
            val t = line.trimStart()
            t.startsWith("<") ||
                t.startsWith("#!") ||
                t.startsWith("import ") ||
                t.startsWith("from ") ||
                t.startsWith("def ") ||
                t.startsWith("class ") ||
                t.startsWith("function ") ||
                t.startsWith("const ") ||
                t.startsWith("let ") ||
                t.startsWith("var ")
        }
        val trimmed = if (startIndex > 0) lines.drop(startIndex).joinToString("\n") else raw
        return trimmed
            .replace(Regex("(?m)^\\s*File\\s*:.*$"), "")
            .replace(Regex("(?m)^\\s*TARGET\\s+LANGUAGE\\s*:.*$"), "")
            .replace(Regex("(?m)^\\s*TARGET\\s+RULE\\s*:.*$"), "")
            .trim()
    }

    private fun detectLanguageFromContent(content: String): CodeLanguage {
        return when {
            content.contains("<!DOCTYPE html", ignoreCase = true) || content.contains("<html", ignoreCase = true) -> CodeLanguage.HTML
            content.contains("def ", ignoreCase = true) || content.contains("import ", ignoreCase = true) -> CodeLanguage.PYTHON
            content.contains("function ", ignoreCase = true) || content.contains("const ", ignoreCase = true) || content.contains("var ", ignoreCase = true) -> CodeLanguage.JAVASCRIPT
            else -> CodeLanguage.UNKNOWN
        }
    }
    
    /**
     * Cancel ongoing code generation
     */
    fun cancelGeneration() {
        viewModelScope.launch {
            cancelGenerationInternal()
        }
    }

    /**
     * Safe cleanup path when leaving Vibe screen:
     * stop streaming generation first, then unload model.
     */
    fun stopAndUnloadOnExit() {
        viewModelScope.launch {
            try {
                cancelGenerationInternal()
                inferenceService.unloadModel()
                _isModelLoaded.value = false
            } catch (e: Exception) {
                Log.w("VibeCoderVM", "stopAndUnloadOnExit failed: ${e.message}")
            }
        }
    }

    private suspend fun cancelGenerationInternal() {
        val activeJob = processingJob
        if (activeJob != null) {
            activeJob.cancel()
            try {
                activeJob.cancelAndJoin()
            } catch (_: Exception) {
            }
            processingJob = null
        }
        _isProcessing.value = false
        _isPlanning.value = false
        try {
            inferenceService.setGenerationParameters(null, null, null, null)
        } catch (_: Exception) {
        }
    }
    
    /**
     * Clear generated code
     */
    fun clearCode() {
        _generatedCode.value = ""
        _codeLanguage.value = CodeLanguage.UNKNOWN
        _isDirty.value = true
        _pendingProposal.value = null
        currentSpec = ""
    }

    /**
     * Update generated code (user edits)
     */
    fun updateGeneratedCode(code: String) {
        _generatedCode.value = code
        _isDirty.value = true
        saveSettings()
    }

    /**
     * Clear error message
     */
    fun clearError() {
        _errorMessage.value = null
    }

    fun setError(message: String) {
        _errorMessage.value = message
    }
    
    /**
     * Build the Architect Spec Prompt (Step 1)
     * Simplified to a "Technical Assistant" role that generates a concise Requirements List.
     */
    private fun buildSpecPrompt(userRequest: String, currentCode: String): String {
        val isRevision = currentCode.isNotBlank()
        return """
            You are a helpful Technical Assistant.
            Your goal is to expand the user's request into a clear, concise list of functional requirements.

            CONTEXT:
            ${if (isRevision) "The user wants to MODIFY this existing code:\n$currentCode" else "This is a NEW project request."}

            USER REQUEST: "$userRequest"

            TASK:
            1. Identify the core features needed.
            2. List specific UI elements required (buttons, inputs, displays).
            3. Define the basic logic flow (e.g., "User clicks -> Update Score").
            4. Keep it brief and actionable.

            OUTPUT FORMAT:
            - Feature: [Description]
            - UI: [Element]
            - Logic: [Rule]

            Output ONLY the list. Do not write code or introductions.
            IMPORTANT: Respond in the same language as the user's request.
        """.trimIndent()
    }

    /**
     * Build the Developer Implementation Prompt (Step 2)
     */
    /**
     * Build the Developer Implementation Prompt (Step 2 - New Project)
     */
    private fun buildImplementationPrompt(requirements: String): String {
        val config = languagePromptConfig()
            ?: throw IllegalStateException("Unsupported file extension for code generation")
        val (languageName, targetRule, fenceLanguage) = config
        return """
            You are a senior software engineer.
            Produce exactly what the user asked for as working code.
            REQUIREMENTS:
            $requirements

            TARGET LANGUAGE: $languageName
            TARGET RULE: $targetRule

            RULES:
            - Return only code in one markdown code block fenced as ```$fenceLanguage
            - No explanations.
            - Language in UI/messages must match user language.
            - Include reset behavior where stateful interactions exist.
        """.trimIndent()
    }

    /**
     * Build the Developer Modification Prompt (Step 2 - Revision)
     * Direct Code Modification skipping the Architect.
     */
    private fun buildModificationPrompt(userRequest: String, currentCode: String): String {
        val config = languagePromptConfig()
            ?: throw IllegalStateException("Unsupported file extension for code generation")
        val (languageName, targetRule, fenceLanguage) = config
        // Keep prompt size bounded to avoid context overflow during refine mode.
        val maxChars = 12_000
        val trimmedCode = if (currentCode.length > maxChars) {
            currentCode.take(maxChars) + "\n\n/* ... truncated for prompt size ... */"
        } else {
            currentCode
        }
        return """
            You are a senior software engineer.
            Rewrite the full code to satisfy the user's modification request.
            EXISTING CODE:
            ```
            $trimmedCode
            ```
            
            USER REQUEST: "$userRequest"

            TARGET LANGUAGE: $languageName
            TARGET RULE: $targetRule

            RULES:
            - Return the full updated code, not a diff.
            - Return only code in one markdown code block fenced as ```$fenceLanguage
            - No explanations.
            - Preserve existing behavior unless user asked to change it.
            - Keep text/output language aligned with user request.
            - If input code was truncated, infer missing parts conservatively and output a complete working file.
        """.trimIndent()
    }

    /**
     * Legacy Prompt Builder (Fallback for v0.4 behavior)
     * Used when Planning Phase fails or times out.
     */
    private fun buildPrompt(userPrompt: String): String {
        val config = languagePromptConfig()
            ?: throw IllegalStateException("Unsupported file extension for code generation")
        val (languageName, targetRule, fenceLanguage) = config
        return """
            You are a senior software engineer.
            Generate code that directly fulfills the user's request.

            User request: $userPrompt

            TARGET LANGUAGE: $languageName
            TARGET RULE: $targetRule

            RULES:
            - Return only code in one markdown code block fenced as ```$fenceLanguage
            - No explanations.
            - Keep UI/messages in the user's language.
            - For interactive apps, include clear state and reset behavior.
        """.trimIndent()
    }

    private fun buildFileAwareEditPrompt(userPrompt: String, currentCode: String): String {
        val config = languagePromptConfig()
            ?: throw IllegalStateException("Unsupported file extension for code generation")
        val (languageName, targetRule, fenceLanguage) = config
        val fileName = _currentFileName.value ?: "untitled"
        val codeSection = if (currentCode.isBlank()) {
            "FILE_IS_EMPTY"
        } else {
            currentCode.take(20_000)
        }
        return """
            You are an expert coding assistant working on a real file.
            FILE: $fileName
            TARGET LANGUAGE: $languageName
            TARGET RULE: $targetRule

            USER REQUEST:
            $userPrompt

            CURRENT FILE CONTENT:
            ```$fenceLanguage
            $codeSection
            ```

            INSTRUCTIONS:
            - Produce the FULL updated file content.
            - Do not return partial snippets or patch hunks.
            - Wrap the final full file between markers:
              <<<FULL_FILE_START>>>
              [full file content]
              <<<FULL_FILE_END>>>
            - Respect the FILE extension/language exactly.
            - If file is empty, create a complete starter implementation for this request.
            - Do not output explanations.
            - Output only one fenced code block using ```$fenceLanguage.
        """.trimIndent()
    }
    
    override fun onCleared() {
        super.onCleared()
        viewModelScope.launch {
            try { inferenceService.unloadModel() } catch (_: Exception) {}
            inferenceService.onCleared()
        }
    }
}
