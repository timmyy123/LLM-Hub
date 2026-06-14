package com.llmhub.llmhub.inference

import android.content.Context
import android.content.res.Configuration
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ToolProvider
import com.google.ai.edge.litertlm.tool
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.utils.KidModeManager
import com.llmhub.llmhub.utils.LocaleHelper
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.SearchIntentDetector
import com.llmhub.llmhub.websearch.WebSearchService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * LiteRT-LM based inference service for .litertlm format models (Gemma-3n, Gemma-4, etc.).
 *
 * Uses the Engine / Conversation API from litertlm-android, which is the recommended
 * replacement for tasks-genai for LiteRT model files.
 *
 * A fresh Conversation is created per generation call (stateless sessions) so that the
 * full conversation history managed by ChatViewModel is re-processed each turn.
 * This avoids all the "Previous invocation still processing" and DetokenizerCalculator
 * issues inherent to the old tasks-genai session model.
 *
 * GPU NOTE: litertlm-android 0.10.0 has a known GPU decode crash on Android
 * (missing libLiteRtTopKOpenClSampler.so). The fix ships in 0.10.1 (not yet on Maven
 * as of April 2026). GPU is currently forced to CPU — see mapBackend(). Remove the
 * workaround and uncomment the proper mapping once 0.10.1 is available on Maven.
 */
class LiteRtLmInferenceService(private val applicationContext: Context) : InferenceService {

    private var engine: Engine? = null
    private var currentModel: LLMModel? = null
    private var isVisionDisabled: Boolean = false
    private var isAudioDisabled: Boolean = false
    private var currentBackendIsGpu: Boolean = false

    // Generation parameters from UI
    // NOTE: contextWindow is stored but not forwarded to the Engine — the LiteRT-LM API
    // manages KV cache automatically. topK/topP/temperature are applied per Conversation.
    private var overrideMaxTokens: Int? = null
    private var overrideContextWindow: Int? = null
    private var overrideTopK: Int? = null
    private var overrideTopP: Float? = null
    private var overrideTemperature: Float? = null
    private var overrideEnableThinking: Boolean = true

    private val engineMutex = Mutex()
    private val sessionResetTimes = mutableMapOf<String, Long>()
    private val webSearchService: WebSearchService = DuckDuckGoSearchService()

    // Agent skills tools — set by UnifiedInferenceService when a Gemma-4 model is loaded.
    // null = no tools (default for all other models).
    private var agentTools: ChatAgentSkillsTools? = null

    fun setAgentTools(tools: ChatAgentSkillsTools?) {
        agentTools = tools
        Log.d(TAG, if (tools != null) "Agent tools enabled" else "Agent tools disabled")
    }

    companion object {
        private const val TAG = "LiteRtLmInference"
        private const val DEFAULT_TOP_K = 40
        private const val DEFAULT_TOP_P = 0.95f
        private const val DEFAULT_TEMPERATURE = 1.0f

        /** Sentinel markers used by ChatComponents.kt to render expandable thinking panel. */
        const val SENTINEL_THINK = "\u200B\u200BTHINK\u200B\u200B"
        const val SENTINEL_ENDTHINK = "\u200B\u200BENDTHINK\u200B\u200B"

        @JvmStatic
        fun getMaxTokensForModelStatic(model: LLMModel): Int = model.contextWindowSize
    }

    /**
     * Map MediaPipe backend enum → LiteRT-LM Backend sealed class.
     *
     * TODO: Remove CPU-only workaround once litertlm-android 0.10.1 is on Maven and
     *  the OpenCL sampler GPU decode crash is fixed. Then restore the commented block.
     */
    private fun mapBackend(preferredBackend: LlmInference.Backend?, supportsGpu: Boolean): Backend {
        return when (preferredBackend) {
            LlmInference.Backend.GPU -> Backend.GPU()
            LlmInference.Backend.CPU -> Backend.CPU()
            null -> if (supportsGpu) Backend.GPU() else Backend.CPU()
            else -> Backend.CPU()
        }
    }

    override fun getEffectiveMaxTokens(model: LLMModel): Int {
        val isGemma4_12B = model.name.contains("Gemma-4 12B", ignoreCase = true) || model.name.contains("Gemma 4 12B", ignoreCase = true)
        val limit = if (isGemma4_12B) 4096 else model.contextWindowSize
        val contextWindow = overrideContextWindow?.coerceIn(1, limit)
            ?: limit
        return overrideMaxTokens?.coerceIn(1, contextWindow) ?: contextWindow
    }

    override suspend fun loadModel(
        model: LLMModel,
        preferredBackend: LlmInference.Backend?,
        deviceId: String?
    ): Boolean = loadModel(model, preferredBackend, disableVision = false, disableAudio = false, deviceId = deviceId)

    override suspend fun loadModel(
        model: LLMModel,
        preferredBackend: LlmInference.Backend?,
        disableVision: Boolean,
        disableAudio: Boolean,
        deviceId: String?
    ): Boolean {
        return try {
            ensureEngineLoaded(model, preferredBackend, disableVision, disableAudio)
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model ${model.name}: ${e.message}", e)
            false
        }
    }

    private suspend fun resolveModelFile(model: LLMModel): File {
        if (model.source == "Custom" && model.url.startsWith("content://")) {
            val target = File(applicationContext.filesDir, "models/${model.localFileName()}")
            target.parentFile?.mkdirs()
            if (!target.exists()) {
                applicationContext.contentResolver.openInputStream(Uri.parse(model.url))?.use { input ->
                    target.outputStream().use { input.copyTo(it) }
                } ?: throw IllegalStateException("Cannot open URI: ${model.url}")
            }
            return target
        }

        val assetPath = if (model.url.startsWith("file://models/")) {
            model.url.removePrefix("file://")
        } else {
            "models/${model.localFileName()}"
        }

        return try {
            applicationContext.assets.open(assetPath).use { _ ->
                val target = File(applicationContext.filesDir, "models/${model.localFileName()}")
                target.parentFile?.mkdirs()
                if (!target.exists()) {
                    target.outputStream().use { out ->
                        applicationContext.assets.open(assetPath).use { it.copyTo(out) }
                    }
                }
                target
            }
        } catch (e: Exception) {
            val f = File(applicationContext.filesDir, assetPath)
            if (f.exists()) f
            else throw IllegalStateException("Model not found: $assetPath")
        }
    }

    @OptIn(ExperimentalApi::class)
    private suspend fun ensureEngineLoaded(
        model: LLMModel,
        preferredBackend: LlmInference.Backend? = null,
        disableVision: Boolean = false,
        disableAudio: Boolean = false
    ) {
        engineMutex.withLock {
            if (currentModel?.name == model.name && engine != null) return@withLock

            engine?.let {
                try { it.close() } catch (e: Exception) { Log.w(TAG, "Error closing engine: ${e.message}") }
            }
            engine = null

            val modelFile = withContext(Dispatchers.IO) { resolveModelFile(model) }
            isVisionDisabled = disableVision
            isAudioDisabled = disableAudio

            val isGemma4_12B = model.name.contains("Gemma-4 12B", ignoreCase = true) || model.name.contains("Gemma 4 12B", ignoreCase = true)

            val backend = if (isGemma4_12B) Backend.GPU() else mapBackend(preferredBackend, model.supportsGpu)
            currentBackendIsGpu = backend is Backend.GPU

            // Enable Multi-Token Prediction (MTP) via speculative decoding when running on GPU, except for Gemma-4 12B
            ExperimentalFlags.enableSpeculativeDecoding = currentBackendIsGpu && !isGemma4_12B

            val engineConfig = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = backend,
                visionBackend = if (model.supportsVision && !disableVision && !isGemma4_12B) backend else null,
                audioBackend = if (model.supportsAudio && !disableAudio && !isGemma4_12B) Backend.CPU() else null,
                maxNumTokens = if (isGemma4_12B) getEffectiveMaxTokens(model) else null,
                cacheDir = applicationContext.cacheDir.path
            )

            Log.d(TAG, "Initializing engine for ${model.name} | backend=$backend | vision=${model.supportsVision && !disableVision} | audio=${model.supportsAudio && !disableAudio}")

            val newEngine = Engine(engineConfig)
            withContext(Dispatchers.IO) { newEngine.initialize() }
            engine = newEngine
            currentModel = model

            Log.d(TAG, "Engine ready for ${model.name}")
        }
    }

    override suspend fun unloadModel() {
        engineMutex.withLock {
            engine?.let {
                try { it.close() } catch (e: Exception) { Log.w(TAG, "Error unloading engine: ${e.message}") }
            }
            engine = null
            currentModel = null
            Log.d(TAG, "Engine unloaded")
        }
    }

    override suspend fun resetChatSession(chatId: String) {
        // With fresh-Conversation-per-generation there is no persistent session state to clear.
        recordSessionReset(chatId)
        Log.d(TAG, "Session reset recorded for chat $chatId")
    }

    override suspend fun onCleared() {
        withContext(Dispatchers.IO) {
            engineMutex.withLock {
                engine?.let {
                    try { it.close() } catch (e: Exception) { Log.w(TAG, "Error on onCleared: ${e.message}") }
                }
                engine = null
                currentModel = null
            }
        }
    }

    override fun getCurrentlyLoadedModel(): LLMModel? = currentModel

    override fun getCurrentlyLoadedBackend(): LlmInference.Backend? {
        if (currentModel == null) return null
        return if (currentBackendIsGpu) LlmInference.Backend.GPU else LlmInference.Backend.CPU
    }

    override fun isVisionCurrentlyDisabled(): Boolean = isVisionDisabled
    override fun isAudioCurrentlyDisabled(): Boolean = isAudioDisabled
    override fun isGpuBackendEnabled(): Boolean = currentBackendIsGpu

    override fun setGenerationParameters(
        maxTokens: Int?,
        topK: Int?,
        topP: Float?,
        temperature: Float?,
        nGpuLayers: Int?,
        enableThinking: Boolean?,
        contextWindow: Int?
    ) {
        overrideMaxTokens = maxTokens
        overrideContextWindow = contextWindow
        overrideTopK = topK
        overrideTopP = topP
        overrideTemperature = temperature
        if (enableThinking != null) overrideEnableThinking = enableThinking
        Log.d(TAG, "Generation params: contextWindow=$contextWindow maxTokens=$maxTokens topK=$topK topP=$topP temperature=$temperature enableThinking=$enableThinking")
    }

    override fun getMemoryWarningForImages(images: List<Bitmap>): String? = null

    override fun wasSessionRecentlyReset(chatId: String): Boolean {
        val t = sessionResetTimes[chatId] ?: return false
        return (System.currentTimeMillis() - t) < 10_000
    }

    private fun recordSessionReset(chatId: String) {
        sessionResetTimes[chatId] = System.currentTimeMillis()
    }

    private fun isGemma4Model(): Boolean =
        currentModel?.name?.contains("Gemma-4", ignoreCase = true) == true

    private fun buildConversationConfig(): ConversationConfig {
        val sysInstruction = if (isGemma4Model() && overrideEnableThinking) Contents.of("<|think|>") else null
        return ConversationConfig(
            samplerConfig = SamplerConfig(
                topK = overrideTopK ?: DEFAULT_TOP_K,
                topP = (overrideTopP ?: DEFAULT_TOP_P).toDouble(),
                temperature = (overrideTemperature ?: DEFAULT_TEMPERATURE).toDouble()
            ),
            systemInstruction = sysInstruction
        )
    }

    /**
     * Build a ConversationConfig wired with agent tools and system instruction.
     * Call only when [agentTools] is non-null (Gemma-4 loaded).
     */
    @OptIn(ExperimentalApi::class)
    private fun buildAgentConversationConfig(tools: ChatAgentSkillsTools): ConversationConfig {
        val toolProviders: List<ToolProvider> = listOf(tool(tools))
        val sysText = if (isGemma4Model() && overrideEnableThinking) "<|think|>\n${ChatAgentSkillsTools.AGENT_SYSTEM_PROMPT}" else ChatAgentSkillsTools.AGENT_SYSTEM_PROMPT
        val sysInstruction = Contents.of(sysText)
        return ConversationConfig(
            samplerConfig = SamplerConfig(
                topK = overrideTopK ?: DEFAULT_TOP_K,
                topP = (overrideTopP ?: DEFAULT_TOP_P).toDouble(),
                temperature = (overrideTemperature ?: DEFAULT_TEMPERATURE).toDouble()
            ),
            systemInstruction = sysInstruction,
            tools = toolProviders
        )
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        ensureEngineLoaded(model)
        val eng = engine ?: throw IllegalStateException("No engine loaded")
        return withContext(Dispatchers.IO) {
            val sb = StringBuilder()
            eng.createConversation(buildConversationConfig()).use { conv ->
                conv.sendMessageAsync(prompt).collect { msg ->
                    sb.append(msg.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text })
                }
            }
            sb.toString()
        }
    }

    override suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String> {
        ensureEngineLoaded(model)
        val eng = engine ?: throw IllegalStateException("No engine loaded")
        return flow {
            eng.createConversation(buildConversationConfig()).use { conv ->
                conv.sendMessageAsync(prompt).collect { msg ->
                    val chunk = msg.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }
                    if (chunk.isNotEmpty()) emit(chunk)
                }
            }
        }.flowOn(Dispatchers.IO)
    }

    override suspend fun generateResponseStreamWithSession(
        prompt: String,
        model: LLMModel,
        chatId: String,
        images: List<Bitmap>,
        audioData: ByteArray?,
        webSearchEnabled: Boolean,
        imagePaths: List<String>
    ): Flow<String> {
        ensureEngineLoaded(model)
        val eng = engine ?: throw IllegalStateException("No engine loaded")

        val localCtx = run {
            val locale = LocaleHelper.getCurrentLocale(applicationContext)
            val cfg = Configuration(applicationContext.resources.configuration)
            cfg.setLocale(locale)
            applicationContext.createConfigurationContext(cfg)
        }

        return flow {
            val currentUserMessage = extractCurrentUserMessage(prompt)
            val needsWebSearch = webSearchEnabled
            var enhancedPrompt = prompt

            if (needsWebSearch) {
                emit(localCtx.getString(R.string.web_searching))
                try {
                    val query = SearchIntentDetector.extractSearchQuery(currentUserMessage)
                    val results = webSearchService.search(query, maxResults = 5)
                    if (results.isNotEmpty()) {
                        emit(localCtx.getString(R.string.web_search_found_results, results.size))
                        val resultsText = results.joinToString("\n\n") {
                            "SOURCE: ${it.source}\nTITLE: ${it.title}\nCONTENT: ${it.snippet}\n---"
                        }
                        enhancedPrompt = "CURRENT WEB SEARCH RESULTS:\n$resultsText\n\n" +
                            "Based on the above, answer: \"$currentUserMessage\"\n\n" +
                            "Answer directly and clearly:"
                    } else {
                        emit(localCtx.getString(R.string.web_search_no_results) + "\n\n")
                    }
                } catch (e: Exception) {
                    emit(localCtx.getString(R.string.web_search_failed, e.message ?: "Unknown error") + "\n\n")
                }
            }

            // Kid mode
            val kidModeManager = KidModeManager(applicationContext)
            if (kidModeManager.isKidModeEnabled.value) {
                val kidInstruction = KidModeManager.SYSTEM_INSTRUCTION
                enhancedPrompt = if (enhancedPrompt.startsWith("system:")) {
                    enhancedPrompt.replaceFirst("system:", "system: $kidInstruction\n\n")
                } else {
                    "system: $kidInstruction\n\n$enhancedPrompt"
                }
            }

            // Build content parts list (text first, then images, then audio — per API requirement)
            val contentParts = mutableListOf<Content>()
            if (enhancedPrompt.trim().isNotEmpty()) {
                contentParts.add(Content.Text(enhancedPrompt))
            }

            // Bitmaps: compress to JPEG and pass as ImageFile via temp file
            if (images.isNotEmpty() && model.supportsVision && !isVisionDisabled) {
                for ((i, bitmap) in images.withIndex()) {
                    try {
                        val tmpFile = File(applicationContext.cacheDir, "litert_img_${chatId}_$i.jpg")
                        tmpFile.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
                        contentParts.add(Content.ImageFile(tmpFile.absolutePath))
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to write image $i for chat $chatId: ${e.message}")
                    }
                }
            } else if (images.isNotEmpty() && (model.supportsVision == false || isVisionDisabled)) {
                Log.w(TAG, "Ignoring ${images.size} images: vision not supported or disabled")
            }

            // Image file paths
            if (imagePaths.isNotEmpty() && model.supportsVision && !isVisionDisabled) {
                for (path in imagePaths) contentParts.add(Content.ImageFile(path))
            }

            // Audio
            if (audioData != null && model.supportsAudio && !isAudioDisabled) {
                contentParts.add(Content.AudioBytes(audioData))
            }

            if (contentParts.isEmpty()) {
                Log.w(TAG, "No content parts for chat $chatId — skipping generation")
                return@flow
            }

            val contents = Contents.of(*contentParts.toTypedArray<Content>())

            val localAgentTools = agentTools
            val useThinking = isGemma4Model() && overrideEnableThinking
            @OptIn(ExperimentalApi::class)
            if (localAgentTools != null) {
                // Gemma-4 agent mode: enable constrained decoding for reliable function-call
                // extraction, then create conversation with tools + system instruction.
                ExperimentalFlags.enableConversationConstrainedDecoding = true
                val agentConfig = buildAgentConversationConfig(localAgentTools)
                ExperimentalFlags.enableConversationConstrainedDecoding = false
                Log.d(TAG, "Using agent conversation config with ${localAgentTools.javaClass.simpleName}")
                var sentThinkOpen = false
                var sentThinkClose = false
                eng.createConversation(agentConfig).use { conv ->
                    conv.sendMessageAsync(contents).collect { msg ->
                        val chunk = msg.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }
                        val thinkingChunk = if (useThinking) {
                            try { msg.channels?.get("thought") } catch (_: Exception) { null }
                        } else null
                        val isThinking = thinkingChunk != null && thinkingChunk.isNotEmpty()
                        if (useThinking && isThinking) {
                            if (!sentThinkOpen) { emit(SENTINEL_THINK); sentThinkOpen = true }
                            val (cleanedThinking, _) = processLlamaStopTokens(thinkingChunk!!)
                            if (cleanedThinking.isNotEmpty()) emit(cleanedThinking)
                        } else {
                            val (cleaned, _) = processLlamaStopTokens(chunk)
                            if (sentThinkOpen && !sentThinkClose) { emit(SENTINEL_ENDTHINK); sentThinkClose = true }
                            if (cleaned.isNotEmpty()) emit(cleaned)
                        }
                    }
                }
                if (useThinking && sentThinkOpen && !sentThinkClose) emit(SENTINEL_ENDTHINK)
            } else {
                var sentThinkOpen = false
                var sentThinkClose = false
                eng.createConversation(buildConversationConfig()).use { conv ->
                    conv.sendMessageAsync(contents).collect { msg ->
                        val chunk = msg.contents.contents.filterIsInstance<Content.Text>().joinToString("") { it.text }
                        val thinkingChunk = if (useThinking) {
                            try { msg.channels?.get("thought") } catch (_: Exception) { null }
                        } else null
                        val isThinking = thinkingChunk != null && thinkingChunk.isNotEmpty()
                        if (useThinking && isThinking) {
                            if (!sentThinkOpen) { emit(SENTINEL_THINK); sentThinkOpen = true }
                            val (cleanedThinking, _) = processLlamaStopTokens(thinkingChunk!!)
                            if (cleanedThinking.isNotEmpty()) emit(cleanedThinking)
                        } else {
                            val (cleaned, _) = processLlamaStopTokens(chunk)
                            if (sentThinkOpen && !sentThinkClose) { emit(SENTINEL_ENDTHINK); sentThinkClose = true }
                            if (cleaned.isNotEmpty()) emit(cleaned)
                        }
                    }
                }
                if (useThinking && sentThinkOpen && !sentThinkClose) emit(SENTINEL_ENDTHINK)
            }

        }.flowOn(Dispatchers.IO)
    }

    private fun extractCurrentUserMessage(prompt: String): String {
        val lines = prompt.trim().split('\n')
        for (i in lines.lastIndex downTo 0) {
            val line = lines[i].trim()
            if (line.startsWith("user:")) return line.removePrefix("user:").trim()
        }
        if (!prompt.contains("assistant:") && !prompt.contains("user:")) return prompt.trim()
        for (i in lines.lastIndex downTo 0) {
            val line = lines[i].trim()
            if (line.isNotEmpty() && !line.startsWith("assistant:")) return line
        }
        return prompt.trim()
    }

    private fun processLlamaStopTokens(text: String): Pair<String, Boolean> {
        val stopTokens = listOf("<|eot_id|>", "<|end_of_text|>", "<|end|>", "</s>")
        var cleaned = text
        var shouldStop = false
        for (stopToken in stopTokens) {
            if (cleaned.contains(stopToken)) {
                shouldStop = true
                cleaned = cleaned.substring(0, cleaned.indexOf(stopToken))
                break
            }
        }
        cleaned = cleaned
            .replace("<|start_header_id|>", "")
            .replace("<|end_header_id|>", "")
        return Pair(cleaned, shouldStop)
    }

}
