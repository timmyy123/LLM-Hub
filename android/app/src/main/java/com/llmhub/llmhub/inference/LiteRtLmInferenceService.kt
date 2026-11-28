package com.llmhub.llmhub.inference

import android.content.Context
import android.content.res.Configuration
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import com.google.ai.edge.litertlm.ToolProvider
import com.google.ai.edge.litertlm.tool
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.ModelData
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.utils.LocaleHelper
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.SearchIntentDetector
import com.llmhub.llmhub.websearch.WebSearchCitationStore
import com.llmhub.llmhub.websearch.WebSearchService
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.ModelInfo
import com.google.ai.edge.litertlm.MessageCallback
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
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
    // contextWindow becomes the Engine's maxNumTokens when set; otherwise the model file's default applies
    // (4,096 for Gemma 4 E2B/E4B, the same as the default config). topK/topP/temperature are applied per Conversation.
    private var overrideMaxTokens: Int? = null
    private var overrideContextWindow: Int? = null
    /** The engine's maxNumTokens when LLM Hub set it; null when the model file's default applies. */
    private var loadedContextSize: Int? = null
    private var overrideTopK: Int? = null
    private var overrideTopP: Float? = null
    private var overrideTemperature: Float? = null
    private var overrideEnableThinking: Boolean = true

    private val engineMutex = Mutex()
    // Conversations whose reply may still run on LiteRT-LM's worker thread; see closeEngineLocked.
    private val liveConversations = java.util.concurrent.ConcurrentHashMap.newKeySet<Conversation>()
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
        /** Whether a downloaded .litertlm carries an MTP drafter (read from its header; call off the main thread). */
        @OptIn(ExperimentalApi::class)
        fun fileHasMtpDrafter(file: File): Boolean = file.isFile && runCatching {
            ModelInfo.from(file.absolutePath).use { (it as? ModelInfo.Llm)?.hasSpeculativeDecodingSupport() == true }
        }.onFailure { Log.w(TAG, "Could not read model info of ${file.name}", it) }.getOrDefault(false)

        /** Whether [model]'s downloaded file has an MTP drafter, so its settings show the MTP switch. */
        fun supportsMtpSwitch(context: Context, model: LLMModel?): Boolean = model?.modelFormat == "litertlm" &&
            fileHasMtpDrafter(File(File(context.filesDir, "models"), model.localFileName()))

        private const val TAG = "LiteRtLmInference"
        private const val DEFAULT_TOP_K = 40
        private const val DEFAULT_TOP_P = 0.95f
        private const val DEFAULT_TEMPERATURE = 1.0f

        /** Sentinel markers used by ChatComponents.kt to render expandable thinking panel. */
        const val SENTINEL_THINK = "\u200B\u200BTHINK\u200B\u200B"
        const val SENTINEL_ENDTHINK = "\u200B\u200BENDTHINK\u200B\u200B"

        @JvmStatic
        fun getMaxTokensForModelStatic(model: LLMModel): Int = model.contextWindowSize

        /**
         * LiteRT-LM rejects a prompt longer than the engine's maxNumTokens with a LiteRtLmJniException
         * ("Input token ids are too long. Exceeding the maximum number of tokens allowed").
         */
        @JvmStatic
        fun isContextWindowOverflow(error: Throwable): Boolean =
            generateSequence(error) { it.cause }.any { e ->
                val message = e.message ?: return@any false
                message.contains("Input token ids are too long", ignoreCase = true) ||
                    message.contains("maximum number of tokens", ignoreCase = true)
            }
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

    override fun getLoadedContextSize(): Int? = loadedContextSize?.takeIf { engine != null }

    override fun getEffectiveMaxTokens(model: LLMModel): Int {
        val limit = model.contextWindowSize
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
            if (currentModel?.name == model.name && engine != null && loadedMtpSetting == mtpSetting(model)) return@withLock

            closeEngineLocked()

            val modelFile = withContext(Dispatchers.IO) { resolveModelFile(model) }
            isVisionDisabled = disableVision
            isAudioDisabled = disableAudio

            val isGemma4_12B = isGemma4_12B(model)

            // Gemma-4 12B is too slow on the CPU, so it always runs on the GPU (as on iOS).
            val backend = if (isGemma4_12B) Backend.GPU() else mapBackend(preferredBackend, model.supportsGpu)
            currentBackendIsGpu = backend is Backend.GPU

            // Multi-Token Prediction (MTP) via speculative decoding when running on the GPU; LiteRT-LM uses it only when
            // the file carries an MTP drafter.
            val mtp = mtpSetting(model)
            loadedMtpSetting = mtp
            ExperimentalFlags.enableSpeculativeDecoding = mtp && currentBackendIsGpu && fileHasMtpDrafter(modelFile)
            Log.d(TAG, "MTP for ${model.name}: switch=$mtp, on=${ExperimentalFlags.enableSpeculativeDecoding}")
            // LiteRT-LM's own prefill/decode token counts and speeds, read after each chat reply.
            ExperimentalFlags.enableBenchmark = true

            if (ModelData.isLiteRtLmNpuChatModel(model)) {
                loadNpuEngine(model, modelFile, disableVision, disableAudio)
                return@withLock
            }

            val engineConfig = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = backend,
                // Gemma-4 12B's vision encoder runs on the CPU, as on iOS, which leaves the GPU's memory to the 12B
                // decoder; the other models' encoders share their main backend.
                visionBackend = if (model.supportsVision && !disableVision) (if (isGemma4_12B) Backend.CPU() else backend) else null,
                audioBackend = if (model.supportsAudio && !disableAudio) Backend.CPU() else null,
                maxNumTokens = overrideContextWindow?.coerceIn(1, model.contextWindowSize),
                cacheDir = applicationContext.cacheDir.path
            )

            Log.d(TAG, "Initializing engine for ${model.name} | backend=$backend | vision=${model.supportsVision && !disableVision} | audio=${model.supportsAudio && !disableAudio}")

            loadedContextSize = engineConfig.maxNumTokens
            val newEngine = Engine(engineConfig)
            withContext(Dispatchers.IO) { newEngine.initialize() }
            engine = newEngine
            currentModel = model

            Log.d(TAG, "Engine ready for ${model.name}")
        }
    }

    /**
     * A Gemma 4 build compiled for this chip's NPU. The text model runs only there; its vision encoder is tried on
     * the GPU, then the CPU, and audio stays on the CPU. Its context length is fixed by the compiled graphs, so no
     * token limit is passed. Called with [engineMutex] held.
     */
    @OptIn(ExperimentalApi::class)
    private suspend fun loadNpuEngine(model: LLMModel, modelFile: File, disableVision: Boolean, disableAudio: Boolean) {
        val libDir = LiteRtLmNpu.libraryDir(applicationContext)
        val vision = model.supportsVision && !disableVision
        val visionBackends: List<Backend?> = if (vision) listOf(Backend.GPU(), Backend.CPU()) else listOf(null)
        currentBackendIsGpu = false
        // Google's Tensor NPU builds of Gemma 4 E2B carry an MTP drafter; the user's MTP switch decides (on a Pixel 11,
        // Tensor G6, a 174-token reply ran 22.0 tok/s without MTP and 16.7 tok/s with it, 2026-11-28).
        val mtp = mtpSetting(model)
        loadedMtpSetting = mtp
        ExperimentalFlags.enableSpeculativeDecoding = mtp && fileHasMtpDrafter(modelFile)
        Log.d(TAG, "MTP for ${model.name} on the NPU: switch=$mtp, on=${ExperimentalFlags.enableSpeculativeDecoding}")
        ExperimentalFlags.enableBenchmark = true
        var failure: Exception? = null
        for (visionBackend in visionBackends) {
            val config = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.NPU(libDir),
                visionBackend = visionBackend,
                audioBackend = if (model.supportsAudio && !disableAudio) Backend.CPU() else null,
                cacheDir = applicationContext.cacheDir.path
            )
            Log.d(TAG, "Initializing engine for ${model.name} | backend=NPU | vision=$visionBackend | audio=${config.audioBackend}")
            val candidate = Engine(config)
            try {
                withContext(Dispatchers.IO) { candidate.initialize() }
            } catch (e: Exception) {
                Log.w(TAG, "${model.name} failed to load on the NPU with vision on $visionBackend", e)
                runCatching { candidate.close() }
                failure = e
                continue
            }
            loadedContextSize = null
            engine = candidate
            currentModel = model
            Log.d(TAG, "Engine ready for ${model.name} on the NPU")
            return
        }
        throw failure ?: IllegalStateException("${model.name} did not load on the NPU")
    }

    override suspend fun unloadModel() {
        // Counts as unloaded at once, so the next screen (e.g. AI Chat after Writing Aid) never sees it as loaded.
        currentModel = null
        // On the caller's thread, not deferred: with nothing generating the engine closes right away. A reply
        // still generating is cancelled and awaited first (closeEngineLocked), since freeing the engine under
        // LiteRT-LM's worker thread crashed in nativeDeleteEngine (4.5.7). NonCancellable: leaving the feature
        // cancels the screen's scope, which must not stop the close halfway and leave the engine in memory.
        withContext(NonCancellable) {
            engineMutex.withLock {
                closeEngineLocked()
                currentModel = null
                Log.d(TAG, "Engine unloaded")
            }
        }
    }

    /**
     * Call with [engineMutex] held. Replies still running on this engine are cancelled and awaited
     * first: freeing the engine under LiteRT-LM's worker thread crashed in nativeDeleteEngine (4.5.7).
     */
    private suspend fun closeEngineLocked() {
        val eng = engine ?: return
        liveConversations.forEach { runCatching { it.cancelProcess() } }
        withTimeoutOrNull(10_000) { while (liveConversations.isNotEmpty()) delay(50) }
        try { eng.close() } catch (e: Exception) { Log.w(TAG, "Error closing engine: ${e.message}") }
        engine = null
    }

    /** A Conversation registered in [liveConversations] until [block] returns and it is closed. */
    private inline fun <R> Engine.withConversation(config: ConversationConfig, block: (Conversation) -> R): R {
        val conv = createConversation(config)
        liveConversations += conv
        try {
            return block(conv)
        } finally {
            // Freed before it leaves the set: closeEngineLocked closes the engine once the set is empty, and closing
            // it while this Conversation was still being freed crashed in LiteRtDestroyTensorBuffer (4.7.0 vitals).
            try { conv.close() } finally { liveConversations -= conv }
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
                closeEngineLocked()
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

    /**
     * Whether the .litertlm carries an MTP drafter (as iOS checks with ModelInfo). Turning speculative decoding on for
     * a file without one fails engine creation ("Speculative decoding requires embedding lookup"), as with Gemma-3n,
     * Phi-4 and Gemma-3 1B.
     */
    private fun fileHasMtpDrafter(file: File): Boolean = Companion.fileHasMtpDrafter(file)

    /** The MTP switch as loaded into the current engine. */
    @Volatile private var loadedMtpSetting: Boolean? = null

    /** The model's MTP switch from its saved settings (Chat, Writing Aid, Translator and every other feature share it). */
    private suspend fun mtpSetting(model: LLMModel): Boolean =
        runCatching { com.llmhub.llmhub.data.ModelPreferences(applicationContext).getModelConfig(model.name)?.mtpEnabled }
            .getOrNull() ?: ModelData.defaultMtpEnabled(model)

    /** True when the loaded engine was built with a different MTP setting than the model's saved one. */
    suspend fun mtpSettingChanged(model: LLMModel): Boolean =
        engine != null && currentModel?.name == model.name && loadedMtpSetting != null && loadedMtpSetting != mtpSetting(model)

    private fun isGemma4Model(): Boolean =
        currentModel?.name?.contains("Gemma-4", ignoreCase = true) == true

    private fun isGemma4_12B(model: LLMModel?): Boolean =
        model?.name?.contains("Gemma-4 12B", ignoreCase = true) == true || model?.name?.contains("Gemma 4 12B", ignoreCase = true) == true

    /**
     * Gemma-4 12B thinks through LiteRT-LM's own ThinkingConfig; E2B/E4B get "<|think|>" in the system prompt so MTP
     * keeps working on the main token stream (as on iOS).
     */
    private fun usesPromptThinking() = isGemma4Model() && thinkingEnabled() && !isGemma4_12B(currentModel)

    /**
     * Gemma-4 12B with agent tools always thinks: with tools on and thinking off, LiteRT-LM fails at the end of a
     * reply ("The new rendered string is shorter than the previous rendered string").
     */
    private fun thinkingEnabled() = overrideEnableThinking || (isGemma4_12B(currentModel) && agentTools != null)

    private fun thinkingConfig(): ThinkingConfig? =
        if (thinkingEnabled() && isGemma4_12B(currentModel)) ThinkingConfig(enableThinking = true) else null

    /** A chat transcript split for LiteRT-LM: system text, earlier turns, and the newest user message. */
    private class Transcript(val system: String?, val history: List<Message>, val current: String)

    private fun splitTranscript(prompt: String): Transcript {
        val turns = parseVlmPromptTurns(prompt)
        val lastUser = turns.indexOfLast { it.role == "user" }
        if (turns.isEmpty() || lastUser < 0) return Transcript(null, emptyList(), prompt)
        val system = turns.filter { it.role == "system" }.joinToString("\n\n") { it.text }.ifBlank { null }
        val history = turns.subList(0, lastUser).mapNotNull { turn ->
            when (turn.role) {
                "user" -> Message.user(turn.text)
                "assistant" -> Message.model(turn.text)
                else -> null
            }
        }
        // Anything after the newest user turn (there should be nothing) stays with it.
        val current = turns.subList(lastUser, turns.size).filter { it.role != "system" }.joinToString("\n\n") { it.text }
        return Transcript(system, history, current)
    }

    private fun buildConversationConfig(transcript: Transcript? = null): ConversationConfig {
        val system = transcript?.system
        val sysInstruction = when {
            usesPromptThinking() -> Contents.of(if (system != null) "<|think|>\n$system" else "<|think|>")
            system != null -> Contents.of(system)
            else -> null
        }
        return ConversationConfig(
            initialMessages = transcript?.history.orEmpty(),
            samplerConfig = SamplerConfig(
                topK = overrideTopK ?: DEFAULT_TOP_K,
                topP = (overrideTopP ?: DEFAULT_TOP_P).toDouble(),
                temperature = (overrideTemperature ?: DEFAULT_TEMPERATURE).toDouble()
            ),
            systemInstruction = sysInstruction,
            thinkingConfig = thinkingConfig()
        )
    }

    /**
     * Build a ConversationConfig wired with agent tools and system instruction.
     * Call only when [agentTools] is non-null (Gemma-4 loaded).
     */
    @OptIn(ExperimentalApi::class)
    private fun buildAgentConversationConfig(tools: ChatAgentSkillsTools, transcript: Transcript? = null): ConversationConfig {
        val toolProviders: List<ToolProvider> = listOf(tool(tools))
        // The user's own system prompt replaces the agent prompt when set (as on iOS).
        val base = transcript?.system ?: ChatAgentSkillsTools.AGENT_SYSTEM_PROMPT
        val sysText = if (usesPromptThinking()) "<|think|>\n$base" else base
        val sysInstruction = Contents.of(sysText)
        return ConversationConfig(
            initialMessages = transcript?.history.orEmpty(),
            samplerConfig = SamplerConfig(
                topK = overrideTopK ?: DEFAULT_TOP_K,
                topP = (overrideTopP ?: DEFAULT_TOP_P).toDouble(),
                temperature = (overrideTemperature ?: DEFAULT_TEMPERATURE).toDouble()
            ),
            systemInstruction = sysInstruction,
            tools = toolProviders,
            thinkingConfig = thinkingConfig()
        )
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        ensureEngineLoaded(model)
        val eng = engine ?: throw IllegalStateException("No engine loaded")
        return withContext(Dispatchers.IO) {
            val sb = StringBuilder()
            eng.withConversation(buildConversationConfig()) { conv ->
                conv.sendMessageAsyncWithCallback(prompt).collect { msg ->
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
            eng.withConversation(buildConversationConfig()) { conv ->
                conv.sendMessageAsyncWithCallback(prompt).collect { msg ->
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
            lastDecodeSpeed = null
            lastGeneratedTokens = null
            val currentUserMessage = extractCurrentUserMessage(prompt)
            val needsWebSearch = webSearchEnabled
            var enhancedPrompt = prompt

            if (needsWebSearch) {
                emit(localCtx.getString(R.string.web_searching))
                try {
                    val query = SearchIntentDetector.extractSearchQuery(currentUserMessage)
                    val results = webSearchService.search(query, maxResults = 5)
                    WebSearchCitationStore.put(chatId, results)
                    if (results.isNotEmpty()) {
                        emit(localCtx.getString(R.string.web_search_found_results, results.size))
                        val resultsText = results.joinToString("\n\n") {
                            "SOURCE: ${it.source}\nTITLE: ${it.title}\nURL: ${it.url}\nCONTENT: ${it.snippet}\n---"
                        }
                        enhancedPrompt = "CURRENT WEB SEARCH RESULTS:\n$resultsText\n\n" +
                            "Based on the above, answer: \"$currentUserMessage\"\n\n" +
                            "Answer directly and clearly. Cite factual claims with the provided source URLs when possible:"
                    } else {
                        emit(localCtx.getString(R.string.web_search_no_results) + "\n\n")
                    }
                } catch (e: Exception) {
                    emit(localCtx.getString(R.string.web_search_failed, e.message ?: "Unknown error") + "\n\n")
                }
            }

            // The chat screen sends a "user: …\nassistant: …" transcript, which every model gets as one message (as on
            // iOS) except Phi-4 and the Gemma-3 1B NPU builds: sent that way Phi-4 continued the transcript with an
            // invented "user:" turn instead of ending its reply, and Gemma-3 1B ignored the earlier turns, so their
            // earlier turns go to LiteRT-LM as history for the file's own template.
            val splitsHistory = model.name.contains("Phi-4", ignoreCase = true) ||
                (ModelData.isLiteRtLmNpuChatModel(model) && model.name.contains("Gemma-3 1B", ignoreCase = true))
            val transcript = if (splitsHistory) splitTranscript(enhancedPrompt)
                else Transcript(null, emptyList(), enhancedPrompt)

            // Build content parts list (text first, then images, then audio — per API requirement)
            val contentParts = mutableListOf<Content>()
            if (transcript.current.trim().isNotEmpty()) {
                contentParts.add(Content.Text(transcript.current))
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
            val useThinking = isGemma4Model() && thinkingEnabled()
            @OptIn(ExperimentalApi::class)
            if (localAgentTools != null) {
                // Gemma-4 agent mode: enable constrained decoding for reliable function-call
                // extraction, then create conversation with tools + system instruction.
                ExperimentalFlags.enableConversationConstrainedDecoding = true
                val agentConfig = buildAgentConversationConfig(localAgentTools, transcript)
                ExperimentalFlags.enableConversationConstrainedDecoding = false
                Log.d(TAG, "Using agent conversation config with ${localAgentTools.javaClass.simpleName}")
                var sentThinkOpen = false
                var sentThinkClose = false
                eng.withConversation(agentConfig) { conv ->
                    try {
                    conv.sendMessageAsyncWithCallback(contents).collect { msg ->
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
                    } finally {
                        recordBenchmark(conv)
                    }
                }
                if (useThinking && sentThinkOpen && !sentThinkClose) emit(SENTINEL_ENDTHINK)
            } else {
                var sentThinkOpen = false
                var sentThinkClose = false
                eng.withConversation(buildConversationConfig(transcript)) { conv ->
                    try {
                    conv.sendMessageAsyncWithCallback(contents).collect { msg ->
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
                    } finally {
                        recordBenchmark(conv)
                    }
                }
                if (useThinking && sentThinkOpen && !sentThinkClose) emit(SENTINEL_ENDTHINK)
            }

        }.flowOn(Dispatchers.IO)
    }

    @Volatile private var lastDecodeSpeed: Double? = null
    @Volatile private var lastGeneratedTokens: Int? = null

    /** LiteRT-LM's own decode token count and speed for the reply this Conversation just produced. */
    @OptIn(ExperimentalApi::class)
    private fun recordBenchmark(conv: Conversation) {
        val info = runCatching { conv.getBenchmarkInfo() }.getOrNull() ?: return
        lastGeneratedTokens = info.lastDecodeTokenCount.takeIf { it > 0 }
        lastDecodeSpeed = info.lastDecodeTokensPerSecond.takeIf { it > 0.0 }
        Log.i(
            TAG,
            "LiteRT-LM reply: prefill ${info.lastPrefillTokenCount} tokens at ${info.lastPrefillTokensPerSecond} tok/s, " +
                "decode ${info.lastDecodeTokenCount} tokens at ${info.lastDecodeTokensPerSecond} tok/s",
        )
    }

    override fun getLastDecodeSpeedTokPerSec(): Double? = lastDecodeSpeed

    override fun getLastGeneratedTokenCount(): Int? = lastGeneratedTokens

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

    private fun Conversation.sendMessageAsyncWithCallback(prompt: String): Flow<Message> = callbackFlow {
        val finished = CompletableDeferred<Unit>()
        try {
            sendMessageAsync(prompt, object : MessageCallback {
                override fun onMessage(message: Message) {
                    trySend(message)
                }
                override fun onDone() {
                    finished.complete(Unit)
                    close()
                }
                override fun onError(throwable: Throwable) {
                    finished.complete(Unit)
                    close(throwable)
                }
            })
        } catch (e: Exception) {
            finished.complete(Unit)
            close(e)
        }
        // A cancelled collector (Stop, leaving the screen) must not let the caller close the
        // Conversation while the worker thread still generates into it (SIGSEGV in liblitertlm_jni,
        // 4.5.8): stop the native process and wait until it reports done.
        awaitClose {
            if (!finished.isCompleted) {
                runCatching { cancelProcess() }
                runBlocking { withTimeoutOrNull(10_000) { finished.await() } }
            }
        }
    }

    private fun Conversation.sendMessageAsyncWithCallback(contents: Contents): Flow<Message> = callbackFlow {
        val finished = CompletableDeferred<Unit>()
        try {
            sendMessageAsync(contents, object : MessageCallback {
                override fun onMessage(message: Message) {
                    trySend(message)
                }
                override fun onDone() {
                    finished.complete(Unit)
                    close()
                }
                override fun onError(throwable: Throwable) {
                    finished.complete(Unit)
                    close(throwable)
                }
            })
        } catch (e: Exception) {
            finished.complete(Unit)
            close(e)
        }
        // A cancelled collector (Stop, leaving the screen) must not let the caller close the
        // Conversation while the worker thread still generates into it (SIGSEGV in liblitertlm_jni,
        // 4.5.8): stop the native process and wait until it reports done.
        awaitClose {
            if (!finished.isCompleted) {
                runCatching { cancelProcess() }
                runBlocking { withTimeoutOrNull(10_000) { finished.await() } }
            }
        }
    }

}
