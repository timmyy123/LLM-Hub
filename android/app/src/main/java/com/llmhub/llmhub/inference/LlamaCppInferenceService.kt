package com.llmhub.llmhub.inference

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.llmhub.llmhub.R
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.data.localFileName
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.SearchIntentDetector
import com.llmhub.llmhub.websearch.WebSearchCitationStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** JNI entry points for the statically linked, baseline ARMv8 llama.cpp CPU runtime. */
internal object LlamaCppNative {
    init {
        System.loadLibrary("llmhub_llama_cpu")
    }

    external fun nativeInit(): Int
    external fun nativeLoadModel(
        modelPath: String,
        mmprojPath: String?,
        contextSize: Int,
        threadCount: Int,
    ): Int
    external fun nativeSupportsVision(): Boolean
    external fun nativeMediaMarker(): String
    external fun nativeFormatChat(roles: Array<String>, contents: Array<String>): String?
    external fun nativeStartCompletion(
        formattedPrompt: String,
        imagePaths: Array<String>,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
    ): Int
    external fun nativeNextToken(): String?
    external fun nativeStop()
    external fun nativeReset()
    external fun nativeUnload()
    external fun nativeDecodeSpeed(): Double
}

/**
 * CPU safety net for text and vision GGUF models.
 *
 * GenieX remains the primary engine. UnifiedInferenceService selects this implementation only
 * when GenieX cannot initialize or cannot load the selected model. llama.cpp and ggml are linked
 * statically into a uniquely named JNI library, so none of their library names collide with the
 * copies bundled by GenieX.
 */
private enum class LlamaHarmonyState { BEFORE_HEADER, IN_ANALYSIS, IN_TRANSITION, IN_FINAL }
private enum class LlamaMuseState { BEFORE_HEADER, IN_REASONING, IN_TRANSITION, IN_FINAL }

class LlamaCppInferenceService(private val context: Context) : InferenceService {
    companion object {
        private const val TAG = "LlamaCppFallback"
        private const val DEFAULT_MAX_TOKENS = 1024
        private const val MAX_FALLBACK_CONTEXT = 8192
        private const val MAX_IMAGE_DIMENSION = 300
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private val dispatcher = Dispatchers.IO.limitedParallelism(1)
    private val webSearchService = DuckDuckGoSearchService()

    private var nativeInitialized = false
    private var currentModel: LLMModel? = null
    private var contextSize = 4096
    private var lastDecodeSpeed: Double? = null
    private var visionDisabled = true
    private var audioDisabled = true
    private val chatSessions = mutableMapOf<String, MutableList<VlmPromptTurn>>()
    private val sentinelThink = "\u200B\u200BTHINK\u200B\u200B"
    private val sentinelEndThink = "\u200B\u200BENDTHINK\u200B\u200B"
    private val harmonyBuffer = StringBuilder()
    private var harmonyState = LlamaHarmonyState.BEFORE_HEADER
    private val museBuffer = StringBuilder()
    private var museState = LlamaMuseState.BEFORE_HEADER
    private var sentInitialThinkingSentinel = false

    private var overrideMaxTokens: Int? = null
    private var overrideContextWindow: Int? = null
    private var overrideTopK: Int? = null
    private var overrideTopP: Float? = null
    private var overrideTemperature: Float? = null
    private var overrideEnableThinking: Boolean? = null

    override suspend fun loadModel(
        model: LLMModel,
        preferredBackend: LlmInference.Backend?,
        deviceId: String?,
    ): Boolean = loadModel(model, preferredBackend, false, false, deviceId)

    override suspend fun loadModel(
        model: LLMModel,
        preferredBackend: LlmInference.Backend?,
        disableVision: Boolean,
        disableAudio: Boolean,
        deviceId: String?,
    ): Boolean = withContext(dispatcher) {
        try {
            if (!nativeInitialized) {
                check(LlamaCppNative.nativeInit() == 0) { "llama.cpp initialization failed" }
                nativeInitialized = true
            }

            val modelFile = resolveModelFile(model)
            if (!modelFile.isFile || !modelFile.canRead()) {
                Log.e(TAG, "GGUF file is missing or unreadable: ${modelFile.absolutePath}")
                return@withContext false
            }

            if (currentModel != null) LlamaCppNative.nativeUnload()
            contextSize = (overrideContextWindow ?: model.contextWindowSize)
                .coerceIn(512, MAX_FALLBACK_CONTEXT)
            val threads = (Runtime.getRuntime().availableProcessors() - 2).coerceIn(2, 8)
            val modelDir = modelFile.parentFile ?: File(context.filesDir, "models")
            val mmprojFile = if (model.supportsVision && !disableVision) {
                findMmprojFile(modelDir, modelFile, model)
            } else {
                null
            }
            if (model.supportsVision && !disableVision && mmprojFile == null) {
                Log.w(TAG, "Vision projector not found for '${model.name}'; loading text-only")
            }
            Log.i(
                TAG,
                "Loading '${model.name}' with llama.cpp CPU fallback: context=$contextSize " +
                    "threads=$threads mmproj=${mmprojFile?.name ?: "none"}",
            )
            val result = LlamaCppNative.nativeLoadModel(
                modelFile.absolutePath,
                mmprojFile?.absolutePath,
                contextSize,
                threads,
            )
            if (result != 0) {
                Log.e(TAG, "llama.cpp model load failed with code $result")
                currentModel = null
                return@withContext false
            }

            currentModel = model
            chatSessions.clear()
            visionDisabled = disableVision || !model.supportsVision || !LlamaCppNative.nativeSupportsVision()
            audioDisabled = true
            lastDecodeSpeed = null
            Log.i(TAG, "Loaded '${model.name}' using llama.cpp CPU fallback")
            true
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (fatal: VirtualMachineError) {
            throw fatal
        } catch (error: Throwable) {
            Log.e(TAG, "llama.cpp CPU fallback failed to load '${model.name}'", error)
            currentModel = null
            false
        }
    }

    override suspend fun unloadModel() = withContext(dispatcher) {
        if (nativeInitialized) LlamaCppNative.nativeUnload()
        currentModel = null
        lastDecodeSpeed = null
        chatSessions.clear()
    }

    override suspend fun generateResponse(prompt: String, model: LLMModel): String {
        val output = StringBuilder()
        generateResponseStream(prompt, model).collect { output.append(it) }
        return output.toString()
    }

    override suspend fun generateResponseStream(prompt: String, model: LLMModel): Flow<String> =
        generateInternal(prompt, model, webSearchEnabled = false, chatId = "")

    override suspend fun generateResponseStreamWithSession(
        prompt: String,
        model: LLMModel,
        chatId: String,
        images: List<Bitmap>,
        audioData: ByteArray?,
        webSearchEnabled: Boolean,
        imagePaths: List<String>,
    ): Flow<String> {
        if (audioData != null) {
            return flow {
                throw UnsupportedOperationException(
                    "The llama.cpp CPU fallback does not currently support audio input",
                )
            }
        }
        if ((images.isNotEmpty() || imagePaths.isNotEmpty()) &&
            (!model.supportsVision || visionDisabled)
        ) {
            return flow {
                throw UnsupportedOperationException(
                    "Vision is unavailable because this model's multimodal projector is not loaded",
                )
            }
        }
        return flow {
            val generatedFiles = mutableListOf<File>()
            try {
                val allImagePaths = imagePaths.toMutableList()
                images.forEachIndexed { index, bitmap ->
                    val scaled = if (
                        bitmap.width > MAX_IMAGE_DIMENSION || bitmap.height > MAX_IMAGE_DIMENSION
                    ) {
                        val scale = MAX_IMAGE_DIMENSION.toFloat() / maxOf(bitmap.width, bitmap.height)
                        Bitmap.createScaledBitmap(
                            bitmap,
                            (bitmap.width * scale).toInt().coerceAtLeast(1),
                            (bitmap.height * scale).toInt().coerceAtLeast(1),
                            true,
                        )
                    } else {
                        bitmap
                    }
                    val file = File(
                        context.cacheDir,
                        "llamacpp_vlm_${System.currentTimeMillis()}_$index.jpg",
                    )
                    file.outputStream().use { scaled.compress(Bitmap.CompressFormat.JPEG, 70, it) }
                    if (scaled !== bitmap) scaled.recycle()
                    generatedFiles += file
                    allImagePaths += file.absolutePath
                }
                generateInternal(prompt, model, webSearchEnabled, chatId, allImagePaths)
                    .collect { emit(it) }
            } finally {
                generatedFiles.forEach { it.delete() }
            }
        }.flowOn(dispatcher)
    }

    private fun generateInternal(
        prompt: String,
        model: LLMModel,
        webSearchEnabled: Boolean,
        chatId: String,
        imagePaths: List<String> = emptyList(),
    ): Flow<String> = flow {
        check(currentModel?.name == model.name) { "Model not loaded in llama.cpp CPU fallback" }

        var effectivePrompt = prompt
        if (webSearchEnabled) {
            val userText = extractUserTextForSearch(prompt)
            Log.d(TAG, "Web search detected for chat $chatId. Current message: '$userText'")
            emit(context.getString(R.string.web_searching))
            try {
                val query = SearchIntentDetector.extractSearchQuery(userText)
                Log.d(TAG, "Extracted search query: '$query'")
                val results = webSearchService.search(query, maxResults = 5)
                WebSearchCitationStore.put(chatId, results)
                if (results.isNotEmpty()) {
                    Log.d(TAG, "Found ${results.size} search results")
                    emit(context.getString(R.string.web_search_found_results, results.size))
                    val sources = results.joinToString("\n\n") {
                        "SOURCE: ${it.source}\nTITLE: ${it.title}\nURL: ${it.url}\nCONTENT: ${it.snippet}\n---"
                    }
                    effectivePrompt = """
                        CURRENT WEB SEARCH RESULTS:
                        $sources

                        Based on the above current web search results, please answer the user's question: "$userText"

                        IMPORTANT INSTRUCTIONS:
                        - Use ONLY the information from the web search results above
                        - If the search results contain the answer, provide a clear and specific response
                        - If the search results don't contain enough information, say so clearly
                        - For dates and events, be specific based on what you find in the results
                        - Do not make up information not found in the search results
                        - Cite factual claims with the provided source URLs when possible

                        Answer the question directly and clearly:
                    """.trimIndent()
                    Log.d(TAG, "Enhanced prompt created with ${results.size} search results")
                } else {
                    Log.w(TAG, "No search results found for query: '$query'")
                    emit(context.getString(R.string.web_search_no_results) + "\n\n")
                }
            } catch (error: Exception) {
                Log.e(TAG, "Web search failed for chat $chatId", error)
                emit(context.getString(R.string.web_search_failed, error.message ?: "Unknown error") + "\n\n")
            }
        }

        val sessionPrompt = mergePromptIntoChatSession(chatId, effectivePrompt)
        val promptWithMedia = if (imagePaths.isNotEmpty()) {
            injectMediaMarkersIntoLatestUser(sessionPrompt, imagePaths.size)
        } else {
            sessionPrompt
        }
        val thinkingEnabled = overrideEnableThinking ?: true
        val formattedPrompt = formatPromptLikeGeniex(promptWithMedia, model, thinkingEnabled)
        resetTextOutputBehavior(model, thinkingEnabled)

        // Pass the configuration-sheet value through unchanged. Native llama.cpp tokenizes the
        // completed prompt and caps generation only to the context space that actually remains.
        val maxTokens = overrideMaxTokens ?: DEFAULT_MAX_TOKENS
        val temperature = overrideTemperature ?: 0.7f
        val topK = overrideTopK ?: 40
        val topP = overrideTopP ?: 0.9f
        Log.d(
            TAG,
            "Generation config: maxTokens=$maxTokens context=$contextSize " +
                "temperature=$temperature topK=$topK topP=$topP",
        )
        val startResult = LlamaCppNative.nativeStartCompletion(
            formattedPrompt,
            imagePaths.toTypedArray(),
            maxTokens,
            temperature,
            topK,
            topP,
        )
        check(startResult == 0) { "llama.cpp prompt evaluation failed with code $startResult" }

        try {
            val assistantResponse = StringBuilder()
            while (true) {
                val token = LlamaCppNative.nativeNextToken() ?: break
                if (token.isNotEmpty()) {
                    filterTextOutputToken(token, model, thinkingEnabled)
                        .forEach {
                            assistantResponse.append(it)
                            emit(it)
                        }
                }
            }
            finishTextOutput(model).forEach {
                assistantResponse.append(it)
                emit(it)
            }
            rememberAssistantResponse(chatId, assistantResponse.toString())
            lastDecodeSpeed = LlamaCppNative.nativeDecodeSpeed().takeIf { it > 0.0 }
            Log.i(TAG, "CPU fallback generation completed at ${lastDecodeSpeed ?: 0.0} tok/s")
        } catch (cancelled: CancellationException) {
            LlamaCppNative.nativeStop()
            throw cancelled
        } finally {
            LlamaCppNative.nativeStop()
        }
    }.flowOn(dispatcher)

    override suspend fun resetChatSession(chatId: String) = withContext(dispatcher) {
        if (nativeInitialized) LlamaCppNative.nativeReset()
        chatSessions.remove(chatId)
        Unit
    }

    override suspend fun onCleared() {
        unloadModel()
    }

    override fun getCurrentlyLoadedModel(): LLMModel? = currentModel
    override fun getCurrentlyLoadedBackend(): LlmInference.Backend? =
        if (currentModel != null) LlmInference.Backend.CPU else null
    override fun getMemoryWarningForImages(images: List<Bitmap>): String? = null
    override fun wasSessionRecentlyReset(chatId: String): Boolean = false
    override fun setGenerationParameters(
        maxTokens: Int?,
        topK: Int?,
        topP: Float?,
        temperature: Float?,
        nGpuLayers: Int?,
        enableThinking: Boolean?,
        contextWindow: Int?,
    ) {
        overrideMaxTokens = maxTokens
        overrideTopK = topK
        overrideTopP = topP
        overrideTemperature = temperature
        overrideEnableThinking = enableThinking
        overrideContextWindow = contextWindow
    }
    override fun isVisionCurrentlyDisabled(): Boolean = visionDisabled
    override fun isAudioCurrentlyDisabled(): Boolean = audioDisabled
    override fun isGpuBackendEnabled(): Boolean = false
    override fun isNpuBackendEnabled(): Boolean = false
    override fun getLastDecodeSpeedTokPerSec(): Double? = lastDecodeSpeed
    override fun getEffectiveMaxTokens(model: LLMModel): Int =
        overrideMaxTokens ?: DEFAULT_MAX_TOKENS

    /** Keep web-search extraction byte-for-byte equivalent to GeniexInferenceService. */
    private fun extractUserTextForSearch(prompt: String): String {
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

    /**
     * Maintain real chat state for callers that send only the newest turn. When ChatViewModel
     * supplies a complete/trimmed transcript, that transcript becomes the source of truth; when
     * it supplies one user turn, it is appended to the existing chatId session.
     */
    private fun mergePromptIntoChatSession(chatId: String, prompt: String): String {
        if (chatId.isBlank()) return prompt
        val incoming = parseVlmPromptTurns(prompt).toMutableList().apply {
            if (isEmpty() && prompt.isNotBlank()) add(VlmPromptTurn("user", prompt.trim()))
        }
        val session = chatSessions.getOrPut(chatId) { mutableListOf() }
        val containsTranscript = incoming.size > 1 || incoming.any { it.role != "user" }
        if (containsTranscript) {
            session.clear()
            session.addAll(incoming)
        } else {
            incoming.firstOrNull()?.let { currentUser ->
                if (session.lastOrNull()?.role == "user") {
                    session[session.lastIndex] = currentUser
                } else {
                    session += currentUser
                }
            }
        }
        return buildString {
            session.forEach { turn ->
                append(turn.role).append(": ").append(turn.text).append("\n\n")
            }
            append("assistant:")
        }
    }

    /** Put mtmd media markers inside the newest user turn before applying the model chat template. */
    private fun injectMediaMarkersIntoLatestUser(prompt: String, imageCount: Int): String {
        if (imageCount <= 0) return prompt
        val userMarker = "user: "
        val lastUser = prompt.lastIndexOf(userMarker)
        check(lastUser >= 0) { "Cannot attach an image because the prompt has no user turn" }
        val insertAt = lastUser + userMarker.length
        val mediaMarkers = buildString {
            repeat(imageCount) { append(LlamaCppNative.nativeMediaMarker()) }
            append('\n')
        }
        return prompt.substring(0, insertAt) + mediaMarkers + prompt.substring(insertAt)
    }

    private fun rememberAssistantResponse(chatId: String, response: String) {
        if (chatId.isBlank() || response.isBlank()) return
        val session = chatSessions.getOrPut(chatId) { mutableListOf() }
        if (session.lastOrNull()?.role == "assistant") {
            session[session.lastIndex] = VlmPromptTurn("assistant", response)
        } else {
            session += VlmPromptTurn("assistant", response)
        }
    }

    private fun resetTextOutputBehavior(model: LLMModel, thinkingEnabled: Boolean) {
        val harmony = isHarmonyModel(model)
        val muse = isMuseModel(model)
        sentInitialThinkingSentinel = false
        harmonyBuffer.clear()
        harmonyState = if (!thinkingEnabled && harmony) LlamaHarmonyState.IN_FINAL else LlamaHarmonyState.BEFORE_HEADER
        museBuffer.clear()
        museState = if (!thinkingEnabled && muse) LlamaMuseState.IN_FINAL else LlamaMuseState.BEFORE_HEADER
    }

    private fun filterTextOutputToken(
        text: String,
        model: LLMModel,
        thinkingEnabled: Boolean,
    ): List<String> {
        val output = mutableListOf<String>()
        val isGranite42 = isGranite42Model(model)
        val thinking = (((model.name.contains("Thinking", true) ||
            model.name.contains("Reasoning", true) ||
            model.name.contains("LFM2.5-8B-A1B", true) ||
            model.name.contains("LFM-2.5 2.6B", true)) && !model.name.contains("muse", true)) ||
            (isGranite42 && thinkingEnabled))
        when {
            isMuseModel(model) -> emitMuse(text, thinkingEnabled, output::add)
            isHarmonyModel(model) -> emitHarmony(text, output::add)
            thinking -> {
                var token = text
                if (!sentInitialThinkingSentinel && !token.contains("<think>")) {
                    sentInitialThinkingSentinel = true
                    token = sentinelThink + token
                }
                if (token.contains("<think>")) {
                    sentInitialThinkingSentinel = true
                    token = token.replace("<think>", sentinelThink)
                }
                if (token.contains("</think>")) token = token.replace("</think>", sentinelEndThink)
                if (token.contains("<|im_end|>")) token = token.replace("<|im_end|>", "")
                output += token
            }
            else -> {
                var token = text
                if (isGranite42 && token.contains("<|im_end|>")) {
                    token = token.replace("<|im_end|>", "")
                }
                output += token
            }
        }
        return output
    }

    private fun finishTextOutput(model: LLMModel): List<String> {
        val output = mutableListOf<String>()
        sentInitialThinkingSentinel = false
        if (isMuseModel(model)) {
            if (museState == LlamaMuseState.IN_REASONING) {
                var remaining = museBuffer.toString().replace("<|eot|>", "").replace("<|end_of_text|>", "")
                val userIndex = findMuseUserHeaderIndex(remaining)
                if (userIndex >= 0) remaining = remaining.substring(0, userIndex)
                if (remaining.isNotEmpty()) output += remaining
                output += sentinelEndThink
            } else if (museState == LlamaMuseState.IN_FINAL || museState == LlamaMuseState.IN_TRANSITION) {
                var remaining = museBuffer.toString().replace("<|eot|>", "").replace("<|end_of_text|>", "")
                stripLeadingMuseUserHeader(remaining)?.let { remaining = it }
                if (remaining.startsWith("<|message|>")) {
                    remaining = remaining.removePrefix("<|message|>").trimStart('\n', '\r', ' ')
                }
                if (remaining.isNotEmpty()) output += remaining
            }
            museBuffer.clear()
            museState = LlamaMuseState.BEFORE_HEADER
        }
        return output
    }

    private fun emitHarmony(tokenText: String, send: (String) -> Unit) {
        harmonyBuffer.append(tokenText)
        val analysisHeader = "<|channel|>analysis<|message|>"
        val endTag = "<|end|>"
        val finalHeader = "<|start|>assistant<|channel|>final<|message|>"
        when (harmonyState) {
            LlamaHarmonyState.BEFORE_HEADER -> {
                val headerIndex = harmonyBuffer.indexOf(analysisHeader)
                if (headerIndex >= 0) {
                    val remainder = harmonyBuffer.substring(headerIndex + analysisHeader.length)
                    harmonyBuffer.clear().append(remainder)
                    harmonyState = LlamaHarmonyState.IN_ANALYSIS
                    send(sentinelThink)
                    if (harmonyBuffer.isNotEmpty()) emitHarmony("", send)
                }
            }
            LlamaHarmonyState.IN_ANALYSIS -> {
                val value = harmonyBuffer.toString()
                val endIndex = value.indexOf(endTag)
                if (endIndex >= 0) {
                    value.substring(0, endIndex).takeIf { it.isNotEmpty() }?.let(send)
                    send(sentinelEndThink)
                    harmonyBuffer.clear().append(value.substring(endIndex + endTag.length))
                    harmonyState = LlamaHarmonyState.IN_TRANSITION
                    if (harmonyBuffer.isNotEmpty()) emitHarmony("", send)
                } else {
                    val safeLength = (value.length - (endTag.length - 1)).coerceAtLeast(0)
                    if (safeLength > 0) {
                        send(value.substring(0, safeLength))
                        harmonyBuffer.delete(0, safeLength)
                    }
                }
            }
            LlamaHarmonyState.IN_TRANSITION -> {
                val value = harmonyBuffer.toString()
                val finalIndex = value.indexOf(finalHeader)
                if (finalIndex >= 0) {
                    harmonyBuffer.clear().append(value.substring(finalIndex + finalHeader.length))
                    harmonyState = LlamaHarmonyState.IN_FINAL
                    if (harmonyBuffer.isNotEmpty()) emitHarmony("", send)
                }
            }
            LlamaHarmonyState.IN_FINAL -> if (harmonyBuffer.isNotEmpty()) {
                send(harmonyBuffer.toString())
                harmonyBuffer.clear()
            }
        }
    }

    private fun emitMuse(tokenText: String, thinkingEnabled: Boolean, send: (String) -> Unit) {
        museBuffer.append(tokenText)
        when (museState) {
            LlamaMuseState.BEFORE_HEADER -> {
                val value = museBuffer.toString()
                if (!thinkingEnabled) {
                    val cleaned = stripLeadingMuseUserHeader(value)
                    if (cleaned != null) {
                        museBuffer.clear().append(cleaned)
                        museState = LlamaMuseState.IN_FINAL
                        if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    } else if (!isPotentialMuseHeaderPrefix(value)) {
                        museState = LlamaMuseState.IN_FINAL
                        if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    }
                    return
                }
                val self = stripLeadingMuseSelfHeader(value)
                if (self != null) {
                    museBuffer.clear().append(self)
                    museState = LlamaMuseState.IN_REASONING
                    send(sentinelThink)
                    if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    return
                }
                val user = stripLeadingMuseUserHeader(value)
                if (user != null) {
                    museBuffer.clear().append(user)
                    museState = LlamaMuseState.IN_FINAL
                    if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    return
                }
                if (isPotentialMuseHeaderPrefix(value)) return
                if (value.length >= 40) {
                    museState = LlamaMuseState.IN_REASONING
                    send(sentinelThink)
                    val safeLength = (museBuffer.length - 20).coerceAtLeast(0)
                    if (safeLength > 0) {
                        send(museBuffer.substring(0, safeLength))
                        museBuffer.delete(0, safeLength)
                    }
                }
            }
            LlamaMuseState.IN_REASONING -> {
                val value = museBuffer.toString()
                val eomIndex = value.indexOf("<|eom|>")
                if (eomIndex >= 0) {
                    value.substring(0, eomIndex).takeIf { it.isNotEmpty() }?.let(send)
                    send(sentinelEndThink)
                    museBuffer.clear().append(value.substring(eomIndex + "<|eom|>".length))
                    museState = LlamaMuseState.IN_TRANSITION
                    if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    return
                }
                val userIndex = findMuseUserHeaderIndex(value)
                if (userIndex >= 0) {
                    value.substring(0, userIndex).takeIf { it.isNotEmpty() }?.let(send)
                    send(sentinelEndThink)
                    museBuffer.clear().append(value.substring(userIndex))
                    museState = LlamaMuseState.IN_TRANSITION
                    if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    return
                }
                val eotIndex = value.indexOf("<|eot|>")
                if (eotIndex >= 0) {
                    value.substring(0, eotIndex).takeIf { it.isNotEmpty() }?.let(send)
                    send(sentinelEndThink)
                    museBuffer.clear()
                    museState = LlamaMuseState.IN_FINAL
                    return
                }
                val safeLength = (value.length - 45).coerceAtLeast(0)
                if (safeLength > 0) {
                    send(value.substring(0, safeLength))
                    museBuffer.delete(0, safeLength)
                }
            }
            LlamaMuseState.IN_TRANSITION -> {
                val value = museBuffer.toString()
                val cleaned = stripLeadingMuseUserHeader(value)
                if (cleaned != null) {
                    museBuffer.clear().append(cleaned)
                    museState = LlamaMuseState.IN_FINAL
                    if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    return
                }
                val messageIndex = value.indexOf("<|message|>")
                if (messageIndex >= 0) {
                    val remainder = value.substring(messageIndex + "<|message|>".length).trimStart('\n', '\r', ' ')
                    museBuffer.clear().append(remainder)
                    museState = LlamaMuseState.IN_FINAL
                    if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    return
                }
                if (isPotentialMuseUserHeaderPrefix(value)) return
                museBuffer.clear().append(value.trimStart('\n', '\r', ' '))
                museState = LlamaMuseState.IN_FINAL
                if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
            }
            LlamaMuseState.IN_FINAL -> {
                val value = museBuffer.toString()
                val cleaned = stripLeadingMuseUserHeader(value)
                if (cleaned != null) {
                    museBuffer.clear().append(cleaned)
                    if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    return
                }
                if (value.startsWith("<|message|>")) {
                    museBuffer.clear().append(value.removePrefix("<|message|>").trimStart('\n', '\r', ' '))
                    if (museBuffer.isNotEmpty()) emitMuse("", thinkingEnabled, send)
                    return
                }
                val eot = value.indexOf("<|eot|>")
                val end = value.indexOf("<|end_of_text|>")
                val stop = when {
                    eot >= 0 && end >= 0 -> minOf(eot, end)
                    eot >= 0 -> eot
                    else -> end
                }
                if (stop >= 0) {
                    value.substring(0, stop).takeIf { it.isNotEmpty() }?.let(send)
                    museBuffer.clear()
                    return
                }
                val safeLength = (value.length - 15).coerceAtLeast(0)
                if (safeLength > 0) {
                    send(value.substring(0, safeLength))
                    museBuffer.delete(0, safeLength)
                }
            }
        }
    }

    private fun stripLeadingMuseUserHeader(text: String): String? = stripMuseHeader(text, "user")
    private fun stripLeadingMuseSelfHeader(text: String): String? = stripMuseHeader(text, "self")

    private fun stripMuseHeader(text: String, recipient: String): String? {
        val trimmed = text.trimStart('\n', '\r', ' ')
        if (recipient == "user" && trimmed.startsWith("<|message|>")) {
            return trimmed.removePrefix("<|message|>").trimStart('\n', '\r', ' ')
        }
        val headers = listOf(
            "<|start|>assistant to=$recipient<|message|>",
            "<|start|>assistant to=$recipient\n<|message|>",
            "<|start|>assistant\nto=$recipient<|message|>",
            "<|start|>assistant\nto=$recipient\n<|message|>",
            "assistant to=$recipient<|message|>",
            "assistant to=$recipient\n<|message|>",
            "to=$recipient<|message|>",
            "to=$recipient\n<|message|>",
        )
        headers.firstOrNull { trimmed.startsWith(it) }?.let {
            return trimmed.substring(it.length).trimStart('\n', '\r', ' ')
        }
        val recipientIndex = trimmed.indexOf("to=$recipient")
        if (recipientIndex in 0..25) {
            val messageIndex = trimmed.indexOf("<|message|>", recipientIndex)
            if (messageIndex >= 0 && messageIndex - recipientIndex < 30) {
                return trimmed.substring(messageIndex + "<|message|>".length).trimStart('\n', '\r', ' ')
            }
        }
        return null
    }

    private fun isPotentialMuseUserHeaderPrefix(text: String): Boolean {
        val trimmed = text.trimStart('\n', '\r', ' ')
        if (trimmed.isEmpty()) return true
        return listOf(
            "<|start|>assistant to=user<|message|>",
            "assistant to=user<|message|>",
            "to=user<|message|>",
            "<|message|>",
        ).any { it.startsWith(trimmed) }
    }

    private fun isPotentialMuseHeaderPrefix(text: String): Boolean {
        val trimmed = text.trimStart('\n', '\r', ' ')
        if (trimmed.isEmpty()) return true
        return listOf(
            "<|start|>assistant to=self<|message|>",
            "<|start|>assistant to=user<|message|>",
            "assistant to=self<|message|>",
            "assistant to=user<|message|>",
            "to=self<|message|>",
            "to=user<|message|>",
            "<|message|>",
        ).any { it.startsWith(trimmed) }
    }

    private fun findMuseUserHeaderIndex(text: String): Int = listOf(
        "<|start|>assistant to=user<|message|>",
        "assistant to=user<|message|>",
        "to=user<|message|>",
        "<|start|>assistant to=user",
        "assistant to=user",
        "to=user",
    ).map { text.indexOf(it) }.firstOrNull { it >= 0 } ?: -1

    private fun isHarmonyModel(model: LLMModel): Boolean =
        model.name.contains("gpt-oss", true) || model.name.contains("gpt_oss", true)

    private fun isMuseModel(model: LLMModel): Boolean =
        model.name.contains("muse glimmer", true) || model.name.contains("muse-glimmer", true)

    private fun isGranite42Model(model: LLMModel): Boolean =
        model.name.contains("granite-4.2", true) || model.name.contains("granite 4.2", true)

    /**
     * GenieX's text prompt pipeline, using llama.cpp only for the model-owned chat template.
     * Conversation parsing, feature/RAG preservation, thinking controls and manual fallbacks are
     * deliberately kept equivalent to GeniexInferenceService.formatPrompt.
     */
    private fun formatPromptLikeGeniex(
        prompt: String,
        model: LLMModel,
        thinkingEnabled: Boolean,
    ): String {
        val cleanPrompt = if (prompt.trimEnd().endsWith("assistant:")) {
            prompt.substringBeforeLast("assistant:").trimEnd()
        } else {
            prompt
        }
        val messages = parseMessagesLikeGeniex(cleanPrompt)

        val isMuseGlimmer = model.name.contains("muse glimmer", ignoreCase = true) ||
            model.name.contains("muse-glimmer", ignoreCase = true)
        if (isMuseGlimmer) {
            return buildMuseGlimmerPrompt(messages, cleanPrompt, thinkingEnabled)
        }

        if (isGranite42Model(model)) {
            return buildGranite42Prompt(messages, cleanPrompt, thinkingEnabled)
        }

        val formatted = LlamaCppNative.nativeFormatChat(
            messages.map { it.role }.toTypedArray(),
            messages.map { it.text }.toTypedArray(),
        )
        if (!formatted.isNullOrEmpty()) {
            val isThinkingModel = model.name.contains("Thinking", ignoreCase = true) ||
                model.name.contains("Reasoning", ignoreCase = true) ||
                model.name.contains("LFM2.5-8B-A1B", ignoreCase = true)
            val isHarmonyModel = model.name.contains("gpt-oss", ignoreCase = true) ||
                model.name.contains("gpt_oss", ignoreCase = true)
            var result = if (!thinkingEnabled && isThinkingModel && !isHarmonyModel) {
                injectNoThinkIntoFormatted(formatted)
            } else {
                formatted
            }
            if (!thinkingEnabled && isHarmonyModel) {
                result = result.trimEnd() +
                    "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
            }
            return result
        }

        if (model.name.contains("Ministral", ignoreCase = true) ||
            model.name.contains("Mistral", ignoreCase = true)
        ) {
            val sb = StringBuilder("<s>")
            val isReasoning = model.name.contains("Reasoning", ignoreCase = true) ||
                model.name.contains("Thinking", ignoreCase = true)
            var systemInstructions = if (isReasoning) {
                "You are a reasoning model. Always output your internal thought process within <think> and </think> tags before your final answer.\n\n"
            } else {
                ""
            }
            messages.filter { it.role == "system" }.forEach {
                systemInstructions += it.text + "\n\n"
            }
            var isFirstUser = true
            messages.forEachIndexed { index, message ->
                when (message.role) {
                    "system" -> Unit
                    "user" -> {
                        if (!isFirstUser) sb.append(' ')
                        sb.append("[INST] ")
                        if (isFirstUser && systemInstructions.isNotEmpty()) {
                            sb.append(systemInstructions)
                            isFirstUser = false
                        }
                        sb.append(message.text).append(" [/INST]")
                    }
                    "assistant" -> if (!(index == messages.lastIndex && message.text.isEmpty())) {
                        sb.append(' ').append(message.text).append("</s>")
                    }
                }
            }
            return sb.toString()
        }

        val sb = StringBuilder()
        val isThinkingModel = model.name.contains("Thinking", ignoreCase = true) ||
            model.name.contains("Reasoning", ignoreCase = true) ||
            model.name.contains("LFM2.5-8B-A1B", ignoreCase = true)
        sb.append("<|im_start|>system\n")
        if (isThinkingModel) {
            sb.append("You are a reasoning model. Always output your internal thought process within <think> and </think> tags before your final answer.\n")
        } else {
            sb.append("You are a helpful assistant.\n")
        }
        sb.append("<|im_end|>\n")
        messages.forEachIndexed { index, message ->
            if (message.role != "assistant" || message.text.isNotEmpty() || index != messages.lastIndex) {
                sb.append("<|im_start|>${message.role}\n${message.text}<|im_end|>\n")
            }
        }
        sb.append("<|im_start|>assistant\n")
        return sb.toString()
    }

    private fun parseMessagesLikeGeniex(cleanPrompt: String): MutableList<VlmPromptTurn> {
        val messages = mutableListOf<VlmPromptTurn>()
        if (cleanPrompt.contains("user: ") || cleanPrompt.contains("assistant: ")) {
            var systemPromptText = ""
            cleanPrompt.split("\n\n").filter { it.isNotBlank() }.forEach { segment ->
                when {
                    segment.startsWith("system: ") -> {
                        val content = segment.removePrefix("system: ").trim()
                        if (content.isNotEmpty()) systemPromptText += "$content\n\n"
                    }
                    segment.startsWith("user: ") -> {
                        val content = segment.removePrefix("user: ").trim()
                        if (messages.isEmpty() && systemPromptText.isNotEmpty()) {
                            messages += VlmPromptTurn("user", systemPromptText + content)
                            systemPromptText = ""
                        } else {
                            messages += VlmPromptTurn("user", content)
                        }
                    }
                    segment.startsWith("assistant: ") ->
                        messages += VlmPromptTurn("assistant", segment.removePrefix("assistant: ").trim())
                    messages.isEmpty() -> systemPromptText += segment.trim() + "\n\n"
                    else -> {
                        val last = messages.last()
                        messages[messages.lastIndex] = last.copy(text = last.text + "\n\n" + segment.trim())
                    }
                }
            }
            if (systemPromptText.isNotEmpty()) {
                messages.add(0, VlmPromptTurn("system", systemPromptText.trimEnd()))
            }
        }

        if (messages.isEmpty()) {
            val quotedRequest = Regex("""(?is)\bUSER REQUEST\s*:\s*\"([\s\S]*?)\"""").find(cleanPrompt)
            val plainRequest = Regex("""(?im)^\s*User request\s*:\s*(.+)$""").find(cleanPrompt)
            val request = quotedRequest ?: plainRequest
            if (request != null) {
                val userContent = request.groupValues.getOrNull(1)?.trim().orEmpty()
                val systemContent = cleanPrompt.removeRange(request.range).trim()
                if (userContent.isNotEmpty() && systemContent.isNotEmpty()) {
                    messages += VlmPromptTurn("system", systemContent)
                    messages += VlmPromptTurn("user", userContent)
                }
            }

            if (messages.isEmpty()) {
                val creator = Regex(
                    """(?is)\bUser Description\s*:\s*"([\s\S]*?)"\s*(?=\n+\s*Structure your response EXACTLY)""",
                ).find(cleanPrompt)
                if (creator != null) {
                    val userContent = creator.groupValues.getOrNull(1)?.trim().orEmpty()
                    val systemContent = cleanPrompt.removeRange(creator.range).trim()
                    if (userContent.isNotEmpty() && systemContent.isNotEmpty()) {
                        messages += VlmPromptTurn("system", systemContent)
                        messages += VlmPromptTurn("user", userContent)
                    }
                }
            }

            if (messages.isEmpty()) {
                val separators = listOf(
                    "Text to rewrite:\n",
                    "Content to analyze:\n",
                    "Text to translate:\n",
                    "Text to transcribe:\n",
                    "Text to process:\n",
                )
                val separator = separators.firstOrNull { cleanPrompt.contains(it) }
                if (separator != null) {
                    val index = cleanPrompt.indexOf(separator)
                    val instructions = cleanPrompt.substring(0, index).trimEnd()
                    val userContent = cleanPrompt.substring(index + separator.length).trim()
                    if (instructions.isNotEmpty() && userContent.isNotEmpty()) {
                        messages += VlmPromptTurn("system", instructions)
                        messages += VlmPromptTurn("user", userContent)
                    } else {
                        messages += VlmPromptTurn("user", cleanPrompt.trim())
                    }
                } else {
                    messages += VlmPromptTurn("user", cleanPrompt.trim())
                }
            }
        }
        return messages
    }

    private fun injectNoThinkIntoFormatted(formatted: String): String {
        val chatMlMarker = "<|im_start|>user\n"
        val lastChatMl = formatted.lastIndexOf(chatMlMarker)
        if (lastChatMl >= 0) {
            val insert = lastChatMl + chatMlMarker.length
            return formatted.substring(0, insert) + "/no_think " + formatted.substring(insert)
        }
        val lastInst = formatted.lastIndexOf("[INST]")
        if (lastInst >= 0) {
            val insert = lastInst + "[INST]".length + 1
            return formatted.substring(0, insert) + "/no_think " + formatted.substring(insert)
        }
        return "/no_think $formatted"
    }

    private fun buildGranite42Prompt(
        messages: List<VlmPromptTurn>,
        cleanPrompt: String,
        thinkingEnabled: Boolean,
    ): String {
        val systemContent = messages.filter { it.role == "system" }
            .joinToString("\n\n") { it.text }.trim()
        val history = messages.filter { it.role != "system" }
        val sb = StringBuilder()
        if (systemContent.isNotEmpty()) {
            sb.append("<|im_start|>system\n$systemContent<|im_end|>\n")
        }
        if (history.isEmpty()) {
            val userContent = cleanPrompt.trim()
            if (userContent.isNotEmpty()) {
                sb.append("<|im_start|>user\n$userContent<|im_end|>\n")
            }
        } else {
            history.forEach { message ->
                val content = message.text.trim()
                if (content.isNotEmpty()) {
                    if (message.role == "user") {
                        sb.append("<|im_start|>user\n$content<|im_end|>\n")
                    } else {
                        val formattedAssistant = if (content.contains("<think>") && content.contains("</think>")) {
                            "<think></think>" + content.substringAfterLast("</think>").trim()
                        } else if (!content.startsWith("<think>")) {
                            "<think></think>$content"
                        } else {
                            content
                        }
                        sb.append("<|im_start|>assistant\n$formattedAssistant<|im_end|>\n")
                    }
                }
            }
        }
        if (thinkingEnabled) {
            sb.append("<|im_start|>assistant\n<think>\n")
        } else {
            sb.append("<|im_start|>assistant\n<think></think>")
        }
        return sb.toString()
    }

    private fun buildMuseGlimmerPrompt(
        messages: List<VlmPromptTurn>,
        cleanPrompt: String,
        thinkingEnabled: Boolean,
    ): String {
        val systemContent = messages.filter { it.role == "system" }
            .joinToString("\n\n") { it.text }
        val history = messages.filter { it.role != "system" }
        val effectiveSystem = systemContent.ifBlank { "You are a helpful AI assistant." }.trim()
        val currentDate = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
        val validRecipients = if (thinkingEnabled) "\"self\", \"user\"" else "\"user\""
        val sb = StringBuilder()
        sb.append("<|start|>system<|message|>$effectiveSystem\nKnowledge cutoff: 2026-01-04.\nCurrent date: $currentDate.\n\nReasoning strength: high.\n\n# Valid recipients: $validRecipients.<|eot|>")
        if (history.isEmpty()) {
            sb.append("<|start|>user<|message|>${cleanPrompt.trim()}<|eot|>")
        } else {
            history.forEach { message ->
                val content = message.text.trim()
                if (content.isNotEmpty()) {
                    if (message.role == "user") {
                        sb.append("<|start|>user<|message|>$content<|eot|>")
                    } else {
                        sb.append("<|start|>assistant to=user<|message|>$content<|eot|>")
                    }
                }
            }
        }
        if (thinkingEnabled) sb.append("<|start|>assistant")
        else sb.append("<|start|>assistant to=user<|message|>")
        return sb.toString()
    }

    /** Match the same local mmproj/projector discovery rules used by the GenieX backend. */
    private fun findMmprojFile(modelDir: File, modelFile: File, model: LLMModel): File? {
        val modelsDir = File(context.filesDir, "models")

        model.additionalFiles.forEach { urlOrPath ->
            val fileName = urlOrPath.substringAfterLast("/").substringBefore("?").substringBefore("#")
            listOf(
                File(modelDir, fileName),
                File(modelsDir, fileName),
                File(modelDir, urlOrPath),
                File(modelsDir, urlOrPath),
            ).firstOrNull { it.isFile }?.let { return it }
        }

        val searchDirs = buildList {
            if (modelDir.isDirectory) add(modelDir)
            if (modelDir.absolutePath != modelsDir.absolutePath && modelsDir.isDirectory) add(modelsDir)
        }
        val allProjectors = searchDirs.flatMap { directory ->
            directory.listFiles { file ->
                file.isFile && file.extension.equals("gguf", true) &&
                    (file.name.contains("mmproj", true) || file.name.contains("projector", true))
            }?.toList().orEmpty()
        }.distinctBy { it.absolutePath }
        if (allProjectors.isEmpty()) return null
        if (allProjectors.size == 1) return allProjectors.first()

        val variantRegex = Regex(
            """(?:q\d(?:_[a-z0-9]+)?|bf16|f16|f32)""",
            RegexOption.IGNORE_CASE,
        )
        fun normalizedVariant(name: String): String? {
            return when (val value = variantRegex.find(name)?.value?.lowercase()) {
                "f16", "bf16" -> "bf16"
                else -> value
            }
        }
        fun cleanName(name: String): String = name.lowercase()
            .replace("mmproj", "")
            .replace("projector", "")
            .replace("vision", "")
            .replace(
                Regex("""[-_](?:q\d(?:_[a-z0-9]+)?|bf16|f16|f32|ud-[a-z0-9_]+|instruct|it|preview)"""),
                "",
            )
            .replace(Regex("""[^a-z0-9]"""), "")

        val modelCore = cleanName(modelFile.nameWithoutExtension)
        val namedCandidates = allProjectors.filter { candidate ->
            val candidateCore = cleanName(candidate.nameWithoutExtension)
            candidateCore.isNotEmpty() && modelCore.isNotEmpty() &&
                (modelCore.contains(candidateCore) || candidateCore.contains(modelCore))
        }.ifEmpty { allProjectors }

        val modelVariant = normalizedVariant(modelFile.nameWithoutExtension)
        if (modelVariant != null) {
            namedCandidates.firstOrNull {
                normalizedVariant(it.nameWithoutExtension) == modelVariant
            }?.let { return it }
        }
        return namedCandidates.firstOrNull {
            it.name.contains("bf16", true) || it.name.contains("f16", true)
        } ?: namedCandidates.first()
    }

    private fun resolveModelFile(model: LLMModel): File {
        val modelsDir = File(context.filesDir, "models")
        if (model.source == "Custom" && model.url.startsWith("content://")) {
            val target = File(modelsDir, model.localFileName())
            target.parentFile?.mkdirs()
            if (!target.exists()) {
                context.contentResolver.openInputStream(Uri.parse(model.url))?.use { input ->
                    target.outputStream().use { output -> input.copyTo(output) }
                }
            }
            return target
        }

        val modelDirectoryName = model.name
            .replace(" ", "_")
            .replace(Regex("[^a-zA-Z0-9_.-]"), "")
        val modelDir = File(modelsDir, modelDirectoryName).takeIf { it.isDirectory } ?: modelsDir
        val expected = File(modelDir, model.localFileName())
        if (expected.isFile) return expected
        val rootExpected = File(modelsDir, model.localFileName())
        if (rootExpected.isFile) return rootExpected
        return modelDir.listFiles { file ->
            file.isFile && file.extension.equals("gguf", true) &&
                !file.name.contains("mmproj", true) &&
                !file.name.contains("projector", true)
        }?.firstOrNull() ?: expected
    }
}
