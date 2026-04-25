package com.llmhub.llmhub.mimobot.pipeline

import android.util.Log
import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.embedding.RagService
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.mimobot.audio.AudioSink
import com.llmhub.llmhub.mimobot.speech.SpeechToText
import com.llmhub.llmhub.mimobot.speech.Tts
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.SearchIntentDetector
import com.llmhub.llmhub.websearch.WebSearchService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Voice loop wiring: STT → LLM → TTS → AudioSink.
 *
 * Transport-agnostic: pass a [SpeakerSink][com.llmhub.llmhub.mimobot.audio.SpeakerSink]
 * for local dev mode, or a future `BleAudioSink` (Opus-encode + BLE write) for
 * the real device.
 *
 * Reuses existing LLM-Hub services unchanged:
 *   - [InferenceService]  → any of the 4 inference backends
 *   - [RagService]        → optional, passes document context into the prompt
 *   - [WebSearchService]  → optional, gated by [SearchIntentDetector]
 */
class VoicePipeline(
    private val scope: CoroutineScope,
    private val inference: InferenceService,
    private val model: LLMModel,
    private val stt: SpeechToText,
    private val tts: Tts,
    private val sink: AudioSink,
    private val chatId: String = "mimobot",
    private val rag: RagService? = null,
    private val webSearch: WebSearchService? = DuckDuckGoSearchService(),
    private val webSearchEnabled: Boolean = true,
) {

    enum class State { IDLE, LISTENING, THINKING, SPEAKING }

    private val _state = MutableStateFlow(State.IDLE)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _lastTranscript = MutableStateFlow("")
    val lastTranscript: StateFlow<String> = _lastTranscript.asStateFlow()

    private val _lastResponse = MutableStateFlow("")
    val lastResponse: StateFlow<String> = _lastResponse.asStateFlow()

    private var currentTurn: Job? = null
    @Volatile private var cancelRequested = false

    /** Run one listen → think → speak turn. Safe to call only when state == IDLE. */
    fun startTurn() {
        if (currentTurn?.isActive == true) return
        cancelRequested = false
        currentTurn = scope.launch(Dispatchers.Default) {
            try {
                runTurn()
            } catch (t: Throwable) {
                Log.e(TAG, "turn failed", t)
            } finally {
                _state.value = State.IDLE
            }
        }
    }

    /** Cancel an in-flight turn. Mic, LLM stream, and TTS are all torn down. */
    fun cancel() {
        cancelRequested = true
        stt.cancel()
        tts.stop()
        sink.stop()
        currentTurn?.cancel()
        currentTurn = null
        _state.value = State.IDLE
    }

    private suspend fun runTurn() {
        _state.value = State.LISTENING
        val transcript = stt.recognizeTurn().trim()
        if (cancelRequested) return
        _lastTranscript.value = transcript
        if (transcript.isBlank()) {
            Log.d(TAG, "empty transcript, abandoning turn")
            return
        }

        _state.value = State.THINKING
        val prompt = augmentPrompt(transcript)

        _state.value = State.SPEAKING
        _lastResponse.value = ""
        sink.start()

        // Sentence channel decouples LLM streaming from TTS synthesis so the first
        // audio frame starts rendering before the model is done generating.
        val sentences = Channel<String>(Channel.UNLIMITED)
        val speakJob = scope.launch(Dispatchers.IO) {
            for (sentence in sentences) {
                if (cancelRequested) break
                try {
                    tts.speakToPcm(sentence).collect { frame ->
                        if (cancelRequested) return@collect
                        sink.write(frame)
                    }
                } catch (t: Throwable) {
                    Log.w(TAG, "tts emit failed: ${t.message}")
                }
            }
        }

        try {
            val buf = StringBuilder()
            inference.generateResponseStream(prompt, model).collect { token ->
                if (cancelRequested) return@collect
                buf.append(token)
                _lastResponse.value = buf.toString()
                while (true) {
                    val end = findSentenceEnd(buf)
                    if (end < 0) break
                    val sentence = buf.substring(0, end + 1).trim()
                    buf.delete(0, end + 1)
                    if (sentence.isNotEmpty()) sentences.send(sentence)
                }
            }
            val tail = buf.toString().trim()
            if (tail.isNotEmpty() && !cancelRequested) sentences.send(tail)
        } finally {
            sentences.close()
            speakJob.join()
            sink.stop()
        }
    }

    /** Find the index of the first sentence-ending char in [buf], or -1. */
    private fun findSentenceEnd(buf: StringBuilder): Int {
        // Allow a short minimum so "Hi." still gets spoken, but prefer longer chunks
        // so we don't chop in the middle of a short phrase.
        if (buf.length < 12) return -1
        for (i in buf.indices) {
            val c = buf[i]
            if (c == '.' || c == '!' || c == '?' || c == '\n') return i
        }
        return -1
    }

    private suspend fun augmentPrompt(userQuery: String): String {
        val sb = StringBuilder()

        rag?.let { r ->
            val chunks = r.searchRelevantContext(
                chatId = chatId,
                query = userQuery,
                maxResults = 3,
                relaxedLexicalFallback = false,
                queryEmbedding = null,
            )
            if (chunks.isNotEmpty()) {
                sb.append("CONTEXT FROM YOUR DOCUMENTS:\n")
                chunks.forEach { sb.append("- ").append(it.content).append('\n') }
                sb.append('\n')
            }
        }

        if (webSearchEnabled && webSearch != null &&
            SearchIntentDetector.needsWebSearch(userQuery)
        ) {
            try {
                val q = SearchIntentDetector.extractSearchQuery(userQuery)
                val results = webSearch.search(q, maxResults = 3)
                if (results.isNotEmpty()) {
                    sb.append("CURRENT WEB SEARCH RESULTS:\n")
                    results.forEach { r ->
                        sb.append("- ").append(r.title).append(": ").append(r.snippet).append('\n')
                    }
                    sb.append('\n')
                }
            } catch (t: Throwable) {
                Log.w(TAG, "web search failed: ${t.message}")
            }
        }

        sb.append("system: You are a small voice companion. Answer in one or two short sentences. Never output Markdown, code blocks, or emoji.\n\n")
        sb.append("user: ").append(userQuery)
        return sb.toString()
    }

    companion object { private const val TAG = "MimoVoicePipeline" }
}
