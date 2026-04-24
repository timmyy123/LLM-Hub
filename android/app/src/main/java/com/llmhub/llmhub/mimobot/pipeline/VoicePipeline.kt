package com.llmhub.llmhub.mimobot.pipeline

import com.llmhub.llmhub.data.LLMModel
import com.llmhub.llmhub.embedding.RagService
import com.llmhub.llmhub.inference.InferenceService
import com.llmhub.llmhub.mimobot.audio.OpusDecoder
import com.llmhub.llmhub.mimobot.audio.OpusEncoder
import com.llmhub.llmhub.mimobot.speech.Tts
import com.llmhub.llmhub.mimobot.speech.WhisperStt
import com.llmhub.llmhub.mimobot.transport.MimoTransport
import com.llmhub.llmhub.websearch.DuckDuckGoSearchService
import com.llmhub.llmhub.websearch.SearchIntentDetector
import com.llmhub.llmhub.websearch.WebSearchService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job

/**
 * Orchestrates the full voice loop, reusing existing LLM-Hub services:
 *   - [InferenceService]       ← LLM (MediaPipe / LiteRT / Nexa / ONNX)
 *   - [RagService]             ← in-memory semantic search over documents
 *   - [WebSearchService]       ← DuckDuckGo fallback
 *   - [SearchIntentDetector]   ← decides when a query warrants web search
 *
 * None of these services are changed. We just assemble a voice front end on
 * top and feed the augmented text prompt into generateResponseStream().
 *
 * v0 flow (PTT):
 *
 *   transport.audioUp  (Opus) ──► decoder ──► PCM buffer
 *       ▲ user releases PTT                     │
 *       │                                        ▼
 *   transport.control ("end") ───► Whisper ──► transcript
 *                                                  │
 *                                                  ▼
 *                                  augment(transcript) with:
 *                                     • RagService.searchRelevantContext(chatId, q)
 *                                     • WebSearchService.search(q)  (if intent detected)
 *                                                  │
 *                                                  ▼
 *                              InferenceService.generateResponseStream
 *                                  (sentence-buffered — emit to TTS on each
 *                                   punctuation boundary for low first-audio
 *                                   latency)
 *                                                  │
 *                                                  ▼
 *                                          Tts.speakToPcm
 *                                                  │
 *                                                  ▼
 *                                          OpusEncoder
 *                                                  │
 *                                                  ▼
 *                                   transport.sendAudioDown(opusFrame)
 *
 * TODO(pipeline): implement.
 *
 * Implementation notes:
 *   - Run the LLM stream and the TTS synth as two coroutines connected by a
 *     Channel<String> of sentence-sized chunks. This is what lets you start
 *     speaking before the model is done generating.
 *   - On `{"t":"barge_in"}` from the device, call tts.stop(), drop the LLM
 *     stream, and return to listening state.
 */
class VoicePipeline(
    private val scope: CoroutineScope,
    private val transport: MimoTransport,
    private val inference: InferenceService,
    private val whisper: WhisperStt,
    private val tts: Tts,
    private val model: LLMModel,
    private val chatId: String = "mimobot",
    // Optional capabilities — pass null to disable that feature in this session.
    private val rag: RagService? = null,
    private val webSearch: WebSearchService? = DuckDuckGoSearchService(),
    private val webSearchEnabled: Boolean = true,
) {
    private var loopJob: Job? = null
    private val encoder = OpusEncoder()
    private val decoder = OpusDecoder()

    fun start() {
        TODO("""
            1. collect transport.control → dispatch PTT / barge-in / end events
            2. collect transport.audioUp → decoder → PCM ring buffer
            3. on end: whisper.transcribe(pcm) → transcript
            4. augmentPrompt(transcript) (see helper below)
            5. inference.generateResponseStream(augmented, model)
               .bufferByPunctuation()
               .collect { sentence ->
                   tts.speakToPcm(sentence).collect { pcm ->
                       transport.sendAudioDown(encoder.encode(pcm))
                   }
               }
        """.trimIndent())
    }

    /**
     * Builds the text prompt fed to the LLM. Mirrors the logic in
     * InferenceService.generateResponseStreamWithSession (lines 745-795) but
     * for a headless voice pipeline — no Markdown, short responses, and
     * RAG/web results already inlined before the question reaches the model.
     */
    @Suppress("unused")
    private suspend fun augmentPrompt(userQuery: String): String {
        val sb = StringBuilder()

        // ── RAG: pull any pre-ingested document context
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

        // ── Web search: reuse SearchIntentDetector exactly like InferenceService does
        if (webSearchEnabled && webSearch != null &&
            SearchIntentDetector.needsWebSearch(userQuery)
        ) {
            val q = SearchIntentDetector.extractSearchQuery(userQuery)
            val results = webSearch.search(q, maxResults = 5)
            if (results.isNotEmpty()) {
                sb.append("CURRENT WEB SEARCH RESULTS:\n")
                results.forEach { r ->
                    sb.append("- ").append(r.title).append(": ").append(r.snippet).append('\n')
                }
                sb.append('\n')
            }
        }

        // Voice-specific system hint — keep responses short, no Markdown.
        sb.append("system: You are a small voice companion. Answer in one or two short sentences. Never output Markdown, code blocks, or emoji.\n\n")
        sb.append("user: ").append(userQuery)
        return sb.toString()
    }

    fun stop() {
        loopJob?.cancel()
        loopJob = null
        tts.stop()
        encoder.close()
        decoder.close()
    }
}
