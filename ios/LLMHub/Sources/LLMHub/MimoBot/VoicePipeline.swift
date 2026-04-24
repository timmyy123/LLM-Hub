import Foundation

/// Orchestrates the full voice loop, reusing existing LLM-Hub services:
///   - `LLMBackend`     ← LLM via RunAnywhere (llama.cpp / Apple Foundation)
///   - `RagService`     ← actor in RagService.swift, already present
///   - Web search       ← currently file-private inside `ChatScreen.swift`;
///                        see TODO below — either hoist it into its own file
///                        or re-implement a slim version here.
///
/// v0 flow (PTT):
///
///   transport.audioUp (Opus) → decoder → PCM ring buffer
///       ▲ user releases PTT              │
///       │                                 ▼
///   transport.control ("end") → Whisper → transcript
///                                           │
///                                           ▼
///                             augment(transcript):
///                                • RagService.search(chatId, query, …)
///                                • DuckDuckGo search (when intent matches)
///                                           │
///                                           ▼
///                             LLMBackend.shared.generate(prompt, …)
///                                (sentence-buffered → TTS on punctuation
///                                 boundary for low first-audio latency)
///                                           │
///                                           ▼
///                                    TTS.speakToPCM
///                                           │
///                                           ▼
///                                    OpusEncoderWrap
///                                           │
///                                           ▼
///                          transport.sendAudioDown(opusFrame)
///
/// TODO(pipeline): implement.
@MainActor
final class VoicePipeline {
    private let transport: MimoTransport
    private let whisper: WhisperSTT
    private let tts: TTS
    private let encoder = OpusEncoderWrap()
    private let decoder = OpusDecoderWrap()
    private let chatId: String
    private let rag: RagService?
    private let webSearchEnabled: Bool

    init(
        transport: MimoTransport,
        whisper: WhisperSTT,
        tts: TTS,
        chatId: String = "mimobot",
        rag: RagService? = nil,
        webSearchEnabled: Bool = true
    ) {
        self.transport = transport
        self.whisper = whisper
        self.tts = tts
        self.chatId = chatId
        self.rag = rag
        self.webSearchEnabled = webSearchEnabled
    }

    func start() {
        // TODO: spawn Tasks that drain transport.audioUp + transport.control,
        // then whisper → augmentPrompt → LLMBackend.shared.generate(...) →
        // sentence-buffered → tts.speakToPCM → encoder → transport.sendAudioDown.
    }

    /// Builds the prompt that goes to LLMBackend. Mirrors what the Android
    /// `VoicePipeline.augmentPrompt` does — inlines RAG context and web
    /// search results before the user's question.
    ///
    /// TODO(augment):
    ///   1. Hoist `WebSearchService` out of `ChatScreen.swift` into its own
    ///      file so this pipeline (and anything else headless) can import it.
    ///   2. Add a tiny iOS twin of `SearchIntentDetector` — or ship the
    ///      Android keyword list in a shared resource if we ever set up
    ///      shared-resource tooling.
    private func augmentPrompt(userQuery: String) async -> String {
        var sb = ""

        if let rag {
            // RagService.search returns [ContextChunk]; see RagService.swift
            let chunks = await rag.search(
                chatId: chatId,
                query: userQuery,
                queryEmbedding: nil,
                maxResults: 3
            )
            if !chunks.isEmpty {
                sb += "CONTEXT FROM YOUR DOCUMENTS:\n"
                for c in chunks { sb += "- \(c.content)\n" }
                sb += "\n"
            }
        }

        if webSearchEnabled {
            // TODO(augment-web): run DuckDuckGo search when intent matches.
            // Currently the iOS implementation lives inside ChatScreen.swift as
            // `fileprivate actor WebSearchService` — hoist it first.
        }

        sb += "system: You are a small voice companion. Answer in one or two short sentences. Never output Markdown, code blocks, or emoji.\n\n"
        sb += "user: \(userQuery)"
        return sb
    }

    func stop() {
        tts.stop()
        Task { await transport.disconnect() }
    }
}
