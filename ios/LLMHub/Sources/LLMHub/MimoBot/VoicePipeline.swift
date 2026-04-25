import Foundation
import SwiftUI

/// Voice loop wiring: STT → LLM → TTS → AudioSink.
///
/// Transport-agnostic: pass a `SpeakerSink` for local dev mode or a future
/// `BLEAudioSink` for the real device. Reuses `LLMBackend.shared` for
/// generation (which itself wraps RunAnywhere + llama.cpp / Apple Foundation).
@MainActor
final class VoicePipeline: ObservableObject {

    enum State: String { case idle, listening, thinking, speaking }

    /// A single completed exchange.
    struct Turn: Identifiable, Equatable {
        let id = UUID()
        let user: String
        let assistant: String
    }

    @Published private(set) var state: State = .idle
    @Published private(set) var lastTranscript: String = ""
    @Published private(set) var lastResponse: String = ""
    @Published private(set) var history: [Turn] = []

    private var stt: SpeechToText
    private var tts: TTS
    private var sink: AudioSink
    private let rag: RagService?
    private let webSearchEnabled: Bool
    private let chatId: String
    private let maxHistoryTurns: Int

    private var currentTurn: Task<Void, Never>?

    init(
        stt: SpeechToText,
        tts: TTS,
        sink: AudioSink,
        chatId: String = "mimobot",
        rag: RagService? = nil,
        webSearchEnabled: Bool = true,
        maxHistoryTurns: Int = 6
    ) {
        self.stt = stt
        self.tts = tts
        self.sink = sink
        self.chatId = chatId
        self.rag = rag
        self.webSearchEnabled = webSearchEnabled
        self.maxHistoryTurns = maxHistoryTurns
    }

    /// Hot-swap the TTS impl. Stops any in-flight synth on the old one first.
    func setTTS(_ newTTS: TTS) {
        tts.stop()
        tts = newTTS
    }

    /// Wipe the conversation memory. Called from the test screen's Clear button.
    func clearHistory() { history = [] }

    func startTurn() {
        if currentTurn != nil { return }
        lastResponse = ""
        currentTurn = Task { [weak self] in
            guard let self else { return }
            await self.runTurn()
            await MainActor.run { self.currentTurn = nil; self.state = .idle }
        }
    }

    func cancel() {
        stt.cancel()
        tts.stop()
        sink.stop()
        currentTurn?.cancel()
        currentTurn = nil
        state = .idle
    }

    private func runTurn() async {
        state = .listening
        let transcript = await stt.recognizeTurn(languageHint: "en-US")
        if Task.isCancelled { return }
        lastTranscript = transcript
        if transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return }

        state = .thinking
        let prompt = await augmentPrompt(userQuery: transcript)

        state = .speaking
        sink.start()

        // Decouple LLM streaming from TTS synthesis so the first audio frame
        // starts rendering before the model is finished generating.
        let (sentences, sentenceCont) = AsyncStream<String>.makeStream()

        let speakTask = Task { [weak self] in
            guard let self else { return }
            for await sentence in sentences {
                if Task.isCancelled { break }
                for await frame in self.tts.speakToPCM(text: sentence, language: "en-US") {
                    if Task.isCancelled { break }
                    await self.sink.write(frame)
                }
            }
            await MainActor.run { self.sink.stop() }
        }

        // Bridge the callback-driven LLMBackend.generate API into an AsyncStream
        // so the rest of the loop can iterate over cumulative-text snapshots
        // without any @escaping mutable captures.
        let (tokenStream, tokenCont) = AsyncStream<String>.makeStream()
        let generateTask = Task.detached {
            do {
                try await LLMBackend.shared.generate(
                    prompt: prompt,
                    systemPrompt: nil,
                    onUpdate: { cumulativeText, _, _ in
                        tokenCont.yield(cumulativeText)
                    }
                )
            } catch {
                print("⚠️ VoicePipeline: LLM generate failed: \(error)")
            }
            tokenCont.finish()
        }

        var buf = ""
        var sent = 0  // chars of cumulative text already chunked
        var fullReply = ""  // running total for committing to history
        for await cumulative in tokenStream {
            if Task.isCancelled { break }
            if cumulative.count > sent {
                let idx = cumulative.index(cumulative.startIndex, offsetBy: sent)
                buf += String(cumulative[idx...])
                sent = cumulative.count
                lastResponse = cumulative
                fullReply = cumulative
                while let end = Self.findSentenceEnd(buf) {
                    let sentence = String(buf[..<buf.index(after: end)])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    buf.removeSubrange(..<buf.index(after: end))
                    if !sentence.isEmpty { sentenceCont.yield(sentence) }
                }
            }
        }
        await generateTask.value

        let tail = buf.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { sentenceCont.yield(tail) }
        sentenceCont.finish()
        await speakTask.value

        // Commit this turn to history once we've finished speaking it.
        let trimmedReply = fullReply.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedReply.isEmpty, !Task.isCancelled {
            var updated = history + [Turn(user: transcript, assistant: trimmedReply)]
            if updated.count > maxHistoryTurns {
                updated.removeFirst(updated.count - maxHistoryTurns)
            }
            history = updated
        }
    }

    private static func findSentenceEnd(_ s: String) -> String.Index? {
        guard s.count >= 12 else { return nil }
        return s.firstIndex { c in c == "." || c == "!" || c == "?" || c == "\n" }
    }

    private func augmentPrompt(userQuery: String) async -> String {
        var sb = ""

        if let rag {
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

        // TODO(augment-web): iOS web-search lives inside ChatScreen.swift as
        // a fileprivate actor; hoist it before wiring here. See TODO in
        // ios/LLMHub/Sources/LLMHub/MimoBot/README-style TODOs.
        _ = webSearchEnabled

        sb += "system: You are a small voice companion. Answer in one or two short sentences. Never output Markdown, code blocks, or emoji.\n\n"

        // Replay conversation memory so the model sees previous turns. Format
        // matches what LLMBackend.generate's prompt parser expects ("user: …"
        // / "assistant: …" lines) — Harmony rebuilding stays correct because
        // both pieces follow the same convention.
        for turn in history {
            sb += "user: \(turn.user)\n"
            sb += "assistant: \(turn.assistant)\n"
        }

        sb += "user: \(userQuery)"
        return sb
    }
}
