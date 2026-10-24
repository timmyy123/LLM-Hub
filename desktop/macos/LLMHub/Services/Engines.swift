import Foundation
import FoundationModels
import LiteRTLM
import Observation

/// Loads a model on the right engine (llama.cpp for GGUF, LiteRT-LM for .litertlm, Apple Intelligence for the
/// system model) and streams replies. Chat and the Local API both go through here.
@MainActor @Observable
final class EngineHub {
    let llama: LlamaEngine
    private let liteRt = LiteRtEngine()
    private(set) var liteRtLoaded: InstalledModel?
    private(set) var appleActive = false
    private var activeFormat = "gguf"
    private var overrideStatus: (EngineState, String)?

    init(llama: LlamaEngine) { self.llama = llama }

    var state: EngineState { activeFormat == "gguf" ? llama.state : overrideStatus?.0 ?? .idle }
    var statusText: String { activeFormat == "gguf" ? llama.statusText : overrideStatus?.1 ?? L.s("EngineIdle") }
    var loadedModel: InstalledModel? {
        switch activeFormat {
        case "litertlm": return liteRtLoaded
        case "apple": return appleActive ? AppleModel.installed : nil
        default: return llama.loadedModel
        }
    }

    var backend: String {
        switch activeFormat {
        case "litertlm": return "LiteRT-LM \(liteRt.backend ?? "")"
        case "apple": return "Apple Intelligence"
        default: return "llama.cpp Metal"
        }
    }

    func isReady(_ model: InstalledModel, contextLength: Int? = nil) -> Bool {
        switch model.format {
        case "litertlm": return activeFormat == "litertlm" && liteRtLoaded?.id == model.id && liteRt.loadedWithMtp == model.mtpEnabled
        case "apple": return activeFormat == "apple" && appleActive
        default:
            let wanted = LlamaEngine.context(for: model, wanted: contextLength ?? AppSettings.current.contextLength)
            return activeFormat == "gguf" && llama.loadedModel?.id == model.id && llama.baseURL != nil && llama.loadedContext >= wanted
        }
    }

    /// <paramref name="contextLength"/>: a larger window than the chat setting (Agent, Design, Local API); llama.cpp only.
    func load(_ model: InstalledModel, contextLength: Int? = nil) async throws {
        if isReady(model, contextLength: contextLength) { return }
        switch model.format {
        case "litertlm":
            llama.stop()
            appleActive = false
            activeFormat = "litertlm"
            overrideStatus = (.loading, L.f("LoadingModel", model.name))
            do {
                try await liteRt.load(model)
                liteRtLoaded = model
                overrideStatus = (.ready, L.f("EngineReady", model.name))
            } catch {
                overrideStatus = (.error, L.f("EngineError", error.localizedDescription))
                throw error
            }
        case "apple":
            llama.stop()
            liteRt.unload()
            liteRtLoaded = nil
            activeFormat = "apple"
            if let reason = AppleModel.unavailableReason {
                overrideStatus = (.error, L.f("AppleModelUnavailable", reason))
                throw NSError(domain: "LLMHub", code: 20, userInfo: [NSLocalizedDescriptionKey: reason])
            }
            appleActive = true
            overrideStatus = (.ready, L.f("EngineReady", L.s("AppleModelName")))
        default:
            liteRt.unload()
            liteRtLoaded = nil
            appleActive = false
            activeFormat = "gguf"
            try await llama.load(model, contextLength: contextLength)
        }
    }

    func stream(_ model: InstalledModel, turns: [ChatTurn], options: ChatOptions = .init()) -> AsyncThrowingStream<ChatClient.Delta, Error> {
        switch model.format {
        case "litertlm": return liteRt.stream(turns: turns, options: options)
        case "apple": return AppleModel.stream(turns: turns)
        default: return ChatClient.stream(baseURL: llama.baseURL!, model: model.id, turns: turns, options: options)
        }
    }

    func stop() {
        llama.stop()
        liteRt.unload()
        liteRtLoaded = nil
        appleActive = false
        overrideStatus = nil
        activeFormat = "gguf"
    }
}

/// Runs .litertlm models through Google's LiteRT-LM (same Swift package as the iOS app). Metal GPU first, CPU fallback.
@MainActor
final class LiteRtEngine {
    private var engine: Engine?
    private(set) var backend: String?
    private(set) var loadedWithMtp = false
    private var model: InstalledModel?
    private var hasVision = false

    func load(_ model: InstalledModel) async throws {
        unload()
        let cache = model.folder.appendingPathComponent("cache", isDirectory: true)
        try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let tokens = min(max(model.contextLength, 4096), AppSettings.current.contextLength)
        var lastError: Error?
        // Speculative decoding does nothing for a file without an MTP drafter; learn that once per model.
        var model = model
        if model.hasMtpDrafter == nil {
            let has = ModelInfo(modelPath: model.primaryPath.path)?.llm?.hasSpeculativeDecodingSupport() == true
            model.hasMtpDrafter = has
            AppModel.shared.modelStore.setHasMtpDrafter(has, for: model.id)
            Log.info("LiteRT-LM \(model.id): file \(has ? "has" : "has no") MTP drafter")
        }
        let fileHasMtp = model.mtpEnabled
        // Vision models also get the image encoder (chat attachments); without it if that cannot start.
        // (Google's "-gpu" Gemma 4 build ships no vision encoder, and a conversation then fails, so ask the file.)
        let wantsVision = model.tags.contains("vision") && ModelInfo(modelPath: model.primaryPath.path)?.inputModalities.vision == true
        for backend in [Backend.gpu, Backend.cpu()] {
            for vision in wantsVision ? [true, false] : [false] {
                do {
                    // MTP (speculative decoding) only on the GPU, and only for files that ship a drafter; it throws otherwise.
                    let mtp = fileHasMtp && backend == .gpu
                    ExperimentalFlags.optIntoExperimentalAPIs()
                    ExperimentalFlags.enableSpeculativeDecoding = mtp ? true : nil
                    // Real decode tokens/s for the reply footer (stream chunks can hold several tokens, especially with MTP).
                    ExperimentalFlags.enableBenchmark = true
                    let config = try EngineConfig(modelPath: model.primaryPath.path, backend: backend, visionBackend: vision ? backend : nil,
                                                  maxNumTokens: tokens, cacheDir: cache.path)
                    let engine = Engine(engineConfig: config)
                    try await engine.initialize()
                    self.engine = engine
                    self.backend = backend.rawValue
                    self.model = model
                    hasVision = vision
                    loadedWithMtp = model.mtpEnabled
                    Log.info("LiteRT-LM loaded \(model.id) on \(backend.rawValue)\(mtp ? " with MTP" : "")\(vision ? " with vision" : "")")
                    return
                } catch {
                    Log.error("LiteRT-LM could not start \(model.id) on \(backend.rawValue)\(vision ? " with vision" : "")", error)
                    lastError = error
                }
            }
        }
        throw lastError ?? NSError(domain: "LLMHub", code: 21)
    }

    func unload() {
        engine = nil
        backend = nil
        model = nil
        hasVision = false
    }

    /// A turn as a LiteRT-LM message: attached images (when the vision encoder is loaded) before the text.
    private func message(_ turn: ChatTurn, role: Role) -> Message {
        let images = hasVision ? turn.images.map { Content.imageFile($0.path) } : []
        return images.isEmpty ? Message(turn.content, role: role) : Message(contents: images + [.text(turn.content)], role: role)
    }

    /// Extended thinking: Gemma 4 12B uses the engine's thinking switch; smaller Gemma 4 models the <|think|> system tag,
    /// which keeps MTP on the main token stream (as in the iOS app). Reasoning arrives on the "thought" channel.
    func stream(turns: [ChatTurn], options: ChatOptions = .init()) -> AsyncThrowingStream<ChatClient.Delta, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    guard let engine else { throw NSError(domain: "LLMHub", code: 22, userInfo: [NSLocalizedDescriptionKey: "No LiteRT-LM model loaded"]) }
                    let id = (self.model?.catalogId ?? self.model?.id ?? "").lowercased()
                    let nativeThinking = options.thinking && id.contains("12b")
                    let tagThinking = options.thinking && !nativeThinking
                    // Every system note (instructions, web search results) goes into the one system message.
                    var systemText = turns.filter { $0.role == "system" }.map(\.content).joined(separator: "\n\n")
                    if tagThinking { systemText = systemText.isEmpty ? "<|think|>" : "<|think|>\n" + systemText }
                    let system = systemText.isEmpty ? nil : Message(systemText, role: .system)
                    let dialog = turns.filter { $0.role != "system" }
                    let history = dialog.dropLast().map { message($0, role: $0.role == "assistant" ? .model : .user) }
                    let conversation = try await engine.createConversation(with: ConversationConfig(
                        systemMessage: system, initialMessages: Array(history),
                        thinkingConfig: nativeThinking ? ThinkingConfig(enableThinking: true) : nil))
                    var firstToken: Date?
                    var pieces = 0
                    let last = dialog.last ?? ChatTurn(role: "user", content: "")
                    for try await chunk in conversation.sendMessageStream(message(last, role: .user)) {
                        try Task.checkCancellation()
                        let thought = options.thinking ? chunk.channels["thought"] ?? "" : ""
                        let text = chunk.toString
                        if !thought.isEmpty || !text.isEmpty {
                            firstToken = firstToken ?? Date()
                            pieces += 1
                            continuation.yield(.init(content: text.isEmpty ? nil : text, reasoning: thought.isEmpty ? nil : thought))
                        }
                    }
                    // What the user feels: decoded tokens over the wall time from first to last token.
                    // (The engine's own decode rate counts steps, and with MTP a step yields several tokens.)
                    let wall = firstToken.map { Date().timeIntervalSince($0) } ?? 0
                    let info = try? conversation.getBenchmarkInfo()
                    let tokens = info?.lastDecodeTokenCount ?? 0
                    Log.info("LiteRT-LM decode: \(tokens) tokens, \(pieces) chunks, \(String(format: "%.2f", wall))s wall, engine rate \(String(format: "%.1f", info?.lastDecodeTokensPerSecond ?? 0)) tok/s")
                    // A reply that arrives in one or two chunks has no meaningful wall time; use the engine's figure then.
                    if pieces >= 3, tokens > 1, wall > 0 {
                        continuation.yield(.init(tokensPerSecond: Double(tokens - 1) / wall))
                    } else if let rate = info?.lastDecodeTokensPerSecond, rate > 0 {
                        continuation.yield(.init(tokensPerSecond: rate))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

/// Apple's on-device foundation model (Apple Intelligence), offered like an installed model that needs no download.
enum AppleModel {
    static let id = "apple.foundation.system"

    @MainActor static var installed: InstalledModel {
        InstalledModel(id: id, name: L.s("AppleModelName"), catalogId: "apple-foundation", repo: "Apple", format: "apple", quant: "on device",
                       files: ["system"], mmproj: nil, sizeBytes: 0, contextLength: SystemLanguageModel.default.contextSize,
                       tags: [], installedAt: .distantPast)
    }

    /// Nil when available; otherwise why not (not eligible, Apple Intelligence off, model still downloading).
    @MainActor static var unavailableReason: String? {
        switch SystemLanguageModel.default.availability {
        case .available: return nil
        case .unavailable(let reason):
            switch reason {
            case .deviceNotEligible: return "this Mac does not support Apple Intelligence"
            case .appleIntelligenceNotEnabled: return "turn on Apple Intelligence in System Settings"
            case .modelNotReady: return "the model is still downloading"
            @unknown default: return "\(reason)"
            }
        }
    }

    /// History is folded into the prompt, as in the iOS app; each request uses a fresh session.
    static func stream(turns: [ChatTurn]) -> AsyncThrowingStream<ChatClient.Delta, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @MainActor in
                do {
                    let system = turns.first { $0.role == "system" }?.content
                    let dialog = turns.filter { $0.role != "system" }
                    let prompt = dialog.count == 1 ? dialog[0].content
                        : dialog.map { "\($0.role == "assistant" ? "Assistant" : "User"): \($0.content)" }.joined(separator: "\n\n") + "\n\nAssistant:"
                    let session = LanguageModelSession(model: .default, instructions: system)
                    var sent = ""
                    let start = Date()
                    for try await snapshot in session.streamResponse(to: prompt) {
                        try Task.checkCancellation()
                        // Snapshots are cumulative; pass on only the new part.
                        let full = snapshot.content
                        if full.hasPrefix(sent) {
                            let delta = String(full.dropFirst(sent.count))
                            if !delta.isEmpty { continuation.yield(.init(content: delta)) }
                        }
                        sent = full
                    }
                    let seconds = Date().timeIntervalSince(start)
                    if seconds > 0, !sent.isEmpty { continuation.yield(.init(tokensPerSecond: Double(sent.count) / 4 / seconds)) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
