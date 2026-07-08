import Foundation
import UIKit
#if canImport(LiteRTLM)
@preconcurrency import LiteRTLM

// MARK: - Errors

enum LiteRTLMError: LocalizedError {
    case engineNotLoaded
    case conversationNotCreated
    case modelFileNotFound(String)

    var errorDescription: String? {
        switch self {
        case .engineNotLoaded:
            return "LiteRT-LM engine is not loaded. Load a model first."
        case .conversationNotCreated:
            return "LiteRT-LM conversation not created. Call createConversation() first."
        case .modelFileNotFound(let path):
            return "LiteRT-LM model file not found at: \(path)"
        }
    }
}

// MARK: - LiteRTLMBackend

/// Wraps the Google LiteRT-LM Swift SDK.
/// Always uses GPU (Metal) backend for text generation.
/// Vision is handled via the built-in encoder (no separate mmproj needed).
@MainActor
final class LiteRTLMBackend {

    static let shared = LiteRTLMBackend()

    private var engine: Engine?
    private var loadedModelPath: String?
    /// The single active conversation. LiteRT-LM only allows one at a time.
    private var currentConversation: Conversation?
    /// Prevents re-entrancy bugs where a new session is created before the old one is destroyed.
    private var generationInProgress = false

    private init() {}

    // MARK: - Model Lifecycle

    /// Load a .litertlm model file from disk, initialise GPU engine with MTP.
    func loadModel(at path: String, supportsVision: Bool, supportsAudio: Bool, maxTokens: Int?) async throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw LiteRTLMError.modelFileNotFound(path)
        }

        // Unload any existing engine first
        await unload()

        print("ℹ️ [LiteRTLMBackend] loadModel path=\(path) vision=\(supportsVision) audio=\(supportsAudio) maxTokens=\(String(describing: maxTokens))")

        let isGemma4_12B = path.lowercased().hasSuffix(".litertlm") && (path.lowercased().contains("gemma-4-12b") || path.lowercased().contains("gemma4_12b"))
        ExperimentalFlags.optIntoExperimentalAPIs()
        ExperimentalFlags.enableSpeculativeDecoding = !isGemma4_12B
        ExperimentalFlags.enableBenchmark = true

        let config = try EngineConfig(
            modelPath: path,
            backend: .gpu,
            visionBackend: (supportsVision && !isGemma4_12B) ? .cpu() : nil,
            audioBackend: (supportsAudio && !isGemma4_12B) ? .cpu() : nil,
            maxNumTokens: maxTokens,
            cacheDir: liteRTCacheDir()
        )
        let eng = Engine(engineConfig: config)
        try await eng.initialize()

        self.engine = eng
        self.loadedModelPath = path
        print("✅ [LiteRTLMBackend] engine ready path=\(path)")
    }

    /// Unload the engine and release all resources.
    func unload() async {
        // Explicitly invalidate the conversation (calls litert_lm_conversation_delete
        // synchronously) before niling the reference, so the C session is freed immediately.
        currentConversation?.invalidate()
        currentConversation = nil
        engine = nil
        loadedModelPath = nil
        print("ℹ️ [LiteRTLMBackend] unloaded")
    }

    var isLoaded: Bool { engine != nil }
    var currentModelPath: String? { loadedModelPath }

    // MARK: - Generation

    /// Create a fresh Conversation and stream a response token by token.
    /// A new Conversation is created per call so the pre-formatted multi-turn
    /// prompt (already built by the caller) is sent as a single user message,
    /// matching the GGUF backend's existing prompt-formatting contract.
    func generateStream(
        prompt: String,
        imageURL: URL?,
        audioURL: URL?,
        systemPrompt: String?,
        temperature: Float,
        topK: Int,
        topP: Float,
        maxTokens: Int,
        useThinking: Bool,
        enableAgentTools: Bool,
        onUpdate: @escaping (String, Int, Double) -> Void
    ) async throws {
        guard let engine else {
            throw LiteRTLMError.engineNotLoaded
        }

        // Wait if a previous generation is still winding down (to avoid 'Session already exists' crashes)
        while generationInProgress {
            try await Task.sleep(nanoseconds: 50_000_000)
            try Task.checkCancellation()
        }
        generationInProgress = true
        defer { generationInProgress = false }

        // Build sampler
        let samplerConfig = try SamplerConfig(
            topK: topK,
            topP: topP,
            temperature: temperature
        )

        // Determine the final system prompt based on agent tools and thinking toggles
        let finalSystemPrompt: String?
        if enableAgentTools {
            let basePrompt = (systemPrompt != nil && !systemPrompt!.isEmpty) ? systemPrompt! : ChatAgentSkillsTools.AGENT_SYSTEM_PROMPT
            if useThinking {
                finalSystemPrompt = "<|think|>\n\(basePrompt)"
            } else {
                finalSystemPrompt = basePrompt
            }
        } else {
            if useThinking {
                if let systemPrompt, !systemPrompt.isEmpty {
                    finalSystemPrompt = "<|think|>\n\(systemPrompt)"
                } else {
                    finalSystemPrompt = "<|think|>"
                }
            } else {
                finalSystemPrompt = systemPrompt
            }
        }

        // LiteRT-LM supports only ONE session at a time.
        // Must call invalidate() EXPLICITLY (not just nil the Swift reference) because
        // the local `conversation` variable on this stack frame keeps the object alive
        // past the ARC deinit — causing FAILED_PRECONDITION on the next createConversation.
        if currentConversation != nil {
            currentConversation?.invalidate()  // synchronously calls litert_lm_conversation_delete
            currentConversation = nil
            print("ℹ️ [LiteRTLMBackend] invalidated previous conversation")
        }

        let conversation: Conversation
        if enableAgentTools {
            ExperimentalFlags.enableConversationConstrainedDecoding = true
            let config = ConversationConfig(
                systemMessage: finalSystemPrompt.map { Message($0) },
                tools: ChatAgentSkillsTools.allTools(),
                samplerConfig: samplerConfig
            )
            conversation = try await engine.createConversation(with: config)
            ExperimentalFlags.enableConversationConstrainedDecoding = false
        } else {
            let config = ConversationConfig(
                systemMessage: finalSystemPrompt.map { Message($0) },
                samplerConfig: samplerConfig
            )
            conversation = try await engine.createConversation(with: config)
        }
        currentConversation = conversation

        // Always release the conversation when generation ends (success, error, or cancellation).
        // Call invalidate() to synchronously free the C session — do NOT just nil the Swift ref,
        // because this defer runs while the local `conversation` var is still on the stack.
        // Also open any URL deferred by tools (Maps/Email/SMS) — they must NOT open mid-stream
        // because that sends the app to background and kills Metal GPU access.
        defer {
            currentConversation?.invalidate()  // frees litert_lm session NOW
            self.currentConversation = nil
            print("ℹ️ [LiteRTLMBackend] conversation invalidated after generation")

            // Open deferred URL now that GPU work is done
            if let urlToOpen = ChatAgentSkillsTools.deferredOpenURL {
                ChatAgentSkillsTools.deferredOpenURL = nil
                UIApplication.shared.open(urlToOpen)
                print("ℹ️ [LiteRTLMBackend] opened deferred URL post-generation: \(urlToOpen)")
            }
        }

        // Build user message — interleave image + audio + text if provided
        var contents: [Content] = []
        if let imageURL {
            contents.append(.imageFile(imageURL.path))
        }
        if let audioURL {
            contents.append(.audioFile(audioURL.path))
        }
        contents.append(.text(prompt))

        let message = Message(contents: contents)

        // Stream response
        var currentOutput = ""
        var sentThinkOpen = false
        var sentThinkClose = false
        var tokenCount = 0
        var generationStartTime: Date? = nil

        for try await chunk in conversation.sendMessageStream(message) {
            try Task.checkCancellation()
            tokenCount += 1
            if generationStartTime == nil {
                generationStartTime = Date()
            }

            let textChunk = chunk.toString
            let thinkingChunk = useThinking ? chunk.channels["thought"] : nil
            let isThinking = thinkingChunk != nil && !thinkingChunk!.isEmpty

            if useThinking && isThinking {
                if !sentThinkOpen {
                    currentOutput += "\u{200B}\u{200B}THINK\u{200B}\u{200B}"
                    sentThinkOpen = true
                }
                let cleanedThinking = processLlamaStopTokens(thinkingChunk!)
                if !cleanedThinking.isEmpty {
                    currentOutput += cleanedThinking
                }
            } else {
                let cleaned = processLlamaStopTokens(textChunk)
                if sentThinkOpen && !sentThinkClose {
                    currentOutput += "\u{200B}\u{200B}ENDTHINK\u{200B}\u{200B}"
                    sentThinkClose = true
                }
                if !cleaned.isEmpty {
                    currentOutput += cleaned
                }
            }
            let elapsed = generationStartTime != nil ? Date().timeIntervalSince(generationStartTime!) : 0.0
            let estimatedTokens = Double(currentOutput.count) / 3.5
            let tps = (elapsed > 0 && estimatedTokens > 0) ? (estimatedTokens / elapsed) : 0.0
            onUpdate(currentOutput, Int(estimatedTokens), tps)
        }

        var finalTokens = tokenCount
        var finalTps = 0.0
        
        let elapsed = generationStartTime != nil ? Date().timeIntervalSince(generationStartTime!) : 0.0
        if elapsed > 0 {
            let estimatedTokens = Double(currentOutput.count) / 3.5
            finalTps = estimatedTokens / elapsed
            finalTokens = Int(estimatedTokens)
        }
        
        if let benchmark = try? conversation.getBenchmarkInfo() {
            if benchmark.lastDecodeTokenCount > 0 {
                finalTokens = benchmark.lastDecodeTokenCount
            }
            if benchmark.lastDecodeTokensPerSecond > 0 {
                finalTps = benchmark.lastDecodeTokensPerSecond
            }
            print("📊 [LiteRTLMBackend] Benchmark metrics: prefillCount=\(benchmark.lastPrefillTokenCount) prefillTps=\(benchmark.lastPrefillTokensPerSecond) decodeCount=\(benchmark.lastDecodeTokenCount) decodeTps=\(benchmark.lastDecodeTokensPerSecond)")
        } else {
            print("⚠️ [LiteRTLMBackend] getBenchmarkInfo returned nil or threw")
        }

        if useThinking && sentThinkOpen && !sentThinkClose {
            currentOutput += "\u{200B}\u{200B}ENDTHINK\u{200B}\u{200B}"
            onUpdate(currentOutput, finalTokens, finalTps)
        } else {
            // Final update with accumulated text
            onUpdate(currentOutput, finalTokens, finalTps)
        }
        print("✅ [LiteRTLMBackend] generation complete chars=\(currentOutput.count)")
    }

    private func processLlamaStopTokens(_ text: String) -> String {
        let stopTokens = ["<|eot_id|>", "<|end_of_text|>", "<|end|>", "</s>"]
        var cleaned = text
        for stopToken in stopTokens {
            if let range = cleaned.range(of: stopToken) {
                cleaned = String(cleaned[..<range.lowerBound])
                break
            }
        }
        cleaned = cleaned
            .replacingOccurrences(of: "<|start_header_id|>", with: "")
            .replacingOccurrences(of: "<|end_header_id|>", with: "")
        return cleaned
    }

    // MARK: - Helpers

    /// Returns (and creates if needed) a writable cache directory for LiteRT-LM
    /// shader compilation artefacts.
    private func liteRTCacheDir() -> String {
        let cacheURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("LiteRTLM", isDirectory: true)
        try? FileManager.default.createDirectory(at: cacheURL, withIntermediateDirectories: true)
        return cacheURL.path
    }
}

#endif // canImport(LiteRTLM)
