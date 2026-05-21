import Foundation
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

    private init() {}

    // MARK: - Model Lifecycle

    /// Load a .litertlm model file from disk, initialise GPU engine with MTP.
    func loadModel(at path: String, supportsVision: Bool) async throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw LiteRTLMError.modelFileNotFound(path)
        }

        // Unload any existing engine first
        await unload()

        print("ℹ️ [LiteRTLMBackend] loadModel path=\(path) vision=\(supportsVision)")

        // Enable Multi-Token Prediction speculative decoding for GPU speed boost
        ExperimentalFlags.optIntoExperimentalAPIs()
        ExperimentalFlags.enableSpeculativeDecoding = true

        let config = try EngineConfig(
            modelPath: path,
            backend: .gpu,
            visionBackend: supportsVision ? .cpu() : nil,
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
        systemPrompt: String?,
        temperature: Float,
        topK: Int,
        topP: Float,
        maxTokens: Int,
        onUpdate: @escaping (String, Int, Double) -> Void
    ) async throws {
        guard let engine else {
            throw LiteRTLMError.engineNotLoaded
        }

        // Build sampler
        let samplerConfig = try SamplerConfig(
            topK: topK,
            topP: topP,
            temperature: temperature
        )

        // Build conversation config with optional system message
        let conversationConfig: ConversationConfig
        if let systemPrompt, !systemPrompt.isEmpty {
            conversationConfig = ConversationConfig(
                systemMessage: Message(systemPrompt),
                samplerConfig: samplerConfig
            )
        } else {
            conversationConfig = ConversationConfig(samplerConfig: samplerConfig)
        }

        let conversation = try await engine.createConversation(with: conversationConfig)

        // Build user message — interleave image + text if vision provided
        let message: Message
        if let imageURL {
            message = Message(contents: [
                Content.imageFile(imageURL.path),
                Content.text(prompt)
            ])
        } else {
            message = Message(prompt)
        }

        // Stream response
        var currentOutput = ""
        for try await chunk in conversation.sendMessageStream(message) {
            try Task.checkCancellation()
            currentOutput += chunk.toString
            onUpdate(currentOutput, 0, 0)
        }

        // Final update with accumulated text
        onUpdate(currentOutput, 0, 0)
        print("✅ [LiteRTLMBackend] generation complete chars=\(currentOutput.count)")
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
