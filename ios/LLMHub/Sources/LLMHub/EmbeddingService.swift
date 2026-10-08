import Foundation
import MagentaRuntime
@preconcurrency import LiteRTLM

// MARK: - EmbeddingService
// Runs EmbeddingGemma either as a raw .tflite file with LiteRT, or as a .litertlm bundle
// (EmbeddingGemma 2, multimodal) with the LiteRT-LM EmbeddingEngine.

enum MediaEmbedPart: Sendable {
    case text(String)
    case image(Data)
    case audio(Data)
}

actor EmbeddingService {

    // EmbeddingGemma 2 prompt format from the model card.
    private static let queryPrefix = "task: search result | query: "
    private static let documentPrefix = "title: none | text: "
    private static let maxTextChars = 6000

    // MARK: - State

    private(set) var isInitialized: Bool = false
    private(set) var currentModelID: String? = nil
    private(set) var currentModelName: String? = nil
    private(set) var embeddingDimension: Int = 0
    private(set) var supportsImage: Bool = false
    private(set) var supportsAudio: Bool = false
    /// "GPU" or "CPU" once a LiteRT-LM engine is loaded.
    private(set) var activeBackendLabel: String? = nil
    private var lmEngineBackendLabel: String? = nil
    private var visionTokensPerImage: Int? = nil
    private var model: LiteRTEmbeddingModel?
    private var lmEngine: LiteRTLM.EmbeddingEngine?

    // MARK: - Init

    init() {}

    // MARK: - Lifecycle

    /// Compile the selected EmbeddingGemma LiteRT model once.
    func initialize(modelID: String, modelPath: String, modelName: String) async throws {
        if currentModelID != modelID {
            await cleanup()
        }
        let loaded = try LiteRTEmbeddingModel(modelURL: URL(fileURLWithPath: modelPath))
        guard loaded.sequenceLength > 2, loaded.dimension > 0 else {
            throw EmbeddingError.modelLoadFailed("invalid embedding tensor shape")
        }

        model = loaded
        isInitialized = true
        currentModelID = modelID
        currentModelName = modelName
        embeddingDimension = loaded.dimension
    }

    /// Load a .litertlm embedding bundle, preferring GPU and falling back to CPU.
    func initializeLiteRTLM(
        modelID: String,
        modelPath: String,
        modelName: String,
        cacheDir: String?,
        supportsImage: Bool = false,
        supportsAudio: Bool = false,
        visionTokensPerImage: Int? = nil,
        maxInputLength: Int? = nil
    ) async throws {
        await cleanup()

        let candidates: [(backend: LiteRTLM.Backend, vision: LiteRTLM.Backend?)] = [
            (.gpu, supportsImage ? .gpu : nil),
            (.cpu(), supportsImage ? .cpu() : nil),
        ]
        var lastError: Error?
        for candidate in candidates {
            let config = LiteRTLM.EmbeddingEngineConfig(
                modelPath: modelPath,
                backend: candidate.backend,
                visionBackend: candidate.vision,
                audioBackend: supportsAudio ? .cpu() : nil,
                cacheDir: cacheDir,
                maxInputLength: maxInputLength,
                visionTokensPerImage: visionTokensPerImage
            )
            let engine = LiteRTLM.EmbeddingEngine(config: config)
            do {
                try await engine.initialize()
                let probe = try await engine.computeEmbedding(
                    contents: [.text(Self.queryPrefix + "hello")],
                    options: LiteRTLM.EmbeddingOptions(normalize: true)
                )
                guard !probe.embedding.isEmpty else {
                    throw EmbeddingError.modelLoadFailed("empty probe embedding")
                }
                lmEngine = engine
                lmEngineBackendLabel = candidate.backend == .gpu ? "GPU" : "CPU"
                embeddingDimension = probe.embedding.count
                print("✅ [Embedding] LiteRT-LM engine ready on \(candidate.backend) dim=\(embeddingDimension)")
                break
            } catch {
                print("⚠️ [Embedding] LiteRT-LM init on \(candidate.backend) failed: \(error.localizedDescription)")
                await engine.close()
                lastError = error
            }
        }
        guard lmEngine != nil else {
            throw EmbeddingError.modelLoadFailed(lastError?.localizedDescription ?? "no backend available")
        }

        isInitialized = true
        currentModelID = modelID
        currentModelName = modelName
        self.supportsImage = supportsImage
        self.supportsAudio = supportsAudio
        self.visionTokensPerImage = visionTokensPerImage
        activeBackendLabel = lmEngineBackendLabel
    }

    func cleanup() async {
        model = nil
        if let lmEngine {
            await lmEngine.close()
        }
        lmEngine = nil
        isInitialized = false
        currentModelID = nil
        currentModelName = nil
        embeddingDimension = 0
        supportsImage = false
        supportsAudio = false
        activeBackendLabel = nil
        lmEngineBackendLabel = nil
        visionTokensPerImage = nil
    }

    // MARK: - Embed

    /// Generate a dense float embedding for the given text.
    func embed(_ text: String, isQuery: Bool = false) async throws -> [Float] {
        guard isInitialized else {
            throw EmbeddingError.notInitialized
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        if let lmEngine {
            let prefixed = (isQuery ? Self.queryPrefix : Self.documentPrefix) + String(trimmed.prefix(Self.maxTextChars))
            return try await lmEngine.computeEmbedding(
                contents: [.text(prefixed)],
                options: LiteRTLM.EmbeddingOptions(normalize: true)
            ).embedding
        }
        guard let model else { throw EmbeddingError.notInitialized }
        return try model.embed(trimmed, isQuery: isQuery)
    }

    /// Embed a JPEG/PNG image into the same vector space as text queries.
    func embedImage(_ data: Data) async throws -> [Float]? {
        try await embedImages([data])
    }

    /// One embedding for several frames, so a video's first and last frame share a vector.
    func embedImages(_ frames: [Data]) async throws -> [Float]? {
        guard isInitialized, supportsImage, let lmEngine, !frames.isEmpty else { return nil }
        return try await lmEngine.computeEmbedding(
            contents: frames.map { .imageData($0) },
            options: LiteRTLM.EmbeddingOptions(normalize: true, visionTokensPerImage: visionTokensPerImage)
        ).embedding
    }

    /// One embedding for a Video Moment Finder window: timestamps, audio slices, and frames.
    func embedMixed(_ parts: [MediaEmbedPart]) async throws -> [Float]? {
        guard isInitialized, let lmEngine, !parts.isEmpty else { return nil }
        let contents: [LiteRTLM.Content] = parts.map { part in
            switch part {
            case .text(let text): return .text(text)
            case .image(let data): return .imageData(data)
            case .audio(let data): return .audioData(data)
            }
        }
        return try await lmEngine.computeEmbedding(
            contents: contents,
            options: LiteRTLM.EmbeddingOptions(normalize: true, visionTokensPerImage: visionTokensPerImage)
        ).embedding
    }

    /// Embed a 16 kHz mono WAV clip into the same vector space as text queries.
    func embedAudio(_ wav: Data) async throws -> [Float]? {
        guard isInitialized, supportsAudio, let lmEngine else { return nil }
        return try await lmEngine.computeEmbedding(
            contents: [.audioData(wav)],
            options: LiteRTLM.EmbeddingOptions(normalize: true)
        ).embedding
    }
}

// MARK: - Errors

enum EmbeddingError: Error, LocalizedError {
    case notInitialized
    case initFailed(String)
    case modelLoadFailed(String)
    case embeddingFailed(String)

    var errorDescription: String? {
        switch self {
        case .notInitialized: return "Embedding service not initialized."
        case .initFailed(let m): return "Embedding init failed: \(m)"
        case .modelLoadFailed(let m): return "Model load failed: \(m)"
        case .embeddingFailed(let m): return "Embedding failed: \(m)"
        }
    }
}
