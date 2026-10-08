import Foundation
import MagentaRuntime
@preconcurrency import LiteRTLM

// MARK: - EmbeddingService
// Runs EmbeddingGemma either as a raw .tflite file with LiteRT, or as a .litertlm bundle
// (EmbeddingGemma 2, multimodal) with the LiteRT-LM EmbeddingEngine.

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
        supportsImage: Bool,
        supportsAudio: Bool
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
                cacheDir: cacheDir
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

    /// Embed an image (PNG/JPEG) or 16 kHz mono WAV, with an optional text note, into the same
    /// vector space as text. Returns nil when the loaded model can't embed that modality.
    func embedMedia(type: String, data: Data, note: String?) async throws -> [Float]? {
        guard isInitialized, let lmEngine else { return nil }
        var contents: [LiteRTLM.Content] = []
        if let note, !note.isEmpty {
            contents.append(.text(Self.documentPrefix + String(note.prefix(Self.maxTextChars))))
        }
        switch type {
        case MemoryMedia.typeImage where supportsImage:
            contents.append(.imageData(data))
        case MemoryMedia.typeAudio where supportsAudio:
            contents.append(.audioData(data))
        default:
            return nil
        }
        return try await lmEngine.computeEmbedding(
            contents: contents,
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
