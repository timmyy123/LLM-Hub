import Foundation
import MagentaRuntime

// MARK: - EmbeddingService
// Runs the downloaded EmbeddingGemma .tflite file with LiteRT.

actor EmbeddingService {

    // MARK: - State

    private(set) var isInitialized: Bool = false
    private(set) var currentModelID: String? = nil
    private(set) var currentModelName: String? = nil
    private(set) var embeddingDimension: Int = 0
    private var model: LiteRTEmbeddingModel?

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

    func cleanup() async {
        model = nil
        isInitialized = false
        currentModelID = nil
        currentModelName = nil
        embeddingDimension = 0
    }

    // MARK: - Embed

    /// Generate a dense float embedding for the given text.
    func embed(_ text: String, isQuery: Bool = false) async throws -> [Float] {
        guard isInitialized, let model else {
            throw EmbeddingError.notInitialized
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        return try model.embed(trimmed, isQuery: isQuery)
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
