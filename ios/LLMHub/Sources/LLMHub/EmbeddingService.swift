import Foundation
import RunAnywhere

// MARK: - EmbeddingService
// Wraps the RunAnywhere embeddings lifecycle and keeps the app-facing API as
// plain float vectors for the in-memory RAG service.

actor EmbeddingService {

    // MARK: - State

    private(set) var isInitialized: Bool = false
    private(set) var currentModelID: String? = nil
    private(set) var currentModelName: String? = nil
    private(set) var embeddingDimension: Int = 0

    // MARK: - Init

    init() {}

    // MARK: - Lifecycle

    /// Initialize the embedding service and load a GGUF embedding model.
    func initialize(modelID: String, modelPath _: String, modelName: String) async throws {
        if currentModelID != modelID {
            await cleanup()
        }

        let testResult = try await RunAnywhere.embeddings.embed("test", modelID: modelID)
        guard let vector = testResult.vectors.first, !vector.values.isEmpty else {
            throw EmbeddingError.modelLoadFailed("empty test embedding")
        }

        isInitialized = true
        currentModelID = modelID
        currentModelName = modelName
        embeddingDimension = testResult.dimension > 0 ? Int(testResult.dimension) : vector.values.count
    }

    func cleanup() async {
        if isInitialized {
            try? await RunAnywhere.embeddings.unload()
        }
        isInitialized = false
        currentModelID = nil
        currentModelName = nil
        embeddingDimension = 0
    }

    // MARK: - Embed

    /// Generate a dense float embedding for the given text.
    func embed(_ text: String) async throws -> [Float] {
        guard isInitialized, let modelID = currentModelID else {
            throw EmbeddingError.notInitialized
        }

        let trimmed = String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(1024))
        guard !trimmed.isEmpty else { return [] }

        let result = try await RunAnywhere.embeddings.embed(trimmed, modelID: modelID)
        guard let output = result.vectors.first?.values, !output.isEmpty else {
            throw EmbeddingError.embeddingFailed("empty result")
        }
        return output
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
