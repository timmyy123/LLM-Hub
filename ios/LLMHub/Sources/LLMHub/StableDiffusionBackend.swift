import Foundation
import CoreML
import MediaGenerationKit
import ModelZoo
#if canImport(UIKit)
import UIKit
#endif

// MARK: - SDError (always compiled)

enum SDError: LocalizedError {
    case notDrawThingsModel
    case modelNotDownloaded
    case pipelineNotLoaded
    case imageToImageUnsupported
    case unavailable

    var errorDescription: String? {
        switch self {
        case .notDrawThingsModel: return "Not a Draw Things image generation model."
        case .modelNotDownloaded: return "Model files not found. Please download the model first."
        case .pipelineNotLoaded: return "Image generation pipeline is not loaded."
        case .imageToImageUnsupported: return "Image-to-image is not supported by this model."
        case .unavailable: return "Image generation is not available in this build."
        }
    }
}

// MARK: - StableDiffusionBackend

@MainActor
final class StableDiffusionBackend: ObservableObject {
    static let shared = StableDiffusionBackend()

    @Published var isLoaded = false
    @Published var isLoading = false
    @Published var isGenerating = false
    @Published var generationStep = 0
    @Published var generationTotalSteps = 20
    @Published var loadedModelId: String? = nil

    private var pipeline: MediaGenerationPipeline?

    private init() {}

    // MARK: - SD Model Directory Helpers

    static func isModelDownloaded(modelId: String) -> Bool {
        guard let model = ModelData.allModels().first(where: { $0.id == modelId }) else { return false }
        return ModelData.isModelFullyAvailableLocally(model)
    }

    static func supportsImageToImage(modelId: String) -> Bool {
        return true
    }

    func loadModel(_ model: AIModel) async throws {
        guard model.isDrawThingsImageGeneration else { throw SDError.notDrawThingsModel }
        guard Self.isModelDownloaded(modelId: model.id) else { throw SDError.modelNotDownloaded }

        if loadedModelId == model.id && isLoaded { return }

        isLoading = true
        isLoaded = false
        pipeline = nil
        loadedModelId = nil

        let modelsDirectory = try drawThingsModelsDirectory()

        do {
            let loadedPipeline = try await MediaGenerationPipeline.fromPretrained(
                model.id,
                backend: .local(directory: modelsDirectory.path)
            )
            self.pipeline = loadedPipeline
            self.loadedModelId = model.id
            self.isLoaded = true
        } catch {
            isLoading = false
            throw error
        }
        isLoading = false
    }

    func unloadModel() {
        pipeline = nil
        loadedModelId = nil
        isLoaded = false
        isGenerating = false
    }

    func generateImage(
        prompt: String,
        steps: Int,
        seed: UInt32,
        inputImage: UIImage? = nil,
        denoiseStrength: Float = 0.7
    ) async throws -> UIImage? {
        guard isLoaded, let pipeline = pipeline else { throw SDError.pipelineNotLoaded }

        generationStep = 0
        generationTotalSteps = steps
        isGenerating = true
        defer { isGenerating = false }

        var configuredPipeline = pipeline
        var config = pipeline.configuration
        config.steps = steps
        config.seed = seed
        if inputImage != nil {
            config.strength = denoiseStrength
        }
        configuredPipeline.configuration = config
        self.pipeline = configuredPipeline

        let inputs: [MediaGenerationPipeline.Input] = inputImage.map { [$0] } ?? []
        let results = try await configuredPipeline.generate(
            prompt: prompt,
            negativePrompt: "ugly, blurry, bad anatomy, bad quality",
            inputs: inputs
        ) { @Sendable state in
            if case .generating(let step, let total) = state {
                Task { @MainActor in
                    StableDiffusionBackend.shared.generationStep = step
                    StableDiffusionBackend.shared.generationTotalSteps = total
                }
            }
        }

        guard let result = results.first else { return nil }
        return UIImage(result)
    }

    func cancelGeneration() {
        isGenerating = false
    }

    private func drawThingsModelsDirectory() throws -> URL {
        return ModelZoo.persistentModelsDirectory()
    }
}
