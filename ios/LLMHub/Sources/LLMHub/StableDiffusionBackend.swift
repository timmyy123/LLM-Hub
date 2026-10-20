import Foundation
import CoreML
import MediaGenerationKit
@preconcurrency import ModelZoo
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
    private var encoderFingerprint = ""
    private var overriddenModelID: String?
    private var previousOverride: ModelZoo.Specification?

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

        let extras = ImageExtrasStore.shared.options
        if loadedModelId == model.id && isLoaded && encoderFingerprint == extras.encoderFingerprint { return }
        unloadModel()
        if var specification = ModelZoo.specificationForModel(model.id) {
            let overrides = extras.assets.filter { $0.enabled && $0.kind.isEncoder }
            if !overrides.isEmpty {
                overriddenModelID = model.id
                previousOverride = ModelZoo.overrideMapping[model.id]
                specification = Self.applyingEncoderOverrides(overrides, to: specification)
                ModelZoo.overrideMapping[model.id] = specification
            }
        }

        isLoading = true
        isLoaded = false
        pipeline = nil
        loadedModelId = nil

        let modelsDirectory = try drawThingsModelsDirectory()

        do {
            var loadedPipeline = try await MediaGenerationPipeline.fromPretrained(
                model.id,
                backend: .local(directory: modelsDirectory.path)
            )
            if let resolution = model.imageGenerationResolution {
                var config = loadedPipeline.configuration
                config.width = resolution
                config.height = resolution
                config.batchSize = 1
                config.batchCount = 1
                loadedPipeline.configuration = config
            }
            self.pipeline = loadedPipeline
            self.loadedModelId = model.id
            self.isLoaded = true
            encoderFingerprint = extras.encoderFingerprint
        } catch {
            unloadModel()
            isLoading = false
            throw error
        }
        isLoading = false
    }

    func unloadModel() {
        if let modelID = overriddenModelID {
            ModelZoo.overrideMapping[modelID] = previousOverride
            overriddenModelID = nil
            previousOverride = nil
        }
        encoderFingerprint = ""
        pipeline = nil
        loadedModelId = nil
        isLoaded = false
        isGenerating = false
    }

    static func applyingEncoderOverrides(_ assets: [ImageExtra], to original: ModelZoo.Specification) -> ModelZoo.Specification {
        var specification = original
        for asset in assets where asset.enabled {
            switch asset.kind {
            case .clipL:
                // Draw Things stores a supplementary CLIP-L in clipEncoder; SD1 uses its primary slot.
                if original.clipEncoder != nil { specification.clipEncoder = asset.file }
                else { specification.textEncoder = asset.file }
            case .clipG: specification.textEncoder = asset.file
            case .textEncoder:
                if original.t5Encoder != nil { specification.t5Encoder = asset.file }
                else { specification.textEncoder = asset.file }
            case .vae: specification.autoencoder = asset.file
            default: break
            }
        }
        return specification
    }

    func generateImage(
        prompt: String,
        steps: Int,
        seed: UInt32,
        inputImage: UIImage? = nil,
        denoiseStrength: Float = 0.7
    ) async throws -> UIImage? {
        let extras = ImageExtrasStore.shared.options
        if encoderFingerprint != extras.encoderFingerprint,
           let model = ModelData.allModels().first(where: { $0.id == loadedModelId }) {
            try await loadModel(model)
        }
        guard isLoaded, let pipeline = pipeline else { throw SDError.pipelineNotLoaded }

        if let modelID = loadedModelId {
            let version = ModelZoo.versionForModel(modelID)
            guard extras.assets.filter({ $0.enabled && $0.kind == .lora }).allSatisfy({ LoRAZoo.versionForModel($0.file) == version }) else {
                throw CocoaError(.fileReadCorruptFile)
            }
        }
        generationStep = 0
        generationTotalSteps = steps
        isGenerating = true
        defer { isGenerating = false }

        var configuredPipeline = pipeline
        var config = pipeline.configuration
        config.steps = loadedModelId?.hasPrefix("sd_xl_turbo") == true ? min(4, max(1, steps)) : steps
        if loadedModelId?.hasPrefix("sd_xl_turbo") == true {
            config.guidanceScale = 1
        }
        config.seed = seed
        config.guidanceScale = loadedModelId?.hasPrefix("sd_xl_turbo") == true ? 1 : extras.guidance
        config.clipSkip = extras.clipSkip
        config.loras = extras.assets.filter { $0.enabled && $0.kind == .lora }
            .map { .init(file: $0.file, weight: $0.weight) }
        config.strength = inputImage == nil ? 1 : denoiseStrength
        configuredPipeline.configuration = config
        self.pipeline = configuredPipeline

        let inputs: [MediaGenerationPipeline.Input] = inputImage.map { [$0] } ?? []
        let results = try await configuredPipeline.generate(
            prompt: extras.translatedPrompt(prompt),
            negativePrompt: extras.translatedPrompt(extras.negativePrompt),
            inputs: inputs
        ) { @Sendable state, _ in
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
