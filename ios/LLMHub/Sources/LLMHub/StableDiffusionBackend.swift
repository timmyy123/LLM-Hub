import Foundation
import CoreML
#if canImport(UIKit)
import UIKit
#endif

// MARK: - SDError (always compiled)

enum SDError: LocalizedError {
    case notCoreMLModel
    case modelNotDownloaded
    case pipelineNotLoaded
    case imageToImageUnsupported
    case unavailable

    var errorDescription: String? {
        switch self {
        case .notCoreMLModel: return "Not a CoreML image generation model."
        case .modelNotDownloaded: return "Model files not found. Please download the model first."
        case .pipelineNotLoaded: return "Stable Diffusion pipeline is not loaded."
        case .imageToImageUnsupported: return "Image-to-image is not supported by this downloaded model bundle on iPhone."
        case .unavailable: return "Stable Diffusion is not available in this build."
        }
    }
}

// MARK: - StableDiffusionBackend

#if canImport(StableDiffusion)
import StableDiffusion

// Sendable wrapper: StableDiffusionPipeline Protocol bridge
private struct SendablePipeline: @unchecked Sendable {
    let value: StableDiffusionPipelineProtocol
}

@MainActor
final class StableDiffusionBackend: ObservableObject {
    static let shared = StableDiffusionBackend()

    @Published var isLoaded = false
    @Published var isLoading = false
    @Published var isGenerating = false
    @Published var generationStep = 0
    @Published var generationTotalSteps = 20
    @Published var loadedModelId: String? = nil

    nonisolated(unsafe) private var pipeline: StableDiffusionPipelineProtocol?
    private var loadedWithANE = false

    private init() {}

    // MARK: - SD Model Directory Helpers

    static func sdBaseDirectory() -> URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?
            .appendingPathComponent("sd_models")
    }

    static func sdModelDirectory(for modelId: String) -> URL? {
        sdBaseDirectory()?.appendingPathComponent(modelId)
    }

    static func isCoreMLModelDownloaded(modelId: String) -> Bool {
        guard let dir = sdModelDirectory(for: modelId) else { return false }
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("_downloaded").path)
    }

    static func supportsImageToImage(modelId: String) -> Bool {
        guard let dir = sdModelDirectory(for: modelId) else { return false }
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("VAEEncoder.mlmodelc").path) else {
            return false
        }
        // Apple official split-einsum iPhone/iPad bundles expose VAEEncoder, but the
        // encoder path still fails on-device for the current catalog. Keep img2img
        // hidden unless a non-split bundle is added and verified to work.
        return !modelId.contains("split-einsum")
    }

    /// Strips pre-compiled ANE / espresso artifacts from all .mlmodelc bundles in `directory`.
    /// Called at download time *and* at load time so models downloaded before this fix
    /// are also cleaned up without requiring a re-download.
    static func stripStaleArtifacts(in directory: URL) {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        let stalePatterns = [
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights",
            "coreml_model.espresso.net",
            "coreml_model.espresso.shape",
            "coreml_model.espresso.weights",
        ]

        for case let url as URL in enumerator {
            guard url.pathExtension == "mlmodelc",
                  (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            else { continue }
            enumerator.skipDescendants()
            for pattern in stalePatterns {
                let target = url.appendingPathComponent(pattern)
                if fm.fileExists(atPath: target.path) {
                    try? fm.removeItem(at: target)
                }
            }
        }
    }

#if canImport(UIKit)
    private static func preparedStartingImage(_ image: UIImage, modelId: String?) -> CGImage? {
        let model = modelId.flatMap { id in ModelData.allModels().first(where: { $0.id == id }) }
        let sideLength = CGFloat(model?.imageGenerationResolution ?? 512)
        let canvasSize = CGSize(width: sideLength, height: sideLength)

        let sourceImage = image.cgImage.map { UIImage(cgImage: $0, scale: image.scale, orientation: image.imageOrientation) } ?? image
        let sourceSize = sourceImage.size
        guard sourceSize.width > 0, sourceSize.height > 0 else { return sourceImage.cgImage }

        let scale = max(canvasSize.width / sourceSize.width, canvasSize.height / sourceSize.height)
        let scaledSize = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
        let drawRect = CGRect(
            x: (canvasSize.width - scaledSize.width) / 2,
            y: (canvasSize.height - scaledSize.height) / 2,
            width: scaledSize.width,
            height: scaledSize.height
        )

        let renderer = UIGraphicsImageRenderer(size: canvasSize)
        return renderer.image { _ in
            sourceImage.draw(in: drawRect)
        }.cgImage
    }
#endif

    // MARK: - SDXL detection helper

    nonisolated private static func isSDXL(modelId: String) -> Bool {
        modelId.contains("sdxl") || modelId.contains("xl-base")
    }

    func loadModel(_ model: AIModel) async throws {
        try await loadModel(model, preferANE: model.id.contains("split-einsum"))
    }

    private func loadModel(_ model: AIModel, preferANE: Bool) async throws {
        guard model.isCoreMLImageGeneration else { throw SDError.notCoreMLModel }
        guard let dir = StableDiffusionBackend.sdModelDirectory(for: model.id),
              FileManager.default.fileExists(atPath: dir.appendingPathComponent("_downloaded").path)
        else { throw SDError.modelNotDownloaded }

        if loadedModelId == model.id && isLoaded && loadedWithANE == preferANE { return }

        isLoading = true
        isLoaded = false
        pipeline = nil
        loadedModelId = nil
        loadedWithANE = false

        let modelDir = dir
        let useANE = preferANE

        // Strip any stale ANE artifacts present in models downloaded before this fix.
        // This is a no-op when the artifacts are already absent.
        Self.stripStaleArtifacts(in: modelDir)

        do {
            let wrapper = try await Self.loadPipeline(from: modelDir, useANE: useANE)
            pipeline = wrapper.value
            loadedModelId = model.id
            loadedWithANE = useANE
            isLoaded = true
        } catch {
            isLoading = false
            throw error
        }
        isLoading = false
    }

    private static func loadPipeline(from modelDir: URL, useANE: Bool) async throws -> SendablePipeline {
        return try await Task.detached(priority: .userInitiated) {
            let cfg = MLModelConfiguration()
            // .all: OS picks best compute unit (CPU/GPU/ANE) and handles ANE
            // recompilation automatically. .cpuAndNeuralEngine requires a valid
            // pre-compiled E5 bundle and fails hard when it's stale.
            // .cpuAndGPU: explicitly excludes ANE.
            cfg.computeUnits = useANE ? .all : .cpuAndGPU

            // For SDXL, instantiate StableDiffusionXLPipeline instead.
            // The Apple Neural Engine (ANE) is highly memory efficient but
            // compiling the massive SDXL models all at once during load causes
            // a massive Jetsam OOM memory crash on iOS.
            // To prevent this, we instantiate the pipeline with `.all` (ANE)
            // but we DO NOT call `loadResources()` or `prewarmResources()`.
            // By relying entirely on `reduceMemory: true` and lazy loading,
            // each model is loaded, compiled, and unloaded sequentially during
            // the generation loop, keeping peak memory below the Jetsam limit.
            if isSDXL(modelId: modelDir.lastPathComponent) {
                let p = try StableDiffusionXLPipeline(
                    resourcesAt: modelDir,
                    configuration: cfg,
                    reduceMemory: true
                )
                // Deliberately skipping `p.loadResources()` here for SDXL!
                return SendablePipeline(value: p)
            } else {
                let p = try StableDiffusionPipeline(
                    resourcesAt: modelDir,
                    controlNet: [],
                    configuration: cfg,
                    reduceMemory: true
                )
                try p.loadResources()
                return SendablePipeline(value: p)
            }
        }.value
    }

    func unloadModel() {
        pipeline = nil
        loadedModelId = nil
        loadedWithANE = false
        isLoaded = false
        isGenerating = false
    }

    private func ensurePipelineConfiguration(for inputImage: UIImage?) async throws {
        guard let loadedModelId,
              let model = ModelData.allModels().first(where: { $0.id == loadedModelId })
        else { return }

        let needsEncoderSafeComputeUnits = inputImage != nil && model.id.contains("split-einsum")
        let preferredANE = !needsEncoderSafeComputeUnits && model.id.contains("split-einsum")

        if !isLoaded || self.loadedWithANE != preferredANE {
            try await loadModel(model, preferANE: preferredANE)
        }
    }

    func generateImage(
        prompt: String,
        steps: Int,
        seed: UInt32,
        inputImage: UIImage? = nil,
        denoiseStrength: Float = 0.7
    ) async throws -> UIImage? {
        if inputImage != nil,
           let loadedModelId,
           !Self.supportsImageToImage(modelId: loadedModelId) {
            throw SDError.imageToImageUnsupported
        }
        try await ensurePipelineConfiguration(for: inputImage)
        guard let pipeline else { throw SDError.pipelineNotLoaded }

        generationStep = 0
        generationTotalSteps = steps
        isGenerating = true
        defer { isGenerating = false }

        var config = PipelineConfiguration(prompt: prompt)
        config.negativePrompt = "ugly, blurry, bad anatomy, bad quality"
        let isXL = loadedModelId.map { Self.isSDXL(modelId: $0) } ?? false
        // The apple/coreml-stable-diffusion-xl-base-ios bundle only supports
        // PNDM. Using DPM-Solver++ on this model causes tensor shape mismatches
        // that result in either a crash or an infinite hang during generation.
        config.schedulerType = isXL ? .pndmScheduler : .dpmSolverMultistepScheduler
        // SDXL needs at least 20 PNDM steps to produce coherent output.
        config.stepCount = isXL ? max(steps, 20) : steps
        config.seed = seed
        config.guidanceScale = isXL ? 5.0 : 7.5
        if let inputImage {
    #if canImport(UIKit)
            config.startingImage = Self.preparedStartingImage(inputImage, modelId: loadedModelId)
    #else
            config.startingImage = inputImage.cgImage
    #endif
            config.strength = denoiseStrength
            // mode is a computed property: returns .imageToImage automatically when startingImage is set
        }

        let capturedWrapper = SendablePipeline(value: pipeline)
        let capturedSteps = steps
        let cgImageOrNil = try await Task.detached(priority: .userInitiated) { [weak self] in
            let images = try capturedWrapper.value.generateImages(configuration: config) { [weak self] progress in
                let s = progress.step
                let t = progress.stepCount > 0 ? progress.stepCount : capturedSteps
                Task { @MainActor [weak self] in
                    self?.generationStep = s
                    self?.generationTotalSteps = t
                }
                return !Task.isCancelled
            }
            return images.first.flatMap { $0 }
        }.value

        guard let cgImage = cgImageOrNil else { return nil }
        return UIImage(cgImage: cgImage)
    }

    func cancelGeneration() {
        isGenerating = false
    }
}

#else

// Stub when StableDiffusion package is not yet resolved / not available.
@MainActor
final class StableDiffusionBackend: ObservableObject {
    static let shared = StableDiffusionBackend()

    @Published var isLoaded = false
    @Published var isLoading = false
    @Published var isGenerating = false
    @Published var generationStep = 0
    @Published var generationTotalSteps = 20
    @Published var loadedModelId: String? = nil

    private init() {}

    // MARK: - SD Model Directory Helpers

    static func sdBaseDirectory() -> URL? {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?
            .appendingPathComponent("sd_models")
    }

    static func sdModelDirectory(for modelId: String) -> URL? {
        sdBaseDirectory()?.appendingPathComponent(modelId)
    }

    static func isCoreMLModelDownloaded(modelId: String) -> Bool {
        guard let dir = sdModelDirectory(for: modelId) else { return false }
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("_downloaded").path)
    }

    static func supportsImageToImage(modelId: String) -> Bool {
        guard let dir = sdModelDirectory(for: modelId) else { return false }
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("VAEEncoder.mlmodelc").path)
    }

    func loadModel(_ model: AIModel) async throws {
        throw SDError.unavailable
    }

    func unloadModel() {}

    func generateImage(
        prompt: String,
        steps: Int,
        seed: UInt32,
        inputImage: UIImage? = nil,
        denoiseStrength: Float = 0.7
    ) async throws -> UIImage? {
        throw SDError.unavailable
    }

    func cancelGeneration() {}
}

#endif
