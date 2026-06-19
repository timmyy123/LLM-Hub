import Foundation
import CoreML
import AVFoundation
import MediaGenerationKit
#if canImport(UIKit)
import UIKit
#endif

enum VideoError: LocalizedError {
    case modelNotDownloaded
    case inputImageRequired
    case generationFailed(String = "")
    case cancelled
    case notSupported

    var errorDescription: String? {
        switch self {
        case .modelNotDownloaded: return "Model files not found. Please download the model first."
        case .inputImageRequired: return "Select an input image before generating."
        case .generationFailed(let message): return message.isEmpty ? "Video generation failed." : message
        case .cancelled: return "Generation cancelled."
        case .notSupported: return "Video generation requires a downloaded Draw Things video model. Download one from the Models screen."
        }
    }
}

@MainActor
final class VideoGeneratorBackend: ObservableObject {
    static let shared = VideoGeneratorBackend()

    @Published var isLoaded = false
    @Published var isLoading = false
    @Published var isGenerating = false
    @Published var generationStep = 0
    @Published var generationTotalSteps = 20
    @Published var loadedModelId: String? = nil

    private var pipeline: MediaGenerationPipeline?

    private init() {}

    func loadModel(url: URL, modelId: String) async throws {
        isLoading = true
        defer { isLoading = false }

        guard let model = ModelData.models.first(where: { $0.id == modelId }),
              ModelData.isModelFullyAvailableLocally(model) else {
            throw VideoError.modelNotDownloaded
        }

        let modelsDirectory = try drawThingsModelsDirectory()
        let loadedPipeline = try await MediaGenerationPipeline.fromPretrained(
            model.id,
            backend: .local(directory: modelsDirectory.path)
        )

        self.pipeline = loadedPipeline
        self.isLoaded = true
        self.loadedModelId = modelId
    }

    func unloadModel() {
        self.pipeline = nil
        self.isLoaded = false
        self.loadedModelId = nil
    }

    func generateVideo(
        prompt: String,
        steps: Int,
        seed: UInt32,
        inputImage: UIImage? = nil,
        motionStrength: Float = 0.7
    ) async throws -> URL {
        guard isLoaded, let pipeline = pipeline else {
            throw VideoError.modelNotDownloaded
        }
        guard let startingImage = inputImage else {
            throw VideoError.inputImageRequired
        }

        isGenerating = true
        generationStep = 0
        generationTotalSteps = steps
        defer { isGenerating = false }

        var configuredPipeline = pipeline
        var config = pipeline.configuration
        config.steps = steps
        config.seed = seed
        config.strength = motionStrength
        configuredPipeline.configuration = config
        self.pipeline = configuredPipeline

        let inputs: [MediaGenerationPipeline.Input] = [startingImage]

        let results = try await configuredPipeline.generate(
            prompt: prompt,
            negativePrompt: "",
            inputs: inputs
        ) { @Sendable state in
            if case .generating(let step, let total) = state {
                Task { @MainActor in
                    VideoGeneratorBackend.shared.generationStep = step
                    VideoGeneratorBackend.shared.generationTotalSteps = total
                }
            }
        }

        guard !results.isEmpty else {
            throw VideoError.generationFailed()
        }

        let uiImages = results.map { UIImage($0) }
        let tempDir = FileManager.default.temporaryDirectory
        let outputURL = tempDir.appendingPathComponent(UUID().uuidString + ".mp4")

        try await compileVideo(from: uiImages, to: outputURL, fps: 10)
        return outputURL
    }

    func cancelGeneration() {
        isGenerating = false
    }

    // MARK: - Video Compilation Helper

    private func drawThingsModelsDirectory() throws -> URL {
        guard let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            throw VideoError.modelNotDownloaded
        }
        let modelsDirectory = docsDir.appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        return modelsDirectory
    }

    private func compileVideo(from images: [UIImage], to outputURL: URL, fps: Int) async throws {
        guard !images.isEmpty else {
            throw NSError(domain: "VideoGenerator", code: -1, userInfo: [NSLocalizedDescriptionKey: "No frames generated"])
        }

        let size = images[0].size
        let width = Int(size.width)
        let height = Int(size.height)

        try? FileManager.default.removeItem(at: outputURL)

        guard let videoWriter = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            throw NSError(domain: "VideoGenerator", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAssetWriter"])
        }

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height
        ]

        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        let pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ]
        )

        guard videoWriter.canAdd(writerInput) else {
            throw NSError(domain: "VideoGenerator", code: -3, userInfo: [NSLocalizedDescriptionKey: "Cannot add writer input"])
        }
        videoWriter.add(writerInput)

        guard videoWriter.startWriting() else {
            throw NSError(domain: "VideoGenerator", code: -4, userInfo: [NSLocalizedDescriptionKey: "Asset writer failed to start writing"])
        }

        videoWriter.startSession(atSourceTime: .zero)

        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))

        for (index, image) in images.enumerated() {
            while !writerInput.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 10_000_000)
            }

            guard let pixelBuffer = pixelBufferFrom(image: image, size: size) else {
                throw NSError(domain: "VideoGenerator", code: -5, userInfo: [NSLocalizedDescriptionKey: "Failed to convert UIImage to CVPixelBuffer"])
            }

            let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(index))
            guard pixelBufferAdaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
                throw NSError(domain: "VideoGenerator", code: -6, userInfo: [NSLocalizedDescriptionKey: "Failed to append pixel buffer"])
            }
        }

        writerInput.markAsFinished()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            videoWriter.finishWriting {
                continuation.resume()
            }
        }
        if videoWriter.status == .failed {
            throw NSError(
                domain: "VideoGenerator",
                code: -7,
                userInfo: [NSLocalizedDescriptionKey: videoWriter.error?.localizedDescription ?? "Failed to finish writing video"]
            )
        }
    }

    private func pixelBufferFrom(image: UIImage, size: CGSize) -> CVPixelBuffer? {
        let width = Int(size.width)
        let height = Int(size.height)

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            [
                kCVPixelBufferCGImageCompatibilityKey as String: true,
                kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
            ] as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        }

        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )

        guard let cgImage = image.cgImage else { return nil }
        context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }
}
