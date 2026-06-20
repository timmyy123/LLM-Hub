import Foundation
import CoreML
import AVFoundation
import MediaGenerationKit
import ModelZoo
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
        config.strength = 1.0
        config.motionScale = Int(motionStrength * 255.0)
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

        // Convert pipeline outputs to standard sRGB UIImages.
        // MediaGenerationPipeline may return images with extended or float color spaces
        // (e.g. Display P3 float16). We must normalize to 8-bit sRGB BEFORE encoding,
        // otherwise the pixel channel bytes are misinterpreted and colors become psychedelic.
        let rawImages = results.map { UIImage($0) }
        let uiImages = await Task.detached(priority: .userInitiated) {
            rawImages.map { Self.normalizeToSRGB($0) }
        }.value

        let tempDir = FileManager.default.temporaryDirectory
        let outputURL = tempDir.appendingPathComponent(UUID().uuidString + ".mp4")

        try await compileVideo(from: uiImages, to: outputURL, fps: 10)
        return outputURL
    }

    func cancelGeneration() {
        isGenerating = false
    }

    // MARK: - Frame Normalization

    /// Renders any UIImage into a fresh standard 8-bit sRGB bitmap.
    /// This collapses extended linear / P3 / float color spaces into the standard
    /// device RGB space that H.264 / AVAssetWriter expect.
    nonisolated private static func normalizeToSRGB(_ image: UIImage) -> UIImage {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return image }
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
    }

    // MARK: - Video Compilation Helper

    private func drawThingsModelsDirectory() throws -> URL {
        return ModelZoo.persistentModelsDirectory()
    }

    private func compileVideo(from images: [UIImage], to outputURL: URL, fps: Int) async throws {
        guard !images.isEmpty else {
            throw NSError(domain: "VideoGenerator", code: -1, userInfo: [NSLocalizedDescriptionKey: "No frames generated"])
        }

        let size = images[0].size
        let width = (Int(size.width) / 16) * 16
        let height = (Int(size.height) / 16) * 16

        try? FileManager.default.removeItem(at: outputURL)

        guard let videoWriter = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            throw NSError(domain: "VideoGenerator", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAssetWriter"])
        }

        let compressionProperties: [String: Any] = [
            AVVideoAverageBitRateKey: width * height * 12, // High bitrate for crisp video quality
            AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
        ]

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: compressionProperties,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        ]

        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)

        // IMPORTANT: sourcePixelBufferAttributes format MUST match what pixelBufferFrom() creates.
        // Using BGRA (32BGRA) = iOS native format. Mismatch causes color corruption and encode errors.
        let pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height
            ]
        )

        guard videoWriter.canAdd(writerInput) else {
            throw NSError(domain: "VideoGenerator", code: -3, userInfo: [NSLocalizedDescriptionKey: "Cannot add writer input"])
        }
        videoWriter.add(writerInput)

        guard videoWriter.startWriting() else {
            throw NSError(domain: "VideoGenerator", code: -4, userInfo: [NSLocalizedDescriptionKey: videoWriter.error?.localizedDescription ?? "Asset writer failed to start writing"])
        }

        videoWriter.startSession(atSourceTime: .zero)

        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))

        for (index, image) in images.enumerated() {
            // Wait for writer to be ready; timeout after 5 seconds to avoid infinite hang
            var waited = 0
            while !writerInput.isReadyForMoreMediaData {
                if videoWriter.status == .failed || waited > 500 {
                    throw NSError(domain: "VideoGenerator", code: -5, userInfo: [
                        NSLocalizedDescriptionKey: videoWriter.error?.localizedDescription ?? "Writer not ready (timeout)"
                    ])
                }
                try await Task.sleep(nanoseconds: 10_000_000) // 10ms
                waited += 1
            }

            guard let pixelBuffer = pixelBufferFrom(image: image, width: width, height: height) else {
                throw NSError(domain: "VideoGenerator", code: -6, userInfo: [NSLocalizedDescriptionKey: "Failed to convert frame \(index) to pixel buffer"])
            }

            let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(index))
            guard pixelBufferAdaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
                throw NSError(domain: "VideoGenerator", code: -7, userInfo: [
                    NSLocalizedDescriptionKey: videoWriter.error?.localizedDescription ?? "Failed to append pixel buffer at frame \(index)"
                ])
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
                code: -8,
                userInfo: [NSLocalizedDescriptionKey: videoWriter.error?.localizedDescription ?? "Failed to finish writing video"]
            )
        }
    }

    /// Converts a standard 8-bit sRGB UIImage to a BGRA CVPixelBuffer.
    /// Input MUST be pre-normalized via normalizeToSRGB() — float or extended color spaces
    /// will produce garbage pixels even with the correct bitmapInfo.
    private func pixelBufferFrom(image: UIImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        // BGRA with little-endian byte order = kCVPixelFormatType_32BGRA
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
        guard let context = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo
        ) else { return nil }

        // UIKit coordinate system is flipped vs CoreGraphics — fix orientation
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        UIGraphicsPushContext(context)
        image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()

        return buffer
    }
}
