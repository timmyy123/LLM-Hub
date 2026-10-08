import CoreGraphics
import CoreImage
@preconcurrency import DataModels
@preconcurrency import Diffusion
import Foundation
import ImageGenerator
import ImageIO
import Logging
import ModelZoo
import NNC
import ScriptDataModels
import UniformTypeIdentifiers

#if canImport(UIKit)
  import UIKit
#endif

/// A typed input passed to ``MediaGenerationPipeline/generate(prompt:negativePrompt:inputs:stateHandler:)``.
///
/// Use ``MediaGenerationPipeline/file(_:)`` or ``MediaGenerationPipeline/data(_:)`` to create image
/// inputs, then wrap them with role-specific helpers such as ``MediaGenerationPipeline/MaskInput``.
public protocol MediaGenerationInput {
  var role: MediaGenerationPipeline.InputRole { get }
}

/// An input that can provide encoded image bytes.
public protocol MediaGenerationImageInput: MediaGenerationInput {}

internal protocol MediaGenerationImageDataSource {
  func mediaGenerationEncodedData() throws -> Data
}

/// A configured generation pipeline for local, remote, or cloud compute inference.
///
/// Build a pipeline with ``fromPretrained(_:backend:)``, adjust ``configuration``, then call
/// ``generate(prompt:negativePrompt:inputs:stateHandler:)``.
public struct MediaGenerationPipeline: Sendable {
  public typealias Input = any MediaGenerationInput
  public typealias ImageInput = any MediaGenerationImageInput

  /// Network location of a remote Draw Things generation server.
  public struct Endpoint: Hashable, Sendable {
    public var host: String
    public var port: Int

    /// Creates a remote server endpoint.
    public init(host: String, port: Int) {
      self.host = host
      self.port = port
    }
  }

  /// Options for plain remote generation.
  public struct RemoteOptions: Sendable {
    /// Whether to use TLS when talking to the remote server.
    public var useTLS: Bool
    /// Optional shared-secret header for password-protected remote servers.
    public var sharedSecret: String?

    /// Creates remote generation options.
    public init(
      useTLS: Bool = true,
      sharedSecret: String? = nil
    ) {
      self.useTLS = useTLS
      self.sharedSecret = sharedSecret
    }
  }

  /// Options for Draw Things cloud compute.
  public struct CloudComputeOptions: Sendable {
    /// Optional override for the Draw Things cloud API base URL.
    public var baseURL: URL?
    /// Optional device name to send to the cloud backend.
    public var deviceName: String?
    /// Optional app-verification token source.
    public var appCheck: AppCheckConfiguration

    /// Creates cloud compute options.
    public init(
      baseURL: URL? = nil,
      deviceName: String? = nil,
      appCheck: AppCheckConfiguration = .none
    ) {
      self.baseURL = baseURL
      self.deviceName = deviceName
      self.appCheck = appCheck
    }
  }

  /// Execution backend for a pipeline.
  public enum Backend: Sendable {
    /// Run locally with models resolved from the default environment or an explicit directory.
    case local(directory: String?)
    /// Run against a remote Draw Things server.
    case remote(_ endpoint: Endpoint, options: RemoteOptions = .init())
    /// Run against Draw Things cloud compute.
    case cloudCompute(apiKey: String? = nil, options: CloudComputeOptions = .init())

    /// Local generation using ``MediaGenerationEnvironment/default``.
    public static var local: Backend {
      .local(directory: nil)
    }
  }

  /// Generation settings derived from the model's recommended configuration.
  ///
  /// The public surface intentionally tracks the canonical `JSGenerationConfiguration` bridge. In
  /// practice, most callers only need to adjust a small subset such as `width`, `height`, `steps`,
  /// `strength`, `loras`, and `controls`.
  public struct Configuration: Sendable {
    // Keep this surface in sync with JSGenerationConfiguration.
    // Configuration is derived from JSGenerationConfiguration(configuration:) and converted back through
    // JSGenerationConfiguration.createGenerationConfiguration() so MediaGenerationKit reuses the canonical
    // Scripting conversion layer instead of duplicating GenerationConfiguration normalization logic.
    public private(set) var model: String

    public var width: Int
    public var height: Int
    public var seed: UInt32
    public var steps: Int
    public var guidanceScale: Float
    public var strength: Float
    public var seedMode: SeedMode
    public var clipSkip: Int
    public var batchCount: Int
    public var batchSize: Int
    public var numFrames: Int
    public var fps: Int
    public var motionScale: Int
    public var sampler: SamplerType
    public var hiresFix: Bool
    public var hiresFixWidth: Int
    public var hiresFixHeight: Int
    public var hiresFixStrength: Float
    public var tiledDecoding: Bool
    public var decodingTileWidth: Int
    public var decodingTileHeight: Int
    public var decodingTileOverlap: Int
    public var tiledDiffusion: Bool
    public var diffusionTileWidth: Int
    public var diffusionTileHeight: Int
    public var diffusionTileOverlap: Int
    public var upscaler: String?
    public var upscalerScaleFactor: Int
    public var imageGuidanceScale: Float
    public var loras: [LoRA]
    public var controls: [Control]
    public var maskBlur: Float
    public var maskBlurOutset: Int
    public var sharpness: Float
    public var faceRestoration: String?
    public var clipWeight: Float
    public var negativePromptForImagePrior: Bool
    public var imagePriorSteps: Int
    public var refinerModel: String?
    public var originalImageHeight: Int
    public var originalImageWidth: Int
    public var cropTop: Int
    public var cropLeft: Int
    public var targetImageHeight: Int
    public var targetImageWidth: Int
    public var aestheticScore: Float
    public var negativeAestheticScore: Float
    public var zeroNegativePrompt: Bool
    public var refinerStart: Float
    public var negativeOriginalImageHeight: Int
    public var negativeOriginalImageWidth: Int
    public var guidingFrameNoise: Float
    public var startFrameGuidance: Float
    public var shift: Float
    public var stage2Steps: Int
    public var stage2Guidance: Float
    public var stage2Shift: Float
    public var stochasticSamplingGamma: Float
    public var preserveOriginalAfterInpaint: Bool
    public var t5TextEncoder: Bool
    public var separateClipL: Bool
    public var clipLText: String?
    public var separateOpenClipG: Bool
    public var openClipGText: String?
    public var speedUpWithGuidanceEmbed: Bool
    public var guidanceEmbed: Float
    public var resolutionDependentShift: Bool
    public var teaCache: Bool
    public var teaCacheStart: Int
    public var teaCacheEnd: Int
    public var teaCacheThreshold: Float
    public var teaCacheMaxSkipSteps: Int
    public var separateT5: Bool
    public var t5Text: String?
    public var causalInference: Int
    public var causalInferencePad: Int
    public var cfgZeroStar: Bool
    public var cfgZeroInitSteps: Int
    public var compressionArtifacts: CompressionMethod
    public var compressionArtifactsQuality: Float?
    public var colorCalibration: ColorCalibration
    public var expandPromptToJson: Bool

    fileprivate init(configuration: GenerationConfiguration, model: String) {
      let options = JSGenerationConfiguration(configuration: configuration)
      self.model = model
      self.width = Int(options.width)
      self.height = Int(options.height)
      self.seed = options.seed >= 0 ? UInt32(options.seed) : configuration.seed
      self.steps = Int(options.steps)
      self.guidanceScale = Float(options.guidanceScale)
      self.strength = Float(options.strength)
      self.seedMode = SeedMode(rawValue: options.seedMode) ?? configuration.seedMode
      self.clipSkip = Int(options.clipSkip)
      self.batchCount = Int(options.batchCount)
      self.batchSize = Int(options.batchSize)
      self.numFrames = Int(options.numFrames)
      self.fps = Int(options.fps)
      self.motionScale = Int(options.motionScale)
      self.sampler = SamplerType(rawValue: options.sampler) ?? configuration.sampler
      self.hiresFix = options.hiresFix
      self.hiresFixWidth = Int(options.hiresFixWidth)
      self.hiresFixHeight = Int(options.hiresFixHeight)
      self.hiresFixStrength = Float(options.hiresFixStrength)
      self.tiledDecoding = options.tiledDecoding
      self.decodingTileWidth = Int(options.decodingTileWidth)
      self.decodingTileHeight = Int(options.decodingTileHeight)
      self.decodingTileOverlap = Int(options.decodingTileOverlap)
      self.tiledDiffusion = options.tiledDiffusion
      self.diffusionTileWidth = Int(options.diffusionTileWidth)
      self.diffusionTileHeight = Int(options.diffusionTileHeight)
      self.diffusionTileOverlap = Int(options.diffusionTileOverlap)
      self.upscaler = options.upscaler
      self.upscalerScaleFactor = Int(options.upscalerScaleFactor)
      self.imageGuidanceScale = Float(options.imageGuidanceScale)
      self.loras = options.loras.map { $0.createLora() }
      self.controls = options.controls.map { $0.createControl() }
      self.maskBlur = Float(options.maskBlur)
      self.maskBlurOutset = Int(options.maskBlurOutset)
      self.sharpness = Float(options.sharpness)
      self.faceRestoration = options.faceRestoration
      self.clipWeight = Float(options.clipWeight)
      self.negativePromptForImagePrior = options.negativePromptForImagePrior
      self.imagePriorSteps = Int(options.imagePriorSteps)
      self.refinerModel = options.refinerModel
      self.originalImageHeight = Int(options.originalImageHeight)
      self.originalImageWidth = Int(options.originalImageWidth)
      self.cropTop = Int(options.cropTop)
      self.cropLeft = Int(options.cropLeft)
      self.targetImageHeight = Int(options.targetImageHeight)
      self.targetImageWidth = Int(options.targetImageWidth)
      self.aestheticScore = Float(options.aestheticScore)
      self.negativeAestheticScore = Float(options.negativeAestheticScore)
      self.zeroNegativePrompt = options.zeroNegativePrompt
      self.refinerStart = Float(options.refinerStart)
      self.negativeOriginalImageHeight = Int(options.negativeOriginalImageHeight)
      self.negativeOriginalImageWidth = Int(options.negativeOriginalImageWidth)
      self.guidingFrameNoise = Float(options.guidingFrameNoise)
      self.startFrameGuidance = Float(options.startFrameGuidance)
      self.shift = Float(options.shift)
      self.stage2Steps = Int(options.stage2Steps)
      self.stage2Guidance = Float(options.stage2Guidance)
      self.stage2Shift = Float(options.stage2Shift)
      self.stochasticSamplingGamma = Float(options.stochasticSamplingGamma)
      self.preserveOriginalAfterInpaint = options.preserveOriginalAfterInpaint
      self.t5TextEncoder = options.t5TextEncoder
      self.separateClipL = options.separateClipL
      self.clipLText = options.clipLText
      self.separateOpenClipG = options.separateOpenClipG
      self.openClipGText = options.openClipGText
      self.speedUpWithGuidanceEmbed = options.speedUpWithGuidanceEmbed
      self.guidanceEmbed = Float(options.guidanceEmbed)
      self.resolutionDependentShift = options.resolutionDependentShift
      self.teaCache = options.teaCache
      self.teaCacheStart = Int(options.teaCacheStart)
      self.teaCacheEnd = Int(options.teaCacheEnd)
      self.teaCacheThreshold = Float(options.teaCacheThreshold)
      self.teaCacheMaxSkipSteps = Int(options.teaCacheMaxSkipSteps)
      self.separateT5 = options.separateT5
      self.t5Text = options.t5Text
      self.causalInference = Int(options.causalInference)
      self.causalInferencePad = Int(options.causalInferencePad)
      self.cfgZeroStar = options.cfgZeroStar
      self.cfgZeroInitSteps = Int(options.cfgZeroInitSteps)
      self.compressionArtifacts = Self.compressionMethod(from: options.compressionArtifacts)
      self.compressionArtifactsQuality =
        options.compressionArtifactsQuality.map { quality in Float(quality) }
      self.colorCalibration = Self.colorCalibration(from: options.colorCalibration)
      self.expandPromptToJson = options.expandPromptToJson
    }

    internal func runtimeConfiguration(template: GenerationConfiguration) throws
      -> GenerationConfiguration
    {
      guard width > 0, height > 0 else {
        throw MediaGenerationKitError.generationFailed(
          "invalid request: width and height must be greater than 0")
      }
      guard width % 64 == 0, height % 64 == 0 else {
        throw MediaGenerationKitError.generationFailed(
          "invalid request: width and height must be multiples of 64")
      }
      guard steps > 0 else {
        throw MediaGenerationKitError.generationFailed(
          "invalid request: steps must be greater than 0")
      }
      guard batchCount > 0, batchSize > 0 else {
        throw MediaGenerationKitError.generationFailed(
          "invalid request: batchCount and batchSize must be greater than 0")
      }

      let options = JSGenerationConfiguration(configuration: template)
      options.model = model
      options.width = try Self.nonNegativeUInt32(width, name: "width")
      options.height = try Self.nonNegativeUInt32(height, name: "height")
      options.seed = Int64(seed)
      options.steps = try Self.nonNegativeUInt32(steps, name: "steps")
      options.guidanceScale = guidanceScale
      options.strength = strength
      options.seedMode = seedMode.rawValue
      options.clipSkip = try Self.nonNegativeUInt32(clipSkip, name: "clipSkip")
      options.batchCount = try Self.nonNegativeUInt32(batchCount, name: "batchCount")
      options.batchSize = try Self.nonNegativeUInt32(batchSize, name: "batchSize")
      options.numFrames = try Self.nonNegativeUInt32(numFrames, name: "numFrames")
      options.fps = try Self.nonNegativeUInt32(fps, name: "fps")
      options.motionScale = try Self.nonNegativeUInt32(motionScale, name: "motionScale")
      options.sampler = sampler.rawValue
      options.hiresFix = hiresFix
      options.hiresFixWidth = try Self.nonNegativeUInt32(hiresFixWidth, name: "hiresFixWidth")
      options.hiresFixHeight = try Self.nonNegativeUInt32(hiresFixHeight, name: "hiresFixHeight")
      options.hiresFixStrength = hiresFixStrength
      options.tiledDecoding = tiledDecoding
      options.decodingTileWidth = try Self.nonNegativeUInt32(
        decodingTileWidth, name: "decodingTileWidth")
      options.decodingTileHeight = try Self.nonNegativeUInt32(
        decodingTileHeight, name: "decodingTileHeight")
      options.decodingTileOverlap = try Self.nonNegativeUInt32(
        decodingTileOverlap, name: "decodingTileOverlap")
      options.tiledDiffusion = tiledDiffusion
      options.diffusionTileWidth = try Self.nonNegativeUInt32(
        diffusionTileWidth, name: "diffusionTileWidth")
      options.diffusionTileHeight = try Self.nonNegativeUInt32(
        diffusionTileHeight, name: "diffusionTileHeight")
      options.diffusionTileOverlap = try Self.nonNegativeUInt32(
        diffusionTileOverlap, name: "diffusionTileOverlap")
      options.upscaler = upscaler
      options.upscalerScaleFactor = try Self.nonNegativeUInt8(
        upscalerScaleFactor, name: "upscalerScaleFactor")
      options.imageGuidanceScale = imageGuidanceScale
      options.loras = loras.map(JSLoRA.init(lora:))
      options.controls = controls.map(JSControl.init(control:))
      options.maskBlur = maskBlur
      options.maskBlurOutset = Int32(maskBlurOutset)
      options.sharpness = sharpness
      options.faceRestoration = faceRestoration
      options.clipWeight = clipWeight
      options.negativePromptForImagePrior = negativePromptForImagePrior
      options.imagePriorSteps = try Self.nonNegativeUInt32(imagePriorSteps, name: "imagePriorSteps")
      options.refinerModel = refinerModel
      options.originalImageHeight = try Self.nonNegativeUInt32(
        originalImageHeight, name: "originalImageHeight")
      options.originalImageWidth = try Self.nonNegativeUInt32(
        originalImageWidth, name: "originalImageWidth")
      options.cropTop = Int32(cropTop)
      options.cropLeft = Int32(cropLeft)
      options.targetImageHeight = try Self.nonNegativeUInt32(
        targetImageHeight, name: "targetImageHeight")
      options.targetImageWidth = try Self.nonNegativeUInt32(
        targetImageWidth, name: "targetImageWidth")
      options.aestheticScore = aestheticScore
      options.negativeAestheticScore = negativeAestheticScore
      options.zeroNegativePrompt = zeroNegativePrompt
      options.refinerStart = refinerStart
      options.negativeOriginalImageHeight = try Self.nonNegativeUInt32(
        negativeOriginalImageHeight, name: "negativeOriginalImageHeight")
      options.negativeOriginalImageWidth = try Self.nonNegativeUInt32(
        negativeOriginalImageWidth, name: "negativeOriginalImageWidth")
      options.guidingFrameNoise = guidingFrameNoise
      options.startFrameGuidance = startFrameGuidance
      options.shift = shift
      options.stage2Steps = try Self.nonNegativeUInt32(stage2Steps, name: "stage2Steps")
      options.stage2Guidance = stage2Guidance
      options.stage2Shift = stage2Shift
      options.stochasticSamplingGamma = stochasticSamplingGamma
      options.preserveOriginalAfterInpaint = preserveOriginalAfterInpaint
      options.t5TextEncoder = t5TextEncoder
      options.separateClipL = separateClipL
      options.clipLText = clipLText
      options.separateOpenClipG = separateOpenClipG
      options.openClipGText = openClipGText
      options.speedUpWithGuidanceEmbed = speedUpWithGuidanceEmbed
      options.guidanceEmbed = guidanceEmbed
      options.resolutionDependentShift = resolutionDependentShift
      options.teaCache = teaCache
      options.teaCacheStart = Int32(teaCacheStart)
      options.teaCacheEnd = Int32(teaCacheEnd)
      options.teaCacheThreshold = teaCacheThreshold
      options.teaCacheMaxSkipSteps = Int32(teaCacheMaxSkipSteps)
      options.separateT5 = separateT5
      options.t5Text = t5Text
      options.causalInference = Int32(causalInference)
      options.causalInferencePad = Int32(causalInferencePad)
      options.cfgZeroStar = cfgZeroStar
      options.cfgZeroInitSteps = Int32(cfgZeroInitSteps)
      options.compressionArtifacts = Self.compressionArtifactsName(for: compressionArtifacts)
      options.compressionArtifactsQuality = compressionArtifactsQuality
      options.colorCalibration = Self.colorCalibrationName(for: colorCalibration)
      options.expandPromptToJson = expandPromptToJson
      return options.createGenerationConfiguration()
    }

    private static func nonNegativeUInt32(_ value: Int, name: String) throws -> UInt32 {
      guard value >= 0 else {
        throw MediaGenerationKitError.generationFailed(
          "invalid request: \(name) must be greater than or equal to 0")
      }
      return UInt32(value)
    }

    private static func nonNegativeUInt8(_ value: Int, name: String) throws -> UInt8 {
      guard value >= 0 else {
        throw MediaGenerationKitError.generationFailed(
          "invalid request: \(name) must be greater than or equal to 0")
      }
      guard value <= Int(UInt8.max) else {
        throw MediaGenerationKitError.generationFailed(
          "invalid request: \(name) must be less than or equal to \(UInt8.max)")
      }
      return UInt8(value)
    }

    private static func compressionMethod(from name: String?) -> CompressionMethod {
      switch name?.lowercased() {
      case "h264":
        return .H264
      case "h265":
        return .H265
      case "jpeg":
        return .jpeg
      default:
        return .disabled
      }
    }

    private static func compressionArtifactsName(for method: CompressionMethod) -> String {
      switch method {
      case .disabled:
        return "disabled"
      case .H264:
        return "h264"
      case .H265:
        return "h265"
      case .jpeg:
        return "jpeg"
      }
    }

    private static func colorCalibration(from name: String?) -> ColorCalibration {
      switch name?.lowercased() {
      case "lab":
        return .lab
      default:
        return .disabled
      }
    }

    private static func colorCalibrationName(for colorCalibration: ColorCalibration) -> String {
      switch colorCalibration {
      case .disabled:
        return "none"
      case .lab:
        return "lab"
      }
    }
  }

  /// Semantic role for an input image.
  public enum InputRole: Sendable, Equatable {
    case image
    case mask
    case moodboard
    case depth
  }

  /// Image input backed by in-memory encoded data.
  public struct DataInput: MediaGenerationImageInput {
    public let data: Data

    /// Creates an input from encoded image bytes.
    public init(_ data: Data) {
      self.data = data
    }
  }

  /// Image input backed by a filesystem path.
  public struct FileInput: MediaGenerationImageInput {
    public let path: String

    /// Creates an input from a filesystem path.
    public init(_ path: String) {
      self.path = path
    }
  }

  /// Wraps an image input as an inpainting mask.
  public struct MaskInput: MediaGenerationInput {
    public let source: any MediaGenerationImageInput

    /// Creates a mask input from an image input.
    public init(_ source: any MediaGenerationImageInput) {
      self.source = source
    }

    public var role: InputRole {
      .mask
    }
  }

  /// Wraps an image input as a moodboard/reference image.
  public struct MoodboardInput: MediaGenerationInput {
    public let source: any MediaGenerationImageInput

    /// Creates a moodboard input from an image input.
    public init(_ source: any MediaGenerationImageInput) {
      self.source = source
    }

    public var role: InputRole {
      .moodboard
    }
  }

  /// Wraps an image input as a depth hint.
  public struct DepthInput: MediaGenerationInput {
    public let source: any MediaGenerationImageInput

    /// Creates a depth input from an image input.
    public init(_ source: any MediaGenerationImageInput) {
      self.source = source
    }

    public var role: InputRole {
      .depth
    }
  }

  /// Progress states emitted during generation.
  public enum State: Sendable {
    case resolvingBackend(Backend)
    case resolvingModel(String)
    case preparing
    case ensuringResources
    case uploading(bytesSent: Int, totalBytes: Int)
    case downloading(bytesReceived: Int, totalBytes: Int)
    case encodingText
    case encodingInputs
    case generating(step: Int, totalSteps: Int)
    case decoding
    case postprocessing
    case cancelling
    case completed
    case cancelled

    internal init(signpost: ImageGeneratorSignpost, totalSteps: Int) {
      switch signpost {
      case .textEncoded:
        self = .encodingText
      case .imageEncoded, .controlsGenerated:
        self = .encodingInputs
      case .sampling(let step):
        let reportedStep = min(max(step + 1, 1), max(totalSteps, 1))
        self = .generating(step: reportedStep, totalSteps: totalSteps)
      case .imageDecoded:
        self = .decoding
      case .secondPassImageEncoded, .secondPassSampling, .secondPassImageDecoded, .faceRestored,
        .imageUpscaled:
        self = .postprocessing
      }
    }
  }

  /// A live preview payload emitted through ``generate(prompt:negativePrompt:inputs:stateHandler:)``.
  ///
  /// The collection exposes one lazily decoded `CGImage` per preview image.
  public struct Preview: RandomAccessCollection, @unchecked Sendable {
    public typealias Index = Int

    private let tensor: Tensor<FloatType>
    private let version: ModelVersion

    internal init(tensor: Tensor<FloatType>, version: ModelVersion) {
      self.tensor = tensor
      self.version = version
    }

    public var startIndex: Int { 0 }
    public var endIndex: Int { count }

    public func index(after i: Int) -> Int {
      i + 1
    }

    public func index(before i: Int) -> Int {
      i - 1
    }

    public var count: Int {
      guard tensor.shape.count == 4 else { return 1 }
      return Swift.max(tensor.shape[0], 1)
    }

    /// Decodes the preview image at `position` on demand.
    public subscript(position: Int) -> CGImage {
      precondition(indices.contains(position))
      let previewTensor: Tensor<FloatType>
      if count == 1 {
        previewTensor = tensor
      } else {
        let shape = tensor.shape
        previewTensor = tensor[
          position..<(position + 1),
          0..<shape[1],
          0..<shape[2],
          0..<shape[3]
        ].copied()
      }
      guard
        let image = MediaGenerationImageCodec.previewImage(from: previewTensor, version: version)
      else {
        preconditionFailure("failed to decode preview image at index \(position)")
      }
      return image
    }
  }

  /// A generated image result backed by a tensor.
  public struct Result: @unchecked Sendable {
    public let tensor: Tensor<FloatType>

    public var width: Int {
      MediaGenerationImageCodec.width(for: tensor)
    }

    public var height: Int {
      MediaGenerationImageCodec.height(for: tensor)
    }

    internal init(tensor: Tensor<FloatType>) {
      self.tensor = tensor
    }

    internal init(encodedData: Data) throws {
      self.tensor = try MediaGenerationImageCodec.decode(encodedData)
    }

    /// Encodes the result image and writes it to disk.
    public func write(
      to url: URL,
      type: UTType,
      options: Data.WritingOptions = []
    ) throws {
      let data = try MediaGenerationImageCodec.encode(tensor, type: type)
      try data.write(to: url, options: options)
    }
  }

  internal final class Storage: @unchecked Sendable {
    let environment: MediaGenerationEnvironment
    let remoteExecutor: MediaGenerationRemoteExecutor?
    let resolvedModel: String
    let recommendedTemplate: GenerationConfiguration

    init(
      environment: MediaGenerationEnvironment,
      remoteExecutor: MediaGenerationRemoteExecutor?,
      resolvedModel: String,
      recommendedTemplate: GenerationConfiguration
    ) {
      self.environment = environment
      self.remoteExecutor = remoteExecutor
      self.resolvedModel = resolvedModel
      self.recommendedTemplate = recommendedTemplate
    }

    func generate(
      prompt: String,
      negativePrompt: String,
      configuration: MediaGenerationPipeline.Configuration,
      inputs: [MediaGenerationPipeline.Input],
      logger: Logger,
      stateHandler: ((MediaGenerationPipeline.State, MediaGenerationPipeline.Preview?) -> Void)?
    ) async throws -> [MediaGenerationPipeline.Result] {
      let executionInputs = try MediaGenerationPipeline.executionInputs(from: inputs)
      let runtimeConfiguration = try configuration.runtimeConfiguration(
        template: recommendedTemplate)
      let previewModelVersion = ModelZoo.versionForModel(
        runtimeConfiguration.model ?? resolvedModel
      )
      if let remoteExecutor {
        try await remoteExecutor.prepareForGeneration()
      }
      let cancellationBridge = MediaGenerationCancellationBridge()
      logger.debug(
        "Starting media generation",
        metadata: [
          "model": "\(runtimeConfiguration.model ?? resolvedModel)",
          "width": "\(configuration.width)",
          "height": "\(configuration.height)",
          "steps": "\(configuration.steps)",
        ]
      )

      let tensors: [Tensor<FloatType>] = try await withTaskCancellationHandler(
        operation: {
          try await withCheckedThrowingContinuation { continuation in
            let feedback: (ImageGeneratorSignpost, Tensor<FloatType>?) -> Bool = {
              signpost, tensor in
              if cancellationBridge.isCancelled {
                return false
              }
              let preview = tensor.map {
                MediaGenerationPipeline.Preview(tensor: $0, version: previewModelVersion)
              }
              stateHandler?(
                MediaGenerationPipeline.State(
                  signpost: signpost,
                  totalSteps: Int(runtimeConfiguration.steps)
                ),
                preview
              )
              return !cancellationBridge.isCancelled
            }
            let completion: (Swift.Result<[Tensor<FloatType>], MediaGenerationKitError>) -> Void = {
              result in
              cancellationBridge.finish()
              continuation.resume(with: result)
            }
            if let remoteExecutor {
              remoteExecutor.generate(
                prompt: prompt,
                negativePrompt: negativePrompt,
                configuration: runtimeConfiguration,
                image: executionInputs.image,
                mask: executionInputs.mask,
                hints: executionInputs.hints,
                cancellationBridge: cancellationBridge,
                uploadProgress: { bytesSent, totalBytes in
                  stateHandler?(.uploading(bytesSent: bytesSent, totalBytes: totalBytes), nil)
                },
                downloadProgress: { bytesReceived, totalBytes in
                  stateHandler?(
                    .downloading(bytesReceived: bytesReceived, totalBytes: totalBytes), nil)
                },
                feedback: feedback,
                completion: completion
              )
            } else {
              do {
                try environment.storage.generateLocally(
                  prompt: prompt,
                  negativePrompt: negativePrompt,
                  configuration: runtimeConfiguration,
                  image: executionInputs.image,
                  mask: executionInputs.mask,
                  hints: executionInputs.hints,
                  cancellationBridge: cancellationBridge,
                  feedback: feedback,
                  completion: completion
                )
              } catch let error as MediaGenerationKitError {
                cancellationBridge.finish()
                continuation.resume(throwing: error)
              } catch {
                cancellationBridge.finish()
                continuation.resume(
                  throwing: MediaGenerationKitError.generationFailed(error.localizedDescription))
              }
            }
          }
        },
        onCancel: {
          cancellationBridge.cancel()
        })

      let outputs = tensors.map(MediaGenerationPipeline.Result.init(tensor:))
      logger.debug(
        "Finished media generation",
        metadata: [
          "model": "\(runtimeConfiguration.model ?? resolvedModel)",
          "outputs": "\(outputs.count)",
        ]
      )
      return outputs
    }
  }

  /// Backend used by this pipeline.
  public let backend: Backend
  /// Mutable generation settings for the next call to ``generate(prompt:negativePrompt:inputs:stateHandler:)``.
  public var configuration: Configuration
  /// Logger used for pipeline messages and progress diagnostics.
  public var logger: Logger
  /// Canonical resolved model identifier for this pipeline.
  public let model: String

  private let storage: Storage

  private init(
    backend: Backend,
    configuration: Configuration,
    logger: Logger = Logger(label: "com.draw-things.media-generation-pipeline"),
    model: String,
    storage: Storage
  ) {
    self.backend = backend
    self.configuration = configuration
    self.logger = logger
    self.model = model
    self.storage = storage
  }

  public static func data(_ data: Data) -> DataInput {
    DataInput(data)
  }

  /// Creates an image input from a filesystem path.
  public static func file(_ path: String) -> FileInput {
    FileInput(path)
  }

  /// Builds a pipeline from a model reference and backend.
  ///
  /// `model` may be a local file id, a catalog name, or a supported Hugging Face reference.
  public static func fromPretrained(_ model: String, backend: Backend) async throws
    -> MediaGenerationPipeline
  {
    let preparedStorage = try await backend.preparedStorage()
    let resolvedModel: String
    if let localResolution = try await ModelResolver.resolve(
      model, offline: preparedStorage.offline)
    {
      resolvedModel = localResolution.file
    } else if backend.allowsUnresolvedModelReference {
      resolvedModel = model
    } else {
      throw MediaGenerationKitError.unresolvedModelReference(
        query: model,
        suggestions: await ModelResolver.suggestions(
          model,
          limit: 5,
          offline: preparedStorage.offline
        )
        .map(\.file)
      )
    }

    let recommendedTemplate = await ConfigurationResolver.recommendedTemplate(
      for: resolvedModel,
      loras: [],
      offline: preparedStorage.offline
    )
    let configuration = Configuration(
      configuration: recommendedTemplate,
      model: recommendedTemplate.model ?? resolvedModel
    )
    let storage = Storage(
      environment: preparedStorage.environment,
      remoteExecutor: preparedStorage.remoteExecutor,
      resolvedModel: resolvedModel,
      recommendedTemplate: recommendedTemplate
    )
    return MediaGenerationPipeline(
      backend: backend,
      configuration: configuration,
      model: configuration.model,
      storage: storage
    )
  }

  /// Runs generation with the current configuration.
  ///
  /// - Parameters:
  ///   - prompt: Positive prompt text.
  ///   - negativePrompt: Optional negative prompt text.
  ///   - inputs: Optional image inputs such as init image, mask, moodboard, or depth.
  ///   - stateHandler: Optional progress callback. `preview` is non-`nil` only for preview-bearing
  ///     progress updates.
  /// - Returns: One or more generated image results.
  @MainActor
  public func generate(
    prompt: String,
    negativePrompt: String = "",
    inputs: [Input] = [],
    stateHandler: ((State, Preview?) -> Void)? = nil
  ) async throws -> [Result] {
    try Task.checkCancellation()
    stateHandler?(.preparing, nil)
    return try await withTaskCancellationHandler {
      do {
        let results = try await storage.generate(
          prompt: prompt,
          negativePrompt: negativePrompt,
          configuration: configuration,
          inputs: inputs,
          logger: logger,
          stateHandler: stateHandler
        )
        stateHandler?(.completed, nil)
        return results
      } catch let error as MediaGenerationKitError {
        if case .cancelled = error {
          stateHandler?(.cancelled, nil)
          throw CancellationError()
        }
        throw error
      } catch is CancellationError {
        stateHandler?(.cancelled, nil)
        throw CancellationError()
      }
    } onCancel: {
      stateHandler?(.cancelling, nil)
    }
  }

  /// Estimates Draw Things cloud compute units for the current configuration and inputs.
  ///
  /// This uses the same runtime-configuration normalization and compute-unit estimator used by the
  /// cloud authentication path.
  public func estimatedComputeUnits(inputs: [Input] = []) throws -> Int? {
    let executionInputs = try Self.executionInputs(from: inputs)
    let runtimeConfiguration = try configuration.runtimeConfiguration(
      template: storage.recommendedTemplate)
    let shuffleCount = executionInputs.hints.reduce(0) { partialResult, hint in
      guard hint.type == .shuffle else { return partialResult }
      return partialResult + hint.images.count
    }
    return ComputeUnits.from(
      runtimeConfiguration,
      hasImage: executionInputs.image != nil,
      shuffleCount: shuffleCount
    )
  }

  internal static func executionInputs(from inputs: [Input]) throws
    -> MediaGenerationExecutionInputs
  {
    var image: Data?
    var mask: Data?
    var moodboardImages: [MediaGenerationExecutionHintImage] = []
    var depthImages: [MediaGenerationExecutionHintImage] = []

    for input in inputs {
      switch input.role {
      case .image:
        guard image == nil else {
          throw MediaGenerationKitError.generationFailed(
            "only one primary image input is supported")
        }
        image = try encodedImageData(from: input)
      case .mask:
        guard mask == nil else {
          throw MediaGenerationKitError.generationFailed("only one mask input is supported")
        }
        mask = try encodedImageData(from: input)
      case .moodboard:
        moodboardImages.append(
          MediaGenerationExecutionHintImage(
            data: try encodedImageData(from: input),
            weight: 1.0
          ))
      case .depth:
        depthImages.append(
          MediaGenerationExecutionHintImage(
            data: try encodedImageData(from: input),
            weight: 1.0
          ))
      }
    }

    var hints: [MediaGenerationExecutionHint] = []
    if !moodboardImages.isEmpty {
      hints.append(MediaGenerationExecutionHint(type: .shuffle, images: moodboardImages))
    }
    if !depthImages.isEmpty {
      hints.append(MediaGenerationExecutionHint(type: .depth, images: depthImages))
    }
    return MediaGenerationExecutionInputs(image: image, mask: mask, hints: hints)
  }

  fileprivate static func encodedImageData(from input: Input) throws -> Data {
    if let source = input as? MediaGenerationImageDataSource {
      return try source.mediaGenerationEncodedData()
    }
    if let masked = input as? MaskInput,
      let source = masked.source as? MediaGenerationImageDataSource
    {
      return try source.mediaGenerationEncodedData()
    }
    if let moodboard = input as? MoodboardInput,
      let source = moodboard.source as? MediaGenerationImageDataSource
    {
      return try source.mediaGenerationEncodedData()
    }
    if let depth = input as? DepthInput,
      let source = depth.source as? MediaGenerationImageDataSource
    {
      return try source.mediaGenerationEncodedData()
    }
    throw MediaGenerationKitError.generationFailed("unsupported input type \(type(of: input))")
  }
}

extension MediaGenerationImageInput {
  public var role: MediaGenerationPipeline.InputRole {
    .image
  }

  public func mask() -> MediaGenerationPipeline.Input {
    MediaGenerationPipeline.MaskInput(self)
  }

  public func moodboard() -> MediaGenerationPipeline.Input {
    MediaGenerationPipeline.MoodboardInput(self)
  }

  public func depth() -> MediaGenerationPipeline.Input {
    MediaGenerationPipeline.DepthInput(self)
  }
}

extension MediaGenerationPipeline.DataInput: MediaGenerationImageDataSource {
  func mediaGenerationEncodedData() throws -> Data {
    data
  }
}

extension MediaGenerationPipeline.FileInput: MediaGenerationImageDataSource {
  func mediaGenerationEncodedData() throws -> Data {
    try Data(contentsOf: URL(fileURLWithPath: path))
  }
}

extension MediaGenerationPipeline.Backend {
  fileprivate struct PreparedStorage {
    let environment: MediaGenerationEnvironment
    let remoteExecutor: MediaGenerationRemoteExecutor?
    let offline: Bool
  }

  fileprivate var allowsUnresolvedModelReference: Bool {
    switch self {
    case .local:
      return false
    case .remote, .cloudCompute:
      return true
    }
  }

  fileprivate func preparedStorage() async throws -> PreparedStorage {
    switch self {
    case .local(let directory):
      let environment = try MediaGenerationEnvironment.local(directory)
      _ = try environment.storage.modelsDirectoryURL()
      return PreparedStorage(
        environment: environment,
        remoteExecutor: nil,
        offline: false
      )
    case .remote(let endpoint, let options):
      let environment = MediaGenerationEnvironment.default
      let authentication: MediaGenerationRemoteAuthenticationMode =
        if let sharedSecret = options.sharedSecret {
          .sharedSecret(sharedSecret)
        } else {
          .none
        }
      let remoteExecutor = try await MediaGenerationRemoteExecutor.create(
        configuration:
          MediaGenerationRemoteConfiguration(
            serverURL: endpoint.host,
            port: endpoint.port,
            useTLS: options.useTLS,
            authentication: authentication
          )
      )
      return PreparedStorage(
        environment: environment,
        remoteExecutor: remoteExecutor,
        offline: false
      )
    case .cloudCompute(let apiKey, let options):
      let environment = MediaGenerationEnvironment.default
      let resolvedAPIKey = try MediaGenerationDefaults.resolveAPIKey(explicitAPIKey: apiKey)
      let resolvedBaseURL = options.baseURL ?? CloudConfiguration.defaultBaseURL
      let remoteExecutor = try await MediaGenerationRemoteExecutor.create(
        configuration:
          MediaGenerationRemoteConfiguration(
            serverURL: "compute.drawthings.ai",
            port: 443,
            useTLS: true,
            authentication: .cloudCompute(
              apiKey: resolvedAPIKey,
              baseURL: resolvedBaseURL,
              appCheck: options.appCheck
            ),
            deviceName: options.deviceName ?? "MediaGenerationKit"
          )
      )
      return PreparedStorage(
        environment: environment,
        remoteExecutor: remoteExecutor,
        offline: false
      )
    }
  }

}

extension CIImage: MediaGenerationImageInput {}

extension CIImage: MediaGenerationImageDataSource {
  func mediaGenerationEncodedData() throws -> Data {
    let context = CIContext(options: nil)
    let extent = self.extent.integral
    guard let cgImage = context.createCGImage(self, from: extent) else {
      throw MediaGenerationKitError.generationFailed("failed to render CIImage input")
    }
    let mutableData = NSMutableData()
    guard
      let destination = CGImageDestinationCreateWithData(
        mutableData,
        UTType.png.identifier as CFString,
        1,
        nil
      )
    else {
      throw MediaGenerationKitError.generationFailed("failed to create PNG destination")
    }
    CGImageDestinationAddImage(destination, cgImage, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw MediaGenerationKitError.generationFailed("failed to encode CIImage input as PNG")
    }
    return mutableData as Data
  }
}

extension CIImage {
  public convenience init(_ result: MediaGenerationPipeline.Result) {
    guard let imageData = try? MediaGenerationImageCodec.encode(result.tensor, type: UTType.png),
      let image = CIImage(data: imageData),
      let cgImage = CIContext(options: nil).createCGImage(image, from: image.extent)
    else {
      preconditionFailure("MediaGenerationPipeline.Result does not contain decodable image data")
    }
    self.init(cgImage: cgImage)
  }
}

#if canImport(UIKit)
  extension UIImage: MediaGenerationImageInput {}

  extension UIImage: MediaGenerationImageDataSource {
    func mediaGenerationEncodedData() throws -> Data {
      guard let data = pngData() else {
        throw MediaGenerationKitError.generationFailed("failed to encode UIImage input as PNG")
      }
      return data
    }
  }

  extension UIImage {
    public convenience init(_ result: MediaGenerationPipeline.Result) {
      guard let imageData = try? MediaGenerationImageCodec.encode(result.tensor, type: UTType.png),
        let image = UIImage(data: imageData)
      else {
        preconditionFailure("MediaGenerationPipeline.Result does not contain decodable image data")
      }
      guard let cgImage = image.cgImage else {
        self.init()
        return
      }
      self.init(cgImage: cgImage)
    }
  }
#endif
