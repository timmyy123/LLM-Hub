import DataModels
import Diffusion
import ModelZoo
import NNC
import Utils

public struct Invocation {
  public enum PrefersDefaultOptional<T> {
    case some(T)
    case none
    case prefersDefault
    public var optional: T? {
      switch self {
      case .some(let value):
        return value
      case .none, .prefersDefault:
        return nil
      }
    }
    public var isDefaultPreferred: Bool {
      switch self {
      case .prefersDefault:
        return true
      case .some(_), .none:
        return false
      }
    }
  }
  public var configuration: GenerationConfiguration
  public let image: PrefersDefaultOptional<Tensor<FloatType>>
  public let prompt: String?
  public let negativePrompt: String?
  public let mask: PrefersDefaultOptional<Tensor<UInt8>>
  public let fromHTTPServer: Bool

  public init(
    image: PrefersDefaultOptional<Tensor<FloatType>>, configuration: GenerationConfiguration,
    prompt: String?, negativePrompt: String?, mask: PrefersDefaultOptional<Tensor<UInt8>>,
    fromHTTPServer: Bool
  ) {
    self.image = image
    self.configuration = configuration
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.mask = mask
    self.fromHTTPServer = fromHTTPServer
  }

  public init(
    faceRestorationModel: String?,
    image: PrefersDefaultOptional<Tensor<FloatType>>, mask: PrefersDefaultOptional<Tensor<UInt8>>,
    parameters: Parameters, resizingOccurred: inout Bool, fromHTTPServer: Bool
  ) throws {
    let model = try unwrapOrThrow(
      parameters.modelParameter.value, errorMessage: "Missing 'model' parameter")
    guard
      let modelSpecification = ModelZoo.specificationForModel(model)
        ?? ModelZoo.specificationForHumanReadableModel(model)
    else {
      throw "Unrecognized model name \"\(model)\" (see --help for a list of available models)"
    }
    let upscaler = parameters.upscalerParameter.value
    let loras = parameters.lorasParameter.value.map { $0.createLora() }
    var zooAndFilesPairs: [(DownloadZoo.Type, [String])] =
      [
        (LoRAZoo.self, loras.compactMap { $0.file }),
        (UpscalerZoo.self, Array([upscaler].compactMap { $0 })),
      ]
    if let faceRestorationModel = faceRestorationModel {
      zooAndFilesPairs.append(
        (
          EverythingZoo.self,
          [faceRestorationModel, EverythingZoo.parsenetForModel(faceRestorationModel)]
        ))
    }

    for (zoo, files) in zooAndFilesPairs {
      for file in files {
        try Validation.validate(
          !file.contains("/") && !file.contains("\\"),
          errorMessage: "Model names may not have any '/' or '\' characters in them")
        try Validation.validate(
          zoo.isModelDownloaded(file),
          errorMessage: "Missing file: \(file) (in CLI, see --help to see the lists of files)")
      }
    }

    try Validation.validate(
      ModelZoo.isModelDownloaded(modelSpecification),
      errorMessage: "Missing files for model, see --help for a list of which files this model needs"
    )

    let refinerModel: String? = parameters.refinerModelParameter.value.flatMap {
      guard
        let modelSpecification = ModelZoo.specificationForModel($0)
          ?? ModelZoo.specificationForHumanReadableModel($0)
      else {
        return nil
      }
      guard ModelZoo.isModelDownloaded(modelSpecification) else { return nil }
      return modelSpecification.file
    }

    self.image = image
    let seed =
      parameters.seedParameter.value == -1
      ? UInt32(Int.random(in: parameters.seedParameter.range))
      : UInt32(parameters.seedParameter.value)
    let startWidth = parameters.widthParameter.value / 64
    let startHeight = parameters.heightParameter.value / 64
    let hiresFixStartWidth = parameters.hiresFixWidthParameter.value / 64
    let hiresFixStartHeight = parameters.hiresFixHeightParameter.value / 64
    self.configuration = GenerationConfiguration(
      id: 1,
      startWidth: UInt16(startWidth),
      startHeight: UInt16(startHeight), seed: seed,
      steps: parameters.stepsParameter.uint32Value(),
      guidanceScale: parameters.guidanceScaleParameter.float32Value(),
      strength: parameters.strengthParameter.float32Value(), model: modelSpecification.file,
      sampler: parameters.samplerParameter.value,
      batchCount: parameters.batchCountParameter.uint32Value(),
      batchSize: parameters.batchSizeParameter.uint32Value(),
      hiresFix: parameters.hiresFixParameter.value,
      hiresFixStartWidth: UInt16(hiresFixStartWidth),
      hiresFixStartHeight: UInt16(hiresFixStartHeight),
      hiresFixStrength: parameters.hiresFixStrengthParameter.float32Value(),
      upscaler: upscaler,
      imageGuidanceScale: parameters.imageGuidanceScaleParameter.float32Value(),
      seedMode: parameters.seedModeParameter.value,
      clipSkip: parameters.clipSkipParameter.uint32Value(),
      controls: parameters.controlsParameter.value.map { $0.createControl() },
      loras: loras, maskBlur: parameters.maskBlurParameter.float32Value(),
      faceRestoration: faceRestorationModel,
      clipWeight: parameters.clipWeightParameter.float32Value(),
      negativePromptForImagePrior: parameters.negativePromptForImagePriorParameter.value,
      imagePriorSteps: parameters.imagePriorStepsParameter.uint32Value(),
      refinerModel: refinerModel,
      originalImageHeight: parameters.originalHeightParameter.uint32Value(),
      originalImageWidth: parameters.originalWidthParameter.uint32Value(),
      cropTop: Int32(parameters.cropTopParameter.value),
      cropLeft: Int32(parameters.cropLeftParameter.value),
      targetImageHeight: parameters.targetHeightParameter.uint32Value(),
      targetImageWidth: parameters.targetWidthParameter.uint32Value(),
      aestheticScore: parameters.aestheticScoreParameter.float32Value(),
      negativeAestheticScore: parameters.negativeAestheticScoreParameter.float32Value(),
      zeroNegativePrompt: parameters.zeroNegativePromptParameter.value,
      refinerStart: parameters.refinerStartParameter.float32Value(),
      negativeOriginalImageHeight: parameters.negativeOriginalHeightParameter.uint32Value(),
      negativeOriginalImageWidth: parameters.negativeOriginalWidthParameter.uint32Value(),
      fpsId: parameters.fpsParameter.uint32Value(),
      motionBucketId: parameters.motionScaleParameter.uint32Value(),
      condAug: parameters.guidingFrameNoiseParameter.float32Value(),
      startFrameCfg: parameters.startFrameGuidanceParameter.float32Value(),
      numFrames: parameters.numFramesParameter.uint32Value(),
      maskBlurOutset: parameters.maskBlurOutsetParameter.int32Value(),
      sharpness: parameters.sharpnessParameter.float32Value(),
      shift: parameters.shiftParameter.float32Value(),
      stage2Steps: parameters.stage2StepsParameter.uint32Value(),
      stage2Cfg: parameters.stage2CfgParameter.float32Value(),
      stage2Shift: parameters.stage2ShiftParameter.float32Value(),
      tiledDecoding: parameters.tiledDecodingParameter.value,
      decodingTileWidth: UInt16(parameters.decodingTileWidthParameter.uint32Value() / 64),
      decodingTileHeight: UInt16(parameters.decodingTileHeightParameter.uint32Value() / 64),
      decodingTileOverlap: UInt16(parameters.decodingTileOverlapParameter.uint32Value() / 64),
      stochasticSamplingGamma: parameters.stochasticSamplingGammaParameter.float32Value(),
      preserveOriginalAfterInpaint: parameters.preserveOriginalAfterInpaintParameter.value,
      tiledDiffusion: parameters.tiledDiffusionParameter.value,
      diffusionTileWidth: UInt16(parameters.diffusionTileWidthParameter.uint32Value() / 64),
      diffusionTileHeight: UInt16(parameters.diffusionTileHeightParameter.uint32Value() / 64),
      diffusionTileOverlap: UInt16(parameters.diffusionTileOverlapParameter.uint32Value() / 64),
      upscalerScaleFactor: UInt8(parameters.upscalerScaleFactorParameter.uint32Value()),
      t5TextEncoder: parameters.t5TextEncoderParameter.value,
      separateClipL: parameters.separateClipLParameter.value,
      clipLText: parameters.clipLTextParameter.value,
      separateOpenClipG: parameters.separateOpenClipGParameter.value,
      openClipGText: parameters.openClipGTextParameter.value,
      speedUpWithGuidanceEmbed: parameters.speedUpWithGuidanceEmbedParameter.value,
      guidanceEmbed: parameters.guidanceEmbedParameter.float32Value(),
      resolutionDependentShift: parameters.resolutionDependentShiftParameter.value,
      teaCacheStart: parameters.teaCacheStartParameter.int32Value(),
      teaCacheEnd: parameters.teaCacheEndParameter.int32Value(),
      teaCacheThreshold: parameters.teaCacheThresholdParameter.float32Value(),
      teaCache: parameters.teaCacheParameter.value,
      separateT5: parameters.separateT5Parameter.value,
      t5Text: parameters.t5TextParameter.value,
      teaCacheMaxSkipSteps: parameters.teaCacheMaxSkipStepsParameter.int32Value(),
      causalInferenceEnabled: parameters.causalInferenceParameter.int32Value() > 0,
      causalInference: parameters.causalInferenceParameter.int32Value(),
      causalInferencePad: max(parameters.causalInferencePadParameter.int32Value(), 0),
      cfgZeroStar: parameters.cfgZeroStarParameter.value,
      cfgZeroInitSteps: parameters.cfgZeroInitStepsParameter.int32Value(),
      compressionArtifacts: parameters.compressionArtifactsParameter.value,
      compressionArtifactsQuality: parameters.compressionArtifactsQualityParameter.float32Value(),
      colorCalibration: parameters.colorCalibrationParameter.value,
      expandPromptToJson: parameters.expandPromptToJsonParameter.value,
      shiftForAudio: parameters.shiftForAudioParameter.float32Value(),
      usesSolAttention: parameters.usesSolAttentionParameter.value,
      solAttentionStart: parameters.solAttentionStartParameter.int32Value(),
      solAttentionTau: parameters.solAttentionTauParameter.float32Value()
    )
    self.prompt = try unwrapOrThrow(
      parameters.promptParameter.value, errorMessage: "Missing prompt")
    self.negativePrompt = parameters.negativePromptParameter.value ?? ""
    resizingOccurred =
      configuration.startWidth * 64 != parameters.widthParameter.value
      || configuration.startHeight * 64 != parameters.heightParameter.value
    self.mask = mask
    self.fromHTTPServer = fromHTTPServer
  }
}

extension Invocation: CustomDebugStringConvertible {
  // For debugging, print one property per line in sorted order to run `diff` between two invocations
  public var debugDescription: String {
    let pairs: [(String, Any)] = [
      ("prompt", prompt as Any),
      ("negativePrompt", negativePrompt as Any),
      ("startWidth", configuration.startWidth),
      ("startHeight", configuration.startHeight),
      ("seed", configuration.seed),
      ("steps", configuration.steps),
      ("guidanceScale", configuration.guidanceScale),
      ("strength", configuration.strength),
      ("model", configuration.model as Any),
      ("sampler", configuration.sampler),
      ("batchCount", configuration.batchCount),
      ("batchSize", configuration.batchSize),
      ("hiresFix", configuration.hiresFix),
      ("hiresFixStartWidth", configuration.hiresFixStartWidth),
      ("hiresFixStartHeight", configuration.hiresFixStartHeight),
      ("hiresFixStrength", configuration.hiresFixStrength),
      ("upscaler", configuration.upscaler as Any),
      ("imageGuidanceScale", configuration.imageGuidanceScale),
      ("seedMode", configuration.seedMode),
      ("clipSkip", configuration.clipSkip),
      ("controls", configuration.controls),
      ("loras", configuration.loras),
      ("maskBlur", configuration.maskBlur),
      ("maskBlurOutset", configuration.maskBlurOutset),
      ("sharpness", configuration.sharpness),
      ("faceRestoration", configuration.faceRestoration as Any),
      ("clipWeight", configuration.clipWeight),
      ("negativePromptForImagePrior", configuration.negativePromptForImagePrior),
      ("imagePriorSteps", configuration.imagePriorSteps),
      ("refinerModel", configuration.refinerModel as Any),
      ("originalImageHeight", configuration.originalImageHeight),
      ("originalImageWidth", configuration.originalImageWidth),
      ("cropTop", configuration.cropTop),
      ("cropLeft", configuration.cropLeft),
      ("targetImageHeight", configuration.targetImageHeight),
      ("targetImageWidth", configuration.targetImageWidth),
      ("aestheticScore", configuration.aestheticScore),
      ("negativeAestheticScore", configuration.negativeAestheticScore),
      ("zeroNegativePrompt", configuration.zeroNegativePrompt),
      ("refinerStart", configuration.refinerStart),
      ("fps", configuration.fpsId),
      ("motionScale", configuration.motionBucketId),
      ("guidingFrameNoise", configuration.condAug),
      ("startFrameGuidance", configuration.startFrameCfg),
      ("numFrames", configuration.numFrames),
      ("T5TextEncoder", configuration.t5TextEncoder),
      ("separateClipL", configuration.separateClipL),
      ("clipLText", configuration.clipLText as Any),
      ("separateOpenClipG", configuration.separateOpenClipG),
      ("openClipGText", configuration.openClipGText as Any),
      ("speedUpWithGuidanceEmbed", configuration.speedUpWithGuidanceEmbed as Any),
      ("guidanceEmbed", configuration.guidanceEmbed as Any),
      ("resolutionDependentShift", configuration.resolutionDependentShift as Any),
      ("teaCacheStart", configuration.teaCacheStart),
      ("teaCacheEnd", configuration.teaCacheEnd),
      ("teaCacheThreshold", configuration.teaCacheThreshold),
      ("teaCache", configuration.teaCache as Any),
      ("usesSolAttention", configuration.usesSolAttention as Any),
      ("solAttentionStart", configuration.solAttentionStart),
      ("solAttentionTau", configuration.solAttentionTau),
      ("separateT5", configuration.separateT5),
      ("t5Text", configuration.t5Text as Any),
      ("compressionArtifacts", configuration.compressionArtifacts),
      ("compressionArtifactsQuality", configuration.compressionArtifactsQuality),
      ("colorCalibration", configuration.colorCalibration),
      ("expandPromptToJson", configuration.expandPromptToJson),
    ]
    return pairs.map { (name, value) in
      "\(name): \(value)"
    }.joined(separator: "\n")
  }
}
