import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

// MARK - SQLiteValue for Enumerations

extension SamplerType: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

extension SeedMode: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

extension ControlMode: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

extension ControlInputType: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

extension LoRAMode: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

extension CompressionMethod: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

extension ColorCalibration: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

// MARK - Serializer

extension Control: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __file = self.file.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __controlMode = zzz_DflatGen_ControlMode(rawValue: self.controlMode.rawValue) ?? .balanced
    var __targetBlocks = [Offset]()
    for i in targetBlocks {
      __targetBlocks.append(flatBufferBuilder.create(string: i))
    }
    let __vector_targetBlocks = flatBufferBuilder.createVector(ofOffsets: __targetBlocks)
    let __inputOverride =
      zzz_DflatGen_ControlInputType(rawValue: self.inputOverride.rawValue) ?? .unspecified
    let start = zzz_DflatGen_Control.startControl(&flatBufferBuilder)
    zzz_DflatGen_Control.add(file: __file, &flatBufferBuilder)
    zzz_DflatGen_Control.add(weight: self.weight, &flatBufferBuilder)
    zzz_DflatGen_Control.add(guidanceStart: self.guidanceStart, &flatBufferBuilder)
    zzz_DflatGen_Control.add(guidanceEnd: self.guidanceEnd, &flatBufferBuilder)
    zzz_DflatGen_Control.add(noPrompt: self.noPrompt, &flatBufferBuilder)
    zzz_DflatGen_Control.add(globalAveragePooling: self.globalAveragePooling, &flatBufferBuilder)
    zzz_DflatGen_Control.add(downSamplingRate: self.downSamplingRate, &flatBufferBuilder)
    zzz_DflatGen_Control.add(controlMode: __controlMode, &flatBufferBuilder)
    zzz_DflatGen_Control.addVectorOf(targetBlocks: __vector_targetBlocks, &flatBufferBuilder)
    zzz_DflatGen_Control.add(inputOverride: __inputOverride, &flatBufferBuilder)
    return zzz_DflatGen_Control.endControl(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == Control {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension LoRA: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __file = self.file.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __mode = zzz_DflatGen_LoRAMode(rawValue: self.mode.rawValue) ?? .all
    let start = zzz_DflatGen_LoRA.startLoRA(&flatBufferBuilder)
    zzz_DflatGen_LoRA.add(file: __file, &flatBufferBuilder)
    zzz_DflatGen_LoRA.add(weight: self.weight, &flatBufferBuilder)
    zzz_DflatGen_LoRA.add(mode: __mode, &flatBufferBuilder)
    return zzz_DflatGen_LoRA.endLoRA(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == LoRA {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension GenerationConfiguration: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __model = self.model.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __sampler = zzz_DflatGen_SamplerType(rawValue: self.sampler.rawValue) ?? .dpmpp2mkarras
    let __upscaler = self.upscaler.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __seedMode = zzz_DflatGen_SeedMode(rawValue: self.seedMode.rawValue) ?? .legacy
    var __controls = [Offset]()
    for i in self.controls {
      __controls.append(i.to(flatBufferBuilder: &flatBufferBuilder))
    }
    let __vector_controls = flatBufferBuilder.createVector(ofOffsets: __controls)
    var __loras = [Offset]()
    for i in self.loras {
      __loras.append(i.to(flatBufferBuilder: &flatBufferBuilder))
    }
    let __vector_loras = flatBufferBuilder.createVector(ofOffsets: __loras)
    let __faceRestoration =
      self.faceRestoration.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __refinerModel = self.refinerModel.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __name = self.name.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __clipLText = self.clipLText.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __openClipGText =
      self.openClipGText.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __t5Text = self.t5Text.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __compressionArtifacts =
      zzz_DflatGen_CompressionMethod(rawValue: self.compressionArtifacts.rawValue) ?? .disabled
    let __colorCalibration =
      zzz_DflatGen_ColorCalibration(rawValue: self.colorCalibration.rawValue) ?? .disabled
    let start = zzz_DflatGen_GenerationConfiguration.startGenerationConfiguration(
      &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(id: self.id, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(startWidth: self.startWidth, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(startHeight: self.startHeight, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(seed: self.seed, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(steps: self.steps, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(guidanceScale: self.guidanceScale, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(strength: self.strength, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(model: __model, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(sampler: __sampler, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(batchCount: self.batchCount, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(batchSize: self.batchSize, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(hiresFix: self.hiresFix, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      hiresFixStartWidth: self.hiresFixStartWidth, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      hiresFixStartHeight: self.hiresFixStartHeight, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      hiresFixStrength: self.hiresFixStrength, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(upscaler: __upscaler, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      imageGuidanceScale: self.imageGuidanceScale, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(seedMode: __seedMode, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(clipSkip: self.clipSkip, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.addVectorOf(
      controls: __vector_controls, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.addVectorOf(loras: __vector_loras, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(maskBlur: self.maskBlur, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(faceRestoration: __faceRestoration, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(clipWeight: self.clipWeight, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      negativePromptForImagePrior: self.negativePromptForImagePrior, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      imagePriorSteps: self.imagePriorSteps, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(refinerModel: __refinerModel, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      originalImageHeight: self.originalImageHeight, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      originalImageWidth: self.originalImageWidth, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(cropTop: self.cropTop, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(cropLeft: self.cropLeft, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      targetImageHeight: self.targetImageHeight, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      targetImageWidth: self.targetImageWidth, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      aestheticScore: self.aestheticScore, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      negativeAestheticScore: self.negativeAestheticScore, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      zeroNegativePrompt: self.zeroNegativePrompt, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(refinerStart: self.refinerStart, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      negativeOriginalImageHeight: self.negativeOriginalImageHeight, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      negativeOriginalImageWidth: self.negativeOriginalImageWidth, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(name: __name, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(fpsId: self.fpsId, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      motionBucketId: self.motionBucketId, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(condAug: self.condAug, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(startFrameCfg: self.startFrameCfg, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(numFrames: self.numFrames, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      maskBlurOutset: self.maskBlurOutset, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(sharpness: self.sharpness, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(shift: self.shift, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(stage2Steps: self.stage2Steps, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(stage2Cfg: self.stage2Cfg, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(stage2Shift: self.stage2Shift, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(tiledDecoding: self.tiledDecoding, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      decodingTileWidth: self.decodingTileWidth, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      decodingTileHeight: self.decodingTileHeight, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      decodingTileOverlap: self.decodingTileOverlap, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      stochasticSamplingGamma: self.stochasticSamplingGamma, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      preserveOriginalAfterInpaint: self.preserveOriginalAfterInpaint, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      tiledDiffusion: self.tiledDiffusion, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      diffusionTileWidth: self.diffusionTileWidth, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      diffusionTileHeight: self.diffusionTileHeight, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      diffusionTileOverlap: self.diffusionTileOverlap, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      upscalerScaleFactor: self.upscalerScaleFactor, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(t5TextEncoder: self.t5TextEncoder, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(separateClipL: self.separateClipL, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(clipLText: __clipLText, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      separateOpenClipG: self.separateOpenClipG, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(openClipGText: __openClipGText, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      speedUpWithGuidanceEmbed: self.speedUpWithGuidanceEmbed, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(guidanceEmbed: self.guidanceEmbed, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      resolutionDependentShift: self.resolutionDependentShift, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(teaCacheStart: self.teaCacheStart, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(teaCacheEnd: self.teaCacheEnd, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      teaCacheThreshold: self.teaCacheThreshold, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(teaCache: self.teaCache, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(separateT5: self.separateT5, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(t5Text: __t5Text, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      teaCacheMaxSkipSteps: self.teaCacheMaxSkipSteps, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      causalInferenceEnabled: self.causalInferenceEnabled, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      causalInference: self.causalInference, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      causalInferencePad: self.causalInferencePad, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(cfgZeroStar: self.cfgZeroStar, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      cfgZeroInitSteps: self.cfgZeroInitSteps, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      compressionArtifacts: __compressionArtifacts, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      compressionArtifactsQuality: self.compressionArtifactsQuality, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      colorCalibration: __colorCalibration, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      expandPromptToJson: self.expandPromptToJson, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(shiftForAudio: self.shiftForAudio, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      usesSolAttention: self.usesSolAttention, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      solAttentionStart: self.solAttentionStart, &flatBufferBuilder)
    zzz_DflatGen_GenerationConfiguration.add(
      solAttentionTau: self.solAttentionTau, &flatBufferBuilder)
    return zzz_DflatGen_GenerationConfiguration.endGenerationConfiguration(
      &flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == GenerationConfiguration {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension GenerationConfiguration {
  public func toData() -> Data {
    var fbb = FlatBufferBuilder()
    let offset = to(flatBufferBuilder: &fbb)
    fbb.finish(offset: offset)
    return fbb.data
  }
}

// MARK - ChangeRequest

public final class GenerationConfigurationChangeRequest: Dflat.ChangeRequest {
  private var _o: GenerationConfiguration?
  public typealias Element = GenerationConfiguration
  public var _type: ChangeRequestType
  public var _rowid: Int64
  public var id: Int64
  public var startWidth: UInt16
  public var startHeight: UInt16
  public var seed: UInt32
  public var steps: UInt32
  public var guidanceScale: Float32
  public var strength: Float32
  public var model: String?
  public var sampler: SamplerType
  public var batchCount: UInt32
  public var batchSize: UInt32
  public var hiresFix: Bool
  public var hiresFixStartWidth: UInt16
  public var hiresFixStartHeight: UInt16
  public var hiresFixStrength: Float32
  public var upscaler: String?
  public var imageGuidanceScale: Float32
  public var seedMode: SeedMode
  public var clipSkip: UInt32
  public var controls: [Control]
  public var loras: [LoRA]
  public var maskBlur: Float32
  public var faceRestoration: String?
  public var clipWeight: Float32
  public var negativePromptForImagePrior: Bool
  public var imagePriorSteps: UInt32
  public var refinerModel: String?
  public var originalImageHeight: UInt32
  public var originalImageWidth: UInt32
  public var cropTop: Int32
  public var cropLeft: Int32
  public var targetImageHeight: UInt32
  public var targetImageWidth: UInt32
  public var aestheticScore: Float32
  public var negativeAestheticScore: Float32
  public var zeroNegativePrompt: Bool
  public var refinerStart: Float32
  public var negativeOriginalImageHeight: UInt32
  public var negativeOriginalImageWidth: UInt32
  public var name: String?
  public var fpsId: UInt32
  public var motionBucketId: UInt32
  public var condAug: Float32
  public var startFrameCfg: Float32
  public var numFrames: UInt32
  public var maskBlurOutset: Int32
  public var sharpness: Float32
  public var shift: Float32
  public var stage2Steps: UInt32
  public var stage2Cfg: Float32
  public var stage2Shift: Float32
  public var tiledDecoding: Bool
  public var decodingTileWidth: UInt16
  public var decodingTileHeight: UInt16
  public var decodingTileOverlap: UInt16
  public var stochasticSamplingGamma: Float32
  public var preserveOriginalAfterInpaint: Bool
  public var tiledDiffusion: Bool
  public var diffusionTileWidth: UInt16
  public var diffusionTileHeight: UInt16
  public var diffusionTileOverlap: UInt16
  public var upscalerScaleFactor: UInt8
  public var t5TextEncoder: Bool
  public var separateClipL: Bool
  public var clipLText: String?
  public var separateOpenClipG: Bool
  public var openClipGText: String?
  public var speedUpWithGuidanceEmbed: Bool
  public var guidanceEmbed: Float32
  public var resolutionDependentShift: Bool
  public var teaCacheStart: Int32
  public var teaCacheEnd: Int32
  public var teaCacheThreshold: Float32
  public var teaCache: Bool
  public var separateT5: Bool
  public var t5Text: String?
  public var teaCacheMaxSkipSteps: Int32
  public var causalInferenceEnabled: Bool
  public var causalInference: Int32
  public var causalInferencePad: Int32
  public var cfgZeroStar: Bool
  public var cfgZeroInitSteps: Int32
  public var compressionArtifacts: CompressionMethod
  public var compressionArtifactsQuality: Float32
  public var colorCalibration: ColorCalibration
  public var expandPromptToJson: Bool
  public var shiftForAudio: Float32
  public var usesSolAttention: Bool
  public var solAttentionStart: Int32
  public var solAttentionTau: Float32
  private init(type _type: ChangeRequestType) {
    _o = nil
    self._type = _type
    _rowid = -1
    id = 0
    startWidth = 0
    startHeight = 0
    seed = 0
    steps = 0
    guidanceScale = 0.0
    strength = 0.0
    model = nil
    sampler = .dPMPP2MKarras
    batchCount = 1
    batchSize = 1
    hiresFix = false
    hiresFixStartWidth = 0
    hiresFixStartHeight = 0
    hiresFixStrength = 0.7
    upscaler = nil
    imageGuidanceScale = 1.5
    seedMode = .legacy
    clipSkip = 1
    controls = []
    loras = []
    maskBlur = 0.0
    faceRestoration = nil
    clipWeight = 1.0
    negativePromptForImagePrior = true
    imagePriorSteps = 5
    refinerModel = nil
    originalImageHeight = 0
    originalImageWidth = 0
    cropTop = 0
    cropLeft = 0
    targetImageHeight = 0
    targetImageWidth = 0
    aestheticScore = 6.0
    negativeAestheticScore = 2.5
    zeroNegativePrompt = false
    refinerStart = 0.7
    negativeOriginalImageHeight = 0
    negativeOriginalImageWidth = 0
    name = nil
    fpsId = 5
    motionBucketId = 127
    condAug = 0.02
    startFrameCfg = 1.0
    numFrames = 14
    maskBlurOutset = 0
    sharpness = 0.0
    shift = 1.0
    stage2Steps = 10
    stage2Cfg = 1.0
    stage2Shift = 1.0
    tiledDecoding = false
    decodingTileWidth = 10
    decodingTileHeight = 10
    decodingTileOverlap = 2
    stochasticSamplingGamma = 0.3
    preserveOriginalAfterInpaint = true
    tiledDiffusion = false
    diffusionTileWidth = 16
    diffusionTileHeight = 16
    diffusionTileOverlap = 2
    upscalerScaleFactor = 0
    t5TextEncoder = true
    separateClipL = false
    clipLText = nil
    separateOpenClipG = false
    openClipGText = nil
    speedUpWithGuidanceEmbed = true
    guidanceEmbed = 3.5
    resolutionDependentShift = true
    teaCacheStart = 5
    teaCacheEnd = -1
    teaCacheThreshold = 0.06
    teaCache = false
    separateT5 = false
    t5Text = nil
    teaCacheMaxSkipSteps = 3
    causalInferenceEnabled = false
    causalInference = 3
    causalInferencePad = 0
    cfgZeroStar = false
    cfgZeroInitSteps = 0
    compressionArtifacts = .disabled
    compressionArtifactsQuality = 43.1
    colorCalibration = .disabled
    expandPromptToJson = false
    shiftForAudio = 3.0
    usesSolAttention = false
    solAttentionStart = 2
    solAttentionTau = 0.5
  }
  private init(type _type: ChangeRequestType, _ _o: GenerationConfiguration) {
    self._o = _o
    self._type = _type
    _rowid = _o._rowid
    id = _o.id
    startWidth = _o.startWidth
    startHeight = _o.startHeight
    seed = _o.seed
    steps = _o.steps
    guidanceScale = _o.guidanceScale
    strength = _o.strength
    model = _o.model
    sampler = _o.sampler
    batchCount = _o.batchCount
    batchSize = _o.batchSize
    hiresFix = _o.hiresFix
    hiresFixStartWidth = _o.hiresFixStartWidth
    hiresFixStartHeight = _o.hiresFixStartHeight
    hiresFixStrength = _o.hiresFixStrength
    upscaler = _o.upscaler
    imageGuidanceScale = _o.imageGuidanceScale
    seedMode = _o.seedMode
    clipSkip = _o.clipSkip
    controls = _o.controls
    loras = _o.loras
    maskBlur = _o.maskBlur
    faceRestoration = _o.faceRestoration
    clipWeight = _o.clipWeight
    negativePromptForImagePrior = _o.negativePromptForImagePrior
    imagePriorSteps = _o.imagePriorSteps
    refinerModel = _o.refinerModel
    originalImageHeight = _o.originalImageHeight
    originalImageWidth = _o.originalImageWidth
    cropTop = _o.cropTop
    cropLeft = _o.cropLeft
    targetImageHeight = _o.targetImageHeight
    targetImageWidth = _o.targetImageWidth
    aestheticScore = _o.aestheticScore
    negativeAestheticScore = _o.negativeAestheticScore
    zeroNegativePrompt = _o.zeroNegativePrompt
    refinerStart = _o.refinerStart
    negativeOriginalImageHeight = _o.negativeOriginalImageHeight
    negativeOriginalImageWidth = _o.negativeOriginalImageWidth
    name = _o.name
    fpsId = _o.fpsId
    motionBucketId = _o.motionBucketId
    condAug = _o.condAug
    startFrameCfg = _o.startFrameCfg
    numFrames = _o.numFrames
    maskBlurOutset = _o.maskBlurOutset
    sharpness = _o.sharpness
    shift = _o.shift
    stage2Steps = _o.stage2Steps
    stage2Cfg = _o.stage2Cfg
    stage2Shift = _o.stage2Shift
    tiledDecoding = _o.tiledDecoding
    decodingTileWidth = _o.decodingTileWidth
    decodingTileHeight = _o.decodingTileHeight
    decodingTileOverlap = _o.decodingTileOverlap
    stochasticSamplingGamma = _o.stochasticSamplingGamma
    preserveOriginalAfterInpaint = _o.preserveOriginalAfterInpaint
    tiledDiffusion = _o.tiledDiffusion
    diffusionTileWidth = _o.diffusionTileWidth
    diffusionTileHeight = _o.diffusionTileHeight
    diffusionTileOverlap = _o.diffusionTileOverlap
    upscalerScaleFactor = _o.upscalerScaleFactor
    t5TextEncoder = _o.t5TextEncoder
    separateClipL = _o.separateClipL
    clipLText = _o.clipLText
    separateOpenClipG = _o.separateOpenClipG
    openClipGText = _o.openClipGText
    speedUpWithGuidanceEmbed = _o.speedUpWithGuidanceEmbed
    guidanceEmbed = _o.guidanceEmbed
    resolutionDependentShift = _o.resolutionDependentShift
    teaCacheStart = _o.teaCacheStart
    teaCacheEnd = _o.teaCacheEnd
    teaCacheThreshold = _o.teaCacheThreshold
    teaCache = _o.teaCache
    separateT5 = _o.separateT5
    t5Text = _o.t5Text
    teaCacheMaxSkipSteps = _o.teaCacheMaxSkipSteps
    causalInferenceEnabled = _o.causalInferenceEnabled
    causalInference = _o.causalInference
    causalInferencePad = _o.causalInferencePad
    cfgZeroStar = _o.cfgZeroStar
    cfgZeroInitSteps = _o.cfgZeroInitSteps
    compressionArtifacts = _o.compressionArtifacts
    compressionArtifactsQuality = _o.compressionArtifactsQuality
    colorCalibration = _o.colorCalibration
    expandPromptToJson = _o.expandPromptToJson
    shiftForAudio = _o.shiftForAudio
    usesSolAttention = _o.usesSolAttention
    solAttentionStart = _o.solAttentionStart
    solAttentionTau = _o.solAttentionTau
  }
  public static func changeRequest(_ o: GenerationConfiguration)
    -> GenerationConfigurationChangeRequest?
  {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.id])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: GenerationConfiguration.self, for: key)
    return u.map { GenerationConfigurationChangeRequest(type: .update, $0) }
  }
  public static func upsertRequest(_ o: GenerationConfiguration)
    -> GenerationConfigurationChangeRequest
  {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.id])
    guard
      let u = transactionContext.objectRepository.object(
        transactionContext.connection, ofType: GenerationConfiguration.self, for: key)
    else {
      return Self.creationRequest(o)
    }
    let changeRequest = GenerationConfigurationChangeRequest(type: .update, o)
    changeRequest._o = u
    changeRequest._rowid = u._rowid
    return changeRequest
  }
  public static func creationRequest(_ o: GenerationConfiguration)
    -> GenerationConfigurationChangeRequest
  {
    let creationRequest = GenerationConfigurationChangeRequest(type: .creation, o)
    creationRequest._rowid = -1
    return creationRequest
  }
  public static func creationRequest() -> GenerationConfigurationChangeRequest {
    return GenerationConfigurationChangeRequest(type: .creation)
  }
  public static func deletionRequest(_ o: GenerationConfiguration)
    -> GenerationConfigurationChangeRequest?
  {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.id])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: GenerationConfiguration.self, for: key)
    return u.map { GenerationConfigurationChangeRequest(type: .deletion, $0) }
  }
  var _atom: GenerationConfiguration {
    let atom = GenerationConfiguration(
      id: id, startWidth: startWidth, startHeight: startHeight, seed: seed, steps: steps,
      guidanceScale: guidanceScale, strength: strength, model: model, sampler: sampler,
      batchCount: batchCount, batchSize: batchSize, hiresFix: hiresFix,
      hiresFixStartWidth: hiresFixStartWidth, hiresFixStartHeight: hiresFixStartHeight,
      hiresFixStrength: hiresFixStrength, upscaler: upscaler,
      imageGuidanceScale: imageGuidanceScale, seedMode: seedMode, clipSkip: clipSkip,
      controls: controls, loras: loras, maskBlur: maskBlur, faceRestoration: faceRestoration,
      clipWeight: clipWeight, negativePromptForImagePrior: negativePromptForImagePrior,
      imagePriorSteps: imagePriorSteps, refinerModel: refinerModel,
      originalImageHeight: originalImageHeight, originalImageWidth: originalImageWidth,
      cropTop: cropTop, cropLeft: cropLeft, targetImageHeight: targetImageHeight,
      targetImageWidth: targetImageWidth, aestheticScore: aestheticScore,
      negativeAestheticScore: negativeAestheticScore, zeroNegativePrompt: zeroNegativePrompt,
      refinerStart: refinerStart, negativeOriginalImageHeight: negativeOriginalImageHeight,
      negativeOriginalImageWidth: negativeOriginalImageWidth, name: name, fpsId: fpsId,
      motionBucketId: motionBucketId, condAug: condAug, startFrameCfg: startFrameCfg,
      numFrames: numFrames, maskBlurOutset: maskBlurOutset, sharpness: sharpness, shift: shift,
      stage2Steps: stage2Steps, stage2Cfg: stage2Cfg, stage2Shift: stage2Shift,
      tiledDecoding: tiledDecoding, decodingTileWidth: decodingTileWidth,
      decodingTileHeight: decodingTileHeight, decodingTileOverlap: decodingTileOverlap,
      stochasticSamplingGamma: stochasticSamplingGamma,
      preserveOriginalAfterInpaint: preserveOriginalAfterInpaint, tiledDiffusion: tiledDiffusion,
      diffusionTileWidth: diffusionTileWidth, diffusionTileHeight: diffusionTileHeight,
      diffusionTileOverlap: diffusionTileOverlap, upscalerScaleFactor: upscalerScaleFactor,
      t5TextEncoder: t5TextEncoder, separateClipL: separateClipL, clipLText: clipLText,
      separateOpenClipG: separateOpenClipG, openClipGText: openClipGText,
      speedUpWithGuidanceEmbed: speedUpWithGuidanceEmbed, guidanceEmbed: guidanceEmbed,
      resolutionDependentShift: resolutionDependentShift, teaCacheStart: teaCacheStart,
      teaCacheEnd: teaCacheEnd, teaCacheThreshold: teaCacheThreshold, teaCache: teaCache,
      separateT5: separateT5, t5Text: t5Text, teaCacheMaxSkipSteps: teaCacheMaxSkipSteps,
      causalInferenceEnabled: causalInferenceEnabled, causalInference: causalInference,
      causalInferencePad: causalInferencePad, cfgZeroStar: cfgZeroStar,
      cfgZeroInitSteps: cfgZeroInitSteps, compressionArtifacts: compressionArtifacts,
      compressionArtifactsQuality: compressionArtifactsQuality, colorCalibration: colorCalibration,
      expandPromptToJson: expandPromptToJson, shiftForAudio: shiftForAudio,
      usesSolAttention: usesSolAttention, solAttentionStart: solAttentionStart,
      solAttentionTau: solAttentionTau)
    atom._rowid = _rowid
    return atom
  }
  public func commit(_ toolbox: PersistenceToolbox) -> UpdatedObject? {
    guard let toolbox = toolbox as? SQLitePersistenceToolbox else { return nil }
    switch _type {
    case .creation:
      let indexSurvey = toolbox.connection.indexSurvey(
        GenerationConfiguration.indexFields, table: GenerationConfiguration.table)
      guard
        let insert = toolbox.connection.prepareStaticStatement(
          "INSERT INTO generationconfiguration (__pk0, p) VALUES (?1, ?2)")
      else { return nil }
      id.bindSQLite(insert, parameterId: 1)
      let atom = self._atom
      toolbox.flatBufferBuilder.clear()
      let offset = atom.to(flatBufferBuilder: &toolbox.flatBufferBuilder)
      toolbox.flatBufferBuilder.finish(offset: offset)
      let byteBuffer = toolbox.flatBufferBuilder.buffer
      let memory = byteBuffer.memory.advanced(by: byteBuffer.reader)
      let SQLITE_STATIC = unsafeBitCast(
        OpaquePointer(bitPattern: 0), to: sqlite3_destructor_type.self)
      sqlite3_bind_blob(insert, 2, memory, Int32(byteBuffer.size), SQLITE_STATIC)
      guard SQLITE_DONE == sqlite3_step(insert) else { return nil }
      _rowid = sqlite3_last_insert_rowid(toolbox.connection.sqlite)
      if indexSurvey.full.contains("f86") {
        guard
          let i0 = toolbox.connection.prepareStaticStatement(
            "INSERT INTO generationconfiguration__f86 (rowid, f86) VALUES (?1, ?2)")
        else { return nil }
        _rowid.bindSQLite(i0, parameterId: 1)
        if let r0 = GenerationConfiguration.name.evaluate(object: atom) {
          r0.bindSQLite(i0, parameterId: 2)
        } else {
          sqlite3_bind_null(i0, 2)
        }
        guard SQLITE_DONE == sqlite3_step(i0) else { return nil }
      }
      _type = .none
      atom._rowid = _rowid
      return .inserted(atom)
    case .update:
      guard let o = _o else { return nil }
      let atom = self._atom
      guard atom != o else {
        _type = .none
        return .identity(atom)
      }
      let indexSurvey = toolbox.connection.indexSurvey(
        GenerationConfiguration.indexFields, table: GenerationConfiguration.table)
      guard
        let update = toolbox.connection.prepareStaticStatement(
          "REPLACE INTO generationconfiguration (__pk0, p, rowid) VALUES (?1, ?2, ?3)")
      else { return nil }
      id.bindSQLite(update, parameterId: 1)
      toolbox.flatBufferBuilder.clear()
      let offset = atom.to(flatBufferBuilder: &toolbox.flatBufferBuilder)
      toolbox.flatBufferBuilder.finish(offset: offset)
      let byteBuffer = toolbox.flatBufferBuilder.buffer
      let memory = byteBuffer.memory.advanced(by: byteBuffer.reader)
      let SQLITE_STATIC = unsafeBitCast(
        OpaquePointer(bitPattern: 0), to: sqlite3_destructor_type.self)
      sqlite3_bind_blob(update, 2, memory, Int32(byteBuffer.size), SQLITE_STATIC)
      _rowid.bindSQLite(update, parameterId: 3)
      guard SQLITE_DONE == sqlite3_step(update) else { return nil }
      if indexSurvey.full.contains("f86") {
        let or0 = GenerationConfiguration.name.evaluate(object: o)
        let r0 = GenerationConfiguration.name.evaluate(object: atom)
        if or0 != r0 {
          guard
            let u0 = toolbox.connection.prepareStaticStatement(
              "REPLACE INTO generationconfiguration__f86 (rowid, f86) VALUES (?1, ?2)")
          else { return nil }
          _rowid.bindSQLite(u0, parameterId: 1)
          if let ur0 = r0 {
            ur0.bindSQLite(u0, parameterId: 2)
          } else {
            sqlite3_bind_null(u0, 2)
          }
          guard SQLITE_DONE == sqlite3_step(u0) else { return nil }
        }
      }
      _type = .none
      return .updated(atom)
    case .deletion:
      guard
        let deletion = toolbox.connection.prepareStaticStatement(
          "DELETE FROM generationconfiguration WHERE rowid=?1")
      else { return nil }
      _rowid.bindSQLite(deletion, parameterId: 1)
      guard SQLITE_DONE == sqlite3_step(deletion) else { return nil }
      if let d0 = toolbox.connection.prepareStaticStatement(
        "DELETE FROM generationconfiguration__f86 WHERE rowid=?1")
      {
        _rowid.bindSQLite(d0, parameterId: 1)
        sqlite3_step(d0)
      }
      _type = .none
      return .deleted(_rowid)
    case .none:
      preconditionFailure()
    }
  }
}
