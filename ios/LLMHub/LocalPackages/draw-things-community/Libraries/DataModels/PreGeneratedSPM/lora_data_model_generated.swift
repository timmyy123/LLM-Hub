import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

public enum LoRATrainableLayer: Int8, DflatFriendlyValue, CaseIterable, Codable {
  case latentsEmbedder = 0
  case contextEmbedder = 1
  case projectOut = 2
  case qkv = 3
  case qkvContext = 4
  case out = 5
  case outContext = 6
  case feedForward = 7
  case feedForwardContext = 8
  public static func < (lhs: LoRATrainableLayer, rhs: LoRATrainableLayer) -> Bool {
    return lhs.rawValue < rhs.rawValue
  }
}

public final class LoRATrainingConfiguration: Dflat.Atom, SQLiteDflat.SQLiteAtom,
  FlatBuffersDecodable, Equatable, Codable
{
  public static func == (lhs: LoRATrainingConfiguration, rhs: LoRATrainingConfiguration) -> Bool {
    guard lhs.id == rhs.id else { return false }
    guard lhs.name == rhs.name else { return false }
    guard lhs.startWidth == rhs.startWidth else { return false }
    guard lhs.startHeight == rhs.startHeight else { return false }
    guard lhs.seed == rhs.seed else { return false }
    guard lhs.trainingSteps == rhs.trainingSteps else { return false }
    guard lhs.baseModel == rhs.baseModel else { return false }
    guard lhs.networkDim == rhs.networkDim else { return false }
    guard lhs.networkScale == rhs.networkScale else { return false }
    guard lhs.unetLearningRate == rhs.unetLearningRate else { return false }
    guard lhs.saveEveryNSteps == rhs.saveEveryNSteps else { return false }
    guard lhs.warmupSteps == rhs.warmupSteps else { return false }
    guard lhs.gradientAccumulationSteps == rhs.gradientAccumulationSteps else { return false }
    guard lhs.cotrainTextModel == rhs.cotrainTextModel else { return false }
    guard lhs.textModelLearningRate == rhs.textModelLearningRate else { return false }
    guard lhs.clipSkip == rhs.clipSkip else { return false }
    guard lhs.noiseOffset == rhs.noiseOffset else { return false }
    guard lhs.denoisingStart == rhs.denoisingStart else { return false }
    guard lhs.denoisingEnd == rhs.denoisingEnd else { return false }
    guard lhs.triggerWord == rhs.triggerWord else { return false }
    guard lhs.autoFillPrompt == rhs.autoFillPrompt else { return false }
    guard lhs.autoCaptioning == rhs.autoCaptioning else { return false }
    guard lhs.cotrainCustomEmbedding == rhs.cotrainCustomEmbedding else { return false }
    guard lhs.customEmbeddingLearningRate == rhs.customEmbeddingLearningRate else { return false }
    guard lhs.customEmbeddingLength == rhs.customEmbeddingLength else { return false }
    guard lhs.stopEmbeddingTrainingAtStep == rhs.stopEmbeddingTrainingAtStep else { return false }
    guard lhs.trainableLayers == rhs.trainableLayers else { return false }
    guard lhs.layerIndices == rhs.layerIndices else { return false }
    guard lhs.shift == rhs.shift else { return false }
    guard lhs.resolutionDependentShift == rhs.resolutionDependentShift else { return false }
    guard lhs.guidanceEmbedLowerBound == rhs.guidanceEmbedLowerBound else { return false }
    guard lhs.guidanceEmbedUpperBound == rhs.guidanceEmbedUpperBound else { return false }
    guard lhs.unetLearningRateLowerBound == rhs.unetLearningRateLowerBound else { return false }
    guard lhs.stepsBetweenRestarts == rhs.stepsBetweenRestarts else { return false }
    guard lhs.captionDropoutRate == rhs.captionDropoutRate else { return false }
    guard lhs.orthonormalLoraDown == rhs.orthonormalLoraDown else { return false }
    guard lhs.maxTextLength == rhs.maxTextLength else { return false }
    guard lhs.useImageAspectRatio == rhs.useImageAspectRatio else { return false }
    guard lhs.additionalScales == rhs.additionalScales else { return false }
    guard lhs.powerEmaLowerBound == rhs.powerEmaLowerBound else { return false }
    guard lhs.powerEmaUpperBound == rhs.powerEmaUpperBound else { return false }
    return true
  }
  public var _rowid: Int64 = -1
  public var _changesTimestamp: Int64 = -1
  public let id: Int64
  public let name: String?
  public let startWidth: UInt16
  public let startHeight: UInt16
  public let seed: UInt32
  public let trainingSteps: UInt32
  public let baseModel: String?
  public let networkDim: UInt16
  public let networkScale: Float32
  public let unetLearningRate: Float32
  public let saveEveryNSteps: UInt32
  public let warmupSteps: UInt32
  public let gradientAccumulationSteps: UInt32
  public let cotrainTextModel: Bool
  public let textModelLearningRate: Float32
  public let clipSkip: UInt32
  public let noiseOffset: Float32
  public let denoisingStart: Float32
  public let denoisingEnd: Float32
  public let triggerWord: String?
  public let autoFillPrompt: String?
  public let autoCaptioning: Bool
  public let cotrainCustomEmbedding: Bool
  public let customEmbeddingLearningRate: Float32
  public let customEmbeddingLength: UInt32
  public let stopEmbeddingTrainingAtStep: UInt32
  public let trainableLayers: [LoRATrainableLayer]
  public let layerIndices: [UInt32]
  public let shift: Float32
  public let resolutionDependentShift: Bool
  public let guidanceEmbedLowerBound: Float32
  public let guidanceEmbedUpperBound: Float32
  public let unetLearningRateLowerBound: Float32
  public let stepsBetweenRestarts: UInt32
  public let captionDropoutRate: Float32
  public let orthonormalLoraDown: Bool
  public let maxTextLength: UInt32
  public let useImageAspectRatio: Bool
  public let additionalScales: [UInt16]
  public let powerEmaLowerBound: Float32
  public let powerEmaUpperBound: Float32
  public init(
    id: Int64, name: String? = nil, startWidth: UInt16? = 0, startHeight: UInt16? = 0,
    seed: UInt32? = 0, trainingSteps: UInt32? = 0, baseModel: String? = nil,
    networkDim: UInt16? = 0, networkScale: Float32? = 0.0, unetLearningRate: Float32? = 0.0,
    saveEveryNSteps: UInt32? = 0, warmupSteps: UInt32? = 0, gradientAccumulationSteps: UInt32? = 0,
    cotrainTextModel: Bool? = false, textModelLearningRate: Float32? = 0.0, clipSkip: UInt32? = 1,
    noiseOffset: Float32? = 0.0, denoisingStart: Float32? = 0.0, denoisingEnd: Float32? = 0.0,
    triggerWord: String? = nil, autoFillPrompt: String? = nil, autoCaptioning: Bool? = false,
    cotrainCustomEmbedding: Bool? = false, customEmbeddingLearningRate: Float32? = 0.05,
    customEmbeddingLength: UInt32? = 4, stopEmbeddingTrainingAtStep: UInt32? = 500,
    trainableLayers: [LoRATrainableLayer]? = [], layerIndices: [UInt32]? = [],
    shift: Float32? = 1.0, resolutionDependentShift: Bool? = false,
    guidanceEmbedLowerBound: Float32? = 3.0, guidanceEmbedUpperBound: Float32? = 4.0,
    unetLearningRateLowerBound: Float32? = 0.0, stepsBetweenRestarts: UInt32? = 200,
    captionDropoutRate: Float32? = 0.0, orthonormalLoraDown: Bool? = false,
    maxTextLength: UInt32? = 512, useImageAspectRatio: Bool? = false,
    additionalScales: [UInt16]? = [], powerEmaLowerBound: Float32? = 0.0,
    powerEmaUpperBound: Float32? = 0.0
  ) {
    self.id = id
    self.name = name ?? nil
    self.startWidth = startWidth ?? 0
    self.startHeight = startHeight ?? 0
    self.seed = seed ?? 0
    self.trainingSteps = trainingSteps ?? 0
    self.baseModel = baseModel ?? nil
    self.networkDim = networkDim ?? 0
    self.networkScale = networkScale ?? 0.0
    self.unetLearningRate = unetLearningRate ?? 0.0
    self.saveEveryNSteps = saveEveryNSteps ?? 0
    self.warmupSteps = warmupSteps ?? 0
    self.gradientAccumulationSteps = gradientAccumulationSteps ?? 0
    self.cotrainTextModel = cotrainTextModel ?? false
    self.textModelLearningRate = textModelLearningRate ?? 0.0
    self.clipSkip = clipSkip ?? 1
    self.noiseOffset = noiseOffset ?? 0.0
    self.denoisingStart = denoisingStart ?? 0.0
    self.denoisingEnd = denoisingEnd ?? 0.0
    self.triggerWord = triggerWord ?? nil
    self.autoFillPrompt = autoFillPrompt ?? nil
    self.autoCaptioning = autoCaptioning ?? false
    self.cotrainCustomEmbedding = cotrainCustomEmbedding ?? false
    self.customEmbeddingLearningRate = customEmbeddingLearningRate ?? 0.05
    self.customEmbeddingLength = customEmbeddingLength ?? 4
    self.stopEmbeddingTrainingAtStep = stopEmbeddingTrainingAtStep ?? 500
    self.trainableLayers = trainableLayers ?? []
    self.layerIndices = layerIndices ?? []
    self.shift = shift ?? 1.0
    self.resolutionDependentShift = resolutionDependentShift ?? false
    self.guidanceEmbedLowerBound = guidanceEmbedLowerBound ?? 3.0
    self.guidanceEmbedUpperBound = guidanceEmbedUpperBound ?? 4.0
    self.unetLearningRateLowerBound = unetLearningRateLowerBound ?? 0.0
    self.stepsBetweenRestarts = stepsBetweenRestarts ?? 200
    self.captionDropoutRate = captionDropoutRate ?? 0.0
    self.orthonormalLoraDown = orthonormalLoraDown ?? false
    self.maxTextLength = maxTextLength ?? 512
    self.useImageAspectRatio = useImageAspectRatio ?? false
    self.additionalScales = additionalScales ?? []
    self.powerEmaLowerBound = powerEmaLowerBound ?? 0.0
    self.powerEmaUpperBound = powerEmaUpperBound ?? 0.0
  }
  public init(_ obj: zzz_DflatGen_LoRATrainingConfiguration) {
    self.id = obj.id
    self.name = obj.name
    self.startWidth = obj.startWidth
    self.startHeight = obj.startHeight
    self.seed = obj.seed
    self.trainingSteps = obj.trainingSteps
    self.baseModel = obj.baseModel
    self.networkDim = obj.networkDim
    self.networkScale = obj.networkScale
    self.unetLearningRate = obj.unetLearningRate
    self.saveEveryNSteps = obj.saveEveryNSteps
    self.warmupSteps = obj.warmupSteps
    self.gradientAccumulationSteps = obj.gradientAccumulationSteps
    self.cotrainTextModel = obj.cotrainTextModel
    self.textModelLearningRate = obj.textModelLearningRate
    self.clipSkip = obj.clipSkip
    self.noiseOffset = obj.noiseOffset
    self.denoisingStart = obj.denoisingStart
    self.denoisingEnd = obj.denoisingEnd
    self.triggerWord = obj.triggerWord
    self.autoFillPrompt = obj.autoFillPrompt
    self.autoCaptioning = obj.autoCaptioning
    self.cotrainCustomEmbedding = obj.cotrainCustomEmbedding
    self.customEmbeddingLearningRate = obj.customEmbeddingLearningRate
    self.customEmbeddingLength = obj.customEmbeddingLength
    self.stopEmbeddingTrainingAtStep = obj.stopEmbeddingTrainingAtStep
    var __trainableLayers = [LoRATrainableLayer]()
    for i: Int32 in 0..<obj.trainableLayersCount {
      guard let o = obj.trainableLayers(at: i) else { break }
      __trainableLayers.append(LoRATrainableLayer(rawValue: o.rawValue) ?? .latentsEmbedder)
    }
    self.trainableLayers = __trainableLayers
    self.layerIndices = obj.layerIndices
    self.shift = obj.shift
    self.resolutionDependentShift = obj.resolutionDependentShift
    self.guidanceEmbedLowerBound = obj.guidanceEmbedLowerBound
    self.guidanceEmbedUpperBound = obj.guidanceEmbedUpperBound
    self.unetLearningRateLowerBound = obj.unetLearningRateLowerBound
    self.stepsBetweenRestarts = obj.stepsBetweenRestarts
    self.captionDropoutRate = obj.captionDropoutRate
    self.orthonormalLoraDown = obj.orthonormalLoraDown
    self.maxTextLength = obj.maxTextLength
    self.useImageAspectRatio = obj.useImageAspectRatio
    self.additionalScales = obj.additionalScales
    self.powerEmaLowerBound = obj.powerEmaLowerBound
    self.powerEmaUpperBound = obj.powerEmaUpperBound
  }
  public static func from(data: Data) -> Self {
    return data.withUnsafeBytes { buffer in
      let bb = ByteBuffer(
        assumingMemoryBound: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
        capacity: buffer.count)
      return Self(zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: bb))
    }
  }
  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: bb))
  }
  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_LoRATrainingConfiguration>.verify(
        &verifier, at: 0, of: zzz_DflatGen_LoRATrainingConfiguration.self)
      return true
    } catch {
      return false
    }
  }
  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
  public static var table: String { "loratrainingconfiguration" }
  public static var indexFields: [String] { [] }
  public static func setUpSchema(_ toolbox: PersistenceToolbox) {
    guard let sqlite = ((toolbox as? SQLitePersistenceToolbox).map { $0.connection }) else {
      return
    }
    sqlite3_exec(
      sqlite.sqlite,
      "CREATE TABLE IF NOT EXISTS loratrainingconfiguration (rowid INTEGER PRIMARY KEY AUTOINCREMENT, __pk0 INTEGER, p BLOB, UNIQUE(__pk0))",
      nil, nil, nil)
  }
  public static func insertIndex(
    _ toolbox: PersistenceToolbox, field: String, rowid: Int64, table: ByteBuffer
  ) -> Bool {
    return true
  }
}

public struct LoRATrainingConfigurationBuilder {
  public var id: Int64
  public var name: String?
  public var startWidth: UInt16
  public var startHeight: UInt16
  public var seed: UInt32
  public var trainingSteps: UInt32
  public var baseModel: String?
  public var networkDim: UInt16
  public var networkScale: Float32
  public var unetLearningRate: Float32
  public var saveEveryNSteps: UInt32
  public var warmupSteps: UInt32
  public var gradientAccumulationSteps: UInt32
  public var cotrainTextModel: Bool
  public var textModelLearningRate: Float32
  public var clipSkip: UInt32
  public var noiseOffset: Float32
  public var denoisingStart: Float32
  public var denoisingEnd: Float32
  public var triggerWord: String?
  public var autoFillPrompt: String?
  public var autoCaptioning: Bool
  public var cotrainCustomEmbedding: Bool
  public var customEmbeddingLearningRate: Float32
  public var customEmbeddingLength: UInt32
  public var stopEmbeddingTrainingAtStep: UInt32
  public var trainableLayers: [LoRATrainableLayer]
  public var layerIndices: [UInt32]
  public var shift: Float32
  public var resolutionDependentShift: Bool
  public var guidanceEmbedLowerBound: Float32
  public var guidanceEmbedUpperBound: Float32
  public var unetLearningRateLowerBound: Float32
  public var stepsBetweenRestarts: UInt32
  public var captionDropoutRate: Float32
  public var orthonormalLoraDown: Bool
  public var maxTextLength: UInt32
  public var useImageAspectRatio: Bool
  public var additionalScales: [UInt16]
  public var powerEmaLowerBound: Float32
  public var powerEmaUpperBound: Float32
  public init(from object: LoRATrainingConfiguration) {
    id = object.id
    name = object.name
    startWidth = object.startWidth
    startHeight = object.startHeight
    seed = object.seed
    trainingSteps = object.trainingSteps
    baseModel = object.baseModel
    networkDim = object.networkDim
    networkScale = object.networkScale
    unetLearningRate = object.unetLearningRate
    saveEveryNSteps = object.saveEveryNSteps
    warmupSteps = object.warmupSteps
    gradientAccumulationSteps = object.gradientAccumulationSteps
    cotrainTextModel = object.cotrainTextModel
    textModelLearningRate = object.textModelLearningRate
    clipSkip = object.clipSkip
    noiseOffset = object.noiseOffset
    denoisingStart = object.denoisingStart
    denoisingEnd = object.denoisingEnd
    triggerWord = object.triggerWord
    autoFillPrompt = object.autoFillPrompt
    autoCaptioning = object.autoCaptioning
    cotrainCustomEmbedding = object.cotrainCustomEmbedding
    customEmbeddingLearningRate = object.customEmbeddingLearningRate
    customEmbeddingLength = object.customEmbeddingLength
    stopEmbeddingTrainingAtStep = object.stopEmbeddingTrainingAtStep
    trainableLayers = object.trainableLayers
    layerIndices = object.layerIndices
    shift = object.shift
    resolutionDependentShift = object.resolutionDependentShift
    guidanceEmbedLowerBound = object.guidanceEmbedLowerBound
    guidanceEmbedUpperBound = object.guidanceEmbedUpperBound
    unetLearningRateLowerBound = object.unetLearningRateLowerBound
    stepsBetweenRestarts = object.stepsBetweenRestarts
    captionDropoutRate = object.captionDropoutRate
    orthonormalLoraDown = object.orthonormalLoraDown
    maxTextLength = object.maxTextLength
    useImageAspectRatio = object.useImageAspectRatio
    additionalScales = object.additionalScales
    powerEmaLowerBound = object.powerEmaLowerBound
    powerEmaUpperBound = object.powerEmaUpperBound
  }
  public func build() -> LoRATrainingConfiguration {
    LoRATrainingConfiguration(
      id: id, name: name, startWidth: startWidth, startHeight: startHeight, seed: seed,
      trainingSteps: trainingSteps, baseModel: baseModel, networkDim: networkDim,
      networkScale: networkScale, unetLearningRate: unetLearningRate,
      saveEveryNSteps: saveEveryNSteps, warmupSteps: warmupSteps,
      gradientAccumulationSteps: gradientAccumulationSteps, cotrainTextModel: cotrainTextModel,
      textModelLearningRate: textModelLearningRate, clipSkip: clipSkip, noiseOffset: noiseOffset,
      denoisingStart: denoisingStart, denoisingEnd: denoisingEnd, triggerWord: triggerWord,
      autoFillPrompt: autoFillPrompt, autoCaptioning: autoCaptioning,
      cotrainCustomEmbedding: cotrainCustomEmbedding,
      customEmbeddingLearningRate: customEmbeddingLearningRate,
      customEmbeddingLength: customEmbeddingLength,
      stopEmbeddingTrainingAtStep: stopEmbeddingTrainingAtStep, trainableLayers: trainableLayers,
      layerIndices: layerIndices, shift: shift, resolutionDependentShift: resolutionDependentShift,
      guidanceEmbedLowerBound: guidanceEmbedLowerBound,
      guidanceEmbedUpperBound: guidanceEmbedUpperBound,
      unetLearningRateLowerBound: unetLearningRateLowerBound,
      stepsBetweenRestarts: stepsBetweenRestarts, captionDropoutRate: captionDropoutRate,
      orthonormalLoraDown: orthonormalLoraDown, maxTextLength: maxTextLength,
      useImageAspectRatio: useImageAspectRatio, additionalScales: additionalScales,
      powerEmaLowerBound: powerEmaLowerBound, powerEmaUpperBound: powerEmaUpperBound)
  }
}

#if compiler(>=5.5) && canImport(_Concurrency)
  extension LoRATrainingConfiguration: @unchecked Sendable {}
#endif
