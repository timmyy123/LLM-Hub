import Dflat
import FlatBuffers

extension LoRATrainingConfiguration {

  private static func _tr__f4(_ table: ByteBuffer) -> Int64? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.id
  }
  private static func _or__f4(_ or0: LoRATrainingConfiguration) -> Int64? {
    return or0.id
  }
  public static let id: FieldExpr<Int64, LoRATrainingConfiguration> = FieldExpr(
    name: "__pk0", primaryKey: true, hasIndex: false, tableReader: _tr__f4, objectReader: _or__f4)

  private static func _tr__f6(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    guard let s = tr0.name else { return nil }
    return s
  }
  private static func _or__f6(_ or0: LoRATrainingConfiguration) -> String? {
    guard let s = or0.name else { return nil }
    return s
  }
  public static let name: FieldExpr<String, LoRATrainingConfiguration> = FieldExpr(
    name: "f6", primaryKey: false, hasIndex: false, tableReader: _tr__f6, objectReader: _or__f6)

  private static func _tr__f8(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.startWidth
  }
  private static func _or__f8(_ or0: LoRATrainingConfiguration) -> UInt16? {
    return or0.startWidth
  }
  public static let startWidth: FieldExpr<UInt16, LoRATrainingConfiguration> = FieldExpr(
    name: "f8", primaryKey: false, hasIndex: false, tableReader: _tr__f8, objectReader: _or__f8)

  private static func _tr__f10(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.startHeight
  }
  private static func _or__f10(_ or0: LoRATrainingConfiguration) -> UInt16? {
    return or0.startHeight
  }
  public static let startHeight: FieldExpr<UInt16, LoRATrainingConfiguration> = FieldExpr(
    name: "f10", primaryKey: false, hasIndex: false, tableReader: _tr__f10, objectReader: _or__f10)

  private static func _tr__f12(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.seed
  }
  private static func _or__f12(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.seed
  }
  public static let seed: FieldExpr<UInt32, LoRATrainingConfiguration> = FieldExpr(
    name: "f12", primaryKey: false, hasIndex: false, tableReader: _tr__f12, objectReader: _or__f12)

  private static func _tr__f14(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.trainingSteps
  }
  private static func _or__f14(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.trainingSteps
  }
  public static let trainingSteps: FieldExpr<UInt32, LoRATrainingConfiguration> = FieldExpr(
    name: "f14", primaryKey: false, hasIndex: false, tableReader: _tr__f14, objectReader: _or__f14)

  private static func _tr__f16(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    guard let s = tr0.baseModel else { return nil }
    return s
  }
  private static func _or__f16(_ or0: LoRATrainingConfiguration) -> String? {
    guard let s = or0.baseModel else { return nil }
    return s
  }
  public static let baseModel: FieldExpr<String, LoRATrainingConfiguration> = FieldExpr(
    name: "f16", primaryKey: false, hasIndex: false, tableReader: _tr__f16, objectReader: _or__f16)

  private static func _tr__f18(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.networkDim
  }
  private static func _or__f18(_ or0: LoRATrainingConfiguration) -> UInt16? {
    return or0.networkDim
  }
  public static let networkDim: FieldExpr<UInt16, LoRATrainingConfiguration> = FieldExpr(
    name: "f18", primaryKey: false, hasIndex: false, tableReader: _tr__f18, objectReader: _or__f18)

  private static func _tr__f20(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.networkScale
  }
  private static func _or__f20(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.networkScale
  }
  public static let networkScale: FieldExpr<Float32, LoRATrainingConfiguration> = FieldExpr(
    name: "f20", primaryKey: false, hasIndex: false, tableReader: _tr__f20, objectReader: _or__f20)

  private static func _tr__f22(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.unetLearningRate
  }
  private static func _or__f22(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.unetLearningRate
  }
  public static let unetLearningRate: FieldExpr<Float32, LoRATrainingConfiguration> = FieldExpr(
    name: "f22", primaryKey: false, hasIndex: false, tableReader: _tr__f22, objectReader: _or__f22)

  private static func _tr__f24(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.saveEveryNSteps
  }
  private static func _or__f24(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.saveEveryNSteps
  }
  public static let saveEveryNSteps: FieldExpr<UInt32, LoRATrainingConfiguration> = FieldExpr(
    name: "f24", primaryKey: false, hasIndex: false, tableReader: _tr__f24, objectReader: _or__f24)

  private static func _tr__f26(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.warmupSteps
  }
  private static func _or__f26(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.warmupSteps
  }
  public static let warmupSteps: FieldExpr<UInt32, LoRATrainingConfiguration> = FieldExpr(
    name: "f26", primaryKey: false, hasIndex: false, tableReader: _tr__f26, objectReader: _or__f26)

  private static func _tr__f28(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.gradientAccumulationSteps
  }
  private static func _or__f28(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.gradientAccumulationSteps
  }
  public static let gradientAccumulationSteps: FieldExpr<UInt32, LoRATrainingConfiguration> =
    FieldExpr(
      name: "f28", primaryKey: false, hasIndex: false, tableReader: _tr__f28, objectReader: _or__f28
    )

  private static func _tr__f30(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.cotrainTextModel
  }
  private static func _or__f30(_ or0: LoRATrainingConfiguration) -> Bool? {
    return or0.cotrainTextModel
  }
  public static let cotrainTextModel: FieldExpr<Bool, LoRATrainingConfiguration> = FieldExpr(
    name: "f30", primaryKey: false, hasIndex: false, tableReader: _tr__f30, objectReader: _or__f30)

  private static func _tr__f32(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.textModelLearningRate
  }
  private static func _or__f32(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.textModelLearningRate
  }
  public static let textModelLearningRate: FieldExpr<Float32, LoRATrainingConfiguration> =
    FieldExpr(
      name: "f32", primaryKey: false, hasIndex: false, tableReader: _tr__f32, objectReader: _or__f32
    )

  private static func _tr__f34(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.clipSkip
  }
  private static func _or__f34(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.clipSkip
  }
  public static let clipSkip: FieldExpr<UInt32, LoRATrainingConfiguration> = FieldExpr(
    name: "f34", primaryKey: false, hasIndex: false, tableReader: _tr__f34, objectReader: _or__f34)

  private static func _tr__f36(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.noiseOffset
  }
  private static func _or__f36(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.noiseOffset
  }
  public static let noiseOffset: FieldExpr<Float32, LoRATrainingConfiguration> = FieldExpr(
    name: "f36", primaryKey: false, hasIndex: false, tableReader: _tr__f36, objectReader: _or__f36)

  private static func _tr__f38(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.denoisingStart
  }
  private static func _or__f38(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.denoisingStart
  }
  public static let denoisingStart: FieldExpr<Float32, LoRATrainingConfiguration> = FieldExpr(
    name: "f38", primaryKey: false, hasIndex: false, tableReader: _tr__f38, objectReader: _or__f38)

  private static func _tr__f40(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.denoisingEnd
  }
  private static func _or__f40(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.denoisingEnd
  }
  public static let denoisingEnd: FieldExpr<Float32, LoRATrainingConfiguration> = FieldExpr(
    name: "f40", primaryKey: false, hasIndex: false, tableReader: _tr__f40, objectReader: _or__f40)

  private static func _tr__f42(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    guard let s = tr0.triggerWord else { return nil }
    return s
  }
  private static func _or__f42(_ or0: LoRATrainingConfiguration) -> String? {
    guard let s = or0.triggerWord else { return nil }
    return s
  }
  public static let triggerWord: FieldExpr<String, LoRATrainingConfiguration> = FieldExpr(
    name: "f42", primaryKey: false, hasIndex: false, tableReader: _tr__f42, objectReader: _or__f42)

  private static func _tr__f44(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    guard let s = tr0.autoFillPrompt else { return nil }
    return s
  }
  private static func _or__f44(_ or0: LoRATrainingConfiguration) -> String? {
    guard let s = or0.autoFillPrompt else { return nil }
    return s
  }
  public static let autoFillPrompt: FieldExpr<String, LoRATrainingConfiguration> = FieldExpr(
    name: "f44", primaryKey: false, hasIndex: false, tableReader: _tr__f44, objectReader: _or__f44)

  private static func _tr__f46(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.autoCaptioning
  }
  private static func _or__f46(_ or0: LoRATrainingConfiguration) -> Bool? {
    return or0.autoCaptioning
  }
  public static let autoCaptioning: FieldExpr<Bool, LoRATrainingConfiguration> = FieldExpr(
    name: "f46", primaryKey: false, hasIndex: false, tableReader: _tr__f46, objectReader: _or__f46)

  private static func _tr__f48(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.cotrainCustomEmbedding
  }
  private static func _or__f48(_ or0: LoRATrainingConfiguration) -> Bool? {
    return or0.cotrainCustomEmbedding
  }
  public static let cotrainCustomEmbedding: FieldExpr<Bool, LoRATrainingConfiguration> = FieldExpr(
    name: "f48", primaryKey: false, hasIndex: false, tableReader: _tr__f48, objectReader: _or__f48)

  private static func _tr__f50(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.customEmbeddingLearningRate
  }
  private static func _or__f50(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.customEmbeddingLearningRate
  }
  public static let customEmbeddingLearningRate: FieldExpr<Float32, LoRATrainingConfiguration> =
    FieldExpr(
      name: "f50", primaryKey: false, hasIndex: false, tableReader: _tr__f50, objectReader: _or__f50
    )

  private static func _tr__f52(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.customEmbeddingLength
  }
  private static func _or__f52(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.customEmbeddingLength
  }
  public static let customEmbeddingLength: FieldExpr<UInt32, LoRATrainingConfiguration> = FieldExpr(
    name: "f52", primaryKey: false, hasIndex: false, tableReader: _tr__f52, objectReader: _or__f52)

  private static func _tr__f54(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.stopEmbeddingTrainingAtStep
  }
  private static func _or__f54(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.stopEmbeddingTrainingAtStep
  }
  public static let stopEmbeddingTrainingAtStep: FieldExpr<UInt32, LoRATrainingConfiguration> =
    FieldExpr(
      name: "f54", primaryKey: false, hasIndex: false, tableReader: _tr__f54, objectReader: _or__f54
    )

  private static func _tr__f60(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.shift
  }
  private static func _or__f60(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.shift
  }
  public static let shift: FieldExpr<Float32, LoRATrainingConfiguration> = FieldExpr(
    name: "f60", primaryKey: false, hasIndex: false, tableReader: _tr__f60, objectReader: _or__f60)

  private static func _tr__f62(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.resolutionDependentShift
  }
  private static func _or__f62(_ or0: LoRATrainingConfiguration) -> Bool? {
    return or0.resolutionDependentShift
  }
  public static let resolutionDependentShift: FieldExpr<Bool, LoRATrainingConfiguration> =
    FieldExpr(
      name: "f62", primaryKey: false, hasIndex: false, tableReader: _tr__f62, objectReader: _or__f62
    )

  private static func _tr__f64(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.guidanceEmbedLowerBound
  }
  private static func _or__f64(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.guidanceEmbedLowerBound
  }
  public static let guidanceEmbedLowerBound: FieldExpr<Float32, LoRATrainingConfiguration> =
    FieldExpr(
      name: "f64", primaryKey: false, hasIndex: false, tableReader: _tr__f64, objectReader: _or__f64
    )

  private static func _tr__f66(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.guidanceEmbedUpperBound
  }
  private static func _or__f66(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.guidanceEmbedUpperBound
  }
  public static let guidanceEmbedUpperBound: FieldExpr<Float32, LoRATrainingConfiguration> =
    FieldExpr(
      name: "f66", primaryKey: false, hasIndex: false, tableReader: _tr__f66, objectReader: _or__f66
    )

  private static func _tr__f68(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.unetLearningRateLowerBound
  }
  private static func _or__f68(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.unetLearningRateLowerBound
  }
  public static let unetLearningRateLowerBound: FieldExpr<Float32, LoRATrainingConfiguration> =
    FieldExpr(
      name: "f68", primaryKey: false, hasIndex: false, tableReader: _tr__f68, objectReader: _or__f68
    )

  private static func _tr__f70(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.stepsBetweenRestarts
  }
  private static func _or__f70(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.stepsBetweenRestarts
  }
  public static let stepsBetweenRestarts: FieldExpr<UInt32, LoRATrainingConfiguration> = FieldExpr(
    name: "f70", primaryKey: false, hasIndex: false, tableReader: _tr__f70, objectReader: _or__f70)

  private static func _tr__f72(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.captionDropoutRate
  }
  private static func _or__f72(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.captionDropoutRate
  }
  public static let captionDropoutRate: FieldExpr<Float32, LoRATrainingConfiguration> = FieldExpr(
    name: "f72", primaryKey: false, hasIndex: false, tableReader: _tr__f72, objectReader: _or__f72)

  private static func _tr__f74(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.orthonormalLoraDown
  }
  private static func _or__f74(_ or0: LoRATrainingConfiguration) -> Bool? {
    return or0.orthonormalLoraDown
  }
  public static let orthonormalLoraDown: FieldExpr<Bool, LoRATrainingConfiguration> = FieldExpr(
    name: "f74", primaryKey: false, hasIndex: false, tableReader: _tr__f74, objectReader: _or__f74)

  private static func _tr__f76(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.maxTextLength
  }
  private static func _or__f76(_ or0: LoRATrainingConfiguration) -> UInt32? {
    return or0.maxTextLength
  }
  public static let maxTextLength: FieldExpr<UInt32, LoRATrainingConfiguration> = FieldExpr(
    name: "f76", primaryKey: false, hasIndex: false, tableReader: _tr__f76, objectReader: _or__f76)

  private static func _tr__f78(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.useImageAspectRatio
  }
  private static func _or__f78(_ or0: LoRATrainingConfiguration) -> Bool? {
    return or0.useImageAspectRatio
  }
  public static let useImageAspectRatio: FieldExpr<Bool, LoRATrainingConfiguration> = FieldExpr(
    name: "f78", primaryKey: false, hasIndex: false, tableReader: _tr__f78, objectReader: _or__f78)

  private static func _tr__f82(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.powerEmaLowerBound
  }
  private static func _or__f82(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.powerEmaLowerBound
  }
  public static let powerEmaLowerBound: FieldExpr<Float32, LoRATrainingConfiguration> = FieldExpr(
    name: "f82", primaryKey: false, hasIndex: false, tableReader: _tr__f82, objectReader: _or__f82)

  private static func _tr__f84(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_LoRATrainingConfiguration.getRootAsLoRATrainingConfiguration(bb: table)
    return tr0.powerEmaUpperBound
  }
  private static func _or__f84(_ or0: LoRATrainingConfiguration) -> Float32? {
    return or0.powerEmaUpperBound
  }
  public static let powerEmaUpperBound: FieldExpr<Float32, LoRATrainingConfiguration> = FieldExpr(
    name: "f84", primaryKey: false, hasIndex: false, tableReader: _tr__f84, objectReader: _or__f84)
}
