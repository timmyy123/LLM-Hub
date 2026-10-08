import Dflat
import FlatBuffers

extension GenerationConfiguration {

  private static func _tr__f4(_ table: ByteBuffer) -> Int64? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.id
  }
  private static func _or__f4(_ or0: GenerationConfiguration) -> Int64? {
    return or0.id
  }
  public static let id: FieldExpr<Int64, GenerationConfiguration> = FieldExpr(
    name: "__pk0", primaryKey: true, hasIndex: false, tableReader: _tr__f4, objectReader: _or__f4)

  private static func _tr__f6(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.startWidth
  }
  private static func _or__f6(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.startWidth
  }
  public static let startWidth: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f6", primaryKey: false, hasIndex: false, tableReader: _tr__f6, objectReader: _or__f6)

  private static func _tr__f8(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.startHeight
  }
  private static func _or__f8(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.startHeight
  }
  public static let startHeight: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f8", primaryKey: false, hasIndex: false, tableReader: _tr__f8, objectReader: _or__f8)

  private static func _tr__f10(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.seed
  }
  private static func _or__f10(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.seed
  }
  public static let seed: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f10", primaryKey: false, hasIndex: false, tableReader: _tr__f10, objectReader: _or__f10)

  private static func _tr__f12(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.steps
  }
  private static func _or__f12(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.steps
  }
  public static let steps: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f12", primaryKey: false, hasIndex: false, tableReader: _tr__f12, objectReader: _or__f12)

  private static func _tr__f14(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.guidanceScale
  }
  private static func _or__f14(_ or0: GenerationConfiguration) -> Float32? {
    return or0.guidanceScale
  }
  public static let guidanceScale: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f14", primaryKey: false, hasIndex: false, tableReader: _tr__f14, objectReader: _or__f14)

  private static func _tr__f16(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.strength
  }
  private static func _or__f16(_ or0: GenerationConfiguration) -> Float32? {
    return or0.strength
  }
  public static let strength: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f16", primaryKey: false, hasIndex: false, tableReader: _tr__f16, objectReader: _or__f16)

  private static func _tr__f18(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    guard let s = tr0.model else { return nil }
    return s
  }
  private static func _or__f18(_ or0: GenerationConfiguration) -> String? {
    guard let s = or0.model else { return nil }
    return s
  }
  public static let model: FieldExpr<String, GenerationConfiguration> = FieldExpr(
    name: "f18", primaryKey: false, hasIndex: false, tableReader: _tr__f18, objectReader: _or__f18)

  private static func _tr__f20(_ table: ByteBuffer) -> SamplerType? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return SamplerType(rawValue: tr0.sampler.rawValue)!
  }
  private static func _or__f20(_ or0: GenerationConfiguration) -> SamplerType? {
    return or0.sampler
  }
  public static let sampler: FieldExpr<SamplerType, GenerationConfiguration> = FieldExpr(
    name: "f20", primaryKey: false, hasIndex: false, tableReader: _tr__f20, objectReader: _or__f20)

  private static func _tr__f22(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.batchCount
  }
  private static func _or__f22(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.batchCount
  }
  public static let batchCount: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f22", primaryKey: false, hasIndex: false, tableReader: _tr__f22, objectReader: _or__f22)

  private static func _tr__f24(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.batchSize
  }
  private static func _or__f24(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.batchSize
  }
  public static let batchSize: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f24", primaryKey: false, hasIndex: false, tableReader: _tr__f24, objectReader: _or__f24)

  private static func _tr__f26(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.hiresFix
  }
  private static func _or__f26(_ or0: GenerationConfiguration) -> Bool? {
    return or0.hiresFix
  }
  public static let hiresFix: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f26", primaryKey: false, hasIndex: false, tableReader: _tr__f26, objectReader: _or__f26)

  private static func _tr__f28(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.hiresFixStartWidth
  }
  private static func _or__f28(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.hiresFixStartWidth
  }
  public static let hiresFixStartWidth: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f28", primaryKey: false, hasIndex: false, tableReader: _tr__f28, objectReader: _or__f28)

  private static func _tr__f30(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.hiresFixStartHeight
  }
  private static func _or__f30(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.hiresFixStartHeight
  }
  public static let hiresFixStartHeight: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f30", primaryKey: false, hasIndex: false, tableReader: _tr__f30, objectReader: _or__f30)

  private static func _tr__f32(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.hiresFixStrength
  }
  private static func _or__f32(_ or0: GenerationConfiguration) -> Float32? {
    return or0.hiresFixStrength
  }
  public static let hiresFixStrength: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f32", primaryKey: false, hasIndex: false, tableReader: _tr__f32, objectReader: _or__f32)

  private static func _tr__f34(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    guard let s = tr0.upscaler else { return nil }
    return s
  }
  private static func _or__f34(_ or0: GenerationConfiguration) -> String? {
    guard let s = or0.upscaler else { return nil }
    return s
  }
  public static let upscaler: FieldExpr<String, GenerationConfiguration> = FieldExpr(
    name: "f34", primaryKey: false, hasIndex: false, tableReader: _tr__f34, objectReader: _or__f34)

  private static func _tr__f36(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.imageGuidanceScale
  }
  private static func _or__f36(_ or0: GenerationConfiguration) -> Float32? {
    return or0.imageGuidanceScale
  }
  public static let imageGuidanceScale: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f36", primaryKey: false, hasIndex: false, tableReader: _tr__f36, objectReader: _or__f36)

  private static func _tr__f38(_ table: ByteBuffer) -> SeedMode? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return SeedMode(rawValue: tr0.seedMode.rawValue)!
  }
  private static func _or__f38(_ or0: GenerationConfiguration) -> SeedMode? {
    return or0.seedMode
  }
  public static let seedMode: FieldExpr<SeedMode, GenerationConfiguration> = FieldExpr(
    name: "f38", primaryKey: false, hasIndex: false, tableReader: _tr__f38, objectReader: _or__f38)

  private static func _tr__f40(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.clipSkip
  }
  private static func _or__f40(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.clipSkip
  }
  public static let clipSkip: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f40", primaryKey: false, hasIndex: false, tableReader: _tr__f40, objectReader: _or__f40)

  private static func _tr__f46(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.maskBlur
  }
  private static func _or__f46(_ or0: GenerationConfiguration) -> Float32? {
    return or0.maskBlur
  }
  public static let maskBlur: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f46", primaryKey: false, hasIndex: false, tableReader: _tr__f46, objectReader: _or__f46)

  private static func _tr__f48(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    guard let s = tr0.faceRestoration else { return nil }
    return s
  }
  private static func _or__f48(_ or0: GenerationConfiguration) -> String? {
    guard let s = or0.faceRestoration else { return nil }
    return s
  }
  public static let faceRestoration: FieldExpr<String, GenerationConfiguration> = FieldExpr(
    name: "f48", primaryKey: false, hasIndex: false, tableReader: _tr__f48, objectReader: _or__f48)

  private static func _tr__f54(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.clipWeight
  }
  private static func _or__f54(_ or0: GenerationConfiguration) -> Float32? {
    return or0.clipWeight
  }
  public static let clipWeight: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f54", primaryKey: false, hasIndex: false, tableReader: _tr__f54, objectReader: _or__f54)

  private static func _tr__f56(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.negativePromptForImagePrior
  }
  private static func _or__f56(_ or0: GenerationConfiguration) -> Bool? {
    return or0.negativePromptForImagePrior
  }
  public static let negativePromptForImagePrior: FieldExpr<Bool, GenerationConfiguration> =
    FieldExpr(
      name: "f56", primaryKey: false, hasIndex: false, tableReader: _tr__f56, objectReader: _or__f56
    )

  private static func _tr__f58(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.imagePriorSteps
  }
  private static func _or__f58(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.imagePriorSteps
  }
  public static let imagePriorSteps: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f58", primaryKey: false, hasIndex: false, tableReader: _tr__f58, objectReader: _or__f58)

  private static func _tr__f60(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    guard let s = tr0.refinerModel else { return nil }
    return s
  }
  private static func _or__f60(_ or0: GenerationConfiguration) -> String? {
    guard let s = or0.refinerModel else { return nil }
    return s
  }
  public static let refinerModel: FieldExpr<String, GenerationConfiguration> = FieldExpr(
    name: "f60", primaryKey: false, hasIndex: false, tableReader: _tr__f60, objectReader: _or__f60)

  private static func _tr__f62(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.originalImageHeight
  }
  private static func _or__f62(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.originalImageHeight
  }
  public static let originalImageHeight: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f62", primaryKey: false, hasIndex: false, tableReader: _tr__f62, objectReader: _or__f62)

  private static func _tr__f64(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.originalImageWidth
  }
  private static func _or__f64(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.originalImageWidth
  }
  public static let originalImageWidth: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f64", primaryKey: false, hasIndex: false, tableReader: _tr__f64, objectReader: _or__f64)

  private static func _tr__f66(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.cropTop
  }
  private static func _or__f66(_ or0: GenerationConfiguration) -> Int32? {
    return or0.cropTop
  }
  public static let cropTop: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f66", primaryKey: false, hasIndex: false, tableReader: _tr__f66, objectReader: _or__f66)

  private static func _tr__f68(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.cropLeft
  }
  private static func _or__f68(_ or0: GenerationConfiguration) -> Int32? {
    return or0.cropLeft
  }
  public static let cropLeft: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f68", primaryKey: false, hasIndex: false, tableReader: _tr__f68, objectReader: _or__f68)

  private static func _tr__f70(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.targetImageHeight
  }
  private static func _or__f70(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.targetImageHeight
  }
  public static let targetImageHeight: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f70", primaryKey: false, hasIndex: false, tableReader: _tr__f70, objectReader: _or__f70)

  private static func _tr__f72(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.targetImageWidth
  }
  private static func _or__f72(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.targetImageWidth
  }
  public static let targetImageWidth: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f72", primaryKey: false, hasIndex: false, tableReader: _tr__f72, objectReader: _or__f72)

  private static func _tr__f74(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.aestheticScore
  }
  private static func _or__f74(_ or0: GenerationConfiguration) -> Float32? {
    return or0.aestheticScore
  }
  public static let aestheticScore: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f74", primaryKey: false, hasIndex: false, tableReader: _tr__f74, objectReader: _or__f74)

  private static func _tr__f76(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.negativeAestheticScore
  }
  private static func _or__f76(_ or0: GenerationConfiguration) -> Float32? {
    return or0.negativeAestheticScore
  }
  public static let negativeAestheticScore: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f76", primaryKey: false, hasIndex: false, tableReader: _tr__f76, objectReader: _or__f76)

  private static func _tr__f78(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.zeroNegativePrompt
  }
  private static func _or__f78(_ or0: GenerationConfiguration) -> Bool? {
    return or0.zeroNegativePrompt
  }
  public static let zeroNegativePrompt: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f78", primaryKey: false, hasIndex: false, tableReader: _tr__f78, objectReader: _or__f78)

  private static func _tr__f80(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.refinerStart
  }
  private static func _or__f80(_ or0: GenerationConfiguration) -> Float32? {
    return or0.refinerStart
  }
  public static let refinerStart: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f80", primaryKey: false, hasIndex: false, tableReader: _tr__f80, objectReader: _or__f80)

  private static func _tr__f82(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.negativeOriginalImageHeight
  }
  private static func _or__f82(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.negativeOriginalImageHeight
  }
  public static let negativeOriginalImageHeight: FieldExpr<UInt32, GenerationConfiguration> =
    FieldExpr(
      name: "f82", primaryKey: false, hasIndex: false, tableReader: _tr__f82, objectReader: _or__f82
    )

  private static func _tr__f84(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.negativeOriginalImageWidth
  }
  private static func _or__f84(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.negativeOriginalImageWidth
  }
  public static let negativeOriginalImageWidth: FieldExpr<UInt32, GenerationConfiguration> =
    FieldExpr(
      name: "f84", primaryKey: false, hasIndex: false, tableReader: _tr__f84, objectReader: _or__f84
    )

  private static func _tr__f86(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    guard let s = tr0.name else { return nil }
    return s
  }
  private static func _or__f86(_ or0: GenerationConfiguration) -> String? {
    guard let s = or0.name else { return nil }
    return s
  }
  public static let name: FieldExpr<String, GenerationConfiguration> = FieldExpr(
    name: "f86", primaryKey: false, hasIndex: true, tableReader: _tr__f86, objectReader: _or__f86)

  private static func _tr__f88(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.fpsId
  }
  private static func _or__f88(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.fpsId
  }
  public static let fpsId: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f88", primaryKey: false, hasIndex: false, tableReader: _tr__f88, objectReader: _or__f88)

  private static func _tr__f90(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.motionBucketId
  }
  private static func _or__f90(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.motionBucketId
  }
  public static let motionBucketId: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f90", primaryKey: false, hasIndex: false, tableReader: _tr__f90, objectReader: _or__f90)

  private static func _tr__f92(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.condAug
  }
  private static func _or__f92(_ or0: GenerationConfiguration) -> Float32? {
    return or0.condAug
  }
  public static let condAug: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f92", primaryKey: false, hasIndex: false, tableReader: _tr__f92, objectReader: _or__f92)

  private static func _tr__f94(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.startFrameCfg
  }
  private static func _or__f94(_ or0: GenerationConfiguration) -> Float32? {
    return or0.startFrameCfg
  }
  public static let startFrameCfg: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f94", primaryKey: false, hasIndex: false, tableReader: _tr__f94, objectReader: _or__f94)

  private static func _tr__f96(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.numFrames
  }
  private static func _or__f96(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.numFrames
  }
  public static let numFrames: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f96", primaryKey: false, hasIndex: false, tableReader: _tr__f96, objectReader: _or__f96)

  private static func _tr__f98(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.maskBlurOutset
  }
  private static func _or__f98(_ or0: GenerationConfiguration) -> Int32? {
    return or0.maskBlurOutset
  }
  public static let maskBlurOutset: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f98", primaryKey: false, hasIndex: false, tableReader: _tr__f98, objectReader: _or__f98)

  private static func _tr__f100(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.sharpness
  }
  private static func _or__f100(_ or0: GenerationConfiguration) -> Float32? {
    return or0.sharpness
  }
  public static let sharpness: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f100", primaryKey: false, hasIndex: false, tableReader: _tr__f100,
    objectReader: _or__f100)

  private static func _tr__f102(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.shift
  }
  private static func _or__f102(_ or0: GenerationConfiguration) -> Float32? {
    return or0.shift
  }
  public static let shift: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f102", primaryKey: false, hasIndex: false, tableReader: _tr__f102,
    objectReader: _or__f102)

  private static func _tr__f104(_ table: ByteBuffer) -> UInt32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.stage2Steps
  }
  private static func _or__f104(_ or0: GenerationConfiguration) -> UInt32? {
    return or0.stage2Steps
  }
  public static let stage2Steps: FieldExpr<UInt32, GenerationConfiguration> = FieldExpr(
    name: "f104", primaryKey: false, hasIndex: false, tableReader: _tr__f104,
    objectReader: _or__f104)

  private static func _tr__f106(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.stage2Cfg
  }
  private static func _or__f106(_ or0: GenerationConfiguration) -> Float32? {
    return or0.stage2Cfg
  }
  public static let stage2Cfg: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f106", primaryKey: false, hasIndex: false, tableReader: _tr__f106,
    objectReader: _or__f106)

  private static func _tr__f108(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.stage2Shift
  }
  private static func _or__f108(_ or0: GenerationConfiguration) -> Float32? {
    return or0.stage2Shift
  }
  public static let stage2Shift: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f108", primaryKey: false, hasIndex: false, tableReader: _tr__f108,
    objectReader: _or__f108)

  private static func _tr__f110(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.tiledDecoding
  }
  private static func _or__f110(_ or0: GenerationConfiguration) -> Bool? {
    return or0.tiledDecoding
  }
  public static let tiledDecoding: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f110", primaryKey: false, hasIndex: false, tableReader: _tr__f110,
    objectReader: _or__f110)

  private static func _tr__f112(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.decodingTileWidth
  }
  private static func _or__f112(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.decodingTileWidth
  }
  public static let decodingTileWidth: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f112", primaryKey: false, hasIndex: false, tableReader: _tr__f112,
    objectReader: _or__f112)

  private static func _tr__f114(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.decodingTileHeight
  }
  private static func _or__f114(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.decodingTileHeight
  }
  public static let decodingTileHeight: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f114", primaryKey: false, hasIndex: false, tableReader: _tr__f114,
    objectReader: _or__f114)

  private static func _tr__f116(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.decodingTileOverlap
  }
  private static func _or__f116(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.decodingTileOverlap
  }
  public static let decodingTileOverlap: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f116", primaryKey: false, hasIndex: false, tableReader: _tr__f116,
    objectReader: _or__f116)

  private static func _tr__f118(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.stochasticSamplingGamma
  }
  private static func _or__f118(_ or0: GenerationConfiguration) -> Float32? {
    return or0.stochasticSamplingGamma
  }
  public static let stochasticSamplingGamma: FieldExpr<Float32, GenerationConfiguration> =
    FieldExpr(
      name: "f118", primaryKey: false, hasIndex: false, tableReader: _tr__f118,
      objectReader: _or__f118)

  private static func _tr__f120(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.preserveOriginalAfterInpaint
  }
  private static func _or__f120(_ or0: GenerationConfiguration) -> Bool? {
    return or0.preserveOriginalAfterInpaint
  }
  public static let preserveOriginalAfterInpaint: FieldExpr<Bool, GenerationConfiguration> =
    FieldExpr(
      name: "f120", primaryKey: false, hasIndex: false, tableReader: _tr__f120,
      objectReader: _or__f120)

  private static func _tr__f122(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.tiledDiffusion
  }
  private static func _or__f122(_ or0: GenerationConfiguration) -> Bool? {
    return or0.tiledDiffusion
  }
  public static let tiledDiffusion: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f122", primaryKey: false, hasIndex: false, tableReader: _tr__f122,
    objectReader: _or__f122)

  private static func _tr__f124(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.diffusionTileWidth
  }
  private static func _or__f124(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.diffusionTileWidth
  }
  public static let diffusionTileWidth: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f124", primaryKey: false, hasIndex: false, tableReader: _tr__f124,
    objectReader: _or__f124)

  private static func _tr__f126(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.diffusionTileHeight
  }
  private static func _or__f126(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.diffusionTileHeight
  }
  public static let diffusionTileHeight: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f126", primaryKey: false, hasIndex: false, tableReader: _tr__f126,
    objectReader: _or__f126)

  private static func _tr__f128(_ table: ByteBuffer) -> UInt16? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.diffusionTileOverlap
  }
  private static func _or__f128(_ or0: GenerationConfiguration) -> UInt16? {
    return or0.diffusionTileOverlap
  }
  public static let diffusionTileOverlap: FieldExpr<UInt16, GenerationConfiguration> = FieldExpr(
    name: "f128", primaryKey: false, hasIndex: false, tableReader: _tr__f128,
    objectReader: _or__f128)

  private static func _tr__f130(_ table: ByteBuffer) -> UInt8? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.upscalerScaleFactor
  }
  private static func _or__f130(_ or0: GenerationConfiguration) -> UInt8? {
    return or0.upscalerScaleFactor
  }
  public static let upscalerScaleFactor: FieldExpr<UInt8, GenerationConfiguration> = FieldExpr(
    name: "f130", primaryKey: false, hasIndex: false, tableReader: _tr__f130,
    objectReader: _or__f130)

  private static func _tr__f132(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.t5TextEncoder
  }
  private static func _or__f132(_ or0: GenerationConfiguration) -> Bool? {
    return or0.t5TextEncoder
  }
  public static let t5TextEncoder: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f132", primaryKey: false, hasIndex: false, tableReader: _tr__f132,
    objectReader: _or__f132)

  private static func _tr__f134(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.separateClipL
  }
  private static func _or__f134(_ or0: GenerationConfiguration) -> Bool? {
    return or0.separateClipL
  }
  public static let separateClipL: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f134", primaryKey: false, hasIndex: false, tableReader: _tr__f134,
    objectReader: _or__f134)

  private static func _tr__f136(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    guard let s = tr0.clipLText else { return nil }
    return s
  }
  private static func _or__f136(_ or0: GenerationConfiguration) -> String? {
    guard let s = or0.clipLText else { return nil }
    return s
  }
  public static let clipLText: FieldExpr<String, GenerationConfiguration> = FieldExpr(
    name: "f136", primaryKey: false, hasIndex: false, tableReader: _tr__f136,
    objectReader: _or__f136)

  private static func _tr__f138(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.separateOpenClipG
  }
  private static func _or__f138(_ or0: GenerationConfiguration) -> Bool? {
    return or0.separateOpenClipG
  }
  public static let separateOpenClipG: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f138", primaryKey: false, hasIndex: false, tableReader: _tr__f138,
    objectReader: _or__f138)

  private static func _tr__f140(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    guard let s = tr0.openClipGText else { return nil }
    return s
  }
  private static func _or__f140(_ or0: GenerationConfiguration) -> String? {
    guard let s = or0.openClipGText else { return nil }
    return s
  }
  public static let openClipGText: FieldExpr<String, GenerationConfiguration> = FieldExpr(
    name: "f140", primaryKey: false, hasIndex: false, tableReader: _tr__f140,
    objectReader: _or__f140)

  private static func _tr__f142(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.speedUpWithGuidanceEmbed
  }
  private static func _or__f142(_ or0: GenerationConfiguration) -> Bool? {
    return or0.speedUpWithGuidanceEmbed
  }
  public static let speedUpWithGuidanceEmbed: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f142", primaryKey: false, hasIndex: false, tableReader: _tr__f142,
    objectReader: _or__f142)

  private static func _tr__f144(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.guidanceEmbed
  }
  private static func _or__f144(_ or0: GenerationConfiguration) -> Float32? {
    return or0.guidanceEmbed
  }
  public static let guidanceEmbed: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f144", primaryKey: false, hasIndex: false, tableReader: _tr__f144,
    objectReader: _or__f144)

  private static func _tr__f146(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.resolutionDependentShift
  }
  private static func _or__f146(_ or0: GenerationConfiguration) -> Bool? {
    return or0.resolutionDependentShift
  }
  public static let resolutionDependentShift: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f146", primaryKey: false, hasIndex: false, tableReader: _tr__f146,
    objectReader: _or__f146)

  private static func _tr__f148(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.teaCacheStart
  }
  private static func _or__f148(_ or0: GenerationConfiguration) -> Int32? {
    return or0.teaCacheStart
  }
  public static let teaCacheStart: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f148", primaryKey: false, hasIndex: false, tableReader: _tr__f148,
    objectReader: _or__f148)

  private static func _tr__f150(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.teaCacheEnd
  }
  private static func _or__f150(_ or0: GenerationConfiguration) -> Int32? {
    return or0.teaCacheEnd
  }
  public static let teaCacheEnd: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f150", primaryKey: false, hasIndex: false, tableReader: _tr__f150,
    objectReader: _or__f150)

  private static func _tr__f152(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.teaCacheThreshold
  }
  private static func _or__f152(_ or0: GenerationConfiguration) -> Float32? {
    return or0.teaCacheThreshold
  }
  public static let teaCacheThreshold: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f152", primaryKey: false, hasIndex: false, tableReader: _tr__f152,
    objectReader: _or__f152)

  private static func _tr__f154(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.teaCache
  }
  private static func _or__f154(_ or0: GenerationConfiguration) -> Bool? {
    return or0.teaCache
  }
  public static let teaCache: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f154", primaryKey: false, hasIndex: false, tableReader: _tr__f154,
    objectReader: _or__f154)

  private static func _tr__f156(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.separateT5
  }
  private static func _or__f156(_ or0: GenerationConfiguration) -> Bool? {
    return or0.separateT5
  }
  public static let separateT5: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f156", primaryKey: false, hasIndex: false, tableReader: _tr__f156,
    objectReader: _or__f156)

  private static func _tr__f158(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    guard let s = tr0.t5Text else { return nil }
    return s
  }
  private static func _or__f158(_ or0: GenerationConfiguration) -> String? {
    guard let s = or0.t5Text else { return nil }
    return s
  }
  public static let t5Text: FieldExpr<String, GenerationConfiguration> = FieldExpr(
    name: "f158", primaryKey: false, hasIndex: false, tableReader: _tr__f158,
    objectReader: _or__f158)

  private static func _tr__f160(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.teaCacheMaxSkipSteps
  }
  private static func _or__f160(_ or0: GenerationConfiguration) -> Int32? {
    return or0.teaCacheMaxSkipSteps
  }
  public static let teaCacheMaxSkipSteps: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f160", primaryKey: false, hasIndex: false, tableReader: _tr__f160,
    objectReader: _or__f160)

  private static func _tr__f162(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.causalInferenceEnabled
  }
  private static func _or__f162(_ or0: GenerationConfiguration) -> Bool? {
    return or0.causalInferenceEnabled
  }
  public static let causalInferenceEnabled: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f162", primaryKey: false, hasIndex: false, tableReader: _tr__f162,
    objectReader: _or__f162)

  private static func _tr__f164(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.causalInference
  }
  private static func _or__f164(_ or0: GenerationConfiguration) -> Int32? {
    return or0.causalInference
  }
  public static let causalInference: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f164", primaryKey: false, hasIndex: false, tableReader: _tr__f164,
    objectReader: _or__f164)

  private static func _tr__f166(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.causalInferencePad
  }
  private static func _or__f166(_ or0: GenerationConfiguration) -> Int32? {
    return or0.causalInferencePad
  }
  public static let causalInferencePad: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f166", primaryKey: false, hasIndex: false, tableReader: _tr__f166,
    objectReader: _or__f166)

  private static func _tr__f168(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.cfgZeroStar
  }
  private static func _or__f168(_ or0: GenerationConfiguration) -> Bool? {
    return or0.cfgZeroStar
  }
  public static let cfgZeroStar: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f168", primaryKey: false, hasIndex: false, tableReader: _tr__f168,
    objectReader: _or__f168)

  private static func _tr__f170(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.cfgZeroInitSteps
  }
  private static func _or__f170(_ or0: GenerationConfiguration) -> Int32? {
    return or0.cfgZeroInitSteps
  }
  public static let cfgZeroInitSteps: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f170", primaryKey: false, hasIndex: false, tableReader: _tr__f170,
    objectReader: _or__f170)

  private static func _tr__f172(_ table: ByteBuffer) -> CompressionMethod? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return CompressionMethod(rawValue: tr0.compressionArtifacts.rawValue)!
  }
  private static func _or__f172(_ or0: GenerationConfiguration) -> CompressionMethod? {
    return or0.compressionArtifacts
  }
  public static let compressionArtifacts: FieldExpr<CompressionMethod, GenerationConfiguration> =
    FieldExpr(
      name: "f172", primaryKey: false, hasIndex: false, tableReader: _tr__f172,
      objectReader: _or__f172)

  private static func _tr__f174(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.compressionArtifactsQuality
  }
  private static func _or__f174(_ or0: GenerationConfiguration) -> Float32? {
    return or0.compressionArtifactsQuality
  }
  public static let compressionArtifactsQuality: FieldExpr<Float32, GenerationConfiguration> =
    FieldExpr(
      name: "f174", primaryKey: false, hasIndex: false, tableReader: _tr__f174,
      objectReader: _or__f174)

  private static func _tr__f176(_ table: ByteBuffer) -> ColorCalibration? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return ColorCalibration(rawValue: tr0.colorCalibration.rawValue)!
  }
  private static func _or__f176(_ or0: GenerationConfiguration) -> ColorCalibration? {
    return or0.colorCalibration
  }
  public static let colorCalibration: FieldExpr<ColorCalibration, GenerationConfiguration> =
    FieldExpr(
      name: "f176", primaryKey: false, hasIndex: false, tableReader: _tr__f176,
      objectReader: _or__f176)

  private static func _tr__f178(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.expandPromptToJson
  }
  private static func _or__f178(_ or0: GenerationConfiguration) -> Bool? {
    return or0.expandPromptToJson
  }
  public static let expandPromptToJson: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f178", primaryKey: false, hasIndex: false, tableReader: _tr__f178,
    objectReader: _or__f178)

  private static func _tr__f180(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.shiftForAudio
  }
  private static func _or__f180(_ or0: GenerationConfiguration) -> Float32? {
    return or0.shiftForAudio
  }
  public static let shiftForAudio: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f180", primaryKey: false, hasIndex: false, tableReader: _tr__f180,
    objectReader: _or__f180)

  private static func _tr__f182(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.usesSolAttention
  }
  private static func _or__f182(_ or0: GenerationConfiguration) -> Bool? {
    return or0.usesSolAttention
  }
  public static let usesSolAttention: FieldExpr<Bool, GenerationConfiguration> = FieldExpr(
    name: "f182", primaryKey: false, hasIndex: false, tableReader: _tr__f182,
    objectReader: _or__f182)

  private static func _tr__f184(_ table: ByteBuffer) -> Int32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.solAttentionStart
  }
  private static func _or__f184(_ or0: GenerationConfiguration) -> Int32? {
    return or0.solAttentionStart
  }
  public static let solAttentionStart: FieldExpr<Int32, GenerationConfiguration> = FieldExpr(
    name: "f184", primaryKey: false, hasIndex: false, tableReader: _tr__f184,
    objectReader: _or__f184)

  private static func _tr__f186(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_GenerationConfiguration.getRootAsGenerationConfiguration(bb: table)
    return tr0.solAttentionTau
  }
  private static func _or__f186(_ or0: GenerationConfiguration) -> Float32? {
    return or0.solAttentionTau
  }
  public static let solAttentionTau: FieldExpr<Float32, GenerationConfiguration> = FieldExpr(
    name: "f186", primaryKey: false, hasIndex: false, tableReader: _tr__f186,
    objectReader: _or__f186)
}
