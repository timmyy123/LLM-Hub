import Dflat
import FlatBuffers

extension ModelMixingMetadata {

  private static func _tr__f4(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: table)
    return tr0.name!
  }
  private static func _or__f4(_ or0: ModelMixingMetadata) -> String? {
    return or0.name
  }
  public static let name: FieldExpr<String, ModelMixingMetadata> = FieldExpr(
    name: "__pk0", primaryKey: true, hasIndex: false, tableReader: _tr__f4, objectReader: _or__f4)

  private static func _tr__f6(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: table)
    guard let s = tr0.triggerWord else { return nil }
    return s
  }
  private static func _or__f6(_ or0: ModelMixingMetadata) -> String? {
    guard let s = or0.triggerWord else { return nil }
    return s
  }
  public static let triggerWord: FieldExpr<String, ModelMixingMetadata> = FieldExpr(
    name: "f6", primaryKey: false, hasIndex: false, tableReader: _tr__f6, objectReader: _or__f6)

  private static func _tr__f8(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: table)
    return tr0.vPrediction
  }
  private static func _or__f8(_ or0: ModelMixingMetadata) -> Bool? {
    return or0.vPrediction
  }
  public static let vPrediction: FieldExpr<Bool, ModelMixingMetadata> = FieldExpr(
    name: "f8", primaryKey: false, hasIndex: false, tableReader: _tr__f8, objectReader: _or__f8)

  private static func _tr__f10(_ table: ByteBuffer) -> Bool? {
    let tr0 = zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: table)
    return tr0.upcastAttention
  }
  private static func _or__f10(_ or0: ModelMixingMetadata) -> Bool? {
    return or0.upcastAttention
  }
  public static let upcastAttention: FieldExpr<Bool, ModelMixingMetadata> = FieldExpr(
    name: "f10", primaryKey: false, hasIndex: false, tableReader: _tr__f10, objectReader: _or__f10)

  private static func _tr__f12(_ table: ByteBuffer) -> ModelMixingMode? {
    let tr0 = zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: table)
    return ModelMixingMode(rawValue: tr0.mode.rawValue)!
  }
  private static func _or__f12(_ or0: ModelMixingMetadata) -> ModelMixingMode? {
    return or0.mode
  }
  public static let mode: FieldExpr<ModelMixingMode, ModelMixingMetadata> = FieldExpr(
    name: "f12", primaryKey: false, hasIndex: false, tableReader: _tr__f12, objectReader: _or__f12)

  private static func _tr__f16(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: table)
    guard let s = tr0.note else { return nil }
    return s
  }
  private static func _or__f16(_ or0: ModelMixingMetadata) -> String? {
    guard let s = or0.note else { return nil }
    return s
  }
  public static let note: FieldExpr<String, ModelMixingMetadata> = FieldExpr(
    name: "f16", primaryKey: false, hasIndex: false, tableReader: _tr__f16, objectReader: _or__f16)

  private static func _tr__f18(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: table)
    guard let s = tr0.encoder else { return nil }
    return s
  }
  private static func _or__f18(_ or0: ModelMixingMetadata) -> String? {
    guard let s = or0.encoder else { return nil }
    return s
  }
  public static let encoder: FieldExpr<String, ModelMixingMetadata> = FieldExpr(
    name: "f18", primaryKey: false, hasIndex: false, tableReader: _tr__f18, objectReader: _or__f18)

  private static func _tr__f20(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: table)
    guard let s = tr0.decoder else { return nil }
    return s
  }
  private static func _or__f20(_ or0: ModelMixingMetadata) -> String? {
    guard let s = or0.decoder else { return nil }
    return s
  }
  public static let decoder: FieldExpr<String, ModelMixingMetadata> = FieldExpr(
    name: "f20", primaryKey: false, hasIndex: false, tableReader: _tr__f20, objectReader: _or__f20)
}
