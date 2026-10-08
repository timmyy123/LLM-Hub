import Dflat
import FlatBuffers

extension TrainingData {

  private static func _tr__f4(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_TrainingData.getRootAsTrainingData(bb: table)
    return tr0.id!
  }
  private static func _or__f4(_ or0: TrainingData) -> String? {
    return or0.id
  }
  public static let id: FieldExpr<String, TrainingData> = FieldExpr(
    name: "__pk0", primaryKey: true, hasIndex: false, tableReader: _tr__f4, objectReader: _or__f4)

  private static func _tr__f6(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_TrainingData.getRootAsTrainingData(bb: table)
    guard let s = tr0.caption else { return nil }
    return s
  }
  private static func _or__f6(_ or0: TrainingData) -> String? {
    guard let s = or0.caption else { return nil }
    return s
  }
  public static let caption: FieldExpr<String, TrainingData> = FieldExpr(
    name: "f6", primaryKey: false, hasIndex: false, tableReader: _tr__f6, objectReader: _or__f6)
}
