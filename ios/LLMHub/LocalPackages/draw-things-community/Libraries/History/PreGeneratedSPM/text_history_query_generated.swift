import Dflat
import FlatBuffers

extension TextHistoryNode {

  private static func _tr__f4(_ table: ByteBuffer) -> Int64? {
    let tr0 = zzz_DflatGen_TextHistoryNode.getRootAsTextHistoryNode(bb: table)
    return tr0.lineage
  }
  private static func _or__f4(_ or0: TextHistoryNode) -> Int64? {
    return or0.lineage
  }
  public static let lineage: FieldExpr<Int64, TextHistoryNode> = FieldExpr(
    name: "__pk0", primaryKey: true, hasIndex: false, tableReader: _tr__f4, objectReader: _or__f4)

  private static func _tr__f6(_ table: ByteBuffer) -> Int64? {
    let tr0 = zzz_DflatGen_TextHistoryNode.getRootAsTextHistoryNode(bb: table)
    return tr0.logicalTime
  }
  private static func _or__f6(_ or0: TextHistoryNode) -> Int64? {
    return or0.logicalTime
  }
  public static let logicalTime: FieldExpr<Int64, TextHistoryNode> = FieldExpr(
    name: "__pk1", primaryKey: true, hasIndex: false, tableReader: _tr__f6, objectReader: _or__f6)

  private static func _tr__f8(_ table: ByteBuffer) -> Int64? {
    let tr0 = zzz_DflatGen_TextHistoryNode.getRootAsTextHistoryNode(bb: table)
    return tr0.startEdits
  }
  private static func _or__f8(_ or0: TextHistoryNode) -> Int64? {
    return or0.startEdits
  }
  public static let startEdits: FieldExpr<Int64, TextHistoryNode> = FieldExpr(
    name: "f8", primaryKey: false, hasIndex: false, tableReader: _tr__f8, objectReader: _or__f8)

  private static func _tr__f10(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_TextHistoryNode.getRootAsTextHistoryNode(bb: table)
    guard let s = tr0.startPositiveText else { return nil }
    return s
  }
  private static func _or__f10(_ or0: TextHistoryNode) -> String? {
    guard let s = or0.startPositiveText else { return nil }
    return s
  }
  public static let startPositiveText: FieldExpr<String, TextHistoryNode> = FieldExpr(
    name: "f10", primaryKey: false, hasIndex: false, tableReader: _tr__f10, objectReader: _or__f10)

  private static func _tr__f12(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_TextHistoryNode.getRootAsTextHistoryNode(bb: table)
    guard let s = tr0.startNegativeText else { return nil }
    return s
  }
  private static func _or__f12(_ or0: TextHistoryNode) -> String? {
    guard let s = or0.startNegativeText else { return nil }
    return s
  }
  public static let startNegativeText: FieldExpr<String, TextHistoryNode> = FieldExpr(
    name: "f12", primaryKey: false, hasIndex: false, tableReader: _tr__f12, objectReader: _or__f12)
}
