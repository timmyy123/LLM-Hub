import Dflat
import FlatBuffers

extension TextLineageNode {

  private static func _tr__f4(_ table: ByteBuffer) -> Int64? {
    let tr0 = zzz_DflatGen_TextLineageNode.getRootAsTextLineageNode(bb: table)
    return tr0.lineage
  }
  private static func _or__f4(_ or0: TextLineageNode) -> Int64? {
    return or0.lineage
  }
  public static let lineage: FieldExpr<Int64, TextLineageNode> = FieldExpr(
    name: "__pk0", primaryKey: true, hasIndex: false, tableReader: _tr__f4, objectReader: _or__f4)

  private static func _tr__f6(_ table: ByteBuffer) -> Int64? {
    let tr0 = zzz_DflatGen_TextLineageNode.getRootAsTextLineageNode(bb: table)
    return tr0.pointTo
  }
  private static func _or__f6(_ or0: TextLineageNode) -> Int64? {
    return or0.pointTo
  }
  public static let pointTo: FieldExpr<Int64, TextLineageNode> = FieldExpr(
    name: "f6", primaryKey: false, hasIndex: true, tableReader: _tr__f6, objectReader: _or__f6)
}
