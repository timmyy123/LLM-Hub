import Dflat
import FlatBuffers

extension PeerConnectionId {

  private static func _tr__f4(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_PeerConnectionId.getRootAsPeerConnectionId(bb: table)
    return tr0.id!
  }
  private static func _or__f4(_ or0: PeerConnectionId) -> String? {
    return or0.id
  }
  public static let id: FieldExpr<String, PeerConnectionId> = FieldExpr(
    name: "__pk0", primaryKey: true, hasIndex: false, tableReader: _tr__f4, objectReader: _or__f4)

  private static func _tr__f6(_ table: ByteBuffer) -> String? {
    let tr0 = zzz_DflatGen_PeerConnectionId.getRootAsPeerConnectionId(bb: table)
    guard let s = tr0.updatedAt else { return nil }
    return s
  }
  private static func _or__f6(_ or0: PeerConnectionId) -> String? {
    guard let s = or0.updatedAt else { return nil }
    return s
  }
  public static let updatedAt: FieldExpr<String, PeerConnectionId> = FieldExpr(
    name: "f6", primaryKey: false, hasIndex: false, tableReader: _tr__f6, objectReader: _or__f6)
}
