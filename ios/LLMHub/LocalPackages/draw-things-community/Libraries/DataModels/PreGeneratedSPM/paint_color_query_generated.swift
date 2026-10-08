import Dflat
import FlatBuffers

extension PaintColor {

  private static func _tr__f4(_ table: ByteBuffer) -> Int64? {
    let tr0 = zzz_DflatGen_PaintColor.getRootAsPaintColor(bb: table)
    return tr0.index
  }
  private static func _or__f4(_ or0: PaintColor) -> Int64? {
    return or0.index
  }
  public static let index: FieldExpr<Int64, PaintColor> = FieldExpr(
    name: "__pk0", primaryKey: true, hasIndex: false, tableReader: _tr__f4, objectReader: _or__f4)

  private static func _tr__f6(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_PaintColor.getRootAsPaintColor(bb: table)
    return tr0.r
  }
  private static func _or__f6(_ or0: PaintColor) -> Float32? {
    return or0.r
  }
  public static let r: FieldExpr<Float32, PaintColor> = FieldExpr(
    name: "f6", primaryKey: false, hasIndex: false, tableReader: _tr__f6, objectReader: _or__f6)

  private static func _tr__f8(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_PaintColor.getRootAsPaintColor(bb: table)
    return tr0.g
  }
  private static func _or__f8(_ or0: PaintColor) -> Float32? {
    return or0.g
  }
  public static let g: FieldExpr<Float32, PaintColor> = FieldExpr(
    name: "f8", primaryKey: false, hasIndex: false, tableReader: _tr__f8, objectReader: _or__f8)

  private static func _tr__f10(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_PaintColor.getRootAsPaintColor(bb: table)
    return tr0.b
  }
  private static func _or__f10(_ or0: PaintColor) -> Float32? {
    return or0.b
  }
  public static let b: FieldExpr<Float32, PaintColor> = FieldExpr(
    name: "f10", primaryKey: false, hasIndex: false, tableReader: _tr__f10, objectReader: _or__f10)

  private static func _tr__f12(_ table: ByteBuffer) -> Float32? {
    let tr0 = zzz_DflatGen_PaintColor.getRootAsPaintColor(bb: table)
    return tr0.a
  }
  private static func _or__f12(_ or0: PaintColor) -> Float32? {
    return or0.a
  }
  public static let a: FieldExpr<Float32, PaintColor> = FieldExpr(
    name: "f12", primaryKey: false, hasIndex: false, tableReader: _tr__f12, objectReader: _or__f12)
}
