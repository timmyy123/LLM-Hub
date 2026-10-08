import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

public final class PaintColor: Dflat.Atom, SQLiteDflat.SQLiteAtom, FlatBuffersDecodable, Equatable {
  public static func == (lhs: PaintColor, rhs: PaintColor) -> Bool {
    guard lhs.index == rhs.index else { return false }
    guard lhs.r == rhs.r else { return false }
    guard lhs.g == rhs.g else { return false }
    guard lhs.b == rhs.b else { return false }
    guard lhs.a == rhs.a else { return false }
    return true
  }
  public var _rowid: Int64 = -1
  public var _changesTimestamp: Int64 = -1
  public let index: Int64
  public let r: Float32
  public let g: Float32
  public let b: Float32
  public let a: Float32
  public init(
    index: Int64, r: Float32? = 0.0, g: Float32? = 0.0, b: Float32? = 0.0, a: Float32? = 0.0
  ) {
    self.index = index
    self.r = r ?? 0.0
    self.g = g ?? 0.0
    self.b = b ?? 0.0
    self.a = a ?? 0.0
  }
  public init(_ obj: zzz_DflatGen_PaintColor) {
    self.index = obj.index
    self.r = obj.r
    self.g = obj.g
    self.b = obj.b
    self.a = obj.a
  }
  public static func from(data: Data) -> Self {
    return data.withUnsafeBytes { buffer in
      let bb = ByteBuffer(
        assumingMemoryBound: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
        capacity: buffer.count)
      return Self(zzz_DflatGen_PaintColor.getRootAsPaintColor(bb: bb))
    }
  }
  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_PaintColor.getRootAsPaintColor(bb: bb))
  }
  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_PaintColor>.verify(
        &verifier, at: 0, of: zzz_DflatGen_PaintColor.self)
      return true
    } catch {
      return false
    }
  }
  public static var flatBuffersSchemaVersion: String? {
    return "1"
  }
  public static var table: String { "paintcolor_v1" }
  public static var indexFields: [String] { [] }
  public static func setUpSchema(_ toolbox: PersistenceToolbox) {
    guard let sqlite = ((toolbox as? SQLitePersistenceToolbox).map { $0.connection }) else {
      return
    }
    sqlite3_exec(
      sqlite.sqlite,
      "CREATE TABLE IF NOT EXISTS paintcolor_v1 (rowid INTEGER PRIMARY KEY AUTOINCREMENT, __pk0 INTEGER, p BLOB, UNIQUE(__pk0))",
      nil, nil, nil)
  }
  public static func insertIndex(
    _ toolbox: PersistenceToolbox, field: String, rowid: Int64, table: ByteBuffer
  ) -> Bool {
    return true
  }
}

public struct PaintColorBuilder {
  public var index: Int64
  public var r: Float32
  public var g: Float32
  public var b: Float32
  public var a: Float32
  public init(from object: PaintColor) {
    index = object.index
    r = object.r
    g = object.g
    b = object.b
    a = object.a
  }
  public func build() -> PaintColor {
    PaintColor(index: index, r: r, g: g, b: b, a: a)
  }
}

#if compiler(>=5.5) && canImport(_Concurrency)
  extension PaintColor: @unchecked Sendable {}
#endif
