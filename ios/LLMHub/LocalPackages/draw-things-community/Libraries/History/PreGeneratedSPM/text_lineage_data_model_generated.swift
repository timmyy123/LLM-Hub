import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

public final class TextLineageNode: Dflat.Atom, SQLiteDflat.SQLiteAtom, FlatBuffersDecodable,
  Equatable
{
  public static func == (lhs: TextLineageNode, rhs: TextLineageNode) -> Bool {
    guard lhs.lineage == rhs.lineage else { return false }
    guard lhs.pointTo == rhs.pointTo else { return false }
    return true
  }
  public var _rowid: Int64 = -1
  public var _changesTimestamp: Int64 = -1
  public let lineage: Int64
  public let pointTo: Int64
  public init(lineage: Int64, pointTo: Int64? = 0) {
    self.lineage = lineage
    self.pointTo = pointTo ?? 0
  }
  public init(_ obj: zzz_DflatGen_TextLineageNode) {
    self.lineage = obj.lineage
    self.pointTo = obj.pointTo
  }
  public static func from(data: Data) -> Self {
    return data.withUnsafeBytes { buffer in
      let bb = ByteBuffer(
        assumingMemoryBound: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
        capacity: buffer.count)
      return Self(zzz_DflatGen_TextLineageNode.getRootAsTextLineageNode(bb: bb))
    }
  }
  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_TextLineageNode.getRootAsTextLineageNode(bb: bb))
  }
  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_TextLineageNode>.verify(
        &verifier, at: 0, of: zzz_DflatGen_TextLineageNode.self)
      return true
    } catch {
      return false
    }
  }
  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
  public static var table: String { "textlineagenode" }
  public static var indexFields: [String] { ["f6"] }
  public static func setUpSchema(_ toolbox: PersistenceToolbox) {
    guard let sqlite = ((toolbox as? SQLitePersistenceToolbox).map { $0.connection }) else {
      return
    }
    sqlite3_exec(
      sqlite.sqlite,
      "CREATE TABLE IF NOT EXISTS textlineagenode (rowid INTEGER PRIMARY KEY AUTOINCREMENT, __pk0 INTEGER, p BLOB, UNIQUE(__pk0))",
      nil, nil, nil)
    sqlite3_exec(
      sqlite.sqlite,
      "CREATE TABLE IF NOT EXISTS textlineagenode__f6 (rowid INTEGER PRIMARY KEY, f6 INTEGER)", nil,
      nil, nil)
    sqlite3_exec(
      sqlite.sqlite,
      "CREATE INDEX IF NOT EXISTS index__textlineagenode__f6 ON textlineagenode__f6 (f6)", nil, nil,
      nil)
    sqlite.clearIndexStatus(for: Self.table)
  }
  public static func insertIndex(
    _ toolbox: PersistenceToolbox, field: String, rowid: Int64, table: ByteBuffer
  ) -> Bool {
    guard let sqlite = ((toolbox as? SQLitePersistenceToolbox).map { $0.connection }) else {
      return false
    }
    switch field {
    case "f6":
      guard
        let insert = sqlite.prepareStaticStatement(
          "INSERT INTO textlineagenode__f6 (rowid, f6) VALUES (?1, ?2)")
      else { return false }
      rowid.bindSQLite(insert, parameterId: 1)
      if let retval = TextLineageNode.pointTo.evaluate(byteBuffer: table) {
        retval.bindSQLite(insert, parameterId: 2)
      } else {
        sqlite3_bind_null(insert, 2)
      }
      guard SQLITE_DONE == sqlite3_step(insert) else { return false }
    default:
      break
    }
    return true
  }
}

public struct TextLineageNodeBuilder {
  public var lineage: Int64
  public var pointTo: Int64
  public init(from object: TextLineageNode) {
    lineage = object.lineage
    pointTo = object.pointTo
  }
  public func build() -> TextLineageNode {
    TextLineageNode(lineage: lineage, pointTo: pointTo)
  }
}

#if compiler(>=5.5) && canImport(_Concurrency)
  extension TextLineageNode: @unchecked Sendable {}
#endif
