import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

public final class PeerConnectionId: Dflat.Atom, SQLiteDflat.SQLiteAtom, FlatBuffersDecodable,
  Equatable
{
  public static func == (lhs: PeerConnectionId, rhs: PeerConnectionId) -> Bool {
    guard lhs.id == rhs.id else { return false }
    guard lhs.updatedAt == rhs.updatedAt else { return false }
    return true
  }
  public var _rowid: Int64 = -1
  public var _changesTimestamp: Int64 = -1
  public let id: String
  public let updatedAt: String?
  public init(id: String, updatedAt: String? = nil) {
    self.id = id
    self.updatedAt = updatedAt ?? nil
  }
  public init(_ obj: zzz_DflatGen_PeerConnectionId) {
    self.id = obj.id!
    self.updatedAt = obj.updatedAt
  }
  public static func from(data: Data) -> Self {
    return data.withUnsafeBytes { buffer in
      let bb = ByteBuffer(
        assumingMemoryBound: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
        capacity: buffer.count)
      return Self(zzz_DflatGen_PeerConnectionId.getRootAsPeerConnectionId(bb: bb))
    }
  }
  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_PeerConnectionId.getRootAsPeerConnectionId(bb: bb))
  }
  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_PeerConnectionId>.verify(
        &verifier, at: 0, of: zzz_DflatGen_PeerConnectionId.self)
      return true
    } catch {
      return false
    }
  }
  public static var flatBuffersSchemaVersion: String? {
    return "1"
  }
  public static var table: String { "peerconnectionid_v1" }
  public static var indexFields: [String] { [] }
  public static func setUpSchema(_ toolbox: PersistenceToolbox) {
    guard let sqlite = ((toolbox as? SQLitePersistenceToolbox).map { $0.connection }) else {
      return
    }
    sqlite3_exec(
      sqlite.sqlite,
      "CREATE TABLE IF NOT EXISTS peerconnectionid_v1 (rowid INTEGER PRIMARY KEY AUTOINCREMENT, __pk0 TEXT, p BLOB, UNIQUE(__pk0))",
      nil, nil, nil)
  }
  public static func insertIndex(
    _ toolbox: PersistenceToolbox, field: String, rowid: Int64, table: ByteBuffer
  ) -> Bool {
    return true
  }
}

public struct PeerConnectionIdBuilder {
  public var id: String
  public var updatedAt: String?
  public init(from object: PeerConnectionId) {
    id = object.id
    updatedAt = object.updatedAt
  }
  public func build() -> PeerConnectionId {
    PeerConnectionId(id: id, updatedAt: updatedAt)
  }
}

#if compiler(>=5.5) && canImport(_Concurrency)
  extension PeerConnectionId: @unchecked Sendable {}
#endif
