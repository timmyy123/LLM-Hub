import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

public final class TrainingData: Dflat.Atom, SQLiteDflat.SQLiteAtom, FlatBuffersDecodable, Equatable
{
  public static func == (lhs: TrainingData, rhs: TrainingData) -> Bool {
    guard lhs.id == rhs.id else { return false }
    guard lhs.caption == rhs.caption else { return false }
    return true
  }
  public var _rowid: Int64 = -1
  public var _changesTimestamp: Int64 = -1
  public let id: String
  public let caption: String?
  public init(id: String, caption: String? = nil) {
    self.id = id
    self.caption = caption ?? nil
  }
  public init(_ obj: zzz_DflatGen_TrainingData) {
    self.id = obj.id!
    self.caption = obj.caption
  }
  public static func from(data: Data) -> Self {
    return data.withUnsafeBytes { buffer in
      let bb = ByteBuffer(
        assumingMemoryBound: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
        capacity: buffer.count)
      return Self(zzz_DflatGen_TrainingData.getRootAsTrainingData(bb: bb))
    }
  }
  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_TrainingData.getRootAsTrainingData(bb: bb))
  }
  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_TrainingData>.verify(
        &verifier, at: 0, of: zzz_DflatGen_TrainingData.self)
      return true
    } catch {
      return false
    }
  }
  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
  public static var table: String { "trainingdata" }
  public static var indexFields: [String] { [] }
  public static func setUpSchema(_ toolbox: PersistenceToolbox) {
    guard let sqlite = ((toolbox as? SQLitePersistenceToolbox).map { $0.connection }) else {
      return
    }
    sqlite3_exec(
      sqlite.sqlite,
      "CREATE TABLE IF NOT EXISTS trainingdata (rowid INTEGER PRIMARY KEY AUTOINCREMENT, __pk0 TEXT, p BLOB, UNIQUE(__pk0))",
      nil, nil, nil)
  }
  public static func insertIndex(
    _ toolbox: PersistenceToolbox, field: String, rowid: Int64, table: ByteBuffer
  ) -> Bool {
    return true
  }
}

public struct TrainingDataBuilder {
  public var id: String
  public var caption: String?
  public init(from object: TrainingData) {
    id = object.id
    caption = object.caption
  }
  public func build() -> TrainingData {
    TrainingData(id: id, caption: caption)
  }
}

#if compiler(>=5.5) && canImport(_Concurrency)
  extension TrainingData: @unchecked Sendable {}
#endif
