import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

// MARK - SQLiteValue for Enumerations

// MARK - Serializer

extension TrainingData: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __id = flatBufferBuilder.create(string: self.id)
    let __caption = self.caption.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let start = zzz_DflatGen_TrainingData.startTrainingData(&flatBufferBuilder)
    zzz_DflatGen_TrainingData.add(id: __id, &flatBufferBuilder)
    zzz_DflatGen_TrainingData.add(caption: __caption, &flatBufferBuilder)
    return zzz_DflatGen_TrainingData.endTrainingData(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == TrainingData {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension TrainingData {
  public func toData() -> Data {
    var fbb = FlatBufferBuilder()
    let offset = to(flatBufferBuilder: &fbb)
    fbb.finish(offset: offset)
    return fbb.data
  }
}

// MARK - ChangeRequest

public final class TrainingDataChangeRequest: Dflat.ChangeRequest {
  private var _o: TrainingData?
  public typealias Element = TrainingData
  public var _type: ChangeRequestType
  public var _rowid: Int64
  public var id: String
  public var caption: String?
  private init(type _type: ChangeRequestType) {
    _o = nil
    self._type = _type
    _rowid = -1
    id = ""
    caption = nil
  }
  private init(type _type: ChangeRequestType, _ _o: TrainingData) {
    self._o = _o
    self._type = _type
    _rowid = _o._rowid
    id = _o.id
    caption = _o.caption
  }
  public static func changeRequest(_ o: TrainingData) -> TrainingDataChangeRequest? {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.id])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: TrainingData.self, for: key)
    return u.map { TrainingDataChangeRequest(type: .update, $0) }
  }
  public static func upsertRequest(_ o: TrainingData) -> TrainingDataChangeRequest {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.id])
    guard
      let u = transactionContext.objectRepository.object(
        transactionContext.connection, ofType: TrainingData.self, for: key)
    else {
      return Self.creationRequest(o)
    }
    let changeRequest = TrainingDataChangeRequest(type: .update, o)
    changeRequest._o = u
    changeRequest._rowid = u._rowid
    return changeRequest
  }
  public static func creationRequest(_ o: TrainingData) -> TrainingDataChangeRequest {
    let creationRequest = TrainingDataChangeRequest(type: .creation, o)
    creationRequest._rowid = -1
    return creationRequest
  }
  public static func creationRequest() -> TrainingDataChangeRequest {
    return TrainingDataChangeRequest(type: .creation)
  }
  public static func deletionRequest(_ o: TrainingData) -> TrainingDataChangeRequest? {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.id])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: TrainingData.self, for: key)
    return u.map { TrainingDataChangeRequest(type: .deletion, $0) }
  }
  var _atom: TrainingData {
    let atom = TrainingData(id: id, caption: caption)
    atom._rowid = _rowid
    return atom
  }
  public func commit(_ toolbox: PersistenceToolbox) -> UpdatedObject? {
    guard let toolbox = toolbox as? SQLitePersistenceToolbox else { return nil }
    switch _type {
    case .creation:
      guard
        let insert = toolbox.connection.prepareStaticStatement(
          "INSERT INTO trainingdata (__pk0, p) VALUES (?1, ?2)")
      else { return nil }
      id.bindSQLite(insert, parameterId: 1)
      let atom = self._atom
      toolbox.flatBufferBuilder.clear()
      let offset = atom.to(flatBufferBuilder: &toolbox.flatBufferBuilder)
      toolbox.flatBufferBuilder.finish(offset: offset)
      let byteBuffer = toolbox.flatBufferBuilder.buffer
      let memory = byteBuffer.memory.advanced(by: byteBuffer.reader)
      let SQLITE_STATIC = unsafeBitCast(
        OpaquePointer(bitPattern: 0), to: sqlite3_destructor_type.self)
      sqlite3_bind_blob(insert, 2, memory, Int32(byteBuffer.size), SQLITE_STATIC)
      guard SQLITE_DONE == sqlite3_step(insert) else { return nil }
      _rowid = sqlite3_last_insert_rowid(toolbox.connection.sqlite)
      _type = .none
      atom._rowid = _rowid
      return .inserted(atom)
    case .update:
      guard let o = _o else { return nil }
      let atom = self._atom
      guard atom != o else {
        _type = .none
        return .identity(atom)
      }
      guard
        let update = toolbox.connection.prepareStaticStatement(
          "REPLACE INTO trainingdata (__pk0, p, rowid) VALUES (?1, ?2, ?3)")
      else { return nil }
      id.bindSQLite(update, parameterId: 1)
      toolbox.flatBufferBuilder.clear()
      let offset = atom.to(flatBufferBuilder: &toolbox.flatBufferBuilder)
      toolbox.flatBufferBuilder.finish(offset: offset)
      let byteBuffer = toolbox.flatBufferBuilder.buffer
      let memory = byteBuffer.memory.advanced(by: byteBuffer.reader)
      let SQLITE_STATIC = unsafeBitCast(
        OpaquePointer(bitPattern: 0), to: sqlite3_destructor_type.self)
      sqlite3_bind_blob(update, 2, memory, Int32(byteBuffer.size), SQLITE_STATIC)
      _rowid.bindSQLite(update, parameterId: 3)
      guard SQLITE_DONE == sqlite3_step(update) else { return nil }
      _type = .none
      return .updated(atom)
    case .deletion:
      guard
        let deletion = toolbox.connection.prepareStaticStatement(
          "DELETE FROM trainingdata WHERE rowid=?1")
      else { return nil }
      _rowid.bindSQLite(deletion, parameterId: 1)
      guard SQLITE_DONE == sqlite3_step(deletion) else { return nil }
      _type = .none
      return .deleted(_rowid)
    case .none:
      preconditionFailure()
    }
  }
}
