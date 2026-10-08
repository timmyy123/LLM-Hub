import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

// MARK - SQLiteValue for Enumerations

// MARK - Serializer

extension PaintColor: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let start = zzz_DflatGen_PaintColor.startPaintColor(&flatBufferBuilder)
    zzz_DflatGen_PaintColor.add(index: self.index, &flatBufferBuilder)
    zzz_DflatGen_PaintColor.add(r: self.r, &flatBufferBuilder)
    zzz_DflatGen_PaintColor.add(g: self.g, &flatBufferBuilder)
    zzz_DflatGen_PaintColor.add(b: self.b, &flatBufferBuilder)
    zzz_DflatGen_PaintColor.add(a: self.a, &flatBufferBuilder)
    return zzz_DflatGen_PaintColor.endPaintColor(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == PaintColor {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension PaintColor {
  public func toData() -> Data {
    var fbb = FlatBufferBuilder()
    let offset = to(flatBufferBuilder: &fbb)
    fbb.finish(offset: offset)
    return fbb.data
  }
}

// MARK - ChangeRequest

public final class PaintColorChangeRequest: Dflat.ChangeRequest {
  private var _o: PaintColor?
  public typealias Element = PaintColor
  public var _type: ChangeRequestType
  public var _rowid: Int64
  public var index: Int64
  public var r: Float32
  public var g: Float32
  public var b: Float32
  public var a: Float32
  private init(type _type: ChangeRequestType) {
    _o = nil
    self._type = _type
    _rowid = -1
    index = 0
    r = 0.0
    g = 0.0
    b = 0.0
    a = 0.0
  }
  private init(type _type: ChangeRequestType, _ _o: PaintColor) {
    self._o = _o
    self._type = _type
    _rowid = _o._rowid
    index = _o.index
    r = _o.r
    g = _o.g
    b = _o.b
    a = _o.a
  }
  public static func changeRequest(_ o: PaintColor) -> PaintColorChangeRequest? {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.index])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: PaintColor.self, for: key)
    return u.map { PaintColorChangeRequest(type: .update, $0) }
  }
  public static func upsertRequest(_ o: PaintColor) -> PaintColorChangeRequest {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.index])
    guard
      let u = transactionContext.objectRepository.object(
        transactionContext.connection, ofType: PaintColor.self, for: key)
    else {
      return Self.creationRequest(o)
    }
    let changeRequest = PaintColorChangeRequest(type: .update, o)
    changeRequest._o = u
    changeRequest._rowid = u._rowid
    return changeRequest
  }
  public static func creationRequest(_ o: PaintColor) -> PaintColorChangeRequest {
    let creationRequest = PaintColorChangeRequest(type: .creation, o)
    creationRequest._rowid = -1
    return creationRequest
  }
  public static func creationRequest() -> PaintColorChangeRequest {
    return PaintColorChangeRequest(type: .creation)
  }
  public static func deletionRequest(_ o: PaintColor) -> PaintColorChangeRequest? {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.index])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: PaintColor.self, for: key)
    return u.map { PaintColorChangeRequest(type: .deletion, $0) }
  }
  var _atom: PaintColor {
    let atom = PaintColor(index: index, r: r, g: g, b: b, a: a)
    atom._rowid = _rowid
    return atom
  }
  public func commit(_ toolbox: PersistenceToolbox) -> UpdatedObject? {
    guard let toolbox = toolbox as? SQLitePersistenceToolbox else { return nil }
    switch _type {
    case .creation:
      guard
        let insert = toolbox.connection.prepareStaticStatement(
          "INSERT INTO paintcolor_v1 (__pk0, p) VALUES (?1, ?2)")
      else { return nil }
      index.bindSQLite(insert, parameterId: 1)
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
          "REPLACE INTO paintcolor_v1 (__pk0, p, rowid) VALUES (?1, ?2, ?3)")
      else { return nil }
      index.bindSQLite(update, parameterId: 1)
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
          "DELETE FROM paintcolor_v1 WHERE rowid=?1")
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
