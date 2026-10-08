import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

// MARK - SQLiteValue for Enumerations

// MARK - Serializer

extension TextLineageNode: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let start = zzz_DflatGen_TextLineageNode.startTextLineageNode(&flatBufferBuilder)
    zzz_DflatGen_TextLineageNode.add(lineage: self.lineage, &flatBufferBuilder)
    zzz_DflatGen_TextLineageNode.add(pointTo: self.pointTo, &flatBufferBuilder)
    return zzz_DflatGen_TextLineageNode.endTextLineageNode(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == TextLineageNode {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension TextLineageNode {
  public func toData() -> Data {
    var fbb = FlatBufferBuilder()
    let offset = to(flatBufferBuilder: &fbb)
    fbb.finish(offset: offset)
    return fbb.data
  }
}

// MARK - ChangeRequest

public final class TextLineageNodeChangeRequest: Dflat.ChangeRequest {
  private var _o: TextLineageNode?
  public typealias Element = TextLineageNode
  public var _type: ChangeRequestType
  public var _rowid: Int64
  public var lineage: Int64
  public var pointTo: Int64
  private init(type _type: ChangeRequestType) {
    _o = nil
    self._type = _type
    _rowid = -1
    lineage = 0
    pointTo = 0
  }
  private init(type _type: ChangeRequestType, _ _o: TextLineageNode) {
    self._o = _o
    self._type = _type
    _rowid = _o._rowid
    lineage = _o.lineage
    pointTo = _o.pointTo
  }
  public static func changeRequest(_ o: TextLineageNode) -> TextLineageNodeChangeRequest? {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.lineage])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: TextLineageNode.self, for: key)
    return u.map { TextLineageNodeChangeRequest(type: .update, $0) }
  }
  public static func upsertRequest(_ o: TextLineageNode) -> TextLineageNodeChangeRequest {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.lineage])
    guard
      let u = transactionContext.objectRepository.object(
        transactionContext.connection, ofType: TextLineageNode.self, for: key)
    else {
      return Self.creationRequest(o)
    }
    let changeRequest = TextLineageNodeChangeRequest(type: .update, o)
    changeRequest._o = u
    changeRequest._rowid = u._rowid
    return changeRequest
  }
  public static func creationRequest(_ o: TextLineageNode) -> TextLineageNodeChangeRequest {
    let creationRequest = TextLineageNodeChangeRequest(type: .creation, o)
    creationRequest._rowid = -1
    return creationRequest
  }
  public static func creationRequest() -> TextLineageNodeChangeRequest {
    return TextLineageNodeChangeRequest(type: .creation)
  }
  public static func deletionRequest(_ o: TextLineageNode) -> TextLineageNodeChangeRequest? {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.lineage])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: TextLineageNode.self, for: key)
    return u.map { TextLineageNodeChangeRequest(type: .deletion, $0) }
  }
  var _atom: TextLineageNode {
    let atom = TextLineageNode(lineage: lineage, pointTo: pointTo)
    atom._rowid = _rowid
    return atom
  }
  public func commit(_ toolbox: PersistenceToolbox) -> UpdatedObject? {
    guard let toolbox = toolbox as? SQLitePersistenceToolbox else { return nil }
    switch _type {
    case .creation:
      let indexSurvey = toolbox.connection.indexSurvey(
        TextLineageNode.indexFields, table: TextLineageNode.table)
      guard
        let insert = toolbox.connection.prepareStaticStatement(
          "INSERT INTO textlineagenode (__pk0, p) VALUES (?1, ?2)")
      else { return nil }
      lineage.bindSQLite(insert, parameterId: 1)
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
      if indexSurvey.full.contains("f6") {
        guard
          let i0 = toolbox.connection.prepareStaticStatement(
            "INSERT INTO textlineagenode__f6 (rowid, f6) VALUES (?1, ?2)")
        else { return nil }
        _rowid.bindSQLite(i0, parameterId: 1)
        if let r0 = TextLineageNode.pointTo.evaluate(object: atom) {
          r0.bindSQLite(i0, parameterId: 2)
        } else {
          sqlite3_bind_null(i0, 2)
        }
        guard SQLITE_DONE == sqlite3_step(i0) else { return nil }
      }
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
      let indexSurvey = toolbox.connection.indexSurvey(
        TextLineageNode.indexFields, table: TextLineageNode.table)
      guard
        let update = toolbox.connection.prepareStaticStatement(
          "REPLACE INTO textlineagenode (__pk0, p, rowid) VALUES (?1, ?2, ?3)")
      else { return nil }
      lineage.bindSQLite(update, parameterId: 1)
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
      if indexSurvey.full.contains("f6") {
        let or0 = TextLineageNode.pointTo.evaluate(object: o)
        let r0 = TextLineageNode.pointTo.evaluate(object: atom)
        if or0 != r0 {
          guard
            let u0 = toolbox.connection.prepareStaticStatement(
              "REPLACE INTO textlineagenode__f6 (rowid, f6) VALUES (?1, ?2)")
          else { return nil }
          _rowid.bindSQLite(u0, parameterId: 1)
          if let ur0 = r0 {
            ur0.bindSQLite(u0, parameterId: 2)
          } else {
            sqlite3_bind_null(u0, 2)
          }
          guard SQLITE_DONE == sqlite3_step(u0) else { return nil }
        }
      }
      _type = .none
      return .updated(atom)
    case .deletion:
      guard
        let deletion = toolbox.connection.prepareStaticStatement(
          "DELETE FROM textlineagenode WHERE rowid=?1")
      else { return nil }
      _rowid.bindSQLite(deletion, parameterId: 1)
      guard SQLITE_DONE == sqlite3_step(deletion) else { return nil }
      if let d0 = toolbox.connection.prepareStaticStatement(
        "DELETE FROM textlineagenode__f6 WHERE rowid=?1")
      {
        _rowid.bindSQLite(d0, parameterId: 1)
        sqlite3_step(d0)
      }
      _type = .none
      return .deleted(_rowid)
    case .none:
      preconditionFailure()
    }
  }
}
