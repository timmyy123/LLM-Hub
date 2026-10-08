import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

// MARK - SQLiteValue for Enumerations

extension TextType: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

// MARK - Serializer

extension TextRange: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    flatBufferBuilder.create(struct: zzz_DflatGen_TextRange(self))
  }
}

extension zzz_DflatGen_TextRange {
  init(_ obj: TextRange) {
    self.init(location: obj.location, length: obj.length)
  }
  init?(_ obj: TextRange?) {
    guard let obj = obj else { return nil }
    self.init(obj)
  }
}

extension TextModification: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __type = zzz_DflatGen_TextType(rawValue: self.type.rawValue) ?? .positivetext
    let __text = self.text.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let start = zzz_DflatGen_TextModification.startTextModification(&flatBufferBuilder)
    zzz_DflatGen_TextModification.add(type: __type, &flatBufferBuilder)
    let __range = zzz_DflatGen_TextRange(self.range)
    zzz_DflatGen_TextModification.add(range: __range, &flatBufferBuilder)
    zzz_DflatGen_TextModification.add(text: __text, &flatBufferBuilder)
    return zzz_DflatGen_TextModification.endTextModification(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == TextModification {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension TextHistoryNode: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __startPositiveText =
      self.startPositiveText.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __startNegativeText =
      self.startNegativeText.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    var __modifications = [Offset]()
    for i in self.modifications {
      __modifications.append(i.to(flatBufferBuilder: &flatBufferBuilder))
    }
    let __vector_modifications = flatBufferBuilder.createVector(ofOffsets: __modifications)
    let start = zzz_DflatGen_TextHistoryNode.startTextHistoryNode(&flatBufferBuilder)
    zzz_DflatGen_TextHistoryNode.add(lineage: self.lineage, &flatBufferBuilder)
    zzz_DflatGen_TextHistoryNode.add(logicalTime: self.logicalTime, &flatBufferBuilder)
    zzz_DflatGen_TextHistoryNode.add(startEdits: self.startEdits, &flatBufferBuilder)
    zzz_DflatGen_TextHistoryNode.add(startPositiveText: __startPositiveText, &flatBufferBuilder)
    zzz_DflatGen_TextHistoryNode.add(startNegativeText: __startNegativeText, &flatBufferBuilder)
    zzz_DflatGen_TextHistoryNode.addVectorOf(
      modifications: __vector_modifications, &flatBufferBuilder)
    return zzz_DflatGen_TextHistoryNode.endTextHistoryNode(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == TextHistoryNode {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension TextHistoryNode {
  public func toData() -> Data {
    var fbb = FlatBufferBuilder()
    let offset = to(flatBufferBuilder: &fbb)
    fbb.finish(offset: offset)
    return fbb.data
  }
}

// MARK - ChangeRequest

public final class TextHistoryNodeChangeRequest: Dflat.ChangeRequest {
  private var _o: TextHistoryNode?
  public typealias Element = TextHistoryNode
  public var _type: ChangeRequestType
  public var _rowid: Int64
  public var lineage: Int64
  public var logicalTime: Int64
  public var startEdits: Int64
  public var startPositiveText: String?
  public var startNegativeText: String?
  public var modifications: [TextModification]
  private init(type _type: ChangeRequestType) {
    _o = nil
    self._type = _type
    _rowid = -1
    lineage = 0
    logicalTime = 0
    startEdits = 0
    startPositiveText = nil
    startNegativeText = nil
    modifications = []
  }
  private init(type _type: ChangeRequestType, _ _o: TextHistoryNode) {
    self._o = _o
    self._type = _type
    _rowid = _o._rowid
    lineage = _o.lineage
    logicalTime = _o.logicalTime
    startEdits = _o.startEdits
    startPositiveText = _o.startPositiveText
    startNegativeText = _o.startNegativeText
    modifications = _o.modifications
  }
  public static func changeRequest(_ o: TextHistoryNode) -> TextHistoryNodeChangeRequest? {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey =
      o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.lineage, o.logicalTime])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: TextHistoryNode.self, for: key)
    return u.map { TextHistoryNodeChangeRequest(type: .update, $0) }
  }
  public static func upsertRequest(_ o: TextHistoryNode) -> TextHistoryNodeChangeRequest {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey =
      o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.lineage, o.logicalTime])
    guard
      let u = transactionContext.objectRepository.object(
        transactionContext.connection, ofType: TextHistoryNode.self, for: key)
    else {
      return Self.creationRequest(o)
    }
    let changeRequest = TextHistoryNodeChangeRequest(type: .update, o)
    changeRequest._o = u
    changeRequest._rowid = u._rowid
    return changeRequest
  }
  public static func creationRequest(_ o: TextHistoryNode) -> TextHistoryNodeChangeRequest {
    let creationRequest = TextHistoryNodeChangeRequest(type: .creation, o)
    creationRequest._rowid = -1
    return creationRequest
  }
  public static func creationRequest() -> TextHistoryNodeChangeRequest {
    return TextHistoryNodeChangeRequest(type: .creation)
  }
  public static func deletionRequest(_ o: TextHistoryNode) -> TextHistoryNodeChangeRequest? {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey =
      o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.lineage, o.logicalTime])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: TextHistoryNode.self, for: key)
    return u.map { TextHistoryNodeChangeRequest(type: .deletion, $0) }
  }
  var _atom: TextHistoryNode {
    let atom = TextHistoryNode(
      lineage: lineage, logicalTime: logicalTime, startEdits: startEdits,
      startPositiveText: startPositiveText, startNegativeText: startNegativeText,
      modifications: modifications)
    atom._rowid = _rowid
    return atom
  }
  public func commit(_ toolbox: PersistenceToolbox) -> UpdatedObject? {
    guard let toolbox = toolbox as? SQLitePersistenceToolbox else { return nil }
    switch _type {
    case .creation:
      guard
        let insert = toolbox.connection.prepareStaticStatement(
          "INSERT INTO texthistorynode (__pk0, __pk1, p) VALUES (?1, ?2, ?3)")
      else { return nil }
      lineage.bindSQLite(insert, parameterId: 1)
      logicalTime.bindSQLite(insert, parameterId: 2)
      let atom = self._atom
      toolbox.flatBufferBuilder.clear()
      let offset = atom.to(flatBufferBuilder: &toolbox.flatBufferBuilder)
      toolbox.flatBufferBuilder.finish(offset: offset)
      let byteBuffer = toolbox.flatBufferBuilder.buffer
      let memory = byteBuffer.memory.advanced(by: byteBuffer.reader)
      let SQLITE_STATIC = unsafeBitCast(
        OpaquePointer(bitPattern: 0), to: sqlite3_destructor_type.self)
      sqlite3_bind_blob(insert, 3, memory, Int32(byteBuffer.size), SQLITE_STATIC)
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
          "REPLACE INTO texthistorynode (__pk0, __pk1, p, rowid) VALUES (?1, ?2, ?3, ?4)")
      else { return nil }
      lineage.bindSQLite(update, parameterId: 1)
      logicalTime.bindSQLite(update, parameterId: 2)
      toolbox.flatBufferBuilder.clear()
      let offset = atom.to(flatBufferBuilder: &toolbox.flatBufferBuilder)
      toolbox.flatBufferBuilder.finish(offset: offset)
      let byteBuffer = toolbox.flatBufferBuilder.buffer
      let memory = byteBuffer.memory.advanced(by: byteBuffer.reader)
      let SQLITE_STATIC = unsafeBitCast(
        OpaquePointer(bitPattern: 0), to: sqlite3_destructor_type.self)
      sqlite3_bind_blob(update, 3, memory, Int32(byteBuffer.size), SQLITE_STATIC)
      _rowid.bindSQLite(update, parameterId: 4)
      guard SQLITE_DONE == sqlite3_step(update) else { return nil }
      _type = .none
      return .updated(atom)
    case .deletion:
      guard
        let deletion = toolbox.connection.prepareStaticStatement(
          "DELETE FROM texthistorynode WHERE rowid=?1")
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
