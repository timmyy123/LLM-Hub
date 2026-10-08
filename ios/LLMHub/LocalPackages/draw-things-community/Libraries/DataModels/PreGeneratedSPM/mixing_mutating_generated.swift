import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

// MARK - SQLiteValue for Enumerations

extension ModelMixingMode: SQLiteValue {
  public func bindSQLite(_ query: OpaquePointer, parameterId: Int32) {
    self.rawValue.bindSQLite(query, parameterId: parameterId)
  }
}

// MARK - Serializer

extension ModelMixingItem: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __name = self.name.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let start = zzz_DflatGen_ModelMixingItem.startModelMixingItem(&flatBufferBuilder)
    zzz_DflatGen_ModelMixingItem.add(name: __name, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingItem.add(weight: self.weight, &flatBufferBuilder)
    return zzz_DflatGen_ModelMixingItem.endModelMixingItem(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == ModelMixingItem {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension ModelMixingLoRA: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __file = self.file.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let start = zzz_DflatGen_ModelMixingLoRA.startModelMixingLoRA(&flatBufferBuilder)
    zzz_DflatGen_ModelMixingLoRA.add(file: __file, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingLoRA.add(weight: self.weight, &flatBufferBuilder)
    return zzz_DflatGen_ModelMixingLoRA.endModelMixingLoRA(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == ModelMixingLoRA {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension ModelMixingMetadata: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let __name = flatBufferBuilder.create(string: self.name)
    let __triggerWord = self.triggerWord.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __mode = zzz_DflatGen_ModelMixingMode(rawValue: self.mode.rawValue) ?? .weightedsum
    var __items = [Offset]()
    for i in self.items {
      __items.append(i.to(flatBufferBuilder: &flatBufferBuilder))
    }
    let __vector_items = flatBufferBuilder.createVector(ofOffsets: __items)
    let __note = self.note.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __encoder = self.encoder.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    let __decoder = self.decoder.map { flatBufferBuilder.create(string: $0) } ?? Offset()
    var __loras = [Offset]()
    for i in self.loras {
      __loras.append(i.to(flatBufferBuilder: &flatBufferBuilder))
    }
    let __vector_loras = flatBufferBuilder.createVector(ofOffsets: __loras)
    let start = zzz_DflatGen_ModelMixingMetadata.startModelMixingMetadata(&flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.add(name: __name, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.add(triggerWord: __triggerWord, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.add(vPrediction: self.vPrediction, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.add(upcastAttention: self.upcastAttention, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.add(mode: __mode, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.addVectorOf(items: __vector_items, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.add(note: __note, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.add(encoder: __encoder, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.add(decoder: __decoder, &flatBufferBuilder)
    zzz_DflatGen_ModelMixingMetadata.addVectorOf(loras: __vector_loras, &flatBufferBuilder)
    return zzz_DflatGen_ModelMixingMetadata.endModelMixingMetadata(&flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == ModelMixingMetadata {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

extension ModelMixingMetadata {
  public func toData() -> Data {
    var fbb = FlatBufferBuilder()
    let offset = to(flatBufferBuilder: &fbb)
    fbb.finish(offset: offset)
    return fbb.data
  }
}

// MARK - ChangeRequest

public final class ModelMixingMetadataChangeRequest: Dflat.ChangeRequest {
  private var _o: ModelMixingMetadata?
  public typealias Element = ModelMixingMetadata
  public var _type: ChangeRequestType
  public var _rowid: Int64
  public var name: String
  public var triggerWord: String?
  public var vPrediction: Bool
  public var upcastAttention: Bool
  public var mode: ModelMixingMode
  public var items: [ModelMixingItem]
  public var note: String?
  public var encoder: String?
  public var decoder: String?
  public var loras: [ModelMixingLoRA]
  private init(type _type: ChangeRequestType) {
    _o = nil
    self._type = _type
    _rowid = -1
    name = ""
    triggerWord = nil
    vPrediction = false
    upcastAttention = false
    mode = .weightedSum
    items = []
    note = nil
    encoder = nil
    decoder = nil
    loras = []
  }
  private init(type _type: ChangeRequestType, _ _o: ModelMixingMetadata) {
    self._o = _o
    self._type = _type
    _rowid = _o._rowid
    name = _o.name
    triggerWord = _o.triggerWord
    vPrediction = _o.vPrediction
    upcastAttention = _o.upcastAttention
    mode = _o.mode
    items = _o.items
    note = _o.note
    encoder = _o.encoder
    decoder = _o.decoder
    loras = _o.loras
  }
  public static func changeRequest(_ o: ModelMixingMetadata) -> ModelMixingMetadataChangeRequest? {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.name])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: ModelMixingMetadata.self, for: key)
    return u.map { ModelMixingMetadataChangeRequest(type: .update, $0) }
  }
  public static func upsertRequest(_ o: ModelMixingMetadata) -> ModelMixingMetadataChangeRequest {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.name])
    guard
      let u = transactionContext.objectRepository.object(
        transactionContext.connection, ofType: ModelMixingMetadata.self, for: key)
    else {
      return Self.creationRequest(o)
    }
    let changeRequest = ModelMixingMetadataChangeRequest(type: .update, o)
    changeRequest._o = u
    changeRequest._rowid = u._rowid
    return changeRequest
  }
  public static func creationRequest(_ o: ModelMixingMetadata) -> ModelMixingMetadataChangeRequest {
    let creationRequest = ModelMixingMetadataChangeRequest(type: .creation, o)
    creationRequest._rowid = -1
    return creationRequest
  }
  public static func creationRequest() -> ModelMixingMetadataChangeRequest {
    return ModelMixingMetadataChangeRequest(type: .creation)
  }
  public static func deletionRequest(_ o: ModelMixingMetadata) -> ModelMixingMetadataChangeRequest?
  {
    let transactionContext = SQLiteTransactionContext.current!
    let key: SQLiteObjectKey = o._rowid >= 0 ? .rowid(o._rowid) : .primaryKey([o.name])
    let u = transactionContext.objectRepository.object(
      transactionContext.connection, ofType: ModelMixingMetadata.self, for: key)
    return u.map { ModelMixingMetadataChangeRequest(type: .deletion, $0) }
  }
  var _atom: ModelMixingMetadata {
    let atom = ModelMixingMetadata(
      name: name, triggerWord: triggerWord, vPrediction: vPrediction,
      upcastAttention: upcastAttention, mode: mode, items: items, note: note, encoder: encoder,
      decoder: decoder, loras: loras)
    atom._rowid = _rowid
    return atom
  }
  public func commit(_ toolbox: PersistenceToolbox) -> UpdatedObject? {
    guard let toolbox = toolbox as? SQLitePersistenceToolbox else { return nil }
    switch _type {
    case .creation:
      guard
        let insert = toolbox.connection.prepareStaticStatement(
          "INSERT INTO modelmixingmetadata (__pk0, p) VALUES (?1, ?2)")
      else { return nil }
      name.bindSQLite(insert, parameterId: 1)
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
          "REPLACE INTO modelmixingmetadata (__pk0, p, rowid) VALUES (?1, ?2, ?3)")
      else { return nil }
      name.bindSQLite(update, parameterId: 1)
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
          "DELETE FROM modelmixingmetadata WHERE rowid=?1")
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
