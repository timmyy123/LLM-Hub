import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

public enum ModelMixingMode: Int8, DflatFriendlyValue, CaseIterable {
  case weightedSum = 0
  case addDifference = 1
  case freeform = 2
  public static func < (lhs: ModelMixingMode, rhs: ModelMixingMode) -> Bool {
    return lhs.rawValue < rhs.rawValue
  }
}

public struct ModelMixingItem: Equatable, FlatBuffersDecodable {
  public var name: String?
  public var weight: Float32
  public init(name: String? = nil, weight: Float32? = 0.0) {
    self.name = name ?? nil
    self.weight = weight ?? 0.0
  }
  public init(_ obj: zzz_DflatGen_ModelMixingItem) {
    self.name = obj.name
    self.weight = obj.weight
  }

  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_ModelMixingItem.getRootAsModelMixingItem(bb: bb))
  }

  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_ModelMixingItem>.verify(
        &verifier, at: 0, of: zzz_DflatGen_ModelMixingItem.self)
      return true
    } catch {
      return false
    }
  }

  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
}

public struct ModelMixingLoRA: Equatable, FlatBuffersDecodable {
  public var file: String?
  public var weight: Float32
  public init(file: String? = nil, weight: Float32? = 0.6) {
    self.file = file ?? nil
    self.weight = weight ?? 0.6
  }
  public init(_ obj: zzz_DflatGen_ModelMixingLoRA) {
    self.file = obj.file
    self.weight = obj.weight
  }

  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_ModelMixingLoRA.getRootAsModelMixingLoRA(bb: bb))
  }

  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_ModelMixingLoRA>.verify(
        &verifier, at: 0, of: zzz_DflatGen_ModelMixingLoRA.self)
      return true
    } catch {
      return false
    }
  }

  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
}

public final class ModelMixingMetadata: Dflat.Atom, SQLiteDflat.SQLiteAtom, FlatBuffersDecodable,
  Equatable
{
  public static func == (lhs: ModelMixingMetadata, rhs: ModelMixingMetadata) -> Bool {
    guard lhs.name == rhs.name else { return false }
    guard lhs.triggerWord == rhs.triggerWord else { return false }
    guard lhs.vPrediction == rhs.vPrediction else { return false }
    guard lhs.upcastAttention == rhs.upcastAttention else { return false }
    guard lhs.mode == rhs.mode else { return false }
    guard lhs.items == rhs.items else { return false }
    guard lhs.note == rhs.note else { return false }
    guard lhs.encoder == rhs.encoder else { return false }
    guard lhs.decoder == rhs.decoder else { return false }
    guard lhs.loras == rhs.loras else { return false }
    return true
  }
  public var _rowid: Int64 = -1
  public var _changesTimestamp: Int64 = -1
  public let name: String
  public let triggerWord: String?
  public let vPrediction: Bool
  public let upcastAttention: Bool
  public let mode: ModelMixingMode
  public let items: [ModelMixingItem]
  public let note: String?
  public let encoder: String?
  public let decoder: String?
  public let loras: [ModelMixingLoRA]
  public init(
    name: String, triggerWord: String? = nil, vPrediction: Bool? = false,
    upcastAttention: Bool? = false, mode: ModelMixingMode? = .weightedSum,
    items: [ModelMixingItem]? = [], note: String? = nil, encoder: String? = nil,
    decoder: String? = nil, loras: [ModelMixingLoRA]? = []
  ) {
    self.name = name
    self.triggerWord = triggerWord ?? nil
    self.vPrediction = vPrediction ?? false
    self.upcastAttention = upcastAttention ?? false
    self.mode = mode ?? .weightedSum
    self.items = items ?? []
    self.note = note ?? nil
    self.encoder = encoder ?? nil
    self.decoder = decoder ?? nil
    self.loras = loras ?? []
  }
  public init(_ obj: zzz_DflatGen_ModelMixingMetadata) {
    self.name = obj.name!
    self.triggerWord = obj.triggerWord
    self.vPrediction = obj.vPrediction
    self.upcastAttention = obj.upcastAttention
    self.mode = ModelMixingMode(rawValue: obj.mode.rawValue) ?? .weightedSum
    var __items = [ModelMixingItem]()
    for i: Int32 in 0..<obj.itemsCount {
      guard let o = obj.items(at: i) else { break }
      __items.append(ModelMixingItem(o))
    }
    self.items = __items
    self.note = obj.note
    self.encoder = obj.encoder
    self.decoder = obj.decoder
    var __loras = [ModelMixingLoRA]()
    for i: Int32 in 0..<obj.lorasCount {
      guard let o = obj.loras(at: i) else { break }
      __loras.append(ModelMixingLoRA(o))
    }
    self.loras = __loras
  }
  public static func from(data: Data) -> Self {
    return data.withUnsafeBytes { buffer in
      let bb = ByteBuffer(
        assumingMemoryBound: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
        capacity: buffer.count)
      return Self(zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: bb))
    }
  }
  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_ModelMixingMetadata.getRootAsModelMixingMetadata(bb: bb))
  }
  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_ModelMixingMetadata>.verify(
        &verifier, at: 0, of: zzz_DflatGen_ModelMixingMetadata.self)
      return true
    } catch {
      return false
    }
  }
  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
  public static var table: String { "modelmixingmetadata" }
  public static var indexFields: [String] { [] }
  public static func setUpSchema(_ toolbox: PersistenceToolbox) {
    guard let sqlite = ((toolbox as? SQLitePersistenceToolbox).map { $0.connection }) else {
      return
    }
    sqlite3_exec(
      sqlite.sqlite,
      "CREATE TABLE IF NOT EXISTS modelmixingmetadata (rowid INTEGER PRIMARY KEY AUTOINCREMENT, __pk0 TEXT, p BLOB, UNIQUE(__pk0))",
      nil, nil, nil)
  }
  public static func insertIndex(
    _ toolbox: PersistenceToolbox, field: String, rowid: Int64, table: ByteBuffer
  ) -> Bool {
    return true
  }
}

public struct ModelMixingMetadataBuilder {
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
  public init(from object: ModelMixingMetadata) {
    name = object.name
    triggerWord = object.triggerWord
    vPrediction = object.vPrediction
    upcastAttention = object.upcastAttention
    mode = object.mode
    items = object.items
    note = object.note
    encoder = object.encoder
    decoder = object.decoder
    loras = object.loras
  }
  public func build() -> ModelMixingMetadata {
    ModelMixingMetadata(
      name: name, triggerWord: triggerWord, vPrediction: vPrediction,
      upcastAttention: upcastAttention, mode: mode, items: items, note: note, encoder: encoder,
      decoder: decoder, loras: loras)
  }
}

#if compiler(>=5.5) && canImport(_Concurrency)
  extension ModelMixingMetadata: @unchecked Sendable {}
#endif
