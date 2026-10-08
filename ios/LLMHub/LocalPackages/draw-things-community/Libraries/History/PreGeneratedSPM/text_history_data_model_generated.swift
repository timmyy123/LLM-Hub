import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

public enum TextType: Int8, DflatFriendlyValue, CaseIterable {
  case positiveText = 0
  case negativeText = 1
  public static func < (lhs: TextType, rhs: TextType) -> Bool {
    return lhs.rawValue < rhs.rawValue
  }
}

public struct TextRange: Equatable, FlatBuffersDecodable {
  public var location: Int32
  public var length: Int32
  public init(location: Int32? = 0, length: Int32? = 0) {
    self.location = location ?? 0
    self.length = length ?? 0
  }
  public init(_ obj: zzz_DflatGen_TextRange) {
    self.location = obj.location
    self.length = obj.length
  }

  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    // Assuming this is the root
    Self(
      bb.read(
        def: zzz_DflatGen_TextRange.self,
        position: Int(bb.read(def: UOffset.self, position: bb.reader)) + bb.reader))
  }

  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_TextRange>.verify(
        &verifier, at: 0, of: zzz_DflatGen_TextRange.self)
      return true
    } catch {
      return false
    }
  }

  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
}

public struct TextModification: Equatable, FlatBuffersDecodable {
  public var type: TextType
  public var range: TextRange?
  public var text: String?
  public init(type: TextType? = .positiveText, range: TextRange? = nil, text: String? = nil) {
    self.type = type ?? .positiveText
    self.range = range ?? nil
    self.text = text ?? nil
  }
  public init(_ obj: zzz_DflatGen_TextModification) {
    self.type = TextType(rawValue: obj.type.rawValue) ?? .positiveText
    self.range = obj.range.map { TextRange($0) }
    self.text = obj.text
  }

  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_TextModification.getRootAsTextModification(bb: bb))
  }

  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_TextModification>.verify(
        &verifier, at: 0, of: zzz_DflatGen_TextModification.self)
      return true
    } catch {
      return false
    }
  }

  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
}

public final class TextHistoryNode: Dflat.Atom, SQLiteDflat.SQLiteAtom, FlatBuffersDecodable,
  Equatable
{
  public static func == (lhs: TextHistoryNode, rhs: TextHistoryNode) -> Bool {
    guard lhs.lineage == rhs.lineage else { return false }
    guard lhs.logicalTime == rhs.logicalTime else { return false }
    guard lhs.startEdits == rhs.startEdits else { return false }
    guard lhs.startPositiveText == rhs.startPositiveText else { return false }
    guard lhs.startNegativeText == rhs.startNegativeText else { return false }
    guard lhs.modifications == rhs.modifications else { return false }
    return true
  }
  public var _rowid: Int64 = -1
  public var _changesTimestamp: Int64 = -1
  public let lineage: Int64
  public let logicalTime: Int64
  public let startEdits: Int64
  public let startPositiveText: String?
  public let startNegativeText: String?
  public let modifications: [TextModification]
  public init(
    lineage: Int64, logicalTime: Int64, startEdits: Int64? = 0, startPositiveText: String? = nil,
    startNegativeText: String? = nil, modifications: [TextModification]? = []
  ) {
    self.lineage = lineage
    self.logicalTime = logicalTime
    self.startEdits = startEdits ?? 0
    self.startPositiveText = startPositiveText ?? nil
    self.startNegativeText = startNegativeText ?? nil
    self.modifications = modifications ?? []
  }
  public init(_ obj: zzz_DflatGen_TextHistoryNode) {
    self.lineage = obj.lineage
    self.logicalTime = obj.logicalTime
    self.startEdits = obj.startEdits
    self.startPositiveText = obj.startPositiveText
    self.startNegativeText = obj.startNegativeText
    var __modifications = [TextModification]()
    for i: Int32 in 0..<obj.modificationsCount {
      guard let o = obj.modifications(at: i) else { break }
      __modifications.append(TextModification(o))
    }
    self.modifications = __modifications
  }
  public static func from(data: Data) -> Self {
    return data.withUnsafeBytes { buffer in
      let bb = ByteBuffer(
        assumingMemoryBound: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
        capacity: buffer.count)
      return Self(zzz_DflatGen_TextHistoryNode.getRootAsTextHistoryNode(bb: bb))
    }
  }
  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_TextHistoryNode.getRootAsTextHistoryNode(bb: bb))
  }
  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_TextHistoryNode>.verify(
        &verifier, at: 0, of: zzz_DflatGen_TextHistoryNode.self)
      return true
    } catch {
      return false
    }
  }
  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
  public static var table: String { "texthistorynode" }
  public static var indexFields: [String] { [] }
  public static func setUpSchema(_ toolbox: PersistenceToolbox) {
    guard let sqlite = ((toolbox as? SQLitePersistenceToolbox).map { $0.connection }) else {
      return
    }
    sqlite3_exec(
      sqlite.sqlite,
      "CREATE TABLE IF NOT EXISTS texthistorynode (rowid INTEGER PRIMARY KEY AUTOINCREMENT, __pk0 INTEGER, __pk1 INTEGER, p BLOB, UNIQUE(__pk0, __pk1))",
      nil, nil, nil)
  }
  public static func insertIndex(
    _ toolbox: PersistenceToolbox, field: String, rowid: Int64, table: ByteBuffer
  ) -> Bool {
    return true
  }
}

public struct TextHistoryNodeBuilder {
  public var lineage: Int64
  public var logicalTime: Int64
  public var startEdits: Int64
  public var startPositiveText: String?
  public var startNegativeText: String?
  public var modifications: [TextModification]
  public init(from object: TextHistoryNode) {
    lineage = object.lineage
    logicalTime = object.logicalTime
    startEdits = object.startEdits
    startPositiveText = object.startPositiveText
    startNegativeText = object.startNegativeText
    modifications = object.modifications
  }
  public func build() -> TextHistoryNode {
    TextHistoryNode(
      lineage: lineage, logicalTime: logicalTime, startEdits: startEdits,
      startPositiveText: startPositiveText, startNegativeText: startNegativeText,
      modifications: modifications)
  }
}

#if compiler(>=5.5) && canImport(_Concurrency)
  extension TextHistoryNode: @unchecked Sendable {}
#endif
