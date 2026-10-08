import Foundation
import Guaka
import Localization
import Utils

public protocol Parameter {
  var commandLineFlag: String { get }
  var title: String { get }
  var additionalJsonKeys: [String] { get }
  func createFlag() -> Flag
  func validate() throws
  func encode(to container: inout KeyedEncodingContainer<JSONKey>, forKey: JSONKey) throws
  func decode(from container: KeyedDecodingContainer<JSONKey>, forKey: JSONKey) throws
  func decode(fromFlagValue flagValue: FlagValue)
}

public final class CaseIterableEnum<T: CaseIterable & CommandLineAbbreviatable>: FlagValue {
  public let value: T

  init(value: T) {
    self.value = value
  }

  public static func fromString(flagValue value: String) throws -> Self {
    for enumCase in T.allCases {
      if enumCase.commandLineAbbreviation == value {
        return Self.init(value: enumCase)
      }
    }
    throw FlagValueError.conversionError(
      "Value \"\(value)\" doesn't match any of the valid options: \(typeDescription)")
  }

  public static var typeDescription: String {
    let options = T.allCases.map { $0.commandLineAbbreviation }
    return "\(options.map { "\"\($0)\"" }.joined(separator: ", "))"
  }
}

public final class EnumParameter<T: CaseIterable & CommandLineAbbreviatable>: Parameter {
  public let title: String
  public let commandLineFlag: String
  public let explanationKey: String?
  public let additionalJsonKeys: [String]
  public var value: T
  public let defaultValue: T

  init(
    titleKey: String, explanationKey: String?, defaultValue: T, commandLineFlag: String,
    additionalJsonKeys: [String] = []
  ) {
    self.title = LocalizedString.forKey(titleKey)
    self.commandLineFlag = commandLineFlag
    self.explanationKey = explanationKey
    self.value = defaultValue
    self.defaultValue = defaultValue
    self.additionalJsonKeys = additionalJsonKeys
  }

  public func validate() throws {
  }

  public func encode(to container: inout KeyedEncodingContainer<JSONKey>, forKey key: JSONKey)
    throws
  {
    try container.encode(value.commandLineAbbreviation, forKey: key)
  }

  public func decode(from container: KeyedDecodingContainer<JSONKey>, forKey key: JSONKey) throws {
    let string = try container.decode(String.self, forKey: key)
    guard let value = T.allCases.first(where: { $0.commandLineAbbreviation == string }) else {
      throw
        "Invalid value for \(key.stringValue) (options: \(T.allCases.map(\.commandLineAbbreviation)))"
    }
    self.value = value
  }

  public func decode(fromFlagValue flagValue: FlagValue) {
    value = (flagValue as! CaseIterableEnum<T>).value
  }

  func description() -> String {
    var description = title
    if let explanationKey = explanationKey {
      description += " - \(LocalizedString.forKey(explanationKey))"
    }
    description += " (options: \(T.allCases.map(\.commandLineAbbreviation)))"
    return description
  }

  public func createFlag() -> Flag {
    return Flag(
      longName: commandLineFlag, type: CaseIterableEnum<T>.self, description: description())
  }
}

public class StringParameter: Parameter {
  public let title: String
  public let explanation: String?
  public let additionalJsonKeys: [String]
  public let commandLineFlag: String
  public var value: String?

  init(
    title: String, explanation: String?, defaultValue: String?, commandLineFlag: String,
    additionalJsonKeys: [String] = []
  ) {
    self.title = title
    self.explanation = explanation
    self.commandLineFlag = commandLineFlag
    self.additionalJsonKeys = additionalJsonKeys
    self.value = defaultValue
  }

  convenience init(
    titleKey: String, explanationKey: String?, defaultValue: String?, commandLineFlag: String,
    additionalJsonKeys: [String] = []
  ) {
    let explanation = explanationKey.map { LocalizedString.forKey($0) }
    self.init(
      title: LocalizedString.forKey(titleKey), explanation: explanation, defaultValue: defaultValue,
      commandLineFlag: commandLineFlag, additionalJsonKeys: additionalJsonKeys)
  }

  func description() -> String {
    var description = title
    if let explanation = explanation {
      description += " - \(explanation)"
    }
    return description
  }

  public func validate() throws {
  }

  public func createFlag() -> Flag {
    return Flag(longName: commandLineFlag, type: String.self, description: description())
  }

  public func encode(to container: inout KeyedEncodingContainer<JSONKey>, forKey key: JSONKey)
    throws
  {
    try container.encode(value, forKey: key)
  }

  public func decode(from container: KeyedDecodingContainer<JSONKey>, forKey key: JSONKey) throws {
    if container.contains(key) {
      value = try container.decode(String?.self, forKey: key)
    }
  }

  public func decode(fromFlagValue flagValue: FlagValue) {
    value = (flagValue as? String)!
  }
}

public class DoubleParameter: Parameter {
  public let title: String
  public let explanationKey: String?
  public let range: ClosedRange<Double>
  public let additionalJsonKeys: [String]
  public let commandLineFlag: String
  public var value: Double
  public let defaultValue: Double
  public var isOverridden: Bool = false

  init(
    titleKey: String, explanationKey: String?, defaultValue: Double, range: ClosedRange<Double>,
    commandLineFlag: String,
    additionalJsonKeys: [String] = []
  ) {
    self.title = LocalizedString.forKey(titleKey)
    self.explanationKey = explanationKey
    self.range = range
    self.value = defaultValue
    self.defaultValue = defaultValue
    self.commandLineFlag = commandLineFlag
    self.additionalJsonKeys = additionalJsonKeys
  }

  func description() -> String {
    var description = title
    if let explanationKey = explanationKey {
      description += " - \(LocalizedString.forKey(explanationKey))"
    }
    description +=
      " (default: \(defaultValue), range: \(range.lowerBound)-\(range.upperBound), inclusive)"
    return description
  }

  public func validate() throws {
    try Validation.validate(
      range.contains(value),
      errorMessage:
        "Value for \(title) must be between \(range.lowerBound) and \(range.upperBound), inclusive (was \(value))"
    )
  }

  func float32Value() -> Float32 {
    return Float32(value)
  }

  public func createFlag() -> Flag {
    return Flag(longName: commandLineFlag, type: Double.self, description: description())
  }

  public func encode(to container: inout KeyedEncodingContainer<JSONKey>, forKey key: JSONKey)
    throws
  {
    try container.encode(value, forKey: key)
  }

  public func decode(from container: KeyedDecodingContainer<JSONKey>, forKey key: JSONKey) throws {
    value = try container.decodeIfPresent(Double.self, forKey: key) ?? value
    isOverridden = true
  }

  public func decode(fromFlagValue flagValue: FlagValue) {
    value = (flagValue as? Double)!
    isOverridden = true
  }
}

class BoolParameter: Parameter {
  public let title: String
  public let explanationKey: String?
  public let additionalJsonKeys: [String]
  public let commandLineFlag: String
  public var value: Bool = false

  init(
    titleKey: String, explanationKey: String?, commandLineFlag: String,
    additionalJsonKeys: [String] = [], defaultValue: Bool
  ) {
    self.title = LocalizedString.forKey(titleKey)
    self.explanationKey = explanationKey
    self.commandLineFlag = commandLineFlag
    self.additionalJsonKeys = additionalJsonKeys
    self.value = defaultValue
  }

  func description() -> String {
    var description = title
    if let explanationKey = explanationKey {
      description += " - \(LocalizedString.forKey(explanationKey))"
    }
    return description
  }

  func validate() throws {
  }

  func createFlag() -> Flag {
    return Flag(longName: commandLineFlag, type: BoolFlagValue.self, description: description())
  }

  public func encode(to container: inout KeyedEncodingContainer<JSONKey>, forKey key: JSONKey)
    throws
  {
    try container.encode(value, forKey: key)
  }

  public func decode(from container: KeyedDecodingContainer<JSONKey>, forKey key: JSONKey) throws {
    if let decodedValue = try container.decodeIfPresent(Bool.self, forKey: key) {
      value = decodedValue
    }
  }

  func decode(fromFlagValue flagValue: FlagValue) {
    value = (flagValue as! BoolFlagValue).value
  }
}

public final class IntParameter: Parameter {
  public let title: String
  public let explanationKey: String?
  public let range: ClosedRange<Int>
  public let additionalJsonKeys: [String]
  public let commandLineFlag: String
  public var value: Int
  public let defaultValue: Int

  init(
    titleKey: String, explanationKey: String?, defaultValue: Int, range: ClosedRange<Int>,
    commandLineFlag: String,
    additionalJsonKeys: [String] = []
  ) {
    self.title = LocalizedString.forKey(titleKey)
    self.defaultValue = defaultValue
    self.value = defaultValue
    self.explanationKey = explanationKey
    self.range = range
    self.commandLineFlag = commandLineFlag
    self.additionalJsonKeys = additionalJsonKeys
  }

  func description() -> String {
    var description = title
    if let explanationKey = explanationKey {
      description += " - \(LocalizedString.forKey(explanationKey))"
    }
    description +=
      " (default: \(defaultValue), range: \(range.lowerBound)-\(range.upperBound), inclusive)"
    return description
  }

  public func validate() throws {
    try Validation.validate(
      range.contains(value),
      errorMessage:
        "Value for \(title) must be between \(range.lowerBound) and \(range.upperBound), inclusive (was \(value))"
    )
  }

  func uint32Value() -> UInt32 {
    return UInt32(value)
  }

  func int32Value() -> Int32 {
    return Int32(value)
  }

  public func createFlag() -> Flag {
    return Flag(longName: commandLineFlag, type: Int.self, description: description())
  }

  public func encode(to container: inout KeyedEncodingContainer<JSONKey>, forKey key: JSONKey)
    throws
  {
    try container.encode(value, forKey: key)
  }

  public func decode(from container: KeyedDecodingContainer<JSONKey>, forKey key: JSONKey) throws {
    value = try container.decodeIfPresent(Int.self, forKey: key) ?? value
  }

  public func decode(fromFlagValue flagValue: FlagValue) {
    value = (flagValue as? Int)!
  }
}

final class JSONParameter<T: Codable>: Parameter {

  public let title: String
  public let explanationKey: String?
  public let additionalJsonKeys: [String]
  public let commandLineFlag: String
  public var value: T
  public let defaultValue: T

  init(
    title: String, explanationKey: String?, defaultValue: T,
    commandLineFlag: String, additionalJsonKeys: [String] = []
  ) {
    self.title = title
    self.defaultValue = defaultValue
    self.value = defaultValue
    self.explanationKey = explanationKey
    self.commandLineFlag = commandLineFlag
    self.additionalJsonKeys = additionalJsonKeys
  }

  var description: String {
    var description = title
    if let explanationKey = explanationKey {
      description += " - \(LocalizedString.forKey(explanationKey))"
    }
    description +=
      " (default: \(defaultValue))"
    return description
  }

  func validate() throws {
    // No validation can be done for this.
  }

  func createFlag() -> Flag {
    return Flag(longName: commandLineFlag, type: String.self, description: description)
  }

  public func encode(to container: inout KeyedEncodingContainer<JSONKey>, forKey key: JSONKey)
    throws
  {
    try container.encode(value, forKey: key)
  }

  public func decode(from container: KeyedDecodingContainer<JSONKey>, forKey key: JSONKey) throws {
    guard let value = try container.decodeIfPresent(T.self, forKey: key) else { return }
    self.value = value
  }

  func decode(fromFlagValue flagValue: Guaka.FlagValue) {
    guard let flagValue = flagValue as? String else { return }
    guard let jsonData = flagValue.data(using: .utf8) else { return }
    let decoder = JSONDecoder()
    guard let value = try? decoder.decode(T.self, from: jsonData) else { return }
    self.value = value
  }

}
