import Guaka
import Utils

extension Double: FlagValue {
  public static var typeDescription: String {
    return "Double"
  }

  public static func fromString(flagValue value: String) throws -> Double {
    return try unwrapOrThrow(Double(value), errorMessage: "Can't convert \(value) to double")
  }
}

public final class BoolFlagValue: FlagValue {
  public let value: Bool

  init(value: Bool) {
    self.value = value
  }

  public static func fromString(flagValue value: String) throws -> Self {
    if value == "true" {
      return Self.init(value: true)
    } else if value == "false" {
      return Self.init(value: false)
    } else {
      throw FlagValueError.conversionError(
        "Must pass either \"true\" or \"false\" for boolean argument")
    }
  }

  public static var typeDescription: String {
    return "\"true\", \"false\""
  }
}
