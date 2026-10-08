import Foundation

public struct JSONKey: CodingKey {
  public let intValue: Int? = nil
  public let stringValue: String

  public init(_ stringValue: String) {
    self.init(stringValue: stringValue)!
  }

  public init?(stringValue: String) {
    self.stringValue = stringValue
  }

  public init?(intValue: Int) {
    fatalError()
  }
}
