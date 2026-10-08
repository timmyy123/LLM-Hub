public protocol DefaultCreatable {
  static func defaultInstance() -> Self
}

extension Dictionary: DefaultCreatable {
  public static func defaultInstance() -> [Key: Value] {
    return [:]
  }
}

extension Array: DefaultCreatable {
  public static func defaultInstance() -> [Element] {
    return []
  }
}

extension Int: DefaultCreatable {
  public static func defaultInstance() -> Int {
    return 0
  }
}

extension String: DefaultCreatable {
  public static func defaultInstance() -> String {
    return ""
  }
}

extension Optional: DefaultCreatable {
  public static func defaultInstance() -> Wrapped? {
    return nil
  }
}
