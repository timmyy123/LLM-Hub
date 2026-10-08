public func unwrapOrThrow<T>(_ object: T?, errorMessage: String) throws
  -> T
{
  if let object = object {
    return object
  } else {
    throw errorMessage
  }
}

public enum Validation {
  public static func validate(_ condition: Bool, errorMessage: String) throws {
    if !condition {
      throw errorMessage
    }
  }
}
