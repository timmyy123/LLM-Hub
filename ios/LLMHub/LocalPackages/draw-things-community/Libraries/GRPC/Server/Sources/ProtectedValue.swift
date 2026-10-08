import NIOConcurrencyHelpers

public struct ProtectedValue<T> {
  private var value: T
  private let lock: NIOLock
  public init(_ value: T) {
    self.value = value
    lock = NIOLock()
  }
  public mutating func modify(_ body: (inout T) throws -> Void) rethrows {
    lock.lock()
    defer { lock.unlock() }
    try body(&value)
  }
}
