import NNC

public protocol TensorArchive {
  func with<T>(_: TensorDescriptor, block: (AnyTensor) throws -> T) throws -> T
}

extension TensorArchive {
  public func with<T>(_ tensorDescriptors: [TensorDescriptor], block: ([AnyTensor]) throws -> T)
    throws -> T
  {
    guard tensorDescriptors.count > 1 else {
      guard let first = tensorDescriptors.first else {
        return try block([])
      }
      return try with(first) {
        return try block([$0])
      }
    }
    func unzip(
      _ array: [TensorDescriptor], index: Int, tensors: inout [AnyTensor],
      block: ([AnyTensor]) throws -> T
    ) throws -> T {
      guard index < array.count else {
        return try block(tensors)
      }
      return try with(array[index]) {
        tensors.append($0)
        return try unzip(array, index: index + 1, tensors: &tensors, block: block)
      }
    }
    var tensors = [AnyTensor]()
    return try unzip(tensorDescriptors, index: 0, tensors: &tensors, block: block)
  }
}
