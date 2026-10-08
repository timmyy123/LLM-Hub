import Fickling
import Foundation
import NNC
import ZIPFoundation

#if canImport(C_ccv)
  import C_ccv
#elseif canImport(C_swiftpm_ccv)
  import C_swiftpm_ccv
#endif

extension Archive: TensorArchive {
  public func with<T>(_ tensorDescriptor: TensorDescriptor, block: (AnyTensor) throws -> T) throws
    -> T
  {
    var flag = false
    var v = 1
    for i in stride(from: tensorDescriptor.shape.count - 1, through: 0, by: -1) {
      // Skip check if the shape is 1 in the beginning.
      if !flag && tensorDescriptor.shape[i] == 1 {
        continue
      }
      flag = true
      precondition(tensorDescriptor.strides[i] == v)
      v *= tensorDescriptor.shape[i]
    }
    guard let entry = (first { $0.path.hasSuffix("/data/\(tensorDescriptor.storage.name)") }) else {
      throw InflateError.tensorNotFound
    }
    if Interpreter.inflateInterrupter?() ?? false {
      throw InflateError.interrupted
    }
    var data = Data()
    let _ = try extract(entry) { data.append($0) }
    return try data.withUnsafeMutableBytes {
      guard let address = $0.baseAddress else { throw InflateError.dataNoBaseAddress }
      let tensor: AnyTensor
      #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
        if tensorDescriptor.storage.dataType == .Float16 {
          if tensorDescriptor.storage.BF16 {
            let count = tensorDescriptor.strides[0] * tensorDescriptor.shape[0]
            let u16 = UnsafeMutablePointer<UInt16>.allocate(capacity: count * 2)
            let bf16 = (address + tensorDescriptor.storageOffset).assumingMemoryBound(
              to: UInt16.self)
            for i in 0..<count {
              u16[i * 2] = 0
              u16[i * 2 + 1] = bf16[i]
            }
            tensor = Tensor<Float>(
              .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
              unsafeMutablePointer: UnsafeMutableRawPointer(u16).assumingMemoryBound(
                to: Float.self), bindLifetimeOf: self
            ).copied()
            u16.deallocate()
          } else {
            tensor = Tensor<Float16>(
              .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
              unsafeMutablePointer: (address + tensorDescriptor.storageOffset).assumingMemoryBound(
                to: Float16.self),
              bindLifetimeOf: entry
            )
          }
        } else if tensorDescriptor.storage.dataType == .Float64 {
          tensor = Tensor<Double>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: (address + tensorDescriptor.storageOffset).assumingMemoryBound(
              to: Double.self),
            bindLifetimeOf: entry
          )
        } else {
          tensor = Tensor<Float>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: (address + tensorDescriptor.storageOffset).assumingMemoryBound(
              to: Float.self), bindLifetimeOf: entry
          )
        }
      #else
        if tensorDescriptor.storage.dataType == .Float16 {
          let count = tensorDescriptor.strides[0] * tensorDescriptor.shape[0]
          let u16 = UnsafeMutablePointer<UInt16>.allocate(capacity: count * 2)
          let f16 = (address + tensorDescriptor.storageOffset).assumingMemoryBound(
            to: UInt16.self)
          if tensorDescriptor.storage.BF16 {
            for i in 0..<count {
              u16[i * 2] = 0
              u16[i * 2 + 1] = f16[i]
            }
          } else {
            ccv_half_precision_to_float(
              f16, UnsafeMutableRawPointer(u16).assumingMemoryBound(to: Float.self), count)
          }
          tensor = Tensor<Float>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: UnsafeMutableRawPointer(u16).assumingMemoryBound(
              to: Float.self), bindLifetimeOf: self
          ).copied()
          u16.deallocate()
        } else if tensorDescriptor.storage.dataType == .Float64 {
          tensor = Tensor<Double>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: (address + tensorDescriptor.storageOffset).assumingMemoryBound(
              to: Double.self),
            bindLifetimeOf: entry
          )
        } else {
          tensor = Tensor<Float>(
            .CPU, format: .NCHW, shape: TensorShape(tensorDescriptor.shape),
            unsafeMutablePointer: (address + tensorDescriptor.storageOffset).assumingMemoryBound(
              to: Float.self), bindLifetimeOf: entry
          )
        }
      #endif
      return try block(tensor)
    }
  }
}
