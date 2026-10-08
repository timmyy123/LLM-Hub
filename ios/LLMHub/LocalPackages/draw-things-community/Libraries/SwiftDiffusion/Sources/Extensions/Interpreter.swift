import DiffusionMappings
import Fickling
import Foundation
import ZIPFoundation

public typealias PythonReader = ([String: TensorDescriptor], TensorArchive) throws -> Void

public enum UnpickleError: Error {
  case dataNotFound
  case tensorNotFound
  case noRootObject
}

extension Interpreter {
  public static var inflateInterrupter: (() -> Bool)? = nil
  public static func unpickle(zip archive: Archive) throws -> Interpreter.Dictionary {
    guard let entry = (archive.first { $0.path.hasSuffix("/data.pkl") }) else {
      throw UnpickleError.dataNotFound
    }
    var data = Data()
    let _ = try archive.extract(entry) { data.append($0) }
    return try unpickle(data: data, fileReadDirectly: false)
  }
  public static func unpickle(data: Data, fileReadDirectly: Bool) throws -> Interpreter.Dictionary {
    let interpreter = Interpreter.from(data: data)
    interpreter.intercept(module: "UNPICKLER", function: "persistent_load") {
      module, function, args in
      guard args.count >= 5, let global = args[1] as? Interpreter.GlobalObject,
        let name = args[2] as? String, let size = args[4] as? Int
      else { return [nil] }
      guard
        global.function == "HalfStorage" || global.function == "FloatStorage"
          || global.function == "DoubleStorage" || global.function == "BFloat16Storage"
      else {
        return [nil]
      }
      let storage: Storage
      if global.function == "HalfStorage" {
        storage = Storage(
          name: name, size: size, dataType: .Float16, BF16: false, FP8_E4M3: false, FP8_E5M2: false)
      } else if global.function == "BFloat16Storage" {
        storage = Storage(
          name: name, size: size, dataType: .Float16, BF16: true, FP8_E4M3: false, FP8_E5M2: false)
      } else if global.function == "DoubleStorage" {
        storage = Storage(
          name: name, size: size, dataType: .Float64, BF16: false, FP8_E4M3: false, FP8_E5M2: false)
      } else {
        storage = Storage(
          name: name, size: size, dataType: .Float32, BF16: false, FP8_E4M3: false, FP8_E5M2: false)
      }
      return [storage]
    }
    interpreter.intercept(module: "torch.nn.modules.container", function: "ParameterDict") {
      module, function, _ in
      return [Interpreter.Dictionary(.unordered)]
    }
    interpreter.intercept(module: "torch._utils", function: "_rebuild_tensor_v2") {
      module, function, args in
      guard args.count >= 5, let storage = args[0] as? Storage, let storageOffset = args[1] as? Int,
        let shape = args[2] as? [Int],
        let strides = args[3] as? [Int]
      else { return [nil] }
      let storeageOffsetInBytes: Int
      switch storage.dataType {
      case .Float16, .BFloat16:
        storeageOffsetInBytes = storageOffset * 2
      case .Float32:
        storeageOffsetInBytes = storageOffset * 4
      case .Float64:
        storeageOffsetInBytes = storageOffset * 8
      case .UInt8, .Int32, .Int64:
        fatalError()
      }
      let tensorDescriptor = TensorDescriptor(
        storage: storage, storageOffset: storeageOffsetInBytes, shape: shape, strides: strides)
      return [tensorDescriptor]
    }
    interpreter.intercept(module: "torch._utils", function: "_rebuild_parameter") { _, _, args in
      guard let tensorDescriptor = args.first as? TensorDescriptor else { return [nil] }
      return [tensorDescriptor]
    }
    guard fileReadDirectly else {
      // This uses the other containers (zip or tar), so we can read all opcode and do this.
      while try interpreter.step() {}
      guard let rootObject = (interpreter.rootObject as? Interpreter.Dictionary) else {
        // Check if it is old format, which contains dictionary and array.
        throw UnpickleError.noRootObject
      }
      return rootObject
    }
    // This is the legacy implementation, it will first read magic number, protocol version, sys info, and then dictionary and key order.
    // Read more in https://github.com/pytorch/pytorch/blob/v2.6.0/torch/serialization.py#L1744
    let stopOffset = interpreter.stopOffset
    while try interpreter.step(onStop: true) {}
    let _ = interpreter.pop()
    while try interpreter.step(onStop: true) {}
    let _ = interpreter.pop()
    while try interpreter.step(onStop: true) {}
    let _ = interpreter.pop()
    while try interpreter.step(onStop: true) {}
    guard let rootObject = interpreter.pop() as? Interpreter.Dictionary else {
      throw UnpickleError.noRootObject
    }
    while try interpreter.step(onStop: true) {}
    guard
      let deserializationKeys =
        ((interpreter.rootObject as? Interpreter.Array).flatMap { $0.array as? [String] })
    else { throw UnpickleError.noRootObject }
    var storageKeys = [String: String]()
    rootObject.forEach { key, value in
      guard let descriptor = value as? TensorDescriptor else { return }
      storageKeys[descriptor.storage.name] = key
    }
    var offset = stopOffset
    for key in deserializationKeys {
      guard let name = storageKeys[key], var descriptor = rootObject[name] as? TensorDescriptor
      else { continue }
      let size =
        offset < data.count
        ? (data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: Int64.self) }) : 0
      descriptor.storageOffset = offset + MemoryLayout<Int64>.size
      let elementSize = {
        if descriptor.storage.FP8_E4M3 || descriptor.storage.FP8_E5M2 {
          return 1
        }
        switch descriptor.storage.dataType {
        case .Float16, .BFloat16:
          return 2
        case .Float32, .Int32:
          return 4
        case .Float64, .Int64:
          return 8
        case .UInt8:
          return 1
        }
      }()
      offset += MemoryLayout<Int64>.size + Int(size) * elementSize
      rootObject[name] = descriptor
    }
    return rootObject
  }
}
