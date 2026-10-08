import Foundation
import NNC

public enum TensorData {
  public static func externalStore(filePath: String) -> String {
    return filePath.appending("-tensordata")
  }

  public static func externalStoreExists(filePath: String) -> Bool {
    let externalStoreFilePath = externalStore(filePath: filePath)
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: externalStoreFilePath) else { return false }
    // Now we know external data file path exists, but we want to check if majority of data is inside the external data file path now.
    guard
      let fileSize = (try? URL(fileURLWithPath: filePath).resourceValues(forKeys: [.fileSizeKey]))?
        .fileSize,
      let externalFileSize =
        (try? URL(fileURLWithPath: externalStoreFilePath).resourceValues(forKeys: [.fileSizeKey]))?
        .fileSize
    else { return false }
    return fileSize < externalFileSize
  }

  public static func makeCompactStore(
    for filePath: String, graph: DynamicGraph, progress: ((Double) -> Void)? = nil
  ) {
    // Only move it back if the external store exists.
    guard TensorData.externalStoreExists(filePath: filePath) else { return }
    // Trailer store is read-only.
    guard !DynamicGraph.Store.isTrailerStore(filePath) else { return }
    let externalStoreFilePath = TensorData.externalStore(filePath: filePath)
    let fileManager = FileManager.default
    guard
      let externalFileSize =
        (try? URL(fileURLWithPath: externalStoreFilePath).resourceValues(forKeys: [.fileSizeKey]))?
        .fileSize, externalFileSize > 0
    else {
      try? fileManager.removeItem(atPath: externalStoreFilePath)
      return
    }
    // We need to open it and move back to the main storage.
    do {
      try graph.openStore(filePath, externalStore: externalStoreFilePath) { store in
        let keys = store.keys
        try store.withTransaction {
          let total = keys.count
          for (i, key) in keys.enumerated() {
            guard var codec = store.codec(for: key) else { continue }
            // Only keep the other attributes for codec.
            guard codec.contains(.externalData) || codec.contains(.externalOnDemand) else {
              continue
            }
            codec.subtract([.externalData, .jit, .externalOnDemand])
            guard
              let tensor = store.read(key, kind: .CPU, codec: codec.union([.jit, .externalData]))
            else {
              continue
            }
            // Move this tensor to main SQLite file.
            try store.write(key, tensor: tensor, strict: true, codec: codec)
            progress?(Double(i + 1) / Double(total))
          }
        }
        store.vacuum()
      }
      try fileManager.removeItem(atPath: externalStoreFilePath)
    } catch {
      // There are some errors, at least the external store is not deleted.
    }
  }

  public static func makeExternalData(
    for filePath: String, graph: DynamicGraph, progress: ((Double) -> Void)? = nil
  ) {
    // Before load, we move most of the tensors to -tensordata file if it doesn't exist yet.
    guard !TensorData.externalStoreExists(filePath: filePath) else { return }
    // Trailer store is read-only.
    guard !DynamicGraph.Store.isTrailerStore(filePath) else { return }
    // First, checking the externalStore, if the file has size, we need to move data back into the database prior to move it out (we cannot write to the externalStore while reading from it).
    let externalStoreFilePath = TensorData.externalStore(filePath: filePath)
    if let externalFileSize =
      (try? URL(fileURLWithPath: externalStoreFilePath).resourceValues(forKeys: [.fileSizeKey]))?
      .fileSize, externalFileSize > 0
    {
      // We need to open it and move back to the main storage first.
      do {
        try graph.openStore(filePath, externalStore: externalStoreFilePath) { store in
          let keys = store.keys
          try store.withTransaction {
            for key in keys {
              guard var codec = store.codec(for: key) else { continue }
              // Only keep the other attributes for codec.
              guard codec.contains(.externalData) || codec.contains(.externalOnDemand) else {
                continue
              }
              codec.subtract([.externalData, .jit, .externalOnDemand])
              guard
                let tensor = store.read(key, kind: .CPU, codec: codec.union([.jit, .externalData]))
              else {
                continue
              }
              // Move this tensor to main SQLite file.
              try store.write(key, tensor: tensor, strict: true, codec: codec)
            }
          }
          // No need to vacuum as we will move it out momentarily.
        }
      } catch {
        // If we cannot write tensors back to the SQLite file, there is nothing we can do.
        return
      }
    }
    graph.openStore(filePath, externalStore: externalStoreFilePath) { store in
      let keys = store.keys
      do {
        // Move ada_ln related weights to the beginning.
        func first(_ key: String) -> Bool {
          return key.contains("ada_ln")
        }
        try store.withTransaction {
          let total = keys.count
          var i = 0
          // First move the ada_ln related weights (mostly used by UNetFixed).
          for key in keys {
            guard first(key) else { continue }
            i += 1
            guard var codec = store.codec(for: key) else { continue }
            // Only keep the other attributes for codec.
            codec.subtract([.externalData, .jit, .externalOnDemand])
            guard let tensor = store.read(key, kind: .CPU, codec: codec.union([.jit])) else {
              continue
            }
            let shape = tensor.shape
            // Now, check if we want ot move it to external storage. We check the shape of the tensor.
            let squeezedDims = shape.reduce(0) { return $0 + ($1 > 1 ? 1 : 0) }  // Check how many axis this tensor has.
            guard squeezedDims > 1 else { continue }
            // Move this tensor to external storage.
            try store.write(key, tensor: tensor, strict: true, codec: codec.union([.externalData]))
            progress?(Double(i + 1) / Double(total))
          }
          // Then move the other weights (mostly used by UNet).
          for key in keys {
            guard !first(key) else { continue }
            i += 1
            guard var codec = store.codec(for: key) else { continue }
            // Only keep the other attributes for codec.
            codec.subtract([.externalData, .jit, .externalOnDemand])
            guard let tensor = store.read(key, kind: .CPU, codec: codec.union([.jit])) else {
              continue
            }
            let shape = tensor.shape
            // Now, check if we want ot move it to external storage. We check the shape of the tensor.
            let squeezedDims = shape.reduce(0) { return $0 + ($1 > 1 ? 1 : 0) }  // Check how many axis this tensor has.
            guard squeezedDims > 1 else { continue }
            // Move this tensor to external storage.
            try store.write(key, tensor: tensor, strict: true, codec: codec.union([.externalData]))
            progress?(Double(i + 1) / Double(total))
          }
        }
        store.vacuum()
      } catch {
        // This is failed, we need to figure out what to do.
      }
    }
  }
}
