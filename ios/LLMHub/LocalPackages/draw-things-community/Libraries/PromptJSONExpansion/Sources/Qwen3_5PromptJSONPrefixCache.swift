import CryptoKit
import Foundation
import LLM
import NNC

// This intentionally caches only the chunk-aligned Ideogram 4 system-prompt prefix. Unlike
// Local Code's context cache, it does not manage arbitrary prompts or incremental checkpoints.
final class Qwen3_5PromptJSONPrefixCache<
  FloatType: TensorNumeric & BinaryFloatingPoint
>: Qwen3_5TextGenerationPrefixCache {
  let tokenIds: [Int32]
  let fingerprint: String

  private let cacheDirectory: URL
  private let storePath: String

  init?(modelFilePath: String, tokenIds: [Int32], cacheDirectoryURL: URL? = nil) {
    guard !tokenIds.isEmpty else { return nil }
    let cacheDirectory: URL
    if let requestedCacheDirectory = cacheDirectoryURL {
      cacheDirectory = requestedCacheDirectory
    } else {
      guard
        let cachesDirectory = FileManager.default.urls(
          for: .cachesDirectory, in: .userDomainMask
        ).first
      else { return nil }
      cacheDirectory = cachesDirectory.appendingPathComponent("qwen3_5", isDirectory: true)
    }
    guard
      (try? FileManager.default.createDirectory(
        at: cacheDirectory, withIntermediateDirectories: true)) != nil
    else { return nil }
    self.tokenIds = tokenIds
    self.fingerprint = Self.fingerprint(modelFilePath: modelFilePath, tokenIds: tokenIds)
    self.cacheDirectory = cacheDirectory
    self.storePath = cacheDirectory.appendingPathComponent("context_cache.sqlite3").path
  }

  static func fingerprint(modelFilePath: String, tokenIds: [Int32]) -> String {
    var sha256 = SHA256()
    let modelFileName = URL(fileURLWithPath: modelFilePath).lastPathComponent
    sha256.update(data: Data(modelFileName.utf8))
    var data = Data()
    data.reserveCapacity(tokenIds.count * MemoryLayout<Int32>.size)
    for token in tokenIds {
      var littleEndianToken = token.littleEndian
      withUnsafeBytes(of: &littleEndianToken) { bytes in
        data.append(contentsOf: bytes)
      }
    }
    sha256.update(data: data)
    return sha256.finalize().map { String(format: "%02x", $0) }.joined()
  }

  func restore(
    graph: DynamicGraph, configuration: Qwen3_5ModelConfiguration,
    caches: inout [DynamicGraph.AnyTensor]
  ) -> Bool {
    guard caches.count == configuration.layers * 2,
      FileManager.default.fileExists(atPath: storePath)
    else { return false }
    let result = graph.openStore(storePath, flags: .readOnly) { store -> Bool in
      var fullAttentionStates = [
        (cursor: Int, key: Tensor<FloatType>, value: Tensor<FloatType>)
      ]()
      var linearAttentionStates = [
        (cursor: Int, conv: Tensor<FloatType>, recurrent: Tensor<Float>)
      ]()
      var cursor = 0
      for layerIndex in 0..<configuration.layers {
        if configuration.isLinearAttentionLayer(layerIndex) {
          let convShape = [
            1, configuration.linearConvKernel - 1, 1, configuration.linearConvDim,
          ]
          let recurrentShape = [
            1, configuration.linearNumValueHeads, configuration.linearValueHeadDim,
            configuration.linearKeyHeadDim,
          ]
          guard Array(caches[cursor].shape) == convShape,
            Array(caches[cursor + 1].shape) == recurrentShape,
            let convStored = store.read(
              "\(fingerprint)/linear/\(layerIndex)/conv", kind: .CPU),
            Array(convStored.shape) == convShape,
            let recurrentStored = store.read(
              "\(fingerprint)/linear/\(layerIndex)/recurrent", kind: .CPU),
            Array(recurrentStored.shape) == recurrentShape
          else { return false }
          linearAttentionStates.append(
            (
              cursor: cursor, conv: Tensor<FloatType>(from: convStored),
              recurrent: Tensor<Float>(from: recurrentStored)
            ))
        } else {
          let storedShape = [
            1, tokenIds.count, configuration.keyValueHeads, configuration.attentionHeadDim,
          ]
          let cacheShape = Array(caches[cursor].shape)
          guard cacheShape.count == 4, cacheShape[0] == 1,
            cacheShape[1] >= tokenIds.count,
            cacheShape[2] == configuration.keyValueHeads,
            cacheShape[3] == configuration.attentionHeadDim,
            Array(caches[cursor + 1].shape) == cacheShape,
            let keyStored = store.read("\(fingerprint)/kv/\(layerIndex)/k", kind: .CPU),
            Array(keyStored.shape) == storedShape,
            let valueStored = store.read("\(fingerprint)/kv/\(layerIndex)/v", kind: .CPU),
            Array(valueStored.shape) == storedShape
          else { return false }
          fullAttentionStates.append(
            (
              cursor: cursor, key: Tensor<FloatType>(from: keyStored),
              value: Tensor<FloatType>(from: valueStored)
            ))
        }
        cursor += 2
      }
      for state in fullAttentionStates {
        let key = graph.variable(state.key.toGPU(0))
        let value = graph.variable(state.value.toGPU(0))
        var keyCache = caches[state.cursor].as(of: FloatType.self)
        var valueCache = caches[state.cursor + 1].as(of: FloatType.self)
        keyCache[
          0..<1, 0..<tokenIds.count, 0..<configuration.keyValueHeads,
          0..<configuration.attentionHeadDim
        ] = key
        valueCache[
          0..<1, 0..<tokenIds.count, 0..<configuration.keyValueHeads,
          0..<configuration.attentionHeadDim
        ] = value
      }
      for state in linearAttentionStates {
        var convCache = caches[state.cursor].as(of: FloatType.self)
        var recurrentCache = caches[state.cursor + 1].as(of: Float.self)
        convCache[
          0..<1, 0..<configuration.linearConvKernel - 1, 0..<1,
          0..<configuration.linearConvDim
        ] = graph.variable(state.conv.toGPU(0))
        recurrentCache[
          0..<1, 0..<configuration.linearNumValueHeads,
          0..<configuration.linearValueHeadDim, 0..<configuration.linearKeyHeadDim
        ] = graph.variable(state.recurrent.toGPU(0))
      }
      return true
    }
    switch result {
    case .success(let restored):
      return restored
    case .failure:
      return false
    }
  }

  func save(
    graph: DynamicGraph, configuration: Qwen3_5ModelConfiguration,
    caches: [DynamicGraph.AnyTensor]
  ) {
    guard caches.count == configuration.layers * 2 else { return }
    guard
      (try? FileManager.default.createDirectory(
        at: cacheDirectory, withIntermediateDirectories: true)) != nil
    else { return }
    for suffix in ["", "-shm", "-wal", "-journal"] {
      let path = storePath + suffix
      if FileManager.default.fileExists(atPath: path) {
        try? FileManager.default.removeItem(atPath: path)
      }
    }
    _ = graph.openStore(storePath, flags: []) { store in
      store.withTransaction {
        var cursor = 0
        for layerIndex in 0..<configuration.layers {
          if configuration.isLinearAttentionLayer(layerIndex) {
            store.write(
              "\(fingerprint)/linear/\(layerIndex)/conv",
              tensor: caches[cursor].as(of: FloatType.self).rawValue.toCPU())
            store.write(
              "\(fingerprint)/linear/\(layerIndex)/recurrent",
              tensor: caches[cursor + 1].as(of: Float.self).rawValue.toCPU())
          } else {
            let keyCache = caches[cursor].as(of: FloatType.self)
            let valueCache = caches[cursor + 1].as(of: FloatType.self)
            let key = keyCache.reshaped(
              .NHWC(
                1, tokenIds.count, configuration.keyValueHeads,
                configuration.attentionHeadDim),
              offset: [0, 0, 0, 0],
              strides: [
                keyCache.shape[1] * configuration.keyValueHeads
                  * configuration.attentionHeadDim,
                configuration.keyValueHeads * configuration.attentionHeadDim,
                configuration.attentionHeadDim, 1,
              ]
            ).copied()
            let value = valueCache.reshaped(
              .NHWC(
                1, tokenIds.count, configuration.keyValueHeads,
                configuration.attentionHeadDim),
              offset: [0, 0, 0, 0],
              strides: [
                valueCache.shape[1] * configuration.keyValueHeads
                  * configuration.attentionHeadDim,
                configuration.keyValueHeads * configuration.attentionHeadDim,
                configuration.attentionHeadDim, 1,
              ]
            ).copied()
            store.write("\(fingerprint)/kv/\(layerIndex)/k", tensor: key.rawValue.toCPU())
            store.write("\(fingerprint)/kv/\(layerIndex)/v", tensor: value.rawValue.toCPU())
          }
          cursor += 2
        }
      }
    }
  }
}
