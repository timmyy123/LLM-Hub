import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
import ZIPFoundation

public enum EmbeddingImporter {
  public static func `import`(
    downloadedFile: String, name: String, filename: String, keyword: String
  ) throws -> (multivector: Int, version: ModelVersion) {
    let filePath =
      downloadedFile.starts(with: "/")
      ? downloadedFile : ModelZoo.filePathForDownloadedFile(downloadedFile)
    let archive: TensorArchive
    var stateDict: [String: TensorDescriptor]
    if let safeTensors = SafeTensors(url: URL(fileURLWithPath: filePath)) {
      archive = safeTensors
      let states = safeTensors.states
      stateDict = states
    } else if let zipArchive = Archive(url: URL(fileURLWithPath: filePath), accessMode: .read) {
      archive = zipArchive
      let rootObject = try Interpreter.unpickle(zip: zipArchive)
      let originalStateDict = rootObject["string_to_param"] as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      if let parametersDict = originalStateDict["_parameters"] as? Interpreter.Dictionary {
        parametersDict.forEach { key, value in
          guard let value = value as? TensorDescriptor else { return }
          stateDict[key] = value
        }
      }
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        stateDict[key] = value
      }
    } else {
      throw UnpickleError.dataNotFound
    }
    let multivector: Int
    let modelVersion: ModelVersion
    if let tensorDesc = stateDict["*"],
      tensorDesc.shape.count >= 1 && tensorDesc.shape.count <= 2
        && (tensorDesc.shape[tensorDesc.shape.count - 1] == 768
          || tensorDesc.shape[tensorDesc.shape.count - 1] == 1024)
    {
      multivector = tensorDesc.shape.count > 1 ? tensorDesc.shape[0] : 1
      modelVersion = tensorDesc.shape[tensorDesc.shape.count - 1] == 768 ? .v1 : .v2
      try archive.with(tensorDesc) {
        let tensor = Tensor<FloatType>(from: $0)
        let graph = DynamicGraph()
        graph.openStore(
          TextualInversionZoo.filePathForModelDownloaded(filename), flags: [.truncateWhenClose]
        ) {
          $0.removeAll()
          $0.write("string_to_param", tensor: tensor)
        }
      }
    } else if let tensorClipGDesc = stateDict["clip_g"], let tensorClipLDesc = stateDict["clip_l"],
      tensorClipGDesc.shape[tensorClipGDesc.shape.count - 1] == 1280
        && tensorClipLDesc.shape[tensorClipLDesc.shape.count - 1] == 768
    {
      let multivectorClipG = tensorClipGDesc.shape.count > 1 ? tensorClipGDesc.shape[0] : 1
      let multivectorClipL = tensorClipLDesc.shape.count > 1 ? tensorClipLDesc.shape[0] : 1
      guard multivectorClipG == multivectorClipL else {
        throw UnpickleError.tensorNotFound
      }
      multivector = multivectorClipG
      modelVersion = .sdxlBase
      let graph = DynamicGraph()
      try graph.openStore(
        TextualInversionZoo.filePathForModelDownloaded(filename), flags: [.truncateWhenClose]
      ) { store in
        store.removeAll()
        try archive.with(tensorClipGDesc) {
          let tensorClipG = Tensor<FloatType>(from: $0)
          store.write("string_to_param_clip_g", tensor: tensorClipG)
        }
        try archive.with(tensorClipLDesc) {
          let tensorClipL = Tensor<FloatType>(from: $0)
          store.write("string_to_param_clip_l", tensor: tensorClipL)
        }
      }
    } else {
      var tensorDesc: TensorDescriptor? = nil
      for value in stateDict.values {
        if value.shape.count >= 1 && value.shape.count <= 2
          && (value.shape[value.shape.count - 1] == 768
            || value.shape[value.shape.count - 1] == 1024)
          && (value.shape.count == 1 || value.shape[0] < 77)
        {
          tensorDesc = value
        }
      }
      guard let tensorDesc = tensorDesc else {
        throw UnpickleError.tensorNotFound
      }
      multivector = tensorDesc.shape.count > 1 ? tensorDesc.shape[0] : 1
      modelVersion = tensorDesc.shape[tensorDesc.shape.count - 1] == 768 ? .v1 : .v2
      try archive.with(tensorDesc) {
        let tensor = Tensor<FloatType>(from: $0)
        let graph = DynamicGraph()
        graph.openStore(
          TextualInversionZoo.filePathForModelDownloaded(filename), flags: [.truncateWhenClose]
        ) {
          $0.removeAll()
          $0.write("string_to_param", tensor: tensor)
        }
      }
    }
    return (multivector, modelVersion)
  }
}
