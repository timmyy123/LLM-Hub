import DataModels
import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
import Upscaler
import ZIPFoundation

public final class UpscalerImporter {
  public enum Error: Swift.Error {
    case tensorWritesFailed
    case parsingFailed
  }
  final class DataArchive: TensorDataArchive {
    let data: Data
    let bufferStart: Int
    init(data: Data, bufferStart: Int) {
      self.data = data
      self.bufferStart = bufferStart
    }
  }
  public let name: String
  private let filePath: String
  public init(name: String, filePath: String) {
    self.name = name
    self.filePath = filePath
  }
  public func `import`(
    progress: @escaping (Float) -> Void
  ) throws -> (file: String, numberOfBlocks: Int, upscaleFactor: UpscaleFactor) {
    let archive: TensorArchive
    var stateDict: [String: TensorDescriptor]
    if let safeTensors = SafeTensors(url: URL(fileURLWithPath: filePath)) {
      archive = safeTensors
      let states = safeTensors.states
      stateDict = states
    } else if let zipArchive = Archive(url: URL(fileURLWithPath: filePath), accessMode: .read) {
      archive = zipArchive
      let rootObject = try Interpreter.unpickle(zip: zipArchive)
      let originalStateDict =
        rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["params_ema"]
        as? Interpreter.Dictionary ?? rootObject["params"] as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        stateDict[key] = value
      }
    } else if let data = try? Data(
      contentsOf: URL(fileURLWithPath: filePath), options: .mappedIfSafe)
    {
      archive = DataArchive(data: data, bufferStart: 0)
      let rootObject = try Interpreter.unpickle(data: data, fileReadDirectly: true)
      let originalStateDict =
        rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["params_ema"]
        as? Interpreter.Dictionary ?? rootObject["params"] as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        stateDict[key] = value
      }
    } else {
      throw UnpickleError.dataNotFound
    }
    let keys = Array(stateDict.keys)
    let numberOfBlocks = {
      for i in stride(from: 22, through: 0, by: -1) {
        if keys.contains(where: {
          $0.hasPrefix("body.\(i)") || $0.hasPrefix("model.1.sub.\(i)")
        }) {
          return i + 1
        }
      }
      return 0
    }()
    guard numberOfBlocks > 0 else {
      throw Error.parsingFailed
    }
    let (mapper, rrdbnet) = RRDBNet(
      numberOfOutputChannels: 3, numberOfFeatures: 64, numberOfBlocks: numberOfBlocks,
      numberOfGrowChannels: 32)
    let graph = DynamicGraph()
    let inputChannels: Int
    if let descriptor = stateDict["conv_first.weight"] ?? stateDict["model.0.weight"],
      descriptor.shape.count == 4
    {
      inputChannels = descriptor.shape[1]
    } else {
      inputChannels = 3
    }
    let upscaleFactor: UpscaleFactor
    let z = graph.variable(.GPU(0), .NHWC(1, 128, 128, inputChannels), of: FloatType.self)
    rrdbnet.compile(inputs: z)
    let modelKey: String
    if inputChannels == 12 {
      upscaleFactor = .x2
    } else {
      upscaleFactor = .x4
    }
    if numberOfBlocks == 6 {
      modelKey = "realesrgan_x4plus_6b"
    } else if upscaleFactor == .x2 {
      modelKey = "realesrgan_x2plus"
    } else {
      modelKey = "realesrgan_x4plus"
    }
    let mapping = mapper()
    let filePath = UpscalerZoo.filePathForModelDownloaded(name)
    try graph.openStore(filePath, flags: .truncateWhenClose) { store in
      store.removeAll()
      for (key, values) in mapping {
        guard let descriptor = stateDict[key] else { continue }
        try archive.with(descriptor) {
          let tensor = Tensor<FloatType>(from: $0)
          if values.count > 1 {
            let shape = tensor.shape
            if shape.count == 4 && shape[1] == 64 + (values.count - 1) * 32 {
              store.write(
                "__\(modelKey)__[\(values[0])]",
                tensor: tensor[0..<shape[0], 0..<64, 0..<shape[2], 0..<shape[3]].contiguous())
              for i in 1..<values.count {
                store.write(
                  "__\(modelKey)__[\(values[i])]",
                  tensor: tensor[
                    0..<shape[0], (64 + (i - 1) * 32)..<(64 + i * 32), 0..<shape[2], 0..<shape[3]
                  ].contiguous())
              }
            } else {
              store.write("__\(modelKey)__[\(values[0])]", tensor: tensor)
            }
          } else if let value = values.first {
            store.write("__\(modelKey)__[\(value)]", tensor: tensor)
          }
        }
      }
      if store.keys.count != 12 + 60 * numberOfBlocks {
        throw Error.tensorWritesFailed
      }
    }
    return (name, numberOfBlocks, upscaleFactor)
  }
}
