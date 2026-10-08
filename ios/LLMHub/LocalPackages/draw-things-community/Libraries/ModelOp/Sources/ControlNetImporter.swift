import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
import ZIPFoundation

public final class ControlNetImporter {
  private var access: Int = 0
  private var expectedTotalAccess: Int = 0
  private var progress: ((Float) -> Void)? = nil

  public init() {}

  private func interrupt() -> Bool {
    access += 1
    progress?(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
    return false
  }

  private static func numberOfTransformBlocks(
    prefix: String, attn: String, upTo: Int, keys: [String]
  ) -> Int {
    for i in 0..<upTo {
      let suffix = "\(prefix).transformer_blocks.\(i).\(attn).to_k.weight"
      if !keys.contains(where: { $0.hasSuffix(suffix) }) {
        return i
      }
    }
    return upTo
  }

  private static func numberOfSingleTransformerBlocks(
    prefix: String, attn: String, upTo: Int, keys: [String]
  ) -> Int {
    for i in 0..<upTo {
      let suffix = "\(prefix).single_transformer_blocks.\(i).\(attn).to_k.weight"
      if !keys.contains(where: { $0.hasSuffix(suffix) }) {
        return i
      }
    }
    return upTo
  }

  public func importFlux1(
    archive: TensorArchive, stateDict: [String: TensorDescriptor], filename: String,
    progress: @escaping (Float) -> Void
  ) throws -> (transformerBlocks: [Int], version: ModelVersion, type: ControlType) {
    let keys = Array(stateDict.keys)
    guard
      (keys.contains {
        $0.contains("transformer_blocks.0.")
      })
    else {
      throw UnpickleError.tensorNotFound
    }
    let doubleControls = Self.numberOfTransformBlocks(
      prefix: "model.control_model", attn: "attn", upTo: 19, keys: keys)
    let singleControls = Self.numberOfSingleTransformerBlocks(
      prefix: "model.control_model", attn: "attn", upTo: 38, keys: keys)
    guard doubleControls > 0 || singleControls > 0 else {
      throw UnpickleError.tensorNotFound
    }
    let union = keys.contains { $0.hasSuffix("controlnet_mode_embedder.weight") }
    // Union is handled in ControlNetFlux1Fixed, the union provided here is only about +1 to tokenLength, which will be counterproductive if ew use Flux1FixedOutputShapes to get the proper shape.
    let (controlNetMapper, controlNet) = ControlNetFlux1(
      union: false, batchSize: 1, tokenLength: 256, height: 64, width: 64, channels: 3072,
      layers: (doubleControls, singleControls), usesFlashAttention: .none)
    let (controlNetFixedMapper, controlNetFixed) = ControlNetFlux1Fixed(
      union: union, batchSize: (1, 1), channels: 3072, layers: (doubleControls, singleControls),
      guidanceEmbed: true, of: FloatType.self)
    let graph = DynamicGraph()
    var crossattn: [DynamicGraph_Any] = [
      graph.variable(.CPU, .HWC(1, 256, 4096), of: FloatType.self),
      graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
      graph.variable(.CPU, .WC(1, 768), of: FloatType.self),
      graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
    ]
    if union {
      crossattn.insert(graph.variable(.CPU, format: .NHWC, shape: [1], of: Int32.self), at: 0)
    }
    controlNetFixed.compile(inputs: crossattn)
    let controlNetFixedMapping = controlNetFixedMapper(.diffusers)
    let flux1FixedOutputShapes = Flux1FixedOutputShapes(
      batchSize: (1, 1), tokenLength: 256, channels: 3072, layers: (doubleControls, singleControls),
      contextPreloaded: true)
    let cArr =
      [
        graph.variable(
          Tensor<FloatType>(
            from: Flux1RotaryPositionEmbedding(
              height: 32, width: 32, tokenLength: 256, referenceSizes: [], channels: 128)))
      ]
      + flux1FixedOutputShapes.map {
        graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
      }
    let xTensor = graph.variable(.CPU, .NHWC(1, 64, 64, 16), of: FloatType.self)
    // Remove the last two because for Flux1, there is additional scale / shift and not for ControlNetFlux1.
    controlNet.compile(inputs: [xTensor, xTensor] + cArr.prefix(upTo: cArr.count - 2))
    let controlNetMapping = controlNetMapper(.diffusers)
    try graph.openStore(ModelZoo.filePathForModelDownloaded(filename)) { store in
      store.removeAll()
      expectedTotalAccess = controlNetMapping.count + controlNetFixedMapping.count
      access = 0
      try store.withTransaction {
        for (key, value) in controlNetMapping {
          guard let tensorDescriptor = stateDict[key] else {
            continue
          }
          try archive.with(tensorDescriptor) { tensor in
            let tensor = {
              // If it is a zero conv bias, because we scale the output by 1/8, we should scale the bias by the same amount too.
              guard key.hasSuffix(".bias") && (value.contains { $0.contains("zero_conv") }) else {
                return Tensor<FloatType>(from: tensor)
              }
              let f32 = graph.variable(Tensor<Float>(from: tensor))
              return Tensor<FloatType>(from: f32.scaled(by: 1.0 / 8).rawValue)
            }()
            if value.count > 1 {
              value.write(
                graph: graph,
                to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                isDiagonalDown: false
              ) {
                return "__controlnet__[\($0)]"
              }
            } else if let name = value.first {
              store.write("__controlnet__[\(name)]", tensor: tensor)
            }
          }
          progress(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
        }
        for (key, value) in controlNetFixedMapping {
          guard let tensorDescriptor = stateDict[key] else {
            continue
          }
          try archive.with(tensorDescriptor) { tensor in
            let tensor = Tensor<FloatType>(from: tensor)
            if value.count > 1 {
              value.write(
                graph: graph,
                to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                isDiagonalDown: false
              ) {
                return "__controlnet__[\($0)]"
              }
            } else if let name = value.first {
              store.write("__controlnet__[\(name)]", tensor: tensor)
            }
          }
          progress(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
        }
      }
    }
    return (
      transformerBlocks: [doubleControls, singleControls], version: .flux1,
      type: union ? .controlnetunion : .controlnet
    )
  }

  private func ControlAddEmbed() -> (Model, ModelWeightMapper) {
    let (fc1, fc2, model) = Diffusion.ControlAddEmbed(modelChannels: 320)
    let mapper: ModelWeightMapper = { _ in
      var mapping = [String: ModelWeightElement]()
      mapping["control_add_embedding.linear_1.weight"] = [fc1.weight.name]
      mapping["control_add_embedding.linear_1.bias"] = [fc1.bias.name]
      mapping["control_add_embedding.linear_2.weight"] = [fc2.weight.name]
      mapping["control_add_embedding.linear_2.bias"] = [fc2.bias.name]
      return mapping
    }
    return (model, mapper)
  }

  public func `import`(
    downloadedFile: String, name: String, filename: String, modifier: ControlHintType?,
    progress: @escaping (Float) -> Void
  ) throws -> (transformerBlocks: [Int], version: ModelVersion, type: ControlType) {
    Interpreter.inflateInterrupter = interrupt
    defer {
      Interpreter.inflateInterrupter = nil
    }
    let filePath =
      downloadedFile.starts(with: "/")
      ? downloadedFile : ModelZoo.filePathForDownloadedFile(downloadedFile)
    let archive: TensorArchive
    var stateDict: [String: TensorDescriptor]
    let removePrefix = "control_model."
    if let safeTensors = SafeTensors(url: URL(fileURLWithPath: filePath)) {
      archive = safeTensors
      let states = safeTensors.states
      stateDict = states
      for (key, value) in states {
        var newKey = key
        if key.hasPrefix(removePrefix) {
          let prefixRange = key.startIndex..<key.index(key.startIndex, offsetBy: removePrefix.count)
          newKey = String(key[prefixRange.upperBound...])
        }
        if !key.hasPrefix("model.control_model.") {
          stateDict["model.control_model.\(newKey)"] = value
        }
      }
    } else if let zipArchive = Archive(url: URL(fileURLWithPath: filePath), accessMode: .read) {
      archive = zipArchive
      let rootObject = try Interpreter.unpickle(zip: zipArchive)
      let originalStateDict = rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        var newKey = key
        if key.hasPrefix(removePrefix) {
          let prefixRange = key.startIndex..<key.index(key.startIndex, offsetBy: removePrefix.count)
          newKey = String(key[prefixRange.upperBound...])
        }
        if !key.hasPrefix("model.control_model.") {
          stateDict["model.control_model.\(newKey)"] = value
        } else {
          stateDict[key] = value
        }
      }
    } else {
      throw UnpickleError.dataNotFound
    }
    progress(0.05)
    guard
      let tokey = stateDict.first(where: {
        $0.key.hasSuffix("transformer_blocks.0.attn2.to_k.weight")
          // In case there is no transformer blocks at all, we probing the label emb layer for SDXL models.
          || $0.key.hasSuffix("add_embedding.linear_1.weight")
          || $0.key.hasSuffix("label_emb.0.0.weight")
          // Probing for ControlNet LoRA.
          || $0.key.hasSuffix("transformer_blocks.0.attn2.to_k.down")
          || $0.key.hasSuffix("label_emb.0.0.down")
      })?.value
    else {
      // See if we can import this as Flux1. If we support more ControlNet in the future, we can do detection there and keep this method to import SD v1.5 / SDXL only.
      return try importFlux1(
        archive: archive, stateDict: stateDict, filename: filename, progress: progress)
    }
    var modelVersion: ModelVersion = .v1
    switch tokey.shape.last {
    case 768:
      modelVersion = .v1
    case 1024:
      modelVersion = .v2
    case 2048, 2816:
      modelVersion = .sdxlBase
    case 1280, 2560:
      modelVersion = .sdxlRefiner
      throw UnpickleError.tensorNotFound
    default:
      throw UnpickleError.tensorNotFound
    }
    let graph = DynamicGraph()
    let (hintNet, reader, mapper) = HintNet(channels: 320)
    let tokensTensor = graph.variable(.CPU, .NHWC(1, 512, 512, 3), of: FloatType.self)
    let x = graph.variable(.CPU, .NHWC(2, 64, 64, 4), of: FloatType.self)
    let hint = graph.variable(.CPU, .NHWC(2, 64, 64, 320), of: FloatType.self)
    let tembed = graph.variable(.CPU, .WC(2, 320), of: FloatType.self)
    let dim: Int
    switch modelVersion {
    case .longcatVideoAvatar1_5, .minimaxH3:
      fatalError()
    case .v1:
      dim = 768
    case .v2:
      dim = 1024
    case .sdxlBase, .ssd1b:
      dim = 2048
    case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .sdxlRefiner,
      .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1,
      .hiDreamO1, .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2,
      .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
      fatalError()
    }
    let c = graph.variable(.CPU, .HWC(2, 77, dim), of: FloatType.self)
    hintNet.compile(inputs: tokensTensor)
    let controlNet: Model
    let controlNetReader: PythonReader?
    let controlNetMapper: ModelWeightMapper?
    let controlNetFixed: Model?
    let controlNetFixedMapper: ModelWeightMapper?
    let isDiffusersFormat = stateDict.keys.contains { $0.hasPrefix("mid_block.") }
    let isControlNetLoRA = stateDict.keys.contains {
      $0.hasSuffix("label_emb.0.0.down") || $0.hasSuffix("transformer_blocks.0.attn2.to_k.down")
    }
    let isControlUnion = stateDict["control_add_embedding.linear_1.weight"] != nil
    var transformerBlocks = [Int]()
    switch modelVersion {
    case .longcatVideoAvatar1_5, .minimaxH3:
      fatalError()
    case .v1:
      (controlNet, controlNetReader) = ControlNet(
        batchSize: 2, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
        usesFlashAttention: .scaleMerged)
      controlNetMapper = nil
      controlNetFixed = nil
      controlNetFixedMapper = nil
    case .v2:
      (controlNet, controlNetReader) = ControlNetv2(
        batchSize: 2, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
        upcastAttention: false, usesFlashAttention: .scaleMerged)
      controlNetMapper = nil
      controlNetFixed = nil
      controlNetFixedMapper = nil
    case .sdxlBase:
      let inputAttentionRes: KeyValuePairs<Int, [Int]>
      let middleAttentionBlocks: Int
      if isDiffusersFormat {
        // For diffusers format ControlNet, there is small / mid / full size.
        // For small size, there is no transformer blocks, for mid size, there is one transformer blocks.
        // To make this generic, we will get "transform_blocks" and configure the network accordingly.
        let keys = Array(stateDict.keys)
        let downBlocks1AttentionRes = Self.numberOfTransformBlocks(
          prefix: "down_blocks.1.attentions.0", attn: "attn2", upTo: 2, keys: keys)
        let downBlocks2AttentionRes = Self.numberOfTransformBlocks(
          prefix: "down_blocks.2.attentions.0", attn: "attn2", upTo: 10, keys: keys)
        inputAttentionRes = [
          2: [downBlocks1AttentionRes, downBlocks1AttentionRes],
          4: [downBlocks2AttentionRes, downBlocks2AttentionRes],
        ]
        middleAttentionBlocks = Self.numberOfTransformBlocks(
          prefix: "mid_block.attentions.0", attn: "attn2", upTo: 10, keys: keys)
        if downBlocks1AttentionRes != 2 || downBlocks2AttentionRes != 10
          || middleAttentionBlocks != 10
        {
          transformerBlocks = [
            0, downBlocks1AttentionRes, downBlocks2AttentionRes, middleAttentionBlocks,
          ]
        }
      } else {
        inputAttentionRes = [2: [2, 2], 4: [10, 10]]
        middleAttentionBlocks = 10
      }
      if isControlNetLoRA {
        let LoRAConfiguration = LoRANetworkConfiguration(rank: 64, scale: 1, highPrecision: false)
        (controlNet, controlNetMapper) = LoRAControlNetXL(
          batchSize: 2, startWidth: 64, startHeight: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: inputAttentionRes,
          middleAttentionBlocks: middleAttentionBlocks, usesFlashAttention: .none,
          LoRAConfiguration: LoRAConfiguration)
        if transformerBlocks.isEmpty || transformerBlocks.contains(where: { $0 > 0 }) {
          (controlNetFixed, controlNetFixedMapper) = LoRAControlNetXLFixed(
            batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
            embeddingLength: (77, 77), inputAttentionRes: inputAttentionRes,
            middleAttentionBlocks: middleAttentionBlocks, usesFlashAttention: .none,
            LoRAConfiguration: LoRAConfiguration)
        } else {
          controlNetFixed = nil
          controlNetFixedMapper = nil
        }
      } else {
        (controlNet, _, controlNetMapper) = ControlNetXL(
          batchSize: 2, startWidth: 64, startHeight: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: inputAttentionRes,
          middleAttentionBlocks: middleAttentionBlocks, union: isControlUnion,
          usesFlashAttention: .none)
        if transformerBlocks.isEmpty || transformerBlocks.contains(where: { $0 > 0 }) {
          (controlNetFixed, _, controlNetFixedMapper) = ControlNetXLFixed(
            batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
            embeddingLength: (77, 77), inputAttentionRes: inputAttentionRes,
            middleAttentionBlocks: middleAttentionBlocks, usesFlashAttention: .none)
        } else {
          controlNetFixed = nil
          controlNetFixedMapper = nil
        }
      }
      controlNetReader = nil
    case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .ssd1b, .svdI2v, .sdxlRefiner,
      .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1,
      .hiDreamO1, .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2,
      .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
      fatalError()
    }

    if let controlNetMapper = controlNetMapper {
      try graph.openStore(ModelZoo.filePathForModelDownloaded(filename)) { store in
        let mapping: ModelWeightMapping
        let mappingFixed: ModelWeightMapping
        let hintMapping: ModelWeightMapping
        switch modelVersion {
        case .longcatVideoAvatar1_5, .minimaxH3:
          fatalError()
        case .sdxlBase:
          let vector = graph.variable(.GPU(0), .WC(2, 2816), of: FloatType.self)
          // These values doesn't matter, it won't affect the model shape, just the input vector.
          let kvs: [DynamicGraph.Tensor<FloatType>]
          if let controlNetFixed = controlNetFixed {
            let crossattn = graph.variable(.GPU(0), .HWC(2, 77, 2048), of: FloatType.self)
            kvs = controlNetFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
          } else {
            kvs = []
          }
          var cArr = [vector]
          if !isControlNetLoRA && isControlUnion {
            let controlEmb = graph.variable(.GPU(0), .WC(2, 1280), of: FloatType.self)
            let taskEmbedding = graph.variable(.GPU(0), .WC(2, 320), of: FloatType.self)
            cArr += [controlEmb, taskEmbedding]
          }
          cArr += kvs
          controlNet.compile(inputs: [x, hint, tembed] + cArr)
          if isDiffusersFormat {
            mapping = controlNetMapper(.diffusers)
            if let controlNetFixedMapper = controlNetFixedMapper {
              mappingFixed = controlNetFixedMapper(.diffusers)
            } else {
              mappingFixed = [:]
            }
            hintMapping = mapper(.diffusers)
          } else {
            mapping = controlNetMapper(.generativeModels)
            if let controlNetFixedMapper = controlNetFixedMapper {
              mappingFixed = controlNetFixedMapper(.generativeModels)
            } else {
              mappingFixed = [:]
            }
            hintMapping = mapper(.generativeModels)
          }
        case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .sdxlRefiner, .ssd1b, .v1, .v2,
          .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b,
          .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b,
          .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b,
          .seedvr2_7b, .ideogram4, .krea2:
          fatalError()
        }
        store.removeAll()
        expectedTotalAccess = mapping.count + mappingFixed.count + hintMapping.count
        access = 0
        try store.withTransaction {
          if let encoderHidProjWeightDescriptor = stateDict["encoder_hid_proj.weight"],
            let encoderHidProjBiasDescriptor = stateDict["encoder_hid_proj.bias"]
          {
            try archive.with(encoderHidProjWeightDescriptor) { tensor in
              let tensor = Tensor<FloatType>(from: tensor)
              store.write("__encoder_hid_proj__[t-0-0]", tensor: tensor)
            }
            try archive.with(encoderHidProjBiasDescriptor) { tensor in
              let tensor = Tensor<FloatType>(from: tensor)
              store.write("__encoder_hid_proj__[t-0-1]", tensor: tensor)
            }
          }
          if stateDict["control_add_embedding.linear_1.weight"] != nil {
            let (controlAddEmbedding, controlAddEmbeddingMapper) = ControlAddEmbed()
            let controlType = graph.variable(.GPU(0), .WC(2, 256 * 8), of: FloatType.self)
            controlAddEmbedding.compile(inputs: controlType)
            let mapping = controlAddEmbeddingMapper(.diffusers)
            for (key, value) in mapping {
              guard let tensorDescriptor = stateDict[key] else {
                continue
              }
              try archive.with(tensorDescriptor) { tensor in
                let tensor = Tensor<FloatType>(from: tensor)
                for name in value {
                  store.write("__control_add_embed__[\(name)]", tensor: tensor)
                }
              }
            }
          }
          if let taskEmbedding = stateDict["task_embedding"] {
            try archive.with(taskEmbedding) { tensor in
              let tensor = Tensor<FloatType>(from: tensor)
              store.write("task_embedding", tensor: tensor)
            }
          }
          for (key, value) in mapping {
            guard let tensorDescriptor = stateDict[key] else {
              continue
            }
            try archive.with(tensorDescriptor) { tensor in
              let tensor = Tensor<FloatType>(from: tensor)
              if value.count > 1 {
                if key.hasSuffix(".down") {
                  for name in value {
                    store.write("__controlnet__[\(name)]", tensor: tensor)
                  }
                } else {
                  let count = tensor.shape[0] / value.count
                  for (i, name) in value.enumerated() {
                    if tensor.shape.count > 1 {
                      store.write(
                        "__controlnet__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count), 0..<tensor.shape[1]]
                          .copied())
                    } else {
                      store.write(
                        "__controlnet__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                    }
                  }
                }
              } else if let name = value.first {
                store.write("__controlnet__[\(name)]", tensor: tensor)
              }
            }
            progress(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
          }
          for (key, value) in mappingFixed {
            guard let tensorDescriptor = stateDict[key] else {
              continue
            }
            try archive.with(tensorDescriptor) { tensor in
              let tensor = Tensor<FloatType>(from: tensor)
              if value.count > 1 {
                if key.hasSuffix(".down") {
                  for name in value {
                    store.write("__controlnet_fixed__[\(name)]", tensor: tensor)
                  }
                } else {
                  let count = tensor.shape[0] / value.count
                  for (i, name) in value.enumerated() {
                    if tensor.shape.count > 1 {
                      store.write(
                        "__controlnet_fixed__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count), 0..<tensor.shape[1]]
                          .copied())
                    } else {
                      store.write(
                        "__controlnet_fixed__[\(name)]",
                        tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                    }
                  }
                }
              } else if let name = value.first {
                store.write("__controlnet_fixed__[\(name)]", tensor: tensor)
              }
            }
            progress(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
          }
          for (key, value) in hintMapping {
            guard let tensorDescriptor = stateDict[key] else {
              continue
            }
            try archive.with(tensorDescriptor) { tensor in
              let tensor = Tensor<FloatType>(from: tensor)
              if value.count > 1 {
                let count = tensor.shape[0] / value.count
                for (i, name) in value.enumerated() {
                  if tensor.shape.count > 1 {
                    store.write(
                      "__hintnet__[\(name)]",
                      tensor: tensor[(i * count)..<((i + 1) * count), 0..<tensor.shape[1]]
                        .copied())
                  } else {
                    store.write(
                      "__hintnet__[\(name)]",
                      tensor: tensor[(i * count)..<((i + 1) * count)].copied())
                  }
                }
              } else if let name = value.first {
                store.write("__hintnet__[\(name)]", tensor: tensor)
              }
            }
            progress(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
          }
        }
      }
    } else if let controlNetReader = controlNetReader {
      controlNet.compile(inputs: x, hint, tembed, c)
      access = 0
      // This is the same for v1, v2.
      expectedTotalAccess = 340
      self.progress = progress
      try reader(stateDict, archive)
      try controlNetReader(stateDict, archive)
      self.progress = nil
      graph.openStore(ModelZoo.filePathForModelDownloaded(filename)) { store in
        store.removeAll()
        store.write("hintnet", model: hintNet)
        store.write("controlnet", model: controlNet)
      }
      progress(1)
    }
    return (
      transformerBlocks, modelVersion,
      isControlUnion ? .controlnetunion : (isControlNetLoRA ? .controlnetlora : .controlnet)
    )
  }
}
