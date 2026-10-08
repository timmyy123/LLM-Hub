import Diffusion
import NNC
import TensorBoard

public struct LoRATrainerCheckpoint {
  public var version: ModelVersion
  public var textModel1: AnyModel?
  public var textModel2: AnyModel?
  public var unetFixed: AnyModel?
  public var unet: AnyModel
  public var textEmbedding1: DynamicGraph.Tensor<Float>?
  public var textEmbedding2: DynamicGraph.Tensor<Float>?
  public var step: Int
  public struct ExponentialMovingAverage {
    public var textModel1: [String: Tensor<Float>]
    public var textModel2: [String: Tensor<Float>]
    public var unetFixed: [String: Tensor<Float>]
    public var unet: [String: Tensor<Float>]
    public var textEmbedding1: Tensor<Float>?
    public var textEmbedding2: Tensor<Float>?
    public init(
      textModel1: [String: Tensor<Float>],
      textModel2: [String: Tensor<Float>],
      unetFixed: [String: Tensor<Float>], unet: [String: Tensor<Float>],
      textEmbedding1: Tensor<Float>? = nil,
      textEmbedding2: Tensor<Float>? = nil
    ) {
      self.textModel1 = textModel1
      self.textModel2 = textModel2
      self.unetFixed = unetFixed
      self.unet = unet
      self.textEmbedding1 = textEmbedding1
      self.textEmbedding2 = textEmbedding2
    }
  }
  public var exponentialMovingAverageLowerBound: ExponentialMovingAverage?
  public var exponentialMovingAverageUpperBound: ExponentialMovingAverage?
  public init(
    version: ModelVersion, textModel1: AnyModel? = nil, textModel2: AnyModel? = nil,
    unetFixed: AnyModel? = nil, unet: AnyModel, textEmbedding1: DynamicGraph.Tensor<Float>? = nil,
    textEmbedding2: DynamicGraph.Tensor<Float>? = nil, step: Int,
    exponentialMovingAverageLowerBound: ExponentialMovingAverage? = nil,
    exponentialMovingAverageUpperBound: ExponentialMovingAverage? = nil
  ) {
    self.version = version
    self.textModel1 = textModel1
    self.textModel2 = textModel2
    self.unetFixed = unetFixed
    self.unet = unet
    self.textEmbedding1 = textEmbedding1
    self.textEmbedding2 = textEmbedding2
    self.step = step
    self.exponentialMovingAverageLowerBound = exponentialMovingAverageLowerBound
    self.exponentialMovingAverageUpperBound = exponentialMovingAverageUpperBound
  }
}

extension LoRATrainerCheckpoint {
  // This is a very easy way to quickly save a session.
  public func write(to filePath: String) {
    let graph = textEmbedding1?.graph ?? DynamicGraph()
    graph.openStore(filePath) {
      if let textEmbedding1 = textEmbedding1 {
        switch version {
        case .longcatVideoAvatar1_5, .minimaxH3:
          fatalError()
        case .v1, .v2, .kandinsky21, .svdI2v, .pixart, .auraflow, .flux1, .hunyuanVideo,
          .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage2_1, .qwenImage, .cosmos2_5_2b,
          .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3:
          $0.write("string_to_param", variable: textEmbedding1)
        case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC, .wurstchenStageB:
          $0.write("string_to_param_clip_g", variable: textEmbedding1)
          if let textEmbedding2 = textEmbedding2 {
            $0.write("string_to_param_clip_l", variable: textEmbedding2)
          }
        case .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
          fatalError()
        }
      }
      let textModels = (textModel1.map { [$0] } ?? []) + (textModel2.map { [$0] } ?? [])
      for (i, textModel) in textModels.enumerated() {
        $0.write("te\(i)__text_model", model: textModel) { name, _ in
          guard name.contains("[i-") || name.contains("lora") else {
            return .skip
          }
          return .continue(name)
        }
      }
      if let unetFixed = unetFixed {
        $0.write("unet_fixed", model: unetFixed) { name, _ in
          guard name.contains("[i-") || name.contains("lora") else {
            return .skip
          }
          return .continue(name)
        }
      }
      let modelName: String
      switch version {
      case .longcatVideoAvatar1_5, .minimaxH3:
        fatalError()
      case .v1, .v2, .ssd1b, .sdxlBase, .sdxlRefiner:
        modelName = "unet"
      case .sd3, .pixart, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1,
        .hiDreamO1, .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage,
        .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .krea2:
        modelName = "dit"
      case .auraflow, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .seedvr2_3b,
        .seedvr2_7b, .ideogram4:
        fatalError()
      }
      $0.write(modelName, model: unet) { name, _ in
        guard name.contains("[i-") || name.contains("lora") else {
          return .skip
        }
        return .continue(name)
      }
      var stepTensor = Tensor<Int32>(.CPU, .C(1))
      stepTensor[0] = Int32(step)
      $0.write("current_step", tensor: stepTensor)
    }
  }
}

extension LoRATrainerCheckpoint {
  public func makeLoRA(to filePath: String, scale: Float) {
    let graph = textEmbedding1?.graph ?? DynamicGraph()
    graph.openStore(filePath) { store in
      // Remove all values first.
      store.removeAll()
      if let textEmbedding1 = textEmbedding1 {
        switch version {
        case .longcatVideoAvatar1_5, .minimaxH3:
          fatalError()
        case .v1, .v2, .kandinsky21, .svdI2v, .pixart, .auraflow, .flux1, .hunyuanVideo,
          .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage2_1, .qwenImage, .cosmos2_5_2b,
          .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3:
          store.write("string_to_param", variable: textEmbedding1)
        case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC, .wurstchenStageB:
          store.write("string_to_param_clip_g", variable: textEmbedding1)
          if let textEmbedding2 = textEmbedding2 {
            store.write("string_to_param_clip_l", variable: textEmbedding2)
          }
        case .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
          fatalError()
        }
      }
      if let textModel1 = textModel1 {
        let textModelMapping: [Int: Int]
        switch version {
        case .longcatVideoAvatar1_5, .minimaxH3:
          fatalError()
        case .v1:
          textModelMapping = LoRAMapping.CLIPTextModel
        case .v2:
          textModelMapping = LoRAMapping.OpenCLIPTextModel
        case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
          .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1,
          .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2,
          .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
          fatalError()
        case .sdxlBase, .ssd1b, .sdxlRefiner:
          textModelMapping = LoRAMapping.OpenCLIPTextModelG
        }
        store.write((textModel2 != nil ? "te2__" : "") + "text_model", model: textModel1) {
          name, tensor in
          guard name.contains("lora") else { return .skip }
          let isUp = name.contains("lora_up")
          let updatedName = LoRATrainer.originalLoRA(name: name, LoRAMapping: textModelMapping)
          if scale != 1 && !isUp {
            let tensor = graph.withNoGrad {
              (scale * graph.variable(Tensor<Float>(from: tensor))).rawValue
            }
            store.write(updatedName, tensor: tensor)
            return .skip
          }
          return .continue(updatedName)
        }
      }
      if let textModel2 = textModel2 {
        let textModelMapping = LoRAMapping.CLIPTextModel
        store.write("text_model", model: textModel2) { name, tensor in
          guard name.contains("lora") else { return .skip }
          let isUp = name.contains("lora_up")
          let updatedName = LoRATrainer.originalLoRA(name: name, LoRAMapping: textModelMapping)
          if scale != 1 && !isUp {
            let tensor = graph.withNoGrad {
              (scale * graph.variable(Tensor<Float>(from: tensor))).rawValue
            }
            store.write(updatedName, tensor: tensor)
            return .skip
          }
          return .continue(updatedName)
        }
      }
      if let unetFixed = unetFixed {
        store.write("unet_fixed", model: unetFixed) { name, tensor in
          guard name.contains("lora") else { return .skip }
          let isUp = name.contains("lora_up")
          // Every parameter in unetFixed is trainable.
          let updatedName = LoRATrainer.originalLoRA(name: name, LoRAMapping: nil)
          if scale != 1 && !isUp {
            let tensor = graph.withNoGrad {
              (scale * graph.variable(Tensor<Float>(from: tensor))).rawValue
            }
            store.write(updatedName, tensor: tensor)
            return .skip
          }
          return .continue(updatedName)
        }
      }
      let modelName: String
      let UNetMapping: [Int: Int]
      switch version {
      case .longcatVideoAvatar1_5, .minimaxH3:
        fatalError()
      case .v1, .v2:
        UNetMapping = LoRAMapping.SDUNet
        modelName = "unet"
      case .sd3:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<24).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .pixart:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<28).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .flux1:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<(19 + 38)).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .flux2:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<(8 + 48)).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .flux2_9b:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<(8 + 24)).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .flux2_4b:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<(5 + 20)).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .krea2:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<28).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .sd3Large:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<38).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .qwenImage:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<60).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .zImage:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<30).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .ernieImage:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<36).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .cosmos2_5_2b:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<28).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .auraflow, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo,
        .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage2_1, .wan22_5b, .ltx2, .ltx2_3,
        .seedvr2_3b, .seedvr2_7b, .ideogram4:
        fatalError()
      case .ssd1b:
        UNetMapping = LoRAMapping.SDUNetXLSSD1B
        modelName = "unet"
      case .sdxlBase:
        UNetMapping = LoRAMapping.SDUNetXLBase
        modelName = "unet"
      case .sdxlRefiner:
        UNetMapping = LoRAMapping.SDUNetXLRefiner
        modelName = "unet"
      }
      store.write(modelName, model: unet) { name, tensor in
        guard name.contains("lora") else { return .skip }
        let isUp = name.contains("lora_up")
        let updatedName = LoRATrainer.originalLoRA(name: name, LoRAMapping: UNetMapping)
        if scale != 1 && !isUp {
          let tensor = graph.withNoGrad {
            (scale * graph.variable(Tensor<Float>(from: tensor))).rawValue
          }
          store.write(updatedName, tensor: tensor)
          return .skip
        }
        return .continue(updatedName)
      }
    }
  }

  public func makeLoRA(
    _ exponentialMovingAverage: ExponentialMovingAverage, to filePath: String, scale: Float
  ) {
    let graph = DynamicGraph()
    graph.openStore(filePath) { store in
      // Remove all values first.
      store.removeAll()
      if let textEmbedding1 = exponentialMovingAverage.textEmbedding1 {
        switch version {
        case .longcatVideoAvatar1_5, .minimaxH3:
          fatalError()
        case .v1, .v2, .kandinsky21, .svdI2v, .pixart, .auraflow, .flux1, .hunyuanVideo,
          .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage2_1, .qwenImage, .cosmos2_5_2b,
          .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3:
          store.write("string_to_param", tensor: textEmbedding1)
        case .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC, .wurstchenStageB:
          store.write("string_to_param_clip_g", tensor: textEmbedding1)
          if let textEmbedding2 = exponentialMovingAverage.textEmbedding2 {
            store.write("string_to_param_clip_l", tensor: textEmbedding2)
          }
        case .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
          fatalError()
        }
      }
      if !exponentialMovingAverage.textModel1.isEmpty {
        let textModelMapping: [Int: Int]
        switch version {
        case .longcatVideoAvatar1_5, .minimaxH3:
          fatalError()
        case .v1:
          textModelMapping = LoRAMapping.CLIPTextModel
        case .v2:
          textModelMapping = LoRAMapping.OpenCLIPTextModel
        case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
          .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1,
          .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2,
          .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
          fatalError()
        case .sdxlBase, .ssd1b, .sdxlRefiner:
          textModelMapping = LoRAMapping.OpenCLIPTextModelG
        }
        let modelName = (textModel2 != nil ? "te2__" : "") + "text_model"
        for (name, tensor) in exponentialMovingAverage.textModel1 {
          guard name.contains("lora") else { continue }
          let isUp = name.contains("lora_up")
          let updatedName =
            "__\(modelName)__["
            + LoRATrainer.originalLoRA(name: name, LoRAMapping: textModelMapping)
          if scale != 1 && !isUp {
            let tensor = graph.withNoGrad {
              (scale * graph.variable(tensor)).rawValue
            }
            store.write(updatedName, tensor: tensor)
            continue
          }
          store.write(updatedName, tensor: tensor)
        }
      }
      if !exponentialMovingAverage.textModel2.isEmpty {
        let textModelMapping = LoRAMapping.CLIPTextModel
        let modelName = "text_model"
        for (name, tensor) in exponentialMovingAverage.textModel2 {
          guard name.contains("lora") else { continue }
          let isUp = name.contains("lora_up")
          let updatedName =
            "__\(modelName)__["
            + LoRATrainer.originalLoRA(name: name, LoRAMapping: textModelMapping)
          if scale != 1 && !isUp {
            let tensor = graph.withNoGrad {
              (scale * graph.variable(tensor)).rawValue
            }
            store.write(updatedName, tensor: tensor)
            continue
          }
          store.write(updatedName, tensor: tensor)
        }
      }
      if !exponentialMovingAverage.unetFixed.isEmpty {
        let modelName = "unet_fixed"
        for (name, tensor) in exponentialMovingAverage.unetFixed {
          guard name.contains("lora") else { continue }
          let isUp = name.contains("lora_up")
          // Every parameter in unetFixed is trainable.
          let updatedName =
            "__\(modelName)__[" + LoRATrainer.originalLoRA(name: name, LoRAMapping: nil)
          if scale != 1 && !isUp {
            let tensor = graph.withNoGrad {
              (scale * graph.variable(tensor)).rawValue
            }
            store.write(updatedName, tensor: tensor)
            continue
          }
          store.write(updatedName, tensor: tensor)
        }
      }
      let modelName: String
      let UNetMapping: [Int: Int]
      switch version {
      case .longcatVideoAvatar1_5, .minimaxH3:
        fatalError()
      case .v1, .v2:
        UNetMapping = LoRAMapping.SDUNet
        modelName = "unet"
      case .sd3:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<24).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .pixart:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<28).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .flux1:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<(19 + 38)).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .flux2:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<(8 + 48)).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .flux2_9b:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<(8 + 24)).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .flux2_4b:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<(5 + 20)).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .krea2:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<28).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .sd3Large:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<38).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .qwenImage:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<60).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .zImage:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<30).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .ernieImage:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<36).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .cosmos2_5_2b:
        UNetMapping = [Int: Int](
          uniqueKeysWithValues: (0..<28).map {
            return ($0, $0)
          })
        modelName = "dit"
      case .auraflow, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .hunyuanVideo,
        .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage2_1, .wan22_5b, .ltx2, .ltx2_3,
        .seedvr2_3b, .seedvr2_7b, .ideogram4:
        fatalError()
      case .ssd1b:
        UNetMapping = LoRAMapping.SDUNetXLSSD1B
        modelName = "unet"
      case .sdxlBase:
        UNetMapping = LoRAMapping.SDUNetXLBase
        modelName = "unet"
      case .sdxlRefiner:
        UNetMapping = LoRAMapping.SDUNetXLRefiner
        modelName = "unet"
      }
      for (name, tensor) in exponentialMovingAverage.unet {
        guard name.contains("lora") else { continue }
        let isUp = name.contains("lora_up")
        let updatedName =
          "__\(modelName)__[" + LoRATrainer.originalLoRA(name: name, LoRAMapping: UNetMapping)
        if scale != 1 && !isUp {
          let tensor = graph.withNoGrad {
            (scale * graph.variable(tensor)).rawValue
          }
          store.write(updatedName, tensor: tensor)
          continue
        }
        store.write(updatedName, tensor: tensor)
      }
    }
  }

  public func write(to summaryWriter: SummaryWriter) {
    summaryWriter.addParameters(
      "lora_up", unet.parameters.filter(where: { $0.contains("lora_up") }), step: step)
    summaryWriter.addParameters(
      "lora_down", unet.parameters.filter(where: { $0.contains("lora_down") }), step: step)
  }
}
