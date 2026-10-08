import Diffusion
import Foundation
import ModelZoo
import NNC

public enum LoRAExporter {

  public static func export(file: String, outputFile: String, progress: (Int, Int) -> Void) throws {
    let humanName = LoRAZoo.humanReadableNameForModel(file)
    let version = LoRAZoo.versionForModel(file)
    let prefix = LoRAZoo.textPrefixForModel(file)
    enum ModelExportFormat {
      case kohyaTrainer
      case comfyUI
    }
    let modelFixedPrefix: String
    let modelPrefix: String
    var modelExportFormat: ModelExportFormat = .kohyaTrainer
    var modelWeightFormats: [ModelWeightFormat] = [.generativeModels]
    var textualInversionExport: (storeKey: String, safetensorsKey: String)? = nil
    var metadata = [String: String]()
    metadata["name"] = humanName
    switch version {
    case .longcatVideoAvatar1_5, .minimaxH3:
      fatalError()
    case .v1:
      metadata["version"] = "v1"
      metadata["ss_base_model_version"] = "sd_v1"
      modelFixedPrefix = "unet_fixed"
      modelPrefix = "unet"
    case .v2:
      metadata["version"] = "v2"
      metadata["ss_base_model_version"] = "sd_v2"
      metadata["ss_v2"] = "True"
      modelFixedPrefix = "unet_fixed"
      modelPrefix = "unet"
    case .sdxlBase:
      metadata["version"] = "sdxl_base"
      metadata["ss_base_model_version"] = "sdxl_base"
      modelFixedPrefix = "unet_fixed"
      modelPrefix = "unet"
    case .sdxlRefiner:
      metadata["version"] = "sdxl_refiner"
      metadata["ss_base_model_version"] = "sdxl_refiner"
      modelFixedPrefix = "unet_fixed"
      modelPrefix = "unet"
    case .ssd1b:
      metadata["version"] = "ssd_1b"
      metadata["ss_base_model_version"] = "ssd_1b"
      modelFixedPrefix = "unet_fixed"
      modelPrefix = "unet"
    case .sd3:
      metadata["version"] = "sd3_medium"
      metadata["ss_base_model_version"] = "sd3_medium"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
    case .sd3Large:
      metadata["version"] = "sd3_large"
      metadata["ss_base_model_version"] = "sd3_large"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
    case .pixart:
      metadata["version"] = "pixart"
      metadata["ss_base_model_version"] = "pixart"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
    case .auraflow:
      metadata["version"] = "auraflow_v0.2"
      metadata["ss_base_model_version"] = "auraflow_v0.2"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
    case .flux1:
      metadata["version"] = "flux_1"
      metadata["ss_base_model_version"] = "flux_1"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
    case .hunyuanVideo:
      metadata["version"] = "hunyuan_video"
      metadata["ss_base_model_version"] = "hunyuan_video"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
    case .wan21_1_3b, .wan21_14b:
      metadata["version"] = "wan_v2.1_1.3b"
      metadata["ss_base_model_version"] = "wan_v2.1_1.3b"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
    case .qwenImage:
      metadata["version"] = "qwen_image"
      metadata["ss_base_model_version"] = "qwen_image"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
      modelExportFormat = .comfyUI
      modelWeightFormats = [.generativeModels]
      textualInversionExport = ("string_to_param_qwen25", "qwen25")
    case .cosmos2_5_2b:
      metadata["version"] = "cosmos2.5_2b"
      metadata["ss_base_model_version"] = "cosmos2.5_2b"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
      modelExportFormat = .comfyUI
      modelWeightFormats = [.generativeModels]
      textualInversionExport = ("string_to_param_qwen3", "qwen3")
    case .zImage:
      metadata["version"] = "z_image"
      metadata["ss_base_model_version"] = "z_image"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
      modelExportFormat = .comfyUI
      modelWeightFormats = [.diffusers]
      textualInversionExport = ("string_to_param_qwen3", "qwen3")
    case .ernieImage:
      metadata["version"] = "ernie_image"
      metadata["ss_base_model_version"] = "ernie_image"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
      modelExportFormat = .comfyUI
      modelWeightFormats = [.generativeModels]
      textualInversionExport = ("string_to_param_mistral3", "mistral3")
    case .wan22_5b:
      fatalError()
    case .hiDreamI1, .hiDreamO1, .qwenImage2_1:
      fatalError()
    case .flux2_4b:
      metadata["version"] = "flux_2_4b"
      metadata["ss_base_model_version"] = "flux_2_4b"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
      modelExportFormat = .comfyUI
      modelWeightFormats = [.generativeModels]
      textualInversionExport = ("string_to_param_mistral3", "mistral3")
    case .flux2_9b:
      metadata["version"] = "flux_2_9b"
      metadata["ss_base_model_version"] = "flux_2_9b"
      modelFixedPrefix = "dit"
      modelPrefix = "dit"
      modelExportFormat = .comfyUI
      modelWeightFormats = [.generativeModels]
      textualInversionExport = ("string_to_param_mistral3", "mistral3")
    case .flux2:
      fatalError()
    case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
      fatalError()
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
      fatalError()
    }
    if prefix.count > 0 {
      metadata["trigger_word"] = prefix
    }
    var json = [String: Any]()
    json["__metadata__"] = metadata
    let graph = DynamicGraph()
    let textEncoderKeys: [String]
    let textEncoderKeysMapping: ModelWeightMapping
    let textEncoderKeys2: [String]?
    let textEncoderKeysMapping2: ModelWeightMapping?
    switch version {
    case .longcatVideoAvatar1_5, .minimaxH3:
      fatalError()
    case .v1:
      textEncoderKeys = StableDiffusionMapping.CLIPTextModel.keys.sorted()
      textEncoderKeysMapping = StableDiffusionMapping.CLIPTextModel
      textEncoderKeys2 = nil
      textEncoderKeysMapping2 = nil
    case .v2:
      textEncoderKeys = StableDiffusionMapping.OpenCLIPTextModel.keys.sorted()
      textEncoderKeysMapping = StableDiffusionMapping.OpenCLIPTextModel
      textEncoderKeys2 = nil
      textEncoderKeysMapping2 = nil
    case .sdxlBase, .ssd1b:
      textEncoderKeys = StableDiffusionMapping.OpenCLIPTextModelG.keys.sorted()
      textEncoderKeysMapping = StableDiffusionMapping.OpenCLIPTextModelG
      textEncoderKeys2 = StableDiffusionMapping.CLIPTextModel.keys.sorted()
      textEncoderKeysMapping2 = StableDiffusionMapping.CLIPTextModel
    case .sdxlRefiner:
      textEncoderKeys = StableDiffusionMapping.OpenCLIPTextModelG.keys.sorted()
      textEncoderKeysMapping = StableDiffusionMapping.OpenCLIPTextModelG
      textEncoderKeys2 = nil
      textEncoderKeysMapping2 = nil
    case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1,
      .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b,
      .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
      textEncoderKeys = []
      textEncoderKeysMapping = [:]
      textEncoderKeys2 = nil
      textEncoderKeysMapping2 = nil
    }
    let qkNorm = false
    let dualAttentionLayers = [Int]()
    let (modelKeysMapping2, modelKeysMapping, _, _) = LoRAImporter.modelWeightsMapping(
      by: version, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      format: modelWeightFormats)
    let modelKeys = modelKeysMapping.keys.sorted()
    let modelKeys2 = modelKeysMapping2.keys.sorted()
    let total =
      (textEncoderKeys.count + (textEncoderKeys2?.count ?? 0) + modelKeys.count + modelKeys2.count)
      * 2
    var offset = 0
    progress(0, total)
    var progressed = 0
    graph.openStore(
      LoRAZoo.filePathForModelDownloaded(file), flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: LoRAZoo.filePathForModelDownloaded(file))
    ) {
      store in
      if let tensor = store.read("string_to_param") {
        var tensorDescriptor = [String: Any]()
        var bytesPerElement: Int
        switch tensor.dataType {
        case .Float16:
          tensorDescriptor["dtype"] = "F16"
          bytesPerElement = 2
        case .Float32:
          tensorDescriptor["dtype"] = "F32"
          bytesPerElement = 4
        default:
          bytesPerElement = 0
        }
        tensorDescriptor["shape"] = [Int](tensor.shape)
        let size = tensor.shape.reduce(bytesPerElement, *)
        tensorDescriptor["data_offsets"] = [offset, offset + size]
        offset += size
        json["emb_params"] = tensorDescriptor
      } else if let tensorClipG = store.read("string_to_param_clip_g"),
        let tensorClipL = store.read("string_to_param_clip_l")
      {
        var tensorClipGDescriptor = [String: Any]()
        var bytesPerElement: Int
        switch tensorClipG.dataType {
        case .Float16:
          tensorClipGDescriptor["dtype"] = "F16"
          bytesPerElement = 2
        case .Float32:
          tensorClipGDescriptor["dtype"] = "F32"
          bytesPerElement = 4
        default:
          bytesPerElement = 0
        }
        tensorClipGDescriptor["shape"] = [Int](tensorClipG.shape)
        let clipGSize = tensorClipG.shape.reduce(bytesPerElement, *)
        tensorClipGDescriptor["data_offsets"] = [offset, offset + clipGSize]
        offset += clipGSize
        json["clip_g"] = tensorClipGDescriptor
        var tensorClipLDescriptor = [String: Any]()
        switch tensorClipL.dataType {
        case .Float16:
          tensorClipLDescriptor["dtype"] = "F16"
          bytesPerElement = 2
        case .Float32:
          tensorClipLDescriptor["dtype"] = "F32"
          bytesPerElement = 4
        default:
          bytesPerElement = 0
        }
        tensorClipLDescriptor["shape"] = [Int](tensorClipL.shape)
        let clipLSize = tensorClipL.shape.reduce(bytesPerElement, *)
        tensorClipLDescriptor["data_offsets"] = [offset, offset + clipLSize]
        offset += clipLSize
        json["clip_g"] = tensorClipLDescriptor
      } else if let textualInversionExport = textualInversionExport,
        let tensor = store.read(textualInversionExport.storeKey)
      {
        var tensorDescriptor = [String: Any]()
        var bytesPerElement: Int
        switch tensor.dataType {
        case .Float16:
          tensorDescriptor["dtype"] = "F16"
          bytesPerElement = 2
        case .Float32:
          tensorDescriptor["dtype"] = "F32"
          bytesPerElement = 4
        default:
          bytesPerElement = 0
        }
        tensorDescriptor["shape"] = [Int](tensor.shape)
        let size = tensor.shape.reduce(bytesPerElement, *)
        tensorDescriptor["data_offsets"] = [offset, offset + size]
        offset += size
        json[textualInversionExport.safetensorsKey] = tensorDescriptor
      }
      func loraPrefix(forModelKey key: String) -> String? {
        var components = key.split(separator: ".")
        guard components.count > 1 else { return nil }
        components.removeLast()
        switch modelExportFormat {
        case .kohyaTrainer:
          if components.count > 2 && components[0] == "model" && components[1] == "diffusion_model"
          {
            components.removeFirst(2)
          } else if components.count > 1
            && (components[0] == "diffusion_model" || components[0] == "model")
          {
            components.removeFirst()
          }
          return "lora_unet_" + components.joined(separator: "_")
        case .comfyUI:
          if components.first == "model" {
            components.removeFirst()
          }
          if components.first != "diffusion_model" {
            components.insert("diffusion_model", at: 0)
          }
          return components.joined(separator: ".")
        }
      }
      for (i, key) in textEncoderKeys.enumerated() {
        guard key.hasSuffix(".weight") else { continue }
        guard let name = textEncoderKeysMapping[key]?.first else { continue }
        let prefix = textEncoderKeys2 != nil ? "__te2" : ""
        guard
          let upTensor = store.read(
            prefix + "__text_model__[\(name)]__up__", codec: [.q6p, .q8p, .ezm7, .externalData]),
          let downTensor = store.read(
            prefix + "__text_model__[\(name)]__down__", codec: [.q6p, .q8p, .ezm7, .externalData])
        else { continue }
        var components = key.split(separator: ".")
        guard components.count > 3 else { continue }
        components.removeFirst(2)
        components.removeLast()
        let loraPrefix =
          (textEncoderKeys2 != nil ? "lora_te2_" : "lora_te_") + components.joined(separator: "_")
        // Generate lora name. For text encoder, that is lora_te_text_model_xxx.lora_up.weight, alpha, lora_down.weight
        var tensorDescriptor = [String: Any]()
        var bytesPerElement: Int
        switch upTensor.dataType {
        case .Float16:
          tensorDescriptor["dtype"] = "F16"
          bytesPerElement = 2
        case .Float32:
          tensorDescriptor["dtype"] = "F32"
          bytesPerElement = 4
        default:
          bytesPerElement = 0
        }
        tensorDescriptor["shape"] = [Int](upTensor.shape)
        let upSize = upTensor.shape.reduce(bytesPerElement, *)
        tensorDescriptor["data_offsets"] = [offset, offset + upSize]
        offset += upSize
        json[loraPrefix + ".lora_up.weight"] = tensorDescriptor
        switch downTensor.dataType {
        case .Float16:
          tensorDescriptor["dtype"] = "F16"
          bytesPerElement = 2
        case .Float32:
          tensorDescriptor["dtype"] = "F32"
          bytesPerElement = 4
        default:
          bytesPerElement = 0
        }
        tensorDescriptor["shape"] = [Int](downTensor.shape)
        let downSize = downTensor.shape.reduce(bytesPerElement, *)
        tensorDescriptor["data_offsets"] = [offset, offset + downSize]
        offset += downSize
        json[loraPrefix + ".lora_down.weight"] = tensorDescriptor
        tensorDescriptor["shape"] = [Int]()
        tensorDescriptor["data_offsets"] = [offset, offset + bytesPerElement]
        offset += bytesPerElement
        json[loraPrefix + ".alpha"] = tensorDescriptor
        progress(i + 1, total)
      }
      progressed += textEncoderKeys.count
      if let textEncoderKeys2 = textEncoderKeys2,
        let textEncoderKeysMapping2 = textEncoderKeysMapping2
      {
        // Need to save for CLIP text model too.
        for (i, key) in textEncoderKeys2.enumerated() {
          guard key.hasSuffix(".weight") else { continue }
          guard let name = textEncoderKeysMapping2[key]?.first else { continue }
          guard
            let upTensor = store.read(
              "__text_model__[\(name)]__up__", codec: [.q6p, .q8p, .ezm7, .externalData]),
            let downTensor = store.read(
              "__text_model__[\(name)]__down__", codec: [.q6p, .q8p, .ezm7, .externalData])
          else { continue }
          var components = key.split(separator: ".")
          guard components.count > 3 else { continue }
          components.removeFirst(2)
          components.removeLast()
          let loraPrefix = "lora_te1_" + components.joined(separator: "_")
          // Generate lora name. For text encoder, that is lora_te_text_model_xxx.lora_up.weight, alpha, lora_down.weight
          var tensorDescriptor = [String: Any]()
          var bytesPerElement: Int
          switch upTensor.dataType {
          case .Float16:
            tensorDescriptor["dtype"] = "F16"
            bytesPerElement = 2
          case .Float32:
            tensorDescriptor["dtype"] = "F32"
            bytesPerElement = 4
          default:
            bytesPerElement = 0
          }
          tensorDescriptor["shape"] = [Int](upTensor.shape)
          let upSize = upTensor.shape.reduce(bytesPerElement, *)
          tensorDescriptor["data_offsets"] = [offset, offset + upSize]
          offset += upSize
          json[loraPrefix + ".lora_up.weight"] = tensorDescriptor
          switch downTensor.dataType {
          case .Float16:
            tensorDescriptor["dtype"] = "F16"
            bytesPerElement = 2
          case .Float32:
            tensorDescriptor["dtype"] = "F32"
            bytesPerElement = 4
          default:
            bytesPerElement = 0
          }
          tensorDescriptor["shape"] = [Int](downTensor.shape)
          let downSize = downTensor.shape.reduce(bytesPerElement, *)
          tensorDescriptor["data_offsets"] = [offset, offset + downSize]
          offset += downSize
          json[loraPrefix + ".lora_down.weight"] = tensorDescriptor
          tensorDescriptor["shape"] = [Int]()
          tensorDescriptor["data_offsets"] = [offset, offset + bytesPerElement]
          offset += bytesPerElement
          json[loraPrefix + ".alpha"] = tensorDescriptor
          progress(progressed + i + 1, total)
        }
        progressed += textEncoderKeys2.count
      }
      for (i, key) in modelKeys.enumerated() {
        guard key.hasSuffix(".weight") else { continue }
        guard let names = modelKeysMapping[key] else { continue }
        guard let loraPrefix = loraPrefix(forModelKey: key) else { continue }
        if names.count > 1 {
          var upDataType: DataType? = nil
          var upShape: [Int]? = nil
          var downDataType: DataType? = nil
          var downShape: [Int]? = nil
          for name in names {
            guard
              let upTensor = store.read(
                "__\(modelPrefix)__[\(name)]__up__", codec: [.q6p, .q8p, .ezm7, .externalData]),
              let downTensor = store.read(
                "__\(modelPrefix)__[\(name)]__down__", codec: [.q6p, .q8p, .ezm7, .externalData])
            else {
              continue
            }
            upDataType = upTensor.dataType
            if let oldShape = upShape {
              upShape = zip(oldShape, upTensor.shape).enumerated().map {
                ($0 == 0 && names.format == .O) || $0 == 1 ? $1.0 + $1.1 : $1.0
              }
            } else {
              upShape = [Int](upTensor.shape)
            }
            downDataType = downTensor.dataType
            if let oldShape = downShape {
              // If the format is I, we need both up and down to be diag.
              downShape = zip(oldShape, downTensor.shape).enumerated().map {
                $0 <= (names.format == .I ? 1 : 0) ? $1.0 + $1.1 : $1.0
              }
            } else {
              downShape = [Int](downTensor.shape)
            }
          }
          if let upDataType = upDataType, let upShape = upShape, let downDataType = downDataType,
            let downShape = downShape
          {
            var tensorDescriptor = [String: Any]()
            var bytesPerElement: Int
            switch upDataType {
            case .Float16:
              tensorDescriptor["dtype"] = "F16"
              bytesPerElement = 2
            case .Float32:
              tensorDescriptor["dtype"] = "F32"
              bytesPerElement = 4
            default:
              bytesPerElement = 0
            }
            if key.contains(".in_layers.") || key.contains(".out_layers.")
              || key.contains(".out.0.")
            {
              tensorDescriptor["shape"] = upShape.compactMap { $0 == 1 ? nil : $0 }
            } else {
              tensorDescriptor["shape"] = upShape
            }
            let upSize = upShape.reduce(bytesPerElement, *)
            tensorDescriptor["data_offsets"] = [offset, offset + upSize]
            offset += upSize
            json[loraPrefix + ".lora_up.weight"] = tensorDescriptor
            switch downDataType {
            case .Float16:
              tensorDescriptor["dtype"] = "F16"
              bytesPerElement = 2
            case .Float32:
              tensorDescriptor["dtype"] = "F32"
              bytesPerElement = 4
            default:
              bytesPerElement = 0
            }
            if key.contains(".in_layers.") || key.contains(".out_layers.")
              || key.contains(".out.0.")
            {
              tensorDescriptor["shape"] = downShape.compactMap { $0 == 1 ? nil : $0 }
            } else {
              tensorDescriptor["shape"] = downShape
            }
            let downSize = downShape.reduce(bytesPerElement, *)
            tensorDescriptor["data_offsets"] = [offset, offset + downSize]
            offset += downSize
            json[loraPrefix + ".lora_down.weight"] = tensorDescriptor
            tensorDescriptor["shape"] = [Int]()
            tensorDescriptor["data_offsets"] = [offset, offset + bytesPerElement]
            offset += bytesPerElement
            json[loraPrefix + ".alpha"] = tensorDescriptor
          }
        } else if let name = names.first {
          guard
            let upTensor = store.read(
              "__\(modelPrefix)__[\(name)]__up__", codec: [.q6p, .q8p, .ezm7, .externalData]),
            let downTensor = store.read(
              "__\(modelPrefix)__[\(name)]__down__", codec: [.q6p, .q8p, .ezm7, .externalData])
          else {
            continue
          }
          var tensorDescriptor = [String: Any]()
          var bytesPerElement: Int
          switch upTensor.dataType {
          case .Float16:
            tensorDescriptor["dtype"] = "F16"
            bytesPerElement = 2
          case .Float32:
            tensorDescriptor["dtype"] = "F32"
            bytesPerElement = 4
          default:
            bytesPerElement = 0
          }
          if key.contains(".in_layers.") || key.contains(".out_layers.") || key.contains(".out.0.")
          {
            tensorDescriptor["shape"] = upTensor.shape.compactMap { $0 == 1 ? nil : $0 }
          } else {
            tensorDescriptor["shape"] = [Int](upTensor.shape)
          }
          let upSize = upTensor.shape.reduce(bytesPerElement, *)
          tensorDescriptor["data_offsets"] = [offset, offset + upSize]
          offset += upSize
          json[loraPrefix + ".lora_up.weight"] = tensorDescriptor
          switch downTensor.dataType {
          case .Float16:
            tensorDescriptor["dtype"] = "F16"
            bytesPerElement = 2
          case .Float32:
            tensorDescriptor["dtype"] = "F32"
            bytesPerElement = 4
          default:
            bytesPerElement = 0
          }
          if key.contains(".in_layers.") || key.contains(".out_layers.") || key.contains(".out.0.")
          {
            tensorDescriptor["shape"] = downTensor.shape.compactMap { $0 == 1 ? nil : $0 }
          } else {
            tensorDescriptor["shape"] = [Int](downTensor.shape)
          }
          let downSize = downTensor.shape.reduce(bytesPerElement, *)
          tensorDescriptor["data_offsets"] = [offset, offset + downSize]
          offset += downSize
          json[loraPrefix + ".lora_down.weight"] = tensorDescriptor
          tensorDescriptor["shape"] = [Int]()
          tensorDescriptor["data_offsets"] = [offset, offset + bytesPerElement]
          offset += bytesPerElement
          json[loraPrefix + ".alpha"] = tensorDescriptor
        }
        progress(progressed + i + 1, total)
      }
      progressed += modelKeys.count
      if !modelKeys2.isEmpty {
        for (i, key) in modelKeys2.enumerated() {
          guard key.hasSuffix(".weight") else { continue }
          guard let names = modelKeysMapping2[key] else { continue }
          guard let loraPrefix = loraPrefix(forModelKey: key) else { continue }
          if names.count > 1 {
            var upDataType: DataType? = nil
            var upShape: [Int]? = nil
            var downDataType: DataType? = nil
            var downShape: [Int]? = nil
            for name in names {
              guard
                let upTensor = store.read(
                  "__\(modelFixedPrefix)__[\(name)]__up__",
                  codec: [.q6p, .q8p, .ezm7, .externalData]),
                let downTensor = store.read(
                  "__\(modelFixedPrefix)__[\(name)]__down__",
                  codec: [.q6p, .q8p, .ezm7, .externalData])
              else {
                continue
              }
              upDataType = upTensor.dataType
              if let oldShape = upShape {
                upShape = zip(oldShape, upTensor.shape).enumerated().map {
                  ($0 == 0 && names.format == .O) || $0 == 1 ? $1.0 + $1.1 : $1.0
                }
              } else {
                upShape = [Int](upTensor.shape)
              }
              downDataType = downTensor.dataType
              if let oldShape = downShape {
                // If the format is I, we need both up and down to be diag.
                downShape = zip(oldShape, downTensor.shape).enumerated().map {
                  $0 <= (names.format == .I ? 1 : 0) ? $1.0 + $1.1 : $1.0
                }
              } else {
                downShape = [Int](downTensor.shape)
              }
            }
            if let upDataType = upDataType, let upShape = upShape, let downDataType = downDataType,
              let downShape = downShape
            {
              var tensorDescriptor = [String: Any]()
              var bytesPerElement: Int
              switch upDataType {
              case .Float16:
                tensorDescriptor["dtype"] = "F16"
                bytesPerElement = 2
              case .Float32:
                tensorDescriptor["dtype"] = "F32"
                bytesPerElement = 4
              default:
                bytesPerElement = 0
              }
              if key.contains(".in_layers.") || key.contains(".out_layers.")
                || key.contains(".out.0.")
              {
                tensorDescriptor["shape"] = upShape.compactMap { $0 == 1 ? nil : $0 }
              } else {
                tensorDescriptor["shape"] = upShape
              }
              let upSize = upShape.reduce(bytesPerElement, *)
              tensorDescriptor["data_offsets"] = [offset, offset + upSize]
              offset += upSize
              json[loraPrefix + ".lora_up.weight"] = tensorDescriptor
              switch downDataType {
              case .Float16:
                tensorDescriptor["dtype"] = "F16"
                bytesPerElement = 2
              case .Float32:
                tensorDescriptor["dtype"] = "F32"
                bytesPerElement = 4
              default:
                bytesPerElement = 0
              }
              if key.contains(".in_layers.") || key.contains(".out_layers.")
                || key.contains(".out.0.")
              {
                tensorDescriptor["shape"] = downShape.compactMap { $0 == 1 ? nil : $0 }
              } else {
                tensorDescriptor["shape"] = downShape
              }
              let downSize = downShape.reduce(bytesPerElement, *)
              tensorDescriptor["data_offsets"] = [offset, offset + downSize]
              offset += downSize
              json[loraPrefix + ".lora_down.weight"] = tensorDescriptor
              tensorDescriptor["shape"] = [Int]()
              tensorDescriptor["data_offsets"] = [offset, offset + bytesPerElement]
              offset += bytesPerElement
              json[loraPrefix + ".alpha"] = tensorDescriptor
            }
          } else if let name = names.first {
            guard
              let upTensor = store.read(
                "__\(modelFixedPrefix)__[\(name)]__up__", codec: [.q6p, .q8p, .ezm7, .externalData]),
              let downTensor = store.read(
                "__\(modelFixedPrefix)__[\(name)]__down__",
                codec: [.q6p, .q8p, .ezm7, .externalData])
            else {
              continue
            }
            var tensorDescriptor = [String: Any]()
            var bytesPerElement: Int
            switch upTensor.dataType {
            case .Float16:
              tensorDescriptor["dtype"] = "F16"
              bytesPerElement = 2
            case .Float32:
              tensorDescriptor["dtype"] = "F32"
              bytesPerElement = 4
            default:
              bytesPerElement = 0
            }
            if key.contains(".in_layers.") || key.contains(".out_layers.")
              || key.contains(".out.0.")
            {
              tensorDescriptor["shape"] = upTensor.shape.compactMap { $0 == 1 ? nil : $0 }
            } else {
              tensorDescriptor["shape"] = [Int](upTensor.shape)
            }
            let upSize = upTensor.shape.reduce(bytesPerElement, *)
            tensorDescriptor["data_offsets"] = [offset, offset + upSize]
            offset += upSize
            json[loraPrefix + ".lora_up.weight"] = tensorDescriptor
            switch downTensor.dataType {
            case .Float16:
              tensorDescriptor["dtype"] = "F16"
              bytesPerElement = 2
            case .Float32:
              tensorDescriptor["dtype"] = "F32"
              bytesPerElement = 4
            default:
              bytesPerElement = 0
            }
            if key.contains(".in_layers.") || key.contains(".out_layers.")
              || key.contains(".out.0.")
            {
              tensorDescriptor["shape"] = downTensor.shape.compactMap { $0 == 1 ? nil : $0 }
            } else {
              tensorDescriptor["shape"] = [Int](downTensor.shape)
            }
            let downSize = downTensor.shape.reduce(bytesPerElement, *)
            tensorDescriptor["data_offsets"] = [offset, offset + downSize]
            offset += downSize
            json[loraPrefix + ".lora_down.weight"] = tensorDescriptor
            tensorDescriptor["shape"] = [Int]()
            tensorDescriptor["data_offsets"] = [offset, offset + bytesPerElement]
            offset += bytesPerElement
            json[loraPrefix + ".alpha"] = tensorDescriptor
          }
          progress(progressed + i + 1, total)
        }
        progressed += modelKeys2.count
      }
    }
    let data = try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
    let w = fopen(outputFile, "wb")
    var size = data.count
    fwrite(&size, 8, 1, w)
    let _ = data.withUnsafeBytes {
      fwrite($0.baseAddress, 1, $0.count, w)
    }
    graph.openStore(LoRAZoo.filePathForModelDownloaded(file), flags: .readOnly) {
      store in
      if let tensor = store.read("string_to_param") {
        switch tensor.dataType {
        case .Float16:
          #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
            let _ = Tensor<Float16>(tensor).toCPU().withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          #endif
        case .Float32:
          let _ = Tensor<Float32>(tensor).toCPU().withUnsafeBytes {
            fwrite($0.baseAddress, 1, $0.count, w)
          }
        default:
          break
        }
      } else if let tensorClipG = store.read("string_to_param_clip_g"),
        let tensorClipL = store.read("string_to_param_clip_l")
      {
        switch tensorClipG.dataType {
        case .Float16:
          #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
            let _ = Tensor<Float16>(tensorClipG).toCPU().withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          #endif
        case .Float32:
          let _ = Tensor<Float32>(tensorClipG).toCPU().withUnsafeBytes {
            fwrite($0.baseAddress, 1, $0.count, w)
          }
        default:
          break
        }
        switch tensorClipL.dataType {
        case .Float16:
          #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
            let _ = Tensor<Float16>(tensorClipL).toCPU().withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          #endif
        case .Float32:
          let _ = Tensor<Float32>(tensorClipL).toCPU().withUnsafeBytes {
            fwrite($0.baseAddress, 1, $0.count, w)
          }
        default:
          break
        }
      } else if let textualInversionExport = textualInversionExport,
        let tensor = store.read(textualInversionExport.storeKey)
      {
        switch tensor.dataType {
        case .Float16:
          #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
            let _ = Tensor<Float16>(tensor).toCPU().withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          #endif
        case .Float32:
          let _ = Tensor<Float32>(tensor).toCPU().withUnsafeBytes {
            fwrite($0.baseAddress, 1, $0.count, w)
          }
        default:
          break
        }
      }
      func exportUpTensor<T: TensorNumeric>(
        _ tensor: Tensor<T>, for names: ModelWeightElement, at index: Int = 0
      ) -> Tensor<T> {
        var tensor = tensor.toCPU()
        switch names.format {
        case .O:
          let numberOfHeads = names.numberOfHeads
          let headDimension = names.headDimension
          if names.interleavedIndices.contains(index), numberOfHeads > 0, headDimension > 0 {
            // Inverse of TensorDescriptor's interleaved .O write transform.
            if let dimension = names.interleavedDimension, dimension != headDimension {
              precondition(dimension > 0 && dimension <= headDimension && dimension % 2 == 0)
              let rows = tensor.reshaped(
                format: tensor.format, shape: [numberOfHeads, headDimension, -1])
              let width = rows.shape[2]
              var result = rows.copied()
              result[0..<numberOfHeads, 0..<dimension, 0..<width] = graph.withNoGrad {
                graph.variable(
                  rows[0..<numberOfHeads, 0..<dimension, 0..<width].copied().reshaped(
                    format: tensor.format, shape: [numberOfHeads, dimension / 2, 2, width]
                  ).toGPU()
                ).transposed(1, 2).reshaped(
                  format: tensor.format, shape: [numberOfHeads, dimension, width]
                ).toCPU().rawValue
              }
              return result.reshaped(format: tensor.format, shape: tensor.shape)
            }
            tensor = graph.withNoGrad {
              Tensor<T>(
                from: graph.variable(
                  tensor.reshaped(
                    format: tensor.format,
                    shape: [numberOfHeads, headDimension / 2, 2, -1]
                  ).toGPU()
                ).transposed(1, 2).reshaped(
                  format: tensor.format, shape: [numberOfHeads * headDimension, -1]
                ).toCPU().rawValue)
            }
          }
        case .I:
          break
        }
        return tensor
      }
      for (i, key) in textEncoderKeys.enumerated() {
        guard key.hasSuffix(".weight") else { continue }
        guard let name = textEncoderKeysMapping[key]?.first else {
          continue
        }
        let prefix = textEncoderKeys2 != nil ? "__te2" : ""
        guard
          let upTensor = store.read(
            prefix + "__text_model__[\(name)]__up__", codec: [.q6p, .q8p, .ezm7, .externalData]),
          let downTensor = store.read(
            prefix + "__text_model__[\(name)]__down__", codec: [.q6p, .q8p, .ezm7, .externalData])
        else { continue }
        switch upTensor.dataType {
        case .Float16:
          #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
            let _ = Tensor<Float16>(upTensor).toCPU().withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          #endif
        case .Float32:
          let _ = Tensor<Float32>(upTensor).toCPU().withUnsafeBytes {
            fwrite($0.baseAddress, 1, $0.count, w)
          }
        default:
          break
        }
        switch downTensor.dataType {
        case .Float16:
          #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
            let _ = Tensor<Float16>(downTensor).toCPU().withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
            var alpha = Tensor<Float16>(.CPU, .C(1))
            alpha[0] = Float16(downTensor.shape[0])
            let _ = alpha.withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          #endif
        case .Float32:
          let _ = Tensor<Float32>(downTensor).toCPU().withUnsafeBytes {
            fwrite($0.baseAddress, 1, $0.count, w)
          }
          var alpha = Tensor<Float32>(.CPU, .C(1))
          alpha[0] = Float32(downTensor.shape[0])
          let _ = alpha.withUnsafeBytes {
            fwrite($0.baseAddress, 1, $0.count, w)
          }
        default:
          break
        }
        progress(progressed + i + 1, total)
      }
      progressed += textEncoderKeys.count
      if let textEncoderKeys2 = textEncoderKeys2,
        let textEncoderKeysMapping2 = textEncoderKeysMapping2
      {
        for (i, key) in textEncoderKeys2.enumerated() {
          guard key.hasSuffix(".weight") else { continue }
          guard let name = textEncoderKeysMapping2[key]?.first else {
            continue
          }
          guard let upTensor = store.read("__text_model__[\(name)]__up__"),
            let downTensor = store.read("__text_model__[\(name)]__down__")
          else { continue }
          switch upTensor.dataType {
          case .Float16:
            #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
              let _ = Tensor<Float16>(upTensor).toCPU().withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
            #endif
          case .Float32:
            let _ = Tensor<Float32>(upTensor).toCPU().withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          default:
            break
          }
          switch downTensor.dataType {
          case .Float16:
            #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
              let _ = Tensor<Float16>(downTensor).toCPU().withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
              var alpha = Tensor<Float16>(.CPU, .C(1))
              alpha[0] = Float16(downTensor.shape[0])
              let _ = alpha.withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
            #endif
          case .Float32:
            let _ = Tensor<Float32>(downTensor).toCPU().withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
            var alpha = Tensor<Float32>(.CPU, .C(1))
            alpha[0] = Float32(downTensor.shape[0])
            let _ = alpha.withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          default:
            break
          }
          progress(progressed + i + 1, total)
        }
        progressed += textEncoderKeys2.count
      }
      for (i, key) in modelKeys.enumerated() {
        guard key.hasSuffix(".weight") else { continue }
        guard let names = modelKeysMapping[key] else {
          continue
        }
        if names.count > 1 {
          var upTensors = [AnyTensor]()
          var upDataType: DataType? = nil
          var upShape: [Int]? = nil
          for name in names {
            guard
              let upTensor = store.read(
                "__\(modelPrefix)__[\(name)]__up__", codec: [.q6p, .q8p, .ezm7, .externalData])
            else {
              continue
            }
            upDataType = upTensor.dataType
            upTensors.append(upTensor)
            if let oldShape = upShape {
              upShape = zip(oldShape, upTensor.shape).enumerated().map {
                ($0 == 0 && names.format == .O) || $0 == 1 ? $1.0 + $1.1 : $1.0
              }
            } else {
              upShape = [Int](upTensor.shape)
            }
          }
          if let upDataType = upDataType, let upShape = upShape, upTensors.count > 1 {
            switch upDataType {
            case .Float16:
              #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                var tensor = Tensor<Float16>(.CPU, .NC(upShape[0], upShape[1]))
                switch names.format {
                case .O:
                  let _ = tensor.withUnsafeMutableBytes {
                    memset($0.baseAddress!, 0, $0.count)
                  }
                  var shapeStart0 = 0
                  var shapeStart1 = 0
                  for (j, upTensor) in upTensors.enumerated() {
                    let shape = upTensor.shape
                    tensor[
                      shapeStart0..<(shapeStart0 + shape[0]), shapeStart1..<(shapeStart1 + shape[1])
                    ] =
                      exportUpTensor(Tensor<Float16>(from: upTensor), for: names, at: j)
                    shapeStart0 = shapeStart0 + shape[0]
                    shapeStart1 = shapeStart1 + shape[1]
                  }
                case .I:
                  var shapeStart1 = 0
                  for (j, upTensor) in upTensors.enumerated() {
                    let shape = upTensor.shape
                    tensor[0..<shape[0], shapeStart1..<(shapeStart1 + shape[1])] =
                      exportUpTensor(Tensor<Float16>(from: upTensor), for: names, at: j)
                    shapeStart1 = shapeStart1 + shape[1]
                  }
                }
                let _ = tensor.withUnsafeBytes {
                  fwrite($0.baseAddress, 1, $0.count, w)
                }
              #endif
            case .Float32:
              var tensor = Tensor<Float32>(.CPU, .NC(upShape[0], upShape[1]))
              switch names.format {
              case .O:
                let _ = tensor.withUnsafeMutableBytes {
                  memset($0.baseAddress!, 0, $0.count)
                }
                var shapeStart0 = 0
                var shapeStart1 = 0
                for (j, upTensor) in upTensors.enumerated() {
                  let shape = upTensor.shape
                  tensor[
                    shapeStart0..<(shapeStart0 + shape[0]), shapeStart1..<(shapeStart1 + shape[1])] =
                    exportUpTensor(Tensor<Float32>(from: upTensor), for: names, at: j)
                  shapeStart0 = shapeStart0 + shape[0]
                  shapeStart1 = shapeStart1 + shape[1]
                }
              case .I:
                var shapeStart1 = 0
                for (j, upTensor) in upTensors.enumerated() {
                  let shape = upTensor.shape
                  tensor[0..<shape[0], shapeStart1..<(shapeStart1 + shape[1])] =
                    exportUpTensor(Tensor<Float32>(from: upTensor), for: names, at: j)
                  shapeStart1 = shapeStart1 + shape[1]
                }
              }
              let _ = tensor.withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
            default:
              break
            }
          }
          var dim: Int = 0
          var downTensors = [AnyTensor]()
          var downDataType: DataType? = nil
          var downShape: [Int]? = nil
          for name in names {
            guard
              let downTensor = store.read(
                "__\(modelPrefix)__[\(name)]__down__", codec: [.q6p, .q8p, .ezm7, .externalData])
            else {
              continue
            }
            downDataType = downTensor.dataType
            dim += downTensor.shape[0]
            downTensors.append(downTensor)
            if let oldShape = downShape {
              downShape = zip(oldShape, downTensor.shape).enumerated().map {
                $0 <= (names.format == .I ? 1 : 0) ? $1.0 + $1.1 : $1.0
              }
            } else {
              downShape = [Int](downTensor.shape)
            }
          }
          switch names.format {
          case .I:
            if let downDataType = downDataType, let downShape = downShape, downTensors.count > 1 {
              switch downDataType {
              case .Float16:
                #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                  var tensor = Tensor<Float16>(.CPU, .NC(downShape[0], downShape[1]))
                  let _ = tensor.withUnsafeMutableBytes {
                    memset($0.baseAddress!, 0, $0.count)
                  }
                  var shapeStart0 = 0
                  var shapeStart1 = 0
                  for downTensor in downTensors {
                    let shape = downTensor.shape
                    tensor[
                      shapeStart0..<(shapeStart0 + shape[0]), shapeStart1..<(shapeStart1 + shape[1])
                    ] =
                      Tensor<Float16>(from: downTensor).toCPU()
                    shapeStart0 = shapeStart0 + shape[0]
                    shapeStart1 = shapeStart1 + shape[1]
                  }
                  let _ = tensor.withUnsafeBytes {
                    fwrite($0.baseAddress, 1, $0.count, w)
                  }
                #endif
              case .Float32:
                var tensor = Tensor<Float32>(.CPU, .NC(downShape[0], downShape[1]))
                let _ = tensor.withUnsafeMutableBytes {
                  memset($0.baseAddress!, 0, $0.count)
                }
                var shapeStart0 = 0
                var shapeStart1 = 0
                for downTensor in downTensors {
                  let shape = downTensor.shape
                  tensor[
                    shapeStart0..<(shapeStart0 + shape[0]), shapeStart1..<(shapeStart1 + shape[1])] =
                    Tensor<Float32>(from: downTensor).toCPU()
                  shapeStart0 = shapeStart0 + shape[0]
                  shapeStart1 = shapeStart1 + shape[1]
                }
                let _ = tensor.withUnsafeBytes {
                  fwrite($0.baseAddress, 1, $0.count, w)
                }
              default:
                break
              }
            }
          case .O:
            for downTensor in downTensors {
              switch downTensor.dataType {
              case .Float16:
                #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                  let _ = Tensor<Float16>(downTensor).toCPU().withUnsafeBytes {
                    fwrite($0.baseAddress, 1, $0.count, w)
                  }
                #endif
              case .Float32:
                let _ = Tensor<Float32>(downTensor).toCPU().withUnsafeBytes {
                  fwrite($0.baseAddress, 1, $0.count, w)
                }
              default:
                break
              }
            }
          }
          if let downDataType = downDataType, dim > 0 {
            switch downDataType {
            case .Float16:
              #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                var alpha = Tensor<Float16>(.CPU, .C(1))
                alpha[0] = Float16(dim)
                let _ = alpha.withUnsafeBytes {
                  fwrite($0.baseAddress, 1, $0.count, w)
                }
              #endif
            case .Float32:
              var alpha = Tensor<Float32>(.CPU, .C(1))
              alpha[0] = Float32(dim)
              let _ = alpha.withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
            default:
              break
            }
          }
        } else if let name = names.first {
          guard
            let upTensor = store.read(
              "__\(modelPrefix)__[\(name)]__up__", codec: [.q6p, .q8p, .ezm7, .externalData]),
            let downTensor = store.read(
              "__\(modelPrefix)__[\(name)]__down__", codec: [.q6p, .q8p, .ezm7, .externalData])
          else {
            continue
          }
          switch upTensor.dataType {
          case .Float16:
            #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
              let _ = exportUpTensor(Tensor<Float16>(upTensor), for: names).withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
            #endif
          case .Float32:
            let _ = exportUpTensor(Tensor<Float32>(upTensor), for: names).withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          default:
            break
          }
          switch downTensor.dataType {
          case .Float16:
            #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
              let _ = Tensor<Float16>(downTensor).toCPU().withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
              var alpha = Tensor<Float16>(.CPU, .C(1))
              alpha[0] = Float16(downTensor.shape[0])
              let _ = alpha.withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
            #endif
          case .Float32:
            let _ = Tensor<Float32>(downTensor).toCPU().withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
            var alpha = Tensor<Float32>(.CPU, .C(1))
            alpha[0] = Float32(downTensor.shape[0])
            let _ = alpha.withUnsafeBytes {
              fwrite($0.baseAddress, 1, $0.count, w)
            }
          default:
            break
          }
        }
        progress(progressed + i + 1, total)
      }
      progressed += modelKeys.count
      if !modelKeys2.isEmpty {
        for (i, key) in modelKeys2.enumerated() {
          guard key.hasSuffix(".weight") else { continue }
          guard let names = modelKeysMapping2[key] else {
            continue
          }
          if names.count > 1 {
            var upTensors = [AnyTensor]()
            var upDataType: DataType? = nil
            var upShape: [Int]? = nil
            for name in names {
              guard
                let upTensor = store.read(
                  "__\(modelFixedPrefix)__[\(name)]__up__",
                  codec: [.q6p, .q8p, .ezm7, .externalData])
              else {
                continue
              }
              upDataType = upTensor.dataType
              upTensors.append(upTensor)
              if let oldShape = upShape {
                upShape = zip(oldShape, upTensor.shape).enumerated().map {
                  ($0 == 0 && names.format == .O) || $0 == 1 ? $1.0 + $1.1 : $1.0
                }
              } else {
                upShape = [Int](upTensor.shape)
              }
            }
            if let upDataType = upDataType, let upShape = upShape, upTensors.count > 1 {
              switch upDataType {
              case .Float16:
                #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                  var tensor = Tensor<Float16>(.CPU, .NC(upShape[0], upShape[1]))
                  switch names.format {
                  case .O:
                    let _ = tensor.withUnsafeMutableBytes {
                      memset($0.baseAddress!, 0, $0.count)
                    }
                    var shapeStart0 = 0
                    var shapeStart1 = 0
                    for (j, upTensor) in upTensors.enumerated() {
                      let shape = upTensor.shape
                      tensor[
                        shapeStart0..<(shapeStart0 + shape[0]),
                        shapeStart1..<(shapeStart1 + shape[1])
                      ] =
                        exportUpTensor(Tensor<Float16>(from: upTensor), for: names, at: j)
                      shapeStart0 = shapeStart0 + shape[0]
                      shapeStart1 = shapeStart1 + shape[1]
                    }
                  case .I:
                    var shapeStart1 = 0
                    for (j, upTensor) in upTensors.enumerated() {
                      let shape = upTensor.shape
                      tensor[0..<shape[0], shapeStart1..<(shapeStart1 + shape[1])] =
                        exportUpTensor(Tensor<Float16>(from: upTensor), for: names, at: j)
                      shapeStart1 = shapeStart1 + shape[1]
                    }
                  }
                  let _ = tensor.withUnsafeBytes {
                    fwrite($0.baseAddress, 1, $0.count, w)
                  }
                #endif
              case .Float32:
                var tensor = Tensor<Float32>(.CPU, .NC(upShape[0], upShape[1]))
                switch names.format {
                case .O:
                  let _ = tensor.withUnsafeMutableBytes {
                    memset($0.baseAddress!, 0, $0.count)
                  }
                  var shapeStart0 = 0
                  var shapeStart1 = 0
                  for (j, upTensor) in upTensors.enumerated() {
                    let shape = upTensor.shape
                    tensor[
                      shapeStart0..<(shapeStart0 + shape[0]), shapeStart1..<(shapeStart1 + shape[1])
                    ] =
                      exportUpTensor(Tensor<Float32>(from: upTensor), for: names, at: j)
                    shapeStart0 = shapeStart0 + shape[0]
                    shapeStart1 = shapeStart1 + shape[1]
                  }
                case .I:
                  var shapeStart1 = 0
                  for (j, upTensor) in upTensors.enumerated() {
                    let shape = upTensor.shape
                    tensor[0..<shape[0], shapeStart1..<(shapeStart1 + shape[1])] =
                      exportUpTensor(Tensor<Float32>(from: upTensor), for: names, at: j)
                    shapeStart1 = shapeStart1 + shape[1]
                  }
                }
                let _ = tensor.withUnsafeBytes {
                  fwrite($0.baseAddress, 1, $0.count, w)
                }
              default:
                break
              }
            }
            var dim: Int = 0
            var downTensors = [AnyTensor]()
            var downDataType: DataType? = nil
            var downShape: [Int]? = nil
            for name in names {
              guard
                let downTensor = store.read(
                  "__\(modelFixedPrefix)__[\(name)]__down__",
                  codec: [.q6p, .q8p, .ezm7, .externalData])
              else {
                continue
              }
              downDataType = downTensor.dataType
              dim += downTensor.shape[0]
              downTensors.append(downTensor)
              if let oldShape = downShape {
                downShape = zip(oldShape, downTensor.shape).enumerated().map {
                  $0 <= (names.format == .I ? 1 : 0) ? $1.0 + $1.1 : $1.0
                }
              } else {
                downShape = [Int](downTensor.shape)
              }
            }
            switch names.format {
            case .I:
              if let downDataType = downDataType, let downShape = downShape, downTensors.count > 1 {
                switch downDataType {
                case .Float16:
                  #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                    var tensor = Tensor<Float16>(.CPU, .NC(downShape[0], downShape[1]))
                    let _ = tensor.withUnsafeMutableBytes {
                      memset($0.baseAddress!, 0, $0.count)
                    }
                    var shapeStart0 = 0
                    var shapeStart1 = 0
                    for downTensor in downTensors {
                      let shape = downTensor.shape
                      tensor[
                        shapeStart0..<(shapeStart0 + shape[0]),
                        shapeStart1..<(shapeStart1 + shape[1])
                      ] =
                        Tensor<Float16>(from: downTensor).toCPU()
                      shapeStart0 = shapeStart0 + shape[0]
                      shapeStart1 = shapeStart1 + shape[1]
                    }
                    let _ = tensor.withUnsafeBytes {
                      fwrite($0.baseAddress, 1, $0.count, w)
                    }
                  #endif
                case .Float32:
                  var tensor = Tensor<Float32>(.CPU, .NC(downShape[0], downShape[1]))
                  let _ = tensor.withUnsafeMutableBytes {
                    memset($0.baseAddress!, 0, $0.count)
                  }
                  var shapeStart0 = 0
                  var shapeStart1 = 0
                  for downTensor in downTensors {
                    let shape = downTensor.shape
                    tensor[
                      shapeStart0..<(shapeStart0 + shape[0]), shapeStart1..<(shapeStart1 + shape[1])
                    ] =
                      Tensor<Float32>(from: downTensor).toCPU()
                    shapeStart0 = shapeStart0 + shape[0]
                    shapeStart1 = shapeStart1 + shape[1]
                  }
                  let _ = tensor.withUnsafeBytes {
                    fwrite($0.baseAddress, 1, $0.count, w)
                  }
                default:
                  break
                }
              }
            case .O:
              for downTensor in downTensors {
                switch downTensor.dataType {
                case .Float16:
                  #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                    let _ = Tensor<Float16>(downTensor).toCPU().withUnsafeBytes {
                      fwrite($0.baseAddress, 1, $0.count, w)
                    }
                  #endif
                case .Float32:
                  let _ = Tensor<Float32>(downTensor).toCPU().withUnsafeBytes {
                    fwrite($0.baseAddress, 1, $0.count, w)
                  }
                default:
                  break
                }
              }
            }
            if let downDataType = downDataType, dim > 0 {
              switch downDataType {
              case .Float16:
                #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                  var alpha = Tensor<Float16>(.CPU, .C(1))
                  alpha[0] = Float16(dim)
                  let _ = alpha.withUnsafeBytes {
                    fwrite($0.baseAddress, 1, $0.count, w)
                  }
                #endif
              case .Float32:
                var alpha = Tensor<Float32>(.CPU, .C(1))
                alpha[0] = Float32(dim)
                let _ = alpha.withUnsafeBytes {
                  fwrite($0.baseAddress, 1, $0.count, w)
                }
              default:
                break
              }
            }
          } else if let name = names.first {
            guard
              let upTensor = store.read(
                "__\(modelFixedPrefix)__[\(name)]__up__", codec: [.q6p, .q8p, .ezm7, .externalData]),
              let downTensor = store.read(
                "__\(modelFixedPrefix)__[\(name)]__down__",
                codec: [.q6p, .q8p, .ezm7, .externalData])
            else {
              continue
            }
            switch upTensor.dataType {
            case .Float16:
              #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                let _ = exportUpTensor(Tensor<Float16>(upTensor), for: names).withUnsafeBytes {
                  fwrite($0.baseAddress, 1, $0.count, w)
                }
              #endif
            case .Float32:
              let _ = exportUpTensor(Tensor<Float32>(upTensor), for: names).withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
            default:
              break
            }
            switch downTensor.dataType {
            case .Float16:
              #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
                let _ = Tensor<Float16>(downTensor).toCPU().withUnsafeBytes {
                  fwrite($0.baseAddress, 1, $0.count, w)
                }
                var alpha = Tensor<Float16>(.CPU, .C(1))
                alpha[0] = Float16(downTensor.shape[0])
                let _ = alpha.withUnsafeBytes {
                  fwrite($0.baseAddress, 1, $0.count, w)
                }
              #endif
            case .Float32:
              let _ = Tensor<Float32>(downTensor).toCPU().withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
              var alpha = Tensor<Float32>(.CPU, .C(1))
              alpha[0] = Float32(downTensor.shape[0])
              let _ = alpha.withUnsafeBytes {
                fwrite($0.baseAddress, 1, $0.count, w)
              }
            default:
              break
            }
          }
          progress(progressed + i + 1, total)
        }
      }
    }
    fclose(w)
  }
}
