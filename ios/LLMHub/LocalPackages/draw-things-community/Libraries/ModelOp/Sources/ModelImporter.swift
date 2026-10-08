import DataModels
import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
import WeightsCache
import ZIPFoundation

public final class ModelImporter {
  public enum Error: Swift.Error {
    case tensorWritesFailed
    case textEncoder(Swift.Error)
    case autoencoder(Swift.Error)
    case noTextEncoder
  }
  private let filePath: String
  private let modelName: String
  private let isTextEncoderCustomized: Bool
  private let autoencoderFilePath: String?
  private let textEncoderFilePath: String?
  private let textEncoder2FilePath: String?
  private var access: Int = 0
  private var expectedTotalAccess: Int = 0
  private var progress: ((Float) -> Void)? = nil
  public init(
    filePath: String, modelName: String, isTextEncoderCustomized: Bool,
    autoencoderFilePath: String?, textEncoderFilePath: String?,
    textEncoder2FilePath: String?
  ) {
    self.filePath = filePath
    self.modelName = modelName
    self.isTextEncoderCustomized = isTextEncoderCustomized
    self.autoencoderFilePath = autoencoderFilePath
    self.textEncoderFilePath = textEncoderFilePath
    self.textEncoder2FilePath = textEncoder2FilePath
  }

  public func `import`(
    versionCheck: @escaping (ModelVersion) -> Void, progress: @escaping (Float) -> Void
  ) throws -> ([String], ModelVersion, SamplerModifier, InspectionResult) {
    self.progress = progress
    defer { self.progress = nil }
    Interpreter.inflateInterrupter = self.interrupt
    defer { Interpreter.inflateInterrupter = nil }
    self.access = 0
    return try self.internalImport(versionCheck: versionCheck)
  }

  private func interrupt() -> Bool {
    access += 1
    progress?(0.05 + Float(access) / Float(max(expectedTotalAccess, 1)) * 0.95)
    return false
  }

  public struct InspectionResult {
    public var version: ModelVersion
    public var archive: TensorArchive
    public var stateDict: [String: TensorDescriptor]
    public var modifier: SamplerModifier
    public var inputChannels: Int
    public var isDiffusersFormat: Bool
    public var hasEncoderHidProj: Bool
    public var hasGuidanceEmbed: Bool
    public var numberOfTensors: Int
    public var qkNorm: Bool
    public var dualAttentionLayers: [Int]
    public var distilledGuidanceLayers: Int
    public init(
      version: ModelVersion, archive: TensorArchive, stateDict: [String: TensorDescriptor],
      modifier: SamplerModifier, inputChannels: Int, isDiffusersFormat: Bool,
      hasEncoderHidProj: Bool, hasGuidanceEmbed: Bool, qkNorm: Bool, dualAttentionLayers: [Int],
      distilledGuidanceLayers: Int, numberOfTensors: Int
    ) {
      self.version = version
      self.archive = archive
      self.stateDict = stateDict
      self.modifier = modifier
      self.inputChannels = inputChannels
      self.isDiffusersFormat = isDiffusersFormat
      self.hasEncoderHidProj = hasEncoderHidProj
      self.hasGuidanceEmbed = hasGuidanceEmbed
      self.qkNorm = qkNorm
      self.dualAttentionLayers = dualAttentionLayers
      self.distilledGuidanceLayers = distilledGuidanceLayers
      self.numberOfTensors = numberOfTensors
    }
  }

  public func inspect() throws -> InspectionResult {
    var archive: TensorArchive
    var stateDict: [String: TensorDescriptor]
    if let safeTensors = SafeTensors(url: URL(fileURLWithPath: filePath)) {
      archive = safeTensors
      let states = safeTensors.states
      stateDict = states
      for (key, value) in states {
        if key.hasPrefix("encoder.") || key.hasPrefix("decoder.")
          || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.")
        {
          stateDict["first_stage_model.\(key)"] = value
        }
      }
    } else if let zipArchive = Archive(url: URL(fileURLWithPath: filePath), accessMode: .read) {
      archive = zipArchive
      let rootObject = try Interpreter.unpickle(zip: zipArchive)
      let originalStateDict =
        rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["module"]
        as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        stateDict[key] = value
        if key.hasPrefix("encoder.") || key.hasPrefix("decoder.")
          || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.")
        {
          stateDict["first_stage_model.\(key)"] = value
        }
      }
    } else {
      throw UnpickleError.dataNotFound
    }
    let isSvdI2v = stateDict.keys.contains { $0.contains("time_mixer") }
    var isWurstchenStageC = stateDict.keys.contains { $0.contains("clip_txt_mapper.") }
    var isSD3Large = stateDict.keys.contains {
      $0.contains("joint_blocks.37.context_block.")
        || $0.contains("transformer_blocks.36.ff_context.")
    }
    var isSD3Medium =
      !isSD3Large
      && stateDict.keys.contains {
        ($0.contains("joint_blocks.23.context_block.")
          || $0.contains("transformer_blocks.22.ff_context."))
      }
    let isLongCatVideoAvatar = stateDict.keys.contains {
      $0.contains("blocks.47.audio_cross_attn.kv_linear.")
    }
    var isPixArtSigmaXL =
      !isSD3Large && !isLongCatVideoAvatar
      && stateDict.keys.contains {
        $0.contains("blocks.27.cross_attn.kv_") || $0.contains("transformer_blocks.27.attn2.to_")
      }
    var isHunyuan = stateDict.keys.contains {
      $0.contains("double_blocks.19.img_attn_qkv.")
        || $0.contains("single_transformer_blocks.39.linear1.")
    }
    var isAuraFlow =
      (stateDict.keys.contains {
        $0.contains("double_layers.3.attn.w2q.")
          || $0.contains("transformer_blocks.3.attn.add_q_proj.")
      })
      && (stateDict.keys.contains {
        $0.contains("single_layers.31.attn.w1q.")
          || $0.contains("single_transformer_blocks.31.attn.to_q.")
      })
    var isFlux2 = stateDict.keys.contains {
      $0.contains("single_blocks.39.linear1.")
        || $0.contains("single_transformer_blocks.39.attn.to_qkv_mlp_proj.")
    }
    var isFlux2_9B =
      !isFlux2
      && stateDict.keys.contains {
        $0.contains("single_blocks.23.linear1.")
          || $0.contains("single_transformer_blocks.23.attn.to_qkv_mlp_proj.")
      }
    var isFlux2_4B =
      !isFlux2 && !isFlux2_9B
      && stateDict.keys.contains {
        $0.contains("single_blocks.19.linear1.")
          || $0.contains("single_transformer_blocks.19.attn.to_qkv_mlp_proj.")
      }
    var isFlux1 =
      !isHunyuan && !isFlux2
      && stateDict.keys.contains {
        $0.contains("double_blocks.18.img_attn.qkv.")
          || $0.contains("single_transformer_blocks.37.proj_mlp.")
      }
    var isWan21_14B = stateDict.keys.contains {
      $0.contains("blocks.39.cross_attn.v.") || $0.contains("blocks.39.attn2.to_v.")
    }
    var isWan21_1_3B =
      !isWan21_14B
      && stateDict.keys.contains {
        $0.contains("blocks.29.cross_attn.v.") || $0.contains("blocks.29.attn2.to_v.")
      }
    var isHiDream = stateDict.keys.contains {
      $0.contains("double_stream_blocks.15.block.ff_i.experts.0.")
        || $0.contains("single_stream_blocks.31.block.ff_i.experts.0.")
    }
    var isQwenImage = stateDict.keys.contains {
      $0.contains("transformer_blocks.59.txt_mlp.")
    }
    var isZImage = stateDict.keys.contains {
      $0.contains("layers.29.feed_forward.w3.")
    }
    var isErnieImage = stateDict.keys.contains {
      $0.contains("layers.35.mlp.linear_fc2.") || $0.contains("layers_35_mlp_linear_fc2")
    }
    var isCosmosAnima = stateDict.keys.contains {
      $0.contains("patch_embed.proj.") || $0.contains("patch_embed_proj")
        || $0.contains("blocks.5.self_attn.q_proj.")
        || $0.contains("blocks_5_self_attn_q_proj")
    }
    var isIdeogram4 =
      stateDict.keys.contains {
        $0.contains("layers.33.attention.to_q.weight")
          || $0.contains("layers.33.attention.qkv.weight")
      }
      && stateDict.keys.contains {
        $0.contains("final_layer.adaln_modulation.weight")
      }
    if isIdeogram4 {
      isZImage = false
    }
    var isKrea2 =
      stateDict.keys.contains {
        $0.contains("blocks.27.attn.wq.weight")
          || $0.contains("transformer_blocks.27.attn.to_q.weight")
      }
      && stateDict.keys.contains {
        $0.contains("txtfusion.projector.weight") || $0.contains("text_fusion.projector.weight")
      }
    let modifier: SamplerModifier
    let modelVersion: ModelVersion
    let inputDim: Int
    let isDiffusersFormat: Bool
    let expectedTotalAccess: Int
    var distilledGuidanceLayers: Int = 0
    // This is for SD v1, v2 and SDXL.
    if let tokey = stateDict[
      "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"]
      ?? stateDict["down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight"],
      let inputConv2d = stateDict["model.diffusion_model.input_blocks.0.0.weight"]
        ?? stateDict["conv_in.weight"]
    {

      inputDim = inputConv2d.shape.count >= 2 ? inputConv2d.shape[1] : 4
      switch inputDim {
      case 9:
        modifier = .inpainting
      case 8:
        if isSvdI2v {
          // For Stable Video Diffusion.
          modifier = .none
        } else {
          modifier = .editing
        }
      case 5:
        modifier = .depth
      default:
        modifier = .none
      }
      switch tokey.shape.last {
      case 2048:
        if !stateDict.keys.contains(where: {
          $0.contains("mid_block.attentions.0.transformer_blocks.")
            || $0.contains("middle_block.1.transformer_blocks.")
        }) {
          modelVersion = .ssd1b
          expectedTotalAccess = 944
        } else {
          modelVersion = .sdxlBase
          expectedTotalAccess = 1680
        }
      case 1280:
        modelVersion = .sdxlRefiner
        expectedTotalAccess = 1220
      case 1024:
        if isSvdI2v {
          modelVersion = .svdI2v
          expectedTotalAccess = 686
        } else {
          modelVersion = .v2
          expectedTotalAccess = 686
        }
      case 768:
        modelVersion = .v1
        expectedTotalAccess = 686
      default:
        throw UnpickleError.tensorNotFound
      }
      isDiffusersFormat = stateDict.keys.contains { $0.hasPrefix("mid_block.") }
      isWurstchenStageC = false
      isPixArtSigmaXL = false
      isSD3Medium = false
      isSD3Large = false
      isFlux1 = false
      isHunyuan = false
      isWan21_14B = false
      isWan21_1_3B = false
      isHiDream = false
      isQwenImage = false
      isAuraFlow = false
      isZImage = false
      isErnieImage = false
      isCosmosAnima = false
      isFlux2 = false
      isFlux2_9B = false
      isFlux2_4B = false
      isIdeogram4 = false
      isKrea2 = false
    } else if isWurstchenStageC {
      modelVersion = .wurstchenStageC
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1550
      isDiffusersFormat = false
    } else if isPixArtSigmaXL {
      modelVersion = .pixart
      modifier = .none
      inputDim = 4
      expectedTotalAccess = 754
      isDiffusersFormat = stateDict.keys.contains { $0.contains("transformer_blocks.27.") }
    } else if isSD3Medium {
      modelVersion = .sd3
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1157
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("transformer_blocks.22.ff_context.")
      }
    } else if isSD3Large {
      modelVersion = .sd3Large
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1981
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("transformer_blocks.36.ff_context.")
      }
    } else if isFlux1 {
      modelVersion = .flux1
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1732
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("single_transformer_blocks.37.")
      }
      // If it is flux, check for distilled guidance layer (Chroma).
      distilledGuidanceLayers = 0
      while stateDict.keys.contains(where: {
        $0.contains("distilled_guidance_layer.layers.\(distilledGuidanceLayers).")
      }) {
        distilledGuidanceLayers += 1
      }
    } else if isHunyuan {
      modelVersion = .hunyuanVideo
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1870
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("single_transformer_blocks.39.")
      }
    } else if isLongCatVideoAvatar {
      modelVersion = .longcatVideoAvatar1_5
      modifier = .kontext
      inputDim = 16
      expectedTotalAccess = 1608
      isDiffusersFormat = false
    } else if isWan21_14B {
      modelVersion = .wan21_14b
      modifier =
        stateDict.keys.contains {
          $0.contains("blocks.39.cross_attn.v_img.") || $0.contains("blocks.39.attn2.add_v_proj.")
        } ? .inpainting : .none
      inputDim = 16
      expectedTotalAccess = 1306 + (modifier == .inpainting ? 208 : 0)
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("blocks.39.attn2.to_v.")
      }
    } else if isWan21_1_3B {
      if let patchEmbeddingWeight =
        (stateDict.first {
          $0.key.contains("patch_embedding.weight")
        }), patchEmbeddingWeight.value.shape[0] == 3_072
      {
        modelVersion = .wan22_5b
        modifier = .none
        inputDim = 48
        expectedTotalAccess = 986
      } else {
        modelVersion = .wan21_1_3b
        modifier =
          stateDict.keys.contains {
            $0.contains("blocks.29.cross_attn.v_img.") || $0.contains("blocks.29.attn2.add_v_proj.")
          } ? .inpainting : .none
        inputDim = 16
        expectedTotalAccess = 986 + (modifier == .inpainting ? 158 : 0)
      }
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("blocks.29.attn2.to_v.")
      }
    } else if isHiDream {
      modelVersion = .hiDreamI1
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1857
      isDiffusersFormat = true  // Only Diffusers format available.
    } else if isQwenImage {
      modelVersion = .qwenImage
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 3121
      isDiffusersFormat = true  // Only Diffusers format available.
    } else if isAuraFlow {
      modelVersion = .auraflow
      modifier = .none
      inputDim = 4
      expectedTotalAccess = 532
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("single_transformer_blocks.31.attn.")
      }
    } else if isZImage {
      modelVersion = .zImage
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 1652
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("layers.29.attention.to_out.0.")
      }
    } else if isErnieImage {
      modelVersion = .ernieImage
      modifier = .none
      inputDim = 32
      expectedTotalAccess = 421
      isDiffusersFormat = false
    } else if isCosmosAnima {
      modelVersion = .cosmos2_5_2b
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 856
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("patch_embed.proj.")
      }
    } else if isFlux2 {
      modelVersion = .flux2
      modifier = .kontext
      inputDim = 32
      expectedTotalAccess = 1061
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("single_transformer_blocks.39.attn.to_qkv_mlp_proj.")
      }
    } else if isFlux2_9B {
      modelVersion = .flux2_9b
      modifier = .kontext
      inputDim = 32
      expectedTotalAccess = 697
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("single_transformer_blocks.23.attn.to_qkv_mlp_proj.")
      }
    } else if isFlux2_4B {
      modelVersion = .flux2_4b
      modifier = .kontext
      inputDim = 32
      expectedTotalAccess = 523
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("single_transformer_blocks.19.attn.to_qkv_mlp_proj.")
      }
    } else if isIdeogram4 {
      modelVersion = .ideogram4
      modifier = .none
      inputDim = 32
      expectedTotalAccess = 526
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("layers.33.attention.to_q.weight")
      }
    } else if isKrea2 {
      modelVersion = .krea2
      modifier = .none
      inputDim = 16
      expectedTotalAccess = 430
      isDiffusersFormat = stateDict.keys.contains {
        $0.contains("transformer_blocks.27.attn.to_q.weight")
      }
    } else {
      throw UnpickleError.tensorNotFound
    }
    let keys = stateDict.keys
    let hasEncoderHidProj = keys.contains { $0 == "encoder_hid_proj.weight" }
    let hasGuidanceEmbed = keys.contains {
      $0.contains(".guidance_embedder.") || $0.contains("guidance_in.")
    }
    let qkNorm = keys.contains {
      $0.contains(".ln_k.") || $0.contains(".ln_q.") || $0.contains(".norm_k.")
        || $0.contains(".norm_q.")
    }
    let dualAttentionLayers = (0..<38).filter { i in
      keys.contains {
        $0.contains(".\(i).x_block.attn2.") || $0.contains("_blocks.\(i).attn2.")
      }
    }
    return InspectionResult(
      version: modelVersion, archive: archive, stateDict: stateDict, modifier: modifier,
      inputChannels: inputDim, isDiffusersFormat: isDiffusersFormat,
      hasEncoderHidProj: hasEncoderHidProj, hasGuidanceEmbed: hasGuidanceEmbed,
      qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      distilledGuidanceLayers: distilledGuidanceLayers, numberOfTensors: expectedTotalAccess
    )
  }

  private func internalImport(versionCheck: @escaping (ModelVersion) -> Void) throws -> (
    [String], ModelVersion, SamplerModifier, InspectionResult
  ) {
    let inspectionResult = try inspect()
    var archive = inspectionResult.archive
    var stateDict = inspectionResult.stateDict
    var modifier = inspectionResult.modifier
    let modelVersion = inspectionResult.version
    let inputDim = inspectionResult.inputChannels
    let isDiffusersFormat = inspectionResult.isDiffusersFormat
    let hasEncoderHidProj = inspectionResult.hasEncoderHidProj
    let qkNorm = inspectionResult.qkNorm
    let dualAttentionLayers = inspectionResult.dualAttentionLayers
    let distilledGuidanceLayers = inspectionResult.distilledGuidanceLayers
    expectedTotalAccess = inspectionResult.numberOfTensors
    versionCheck(modelVersion)
    progress?(0.05)
    if isTextEncoderCustomized {
      switch modelVersion {
      case .longcatVideoAvatar1_5:
        throw Error.noTextEncoder
      case .v1:
        expectedTotalAccess += 196
      case .v2:
        expectedTotalAccess += 280
      case .sdxlBase, .ssd1b:
        expectedTotalAccess += 388 + 196
      case .sdxlRefiner:
        expectedTotalAccess += 388
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .hunyuanVideo:
        throw Error.noTextEncoder
      case .svdI2v:
        throw Error.noTextEncoder
      case .wurstchenStageC, .wurstchenStageB:
        throw Error.noTextEncoder
      case .wan21_1_3b, .wan21_14b, .wan22_5b:
        throw Error.noTextEncoder
      case .qwenImage, .cosmos2_5_2b:
        throw Error.noTextEncoder
      case .hiDreamI1, .hiDreamO1, .qwenImage2_1:
        throw Error.noTextEncoder
      case .zImage:
        throw Error.noTextEncoder
      case .ernieImage, .flux2, .flux2_9b, .flux2_4b:
        throw Error.noTextEncoder
      case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
        throw Error.noTextEncoder
      case .kandinsky21, .minimaxH3:
        fatalError()
      }
    }
    if autoencoderFilePath != nil {
      expectedTotalAccess += 248
    }
    let graph = DynamicGraph()
    var filePaths = [String]()
    if isTextEncoderCustomized {
      do {
        try graph.withNoGrad {
          let tokensTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
          let positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
          let causalAttentionMask = graph.variable(.CPU, .NHWC(1, 1, 77, 77), of: FloatType.self)
          var textModel: Model
          var textModelReader: PythonReader
          var textEncoderArchive = archive
          var filePath: String
          var textEncoderStateDict = stateDict
          if let textEncoderFilePath = textEncoderFilePath {
            if let safeTensors = SafeTensors(url: URL(fileURLWithPath: textEncoderFilePath)) {
              textEncoderArchive = safeTensors
              textEncoderStateDict = safeTensors.states
              for (key, value) in textEncoderStateDict {
                textEncoderStateDict["cond_stage_model.transformer.\(key)"] = value
              }
            } else if let zipArchive = Archive(
              url: URL(fileURLWithPath: textEncoderFilePath), accessMode: .read)
            {
              textEncoderArchive = zipArchive
              let rootObject = try Interpreter.unpickle(zip: zipArchive)
              let originalStateDict =
                rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["module"]
                as? Interpreter.Dictionary ?? rootObject
              textEncoderStateDict = [String: TensorDescriptor]()
              originalStateDict.forEach { key, value in
                guard let value = value as? TensorDescriptor else { return }
                textEncoderStateDict["cond_stage_model.transformer.\(key)"] = value
                textEncoderStateDict[key] = value
              }
            } else {
              throw UnpickleError.dataNotFound
            }
          }
          switch modelVersion {
          case .longcatVideoAvatar1_5, .minimaxH3:
            fatalError()
          case .v1, .sdxlBase, .ssd1b:
            (textModel, textModelReader) = CLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 768,
              numLayers: 12, numHeads: 12,
              batchSize: 2, intermediateSize: 3072, usesFlashAttention: false)
            filePath = ModelZoo.filePathForModelDownloaded("\(modelName)_clip_vit_l14_f16.ckpt")
          case .v2:
            (textModel, textModelReader) = OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1024,
              numLayers: 23, numHeads: 16,
              batchSize: 2, intermediateSize: 4096, usesFlashAttention: false)
            filePath = ModelZoo.filePathForModelDownloaded(
              "\(modelName)_open_clip_vit_h14_f16.ckpt")
          case .sdxlRefiner:
            (textModel, textModelReader) = OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1280,
              numLayers: 32, numHeads: 20, batchSize: 2,
              intermediateSize: 5120, usesFlashAttention: false, outputPenultimate: true)
            filePath = ModelZoo.filePathForModelDownloaded(
              "\(modelName)_open_clip_vit_bigg14_f16.ckpt")
          case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
            .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1,
            .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2,
            .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
            fatalError()
          }
          if modelVersion == .sdxlBase || modelVersion == .sdxlRefiner {
            for (key, value) in textEncoderStateDict {
              if key.hasPrefix("conditioner.embedders.0.") {
                textEncoderStateDict["cond_stage_model." + key.dropFirst(24)] = value
              }
            }
          }
          textModel.compile(inputs: tokensTensor, positionTensor, causalAttentionMask)
          try textModelReader(textEncoderStateDict, textEncoderArchive)

          try graph.openStore(filePath) { store in
            store.removeAll()
            if let text_projection = textEncoderStateDict["cond_stage_model.model.text_projection"]
              ?? textEncoderStateDict["cond_stage_model.text_projection"]
            {
              try textEncoderArchive.with(text_projection) {
                store.write("text_projection", tensor: $0)
              }
            }
            store.write("text_model", model: textModel)
          }
          try graph.openStore(filePath) {
            switch modelVersion {
            case .longcatVideoAvatar1_5, .minimaxH3:
              fatalError()
            case .v1, .sdxlBase, .ssd1b:
              if $0.keys.count < 196 {
                throw Error.tensorWritesFailed
              }
            case .v2:
              if $0.keys.count < 372 {
                throw Error.tensorWritesFailed
              }
            case .sdxlRefiner:
              if $0.keys.count < 517 {
                throw Error.tensorWritesFailed
              }
            case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v,
              .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
              .hiDreamI1, .hiDreamO1, .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b, .zImage,
              .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b,
              .ideogram4, .krea2:
              fatalError()
            }
          }
          filePaths.append(filePath)
          if modelVersion == .sdxlBase {
            // SDXL Base has two text encoders.
            (textModel, textModelReader) = OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1280,
              numLayers: 32, numHeads: 20, batchSize: 2,
              intermediateSize: 5120, usesFlashAttention: false, outputPenultimate: true)
            filePath = ModelZoo.filePathForModelDownloaded(
              "\(modelName)_open_clip_vit_bigg14_f16.ckpt")
            var textEncoder2StateDict = stateDict
            var textEncoder2Archive: TensorArchive = archive
            var textEncoder2TextProjectionTransposed = false
            if let textEncoder2FilePath = textEncoder2FilePath {
              if let safeTensors = SafeTensors(url: URL(fileURLWithPath: textEncoder2FilePath)) {
                textEncoder2Archive = safeTensors
                textEncoder2StateDict = safeTensors.states
                for (key, value) in textEncoder2StateDict {
                  if key.hasPrefix("text_projection") {
                    textEncoder2StateDict["cond_stage_model.model.text_projection"] = value
                    if key.hasSuffix(".weight") {
                      textEncoder2TextProjectionTransposed = true
                    }
                  } else {
                    textEncoder2StateDict["cond_stage_model.transformer.\(key)"] = value
                  }
                }
              } else if let zipArchive = Archive(
                url: URL(fileURLWithPath: textEncoder2FilePath), accessMode: .read)
              {
                textEncoder2Archive = zipArchive
                let rootObject = try Interpreter.unpickle(zip: zipArchive)
                textEncoder2StateDict = [String: TensorDescriptor]()
                let originalStateDict =
                  rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["module"]
                  as? Interpreter.Dictionary ?? rootObject
                originalStateDict.forEach { key, value in
                  guard let value = value as? TensorDescriptor else { return }
                  if key.hasPrefix("text_projection") {
                    textEncoder2StateDict["cond_stage_model.model.text_projection"] = value
                    if key.hasSuffix(".weight") {
                      textEncoder2TextProjectionTransposed = true
                    }
                  } else {
                    textEncoder2StateDict["cond_stage_model.transformer.\(key)"] = value
                  }
                  textEncoder2StateDict[key] = value
                }
              } else {
                throw UnpickleError.dataNotFound
              }
            }
            for (key, value) in textEncoder2StateDict {
              if key.hasPrefix("conditioner.embedders.1.") {
                textEncoder2StateDict["cond_stage_model." + key.dropFirst(24)] = value
              }
            }
            textModel.compile(inputs: tokensTensor, positionTensor, causalAttentionMask)
            try textModelReader(textEncoder2StateDict, textEncoder2Archive)
            try graph.openStore(filePath) { store in
              store.removeAll()
              if let text_projection = textEncoder2StateDict[
                "cond_stage_model.model.text_projection"]
                ?? textEncoder2StateDict["cond_stage_model.text_projection"]
              {
                if textEncoder2TextProjectionTransposed {
                  try textEncoder2Archive.with(text_projection) { tensor in
                    let textProjection = graph.variable(
                      Tensor<FloatType>(
                        from: tensor)
                    ).toGPU(0)
                    store.write(
                      "text_projection", tensor: textProjection.transposed(0, 1).toCPU().rawValue)
                  }
                } else {
                  try textEncoder2Archive.with(text_projection) {
                    store.write("text_projection", tensor: $0)
                  }
                }
              }
              store.write("text_model", model: textModel)
            }
            try graph.openStore(filePath) {
              if $0.keys.count != 517 {
                throw Error.tensorWritesFailed
              }
            }
            filePaths.append(filePath)
          }
        }
      } catch {
        throw Error.textEncoder(error)
      }
    }
    let conditionalLength: Int
    let batchSize: Int
    switch modelVersion {
    case .longcatVideoAvatar1_5:
      conditionalLength = 4096
      batchSize = 1
    case .v1:
      conditionalLength = 768
      batchSize = 2
    case .v2:
      conditionalLength = 1024
      batchSize = 2
    case .svdI2v:
      conditionalLength = 1024
      batchSize = 1
    case .sdxlBase, .sdxlRefiner, .ssd1b:
      conditionalLength = 1280
      batchSize = 2
    case .wurstchenStageC:
      conditionalLength = 1280
      batchSize = 2
    case .pixart:
      conditionalLength = 4096
      batchSize = 2
    case .sd3, .sd3Large:
      conditionalLength = 4096
      batchSize = 2
    case .auraflow:
      conditionalLength = 2048
      batchSize = 1
    case .flux1:
      conditionalLength = 4096
      batchSize = 1
    case .hunyuanVideo:
      conditionalLength = 4096
      batchSize = 1
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      conditionalLength = 4096
      batchSize = 1
    case .hiDreamI1:
      conditionalLength = 4096
      batchSize = 1
    case .hiDreamO1, .qwenImage2_1, .minimaxH3:
      fatalError()
    case .qwenImage:
      conditionalLength = 3854
      batchSize = 1
    case .zImage:
      conditionalLength = 2560
      batchSize = 1
    case .ernieImage:
      conditionalLength = 3072
      batchSize = 1
    case .cosmos2_5_2b:
      conditionalLength = 1024
      batchSize = 1
    case .flux2:
      conditionalLength = 15360
      batchSize = 1
    case .flux2_9b:
      conditionalLength = 12288
      batchSize = 1
    case .flux2_4b:
      conditionalLength = 7680
      batchSize = 1
    case .ideogram4:
      conditionalLength = 13 * 4_096
      batchSize = 1
    case .krea2:
      conditionalLength = 2560
      batchSize = 1
    case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b:
      fatalError()
    case .kandinsky21, .wurstchenStageB:
      fatalError()
    }
    try graph.withNoGrad {
      var tEmb: DynamicGraph.Tensor<FloatType>? = graph.variable(
        Tensor<FloatType>(
          from: timeEmbedding(
            timestep: 981, batchSize: batchSize,
            embeddingSize: modelVersion == .sdxlRefiner ? 384 : 320,
            maxPeriod: 10_000)
        ))
      let xTensor = graph.variable(.CPU, .NHWC(batchSize, 64, 64, inputDim), of: FloatType.self)
      let cTensor = graph.variable(.CPU, .HWC(batchSize, 77, conditionalLength), of: FloatType.self)
      var cArr = [cTensor]
      let unet: Model
      var unetReader: PythonReader?
      let unetMapper: ModelWeightMapper?
      let filePath = ModelZoo.filePathForModelDownloaded("\(self.modelName)_f16.ckpt")
      switch modelVersion {
      case .longcatVideoAvatar1_5:
        let rot = Tensor<FloatType>(
          from: WanRotaryPositionEmbedding(
            height: 32, width: 32, time: 1, channels: 128)
        )
        cArr =
          [graph.variable(rot)]
          + LongCatVideoAvatarFixedOutputShapes(
            timesteps: 1, time: 1, condFrames: 0, channels: 4_096, layers: 48, textLength: 512,
            audioTokens: 32
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageB, .wurstchenStageC, .pixart,
        .sd3, .sd3Large, .auraflow:
        let fixedEncoder = UNetFixedEncoder<FloatType>(
          filePath: "", version: modelVersion, modifier: .none,
          dualAttentionLayers: dualAttentionLayers, activationQkScaling: [:],
          activationProjScaling: [:], activationFfnProjUpScaling: [:],
          activationFfnScaling: [:],
          usesFlashAttention: .none, zeroNegativePrompt: false,
          isQuantizedModel: false, canRunLoRASeparately: false, externalOnDemand: false,
          deviceProperties: DeviceProperties(
            isFreadPreferred: true, memoryCapacity: .high, isNHWCPreferred: true,
            cacheUri: URL(fileURLWithPath: NSTemporaryDirectory()), isPartialOffloadPreferred: false
          ),
          weightsCache: WeightsCache(maxTotalCacheSize: 0, memorySubsystem: .UMA))
        cArr.insert(
          graph.variable(.CPU, .HWC(batchSize, 77, 768), of: FloatType.self),
          at: 0)
        cArr.append(
          graph.variable(.CPU, .WC(batchSize, 1280), of: FloatType.self))
        for c in cArr {
          c.full(0)
        }
        let vectors: [DynamicGraph.Tensor<FloatType>]
        switch modelVersion {
        case .longcatVideoAvatar1_5, .minimaxH3:
          fatalError()
        case .sdxlBase, .ssd1b:
          vectors = [graph.variable(.CPU, .WC(batchSize, 2816), of: FloatType.self)]
        case .sdxlRefiner:
          vectors = [graph.variable(.CPU, .WC(batchSize, 2560), of: FloatType.self)]
        case .svdI2v:
          vectors = [graph.variable(.CPU, .WC(batchSize, 768), of: FloatType.self)]
        case .wurstchenStageC, .wurstchenStageB, .pixart, .sd3, .sd3Large, .auraflow, .flux1,
          .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage2_1, .qwenImage,
          .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2,
          .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
          vectors = []
        case .kandinsky21, .v1, .v2:
          fatalError()
        }
        // These values doesn't matter, it won't affect the model shape, just the input vector.
        cArr =
          vectors
          + fixedEncoder.encode(
            isCfgEnabled: true, textGuidanceScale: 3.5, guidanceEmbed: 3.5,
            isGuidanceEmbedEnabled: false, distilledGuidanceLayers: 0, modifier: .none,
            textEncoding: cArr.map({ $0.toGPU(0) }), timesteps: [0], audioShiftRatio: 1,
            batchSize: batchSize,
            startHeight: 64, startWidth: 64,
            tokenLengthUncond: 77, tokenLengthCond: 77, lora: [],
            tiledDiffusion: TiledConfiguration(
              isEnabled: false, tileSize: .init(width: 0, height: 0), tileOverlap: 0),
            teaCache: TeaCacheConfiguration(
              coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0),
            isBF16: false, injectedControls: [], referenceImages: [], referenceAudios: []
          ).0.map({ DynamicGraph.Tensor<FloatType>($0).toCPU() })
        if modelVersion == .svdI2v {
          // Only take the first half (positive part).
          cArr = Array(cArr[0..<(1 + (cArr.count - 1) / 2)])
        }
      case .flux1:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: Flux1RotaryPositionEmbedding(
                  height: 32, width: 32, tokenLength: 256, referenceSizes: [], channels: 128)))
          ]
          + Flux1FixedOutputShapes(
            batchSize: (1, 1), tokenLength: 256, channels: 3072, layers: (19, 38),
            contextPreloaded: true
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .hunyuanVideo:
        let (rot0, rot1) = HunyuanRotaryPositionEmbedding(
          height: 64, width: 64, time: 1, tokenLength: 20, channels: 128)
        cArr =
          [
            graph.variable(Tensor<FloatType>(from: rot0)),
            graph.variable(Tensor<FloatType>(from: rot1)),
          ]
          + HunyuanFixedOutputShapes(
            batchSize: batchSize, channels: 3072, layers: (20, 40), textLength: 20
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .wan21_1_3b:
        let rot = Tensor<FloatType>(
          from: WanRotaryPositionEmbedding(
            height: 64, width: 64, time: 1, channels: 128)
        )
        cArr =
          [graph.variable(rot)]
          + WanFixedOutputShapes(
            timesteps: 1, batchSize: (1, 1), channels: 1_536, layers: 30, textLength: 512,
            injectImage: true
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .wan21_14b:
        let rot = Tensor<FloatType>(
          from: WanRotaryPositionEmbedding(
            height: 64, width: 64, time: 1, channels: 128)
        )
        cArr =
          [graph.variable(rot)]
          + WanFixedOutputShapes(
            timesteps: 1, batchSize: (1, 1), channels: 5_120, layers: 40, textLength: 512,
            injectImage: true
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .wan22_5b:
        let rot = Tensor<FloatType>(
          from: WanRotaryPositionEmbedding(
            height: 64, width: 64, time: 1, channels: 128)
        )
        cArr =
          [graph.variable(rot)]
          + WanFixedOutputShapes(
            timesteps: 1, batchSize: (1, 1), channels: 3_072, layers: 30, textLength: 512,
            injectImage: true
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .hiDreamI1:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: HiDreamRotaryPositionEmbedding(
                  height: 32, width: 32, tokenLength: 128, channels: 128)))
          ]
          + HiDreamFixedOutputShapes(
            timesteps: 1, layers: (16, 32), textLength: 128
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .hiDreamO1, .qwenImage2_1, .minimaxH3:
        fatalError()
      case .qwenImage:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: QwenImageRotaryPositionEmbedding(
                  height: 32, width: 32, tokenLength: 128, referenceSizes: [], channels: 128)))
          ]
          + QwenImageFixedOutputShapes(
            batchSize: 1, textLength: 128, channels: 3072, layers: 60
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .zImage:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: ZImageRotaryPositionEmbedding(
                  height: 32, width: 32, tokenLength: 32, imagePaddedLength: 0)))
          ]
          + ZImageFixedOutputShapes(
            batchSize: 1, tokenLength: 32, channels: 3840, layers: 30
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .ernieImage:
        let (rotCos, rotSin) = ErnieImageRotaryPositionEmbedding(
          height: 32, width: 32, textLength: 512)
        cArr =
          [
            graph.variable(Tensor<FloatType>(from: rotCos)),
            graph.variable(Tensor<FloatType>(from: rotSin)),
          ]
          + ErnieImageFixedOutputShapes(
            tokenLength: 512, channels: 4096
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .cosmos2_5_2b:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: CosmosRotaryPositionEmbedding(height: 32, width: 32)))
          ]
          + CosmosFixedOutputShapes(
            timesteps: 1, batchSize: 1, textLength: 512
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .flux2:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: Flux2RotaryPositionEmbedding(
                  height: 32, width: 32, tokenLength: 512, referenceSizes: [], channels: 128)))
          ]
          + Flux2FixedOutputShapes(
            tokenLength: 512, channels: 6144
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .flux2_9b:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: Flux2RotaryPositionEmbedding(
                  height: 32, width: 32, tokenLength: 512, referenceSizes: [], channels: 128)))
          ]
          + Flux2FixedOutputShapes(
            tokenLength: 512, channels: 4096
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .flux2_4b:
        cArr =
          [
            graph.variable(
              Tensor<FloatType>(
                from: Flux2RotaryPositionEmbedding(
                  height: 32, width: 32, tokenLength: 512, referenceSizes: [], channels: 128)))
          ]
          + Flux2FixedOutputShapes(
            tokenLength: 512, channels: 3072
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .krea2:
        cArr =
          [
            graph.variable(.CPU, .HWC(1, 32, 6_144), of: FloatType.self),
            graph.variable(
              Tensor<FloatType>(
                from: Krea2RotaryPositionEmbedding(
                  textLength: 32, gridHeight: 32, gridWidth: 32, of: FloatType.self))),
          ]
          + (0..<7).map { _ in
            graph.variable(.CPU, .HWC(1, 1, 6_144), of: FloatType.self)
          }
      case .ideogram4:
        cArr =
          [
            graph.variable(.CPU, .HWC(1, 32, 4_608), of: FloatType.self),
            graph.variable(.CPU, .WC(32 * 32, 4_608), of: FloatType.self),
            graph.variable(
              Tensor<FloatType>(
                from: Ideogram4RotaryPositionEmbedding(
                  textLength: 32, gridHeight: 32, gridWidth: 32, of: FloatType.self))),
          ]
          + (0..<(34 * 4 + 1)).map { _ in
            graph.variable(.CPU, .WC(1, 4_608), of: FloatType.self)
          }
      case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b:
        fatalError()
      case .kandinsky21, .v1, .v2:
        break
      }
      let unetFixed: Model?
      let unetFixedMapper: ModelWeightMapper?
      var additionalMappings = [(prefix: String, mapping: ModelWeightMapping)]()
      switch modelVersion {
      case .longcatVideoAvatar1_5:
        (unetMapper, unet) = LongCatVideoAvatar(
          time: 1, height: 64, width: 64, channels: 4_096, layers: 48, intermediateSize: 11_008,
          textLength: 512, audioTokens: 32, condFrames: 0, usesFlashAttention: .scale1)
        unetReader = nil
        (unetFixedMapper, unetFixed) = LongCatVideoAvatarFixed(
          timesteps: 1, time: 2, condFrames: 0, channels: 4_096, layers: 48, textLength: 512,
          audioWindow: 5, audioBlocks: 5, audioChannels: 1_280, audioTokens: 32,
          audioIntermediateDim: 512, audioOutputDim: 768)
      case .v1:
        (unet, unetReader) = UNet(
          batchSize: batchSize, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
          usesFlashAttention: .none, injectControls: false, injectT2IAdapters: false,
          injectIPAdapterLengths: [], injectAttentionKV: false)
        unetMapper = nil
        (unetFixed, unetFixedMapper) = (nil, nil)
      case .v2:
        (unet, unetReader) = UNetv2(
          batchSize: batchSize, embeddingLength: (77, 77), startWidth: 64, startHeight: 64,
          upcastAttention: false, usesFlashAttention: .none, injectControls: false)
        unetMapper = nil
        (unetFixed, unetFixedMapper) = (nil, nil)
      case .sdxlBase:
        (unet, _, unetMapper) = UNetXL(
          batchSize: batchSize, startHeight: 64, startWidth: 64,
          channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
          middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
          embeddingLength: (77, 77), injectIPAdapterLengths: [],
          upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
          isTemporalMixEnabled: false, of: FloatType.self)
        unetReader = nil
        (unetFixed, _, unetFixedMapper) = UNetXLFixed(
          batchSize: batchSize, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [10, 10]],
          middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
          usesFlashAttention: .none, isTemporalMixEnabled: false)
      case .ssd1b:
        (unet, _, unetMapper) = UNetXL(
          batchSize: batchSize, startHeight: 64, startWidth: 64,
          channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
          middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
          embeddingLength: (77, 77), injectIPAdapterLengths: [],
          upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
          isTemporalMixEnabled: false, of: FloatType.self)
        unetReader = nil
        (unetFixed, _, unetFixedMapper) = UNetXLFixed(
          batchSize: batchSize, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
          embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [4, 4]],
          middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
          usesFlashAttention: .none, isTemporalMixEnabled: false)
      case .sdxlRefiner:
        (unet, _, unetMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: 64, startWidth: 64,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (77, 77), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
            isTemporalMixEnabled: false, of: FloatType.self
          )
        unetReader = nil
        (unetFixed, _, unetFixedMapper) = UNetXLFixed(
          batchSize: batchSize, startHeight: 64, startWidth: 64, channels: [384, 768, 1536, 1536],
          embeddingLength: (77, 77), inputAttentionRes: [2: [4, 4], 4: [4, 4]],
          middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
          usesFlashAttention: .none, isTemporalMixEnabled: false)
      case .svdI2v:
        (unet, _, unetMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: 64, startWidth: 64,
            channels: [320, 640, 1280, 1280],
            inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
            outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], embeddingLength: (1, 1),
            injectIPAdapterLengths: [], upcastAttention: ([:], false, [1: [0, 1, 2]]),
            usesFlashAttention: .none, injectControls: false,
            isTemporalMixEnabled: true, of: FloatType.self
          )
        unetReader = nil
        (unetFixed, _, unetFixedMapper) = UNetXLFixed(
          batchSize: batchSize, startHeight: 64, startWidth: 64,
          channels: [320, 640, 1280, 1280], embeddingLength: (1, 1),
          inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
          outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], usesFlashAttention: .none,
          isTemporalMixEnabled: true)
      case .wurstchenStageC:
        (unet, unetMapper) = WurstchenStageC(
          batchSize: batchSize, height: 24, width: 24,
          t: (77 + 8, 77 + 8),
          usesFlashAttention: .none)
        unetReader = nil
        (unetFixed, unetFixedMapper) = WurstchenStageCFixed(
          batchSize: batchSize, t: (77 + 8, 77 + 8),
          usesFlashAttention: .none)
      case .pixart:
        (unetMapper, unet) = PixArt(
          batchSize: batchSize, height: 64, width: 64, channels: 1152, layers: 28,
          tokenLength: (77, 77), usesFlashAttention: .none, of: FloatType.self)
        unetReader = nil
        (unetFixedMapper, unetFixed) = PixArtFixed(
          batchSize: batchSize, channels: 1152, layers: 28, tokenLength: (77, 77),
          usesFlashAttention: false, of: FloatType.self)
      case .sd3:
        (unetMapper, unet) = MMDiT(
          batchSize: batchSize, t: 77, height: 64, width: 64, channels: 1536, layers: 24,
          upcast: false, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
          posEmbedMaxSize: 192, usesFlashAttention: .none, of: FloatType.self)
        unetReader = nil
        (unetFixedMapper, unetFixed) = MMDiTFixed(
          batchSize: batchSize, channels: 1536, layers: 24, dualAttentionLayers: dualAttentionLayers
        )
      case .sd3Large:
        (unetMapper, unet) = MMDiT(
          batchSize: batchSize, t: 77, height: 64, width: 64, channels: 2432, layers: 38,
          upcast: true, qkNorm: true, dualAttentionLayers: [], posEmbedMaxSize: 192,
          usesFlashAttention: .none, of: FloatType.self)
        unetReader = nil
        (unetFixedMapper, unetFixed) = MMDiTFixed(
          batchSize: batchSize, channels: 2432, layers: 38, dualAttentionLayers: [])
      case .flux1:
        (unetMapper, unet) = Flux1(
          batchSize: batchSize, tokenLength: 256, referenceSequenceLength: 0, height: 64, width: 64,
          channels: 3072,
          layers: (19, 38), usesFlashAttention: .scaleMerged, contextPreloaded: true,
          injectControls: false, injectIPAdapterLengths: [:], outputResidual: false,
          inputResidual: false)
        if distilledGuidanceLayers > 0 {
          (unetFixedMapper, unetFixed) = ChromaFixed(
            channels: 3072, distilledGuidanceLayers: distilledGuidanceLayers, layers: (19, 38),
            contextPreloaded: true)
        } else {
          (unetFixedMapper, unetFixed) = Flux1Fixed(
            batchSize: (batchSize, batchSize), channels: 3072, layers: (19, 38),
            contextPreloaded: true, numberOfReferenceImages: 0, guidanceEmbed: true)
        }
      case .hunyuanVideo:
        (unetMapper, unet) = Hunyuan(
          time: 1, height: 64, width: 64, textLength: 20, channels: 3072, layers: (20, 40),
          usesFlashAttention: .scaleMerged, outputResidual: false, inputResidual: false)
        (unetFixedMapper, unetFixed) = HunyuanFixed(
          timesteps: batchSize, channels: 3072, layers: (20, 40), textLength: (0, 20))
      case .wan21_1_3b:
        (unetMapper, unet) = Wan(
          channels: 1_536, layers: 30, vaceLayers: [], intermediateSize: 8_960, time: 1, height: 64,
          width: 64,
          textLength: 512, causalInference: (0, 0), injectImage: true,
          usesFlashAttention: .scale1,
          outputResidual: false, inputResidual: false, outputChannels: 16)
        (unetFixedMapper, unetFixed) = WanFixed(
          timesteps: 1, batchSize: (1, 1), channels: 1_536, layers: 30, vaceLayers: [],
          textLength: 512, injectImage: true)
      case .wan21_14b:
        (unetMapper, unet) = Wan(
          channels: 5_120, layers: 40, vaceLayers: [], intermediateSize: 13_824, time: 1,
          height: 64, width: 64,
          textLength: 512, causalInference: (0, 0), injectImage: true,
          usesFlashAttention: .scale1,
          outputResidual: false, inputResidual: false, outputChannels: 16)
        (unetFixedMapper, unetFixed) = WanFixed(
          timesteps: 1, batchSize: (1, 1), channels: 5_120, layers: 40, vaceLayers: [],
          textLength: 512, injectImage: true)
      case .wan22_5b:
        (unetMapper, unet) = Wan(
          channels: 3_072, layers: 30, vaceLayers: [], intermediateSize: 14_336,
          time: 1, height: 64, width: 64,
          textLength: 512, causalInference: (0, 0), injectImage: true,
          usesFlashAttention: .scale1, outputResidual: false,
          inputResidual: false, outputChannels: 48
        )
        (unetFixedMapper, unetFixed) = WanFixed(
          timesteps: 1, batchSize: (1, 1), channels: 3_072,
          layers: 30, vaceLayers: [], textLength: 512, injectImage: true
        )
      case .hiDreamI1:
        (unet, unetMapper) = HiDream(
          batchSize: 1, height: 64, width: 64, textLength: (128, 128), layers: (16, 32),
          usesFlashAttention: .scale1, outputResidual: false, inputResidual: false)
        (unetFixed, unetFixedMapper) = HiDreamFixed(
          timesteps: 1, layers: (16, 32), outputTimesteps: false)
      case .hiDreamO1, .qwenImage2_1, .minimaxH3:
        fatalError()
      case .qwenImage:
        (unetMapper, unet) = QwenImage(
          batchSize: 1, height: 64, width: 64, textLength: 128, referenceSequenceLength: 0,
          channels: 3_072, layers: 60, usesFlashAttention: .scale1, isBF16: true,
          isQwenImageLayered: false, zeroTimestepForReference: false, activationQkScaling: [:],
          activationProjScaling: [:], activationFfnProjUpScaling: [:], activationFfnScaling: [:])
        (unetFixedMapper, unetFixed) = QwenImageFixed(
          FloatType.self,
          timesteps: 1, channels: 3_072, layers: 60, isBF16: true, activationQkScaling: [:],
          activationProjScaling: [:], activationFfnProjUpScaling: [:],
          activationFfnScaling: [:], numberOfReferenceImages: 0, useAdditionalTCond: false)
      case .auraflow:
        (unetMapper, unet) = AuraFlow(
          batchSize: 1, tokenLength: 256,
          height: 64, width: 64, maxSequence: 64, channels: 3072, layers: (4, 32),
          usesFlashAttention: .scaleMerged, of: FloatType.self
        )
        (unetFixedMapper, unetFixed) = AuraFlowFixed(
          batchSize: (1, 1), channels: 3072, layers: (4, 32),
          of: FloatType.self)
      case .zImage:
        (unet, unetMapper) = ZImage(
          batchSize: 1, height: 64, width: 64, textLength: 32, channels: 3840, layers: 30,
          activationQkScaling: [:], activationProjScaling: [:], activationFfnProjUpScaling: [:],
          activationFfnScaling: [:], usesFlashAttention: .scale1, isBF16: true)
        (unetFixed, unetFixedMapper) = ZImageFixed(
          batchSize: 1, tokenLength: (0, 32), channels: 3840, layers: 30, activationQkScaling: [:],
          activationProjScaling: [:], activationFfnProjUpScaling: [:], activationFfnScaling: [:],
          usesFlashAttention: .scale1, isBF16: true
        )
      case .ernieImage:
        (unetMapper, unet) = ErnieImage(
          batchSize: 1, height: 64, width: 64, textLength: 512, layers: 36, channels: 4_096,
          usesFlashAttention: .scale1)
        (unetFixedMapper, unetFixed) = ErnieImageFixed(timesteps: 1, channels: 4_096)
      case .cosmos2_5_2b:
        (unetMapper, unet) = Cosmos(
          batchSize: 1, height: 64, width: 64, textLength: 512,
          usesFlashAttention: .scale1)
        (unetFixedMapper, unetFixed) = CosmosFixed(
          timesteps: 1, batchSize: 1, textLength: 512, usesFlashAttention: .scale1)
        let (adapterMapper, adapter) = AnimaLLMAdapter(
          targetLength: (16, 16), sourceLength: (16, 16), usesFlashAttention: true)
        let sourceHiddenStates = graph.variable(.CPU, .WC(32, 1024), of: FloatType.self)
        let targetInputIDs = graph.variable(.CPU, format: .NHWC, shape: [32], of: Int32.self)
        let targetRot = graph.variable(
          Tensor<FloatType>(from: AnimaRotaryPositionEmbedding(sequenceLengths: (16, 16))))
        let sourceRot = graph.variable(
          Tensor<FloatType>(from: AnimaRotaryPositionEmbedding(sequenceLengths: (16, 16))))
        adapter.compile(inputs: [sourceHiddenStates, targetInputIDs, targetRot, sourceRot])
        var adapterMapping = ModelWeightMapping()
        for (key, value) in adapterMapper(.generativeModels) {
          adapterMapping["llm_adapter.\(key)"] = value
        }
        additionalMappings.append((prefix: "llm_adapter", mapping: adapterMapping))
      case .flux2:
        (unetMapper, unet) = Flux2(
          batchSize: 1, tokenLength: 512, referenceSequenceLength: 0, height: 64, width: 64,
          channels: 6144, layers: (8, 48), usesFlashAttention: .scale1, kvCache: false)
        (unetFixedMapper, unetFixed) = Flux2Fixed(
          timesteps: 1,
          channels: 6144, layers: (8, 48), numberOfReferenceImages: 0, guidanceEmbed: true,
          usesFlashAttention: .scale1, kvCache: false)
      case .flux2_9b:
        (unetMapper, unet) = Flux2(
          batchSize: 1, tokenLength: 512, referenceSequenceLength: 0, height: 64, width: 64,
          channels: 4096, layers: (8, 24), usesFlashAttention: .scale1, kvCache: false)
        (unetFixedMapper, unetFixed) = Flux2Fixed(
          timesteps: 1,
          channels: 4096, layers: (8, 24), numberOfReferenceImages: 0, guidanceEmbed: true,
          usesFlashAttention: .scale1, kvCache: false)
      case .flux2_4b:
        (unetMapper, unet) = Flux2(
          batchSize: 1, tokenLength: 512, referenceSequenceLength: 0, height: 64, width: 64,
          channels: 3072, layers: (5, 20), usesFlashAttention: .scale1, kvCache: false)
        (unetFixedMapper, unetFixed) = Flux2Fixed(
          timesteps: 1,
          channels: 3072, layers: (5, 20), numberOfReferenceImages: 0, guidanceEmbed: true,
          usesFlashAttention: .scale1, kvCache: false)
      case .ideogram4:
        (unetMapper, unet) = Ideogram4(
          batchSize: 1, height: 64, width: 64, textLength: 32, usesFlashAttention: .scale1)
        (unetFixedMapper, unetFixed) = Ideogram4Fixed(timesteps: 1)
      case .krea2:
        (unetMapper, unet) = Krea2(
          batchSize: 1, height: 64, width: 64, textLength: 32, usesFlashAttention: .scale1)
        (unetFixedMapper, unetFixed) = Krea2Fixed(timesteps: 1)
        let (textFusionMapper, textFusion) = Krea2TextFusionAdapter(
          batchSize: 1, textLength: (0, 32), usesFlashAttention: .scale1)
        let textFusionInput = graph.variable(
          .CPU, .HWC(1, 32, 12 * 2_560), of: FloatType.self)
        textFusion.compile(inputs: textFusionInput)
        additionalMappings.append(
          (
            prefix: "dit",
            mapping: textFusionMapper(isDiffusersFormat ? .diffusers : .generativeModels)
          ))
      case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b:
        fatalError()
      case .kandinsky21, .wurstchenStageB:
        fatalError()
      }
      let crossattn: [DynamicGraph.AnyTensor]
      switch modelVersion {
      case .longcatVideoAvatar1_5:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(2, 256), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 1, 5 * 5 * 1_280), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 1, 8 * 5 * 1_280), of: FloatType.self),
        ]
        tEmb = nil
      case .sdxlBase, .ssd1b:
        crossattn = [graph.variable(.CPU, .HWC(batchSize, 77, 2048), of: FloatType.self)]
      case .sdxlRefiner:
        crossattn = [graph.variable(.CPU, .HWC(batchSize, 77, 1280), of: FloatType.self)]
      case .svdI2v:
        let numFramesEmb = [320, 640, 1280, 1280].map { embeddingSize in
          let tensors = (0..<batchSize).map {
            graph.variable(
              timeEmbedding(
                timestep: Float($0), batchSize: 1, embeddingSize: embeddingSize, maxPeriod: 10_000)
            )
          }
          return DynamicGraph.Tensor<FloatType>(
            from: Concat(axis: 0)(inputs: tensors[0], Array(tensors[1...]))[0].as(of: Float.self))
        }
        crossattn =
          [graph.variable(.CPU, .HWC(batchSize, 1, 1024), of: FloatType.self)] + numFramesEmb
      case .wurstchenStageC:
        crossattn = [
          graph.variable(.CPU, .HWC(batchSize, 77, 1280), of: FloatType.self),
          graph.variable(.CPU, .HWC(batchSize, 1, 1280), of: FloatType.self),
          graph.variable(.CPU, .HWC(batchSize, 1, 1280), of: FloatType.self),
        ]
      case .pixart:
        crossattn = [
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
            ).reshaped(.HWC(1, 1, 256))
          ), graph.variable(.CPU, .HWC(batchSize, 77, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .sd3, .sd3Large:
        crossattn = [
          graph.variable(.CPU, .WC(batchSize, 2048), of: FloatType.self),
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: batchSize, embeddingSize: 256, maxPeriod: 10_000)
            ).reshaped(.WC(batchSize, 256))
          ), graph.variable(.CPU, .HWC(batchSize, 154, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .flux1:
        if distilledGuidanceLayers > 0 {
          crossattn = [
            graph.variable(.CPU, .HWC(batchSize, 256, 4096), of: FloatType.self),
            graph.variable(.CPU, .HWC(batchSize, 344, 64), of: FloatType.self),
          ]
        } else {
          crossattn = [
            graph.variable(.CPU, .HWC(batchSize, 256, 4096), of: FloatType.self),
            graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
            graph.variable(.CPU, .WC(batchSize, 768), of: FloatType.self),
            graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
          ]
        }
        tEmb = nil
      case .hunyuanVideo:
        crossattn = [
          graph.variable(.CPU, .HWC(batchSize, 20, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(batchSize, 768), of: FloatType.self),
          graph.variable(.CPU, .WC(batchSize, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .wan21_1_3b, .wan21_14b, .wan22_5b:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 257, 1280), of: FloatType.self),
        ]
        tEmb = nil
      case .hiDreamI1:
        crossattn =
          [
            graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
            graph.variable(.CPU, .WC(1, 2048), of: FloatType.self),
            graph.variable(.CPU, .HWC(1, 128, 4096), of: FloatType.self),
          ]
          + (0..<32).map { _ in
            graph.variable(.CPU, .HWC(1, 128, 4096), of: FloatType.self)  // Llama encoder hidden states.
          }
        tEmb = nil
      case .hiDreamO1, .qwenImage2_1, .minimaxH3:
        fatalError()
      case .qwenImage:
        crossattn =
          [
            graph.variable(.CPU, .HWC(1, 256, 3854), of: FloatType.self),
            graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          ]
        tEmb = nil
      case .zImage:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 32, 2560), of: FloatType.self),
          graph.variable(
            Tensor<FloatType>(
              from: ZImageRotaryPositionEmbedding(
                height: 0, width: 0, tokenLength: 32, imagePaddedLength: 0))),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .ernieImage:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 3072), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 1, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .cosmos2_5_2b:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 1024), of: FloatType.self),
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: 1, embeddingSize: 2048, maxPeriod: 10_000))),
        ]
        tEmb = nil
      case .auraflow:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 256, 2048), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .flux2:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 15360), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .flux2_9b:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 12288), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .flux2_4b:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 7680), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .ideogram4:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 32, 13 * 4_096), of: FloatType.self),
          graph.variable(
            Tensor<Int32>(
              from: Ideogram4IndicatorIDs(textLength: 32, imageLength: 0))),
          graph.variable(
            Tensor<Int32>(
              from: Ideogram4IndicatorIDs(textLength: 0, imageLength: 32 * 32))),
          graph.variable(
            Tensor<FloatType>(
              from: Ideogram4TimeEmbedding(timestep: 0, of: FloatType.self))),
        ]
        tEmb = nil
      case .krea2:
        crossattn = [
          graph.variable(.CPU, .HWC(1, 32, 2_560), of: FloatType.self),
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
            ).reshaped(.HWC(1, 1, 256))),
        ]
        tEmb = nil
      case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b:
        fatalError()
      case .v1, .v2, .kandinsky21, .wurstchenStageB:
        crossattn = []
      }
      if modelVersion == .v1 || modelVersion == .v2 {
        for (key, value) in stateDict {
          if !key.hasPrefix("model.diffusion_model.") {
            stateDict["model.diffusion_model.\(key)"] = value
          }
        }
        for (key, value) in stateDict {
          for name in DiffusersMapping.UNetPartials {
            if key.contains(name.1) {
              var newKey = key.replacingOccurrences(of: name.1, with: name.0)
              if key.contains(".resnets.") {
                for name in DiffusersMapping.ResNetsPartials {
                  if newKey.contains(name.1) {
                    newKey = newKey.replacingOccurrences(of: name.1, with: name.0)
                    break
                  }
                }
              }
              stateDict[newKey] = value
              break
            }
          }
        }
      } else if modelVersion == .wurstchenStageC {
        // Remove the model.diffusion_model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          }
        }
      } else if modelVersion == .sd3 || modelVersion == .sd3Large {
        // Remove the model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .pixart {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .flux1 {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .hiDreamI1 {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .hunyuanVideo {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .wan21_1_3b || modelVersion == .wan21_14b
        || modelVersion == .wan22_5b
      {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .qwenImage {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .auraflow {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .zImage {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .ernieImage {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .cosmos2_5_2b {
        // Remove the model.diffusion_model / model / net prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          } else if key.hasPrefix("net.") {
            stateDict[String(key.dropFirst(4))] = value
          }
        }
      } else if modelVersion == .ideogram4 {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      } else if modelVersion == .krea2 {
        // Remove the model.diffusion_model / model prefix.
        for (key, value) in stateDict {
          if key.hasPrefix("model.diffusion_model.") {
            stateDict[String(key.dropFirst(22))] = value
          } else if key.hasPrefix("model.") {
            stateDict[String(key.dropFirst(6))] = value
          }
        }
      }
      // In case it is not on high performance device and it is SDXL model, read the parameters directly from the mapping.
      if let unetReader = unetReader {
        let inputs: [DynamicGraph.Tensor<FloatType>] = [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
        unet.compile(inputs: inputs)
        try! unetReader(stateDict, archive)
        graph.openStore(filePath) {
          if unetFixed == nil {
            $0.removeAll()
          }
          $0.write("unet", model: unet)
        }
      } else if let unetMapper = unetMapper, let unetFixedMapper = unetFixedMapper,
        let unetFixed = unetFixed
      {
        try graph.openStore(filePath) { store in
          let UNetMapping: ModelWeightMapping
          let UNetMappingFixed: ModelWeightMapping
          let modelPrefix: String
          let modelPrefixFixed: String
          switch modelVersion {
          case .sdxlBase:
            if isDiffusersFormat {
              let inputs: [DynamicGraph.Tensor<FloatType>] =
                [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
              unet.compile(inputs: inputs)
              unetFixed.compile(inputs: crossattn)
              UNetMapping = unetMapper(.diffusers)
              UNetMappingFixed = unetFixedMapper(.diffusers)
            } else {
              UNetMapping = StableDiffusionMapping.UNetXLBase
              UNetMappingFixed = StableDiffusionMapping.UNetXLBaseFixed
            }
            modelPrefix = "unet"
            modelPrefixFixed = "unet_fixed"
          case .sdxlRefiner:
            if isDiffusersFormat {
              let inputs: [DynamicGraph.Tensor<FloatType>] =
                [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
              unet.compile(inputs: inputs)
              unetFixed.compile(inputs: crossattn)
              UNetMapping = unetMapper(.diffusers)
              UNetMappingFixed = unetFixedMapper(.diffusers)
            } else {
              UNetMapping = StableDiffusionMapping.UNetXLRefiner
              UNetMappingFixed = StableDiffusionMapping.UNetXLRefinerFixed
            }
            modelPrefix = "unet"
            modelPrefixFixed = "unet_fixed"
          case .ssd1b:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            UNetMappingFixed = unetFixedMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            modelPrefix = "unet"
            modelPrefixFixed = "unet_fixed"
          case .svdI2v:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            UNetMappingFixed = unetFixedMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            modelPrefix = "unet"
            modelPrefixFixed = "unet_fixed"
          case .wurstchenStageC:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(.generativeModels)
            UNetMappingFixed = unetFixedMapper(.generativeModels)
            modelPrefix = "stage_c"
            modelPrefixFixed = "stage_c_fixed"
          case .pixart, .sd3, .sd3Large, .flux1, .hunyuanVideo, .wan21_14b, .wan21_1_3b, .hiDreamI1,
            .hiDreamO1, .qwenImage2_1, .wan22_5b, .qwenImage, .cosmos2_5_2b, .auraflow, .zImage,
            .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .longcatVideoAvatar1_5,
            .ideogram4, .krea2:
            let inputs: [DynamicGraph.Tensor<FloatType>] =
              [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
            unet.compile(inputs: inputs)
            unetFixed.compile(inputs: crossattn)
            UNetMapping = unetMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            UNetMappingFixed = unetFixedMapper(isDiffusersFormat ? .diffusers : .generativeModels)
            modelPrefix = "dit"
            modelPrefixFixed = "dit"
          case .v1, .v2, .kandinsky21, .wurstchenStageB, .seedvr2_3b, .seedvr2_7b, .minimaxH3:
            fatalError()
          }
          func reverseMapping(original: ModelWeightMapping) -> [String: [String]] {
            var reversed: [String: [(Int, String)]] = [:]
            for (key, values) in original {
              for value in values {
                reversed[value, default: []].append((values.index, key))
              }
            }
            return reversed.mapValues {
              $0.sorted(by: { $0.0 < $1.0 }).map(\.1)
            }
          }
          let reverseUNetMapping = reverseMapping(original: UNetMapping)
          let reverseUNetMappingFixed = reverseMapping(original: UNetMappingFixed)
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
            if modelVersion == .zImage, let xPadTokenDescriptor = stateDict["x_pad_token"],
              let capPadTokenDescriptor = stateDict["cap_pad_token"]
            {
              try archive.with(xPadTokenDescriptor) { tensor in
                let tensor = Tensor<FloatType>(from: tensor)
                store.write("x_pad_token", tensor: tensor)
              }
              try archive.with(capPadTokenDescriptor) { tensor in
                let tensor = Tensor<FloatType>(from: tensor)
                store.write("cap_pad_token", tensor: tensor)
              }
            }
            var consumed = Set<String>()
            for (key, value) in UNetMapping.sorted(by: { $0.key < $1.key }) {
              guard let _ = stateDict[key], !consumed.contains(key) else {
                continue
              }
              let values = value.count == 1 ? (reverseUNetMapping[value[0]] ?? [key]) : [key]
              consumed.formUnion(values)
              let tensorDescriptors = values.compactMap { stateDict[$0] }
              try archive.with(tensorDescriptors) { tensors in
                guard !tensors.isEmpty else { return }
                let tensor: Tensor<FloatType>
                if tensors.count == 1 {
                  tensor = Tensor<FloatType>(from: tensors[0])
                } else {
                  let shape = [tensors.count] + Array(tensors[0].shape)
                  var combined = Tensor<FloatType>(.CPU, format: .NCHW, shape: TensorShape(shape))
                  for (i, tensor) in tensors.enumerated() {
                    let shape = tensor.shape
                    if shape.count == 4 {
                      combined[
                        i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                        Tensor<FloatType>(from: tensor).reshaped(
                          format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 3 {
                      combined[i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2]] = Tensor<
                        FloatType
                      >(from: tensor).reshaped(
                        format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 2 {
                      combined[i..<(i + 1), 0..<shape[0], 0..<shape[1]] = Tensor<FloatType>(
                        from: tensor
                      ).reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 1 {
                      combined[i..<(i + 1), 0..<shape[0]] = Tensor<FloatType>(from: tensor)
                        .reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    }
                  }
                  tensor = combined
                }
                if value.count > 1 {
                  value.write(
                    graph: graph,
                    to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                    isDiagonalDown: false
                  ) {
                    let _ = interrupt()
                    return "__\(modelPrefix)__[\($0)]"
                  }
                } else if let name = value.first {
                  // For FLUX.1 model, we can inspect x_embedder for whether it is a inpainting, depth, or canny.
                  if name == "t-x_embedder-0-0" && modelVersion == .flux1 {
                    let shape = tensor.shape
                    if shape[1] == 384 {
                      // This is an inpainting model.
                      modifier = .inpainting
                    } else if shape[1] == 128 {
                      modifier = .depth  // We cannot differentiate depth or canny, assuming it is depth.
                    }
                    store.write("__\(modelPrefix)__[\(name)]", tensor: tensor)
                  } else if name.contains("time_mixer") {
                    var f32Tensor = Tensor<Float>(from: tensor)
                    // Apply sigmoid transformation.
                    f32Tensor[0] = 1.0 / (1.0 + expf(-f32Tensor[0]))
                    store.write(
                      "__\(modelPrefix)__[\(name)]", tensor: Tensor<FloatType>(from: f32Tensor))
                  } else {
                    value.write(
                      graph: graph,
                      to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                      isDiagonalDown: false
                    ) {
                      let _ = interrupt()
                      return "__\(modelPrefix)__[\($0)]"
                    }
                  }
                  let _ = interrupt()
                }
              }
            }
            for (key, value) in UNetMappingFixed.sorted(by: { $0.key < $1.key }) {
              guard let _ = stateDict[key], !consumed.contains(key) else {
                continue
              }
              let values = value.count == 1 ? (reverseUNetMappingFixed[value[0]] ?? [key]) : [key]
              consumed.formUnion(values)
              let tensorDescriptors = values.compactMap { stateDict[$0] }
              try archive.with(tensorDescriptors) { tensors in
                guard !tensors.isEmpty else { return }
                let tensor: Tensor<FloatType>
                if tensors.count == 1 {
                  tensor = Tensor<FloatType>(from: tensors[0])
                } else {
                  let shape = [tensors.count] + Array(tensors[0].shape)
                  var combined = Tensor<FloatType>(.CPU, format: .NCHW, shape: TensorShape(shape))
                  for (i, tensor) in tensors.enumerated() {
                    let shape = tensor.shape
                    if shape.count == 3 {
                      combined[
                        i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                        Tensor<FloatType>(from: tensor).reshaped(
                          format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 3 {
                      combined[i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2]] = Tensor<
                        FloatType
                      >(from: tensor).reshaped(
                        format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 2 {
                      combined[i..<(i + 1), 0..<shape[0], 0..<shape[1]] = Tensor<FloatType>(
                        from: tensor
                      ).reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    } else if shape.count == 1 {
                      combined[i..<(i + 1), 0..<shape[0]] = Tensor<FloatType>(from: tensor)
                        .reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                    }
                  }
                  tensor = combined
                }
                if value.count > 1 {
                  value.write(
                    graph: graph,
                    to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                    isDiagonalDown: false
                  ) {
                    let _ = interrupt()
                    return "__\(modelPrefixFixed)__[\($0)]"
                  }
                } else if let name = value.first, modelVersion == .ideogram4,
                  name == "t-indicator_embedding-0-0"
                {
                  value.write(
                    graph: graph,
                    to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                    isDiagonalDown: false
                  ) { _ in
                    let _ = interrupt()
                    return "indicator_embedding"
                  }
                  let _ = interrupt()
                } else if !value.isEmpty {
                  value.write(
                    graph: graph,
                    to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                    isDiagonalDown: false
                  ) {
                    let _ = interrupt()
                    return "__\(modelPrefixFixed)__[\($0)]"
                  }
                  let _ = interrupt()
                }
              }
            }
            for additionalMapping in additionalMappings {
              let reverseAdditionalMapping = reverseMapping(original: additionalMapping.mapping)
              for (key, value) in additionalMapping.mapping.sorted(by: { $0.key < $1.key }) {
                guard let _ = stateDict[key], !consumed.contains(key) else {
                  continue
                }
                let values =
                  value.count == 1
                  ? (reverseAdditionalMapping[value[0]] ?? [key])
                  : [
                    key
                  ]
                consumed.formUnion(values)
                let tensorDescriptors = values.compactMap { stateDict[$0] }
                try archive.with(tensorDescriptors) { tensors in
                  guard !tensors.isEmpty else { return }
                  let tensor: Tensor<FloatType>
                  if tensors.count == 1 {
                    tensor = Tensor<FloatType>(from: tensors[0])
                  } else {
                    let shape = [tensors.count] + Array(tensors[0].shape)
                    var combined = Tensor<FloatType>(.CPU, format: .NCHW, shape: TensorShape(shape))
                    for (i, tensor) in tensors.enumerated() {
                      let shape = tensor.shape
                      if shape.count == 4 {
                        combined[
                          i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]] =
                          Tensor<FloatType>(from: tensor).reshaped(
                            format: .NCHW, shape: TensorShape([1] + Array(shape)))
                      } else if shape.count == 3 {
                        combined[i..<(i + 1), 0..<shape[0], 0..<shape[1], 0..<shape[2]] = Tensor<
                          FloatType
                        >(from: tensor).reshaped(
                          format: .NCHW, shape: TensorShape([1] + Array(shape)))
                      } else if shape.count == 2 {
                        combined[i..<(i + 1), 0..<shape[0], 0..<shape[1]] = Tensor<FloatType>(
                          from: tensor
                        ).reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                      } else if shape.count == 1 {
                        combined[i..<(i + 1), 0..<shape[0]] = Tensor<FloatType>(from: tensor)
                          .reshaped(format: .NCHW, shape: TensorShape([1] + Array(shape)))
                      }
                    }
                    tensor = combined
                  }
                  value.write(
                    graph: graph,
                    to: store, tensor: tensor, format: value.format, isDiagonalUp: false,
                    isDiagonalDown: false
                  ) {
                    let _ = interrupt()
                    return "__\(additionalMapping.prefix)__[\($0)]"
                  }
                  let _ = interrupt()
                }
              }
            }
          }
        }
      }
      try graph.openStore(filePath) {
        switch modelVersion {
        case .longcatVideoAvatar1_5:
          if $0.keys.count < 2_600 {
            throw Error.tensorWritesFailed
          }
        case .v1, .v2:
          if $0.keys.count != 718 {
            throw Error.tensorWritesFailed
          }
        case .sdxlBase:
          if $0.keys.count != 1820 + (hasEncoderHidProj ? 2 : 0) {
            throw Error.tensorWritesFailed
          }
        case .ssd1b:
          if $0.keys.count != 1012 + (hasEncoderHidProj ? 2 : 0) {
            throw Error.tensorWritesFailed
          }
        case .sdxlRefiner:
          if $0.keys.count != 1308 + (hasEncoderHidProj ? 2 : 0) {
            throw Error.tensorWritesFailed
          }
        case .svdI2v:
          if $0.keys.count != 1396 {
            throw Error.tensorWritesFailed
          }
        case .wurstchenStageC:
          // The first number is the number of tensors in stage c file, the second is the ones with effnet and previewer.
          if $0.keys.count != 1550 && $0.keys.count != 1550 + 374 {
            throw Error.tensorWritesFailed
          }
        case .pixart:
          if $0.keys.count != 754 {
            throw Error.tensorWritesFailed
          }
        case .sd3:
          if $0.keys.count != 1157 + (qkNorm ? 24 * 4 : 0) + dualAttentionLayers.count
            * (4 * 2 + 2 + 3 * 2)
          {
            throw Error.tensorWritesFailed
          }
        case .sd3Large:
          if $0.keys.count != 1981 {
            throw Error.tensorWritesFailed
          }
        case .flux1:
          let count = $0.keys.count
          if count != 1732 && count != 1728 && count != 1041 + max(distilledGuidanceLayers, 1) * 4 {
            throw Error.tensorWritesFailed
          }
        case .hunyuanVideo:
          if $0.keys.count != 1870 {
            throw Error.tensorWritesFailed
          }
        case .wan21_1_3b:
          let count = $0.keys.count
          if count != 986 && count != 1144 {
            throw Error.tensorWritesFailed
          }
        case .wan21_14b:
          let count = $0.keys.count
          if count != 1306 && count != 1514 {
            throw Error.tensorWritesFailed
          }
        case .hiDreamI1:
          if $0.keys.count != 1857 {
            throw Error.tensorWritesFailed
          }
        case .hiDreamO1, .qwenImage2_1, .minimaxH3:
          fatalError()
        case .wan22_5b:
          let count = $0.keys.count
          if count != 986 {
            throw Error.tensorWritesFailed
          }
        case .qwenImage:
          if $0.keys.count != 3121 {
            throw Error.tensorWritesFailed
          }
        case .auraflow:
          if $0.keys.count != 532 {
            throw Error.tensorWritesFailed
          }
        case .zImage:
          if $0.keys.count != 713 {
            throw Error.tensorWritesFailed
          }
        case .ernieImage:
          if $0.keys.count != 421 {
            throw Error.tensorWritesFailed
          }
        case .cosmos2_5_2b:
          if $0.keys.count != 856 {
            throw Error.tensorWritesFailed
          }
        case .flux2:
          if $0.keys.count != 600 {
            throw Error.tensorWritesFailed
          }
        case .flux2_9b:
          if $0.keys.count != 382 && $0.keys.count != 384 {
            throw Error.tensorWritesFailed
          }
        case .flux2_4b:
          if $0.keys.count != 292 && $0.keys.count != 294 {
            throw Error.tensorWritesFailed
          }
        case .ideogram4:
          if $0.keys.count != 730 {
            throw Error.tensorWritesFailed
          }
        case .krea2:
          if $0.keys.count != 581 {
            throw Error.tensorWritesFailed
          }
        case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b:
          fatalError()
        case .kandinsky21, .wurstchenStageB:
          fatalError()
        }
      }
      filePaths.append(filePath)
    }
    if let autoencoderFilePath = autoencoderFilePath {
      do {
        if autoencoderFilePath != filePath {
          // Redo the stateDict
          stateDict.removeAll()
          if let safeTensors = SafeTensors(url: URL(fileURLWithPath: autoencoderFilePath)) {
            archive = safeTensors
            let states = safeTensors.states
            stateDict = states
            for (key, value) in states {
              if key.hasPrefix("encoder.") || key.hasPrefix("decoder.")
                || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.")
              {
                stateDict["first_stage_model.\(key)"] = value
              }
            }
          } else if let zipArchive = Archive(
            url: URL(fileURLWithPath: autoencoderFilePath), accessMode: .read)
          {
            archive = zipArchive
            let rootObject = try Interpreter.unpickle(zip: zipArchive)
            guard
              let originalStateDict =
                (rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject["module"]
                  as? Interpreter.Dictionary)
            else {
              throw UnpickleError.dataNotFound
            }
            originalStateDict.forEach { key, value in
              guard let value = value as? TensorDescriptor else { return }
              stateDict[key] = value
              if key.hasPrefix("encoder.") || key.hasPrefix("decoder.")
                || key.hasPrefix("post_quant_conv.") || key.hasPrefix("quant_conv.")
              {
                stateDict["first_stage_model.\(key)"] = value
              }
            }
          } else {
            throw UnpickleError.dataNotFound
          }
        }
        try graph.withNoGrad {
          let encoderTensor = graph.variable(.CPU, .NHWC(1, 512, 512, 3), of: FloatType.self)
          let decoderTensor = graph.variable(.CPU, .NHWC(1, 64, 64, inputDim), of: FloatType.self)
          let (encoder, encoderReader, encoderWeightMapper) = Encoder(
            channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64,
            startHeight: 64, usesFlashAttention: false, format: .NHWC, outputChannels: inputDim)
          let (decoder, decoderReader, decoderWeightMapper) = Decoder(
            channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 64,
            startHeight: 64, inputChannels: inputDim, highPrecisionKeysAndValues: false,
            usesFlashAttention: false, paddingFinalConvLayer: false, format: .NHWC)
          decoder.compile(inputs: decoderTensor)
          encoder.compile(inputs: encoderTensor)
          let filePath = ModelZoo.filePathForModelDownloaded("\(modelName)_vae_f16.ckpt")
          // This is the name of diffusers.
          if stateDict.keys.contains(where: { $0.contains(".mid_block.") }) {
            // For diffusers, we use the weight mapper.
            try graph.openStore(filePath) { store in
              store.removeAll()
              try store.withTransaction {
                let decoderMapping = decoderWeightMapper(.diffusers)
                for (key, value) in decoderMapping {
                  guard let tensorDescriptor = stateDict[key] else {
                    continue
                  }
                  try archive.with(tensorDescriptor) { tensor in
                    let tensor = Tensor<FloatType>(from: tensor)
                    if let name = value.first {
                      store.write("__decoder__[\(name)]", tensor: tensor)
                    }
                  }
                }
                let encoderMapping = encoderWeightMapper(.diffusers)
                for (key, value) in encoderMapping {
                  guard let tensorDescriptor = stateDict[key] else {
                    continue
                  }
                  try archive.with(tensorDescriptor) { tensor in
                    let tensor = Tensor<FloatType>(from: tensor)
                    if let name = value.first {
                      store.write("__encoder__[\(name)]", tensor: tensor)
                    }
                  }
                }
              }
            }
          } else {
            try decoderReader(stateDict, archive)
            try encoderReader(stateDict, archive)
            graph.openStore(filePath) {
              $0.removeAll()
              $0.write("decoder", model: decoder)
              $0.write("encoder", model: encoder)
            }
          }
          try graph.openStore(filePath) {
            if $0.keys.count != 248 {
              throw Error.tensorWritesFailed
            }
          }
          filePaths.append(filePath)
        }
      } catch {
        throw Error.autoencoder(error)
      }
    }
    return (filePaths, modelVersion, modifier, inspectionResult)
  }
}

extension ModelImporter {
  public static func merge(efficientNetAndPreviewer: String, into file: String) {
    let graph = DynamicGraph()
    graph.openStore(efficientNetAndPreviewer, flags: .readOnly) { fromStore in
      let keys = fromStore.keys.filter {
        $0.hasPrefix("__effnet__[") || $0.hasPrefix("__previewer__[")
      }
      graph.openStore(file, flags: .truncateWhenClose) { toStore in
        for key in keys {
          guard let tensor = fromStore.read(key, codec: [.q6p, .q8p, .ezm7, .externalData]) else {
            continue
          }
          toStore.write(key, tensor: tensor)
        }
      }
    }
  }
}

extension ModelImporter {

  public static func inferModelSpecification(
    modelName: String,
    fileName: String,
    fileNames: [String],
    modelVersion: ModelVersion,
    modifier: SamplerModifier,
    inspectionResult: ModelImporter.InspectionResult,
    prefix: String,
    objective: Denoiser.Objective?,
    conditioning: Denoiser.Conditioning?,
    noiseDiscretization: ModelZoo.NoiseDiscretization?,
    upcastAttention: Bool,
    finetuneScale: UInt16
  ) -> (
    specification: ModelZoo.Specification,
    additionalModels: [(name: String, subtitle: String, file: String)],
    wurstchenStageCEffNetAndPreviewer: String?
  ) {

    // Extract text encoders based on model version
    var clipEncoder: String? = nil
    var additionalClipEncoders: [String]? = nil
    var t5Encoder: String? = nil
    let textEncoder: String?

    switch modelVersion {
    case .longcatVideoAvatar1_5:
      textEncoder = "umt5_xxl_encoder_q8p.ckpt"
    case .v1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .v2, .svdI2v:
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_h14_f16.ckpt")
      }
    case .sdxlBase, .ssd1b:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
    case .sd3:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
    case .sd3Large:
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
    case .pixart:
      textEncoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
    case .auraflow:
      textEncoder = fileNames.first {
        $0.hasSuffix("_pile_t5_xl_encoder_f16.ckpt")
      }
    case .sdxlRefiner:
      textEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_vit_bigg14_f16.ckpt")
      }
    case .flux1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .hunyuanVideo:
      textEncoder = fileNames.first {
        $0.hasSuffix("_llava_llama_3_8b_v1.1_q8p.ckpt")
      }
      clipEncoder = fileNames.first {
        $0.hasSuffix("_clip_vit_l14_f16.ckpt")
      }
    case .wan21_1_3b, .wan21_14b:
      textEncoder = fileNames.first {
        $0.hasSuffix("_umt5_xxl_encoder_f16.ckpt")
      }
      clipEncoder = fileNames.first {
        $0.hasSuffix("_open_clip_xlm_roberta_large_vit_h14_f16.ckpt")
      }
    case .hiDreamI1:
      textEncoder = fileNames.first {
        $0.hasSuffix("_llama_3.1_8b_instruct_f16.ckpt")
      }
      clipEncoder = fileNames.first {
        $0.hasSuffix("_long_clip_vit_l14_f16.ckpt")
      }
      additionalClipEncoders =
        (fileNames.first {
          $0.hasSuffix("_long_open_clip_vit_bigg14_f16.ckpt")
        }).map { [$0] }
      t5Encoder = fileNames.first {
        $0.hasSuffix("_t5_xxl_encoder_f16.ckpt")
      }
    case .hiDreamO1, .qwenImage2_1:
      textEncoder = nil
    case .wan22_5b:
      textEncoder = fileNames.first {
        $0.hasSuffix("_umt5_xxl_encoder_f16.ckpt")
      }
    case .qwenImage:
      textEncoder = fileNames.first {
        $0.hasSuffix("_qwen_2.5_vl_7b_f16.ckpt")
      }
    case .cosmos2_5_2b:
      textEncoder = fileNames.first {
        $0.hasSuffix("_qwen_3_0.6b_q8p.ckpt")
      }
    case .zImage:
      textEncoder = fileNames.first {
        $0.hasSuffix("_qwen_3_vl_4b_instruct_f16.ckpt")
      }
    case .ernieImage:
      textEncoder = fileNames.first {
        $0.hasSuffix("_ministral_3_3b_f16.ckpt")
      }
    case .flux2:
      textEncoder = fileNames.first {
        $0.hasSuffix("_mistral_small_3.2_24b_instruct_2506_f16.ckpt")
      }
    case .flux2_9b:
      textEncoder = fileNames.first {
        $0.hasSuffix("_qwen_3_8b_f16.ckpt")
      }
    case .flux2_4b:
      textEncoder = fileNames.first {
        $0.hasSuffix("_qwen_3_4b_f16.ckpt")
      }
    case .ltx2, .ltx2_3:
      textEncoder = fileNames.first {
        $0.hasSuffix("_gemma_3_12b_it_qat_q8p.ckpt")
      }
    case .seedvr2_3b, .seedvr2_7b:
      textEncoder = nil
    case .ideogram4:
      textEncoder = fileNames.first {
        $0.hasSuffix("_qwen_3_vl_8b_instruct_f16.ckpt")
          || $0.hasSuffix("_qwen_3_vl_8b_instruct_q8p.ckpt")
      }
    case .krea2:
      textEncoder = fileNames.first {
        $0.hasSuffix("_qwen_3_vl_4b_f16.ckpt")
          || $0.hasSuffix("_qwen_3_vl_4b_q8p.ckpt")
      }
      clipEncoder = "\(fileName)_f16.ckpt"
    case .wurstchenStageC:
      textEncoder = nil
    case .kandinsky21, .wurstchenStageB, .minimaxH3:
      fatalError()
    }

    let autoencoder = fileNames.first { $0.hasSuffix("_vae_f16.ckpt") }

    // Create base specification
    var specification = ModelZoo.Specification(
      name: modelName, file: "\(fileName)_f16.ckpt", prefix: prefix, version: modelVersion,
      upcastAttention: upcastAttention, defaultScale: finetuneScale,
      textEncoder: textEncoder, autoencoder: autoencoder, modifier: modifier,
      clipEncoder: clipEncoder, additionalClipEncoders: additionalClipEncoders,
      t5Encoder: t5Encoder, conditioning: conditioning, objective: objective,
      noiseDiscretization: noiseDiscretization
    )

    // Apply version-specific configurations
    var wurstchenStageCEffNetAndPreviewer: String? = nil
    var additionalModels = [(name: String, subtitle: String, file: String)]()

    switch modelVersion {
    case .longcatVideoAvatar1_5:
      if specification.textEncoder == nil {
        specification.textEncoder = "umt5_xxl_encoder_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "wan_v2.1_video_vae_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      specification.isConsistencyModel = true
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
    case .v1:
      break
    case .v2:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_h14_f16.ckpt"
      }
    case .sdxlBase, .sdxlRefiner, .ssd1b:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_bigg14_f16.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "clip_vit_l14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
    case .svdI2v:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_h14_vision_model_f16.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "open_clip_vit_h14_visual_proj_f16.ckpt"
      }
      specification.conditioning = .noise
      specification.objective = .v
      specification.noiseDiscretization = .edm(.init(sigmaMax: 700.0))
    case .wurstchenStageC:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_bigg14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "wurstchen_3.0_stage_a_hq_f16.ckpt"
      }
      specification.stageModels = ["wurstchen_3.0_stage_b_q6p_q8p.ckpt"]
      if ModelZoo.isModelDownloaded("wurstchen_3.0_stage_c_f32_f16.ckpt") {
        wurstchenStageCEffNetAndPreviewer = "wurstchen_3.0_stage_c_f32_f16.ckpt"
      } else if ModelZoo.isModelDownloaded("wurstchen_3.0_stage_c_f32_q6p_q8p.ckpt") {
        wurstchenStageCEffNetAndPreviewer = "wurstchen_3.0_stage_c_f32_q6p_q8p.ckpt"
      } else {
        wurstchenStageCEffNetAndPreviewer = "wurstchen_3.0_stage_c_effnet_previewer_f32_f16.ckpt"
        additionalModels.append(
          (
            name: "Stable Cascade (Würstchen v3.0) EfficientNet and Previewer",
            subtitle: ModelZoo.humanReadableNameForVersion(.wurstchenStageC),
            file: "wurstchen_3.0_stage_c_effnet_previewer_f32_f16.ckpt"
          ))
      }
    case .ideogram4:
      if specification.textEncoder == nil {
        specification.textEncoder = "qwen_3_vl_8b_instruct_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_2_vae_f16.ckpt"
      }
    case .krea2:
      if specification.textEncoder == nil {
        specification.textEncoder = "qwen_3_vl_4b_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "qwen_image_vae_f16.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "\(fileName)_f16.ckpt"
      }
    case .pixart:
      if specification.textEncoder == nil {
        specification.textEncoder = "t5_xxl_encoder_q6p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
      specification.noiseDiscretization = .ddpm(
        .init(linearStart: 0.0001, linearEnd: 0.02, timesteps: 1_000, linspace: .linearWrtBeta))
    case .auraflow:
      if specification.textEncoder == nil {
        specification.textEncoder = "pile_t5_xl_encoder_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "sdxl_vae_v1.0_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
    case .flux1:
      if specification.textEncoder == nil {
        specification.textEncoder = "t5_xxl_encoder_q6p.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "clip_vit_l14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_1_vae_f16.ckpt"
      }
      specification.highPrecisionAutoencoder = true
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      if inspectionResult.hasGuidanceEmbed {
        specification.guidanceEmbed = true
      }
      // For FLUX.1, the hires fix trigger scale is 1.5 of the finetune scale.
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
      if inspectionResult.distilledGuidanceLayers > 0 {
        specification.mmdit = ModelZoo.Specification.MMDiT(
          qkNorm: true, dualAttentionLayers: [],
          distilledGuidanceLayers: inspectionResult.distilledGuidanceLayers)
      }
    case .hunyuanVideo:
      if specification.textEncoder == nil {
        specification.textEncoder = "llava_llama_3_8b_v1.1_q8p.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "clip_vit_l14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "hunyuan_video_vae_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      specification.guidanceEmbed = true
      // For Hunyuan, the hires fix trigger scale is 1.5 of the finetune scale.
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
    case .wan21_1_3b, .wan21_14b:
      if specification.textEncoder == nil {
        specification.textEncoder = "umt5_xxl_encoder_q8p.ckpt"
      }
      if modifier == .inpainting && specification.clipEncoder == nil {
        specification.clipEncoder = "open_clip_xlm_roberta_large_vit_h14_f16.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "wan_v2.1_video_vae_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      // For Wan, the hires fix trigger scale is 1.5 of the finetune scale.
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
    case .wan22_5b:
      if specification.textEncoder == nil {
        specification.textEncoder = "umt5_xxl_encoder_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "wan_v2.2_video_vae_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      // For Wan, the hires fix trigger scale is 1.5 of the finetune scale.
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
    case .hiDreamI1:
      if specification.textEncoder == nil {
        specification.textEncoder = "llama_3.1_8b_instruct_q8p.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "long_clip_vit_l14_f16.ckpt"
      }
      if specification.additionalClipEncoders == nil {
        specification.additionalClipEncoders = ["long_open_clip_vit_bigg14_f16.ckpt"]
      }
      if specification.t5Encoder == nil {
        specification.t5Encoder = "t5_xxl_encoder_q6p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_1_vae_f16.ckpt"
      }
      specification.highPrecisionAutoencoder = true
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      // For HiDream, the hires fix trigger scale is 1.5 of the finetune scale.
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
    case .hiDreamO1, .qwenImage2_1:
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      specification.noiseScalingFactor = 7.5
      specification.paddedTextEncodingLength = 1024
    case .qwenImage:
      if specification.textEncoder == nil {
        specification.textEncoder = "qwen_2.5_vl_7b_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "qwen_image_vae_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
      specification.isBf16 = true
    case .cosmos2_5_2b:
      if specification.textEncoder == nil {
        specification.textEncoder = "qwen_3_0.6b_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "qwen_image_vae_f16.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "\(fileName)_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1)
      specification.noiseDiscretization = .rf(.init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1))
      specification.paddedTextEncodingLength = 512
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
    case .sd3, .sd3Large:
      if specification.textEncoder == nil {
        specification.textEncoder = "open_clip_vit_bigg14_f16.ckpt"
      }
      if specification.clipEncoder == nil {
        specification.clipEncoder = "clip_vit_l14_f16.ckpt"
      }
      if specification.t5Encoder == nil {
        specification.t5Encoder = "t5_xxl_encoder_q6p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "sd3_vae_f16.ckpt"
      }
      specification.mmdit = ModelZoo.Specification.MMDiT(
        qkNorm: inspectionResult.qkNorm, dualAttentionLayers: inspectionResult.dualAttentionLayers
      )
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
    case .zImage:
      if specification.textEncoder == nil {
        specification.textEncoder = "qwen_3_vl_4b_instruct_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_1_vae_f16.ckpt"
      }
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      specification.hiresFixScale = (finetuneScale * 3 + 1) / 2
      specification.mmdit = ModelZoo.Specification.MMDiT(
        qkNorm: true, dualAttentionLayers: [], activationQkScaling: [:],
        activationFfnProjUpScaling: [:])
      specification.isBf16 = true
    case .ernieImage:
      if specification.textEncoder == nil {
        specification.textEncoder = "ministral_3_3b_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_2_vae_f16.ckpt"
      }
      specification.highPrecisionAutoencoder = true
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1000))
      specification.hiresFixScale = finetuneScale * 2
    case .flux2:
      if specification.textEncoder == nil {
        specification.textEncoder = "mistral_small_3.2_24b_instruct_2506_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_2_vae_f16.ckpt"
      }
      specification.highPrecisionAutoencoder = true
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      if inspectionResult.hasGuidanceEmbed {
        specification.guidanceEmbed = true
      }
      // For FLUX.2, the hires fix trigger scale is 2 of the finetune scale.
      specification.hiresFixScale = finetuneScale * 2
    case .flux2_9b:
      if specification.textEncoder == nil {
        specification.textEncoder = "qwen_3_8b_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_2_vae_f16.ckpt"
      }
      specification.highPrecisionAutoencoder = true
      specification.paddedTextEncodingLength = 512
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      if inspectionResult.hasGuidanceEmbed {
        specification.guidanceEmbed = true
      }
      // For FLUX.2, the hires fix trigger scale is 2 of the finetune scale.
      specification.hiresFixScale = finetuneScale * 2
    case .flux2_4b:
      if specification.textEncoder == nil {
        specification.textEncoder = "qwen_3_4b_q8p.ckpt"
      }
      if specification.autoencoder == nil {
        specification.autoencoder = "flux_2_vae_f16.ckpt"
      }
      specification.highPrecisionAutoencoder = true
      specification.paddedTextEncodingLength = 512
      specification.objective = .u(conditionScale: 1000)
      specification.noiseDiscretization = .rf(
        .init(sigmaMin: 0, sigmaMax: 1, conditionScale: 1_000))
      if inspectionResult.hasGuidanceEmbed {
        specification.guidanceEmbed = true
      }
      // For FLUX.2, the hires fix trigger scale is 2 of the finetune scale.
      specification.hiresFixScale = finetuneScale * 2
    case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .minimaxH3:
      fatalError()
    case .kandinsky21, .wurstchenStageB:
      fatalError()
    }

    return (specification, additionalModels, wurstchenStageCEffNetAndPreviewer)
  }
}
