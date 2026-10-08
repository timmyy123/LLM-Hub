import Diffusion
import Fickling
import Foundation
import ModelZoo
import NNC
import WeightsCache
import ZIPFoundation

public enum LoRAImporter {
  public enum Error: Swift.Error {
    case modelVersionFailed
  }
  public static func modelWeightsMapping(
    by version: ModelVersion, qkNorm: Bool, dualAttentionLayers: [Int], format: [ModelWeightFormat]
  ) -> (ModelWeightMapping, ModelWeightMapping, [String: [String]], [(String, ModelWeightMapping)])
  {
    let graph = DynamicGraph()
    var UNetMapping = ModelWeightMapping()
    var UNetMappingFixed = ModelWeightMapping()
    guard version != .v1 && version != .v2 else {
      // Legacy compatibility.
      UNetMapping = StableDiffusionMapping.UNet
      let UNetMappingKeys = UNetMapping.keys
      if format.contains(.diffusers) {
        for key in UNetMappingKeys {
          var diffusersKey = key
          for name in DiffusersMapping.UNetPartials {
            if key.contains(name.0) {
              diffusersKey = key.replacingOccurrences(of: name.0, with: name.1)
              break
            }
          }
          if diffusersKey.contains(".resnets.") {
            for name in DiffusersMapping.ResNetsPartials {
              if diffusersKey.contains(name.0) {
                diffusersKey = diffusersKey.replacingOccurrences(of: name.0, with: name.1)
                break
              }
            }
          }
          if diffusersKey != key {
            UNetMapping[diffusersKey] = UNetMapping[key]
          }
        }
      }
      return (UNetMappingFixed, UNetMapping, [:], [])
    }
    let unet: Model
    let unetFixed: Model
    let unetMapper: ModelWeightMapper
    let unetFixedMapper: ModelWeightMapper
    switch version {
    case .longcatVideoAvatar1_5:
      fatalError()
    case .minimaxH3:
      (unetMapper, unet) = MiniMaxH3(
        hiddenSize: 5_376, layers: 50, textLength: 2, audioLength: 4, videoFrames: 1,
        videoHeight: 64, videoWidth: 64, usesFlashAttention: .scale1)
      (unetFixedMapper, unetFixed) = MiniMaxH3Fixed(
        timesteps: 1, hiddenSize: 5_376, layers: 50, textLength: (0, 2),
        usesFlashAttention: .scale1)
    case .sdxlBase:
      UNetMapping = StableDiffusionMapping.UNetXLBase
      UNetMappingFixed = StableDiffusionMapping.UNetXLBaseFixed
      (unet, _, unetMapper) = UNetXL(
        batchSize: 2, startHeight: 64, startWidth: 64,
        channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
        middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
        embeddingLength: (77, 77), injectIPAdapterLengths: [],
        upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
        isTemporalMixEnabled: false, of: FloatType.self)
      (unetFixed, _, unetFixedMapper) = UNetXLFixed(
        batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
        embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [10, 10]],
        middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
        usesFlashAttention: .none, isTemporalMixEnabled: false)
    case .sdxlRefiner:
      UNetMapping = StableDiffusionMapping.UNetXLRefiner
      UNetMappingFixed = StableDiffusionMapping.UNetXLRefinerFixed
      (unet, _, unetMapper) =
        UNetXL(
          batchSize: 2, startHeight: 64, startWidth: 64,
          channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
          middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
          embeddingLength: (77, 77), injectIPAdapterLengths: [],
          upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
          isTemporalMixEnabled: false, of: FloatType.self
        )
      (unetFixed, _, unetFixedMapper) = UNetXLFixed(
        batchSize: 2, startHeight: 64, startWidth: 64, channels: [384, 768, 1536, 1536],
        embeddingLength: (77, 77), inputAttentionRes: [2: [4, 4], 4: [4, 4]],
        middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
        usesFlashAttention: .none, isTemporalMixEnabled: false)
    case .sd3:
      (unetMapper, unet) = MMDiT(
        batchSize: 2, t: 77, height: 64, width: 64, channels: 1536, layers: 24,
        upcast: false, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
        posEmbedMaxSize: 192 /* This value doesn't matter */,
        usesFlashAttention: .none, of: FloatType.self)
      (unetFixedMapper, unetFixed) = MMDiTFixed(
        batchSize: 2, channels: 1536, layers: 24, dualAttentionLayers: dualAttentionLayers)
    case .sd3Large:
      (unetMapper, unet) = MMDiT(
        batchSize: 2, t: 77, height: 64, width: 64, channels: 2432, layers: 38,
        upcast: true, qkNorm: true, dualAttentionLayers: [], posEmbedMaxSize: 192,
        usesFlashAttention: .none, of: FloatType.self)
      (unetFixedMapper, unetFixed) = MMDiTFixed(
        batchSize: 2, channels: 2432, layers: 38, dualAttentionLayers: [])
    case .pixart:
      (unetMapper, unet) = PixArt(
        batchSize: 2, height: 64, width: 64, channels: 1152, layers: 28,
        tokenLength: (77, 77), usesFlashAttention: .none, of: FloatType.self)
      (unetFixedMapper, unetFixed) = PixArtFixed(
        batchSize: 2, channels: 1152, layers: 28, tokenLength: (77, 77),
        usesFlashAttention: false, of: FloatType.self)
    case .flux1:
      (unetMapper, unet) = Flux1(
        batchSize: 1, tokenLength: 256, referenceSequenceLength: 0, height: 64, width: 64,
        channels: 3072, layers: (19, 38),
        usesFlashAttention: .scaleMerged, contextPreloaded: true, injectControls: false,
        injectIPAdapterLengths: [:], outputResidual: false, inputResidual: false)
      (unetFixedMapper, unetFixed) = Flux1Fixed(
        batchSize: (1, 1), channels: 3072, layers: (19, 38), contextPreloaded: true,
        numberOfReferenceImages: 0, guidanceEmbed: true)
    case .hunyuanVideo:
      (unetMapper, unet) = Hunyuan(
        time: 1, height: 64, width: 64, textLength: 20, channels: 3072, layers: (20, 40),
        usesFlashAttention: .scaleMerged, outputResidual: false, inputResidual: false)
      (unetFixedMapper, unetFixed) = HunyuanFixed(
        timesteps: 1, channels: 3072, layers: (20, 40), textLength: (0, 20))
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
        channels: 5_120, layers: 40, vaceLayers: [], intermediateSize: 13_824, time: 1, height: 64,
        width: 64,
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
    case .hiDreamO1:
      fatalError()
    case .qwenImage2_1:
      (unetMapper, unet) = QwenImage2_1(
        FloatType.self, batchSize: 1, height: 64, width: 64, prefixLength: 132,
        channels: 4096, layers: 32, usesFlashAttention: .scaleMerged)
      (unetFixedMapper, unetFixed) = QwenImage2_1Fixed(
        FloatType.self, batchSize: 1, textLength: 128, referenceLength: 4, timesteps: 1,
        channels: 4096, layers: 32, segments: [(128, false), (4, true)],
        usesFlashAttention: .scaleMerged)
    case .qwenImage:
      (unetMapper, unet) = QwenImage(
        batchSize: 1, height: 64, width: 64, textLength: 128, referenceSequenceLength: 0,
        channels: 3_072, layers: 60, usesFlashAttention: .scale1, isBF16: false,
        isQwenImageLayered: false, zeroTimestepForReference: false, activationQkScaling: [:],
        activationProjScaling: [:], activationFfnProjUpScaling: [:], activationFfnScaling: [:])
      (unetFixedMapper, unetFixed) = QwenImageFixed(
        FloatType.self,
        timesteps: 1, channels: 3_072, layers: 60, isBF16: false, activationQkScaling: [:],
        activationProjScaling: [:], activationFfnProjUpScaling: [:],
        activationFfnScaling: [:], numberOfReferenceImages: 0, useAdditionalTCond: false)
    case .zImage:
      (unet, unetMapper) = ZImage(
        batchSize: 1, height: 64, width: 64, textLength: 32, channels: 3840, layers: 30,
        activationQkScaling: [:], activationProjScaling: [:], activationFfnProjUpScaling: [:],
        activationFfnScaling: [:], usesFlashAttention: .scale1, isBF16: false)
      (unetFixed, unetFixedMapper) = ZImageFixed(
        batchSize: 1, tokenLength: (0, 32), channels: 3840, layers: 30, activationQkScaling: [:],
        activationProjScaling: [:], activationFfnProjUpScaling: [:], activationFfnScaling: [:],
        usesFlashAttention: .scale1, isBF16: false
      )
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
    case .ltx2:
      (unetMapper, unet) = LTX2(
        time: 16, h: 16, w: 16, textLength: 1024, audioFrames: 121, channels: (4096, 2048),
        layers: 48, tokenModulation: false, KV: true, usesFlashAttention: .scale1,
        useGatedAttention: false,
        textCrossAttentionAdaLN: false)
      (unetFixedMapper, unetFixed) = LTX2Fixed(
        time: 16, textLength: 1024, audioFrames: 121, timesteps: 1, channels: (4096, 2048),
        layers: 48, contextProjection: true, textCrossAttentionAdaLN: false, KV: true,
        usesFlashAttention: .scale1)
    case .ltx2_3:
      (unetMapper, unet) = LTX2(
        time: 16, h: 16, w: 16, textLength: 1024, audioFrames: 121, channels: (4096, 2048),
        layers: 48, tokenModulation: false, KV: false, usesFlashAttention: .scale1,
        useGatedAttention: true,
        textCrossAttentionAdaLN: true)
      (unetFixedMapper, unetFixed) = LTX2Fixed(
        time: 16, textLength: 1024, audioFrames: 121, timesteps: 1, channels: (4096, 2048),
        layers: 48, contextProjection: false, textCrossAttentionAdaLN: true, KV: false,
        usesFlashAttention: .scale1)
    case .ideogram4:
      (unetMapper, unet) = Ideogram4(
        batchSize: 1, height: 64, width: 64, textLength: 128, usesFlashAttention: .scale1)
      (unetFixedMapper, unetFixed) = Ideogram4Fixed(timesteps: 1)
    case .krea2:
      (unetMapper, unet) = Krea2(
        batchSize: 1, height: 64, width: 64, textLength: 128, usesFlashAttention: .scale1)
      (unetFixedMapper, unetFixed) = Krea2Fixed(timesteps: 1)
    case .auraflow:
      fatalError()
    case .v1, .v2, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .seedvr2_3b,
      .seedvr2_7b:
      fatalError()
    case .ssd1b:
      (unet, _, unetMapper) = UNetXL(
        batchSize: 2, startHeight: 64, startWidth: 64,
        channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
        middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
        embeddingLength: (77, 77), injectIPAdapterLengths: [],
        upcastAttention: ([:], false, [:]), usesFlashAttention: .none, injectControls: false,
        isTemporalMixEnabled: false, of: FloatType.self)
      (unetFixed, _, unetFixedMapper) = UNetXLFixed(
        batchSize: 2, startHeight: 64, startWidth: 64, channels: [320, 640, 1280],
        embeddingLength: (77, 77), inputAttentionRes: [2: [2, 2], 4: [4, 4]],
        middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
        usesFlashAttention: .none, isTemporalMixEnabled: false)
    }
    var otherMappings = [(String, ModelWeightMapping)]()
    return graph.withNoGrad {
      let inputDim: Int
      let conditionalLength: Int
      switch version {
      case .longcatVideoAvatar1_5:
        fatalError()
      case .minimaxH3:
        inputDim = 24
        conditionalLength = 5_120
      case .v1:
        inputDim = 4
        conditionalLength = 768
      case .v2:
        inputDim = 4
        conditionalLength = 1024
      case .sdxlBase, .sdxlRefiner, .ssd1b:
        inputDim = 4
        conditionalLength = 1280
      case .sd3, .sd3Large:
        inputDim = 16
        conditionalLength = 4096
      case .pixart:
        inputDim = 4
        conditionalLength = 4096
      case .auraflow:
        inputDim = 4
        conditionalLength = 2048
      case .flux1:
        inputDim = 16
        conditionalLength = 4096
      case .hunyuanVideo:
        inputDim = 16
        conditionalLength = 4096
      case .wan21_1_3b, .wan21_14b:
        inputDim = 16
        conditionalLength = 4096
      case .wan22_5b:
        inputDim = 48
        conditionalLength = 4096
      case .hiDreamI1:
        inputDim = 16
        conditionalLength = 4096
      case .hiDreamO1:
        fatalError()
      case .qwenImage2_1:
        inputDim = 64
        conditionalLength = 4096
      case .qwenImage:
        inputDim = 16
        conditionalLength = 3854
      case .zImage:
        inputDim = 16
        conditionalLength = 2560
      case .flux2:
        inputDim = 32
        conditionalLength = 15360
      case .flux2_9b:
        inputDim = 32
        conditionalLength = 12288
      case .flux2_4b:
        inputDim = 32
        conditionalLength = 7680
      case .ernieImage:
        inputDim = 32
        conditionalLength = 4096
      case .cosmos2_5_2b:
        inputDim = 16
        conditionalLength = 1024
      case .ltx2:
        inputDim = 128
        conditionalLength = 3840
      case .ltx2_3:
        inputDim = 128
        conditionalLength = 6144
      case .ideogram4:
        inputDim = 32
        conditionalLength = 4096
      case .krea2:
        inputDim = 16
        conditionalLength = 6144
      case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .seedvr2_3b, .seedvr2_7b:
        fatalError()
      }
      let crossattn: [DynamicGraph.AnyTensor]
      let tEmb: DynamicGraph.Tensor<FloatType>?
      let isCfgEnabled: Bool
      let isGuidanceEmbedEnabled: Bool
      switch version {
      case .longcatVideoAvatar1_5:
        fatalError()
      case .minimaxH3:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        tEmb = nil
        crossattn = [
          graph.variable(.CPU, .HWC(1, 2, 5_120), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 2, 256), of: FloatType.self),
        ]
      case .sdxlBase, .ssd1b:
        isCfgEnabled = true
        isGuidanceEmbedEnabled = false
        crossattn = [graph.variable(.CPU, .HWC(2, 77, 2048), of: FloatType.self)]
        tEmb = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: 981, batchSize: 2, embeddingSize: 320,
              maxPeriod: 10_000)
          ))
      case .sdxlRefiner:
        isCfgEnabled = true
        isGuidanceEmbedEnabled = false
        crossattn = [graph.variable(.CPU, .HWC(2, 77, 1280), of: FloatType.self)]
        tEmb = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: 981, batchSize: 2, embeddingSize: 384,
              maxPeriod: 10_000)
          ))
      case .pixart:
        isCfgEnabled = true
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000)
            ).reshaped(.HWC(1, 1, 256))
          ), graph.variable(.CPU, .HWC(2, 77, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .sd3, .sd3Large:
        isCfgEnabled = true
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(.CPU, .WC(2, 2048), of: FloatType.self),
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: 2, embeddingSize: 256, maxPeriod: 10_000)
            ).reshaped(.WC(2, 256))
          ), graph.variable(.CPU, .HWC(2, 154, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .flux1:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = true
        crossattn = [
          graph.variable(.CPU, .HWC(1, 256, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 768), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .hunyuanVideo:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = true
        crossattn = [
          graph.variable(.CPU, .HWC(1, 20, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 768), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .wan21_1_3b, .wan21_14b, .wan22_5b:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 257, 1280), of: FloatType.self),
        ]
        tEmb = nil
      case .hiDreamI1:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
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
      case .hiDreamO1:
        fatalError()
      case .qwenImage2_1:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(.CPU, .HWC(1, 128, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(2, 256), of: FloatType.self),
          graph.variable(.CPU, .NHWC(1, 132, 1, 128), of: FloatType.self),
          graph.variable(.CPU, .WC(4, 64), of: FloatType.self),
        ]
        tEmb = nil
      case .qwenImage:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        crossattn =
          [
            graph.variable(.CPU, .HWC(1, 256, 3854), of: FloatType.self),
            graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          ]
        tEmb = nil
      case .zImage:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(.CPU, .HWC(1, 32, 2560), of: FloatType.self),
          graph.variable(
            Tensor<FloatType>(
              from: ZImageRotaryPositionEmbedding(
                height: 0, width: 0, tokenLength: 32, imagePaddedLength: 0))),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .flux2:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = true
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 15360), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .flux2_9b:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = true
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 12288), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .flux2_4b:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = true
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 7680), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 256), of: FloatType.self),
        ]
        tEmb = nil
      case .ernieImage:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 3072), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 1, 4096), of: FloatType.self),
        ]
        tEmb = nil
      case .cosmos2_5_2b:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(.CPU, .HWC(1, 512, 1024), of: FloatType.self),
          graph.variable(
            Tensor<FloatType>(
              from: timeEmbedding(
                timestep: 1000, batchSize: 1, embeddingSize: 2048, maxPeriod: 10_000))
          ),
        ]
        tEmb = nil
        let (adapterMapper, adapter) = AnimaLLMAdapter(
          targetLength: (16, 16), sourceLength: (16, 16), usesFlashAttention: true)
        let sourceHiddenStates = graph.variable(.CPU, .WC(32, 1024), of: FloatType.self)
        let targetInputIDs = graph.variable(.CPU, format: .NHWC, shape: [32], of: Int32.self)
        let targetRot = graph.variable(
          Tensor<FloatType>(from: AnimaRotaryPositionEmbedding(sequenceLengths: (16, 16))))
        let sourceRot = graph.variable(
          Tensor<FloatType>(from: AnimaRotaryPositionEmbedding(sequenceLengths: (16, 16))))
        adapter.compile(inputs: [sourceHiddenStates, targetInputIDs, targetRot, sourceRot])
        otherMappings.append(
          (
            "llm_adapter",
            adapterMapper(format.contains(.diffusers) ? .diffusers : .generativeModels)
          ))
      case .ltx2:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        crossattn = [
          graph.variable(.CPU, .HWC(1, 1024, 3840), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 1024, 3840), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 1, 256), of: FloatType.self),
        ]
        tEmb = nil
        let (videoConnector, videoConnectorMapper) = Embedding1DConnector(
          prefix: "embeddings_connector", layers: 2, batchSize: 1,
          tokenLength: 1024, headDimension: 128, numberOfHeads: 30, usesFlashAttention: true,
          useGatedAttention: false
        )
        let (audioConnector, audioConnectorMapper) = Embedding1DConnector(
          prefix: "audio_embeddings_connector", layers: 2, batchSize: 1,
          tokenLength: 1024, headDimension: 128, numberOfHeads: 30, usesFlashAttention: true,
          useGatedAttention: false
        )
        let rotaryEmbedding1D = graph.variable(
          Tensor<FloatType>(
            from: LTX2RotaryPositionEmbedding1D(
              tokenLength: 1024, maxLength: 4096, channels: 3840, headDimension: 128
            )))
        let hiddenStates = graph.variable(.CPU, .HWC(1, 1024, 3840), of: FloatType.self)
        videoConnector.compile(inputs: hiddenStates, rotaryEmbedding1D)
        audioConnector.compile(inputs: hiddenStates, rotaryEmbedding1D)
        if format.contains(.diffusers) {
          otherMappings.append(("text_video_connector", videoConnectorMapper(.diffusers)))
          otherMappings.append(("text_audio_connector", audioConnectorMapper(.diffusers)))
        } else {
          otherMappings.append(("text_video_connector", videoConnectorMapper(.generativeModels)))
          otherMappings.append(("text_audio_connector", audioConnectorMapper(.generativeModels)))
        }
        var textFeatureExtractorMapping = ModelWeightMapping()
        textFeatureExtractorMapping["text_embedding_projection.aggregate_embed.weight"] = ["t-0-0"]
        otherMappings.append(("text_feature_extractor", textFeatureExtractorMapping))
      case .ltx2_3:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        crossattn = [graph.variable(.CPU, .HWC(1, 1, 256), of: FloatType.self)]
        tEmb = nil
        let (videoConnector, videoConnectorMapper) = Embedding1DConnector(
          prefix: "model.diffusion_model.video_embeddings_connector", layers: 8, batchSize: 1,
          tokenLength: 1024, headDimension: 128, numberOfHeads: 32, usesFlashAttention: true,
          useGatedAttention: true
        )
        let (audioConnector, audioConnectorMapper) = Embedding1DConnector(
          prefix: "model.diffusion_model.audio_embeddings_connector", layers: 8, batchSize: 1,
          tokenLength: 1024, headDimension: 64, numberOfHeads: 32, usesFlashAttention: true,
          useGatedAttention: true
        )
        let videoRotaryEmbedding1D = graph.variable(
          Tensor<FloatType>(
            from: LTX2RotaryPositionEmbedding1D(
              tokenLength: 1024, maxLength: 4096, channels: 4096, headDimension: 128
            )))
        let audioRotaryEmbedding1D = graph.variable(
          Tensor<FloatType>(
            from: LTX2RotaryPositionEmbedding1D(
              tokenLength: 1024, maxLength: 4096, channels: 2048, headDimension: 64
            )))
        let videoHiddenStates = graph.variable(.CPU, .HWC(1, 1024, 4096), of: FloatType.self)
        let audioHiddenStates = graph.variable(.CPU, .HWC(1, 1024, 2048), of: FloatType.self)
        videoConnector.compile(inputs: videoHiddenStates, videoRotaryEmbedding1D)
        audioConnector.compile(inputs: audioHiddenStates, audioRotaryEmbedding1D)
        if format.contains(.diffusers) {
          otherMappings.append(("text_video_connector", videoConnectorMapper(.diffusers)))
          otherMappings.append(("text_audio_connector", audioConnectorMapper(.diffusers)))
        } else {
          otherMappings.append(("text_video_connector", videoConnectorMapper(.generativeModels)))
          otherMappings.append(("text_audio_connector", audioConnectorMapper(.generativeModels)))
        }
        let textFeatureInput = Input()
        let textVideoAggregateEmbed = Dense(count: 4096, name: "video_aggregate_embed")
        let textAudioAggregateEmbed = Dense(count: 2048, name: "audio_aggregate_embed")
        let concat = Concat(axis: 1)
        concat.flags = [.disableOpt]
        let textFeatureExtractor = Model(
          [textFeatureInput],
          [
            concat([
              textVideoAggregateEmbed(textFeatureInput), textAudioAggregateEmbed(textFeatureInput),
            ])
          ]
        )
        let textFeatureExtractorInput = graph.variable(.CPU, .WC(1, 3840 * 49), of: FloatType.self)
        textFeatureExtractor.compile(inputs: textFeatureExtractorInput)
        var textFeatureExtractorMapping = ModelWeightMapping()
        textFeatureExtractorMapping["text_embedding_projection.video_aggregate_embed.weight"] = [
          textVideoAggregateEmbed.weight.name
        ]
        textFeatureExtractorMapping["text_embedding_projection.video_aggregate_embed.bias"] = [
          textVideoAggregateEmbed.bias.name
        ]
        textFeatureExtractorMapping["text_embedding_projection.audio_aggregate_embed.weight"] = [
          textAudioAggregateEmbed.weight.name
        ]
        textFeatureExtractorMapping["text_embedding_projection.audio_aggregate_embed.bias"] = [
          textAudioAggregateEmbed.bias.name
        ]
        otherMappings.append(("text_feature_extractor", textFeatureExtractorMapping))
      case .ideogram4:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        let textLength = 128
        let imageLength = 32 * 32
        crossattn = [
          graph.variable(.CPU, .HWC(1, textLength, 4096), of: FloatType.self),
          graph.variable(Ideogram4IndicatorIDs(textLength: textLength, imageLength: 0)),
          graph.variable(Ideogram4IndicatorIDs(textLength: 0, imageLength: imageLength)),
          graph.variable(Ideogram4TimeEmbedding(timestep: 1, of: FloatType.self)),
        ]
        tEmb = nil
      case .krea2:
        isCfgEnabled = false
        isGuidanceEmbedEnabled = false
        let textLength = 128
        crossattn = [
          graph.variable(.CPU, .HWC(1, textLength, 2560), of: FloatType.self),
          graph.variable(.CPU, .HWC(1, 1, 256), of: FloatType.self),
        ]
        tEmb = nil
        let (adapterMapper, adapter) = Krea2TextFusionAdapter(
          batchSize: 1, textLength: (0, textLength), usesFlashAttention: .scale1)
        let hiddenStates = graph.variable(.CPU, .HWC(1, textLength, 12 * 2560), of: FloatType.self)
        adapter.compile(inputs: hiddenStates)
        var adapterMapping = ModelWeightMapping()
        if format.contains(.generativeModels) {
          adapterMapping.merge(adapterMapper(.generativeModels)) { v, _ in v }
        }
        if format.contains(.diffusers) {
          adapterMapping.merge(adapterMapper(.diffusers)) { v, _ in v }
        }
        otherMappings.append(("dit", adapterMapping))
      case .auraflow:
        fatalError()
      case .v1, .v2, .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .seedvr2_3b,
        .seedvr2_7b:
        fatalError()
      }
      unetFixed.compile(inputs: crossattn)
      let xTensor = graph.variable(
        .CPU, .NHWC(isCfgEnabled ? 2 : 1, 64, 64, inputDim), of: FloatType.self)
      var cArr: [DynamicGraph.Tensor<FloatType>]
      if version == .flux1 {  // This logic should be switch and for auraflow too.
        cArr = [
          graph.variable(.CPU, .HWC(1, 256, 4096), of: FloatType.self),
          graph.variable(.CPU, .WC(1, 768), of: FloatType.self),
        ]
      } else {
        let cTensor = graph.variable(
          .CPU, .HWC(isCfgEnabled ? 2 : 1, 77, conditionalLength), of: FloatType.self)
        cArr = [cTensor]
        cArr.insert(
          graph.variable(.CPU, .HWC(isCfgEnabled ? 2 : 1, 77, 768), of: FloatType.self),
          at: 0)
        cArr.append(
          graph.variable(.CPU, .WC(isCfgEnabled ? 2 : 1, 1280), of: FloatType.self))
      }
      let fixedEncoder = UNetFixedEncoder<FloatType>(
        filePath: "", version: version, modifier: .none, dualAttentionLayers: dualAttentionLayers,
        activationQkScaling: [:],
        activationProjScaling: [:], activationFfnProjUpScaling: [:], activationFfnScaling: [:],
        usesFlashAttention: .none,
        zeroNegativePrompt: false,
        isQuantizedModel: false,
        canRunLoRASeparately: false, externalOnDemand: false,
        deviceProperties: DeviceProperties(
          isFreadPreferred: true, memoryCapacity: .high, isNHWCPreferred: true,
          cacheUri: URL(fileURLWithPath: NSTemporaryDirectory()), isPartialOffloadPreferred: false),
        weightsCache: WeightsCache(maxTotalCacheSize: 0, memorySubsystem: .UMA))
      for c in cArr {
        c.full(0)
      }
      // These values doesn't matter, it won't affect the model shape, just the input vector.
      let vectors: [DynamicGraph.Tensor<FloatType>]
      switch version {
      case .longcatVideoAvatar1_5:
        fatalError()
      case .sdxlBase, .ssd1b:
        vectors = [graph.variable(.CPU, .WC(2, 2816), of: FloatType.self)]
      case .sdxlRefiner:
        vectors = [graph.variable(.CPU, .WC(2, 2560), of: FloatType.self)]
      case .svdI2v:
        vectors = [graph.variable(.CPU, .WC(2, 768), of: FloatType.self)]
      case .wurstchenStageC, .wurstchenStageB, .pixart, .sd3, .sd3Large, .auraflow, .flux1,
        .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage2_1, .qwenImage,
        .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3,
        .cosmos2_5_2b, .ideogram4, .krea2, .minimaxH3:
        vectors = []
      case .kandinsky21, .v1, .v2, .seedvr2_3b, .seedvr2_7b:
        fatalError()
      }
      switch version {
      case .longcatVideoAvatar1_5:
        fatalError()
      case .minimaxH3:
        cArr =
          [
            graph.variable(.CPU, .HWC(1, 4, 32), of: FloatType.self),
            graph.variable(.CPU, .HWC(1, 2, 5_376), of: FloatType.self),
            graph.variable(.CPU, .NHWC(1, 1_030, 1, 128), of: FloatType.self),
          ]
          + (0..<(50 * 18 + 4)).map { _ in
            graph.variable(.CPU, .HWC(1, 1, 5_376), of: FloatType.self)
          }
      case .sdxlBase, .ssd1b, .sdxlRefiner, .svdI2v, .wurstchenStageC, .wurstchenStageB, .pixart,
        .sd3, .sd3Large, .auraflow:
        // These values doesn't matter, it won't affect the model shape, just the input vector.
        cArr =
          vectors
          + fixedEncoder.encode(
            isCfgEnabled: isCfgEnabled, textGuidanceScale: 3.5, guidanceEmbed: 3.5,
            isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, distilledGuidanceLayers: 0,
            modifier: .none,
            textEncoding: cArr.map({ $0.toGPU(0) }), timesteps: [0], audioShiftRatio: 1,
            batchSize: isCfgEnabled ? 2 : 1, startHeight: 64,
            startWidth: 64,
            tokenLengthUncond: 77, tokenLengthCond: 77, lora: [],
            tiledDiffusion: TiledConfiguration(
              isEnabled: false, tileSize: .init(width: 0, height: 0), tileOverlap: 0),
            teaCache: TeaCacheConfiguration(
              coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0),
            isBF16: false, injectedControls: [], referenceImages: [], referenceAudios: []
          ).0.map({ DynamicGraph.Tensor<FloatType>($0).toCPU() })
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
            batchSize: 1, channels: 3072, layers: (20, 40), textLength: 20
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
      case .hiDreamO1:
        fatalError()
      case .qwenImage2_1:
        cArr =
          [graph.variable(.CPU, .NHWC(1, 64 * 64, 1, 128), of: FloatType.self)]
          + (0..<5).map { _ in graph.variable(.CPU, .WC(1, 4096), of: FloatType.self) }
          + (0..<64).map { _ in
            graph.variable(.CPU, .NHWC(1, 132, 32, 128), of: FloatType.self)
          }
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
      case .ltx2:
        let (rotaryEmbedding, rotaryEmbeddingAudio, rotaryEmbeddingVideoToAudio) =
          LTX2VideoAudioRotaryPositionEmbedding(
            time: 16, height: 16, width: 16, audioTime: 121,
            channels: (4096, 2048), numberOfHeads: 32)
        cArr =
          [
            graph.variable(.CPU, format: .NHWC, shape: [1, 121, 128], of: FloatType.self),  // audio
            graph.variable(Tensor<FloatType>(from: rotaryEmbedding)),
            graph.variable(Tensor<FloatType>(from: rotaryEmbeddingAudio)),
            graph.variable(Tensor<FloatType>(from: rotaryEmbeddingVideoToAudio)),
          ]
          + LTX2FixedOutputShapes(
            time: 16, textLength: 1024, audioFrames: 121, timesteps: 1, channels: (4096, 2048),
            layers: 48, textCrossAttentionAdaLN: false, KV: true
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .ltx2_3:
        let (rotaryEmbedding, rotaryEmbeddingAudio, rotaryEmbeddingVideoToAudio) =
          LTX2VideoAudioRotaryPositionEmbedding(
            time: 16, height: 16, width: 16, audioTime: 121,
            channels: (4096, 2048), numberOfHeads: 32)
        cArr =
          [
            graph.variable(.CPU, format: .NHWC, shape: [1, 121, 128], of: FloatType.self),  // audio
            graph.variable(Tensor<FloatType>(from: rotaryEmbedding)),
            graph.variable(Tensor<FloatType>(from: rotaryEmbeddingAudio)),
            graph.variable(Tensor<FloatType>(from: rotaryEmbeddingVideoToAudio)),
          ]
          + LTX2FixedOutputShapes(
            time: 16, textLength: 1024, audioFrames: 121, timesteps: 1, channels: (4096, 2048),
            layers: 48, textCrossAttentionAdaLN: true, KV: false
          ).map {
            graph.variable(.CPU, format: .NHWC, shape: $0, of: FloatType.self)
          }
      case .ideogram4:
        let textLength = 128
        let imageLength = 32 * 32
        cArr =
          [
            graph.variable(.CPU, .HWC(1, textLength, 4_608), of: FloatType.self),
            graph.variable(.CPU, .HWC(1, imageLength, 4_608), of: FloatType.self),
            graph.variable(
              Tensor<FloatType>(
                from: Ideogram4RotaryPositionEmbedding(
                  textLength: textLength, gridHeight: 32, gridWidth: 32, of: FloatType.self))),
          ]
          + (0..<(34 * 4 + 1)).map { _ in
            graph.variable(.CPU, .HWC(1, 1, 4_608), of: FloatType.self)
          }
      case .krea2:
        let textLength = 128
        let imageLength = 32 * 32
        cArr =
          [
            graph.variable(.CPU, .HWC(1, textLength, 6_144), of: FloatType.self),
            graph.variable(
              Tensor<FloatType>(
                from: Krea2RotaryPositionEmbedding(
                  textLength: textLength, gridHeight: 32, gridWidth: 32, of: FloatType.self))),
          ]
          + (0..<7).map { _ in
            graph.variable(.CPU, .HWC(1, 1, 6_144), of: FloatType.self)
          }
        precondition(cArr[1].shape[1] == textLength + imageLength)
      case .kandinsky21, .v1, .v2, .seedvr2_3b, .seedvr2_7b:
        fatalError()
      }
      let inputs: [DynamicGraph.Tensor<FloatType>] = [xTensor] + (tEmb.map { [$0] } ?? []) + cArr
      unet.compile(inputs: inputs)
      func reverseMapping(original: ModelWeightMapping) -> [String: [String]] {
        var reversed: [String: [(Int, String)]] = [:]
        for (key, values) in original {
          for value in values {
            reversed[value, default: []].append((values.index, key))
          }
        }
        return reversed.mapValues {
          // If they have the same index, we need to override the map, to avoid accidentally create MoE.
          func unique(_ values: [(Int, String)]) -> [(Int, String)] {
            guard values.count > 1 else { return values }
            var seen: Set<Int> = []
            return values.filter { seen.insert($0.0).inserted }
          }
          return unique($0).sorted(by: { $0.0 < $1.0 }).map(\.1)
        }
      }
      let reverseUNetMapping: [String: [String]]
      // Otherwise we use our fixed dictionary.
      if version != .sdxlBase && version != .sdxlRefiner {
        let mappingFixed = unetFixedMapper(.generativeModels)
        let mapping = unetMapper(.generativeModels)
        reverseUNetMapping = reverseMapping(original: mapping)
        UNetMappingFixed = [:]
        for (key, value) in mappingFixed {
          guard !key.hasPrefix("diffusion_model.") else {
            UNetMappingFixed[key] = value
            continue
          }
          UNetMappingFixed["diffusion_model.\(key)"] = value
        }
        UNetMapping = [:]
        for (key, value) in mapping {
          guard !key.hasPrefix("diffusion_model.") else {
            UNetMapping[key] = value
            continue
          }
          UNetMapping["diffusion_model.\(key)"] = value
        }
      } else {
        reverseUNetMapping = [:]
      }
      if format.contains(.diffusers) {
        let diffusersUNetMappingFixed = unetFixedMapper(.diffusers)
        let diffusersUNetMapping = unetMapper(.diffusers)
        if !format.contains(.generativeModels) {
          // Remove generativeModels ones.
          UNetMappingFixed = diffusersUNetMappingFixed
          UNetMapping = diffusersUNetMapping
        } else {
          UNetMappingFixed.merge(diffusersUNetMappingFixed) { v, _ in v }
          UNetMapping.merge(diffusersUNetMapping) { v, _ in v }
        }
      }
      return (UNetMappingFixed, UNetMapping, reverseUNetMapping, otherMappings)
    }
  }

  private static func isDiagonalUp(_ tensor: Tensor<FloatType>, weights: ModelWeightElement) -> Bool
  {
    // Check if this tensor is diagonal. If it is, marking it and prepare to slice the corresponding lora_down.weight.
    // First, check if the shape is right.
    guard weights.format == .O else { return false }
    let shape = tensor.shape
    var isDiagonal = false
    if shape.count == 2 && (shape[1] % weights.count) == 0 {
      let estimatedInputShape0 = shape[0] / weights.count
      let inputShape1 = shape[1] / weights.count
      isDiagonal = true
      for i in 0..<weights.count {
        let startOffset = weights.offsets?[i] ?? i * estimatedInputShape0
        let endOffset =
          i < weights.count - 1
          ? (weights.offsets?[i + 1] ?? (i + 1) * estimatedInputShape0) : shape[0]
        for j in 0..<weights.count {
          // Now, only diag should have any value.
          guard i != j else { continue }
          let offDiagTensor = tensor[
            startOffset..<endOffset,
            (j * inputShape1)..<((j + 1) * inputShape1)
          ].copied()
          isDiagonal = offDiagTensor.withUnsafeBytes {
            guard let ptr = $0.baseAddress else { return false }
            let val = ptr.assumingMemoryBound(to: UInt8.self)
            for k in 0..<$0.count {
              if val[k] != 0 {
                return false
              }
            }
            return true
          }
          if !isDiagonal {
            break
          }
        }
        if !isDiagonal {
          break
        }
      }
    }
    return isDiagonal
  }

  private static func isDiagonalDown(_ tensor: Tensor<FloatType>, weights: ModelWeightElement)
    -> Bool
  {
    // Check if this tensor is diagonal. If it is, marking it and prepare to slice the corresponding lora_down.weight.
    // First, check if the shape is right.
    guard weights.format == .I else { return false }
    let shape = tensor.shape
    var isDiagonal = false
    if shape.count == 2 && (shape[1] % weights.count) == 0 {
      let inputShape0 = shape[0] / weights.count
      let estimatedInputShape1 = shape[1] / weights.count
      isDiagonal = true
      for i in 0..<weights.count {
        for j in 0..<weights.count {
          // Now, only diag should have any value.
          guard i != j else { continue }
          let startOffset = weights.offsets?[j] ?? j * estimatedInputShape1
          let endOffset =
            j < weights.count - 1
            ? (weights.offsets?[j + 1] ?? (j + 1) * estimatedInputShape1) : shape[1]
          let offDiagTensor = tensor[
            (i * inputShape0)..<((i + 1) * inputShape0),
            startOffset..<endOffset
          ].copied()
          isDiagonal = offDiagTensor.withUnsafeBytes {
            guard let ptr = $0.baseAddress else { return false }
            let val = ptr.assumingMemoryBound(to: UInt8.self)
            for k in 0..<$0.count {
              if val[k] != 0 {
                return false
              }
            }
            return true
          }
          if !isDiagonal {
            break
          }
        }
        if !isDiagonal {
          break
        }
      }
    }
    return isDiagonal
  }

  static private func findKeysAndValues(_ dictionary: ModelWeightMapping, keys: [String]) -> (
    String, ModelWeightElement
  )? {
    for key in keys {
      if let value = dictionary[key] {
        return (key, value)
      }
      let parts = key.components(separatedBy: ".")
      if parts.first == "model" {
        if parts.count > 3 && parts[1] == "diffusion_model" {
          let strippedKey = parts[2..<parts.count].joined(separator: ".")
          if let value = dictionary[strippedKey] {
            return (strippedKey, value)
          }
        }
        if parts.count > 2 {
          let strippedKey = parts[1..<parts.count].joined(separator: ".")
          if let value = dictionary[strippedKey] {
            return (strippedKey, value)
          }
        }
      } else if parts.first == "diffusion_model", parts.count > 2 {
        let strippedKey = parts[1..<parts.count].joined(separator: ".")
        if let value = dictionary[strippedKey] {
          return (strippedKey, value)
        }
      }
    }
    return nil
  }

  public static func `import`(
    downloadedFile: String, name: String, filename: String, scaleFactor: Double,
    forceVersion: ModelVersion?, progress: (Float) -> Void
  ) throws -> (ModelVersion, Bool, Int, Bool) {
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
      let originalStateDict = rootObject["state_dict"] as? Interpreter.Dictionary ?? rootObject
      stateDict = [String: TensorDescriptor]()
      originalStateDict.forEach { key, value in
        guard let value = value as? TensorDescriptor else { return }
        stateDict[key] = value
      }
    } else {
      throw UnpickleError.dataNotFound
    }
    let keys = stateDict.keys
    func copyAlphaIfPresent(
      from oldKey: String, stripping suffix: String, alpha: String, to newKey: String
    ) {
      guard oldKey.hasSuffix(suffix) else { return }
      let alphaBaseKey = String(oldKey.dropLast(suffix.count))
      guard let alphaTensor = stateDict[alphaBaseKey + alpha] else { return }
      stateDict[newKey + ".alpha"] = alphaTensor
    }
    // Fix for one LoRA formulation (commonly found in LyCORIS)
    for key in keys {
      guard key.hasSuffix(".lora_A.weight") || key.hasSuffix(".lora_B.weight") else { continue }
      var components = key.components(separatedBy: ".")
      guard components.count >= 3 else { continue }
      if components[1] == "base_model" {
        components.remove(at: 1)
      }
      if components[0] == "base_model" {
        components.remove(at: 0)
      }
      let newKey = components[0..<(components.count - 2)].joined(separator: "_")
      let isUp = key.hasSuffix(".lora_B.weight")
      if isUp {
        copyAlphaIfPresent(from: key, stripping: ".lora_B.weight", alpha: ".alpha", to: newKey)
        stateDict[newKey + ".lora_up.weight"] = stateDict[key]
      } else {
        stateDict[newKey + ".lora_down.weight"] = stateDict[key]
      }
    }
    // Fix the LoRA formulation for diffusers.
    for key in keys {
      guard key.hasSuffix(".lora.up.weight") || key.hasSuffix(".lora.down.weight") else { continue }
      var components = key.components(separatedBy: ".")
      guard components.count >= 4 else { continue }
      if components[1] == "base_model" {
        components.remove(at: 1)
      }
      if components[0] == "base_model" {
        components.remove(at: 0)
      }
      let newKey = components[0..<(components.count - 3)].joined(separator: "_")
      let isUp = key.hasSuffix(".lora.up.weight")
      if isUp {
        copyAlphaIfPresent(from: key, stripping: ".lora.up.weight", alpha: ".alpha", to: newKey)
        stateDict[newKey + ".lora_up.weight"] = stateDict[key]
      } else {
        stateDict[newKey + ".lora_down.weight"] = stateDict[key]
      }
    }
    // Fix the LoRA formulation from OneTrainer.
    for key in keys {
      guard key.hasSuffix(".lora_up.weight") || key.hasSuffix(".lora_down.weight")
      else { continue }
      var components = key.components(separatedBy: ".")
      guard components.count >= 4 else { continue }
      if components[0] == "transformer" {
        components.remove(at: 0)
      }
      let newKey = components[0..<(components.count - 2)].joined(separator: "_")
      let isUp = key.hasSuffix(".lora_up.weight")
      if isUp {
        copyAlphaIfPresent(from: key, stripping: ".lora_up.weight", alpha: ".alpha", to: newKey)
        stateDict[newKey + ".lora_up.weight"] = stateDict[key]
      } else {
        stateDict[newKey + ".lora_down.weight"] = stateDict[key]
      }
    }
    // Fix for another LoRA formulation (commonly found in SD-Forge, particularly, layerdiffuse).
    for key in keys {
      guard key.hasSuffix("::lora::0") || key.hasSuffix("::lora::1") else { continue }
      let components = key.components(separatedBy: ".")
      guard components.count > 2 else { continue }
      let newKey = components[0..<(components.count - 1)].joined(separator: "_")
      let isUp = key.hasSuffix("::lora::0")
      if isUp {
        copyAlphaIfPresent(from: key, stripping: "::lora::0", alpha: "::alpha", to: newKey)
        stateDict[newKey + ".lora_up.weight"] = stateDict[key]
      } else {
        stateDict[newKey + ".lora_down.weight"] = stateDict[key]
      }
    }
    // Fix for another LoRA formulation (Foocus, SD-Forge, diff based).
    for key in keys {
      guard key.hasSuffix("::diff::0") else { continue }
      let components = key.components(separatedBy: ".")
      guard components.count > 2 else { continue }
      let newKey =
        components[0..<(components.count - 1)].joined(separator: "_") + "."
        + components[components.count - 1]
      stateDict[newKey.dropLast(9) + ".diff"] = stateDict[key]
    }
    // Fix for one more LoRA formulation (found in HiDream E1 LoRA.)
    for key in keys {
      guard key.hasSuffix(".lora_A.default.weight") || key.hasSuffix(".lora_B.default.weight")
      else { continue }
      var components = key.components(separatedBy: ".")
      guard components.count >= 3 else { continue }
      if components[1] == "base_model" {
        components.remove(at: 1)
      }
      if components[0] == "base_model" {
        components.remove(at: 0)
      }
      let newKey = components[0..<(components.count - 3)].joined(separator: "_")
      let isUp = key.hasSuffix(".lora_B.default.weight")
      if isUp {
        copyAlphaIfPresent(
          from: key, stripping: ".lora_B.default.weight", alpha: ".alpha", to: newKey)
        stateDict[newKey + ".lora_up.weight"] = stateDict[key]
      } else {
        stateDict[newKey + ".lora_down.weight"] = stateDict[key]
      }
    }
    // Fix LoRA formulation for Z Image Base Fun Distill.
    for key in keys {
      guard key.hasPrefix("lora_unet__") else { continue }
      var newKey = key
      newKey.removeFirst(11)
      stateDict[newKey] = stateDict[key]
    }
    let modelVersion: ModelVersion = try {
      let isSD3Large = stateDict.keys.contains {
        $0.contains("joint_blocks_37_attn_")
          || $0.contains("transformer_blocks_37_attn_")
      }
      let isSD3Medium = stateDict.keys.contains {
        $0.contains("joint_blocks_23_context_block_")
          || $0.contains("transformer_blocks_22_ff_context_")
      }
      let isPixArtSigmaXL = stateDict.keys.contains {
        $0.contains("blocks_27_cross_attn_kv_") || $0.contains("transformer_blocks_27_attn2_to_")
      }
      let isFlux1 = stateDict.keys.contains {
        $0.contains("double_blocks.18.img_attn.qkv.")
          || $0.contains("double_blocks_18_img_attn_qkv.")
          || $0.contains("single_transformer_blocks_37_")
      }
      let isHunyuan = stateDict.keys.contains {
        $0.contains("double_blocks.19.img_attn_qkv.")
          || $0.contains("double_blocks_19_img_attn_qkv.")
          || $0.contains("double_blocks.19.img_attn.qkv.")
          || $0.contains("single_transformer_blocks_39_")
      }
      let isWan21_14B = stateDict.keys.contains {
        $0.contains("blocks.39.cross_attn.v.") || $0.contains("blocks_39_cross_attn_v.")
      }
      let isWan21_1_3B = stateDict.keys.contains {
        $0.contains("blocks.29.cross_attn.v.") || $0.contains("blocks_29_cross_attn_v.")
      }
      let isWan22_5B = stateDict.keys.contains {
        $0.contains("blocks.29.cross_attn.v.") || $0.contains("blocks_29_cross_attn_v.")
      }
      let isHiDream = stateDict.keys.contains {
        $0.contains("double_stream_blocks.15.block.ff_i.shared_experts.w1.")
          || $0.contains("single_stream_blocks.31.block.ff_i.shared_experts.w1.")
      }
      let isQwenImage = stateDict.keys.contains {
        $0.contains("transformer_blocks_37_attn_") || $0.contains("transformer_blocks.59.attn.")
      }
      let isZImage = stateDict.keys.contains {
        $0.contains("layers_29_feed_forward_w3") || $0.contains("layers.29.feed_forward.w3.")
      }
      let isFlux2 = stateDict.keys.contains {
        $0.contains("single_blocks.39.linear1.") || $0.contains("single_blocks_39_linear1")
          || $0.contains("single_transformer_blocks.39.attn.to_qkv_mlp_proj.")
          || $0.contains("single_transformer_blocks_39_attn_to_qkv_mlp_proj")
      }
      let isErnieImage = stateDict.keys.contains {
        $0.contains("layers.35.self_attention.to_q.")
          || $0.contains("layers_35_self_attention_to_q")
          || $0.contains("layers.35.mlp.linear_fc2.")
          || $0.contains("layers_35_mlp_linear_fc2")
      }
      let isCosmosAnima = stateDict.keys.contains {
        $0.contains("patch_embed.proj.") || $0.contains("patch_embed_proj")
          || $0.contains("blocks.5.self_attn.q_proj.")
          || $0.contains("blocks_5_self_attn_q_proj")
      }
      let isSDOrSDXL = stateDict.keys.contains {
        $0.hasSuffix(
          "down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight")
          || $0.hasSuffix(
            "up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight")
          || $0.hasSuffix("input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight")
          || $0.hasSuffix(
            "down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.hada_w1_b")
          || $0.hasSuffix("input_blocks_4_1_transformer_blocks_0_attn2_to_k.hada_w1_b")
        // For models we can only infer from CLIP, we let user pick which model that should be.
        // || $0.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
        // || $0.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b")
      }
      func forceVersionOr(_ version: ModelVersion) -> ModelVersion {
        if let forceVersion = forceVersion {
          return forceVersion
        }
        return version
      }
      let isMiniMaxH3 = stateDict.contains { key, descriptor in
        descriptor.shape.last == 5_376 && key.hasSuffix(".lora_down.weight")
          && (key.contains("token_refiner_refiner_blocks_")
            || key.contains("token_refiner_blocks_")
            || key.contains("transformer_blocks_49_attn_")
            || key.contains("blocks_49_attn_qkv_proj"))
      }
      if isMiniMaxH3 {
        return forceVersionOr(.minimaxH3)
      }
      let isQwenImage2_1 = stateDict.contains { key, descriptor in
        descriptor.shape.last == 4096 && key.hasSuffix(".lora_down.weight")
          && (key.contains("transformer_blocks_31_attn_to_")
            || key.contains("transformer_blocks_31_img_mlp_"))
      }
      if isQwenImage2_1 {
        return forceVersionOr(.qwenImage2_1)
      }
      if isErnieImage {
        return forceVersionOr(.ernieImage)
      }
      if isCosmosAnima {
        return forceVersionOr(.cosmos2_5_2b)
      }
      // Only confident about these if it is SD or SDXL. In any other cases, prefer the force version.
      switch (
        isSDOrSDXL, isSD3Medium, isSD3Large, isPixArtSigmaXL, isFlux1, isHunyuan, isWan21_1_3B,
        isWan22_5B, isWan21_14B, isHiDream, isQwenImage, isZImage, isFlux2
      ) {
      case (
        true, false, false, false, false, false, false, false, false, false, false, false, false
      ):
        if let tokey = stateDict.first(where: {
          $0.key.hasSuffix(
            "down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight")
            || $0.key.hasSuffix(
              "up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight")
            || $0.key.hasSuffix("input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight")
            || $0.key.hasSuffix(
              "down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.hada_w1_b")
            || $0.key.hasSuffix("input_blocks_4_1_transformer_blocks_0_attn2_to_k.hada_w1_b")
        })?.value
          ?? stateDict.first(where: {
            $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
              || $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b")
          })?.value
        {
          switch tokey.shape.last {
          case 2048:
            if !stateDict.keys.contains(where: {
              $0.contains("mid_block_attentions_0_transformer_blocks_")
                || $0.contains("middle_block_1_transformer_blocks_")
            }) {
              return forceVersionOr(.ssd1b)
            } else {
              return .sdxlBase
            }
          case 1280:
            return .sdxlRefiner
          case 1024:
            return .v2
          case 768:
            // Check if it has lora_te2, if it does, this might be text-encoder only SDXL Base.
            if stateDict.contains(where: { $0.key.hasPrefix("lora_te2_") }) {
              return .sdxlBase
            } else {
              return .v1
            }
          default:
            if let forceVersion = forceVersion {
              return forceVersion
            }
            throw Error.modelVersionFailed
          }
        } else {
          if let forceVersion = forceVersion {
            return forceVersion
          }
          throw Error.modelVersionFailed
        }
      case (
        false, true, false, false, false, false, false, false, false, false, false, false, false
      ):
        return forceVersionOr(.sd3)
      case (
        false, false, true, false, false, false, false, false, false, false, false, false, false
      ):
        return forceVersionOr(.sd3Large)
      case (
        false, false, false, true, false, false, false, false, false, false, false, false, false
      ):
        return forceVersionOr(.pixart)
      case (
        false, false, false, false, true, false, false, false, false, false, false, false, false
      ):
        return forceVersionOr(.flux1)
      case (
        false, false, false, false, false, true, false, false, false, false, false, false, false
      ):
        return forceVersionOr(.hunyuanVideo)
      case (
        false, false, false, false, false, false, true, false, false, false, false, false, false
      ):
        return forceVersionOr(.wan21_1_3b)
      case (
        false, false, false, false, false, false, false, true, false, false, false, false, false
      ):
        return forceVersionOr(.wan22_5b)
      case (
        false, false, false, false, false, false, false, false, true, false, false, false, false
      ),
        (false, false, false, false, false, false, true, true, true, false, false, false, false):
        return forceVersionOr(.wan21_14b)
      case (
        false, false, false, false, false, false, false, false, false, true, false, false, false
      ):
        return forceVersionOr(.hiDreamI1)
      case (
        false, false, false, false, false, false, false, false, false, false, true, false, false
      ):
        return forceVersionOr(.qwenImage)
      case (
        false, false, false, false, false, false, false, false, false, false, false, true, false
      ):
        return forceVersionOr(.zImage)
      case (
        false, false, false, false, false, false, false, false, false, false, false, false, true
      ):
        return forceVersionOr(.flux2)
      default:
        if let forceVersion = forceVersion {
          return forceVersion
        }
        throw Error.modelVersionFailed
      }
    }()
    let graph = DynamicGraph()
    var textModelMapping1: ModelWeightMapping
    var textModelMapping2: ModelWeightMapping
    switch modelVersion {
    case .longcatVideoAvatar1_5:
      fatalError()
    case .v1:
      textModelMapping1 = StableDiffusionMapping.CLIPTextModel
      textModelMapping2 = [:]
    case .v2:
      textModelMapping1 = StableDiffusionMapping.OpenCLIPTextModel
      textModelMapping2 = [:]
    case .sdxlBase, .ssd1b:
      textModelMapping1 = StableDiffusionMapping.CLIPTextModel
      textModelMapping2 = StableDiffusionMapping.OpenCLIPTextModelG
      textModelMapping2.merge(StableDiffusionMapping.OpenCLIPTextModelGTransformers) { v, _ in v }
    case .sdxlRefiner:
      textModelMapping1 = StableDiffusionMapping.OpenCLIPTextModelG
      textModelMapping1.merge(StableDiffusionMapping.OpenCLIPTextModelGTransformers) { v, _ in v }
      textModelMapping2 = [:]
    case .sd3, .sd3Large:
      // We don't support LoRA on text encoder for SD3 yet.
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .pixart:
      textModelMapping1 = graph.withNoGrad {
        let (t5Mapper, t5) = T5ForConditionalGeneration(
          b: 1, t: 2, attentionMask: false, of: FloatType.self)
        let relativePositionBuckets = relativePositionBuckets(
          sequenceLength: 2, numBuckets: 32, maxDistance: 128
        ).toGPU(0)
        let tokens = Tensor<Int32>([0, 0], .GPU(0), .C(2))
        t5.compile(inputs: graph.variable(tokens), graph.variable(relativePositionBuckets))
        return t5Mapper(.generativeModels)
      }
      textModelMapping2 = [:]
    case .flux1:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .hunyuanVideo:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .hiDreamI1, .hiDreamO1, .qwenImage2_1:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .qwenImage:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .cosmos2_5_2b:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .zImage:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .ernieImage, .flux2, .flux2_9b, .flux2_4b:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2, .minimaxH3:
      textModelMapping1 = [:]
      textModelMapping2 = [:]
    case .auraflow:
      fatalError()
    case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB:
      fatalError()
    }
    var swapTE2 = false
    if modelVersion == .sdxlBase {
      // Inspect further to see if it has both te1 / te2 model, and whether one of them is CLIP or OpenCLIPTextModelG.
      // If both te2 and te1 exists, check if te2 is 2048, if not, then check if t1 is. If it is,
      // we need to swap.
      // Otherwise, if only te2 exists, check if it is 768, if it is, it is CLIP, we need to swap.
      // If te2 doesn't exists, then we need to decide whether te1 / te is 2048, if it is, we need
      // to swap too.
      let te1tokeys = stateDict.first(where: {
        ($0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
          || $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b"))
          && $0.key.contains("_te1_")
      })?.value
      let te2tokeys = stateDict.first(where: {
        ($0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
          || $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b"))
          && $0.key.contains("_te2_")
      })?.value
      let tetokeys = stateDict.first(where: {
        ($0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.lora_down.weight")
          || $0.key.hasSuffix("encoder_layers_0_self_attn_k_proj.hada_w1_b"))
          && $0.key.contains("_te_")
      })?.value
      if let te1tokeys = te1tokeys, let te2tokeys = te2tokeys {
        if te2tokeys.shape.last == 1280 {
          // Everything matches.
        } else if te1tokeys.shape.last == 1280 {
          // We need to swap te1 / te2.
          swapTE2 = true
        }
      } else if let te2tokeys = te2tokeys {
        if te2tokeys.shape.last == 768 {
          swapTE2 = true
        }
      } else if let te1tokeys = te1tokeys, te1tokeys.shape.last == 1280 {
        // In this case, we treat the te1 as OpenCLIP-G
        swapTE2 = true
      } else if let tetokeys = tetokeys, tetokeys.shape.last == 1280 {
        // In this case, we treat the te as OpenCLIP-G
        swapTE2 = true
      }
    }
    let textModelMapping1Keys = textModelMapping1.keys
    for key in textModelMapping1Keys {
      let value = textModelMapping1[key]
      let parts = key.components(separatedBy: ".")
      for i in 0...2 {
        guard parts.count - 1 > i else { break }
        textModelMapping1[
          parts[i..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
          value
      }
    }
    let textModelMapping2Keys = textModelMapping2.keys
    for key in textModelMapping2Keys {
      let value = textModelMapping2[key]
      let parts = key.components(separatedBy: ".")
      for i in 0...2 {
        guard parts.count - 1 > i else { break }
        textModelMapping2[
          parts[i..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
          value
      }
    }
    let qkNorm = keys.contains {
      ($0.contains(".ln_k.") || $0.contains(".ln_q.") || $0.contains(".norm_k.")
        || $0.contains(".norm_q."))
        || ($0.contains("_ln_k_") || $0.contains("_ln_q_") || $0.contains("_norm_k_")
          || $0.contains("_norm_q_"))
    }
    let dualAttentionLayers = (0..<38).filter { i in
      if let key = keys.first(where: {
        ($0.contains("_\(i)_x_block_adaLN_modulation_1")
          || $0.contains(".\(i).x_block.adaLN_modulation_1")
          || $0.contains("_\(i)_norm1_linear") || $0.contains(".\(i).norm1.linear"))
          && $0.contains("lora_up")
      }),
        let value = stateDict[key]
      {
        return value.shape[0] == 1536 * 9
      }
      return keys.contains {
        ($0.contains(".\(i).x_block.attn2.") || $0.contains("_blocks.\(i).attn2."))
          || ($0.contains("_\(i)_x_block_attn2_") || $0.contains("_blocks_\(i)_attn2_"))
      }
    }
    // Prepare UNet mapping.
    var (UNetMappingFixed, UNetMapping, reverseUNetMapping, otherMappings) = modelWeightsMapping(
      by: modelVersion, qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      format: [.diffusers, .generativeModels])
    let UNetMappingKeys = UNetMapping.keys
    for key in UNetMappingKeys {
      let value = UNetMapping[key]
      let parts = key.components(separatedBy: ".")
      if parts[0] == "model" {
        if parts.count > 3 {
          UNetMapping[
            parts[2..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
        if parts.count > 2 {
          UNetMapping[
            parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      } else if parts[0] == "diffusion_model" {
        if parts.count > 2 {
          UNetMapping[
            parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      } else {
        if parts.count > 1 {
          UNetMapping[
            parts[0..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      }
    }
    let UNetMappingFixedKeys = UNetMappingFixed.keys
    for key in UNetMappingFixedKeys {
      let value = UNetMappingFixed[key]
      let parts = key.components(separatedBy: ".")
      if parts[0] == "model" {
        if parts.count > 3 {
          UNetMappingFixed[
            parts[2..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
        if parts.count > 2 {
          UNetMappingFixed[
            parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      } else if parts[0] == "diffusion_model" {
        if parts.count > 2 {
          UNetMappingFixed[
            parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      } else {
        if parts.count > 1 {
          UNetMappingFixed[
            parts[0..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
            value
        }
      }
    }
    let reverseUNetMappingKeys = reverseUNetMapping.keys
    for key in reverseUNetMappingKeys {
      guard let values = reverseUNetMapping[key] else { continue }
      reverseUNetMapping[key] = values.map {
        let parts = $0.components(separatedBy: ".")
        if parts[0] == "model" {
          if parts.count > 3 {
            return parts[2..<(parts.count - 1)].joined(separator: "_") + "."
              + parts[parts.count - 1]
          }
          if parts.count > 2 {
            return parts[1..<(parts.count - 1)].joined(separator: "_") + "."
              + parts[parts.count - 1]
          }
        } else if parts[0] == "diffusion_model" {
          if parts.count > 2 {
            return parts[1..<(parts.count - 1)].joined(separator: "_") + "."
              + parts[parts.count - 1]
          }
        } else {
          if parts.count > 1 {
            return parts[0..<(parts.count - 1)].joined(separator: "_") + "."
              + parts[parts.count - 1]
          }
        }
        return $0
      }
    }
    otherMappings = otherMappings.map {
      var mapping = $0.1
      let mappingKeys = mapping.keys
      for key in mappingKeys {
        let value = mapping[key]
        let parts = key.components(separatedBy: ".")
        if parts[0] == "model" {
          if parts.count > 3 {
            mapping[
              parts[2..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
              value
          }
          if parts.count > 2 {
            mapping[
              parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
              value
          }
        } else if parts[0] == "diffusion_model" {
          if parts.count > 2 {
            mapping[
              parts[1..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
              value
          }
        } else {
          if parts.count > 1 {
            mapping[
              parts[0..<(parts.count - 1)].joined(separator: "_") + "." + parts[parts.count - 1]] =
              value
          }
        }
      }
      return ($0.0, mapping)
    }
    var didImportTIEmbedding = false
    var textEmbeddingLength = 0
    var isLoHa = false
    try graph.openStore(LoRAZoo.filePathForModelDownloaded(filename)) { store in
      store.removeAll()
      switch modelVersion {
      case .longcatVideoAvatar1_5:
        fatalError()
      case .minimaxH3:
        break
      case .v1, .v2, .kandinsky21, .svdI2v:
        if let tensorDesc = stateDict["emb_params"] {
          try archive.with(tensorDesc) {
            let tensor = Tensor<FloatType>(from: $0)
            textEmbeddingLength = tensorDesc.shape.count > 1 ? tensorDesc.shape[0] : 1
            store.write("string_to_param", tensor: tensor)
            didImportTIEmbedding = true
          }
        }
      case .sd3, .sd3Large:
        if let tensorDescClipG = stateDict["clip_g"] {
          try archive.with(tensorDescClipG) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_clip_g", tensor: tensor)
            textEmbeddingLength = tensorDescClipG.shape.count > 1 ? tensorDescClipG.shape[0] : 1
            didImportTIEmbedding = true
          }
        }
        if let tensorDescClipL = stateDict["clip_l"] {
          try archive.with(tensorDescClipL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_clip_l", tensor: tensor)
            let textEmbeddingLengthClipL =
              tensorDescClipL.shape.count > 1 ? tensorDescClipL.shape[0] : 1
            guard textEmbeddingLengthClipL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
        if let tensorDescT5XXL = stateDict["t5_xxl"] {
          try archive.with(tensorDescT5XXL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_t5_xxl", tensor: tensor)
            let textEmbeddingLengthT5XXL =
              tensorDescT5XXL.shape.count > 1 ? tensorDescT5XXL.shape[0] : 1
            guard textEmbeddingLengthT5XXL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .pixart:
        if let tensorDescT5XXL = stateDict["t5_xxl"] {
          try archive.with(tensorDescT5XXL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_t5_xxl", tensor: tensor)
            let textEmbeddingLengthT5XXL =
              tensorDescT5XXL.shape.count > 1 ? tensorDescT5XXL.shape[0] : 1
            guard textEmbeddingLengthT5XXL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .flux1:
        if let tensorDescT5XXL = stateDict["t5_xxl"] {
          try archive.with(tensorDescT5XXL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_t5_xxl", tensor: tensor)
            let textEmbeddingLengthT5XXL =
              tensorDescT5XXL.shape.count > 1 ? tensorDescT5XXL.shape[0] : 1
            guard textEmbeddingLengthT5XXL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .hunyuanVideo:
        if let tensorDescLlama = stateDict["llama"] {
          try archive.with(tensorDescLlama) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_llama", tensor: tensor)
            let textEmbeddingLengthLlama =
              tensorDescLlama.shape.count > 1 ? tensorDescLlama.shape[0] : 1
            guard textEmbeddingLengthLlama == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .wan21_1_3b, .wan21_14b, .wan22_5b:
        if let tensorDescUMT5XXL = stateDict["umt5_xxl"] {
          try archive.with(tensorDescUMT5XXL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_umt5_xxl", tensor: tensor)
            let textEmbeddingLengthUMT5XXL =
              tensorDescUMT5XXL.shape.count > 1 ? tensorDescUMT5XXL.shape[0] : 1
            guard textEmbeddingLengthUMT5XXL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .hiDreamI1:
        if let tensorDescLlama = stateDict["llama"] {
          try archive.with(tensorDescLlama) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_llama", tensor: tensor)
            let textEmbeddingLengthLlama =
              tensorDescLlama.shape.count > 1 ? tensorDescLlama.shape[0] : 1
            guard textEmbeddingLengthLlama == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .hiDreamO1, .qwenImage2_1:
        break
      case .qwenImage:
        if let tensorDescQwen25 = stateDict["qwen25"] {
          try archive.with(tensorDescQwen25) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_qwen25", tensor: tensor)
            let textEmbeddingLengthQwen25 =
              tensorDescQwen25.shape.count > 1 ? tensorDescQwen25.shape[0] : 1
            guard textEmbeddingLengthQwen25 == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .zImage, .cosmos2_5_2b:
        if let tensorDescQwen3 = stateDict["qwen3"] {
          try archive.with(tensorDescQwen3) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_qwen3", tensor: tensor)
            let textEmbeddingLengthQwen3 =
              tensorDescQwen3.shape.count > 1 ? tensorDescQwen3.shape[0] : 1
            guard textEmbeddingLengthQwen3 == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .ernieImage, .flux2, .flux2_9b, .flux2_4b:
        if let tensorDescMistral3 = stateDict["mistral3"] {
          try archive.with(tensorDescMistral3) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_mistral3", tensor: tensor)
            let textEmbeddingLengthMistral3 =
              tensorDescMistral3.shape.count > 1 ? tensorDescMistral3.shape[0] : 1
            guard textEmbeddingLengthMistral3 == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
        if let tensorDescGemma3 = stateDict["gemma3"] {
          try archive.with(tensorDescGemma3) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_gemma3", tensor: tensor)
            let textEmbeddingLengthGemma3 =
              tensorDescGemma3.shape.count > 1 ? tensorDescGemma3.shape[0] : 1
            guard textEmbeddingLengthGemma3 == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      case .auraflow:
        fatalError()
      case .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC, .wurstchenStageB:
        if let tensorDescClipG = stateDict["clip_g"] {
          try archive.with(tensorDescClipG) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_clip_g", tensor: tensor)
            textEmbeddingLength = tensorDescClipG.shape.count > 1 ? tensorDescClipG.shape[0] : 1
            didImportTIEmbedding = true
          }
        }
        if let tensorDescClipL = stateDict["clip_l"] {
          try archive.with(tensorDescClipL) {
            let tensor = Tensor<FloatType>(from: $0)
            store.write("string_to_param_clip_l", tensor: tensor)
            let textEmbeddingLengthClipL =
              tensorDescClipL.shape.count > 1 ? tensorDescClipL.shape[0] : 1
            guard textEmbeddingLengthClipL == textEmbeddingLength else {
              textEmbeddingLength = 0
              return
            }
            didImportTIEmbedding = true
          }
        }
      }
      let modelPrefix: String
      let modelPrefixFixed: String
      switch modelVersion {
      case .longcatVideoAvatar1_5:
        fatalError()
      case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v:
        modelPrefix = "unet"
        modelPrefixFixed = "unet_fixed"
      case .wurstchenStageC, .wurstchenStageB:
        modelPrefix = "stage_c"
        modelPrefixFixed = "stage_c_fixed"
      case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
        .hiDreamI1, .hiDreamO1, .qwenImage2_1, .qwenImage, .cosmos2_5_2b, .wan22_5b, .zImage,
        .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b,
        .ideogram4, .krea2, .minimaxH3:
        modelPrefix = "dit"
        modelPrefixFixed = "dit"
      }
      try store.withTransaction {
        let total = stateDict.count
        var diagonalUpMatrixKeys = Set<String>()
        func deriveUpKeys(_ parts: [String]) -> [String] {
          return (0...2).compactMap {
            let newParts = String(parts[$0..<parts.count].joined(separator: "_")).components(
              separatedBy: ".")  // Remove the first two.
            guard newParts.count > 1 else { return nil }
            return newParts[0..<(newParts.count > 2 ? newParts.count - 2 : newParts.count - 1)]
              .joined(
                separator: ".") + ".weight"
          }
        }
        func mappingUpKeys(original: [String], weightsMapping: ModelWeightMapping) -> [String:
          String]
        {
          var mapping = [String: String]()
          for key in original.sorted(by: { $0 < $1 }) {
            let parts = key.components(separatedBy: "_")
            guard parts.count > 2 else { continue }
            let newKeys = deriveUpKeys(parts)
            if let (newKey, _) = Self.findKeysAndValues(weightsMapping, keys: newKeys) {
              if key.hasSuffix("up.weight") {
                mapping[newKey] = key
              }
            }
          }
          return mapping
        }
        let UNetUpKeysMapping = mappingUpKeys(
          original: Array(stateDict.keys), weightsMapping: UNetMapping)
        var diagonalDownsForRedefine = [(String, String, ModelWeightElement)]()
        var consumed = Set<String>()
        for (i, (key, descriptor)) in stateDict.sorted(by: { $0.key < $1.key }).enumerated() {
          let parts = key.components(separatedBy: "_")
          guard parts.count > 2 else { continue }
          let te2 = parts[1] == "te2"
          // Try to remove the prefixes.
          let newKeys = deriveUpKeys(parts)
          if let (newKey, unetParams) = Self.findKeysAndValues(UNetMapping, keys: newKeys) {
            if key.hasSuffix("up.weight"), !consumed.contains(newKey) {
              let values =
                unetParams.count == 1 ? (reverseUNetMapping[unetParams[0]] ?? [newKey]) : [newKey]
              consumed.formUnion(values)
              var tensorDescriptors = values.compactMap {
                UNetUpKeysMapping[$0].flatMap { stateDict[$0] }
              }
              if tensorDescriptors.isEmpty {  // Fallback.
                tensorDescriptors = [descriptor]
              }
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -14))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              try archive.with(tensorDescriptors) { tensors in
                guard !tensors.isEmpty else { return }
                var tensor: Tensor<FloatType>
                let loraDim = Float(tensors[0].shape[1])
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
                let isDiagonalUp =
                  unetParams.count > 1 ? Self.isDiagonalUp(tensor, weights: unetParams) : false
                if let scalar = scalar, abs(scalar - loraDim) > 1e-5 {
                  tensor = Tensor<FloatType>(
                    from: (Float(Double(scalar / loraDim) * scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                if isDiagonalUp {
                  diagonalUpMatrixKeys.insert(newKey)
                }
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .O, isDiagonalUp: isDiagonalUp,
                  isDiagonalDown: false
                ) {
                  "__\(modelPrefix)__[\($0)]__up__"
                }
              }
            } else if key.hasSuffix("hada_w1_a") || key.hasSuffix("hada_w2_a") {
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -9))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              let wSuffix = key.hasSuffix("hada_w1_a") ? "w1_a" : "w2_a"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let loraDim = Float(tensor.shape[1])
                if let scalar = scalar {
                  tensor = Tensor<FloatType>(
                    from: (Float(
                      Double(scalar / loraDim).squareRoot()
                        * scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  "__\(modelPrefix)__[\($0)]__\(wSuffix)__"
                }
              }
              isLoHa = true
            }
          } else if let (newKey, unetParams) = Self.findKeysAndValues(
            UNetMappingFixed, keys: newKeys)
          {
            if key.hasSuffix("up.weight") {
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -14))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let loraDim = Float(tensor.shape[1])
                let isDiagonalUp =
                  unetParams.count > 1 ? Self.isDiagonalUp(tensor, weights: unetParams) : false
                if let scalar = scalar, abs(scalar - loraDim) > 1e-5 {
                  tensor = Tensor<FloatType>(
                    from: (Float(Double(scalar / loraDim) * scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                if isDiagonalUp {
                  diagonalUpMatrixKeys.insert(newKey)
                }
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .O, isDiagonalUp: isDiagonalUp,
                  isDiagonalDown: false
                ) {
                  "__\(modelPrefixFixed)__[\($0)]__up__"
                }
              }
            } else if key.hasSuffix("hada_w1_a") || key.hasSuffix("hada_w2_a") {
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -9))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              let wSuffix = key.hasSuffix("hada_w1_a") ? "w1_a" : "w2_a"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let loraDim = Float(tensor.shape[1])
                if let scalar = scalar {
                  tensor = Tensor<FloatType>(
                    from: (Float(
                      Double(scalar / loraDim).squareRoot()
                        * scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  "__\(modelPrefixFixed)__[\($0)]__\(wSuffix)__"
                }
              }
              isLoHa = true
            }
          } else if let (model, mapping) =
            (otherMappings.first {
              Self.findKeysAndValues($0.1, keys: newKeys) != nil
            }), let (newKey, unetParams) = Self.findKeysAndValues(mapping, keys: newKeys)
          {
            if key.hasSuffix("up.weight") {
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -14))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let loraDim = Float(tensor.shape[1])
                let isDiagonalUp =
                  unetParams.count > 1 ? Self.isDiagonalUp(tensor, weights: unetParams) : false
                if let scalar = scalar, abs(scalar - loraDim) > 1e-5 {
                  tensor = Tensor<FloatType>(
                    from: (Float(Double(scalar / loraDim) * scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                if isDiagonalUp {
                  diagonalUpMatrixKeys.insert(newKey)
                }
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .O, isDiagonalUp: isDiagonalUp,
                  isDiagonalDown: false
                ) {
                  "__\(model)__[\($0)]__up__"
                }
              }
            } else if key.hasSuffix("hada_w1_a") || key.hasSuffix("hada_w2_a") {
              let scalar = try stateDict[
                String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -9))) + "alpha"
              ].map {
                return try archive.with($0) {
                  return Tensor<Float32>(from: $0)[0]
                }
              }
              let wSuffix = key.hasSuffix("hada_w1_a") ? "w1_a" : "w2_a"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let loraDim = Float(tensor.shape[1])
                if let scalar = scalar {
                  tensor = Tensor<FloatType>(
                    from: (Float(
                      Double(scalar / loraDim).squareRoot()
                        * scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                } else if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  "__\(model)__[\($0)]__\(wSuffix)__"
                }
              }
              isLoHa = true
            }
          } else {
            let textModelMapping: ModelWeightMapping
            if te2 != swapTE2 {
              textModelMapping = textModelMapping2
            } else {
              textModelMapping = textModelMapping1
            }
            if let (_, textParams) = Self.findKeysAndValues(textModelMapping, keys: newKeys),
              let name = textParams.first
            {
              if key.hasSuffix("up.weight") {
                let scalar = try stateDict[
                  String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -14))) + "alpha"
                ].map {
                  return try archive.with($0) {
                    return Tensor<Float32>(from: $0)[0]
                  }
                }
                try archive.with(descriptor) {
                  var tensor = Tensor<FloatType>(from: $0)
                  let loraDim = Float(tensor.shape[1])
                  if let scalar = scalar, abs(scalar - loraDim) > 1e-5 {
                    tensor = Tensor<FloatType>(
                      from: (Float(Double(scalar / loraDim) * scaleFactor.squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  } else if scaleFactor != 1 {
                    tensor = Tensor<FloatType>(
                      from: (Float(scaleFactor.squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  }
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]__up__", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]__up__", tensor: tensor)
                  }
                }
              } else if key.hasSuffix("hada_w1_a") || key.hasSuffix("hada_w2_a") {
                let scalar = try stateDict[
                  String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -9))) + "alpha"
                ].map {
                  return try archive.with($0) {
                    return Tensor<Float32>(from: $0)[0]
                  }
                }
                let wSuffix = key.hasSuffix("hada_w1_a") ? "w1_a" : "w2_a"
                try archive.with(descriptor) {
                  var tensor = Tensor<FloatType>(from: $0)
                  let loraDim = Float(tensor.shape[1])
                  if let scalar = scalar {
                    tensor = Tensor<FloatType>(
                      from: (Float(
                        Double(scalar / loraDim).squareRoot()
                          * scaleFactor.squareRoot().squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  } else if scaleFactor != 1 {
                    tensor = Tensor<FloatType>(
                      from: (Float(scaleFactor.squareRoot().squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  }
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]__\(wSuffix)__", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]__\(wSuffix)__", tensor: tensor)
                  }
                }
                isLoHa = true
              }
            }
          }
          progress(Float(i + 1) / Float(total * 2))
        }
        func deriveDownKeys(_ parts: [String]) -> [String] {
          return (0...2).flatMap { (index) -> [String] in
            let newParts = String(parts[index..<parts.count].joined(separator: "_")).components(
              separatedBy: ".")  // Remove the first two.
            guard newParts.count > 1 else { return [] }
            var newKey =
              newParts[0..<(newParts.count > 2 ? newParts.count - 2 : newParts.count - 1)].joined(
                separator: ".")
            if (newParts[newParts.count - 1] == "diff" || newParts[newParts.count - 1] == "diff_b")
              && newParts.count > 2
            {
              newKey = newKey + "." + newParts[newParts.count - 2]
              if newParts[newParts.count - 1] == "diff_b" {
                return [newKey, newKey + ".bias"]
              }
              return [newKey, newKey + ".weight"]
            } else {
              newKey = newKey + ".weight"
              return [newKey]
            }
          }
        }
        func mappingDownKeys(original: [String], weightsMapping: ModelWeightMapping) -> [String:
          String]
        {
          var mapping = [String: String]()
          for key in original.sorted(by: { $0 < $1 }) {
            let parts = key.components(separatedBy: "_")
            guard parts.count > 2 else { continue }
            let newKeys = deriveDownKeys(parts)
            if let (newKey, _) = Self.findKeysAndValues(weightsMapping, keys: newKeys) {
              if key.hasSuffix("down.weight") {
                mapping[newKey] = key
              }
            }
          }
          return mapping
        }
        let UNetDownKeysMapping = mappingDownKeys(
          original: Array(stateDict.keys), weightsMapping: UNetMapping)
        consumed.removeAll()
        for (i, (key, descriptor)) in stateDict.sorted(by: { $0.key < $1.key }).enumerated() {
          let parts = key.components(separatedBy: "_")
          guard parts.count > 2 else { continue }
          let te2 = parts[1] == "te2"
          // Try to remove the prefixes.
          let newKeys = deriveDownKeys(parts)
          if let (newKey, unetParams) = Self.findKeysAndValues(UNetMapping, keys: newKeys) {
            if key.hasSuffix("down.weight"), !consumed.contains(newKey) {
              let values =
                unetParams.count == 1 ? (reverseUNetMapping[unetParams[0]] ?? [newKey]) : [newKey]
              consumed.formUnion(values)
              var tensorDescriptors = values.compactMap {
                UNetDownKeysMapping[$0].flatMap { stateDict[$0] }
              }
              if tensorDescriptors.isEmpty {  // Fallback.
                tensorDescriptors = [descriptor]
              }
              try archive.with(tensorDescriptors) { tensors in
                guard !tensors.isEmpty else { return }
                var tensor: Tensor<FloatType>
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
                let isDiagonalDown = Self.isDiagonalDown(tensor, weights: unetParams)
                if isDiagonalDown {
                  diagonalDownsForRedefine.append(
                    (
                      String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -16))), modelPrefix,
                      unetParams
                    ))
                }
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .I,
                  isDiagonalUp: diagonalUpMatrixKeys.contains(newKey),
                  isDiagonalDown: isDiagonalDown
                ) {
                  return "__\(modelPrefix)__[\($0)]__down__"
                }
              }
            } else if key.hasSuffix("mid.weight") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                for name in unetParams {
                  store.write("__\(modelPrefix)__[\(name)]__mid__", tensor: tensor)
                }
              }
            } else if key.hasSuffix("hada_w1_b") || key.hasSuffix("hada_w2_b") {
              let wSuffix = key.hasSuffix("hada_w1_b") ? "w1_b" : "w2_b"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                for name in unetParams {
                  store.write("__\(modelPrefix)__[\(name)]__\(wSuffix)__", tensor: tensor)
                }
              }
              isLoHa = true
            } else if key.hasSuffix(".diff") || key.hasSuffix(".diff_b") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                /* One-off code to import 8-channel to 9-channel as if it is a inpainting LoRA.
                let shape = tensor.shape
                if shape[1] == 8 && shape[2] == 3 && shape[3] == 3 {
                  // Expand this tensor to 9 in the middle section.
                  var newTensor = Tensor<FloatType>(.CPU, .NCHW(shape[0], 9, 3, 3))
                  newTensor.withUnsafeMutableBytes {
                    guard let f16 = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
                    memset(f16, 0, 2 * shape[0] * 9 * 3 * 3)
                  }
                  newTensor[0..<shape[0], 0..<4, 0..<3, 0..<3] = tensor[0..<shape[0], 0..<4, 0..<3, 0..<3]
                  newTensor[0..<shape[0], 5..<9, 0..<3, 0..<3] = tensor[0..<shape[0], 4..<8, 0..<3, 0..<3]
                  tensor = newTensor
                }
                 */
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  return "__\(modelPrefix)__[\($0)]"
                }
              }
            }
          } else if let (newKey, unetParams) = Self.findKeysAndValues(
            UNetMappingFixed, keys: newKeys)
          {
            if key.hasSuffix("down.weight") {
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let isDiagonalDown = Self.isDiagonalDown(tensor, weights: unetParams)
                if isDiagonalDown {
                  diagonalDownsForRedefine.append(
                    (
                      String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -16))),
                      modelPrefixFixed, unetParams
                    ))
                }
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .I,
                  isDiagonalUp: diagonalUpMatrixKeys.contains(newKey),
                  isDiagonalDown: isDiagonalDown
                ) {
                  return "__\(modelPrefixFixed)__[\($0)]__down__"
                }
              }
            } else if key.hasSuffix("mid.weight") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                for name in unetParams {
                  store.write("__\(modelPrefixFixed)__[\(name)]__mid__", tensor: tensor)
                }
              }
            } else if key.hasSuffix("hada_w1_b") || key.hasSuffix("hada_w2_b") {
              let wSuffix = key.hasSuffix("hada_w1_b") ? "w1_b" : "w2_b"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                for name in unetParams {
                  store.write("__\(modelPrefixFixed)__[\(name)]__\(wSuffix)__", tensor: tensor)
                }
              }
              isLoHa = true
            } else if key.hasSuffix(".diff") || key.hasSuffix(".diff_b") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  return "__\(modelPrefixFixed)__[\($0)]"
                }
              }
            }
          } else if let (model, mapping) =
            (otherMappings.first {
              Self.findKeysAndValues($0.1, keys: newKeys) != nil
            }), let (newKey, unetParams) = Self.findKeysAndValues(mapping, keys: newKeys)
          {
            if key.hasSuffix("down.weight") {
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                let isDiagonalDown = Self.isDiagonalDown(tensor, weights: unetParams)
                if isDiagonalDown {
                  diagonalDownsForRedefine.append(
                    (
                      String(key.prefix(upTo: key.index(key.endIndex, offsetBy: -16))),
                      model, unetParams
                    ))
                }
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .I,
                  isDiagonalUp: diagonalUpMatrixKeys.contains(newKey),
                  isDiagonalDown: isDiagonalDown
                ) {
                  return "__\(model)__[\($0)]__down__"
                }
              }
            } else if key.hasSuffix("mid.weight") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                for name in unetParams {
                  store.write("__\(model)__[\(name)]__mid__", tensor: tensor)
                }
              }
            } else if key.hasSuffix("hada_w1_b") || key.hasSuffix("hada_w2_b") {
              let wSuffix = key.hasSuffix("hada_w1_b") ? "w1_b" : "w2_b"
              try archive.with(descriptor) {
                var tensor = Tensor<FloatType>(from: $0)
                if scaleFactor != 1 {
                  tensor = Tensor<FloatType>(
                    from: (Float(scaleFactor.squareRoot().squareRoot())
                      * graph.variable(Tensor<Float>(from: tensor)))
                      .rawValue)
                }
                for name in unetParams {
                  store.write("__\(model)__[\(name)]__\(wSuffix)__", tensor: tensor)
                }
              }
              isLoHa = true
            } else if key.hasSuffix(".diff") || key.hasSuffix(".diff_b") {
              try archive.with(descriptor) {
                let tensor = Tensor<FloatType>(from: $0)
                unetParams.write(
                  graph: graph,
                  to: store, tensor: tensor, format: .O, isDiagonalUp: false, isDiagonalDown: false
                ) {
                  return "__\(model)__[\($0)]"
                }
              }
            }
          } else {
            let textModelMapping: ModelWeightMapping
            if te2 != swapTE2 {
              textModelMapping = textModelMapping2
            } else {
              textModelMapping = textModelMapping1
            }
            if let (_, textParams) = Self.findKeysAndValues(textModelMapping, keys: newKeys),
              let name = textParams.first
            {
              if key.hasSuffix("down.weight") {
                try archive.with(descriptor) {
                  var tensor = Tensor<FloatType>(from: $0)
                  if scaleFactor != 1 {
                    tensor = Tensor<FloatType>(
                      from: (Float(scaleFactor.squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  }
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]__down__", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]__down__", tensor: tensor)
                  }
                }
              } else if key.hasSuffix("hada_w1_b") || key.hasSuffix("hada_w2_b") {
                let wSuffix = key.hasSuffix("hada_w1_b") ? "w1_b" : "w2_b"
                try archive.with(descriptor) {
                  var tensor = Tensor<FloatType>(from: $0)
                  if scaleFactor != 1 {
                    tensor = Tensor<FloatType>(
                      from: (Float(scaleFactor.squareRoot().squareRoot())
                        * graph.variable(Tensor<Float>(from: tensor)))
                        .rawValue)
                  }
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]__\(wSuffix)__", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]__\(wSuffix)__", tensor: tensor)
                  }
                }
                isLoHa = true
              } else if key.hasSuffix(".diff") || key.hasSuffix(".diff_b") {
                try archive.with(descriptor) {
                  let tensor = Tensor<FloatType>(from: $0)
                  if te2 != swapTE2 {
                    store.write("__te2__text_model__[\(name)]", tensor: tensor)
                  } else {
                    store.write("__text_model__[\(name)]", tensor: tensor)
                  }
                }
              }
            }
          }
          progress(Float(i + total + 1) / Float(total * 2))
        }
        for diagonalDownForRedefine in diagonalDownsForRedefine {
          guard let descriptor = stateDict[diagonalDownForRedefine.0 + "lora_up.weight"] else {
            continue
          }
          let scalar = try stateDict[
            diagonalDownForRedefine.0 + "alpha"
          ].map {
            return try archive.with($0) {
              return Tensor<Float32>(from: $0)[0]
            }
          }
          try archive.with(descriptor) {
            var tensor = Tensor<FloatType>(from: $0)
            let loraDim = Float(tensor.shape[1])
            let unetParams = diagonalDownForRedefine.2
            if let scalar = scalar, abs(scalar - loraDim) > 1e-5 {
              tensor = Tensor<FloatType>(
                from: (Float(Double(scalar / loraDim) * scaleFactor.squareRoot())
                  * graph.variable(Tensor<Float>(from: tensor)))
                  .rawValue)
            } else if scaleFactor != 1 {
              tensor = Tensor<FloatType>(
                from: (Float(scaleFactor.squareRoot())
                  * graph.variable(Tensor<Float>(from: tensor)))
                  .rawValue)
            }
            let prefix = diagonalDownForRedefine.1
            unetParams.write(
              graph: graph,
              to: store, tensor: tensor, format: .O, isDiagonalUp: false,
              isDiagonalDown: true
            ) {
              "__\(prefix)__[\($0)]__up__"
            }
          }
        }
      }
    }
    return (modelVersion, didImportTIEmbedding, textEmbeddingLength, isLoHa)
  }
}
