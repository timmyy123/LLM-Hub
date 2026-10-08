import DataModels
import Diffusion
import Foundation

public enum ComputeUnits {
  private struct ResolvedModelContext {
    let modelName: String
    let modelVersion: ModelVersion
    let samplerModifier: SamplerModifier
    let isGuidanceEmbedEnabled: Bool
    let isConsistencyModel: Bool
  }

  private static func resolveModelContext(
    _ configuration: GenerationConfiguration,
    overrideMapping: (
      model: [String: ModelZoo.Specification], lora: [String: LoRAZoo.Specification]
    )? = nil
  ) -> ResolvedModelContext? {
    guard let model = configuration.model else {
      return nil
    }
    let modelVersion: ModelVersion
    let samplerModifier: SamplerModifier
    let isGuidanceEmbedEnabled: Bool
    let isConsistencyModel: Bool

    if let overrideMapping = overrideMapping, let specification = overrideMapping.model[model] {
      modelVersion = specification.version
      var modifier = specification.modifier ?? .none
      if modifier == .none {
        for lora in configuration.loras {
          guard let file = lora.file, let specification = overrideMapping.lora[file],
            let loraModifier = specification.modifier
          else { continue }
          if loraModifier != .none {
            modifier = loraModifier
            break
          }
        }
      }
      samplerModifier = modifier
      isGuidanceEmbedEnabled =
        (specification.guidanceEmbed ?? false)
        && configuration.speedUpWithGuidanceEmbed
      isConsistencyModel = specification.isConsistencyModel ?? false
    } else {
      modelVersion = ModelZoo.versionForModel(model)
      var modifier = ModelZoo.modifierForModel(model)
      for lora in configuration.loras {
        guard let file = lora.file else { continue }
        let loraModifier = LoRAZoo.modifierForModel(file)
        if loraModifier != .none {
          modifier = loraModifier
        }
      }
      samplerModifier = modifier
      isGuidanceEmbedEnabled =
        ModelZoo.guidanceEmbedForModel(model) && configuration.speedUpWithGuidanceEmbed
      isConsistencyModel = ModelZoo.isConsistencyModelForModel(model)
    }
    return ResolvedModelContext(
      modelName: model,
      modelVersion: modelVersion,
      samplerModifier: samplerModifier,
      isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
      isConsistencyModel: isConsistencyModel
    )
  }

  private static func cfgChannels(
    configuration: GenerationConfiguration, context: ResolvedModelContext
  ) -> Int {
    let isCfgEnabled =
      (!context.isConsistencyModel && !context.isGuidanceEmbedEnabled)
      && isCfgEnabled(
        textGuidanceScale: configuration.guidanceScale,
        imageGuidanceScale: configuration.imageGuidanceScale,
        startFrameCfg: configuration.startFrameCfg, version: context.modelVersion,
        modifier: context.samplerModifier
      )
    let (cfgChannels, _) = cfgChannelsAndInputChannels(
      channels: 0, conditionShape: nil, isCfgEnabled: isCfgEnabled,
      textGuidanceScale: configuration.guidanceScale,
      imageGuidanceScale: configuration.imageGuidanceScale, version: context.modelVersion,
      modifier: context.samplerModifier)
    return cfgChannels
  }

  private static func batchSizeAndNumFrames(
    configuration: GenerationConfiguration,
    context: ResolvedModelContext,
    cfgChannels: Int
  ) -> (batchSize: Int, numFrames: Int) {
    switch context.modelVersion {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large, .flux1, .qwenImage, .qwenImage2_1,
      .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .hiDreamI1, .hiDreamO1,
      .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
      return (max(1, Int(configuration.batchSize)) * cfgChannels, 1)
    case .svdI2v:
      return (cfgChannels, max(1, Int(configuration.numFrames)))
    case .hunyuanVideo:
      return (cfgChannels, max(1, (Int(configuration.numFrames) - 1) / 4 + 1))
    case .wan21_1_3b, .wan21_14b, .wan22_5b, .longcatVideoAvatar1_5:
      return (cfgChannels, max(1, (Int(configuration.numFrames) - 1) / 4 + 1))
    case .ltx2, .ltx2_3:
      return (cfgChannels, max(1, (Int(configuration.numFrames) - 1) / 8 + 1))
    case .minimaxH3:
      let frames = Int(configuration.numFrames)
      return (cfgChannels, frames == 1 ? 1 : ((max(frames, 5) - 5 + 16) / 17) * 5 + 2)
    }
  }

  private static func referenceImageCount(
    context: ResolvedModelContext,
    hasImage: Bool,
    shuffleCount: Int
  ) -> Int {
    let baseReferenceCount = hasImage ? 1 : 0
    let extraReferenceCount = max(0, shuffleCount)
    switch context.samplerModifier {
    case .qwenimageLayered:
      return baseReferenceCount
    case .kontext, .kontextKv, .qwenimageEditPlus, .qwenimageEdit2511, .ref2va:
      return baseReferenceCount + extraReferenceCount
    case .fl2va:
      return hasImage ? 1 + min(extraReferenceCount, 1) : 0
    default:
      return 0
    }
  }

  private static func defaultTokenLength(modelName: String, version: ModelVersion) -> Int {
    let paddedLength = ModelZoo.paddedTextEncodingLengthForModel(modelName)
    if paddedLength > 0 {
      return paddedLength
    }
    switch version {
    case .flux1, .minimaxH3:
      return 512
    case .hunyuanVideo:
      return 256
    case .qwenImage, .qwenImage2_1, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ideogram4:
      return 512
    case .krea2:
      return 256
    case .ltx2, .ltx2_3:
      return 128
    case .seedvr2_3b, .seedvr2_7b:
      return 77
    case .hiDreamI1:
      return 128
    default:
      return 77
    }
  }

  private static func modelStartSizeFromConfiguration(
    _ configuration: GenerationConfiguration,
    version: ModelVersion
  ) -> (width: Int, height: Int) {
    let rawWidth = max(1, Int(configuration.startWidth))
    let rawHeight = max(1, Int(configuration.startHeight))
    switch version {
    case .wurstchenStageC:
      return (
        Int((Double(rawWidth) * 3.0 / 2.0).rounded(.up)),
        Int((Double(rawHeight) * 3.0 / 2.0).rounded(.up))
      )
    case .wan22_5b, .minimaxH3, .qwenImage2_1:
      return (rawWidth * 4, rawHeight * 4)
    case .ltx2, .ltx2_3:
      return (rawWidth * 2, rawHeight * 2)
    case .hiDreamO1:
      return (rawWidth * 2, rawHeight * 2)
    default:
      return (rawWidth * 8, rawHeight * 8)
    }
  }

  private static func wurstchenStageBCountsForStageC(
    _ configuration: GenerationConfiguration,
    batchSize: Int
  ) -> (main: Int, fixed: Int) {
    let width = max(1, Int(configuration.startWidth)) * 16
    let height = max(1, Int(configuration.startHeight)) * 16
    let mainCount = WurstchenStageBInstructionCount(
      batchSize: 1, cIn: 4, height: height, width: width)
    let fixedCount = WurstchenStageBFixedInstructionCount(
      batchSize: 1, height: height, width: width, effnetHeight: height,
      effnetWidth: width)
    return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
  }

  private static func instructionCountFromResolvedModel(
    _ configuration: GenerationConfiguration,
    context: ResolvedModelContext,
    hasImage: Bool,
    shuffleCount: Int
  ) -> (main: Int, fixed: Int) {
    let modelStartSize = modelStartSizeFromConfiguration(
      configuration, version: context.modelVersion)
    let startWidth = modelStartSize.width
    let startHeight = modelStartSize.height
    let cfgChannels = cfgChannels(configuration: configuration, context: context)
    let (batchSize, numFrames) = batchSizeAndNumFrames(
      configuration: configuration, context: context, cfgChannels: cfgChannels)
    let baseTokenLength = max(
      1, defaultTokenLength(modelName: context.modelName, version: context.modelVersion))
    let tokenLengths = (baseTokenLength, baseTokenLength)
    let referenceLengthPerImage = max(1, (startHeight / 2) * (startWidth / 2))
    let referenceSequenceLength =
      referenceLengthPerImage
      * referenceImageCount(context: context, hasImage: hasImage, shuffleCount: shuffleCount)

    switch context.modelVersion {
    case .minimaxH3:
      let frames = numFrames == 1 ? 1 : (numFrames - 2) / 5 * 17 + 5
      let audioLength =
        2
        * Int(
          (Double(frames) / Double(MiniMaxH3Configuration.framesPerSecond) * 40).rounded())
      let mainCount = MiniMaxH3InstructionCount(
        videoTime: numFrames, videoHeight: startHeight, videoWidth: startWidth,
        audioLength: audioLength, textLength: baseTokenLength,
        referenceSequenceLength: referenceSequenceLength)
      let fixedCount = MiniMaxH3FixedInstructionCount(
        timesteps: max(1, Int(configuration.steps)), hiddenSize: 5_376, layers: 50,
        textLength: (cfgChannels > 1 ? baseTokenLength : 0, baseTokenLength),
        referenceSequenceLength: referenceSequenceLength)
      return (main: mainCount * batchSize, fixed: fixedCount)
    case .v1:
      let mainCount = UNetInstructionCount(
        batchSize: 1, embeddingLength: tokenLengths, startWidth: startWidth,
        startHeight: startHeight, injectIPAdapterLengths: [], injectAttentionKV: false)
      return (main: mainCount * batchSize, fixed: 0)
    case .v2:
      let mainCount = UNetv2InstructionCount(
        batchSize: 1, embeddingLength: tokenLengths, startWidth: startWidth,
        startHeight: startHeight)
      return (main: mainCount * batchSize, fixed: 0)
    case .svdI2v:
      let mainCount = UNetXLInstructionCount(
        batchSize: numFrames, startHeight: startHeight, startWidth: startWidth,
        channels: [320, 640, 1280, 1280],
        inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
        outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], embeddingLength: (1, 1),
        injectIPAdapterLengths: [], isTemporalMixEnabled: true)
      let fixedCount = UNetXLFixedInstructionCount(
        batchSize: numFrames, startHeight: startHeight, startWidth: startWidth,
        channels: [320, 640, 1280, 1280], embeddingLength: (1, 1),
        inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
        outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], isTemporalMixEnabled: true,
        temporalFrameEmbeddingRows: numFrames)
      return (
        main: mainCount * batchSize,
        fixed: fixedCount * batchSize
      )
    case .kandinsky21:
      let mainCount = UNetKandinskyInstructionCount(
        batchSize: 1, channels: 384, outChannels: 8, channelMult: [1, 2, 3, 4],
        numResBlocks: 3, numHeadChannels: 64, t: 87, startHeight: startHeight,
        startWidth: startWidth, attentionResolutions: Set([2, 4, 8]))
      return (main: mainCount * batchSize, fixed: 0)
    case .sdxlBase:
      let mainCount = UNetXLInstructionCount(
        batchSize: 1, startHeight: startHeight, startWidth: startWidth,
        channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
        middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
        embeddingLength: tokenLengths, injectIPAdapterLengths: [], isTemporalMixEnabled: false)
      let fixedCount = UNetXLFixedInstructionCount(
        batchSize: 1, startHeight: startHeight, startWidth: startWidth,
        channels: [320, 640, 1280], embeddingLength: tokenLengths,
        inputAttentionRes: [2: [2, 2], 4: [10, 10]], middleAttentionBlocks: 10,
        outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]], isTemporalMixEnabled: false)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .sdxlRefiner:
      let mainCount = UNetXLInstructionCount(
        batchSize: 1, startHeight: startHeight, startWidth: startWidth,
        channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
        middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
        embeddingLength: tokenLengths, injectIPAdapterLengths: [], isTemporalMixEnabled: false)
      let fixedCount = UNetXLFixedInstructionCount(
        batchSize: 1, startHeight: startHeight, startWidth: startWidth,
        channels: [384, 768, 1536, 1536], embeddingLength: tokenLengths,
        inputAttentionRes: [2: [4, 4], 4: [4, 4]], middleAttentionBlocks: 4,
        outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]], isTemporalMixEnabled: false)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .ssd1b:
      let mainCount = UNetXLInstructionCount(
        batchSize: 1, startHeight: startHeight, startWidth: startWidth,
        channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
        middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
        embeddingLength: tokenLengths, injectIPAdapterLengths: [], isTemporalMixEnabled: false)
      let fixedCount = UNetXLFixedInstructionCount(
        batchSize: 1, startHeight: startHeight, startWidth: startWidth,
        channels: [320, 640, 1280], embeddingLength: tokenLengths,
        inputAttentionRes: [2: [2, 2], 4: [4, 4]], middleAttentionBlocks: 0,
        outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]], isTemporalMixEnabled: false)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .wurstchenStageC:
      let mainCount = WurstchenStageCInstructionCount(
        batchSize: 1, height: startHeight, width: startWidth,
        t: (baseTokenLength + 8, baseTokenLength + 8))
      let fixedCount = WurstchenStageCFixedInstructionCount(
        batchSize: 1, t: (baseTokenLength + 8, baseTokenLength + 8))
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .wurstchenStageB:
      let mainCount = WurstchenStageBInstructionCount(
        batchSize: 1, cIn: 4, height: startHeight, width: startWidth)
      let fixedCount = WurstchenStageBFixedInstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, effnetHeight: startHeight,
        effnetWidth: startWidth)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .sd3:
      let dualAttentionLayers =
        ModelZoo.specificationForModel(context.modelName)?.mmdit?
        .dualAttentionLayers ?? []
      let mainCount = MMDiTInstructionCount(
        batchSize: 1, t: baseTokenLength, height: startHeight, width: startWidth,
        channels: 1536, layers: 24, dualAttentionLayers: dualAttentionLayers)
      let fixedCount = MMDiTFixedInstructionCount(
        batchSize: 1, contextLength: baseTokenLength, channels: 1536, layers: 24,
        dualAttentionLayers: dualAttentionLayers)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .sd3Large:
      let mainCount = MMDiTInstructionCount(
        batchSize: 1, t: baseTokenLength, height: startHeight, width: startWidth,
        channels: 2432, layers: 38, dualAttentionLayers: [])
      let fixedCount = MMDiTFixedInstructionCount(
        batchSize: 1, contextLength: baseTokenLength, channels: 2432, layers: 38,
        dualAttentionLayers: [])
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .pixart:
      let mainCount = PixArtInstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, channels: 1152, layers: 28,
        tokenLength: tokenLengths)
      let fixedCount = PixArtFixedInstructionCount(
        batchSize: 1, channels: 1152, layers: 28, tokenLength: tokenLengths)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .auraflow:
      let tokenLength = max(256, baseTokenLength)
      let mainCount = AuraFlowInstructionCount(
        batchSize: 1, tokenLength: tokenLength, height: startHeight,
        width: startWidth, channels: 3072, layers: (4, 32))
      let fixedCount = AuraFlowFixedInstructionCount(
        batchSize: (1, 1), tokenLength: tokenLength, channels: 3072, layers: (4, 32))
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .flux1:
      let mainCount = Flux1InstructionCount(
        batchSize: 1, tokenLength: baseTokenLength,
        referenceSequenceLength: referenceSequenceLength,
        height: startHeight, width: startWidth, channels: 3072, layers: (19, 38),
        contextPreloaded: true)
      let fixedCount = Flux1FixedInstructionCount(
        batchSize: (1, 1), tokenLength: baseTokenLength,
        referenceSequenceLength: referenceSequenceLength, channels: 3072, layers: (19, 38),
        contextPreloaded: true, guidanceEmbed: context.isGuidanceEmbedEnabled)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .hunyuanVideo:
      let fixedTextLength: (Int, Int) =
        cfgChannels > 1
        ? (baseTokenLength, baseTokenLength)
        : (0, baseTokenLength)
      let mainCount = HunyuanInstructionCount(
        time: numFrames, height: startHeight, width: startWidth, textLength: baseTokenLength,
        channels: 3072, layers: (20, 40))
      let fixedCount = HunyuanFixedInstructionCount(
        timesteps: 1, channels: 3072, layers: (20, 40), textLength: fixedTextLength)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .wan21_1_3b:
      let causalInference: (Int, pad: Int) =
        configuration.causalInferenceEnabled
        ? (
          max(0, Int(configuration.causalInference)), max(0, Int(configuration.causalInferencePad))
        )
        : (0, 0)
      let mainCount = WanInstructionCount(
        channels: 1536, layers: 30, vaceLayers: [], intermediateSize: 8960,
        time: numFrames, height: startHeight, width: startWidth, textLength: baseTokenLength,
        causalInference: causalInference, injectImage: hasImage, outputChannels: 16)
      let fixedCount = WanFixedInstructionCount(
        timesteps: 1, batchSize: (1, 1), channels: 1536, layers: 30, vaceLayers: [],
        textLength: baseTokenLength, injectImage: hasImage)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .longcatVideoAvatar1_5:
      let mainCount = WanInstructionCount(
        channels: 4096, layers: 48, vaceLayers: [], intermediateSize: 11_008,
        time: numFrames, height: startHeight, width: startWidth, textLength: baseTokenLength,
        causalInference: (0, 0), injectImage: false, outputChannels: 16)
      let fixedCount = WanFixedInstructionCount(
        timesteps: 1, batchSize: (1, 1), channels: 4096, layers: 48, vaceLayers: [],
        textLength: baseTokenLength, injectImage: false)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .wan21_14b:
      let causalInference: (Int, pad: Int) =
        configuration.causalInferenceEnabled
        ? (
          max(0, Int(configuration.causalInference)), max(0, Int(configuration.causalInferencePad))
        )
        : (0, 0)
      let mainCount = WanInstructionCount(
        channels: 5120, layers: 40, vaceLayers: [], intermediateSize: 13_824,
        time: numFrames, height: startHeight, width: startWidth, textLength: baseTokenLength,
        causalInference: causalInference, injectImage: hasImage, outputChannels: 16)
      let fixedCount = WanFixedInstructionCount(
        timesteps: 1, batchSize: (1, 1), channels: 5120, layers: 40, vaceLayers: [],
        textLength: baseTokenLength, injectImage: hasImage)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .wan22_5b:
      let causalInference: (Int, pad: Int) =
        configuration.causalInferenceEnabled
        ? (
          max(0, Int(configuration.causalInference)), max(0, Int(configuration.causalInferencePad))
        )
        : (0, 0)
      let mainCount = WanInstructionCount(
        channels: 3072, layers: 30, vaceLayers: [], intermediateSize: 14_336,
        time: numFrames, height: startHeight, width: startWidth, textLength: baseTokenLength,
        causalInference: causalInference, injectImage: false, outputChannels: 48)
      let fixedCount = WanFixedInstructionCount(
        timesteps: 1, batchSize: (1, 1), channels: 3072, layers: 30, vaceLayers: [],
        textLength: baseTokenLength, injectImage: false)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .qwenImage2_1:
      let references = referenceImageCount(
        context: context, hasImage: hasImage, shuffleCount: shuffleCount)
      // Estimate the text before the image blocks; exact interleaving comes from the prompt at runtime.
      let segments: [(length: Int, image: Bool)] =
        [(baseTokenLength, false)]
        + Array(repeating: (4096, true), count: references)
      return (
        main: QwenImage2_1InstructionCount(
          batchSize: batchSize, height: startHeight, width: startWidth,
          prefixLength: baseTokenLength + 4096 * references),
        fixed: QwenImage2_1FixedInstructionCount(
          batchSize: cfgChannels,
          timesteps: max(1, Int(configuration.steps)), segments: segments)
      )
    case .qwenImage:
      let mainCount = QwenImageInstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, textLength: baseTokenLength,
        referenceSequenceLength: referenceSequenceLength, channels: 3072, layers: 60,
        isQwenImageLayered: context.samplerModifier == .qwenimageLayered)
      let fixedCount = QwenImageFixedInstructionCount(
        timesteps: 1, batchSize: 1, textLength: baseTokenLength, channels: 3072, layers: 60,
        referenceSequenceLength: referenceSequenceLength)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .zImage:
      let mainCount = ZImageInstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, textLength: baseTokenLength,
        channels: 3840, layers: 30)
      let fixedCount = ZImageFixedInstructionCount(
        batchSize: 1, timesteps: 1, tokenLength: tokenLengths, channels: 3840, layers: 30)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .ernieImage:
      let mainCount = ErnieImageInstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, textLength: baseTokenLength,
        channels: 4_096, layers: 36, intermediateSize: 12_288)
      let fixedCount = ErnieImageFixedInstructionCount(
        batchSize: 1, textLength: baseTokenLength, channels: 4_096)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .ideogram4:
      let mainCount = Ideogram4InstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, textLength: baseTokenLength,
        channels: 4_608, layers: 34, intermediateSize: 12_288)
      let fixedCount = Ideogram4FixedInstructionCount(
        timesteps: 1, batchSize: 1, textLength: baseTokenLength, channels: 4_608, layers: 34)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .krea2:
      let fixedTextLength: (Int, Int) =
        cfgChannels > 1
        ? (baseTokenLength, baseTokenLength)
        : (0, baseTokenLength)
      let totalFixedTextLength = fixedTextLength.0 + fixedTextLength.1
      let mainCount = Krea2InstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, textLength: baseTokenLength,
        hiddenSize: 6_144, layers: 28, intermediateSize: 16_384)
      let textFusionCount = Krea2TextFusionAdapterInstructionCount(
        batchSize: 1, textLength: fixedTextLength)
      let fixedCount = Krea2FixedInstructionCount(
        batchSize: 1, timesteps: 1, textLength: totalFixedTextLength)
      return (main: mainCount * batchSize, fixed: textFusionCount + fixedCount * batchSize)
    case .seedvr2_3b, .seedvr2_7b:
      let configuration: SeedVR2DiTConfiguration =
        context.modelVersion == .seedvr2_7b ? ._7B : ._3B
      let mainCount = ErnieImageInstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, textLength: baseTokenLength,
        channels: configuration.hiddenSize, layers: configuration.layers,
        intermediateSize: configuration.mlpHiddenSize)
      return (main: mainCount * batchSize, fixed: 0)
    case .cosmos2_5_2b:
      let mainCount = ZImageInstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, textLength: baseTokenLength,
        channels: 2048, layers: 28)
      return (main: mainCount * batchSize, fixed: 0)
    case .flux2, .flux2_9b, .flux2_4b:
      let channels: Int
      let layers: (Int, Int)
      if context.modelVersion == .flux2_9b {
        channels = 4096
        layers = (8, 24)
      } else if context.modelVersion == .flux2_4b {
        channels = 3072
        layers = (5, 20)
      } else {
        channels = 6144
        layers = (8, 48)
      }
      let kvCache = context.samplerModifier == .kontextKv && referenceSequenceLength > 0
      let mainCount = Flux2InstructionCount(
        batchSize: 1, tokenLength: baseTokenLength,
        referenceSequenceLength: kvCache ? 0 : referenceSequenceLength,
        cachedReferenceSequenceLength: kvCache ? referenceSequenceLength : 0,
        height: startHeight, width: startWidth, channels: channels, layers: layers)
      let fixedCount = Flux2FixedInstructionCount(
        contextBatchSize: 1, tokenLength: baseTokenLength, timestepCount: kvCache ? 2 : 1,
        referenceSequenceLength: referenceSequenceLength, channels: channels, layers: layers,
        guidanceEmbed: context.isGuidanceEmbedEnabled, kvCache: kvCache)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .hiDreamI1:
      let hiDreamWidth =
        context.samplerModifier == .editing
        ? startWidth * 2
        : startWidth
      let mainCount = HiDreamInstructionCount(
        batchSize: 1, height: startHeight, width: hiDreamWidth, textLength: tokenLengths,
        layers: (16, 32))
      let fixedCount = HiDreamFixedInstructionCount(
        timesteps: 1, layers: (16, 32), t5TextLength: baseTokenLength,
        llamaTextLength: baseTokenLength)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .hiDreamO1:
      let mainCount = HiDreamO1InstructionCount(
        batchSize: 1, height: startHeight, width: startWidth, textLength: baseTokenLength)
      let fixedCount = HiDreamO1FixedInstructionCount(batchSize: 1, textLength: baseTokenLength)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .ltx2:
      let videoFrames = numFrames
      let audioFrames = (videoFrames - 1) * 8 + 1
      let paddedTextLength = max(baseTokenLength, 1024)
      let mainCount = LTX2InstructionCount(
        time: videoFrames, h: startHeight, w: startWidth, textLength: paddedTextLength,
        audioFrames: audioFrames, channels: (4096, 2048), layers: 48,
        tokenModulation: hasImage, KV: true, useGatedAttention: false,
        textCrossAttentionAdaLN: false)
      let fixedCount = LTX2FixedInstructionCount(
        time: videoFrames, textLength: paddedTextLength, audioFrames: audioFrames, timesteps: 1,
        channels: (4096, 2048), layers: 48, contextProjection: true,
        textCrossAttentionAdaLN: false, KV: true)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    case .ltx2_3:
      let videoFrames = numFrames
      let audioFrames = (videoFrames - 1) * 8 + 1
      let paddedTextLength = max(baseTokenLength, 1024)
      let mainCount = LTX2InstructionCount(
        time: videoFrames, h: startHeight, w: startWidth, textLength: paddedTextLength,
        audioFrames: audioFrames, channels: (4096, 2048), layers: 48,
        tokenModulation: hasImage, KV: false, useGatedAttention: true,
        textCrossAttentionAdaLN: true)
      let fixedCount = LTX2FixedInstructionCount(
        time: videoFrames, textLength: paddedTextLength, audioFrames: audioFrames, timesteps: 1,
        channels: (4096, 2048), layers: 48, contextProjection: false,
        textCrossAttentionAdaLN: true, KV: false)
      return (main: mainCount * batchSize, fixed: fixedCount * batchSize)
    }
  }

  // Single global calibration scalar from instruction-count domain to legacy compute-unit domain.
  // Calibrated against Flux1 / Flux2_9B with a 1024x1024 focus.
  private static let instructionCalibrationScale: Double = 5.14816e-12

  // Policy-side multiplier to bias compute-unit cost by model family.
  // This is intentionally non-physical and is used to steer usage toward preferred models.
  private static func modelUsageBiasMultiplier(for version: ModelVersion) -> Double {
    switch version {
    case .v1, .v2, .kandinsky21, .wurstchenStageC, .wurstchenStageB, .sdxlBase, .sdxlRefiner,
      .ssd1b,
      .svdI2v:
      return 2.5
    default:
      return 1.0
    }
  }

  private static func from(
    _ configuration: GenerationConfiguration,
    context: ResolvedModelContext,
    hasImage: Bool,
    shuffleCount: Int
  ) -> Int {
    if context.modelVersion == .wurstchenStageC {
      let firstPassCounts = instructionCountFromResolvedModel(
        configuration, context: context, hasImage: hasImage, shuffleCount: shuffleCount)
      let cfgChannels = cfgChannels(configuration: configuration, context: context)
      let (batchSize, _) = batchSizeAndNumFrames(
        configuration: configuration, context: context, cfgChannels: cfgChannels)
      let secondPassCounts = wurstchenStageBCountsForStageC(configuration, batchSize: batchSize)
      let strength = Double(max(configuration.strength, 0.05))
      let mainEstimate =
        (Double(firstPassCounts.main) * Double(configuration.steps)
          + Double(secondPassCounts.main) * Double(configuration.stage2Steps))
        * instructionCalibrationScale * strength
      let fixedEstimate =
        Double(firstPassCounts.fixed + secondPassCounts.fixed) * instructionCalibrationScale
      let biasedEstimate =
        (mainEstimate + fixedEstimate) * modelUsageBiasMultiplier(for: context.modelVersion)
      guard biasedEstimate.isFinite, biasedEstimate > 0 else { return 0 }
      return Int(min(Double(Int.max), biasedEstimate.rounded(.up)))
    }
    let instructionCounts = instructionCountFromResolvedModel(
      configuration, context: context, hasImage: hasImage, shuffleCount: shuffleCount)
    let mainEstimate =
      Double(instructionCounts.main) * instructionCalibrationScale
      * Double(configuration.steps) * Double(max(configuration.strength, 0.05))
    let fixedEstimate = Double(instructionCounts.fixed) * instructionCalibrationScale
    let biasedEstimate =
      (mainEstimate + fixedEstimate) * modelUsageBiasMultiplier(for: context.modelVersion)
    guard biasedEstimate.isFinite, biasedEstimate > 0 else { return 0 }
    return Int(min(Double(Int.max), biasedEstimate.rounded(.up)))
  }

  public static func thresholdAfterBoost(
    policyThreshold: Int,
    boostsToSpend: Int
  ) -> Int {
    return max(
      policyThreshold,
      (boostsToSpend - 1) * ComputeUnits.perBoost + max(policyThreshold, ComputeUnits.perBoost))
  }

  public static func from(
    _ configuration: GenerationConfiguration,
    hasImage: Bool, shuffleCount: Int,
    overrideMapping: (
      model: [String: ModelZoo.Specification], lora: [String: LoRAZoo.Specification]
    )? = nil
  ) -> Int? {
    guard
      let context = resolveModelContext(configuration, overrideMapping: overrideMapping)
    else {
      return nil
    }
    return from(
      configuration, context: context, hasImage: hasImage, shuffleCount: shuffleCount)
  }

  public static func threshold(for priority: String?) -> Int {
    switch priority {
    case "community":
      return 10_000  // around 120s
    case "plus":
      return 40000  // around 300s
    case nil:
      return 10_000
    default:
      return 10_000
    }
  }

  public static func threshold(
    for priority: String?, computeUnitPolicy: [String: Int]?, expirationTimestamp: Date?
  ) -> Int {
    // Check if we have a valid policy and it's not expired
    let currentTimestamp = Date()
    if let policy = computeUnitPolicy,
      let expiration = expirationTimestamp,
      currentTimestamp < expiration, let priority = priority,
      let policyValue = policy[priority]
    {
      return policyValue
    }

    // Fallback to original default setup
    return threshold(for: priority)
  }

  public static let perBoost: Int = 60000

  public static func boost(for computeUnits: Int, threshold: Int) -> Int {
    return 1
      + Int(
        (Double(computeUnits - max(threshold, Self.perBoost)) / Double(Self.perBoost)).rounded(.up))
  }
}
