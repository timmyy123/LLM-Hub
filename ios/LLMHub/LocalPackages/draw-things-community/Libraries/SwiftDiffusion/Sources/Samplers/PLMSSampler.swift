import Foundation
import NNC
import WeightsCache

private func xPrevAndPredX0<T: TensorNumeric>(
  x: DynamicGraph.Tensor<T>, et: DynamicGraph.Tensor<T>, alpha: Double,
  alphaPrev: Double
) -> (DynamicGraph.Tensor<T>, DynamicGraph.Tensor<T>) {
  let predX0 = Float(1 / alpha.squareRoot()) * (x - Float((1 - alpha).squareRoot()) * et)
  let dirXt = Float((1 - alphaPrev).squareRoot()) * et
  let xPrev = Float(alphaPrev.squareRoot()) * predX0 + dirXt
  return (xPrev, predX0)
}

public struct PLMSSampler<
  FloatType: TensorNumeric & BinaryFloatingPoint, UNet: UNetProtocol,
  Discretization: Denoiser.Discretization
>
where UNet.FloatType == FloatType {
  public let filePath: String
  public let modifier: SamplerModifier
  public let version: ModelVersion
  public let audioShiftRatio: Float
  public let qkNorm: Bool
  public let dualAttentionLayers: [Int]
  public let distilledGuidanceLayers: Int
  public let activationQkScaling: [Int: Int]
  public let activationProjScaling: [Int: Int]
  public let activationFfnProjUpScaling: [Int: Int]
  public let activationFfnScaling: [Int: Int]
  public let usesFlashAttention: UseFlashAttention
  public let usesSolAttention: Bool
  public let solAttentionStart: Int
  public let solAttentionTau: Float
  public let upcastAttention: Bool
  public let externalOnDemand: Bool
  public let injectControls: Bool
  public let injectT2IAdapters: Bool
  public let injectAttentionKV: Bool

  public let injectIPAdapterLengths: [Int]
  public let lora: [LoRAConfiguration]
  public let classifierFreeGuidance: Bool
  public let isGuidanceEmbedEnabled: Bool
  public let isQuantizedModel: Bool
  public let canRunLoRASeparately: Bool
  public let deviceProperties: DeviceProperties
  public let conditioning: Denoiser.Conditioning
  public let tiledDiffusion: TiledConfiguration
  public let teaCache: TeaCacheConfiguration
  public let causalInference: (Int, pad: Int)
  public let cfgZeroStar: CfgZeroStarConfiguration
  public let isBF16: Bool
  private let discretization: Discretization
  private let weightsCache: WeightsCache
  public init(
    filePath: String, modifier: SamplerModifier, version: ModelVersion,
    audioShiftRatio: Float, qkNorm: Bool,
    dualAttentionLayers: [Int], distilledGuidanceLayers: Int, activationQkScaling: [Int: Int],
    activationProjScaling: [Int: Int], activationFfnProjUpScaling: [Int: Int],
    activationFfnScaling: [Int: Int],
    usesFlashAttention: UseFlashAttention,
    usesSolAttention: Bool = false,
    solAttentionStart: Int = 2, solAttentionTau: Float = 0.5,
    upcastAttention: Bool, externalOnDemand: Bool, injectControls: Bool,
    injectT2IAdapters: Bool, injectAttentionKV: Bool, injectIPAdapterLengths: [Int],
    lora: [LoRAConfiguration],
    classifierFreeGuidance: Bool, isGuidanceEmbedEnabled: Bool, isQuantizedModel: Bool,
    canRunLoRASeparately: Bool, deviceProperties: DeviceProperties,
    conditioning: Denoiser.Conditioning, tiledDiffusion: TiledConfiguration,
    teaCache: TeaCacheConfiguration, causalInference: (Int, pad: Int),
    cfgZeroStar: CfgZeroStarConfiguration, isBF16: Bool, discretization: Discretization,
    weightsCache: WeightsCache
  ) {
    self.filePath = filePath
    self.modifier = modifier
    self.version = version
    self.audioShiftRatio = audioShiftRatio
    self.qkNorm = qkNorm
    self.dualAttentionLayers = dualAttentionLayers
    self.distilledGuidanceLayers = distilledGuidanceLayers
    self.activationQkScaling = activationQkScaling
    self.activationProjScaling = activationProjScaling
    self.activationFfnProjUpScaling = activationFfnProjUpScaling
    self.activationFfnScaling = activationFfnScaling
    self.usesFlashAttention = usesFlashAttention
    self.usesSolAttention = usesSolAttention
    self.solAttentionStart = solAttentionStart
    self.solAttentionTau = solAttentionTau
    self.upcastAttention = upcastAttention
    self.externalOnDemand = externalOnDemand
    self.injectControls = injectControls
    self.injectT2IAdapters = injectT2IAdapters
    self.injectAttentionKV = injectAttentionKV

    self.injectIPAdapterLengths = injectIPAdapterLengths
    self.lora = lora
    self.classifierFreeGuidance = classifierFreeGuidance
    self.isGuidanceEmbedEnabled = isGuidanceEmbedEnabled
    self.isQuantizedModel = isQuantizedModel
    self.canRunLoRASeparately = canRunLoRASeparately
    self.deviceProperties = deviceProperties
    self.conditioning = conditioning
    self.tiledDiffusion = tiledDiffusion
    self.teaCache = teaCache
    self.causalInference = causalInference
    self.cfgZeroStar = cfgZeroStar
    self.isBF16 = isBF16
    self.discretization = discretization

    self.weightsCache = weightsCache
  }
}

extension PLMSSampler: Sampler {
  public func sample(
    _ x_T: DynamicGraph.Tensor<FloatType>, unets existingUNets: [UNet?],
    sample: DynamicGraph.Tensor<FloatType>?, conditionImage: DynamicGraph.Tensor<FloatType>?,
    referenceImages: [DynamicGraph.Tensor<FloatType>],
    referenceAudios: [DynamicGraph.Tensor<FloatType>],
    mask: DynamicGraph.Tensor<FloatType>?, negMask: DynamicGraph.Tensor<FloatType>?,
    conditioning c: [DynamicGraph.Tensor<FloatType>], tokenLengthUncond: Int, tokenLengthCond: Int,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControls: [(
      model: ControlModel<FloatType>, hints: [([DynamicGraph.Tensor<FloatType>], Float)]
    )],
    textGuidanceScale: Float, imageGuidanceScale: Float, guidanceEmbed: Float,
    startStep: (integral: Int, fractional: Float), endStep: (integral: Int, fractional: Float),
    originalSize: (width: Int, height: Int), cropTopLeft: (top: Int, left: Int),
    targetSize: (width: Int, height: Int), aestheticScore: Float,
    negativeOriginalSize: (width: Int, height: Int), negativeAestheticScore: Float,
    zeroNegativePrompt: Bool, refiner: Refiner?, fpsId: Int, motionBucketId: Int, condAug: Float,
    startFrameCfg: Float, sharpness: Float, sampling: Sampling,
    cancellation: (@escaping () -> Void) -> Void, feedback: (Int, Tensor<FloatType>?) -> Bool
  ) -> Result<SamplerOutput<FloatType, UNet>, Error> {
    guard endStep.integral > startStep.integral else {
      return .success(SamplerOutput(x: x_T, unets: [nil]))
    }
    let startStep = startStep.integral
    let endStep = endStep.integral
    var x = x_T
    let batchSize = x.shape[0]
    let startHeight = x.shape[1]
    let startWidth = x.shape[2]
    let channels = x.shape[3]
    let graph = x.graph
    let isCfgEnabled =
      classifierFreeGuidance
      && isCfgEnabled(
        textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
        startFrameCfg: startFrameCfg, version: version, modifier: modifier)
    let (cfgChannels, inChannels) = cfgChannelsAndInputChannels(
      channels: channels, conditionShape: conditionImage?.shape, isCfgEnabled: isCfgEnabled,
      textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
      version: version, modifier: modifier)
    let isBatchEnabled = isBatchEnabled(version)
    let zeroNegativePrompt = isCfgEnabled && zeroNegativePrompt
    var xIn = graph.variable(
      .GPU(0), .NHWC(cfgChannels * batchSize, startHeight, startWidth, inChannels),
      of: FloatType.self
    )
    var c = c
    updateCfgInputAndConditions(
      xIn: &xIn, conditions: &c, conditionImage: conditionImage, batchSize: batchSize,
      startHeight: startHeight, startWidth: startWidth, channels: channels,
      isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale, modifier: modifier)
    var extraProjection = extraProjection
    var tokenLengthUncond = tokenLengthUncond
    if !isCfgEnabled && version != .svdI2v && version != .minimaxH3 {
      for i in 0..<c.count {
        let shape = c[i].shape
        guard shape[0] >= batchSize * 2 else { continue }
        if shape.count == 3 {
          let conditionalLength = version == .kandinsky21 ? shape[1] : tokenLengthCond
          // Only tokenLengthCond is used.
          c[i] = c[i][batchSize..<(batchSize * 2), 0..<conditionalLength, 0..<shape[2]].copied()
        } else if shape.count == 2 {
          c[i] = c[i][batchSize..<(batchSize * 2), 0..<shape[1]].copied()
        }
      }
      if var projection = extraProjection {
        let shape = projection.shape
        if shape[0] >= batchSize * 2 {
          if shape.count == 3 {
            // Only tokenLengthCond is used.
            projection = projection[batchSize..<(batchSize * 2), 0..<shape[1], 0..<shape[2]]
              .copied()
          } else if shape.count == 2 {
            projection = projection[batchSize..<(batchSize * 2), 0..<shape[1]].copied()
          }
        }
        extraProjection = projection
      }
      // There is no tokenLengthUncond any more.
      tokenLengthUncond = tokenLengthCond
    }
    let streamContext = StreamContext(.GPU(0))
    let result: Result<SamplerOutput<FloatType, UNet>, Error> = graph.withStream(streamContext) {
      let oldC = c
      var conditions: [DynamicGraph.AnyTensor] = c
      let fixedEncoder = UNetFixedEncoder<FloatType>(
        filePath: filePath, version: version, modifier: modifier,
        dualAttentionLayers: dualAttentionLayers, activationQkScaling: activationQkScaling,
        activationProjScaling: activationProjScaling,
        activationFfnProjUpScaling: activationFfnProjUpScaling,
        activationFfnScaling: activationFfnScaling,
        usesFlashAttention: usesFlashAttention,
        zeroNegativePrompt: zeroNegativePrompt, isQuantizedModel: isQuantizedModel,
        canRunLoRASeparately: canRunLoRASeparately, externalOnDemand: externalOnDemand,
        deviceProperties: deviceProperties, weightsCache: weightsCache)
      let injectedControlsC: [[DynamicGraph.Tensor<FloatType>]]
      let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
      let timesteps = (startStep..<endStep).map {
        let alphaCumprod = alphasCumprod[$0]
        switch conditioning {
        case .noise:
          return discretization.noise(for: alphaCumprod)
        case .timestep:
          return discretization.timestep(for: alphaCumprod)
        }
      }
      let listOfLoras = lora
      var lora = listOfLoras.filter {
        $0.mode == .base || $0.mode == .all
      }
      if UNetFixedEncoder<FloatType>.isFixedEncoderRequired(version: version) {
        let vector = fixedEncoder.vector(
          textEmbedding: c[c.count - 1], originalSize: originalSize,
          cropTopLeft: cropTopLeft,
          targetSize: targetSize, aestheticScore: aestheticScore,
          negativeOriginalSize: negativeOriginalSize,
          negativeAestheticScore: negativeAestheticScore,
          fpsId: fpsId, motionBucketId: motionBucketId, condAug: condAug)
        let (encodings, weightMapper) = fixedEncoder.encode(
          isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale,
          guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          distilledGuidanceLayers: distilledGuidanceLayers, modifier: modifier,
          textEncoding: c, timesteps: timesteps, audioShiftRatio: audioShiftRatio,
          batchSize: batchSize, startHeight: startHeight,
          startWidth: startWidth,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond, lora: lora,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, isBF16: isBF16,
          injectedControls: injectedControls, referenceImages: referenceImages,
          referenceAudios: referenceAudios)
        conditions = vector + encodings
        injectedControlsC = injectedControls.map {
          $0.model.encode(
            isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale,
            guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
            textEncoding: oldC, timesteps: timesteps, vector: vector.first, batchSize: batchSize,
            startHeight: startHeight,
            startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, zeroNegativePrompt: zeroNegativePrompt,
            mainUNetFixed: (fixedEncoder.filePath, weightMapper))
        }
      } else {
        injectedControlsC = injectedControls.map {
          $0.model.encode(
            isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale,
            guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
            textEncoding: oldC, timesteps: timesteps, vector: nil, batchSize: batchSize,
            startHeight: startHeight,
            startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, zeroNegativePrompt: zeroNegativePrompt,
            mainUNetFixed: (fixedEncoder.filePath, nil))
        }
      }
      var unet = existingUNets[0] ?? UNet()
      defer {
        cancellation {}  // In this way, we are not holding the graph any more.
      }
      cancellation {
        unet.cancel()
      }
      let referenceImageCount = referenceImages.count
      let referenceAudioCount = referenceAudios.count
      var controlNets = [Model?](repeating: nil, count: injectedControls.count)
      let injectControlsAndAdapters = InjectControlsAndAdapters(
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV, injectIPAdapterLengths: injectIPAdapterLengths,
        injectControlModels: injectedControls.map { $0.model })
      if existingUNets[0] == nil {
        let firstTimestep =
          discretization.timesteps - discretization.timesteps / Float(sampling.steps) + 1
        let t = unet.timeEmbed(
          graph: graph, batchSize: cfgChannels * batchSize, timestep: firstTimestep,
          version: version)
        let emptyInjectedControlsAndAdapters =
          ControlModel<FloatType>
          .emptyInjectedControlsAndAdapters(
            injecteds: injectedControls, step: 0, version: version, inputs: xIn,
            tiledDiffusion: tiledDiffusion)
        let newC: [DynamicGraph.AnyTensor]
        if version == .svdI2v {
          newC = Array(conditions[0..<(1 + (conditions.count - 1) / 2)])
        } else {
          newC = conditions
        }
        let _ = unet.compileModel(
          filePath: filePath, externalOnDemand: externalOnDemand,
          deviceProperties: deviceProperties,
          version: version,
          modifier: modifier, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
          usesSolAttention: usesSolAttention,
          solAttentionStart: solAttentionStart, solAttentionTau: solAttentionTau,
          injectControlsAndAdapters: injectControlsAndAdapters, lora: lora,
          isQuantizedModel: isQuantizedModel, canRunLoRASeparately: canRunLoRASeparately,
          inputs: xIn, t,
          UNetExtractConditions(
            of: FloatType.self,
            graph: graph, index: 0, batchSize: cfgChannels * batchSize,
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            conditions: newC,
            referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
            version: version, modifier: modifier, isCfgEnabled: isCfgEnabled),
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          isCfgEnabled: isCfgEnabled, extraProjection: extraProjection,
          injectedControlsAndAdapters: emptyInjectedControlsAndAdapters,
          referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          isBF16: isBF16, activationQkScaling: activationQkScaling,
          activationProjScaling: activationProjScaling,
          activationFfnProjUpScaling: activationFfnProjUpScaling,
          activationFfnScaling: activationFfnScaling, weightsCache: weightsCache)
      }
      var noise: DynamicGraph.Tensor<FloatType>? = nil
      if mask != nil {
        noise = graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, channels))
      }
      let condAugFrames: DynamicGraph.Tensor<FloatType>?
      let textGuidanceVector: DynamicGraph.Tensor<FloatType>?
      if version == .svdI2v {
        let scaleCPU = graph.variable(.CPU, .NHWC(batchSize, 1, 1, 1), of: FloatType.self)
        for i in 0..<batchSize {
          scaleCPU[i, 0, 0, 0] = FloatType(
            Float(i) * (textGuidanceScale - startFrameCfg) / Float(batchSize - 1) + startFrameCfg)
        }
        textGuidanceVector = scaleCPU.toGPU(0)
        let maskedImage = conditionImage!
        var frames = graph.variable(
          .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
        for i in 0..<batchSize {
          frames[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<channels] = maskedImage
        }
        if condAug > 0 {
          let noise = graph.variable(like: frames)
          noise.randn(std: condAug)
          frames = frames .+ noise
        }
        condAugFrames = frames
      } else {
        textGuidanceVector = nil
        condAugFrames = nil
      }
      let blur: Model?
      if sharpness > 0 {
        blur = Blur(filters: channels, sigma: 3.0, size: 13, input: x)
      } else {
        blur = nil
      }
      let injecteds = injectedControls
      var refinerKickIn = refiner.map { (1 - $0.start) * discretization.timesteps } ?? -1
      var unets: [UNet?] = [unet]
      var currentModelVersion = version
      var indexOffset = startStep
      var oldEps = [DynamicGraph.Tensor<FloatType>]()
      // Now do PLMS sampling.
      for i in startStep..<endStep {
        let rawValue: Tensor<FloatType>? =
          (i > max(startStep, sampling.steps / 2) || i % 2 == 1) ? x.rawValue.toCPU() : nil
        if i % 5 == 4, let rawValue = rawValue {
          if isNaN(rawValue) {
            return .failure(SamplerError.isNaN)
          }
        }
        guard feedback(i - startStep, rawValue) else {
          return .failure(SamplerError.cancelled(unets))
        }
        let timestep = discretization.timestep(for: alphasCumprod[i])
        if timestep < refinerKickIn, let refiner = refiner {
          let timesteps = (i..<endStep).map {
            let alphaCumprod = alphasCumprod[$0]
            switch conditioning {
            case .noise:
              return discretization.noise(for: alphaCumprod)
            case .timestep:
              return discretization.timestep(for: alphaCumprod)
            }
          }
          unets = [nil]
          unet.unloadModel()
          lora =
            (refiner.builtinLora
              ? [
                LoRAConfiguration(
                  file: refiner.filePath, weight: 1, version: refiner.version, isLoHa: false,
                  modifier: .none, mode: .refiner)
              ] : [])
            + listOfLoras.filter {
              $0.mode == .all || $0.mode == .refiner
            }
          let fixedEncoder = UNetFixedEncoder<FloatType>(
            filePath: refiner.filePath, version: refiner.version,
            modifier: modifier, dualAttentionLayers: refiner.dualAttentionLayers,
            activationQkScaling: refiner.activationQkScaling,
            activationProjScaling: refiner.activationProjScaling,
            activationFfnProjUpScaling: refiner.activationFfnProjUpScaling,
            activationFfnScaling: refiner.activationFfnScaling,
            usesFlashAttention: usesFlashAttention,
            zeroNegativePrompt: zeroNegativePrompt,
            isQuantizedModel: refiner.isQuantizedModel, canRunLoRASeparately: canRunLoRASeparately,
            externalOnDemand: refiner.externalOnDemand, deviceProperties: deviceProperties,
            weightsCache: weightsCache)
          if UNetFixedEncoder<FloatType>.isFixedEncoderRequired(version: refiner.version) {
            conditions = []
            let vector = fixedEncoder.vector(
              textEmbedding: oldC[oldC.count - 1], originalSize: originalSize,
              cropTopLeft: cropTopLeft,
              targetSize: targetSize, aestheticScore: aestheticScore,
              negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug)
            conditions =
              vector
              + fixedEncoder.encode(
                isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale,
                guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
                distilledGuidanceLayers: refiner.distilledGuidanceLayers, modifier: modifier,
                textEncoding: oldC, timesteps: timesteps, audioShiftRatio: audioShiftRatio,
                batchSize: batchSize,
                startHeight: startHeight,
                startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond, lora: lora, tiledDiffusion: tiledDiffusion,
                teaCache: teaCache, isBF16: isBF16, injectedControls: injectedControls,
                referenceImages: referenceImages, referenceAudios: referenceAudios
              ).0
            indexOffset = i
          }
          unet = UNet()
          cancellation {
            unet.cancel()
          }
          currentModelVersion = refiner.version
          let firstTimestep =
            discretization.timesteps - discretization.timesteps / Float(sampling.steps) + 1
          let t = unet.timeEmbed(
            graph: graph, batchSize: cfgChannels * batchSize, timestep: firstTimestep,
            version: currentModelVersion)
          let emptyInjectedControlsAndAdapters =
            ControlModel<FloatType>
            .emptyInjectedControlsAndAdapters(
              injecteds: injectedControls, step: 0, version: refiner.version, inputs: xIn,
              tiledDiffusion: tiledDiffusion)
          let newC: [DynamicGraph.AnyTensor]
          if version == .svdI2v {
            newC = Array(conditions[0..<(1 + (conditions.count - 1) / 2)])
          } else {
            newC = conditions
          }
          let _ = unet.compileModel(
            filePath: refiner.filePath, externalOnDemand: refiner.externalOnDemand,
            deviceProperties: deviceProperties,
            version: refiner.version, modifier: modifier, qkNorm: refiner.qkNorm,
            dualAttentionLayers: refiner.dualAttentionLayers,
            upcastAttention: refiner.upcastAttention,
            usesFlashAttention: usesFlashAttention, usesSolAttention: usesSolAttention,
            solAttentionStart: solAttentionStart, solAttentionTau: solAttentionTau,
            injectControlsAndAdapters: injectControlsAndAdapters,
            lora: lora, isQuantizedModel: refiner.isQuantizedModel,
            canRunLoRASeparately: canRunLoRASeparately,
            inputs: xIn, t,
            UNetExtractConditions(
              of: FloatType.self,
              graph: graph, index: 0, batchSize: cfgChannels * batchSize,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              conditions: newC, referenceImageCount: referenceImageCount,
              referenceAudioCount: referenceAudioCount,
              version: currentModelVersion, modifier: modifier, isCfgEnabled: isCfgEnabled),
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: isCfgEnabled, extraProjection: extraProjection,
            injectedControlsAndAdapters: emptyInjectedControlsAndAdapters,
            referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
            tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
            isBF16: refiner.isBF16, activationQkScaling: refiner.activationQkScaling,
            activationProjScaling: refiner.activationProjScaling,
            activationFfnProjUpScaling: refiner.activationFfnProjUpScaling,
            activationFfnScaling: refiner.activationFfnScaling,
            weightsCache: weightsCache)
          refinerKickIn = -1
          unets.append(unet)
        }
        let nextAlphaCumprod = alphasCumprod[i + 1]
        let cNoise: Float
        let cNoiseNext: Float
        switch conditioning {
        case .noise:
          cNoise = discretization.noise(for: alphasCumprod[i])
          cNoiseNext = discretization.noise(for: nextAlphaCumprod)
        case .timestep:
          cNoise = timestep
          cNoiseNext = discretization.timestep(for: nextAlphaCumprod)
        }
        let t = unet.timeEmbed(
          graph: graph, batchSize: cfgChannels * batchSize, timestep: cNoise,
          version: currentModelVersion)
        let conditions = UNetExtractConditions(
          of: FloatType.self,
          graph: graph, index: i - indexOffset, batchSize: cfgChannels * batchSize,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          conditions: conditions, referenceImageCount: referenceImageCount,
          referenceAudioCount: referenceAudioCount,
          version: currentModelVersion, modifier: modifier, isCfgEnabled: isCfgEnabled)
        var et: DynamicGraph.Tensor<FloatType>
        if version == .svdI2v, let textGuidanceVector = textGuidanceVector,
          let condAugFrames = condAugFrames
        {
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = x
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] =
            condAugFrames
          let injectedIPAdapters = ControlModel<FloatType>
            .injectedIPAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let cCond = Array(conditions[0..<(1 + (conditions.count - 1) / 2)])
          var etCond = unet(
            timestep: (now: cNoise, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
            inputs: xIn, t, cCond,
            extraProjection: extraProjection,
            injectedControlsAndAdapters: injectedControlsAndAdapters,
            injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
            referenceAudioCount: referenceAudioCount,
            step: i, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
            tiledDiffusion: tiledDiffusion,
            controlNets: &controlNets)
          let alpha =
            0.001 * sharpness * (discretization.timesteps - timestep)
            / discretization.timesteps
          if isCfgEnabled {
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)].full(0)
            let cUncond = Array([conditions[0]] + conditions[(1 + (conditions.count - 1) / 2)...])
            let etUncond = unet(
              timestep: (now: cNoise, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
              inputs: xIn, t, cUncond,
              extraProjection: extraProjection,
              injectedControlsAndAdapters: injectedControlsAndAdapters,
              injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
              referenceAudioCount: referenceAudioCount,
              step: i, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
              tiledDiffusion: tiledDiffusion,
              controlNets: &controlNets)
            if let blur = blur {
              let etCondDegraded = blur(inputs: etCond)[0].as(of: FloatType.self)
              etCond = Functional.add(
                left: etCondDegraded, right: etCond, leftScalar: alpha, rightScalar: 1 - alpha)
            }
            et = etUncond + textGuidanceVector .* (etCond - etUncond)
          } else {
            if let blur = blur {
              let etCondDegraded = blur(inputs: etCond)[0].as(of: FloatType.self)
              etCond = Functional.add(
                left: etCondDegraded, right: etCond, leftScalar: alpha, rightScalar: 1 - alpha)
            }
            et = etCond
          }
        } else {
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = x
          if isCfgEnabled {
            xIn[batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels] = x
            if modifier == .editing {
              xIn[
                (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels] =
                x
            }
          }
          let injectedIPAdapters = ControlModel<FloatType>
            .injectedIPAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let etOut = unet(
            timestep: (now: cNoise, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
            inputs: xIn, t, conditions,
            extraProjection: extraProjection,
            injectedControlsAndAdapters: injectedControlsAndAdapters,
            injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
            referenceAudioCount: referenceAudioCount,
            step: i, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
            tiledDiffusion: tiledDiffusion,
            controlNets: &controlNets)
          let alpha =
            0.001 * sharpness * (discretization.timesteps - timestep)
            / discretization.timesteps
          et = applyCfg(
            etOut: etOut, blur: blur, batchSize: batchSize, startHeight: startHeight,
            startWidth: startWidth, channels: channels, isCfgEnabled: isCfgEnabled,
            textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
            alpha: alpha, modifier: modifier, step: i, isBatchEnabled: isBatchEnabled,
            cfgZeroStar: cfgZeroStar)
        }
        let alpha = alphasCumprod[i]
        let alphaPrev = alphasCumprod[i + 1]
        switch discretization.objective {
        case .u(_):
          break
        case .v:
          et = Float(alpha.squareRoot()) * et + Float((1 - alpha).squareRoot()) * x
        case .epsilon:
          break
        case .edm(_):
          break
        }
        let etPrime: DynamicGraph.Tensor<FloatType>
        switch oldEps.count {
        case 0:
          let (xPrev, _) = xPrevAndPredX0(x: x, et: et, alpha: alpha, alphaPrev: alphaPrev)
          let timestepNext = discretization.timestep(for: alphasCumprod[i + 1])
          let tNext = unet.timeEmbed(
            graph: graph, batchSize: cfgChannels * batchSize, timestep: cNoiseNext,
            version: currentModelVersion)
          var etNext: DynamicGraph.Tensor<FloatType>
          if version == .svdI2v, let textGuidanceVector = textGuidanceVector,
            let condAugFrames = condAugFrames
          {
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = xPrev
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] =
              condAugFrames
            let injectedIPAdapters = ControlModel<FloatType>.injectedIPAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let injectedControlsAndAdapters = ControlModel<FloatType>.injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let cCond = Array(conditions[0..<(1 + (conditions.count - 1) / 2)])
            var etNextCond = unet(
              timestep: (now: cNoiseNext, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
              inputs: xIn, tNext, cCond,
              extraProjection: extraProjection,
              injectedControlsAndAdapters: injectedControlsAndAdapters,
              injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
              referenceAudioCount: referenceAudioCount,
              step: i, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
              tiledDiffusion: tiledDiffusion,
              controlNets: &controlNets)
            let alpha =
              0.001 * sharpness * (discretization.timesteps - timestepNext)
              / discretization.timesteps
            if isCfgEnabled {
              xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)].full(0)
              let cUncond = Array([conditions[0]] + conditions[(1 + (conditions.count - 1) / 2)...])
              let etNextUncond = unet(
                timestep: (now: cNoiseNext, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
                inputs: xIn, tNext, cUncond,
                extraProjection: extraProjection,
                injectedControlsAndAdapters: injectedControlsAndAdapters,
                injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
                referenceAudioCount: referenceAudioCount,
                step: i,
                tokenLengthUncond: tokenLengthUncond,
                tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
                tiledDiffusion: tiledDiffusion,
                controlNets: &controlNets)
              if let blur = blur {
                let etNextCondDegraded = blur(inputs: etNextCond)[0].as(of: FloatType.self)
                etNextCond = Functional.add(
                  left: etNextCondDegraded, right: etNextCond, leftScalar: alpha,
                  rightScalar: 1 - alpha)
              }
              etNext = etNextUncond + textGuidanceVector .* (etNextCond - etNextUncond)
            } else {
              if let blur = blur {
                let etNextCondDegraded = blur(inputs: etNextCond)[0].as(of: FloatType.self)
                etNextCond = Functional.add(
                  left: etNextCondDegraded, right: etNextCond, leftScalar: alpha,
                  rightScalar: 1 - alpha)
              }
              etNext = etNextCond
            }
          } else {
            // Compute etNext.
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = xPrev
            if isCfgEnabled {
              xIn[batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels] =
                xPrev
            }
            let injectedIPAdapters = ControlModel<FloatType>.injectedIPAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let injectedControlsAndAdapters = ControlModel<FloatType>.injectedControlsAndAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let etNextOut = unet(
              timestep: (now: cNoiseNext, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
              inputs: xIn, tNext, conditions,
              extraProjection: extraProjection,
              injectedControlsAndAdapters: injectedControlsAndAdapters,
              injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
              referenceAudioCount: referenceAudioCount,
              step: i, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
              tiledDiffusion: tiledDiffusion,
              controlNets: &controlNets)
            let alpha =
              0.001 * sharpness * (discretization.timesteps - timestepNext)
              / discretization.timesteps
            etNext = applyCfg(
              etOut: etNextOut, blur: blur, batchSize: batchSize, startHeight: startHeight,
              startWidth: startWidth, channels: channels, isCfgEnabled: isCfgEnabled,
              textGuidanceScale: textGuidanceScale, imageGuidanceScale: imageGuidanceScale,
              alpha: alpha, modifier: modifier, step: i, isBatchEnabled: isBatchEnabled,
              cfgZeroStar: cfgZeroStar)
          }
          switch discretization.objective {
          case .u(_):
            break
          case .v:
            etNext =
              Float(alphaPrev.squareRoot()) * etNext + Float((1 - alphaPrev).squareRoot()) * xPrev
          case .epsilon:
            break
          case .edm(_):
            break
          }
          etPrime = 0.5 * (et + etNext)
        case 1:
          etPrime = 0.5 * (3 * et - oldEps[0])
        case 2:
          etPrime =
            Float(1) / Float(12) * (Float(23) * et - Float(16) * oldEps[1] + Float(5) * oldEps[0])
        case 3:
          etPrime =
            Float(1) / Float(24)
            * (Float(55) * et - Float(59) * oldEps[2] + Float(37) * oldEps[1] - Float(9) * oldEps[0])
        default:
          fatalError()
        }
        let (xPrev, _) = xPrevAndPredX0(x: x, et: etPrime, alpha: alpha, alphaPrev: alphaPrev)
        x = xPrev
        oldEps.append(et)
        if oldEps.count > 3 {
          oldEps.removeFirst()
        }
        if i < endStep - 1, let noise = noise, let sample = sample, let mask = mask,
          let negMask = negMask
        {
          noise.randn(std: 1, mean: 0)
          let qSample =
            Float(alphaPrev.squareRoot()) * sample + Float((1 - alphaPrev).squareRoot()) * noise
          x = qSample .* negMask + x .* mask
        }
        if i == endStep - 1 {
          if isNaN(x.rawValue.toCPU()) {
            return .failure(SamplerError.isNaN)
          }
        }
      }
      return .success(SamplerOutput(x: x, unets: unets))
    }
    streamContext.joined()
    return result
  }

  public func timestep(for strength: Float, sampling: Sampling) -> (
    timestep: Float, startStep: Float, roundedDownStartStep: Int, roundedUpStartStep: Int
  ) {
    let tEnc = Int(strength * Float(sampling.steps))
    let initTimestep =
      discretization.timesteps - discretization.timesteps
      / Float(sampling.steps * (sampling.steps - tEnc)) + 1
    let startStep = sampling.steps - tEnc
    return (
      timestep: initTimestep, startStep: Float(startStep), roundedDownStartStep: startStep,
      roundedUpStartStep: startStep
    )
  }

  public func sampleScaleFactor(at step: Float, sampling: Sampling) -> Float {
    let step = Int(step.rounded())
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let alphaCumprod = alphasCumprod[max(0, min(alphasCumprod.count - 1, step))]
    return Float(alphaCumprod.squareRoot())
  }

  public func noiseScaleFactor(at step: Float, sampling: Sampling) -> Float {
    let step = Int(step.rounded())
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let alphaCumprod = alphasCumprod[max(0, min(alphasCumprod.count - 1, step))]
    return Float((1 - alphaCumprod).squareRoot())
  }
}
