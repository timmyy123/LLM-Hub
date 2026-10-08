import Foundation
import NNC
import WeightsCache

public struct DPMPPSDESampler<
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

extension DPMPPSDESampler: Sampler {
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
      var allConditions: [DynamicGraph.AnyTensor] = c
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
      let sigmas = alphasCumprod.map { discretization.sigma(from: $0) }
      // Interleave each step and its midpoint for fixed encoders and controls.
      let timesteps = (startStep.integral..<endStep.integral).flatMap { i -> [Float] in
        let alphaCumprod: Double
        if i == startStep.integral && Float(startStep.integral) != startStep.fractional {
          let lowTimestep = discretization.timestep(
            for: alphasCumprod[max(0, min(Int(startStep.integral), alphasCumprod.count - 1))])
          let highTimestep = discretization.timestep(
            for: alphasCumprod[
              max(0, min(Int(startStep.fractional.rounded(.up)), alphasCumprod.count - 1))])
          let timestep =
            lowTimestep
            + Float(highTimestep - lowTimestep) * (startStep.fractional - Float(startStep.integral))
          alphaCumprod = discretization.alphaCumprod(timestep: timestep, shift: 1)
        } else {
          alphaCumprod = discretization.alphaCumprod(from: sigmas[i])
        }
        let sigma = discretization.sigma(from: alphaCumprod)
        let sigmaS = sigmas[i + 1] > 0 ? (sigma * sigmas[i + 1]).squareRoot() : sigma
        return [alphaCumprod, discretization.alphaCumprod(from: sigmaS)].map {
          switch conditioning {
          case .noise:
            return discretization.noise(for: $0)
          case .timestep:
            return discretization.timestep(for: $0)
          }
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
          startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond, lora: lora, tiledDiffusion: tiledDiffusion,
          teaCache: teaCache, isBF16: isBF16, injectedControls: injectedControls,
          referenceImages: referenceImages, referenceAudios: referenceAudios)
        allConditions = vector + encodings
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
          newC = Array(allConditions[0..<(1 + (allConditions.count - 1) / 2)])
        } else {
          newC = allConditions
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
      let noise = graph.variable(
        .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
      var brownianNoise = graph.variable(
        .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
      brownianNoise.randn(std: 1, mean: 0)
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
      var indexOffset = startStep.integral
      // Now do DPM++ SDE Karras sampling.
      if startStep.fractional == 0 && sigmas[0] != 1 {
        x = sampleScaledAdd(x, x, sigma: (now: Float(sigmas[0]), next: 0), version: version) {
          sigma in (sigma.now, 0)
        }
      }
      var oldDenoised: DynamicGraph.Tensor<FloatType>? = nil
      for i in startStep.integral..<endStep.integral {
        let sigma: Double
        if i == startStep.integral && Float(startStep.integral) != startStep.fractional {
          let lowTimestep = discretization.timestep(
            for: alphasCumprod[max(0, min(Int(startStep.integral), alphasCumprod.count - 1))])
          let highTimestep = discretization.timestep(
            for: alphasCumprod[
              max(0, min(Int(startStep.fractional.rounded(.up)), alphasCumprod.count - 1))])
          let timestep =
            lowTimestep
            + Float(highTimestep - lowTimestep) * (startStep.fractional - Float(startStep.integral))
          let alphaCumprod = discretization.alphaCumprod(timestep: timestep, shift: 1)
          sigma = discretization.sigma(from: alphaCumprod)
        } else {
          sigma = sigmas[i]
        }
        if i == startStep.integral {
          brownianNoise = sampleScaledAdd(
            brownianNoise, brownianNoise, sigma: (now: Float(sigma), next: 0), version: version
          ) { sigma in (sigma.now.squareRoot(), 0) }
        }
        let alphaCumprod = discretization.alphaCumprod(from: sigma)
        let sqrtAlphaCumprod = alphaCumprod.squareRoot()
        let input: DynamicGraph.Tensor<FloatType>
        switch discretization.objective {
        case .u(_):
          input = x
        case .v, .epsilon:
          input = Float(sqrtAlphaCumprod) * x
        case .edm(let sigmaData):
          input = Float(1.0 / (sigma * sigma + sigmaData * sigmaData).squareRoot()) * x
        }
        let rawValue: Tensor<FloatType>? =
          (i > max(startStep.integral, sampling.steps / 2) || i % 2 == 1)
          ? (oldDenoised.map { unet.decode($0) })?.rawValue.toCPU() : nil
        if i % 5 == 4, let rawValue = rawValue {
          if isNaN(rawValue) {
            return .failure(SamplerError.isNaN)
          }
        }
        guard feedback(i - startStep.integral, rawValue) else {
          return .failure(SamplerError.cancelled(unets))
        }
        let timestep = discretization.timestep(for: alphaCumprod)
        if timestep < refinerKickIn, let refiner = refiner {
          let refinerTimesteps = Array(timesteps[(2 * (i - startStep.integral))...])
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
            allConditions = []
            let vector = fixedEncoder.vector(
              textEmbedding: oldC[oldC.count - 1], originalSize: originalSize,
              cropTopLeft: cropTopLeft,
              targetSize: targetSize, aestheticScore: aestheticScore,
              negativeOriginalSize: negativeOriginalSize,
              negativeAestheticScore: negativeAestheticScore, fpsId: fpsId,
              motionBucketId: motionBucketId, condAug: condAug)
            allConditions =
              vector
              + fixedEncoder.encode(
                isCfgEnabled: isCfgEnabled, textGuidanceScale: textGuidanceScale,
                guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
                distilledGuidanceLayers: refiner.distilledGuidanceLayers, modifier: modifier,
                textEncoding: oldC, timesteps: refinerTimesteps, audioShiftRatio: audioShiftRatio,
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
            newC = Array(allConditions[0..<(1 + (allConditions.count - 1) / 2)])
          } else {
            newC = allConditions
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
        let nextAlphaCumprod = discretization.alphaCumprod(from: sigmas[i + 1])
        let cNoise: Float
        let cNoiseNext: Float
        switch conditioning {
        case .noise:
          cNoise = discretization.noise(for: alphaCumprod)
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
          graph: graph, index: 2 * (i - indexOffset), batchSize: cfgChannels * batchSize,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          conditions: allConditions, referenceImageCount: referenceImageCount,
          referenceAudioCount: referenceAudioCount,
          version: currentModelVersion, modifier: modifier, isCfgEnabled: isCfgEnabled)
        var et: DynamicGraph.Tensor<FloatType>
        if version == .svdI2v, let textGuidanceVector = textGuidanceVector,
          let condAugFrames = condAugFrames
        {
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = input
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] =
            condAugFrames
          let injectedIPAdapters = ControlModel<FloatType>
            .injectedIPAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: 2 * (i - startStep.integral),
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: 2 * (i - startStep.integral),
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
          xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = input
          if isCfgEnabled {
            xIn[batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels] = input
            if xIn.shape[0] >= batchSize * 3 {
              xIn[
                (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels] =
                input
            }
          }
          let injectedIPAdapters = ControlModel<FloatType>
            .injectedIPAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: 2 * (i - startStep.integral),
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: 2 * (i - startStep.integral),
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
        let stepSigma = (now: Float(sigma), next: Float(sigmas[i + 1]))
        if i < sampling.steps - 1 && sigma > sigmas[i + 1] && sigmas[i + 1] > 0 {
          var denoised: DynamicGraph.Tensor<FloatType>
          switch discretization.objective {
          case .u(_):
            denoised = sampleScaledAdd(x, et, sigma: stepSigma, version: currentModelVersion) {
              sigma in
              // The UNet expresses velocity per video-sigma step.
              let velocityScale = (stepSigma.now - stepSigma.next) / (sigma.now - sigma.next)
              return (1, -sigma.now * velocityScale)
            }
          case .v:
            denoised = Functional.add(
              left: x, right: et, leftScalar: Float(1.0 / (sigma * sigma + 1)),
              rightScalar: Float(-sigma * sqrtAlphaCumprod))
          case .epsilon:
            denoised = Functional.add(left: x, right: et, leftScalar: 1, rightScalar: Float(-sigma))
            if version == .kandinsky21 {
              denoised = clipDenoised(denoised)
            }
          case .edm(let sigmaData):
            let sigmaData2 = sigmaData * sigmaData
            denoised = Functional.add(
              left: x, right: et, leftScalar: Float(sigmaData2 / (sigma * sigma + sigmaData2)),
              rightScalar: Float(sigma * sigmaData / (sigma * sigma + sigmaData2).squareRoot()))
          }
          let sigmaS = (sigma * sigmas[i + 1]).squareRoot()  // exp(log(sigma) - h / 2) == exp(log(sigma) / 2 + log(sigmas[i + 1]) / 2) == sqrt(exp(log(sigma) + log(sigma[i + 1]))
          let firstSigma = (now: Float(sigma), next: Float(sigmaS))
          var x2 = sampleScaledAdd(x, denoised, sigma: firstSigma, version: currentModelVersion) {
            sigma in
            // With eta = 1, sigmaDown = next^2 / now and w = sigmaDown / now.
            let ratio = Double(sigma.next) / Double(sigma.now)
            let w = ratio * ratio
            if case .u(_) = discretization.objective {
              return (Float(w), Float(1 - w + Double(sigma.next) * (ratio - 1)))
            } else {
              return (Float(w), Float(1 - w))
            }
          }
          // Now do brownian sampling to sigma -> sigmaS (right), needs to compute sigmaS -> 0 (left).
          // Formulation borrowed from: https://github.com/google-research/torchsde/blob/master/torchsde/_brownian/brownian_interval.py#L181
          // Because we do brownian sampling, meaning there is a dependency between this observation and the next one.
          // We need to keep leftW and leftW2 in memory (leftW2 in next round) as we split the observations further.
          noise.randn(std: 1, mean: 0)
          let leftW = sampleScaledAdd(
            brownianNoise, noise, sigma: firstSigma, version: currentModelVersion
          ) { sigma in
            let ratio = Double(sigma.next) / Double(sigma.now)
            return (
              Float(ratio), Float(((Double(sigma.now) - Double(sigma.next)) * ratio).squareRoot())
            )
          }
          let rightW = brownianNoise - leftW
          x2 = sampleScaledAdd(x2, rightW, sigma: firstSigma, version: currentModelVersion) {
            sigma in
            // sigmaUp / sqrt(now - next) = (next / now) * sqrt(now + next).
            (
              1,
              Float(
                Double(sigma.next) / Double(sigma.now)
                  * (Double(sigma.now) + Double(sigma.next)).squareRoot())
            )
          }
          // Now run the model again.
          let alphaSCumprod = discretization.alphaCumprod(from: sigmaS)
          let sqrtAlphaSCumprod = alphaSCumprod.squareRoot()
          let input: DynamicGraph.Tensor<FloatType>
          switch discretization.objective {
          case .u(_):
            input = x2
          case .v, .epsilon:
            input = Float(sqrtAlphaSCumprod) * x2
          case .edm(let sigmaData):
            input = Float(1.0 / (sigma * sigma + sigmaData * sigmaData).squareRoot()) * x2
          }
          var et: DynamicGraph.Tensor<FloatType>
          let timestep: Float
          switch conditioning {
          case .noise:
            timestep = discretization.noise(for: alphaSCumprod)
          case .timestep:
            timestep = discretization.timestep(for: alphaSCumprod)
          }
          let t = unet.timeEmbed(
            graph: graph, batchSize: cfgChannels * batchSize, timestep: timestep,
            version: currentModelVersion)
          let conditions = UNetExtractConditions(
            of: FloatType.self, graph: graph, index: 2 * (i - indexOffset) + 1,
            batchSize: cfgChannels * batchSize,
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            conditions: allConditions, referenceImageCount: referenceImageCount,
            referenceAudioCount: referenceAudioCount,
            version: currentModelVersion, modifier: modifier, isCfgEnabled: isCfgEnabled)
          if version == .svdI2v, let textGuidanceVector = textGuidanceVector,
            let condAugFrames = condAugFrames
          {
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = input
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] =
              condAugFrames
            let injectedIPAdapters = ControlModel<FloatType>.injectedIPAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: 2 * (i - startStep.integral) + 1,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let injectedControlsAndAdapters = ControlModel<FloatType>.injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: 2 * (i - startStep.integral) + 1,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let cCond = Array(conditions[0..<(1 + (conditions.count - 1) / 2)])
            var etCond = unet(
              timestep: (now: timestep, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
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
                timestep: (now: timestep, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
                inputs: xIn, t, cUncond,
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
            xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = input
            if isCfgEnabled {
              xIn[batchSize..<(batchSize * 2), 0..<startHeight, 0..<startWidth, 0..<channels] =
                input
              if xIn.shape[0] >= batchSize * 3 {
                xIn[
                  (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels] =
                  input
              }
            }
            let injectedIPAdapters = ControlModel<
              FloatType
            >
            .injectedIPAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: 2 * (i - startStep.integral) + 1,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let injectedControlsAndAdapters = ControlModel<
              FloatType
            >
            .injectedControlsAndAdapters(
              injecteds: injecteds, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: 2 * (i - startStep.integral) + 1,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
            let etOut = unet(
              timestep: (now: timestep, next: cNoiseNext), audioShiftRatio: audioShiftRatio,
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
          var denoised2: DynamicGraph.Tensor<FloatType>
          let secondSigma = (now: Float(sigmaS), next: Float(sigmas[i + 1]))
          switch discretization.objective {
          case .u(_):
            denoised2 = sampleScaledAdd(x2, et, sigma: secondSigma, version: currentModelVersion) {
              sigma in
              let velocityScale = (secondSigma.now - secondSigma.next) / (sigma.now - sigma.next)
              return (1, -sigma.now * velocityScale)
            }
          case .v:
            denoised2 = Functional.add(
              left: x2, right: et, leftScalar: Float(1.0 / (sigmaS * sigmaS + 1)),
              rightScalar: Float(-sigmaS * sqrtAlphaSCumprod))
          case .epsilon:
            denoised2 = Functional.add(
              left: x2, right: et, leftScalar: 1, rightScalar: Float(-sigmaS))
            if version == .kandinsky21 {
              denoised2 = clipDenoised(denoised2)
            }
          case .edm(let sigmaData):
            let sigmaData2 = sigmaData * sigmaData
            denoised2 = Functional.add(
              left: x2, right: et, leftScalar: Float(sigmaData2 / (sigmaS * sigmaS + sigmaData2)),
              rightScalar: Float(sigmaS * sigmaData / (sigmaS * sigmaS + sigmaData2).squareRoot()))
          }
          var denoisedD = denoised2
          if case .u(_) = discretization.objective {
            // The mapped midpoint need not bisect audio's log-sigma interval.
            // denoisedD = denoised + (h / (2 * hMid)) * (denoised2 - denoised).
            let derivative = sampleScaledAdd(
              denoised2, denoised, sigma: firstSigma, version: currentModelVersion
            ) { sigma in
              let hMid = log(Double(sigma.now) / Double(sigma.next))
              return (Float(1 / hMid), Float(-1 / hMid))
            }
            denoisedD = sampleScaledAdd(
              denoised, derivative, sigma: stepSigma, version: currentModelVersion
            ) { sigma in (1, Float(log(Double(sigma.now) / Double(sigma.next)) / 2)) }
          }
          x = sampleScaledAdd(x, denoisedD, sigma: stepSigma, version: currentModelVersion) {
            sigma in
            let ratio = Double(sigma.next) / Double(sigma.now)
            let w = ratio * ratio
            if case .u(_) = discretization.objective {
              return (Float(w), Float(1 - w + Double(sigma.next) * (ratio - 1)))
            } else {
              return (Float(w), Float(1 - w))
            }
          }
          noise.randn(std: 1, mean: 0)
          let leftW2 = sampleScaledAdd(
            leftW, noise, sigma: secondSigma, version: currentModelVersion
          ) {
            sigma in
            let ratio = Double(sigma.next) / Double(sigma.now)
            return (
              Float(ratio), Float(((Double(sigma.now) - Double(sigma.next)) * ratio).squareRoot())
            )
          }
          let rightW2 = leftW - leftW2 + rightW
          x = sampleScaledAdd(x, rightW2, sigma: stepSigma, version: currentModelVersion) { sigma in
            (
              1,
              Float(
                Double(sigma.next) / Double(sigma.now)
                  * (Double(sigma.now) + Double(sigma.next)).squareRoot())
            )
          }
          brownianNoise = leftW2  // On next round, this is the only thing we care.
          oldDenoised = denoised
        } else {
          let dt = sigmas[i + 1] - sigma
          switch discretization.objective {
          case .u(_):
            x = Functional.add(left: x, right: et, leftScalar: 1, rightScalar: Float(dt))
          case .v:
            // denoised = Float(1.0 / (sigma * sigma + 1)) * x - (sigma * sqrtAlphaCumprod) * et
            // d = (x - denoised) / sigma // (x - Float(1.0 / (sigma * sigma + 1)) * x + (sigma * sqrtAlphaCumprod) * et) / sigma = (sigma / (sigma * sigma + 1)) * x + sqrtAlphaCumprod * et
            let d = Functional.add(
              left: x, right: et, leftScalar: Float(sigma / (sigma * sigma + 1)),
              rightScalar: Float(sqrtAlphaCumprod))
            x = Functional.add(left: x, right: d, leftScalar: 1, rightScalar: Float(dt))
          case .epsilon:
            // denoised = x - sigma * et
            // d = (x - denoised) / sigma // (x - x + sigma * et) / sigma = et
            x = Functional.add(left: x, right: et, leftScalar: 1, rightScalar: Float(dt))
          case .edm(let sigmaData):
            let sigmaData2 = sigmaData * sigmaData
            // denoised = sigmaData2 / (sigma * sigma + sigmaData2) * x + (sigma * sigmaData / (sigma * sigma + sigmaData2).squareRoot()) * et
            // d = (x - denoised) / sigma // (x - sigmaData2 / (sigma * sigma + sigmaData2) * x - (sigma * sigmaData / (sigma * sigma + sigmaData2).squareRoot()) * et) / sigma
            let d = Functional.add(
              left: x, right: et, leftScalar: Float(sigma / (sigma * sigma + sigmaData2)),
              rightScalar: Float(-sigmaData / (sigma * sigma + sigmaData2).squareRoot()))
            x = Functional.add(left: x, right: d, leftScalar: 1, rightScalar: Float(dt))
          }
        }
        if i < endStep.integral - 1, let sample = sample, let mask = mask, let negMask = negMask {
          // If you check how we compute sigma, this is basically how we get back to alphaCumprod.
          // alphaPrev = 1 / (sigmas[i + 1] * sigmas[i + 1] + 1)
          // Then, we should compute qSample as alphaPrev.squareRoot() * sample + (1 - alphaPrev).squareRoot() * noise
          // However, because we will multiple back 1 / alphaPrev.squareRoot() again, this effectively become the following.
          noise.randn(std: 1, mean: 0)
          let qSample: DynamicGraph.Tensor<FloatType>
          if case .u(_) = discretization.objective {
            qSample = sampleAdd(
              sample, noise: noise,
              scale: (sample: Float(1 - sigmas[i + 1]), noise: Float(sigmas[i + 1])),
              version: currentModelVersion)
          } else {
            qSample = sample + Float(sigmas[i + 1]) * noise
          }
          x = qSample .* negMask + x .* mask
        }
        if i == endStep.integral - 1 {
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
    let tEnc = strength * discretization.timesteps
    let initTimestep = tEnc
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    var previousTimestep = discretization.timesteps
    for (i, alphaCumprod) in alphasCumprod.enumerated() {
      let timestep = discretization.timestep(for: alphaCumprod)
      if initTimestep >= timestep {
        guard i > 0 else {
          return (
            timestep: timestep, startStep: 0, roundedDownStartStep: 0, roundedUpStartStep: 0
          )
        }
        guard initTimestep > timestep + 1e-3 else {
          return (
            timestep: initTimestep, startStep: Float(i), roundedDownStartStep: i,
            roundedUpStartStep: i
          )
        }
        return (
          timestep: Float(initTimestep),
          startStep: Float(i - 1) + Float(initTimestep - previousTimestep)
            / Float(timestep - previousTimestep), roundedDownStartStep: i - 1, roundedUpStartStep: i
        )
      }
      previousTimestep = timestep
    }
    return (
      timestep: discretization.timestep(for: alphasCumprod[0]),
      startStep: Float(alphasCumprod.count - 1),
      roundedDownStartStep: alphasCumprod.count - 1, roundedUpStartStep: alphasCumprod.count - 1
    )
  }

  public func sampleScaleFactor(at step: Float, sampling: Sampling) -> Float {
    if case .u(_) = discretization.objective {
      return 1 - noiseScaleFactor(at: step, sampling: sampling)
    } else {
      return 1
    }
  }

  public func noiseScaleFactor(at step: Float, sampling: Sampling) -> Float {
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let lowTimestep = discretization.timestep(
      for: alphasCumprod[max(0, min(Int(step.rounded(.down)), alphasCumprod.count - 1))])
    let highTimestep = discretization.timestep(
      for: alphasCumprod[max(0, min(Int(step.rounded(.up)), alphasCumprod.count - 1))])
    let timestep = lowTimestep + (highTimestep - lowTimestep) * (step - Float(step.rounded(.down)))
    let alphaCumprod = discretization.alphaCumprod(timestep: timestep, shift: 1)
    let sigma = discretization.sigma(from: alphaCumprod)
    return Float(sigma)
  }
}
