import Foundation
import NNC
import WeightsCache

public struct UniPCSampler<
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

extension UniPCSampler: Sampler {
  private func uniPBhUpdate(
    prevTimestep t: Int,
    sample x: DynamicGraph.Tensor<Float>, timestepList: [Int],
    outputList: [DynamicGraph.Tensor<Float>], lambdas: [Double], alphas: [Double],
    sigmas: [Double], version: ModelVersion
  ) -> DynamicGraph.Tensor<Float> {
    let s0 = timestepList[timestepList.count - 1]
    let m0 = outputList[outputList.count - 1]
    let lambdat = lambdas[t]
    let lambdas0 = lambdas[s0]
    let alphat = alphas[t]
    let sigmat = sigmas[t]
    let sigmas0 = sigmas[s0]
    if sigmat == 0 { return m0 }
    let h = lambdat - lambdas0
    guard h > 0 else { return x }
    let D1: DynamicGraph.Tensor<Float>?
    if timestepList.count >= 2 && outputList.count >= 2 {
      let si = timestepList[timestepList.count - 2]
      let mi = outputList[outputList.count - 2]
      let lambdasi = lambdas[si]
      let rk = (lambdasi - lambdas0) / h
      D1 = (mi - m0) * Float(1 / rk)
    } else {
      D1 = nil
    }
    let hh = -h
    let hPhi1 = exp(hh) - 1
    let Bh = hPhi1
    let rhosP = 0.5
    let xt_: DynamicGraph.Tensor<Float>
    if case .u(_) = discretization.objective {
      xt_ = sampleScaledAdd(
        x, m0, sigma: (now: Float(sigmas0), next: Float(sigmat)), version: version
      ) { sigma in
        let w = Double(sigma.next) / Double(sigma.now)
        return (Float(w), Float(1 - w))
      }
    } else {
      xt_ = Functional.add(
        left: x, right: m0, leftScalar: Float(sigmat / sigmas0), rightScalar: Float(-alphat * hPhi1)
      )
    }
    if let D1 = D1 {
      if case .u(_) = discretization.objective {
        return sampleScaledAdd(
          xt_, D1, sigma: (now: Float(sigmas0), next: Float(sigmat)), version: version
        ) { sigma in (1, Float((1 - Double(sigma.next) / Double(sigma.now)) * rhosP)) }
      }
      let xt = Functional.add(
        left: xt_, right: D1, leftScalar: 1, rightScalar: Float(-alphat * Bh * rhosP))
      return xt
    } else {
      return xt_
    }
  }
  private func uniCBhUpdate(
    predX0 mt: DynamicGraph.Tensor<Float>, timestep t: Int,
    lastSample x: DynamicGraph.Tensor<Float>, timestepList: [Int],
    outputList: [DynamicGraph.Tensor<Float>], lambdas: [Double], alphas: [Double],
    sigmas: [Double], version: ModelVersion
  ) -> DynamicGraph.Tensor<Float> {
    let s0 = timestepList[timestepList.count - 1]
    let m0 = outputList[outputList.count - 1]
    let lambdat = lambdas[t]
    let lambdas0 = lambdas[s0]
    let alphat = alphas[t]
    let sigmat = sigmas[t]
    let sigmas0 = sigmas[s0]
    if sigmat == 0 { return mt }
    let h = lambdat - lambdas0
    guard h > 0 else { return x }
    let hh = -h
    let hPhi1 = exp(hh) - 1
    let hPhik = hPhi1 / hh - 1
    let Bh = hPhi1
    let D1: DynamicGraph.Tensor<Float>?
    let rhosC0: Double
    let rhosC1: Double
    if timestepList.count >= 2 && outputList.count >= 2 {
      let si = timestepList[timestepList.count - 2]
      let mi = outputList[outputList.count - 2]
      let lambdasi = lambdas[si]
      let rk = (lambdasi - lambdas0) / h
      D1 = (mi - m0) * Float(1 / rk)
      let b0 = hPhik / Bh
      let b1 = (hPhik / hh - 0.5) * 2 / Bh
      rhosC0 = (b0 - b1) / (1 - rk)
      rhosC1 = b0 - rhosC0
    } else {
      D1 = nil
      rhosC0 = 0.5
      rhosC1 = 0.5
    }
    let xt_: DynamicGraph.Tensor<Float>
    if case .u(_) = discretization.objective {
      xt_ = sampleScaledAdd(
        x, m0, sigma: (now: Float(sigmas0), next: Float(sigmat)), version: version
      ) { sigma in
        let w = Double(sigma.next) / Double(sigma.now)
        return (Float(w), Float(1 - w))
      }
    } else {
      xt_ = Functional.add(
        left: x, right: m0, leftScalar: Float(sigmat / sigmas0), rightScalar: Float(-alphat * hPhi1)
      )
    }
    let D1t = mt - m0
    let D1s: DynamicGraph.Tensor<Float>
    if let D1 = D1 {
      D1s = Functional.add(
        left: D1, right: D1t, leftScalar: Float(rhosC0), rightScalar: Float(rhosC1))
    } else {
      D1s = Float(rhosC1) * D1t
    }
    if case .u(_) = discretization.objective {
      return sampleScaledAdd(
        xt_, D1s, sigma: (now: Float(sigmas0), next: Float(sigmat)), version: version
      ) { sigma in (1, Float(1 - Double(sigma.next) / Double(sigma.now))) }
    }
    let xt = Functional.add(left: xt_, right: D1s, leftScalar: 1, rightScalar: Float(-alphat * Bh))
    return xt
  }
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
      var alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
      if Float(startStep.integral) != startStep.fractional {
        let lowTimestep = discretization.timestep(
          for: alphasCumprod[max(0, min(Int(startStep.integral), alphasCumprod.count - 1))])
        let highTimestep = discretization.timestep(
          for: alphasCumprod[
            max(0, min(Int(startStep.fractional.rounded(.up)), alphasCumprod.count - 1))])
        let timestep =
          lowTimestep
          + Float(highTimestep - lowTimestep) * (startStep.fractional - Float(startStep.integral))
        alphasCumprod[startStep.integral] = discretization.alphaCumprod(
          timestep: timestep, shift: 1)
      }
      let timesteps = (startStep.integral..<endStep.integral).map {
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
      let alphas: [Double]
      let sigmas: [Double]
      switch discretization.objective {
      case .u(_):
        alphas = alphasCumprod.map { $0 }
        sigmas = alphasCumprod.map { 1 - $0 }
      case .edm(_), .v, .epsilon:
        alphas = alphasCumprod.map { $0.squareRoot() }
        sigmas = alphasCumprod.map { (1 - $0).squareRoot() }
      }
      // H3's flow shift adds a constant to log-SNR. Differences (h, rk, rhos)
      // are therefore shared; only the sample and clean-prediction coefficients differ.
      let lambdas = zip(alphas, sigmas).map { log($0) - log($1) }
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
      var refinerKickIn = refiner.map { (1 - $0.start) * discretization.timesteps } ?? -1
      var unets: [UNet?] = [unet]
      var currentModelVersion = version
      var indexOffset = startStep.integral
      var timestepList = [Int]()
      var outputList = [DynamicGraph.Tensor<Float>]()
      var lastSample: DynamicGraph.Tensor<Float>? = nil
      for i in startStep.integral..<endStep.integral {
        let rawValue: Tensor<FloatType>? =
          (i > max(startStep.integral, sampling.steps / 2) || i % 2 == 1)
          ? outputList.last.map({ DynamicGraph.Tensor<FloatType>(from: $0) })?.rawValue.toCPU()
          : nil
        if i % 5 == 4, let rawValue = rawValue {
          if isNaN(rawValue) {
            return .failure(SamplerError.isNaN)
          }
        }
        guard feedback(i - startStep.integral, rawValue) else {
          return .failure(SamplerError.cancelled(unets))
        }
        let timestep = discretization.timestep(for: alphasCumprod[i])
        if timestep < refinerKickIn, let refiner = refiner {
          let timesteps = (i..<endStep.integral).map {
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
        let et: DynamicGraph.Tensor<FloatType>
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
              isCfgEnabled: isCfgEnabled, index: i - startStep.integral,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep.integral,
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
            if xIn.shape[0] >= batchSize * 3 {
              xIn[
                (batchSize * 2)..<(batchSize * 3), 0..<startHeight, 0..<startWidth, 0..<channels] =
                x
            }
          }
          let injectedIPAdapters = ControlModel<FloatType>
            .injectedIPAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep.integral,
              mainUNetAndWeightMapper: unet.modelAndWeightMapper,
              controlNets: &controlNets)
          let injectedControlsAndAdapters = ControlModel<FloatType>
            .injectedControlsAndAdapters(
              injecteds: injectedControls, step: i, version: unet.version,
              usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
              tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
              isCfgEnabled: isCfgEnabled, index: i - startStep.integral,
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
        var predX0: DynamicGraph.Tensor<Float>
        var xF32 = DynamicGraph.Tensor<Float>(from: x)
        switch discretization.objective {
        case .u(_):
          let stepSigma = (now: Float(sigmas[i]), next: Float(sigmas[i + 1]))
          predX0 = sampleScaledAdd(
            xF32, DynamicGraph.Tensor<Float>(from: et), sigma: stepSigma,
            version: currentModelVersion
          ) { sigma in
            let delta = sigma.now - sigma.next
            let velocityScale = delta > 0 ? (stepSigma.now - stepSigma.next) / delta : 0
            return (1, -sigma.now * velocityScale)
          }
        case .v:
          predX0 = Functional.add(
            left: xF32, right: DynamicGraph.Tensor<Float>(from: et), leftScalar: Float(alphas[i]),
            rightScalar: Float(-sigmas[i]))
        case .epsilon:
          predX0 = Functional.add(
            left: xF32, right: DynamicGraph.Tensor<Float>(from: et),
            leftScalar: Float(1.0 / alphas[i]),
            rightScalar: Float(-sigmas[i] / alphas[i]))
          if version == .kandinsky21 {
            predX0 = clipDenoised(predX0)
          }
        case .edm(let sigmaData):
          let sigmaData2 = sigmaData * sigmaData
          let alphaCumprod = alphasCumprod[i]
          let sigma = ((1 - alphaCumprod) / alphaCumprod).squareRoot()
          predX0 = Functional.add(
            left: xF32, right: DynamicGraph.Tensor<Float>(from: et),
            leftScalar: Float(sigmaData2 / (sigma * sigma + sigmaData2).squareRoot()),
            rightScalar: Float(sigma * sigmaData / (sigma * sigma + sigmaData2).squareRoot()))
        }
        if let lastSample = lastSample {
          xF32 = uniCBhUpdate(
            predX0: predX0, timestep: i, lastSample: lastSample, timestepList: timestepList,
            outputList: outputList, lambdas: lambdas, alphas: alphas, sigmas: sigmas,
            version: currentModelVersion)
        }
        if timestepList.count < 2 {
          timestepList.append(i)
        } else {
          timestepList[0] = timestepList[1]
          timestepList[1] = i
        }
        if outputList.count < 2 {
          outputList.append(predX0)
        } else {
          outputList[0] = outputList[1]
          outputList[1] = predX0
        }
        let alphaPrev = alphasCumprod[i + 1]
        if i < sampling.steps - 1 {
          let prevTimestep = i + 1
          lastSample = xF32
          xF32 = uniPBhUpdate(
            prevTimestep: prevTimestep, sample: xF32, timestepList: timestepList,
            outputList: outputList, lambdas: lambdas, alphas: alphas, sigmas: sigmas,
            version: currentModelVersion)
        } else {
          xF32 = predX0
        }
        x = DynamicGraph.Tensor<FloatType>(from: xF32)
        if i < endStep.integral - 1, let noise = noise, let sample = sample, let mask = mask,
          let negMask = negMask
        {
          noise.randn(std: 1, mean: 0)
          let qSample: DynamicGraph.Tensor<FloatType>
          if case .u(_) = discretization.objective {
            qSample = sampleAdd(
              sample, noise: noise, scale: (sample: Float(alphaPrev), noise: Float(1 - alphaPrev)),
              version: currentModelVersion)
          } else {
            qSample =
              Float(alphaPrev.squareRoot()) * sample + Float((1 - alphaPrev).squareRoot()) * noise
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
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let lowTimestep = discretization.timestep(
      for: alphasCumprod[max(0, min(Int(step.rounded(.down)), alphasCumprod.count - 1))])
    let highTimestep = discretization.timestep(
      for: alphasCumprod[max(0, min(Int(step.rounded(.up)), alphasCumprod.count - 1))])
    let timestep = lowTimestep + (highTimestep - lowTimestep) * (step - Float(step.rounded(.down)))
    let alphaCumprod = discretization.alphaCumprod(timestep: timestep, shift: 1)
    if case .u(_) = discretization.objective {
      return Float(alphaCumprod)
    } else {
      return Float(alphaCumprod.squareRoot())
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
    if case .u(_) = discretization.objective {
      return Float(1 - alphaCumprod)
    } else {
      return Float((1 - alphaCumprod).squareRoot())
    }
  }
}
