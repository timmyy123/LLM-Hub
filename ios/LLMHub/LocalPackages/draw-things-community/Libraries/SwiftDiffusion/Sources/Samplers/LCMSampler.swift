import Foundation
import NNC
import WeightsCache

public struct LCMSampler<
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
  public let isGuidanceEmbedEnabled: Bool
  public let isQuantizedModel: Bool
  public let canRunLoRASeparately: Bool
  public let deviceProperties: DeviceProperties
  public let conditioning: Denoiser.Conditioning
  public let tiledDiffusion: TiledConfiguration
  public let teaCache: TeaCacheConfiguration
  public let causalInference: (Int, pad: Int)
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
    isGuidanceEmbedEnabled: Bool, isQuantizedModel: Bool, canRunLoRASeparately: Bool,
    deviceProperties: DeviceProperties,
    conditioning: Denoiser.Conditioning, tiledDiffusion: TiledConfiguration,
    teaCache: TeaCacheConfiguration, causalInference: (Int, pad: Int),
    isBF16: Bool, discretization: Discretization, weightsCache: WeightsCache
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
    self.isGuidanceEmbedEnabled = isGuidanceEmbedEnabled
    self.isQuantizedModel = isQuantizedModel
    self.canRunLoRASeparately = canRunLoRASeparately
    self.deviceProperties = deviceProperties
    self.conditioning = conditioning
    self.tiledDiffusion = tiledDiffusion
    self.teaCache = teaCache
    self.causalInference = causalInference
    self.isBF16 = isBF16
    self.discretization = discretization

    self.weightsCache = weightsCache
  }
}

extension LCMSampler: Sampler {
  private func scalingsForBoundaryConditionDiscrete(timestep: Int) -> (Float, Float) {
    // let sigmaData = 0.5
    let scaledTimestep = Double(timestep) * 10
    let cSkip = 0.25 / (scaledTimestep * scaledTimestep + 0.25)
    let cOut = scaledTimestep / (scaledTimestep * scaledTimestep + 0.25).squareRoot()
    return (Float(cSkip), Float(cOut))
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
    let inChannels: Int
    if version == .svdI2v {
      inChannels = channels * 2
    } else {
      switch modifier {
      case .inpainting, .depth, .canny:
        inChannels = channels + (conditionImage?.shape[3] ?? 0)
      case .editing:
        inChannels = channels * 2
      case .double:
        inChannels = channels * 2
      case .none, .kontext, .kontextKv, .qwenimageEditPlus, .qwenimageLayered,
        .qwenimageEdit2511, .fl2va, .ref2va:
        inChannels = channels
      }
    }
    let zeroNegativePrompt = false
    var xIn = graph.variable(
      .GPU(0), .NHWC(batchSize, startHeight, startWidth, inChannels),
      of: FloatType.self
    )
    switch modifier {
    case .inpainting, .depth, .canny:
      if let conditionImage = conditionImage {
        let shape = conditionImage.shape
        for i in stride(from: 0, to: batchSize, by: shape[0]) {
          xIn[
            i..<(i + shape[0]), 0..<startHeight, 0..<startWidth, channels..<(channels + shape[3])] =
            conditionImage
        }
      }
    case .editing:
      let maskedImage = conditionImage!
      for i in 0..<batchSize {
        xIn[i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] = maskedImage
      }
    case .double:
      let maskedImage = conditionImage!
      let maskedImageChannels = maskedImage.shape[3]
      for i in 0..<batchSize {
        xIn[
          i..<(i + 1), 0..<startHeight, 0..<startWidth, channels..<(channels + maskedImageChannels)] =
          maskedImage
      }
    case .none, .kontext, .kontextKv, .qwenimageEditPlus, .qwenimageLayered,
      .qwenimageEdit2511, .fl2va, .ref2va:
      break
    }
    var c = c
    var extraProjection = extraProjection
    // There is no tokenLengthUncond any more.
    let tokenLengthUncond = version == .minimaxH3 ? tokenLengthUncond : tokenLengthCond
    if version != .svdI2v && version != .minimaxH3 {
      for i in 0..<c.count {
        let shape = c[i].shape
        let batchSize = shape[0] / 2
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
        let batchSize = shape[0] / 2
        if shape.count == 3 {
          // Only tokenLengthCond is used.
          projection = projection[batchSize..<(batchSize * 2), 0..<shape[1], 0..<shape[2]].copied()
        } else if shape.count == 2 {
          projection = projection[batchSize..<(batchSize * 2), 0..<shape[1]].copied()
        }
        extraProjection = projection
      }
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
      let alphasCumprod = Array(
        discretization.alphasCumprod(steps: 1000, shift: sampling.shift).reversed())
      let sigmas = alphasCumprod.map { discretization.sigma(from: $0) }
      var timesteps = (0..<sampling.steps).map {
        min(
          Int(
            (Double(1000 - 1 - 20) * Double(sampling.steps - $0)
              / Double(sampling.steps)).rounded()) + 20, 1000 - 1)
      }
      if (Float(startStep.integral) - startStep.fractional).magnitude >= 1e-4 {
        timesteps[startStep.integral] = min(
          Int(
            1000 - startStep.fractional * 1000 / Float(sampling.steps)), 1000 - 1)
      }
      if (Float(endStep.integral) - endStep.fractional).magnitude >= 1e-4
        && endStep.integral < sampling.steps
      {
        timesteps[endStep.integral] = min(
          Int(
            1000 - endStep.fractional * 1000 / Float(sampling.steps)), 1000 - 1)
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
          isCfgEnabled: false, textGuidanceScale: textGuidanceScale, guidanceEmbed: guidanceEmbed,
          isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
          distilledGuidanceLayers: distilledGuidanceLayers, modifier: modifier,
          textEncoding: c,
          timesteps: timesteps[startStep.integral..<endStep.integral].map { Float($0) },
          audioShiftRatio: audioShiftRatio,
          batchSize: batchSize, startHeight: startHeight,
          startWidth: startWidth,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond, lora: lora,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, isBF16: isBF16,
          injectedControls: injectedControls, referenceImages: referenceImages,
          referenceAudios: referenceAudios)
        conditions = vector + encodings
        injectedControlsC = injectedControls.map {
          $0.model.encode(
            isCfgEnabled: false, textGuidanceScale: textGuidanceScale, guidanceEmbed: guidanceEmbed,
            isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, textEncoding: oldC,
            timesteps: timesteps[startStep.integral..<endStep.integral].map { Float($0) },
            vector: vector.first, batchSize: batchSize, startHeight: startHeight,
            startWidth: startWidth, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, zeroNegativePrompt: zeroNegativePrompt,
            mainUNetFixed: (fixedEncoder.filePath, weightMapper))
        }
      } else {
        injectedControlsC = injectedControls.map {
          $0.model.encode(
            isCfgEnabled: false, textGuidanceScale: textGuidanceScale, guidanceEmbed: guidanceEmbed,
            isGuidanceEmbedEnabled: isGuidanceEmbedEnabled, textEncoding: oldC,
            timesteps: timesteps[startStep.integral..<endStep.integral].map { Float($0) },
            vector: nil, batchSize: batchSize, startHeight: startHeight,
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
      var timeEmbeddingSize = version == .kandinsky21 || version == .sdxlRefiner ? 384 : 320
      let injectControlsAndAdapters = InjectControlsAndAdapters(
        injectControls: injectControls, injectT2IAdapters: injectT2IAdapters,
        injectAttentionKV: injectAttentionKV, injectIPAdapterLengths: injectIPAdapterLengths,
        injectControlModels: injectedControls.map { $0.model })
      if existingUNets[0] == nil {
        let firstTimestep =
          discretization.timesteps - discretization.timesteps / Float(sampling.steps) + 1
        let t = unet.timeEmbed(
          graph: graph, batchSize: batchSize, timestep: firstTimestep, version: version)
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
            graph: graph, index: 0, batchSize: batchSize, tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond, conditions: newC,
            referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
            version: version, modifier: modifier,
            isCfgEnabled: false),
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          isCfgEnabled: false, extraProjection: extraProjection,
          injectedControlsAndAdapters: emptyInjectedControlsAndAdapters,
          referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
          tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
          isBF16: isBF16, activationQkScaling: activationQkScaling,
          activationProjScaling: activationProjScaling,
          activationFfnProjUpScaling: activationFfnProjUpScaling,
          activationFfnScaling: activationFfnScaling, weightsCache: weightsCache)
      }
      var noise: DynamicGraph.Tensor<FloatType>? = nil
      if mask != nil || version == .kandinsky21 {
        noise = graph.variable(.GPU(0), .NHWC(batchSize, startHeight, startWidth, channels))
      }
      var refinerKickIn = refiner.map { (1 - $0.start) * discretization.timesteps } ?? -1
      var unets: [UNet?] = [unet]
      let blur: Model?
      if sharpness > 0 {
        blur = Blur(filters: channels, sigma: 3.0, size: 13, input: x)
      } else {
        blur = nil
      }
      var currentModelVersion = version
      var indexOffset = startStep.integral
      let condProj = Dense(count: 320, noBias: true)
      let cfg = graph.variable(
        Tensor<FloatType>(
          from: guidanceScaleEmbedding(guidanceScale: textGuidanceScale - 1, embeddingSize: 256)
        ).toGPU(0))
      condProj.compile(inputs: cfg)
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        store.read("w_cond_proj", model: condProj, codec: [.q6p, .q8p, .ezm7, .externalData]) {
          name, dataType, format, shape in
          guard store.read(name) == nil else {
            return .continue(name)
          }
          switch dataType {
          case .Float16, .BFloat16:
            #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
              var tensor = Tensor<Float16>(.CPU, format: format, shape: shape)
              tensor.withUnsafeMutableBytes {
                let size = shape.reduce(MemoryLayout<Float16>.size, *)
                if let baseAddress = $0.baseAddress {
                  memset(baseAddress, 0, size)
                }
              }
              return .final(tensor)
            #else
              return .continue(name)
            #endif
          case .Float32:
            var tensor = Tensor<Float32>(.CPU, format: format, shape: shape)
            tensor.withUnsafeMutableBytes {
              let size = shape.reduce(MemoryLayout<Float32>.size, *)
              if let baseAddress = $0.baseAddress {
                memset(baseAddress, 0, size)
              }
            }
            return .final(tensor)
          case .Float64, .Int32, .Int64, .UInt8:
            fatalError()
          }
        }
      }
      var cfgCond = condProj(inputs: cfg)[0].as(of: FloatType.self)
      if timeEmbeddingSize != cfgCond.shape[1] {
        cfgCond = graph.variable(.GPU(0), .WC(batchSize, timeEmbeddingSize), of: FloatType.self)
        cfgCond.full(0)
      }
      if version == .svdI2v {
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
        xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)] = frames
      }
      var oldDenoised: DynamicGraph.Tensor<FloatType>? = nil
      for i in startStep.integral..<endStep.integral {
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
        let timestep = timesteps[i]
        if Float(timestep) < refinerKickIn, let refiner = refiner {
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
                isCfgEnabled: false, textGuidanceScale: textGuidanceScale,
                guidanceEmbed: guidanceEmbed, isGuidanceEmbedEnabled: isGuidanceEmbedEnabled,
                distilledGuidanceLayers: refiner.distilledGuidanceLayers, modifier: modifier,
                textEncoding: oldC, timesteps: timesteps[i..<endStep.integral].map { Float($0) },
                audioShiftRatio: audioShiftRatio,
                batchSize: batchSize, startHeight: startHeight,
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
          timeEmbeddingSize =
            refiner.version == .kandinsky21 || refiner.version == .sdxlRefiner ? 384 : 320
          let firstTimestep = timesteps[0]
          let t = unet.timeEmbed(
            graph: graph, batchSize: batchSize, timestep: Float(firstTimestep),
            version: currentModelVersion)
          if let t = t, timeEmbeddingSize != cfgCond.shape[1] || !refiner.isConsistencyModel {
            cfgCond = graph.variable(like: t)
            cfgCond.full(0)
          }
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
              graph: graph, index: 0, batchSize: batchSize, tokenLengthUncond: tokenLengthUncond,
              tokenLengthCond: tokenLengthCond, conditions: newC,
              referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
              version: currentModelVersion, modifier: modifier, isCfgEnabled: false),
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: false, extraProjection: extraProjection,
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
        let sigma = sigmas[timestep]
        let timestepNext = i + 1 < timesteps.count ? timesteps[i + 1] : 0
        let cNoise: Float
        let cNoiseNext: Float
        switch conditioning {
        case .noise:
          cNoise = discretization.noise(for: alphasCumprod[timestep])
          cNoiseNext = discretization.noise(for: alphasCumprod[timestepNext])
        case .timestep:
          cNoise = Float(timestep)
          cNoiseNext = Float(timestepNext)
        }
        let t = unet.timeEmbed(
          graph: graph, batchSize: batchSize, timestep: cNoise, version: currentModelVersion)
        let conditions = UNetExtractConditions(
          of: FloatType.self,
          graph: graph, index: i - indexOffset, batchSize: batchSize,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          conditions: conditions, referenceImageCount: referenceImageCount,
          referenceAudioCount: referenceAudioCount,
          version: currentModelVersion, modifier: modifier, isCfgEnabled: false)
        xIn[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels] = x
        let injectedIPAdapters = ControlModel<FloatType>
          .injectedIPAdapters(
            injecteds: injectedControls, step: i, version: unet.version,
            usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: false, index: i - startStep.integral,
            mainUNetAndWeightMapper: unet.modelAndWeightMapper,
            controlNets: &controlNets)
        let injectedControlsAndAdapters = ControlModel<FloatType>
          .injectedControlsAndAdapters(
            injecteds: injectedControls, step: i, version: unet.version,
            usesFlashAttention: usesFlashAttention != .none, inputs: xIn, t, injectedControlsC,
            tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
            isCfgEnabled: false, index: i - startStep.integral,
            mainUNetAndWeightMapper: unet.modelAndWeightMapper,
            controlNets: &controlNets)
        let newC: [DynamicGraph.AnyTensor]
        if version == .svdI2v {
          newC = Array(conditions[0..<(1 + (conditions.count - 1) / 2)])
        } else {
          newC = conditions
        }
        var etOut = unet(
          timestep: (now: cNoise, next: cNoiseNext), audioShiftRatio: audioShiftRatio, inputs: xIn,
          t.map { $0 + cfgCond }, newC,
          extraProjection: extraProjection,
          injectedControlsAndAdapters: injectedControlsAndAdapters,
          injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
          referenceAudioCount: referenceAudioCount, step: i,
          tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond, isCfgEnabled: false, tiledDiffusion: tiledDiffusion,
          controlNets: &controlNets)

        if channels < etOut.shape[3] {
          etOut = etOut[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels].copied()
        }
        if let blur = blur {
          let alpha =
            0.001 * sharpness * (discretization.timesteps - Float(timestep))
            / discretization.timesteps
          let etOutDegraded = blur(inputs: etOut)[0].as(of: FloatType.self)
          etOut = Functional.add(
            left: etOutDegraded, right: etOut, leftScalar: alpha, rightScalar: 1 - alpha)
        }
        let alpha = alphasCumprod[timestep]
        let predictOriginalSample: DynamicGraph.Tensor<FloatType>
        switch discretization.objective {
        case .u(_):
          predictOriginalSample = Functional.add(
            left: x, right: etOut, leftScalar: 1,
            rightScalar: Float(-sigma))
        case .v:
          let sqrtAlphaCumprod = 1.0 / (sigma * sigma + 1).squareRoot()
          predictOriginalSample = Functional.add(
            left: x, right: etOut, leftScalar: Float(sqrtAlphaCumprod),
            rightScalar: Float(-sigma * sqrtAlphaCumprod))
        case .epsilon:
          predictOriginalSample = Functional.add(
            left: x, right: etOut, leftScalar: Float(1.0 / alpha.squareRoot()),
            rightScalar: Float(-sigma))
        case .edm(let sigmaData):
          let sigmaData2 = sigmaData * sigmaData
          predictOriginalSample = Functional.add(
            left: x, right: etOut,
            leftScalar: Float(sigmaData2 / (sigma * sigma + sigmaData2).squareRoot()),
            rightScalar: Float(sigma * sigmaData / (sigma * sigma + sigmaData2).squareRoot()))
        }
        let (cSkip, cOut) = scalingsForBoundaryConditionDiscrete(timestep: timestep)
        let denoised = Functional.add(
          left: predictOriginalSample, right: x, leftScalar: cOut, rightScalar: cSkip)
        oldDenoised = denoised
        let alphaPrev = alphasCumprod[timestepNext]
        if i < sampling.steps - 1 {
          let noise = graph.variable(like: x)
          noise.randn(std: 1, mean: 0)
          if case .u(_) = discretization.objective {
            x = Functional.add(
              left: denoised, right: noise, leftScalar: Float(alphaPrev),
              rightScalar: Float(1 - alphaPrev))
          } else {
            x = Functional.add(
              left: denoised, right: noise, leftScalar: Float(alphaPrev.squareRoot()),
              rightScalar: Float((1 - alphaPrev).squareRoot()))
          }
        } else {
          x = denoised
        }
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
    let step = Int(step.rounded())
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let alphaCumprod = alphasCumprod[max(0, min(alphasCumprod.count - 1, step))]
    if case .u(_) = discretization.objective {
      return Float(alphaCumprod)
    } else {
      return Float(alphaCumprod.squareRoot())
    }
  }

  public func noiseScaleFactor(at step: Float, sampling: Sampling) -> Float {
    let step = Int(step.rounded())
    let alphasCumprod = discretization.alphasCumprod(steps: sampling.steps, shift: sampling.shift)
    let alphaCumprod = alphasCumprod[max(0, min(alphasCumprod.count - 1, step))]
    if case .u(_) = discretization.objective {
      return Float(1 - alphaCumprod)
    } else {
      return Float((1 - alphaCumprod).squareRoot())
    }
  }
}
