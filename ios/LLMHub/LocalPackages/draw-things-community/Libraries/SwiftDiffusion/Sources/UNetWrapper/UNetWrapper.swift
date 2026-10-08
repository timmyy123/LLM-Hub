import Dflat
import Diffusion
import NNC
import WeightsCache

#if !os(Linux)
  import DiffusionCoreML
#endif

public struct UNetWrapper<FloatType: TensorNumeric & BinaryFloatingPoint>: UNetProtocol {

  private var unetFromNNC = UNetFromNNC<FloatType>()
  #if !os(Linux)
    private var unetFromCoreML = UNetFromCoreML<FloatType>()
  #endif
  private var preferCoreML = false
  public init() {}
  public var version: ModelVersion {
    #if !os(Linux)
      if preferCoreML {
        return unetFromCoreML.version
      } else {
        return unetFromNNC.version
      }
    #else
      return unetFromNNC.version
    #endif
  }
  public var isLoaded: Bool {
    #if !os(Linux)
      return unetFromCoreML.isLoaded || unetFromNNC.isLoaded
    #else
      return unetFromNNC.isLoaded
    #endif
  }
  public var didRunLoRASeparately: Bool {
    #if !os(Linux)
      if preferCoreML {
        return unetFromCoreML.didRunLoRASeparately
      } else {
        return unetFromNNC.didRunLoRASeparately
      }
    #else
      return unetFromNNC.didRunLoRASeparately
    #endif
  }
  public func unloadResources() {
    unetFromNNC.unloadResources()
    #if !os(Linux)
      unetFromCoreML.unloadResources()
    #endif
  }
}

extension UNetWrapper {
  public var model: AnyModel? {
    guard unetFromNNC.isLoaded else { return nil }
    return unetFromNNC.model
  }
  public var modelAndWeightMapper: (AnyModel, ModelWeightMapper)? {
    guard unetFromNNC.isLoaded else { return nil }
    return unetFromNNC.modelAndWeightMapper
  }
  public mutating func unloadModel() {
    unetFromNNC.unloadModel()
  }
  public mutating func compileModel(
    filePath: String, externalOnDemand: Bool, deviceProperties: DeviceProperties,
    version: ModelVersion,
    modifier: SamplerModifier,
    qkNorm: Bool, dualAttentionLayers: [Int], upcastAttention: Bool,
    usesFlashAttention: UseFlashAttention,
    usesSolAttention: Bool,
    solAttentionStart: Int, solAttentionTau: Float,
    injectControlsAndAdapters: InjectControlsAndAdapters<FloatType>, lora: [LoRAConfiguration],
    isQuantizedModel: Bool, canRunLoRASeparately: Bool, inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>?, _ c: [DynamicGraph.AnyTensor],
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: InjectedControlsAndAdapters<FloatType>, referenceImageCount: Int,
    referenceAudioCount: Int,
    tiledDiffusion: TiledConfiguration, teaCache: TeaCacheConfiguration,
    causalInference: (Int, pad: Int), isBF16: Bool, activationQkScaling: [Int: Int],
    activationProjScaling: [Int: Int], activationFfnProjUpScaling: [Int: Int],
    activationFfnScaling: [Int: Int], weightsCache: WeightsCache
  ) -> Bool {
    #if !os(Linux)

      if unetFromCoreML.compileModel(
        filePath: filePath, externalOnDemand: externalOnDemand, deviceProperties: deviceProperties,
        version: version,
        modifier: modifier,
        qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
        usesSolAttention: usesSolAttention,
        solAttentionStart: solAttentionStart, solAttentionTau: solAttentionTau,
        injectControlsAndAdapters: injectControlsAndAdapters, lora: lora,
        isQuantizedModel: isQuantizedModel, canRunLoRASeparately: canRunLoRASeparately, inputs: xT,
        timestep, c,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: isCfgEnabled,
        extraProjection: extraProjection, injectedControlsAndAdapters: injectedControlsAndAdapters,
        referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
        tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
        isBF16: isBF16, activationQkScaling: activationQkScaling,
        activationProjScaling: activationProjScaling,
        activationFfnProjUpScaling: activationFfnProjUpScaling,
        activationFfnScaling: activationFfnScaling, weightsCache: weightsCache)
      {
        preferCoreML = true
        return true
      }
    #endif
    let _ = unetFromNNC.compileModel(
      filePath: filePath, externalOnDemand: externalOnDemand, deviceProperties: deviceProperties,
      version: version, modifier: modifier,
      qkNorm: qkNorm, dualAttentionLayers: dualAttentionLayers,
      upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
      usesSolAttention: usesSolAttention,
      solAttentionStart: solAttentionStart, solAttentionTau: solAttentionTau,
      injectControlsAndAdapters: injectControlsAndAdapters, lora: lora,
      isQuantizedModel: isQuantizedModel, canRunLoRASeparately: canRunLoRASeparately, inputs: xT,
      timestep, c,
      tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
      isCfgEnabled: isCfgEnabled,
      extraProjection: extraProjection, injectedControlsAndAdapters: injectedControlsAndAdapters,
      referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
      tiledDiffusion: tiledDiffusion, teaCache: teaCache, causalInference: causalInference,
      isBF16: isBF16, activationQkScaling: activationQkScaling,
      activationProjScaling: activationProjScaling,
      activationFfnProjUpScaling: activationFfnProjUpScaling,
      activationFfnScaling: activationFfnScaling, weightsCache: weightsCache)
    return true
  }

  public func callAsFunction(
    timestep t: (now: Float, next: Float), audioShiftRatio: Float,
    inputs xT: DynamicGraph.Tensor<FloatType>, _ timestep: DynamicGraph.Tensor<FloatType>?,
    _ c: [DynamicGraph.AnyTensor], extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ restInputs: [DynamicGraph.AnyTensor],
      _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [NNC.DynamicGraph.Tensor<FloatType>]
    ),
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>], referenceImageCount: Int,
    referenceAudioCount: Int, step: Int,
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    tiledDiffusion: TiledConfiguration, controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    #if !os(Linux)

      if preferCoreML {
        return unetFromCoreML(
          timestep: t, audioShiftRatio: audioShiftRatio, inputs: xT, timestep, c,
          extraProjection: extraProjection,
          injectedControlsAndAdapters: injectedControlsAndAdapters,
          injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
          referenceAudioCount: referenceAudioCount,
          step: step, tokenLengthUncond: tokenLengthUncond,
          tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
          tiledDiffusion: tiledDiffusion, controlNets: &controlNets)
      }
    #endif

    return unetFromNNC(
      timestep: t, audioShiftRatio: audioShiftRatio, inputs: xT, timestep, c,
      extraProjection: extraProjection,
      injectedControlsAndAdapters: injectedControlsAndAdapters,
      injectedIPAdapters: injectedIPAdapters, referenceImageCount: referenceImageCount,
      referenceAudioCount: referenceAudioCount, step: step,
      tokenLengthUncond: tokenLengthUncond,
      tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled, tiledDiffusion: tiledDiffusion,
      controlNets: &controlNets)
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    #if !os(Linux)
      if preferCoreML {
        return unetFromCoreML.decode(x)
      }
    #endif

    return unetFromNNC.decode(x)

  }

  public mutating func cancel() {
    unetFromNNC.cancel()
  }
}
