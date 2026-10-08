import Atomics
import DataModels
import Dflat
import Diffusion
import DiffusionCoreMLModelManager
import DiffusionUNetWrapper
import Foundation
import ModelZoo
import NNC
import WeightsCache

extension LoRAConfiguration.Mode {
  public init(from mode: LoRAMode) {
    switch mode {
    case .all:
      self = .all
    case .base:
      self = .base
    case .refiner:
      self = .refiner
    }
  }
}

public enum JITWeightsLoading: Int, Codable {
  case automatic = 0
  case alwaysPartially = 1
  case alwaysFully = 2
  case never = 3
}

public final class ModelPreloader {
  public enum PreloadState: Equatable {
    case countdown(Int)
    case preloadStart(Double, Bool, String, [LoRAConfiguration])
    case preloadDone
  }
  enum Mode {
    case preload
    case yes
    case no
    case coreml
    case unet
    static func from(_ i: Int) -> Self {
      switch i {
      case 0:
        return .no
      case 1:
        return .yes
      case 2:
        return .preload
      case 3:
        return .coreml
      case 4:
        return .unet
      default:
        return .no
      }
    }
  }
  #if canImport(Combine)
    @Published
    public private(set) var preloadState: PreloadState = .preloadDone
  #else
    public private(set) var preloadState: PreloadState = .preloadDone
  #endif
  private let queue: DispatchQueue
  private let weightsCache: WeightsCache
  private var workspace: Workspace
  private var isGenerating: Bool = false
  private var modelFile: String? = nil
  private var imageScale: DeviceCapability.Scale? = nil
  private var batchSize: Int? = nil
  private var clipSkip: Int? = nil
  private var lora: [LoRAConfiguration]? = nil
  private var modelFilePath: String? = nil

  private var firstStageEncoderFilePath: String? = nil
  private var firstStageEncoderExternalOnDemand: Bool? = nil
  private var firstStageEncoderVersion: ModelVersion? = nil
  private var firstStageEncoderScale: DeviceCapability.Scale? = nil
  private var firstStageEncoderHighPrecision: Bool? = nil
  private var firstStageEncoderTiledDiffusion: TiledConfiguration? = nil

  private var firstStageDecoderFilePath: String? = nil
  private var firstStageDecoderExternalOnDemand: Bool? = nil
  private var firstStageDecoderVersion: ModelVersion? = nil
  private var firstStageDecoderScale: DeviceCapability.Scale? = nil
  private var firstStageDecoderHighPrecision: Bool? = nil
  private var firstStageDecoderTiledDecoding: TiledConfiguration? = nil

  private var textEncoderFilePaths: [String]? = nil
  private var textEncoderVersion: ModelVersion? = nil
  private var textEncoderIsCfgEnabled: Bool? = nil
  private var textEncoderUsesFlashAttention: Bool? = nil
  private var textEncoderInjectEmbeddings: Bool = false
  private var textEncoderMaxLength: Int = 77
  private var textEncoderClipSkip: Int = 1
  private var textEncoderLoRA: [LoRAConfiguration]? = nil

  private var unetFilePath: String? = nil
  private var unetModifier: SamplerModifier? = nil
  private var unetVersion: ModelVersion? = nil
  private var unetUpcastAttention: Bool? = nil
  private var unetUsesFlashAttention: UseFlashAttention? = nil
  private var unetUsesSolAttention = false
  private var unetExternalOnDemand: Bool? = nil
  private var unetInjectControls: Bool? = nil
  private var unetInjectT2IAdapters: Bool? = nil
  private var unetInjectIPAdapterLengths: [Int]? = nil
  private var unetTiledDiffusion: TiledConfiguration? = nil
  private var unetLoRA: [LoRAConfiguration]? = nil
  private var unetTokenLengthUncond: Int = 77
  private var unetTokenLengthCond: Int = 77
  private var unetScale: DeviceCapability.Scale? = nil
  private var unetGuidanceEmbed: Bool? = nil

  private var unet = UNetWrapper<FloatType>()
  private var firstStageDecoder: Model? = nil
  private var firstStageEncoder: Model? = nil
  private var textModel: Model? = nil
  private var tiledDecoding: TiledConfiguration? = nil
  private var tiledDiffusion: TiledConfiguration? = nil
  // Subscription tokens.
  private var configurationSubscription: Workspace.Subscription? = nil
  private var keepModelInMemorySubscription: Workspace.Subscription? = nil
  private var useCoreMLSubscription: Workspace.Subscription? = nil
  private var useAppleNeuralEngineFor8BitSModelsSubscription: Workspace.Subscription? = nil
  private var loraUseCoreMLSubscription: Workspace.Subscription? = nil
  private var weightsCacheSizeSubscription: Workspace.Subscription? = nil
  private var externalStoreSubscription: Workspace.Subscription? = nil
  private var useMFASubscription: Workspace.Subscription? = nil
  private var mergeLoRASubscription: Workspace.Subscription? = nil
  private var mode: Mode
  private var activeSentinel: Int = 0
  private let sentinel = ManagedAtomic<Int>(0)
  private var useCoreML: Bool? = nil
  private var useAppleNeuralEngineFor8BitSModels: Bool? = nil
  private var useMFA: Int? = nil
  private var loraUseCoreML: Bool? = nil
  private var externalStore: JITWeightsLoading = .automatic
  private var mergeLoRA: Int? = nil
  init(
    queue: DispatchQueue, weightsCache: WeightsCache,
    configurations: FetchedResult<GenerationConfiguration>, workspace: Workspace
  ) {
    self.queue = queue
    self.weightsCache = weightsCache
    self.workspace = workspace
    if let configuration = configurations.first {
      modelFile = configuration.model ?? ModelZoo.defaultSpecification.file
      imageScale = DeviceCapability.Scale(
        widthScale: configuration.startWidth, heightScale: configuration.startHeight)
      batchSize = Int(configuration.batchSize)
      clipSkip = Int(configuration.clipSkip)
      let modelVersion = ModelZoo.versionForModel(modelFile ?? "")
      let isRefinerEnabled: Bool =
        configuration.refinerModel.map {
          guard $0 != configuration.model, ModelZoo.isModelDownloaded($0) else { return false }
          let version = ModelZoo.versionForModel($0)
          guard ModelZoo.isCompatibleRefiner(modelVersion, refinerVersion: version) else {
            return false
          }
          return true
        } ?? false
      lora = configuration.loras.compactMap {
        guard let file = $0.file, LoRAZoo.isModelDownloaded(file),
          LoRAZoo.versionForModel(file) == modelVersion
        else { return nil }
        guard !isRefinerEnabled || $0.mode == .all || $0.mode == .base else { return nil }
        return LoRAConfiguration(
          file: LoRAZoo.filePathForModelDownloaded(file), weight: $0.weight, version: modelVersion,
          isLoHa: LoRAZoo.isLoHaForModel(file), modifier: LoRAZoo.modifierForModel(file),
          mode: .init(from: $0.mode))
      }
      tiledDecoding = TiledConfiguration(
        isEnabled: configuration.tiledDecoding,
        tileSize: .init(
          width: Int(configuration.decodingTileWidth),
          height: Int(configuration.decodingTileHeight)),
        tileOverlap: Int(configuration.decodingTileOverlap))
      tiledDiffusion = TiledConfiguration(
        isEnabled: configuration.tiledDiffusion,
        tileSize: .init(
          width: Int(configuration.diffusionTileWidth),
          height: Int(configuration.diffusionTileHeight)),
        tileOverlap: Int(configuration.diffusionTileOverlap))
    }
    let mode =
      workspace.dictionary["keep_model_in_memory", Int.self]
      ?? (DeviceCapability.keepModelPreloaded ? 2 : 0)
    self.mode = Mode.from(mode)
    queue.async { [weak self] in
      guard let self = self else { return }
      if let _ = self.workspace.dictionary["preload_guard", Bool.self] {
        if self.mode == .preload || self.mode == .coreml || self.mode == .unet {
          // If default is not to keep model preloaded, just remove the overwrite.
          if !DeviceCapability.keepModelPreloaded {
            self.workspace.dictionary["keep_model_in_memory", Int.self] = nil
          } else {
            // Otherwise, this is more curious, we disable it completely.
            self.workspace.dictionary["keep_model_in_memory", Int.self] = 0
          }
          // Although we subscribe the update, need to do this now because we may immediately start to load in did active.
          self.mode = .no
        }
        self.workspace.dictionary["preload_guard", Bool.self] = nil
      }
      if let _ = self.workspace.dictionary["coreml_guard", Bool.self] {
        if !DeviceCapability.isCoreMLSupported {
          // If we override to CoreML, just remove it.
          self.workspace.dictionary["use_coreml", Bool.self] = nil
        } else {
          // Otherwise, this is curious, but we will disable it completely.
          self.workspace.dictionary["use_coreml", Bool.self] = false
        }
        self.workspace.dictionary["coreml_guard", Bool.self] = nil
      }
      if let _ = self.workspace.dictionary["mfa_guard", Bool.self] {
        if !DeviceCapability.isMFASupported {
          self.workspace.dictionary["use_mfa_v4", Int.self] = nil
        } else {
          self.workspace.dictionary["use_mfa_v4", Int.self] = 0
        }
        self.workspace.dictionary["mfa_guard", Bool.self] = nil
      }
    }
    addLifecycleObservers()
    configurationSubscription = workspace.subscribe(fetchedResult: configurations) {
      [weak self]
      newConfigurations in
      guard let self = self else { return }
      let sentinelValue = self.sentinel.wrappingIncrementThenLoad(ordering: .acquiringAndReleasing)
      queue.async { [weak self] in
        guard let self = self, let newConfiguration = newConfigurations.first else { return }
        guard self.sentinel.load(ordering: .acquiring) == sentinelValue else { return }
        self.newConfiguration(newConfiguration)
      }
    }
    // The outer subscriptions must also capture weakly, or the workspace keeps
    // the preloader and its weights cache alive after the generator is released.
    keepModelInMemorySubscription = workspace.dictionary.subscribe(
      "keep_model_in_memory", of: Int.self
    ) { [weak self] value in
      queue.async { [weak self] in
        guard let self = self else { return }
        switch value {
        case .deleted:
          self.updateKeepModelInMemory(nil)
        case .initial(let value):
          self.updateKeepModelInMemory(value)
        case .updated(let value):
          self.updateKeepModelInMemory(value)
        }
      }
    }
    useCoreMLSubscription = workspace.dictionary.subscribe("use_coreml", of: Bool.self) {
      [weak self] value in
      queue.async { [weak self] in
        guard let self = self else { return }
        switch value {
        case .deleted:
          self.updateUseCoreML(nil)
        case .initial(let value):
          self.updateUseCoreML(value)
        case .updated(let value):
          self.updateUseCoreML(value)
        }
      }
    }
    useAppleNeuralEngineFor8BitSModelsSubscription = workspace.dictionary.subscribe(
      "use_mfa_apple_neural_engine", of: Bool.self
    ) { [weak self] value in
      queue.async { [weak self] in
        guard let self = self else { return }
        switch value {
        case .deleted:
          self.updateUseAppleNeuralEngineFor8BitSModels(nil)
        case .initial(let value):
          self.updateUseAppleNeuralEngineFor8BitSModels(value)
        case .updated(let value):
          self.updateUseAppleNeuralEngineFor8BitSModels(value)
        }
      }
    }
    loraUseCoreMLSubscription = workspace.dictionary.subscribe("lora_use_coreml", of: Bool.self) {
      [weak self] value in
      queue.async { [weak self] in
        guard let self = self else { return }
        switch value {
        case .deleted:
          self.updateLoRAUseCoreML(nil)
        case .initial(let value):
          self.updateLoRAUseCoreML(value)
        case .updated(let value):
          self.updateLoRAUseCoreML(value)
        }
      }
    }
    weightsCacheSizeSubscription = workspace.dictionary.subscribe(
      "total_weights_cache", of: Int.self
    ) { [weak self] value in
      queue.async { [weak self] in
        guard let self = self else { return }
        switch value {
        case .deleted:
          self.updateWeightsCacheSize(nil)
        case .initial(let value):
          self.updateWeightsCacheSize(value)
        case .updated(let value):
          self.updateWeightsCacheSize(value)
        }
      }
    }
    externalStore =
      workspace.dictionary["external_store_v2", Int.self].flatMap {
        JITWeightsLoading(rawValue: $0)
      }
      ?? .automatic
    DeviceCapability.isPartialOffloadPreferred.store(
      externalStore == .alwaysPartially, ordering: .releasing)
    externalStoreSubscription = workspace.dictionary.subscribe("external_store_v2", of: Int.self) {
      [weak self] value in
      queue.async { [weak self] in
        guard let self = self else { return }
        switch value {
        case .deleted:
          self.updateExternalStore(nil)
        case .initial(let value):
          self.updateExternalStore(value)
        case .updated(let value):
          self.updateExternalStore(value)
        }
      }
    }
    useMFASubscription = workspace.dictionary.subscribe("use_mfa_v4", of: Int.self) {
      [weak self] value in
      queue.async { [weak self] in
        guard let self = self else { return }
        switch value {
        case .deleted:
          self.updateUseMFA(nil)
        case .initial(let value):
          self.updateUseMFA(value)
        case .updated(let value):
          self.updateUseMFA(value)
        }
      }
    }
    mergeLoRASubscription = workspace.dictionary.subscribe("merge_lora", of: Int.self) {
      [weak self] value in
      queue.async { [weak self] in
        guard let self = self else { return }
        switch value {
        case .deleted:
          self.updateMergeLoRA(nil)
        case .initial(let value):
          self.updateMergeLoRA(value)
        case .updated(let value):
          self.updateMergeLoRA(value)
        }
      }
    }
    // If it is already active, trigger preload through didBecomeActive call.
    if isAppActive() {
      queue.async { [weak self] in
        guard let self = self else { return }
        self.didBecomeActive()
      }
    }
  }
}

extension ModelPreloader {
  public func beginCoreMLGuard() -> Bool {
    guard CoreMLModelManager.isCoreMLSupported.load(ordering: .acquiring) else { return false }
    workspace.dictionary["coreml_guard", Bool.self] = true
    workspace.dictionary.synchronize()
    return true
  }
  public func endCoreMLGuard() {
    workspace.dictionary["coreml_guard", Bool.self] = nil
  }
  public func beginMFAGuard() -> Bool {
    guard DeviceCapability.isMFAEnabled.load(ordering: .acquiring) > 0 else { return false }
    // For these devices, we are very confident it just works, hence, no need to disable MFA upon crash.
    guard !(DeviceCapability.isMFASupported && DeviceCapability.isMFACausalAttentionMaskSupported)
    else { return true }
    workspace.dictionary["mfa_guard", Bool.self] = true
    workspace.dictionary.synchronize()
    return true
  }
  public func endMFAGuard() {
    workspace.dictionary["mfa_guard", Bool.self] = nil
  }
}

extension ModelPreloader {
  private var isEnabled: Bool {
    #if targetEnvironment(simulator)
      return false
    #else
      guard let modelFile = modelFile else { return false }
      let modelVersion = ModelZoo.versionForModel(modelFile)
      guard
        modelVersion == .v1 || modelVersion == .v2 || modelVersion == .sdxlBase
          || modelVersion == .sdxlRefiner || modelVersion == .ssd1b
      else { return false }
      let textEncoderVersion = ModelZoo.textEncoderVersionForModel(modelFile)
      guard textEncoderVersion == nil else { return false }
      if DeviceCapability.isMaxPerformance {
        return true
      }
      // We cannot cache model if it is SDXL models.
      if modelVersion == .sdxlBase || modelVersion == .sdxlRefiner || modelVersion == .ssd1b {
        return false
      }
      guard let imageScale = imageScale, let batchSize = batchSize,
        let lora = lora, clipSkip != nil
      else {
        return false
      }
      let upcastAttention = ModelZoo.isUpcastAttentionForModel(modelFile)
      var useCoreML = CoreMLModelManager.isCoreMLSupported.load(ordering: .acquiring)
      let loraUseCoreML = CoreMLModelManager.isLoRASupported.load(ordering: .acquiring)
      if useCoreML {
        // Check if we satisfied all conditions to preload CoreML.
        useCoreML =
          imageScale.widthScale == 8 && imageScale.heightScale == 8 && !upcastAttention
          && (lora.count == 0 || loraUseCoreML)
      }
      // We only preload when batchSize is 1 or we use CoreML, or we are at 16GiB RAM.
      guard useCoreML || batchSize == 1 else { return false }
      return imageScale.widthScale * imageScale.heightScale <= 64
    #endif
  }

  private func removeAllCache() {
    unet = UNetWrapper()
    unetExternalOnDemand = nil
    unetUpcastAttention = nil
    unetInjectControls = nil
    unetInjectT2IAdapters = nil
    unetInjectIPAdapterLengths = nil
    unetTiledDiffusion = nil
    unetGuidanceEmbed = nil
    unetLoRA = nil
    unetTokenLengthUncond = 77
    unetTokenLengthCond = 77
    unetFilePath = nil
    unetModifier = nil
    unetVersion = nil
    unetScale = nil
    firstStageEncoder = nil
    firstStageEncoderFilePath = nil
    firstStageEncoderScale = nil
    firstStageEncoderExternalOnDemand = nil
    firstStageEncoderVersion = nil
    firstStageEncoderHighPrecision = nil
    firstStageEncoderTiledDiffusion = nil
    firstStageDecoder = nil
    firstStageDecoderFilePath = nil
    firstStageDecoderScale = nil
    firstStageDecoderExternalOnDemand = nil
    firstStageDecoderVersion = nil
    firstStageDecoderHighPrecision = nil
    firstStageDecoderTiledDecoding = nil
    textModel = nil
    textEncoderVersion = nil
    textEncoderInjectEmbeddings = false
    textEncoderMaxLength = 77
    textEncoderClipSkip = 1
    textEncoderLoRA = nil
    textEncoderFilePaths = nil
    textEncoderIsCfgEnabled = nil
  }

  private func preloadIfPossible() {
    defer { preloadState = .preloadDone }
    guard let modelFile = modelFile, let imageScale = imageScale, let batchSize = batchSize,
      let clipSkip = clipSkip, let lora = lora, let tiledDecoding = tiledDecoding,
      let tiledDiffusion = tiledDiffusion, let useMFA = useMFA, isEnabled
    else {
      return
    }
    // If we crashed, we know, and at that point, we will disable preload.
    workspace.dictionary["preload_guard", Bool.self] = true
    workspace.dictionary.synchronize()
    defer {
      workspace.dictionary["preload_guard", Bool.self] = nil
    }
    let autoencoderModel = ModelZoo.autoencoderForModel(modelFile) ?? "vae_ft_mse_840000_f16.ckpt"
    let textEncoderModel = ModelZoo.textEncoderForModel(modelFile) ?? "clip_vit_l14_f16.ckpt"
    guard
      ModelZoo.isModelDownloaded(modelFile) && ModelZoo.isModelDownloaded(autoencoderModel)
        && ModelZoo.isModelDownloaded(textEncoderModel)
    else { return }
    let modelPath = ModelZoo.filePathForModelDownloaded(modelFile)
    let modelVersion = ModelZoo.versionForModel(modelFile)
    let (qkNorm, dualAttentionLayers) =
      ModelZoo.MMDiTForModel(modelFile).map { return ($0.qkNorm, $0.dualAttentionLayers) } ?? (
        false, []
      )
    let isQuantizedModel = ModelZoo.isQuantizedModel(modelFile)
    let canRunLoRASeparately = canRunLoRASeparately
    let upcastAttention = ModelZoo.isUpcastAttentionForModel(modelFile)
    let modelModifier = ModelZoo.modifierForModel(modelFile)
    let autoencoderPath = ModelZoo.filePathForModelDownloaded(autoencoderModel)
    let textEncoderPath = ModelZoo.filePathForModelDownloaded(textEncoderModel)
    let highPrecisionForAutoencoder = ModelZoo.isHighPrecisionAutoencoderForModel(modelFile)
    let usesFlashAttention: UseFlashAttention
    switch useMFA {
    case 1:
      usesFlashAttention = .quantized
    case 2:
      usesFlashAttention = .sdpa
    default:
      usesFlashAttention = .none
    }
    var useCoreML = CoreMLModelManager.isCoreMLSupported.load(ordering: .acquiring)
    let useLoRACoreML = CoreMLModelManager.isLoRASupported.load(ordering: .acquiring)
    if useCoreML {
      // Check if we satisfied all conditions to preload CoreML.
      useCoreML =
        imageScale.widthScale == 8 && imageScale.heightScale == 8 && !upcastAttention
        && (lora.count == 0 || useLoRACoreML)
    }
    // No need to continue if we are not using CoreML for CoreML preload model.
    if mode == .coreml && !useCoreML {
      return
    }
    preloadState = .preloadStart(useCoreML ? 80 : 2, useCoreML, modelFile, lora)
    // SeedVR2 needs checkpoint-provided text embeddings and rotary side inputs, so the generic
    // SD-style preloader cannot synthesize a representative graph yet.
    if modelVersion == .hiDreamO1 || modelVersion == .seedvr2_3b || modelVersion == .seedvr2_7b
      || modelVersion == .ideogram4 || modelVersion == .krea2
      || modelVersion == .minimaxH3
    {
      return
    }
    let conditionalLength: Int
    switch modelVersion {
    case .v1:
      conditionalLength = 768
    case .v2:
      conditionalLength = 1024
    case .sdxlBase, .sdxlRefiner, .ssd1b:
      conditionalLength = 1280
    case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage,
      .qwenImage2_1, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b,
      .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2, .longcatVideoAvatar1_5,
      .minimaxH3:
      fatalError()
    }
    let cfgChannels: Int
    let numberOfChannels: Int
    switch modelModifier {
    case .fl2va, .ref2va:
      fatalError()
    case .depth, .canny:
      cfgChannels = 2
      numberOfChannels = 5
    case .inpainting:
      cfgChannels = 2
      numberOfChannels = 9
    case .editing:
      cfgChannels = 3
      numberOfChannels = 8
    case .double:
      cfgChannels = 2
      numberOfChannels = 8
    case .none, .kontext, .kontextKv, .qwenimageEditPlus, .qwenimageLayered,
      .qwenimageEdit2511:
      cfgChannels = 2
      numberOfChannels = 4
    }
    let graph = DynamicGraph()
    graph.withNoGrad {
      let startHeight = 8 * Int(imageScale.heightScale)
      let startWidth = 8 * Int(imageScale.widthScale)
      if !unet.isLoaded {
        let externalOnDemand = externalOnDemand(
          version: modelVersion, scale: imageScale, variant: .unet, injectedControls: 0)
        let x = graph.variable(
          .GPU(0), .NHWC(cfgChannels * batchSize, startHeight, startWidth, numberOfChannels),
          of: FloatType.self)
        let c = graph.variable(
          .GPU(0), .HWC(cfgChannels * batchSize, 77, conditionalLength), of: FloatType.self)
        let t = graph.variable(
          Tensor<FloatType>(
            from: timeEmbedding(
              timestep: 981, batchSize: cfgChannels * batchSize, embeddingSize: 320,
              maxPeriod: 10_000)
          ).toGPU(0)
        )
        let teaCache = TeaCacheConfiguration(
          coefficients: (0, 0, 0, 0, 0), steps: 0...0, threshold: 0, maxSkipSteps: 0)
        var cArr = [c]
        var conditions: [DynamicGraph.AnyTensor] = cArr
        if modelVersion == .sdxlBase || modelVersion == .sdxlRefiner || modelVersion == .ssd1b {
          let fixedEncoder = UNetFixedEncoder<FloatType>(
            filePath: modelPath, version: modelVersion, modifier: .none, dualAttentionLayers: [],
            activationQkScaling: [:],
            activationProjScaling: [:], activationFfnProjUpScaling: [:], activationFfnScaling: [:],
            usesFlashAttention: useMFA > 0 ? .sdpa : .none,
            zeroNegativePrompt: false, isQuantizedModel: false, canRunLoRASeparately: false,
            externalOnDemand: false,
            deviceProperties: DeviceCapability.deviceProperties,
            weightsCache: weightsCache)
          cArr.insert(
            graph.variable(.GPU(0), .HWC(cfgChannels * batchSize, 77, 768), of: FloatType.self),
            at: 0)
          cArr.append(
            graph.variable(.GPU(0), .WC(cfgChannels * batchSize, 1280), of: FloatType.self))
          for c in cArr {
            c.full(0)
          }
          // These values doesn't matter, it won't affect the model shape, just the input vector.
          let vector = fixedEncoder.vector(
            textEmbedding: cArr[2], originalSize: (width: 1024, height: 1024),
            cropTopLeft: (top: 0, left: 0),
            targetSize: (width: 1024, height: 1024), aestheticScore: 6,
            negativeOriginalSize: (width: 768, height: 768),
            negativeAestheticScore: 2.5, fpsId: 5, motionBucketId: 127, condAug: 0.02)
          conditions =
            vector
            + fixedEncoder.encode(
              isCfgEnabled: true, textGuidanceScale: 3.5, guidanceEmbed: 3.5,
              isGuidanceEmbedEnabled: false, distilledGuidanceLayers: 0, modifier: .none,
              textEncoding: cArr, timesteps: [0], audioShiftRatio: 1, batchSize: batchSize,
              startHeight: startHeight,
              startWidth: startWidth, tokenLengthUncond: 77, tokenLengthCond: 77, lora: [],
              tiledDiffusion: tiledDiffusion, teaCache: teaCache, isBF16: false,
              injectedControls: [], referenceImages: [], referenceAudios: []
            ).0  // No need to pass lora, one off use.
        }
        let _ = unet.compileModel(
          filePath: modelPath, externalOnDemand: externalOnDemand,
          deviceProperties: DeviceCapability.deviceProperties,
          version: modelVersion, modifier: .none, qkNorm: qkNorm,
          dualAttentionLayers: dualAttentionLayers,
          upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
          usesSolAttention: false, solAttentionStart: 0, solAttentionTau: 0,
          injectControlsAndAdapters: InjectControlsAndAdapters<FloatType>(
            injectControls: false, injectT2IAdapters: false, injectAttentionKV: false,
            injectIPAdapterLengths: [], injectControlModels: []), lora: lora,
          isQuantizedModel: isQuantizedModel,
          canRunLoRASeparately: canRunLoRASeparately,
          inputs: x, t, conditions, tokenLengthUncond: 77, tokenLengthCond: 77, isCfgEnabled: true,
          extraProjection: nil,
          injectedControlsAndAdapters: InjectedControlsAndAdapters<FloatType>(
            injectedControls: [], injectedT2IAdapters: [], injectedIPAdapters: [],
            injectedAttentionKVs: []), referenceImageCount: 0, referenceAudioCount: 0,
          tiledDiffusion: tiledDiffusion,
          teaCache: teaCache, causalInference: (0, 0), isBF16: false,
          activationQkScaling: [:], activationProjScaling: [:],
          activationFfnProjUpScaling: [:], activationFfnScaling: [:],
          weightsCache: weightsCache
        )
        unetFilePath = modelPath
        unetExternalOnDemand = externalOnDemand
        unetInjectControls = false
        unetInjectT2IAdapters = false
        unetInjectIPAdapterLengths = []
        unetTiledDiffusion = tiledDiffusion
        unetGuidanceEmbed = false
        unetVersion = modelVersion
        unetUpcastAttention = upcastAttention
        unetUsesFlashAttention = usesFlashAttention
        unetUsesSolAttention = false
        unetTokenLengthUncond = 77
        unetTokenLengthCond = 77
        unetModifier = modelModifier
        unetScale = imageScale
      }
      // We only load UNet for CoreML only mode.
      guard mode != .coreml && mode != .unet else {
        return
      }
      if firstStageDecoder == nil {
        // Configuration only works for SDXL / SD v1.x / SSD-1B
        let decodingTileSize = (
          width: min(tiledDecoding.tileSize.width * 8, startWidth),
          height: min(tiledDecoding.tileSize.height * 8, startHeight)
        )
        let tiledDecodingIsEnabled =
          tiledDecoding.isEnabled
          && (startWidth > decodingTileSize.width || startHeight > decodingTileSize.height)
        let startWidth = tiledDecodingIsEnabled ? decodingTileSize.width : startWidth
        let startHeight = tiledDecodingIsEnabled ? decodingTileSize.height : startHeight
        let externalOnDemand = externalOnDemand(
          version: modelVersion, scale: imageScale, variant: .autoencoder, injectedControls: 0)
        let z = graph.variable(.GPU(0), .NHWC(1, startHeight, startWidth, 4), of: FloatType.self)
        let decoder = Decoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, inputChannels: 4,
          highPrecisionKeysAndValues: highPrecisionForAutoencoder, usesFlashAttention: false,
          paddingFinalConvLayer: true, format: .NHWC
        ).0
        decoder.compile(inputs: z)
        graph.openStore(
          autoencoderPath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: autoencoderPath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, .externalData])
        }
        firstStageDecoder = decoder
        firstStageDecoderScale = imageScale
        firstStageDecoderFilePath = autoencoderPath
        firstStageDecoderExternalOnDemand = externalOnDemand
        firstStageDecoderVersion = modelVersion
        firstStageDecoderHighPrecision = highPrecisionForAutoencoder
        firstStageDecoderTiledDecoding = tiledDecoding
      }
      if firstStageEncoder == nil {
        let startHeight =
          tiledDiffusion.isEnabled
          ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
        let startWidth =
          tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
        let externalOnDemand = externalOnDemand(
          version: modelVersion, scale: imageScale, variant: .autoencoder, injectedControls: 0)
        let x = graph.variable(
          .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3), of: FloatType.self)
        let encoder = Encoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, usesFlashAttention: false, format: .NHWC
        ).0
        encoder.compile(inputs: x)
        graph.openStore(
          autoencoderPath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: autoencoderPath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, .externalData])
        }
        firstStageEncoder = encoder
        firstStageEncoderScale = imageScale
        firstStageEncoderFilePath = autoencoderPath
        firstStageEncoderExternalOnDemand = externalOnDemand
        firstStageEncoderVersion = modelVersion
        firstStageEncoderHighPrecision = highPrecisionForAutoencoder
        firstStageEncoderTiledDiffusion = tiledDiffusion
      }
      if textModel == nil {
        let textModelToLoad: Model
        let textModelLoRAPrefix: String
        switch modelVersion {
        case .v1:
          textModelToLoad =
            CLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 768,
              numLayers: 13 - min(max(clipSkip, 1), 12),
              numHeads: 12, batchSize: 2, intermediateSize: 3072,
              usesFlashAttention: useMFA > 0 && DeviceCapability.isMFACausalAttentionMaskSupported
            ).0
          textModelLoRAPrefix = ""
        case .v2:
          textModelToLoad =
            OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1024,
              numLayers: 24 - min(max(clipSkip, 1), 23),
              numHeads: 16, batchSize: 2, intermediateSize: 4096,
              usesFlashAttention: useMFA > 0 && DeviceCapability.isMFACausalAttentionMaskSupported
            ).0
          textModelLoRAPrefix = ""
        case .sdxlBase, .sdxlRefiner, .ssd1b:
          textModelToLoad =
            OpenCLIPTextModel(
              FloatType.self, injectEmbeddings: false,
              vocabularySize: 49408, maxLength: 77, maxTokenLength: 77, embeddingSize: 1280,
              numLayers: 32 - min(max(clipSkip - 1, 0), 30), numHeads: 20, batchSize: 2,
              intermediateSize: 5120,
              usesFlashAttention: useMFA > 0 && DeviceCapability.isMFACausalAttentionMaskSupported,
              outputPenultimate: true
            ).0
          textModelLoRAPrefix = "__te2"
        case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v, .wurstchenStageC,
          .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1,
          .qwenImage, .qwenImage2_1, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage, .flux2,
          .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2,
          .longcatVideoAvatar1_5, .minimaxH3:
          fatalError()
        }
        let tokensTensor = graph.variable(.GPU(0), .C(2 * 77), of: Int32.self)
        let positionTensor = graph.variable(.GPU(0), .C(2 * 77), of: Int32.self)
        let causalAttentionMask = graph.variable(.GPU(0), .NHWC(1, 1, 77, 77), of: FloatType.self)
        textModelToLoad.compile(inputs: tokensTensor, positionTensor, causalAttentionMask)
        graph.openStore(
          textEncoderPath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: textEncoderPath)
        ) { store in
          if lora.count > 0 {
            LoRALoader.openStore(graph, lora: lora) { loader in
              if clipSkip > 1 {
                store.read("text_model", model: textModelToLoad, codec: [.jit, .externalData]) {
                  name, dataType, _, shape in
                  // Retrieve the right final layer norm parameters.
                  var name = name
                  switch modelVersion {
                  case .v1:
                    if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                      name = "__text_model__[t-98-0]"
                    } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                      name = "__text_model__[t-98-1]"
                    }
                  case .v2:
                    if name == "__text_model__[t-\(186 - (min(clipSkip, 23) - 1) * 8)-0]" {
                      name = "__text_model__[t-186-0]"
                    } else if name == "__text_model__[t-\(186 - (min(clipSkip, 23) - 1) * 8)-1]" {
                      name = "__text_model__[t-186-1]"
                    }
                  case .sdxlBase, .sdxlRefiner, .ssd1b:
                    if name == "__text_model__[t-\(258 - (min(clipSkip, 31) - 1) * 8)-0]" {
                      name = "__text_model__[t-258-0]"
                    } else if name == "__text_model__[t-\(258 - (min(clipSkip, 31) - 1) * 8)-1]" {
                      name = "__text_model__[t-258-1]"
                    }
                  case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v,
                    .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
                    .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .cosmos2_5_2b, .wan22_5b,
                    .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b,
                    .seedvr2_7b, .ideogram4, .krea2, .longcatVideoAvatar1_5, .minimaxH3:
                    fatalError()
                  }
                  return loader.mergeLoRA(
                    graph, name: name, store: store, dataType: dataType, shape: shape,
                    of: FloatType.self, prefix: textModelLoRAPrefix)
                }
              } else {
                store.read("text_model", model: textModelToLoad, codec: [.jit, .externalData]) {
                  name, dataType, _, shape in
                  return loader.mergeLoRA(
                    graph, name: name, store: store, dataType: dataType, shape: shape,
                    of: FloatType.self, prefix: textModelLoRAPrefix)
                }
              }
            }
          } else {
            if clipSkip > 1 {
              store.read("text_model", model: textModelToLoad, codec: [.jit, .externalData]) {
                name, _, _, _ in
                // Retrieve the right final layer norm parameters.
                var name = name
                switch modelVersion {
                case .v1:
                  if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-0]" {
                    name = "__text_model__[t-98-0]"
                  } else if name == "__text_model__[t-\(98 - (min(clipSkip, 12) - 1) * 8)-1]" {
                    name = "__text_model__[t-98-1]"
                  }
                case .v2:
                  if name == "__text_model__[t-\(186 - (min(clipSkip, 23) - 1) * 8)-0]" {
                    name = "__text_model__[t-186-0]"
                  } else if name == "__text_model__[t-\(186 - (min(clipSkip, 23) - 1) * 8)-1]" {
                    name = "__text_model__[t-186-1]"
                  }
                case .sdxlBase, .sdxlRefiner, .ssd1b:
                  if name == "__text_model__[t-\(258 - (min(clipSkip, 31) - 1) * 8)-0]" {
                    name = "__text_model__[t-258-0]"
                  } else if name == "__text_model__[t-\(258 - (min(clipSkip, 31) - 1) * 8)-1]" {
                    name = "__text_model__[t-258-1]"
                  }
                case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .kandinsky21, .svdI2v,
                  .wurstchenStageC, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
                  .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .cosmos2_5_2b, .wan22_5b,
                  .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b,
                  .seedvr2_7b, .ideogram4, .krea2, .longcatVideoAvatar1_5, .minimaxH3:
                  fatalError()
                }
                return .continue(name)
              }
            } else {
              store.read("text_model", model: textModelToLoad, codec: [.jit, .externalData])
            }
          }
        }
        textEncoderFilePaths = [textEncoderPath]
        textEncoderVersion = modelVersion
        textEncoderIsCfgEnabled = true
        textEncoderUsesFlashAttention =
          useMFA > 0 && DeviceCapability.isMFACausalAttentionMaskSupported
        textEncoderInjectEmbeddings = false
        textEncoderMaxLength = 77
        textEncoderClipSkip = clipSkip
        textEncoderLoRA = lora
      }
    }
  }
}

extension ModelPreloader {
  func didReceiveMemoryWarning() {
    dispatchPrecondition(condition: .onQueue(queue))
    removeAllCache()
  }

  func delayedPreloadIfPossible(activeSentinel: Int, countdown: Int) {
    // Check if we are still in a state that can preload.
    guard
      self.activeSentinel == activeSentinel
        && (mode == .preload || mode == .coreml || mode == .unet)
    else {
      return
    }
    if countdown == 0 {
      preloadIfPossible()
    } else {
      self.preloadState = .countdown(countdown)
      queue.asyncAfter(deadline: .now() + .seconds(1)) { [weak self] in
        guard let self = self else { return }
        // Make sure the preload sequence is not interrupted.
        guard self.preloadState == .countdown(countdown) else { return }
        self.delayedPreloadIfPossible(activeSentinel: activeSentinel, countdown: countdown - 1)
      }
    }
  }

  func startGenerating() {
    dispatchPrecondition(condition: .onQueue(.main))
    isGenerating = true
  }

  func stopGenerating() {
    dispatchPrecondition(condition: .onQueue(.main))
    isGenerating = false
  }

  public func offloadModels() {
    dispatchPrecondition(condition: .onQueue(.main))
    queue.async { [weak self] in
      guard let self = self else { return }
      self.removeAllCache()
      // No more preloading.
      self.preloadState = .preloadDone
    }
  }

  func newConfiguration(_ newConfiguration: GenerationConfiguration) {
    dispatchPrecondition(condition: .onQueue(queue))
    guard mode == .preload || mode == .yes || mode == .coreml || mode == .unet else { return }
    let newModelFile = newConfiguration.model ?? ModelZoo.defaultSpecification.file
    let newImageScale = DeviceCapability.Scale(
      widthScale: newConfiguration.startWidth, heightScale: newConfiguration.startHeight)
    let newBatchSize = Int(newConfiguration.batchSize)
    let newClipSkip = Int(newConfiguration.clipSkip)
    let version = ModelZoo.versionForModel(newModelFile)
    let isRefinerEnabled: Bool =
      newConfiguration.refinerModel.map {
        guard $0 != newModelFile, ModelZoo.isModelDownloaded($0) else { return false }
        let refinerVersion = ModelZoo.versionForModel($0)
        guard ModelZoo.isCompatibleRefiner(version, refinerVersion: refinerVersion) else {
          return false
        }
        return true
      } ?? false
    let newLora: [LoRAConfiguration] = newConfiguration.loras.compactMap {
      guard let file = $0.file, LoRAZoo.isModelDownloaded(file),
        LoRAZoo.versionForModel(file) == version
      else { return nil }
      guard !isRefinerEnabled || $0.mode == .all || $0.mode == .base else { return nil }
      return LoRAConfiguration(
        file: LoRAZoo.filePathForModelDownloaded(file), weight: $0.weight, version: version,
        isLoHa: LoRAZoo.isLoHaForModel(file), modifier: LoRAZoo.modifierForModel(file),
        mode: .init(from: $0.mode))
    }
    let newTiledDecoding = TiledConfiguration(
      isEnabled: newConfiguration.tiledDecoding,
      tileSize: .init(
        width: Int(newConfiguration.decodingTileWidth),
        height: Int(newConfiguration.decodingTileHeight)),
      tileOverlap: Int(newConfiguration.decodingTileOverlap))
    let newTiledDiffusion = TiledConfiguration(
      isEnabled: newConfiguration.tiledDiffusion,
      tileSize: .init(
        width: Int(newConfiguration.diffusionTileWidth),
        height: Int(newConfiguration.diffusionTileHeight)),
      tileOverlap: Int(newConfiguration.diffusionTileOverlap))
    guard
      modelFile != newModelFile || imageScale != newImageScale || batchSize != newBatchSize
        || clipSkip != newClipSkip || lora != newLora || tiledDecoding != newTiledDecoding
        || tiledDiffusion != newTiledDiffusion
    else { return }
    removeAllCache()
    modelFile = newModelFile
    imageScale = newImageScale
    batchSize = newBatchSize
    clipSkip = newClipSkip
    lora = newLora
    tiledDecoding = newTiledDecoding
    tiledDiffusion = newTiledDiffusion
    // Only preload if we are in preload mode.
    if mode == .preload || mode == .coreml || mode == .unet {
      activeSentinel += 1
      delayedPreloadIfPossible(activeSentinel: activeSentinel, countdown: 5)
    }
  }

  func updateKeepModelInMemory(_ value: Int?) {
    dispatchPrecondition(condition: .onQueue(queue))
    let value = value ?? (DeviceCapability.keepModelPreloaded ? 2 : 0)
    let newMode = Mode.from(value)
    guard mode != newMode else { return }
    mode = newMode
    if mode == .preload || mode == .coreml || mode == .unet {
      activeSentinel += 1
      delayedPreloadIfPossible(activeSentinel: activeSentinel, countdown: 4)
    } else if mode == .no {
      removeAllCache()
    }
  }

  func updateUseCoreML(_ value: Bool?) {
    dispatchPrecondition(condition: .onQueue(queue))
    let useCoreML = value ?? DeviceCapability.isCoreMLSupported
    CoreMLModelManager.isCoreMLSupported.store(useCoreML, ordering: .releasing)
    guard self.useCoreML != nil && self.useCoreML != useCoreML else {
      self.useCoreML = useCoreML
      return
    }
    removeAllCache()
    self.useCoreML = useCoreML
    if mode == .preload || mode == .coreml || mode == .unet {
      // If need to preload, we do the countdown preload.
      activeSentinel += 1
      delayedPreloadIfPossible(activeSentinel: activeSentinel, countdown: 4)
    }
  }

  func updateUseAppleNeuralEngineFor8BitSModels(_ value: Bool?) {
    dispatchPrecondition(condition: .onQueue(queue))
    let useAppleNeuralEngineFor8BitSModels =
      value ?? DeviceCapability.isMFAAppleNeuralEngineFaster
    DeviceCapability.isMFAAppleNeuralEngineEnabled.store(
      useAppleNeuralEngineFor8BitSModels, ordering: .releasing)
    guard
      self.useAppleNeuralEngineFor8BitSModels != nil
        && self.useAppleNeuralEngineFor8BitSModels != useAppleNeuralEngineFor8BitSModels
    else {
      self.useAppleNeuralEngineFor8BitSModels = useAppleNeuralEngineFor8BitSModels
      return
    }
    self.useAppleNeuralEngineFor8BitSModels = useAppleNeuralEngineFor8BitSModels
  }

  func updateUseMFA(_ value: Int?) {
    dispatchPrecondition(condition: .onQueue(queue))
    let useMFA = value ?? (DeviceCapability.isMFASupported ? 1 : 0)
    DeviceCapability.isMFAEnabled.store(useMFA, ordering: .releasing)
    guard self.useMFA != nil && self.useMFA != useMFA else {
      self.useMFA = useMFA
      return
    }
    removeAllCache()
    self.useMFA = useMFA
    if mode == .preload || mode == .coreml || mode == .unet {
      // If need to preload, we do the countdown preload.
      activeSentinel += 1
      delayedPreloadIfPossible(activeSentinel: activeSentinel, countdown: 4)
    }
  }

  func updateMergeLoRA(_ value: Int?) {
    dispatchPrecondition(condition: .onQueue(queue))
    let mergeLoRA = value ?? 0
    guard self.mergeLoRA != nil && self.mergeLoRA != mergeLoRA else {
      self.mergeLoRA = mergeLoRA
      return
    }
    removeAllCache()
    self.mergeLoRA = mergeLoRA
    if mode == .preload || mode == .coreml || mode == .unet {
      // If need to preload, we do the countdown preload.
      activeSentinel += 1
      delayedPreloadIfPossible(activeSentinel: activeSentinel, countdown: 4)
    }
  }

  func updateLoRAUseCoreML(_ value: Bool?) {
    dispatchPrecondition(condition: .onQueue(queue))
    let loraUseCoreML = value ?? DeviceCapability.isLoRACoreMLSupported
    CoreMLModelManager.isLoRASupported.store(loraUseCoreML, ordering: .releasing)
    guard self.loraUseCoreML != nil && self.loraUseCoreML != loraUseCoreML else {
      self.loraUseCoreML = loraUseCoreML
      return
    }
    removeAllCache()
    self.loraUseCoreML = loraUseCoreML
    if mode == .preload || mode == .coreml || mode == .unet {
      // If need to preload, we do the countdown preload.
      activeSentinel += 1
      delayedPreloadIfPossible(activeSentinel: activeSentinel, countdown: 4)
    }
  }

  func updateWeightsCacheSize(_ value: Int?) {
    dispatchPrecondition(condition: .onQueue(queue))
    let value =
      value ?? Int(DeviceCapability.maxTotalWeightsCacheSize / (8 * 1_024 * 1_024 * 1_024))
    weightsCache.maxTotalCacheSize = UInt64(max(0, value)) * 8 * 1_024 * 1_024 * 1_024  // Set it to the right value.
  }

  func updateExternalStore(_ value: Int?) {
    dispatchPrecondition(condition: .onQueue(queue))
    let externalStore = value.flatMap { JITWeightsLoading(rawValue: $0) } ?? .automatic
    DeviceCapability.isPartialOffloadPreferred.store(
      externalStore == .alwaysPartially, ordering: .releasing)
    guard externalStore != self.externalStore else { return }
    self.externalStore = externalStore
    removeAllCache()
    if mode == .preload || mode == .coreml || mode == .unet {
      activeSentinel += 1
      delayedPreloadIfPossible(activeSentinel: activeSentinel, countdown: 4)
    }
  }

  enum SupportVariant {
    case unet
    case autoencoder
    case control
    case diffusionMapping
    case textEncoder
  }

  func externalOnDemand(
    version: ModelVersion, scale: DeviceCapability.Scale, variant: SupportVariant,
    injectedControls: Int, suffix: String? = nil, is8BitModel: Bool = false
  ) -> Bool {
    switch externalStore {
    case .alwaysFully:
      return DeviceCapability.externalOnDemand(
        version: version, scale: scale, force: true, suffix: suffix, is8BitModel: is8BitModel)
    case .never:
      return false
    case .automatic, .alwaysPartially:
      if DeviceCapability.isLowPerformance {
        return DeviceCapability.externalOnDemand(
          version: version, scale: scale, force: false, suffix: suffix,
          is8BitModel: is8BitModel)
      } else {
        switch variant {
        case .control:
          if DeviceCapability.isMaxPerformance {
            return DeviceCapability.externalOnDemand(
              version: version, scale: scale, force: false, suffix: suffix,
              is8BitModel: is8BitModel)
          } else if DeviceCapability.isHighPerformance && version == .v1 && injectedControls <= 2 {
            return DeviceCapability.externalOnDemand(
              version: version, scale: scale, force: false, suffix: suffix,
              is8BitModel: is8BitModel)
          } else if DeviceCapability.isGoodPerformance && version == .v1 && injectedControls <= 1 {
            return DeviceCapability.externalOnDemand(
              version: version, scale: scale, force: false, suffix: suffix,
              is8BitModel: is8BitModel)
          } else if version == .v2 && scale.widthScale * scale.heightScale <= 64
            && injectedControls <= 1
          {
            return DeviceCapability.externalOnDemand(
              version: version, scale: scale, force: false, suffix: suffix,
              is8BitModel: is8BitModel)
          }
          // For 3GiB / 4GiB devices, we use file backed control net regardless.
          return DeviceCapability.externalOnDemand(
            version: version, scale: scale, force: true, suffix: suffix,
            is8BitModel: is8BitModel)
        case .textEncoder:
          switch version {
          case .v1, .v2, .kandinsky21, .hiDreamO1:
            return false
          case .sd3, .sd3Large, .pixart, .auraflow, .flux1, .sdxlBase, .sdxlRefiner, .ssd1b,
            .wurstchenStageB, .wurstchenStageC, .svdI2v, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
            .hiDreamI1, .qwenImage, .qwenImage2_1, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage,
            .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4,
            .krea2, .longcatVideoAvatar1_5:
            return DeviceCapability.isLowPerformance
          case .minimaxH3:
            return true
          }
        case .unet:
          return DeviceCapability.externalOnDemand(
            version: version, scale: scale, force: false, suffix: suffix,
            is8BitModel: is8BitModel)
        case .diffusionMapping:
          if DeviceCapability.isGoodPerformance {
            return DeviceCapability.externalOnDemand(
              version: version, scale: scale, force: false, suffix: suffix,
              is8BitModel: is8BitModel)
          }
          // For 3GiB / 4GiB devices, we use file backed net regardless.
          return DeviceCapability.externalOnDemand(
            version: version, scale: scale, force: true, suffix: suffix,
            is8BitModel: is8BitModel)
        case .autoencoder:
          // If it is autoencoder and not low performance, we don't need file backed.
          return false
        }
      }
    }
  }
}

extension ModelPreloader {
  var canRunLoRASeparately: Bool {
    (mergeLoRA ?? 0) == 0
  }
}

extension ModelPreloader {
  func retrieveTextModels(textEncoder: TextEncoder<FloatType>) -> [Model?] {
    guard (mode == .preload || mode == .yes) && isEnabled else {
      textModel = nil
      return [nil]
    }
    guard let textModel = textModel, textEncoderFilePaths == textEncoder.filePaths,
      textEncoderVersion == textEncoder.version,
      textEncoderIsCfgEnabled == textEncoder.isCfgEnabled,
      textEncoderUsesFlashAttention == textEncoder.usesFlashAttention,
      textEncoderInjectEmbeddings == textEncoder.injectEmbeddings,
      textEncoderMaxLength == textEncoder.maxLength,
      textEncoderClipSkip == textEncoder.clipSkip,
      textEncoderLoRA == textEncoder.lora
    else {
      textModel = nil
      return [nil]
    }
    return [textModel]
  }
  func consumeTextModels(
    _ x: ([DynamicGraph.Tensor<FloatType>], [Model]), textEncoder: TextEncoder<FloatType>
  )
    -> [DynamicGraph.Tensor<FloatType>]
  {
    if (mode == .preload || mode == .yes) && isEnabled {
      textModel = x.1[0]
      textEncoderFilePaths = textEncoder.filePaths
      textEncoderVersion = textEncoder.version
      textEncoderIsCfgEnabled = textEncoder.isCfgEnabled
      textEncoderUsesFlashAttention = textEncoder.usesFlashAttention
      textEncoderInjectEmbeddings = textEncoder.injectEmbeddings
      textEncoderMaxLength = textEncoder.maxLength
      textEncoderClipSkip = textEncoder.clipSkip
      textEncoderLoRA = textEncoder.lora
    }
    return x.0
  }
  func retrieveFirstStageEncoder(firstStage: FirstStage<FloatType>, scale: DeviceCapability.Scale)
    -> Model?
  {
    guard (mode == .preload || mode == .yes) && isEnabled else {
      firstStageEncoder = nil
      return nil
    }
    guard let firstStageEncoder = firstStageEncoder,
      firstStageEncoderFilePath == firstStage.filePath,
      firstStageEncoderExternalOnDemand == firstStage.externalOnDemand,
      firstStageEncoderVersion == firstStage.version,
      firstStageEncoderScale == scale,
      firstStageEncoderHighPrecision == firstStage.highPrecisionKeysAndValues,
      firstStageEncoderTiledDiffusion == firstStage.tiledDiffusion
    else {
      firstStageEncoder = nil
      return nil
    }
    return firstStageEncoder
  }
  func consumeFirstStageSample(
    _ x: (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>, Model?),
    firstStage: FirstStage<FloatType>, scale: DeviceCapability.Scale
  ) -> (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>) {
    if (mode == .preload || mode == .yes) && isEnabled {
      firstStageEncoder = x.2
      firstStageEncoderFilePath = firstStage.filePath
      firstStageEncoderExternalOnDemand = firstStage.externalOnDemand
      firstStageEncoderVersion = firstStage.version
      firstStageEncoderScale = scale
      firstStageEncoderHighPrecision = firstStage.highPrecisionKeysAndValues
      firstStageEncoderTiledDiffusion = firstStage.tiledDecoding
    }
    return (x.0, x.1)
  }
  func consumeFirstStageEncode(
    _ x: (DynamicGraph.Tensor<FloatType>, Model?), firstStage: FirstStage<FloatType>,
    scale: DeviceCapability.Scale
  )
    -> DynamicGraph.Tensor<FloatType>
  {
    if (mode == .preload || mode == .yes) && isEnabled {
      firstStageEncoder = x.1
      firstStageEncoderFilePath = firstStage.filePath
      firstStageEncoderExternalOnDemand = firstStage.externalOnDemand
      firstStageEncoderVersion = firstStage.version
      firstStageEncoderScale = scale
      firstStageEncoderHighPrecision = firstStage.highPrecisionKeysAndValues
      firstStageEncoderTiledDiffusion = firstStage.tiledDecoding
    }
    return x.0
  }
  func consumeFirstStageEncode(
    _ x: ([DynamicGraph.Tensor<FloatType>], Model?), firstStage: FirstStage<FloatType>,
    scale: DeviceCapability.Scale
  )
    -> [DynamicGraph.Tensor<FloatType>]
  {
    if (mode == .preload || mode == .yes) && isEnabled {
      firstStageEncoder = x.1
      firstStageEncoderFilePath = firstStage.filePath
      firstStageEncoderExternalOnDemand = firstStage.externalOnDemand
      firstStageEncoderVersion = firstStage.version
      firstStageEncoderScale = scale
      firstStageEncoderHighPrecision = firstStage.highPrecisionKeysAndValues
      firstStageEncoderTiledDiffusion = firstStage.tiledDecoding
    }
    return x.0
  }
  func retrieveFirstStageDecoder(firstStage: FirstStage<FloatType>, scale: DeviceCapability.Scale)
    -> Model?
  {
    guard (mode == .preload || mode == .yes) && isEnabled else {
      firstStageDecoder = nil
      return nil
    }
    guard let firstStageDecoder = firstStageDecoder,
      firstStageDecoderFilePath == firstStage.filePath,
      firstStageDecoderExternalOnDemand == firstStage.externalOnDemand,
      firstStageDecoderVersion == firstStage.version,
      firstStageDecoderScale == scale,
      firstStageDecoderHighPrecision == firstStage.highPrecisionKeysAndValues,
      firstStageDecoderTiledDecoding == firstStage.tiledDecoding
    else {
      firstStageDecoder = nil
      return nil
    }
    return firstStageDecoder
  }
  func consumeFirstStageDecode(
    _ x: (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<Float>?, Model?),
    firstStage: FirstStage<FloatType>,
    scale: DeviceCapability.Scale
  )
    -> (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<Float>?)
  {
    if (mode == .preload || mode == .yes) && isEnabled {
      firstStageDecoder = x.2
      firstStageDecoderFilePath = firstStage.filePath
      firstStageDecoderExternalOnDemand = firstStage.externalOnDemand
      firstStageDecoderVersion = firstStage.version
      firstStageDecoderScale = scale
      firstStageDecoderHighPrecision = firstStage.highPrecisionKeysAndValues
      firstStageDecoderTiledDecoding = firstStage.tiledDecoding
    }
    return (x.0, x.1)
  }
  func retrieveUNet(
    sampler: any Sampler<FloatType, UNetWrapper<FloatType>>, scale: DeviceCapability.Scale,
    tokenLengthUncond: Int, tokenLengthCond: Int
  ) -> [UNetWrapper<FloatType>?] {
    guard (mode == .preload || mode == .yes || mode == .unet) && isEnabled else {
      unet = UNetWrapper()
      return [nil]
    }
    guard unet.isLoaded, unetFilePath == sampler.filePath, unetModifier == sampler.modifier,
      unetVersion == sampler.version, unetUpcastAttention == sampler.upcastAttention,
      unetUsesFlashAttention == sampler.usesFlashAttention,
      unetUsesSolAttention == sampler.usesSolAttention,
      unetExternalOnDemand == sampler.externalOnDemand, unetScale == scale,
      unetInjectControls == sampler.injectControls,
      unetInjectT2IAdapters == sampler.injectT2IAdapters,
      unetInjectIPAdapterLengths == sampler.injectIPAdapterLengths,
      unetTiledDiffusion == sampler.tiledDiffusion,
      unetGuidanceEmbed == sampler.isGuidanceEmbedEnabled,
      unetLoRA == sampler.lora, unetTokenLengthUncond == tokenLengthUncond,
      unetTokenLengthCond == tokenLengthCond
    else {
      unet = UNetWrapper()
      return [nil]
    }
    return [unet]
  }
  func consumeUNet(
    _ x: Result<SamplerOutput<FloatType, UNetWrapper<FloatType>>, Error>,
    sampler: any Sampler<FloatType, UNetWrapper<FloatType>>, scale: DeviceCapability.Scale,
    tokenLengthUncond: Int, tokenLengthCond: Int
  ) throws
    -> DynamicGraph.Tensor<FloatType>
  {
    if (mode == .preload || mode == .yes || mode == .unet) && isEnabled {
      let x = try x.get()
      unet = x.unets[0] ?? UNetWrapper()
      #if !targetEnvironment(macCatalyst)
        // If it is not mac, we need to unload this resource.
        unet.unloadResources()
      #endif
      unetFilePath = sampler.filePath
      unetModifier = sampler.modifier
      unetVersion = sampler.version
      unetUpcastAttention = sampler.upcastAttention
      unetUsesFlashAttention = sampler.usesFlashAttention
      unetUsesSolAttention = sampler.usesSolAttention
      unetExternalOnDemand = sampler.externalOnDemand
      unetInjectControls = sampler.injectControls
      unetInjectT2IAdapters = sampler.injectT2IAdapters
      unetInjectIPAdapterLengths = sampler.injectIPAdapterLengths
      unetTiledDiffusion = sampler.tiledDiffusion
      unetGuidanceEmbed = sampler.isGuidanceEmbedEnabled
      unetLoRA = sampler.lora
      unetScale = scale
      unetTokenLengthUncond = tokenLengthUncond
      unetTokenLengthCond = tokenLengthCond
      return x.x
    }
    switch sampler.version {
    case .auraflow, .flux1, .hiDreamI1, .hiDreamO1, .hunyuanVideo, .sd3, .sd3Large, .wan21_14b,
      .wan21_1_3b, .qwenImage, .qwenImage2_1, .cosmos2_5_2b, .wan22_5b, .zImage, .ernieImage,
      .flux2, .flux2_9b, .flux2_4b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2,
      .longcatVideoAvatar1_5, .minimaxH3:
      let unet: UNetWrapper<FloatType>? = {
        switch x {
        case .success(let x):
          return x.unets[0]
        case .failure(let error):
          guard case .cancelled(let unets) = error as? SamplerError else { return nil }
          return unets[0] as? UNetWrapper<FloatType>
        }
      }()
      if let unet = unet, sampler.lora.isEmpty || unet.didRunLoRASeparately, let model = unet.model
      {
        weightsCache.attach(sampler.filePath + sampler.suffix, from: model.parameters)
      }
      return try x.get().x
    case .kandinsky21, .pixart, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2,
      .wurstchenStageB, .wurstchenStageC:
      return try x.get().x
    }
  }
}

#if canImport(UIKit)
  import UIKit

  extension ModelPreloader {
    func isAppActive() -> Bool {
      if Thread.isMainThread {
        return UIApplication.shared.applicationState == .active
      } else {
        return DispatchQueue.main.sync {
          UIApplication.shared.applicationState == .active
        }
      }
    }

    func addLifecycleObservers() {
      let notificationCenter = NotificationCenter.default
      notificationCenter.addObserver(
        forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: nil
      ) { [weak self] _ in
        guard let self = self else { return }
        guard !self.isGenerating else { return }  // Ignore memory warnings during generation.
        queue.async { [weak self] in
          guard let self = self else { return }
          self.didReceiveMemoryWarning()
        }
      }
      notificationCenter.addObserver(
        forName: UIApplication.didBecomeActiveNotification, object: nil, queue: nil
      ) { [weak self] _ in
        guard let self = self else { return }
        queue.async { [weak self] in
          guard let self = self else { return }
          self.didBecomeActive()
        }
      }
      notificationCenter.addObserver(
        forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: nil
      ) { [weak self] _ in
        guard let self = self else { return }
        self.didEnterBackground()
      }
    }

    private func didEnterBackground() {
      dispatchPrecondition(condition: .onQueue(.main))
      guard !isGenerating else { return }  // Don't hold background task during generating.
      let backgroundTask = UIApplication.shared.beginBackgroundTask()
      queue.async { [weak self] in
        defer { UIApplication.shared.endBackgroundTask(backgroundTask) }
        guard let self = self else {
          return
        }
        self._didEnterBackground()
      }
    }

    private func _didEnterBackground() {
      dispatchPrecondition(condition: .onQueue(queue))
      activeSentinel += 1
      #if !targetEnvironment(macCatalyst)
        removeAllCache()  // Only remove for iOS devices.
      #endif
    }

    private func didBecomeActive() {
      dispatchPrecondition(condition: .onQueue(queue))
      activeSentinel += 1
      if mode == .preload || mode == .coreml {
        delayedPreloadIfPossible(activeSentinel: activeSentinel, countdown: 4)
      }
    }
  }

#else
  extension ModelPreloader {
    func addLifecycleObservers() {
      // No-op
    }

    func isAppActive() -> Bool {
      return true
    }

    private func didBecomeActive() {
      // Don't do any preloading if the app is in non-interactive mode
    }
  }

#endif
