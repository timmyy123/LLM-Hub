import Atomics
import Dispatch
import NNC

public enum MemoryCapacity: Comparable {
  case low
  case medium
  case high
  case veryHigh
}

public struct FirstStage<FloatType: TensorNumeric & BinaryFloatingPoint> {
  public let filePath: String
  public let version: ModelVersion
  public let highPrecisionKeysAndValues: Bool
  public let externalOnDemand: Bool
  public let tiledDecoding: TiledConfiguration
  public let tiledDiffusion: TiledConfiguration
  private let usesFlashAttention: Bool
  private let alternativeFilePath: String?
  private let alternativeDecoderVersion: AlternativeDecoderVersion?
  private let latentsScaling:
    (
      mean: [Float]?, std: [Float]?, scalingFactor: Float, shiftFactor: Float?, audioMean: [Float]?,
      audioStd: [Float]?
    )
  private let highPrecisionFallback: Bool
  private let deviceProperties: DeviceProperties
  private let isCancelled = ManagedAtomic<Bool>(false)
  public init(
    filePath: String, version: ModelVersion,
    latentsScaling: (
      mean: [Float]?, std: [Float]?, scalingFactor: Float, shiftFactor: Float?, audioMean: [Float]?,
      audioStd: [Float]?
    ),
    highPrecisionKeysAndValues: Bool, highPrecisionFallback: Bool,
    tiledDecoding: TiledConfiguration,
    tiledDiffusion: TiledConfiguration, externalOnDemand: Bool, usesFlashAttention: Bool,
    alternativeFilePath: String?, alternativeDecoderVersion: AlternativeDecoderVersion?,
    deviceProperties: DeviceProperties
  ) {
    self.filePath = filePath
    self.version = version
    self.latentsScaling = latentsScaling
    self.highPrecisionKeysAndValues = highPrecisionKeysAndValues
    self.highPrecisionFallback = highPrecisionFallback
    self.externalOnDemand = externalOnDemand
    self.tiledDecoding = tiledDecoding
    self.tiledDiffusion = tiledDiffusion
    self.usesFlashAttention = usesFlashAttention
    self.alternativeFilePath = alternativeFilePath
    self.alternativeDecoderVersion = alternativeDecoderVersion
    self.deviceProperties = deviceProperties
  }
}

private struct TemporalTiledDecodingConfiguration {
  let overlap: Int
  let lookback: Int
  // Override decoded overlap trimming and the sampling offset of the fade ramp.
  var overlapBlending: (trim: Int, offset: Float)? = nil
  // Drop the trailing frames of each interval after temporal blending.
  var frameDrop: (count: Int, interval: Int)? = nil
}

extension FirstStage {

  private func hiDreamO1Patchify(_ x: DynamicGraph.Tensor<FloatType>)
    -> DynamicGraph.Tensor<FloatType>
  {
    let shape = x.shape
    let patchSize = 32
    let imageChannels = 3
    let patchDimension = imageChannels * patchSize * patchSize
    precondition(shape[1] % patchSize == 0)
    precondition(shape[2] % patchSize == 0)
    precondition(shape[3] == imageChannels)
    return x.reshaped(
      format: .NHWC,
      shape: [
        shape[0], shape[1] / patchSize, patchSize, shape[2] / patchSize, patchSize, shape[3],
      ]
    ).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
      .NHWC(shape[0], shape[1] / patchSize, shape[2] / patchSize, patchDimension))
  }

  private func hiDreamO1Unpatchify(
    _ x: DynamicGraph.Tensor<FloatType>
  ) -> DynamicGraph.Tensor<FloatType> {
    let shape = x.shape
    let patchSize = 32
    let imageChannels = 3
    let patchDimension = imageChannels * patchSize * patchSize
    precondition(shape[3] == patchDimension)
    return x.reshaped(
      format: .NHWC,
      shape: [
        shape[0], shape[1], shape[2], imageChannels, patchSize, patchSize,
      ]
    ).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped(
      .NHWC(shape[0], shape[1] * patchSize, shape[2] * patchSize, imageChannels))
  }

  public func upsample(
    _ x: DynamicGraph.Tensor<FloatType>,
    latentsUpscaler: (filePath: String, mode: LTX2SpatialUpscalerMode)
  ) -> DynamicGraph.Tensor<FloatType> {
    precondition(version == .ltx2 || version == .ltx2_3)
    let graph = x.graph
    let shape = x.shape
    let (_, audioHeight) = LTX2ExtractAudioFramesAndHeight(shape)
    let startHeight = shape[1] - audioHeight
    var z = x[0..<shape[0], 0..<startHeight, 0..<shape[2], 0..<shape[3]].contiguous()
    let scalingFactor = latentsScaling.scalingFactor
    if let latentsMean = latentsScaling.mean, let latentsStd = latentsScaling.std,
      latentsMean.count >= 4, latentsStd.count >= 4
    {
      let mean = graph.variable(
        Tensor<FloatType>(
          latentsMean.map { FloatType($0) }, .GPU(0), .NHWC(1, 1, 1, latentsMean.count)))
      let std = graph.variable(
        Tensor<FloatType>(
          latentsStd.map { FloatType($0 / scalingFactor) }, .GPU(0),
          .NHWC(1, 1, 1, latentsStd.count)))
      z = std .* z + mean
    } else if let shiftFactor = latentsScaling.shiftFactor {
      z = z / scalingFactor + shiftFactor
    } else {
      z = z / scalingFactor
    }
    let spatialUpscaler = LTX2SpatialUpscaler3D(
      inChannels: z.shape[3], midChannels: 1024, numBlocks: 4,
      depth: shape[0], height: startHeight, width: shape[2], mode: latentsUpscaler.mode,
      format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
    ).1
    spatialUpscaler.compile(inputs: z)
    graph.openStore(
      latentsUpscaler.filePath, flags: .readOnly,
      externalStore: TensorData.externalStore(filePath: latentsUpscaler.filePath)
    ) { store in
      store.read(
        "spatial_upsampler", model: spatialUpscaler, codec: [.jit, .externalData])
    }
    return scale(spatialUpscaler(inputs: z)[0].as(of: FloatType.self))
  }

  private func decode(
    _ x: DynamicGraph.Tensor<FloatType>, decoder existingDecoder: Model?, highPrecision: Bool,
    cancellation: (@escaping () -> Void) -> Void
  )
    -> (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<Float>?, Model?)
  {
    let shape = x.shape
    let batchSize = shape[0]
    var startHeight = shape[1]
    let startWidth = shape[2]
    let graph = x.graph
    let scalingFactor = latentsScaling.scalingFactor
    var z: DynamicGraph.Tensor<FloatType>
    var audioZ: DynamicGraph.Tensor<Float>?
    switch version {
    case .minimaxH3:
      let latentFrames = shape[0]
      precondition(
        latentFrames == 1 || latentFrames == 2
          || (latentFrames >= 7 && (latentFrames - 2) % 5 == 0))
      let outputFrames = latentFrames == 1 ? 1 : (latentFrames - 2) / 5 * 17 + 5
      let audioRows =
        2
        * Int(
          (Double(outputFrames) / Double(MiniMaxH3Configuration.framesPerSecond) * 40)
            .rounded())
      let audioHeight = MiniMaxH3AudioHeight(
        videoLatentFrames: latentFrames, latentWidth: startWidth)
      startHeight -= audioHeight
      precondition(shape[3] == MiniMaxH3Configuration.videoChannels)
      let audioCapacity =
        latentFrames * audioHeight * startWidth * MiniMaxH3Configuration.videoChannels
        / MiniMaxH3Configuration.audioChannels
      let audioSource = x[
        0..<latentFrames, startHeight..<shape[1], 0..<startWidth,
        0..<MiniMaxH3Configuration.videoChannels
      ].copied().reshaped(.WC(audioCapacity, MiniMaxH3Configuration.audioChannels))
      audioZ = DynamicGraph.Tensor<Float>(
        from: audioSource[0..<audioRows, 0..<32].copied().reshaped(.HWC(2, audioRows / 2, 32)))
      z = x[0..<latentFrames, 0..<startHeight, 0..<startWidth, 0..<shape[3]].copied()
    case .ltx2, .ltx2_3:
      // If it has audio, extract the audio track.
      let (audioFrames, audioHeight) = LTX2ExtractAudioFramesAndHeight(x.shape)
      startHeight = startHeight - audioHeight
      z = x[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<shape[3]].contiguous()
      audioZ = DynamicGraph.Tensor<Float>(
        from: x[0..<batchSize, startHeight..<shape[1], 0..<startWidth, 0..<shape[3]].copied()
          .reshaped(.HWC(1, audioFrames, shape[3])))
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b,
      .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
      .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .seedvr2_3b,
      .seedvr2_7b, .longcatVideoAvatar1_5:
      z = x
      audioZ = nil
    }
    if let latentsMean = latentsScaling.mean, let latentsStd = latentsScaling.std,
      latentsMean.count >= 4, latentsStd.count >= 4
    {
      let mean = graph.variable(
        Tensor<FloatType>(
          latentsMean.map { FloatType($0) }, .GPU(0), .NHWC(1, 1, 1, latentsMean.count)))
      let std = graph.variable(
        Tensor<FloatType>(
          latentsStd.map { FloatType($0 / scalingFactor) }, .GPU(0),
          .NHWC(1, 1, 1, latentsStd.count)))
      if version == .ernieImage || version == .flux2 || version == .flux2_9b || version == .flux2_4b
        || version == .ideogram4
      {
        // FLUX.2 is different, it needs to first reshape x and then apply the scaling.
        let shape = x.shape
        z =
          (std
          .* x.reshaped(
            format: .NHWC, shape: [shape[0], shape[1] / 2, 2, shape[2] / 2, 2, shape[3]]
          ).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
            .NHWC(shape[0], shape[1] / 2, shape[2] / 2, 2 * 2 * shape[3])) + mean).reshaped(
            format: .NHWC, shape: [shape[0], shape[1] / 2, shape[2] / 2, shape[3], 2, 2]
          ).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped(format: .NHWC, shape: shape)
      } else {
        z = std .* z + mean
      }
    } else if let shiftFactor = latentsScaling.shiftFactor {
      z = z / scalingFactor + shiftFactor
    } else {
      z = z / scalingFactor
    }
    if let audioLatentsMean = latentsScaling.audioMean,
      let audioLatentsStd = latentsScaling.audioStd
    {
      let mean = graph.variable(
        Tensor<Float>(
          audioLatentsMean.map { Float($0) }, .GPU(0), .HWC(1, 1, audioLatentsMean.count)))
      let std = graph.variable(
        Tensor<Float>(
          audioLatentsStd.map { Float($0) }, .GPU(0),
          .HWC(1, 1, audioLatentsStd.count)))
      audioZ = audioZ.map { std .* $0 + mean }
    }
    let decoder: Model
    var audioDecoder = [Model]()
    var transparentDecoder: Model? = nil
    let queueWatermark = DynamicGraph.queueWatermark
    if version == .kandinsky21 || version == .hunyuanVideo {
      DynamicGraph.queueWatermark = min(1, queueWatermark)
    }
    defer {
      if version == .kandinsky21 || version == .hunyuanVideo {
        DynamicGraph.queueWatermark = queueWatermark
      }
    }
    let scaleFactorZ: Int
    let scaleFactor: Int
    switch version {
    case .v1, .v2, .sd3, .sd3Large, .pixart, .auraflow, .flux1, .sdxlBase, .sdxlRefiner, .ssd1b,
      .svdI2v, .kandinsky21, .hiDreamI1, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b,
      .ideogram4:
      scaleFactor = 8
      scaleFactorZ = 1
    case .hiDreamO1:
      scaleFactor = 32
      scaleFactorZ = 1
    case .qwenImage2_1:
      scaleFactor = 16
      scaleFactorZ = 1
    case .hunyuanVideo, .wan21_1_3b, .wan21_14b, .qwenImage, .krea2, .cosmos2_5_2b,
      .seedvr2_3b, .seedvr2_7b, .longcatVideoAvatar1_5:
      scaleFactor = 8
      scaleFactorZ = 4
    case .wan22_5b, .minimaxH3:
      scaleFactor = 16
      scaleFactorZ = 4
    case .ltx2, .ltx2_3:
      scaleFactor = 32
      scaleFactorZ = 8
    case .wurstchenStageB, .wurstchenStageC:
      scaleFactor = 4
      scaleFactorZ = 1
    }
    let tiledDecodingConfiguration = self.tiledDecoding
    var temporalTiledDecodingConfiguration: TemporalTiledDecodingConfiguration? = nil
    var decodingTileSize = (
      depth: 0,
      width: min(tiledDecodingConfiguration.tileSize.width * (64 / scaleFactor), startWidth),
      height: min(tiledDecodingConfiguration.tileSize.height * (64 / scaleFactor), startHeight)
    )
    // Cut by half because we will * 2 later.
    var decodingTileOverlap = (tiledDecodingConfiguration.tileOverlap * (64 / scaleFactor) + 1) / 2
    var tiledDecoding =
      tiledDecodingConfiguration.isEnabled
      && (startWidth > decodingTileSize.width || startHeight > decodingTileSize.height)
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand ? .externalOnDemand : .externalData
    let outputChannels: Int
    let causalAttentionMask: DynamicGraph.Tensor<Float>?
    var decoderInputs = [DynamicGraph.AnyTensor]()
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .pixart, .auraflow:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      let decoderUsesFlashAttention =
        usesFlashAttention
        && ((startWidth * startHeight >= 128 * 176 && highPrecisionKeysAndValues)
          || startWidth * startHeight >= 256 * 176)
      decoder =
        existingDecoder
        ?? Decoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, inputChannels: 4,
          highPrecisionKeysAndValues: highPrecisionKeysAndValues,
          usesFlashAttention: decoderUsesFlashAttention,
          paddingFinalConvLayer: true, format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).0
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      if let alternativeFilePath = alternativeFilePath, alternativeDecoderVersion == .transparent {
        let decoder = TransparentVAEDecoder(
          startHeight: startHeight * 8, startWidth: startWidth * 8,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          let pixels = graph.variable(
            .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3), of: Float.self)
          decoder.compile(
            inputs: pixels,
            DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          let pixels = graph.variable(
            .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3), of: FloatType.self)
          decoder.compile(inputs: pixels, z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          alternativeFilePath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: alternativeFilePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, .ezm7, externalData])
        }
        transparentDecoder = decoder
        outputChannels = 4
      } else {
        outputChannels = 3
      }
      causalAttentionMask = nil
    case .sd3, .sd3Large, .flux1, .hiDreamI1, .zImage:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      let decoderUsesFlashAttention =
        usesFlashAttention
        && ((startWidth * startHeight >= 128 * 176 && highPrecisionKeysAndValues)
          || startWidth * startHeight >= 256 * 176)
      decoder =
        existingDecoder
        ?? Decoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, inputChannels: 16,
          highPrecisionKeysAndValues: highPrecisionKeysAndValues,
          usesFlashAttention: decoderUsesFlashAttention,
          paddingFinalConvLayer: true, format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW,
          quantLayer: false
        ).0
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      if let alternativeFilePath = alternativeFilePath, alternativeDecoderVersion == .transparent {
        let decoder = TransparentVAEDecoder(
          startHeight: startHeight * 8, startWidth: startWidth * 8,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          let pixels = graph.variable(
            .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3), of: Float.self)
          decoder.compile(
            inputs: pixels,
            DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          let pixels = graph.variable(
            .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3), of: FloatType.self)
          decoder.compile(inputs: pixels, z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          alternativeFilePath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: alternativeFilePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, .ezm7, externalData])
        }
        transparentDecoder = decoder
        outputChannels = 4
      } else {
        outputChannels = 3
      }
      causalAttentionMask = nil
    case .hiDreamO1:
      // No need to do any model related work.
      return (hiDreamO1Unpatchify(x), nil, nil)
    case .minimaxH3:
      let temporalCount = 7
      decodingTileSize.depth = temporalCount
      temporalTiledDecodingConfiguration = TemporalTiledDecodingConfiguration(
        overlap: 2, lookback: 0, overlapBlending: (trim: 0, offset: 0),
        frameDrop: (count: 3, interval: 20))
      if !tiledDecodingConfiguration.isEnabled {
        // Use 256-pixel spatial tiles with 64-pixel overlap when tiling is disabled.
        decodingTileSize.height = min(startHeight, 16)
        decodingTileSize.width = min(startWidth, 16)
        decodingTileOverlap = 2
      }
      let tileHeight = decodingTileSize.height
      let tileWidth = decodingTileSize.width
      // H3 always needs temporal trimming, including when only one spatial tile is used.
      tiledDecoding = true
      let placeholder = graph.variable(
        .GPU(0), .NHWC(temporalCount, tileHeight, tileWidth, 24), of: FloatType.self)
      let zero = graph.variable(.GPU(0), .HWC(1, 1, 2_048), of: Float.self)
      zero.full(0)
      // The reference decoder normalizes coordinates within each tile, not the full canvas.
      let rotary = graph.variable(
        Tensor<FloatType>(
          from: MiniMaxH3VideoDecoderRotaryEmbedding(
            latentFrames: temporalCount, latentHeight: tileHeight, latentWidth: tileWidth)
        ).toGPU(0))
      decoderInputs = [rotary, zero]
      decoder =
        existingDecoder
        ?? MiniMaxH3VideoDecoder(
          latentFrames: temporalCount, latentHeight: tileHeight, latentWidth: tileWidth,
          hiddenSize: 2_048, layers: 36)
      decoder.maxConcurrency = .limit(1)
      guard let audioZ else { fatalError() }
      audioDecoder = [MiniMaxH3AudioDecoder(latentWidth: audioZ.shape[1])]
      audioDecoder[0].maxConcurrency = .limit(1)
      audioDecoder[0].compile(inputs: audioZ[0..<1, 0..<audioZ.shape[1], 0..<32].copied())
      if existingDecoder == nil {
        if highPrecision {
          decoder.compile(inputs: DynamicGraph.Tensor<Float>(from: placeholder), rotary, zero)
        } else {
          decoder.compile(inputs: placeholder, rotary, zero)
        }
      }
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) { store in
        if existingDecoder == nil {
          store.read(
            "video_decoder", model: decoder,
            codec: [.jit, .q6p, .q8p, .i8x, .ezm7, externalData])
        }
        store.read(
          "audio_decoder", model: audioDecoder[0],
          codec: [.jit, .q6p, .q8p, .i8x, .ezm7, externalData])
      }
      outputChannels = 3
      causalAttentionMask = nil
    case .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ideogram4:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      let decoderUsesFlashAttention =
        usesFlashAttention
        && ((startWidth * startHeight >= 128 * 176 && highPrecisionKeysAndValues)
          || startWidth * startHeight >= 256 * 176)
      decoder =
        existingDecoder
        ?? Decoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, inputChannels: 32,
          highPrecisionKeysAndValues: highPrecisionKeysAndValues,
          usesFlashAttention: decoderUsesFlashAttention,
          paddingFinalConvLayer: true, format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW,
          quantLayer: true, specializingNames: true
        ).0
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData]) { name, _, _, _ in
            guard let firstIndex = name.firstIndex(of: "[") else {
              return .continue(name)
            }
            let extractedName = String(
              name[name.index(after: firstIndex)..<name.index(before: name.endIndex)])
            return .continue(
              "__decoder__[\(DecoderSpecializingNamesMapping[extractedName] ?? extractedName)]")
          }
        }
      }
      if let alternativeFilePath = alternativeFilePath, alternativeDecoderVersion == .transparent {
        let decoder = TransparentVAEDecoder(
          startHeight: startHeight * 8, startWidth: startWidth * 8,
          usesFlashAttention: usesFlashAttention ? .scaleMerged : .none)
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          let pixels = graph.variable(
            .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3), of: Float.self)
          decoder.compile(
            inputs: pixels,
            DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          let pixels = graph.variable(
            .GPU(0), .NHWC(1, startHeight * 8, startWidth * 8, 3), of: FloatType.self)
          decoder.compile(inputs: pixels, z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          alternativeFilePath, flags: .readOnly,
          externalStore: TensorData.externalStore(filePath: alternativeFilePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, .ezm7, externalData])
        }
        transparentDecoder = decoder
        outputChannels = 4
      } else {
        outputChannels = 3
      }
      causalAttentionMask = nil
    case .hunyuanVideo:
      temporalTiledDecodingConfiguration = TemporalTiledDecodingConfiguration(
        overlap: 5, lookback: 0)
      var startDepth = shape[0]
      var startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      var startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      let sizeLimit = deviceProperties.memoryCapacity >= .high ? 32 : 20
      decodingTileSize.depth = 15
      if startWidth > sizeLimit || startHeight > sizeLimit || startDepth > decodingTileSize.depth {
        // We turn on tiled decoding forcefully.
        if !tiledDecoding {
          decodingTileOverlap = 4
        }
        tiledDecoding = true
        startWidth = min(startWidth, sizeLimit)
        startHeight = min(startHeight, sizeLimit)
        decodingTileSize.width = startWidth
        decodingTileSize.height = startHeight
        startDepth = min(startDepth, decodingTileSize.depth)
        decodingTileSize.depth = startDepth
      }
      decoder =
        existingDecoder
        ?? DecoderCausal3D(
          channels: [128, 256, 512, 512], numRepeat: 2, startWidth: startWidth,
          startHeight: startHeight, startDepth: startDepth, paddingFinalConvLayer: true,
          format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).1
      var mask = Tensor<Float>(
        Array(repeating: 0, count: startDepth * startDepth), .CPU,
        .NHWC(startDepth, 1, startDepth, 1))
      for i in 0..<(startDepth - 1) {
        for j in (i + 1)..<startDepth {
          mask[i, 0, j, 0] = -Float.greatestFiniteMagnitude
        }
      }
      causalAttentionMask = graph.variable(mask.toGPU(0))
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: [
              DynamicGraph.Tensor<Float>(
                from: z[0..<startDepth, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
            ] + (causalAttentionMask.map { [$0] } ?? []))
        } else {
          decoder.compile(
            inputs: [z[0..<startDepth, 0..<startHeight, 0..<startWidth, 0..<shape[3]]]
              + (causalAttentionMask.map { [DynamicGraph.Tensor<FloatType>(from: $0)] } ?? []))
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 3
    case .qwenImage2_1:
      let height = tiledDecoding ? decodingTileSize.height : startHeight
      let width = tiledDecoding ? decodingTileSize.width : startWidth
      let decoderUsesFlashAttention = usesFlashAttention && width * height >= 256 * 176
      decoder =
        existingDecoder
        ?? QwenImage2_1Decoder(
          channels: [1152, 1152, 576, 288, 144], height: height, width: width,
          usesFlashAttention: decoderUsesFlashAttention)
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(1)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<height, 0..<width, 0..<64].copied()))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<height, 0..<width, 0..<64].copied())
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 4
      causalAttentionMask = nil
    case .wan21_1_3b, .wan21_14b, .qwenImage, .krea2, .cosmos2_5_2b,
      .longcatVideoAvatar1_5:
      let startDepth =
        version == .qwenImage || version == .krea2 || version == .cosmos2_5_2b ? 1 : shape[0]
      var startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      var startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      let sizeLimit: Int
      if startDepth > 1 {
        switch deviceProperties.memoryCapacity {
        case .high, .veryHigh:
          sizeLimit = 1024  // Practically unlimited.
        case .medium:
          sizeLimit = 104
        case .low:
          sizeLimit = 32
        }
      } else {
        sizeLimit = 1024
      }
      if startWidth > sizeLimit || startHeight > sizeLimit {
        // We turn on tiled decoding forcefully.
        if !tiledDecoding {
          decodingTileOverlap = 4
        }
        tiledDecoding = true
        startWidth = min(startWidth, sizeLimit)
        startHeight = min(startHeight, sizeLimit)
        decodingTileSize.width = startWidth
        decodingTileSize.height = startHeight
        decodingTileSize.depth = startDepth
      }
      decoder =
        existingDecoder
        ?? WanDecoderCausal3D(
          channels: [96, 192, 384, 384], numRepeat: 2, startWidth: startWidth,
          startHeight: startHeight, startDepth: startDepth, paddingFinalConvLayer: true,
          wan22: false,
          outputChannels: alternativeDecoderVersion == .transparent ? 4 : 3,
          highPrecisionFinalNorm: alternativeDecoderVersion == .transparent,
          format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).1
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<startDepth, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(
            inputs: z[0..<startDepth, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          if startDepth > 1 {
            store.read("decoder", model: decoder, codec: [.jit, externalData])
          } else {
            store.read("decoder", model: decoder, codec: [.jit, externalData]) {
              name, dataType, format, shape in
              guard
                name.hasSuffix("-0]")
                  && (name.hasPrefix("__decoder__[t-conv_in-")
                    || name.hasPrefix("__decoder__[t-conv_out-")
                    || name.hasPrefix("__decoder__[t-resnet_conv1-")
                    || name.hasPrefix("__decoder__[t-resnet_conv2-"))
              else {
                return .continue(name)
              }
              guard
                var tensor =
                  (store.read(name, kind: .CPU, codec: [.jit, externalData]).map {
                    Tensor<FloatType>(from: $0)
                  })
              else {
                return .continue(name)
              }
              let channels = tensor.shape.reduce(1, *) / (3 * 3 * 3)  // Input channels and output channels.
              let outChannels = shape.reduce(1, *) / (3 * 3)
              tensor = tensor.reshaped(.NCHW(channels, 3, 3, 3))
              // In case depth = 1, it reduces to 2D convolution.
              guard channels < outChannels else {
                if dataType == .Float16 {
                  return .final(tensor[0..<channels, 2..<3, 0..<3, 0..<3].contiguous())
                } else {
                  return .final(
                    Tensor<Float>(from: tensor[0..<channels, 2..<3, 0..<3, 0..<3].contiguous()))
                }
              }
              // In case we needs more, allocate first.
              var outTensor = Tensor<FloatType>(
                Array(repeating: 0, count: outChannels * 3 * 3), .CPU, .NCHW(outChannels, 1, 3, 3))
              outTensor[0..<channels, 0..<1, 0..<3, 0..<3] = tensor[
                0..<channels, 2..<3, 0..<3, 0..<3
              ].contiguous()
              if dataType == .Float16 {
                return .final(outTensor)
              } else {
                return .final(Tensor<Float>(from: outTensor))
              }
            }
          }
        }
      }
      outputChannels = alternativeDecoderVersion == .transparent ? 4 : 3
      causalAttentionMask = nil
    case .wan22_5b:
      let startDepth = shape[0]
      var startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      var startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      let sizeLimit: Int
      if startDepth > 1 {
        switch deviceProperties.memoryCapacity {
        case .high, .veryHigh:
          sizeLimit = 1024  // Practically unlimited.
        case .medium:
          sizeLimit = 104
        case .low:
          sizeLimit = 32
        }
      } else {
        sizeLimit = 1024
      }
      if startWidth > sizeLimit || startHeight > sizeLimit {
        // We turn on tiled decoding forcefully.
        if !tiledDecoding {
          decodingTileOverlap = 4
        }
        tiledDecoding = true
        startWidth = min(startWidth, sizeLimit)
        startHeight = min(startHeight, sizeLimit)
        decodingTileSize.width = startWidth
        decodingTileSize.height = startHeight
        decodingTileSize.depth = startDepth
      }
      decoder =
        existingDecoder
        ?? WanDecoderCausal3D(
          channels: [256, 512, 1024, 1024], numRepeat: 2, startWidth: startWidth,
          startHeight: startHeight, startDepth: startDepth, paddingFinalConvLayer: true,
          wan22: true, outputChannels: 3, highPrecisionFinalNorm: false,
          format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).1
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<startDepth, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(
            inputs: z[0..<startDepth, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          if startDepth > 1 {
            store.read("decoder", model: decoder, codec: [.jit, externalData])
          } else {
            store.read("decoder", model: decoder, codec: [.jit, externalData]) {
              name, dataType, format, shape in
              guard
                name.hasSuffix("-0]")
                  && (name.hasPrefix("__decoder__[t-conv_in-")
                    || name.hasPrefix("__decoder__[t-conv_out-")
                    || name.hasPrefix("__decoder__[t-resnet_conv1-")
                    || name.hasPrefix("__decoder__[t-resnet_conv2-"))
              else {
                return .continue(name)
              }
              guard
                var tensor =
                  (store.read(name, kind: .CPU, codec: [.jit, externalData]).map {
                    Tensor<FloatType>(from: $0)
                  })
              else {
                return .continue(name)
              }
              let channels = tensor.shape.reduce(1, *) / (3 * 3 * 3)  // Input channels and output channels.
              let outChannels = shape.reduce(1, *) / (3 * 3)
              tensor = tensor.reshaped(.NCHW(channels, 3, 3, 3))
              // In case depth = 1, it reduces to 2D convolution.
              guard channels < outChannels else {
                return .final(tensor[0..<channels, 2..<3, 0..<3, 0..<3].contiguous())
              }
              // In case we needs more, allocate first.
              var outTensor = Tensor<FloatType>(
                Array(repeating: 0, count: outChannels * 3 * 3), .CPU, .NCHW(outChannels, 1, 3, 3))
              outTensor[0..<channels, 0..<1, 0..<3, 0..<3] = tensor[
                0..<channels, 2..<3, 0..<3, 0..<3
              ].contiguous()
              return .final(outTensor)
            }
          }
        }
      }
      outputChannels = 3
      causalAttentionMask = nil
    case .ltx2, .ltx2_3:
      temporalTiledDecodingConfiguration = TemporalTiledDecodingConfiguration(
        overlap: 3, lookback: 1)
      var startDepth = shape[0]
      var startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      var startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      if let audioZ = audioZ {
        let shape = audioZ.shape
        let decodedAudioWidth = (shape[1] - 1) * 4 + 1
        let vocoderLoader: (() -> Void)?
        if version == .ltx2_3 {
          let (_, vocoder, vocoderLoaderLocal) = LTX2VocoderWithBWE(width: decodedAudioWidth)
          audioDecoder = [
            LTX2AudioDecoderCausal2D(
              channels: [128, 256, 512], numRepeat: 3, startWidth: 16, startHeight: shape[1]
            ).1,
            vocoder,
          ]
          vocoderLoader = vocoderLoaderLocal
        } else {
          audioDecoder = [
            LTX2AudioDecoderCausal2D(
              channels: [128, 256, 512], numRepeat: 3, startWidth: 16, startHeight: shape[1]
            ).1,
            LTX2Vocoder(
              width: decodedAudioWidth, initialChannels: 1024,
              layers: [
                (channels: 512, kernelSize: 16, stride: 6, padding: 5),
                (channels: 256, kernelSize: 15, stride: 5, padding: 5),
                (channels: 128, kernelSize: 8, stride: 2, padding: 3),
                (channels: 64, kernelSize: 4, stride: 2, padding: 1),
                (channels: 32, kernelSize: 4, stride: 2, padding: 1),
              ], resblockKernelSizes: [3, 7, 11],
              resblockDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]], activation: .leakyReLU,
              finalConvBias: true, finalActivation: .tanh
            ).1,
          ]
          vocoderLoader = nil
        }
        audioDecoder[0].maxConcurrency = .limit(4)
        audioDecoder[0].compile(inputs: audioZ)
        let decodedAudio = graph.variable(
          .GPU(0),
          format: .NCHW,
          shape: [1, 128, 1, decodedAudioWidth],
          of: Float.self)
        audioDecoder[1].maxConcurrency = .limit(4)
        audioDecoder[1].compile(inputs: decodedAudio)
        vocoderLoader?()
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("audio_decoder", model: audioDecoder[0], codec: [.jit, externalData])
          $0.read("vocoder", model: audioDecoder[1], codec: [.jit, externalData])
        }
      }
      let sizeLimit: Int
      switch deviceProperties.memoryCapacity {
      case .high, .veryHigh:
        sizeLimit = 24
        decodingTileSize.depth = 11
      case .medium:
        sizeLimit = 16
        decodingTileSize.depth = 11
      case .low:
        sizeLimit = 12
        decodingTileSize.depth = 9
      }
      if startWidth > sizeLimit || startHeight > sizeLimit || startDepth > decodingTileSize.depth {
        // We turn on tiled decoding forcefully.
        if !tiledDecoding {
          decodingTileOverlap = 4
          // Shrink each tile only until it would need an extra tile under the current overlap math.
          func optimizedTileSize(_ length: Int) -> Int {
            guard length > sizeLimit else { return length }
            let coveredLength = max(1, length - decodingTileOverlap * 2)
            let maxProgress = max(1, sizeLimit - decodingTileOverlap * 2)
            let tileCount = (coveredLength + maxProgress - 1) / maxProgress
            let minimumProgress = (coveredLength + tileCount - 1) / tileCount
            let minimumTileSize = minimumProgress + decodingTileOverlap * 2
            return min(sizeLimit, ((minimumTileSize + 1) / 2) * 2)
          }
          startWidth = optimizedTileSize(startWidth)
          startHeight = optimizedTileSize(startHeight)
        } else {
          startWidth = min(startWidth, sizeLimit)
          startHeight = min(startHeight, sizeLimit)
        }
        tiledDecoding = true
        startDepth = min(startDepth, decodingTileSize.depth)
        decodingTileSize.width = startWidth
        decodingTileSize.height = startHeight
        decodingTileSize.depth = startDepth
      }
      let decoderLayers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))]
      let decoderPaddingMode: LTX2VideoVAEPaddingMode
      let decoderResidual: Bool
      if version == .ltx2_3 {
        decoderLayers = [
          (channels: 1024, numRepeat: 2, stride: (1, 1, 1)),
          (channels: 512, numRepeat: 2, stride: (2, 2, 2)),
          (channels: 512, numRepeat: 4, stride: (2, 2, 2)),
          (channels: 256, numRepeat: 6, stride: (2, 1, 1)),
          (channels: 128, numRepeat: 4, stride: (1, 2, 2)),
        ]
        decoderPaddingMode = .zero
        decoderResidual = false
      } else {
        decoderLayers = [
          (channels: 1024, numRepeat: 5, stride: (1, 1, 1)),
          (channels: 512, numRepeat: 5, stride: (2, 2, 2)),
          (channels: 256, numRepeat: 5, stride: (2, 2, 2)),
          (channels: 128, numRepeat: 5, stride: (2, 2, 2)),
        ]
        decoderPaddingMode = .reflect
        decoderResidual = true
      }
      decoder =
        existingDecoder
        ?? LTX2VideoDecoderCausal3D(
          layers: decoderLayers, startWidth: startWidth, startHeight: startHeight,
          startDepth: startDepth, paddingMode: decoderPaddingMode, residual: decoderResidual,
          paddingFinalConvLayer: true, format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).1
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(from: z))
        } else {
          decoder.compile(inputs: z)
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 3
      causalAttentionMask = nil
    case .kandinsky21:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      decoder =
        existingDecoder
        ?? MOVQDecoderKandinsky(
          zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2,
          startHeight: startHeight, startWidth: startWidth, attnResolutions: Set([32]))
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("movq", model: decoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 3
      causalAttentionMask = nil
    case .seedvr2_3b, .seedvr2_7b:
      let startDepth = shape[0]
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      let decoderUsesFlashAttention = usesFlashAttention && (startWidth * startHeight >= 256 * 176)
      decoder =
        existingDecoder
        ?? SeedVR2Decoder3D(
          startDepth: startDepth, startHeight: startHeight, startWidth: startWidth,
          usesFlashAttention: decoderUsesFlashAttention)
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(inputs: DynamicGraph.Tensor<Float>(from: z))
        } else {
          decoder.compile(inputs: z)
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          if startDepth > 1 {
            store.read("decoder", model: decoder, codec: [.jit, externalData])
          } else {
            store.read("decoder", model: decoder, codec: [.jit, externalData]) {
              name, dataType, _, shape in
              SeedVR2Conv3DToConv2D(
                graph: graph, name: name, dataType: dataType, shape: shape, store: store,
                externalData: externalData, modelPrefix: "decoder", of: FloatType.self)
                ?? .continue(name)
            }
          }
        }
      }
      outputChannels = 3
      causalAttentionMask = nil
    case .wurstchenStageC, .wurstchenStageB:
      let startWidth = tiledDecoding ? decodingTileSize.width : startWidth
      let startHeight = tiledDecoding ? decodingTileSize.height : startHeight
      decoder =
        existingDecoder
        ?? WurstchenStageADecoder(batchSize: 1, height: startHeight * 2, width: startWidth * 2).0
      if existingDecoder == nil {
        decoder.maxConcurrency = .limit(4)
        if highPrecision {
          decoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]]))
        } else {
          decoder.compile(inputs: z[0..<1, 0..<startHeight, 0..<startWidth, 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("decoder", model: decoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 3
      causalAttentionMask = nil
    }
    isCancelled.store(false, ordering: .releasing)
    let isCancelled = isCancelled
    cancellation {
      isCancelled.store(true, ordering: .releasing)
      decoder.cancel()
      audioDecoder.forEach {
        $0.cancel()
      }
    }
    // Hunyuan / Wan just do the decoding with the batch.
    guard
      batchSize > 1 && version != .hunyuanVideo && version != .wan21_1_3b && version != .wan21_14b
        && version != .wan22_5b && version != .ltx2 && version != .ltx2_3
        && version != .longcatVideoAvatar1_5 && version != .minimaxH3
    else {
      let audio: DynamicGraph.Tensor<Float>?
      if let audioZ = audioZ {
        switch version {
        case .minimaxH3:
          var channels = [DynamicGraph.Tensor<Float>]()
          for channel in 0..<2 {
            channels.append(
              audioDecoder[0](
                inputs: audioZ[
                  channel..<(channel + 1), 0..<audioZ.shape[1], 0..<32
                ].copied())[0].as(of: Float.self).reshaped(.NC(1, audioZ.shape[1] * 800)))
          }
          audio = Functional.concat(axis: 0, channels[0], channels[1])
        case .ltx2, .ltx2_3:
          if audioDecoder.count > 1 {
            let decodedAudio = audioDecoder[0](inputs: audioZ)[0].as(of: Float.self)
            let waveform = audioDecoder[1](inputs: decodedAudio)[0].as(of: Float.self)
            audio = waveform.reshaped(.NC(waveform.shape[1], waveform.shape[3]))
          } else {
            audio = nil
          }
        case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
          .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b,
          .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
          .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .seedvr2_3b,
          .seedvr2_7b, .longcatVideoAvatar1_5:
          audio = nil
        }
      } else {
        audio = nil
      }
      if highPrecision {
        let result: DynamicGraph.Tensor<Float>
        if tiledDecoding {
          result = tiledDecode(
            DynamicGraph.Tensor<Float>(from: z), causalAttentionMask: causalAttentionMask,
            decoder: decoder,
            transparentDecoder: transparentDecoder, tileSize: decodingTileSize,
            tileOverlap: decodingTileOverlap, outputChannels: outputChannels,
            scaleFactor: (scaleFactor, scaleFactorZ),
            temporalTiledDecodingConfiguration: temporalTiledDecodingConfiguration,
            inputs: decoderInputs
          )
        } else {
          result = internalDecode(
            DynamicGraph.Tensor<Float>(from: z), causalAttentionMask: causalAttentionMask,
            decoder: decoder, transparentDecoder: transparentDecoder, inputs: decoderInputs)
        }
        let shape = result.shape
        return (
          DynamicGraph.Tensor<FloatType>(
            from: result[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<outputChannels].contiguous()),
          audio,
          decoder
        )
      } else {
        let result: DynamicGraph.Tensor<FloatType>
        if tiledDecoding {
          result = tiledDecode(
            z,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, decoder: decoder, transparentDecoder: transparentDecoder, tileSize: decodingTileSize,
            tileOverlap: decodingTileOverlap, outputChannels: outputChannels,
            scaleFactor: (scaleFactor, scaleFactorZ),
            temporalTiledDecodingConfiguration: temporalTiledDecodingConfiguration,
            inputs: decoderInputs
          )
        } else {
          result = internalDecode(
            z,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, decoder: decoder, transparentDecoder: transparentDecoder, inputs: decoderInputs)
        }
        let shape = result.shape
        return (
          result[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<outputChannels].copied(), audio,
          decoder
        )
      }
    }
    var result = graph.variable(
      .GPU(0),
      .NHWC(batchSize, startHeight * scaleFactor, startWidth * scaleFactor, outputChannels),
      of: FloatType.self)
    for i in 0..<batchSize {
      let zEnc = z[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<shape[3]].copied()
      if highPrecision {
        let partial: DynamicGraph.Tensor<Float>
        if tiledDecoding {
          partial = tiledDecode(
            DynamicGraph.Tensor<Float>(from: zEnc), causalAttentionMask: causalAttentionMask,
            decoder: decoder,
            transparentDecoder: transparentDecoder, tileSize: decodingTileSize,
            tileOverlap: decodingTileOverlap, outputChannels: outputChannels,
            scaleFactor: (scaleFactor, scaleFactorZ),
            temporalTiledDecodingConfiguration: temporalTiledDecodingConfiguration
          )
        } else {
          partial = internalDecode(
            DynamicGraph.Tensor<Float>(from: zEnc), causalAttentionMask: causalAttentionMask,
            decoder: decoder,
            transparentDecoder: transparentDecoder)
        }
        let shape = partial.shape
        result[
          i..<(i + 1), 0..<(startHeight * scaleFactor), 0..<(startWidth * scaleFactor),
          0..<outputChannels] =
          DynamicGraph
          .Tensor<FloatType>(
            from: partial[0..<1, 0..<shape[1], 0..<shape[2], 0..<outputChannels].contiguous())
      } else {
        let partial: DynamicGraph.Tensor<FloatType>
        if tiledDecoding {
          partial = tiledDecode(
            zEnc,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, decoder: decoder, transparentDecoder: transparentDecoder,
            tileSize: decodingTileSize, tileOverlap: decodingTileOverlap,
            outputChannels: outputChannels, scaleFactor: (scaleFactor, scaleFactorZ),
            temporalTiledDecodingConfiguration: temporalTiledDecodingConfiguration)
        } else {
          partial = internalDecode(
            zEnc,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, decoder: decoder, transparentDecoder: transparentDecoder)
        }
        let shape = partial.shape
        result[
          i..<(i + 1), 0..<(startHeight * scaleFactor), 0..<(startWidth * scaleFactor),
          0..<outputChannels] =
          partial[0..<1, 0..<shape[1], 0..<shape[2], 0..<outputChannels]
      }
    }
    return (result, nil, decoder)
  }

  public func decode(
    _ x: DynamicGraph.Tensor<FloatType>, decoder existingDecoder: Model?,
    cancellation: (@escaping () -> Void) -> Void
  )
    -> (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<Float>?, Model?)
  {
    func rescalePixels(_ result: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
      switch version {
      case .minimaxH3:
        let graph = result.graph
        let mean = graph.variable(
          Tensor<FloatType>([0.485, 0.456, 0.406], .GPU(0), .NHWC(1, 1, 1, 3)))
        let std = graph.variable(
          Tensor<FloatType>([0.229, 0.224, 0.225], .GPU(0), .NHWC(1, 1, 1, 3)))
        return (std .* result + mean).clamped(0...1) * 2 - 1
      case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b,
        .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
        .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .ltx2,
        .ltx2_3, .seedvr2_3b, .seedvr2_7b, .longcatVideoAvatar1_5:
        return result
      }
    }
    let (result, audio, decoder) = decode(
      x, decoder: existingDecoder, highPrecision: false, cancellation: cancellation)
    if highPrecisionFallback && !isCancelled.load(ordering: .acquiring)
      && isNaN(result.rawValue.toCPU())
    {
      let (highPrecisionResult, audio, _) = decode(
        x, decoder: nil, highPrecision: true, cancellation: cancellation)
      return (rescalePixels(highPrecisionResult), audio, decoder)
    }
    return (rescalePixels(result), audio, decoder)
  }

  public func decode(
    _ x: DynamicGraph.Tensor<FloatType>, batchSize: (Int, Int), decoder existingDecoder: Model?,
    cancellation: (@escaping () -> Void) -> Void
  )
    -> (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<Float>?, Model?)
  {
    let shape = x.shape
    if batchSize.1 == 0 {
      if batchSize.0 >= shape[0] {
        return decode(x, decoder: existingDecoder, cancellation: cancellation)
      }
      return decode(
        x[(shape[0] - batchSize.0)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied(),
        decoder: existingDecoder, cancellation: cancellation)
    }
    // Decode first n frames as if they are images.
    var results = [DynamicGraph.Tensor<FloatType>]()
    var decoder: Model? = nil
    for i in 0..<batchSize.1 {
      let result = decode(
        x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied(), decoder: decoder,
        cancellation: cancellation)
      results.append(result.0)
      decoder = result.2
    }
    decoder = nil
    let graph = x.graph
    let result = decode(
      x[batchSize.1..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied(),
      decoder: existingDecoder, cancellation: cancellation)
    var resultShape = result.0.shape
    resultShape[0] += batchSize.1
    var fullResult = graph.variable(.GPU(0), format: .NHWC, shape: resultShape, of: FloatType.self)
    for i in 0..<batchSize.1 {
      fullResult[i..<(i + 1), 0..<resultShape[1], 0..<resultShape[2], 0..<resultShape[3]] =
        results[i]
    }
    fullResult[
      batchSize.1..<resultShape[0], 0..<resultShape[1], 0..<resultShape[2], 0..<resultShape[3]] =
      result.0
    return (fullResult, result.1, result.2)
  }

  public func sampleFromDistribution(
    _ parameters: DynamicGraph.Tensor<FloatType>, noise: DynamicGraph.Tensor<FloatType>? = nil
  ) -> (DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>) {
    let shape = parameters.shape
    let batchSize = shape[0]
    let startHeight = parameters.shape[1]
    let startWidth = parameters.shape[2]
    let graph = parameters.graph
    let channels = parameters.shape[3] / 2
    let mean = parameters[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<channels]
    let logvar = parameters[
      0..<batchSize, 0..<startHeight, 0..<startWidth, channels..<(channels * 2)
    ].copied().clamped(
      -30...20)
    let std = Functional.exp(0.5 * logvar)
    let n: DynamicGraph.Tensor<FloatType>
    if let noise = noise {
      n = noise
    } else {
      n = graph.variable(
        .GPU(0), .NHWC(batchSize, startHeight, startWidth, channels), of: FloatType.self)
      n.randn(std: 1, mean: 0)
    }
    return (scale(mean + std .* n), mean)
  }

  public func sample(
    _ x: DynamicGraph.Tensor<FloatType>, encoder: Model?,
    cancellation: (@escaping () -> Void) -> Void
  ) -> (
    DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>, Model?
  ) {
    let (parameters, encoder) = encode(x, encoder: encoder, cancellation: cancellation)
    guard version != .kandinsky21 && version != .wurstchenStageC && version != .hiDreamO1 else {
      return (parameters, parameters, encoder)
    }
    guard version != .wurstchenStageB && version != .ltx2 && version != .ltx2_3 else {
      // For stage b, we need to scale it properly.
      let sample = scale(parameters)
      return (sample, sample, encoder)
    }
    let (sample, mean) = sampleFromDistribution(parameters)
    return (sample, mean, encoder)
  }

  public func sample(
    _ x: DynamicGraph.Tensor<FloatType>, individualFrames: Int, encoder existingEncoder: Model?,
    cancellation: (@escaping () -> Void) -> Void
  ) -> (
    DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>, Model?
  ) {
    if individualFrames == 0 {
      return sample(x, encoder: existingEncoder, cancellation: cancellation)
    }
    let shape = x.shape
    precondition(individualFrames < shape[0])
    var results = [(DynamicGraph.Tensor<FloatType>, DynamicGraph.Tensor<FloatType>)]()
    var encoder: Model? = nil
    for i in 0..<individualFrames {
      let result = sample(
        x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied(), encoder: encoder,
        cancellation: cancellation)
      results.append((result.0, result.1))
      encoder = result.2
    }
    encoder = nil
    let graph = x.graph
    let result = sample(
      x[individualFrames..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied(),
      encoder: existingEncoder, cancellation: cancellation)
    var sampleShape = result.0.shape
    sampleShape[0] += individualFrames
    var meanShape = result.1.shape
    meanShape[0] += individualFrames
    var sample = graph.variable(.GPU(0), format: .NHWC, shape: sampleShape, of: FloatType.self)
    var mean = graph.variable(.GPU(0), format: .NHWC, shape: meanShape, of: FloatType.self)
    for i in 0..<individualFrames {
      sample[i..<(i + 1), 0..<sampleShape[1], 0..<sampleShape[2], 0..<sampleShape[3]] = results[i].0
      mean[i..<(i + 1), 0..<meanShape[1], 0..<meanShape[2], 0..<meanShape[3]] = results[i].1
    }
    sample[
      individualFrames..<sampleShape[0], 0..<sampleShape[1], 0..<sampleShape[2], 0..<sampleShape[3]] =
      result.0
    mean[individualFrames..<meanShape[0], 0..<meanShape[1], 0..<meanShape[2], 0..<meanShape[3]] =
      result.1
    return (sample, mean, result.2)
  }

  public func scale(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    let graph = x.graph
    let scalingFactor = latentsScaling.scalingFactor
    if let latentsMean = latentsScaling.mean, let latentsStd = latentsScaling.std,
      latentsMean.count >= 4, latentsStd.count >= 4
    {
      let mean = graph.variable(
        Tensor<FloatType>(
          latentsMean.map { FloatType($0) }, .GPU(0), .NHWC(1, 1, 1, latentsMean.count)))
      let invStd = graph.variable(
        Tensor<FloatType>(
          latentsStd.map { FloatType(scalingFactor / $0) }, .GPU(0),
          .NHWC(1, 1, 1, latentsStd.count)))
      if version == .ernieImage || version == .flux2 || version == .flux2_9b || version == .flux2_4b
        || version == .ideogram4
      {
        // FLUX.2 is different, it needs to first reshape x and then apply the scaling.
        let shape = x.shape
        return
          (invStd
          .* (x.reshaped(
            format: .NHWC, shape: [shape[0], shape[1] / 2, 2, shape[2] / 2, 2, shape[3]]
          ).permuted(0, 1, 3, 5, 2, 4).contiguous().reshaped(
            .NHWC(shape[0], shape[1] / 2, shape[2] / 2, 2 * 2 * shape[3])) - mean))
          .reshaped(format: .NHWC, shape: [shape[0], shape[1] / 2, shape[2] / 2, shape[3], 2, 2])
          .permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped(format: .NHWC, shape: shape)
      } else {
        return invStd .* (x - mean)
      }
    } else if let shiftFactor = latentsScaling.shiftFactor {
      return (x - shiftFactor) * scalingFactor
    } else {
      return x * scalingFactor
    }
  }

  public func encode(
    _ x: DynamicGraph.Tensor<FloatType>, encoder existingEncoder: Model?,
    cancellation: (@escaping () -> Void) -> Void
  )
    -> (DynamicGraph.Tensor<FloatType>, Model?)
  {
    let (result, encoder) = encode(
      x, encoder: existingEncoder, highPrecision: false, cancellation: cancellation)
    if highPrecisionFallback && isNaN(result.rawValue.toCPU()) {
      let (highPrecisionResult, _) = encode(
        x, encoder: nil, highPrecision: true, cancellation: cancellation)
      return (highPrecisionResult, encoder)
    }
    return (result, encoder)
  }

  public func encode(
    _ x: [DynamicGraph.Tensor<FloatType>], encoder existingEncoder: Model?,
    cancellation: (@escaping () -> Void) -> Void
  )
    -> ([DynamicGraph.Tensor<FloatType>], Model)
  {
    precondition(!x.isEmpty)
    var encoder = existingEncoder
    return (
      x.map {
        let result: DynamicGraph.Tensor<FloatType>
        (result, encoder) = encode($0, encoder: encoder, cancellation: cancellation)
        return result
      }, encoder!
    )
  }

  private func encode(
    _ x: DynamicGraph.Tensor<FloatType>, encoder existingEncoder: Model?, highPrecision: Bool,
    cancellation: (@escaping () -> Void) -> Void
  )
    -> (DynamicGraph.Tensor<FloatType>, Model?)
  {
    var shape = x.shape
    let batchSize = shape[0]
    precondition(shape[1] % 16 == 0)
    precondition(shape[2] % 16 == 0)
    let graph = x.graph
    let encoder: Model
    let queueWatermark = DynamicGraph.queueWatermark
    if version == .hunyuanVideo {
      DynamicGraph.queueWatermark = min(1, queueWatermark)
    }
    defer {
      if version == .hunyuanVideo {
        DynamicGraph.queueWatermark = queueWatermark
      }
    }
    let outputChannels: Int
    var x = x
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand ? .externalOnDemand : .externalData
    let scaleFactor: Int
    let scaleFactorZ: Int
    switch version {
    case .v1, .v2, .sd3, .sd3Large, .pixart, .auraflow, .flux1, .sdxlBase, .sdxlRefiner, .ssd1b,
      .svdI2v, .kandinsky21, .hiDreamI1, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b,
      .ideogram4:
      scaleFactor = 8
      scaleFactorZ = 1
    case .hiDreamO1:
      scaleFactor = 32
      scaleFactorZ = 1
    case .qwenImage2_1:
      scaleFactor = 16
      scaleFactorZ = 1
    case .hunyuanVideo, .wan21_1_3b, .wan21_14b, .qwenImage, .krea2, .cosmos2_5_2b,
      .seedvr2_3b, .seedvr2_7b, .longcatVideoAvatar1_5:
      scaleFactor = 8
      scaleFactorZ = 4
    case .wan22_5b, .minimaxH3:
      scaleFactor = 16
      scaleFactorZ = 4
    case .ltx2, .ltx2_3:
      scaleFactor = 32
      scaleFactorZ = 8
    case .wurstchenStageC:
      scaleFactor = 32
      scaleFactorZ = 1
    case .wurstchenStageB:
      scaleFactor = 4
      scaleFactorZ = 1
    }
    let startHeight = shape[1] / scaleFactor
    let startWidth = shape[2] / scaleFactor
    var encodingTileSize = (
      depth: 0,
      width: min(tiledDiffusion.tileSize.width * (64 / scaleFactor), startWidth),
      height: min(tiledDiffusion.tileSize.height * (64 / scaleFactor), startHeight)
    )
    // Cut by half because we will * 2 later.
    var encodingTileOverlap = (tiledDiffusion.tileOverlap * (64 / scaleFactor) + 1) / 2
    var tiledEncoding =
      tiledDiffusion.isEnabled
      && (startWidth > encodingTileSize.width || startHeight > encodingTileSize.height)
    var temporalTiledEncodingConfiguration: (overlap: Int, trailingFrameDrop: Int)? = nil
    let causalAttentionMask: DynamicGraph.Tensor<Float>?
    switch version {
    case .minimaxH3:
      // H3 expects ImageNet-normalized RGB, not the app's [-1, 1] pixels.
      let mean = graph.variable(
        Tensor<FloatType>([0.485, 0.456, 0.406], .GPU(0), .NHWC(1, 1, 1, 3)))
      let invStd = graph.variable(
        Tensor<FloatType>(
          [FloatType(1 / 0.229), FloatType(1 / 0.224), FloatType(1 / 0.225)], .GPU(0),
          .NHWC(1, 1, 1, 3)))
      x = invStd .* ((x + 1) * 0.5 - mean)
      let frames = batchSize > 1 ? 17 : 1
      if batchSize > 1 {
        // Independent 17-frame clips produce 5 latents each; drop the final 3 latents.
        if !tiledEncoding {
          encodingTileSize.width = startWidth
          encodingTileSize.height = startHeight
        }
        tiledEncoding = true
        encodingTileSize.depth = 5
        temporalTiledEncodingConfiguration = (overlap: 0, trailingFrameDrop: 3)
      }
      let height = (tiledEncoding ? encodingTileSize.height : startHeight) * 16
      let width = (tiledEncoding ? encodingTileSize.width : startWidth) * 16
      encoder =
        existingEncoder ?? MiniMaxH3VideoEncoder(frames: frames, height: height, width: width)
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        var input = x[0..<min(batchSize, frames), 0..<height, 0..<width, 0..<shape[3]].copied()
        if input.shape[0] < frames {
          input = Pad(.replicate, begin: [0, 0, 0, 0], end: [frames - input.shape[0], 0, 0, 0])(
            inputs: input)[0].as(of: FloatType.self)
        }
        if highPrecision {
          encoder.compile(inputs: DynamicGraph.Tensor<Float>(from: input))
        } else {
          encoder.compile(inputs: input)
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          try! store.read(
            "video_encoder", model: encoder, strict: true, codec: [.jit, externalData])
        }
      }
      outputChannels = 48
      causalAttentionMask = nil
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .pixart, .auraflow:
      let startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      let startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      let encoderUsesFlashAttention = usesFlashAttention && startWidth * startHeight >= 256 * 176
      encoder =
        existingEncoder
        ?? Encoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, usesFlashAttention: encoderUsesFlashAttention,
          format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).0
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]]))
        } else {
          encoder.compile(
            inputs: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 8
      causalAttentionMask = nil
    case .sd3, .sd3Large, .flux1, .hiDreamI1, .zImage:
      let startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      let startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      let encoderUsesFlashAttention = usesFlashAttention && startWidth * startHeight >= 256 * 176
      encoder =
        existingEncoder
        ?? Encoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, usesFlashAttention: encoderUsesFlashAttention,
          format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW, quantLayer: false,
          outputChannels: 16
        ).0
      // Don't use FP32 for SD3 / FLUX.1 encoding pass.
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]]))
        } else {
          encoder.compile(
            inputs: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 32
      causalAttentionMask = nil
    case .hiDreamO1:
      // No need to initialize anything.
      return (hiDreamO1Patchify(x), nil)
    case .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ideogram4:
      let startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      let startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      let encoderUsesFlashAttention = usesFlashAttention && startWidth * startHeight >= 256 * 176
      encoder =
        existingEncoder
        ?? Encoder(
          channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
          startHeight: startHeight, usesFlashAttention: encoderUsesFlashAttention,
          format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW, quantLayer: true,
          outputChannels: 32
        ).0
      // Don't use FP32 for SD3 / FLUX.1 encoding pass.
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]]))
        } else {
          encoder.compile(
            inputs: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 64
      causalAttentionMask = nil
    case .hunyuanVideo:
      var startDepth = (shape[0] - 1) / 4 + 1
      var startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      var startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      let sizeLimit = deviceProperties.memoryCapacity >= .high ? 32 : 20
      if startWidth > sizeLimit || startHeight > sizeLimit || startDepth > 15 {
        // We turn on tiled decoding forcefully.
        if !tiledEncoding {
          encodingTileOverlap = 4
        }
        tiledEncoding = true
        startWidth = min(startWidth, sizeLimit)
        startHeight = min(startHeight, sizeLimit)
        encodingTileSize.width = startWidth
        encodingTileSize.height = startHeight
        startDepth = min(startDepth, 15)
        encodingTileSize.depth = startDepth
      }
      encoder =
        existingEncoder
        ?? EncoderCausal3D(
          channels: [128, 256, 512, 512], numRepeat: 2, startWidth: startWidth,
          startHeight: startHeight, startDepth: startDepth,
          format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).1
      var mask = Tensor<Float>(
        Array(repeating: 0, count: startDepth * startDepth), .CPU,
        .NHWC(startDepth, 1, startDepth, 1))
      for i in 0..<(startDepth - 1) {
        for j in (i + 1)..<startDepth {
          mask[i, 0, j, 0] = -Float.greatestFiniteMagnitude
        }
      }
      let batchSize = (startDepth - 1) * 4 + 1
      causalAttentionMask = graph.variable(mask.toGPU(0))
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(
            inputs: [
              DynamicGraph.Tensor<Float>(
                from: x[0..<batchSize, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
            ] + (causalAttentionMask.map { [$0] } ?? []))
        } else {
          encoder.compile(
            inputs: [x[0..<batchSize, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]]]
              + (causalAttentionMask.map { [DynamicGraph.Tensor<FloatType>(from: $0)] } ?? []))
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 32
    case .qwenImage2_1:
      let height = (tiledEncoding ? encodingTileSize.height : startHeight) * 16
      let width = (tiledEncoding ? encodingTileSize.width : startWidth) * 16
      let encoderUsesFlashAttention =
        usesFlashAttention && (width / 16) * (height / 16) >= 256 * 176
      var rgba = graph.variable(.GPU(0), .NHWC(shape[0], shape[1], shape[2], 4), of: FloatType.self)
      rgba.full(1)
      if shape[3] == 4 {
        rgba[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<3] =
          x[0..<shape[0], 0..<shape[1], 0..<shape[2], 1..<4]
        rgba[0..<shape[0], 0..<shape[1], 0..<shape[2], 3..<4] =
          x[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<1] * 2 - 1
      } else {
        rgba[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<3] = x
      }
      x = rgba
      shape[3] = 4
      encoder =
        existingEncoder
        ?? QwenImage2_1Encoder(
          channels: [96, 192, 384, 768, 768], height: height, width: width,
          usesFlashAttention: encoderUsesFlashAttention)
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(1)
        if highPrecision {
          encoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: x[0..<1, 0..<height, 0..<width, 0..<4].copied()))
        } else {
          encoder.compile(inputs: x[0..<1, 0..<height, 0..<width, 0..<4].copied())
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 128
      causalAttentionMask = nil
    case .wan21_1_3b, .wan21_14b, .qwenImage, .krea2, .cosmos2_5_2b,
      .longcatVideoAvatar1_5:
      let startDepth =
        version == .qwenImage || version == .krea2 || version == .cosmos2_5_2b
        ? 1 : (shape[0] - 1) / 4 + 1
      var startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      var startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      let sizeLimit: Int
      if startDepth > 1 {
        switch deviceProperties.memoryCapacity {
        case .high, .veryHigh:
          sizeLimit = 1024  // Practically unlimited.
        case .medium:
          sizeLimit = 104
        case .low:
          sizeLimit = 32
        }
      } else {
        sizeLimit = 1024
      }
      if alternativeDecoderVersion == .transparent {
        // We need to pad x to 4 channels.
        shape[3] = 4
        var newX = graph.variable(.GPU(0), format: .NHWC, shape: shape, of: FloatType.self)
        newX.full(1)
        newX[0..<shape[0], 0..<shape[1], 0..<shape[2], 0..<x.shape[3]] = x
        x = newX
      }
      if startWidth > sizeLimit || startHeight > sizeLimit {
        // We turn on tiled decoding forcefully.
        if !tiledEncoding {
          encodingTileOverlap = 4
        }
        tiledEncoding = true
        startWidth = min(startWidth, sizeLimit)
        startHeight = min(startHeight, sizeLimit)
        encodingTileSize.width = startWidth
        encodingTileSize.height = startHeight
        encodingTileSize.depth = startDepth
      }
      encoder =
        existingEncoder
        ?? WanEncoderCausal3D(
          channels: [96, 192, 384, 384], numRepeat: 2, startWidth: startWidth,
          startHeight: startHeight, startDepth: startDepth, wan22: false,
          inputChannels: alternativeDecoderVersion == .transparent ? 4 : 3,
          format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).1
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        let batchSize =
          version == .qwenImage || version == .krea2 || version == .cosmos2_5_2b ? 1 : batchSize
        if highPrecision {
          encoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: x[0..<batchSize, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]]))
        } else {
          encoder.compile(
            inputs: x[0..<batchSize, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          if startDepth > 1 {
            store.read("encoder", model: encoder, codec: [.jit, externalData])
          } else {
            store.read("encoder", model: encoder, codec: [.jit, externalData]) {
              name, dataType, format, shape in
              guard
                name.hasSuffix("-0]")
                  && (name.hasPrefix("__encoder__[t-conv_in-")
                    || name.hasPrefix("__encoder__[t-conv_out-")
                    || name.hasPrefix("__encoder__[t-resnet_conv1-")
                    || name.hasPrefix("__encoder__[t-resnet_conv2-"))
              else {
                return .continue(name)
              }
              guard
                var tensor =
                  (store.read(name, kind: .CPU, codec: [.jit, externalData]).map {
                    Tensor<FloatType>(from: $0)
                  })
              else {
                return .continue(name)
              }
              let channels = tensor.shape.reduce(1, *) / (3 * 3 * 3)  // Input channels and output channels.
              let outChannels = shape.reduce(1, *) / (3 * 3)
              tensor = tensor.reshaped(.NCHW(channels, 3, 3, 3))
              // In case depth = 1, it reduces to 2D convolution.
              guard channels < outChannels else {
                return .final(tensor[0..<channels, 2..<3, 0..<3, 0..<3].contiguous())
              }
              // In case we needs more, allocate first.
              var outTensor = Tensor<FloatType>(
                Array(repeating: 0, count: outChannels * 3 * 3), .CPU, .NCHW(outChannels, 1, 3, 3))
              outTensor[0..<channels, 0..<1, 0..<3, 0..<3] = tensor[
                0..<channels, 2..<3, 0..<3, 0..<3
              ].contiguous()
              return .final(outTensor)
            }
          }
        }
      }
      outputChannels = 32
      causalAttentionMask = nil
    case .wan22_5b:
      let startDepth = version == .qwenImage ? 1 : (shape[0] - 1) / 4 + 1
      var startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      var startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      let sizeLimit: Int
      if startDepth > 1 {
        switch deviceProperties.memoryCapacity {
        case .high, .veryHigh:
          sizeLimit = 1024  // Practically unlimited.
        case .medium:
          sizeLimit = 104
        case .low:
          sizeLimit = 32
        }
      } else {
        sizeLimit = 1024
      }
      if startWidth > sizeLimit || startHeight > sizeLimit {
        // We turn on tiled decoding forcefully.
        if !tiledEncoding {
          encodingTileOverlap = 4
        }
        tiledEncoding = true
        startWidth = min(startWidth, sizeLimit)
        startHeight = min(startHeight, sizeLimit)
        encodingTileSize.width = startWidth
        encodingTileSize.height = startHeight
        encodingTileSize.depth = startDepth
      }
      encoder =
        existingEncoder
        ?? WanEncoderCausal3D(
          channels: [160, 320, 640, 640], numRepeat: 2, startWidth: startWidth,
          startHeight: startHeight, startDepth: startDepth, wan22: true,
          inputChannels: 12, format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).1
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(
            inputs: DynamicGraph.Tensor<Float>(
              from: x[0..<batchSize, 0..<(startHeight * 16), 0..<(startWidth * 16), 0..<shape[3]]))
        } else {
          encoder.compile(
            inputs: x[0..<batchSize, 0..<(startHeight * 16), 0..<(startWidth * 16), 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          if startDepth > 1 {
            store.read("encoder", model: encoder, codec: [.jit, externalData])
          } else {
            store.read("encoder", model: encoder, codec: [.jit, externalData]) {
              name, dataType, format, shape in
              guard
                name.hasSuffix("-0]")
                  && (name.hasPrefix("__encoder__[t-conv_in-")
                    || name.hasPrefix("__encoder__[t-conv_out-")
                    || name.hasPrefix("__encoder__[t-resnet_conv1-")
                    || name.hasPrefix("__encoder__[t-resnet_conv2-"))
              else {
                return .continue(name)
              }
              guard
                var tensor =
                  (store.read(name, kind: .CPU, codec: [.jit, externalData]).map {
                    Tensor<FloatType>(from: $0)
                  })
              else {
                return .continue(name)
              }
              let channels = tensor.shape.reduce(1, *) / (3 * 3 * 3)  // Input channels and output channels.
              let outChannels = shape.reduce(1, *) / (3 * 3)
              tensor = tensor.reshaped(.NCHW(channels, 3, 3, 3))
              // In case depth = 1, it reduces to 2D convolution.
              guard channels < outChannels else {
                return .final(tensor[0..<channels, 2..<3, 0..<3, 0..<3].contiguous())
              }
              // In case we needs more, allocate first.
              var outTensor = Tensor<FloatType>(
                Array(repeating: 0, count: outChannels * 3 * 3), .CPU, .NCHW(outChannels, 1, 3, 3))
              outTensor[0..<channels, 0..<1, 0..<3, 0..<3] = tensor[
                0..<channels, 2..<3, 0..<3, 0..<3
              ].contiguous()
              return .final(outTensor)
            }
          }
        }
      }
      outputChannels = 96
      causalAttentionMask = nil
    case .ltx2, .ltx2_3:
      let startDepth = (shape[0] - 1) / 8 + 1
      var startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      var startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      let sizeLimit = deviceProperties.memoryCapacity >= .high ? 16 : 10
      if startWidth > sizeLimit || startHeight > sizeLimit {
        // We turn on tiled decoding forcefully.
        if !tiledEncoding {
          encodingTileOverlap = 4
        }
        tiledEncoding = true
        startWidth = min(startWidth, sizeLimit)
        startHeight = min(startHeight, sizeLimit)
        encodingTileSize.width = startWidth
        encodingTileSize.height = startHeight
        encodingTileSize.depth = startDepth
      }
      let encoderLayers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))]
      let encoderPaddingMode: LTX2VideoVAEPaddingMode
      if version == .ltx2_3 {
        encoderLayers = [
          (channels: 128, numRepeat: 4, stride: (1, 1, 1)),
          (channels: 256, numRepeat: 6, stride: (1, 2, 2)),
          (channels: 512, numRepeat: 4, stride: (2, 1, 1)),
          (channels: 1024, numRepeat: 2, stride: (2, 2, 2)),
          (channels: 1024, numRepeat: 2, stride: (2, 2, 2)),
        ]
        encoderPaddingMode = .zero
      } else {
        encoderLayers = [
          (channels: 128, numRepeat: 4, stride: (1, 1, 1)),
          (channels: 256, numRepeat: 6, stride: (1, 2, 2)),
          (channels: 512, numRepeat: 6, stride: (2, 1, 1)),
          (channels: 1024, numRepeat: 2, stride: (2, 2, 2)),
          (channels: 2048, numRepeat: 2, stride: (2, 2, 2)),
        ]
        encoderPaddingMode = .reflect
      }
      encoder =
        existingEncoder
        ?? LTX2VideoEncoderCausal3D(
          layers: encoderLayers, startWidth: startWidth, startHeight: startHeight,
          startDepth: startDepth, paddingMode: encoderPaddingMode,
          format: deviceProperties.isNHWCPreferred ? .NHWC : .NCHW
        ).1
      let batchSize = (startDepth - 1) * 8 + 1
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(
            inputs:
              DynamicGraph.Tensor<Float>(
                from: x[0..<batchSize, 0..<(startHeight * 32), 0..<(startWidth * 32), 0..<shape[3]])
          )
        } else {
          encoder.compile(
            inputs: x[0..<batchSize, 0..<(startHeight * 32), 0..<(startWidth * 32), 0..<shape[3]])
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 128
      causalAttentionMask = nil
    case .kandinsky21:
      let startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      let startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      encoder =
        existingEncoder
        ?? EncoderKandinsky(
          zChannels: 4, channels: 128, channelMult: [1, 2, 2, 4], numResBlocks: 2,
          startHeight: startHeight * 8, startWidth: startWidth * 8, attnResolutions: Set([32]))
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        encoder.compile(inputs: x[0..<1, 0..<(startHeight * 8), 0..<(startWidth * 8), 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, externalData])
        }
      }
      outputChannels = 4
      causalAttentionMask = nil
    case .wurstchenStageC:
      encodingTileSize.height = startHeight
      encodingTileSize.width = startWidth
      encodingTileOverlap = 0
      encoder = existingEncoder ?? EfficientNetEncoder()
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        encoder.compile(inputs: x[0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("effnet", model: encoder, codec: [.q6p, .q8p, .ezm7, .jit, externalData])
        }
      }
      // Normalize from -1...1 to the EffNet range.
      let mean = graph.variable(
        Tensor<FloatType>(
          [
            FloatType(2 * 0.485 - 1), FloatType(2 * 0.456 - 1),
            FloatType(2 * 0.406 - 1),
          ], .GPU(0), .NHWC(1, 1, 1, 3)))
      let invStd = graph.variable(
        Tensor<FloatType>(
          [
            FloatType(0.5 / 0.229), FloatType(0.5 / 0.224), FloatType(0.5 / 0.225),
          ],
          .GPU(0), .NHWC(1, 1, 1, 3)))
      x = (x - mean) .* invStd
      outputChannels = 16
      causalAttentionMask = nil
    case .wurstchenStageB:
      let startWidth = tiledEncoding ? encodingTileSize.width : startWidth
      let startHeight = tiledEncoding ? encodingTileSize.height : startHeight
      encoder = existingEncoder ?? WurstchenStageAEncoder(batchSize: 1).0
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        encoder.compile(inputs: x[0..<1, 0..<(startHeight * 4), 0..<(startWidth * 4), 0..<shape[3]])
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) {
          $0.read("encoder", model: encoder, codec: [.jit, .q6p, .q8p, .ezm7, externalData])
        }
      }
      x = (x + 1) * 0.5
      outputChannels = 4
      causalAttentionMask = nil
    case .seedvr2_3b, .seedvr2_7b:
      let startDepth = shape[0]
      tiledEncoding = false
      let encoderUsesFlashAttention = usesFlashAttention && (startWidth * startHeight >= 256 * 176)
      encoder =
        existingEncoder
        ?? SeedVR2Encoder3D(
          startDepth: startDepth, startHeight: shape[1], startWidth: shape[2],
          usesFlashAttention: encoderUsesFlashAttention)
      if existingEncoder == nil {
        encoder.maxConcurrency = .limit(4)
        if highPrecision {
          encoder.compile(inputs: DynamicGraph.Tensor<Float>(from: x))
        } else {
          encoder.compile(inputs: x)
        }
        graph.openStore(
          filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
        ) { store in
          if startDepth > 1 {
            store.read("encoder", model: encoder, codec: [.jit, externalData])
          } else {
            store.read("encoder", model: encoder, codec: [.jit, externalData]) {
              name, dataType, _, shape in
              SeedVR2Conv3DToConv2D(
                graph: graph, name: name, dataType: dataType, shape: shape, store: store,
                externalData: externalData, modelPrefix: "encoder", of: FloatType.self)
                ?? .continue(name)
            }
          }
        }
      }
      outputChannels = 32
      causalAttentionMask = nil
    }
    isCancelled.store(false, ordering: .releasing)
    let isCancelled = isCancelled
    cancellation {
      isCancelled.store(true, ordering: .releasing)
      encoder.cancel()
    }
    guard
      batchSize > 1 && version != .hunyuanVideo && version != .wan21_1_3b && version != .wan21_14b
        && version != .wan22_5b && version != .ltx2 && version != .ltx2_3
        && version != .longcatVideoAvatar1_5 && version != .minimaxH3
    else {
      if highPrecision {
        if tiledEncoding {
          return (
            DynamicGraph.Tensor<FloatType>(
              from: tiledEncode(
                DynamicGraph.Tensor<Float>(from: x), causalAttentionMask: causalAttentionMask,
                encoder: encoder,
                tileSize: encodingTileSize,
                tileOverlap: encodingTileOverlap,
                scaleFactor: (scaleFactor, scaleFactorZ), outputChannels: outputChannels,
                temporalTiledEncodingConfiguration: temporalTiledEncodingConfiguration)), encoder
          )
        } else {
          return (
            DynamicGraph.Tensor<FloatType>(
              from: encoder(
                inputs: DynamicGraph.Tensor<Float>(from: x),
                (causalAttentionMask.map { [$0] } ?? []))[0].as(of: Float.self)),
            encoder
          )
        }
      } else {
        if tiledEncoding {
          return (
            tiledEncode(
              x,
              causalAttentionMask: causalAttentionMask.map {
                DynamicGraph.Tensor<FloatType>(from: $0)
              }, encoder: encoder, tileSize: encodingTileSize,
              tileOverlap: encodingTileOverlap, scaleFactor: (scaleFactor, scaleFactorZ),
              outputChannels: outputChannels,
              temporalTiledEncodingConfiguration: temporalTiledEncodingConfiguration),
            encoder
          )
        } else {
          return (
            encoder(
              inputs: x,
              (causalAttentionMask.map { [DynamicGraph.Tensor<FloatType>(from: $0)] } ?? []))[0].as(
                of: FloatType.self), encoder
          )
        }
      }
    }
    var result = graph.variable(
      .GPU(0), .NHWC(batchSize, startHeight, startWidth, outputChannels), of: FloatType.self)
    for i in 0..<batchSize {
      let z = x[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
      if highPrecision {
        if tiledEncoding {
          result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = DynamicGraph
            .Tensor<
              FloatType
            >(
              from: tiledEncode(
                DynamicGraph.Tensor<Float>(from: z), causalAttentionMask: causalAttentionMask,
                encoder: encoder,
                tileSize: encodingTileSize,
                tileOverlap: encodingTileOverlap,
                scaleFactor: (scaleFactor, scaleFactorZ), outputChannels: outputChannels,
                temporalTiledEncodingConfiguration: temporalTiledEncodingConfiguration))
        } else {
          result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = DynamicGraph
            .Tensor<
              FloatType
            >(
              from: encoder(
                inputs: DynamicGraph.Tensor<Float>(from: z),
                (causalAttentionMask.map { [$0] } ?? []))[0].as(
                  of: Float.self))
        }
      } else {
        if tiledEncoding {
          result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = tiledEncode(
            z,
            causalAttentionMask: causalAttentionMask.map {
              DynamicGraph.Tensor<FloatType>(from: $0)
            }, encoder: encoder, tileSize: encodingTileSize,
            tileOverlap: encodingTileOverlap, scaleFactor: (scaleFactor, scaleFactorZ),
            outputChannels: outputChannels,
            temporalTiledEncodingConfiguration: temporalTiledEncodingConfiguration)
        } else {
          result[i..<(i + 1), 0..<startHeight, 0..<startWidth, 0..<outputChannels] = encoder(
            inputs: z,
            (causalAttentionMask.map { [DynamicGraph.Tensor<FloatType>(from: $0)] } ?? []))[0].as(
              of: FloatType.self)
        }
      }
    }
    return (result, encoder)
  }
}

extension FirstStage {
  private func internalDecode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, decoder: Model,
    transparentDecoder: Model?, inputs: [DynamicGraph.AnyTensor] = []
  ) -> DynamicGraph.Tensor<T> {
    var pixel = decoder(inputs: z, (causalAttentionMask.map { [$0] } ?? []) + inputs)[0].as(
      of: T.self)
    if alternativeDecoderVersion == .transparent,
      version == .qwenImage || version == .qwenImage2_1 || version == .krea2
        || version == .cosmos2_5_2b
    {
      // If this is the case, we need to switch alpha channel with rgb channel.
      let shape = pixel.shape
      let rgb = pixel[0..<1, 0..<shape[1], 0..<shape[2], 0..<3].copied()
      let alpha = (pixel[0..<1, 0..<shape[1], 0..<shape[2], 3..<4].contiguous() + 1) * 0.5
      pixel[0..<1, 0..<shape[1], 0..<shape[2], 1..<shape[3]] = rgb
      pixel[0..<1, 0..<shape[1], 0..<shape[2], 0..<1] = alpha
      return pixel
    }
    guard let transparentDecoder = transparentDecoder else { return pixel }
    let pixelShape = pixel.shape
    let rgb = pixel[0..<1, 0..<pixelShape[1], 0..<pixelShape[2], 0..<3].copied()
    // Normalize to between 0 and 1
    var result = transparentDecoder(inputs: (rgb + 1) * 0.5, z)[0].as(of: T.self)
    // At this point, result should have 4 channels. The last 3 is RGB. Do so.
    let shape = result.shape
    // Only scale RGB channels to -1 and 1.
    let alpha = result[0..<1, 0..<shape[1], 0..<shape[2], 0..<1].clamped(0...1)
    result[0..<1, 0..<shape[1], 0..<shape[2], 1..<shape[3]] =
      (result[0..<1, 0..<shape[1], 0..<shape[2], 1..<shape[3]].clamped(0...1) - 0.5) * 2
    result[0..<1, 0..<shape[1], 0..<shape[2], 0..<1] = alpha
    return result
  }

  private func tiledDecode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, decoder: Model,
    transparentDecoder: Model?,
    tileSize: (width: Int, height: Int), tileOverlap: Int, outputChannels: Int,
    scaleFactor: (spatial: Int, temporal: Int), inputs: [DynamicGraph.AnyTensor] = []
  ) -> DynamicGraph.Tensor<T> {
    let shape = z.shape
    let channels = shape[3]
    let batchSize = shape[0]
    // tile overlap shouldn't be bigger than 1/3 of either height or width (otherwise we cannot make much progress).
    let tileOverlap = min(
      min(tileOverlap, Int((Double(tileSize.height) / 6).rounded(.down) * 2)),
      Int((Double(tileSize.width) / 6).rounded(.down) * 2)
    )
    let yTiles =
      (shape[1] - tileOverlap * 2 + (tileSize.height - tileOverlap * 2) - 1)
      / (tileSize.height - tileOverlap * 2)
    let xTiles =
      (shape[2] - tileOverlap * 2 + (tileSize.width - tileOverlap * 2) - 1)
      / (tileSize.width - tileOverlap * 2)
    let graph = z.graph
    var decodedRawValues = [Tensor<T>]()
    let resultBatchSize = (shape[0] - 1) * scaleFactor.temporal + 1
    guard !isCancelled.load(ordering: .acquiring) else {
      return graph.variable(
        Tensor<T>(
          .GPU(0),
          .NHWC(
            resultBatchSize, shape[1] * scaleFactor.spatial, shape[2] * scaleFactor.spatial,
            outputChannels)))
    }
    for y in 0..<yTiles {
      let yOfs = y * (tileSize.height - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
      let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
        iOfs: yOfs, length: shape[1], tileSize: tileSize.height, tileOverlap: tileOverlap)
      for x in 0..<xTiles {
        let xOfs = x * (tileSize.width - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
        let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
          iOfs: xOfs, length: shape[2], tileSize: tileSize.width, tileOverlap: tileOverlap)
        decodedRawValues.append(
          internalDecode(
            z[
              0..<batchSize, inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad,
              0..<channels
            ].copied(), causalAttentionMask: causalAttentionMask, decoder: decoder,
            transparentDecoder: transparentDecoder, inputs: inputs
          ).rawValue.toCPU())
        guard !isCancelled.load(ordering: .acquiring) else {
          return graph.variable(
            Tensor<T>(
              .GPU(0),
              .NHWC(
                resultBatchSize, shape[1] * scaleFactor.spatial, shape[2] * scaleFactor.spatial,
                outputChannels)))
        }
      }
    }
    graph.joined()
    let (xWeightsAndIndexes, yWeightsAndIndexes) = xyTileWeightsAndIndexes(
      width: shape[2] * scaleFactor.spatial, height: shape[1] * scaleFactor.spatial, xTiles: xTiles,
      yTiles: yTiles,
      tileSize: (
        width: tileSize.width * scaleFactor.spatial, height: tileSize.height * scaleFactor.spatial
      ),
      tileOverlap: tileOverlap * scaleFactor.spatial)
    var result = Tensor<T>(
      .CPU,
      .NHWC(
        resultBatchSize, shape[1] * scaleFactor.spatial, shape[2] * scaleFactor.spatial,
        outputChannels))
    let inputChannels = decodedRawValues.first?.shape[3] ?? outputChannels
    result.withUnsafeMutableBytes {
      guard let rfp = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
      withExtendedLifetime(decodedRawValues) {
        DispatchQueue.concurrentPerform(iterations: resultBatchSize) { t in
          var fp =
            rfp + t * shape[1] * scaleFactor.spatial * shape[2] * scaleFactor.spatial
            * outputChannels
          let tOffset =
            t * tileSize.width * scaleFactor.spatial * inputChannels * tileSize.height
            * scaleFactor.spatial
          let vmap = decodedRawValues.map {
            let v: UnsafePointer<T> = $0.withUnsafeBytes {
              let v = $0.baseAddress?.assumingMemoryBound(to: T.self)
              return v! + tOffset
            }
            return v
          }
          for j in 0..<(shape[1] * scaleFactor.spatial) {
            let yWeightAndIndex = yWeightsAndIndexes[j]
            for i in 0..<(shape[2] * scaleFactor.spatial) {
              let xWeightAndIndex = xWeightsAndIndexes[i]
              for k in 0..<outputChannels {
                fp[k] = 0
              }
              for y in yWeightAndIndex {
                let yOffset =
                  y.offset * tileSize.width * scaleFactor.spatial * inputChannels
                for x in xWeightAndIndex {
                  let weight = T(x.weight * y.weight)
                  let index = y.index * xTiles + x.index
                  var v = vmap[index]
                  // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                  v += x.offset * inputChannels + yOffset
                  for k in 0..<outputChannels {
                    fp[k] += v[k] * weight
                  }
                }
              }
              fp += outputChannels
            }
          }
        }
      }
    }
    return graph.variable(result.toGPU(0))
  }

  private func tiledDecode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, decoder: Model,
    transparentDecoder: Model?,
    tileSize: (depth: Int, width: Int, height: Int), tileOverlap: Int, outputChannels: Int,
    scaleFactor: (spatial: Int, temporal: Int),
    temporalTiledDecodingConfiguration: TemporalTiledDecodingConfiguration?,
    inputs: [DynamicGraph.AnyTensor] = []
  ) -> DynamicGraph.Tensor<T> {
    let frameDrop = temporalTiledDecodingConfiguration?.frameDrop
    guard tileSize.depth > 1 && (tileSize.depth < z.shape[0] || frameDrop != nil) else {
      return tiledDecode(
        z, causalAttentionMask: causalAttentionMask, decoder: decoder,
        transparentDecoder: transparentDecoder,
        tileSize: (width: tileSize.width, height: tileSize.height), tileOverlap: tileOverlap,
        outputChannels: outputChannels, scaleFactor: scaleFactor, inputs: inputs)
    }
    let shape = z.shape
    let channels = shape[3]
    let batchSize = shape[0]
    let graph = z.graph
    // tile overlap shouldn't be bigger than 1/3 of either height or width (otherwise we cannot make much progress).
    let tileOverlap = min(
      min(tileOverlap, Int((Double(tileSize.height) / 6).rounded(.down) * 2)),
      Int((Double(tileSize.width) / 6).rounded(.down) * 2)
    )
    let yTiles =
      (shape[1] - tileOverlap * 2 + (tileSize.height - tileOverlap * 2) - 1)
      / (tileSize.height - tileOverlap * 2)
    let xTiles =
      (shape[2] - tileOverlap * 2 + (tileSize.width - tileOverlap * 2) - 1)
      / (tileSize.width - tileOverlap * 2)
    let (xWeightsAndIndexes, yWeightsAndIndexes) = xyTileWeightsAndIndexes(
      width: shape[2] * scaleFactor.spatial, height: shape[1] * scaleFactor.spatial, xTiles: xTiles,
      yTiles: yTiles,
      tileSize: (
        width: tileSize.width * scaleFactor.spatial, height: tileSize.height * scaleFactor.spatial
      ),
      tileOverlap: tileOverlap * scaleFactor.spatial)
    let decodedBatchSize = (shape[0] - 1) * scaleFactor.temporal + 1
    let resultBatchSize: Int
    if let frameDrop {
      resultBatchSize =
        decodedBatchSize / frameDrop.interval * (frameDrop.interval - frameDrop.count)
        + min(decodedBatchSize % frameDrop.interval, frameDrop.interval - frameDrop.count)
    } else {
      resultBatchSize = decodedBatchSize
    }
    var result = Tensor<T>(
      Array(
        repeating: 0,
        count: resultBatchSize * shape[1] * scaleFactor.spatial * shape[2] * scaleFactor.spatial
          * outputChannels), .CPU,
      .NHWC(
        resultBatchSize, shape[1] * scaleFactor.spatial, shape[2] * scaleFactor.spatial,
        outputChannels))
    guard !isCancelled.load(ordering: .acquiring) else {
      return graph.variable(result.toGPU(0))
    }
    let temporalOverlap = temporalTiledDecodingConfiguration?.overlap ?? 5
    let temporalLookback = temporalTiledDecodingConfiguration?.lookback ?? 0
    let tileStep = max(1, tileSize.depth - temporalOverlap - temporalLookback)
    let nominalDepth = max(1, tileSize.depth - temporalLookback)
    let temporalOverlapFrames = max(0, ((temporalOverlap - 1) * scaleFactor.temporal) + 1)
    let overlapTrim =
      temporalTiledDecodingConfiguration?.overlapBlending?.trim
      ?? (temporalOverlapFrames + 1) / 2
    let fadeOffset = temporalTiledDecodingConfiguration?.overlapBlending?.offset ?? 0.5
    let leftFadeFrames: Int
    let rightFadeFrames: Int
    let firstTileValidFrames: Int
    if temporalLookback > 0 {
      leftFadeFrames = max(0, ((temporalOverlap + temporalLookback - 1) * scaleFactor.temporal) + 1)
      rightFadeFrames = temporalOverlap * scaleFactor.temporal
      firstTileValidFrames = max(
        1, ((tileSize.depth - temporalLookback - 1) * scaleFactor.temporal) + 1)
    } else {
      leftFadeFrames = temporalOverlapFrames - overlapTrim
      rightFadeFrames = temporalOverlapFrames - overlapTrim
      firstTileValidFrames = ((tileSize.depth - 1) * scaleFactor.temporal) + 1
    }
    let temporalTileCount =
      batchSize <= nominalDepth ? 1 : ((batchSize - nominalDepth) + tileStep - 1) / tileStep + 1
    for tileIndex in 0..<temporalTileCount {
      let anchor = tileIndex * tileStep
      var decodedRawValues = [Tensor<T>]()
      let unclampedStart = max(0, anchor - temporalLookback)
      let tStart = min(unclampedStart, max(0, batchSize - tileSize.depth))
      let tDecodedStart: Int
      let isLast = tileIndex + 1 >= temporalTileCount
      if temporalLookback > 0 {
        tDecodedStart = (unclampedStart - tStart) * scaleFactor.temporal
      } else if anchor == 0 {
        tDecodedStart = 0
      } else {
        tDecodedStart =
          (unclampedStart - tStart) * scaleFactor.temporal + overlapTrim
      }
      let tileDecodedDepth = ((tileSize.depth - 1) * scaleFactor.temporal) + 1
      var tileDecodedEnd = min(tileDecodedDepth, decodedBatchSize - tStart * scaleFactor.temporal)
      if tileIndex == 0 {
        tileDecodedEnd = min(tileDecodedEnd, firstTileValidFrames)
      }
      guard tDecodedStart < tileDecodedEnd else { break }  // Nothing to copy, break.
      for y in 0..<yTiles {
        let yOfs = y * (tileSize.height - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
        let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
          iOfs: yOfs, length: shape[1], tileSize: tileSize.height, tileOverlap: tileOverlap)
        for x in 0..<xTiles {
          let xOfs = x * (tileSize.width - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
          let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
            iOfs: xOfs, length: shape[2], tileSize: tileSize.width, tileOverlap: tileOverlap)
          var input = z[
            tStart..<min(tStart + tileSize.depth, batchSize), inputStartYPad..<inputEndYPad,
            inputStartXPad..<inputEndXPad, 0..<channels
          ].copied()
          if input.shape[0] < tileSize.depth {
            input = Pad(
              .replicate, begin: [0, 0, 0, 0], end: [tileSize.depth - input.shape[0], 0, 0, 0])(
                inputs: input)[0].as(of: T.self)
          }
          decodedRawValues.append(
            internalDecode(
              input, causalAttentionMask: causalAttentionMask, decoder: decoder,
              transparentDecoder: transparentDecoder, inputs: inputs
            ).rawValue.toCPU())
          guard !isCancelled.load(ordering: .acquiring) else {
            return graph.variable(result.toGPU(0))
          }
        }
      }
      graph.joined()
      let inputChannels = decodedRawValues.first?.shape[3] ?? outputChannels
      result.withUnsafeMutableBytes {
        guard let rfp = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
        withExtendedLifetime(decodedRawValues) {
          DispatchQueue.concurrentPerform(iterations: tileDecodedEnd - tDecodedStart) {
            let tDecoded = $0 + tDecodedStart
            var tOutput = tDecoded + tStart * scaleFactor.temporal
            if let frameDrop {
              let offset = tOutput % frameDrop.interval
              guard offset < frameDrop.interval - frameDrop.count else { return }
              tOutput -= tOutput / frameDrop.interval * frameDrop.count
            }
            var fp =
              rfp + tOutput * shape[1] * scaleFactor.spatial
              * shape[2] * scaleFactor.spatial * outputChannels
            let tOffset =
              tDecoded * tileSize.width * scaleFactor.spatial * inputChannels * tileSize.height
              * scaleFactor.spatial
            let vmap = decodedRawValues.map {
              let v: UnsafePointer<T> = $0.withUnsafeBytes {
                let v = $0.baseAddress?.assumingMemoryBound(to: T.self)
                return v! + tOffset
              }
              return v
            }
            let tWeight: Float
            if tileIndex > 0 && tDecoded - tDecodedStart < leftFadeFrames && leftFadeFrames > 0 {
              tWeight = min(
                (Float(tDecoded - tDecodedStart) + fadeOffset) / Float(leftFadeFrames), 1)
            } else if tileDecodedEnd - tDecoded <= rightFadeFrames && !isLast && rightFadeFrames > 0
            {
              tWeight = min(
                (Float(tileDecodedEnd - tDecoded) - fadeOffset) / Float(rightFadeFrames), 1)
            } else {
              tWeight = 1
            }
            for j in 0..<(shape[1] * scaleFactor.spatial) {
              let yWeightAndIndex = yWeightsAndIndexes[j]
              for i in 0..<(shape[2] * scaleFactor.spatial) {
                let xWeightAndIndex = xWeightsAndIndexes[i]
                for y in yWeightAndIndex {
                  let yOffset =
                    y.offset * tileSize.width * scaleFactor.spatial * inputChannels
                  for x in xWeightAndIndex {
                    let weight = T(x.weight * y.weight * tWeight)
                    let index = y.index * xTiles + x.index
                    var v = vmap[index]
                    // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                    v += x.offset * inputChannels + yOffset
                    for k in 0..<outputChannels {
                      fp[k] += v[k] * weight
                    }
                  }
                }
                fp += outputChannels
              }
            }
          }
        }
      }
    }
    return graph.variable(result.toGPU(0))
  }

  private func tiledEncode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, encoder: Model,
    tileSize: (width: Int, height: Int), tileOverlap: Int,
    scaleFactor: (spatial: Int, temporal: Int),
    outputChannels: Int
  ) -> DynamicGraph.Tensor<T> {
    let shape = z.shape
    let channels = shape[3]
    let batchSize = shape[0]
    // tile overlap shouldn't be bigger than 1/3 of either height or width (otherwise we cannot make much progress).
    let tileOverlap = min(
      min(tileOverlap, Int((Double(tileSize.height) / 6).rounded(.down) * 2)),
      Int((Double(tileSize.width) / 6).rounded(.down) * 2)
    )
    let startHeight = shape[1] / scaleFactor.spatial
    let startWidth = shape[2] / scaleFactor.spatial
    let yTiles =
      (startHeight - tileOverlap * 2 + (tileSize.height - tileOverlap * 2) - 1)
      / (tileSize.height - tileOverlap * 2)
    let xTiles =
      (startWidth - tileOverlap * 2 + (tileSize.width - tileOverlap * 2) - 1)
      / (tileSize.width - tileOverlap * 2)
    let graph = z.graph
    var encodedRawValues = [Tensor<T>]()
    let resultBatchSize = (shape[0] - 1) / scaleFactor.temporal + 1
    guard !isCancelled.load(ordering: .acquiring) else {
      return graph.variable(
        Tensor<T>(.GPU(0), .NHWC(resultBatchSize, startHeight, startWidth, outputChannels)))
    }
    for y in 0..<yTiles {
      let yOfs = y * (tileSize.height - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
      let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
        iOfs: yOfs * scaleFactor.spatial, length: shape[1],
        tileSize: tileSize.height * scaleFactor.spatial,
        tileOverlap: tileOverlap * scaleFactor.spatial)
      for x in 0..<xTiles {
        let xOfs = x * (tileSize.width - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
        let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
          iOfs: xOfs * scaleFactor.spatial, length: shape[2],
          tileSize: tileSize.width * scaleFactor.spatial,
          tileOverlap: tileOverlap * scaleFactor.spatial)
        encodedRawValues.append(
          encoder(
            inputs: z[
              0..<batchSize, inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad,
              0..<channels
            ].copied(), (causalAttentionMask.map { [$0] } ?? []))[0].as(of: T.self).rawValue
            .toCPU())
        guard !isCancelled.load(ordering: .acquiring) else {
          return graph.variable(
            Tensor<T>(.GPU(0), .NHWC(resultBatchSize, startHeight, startWidth, outputChannels)))
        }
      }
    }
    let (xWeightsAndIndexes, yWeightsAndIndexes) = xyTileWeightsAndIndexes(
      width: startWidth, height: startHeight, xTiles: xTiles, yTiles: yTiles,
      tileSize: (width: tileSize.width, height: tileSize.height), tileOverlap: tileOverlap)
    var result = Tensor<T>(.CPU, .NHWC(resultBatchSize, startHeight, startWidth, outputChannels))
    result.withUnsafeMutableBytes {
      guard var fp = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
      for t in 0..<resultBatchSize {
        for j in 0..<startHeight {
          let yWeightAndIndex = yWeightsAndIndexes[j]
          for i in 0..<startWidth {
            let xWeightAndIndex = xWeightsAndIndexes[i]
            for k in 0..<outputChannels {
              fp[k] = 0
            }
            let tOffset = t * tileSize.width * outputChannels * tileSize.height
            for y in yWeightAndIndex {
              let yOffset = y.offset * tileSize.width * outputChannels + tOffset
              for x in xWeightAndIndex {
                let weight = T(x.weight * y.weight)
                let index = y.index * xTiles + x.index
                let tensor = encodedRawValues[index]
                tensor.withUnsafeBytes {
                  guard var v = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
                  // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                  v = v + x.offset * outputChannels + yOffset
                  for k in 0..<outputChannels {
                    fp[k] += v[k] * weight
                  }
                }
              }
            }
            fp += outputChannels
          }
        }
      }
    }
    return graph.variable(result.toGPU(0))
  }

  private func tiledEncode<T: TensorNumeric & BinaryFloatingPoint>(
    _ z: DynamicGraph.Tensor<T>, causalAttentionMask: DynamicGraph.Tensor<T>?, encoder: Model,
    tileSize: (depth: Int, width: Int, height: Int), tileOverlap: Int,
    scaleFactor: (spatial: Int, temporal: Int),
    outputChannels: Int,
    temporalTiledEncodingConfiguration: (overlap: Int, trailingFrameDrop: Int)?
  ) -> DynamicGraph.Tensor<T> {
    let temporalOverlap = temporalTiledEncodingConfiguration?.overlap ?? 5
    guard
      tileSize.depth > 1
        && (temporalOverlap == 0 || tileSize.depth < (z.shape[0] - 1) / scaleFactor.temporal + 1)
    else {
      return tiledEncode(
        z, causalAttentionMask: causalAttentionMask, encoder: encoder,
        tileSize: (width: tileSize.width, height: tileSize.height), tileOverlap: tileOverlap,
        scaleFactor: scaleFactor, outputChannels: outputChannels)
    }
    let shape = z.shape
    let channels = shape[3]
    let batchSize = shape[0]
    // tile overlap shouldn't be bigger than 1/3 of either height or width (otherwise we cannot make much progress).
    let tileOverlap = min(
      min(tileOverlap, Int((Double(tileSize.height) / 6).rounded(.down) * 2)),
      Int((Double(tileSize.width) / 6).rounded(.down) * 2)
    )
    let startHeight = shape[1] / scaleFactor.spatial
    let startWidth = shape[2] / scaleFactor.spatial
    let yTiles =
      (startHeight - tileOverlap * 2 + (tileSize.height - tileOverlap * 2) - 1)
      / (tileSize.height - tileOverlap * 2)
    let xTiles =
      (startWidth - tileOverlap * 2 + (tileSize.width - tileOverlap * 2) - 1)
      / (tileSize.width - tileOverlap * 2)
    let graph = z.graph
    let tileDecodedDepth = ((tileSize.depth - 1) * scaleFactor.temporal) + 1
    let resultBatchSize: Int
    if temporalOverlap == 0 {
      resultBatchSize =
        (batchSize + tileDecodedDepth - 1) / tileDecodedDepth * tileSize.depth
        - (temporalTiledEncodingConfiguration?.trailingFrameDrop ?? 0)
    } else {
      resultBatchSize = (shape[0] - 1) / scaleFactor.temporal + 1
    }
    var result = Tensor<T>(
      Array(
        repeating: 0,
        count: resultBatchSize * startHeight * startWidth * outputChannels), .CPU,
      .NHWC(resultBatchSize, startHeight, startWidth, outputChannels))
    guard !isCancelled.load(ordering: .acquiring) else {
      return graph.variable(result.toGPU(0))
    }
    for t in stride(from: 0, to: resultBatchSize, by: max(1, tileSize.depth - temporalOverlap)) {
      var encodedRawValues = [Tensor<T>]()
      let tDecodedStart: Int
      let tOutputStart: Int
      if temporalOverlap == 0 {
        tDecodedStart = t / tileSize.depth * tileDecodedDepth
        tOutputStart = t
      } else {
        tDecodedStart = min(t * scaleFactor.temporal, batchSize - tileDecodedDepth)
        tOutputStart = tDecodedStart / scaleFactor.temporal
      }
      let tEncodedStart: Int
      let isLast = t + tileSize.depth >= resultBatchSize
      if t == 0 || temporalOverlap == 0 {
        tEncodedStart = 0
      } else if t + tileSize.depth > resultBatchSize {
        tEncodedStart = (t - (resultBatchSize - tileSize.depth)) + 3
      } else {
        tEncodedStart = 3
      }
      guard tEncodedStart < tileSize.depth else { break }  // If nothing to copy, skip.
      for y in 0..<yTiles {
        let yOfs = y * (tileSize.height - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
        let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
          iOfs: yOfs * scaleFactor.spatial, length: shape[1],
          tileSize: tileSize.height * scaleFactor.spatial,
          tileOverlap: tileOverlap * scaleFactor.spatial)
        for x in 0..<xTiles {
          let xOfs = x * (tileSize.width - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
          let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
            iOfs: xOfs * scaleFactor.spatial, length: shape[2],
            tileSize: tileSize.width * scaleFactor.spatial,
            tileOverlap: tileOverlap * scaleFactor.spatial)
          var input = z[
            tDecodedStart..<min(tDecodedStart + tileDecodedDepth, batchSize),
            inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad, 0..<channels
          ].copied()
          if input.shape[0] < tileDecodedDepth {
            input = Pad(
              .replicate, begin: [0, 0, 0, 0], end: [tileDecodedDepth - input.shape[0], 0, 0, 0])(
                inputs: input)[0].as(of: T.self)
          }
          encodedRawValues.append(
            encoder(inputs: input, (causalAttentionMask.map { [$0] } ?? []))[0].as(of: T.self)
              .rawValue
              .toCPU())
          guard !isCancelled.load(ordering: .acquiring) else {
            return graph.variable(result.toGPU(0))
          }
        }
      }
      let (xWeightsAndIndexes, yWeightsAndIndexes) = xyTileWeightsAndIndexes(
        width: startWidth, height: startHeight, xTiles: xTiles, yTiles: yTiles,
        tileSize: (width: tileSize.width, height: tileSize.height), tileOverlap: tileOverlap)
      result.withUnsafeMutableBytes {
        guard let rfp = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
        for tEncoded in tEncodedStart..<min(tileSize.depth, resultBatchSize - tOutputStart) {
          var fp =
            rfp + (tEncoded + tOutputStart) * startWidth * startHeight
            * outputChannels
          let tWeight: Float
          if temporalOverlap == 0 {
            tWeight = 1
          } else if tEncoded - tEncodedStart < 2 && tEncodedStart != 0 {
            tWeight = min((Float(tEncoded - tEncodedStart) + 0.5) / 2, 1)
          } else if tileSize.depth - tEncoded <= 2 && !isLast {
            tWeight = min((Float(tileSize.depth - tEncoded) - 0.5) / 2, 1)
          } else {
            tWeight = 1
          }
          for j in 0..<startHeight {
            let yWeightAndIndex = yWeightsAndIndexes[j]
            for i in 0..<startWidth {
              let xWeightAndIndex = xWeightsAndIndexes[i]
              let tOffset = tEncoded * tileSize.width * outputChannels * tileSize.height
              for y in yWeightAndIndex {
                let yOffset = y.offset * tileSize.width * outputChannels + tOffset
                for x in xWeightAndIndex {
                  let weight = T(x.weight * y.weight * tWeight)
                  let index = y.index * xTiles + x.index
                  let tensor = encodedRawValues[index]
                  tensor.withUnsafeBytes {
                    guard var v = $0.baseAddress?.assumingMemoryBound(to: T.self) else { return }
                    // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                    v = v + x.offset * outputChannels + yOffset
                    for k in 0..<outputChannels {
                      fp[k] += v[k] * weight
                    }
                  }
                }
              }
              fp += outputChannels
            }
          }
        }
      }
    }
    return graph.variable(result.toGPU(0))
  }
}
