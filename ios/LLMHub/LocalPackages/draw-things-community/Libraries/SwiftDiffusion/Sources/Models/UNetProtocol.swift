import Atomics
import Collections
import DiffusionMappings
import Foundation
import NNC
import WeightsCache

public enum UseFlashAttention {
  case none
  case sdpa
  case quantized
}

public enum FlashAttentionLevel {
  case none
  case scale1
  case scaleMerged
  case quantized
}

public struct InjectedControlsAndAdapters<FloatType: TensorNumeric & BinaryFloatingPoint> {
  var injectedControls: [DynamicGraph.Tensor<FloatType>]
  var injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>]
  var injectedIPAdapters: [DynamicGraph.Tensor<FloatType>]
  var injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
  public init(
    injectedControls: [DynamicGraph.Tensor<FloatType>],
    injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>],
    injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
  ) {
    self.injectedControls = injectedControls
    self.injectedT2IAdapters = injectedT2IAdapters
    self.injectedIPAdapters = injectedIPAdapters
    self.injectedAttentionKVs = injectedAttentionKVs
  }
}

public struct InjectControlsAndAdapters<T: TensorNumeric & BinaryFloatingPoint> {
  public var injectControls: Bool
  public var injectT2IAdapters: Bool
  public var injectAttentionKV: Bool
  public var injectIPAdapterLengths: [Int]
  public var injectControlModels: [ControlModel<T>]
  public init(
    injectControls: Bool, injectT2IAdapters: Bool, injectAttentionKV: Bool,
    injectIPAdapterLengths: [Int], injectControlModels: [ControlModel<T>]
  ) {
    self.injectControls = injectControls
    self.injectT2IAdapters = injectT2IAdapters
    self.injectAttentionKV = injectAttentionKV
    self.injectIPAdapterLengths = injectIPAdapterLengths
    self.injectControlModels = injectControlModels
  }
}

public protocol UNetProtocol {
  associatedtype FloatType: TensorNumeric & BinaryFloatingPoint
  init()
  var isLoaded: Bool { get }
  func unloadResources()
  var version: ModelVersion { get }
  var model: AnyModel? { get }
  var modelAndWeightMapper: (AnyModel, ModelWeightMapper)? { get }
  var didRunLoRASeparately: Bool { get }
  mutating func compileModel(
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
  ) -> Bool

  func callAsFunction(
    timestep: (now: Float, next: Float), audioShiftRatio: Float,
    inputs: DynamicGraph.Tensor<FloatType>, _: DynamicGraph.Tensor<FloatType>?,
    _: [DynamicGraph.AnyTensor], extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ restInputs: [DynamicGraph.AnyTensor],
      _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
    ),
    injectedIPAdapters: [DynamicGraph.Tensor<FloatType>], referenceImageCount: Int,
    referenceAudioCount: Int, step: Int,
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    tiledDiffusion: TiledConfiguration, controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType>

  func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType>

  mutating func unloadModel()
  // This is for best-effort.
  mutating func cancel()
}

extension UNetProtocol {
  public func timeEmbed(graph: DynamicGraph, batchSize: Int, timestep: Float, version: ModelVersion)
    -> DynamicGraph.Tensor<FloatType>?
  {
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v:
      let timeEmbeddingSize = version == .kandinsky21 || version == .sdxlRefiner ? 384 : 320
      return graph.variable(
        Tensor<FloatType>(
          from: timeEmbedding(
            timestep: timestep, batchSize: batchSize,
            embeddingSize: timeEmbeddingSize,
            maxPeriod: 10_000)
        ).toGPU(0))
    case .hiDreamO1:
      return graph.variable(
        HiDreamO1TimeEmbedding(
          timestep: max(0, 1 - timestep / 1000), batchSize: batchSize, of: FloatType.self
        )
        .toGPU(0))
    case .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
      .hiDreamI1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b,
      .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .ltx2, .ltx2_3, .longcatVideoAvatar1_5,
      .minimaxH3:
      return nil
    case .seedvr2_3b, .seedvr2_7b:
      return graph.variable(
        Tensor<FloatType>(
          from: SeedVR2TimeEmbedding(
            timestep: timestep, batchSize: batchSize, embeddingSize: 256, maxPeriod: 10_000)
        ).toGPU(0))
    case .wurstchenStageC:
      let rTimeEmbed = rEmbedding(
        timesteps: timestep, batchSize: batchSize, embeddingSize: 64, maxPeriod: 10_000)
      let rZeros = rEmbedding(
        timesteps: 0, batchSize: batchSize, embeddingSize: 64, maxPeriod: 10_000)
      var rEmbed = Tensor<Float>(.CPU, .WC(batchSize, 192))
      rEmbed[0..<batchSize, 0..<64] = rTimeEmbed
      rEmbed[0..<batchSize, 64..<128] = rZeros
      rEmbed[0..<batchSize, 128..<192] = rZeros
      return graph.variable(Tensor<FloatType>(from: rEmbed).toGPU(0))
    case .wurstchenStageB:
      let rTimeEmbed = rEmbedding(
        timesteps: timestep, batchSize: batchSize, embeddingSize: 64, maxPeriod: 10_000)
      let rZeros = rEmbedding(
        timesteps: 0, batchSize: batchSize, embeddingSize: 64, maxPeriod: 10_000)
      var rEmbed = Tensor<Float>(.CPU, .WC(batchSize, 128))
      rEmbed[0..<batchSize, 0..<64] = rTimeEmbed
      rEmbed[0..<batchSize, 64..<128] = rZeros
      return graph.variable(Tensor<FloatType>(from: rEmbed).toGPU(0))
    }
  }
}

public func UNetExtractConditions<FloatType: TensorNumeric & BinaryFloatingPoint>(
  of: FloatType.Type = FloatType.self, graph: DynamicGraph, index: Int, batchSize: Int,
  tokenLengthUncond: Int, tokenLengthCond: Int, conditions: [DynamicGraph.AnyTensor],
  referenceImageCount: Int, referenceAudioCount: Int, version: ModelVersion,
  modifier: SamplerModifier, isCfgEnabled: Bool
)
  -> [DynamicGraph.AnyTensor]
{
  switch version {
  case .minimaxH3:
    let fixedConditionCount =
      50 * (18 + (referenceImageCount > 0 ? 6 : 0) + (referenceAudioCount > 0 ? 6 : 0)) + 4
    let prefixCount = 2 + referenceImageCount + referenceAudioCount
    precondition(conditions.count == fixedConditionCount + prefixCount)
    let fixedConditions = conditions[prefixCount...].map {
      let shape = $0.shape
      return DynamicGraph.Tensor<FloatType>($0)[
        index..<(index + 1), 0..<shape[1], 0..<shape[2]
      ].copied().reshaped(.WC(shape[1], shape[2]))
    }
    return Array(conditions[0..<prefixCount]) + fixedConditions
  case .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .v1, .v2, .wurstchenStageB,
    .wurstchenStageC, .seedvr2_3b, .seedvr2_7b:
    return conditions
  case .ideogram4:
    var extracted =
      Array(conditions[0..<3])
      + conditions[3..<Ideogram4ConditionCount].map {
        let shape = $0.shape
        if shape.count == 2 {
          return DynamicGraph.Tensor<FloatType>($0)[
            index..<(index + 1), 0..<shape[1]
          ].copied()
        }
        return DynamicGraph.Tensor<FloatType>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      }
    if conditions.count > Ideogram4ConditionCount {
      extracted += Array(
        conditions[
          Ideogram4ConditionCount..<(Ideogram4ConditionCount + 2)])
      extracted += conditions[(Ideogram4ConditionCount + 2)..<conditions.count].map {
        let shape = $0.shape
        if shape.count == 2 {
          return DynamicGraph.Tensor<FloatType>($0)[
            index..<(index + 1), 0..<shape[1]
          ].copied()
        }
        return DynamicGraph.Tensor<FloatType>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      }
    }
    return extracted
  case .krea2:
    return Array(conditions[0..<2])
      + conditions[2..<conditions.count].map {
        let shape = $0.shape
        if shape.count == 2 {
          return DynamicGraph.Tensor<FloatType>($0)[
            index..<(index + 1), 0..<shape[1]
          ].copied()
        }
        return DynamicGraph.Tensor<FloatType>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      }
  case .sd3, .auraflow, .sd3Large:
    return [conditions[0]]
      + conditions[1..<conditions.count].map {
        let shape = $0.shape
        return DynamicGraph.Tensor<FloatType>($0)[
          (index * batchSize)..<((index + 1) * batchSize), 0..<shape[1], 0..<shape[2]
        ]
        .copied()
      }
  case .flux1:
    let endIndex = referenceImageCount > 0 ? 3 : 2
    return conditions[0..<endIndex]
      + conditions[endIndex..<conditions.count].map {
        let shape = $0.shape
        return DynamicGraph.Tensor<FloatType>($0)[
          (index * batchSize)..<((index + 1) * batchSize), 0..<shape[1], 0..<shape[2]
        ]
        .copied()
      }
  case .hiDreamO1:
    return conditions
  case .hiDreamI1:
    return conditions[0..<50]
      + conditions[50..<conditions.count].map {
        let shape = $0.shape
        if shape.count == 2 {
          return DynamicGraph.Tensor<Float>($0)[
            (index * batchSize)..<((index + 1) * batchSize), 0..<shape[1]
          ].copied()
        } else {
          return DynamicGraph.Tensor<FloatType>($0)[
            (index * batchSize)..<((index + 1) * batchSize), 0..<shape[1], 0..<shape[2]
          ].copied()
        }
      }
  case .hunyuanVideo:
    return conditions[0..<2]
      + conditions[2..<conditions.count].enumerated().map {
        let shape = $0.1.shape
        if !isCfgEnabled {
          if $0.0 == 0 {
            return DynamicGraph.Tensor<FloatType>($0.1)[
              (index * tokenLengthCond)..<((index + 1) * tokenLengthCond), 0..<shape[1]
            ]
            .reshaped(.HWC(1, tokenLengthCond, shape[1])).copied()
          }
          return DynamicGraph.Tensor<FloatType>($0.1)[
            index..<(index + 1), 0..<shape[1], 0..<shape[2]
          ]
          .copied()
        } else {
          // Note that for Hunyuan, batchSize is num of frames.
          precondition(batchSize % 2 == 0)
          if $0.0 == 0 {
            let timesteps = shape[0] / (tokenLengthUncond + tokenLengthCond)
            return Functional.concat(
              axis: 1,
              DynamicGraph.Tensor<FloatType>($0.1)[
                (index * tokenLengthUncond)..<((index + 1) * tokenLengthUncond), 0..<shape[1]
              ]
              .reshaped(.HWC(1, tokenLengthUncond, shape[1])),
              DynamicGraph.Tensor<FloatType>($0.1)[
                (index * tokenLengthCond + tokenLengthUncond * timesteps)..<((index + 1)
                  * tokenLengthCond + tokenLengthUncond * timesteps), 0..<shape[1]
              ].reshaped(.HWC(1, tokenLengthCond, shape[1])))
          }
          let timesteps = shape[0] / 2
          return Functional.concat(
            axis: 0,
            DynamicGraph.Tensor<FloatType>($0.1)[index..<(index + 1), 0..<shape[1], 0..<shape[2]],
            DynamicGraph.Tensor<FloatType>($0.1)[
              (index + timesteps)..<(index + timesteps + 1), 0..<shape[1], 0..<shape[2]])
        }
      }
  case .longcatVideoAvatar1_5:
    // Layout:
    // [cleanCondLatents?, rot, tEmb, 48 x (4 text/audio KVs + optional 2 clean KVs)].
    // tEmb is [timesteps, time, 512] and is sliced to the current sampling step here.
    let longCatLayerCount = 48
    let longCatPerLayerConditionCount = conditions.count > 240 ? 6 : 4
    let longCatModelInputCount =
      2 + longCatLayerCount * longCatPerLayerConditionCount
    let tEmbIndex = conditions.count - longCatModelInputCount + 1
    let shape = conditions[tEmbIndex].shape
    return conditions[0..<tEmbIndex]
      + [
        DynamicGraph.Tensor<Float>(conditions[tEmbIndex])[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      ] + conditions[(tEmbIndex + 1)...]
  case .wan21_1_3b, .wan21_14b, .wan22_5b:
    return conditions[0..<1]
      + conditions[1..<7].map({
        let shape = $0.shape
        return DynamicGraph.Tensor<Float>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      }) + conditions[7..<(conditions.count - 2)]
      + conditions[(conditions.count - 2)...].map({
        let shape = $0.shape
        return DynamicGraph.Tensor<Float>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      })
  case .qwenImage2_1:
    precondition(conditions.count == 6 + 2 * 32)
    return Array(conditions.prefix(1))
      + conditions[1..<6].map {
        DynamicGraph.Tensor<FloatType>($0)[index..<(index + 1), 0..<4096].copied()
      } + Array(conditions.dropFirst(6))
  case .qwenImage:
    return conditions[0..<(conditions.count - 718)]
      + conditions[(conditions.count - 718)..<(conditions.count - 2)].map {
        let shape = $0.shape
        return DynamicGraph.Tensor<Float>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      }
      + conditions[(conditions.count - 2)..<conditions.count].map {
        let shape = $0.shape
        return DynamicGraph.Tensor<FloatType>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      }
  case .zImage:
    let count = conditions.count
    return conditions[0..<(count - 129)]
      + conditions[(count - 129)..<conditions.count].map {
        let shape = $0.shape
        if shape.count == 2 {
          return DynamicGraph.Tensor<Float>($0)[
            index..<(index + 1), 0..<shape[1]
          ].copied()
        } else {
          return DynamicGraph.Tensor<Float>($0)[
            index..<(index + 1), 0..<shape[1], 0..<shape[2]
          ].copied()
        }
      }
  case .ernieImage:
    let fixedConditionCount = isCfgEnabled ? 5 : 3
    return conditions[0..<fixedConditionCount]
      + Array(conditions[fixedConditionCount..<conditions.count]).enumerated().map {
        let shape = $0.1.shape
        if $0.0 == 2 || $0.0 == 5 {  // These are Float32.
          return DynamicGraph.Tensor<Float>($0.1)[
            index..<(index + 1), 0..<shape[1], 0..<shape[2]
          ].copied()
        } else {
          return DynamicGraph.Tensor<FloatType>($0.1)[
            index..<(index + 1), 0..<shape[1], 0..<shape[2]
          ].copied()
        }
      }
  case .cosmos2_5_2b:
    return conditions[0..<1]
      + conditions[1..<(1 + CosmosFixedTimeConditionCount)].map {
        let shape = $0.shape
        return DynamicGraph.Tensor<Float>($0)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].copied()
      }
      + conditions[(1 + CosmosFixedTimeConditionCount)..<conditions.count]
  case .flux2, .flux2_9b, .flux2_4b:
    if modifier == .kontextKv && referenceImageCount > 0 {
      let cachedKVCount: Int
      if version == .flux2_9b {
        cachedKVCount = (8 + 24) * 2
      } else if version == .flux2_4b {
        cachedKVCount = (5 + 20) * 2
      } else {
        cachedKVCount = (8 + 48) * 2
      }
      let dynamicEndIndex = conditions.count - cachedKVCount
      return conditions[0..<2]
        + conditions[2..<dynamicEndIndex].map {
          let shape = $0.shape
          if shape.count == 2 {
            return DynamicGraph.Tensor<Float>($0)[
              index..<(index + 1), 0..<shape[1]
            ].copied()
          } else {
            return DynamicGraph.Tensor<Float>($0)[
              index..<(index + 1), 0..<shape[1], 0..<shape[2]
            ].copied()
          }
        }
        + Array(conditions[dynamicEndIndex..<conditions.count])
    }
    let endIndex = referenceImageCount > 0 ? 3 : 2
    return conditions[0..<endIndex]
      + conditions[endIndex..<conditions.count].map {
        let shape = $0.shape
        if shape.count == 2 {
          return DynamicGraph.Tensor<Float>($0)[
            index..<(index + 1), 0..<shape[1]
          ].copied()
        } else {
          return DynamicGraph.Tensor<Float>($0)[
            index..<(index + 1), 0..<shape[1], 0..<shape[2]
          ].copied()
        }
      }
  case .ltx2:
    let offset = conditions.count - 1255
    let tokenModulation = referenceImageCount > 0
    let batchSize = isCfgEnabled ? batchSize / 2 : batchSize
    return conditions.enumerated().map {
      guard ($0.0 - 3 - offset) % 26 >= 4 || $0.0 - 3 - offset >= 48 * 26 else {
        return $0.1
      }
      let shape = $0.1.shape
      let modulationIndex =
        $0.0 - 3 - offset >= 48 * 26
        ? $0.0 - 3 - offset - 48 * 26 + 22 : (($0.0 - 3 - offset) % 26) - 4
      // Only do this for video modulation.
      guard tokenModulation,
        [0, 1, 2, 6, 7, 10, 11, 12, 16, 17, 18, 22, 23].contains(modulationIndex)
      else {
        return DynamicGraph.Tensor<Float>($0.1)[index..<(index + 1), 0..<shape[1], 0..<shape[2]]
          .copied()
      }
      var modulation = graph.variable(.GPU(0), .HWC(batchSize, shape[1], shape[2]), of: Float.self)
      modulation[0..<1, 0..<shape[1], 0..<shape[2]] = DynamicGraph.Tensor<Float>($0.1)[
        (shape[0] - 1)..<shape[0], 0..<shape[1], 0..<shape[2]
      ].contiguous()
      for i in 1..<batchSize {
        modulation[i..<(i + 1), 0..<shape[1], 0..<shape[2]] = DynamicGraph.Tensor<Float>($0.1)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].contiguous()
      }
      return modulation
    }
  case .ltx2_3:
    let offset = conditions.count - 1545
    let tokenModulation = referenceImageCount > 0
    let batchSize = isCfgEnabled ? batchSize / 2 : batchSize
    return conditions.enumerated().map {
      let outputIndex = $0.0 - 3 - offset
      guard outputIndex >= 0, outputIndex < 48 * 32 + 4 else {
        return $0.1
      }
      let shape = $0.1.shape
      let modulationIndex =
        outputIndex >= 48 * 32
        ? outputIndex - 48 * 32 + 32 : (outputIndex % 32)
      guard tokenModulation,
        [0, 1, 2, 6, 7, 10, 11, 12, 16, 17, 18, 22, 23, 24, 32, 33].contains(modulationIndex)
      else {
        return DynamicGraph.Tensor<Float>($0.1)[index..<(index + 1), 0..<shape[1], 0..<shape[2]]
          .copied()
      }
      var modulation = graph.variable(.GPU(0), .HWC(batchSize, shape[1], shape[2]), of: Float.self)
      modulation[0..<1, 0..<shape[1], 0..<shape[2]] = DynamicGraph.Tensor<Float>($0.1)[
        (shape[0] - 1)..<shape[0], 0..<shape[1], 0..<shape[2]
      ].contiguous()
      for i in 1..<batchSize {
        modulation[i..<(i + 1), 0..<shape[1], 0..<shape[2]] = DynamicGraph.Tensor<Float>($0.1)[
          index..<(index + 1), 0..<shape[1], 0..<shape[2]
        ].contiguous()
      }
      return modulation
    }
  case .pixart:
    var extractedConditions = [conditions[0]]
    let layers = (conditions.count - 3) / 8
    for i in 0..<layers {
      let shape = conditions[1 + i * 8].shape
      extractedConditions.append(contentsOf: [
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 1])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 2])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        conditions[1 + i * 8 + 3],
        conditions[1 + i * 8 + 4],
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 5])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 6])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
        DynamicGraph.Tensor<FloatType>(conditions[1 + i * 8 + 7])[
          index..<(index + 1), 0..<1, 0..<shape[2]
        ].copied(),
      ])
    }
    let shape = conditions[conditions.count - 2].shape
    extractedConditions.append(contentsOf: [
      DynamicGraph.Tensor<FloatType>(conditions[conditions.count - 2])[
        index..<(index + 1), 0..<1, 0..<shape[2]
      ].copied(),
      DynamicGraph.Tensor<FloatType>(conditions[conditions.count - 1])[
        index..<(index + 1), 0..<1, 0..<shape[2]
      ].copied(),
    ])
    return extractedConditions
  }
}

enum ModelBuilderOrModel {
  case modelBuilder(ModelBuilder<Bool>)
  case model(Model)
  public var unwrapped: AnyModel {
    switch self {
    case .model(let model):
      return model
    case .modelBuilder(let modelBuilder):
      return modelBuilder
    }
  }
  public var maxConcurrency: StreamContext.Concurrency {
    get {
      switch self {
      case .model(let model):
        return model.maxConcurrency
      case .modelBuilder(let modelBuilder):
        return modelBuilder.maxConcurrency
      }
    }
    set {
      switch self {
      case .model(let model):
        model.maxConcurrency = newValue
      case .modelBuilder(let modelBuilder):
        modelBuilder.maxConcurrency = newValue
      }
    }
  }
  public func cancel() {
    switch self {
    case .model(let model):
      model.cancel()
    case .modelBuilder(let modelBuilder):
      modelBuilder.cancel()
    }
  }
  public func compile(_ parameter: Bool = false, inputs: [DynamicGraph_Any], isEager: Bool = false)
  {
    switch self {
    case .model(let model):
      model.compile(inputs: inputs, isEager: isEager)
    case .modelBuilder(let modelBuilder):
      modelBuilder.compile(parameter, inputs: inputs, isEager: isEager)
    }
  }
  public func compile(_ parameter: Bool = false, inputs: DynamicGraph_Any..., isEager: Bool = false)
  {
    compile(parameter, inputs: inputs, isEager: isEager)
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup>(
    _ parameter: Bool = false, inputs firstInput: T, _ restInputs: [DynamicGraph_Any],
    streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    switch self {
    case .model(let model):
      return model(inputs: firstInput, restInputs, streamContext: streamContext)
    case .modelBuilder(let modelBuilder):
      return modelBuilder(parameter, inputs: firstInput, restInputs, streamContext: streamContext)
    }
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup>(
    _ parameter: Bool = false, inputs firstInput: T, _ restInputs: DynamicGraph_Any...,
    streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    return self(parameter, inputs: firstInput, restInputs, streamContext: streamContext)
  }
}

public struct UNetFromNNC<FloatType: TensorNumeric & BinaryFloatingPoint>: UNetProtocol {
  var teaCache: TeaCache<FloatType>? = nil
  var unet: ModelBuilderOrModel? = nil
  var unconditionalUNet: ModelBuilderOrModel? = nil
  private var usesSolAttention = false
  private var solAttentionStart = 2
  var previewer: Model? = nil
  var unetWeightMapper: ModelWeightMapper? = nil
  var timeEmbed: Model? = nil
  var modifier: SamplerModifier = .none
  var yTileWeightsAndIndexes: [[(weight: Float, index: Int, offset: Int)]]? = nil
  var xTileWeightsAndIndexes: [[(weight: Float, index: Int, offset: Int)]]? = nil
  let isCancelled = ManagedAtomic<Bool>(false)
  public private(set) var version: ModelVersion = .v1
  public init() {}
  public var isLoaded: Bool { unet != nil }
  public private(set) var didRunLoRASeparately: Bool = false
  public func unloadResources() {}
}

public func externalOnDemandPartially(
  version: ModelVersion, memoryCapacity: MemoryCapacity, externalOnDemand: Bool,
  isPartialOffloadPreferred: Bool
) -> Bool {
  guard !externalOnDemand else { return false }
  let memoryCapacity = isPartialOffloadPreferred ? MemoryCapacity.medium : memoryCapacity
  switch memoryCapacity {
  case .veryHigh:
    return false
  case .high:
    switch version {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .wan21_1_3b, .wan22_5b, .flux1, .sd3Large,
      .hunyuanVideo, .hiDreamI1, .hiDreamO1, .wan21_14b, .qwenImage, .qwenImage2_1, .zImage,
      .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ltx2, .ltx2_3, .seedvr2_3b,
      .seedvr2_7b, .ideogram4, .krea2, .longcatVideoAvatar1_5:
      return false
    case .minimaxH3:
      return true
    }
  case .medium, .low:
    switch version {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .wan21_1_3b, .wan22_5b:
      return false
    case .flux1, .sd3Large, .hunyuanVideo, .hiDreamI1, .hiDreamO1, .wan21_14b, .qwenImage,
      .qwenImage2_1, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ltx2,
      .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2, .longcatVideoAvatar1_5, .minimaxH3:
      return true
    }
  }
}

extension UNetFromNNC {
  public var model: AnyModel? { return unet?.unwrapped }
  public var modelAndWeightMapper: (AnyModel, ModelWeightMapper)? {
    guard let unet = unet, let unetWeightMapper = unetWeightMapper else { return nil }
    return (unet.unwrapped, unetWeightMapper)
  }

  public mutating func unloadModel() {
    unet = nil
    unconditionalUNet = nil
    teaCache = nil
  }

  public mutating func compileModel(
    filePath: String, externalOnDemand: Bool, deviceProperties: DeviceProperties,
    version: ModelVersion, modifier: SamplerModifier,
    qkNorm: Bool, dualAttentionLayers: [Int], upcastAttention: Bool,
    usesFlashAttention useFlashAttention: UseFlashAttention,
    usesSolAttention: Bool,
    solAttentionStart: Int, solAttentionTau: Float,
    injectControlsAndAdapters: InjectControlsAndAdapters<FloatType>, lora: [LoRAConfiguration],
    isQuantizedModel: Bool, canRunLoRASeparately: Bool, inputs xT: DynamicGraph.Tensor<FloatType>,
    _ timestep: DynamicGraph.Tensor<FloatType>?, _ c: [DynamicGraph.AnyTensor],
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    extraProjection: DynamicGraph.Tensor<FloatType>?,
    injectedControlsAndAdapters: InjectedControlsAndAdapters<FloatType>, referenceImageCount: Int,
    referenceAudioCount: Int,
    tiledDiffusion: TiledConfiguration, teaCache teaCacheConfiguration: TeaCacheConfiguration,
    causalInference: (Int, pad: Int), isBF16: Bool, activationQkScaling: [Int: Int],
    activationProjScaling: [Int: Int], activationFfnProjUpScaling: [Int: Int],
    activationFfnScaling: [Int: Int], weightsCache: WeightsCache
  ) -> Bool {
    guard unet == nil else { return true }
    isCancelled.store(false, ordering: .releasing)
    func valueOr(
      _ usesFlashAttention: UseFlashAttention, _ value: FlashAttentionLevel
    ) -> FlashAttentionLevel {
      switch usesFlashAttention {
      case .none:
        return .none
      case .quantized:
        return .quantized
      case .sdpa:
        return value
      }
    }
    let injectedControls = injectedControlsAndAdapters.injectedControls
    let injectedIPAdapters = injectedControlsAndAdapters.injectedIPAdapters
    let injectedT2IAdapters = injectedControlsAndAdapters.injectedT2IAdapters
    let injectedAttentionKVs = injectedControlsAndAdapters.injectedAttentionKVs
    let shape = xT.shape
    let batchSize = shape[0]
    var startHeight = shape[1]
    let startWidth = shape[2]
    let tiledWidth: Int
    let tiledHeight: Int
    let tiledAudioHeight: Int
    let tileScaleFactor: Int
    let graph = xT.graph
    var unet: ModelBuilderOrModel
    let lora = Array(
      (OrderedDictionary<String, LoRAConfiguration>(
        lora.filter({ $0.version == version }).map {
          ($0.file, $0)
        }
      ) {
        LoRAConfiguration(
          file: $0.file, weight: $0.weight + $1.weight, version: $0.version, isLoHa: $0.isLoHa,
          modifier: $0.modifier, mode: $0.mode)
      })
      .values
    ).filter { $0.weight != 0 }
    let (rankOfLoRA, filesRequireMerge) = LoRALoader.rank(
      graph, of: lora.map { $0.file }, modelFile: filePath)
    let isLoHa = lora.contains { $0.isLoHa }
    var configuration = LoRANetworkConfiguration(rank: rankOfLoRA, scale: 1, highPrecision: false)
    let externalOnDemandPartially = externalOnDemandPartially(
      version: version, memoryCapacity: deviceProperties.memoryCapacity,
      externalOnDemand: externalOnDemand,
      isPartialOffloadPreferred: deviceProperties.isPartialOffloadPreferred)
    let runLoRASeparatelyIsPreferred =
      isQuantizedModel || externalOnDemand || externalOnDemandPartially || isBF16
    let isTeaCacheEnabled = teaCacheConfiguration.threshold > 0
    var unconditionalUNet: ModelBuilderOrModel? = nil
    switch version {
    case .minimaxH3:
      let textLength = isCfgEnabled ? max(tokenLengthUncond, tokenLengthCond) : tokenLengthCond
      let visionLength = c[0].shape[1] - textLength
      let videoLatentFrames = batchSize / (isCfgEnabled ? 2 : 1)
      startHeight -= MiniMaxH3AudioHeight(
        videoLatentFrames: videoLatentFrames, latentWidth: startWidth)
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 4, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 4, startHeight) : startHeight
      tiledAudioHeight = MiniMaxH3AudioHeight(
        videoLatentFrames: videoLatentFrames, latentWidth: tiledWidth)
      tileScaleFactor = 4
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        configuration.keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
      }
      let usesLoRA = didRunLoRASeparately
      let frames = videoLatentFrames == 1 ? 1 : (videoLatentFrames - 2) / 5 * 17 + 5
      let audioLength = 2 * Int((Double(frames) / 24 * 40).rounded())
      let referenceImageSizes = c[2..<(2 + referenceImageCount)].map {
        (height: $0.shape[1] * 2, width: $0.shape[2] * 2)
      }
      let referenceAudioLengths = c[
        (2 + referenceImageCount)..<(2 + referenceImageCount + referenceAudioCount)
      ].map { $0.shape[1] }
      #if os(macOS) || os(iOS) || os(tvOS)
        self.usesSolAttention =
          usesSolAttention && useFlashAttention != .none
          && !DynamicGraph.flags.contains(.disableMFA)
          && !DynamicGraph.flags.contains(.disableMFAAttention)
      #else
        self.usesSolAttention = false
      #endif
      let usesSolAttention = self.usesSolAttention
      self.solAttentionStart = max(0, solAttentionStart)
      unet = .modelBuilder(
        ModelBuilder { eligibleForApproximation, inputs in
          let videoFrames = isTeaCacheEnabled ? videoLatentFrames : inputs[0].shape[0]
          let videoHeight = isTeaCacheEnabled ? tiledHeight : inputs[0].shape[1]
          let videoWidth = isTeaCacheEnabled ? tiledWidth : inputs[0].shape[2]
          let textLength =
            isTeaCacheEnabled
            ? inputs[0].shape[1] - audioLength - videoFrames * videoHeight / 2 * (videoWidth / 2)
              - referenceImageSizes.reduce(0) { $0 + $1.height / 2 * ($1.width / 2) }
              - referenceAudioLengths.reduce(0, +)
            : inputs[2].shape[1]
          let sequenceLength = inputs[isTeaCacheEnabled ? 1 : 3].shape[1]
          let videoLength = videoFrames * videoHeight / 2 * (videoWidth / 2)
          let solAttention: (eligibleForApproximation: Range<Int>, tau: Float)? =
            usesSolAttention
            ? (
              eligibleForApproximation ? (sequenceLength - videoLength)..<sequenceLength : 0..<0,
              solAttentionTau
            )
            : nil
          if usesLoRA {
            return LoRAMiniMaxH3(
              hiddenSize: 5_376, layers: isTeaCacheEnabled ? 49 : 50,
              startLayer: isTeaCacheEnabled ? 1 : 0,
              textLength: textLength, audioLength: audioLength,
              videoFrames: videoFrames, videoHeight: videoHeight, videoWidth: videoWidth,
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              usesSolAttention: solAttention,
              referenceImageSizes: referenceImageSizes,
              referenceAudioLengths: referenceAudioLengths,
              visionLength: visionLength, outputResidual: isTeaCacheEnabled,
              LoRAConfiguration: configuration
            ).1
          }
          return MiniMaxH3(
            hiddenSize: 5_376, layers: isTeaCacheEnabled ? 49 : 50,
            startLayer: isTeaCacheEnabled ? 1 : 0,
            textLength: textLength, audioLength: audioLength,
            videoFrames: videoFrames, videoHeight: videoHeight, videoWidth: videoWidth,
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            usesSolAttention: solAttention,
            referenceImageSizes: referenceImageSizes, referenceAudioLengths: referenceAudioLengths,
            visionLength: visionLength, outputResidual: isTeaCacheEnabled
          ).1
        })
      if isTeaCacheEnabled {
        let reducedModel = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textLength =
              inputs[0].shape[1] - audioLength - videoLatentFrames * tiledHeight / 2
              * (tiledWidth / 2)
              - referenceImageSizes.reduce(0) { $0 + $1.height / 2 * ($1.width / 2) }
              - referenceAudioLengths.reduce(0, +)
            if usesLoRA {
              return LoRAMiniMaxH3(
                hiddenSize: 5_376, layers: 0, startLayer: 1,
                textLength: textLength, audioLength: audioLength,
                videoFrames: videoLatentFrames, videoHeight: tiledHeight, videoWidth: tiledWidth,
                usesFlashAttention: valueOr(useFlashAttention, .scale1),
                referenceImageSizes: referenceImageSizes,
                referenceAudioLengths: referenceAudioLengths,
                visionLength: visionLength, inputResidual: true, LoRAConfiguration: configuration
              ).1
            }
            return MiniMaxH3(
              hiddenSize: 5_376, layers: 0, startLayer: 1,
              textLength: textLength, audioLength: audioLength,
              videoFrames: videoLatentFrames, videoHeight: tiledHeight, videoWidth: tiledWidth,
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              referenceImageSizes: referenceImageSizes,
              referenceAudioLengths: referenceAudioLengths,
              visionLength: visionLength, inputResidual: true
            ).1
          })
        let inferModel = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let videoShape = inputs[0].shape
            if usesLoRA {
              return LoRAMiniMaxH3(
                hiddenSize: 5_376, layers: 1,
                textLength: inputs[2].shape[1], audioLength: audioLength,
                videoFrames: videoShape[0], videoHeight: videoShape[1], videoWidth: videoShape[2],
                usesFlashAttention: valueOr(useFlashAttention, .scale1),
                referenceImageSizes: referenceImageSizes,
                referenceAudioLengths: referenceAudioLengths,
                visionLength: visionLength, outputResidual: true, LoRAConfiguration: configuration
              ).1
            }
            return MiniMaxH3(
              hiddenSize: 5_376, layers: 1,
              textLength: inputs[2].shape[1], audioLength: audioLength,
              videoFrames: videoShape[0], videoHeight: videoShape[1], videoWidth: videoShape[2],
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              referenceImageSizes: referenceImageSizes,
              referenceAudioLengths: referenceAudioLengths,
              visionLength: visionLength, outputResidual: true
            ).1
          })
        teaCache = TeaCache(
          modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
          threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
          maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
          reducedModel: reducedModel, inferModel: inferModel,
          referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount)
      }
    case .ideogram4:
      precondition(c.count >= Ideogram4ConditionCount)
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textShape = inputs[1].shape
            return LoRAIdeogram4(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textShape[1], usesFlashAttention: valueOr(useFlashAttention, .scale1),
              LoRAConfiguration: configuration
            ).1
          })
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textShape = inputs[1].shape
            return Ideogram4(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textShape[1], usesFlashAttention: valueOr(useFlashAttention, .scale1)
            ).1
          })
      }
      if c.count > Ideogram4ConditionCount {
        unconditionalUNet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            return Ideogram4(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth, textLength: 0,
              usesFlashAttention: valueOr(useFlashAttention, .scale1)
            ).1
          })
      }
    case .v1:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        unet =
          ModelBuilderOrModel.model(
            LoRAUNet(
              batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
              startWidth: tiledWidth, startHeight: tiledHeight,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              injectControls: injectControlsAndAdapters.injectControls,
              injectT2IAdapters: injectControlsAndAdapters.injectT2IAdapters,
              injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
              LoRAConfiguration: configuration
            ))
      } else {
        unet =
          ModelBuilderOrModel.model(
            UNet(
              batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
              startWidth: tiledWidth, startHeight: tiledHeight,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              injectControls: injectControlsAndAdapters.injectControls,
              injectT2IAdapters: injectControlsAndAdapters.injectT2IAdapters,
              injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
              injectAttentionKV: injectControlsAndAdapters.injectAttentionKV
            ).0)
      }
    case .v2:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        unet =
          ModelBuilderOrModel.model(
            LoRAUNetv2(
              batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
              startWidth: tiledWidth, startHeight: tiledHeight, upcastAttention: upcastAttention,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              injectControls: injectControlsAndAdapters.injectControls,
              LoRAConfiguration: configuration
            ))
      } else {
        unet =
          ModelBuilderOrModel.model(
            UNetv2(
              batchSize: batchSize, embeddingLength: (tokenLengthUncond, tokenLengthCond),
              startWidth: tiledWidth, startHeight: tiledHeight, upcastAttention: upcastAttention,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              injectControls: injectControlsAndAdapters.injectControls
            ).0)
      }
    case .svdI2v:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      let model: Model
      didRunLoRASeparately = false
      (model, _, unetWeightMapper) =
        UNetXL(
          batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
          channels: [320, 640, 1280, 1280],
          inputAttentionRes: [1: [1, 1], 2: [1, 1], 4: [1, 1]], middleAttentionBlocks: 1,
          outputAttentionRes: [1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 1, 1]], embeddingLength: (1, 1),
          injectIPAdapterLengths: [], upcastAttention: ([:], false, [1: [0, 1, 2]]),
          usesFlashAttention: valueOr(useFlashAttention, .scale1), injectControls: false,
          isTemporalMixEnabled: true, of: FloatType.self
        )
      unet = ModelBuilderOrModel.model(model)
    case .kandinsky21:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately = false
      unet = ModelBuilderOrModel.model(
        UNetKandinsky(
          batchSize: batchSize, channels: 384, outChannels: 8, channelMult: [1, 2, 3, 4],
          numResBlocks: 3, numHeadChannels: 64, t: 87, startHeight: tiledHeight,
          startWidth: tiledWidth, attentionResolutions: Set([2, 4, 8]),
          usesFlashAttention: valueOr(useFlashAttention, .scaleMerged)))
      timeEmbed = timestepEmbedding(prefix: "time_embed", channels: 384 * 4)
    case .sdxlBase:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      let model: Model
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        (model, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
            middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
            upcastAttention: upcastAttention ? ([:], false, [2: [0, 1, 2]]) : ([:], false, [:]),
            usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
            injectControls: injectControlsAndAdapters.injectControls,
            LoRAConfiguration: configuration
          )
      } else {
        (model, _, unetWeightMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [10, 10]],
            middleAttentionBlocks: 10, outputAttentionRes: [2: [2, 2, 2], 4: [10, 10, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
            upcastAttention: upcastAttention ? ([:], false, [2: [0, 1, 2]]) : ([:], false, [:]),
            usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
            injectControls: injectControlsAndAdapters.injectControls, isTemporalMixEnabled: false,
            of: FloatType.self
          )
      }
      unet = ModelBuilderOrModel.model(model)
    case .sdxlRefiner:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      let model: Model
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        (model, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
            injectControls: false, LoRAConfiguration: configuration
          )
      } else {
        (model, _, unetWeightMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [384, 768, 1536, 1536], inputAttentionRes: [2: [4, 4], 4: [4, 4]],
            middleAttentionBlocks: 4, outputAttentionRes: [2: [4, 4, 4], 4: [4, 4, 4]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond), injectIPAdapterLengths: [],
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
            injectControls: false,
            isTemporalMixEnabled: false, of: FloatType.self
          )
      }
      unet = ModelBuilderOrModel.model(model)
    case .ssd1b:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      let model: Model
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        (model, unetWeightMapper) =
          LoRAUNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
            middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            injectControls: false, LoRAConfiguration: configuration
          )
      } else {
        (model, _, unetWeightMapper) =
          UNetXL(
            batchSize: batchSize, startHeight: tiledHeight, startWidth: tiledWidth,
            channels: [320, 640, 1280], inputAttentionRes: [2: [2, 2], 4: [4, 4]],
            middleAttentionBlocks: 0, outputAttentionRes: [2: [2, 1, 1], 4: [4, 4, 10]],
            embeddingLength: (tokenLengthUncond, tokenLengthCond),
            injectIPAdapterLengths: injectControlsAndAdapters.injectIPAdapterLengths,
            upcastAttention: ([:], false, [:]),
            usesFlashAttention: valueOr(useFlashAttention, .scale1), injectControls: false,
            isTemporalMixEnabled: false, of: FloatType.self
          )
      }
      unet = ModelBuilderOrModel.model(model)
    case .wurstchenStageC:
      tiledWidth = startWidth
      tiledHeight = startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 1
      didRunLoRASeparately = false
      unet = ModelBuilderOrModel.model(
        WurstchenStageC(
          batchSize: batchSize, height: startHeight, width: startWidth,
          t: (tokenLengthUncond + 8, tokenLengthCond + 8),
          usesFlashAttention: valueOr(useFlashAttention, .scaleMerged)
        ).0)
      previewer = WurstchenStageCPreviewer()
    case .wurstchenStageB:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 16, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 16, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 16
      didRunLoRASeparately = false
      unet = ModelBuilderOrModel.model(
        WurstchenStageB(
          batchSize: batchSize, cIn: 4, height: tiledHeight, width: tiledWidth,
          usesFlashAttention: valueOr(useFlashAttention, .scaleMerged)
        ).0)
    case .sd3:
      var posEmbedMaxSize = 192
      graph.openStore(
        filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
      ) {
        guard let shape = $0.read(like: "__dit__[t-pos_embed-0-0]")?.shape else { return }
        posEmbedMaxSize = Int(Double(shape.reduce(1, *) / 1536).squareRoot().rounded())
      }
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        unet =
          ModelBuilderOrModel.model(
            LoRAMMDiT(
              batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
              width: tiledWidth, channels: 1536, layers: 24, upcast: false, qkNorm: qkNorm,
              dualAttentionLayers: dualAttentionLayers, posEmbedMaxSize: posEmbedMaxSize,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              LoRAConfiguration: configuration, of: FloatType.self
            ).1)
      } else {
        unet =
          ModelBuilderOrModel.model(
            MMDiT(
              batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
              width: tiledWidth, channels: 1536, layers: 24, upcast: false, qkNorm: qkNorm,
              dualAttentionLayers: dualAttentionLayers, posEmbedMaxSize: posEmbedMaxSize,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              of: FloatType.self
            ).1)
      }
    case .sd3Large:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet =
          ModelBuilderOrModel.model(
            LoRAMMDiT(
              batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
              width: tiledWidth, channels: 2432, layers: 38, upcast: true, qkNorm: true,
              dualAttentionLayers: [], posEmbedMaxSize: 192,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              LoRAConfiguration: configuration, of: FloatType.self
            ).1)
      } else {
        unet =
          ModelBuilderOrModel.model(
            MMDiT(
              batchSize: batchSize, t: c[0].shape[1], height: tiledHeight,
              width: tiledWidth, channels: 2432, layers: 38, upcast: true, qkNorm: true,
              dualAttentionLayers: [], posEmbedMaxSize: 192,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              of: FloatType.self
            ).1)
      }
    case .pixart:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        unet = ModelBuilderOrModel.model(
          LoRAPixArt(
            batchSize: batchSize, height: tiledHeight, width: tiledWidth, channels: 1152,
            layers: 28,
            tokenLength: (tokenLengthUncond, tokenLengthCond),
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            LoRAConfiguration: configuration, of: FloatType.self
          ).1)
      } else {
        unet = ModelBuilderOrModel.model(
          PixArt(
            batchSize: batchSize, height: tiledHeight, width: tiledWidth, channels: 1152,
            layers: 28,
            tokenLength: (tokenLengthUncond, tokenLengthCond),
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            of: FloatType.self
          ).1)
      }
    case .auraflow:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately = false
      let maxSequence =
        (try?
          (graph.openStore(
            filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
          ) {
            guard let tensor = $0.read(like: "__dit__[t-pos_embed-0-0]") else { return 64 }
            let shape = tensor.shape
            let maxSequenceSquared = Double(shape.reduce(1, *) / 3072)
            return Int(maxSequenceSquared.squareRoot().rounded())
          }).get()) ?? 64
      unet = ModelBuilderOrModel.model(
        AuraFlow(
          batchSize: batchSize, tokenLength: max(256, max(tokenLengthCond, tokenLengthUncond)),
          height: tiledHeight, width: tiledWidth, maxSequence: maxSequence, channels: 3072,
          layers: (4, 32),
          usesFlashAttention: valueOr(useFlashAttention, .scaleMerged), of: FloatType.self
        ).1)
    case .flux1:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      var injectIPAdapterLengths = [Int: [Int]]()
      for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18] {
        injectIPAdapterLengths[i] = injectControlsAndAdapters.injectIPAdapterLengths
      }
      for i in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36] {
        injectIPAdapterLengths[19 + i] = injectControlsAndAdapters.injectIPAdapterLengths
      }
      let referenceSequenceLength: Int
      let tokenLength: Int
      if referenceImageCount > 0 {
        referenceSequenceLength = c[1].shape[1]
        tokenLength = c[2].shape[1]
      } else {
        referenceSequenceLength = 0
        tokenLength = c[1].shape[1]
      }
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.model(
          LoRAFlux1(
            batchSize: isTeaCacheEnabled ? 1 : batchSize, tokenLength: tokenLength,
            referenceSequenceLength: referenceSequenceLength,
            height: tiledHeight, width: tiledWidth, channels: 3072, layers: (19, 38),
            usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
            contextPreloaded: true,
            injectControls: injectControlsAndAdapters.injectControls,
            injectIPAdapterLengths: injectIPAdapterLengths, outputResidual: isTeaCacheEnabled,
            inputResidual: false, LoRAConfiguration: configuration
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              LoRAFlux1(
                batchSize: 1, tokenLength: tokenLength,
                referenceSequenceLength: referenceSequenceLength,
                height: tiledHeight, width: tiledWidth, channels: 3072, layers: (0, 0),
                usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
                contextPreloaded: true,
                injectControls: injectControlsAndAdapters.injectControls,
                injectIPAdapterLengths: injectIPAdapterLengths, outputResidual: false,
                inputResidual: true, LoRAConfiguration: configuration
              ).1),
            inferModel: .model(
              LoRAFlux1Norm1(
                batchSize: 1, height: tiledHeight, width: tiledWidth, channels: 3072,
                LoRAConfiguration: configuration)), referenceImageCount: referenceImageCount,
            referenceAudioCount: referenceAudioCount)
        }
      } else {
        unet = ModelBuilderOrModel.model(
          Flux1(
            batchSize: isTeaCacheEnabled ? 1 : batchSize, tokenLength: tokenLength,
            referenceSequenceLength: referenceSequenceLength,
            height: tiledHeight, width: tiledWidth, channels: 3072, layers: (19, 38),
            usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
            contextPreloaded: true,
            injectControls: injectControlsAndAdapters.injectControls,
            injectIPAdapterLengths: injectIPAdapterLengths, outputResidual: isTeaCacheEnabled,
            inputResidual: false
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              Flux1(
                batchSize: 1, tokenLength: tokenLength,
                referenceSequenceLength: referenceSequenceLength,
                height: tiledHeight, width: tiledWidth, channels: 3072, layers: (0, 0),
                usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
                contextPreloaded: true,
                injectControls: injectControlsAndAdapters.injectControls,
                injectIPAdapterLengths: injectIPAdapterLengths, outputResidual: false,
                inputResidual: true
              ).1),
            inferModel: .model(
              Flux1Norm1(
                batchSize: 1, height: tiledHeight,
                width: tiledWidth, channels: 3072)), referenceImageCount: referenceImageCount,
            referenceAudioCount: referenceAudioCount)
        }
      }
    case .hunyuanVideo:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            return LoRAHunyuan(
              time: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: inputs[3].shape[1],
              channels: 3072, layers: (20, 40),
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              outputResidual: isTeaCacheEnabled, inputResidual: false,
              LoRAConfiguration: configuration
            ).1
          })
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              LoRAHunyuan(
                time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
                width: tiledWidth,
                textLength: 0,
                channels: 3072, layers: (0, 0),
                usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
                outputResidual: false, inputResidual: true, LoRAConfiguration: configuration
              ).1),
            inferModel: .model(
              HunyuanNorm1(
                time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
                width: tiledWidth, channels: 3072)))
        }
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            return Hunyuan(
              time: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: inputs[3].shape[1],
              channels: 3072, layers: (20, 40),
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              outputResidual: isTeaCacheEnabled, inputResidual: false
            ).1
          })
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              Hunyuan(
                time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
                width: tiledWidth,
                textLength: 0,
                channels: 3072, layers: (0, 0),
                usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
                outputResidual: false, inputResidual: true
              ).1),
            inferModel: .model(
              HunyuanNorm1(
                time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
                width: tiledWidth, channels: 3072)))
        }
      }
    case .longcatVideoAvatar1_5:
      precondition(!isCfgEnabled, "LongCat-Video-Avatar requires guidance scale 1 (distilled).")
      tiledWidth = startWidth
      tiledHeight = startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      // c layout:
      // [cleanCondLatents?, rot, tEmb, 48 x (4 text/audio KVs + optional 2 clean KVs)].
      // LongCat AI2V uses one clean cond latent; AVC uses one ref latent plus continuation cond latents.
      let longCatLayerCount = 48
      let longCatPerLayerConditionCount = c.count > 240 ? 6 : 4
      let longCatKVStart = c.count - longCatLayerCount * longCatPerLayerConditionCount
      let hasCleanCondLatents = longCatKVStart >= 3
      let condFrames = hasCleanCondLatents ? c[0].shape[0] : 0
      let textLength = c[longCatKVStart].shape[1]
      didRunLoRASeparately = false
      unet = ModelBuilderOrModel.model(
        LongCatVideoAvatar(
          time: batchSize, height: tiledHeight, width: tiledWidth, channels: 4_096, layers: 48,
          intermediateSize: 11_008, textLength: textLength, audioTokens: 32,
          condFrames: condFrames,
          usesFlashAttention: valueOr(useFlashAttention, .scale1),
          kvCache: longCatPerLayerConditionCount == 6
        ).1)
    case .wan21_1_3b:
      let vaceContextExists = (c[7].shape.count == 1 && c[7].shape[0] == 1)
      let vaceLayers: [Int] = vaceContextExists ? (0..<15).map { $0 * 2 } : []
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      let injectImage =
        c.count > 9 + (isCfgEnabled ? 4 : 2) * 30 + vaceLayers.count * (isCfgEnabled ? 4 : 2)
        + (vaceLayers.isEmpty ? 0 : 2)
      let textLength = vaceContextExists ? c[9].shape[1] : c[7].shape[1]
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.model(
          LoRAWan(
            channels: 1_536, layers: 30, vaceLayers: vaceLayers, intermediateSize: 8_960,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, causalInference: causalInference, injectImage: injectImage,
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            outputResidual: isTeaCacheEnabled,
            inputResidual: false, outputChannels: 16, LoRAConfiguration: configuration
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              LoRAWan(
                channels: 1_536, layers: 0, vaceLayers: [], intermediateSize: 8_960,
                time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
                width: tiledWidth, textLength: textLength, causalInference: causalInference,
                injectImage: injectImage,
                usesFlashAttention: valueOr(useFlashAttention, .scale1),
                outputResidual: false, inputResidual: true,
                outputChannels: 16, LoRAConfiguration: configuration
              ).1))
        }
      } else {
        unet = ModelBuilderOrModel.model(
          Wan(
            channels: 1_536, layers: 30, vaceLayers: vaceLayers, intermediateSize: 8_960,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, causalInference: causalInference, injectImage: injectImage,
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            outputResidual: isTeaCacheEnabled,
            inputResidual: false, outputChannels: 16
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              Wan(
                channels: 1_536, layers: 0, vaceLayers: [], intermediateSize: 8_960,
                time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
                width: tiledWidth, textLength: textLength, causalInference: causalInference,
                injectImage: injectImage,
                usesFlashAttention: valueOr(useFlashAttention, .scale1),
                outputResidual: false, inputResidual: true, outputChannels: 16
              ).1))
        }
      }
    case .wan22_5b:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 4, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 4, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 4
      let textLength = c[7].shape[1]
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.model(
          LoRAWan(
            channels: 3_072, layers: 30, vaceLayers: [], intermediateSize: 14_336,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, causalInference: causalInference, injectImage: false,
            usesFlashAttention: valueOr(useFlashAttention, .scale1), outputResidual: false,
            inputResidual: false, outputChannels: 48, LoRAConfiguration: configuration
          ).1)
      } else {
        unet = ModelBuilderOrModel.model(
          Wan(
            channels: 3_072, layers: 30, vaceLayers: [], intermediateSize: 14_336,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, causalInference: causalInference, injectImage: false,
            usesFlashAttention: valueOr(useFlashAttention, .scale1), outputResidual: false,
            inputResidual: false, outputChannels: 48
          ).1)
      }
    case .wan21_14b:
      let vaceContextExists = (c[7].shape.count == 1 && c[7].shape[0] == 1)
      let vaceLayers: [Int] = vaceContextExists ? (0..<8).map { $0 * 5 } : []
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      let injectImage =
        c.count > 9 + (isCfgEnabled ? 4 : 2) * 40 + vaceLayers.count * (isCfgEnabled ? 4 : 2)
        + (vaceLayers.isEmpty ? 0 : 2)
      let textLength = vaceContextExists ? c[9].shape[1] : c[7].shape[1]
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.model(
          LoRAWan(
            channels: 5_120, layers: 40, vaceLayers: vaceLayers, intermediateSize: 13_824,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, causalInference: causalInference, injectImage: injectImage,
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            outputResidual: isTeaCacheEnabled,
            inputResidual: false, outputChannels: 16, LoRAConfiguration: configuration
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              LoRAWan(
                channels: 5_120, layers: 0, vaceLayers: [], intermediateSize: 13_824,
                time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
                width: tiledWidth, textLength: textLength, causalInference: causalInference,
                injectImage: injectImage,
                usesFlashAttention: valueOr(useFlashAttention, .scale1),
                outputResidual: false, inputResidual: true,
                outputChannels: 16, LoRAConfiguration: configuration
              ).1))
        }
      } else {
        unet = ModelBuilderOrModel.model(
          Wan(
            channels: 5_120, layers: 40, vaceLayers: vaceLayers, intermediateSize: 13_824,
            time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight, width: tiledWidth,
            textLength: textLength, causalInference: causalInference, injectImage: injectImage,
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            outputResidual: isTeaCacheEnabled,
            inputResidual: false, outputChannels: 16
          ).1)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              Wan(
                channels: 5_120, layers: 0, vaceLayers: [], intermediateSize: 13_824,
                time: isCfgEnabled ? batchSize / 2 : batchSize, height: tiledHeight,
                width: tiledWidth, textLength: textLength, causalInference: causalInference,
                injectImage: injectImage,
                usesFlashAttention: valueOr(useFlashAttention, .scale1),
                outputResidual: false, inputResidual: true, outputChannels: 16
              ).1))
        }
      }
    case .qwenImage2_1:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 4, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 4, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 4
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        configuration.keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        unet = .modelBuilder(
          ModelBuilder { _, inputs in
            LoRAQwenImage2_1(
              FloatType.self, batchSize: inputs[0].shape[0],
              height: inputs[0].shape[1], width: inputs[0].shape[2],
              prefixLength: inputs[7].shape[1],
              channels: 4096, layers: 32,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged),
              LoRAConfiguration: configuration
            ).1
          })
      } else {
        unet = .modelBuilder(
          ModelBuilder { _, inputs in
            QwenImage2_1(
              FloatType.self, batchSize: inputs[0].shape[0],
              height: inputs[0].shape[1], width: inputs[0].shape[2],
              prefixLength: inputs[7].shape[1],
              channels: 4096, layers: 32,
              usesFlashAttention: valueOr(useFlashAttention, .scaleMerged)
            ).1
          })
      }
    case .qwenImage:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      let isQwenImageLayered = modifier == .qwenimageLayered
      let zeroTimestepForReference = referenceImageCount > 0 && modifier == .qwenimageEdit2511
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let referenceSequenceLength: Int
            let textLength = inputs[inputs.count - 719].shape[1]
            if referenceImageCount > 0 {
              referenceSequenceLength = inputs[2].shape[1]
            } else {
              referenceSequenceLength = 0
            }
            return LoRAQwenImage(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textLength, referenceSequenceLength: referenceSequenceLength,
              channels: 3_072, layers: 60,
              usesFlashAttention: valueOr(useFlashAttention, isBF16 ? .scaleMerged : .scale1),
              isBF16: isBF16, isQwenImageLayered: isQwenImageLayered,
              zeroTimestepForReference: zeroTimestepForReference,
              activationQkScaling: activationQkScaling,
              activationProjScaling: activationProjScaling,
              activationFfnProjUpScaling: activationFfnProjUpScaling,
              activationFfnScaling: activationFfnScaling,
              LoRAConfiguration: configuration
            ).1
          })
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let referenceSequenceLength: Int
            let textLength = inputs[inputs.count - 719].shape[1]
            if referenceImageCount > 0 {
              referenceSequenceLength = inputs[2].shape[1]
            } else {
              referenceSequenceLength = 0
            }
            return QwenImage(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textLength, referenceSequenceLength: referenceSequenceLength,
              channels: 3_072, layers: 60,
              usesFlashAttention: valueOr(useFlashAttention, isBF16 ? .scaleMerged : .scale1),
              isBF16: isBF16, isQwenImageLayered: isQwenImageLayered,
              zeroTimestepForReference: zeroTimestepForReference,
              activationQkScaling: activationQkScaling,
              activationProjScaling: activationProjScaling,
              activationFfnProjUpScaling: activationFfnProjUpScaling,
              activationFfnScaling: activationFfnScaling
            ).1
          })
      }
    case .zImage:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textLength = inputs[inputs.count - 130].shape[1]
            return LoRAZImage(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textLength,
              channels: 3_840, layers: 30, activationQkScaling: activationQkScaling,
              activationProjScaling: activationProjScaling,
              activationFfnProjUpScaling: activationFfnProjUpScaling,
              activationFfnScaling: activationFfnScaling,
              usesFlashAttention: valueOr(useFlashAttention, isBF16 ? .scaleMerged : .scale1),
              isBF16: isBF16,
              LoRAConfiguration: configuration
            ).0
          })
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textLength = inputs[inputs.count - 130].shape[1]
            return ZImage(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textLength,
              channels: 3_840, layers: 30, activationQkScaling: activationQkScaling,
              activationProjScaling: activationProjScaling,
              activationFfnProjUpScaling: activationFfnProjUpScaling,
              activationFfnScaling: activationFfnScaling,
              usesFlashAttention: valueOr(useFlashAttention, isBF16 ? .scaleMerged : .scale1),
              isBF16: isBF16
            ).0
          })
      }
    case .ernieImage:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textLength = inputs[3].shape[1]
            return LoRAErnieImage(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textLength, layers: 36, channels: 4_096,
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              LoRAConfiguration: configuration
            ).0
          })
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textLength = inputs[3].shape[1]
            return ErnieImage(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textLength, layers: 36, channels: 4_096,
              usesFlashAttention: valueOr(useFlashAttention, .scale1)
            ).1
          })
      }
    case .krea2:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textShape = inputs[1].shape
            return LoRAKrea2(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textShape[1], usesFlashAttention: valueOr(useFlashAttention, .scale1),
              LoRAConfiguration: configuration
            ).1
          })
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textShape = inputs[1].shape
            return Krea2(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textShape[1], usesFlashAttention: valueOr(useFlashAttention, .scale1)
            ).1
          })
      }
    case .flux2, .flux2_9b, .flux2_4b:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      let channels: Int
      let layers: (Int, Int)
      if version == .flux2_9b {
        channels = 4_096
        layers = (8, 24)
      } else if version == .flux2_4b {
        channels = 3_072
        layers = (5, 20)
      } else {
        channels = 6_144
        layers = (8, 48)
      }
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let referenceSequenceLength: Int
            let tokenLength: Int
            if modifier == .kontextKv && referenceImageCount > 0 {
              referenceSequenceLength = inputs[inputs.count - 1].shape[1]
              tokenLength = inputs[2].shape[1]
            } else if referenceImageCount > 0 {
              referenceSequenceLength = inputs[2].shape[1]
              tokenLength = inputs[3].shape[1]
            } else {
              referenceSequenceLength = 0
              tokenLength = inputs[2].shape[1]
            }
            return LoRAFlux2(
              batchSize: inputs[0].shape[0], tokenLength: tokenLength,
              referenceSequenceLength: referenceSequenceLength,
              height: tiledHeight, width: tiledWidth, channels: channels, layers: layers,
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              kvCache: modifier == .kontextKv && referenceImageCount > 0,
              LoRAConfiguration: configuration
            ).1
          })
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let referenceSequenceLength: Int
            let tokenLength: Int
            if modifier == .kontextKv && referenceImageCount > 0 {
              referenceSequenceLength = inputs[inputs.count - 1].shape[1]
              tokenLength = inputs[2].shape[1]
            } else if referenceImageCount > 0 {
              referenceSequenceLength = inputs[2].shape[1]
              tokenLength = inputs[3].shape[1]
            } else {
              referenceSequenceLength = 0
              tokenLength = inputs[2].shape[1]
            }
            return Flux2(
              batchSize: inputs[0].shape[0], tokenLength: tokenLength,
              referenceSequenceLength: referenceSequenceLength,
              height: tiledHeight, width: tiledWidth, channels: channels, layers: layers,
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              kvCache: modifier == .kontextKv && referenceImageCount > 0
            ).1
          })
      }
    case .cosmos2_5_2b:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textLength = inputs[2 + CosmosFixedTimeConditionCount].shape[1]
            return LoRACosmos(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textLength, usesFlashAttention: valueOr(useFlashAttention, .scale1),
              LoRAConfiguration: configuration
            ).0
          })
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let textLength = inputs[2 + CosmosFixedTimeConditionCount].shape[1]
            return Cosmos(
              batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
              textLength: textLength, usesFlashAttention: valueOr(useFlashAttention, .scale1)
            ).1
          })
      }
    case .hiDreamO1:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 32, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 32, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 32
      didRunLoRASeparately = false
      unet = ModelBuilderOrModel.modelBuilder(
        ModelBuilder { _, inputs in
          return HiDreamO1(
            batchSize: inputs[0].shape[0], height: tiledHeight, width: tiledWidth,
            textLength: inputs[3].shape[1], layers: 36, hiddenSize: 4_096,
            intermediateSize: 12_288)
        })
    case .hiDreamI1:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      let llama3Length = c[48].shape[1]
      let t5Length = c[49].shape[1] - llama3Length
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.model(
          LoRAHiDream(
            batchSize: 1, height: tiledHeight,
            width: modifier == .editing ? tiledWidth * 2 : tiledWidth,
            textLength: (t5Length, llama3Length), layers: (16, 32),
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            outputResidual: isTeaCacheEnabled,
            inputResidual: false, LoRAConfiguration: configuration
          ).0)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              LoRAHiDream(
                batchSize: 1, height: tiledHeight,
                width: modifier == .editing ? tiledWidth * 2 : tiledWidth,
                textLength: (t5Length, llama3Length), layers: (0, 0),
                usesFlashAttention: valueOr(useFlashAttention, .scale1),
                outputResidual: false, inputResidual: true,
                LoRAConfiguration: configuration
              ).0))
        }
      } else {
        unet = ModelBuilderOrModel.model(
          HiDream(
            batchSize: 1, height: tiledHeight,
            width: modifier == .editing ? tiledWidth * 2 : tiledWidth,
            textLength: (t5Length, llama3Length), layers: (16, 32),
            usesFlashAttention: valueOr(useFlashAttention, .scale1),
            outputResidual: isTeaCacheEnabled,
            inputResidual: false
          ).0)
        if isTeaCacheEnabled {
          teaCache = TeaCache(
            modelVersion: version, coefficients: teaCacheConfiguration.coefficients,
            threshold: teaCacheConfiguration.threshold, steps: teaCacheConfiguration.steps,
            maxSkipSteps: teaCacheConfiguration.maxSkipSteps,
            reducedModel: .model(
              HiDream(
                batchSize: 1, height: tiledHeight,
                width: modifier == .editing ? tiledWidth * 2 : tiledWidth,
                textLength: (t5Length, llama3Length), layers: (0, 0),
                usesFlashAttention: valueOr(useFlashAttention, .scale1),
                outputResidual: false, inputResidual: true
              ).0))
        }
      }
    case .ltx2:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 2, startWidth) : startWidth
      let (_, audioHeight) = Self.audioHeight(
        xT.shape, version: version, isCfgEnabled: isCfgEnabled)
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 2, startHeight - audioHeight)
        : startHeight - audioHeight
      startHeight = startHeight - audioHeight
      tiledAudioHeight =
        Self.audioHeight(
          [batchSize, 1, tiledWidth], version: version, isCfgEnabled: isCfgEnabled
        ).1
      tileScaleFactor = 2
      let tokenModulation = referenceImageCount > 0
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let shape = inputs[0].shape
            let audioFrames = inputs[1].shape[1]
            let textLength = inputs[5].shape[1]
            return LoRALTX2(
              time: shape[0], h: shape[1], w: shape[2], textLength: textLength,
              audioFrames: audioFrames, channels: (4096, 2048), layers: 48,
              tokenModulation: tokenModulation, KV: true,
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              useGatedAttention: false,
              textCrossAttentionAdaLN: false,
              LoRAConfiguration: configuration
            ).1
          })
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let shape = inputs[0].shape
            let audioFrames = inputs[1].shape[1]
            let textLength = inputs[5].shape[1]
            return LTX2(
              time: shape[0], h: shape[1], w: shape[2], textLength: textLength,
              audioFrames: audioFrames, channels: (4096, 2048), layers: 48,
              tokenModulation: tokenModulation, KV: true,
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              useGatedAttention: false,
              textCrossAttentionAdaLN: false
            ).1
          })
      }
    case .ltx2_3:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 2, startWidth) : startWidth
      let (_, audioHeight) = Self.audioHeight(
        xT.shape, version: version, isCfgEnabled: isCfgEnabled)
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 2, startHeight - audioHeight)
        : startHeight - audioHeight
      startHeight = startHeight - audioHeight
      tiledAudioHeight =
        Self.audioHeight(
          [batchSize, 1, tiledWidth], version: version, isCfgEnabled: isCfgEnabled
        ).1
      tileScaleFactor = 2
      let tokenModulation = referenceImageCount > 0
      didRunLoRASeparately =
        !lora.isEmpty && rankOfLoRA > 0 && !isLoHa && runLoRASeparatelyIsPreferred
        && canRunLoRASeparately
      if didRunLoRASeparately {
        let keys = LoRALoader.keys(graph, of: lora.map { $0.file }, modelFile: filePath)
        configuration.keys = keys
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let shape = inputs[0].shape
            let audioFrames = inputs[1].shape[1]
            let textLength = inputs[inputs.count - 2].shape[1]
            return LoRALTX2(
              time: shape[0], h: shape[1], w: shape[2], textLength: textLength,
              audioFrames: audioFrames, channels: (4096, 2048), layers: 48,
              tokenModulation: tokenModulation, KV: false,
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              useGatedAttention: true,
              textCrossAttentionAdaLN: true,
              LoRAConfiguration: configuration
            ).1
          })
      } else {
        unet = ModelBuilderOrModel.modelBuilder(
          ModelBuilder { _, inputs in
            let shape = inputs[0].shape
            let audioFrames = inputs[1].shape[1]
            let textLength = inputs[inputs.count - 2].shape[1]
            return LTX2(
              time: shape[0], h: shape[1], w: shape[2], textLength: textLength,
              audioFrames: audioFrames, channels: (4096, 2048), layers: 48,
              tokenModulation: tokenModulation, KV: false,
              usesFlashAttention: valueOr(useFlashAttention, .scale1),
              useGatedAttention: true,
              textCrossAttentionAdaLN: true
            ).1
          })
      }
    case .seedvr2_3b, .seedvr2_7b:
      tiledWidth =
        tiledDiffusion.isEnabled ? min(tiledDiffusion.tileSize.width * 8, startWidth) : startWidth
      tiledHeight =
        tiledDiffusion.isEnabled
        ? min(tiledDiffusion.tileSize.height * 8, startHeight) : startHeight
      tiledAudioHeight = 0
      tileScaleFactor = 8
      didRunLoRASeparately = false
      let configuration: SeedVR2DiTConfiguration = version == .seedvr2_7b ? ._7B : ._3B
      unet = ModelBuilderOrModel.modelBuilder(
        ModelBuilder { _, inputs in
          SeedVR2DiT(
            configuration: configuration, frames: 1, latentHeight: tiledHeight,
            latentWidth: tiledWidth,
            textLength: inputs[2].shape[0],
            usesFlashAttention: valueOr(useFlashAttention, .scaleMerged))
        })
    }
    // Need to assign version now such that sliceInputs will have the correct version.
    self.version = version
    self.modifier = modifier
    var c = c
    if injectedIPAdapters.count > 0 {
      switch version {
      case .v1:
        let injectIPAdapters = injectedIPAdapters.count / 32
        var newC = [c[0]]
        for i in stride(from: 0, to: 32, by: 2) {
          for j in 0..<injectIPAdapters {
            newC.append(injectedIPAdapters[i + j * 32])  // ip_k
            newC.append(injectedIPAdapters[i + 1 + j * 32])  // ip_v
          }
        }
        c = newC
      case .sdxlBase, .sdxlRefiner, .ssd1b:
        precondition(injectedIPAdapters.count % (c.count - 1) == 0)
        precondition((c.count - 1) % 2 == 0)
        let injectIPAdapters = injectedIPAdapters.count / (c.count - 1)
        var newC = [c[0]]
        for i in stride(from: 0, to: c.count - 1, by: 2) {
          newC.append(c[i + 1])  // k
          newC.append(c[i + 2])  // v
          for j in 0..<injectIPAdapters {
            newC.append(injectedIPAdapters[i + j * (c.count - 1)])  // ip_k
            newC.append(injectedIPAdapters[i + 1 + j * (c.count - 1)])  // ip_v
          }
        }
        c = newC
      case .flux1:
        c.append(contentsOf: injectedIPAdapters)
      case .v2, .sd3, .sd3Large, .pixart, .auraflow, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1,
        .qwenImage, .qwenImage2_1, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b,
        .cosmos2_5_2b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2,
        .longcatVideoAvatar1_5, .minimaxH3:
        fatalError()
      }
    }
    var inputs = [DynamicGraph.AnyTensor]()
    if let extraProjection = extraProjection {
      inputs.append(extraProjection.reshaped(.WC(batchSize, 384 * 4)))
    } else if let timestep = timestep {
      inputs.append(timestep)
    }
    inputs.append(contentsOf: c)
    if injectControlsAndAdapters.injectControls {
      inputs.append(contentsOf: injectedControls)
    }
    if injectControlsAndAdapters.injectT2IAdapters {
      inputs.append(contentsOf: injectedT2IAdapters)
    }
    if !injectedAttentionKVs.isEmpty {
      inputs.append(contentsOf: injectedAttentionKVs)
    }
    unet.maxConcurrency = .limit(4)
    unconditionalUNet?.maxConcurrency = .limit(4)
    let tileOverlap = min(
      min(
        tiledDiffusion.tileOverlap * tileScaleFactor / 2,
        Int((Double(tiledHeight / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor),
      Int((Double(tiledWidth / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor)
    let yTiles =
      (startHeight - tileOverlap * 2 + (tiledHeight - tileOverlap * 2) - 1)
      / (tiledHeight - tileOverlap * 2)
    let xTiles =
      (startWidth - tileOverlap * 2 + (tiledWidth - tileOverlap * 2) - 1)
      / (tiledWidth - tileOverlap * 2)
    if startWidth > tiledWidth || startHeight > tiledHeight {
      let inputs = sliceInputs(
        inputs, originalShape: shape, xyTiles: xTiles * yTiles, index: 0, inputStartYPad: 0,
        inputEndYPad: tiledHeight, inputStartXPad: 0, inputEndXPad: tiledWidth, modifier: modifier,
        referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
        tokenLength: isCfgEnabled ? max(tokenLengthUncond, tokenLengthCond) : tokenLengthCond)
      compile(
        unet, unconditionalUNet: unconditionalUNet, tokenLengthUncond: tokenLengthUncond,
        tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
        referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
        inputs: [xT.reshaped(.NHWC(shape[0], tiledHeight + tiledAudioHeight, tiledWidth, shape[3]))]
          + inputs)
    } else {
      compile(
        unet, unconditionalUNet: unconditionalUNet, tokenLengthUncond: tokenLengthUncond,
        tokenLengthCond: tokenLengthCond, isCfgEnabled: isCfgEnabled,
        referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
        inputs: [xT] + inputs)
    }
    if let timeEmbed = timeEmbed, let timestep = timestep {
      timeEmbed.compile(inputs: timestep)
    }
    let modelKey: String
    switch version {
    case .v1, .v2, .sdxlBase, .sdxlRefiner, .kandinsky21, .ssd1b, .svdI2v:
      modelKey = "unet"
    case .wurstchenStageB:
      modelKey = "stage_b"
    case .wurstchenStageC:
      modelKey = "stage_c"
    case .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b, .wan21_14b,
      .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage, .ernieImage, .flux2,
      .flux2_9b, .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .ltx2, .ltx2_3, .seedvr2_3b,
      .seedvr2_7b, .longcatVideoAvatar1_5, .minimaxH3:
      modelKey = "dit"
    }
    let externalData: DynamicGraph.Store.Codec =
      externalOnDemand
      ? .externalOnDemand : .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap)
    let loadedFromWeightsCache = weightsCache.detach(
      filePath + teaCacheConfiguration.suffix(for: version), to: unet.unwrapped.parameters)

    func shouldOffload(name: String) -> Bool {
      guard externalOnDemandPartially else {
        return false
      }
      guard name.hasSuffix("-0]") else {  // Only weights, not bias.
        return false
      }
      // These are generic for MMDiT.
      if name.contains("c_q") || name.contains("c_k") || name.contains("c_v")  // context q, k, v projection
        || name.contains("x_q") || name.contains("x_k") || name.contains("x_v")  // x q, k, v projection
        || name.contains("c_w1") || name.contains("c_w2")  // HiDream's context FFN, no proj up.
        || name.contains("x_shared") || name.contains("x_moe_w1") || name.contains("x_moe_w2")  // HiDream's x FFN, including MoE, half proj up (w1), all proj down (w2).
        || name.contains("x_linear1") || name.contains("c_linear1")  // Wan 2.1, Hunyuan, FLUX.1 FFNs, proj up.
        || name.contains("-q-") || name.contains("-k-") || name.contains("-v-")  // q, k, v projection for Z Image.
        || name.contains("_gate_proj-") || name.contains("_down_proj-")  // Z Image FFN half proj up (gate), all proj down.
      {
        return true
      }
      if version == .ideogram4 {
        if name.contains("-w3-") {
          return true
        }
      } else if version == .krea2 {
        if name.contains("to_q-") || name.contains("to_k-") || name.contains("to_v-")
          || name.contains("to_out-") || name.contains("-up-")
        {
          return true
        }
      } else if version == .wan21_14b {  // For 14B Wan 2.1, we will be more aggressive and also offload out projection.
        if name.contains("c_o-") || name.contains("x_o-") {
          return true
        }
      } else if version == .minimaxH3 {  // Keep attention output and FFN down projections resident (~5.4 GiB at 8-bit).
        if name.contains("-up-") || name.contains("-gate-") {
          return true
        }
      } else if version == .flux2 || version == .flux2_4b || version == .flux2_9b {  // FLUX.2 is a 32B-parameter model, offload all MLP layers.
        if name.contains("_up_proj-") || name.contains("_w3-") {
          return true
        }
        if version == .flux2 && (name.contains("_w1-") || name.contains("_w2-")) {
          return true
        }
      } else if (version == .ltx2 || version == .ltx2_3)
        && (name.contains("a_q-") || name.contains("a_k-") || name.contains("a_v-")
          || name.contains("cv_q-") || name.contains("cv_k-") || name.contains("cv_v-")
          || name.contains("ca_q-") || name.contains("ca_k-") || name.contains("ca_v-")
          || name.contains("ax_q-") || name.contains("ax_k-") || name.contains("ax_v-")
          || name.contains("xa_q-") || name.contains("xa_k-") || name.contains("xa_v-")
          || name.contains("a_linear1-")
          || name.contains("x_out_proj-") || name.contains("a_out_proj-"))
      {  // LTX-2 uses x/a/cv/ca/ax/xa prefixes for attention projections and x/a for FFN.
        return true
      }
      return false
    }
    func loadWeights(
      _ unet: ModelBuilderOrModel, from store: DynamicGraph.Store, loadedFromWeightsCache: Bool
    ) {
      if !lora.isEmpty && version != .kandinsky21 {
        if didRunLoRASeparately {
          let mapping: [Int: Int] = {
            switch version {
            case .sdxlBase:
              return LoRAMapping.SDUNetXLBase
            case .sdxlRefiner:
              return LoRAMapping.SDUNetXLRefiner
            case .ssd1b:
              return LoRAMapping.SDUNetXLSSD1B
            case .v1, .v2:
              return LoRAMapping.SDUNet
            case .sd3:
              return [Int: Int](
                uniqueKeysWithValues: (0..<24).map {
                  return ($0, $0)
                })
            case .pixart:
              return [Int: Int](
                uniqueKeysWithValues: (0..<28).map {
                  return ($0, $0)
                })
            case .flux1:
              return [Int: Int](
                uniqueKeysWithValues: (0..<(19 + 38)).map {
                  return ($0, $0)
                })
            case .sd3Large:
              return [Int: Int](
                uniqueKeysWithValues: (0..<38).map {
                  return ($0, $0)
                })
            case .hunyuanVideo:
              return [Int: Int](
                uniqueKeysWithValues: (0..<(20 + 40)).map {
                  return ($0, $0)
                })
            case .wan21_1_3b, .wan21_14b, .wan22_5b:
              return [Int: Int](
                uniqueKeysWithValues: (0..<40).map {
                  return ($0, $0)
                })
            case .longcatVideoAvatar1_5:
              return [Int: Int](
                uniqueKeysWithValues: (0..<48).map {
                  return ($0, $0)
                })
            case .hiDreamI1:
              return [Int: Int](
                uniqueKeysWithValues: (0..<(16 + 32)).map {
                  return ($0, $0)
                })
            case .hiDreamO1:
              return [Int: Int](
                uniqueKeysWithValues: (0..<36).map {
                  return ($0, $0)
                })
            case .qwenImage2_1:
              return Dictionary(uniqueKeysWithValues: (0..<32).map { ($0, $0) })
            case .qwenImage:
              return [Int: Int](
                uniqueKeysWithValues: (0..<60).map {
                  return ($0, $0)
                })
            case .zImage:
              return [Int: Int](
                uniqueKeysWithValues: (0..<30).map {
                  return ($0, $0)
                })
            case .ernieImage:
              return [Int: Int](
                uniqueKeysWithValues: (0..<36).map {
                  return ($0, $0)
                })
            case .auraflow:
              return [Int: Int](
                uniqueKeysWithValues: (0..<(4 + 32)).map {
                  return ($0, $0)
                })
            case .flux2, .flux2_9b, .flux2_4b:
              return [Int: Int](
                uniqueKeysWithValues: (0..<(8 + 48)).map {
                  return ($0, $0)
                })
            case .cosmos2_5_2b:
              return [Int: Int](
                uniqueKeysWithValues: (0..<28).map {
                  return ($0, $0)
                })
            case .ltx2, .ltx2_3:
              return [Int: Int](
                uniqueKeysWithValues: (0..<48).map {
                  return ($0, $0)
                })
            case .ideogram4:
              return [Int: Int](
                uniqueKeysWithValues: (0..<34).map {
                  return ($0, $0)
                })
            case .krea2:
              return [Int: Int](
                uniqueKeysWithValues: (0..<28).map {
                  return ($0, $0)
                })
            case .minimaxH3:
              return [Int: Int](uniqueKeysWithValues: (0..<50).map { ($0, $0) })
            case .kandinsky21, .svdI2v, .wurstchenStageC, .wurstchenStageB, .seedvr2_3b,
              .seedvr2_7b:
              fatalError()
            }
          }()
          ControlModelLoader<FloatType>.openStore(
            graph, injectControlModels: injectControlsAndAdapters.injectControlModels,
            version: version
          ) { controlModelLoader in
            LoRALoader.openStore(graph, lora: lora) { loader in
              store.read(
                modelKey, model: unet.unwrapped,
                codec: [.jit, .q6p, .q8p, .ezm7, .i8x, externalData]
              ) {
                name, dataType, format, shape in
                if let result = controlModelLoader.loadMergedWeight(name: name) {
                  if case .continue(let name, _, let store) = result, shouldOffload(name: name) {
                    return .continue(
                      name, codec: [.ezm7, .externalOnDemand, .q6p, .q8p, .jit], store: store)
                  } else {
                    return result
                  }
                }
                // Patch for bias weights which missing a 1/8 scale. Note that this is not needed if we merge this into the model import step like we do for Hunyuan.
                if version == .flux1
                  && (name.hasSuffix("_out_proj-17-1]") || name.hasSuffix("_out_proj-18-1]")),
                  let tensor = store.read(
                    name,
                    codec: [
                      .ezm7, .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
                      .q6p, .q8p, .i8x,
                    ])
                {
                  guard !loadedFromWeightsCache else {
                    return .fail
                  }
                  return .final(
                    graph.withNoGrad {
                      let scaleFactor: Float = 8
                      return
                        ((1 / scaleFactor)
                        * graph.variable(Tensor<FloatType>(from: tensor)).toGPU(0)).rawValue.toCPU()
                    })
                }
                let result: DynamicGraph.Store.ModelReaderResult
                if dataType == .Float32 {
                  // Keeping at higher precision for LoRA loading.
                  result = loader.concatenateLoRA(
                    graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
                    store: store, dataType: dataType, format: format, shape: shape, of: Float32.self
                  )
                } else {
                  result = loader.concatenateLoRA(
                    graph, LoRAMapping: mapping, filesRequireMerge: filesRequireMerge, name: name,
                    store: store, dataType: dataType, format: format, shape: shape,
                    of: FloatType.self)
                }
                switch result {
                case .continue(let updatedName, _, _):
                  guard updatedName == name else {
                    return result
                  }
                  guard !loadedFromWeightsCache else {
                    return .fail
                  }
                  if shouldOffload(name: name) {
                    return .continue(name, codec: [.ezm7, .externalOnDemand, .q6p, .q8p, .jit])
                  }
                  return result
                case .final(_), .fail:
                  return result
                }
              }
            }
          }
        } else {
          ControlModelLoader<FloatType>.openStore(
            graph, injectControlModels: injectControlsAndAdapters.injectControlModels,
            version: version
          ) { controlModelLoader in
            LoRALoader.openStore(graph, lora: lora) { loader in
              store.read(
                modelKey, model: unet.unwrapped,
                codec: [.jit, .q6p, .q8p, .ezm7, .i8x, externalData]
              ) {
                name, dataType, _, shape in
                if let result = controlModelLoader.loadMergedWeight(name: name) {
                  if case .continue(let name, _, let store) = result, shouldOffload(name: name) {
                    return .continue(
                      name, codec: [.ezm7, .externalOnDemand, .q6p, .q8p, .jit], store: store)
                  } else {
                    return result
                  }
                }
                // Patch for bias weights which missing a 1/8 scale. Note that this is not needed if we merge this into the model import step like we do for Hunyuan.
                if version == .flux1
                  && (name.hasSuffix("_out_proj-17-1]") || name.hasSuffix("_out_proj-18-1]")),
                  let tensor = store.read(
                    name,
                    codec: [
                      .ezm7, .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
                      .q6p, .q8p, .i8x,
                    ])
                {
                  guard !loadedFromWeightsCache else {
                    return .fail
                  }
                  return .final(
                    graph.withNoGrad {
                      let scaleFactor: Float = 8
                      return
                        ((1 / scaleFactor)
                        * graph.variable(Tensor<FloatType>(from: tensor)).toGPU(0)).rawValue.toCPU()
                    })
                }
                let result: DynamicGraph.Store.ModelReaderResult
                if dataType == .Float32 {
                  // Keeping at higher precision for LoRA loading.
                  result = loader.mergeLoRA(
                    graph, name: name, store: store, dataType: dataType, shape: shape,
                    of: Float32.self)
                } else {
                  result = loader.mergeLoRA(
                    graph, name: name, store: store, dataType: dataType, shape: shape,
                    of: FloatType.self)
                }
                switch result {
                case .continue(let updatedName, _, _):
                  guard updatedName == name else {
                    return result
                  }
                  guard !loadedFromWeightsCache else {
                    return .fail
                  }
                  if shouldOffload(name: name) {
                    return .continue(name, codec: [.ezm7, .externalOnDemand, .q6p, .q8p, .jit])
                  }
                  return result
                case .final(_), .fail:
                  return result
                }
              }
            }
          }
        }
      } else {
        ControlModelLoader<FloatType>.openStore(
          graph, injectControlModels: injectControlsAndAdapters.injectControlModels,
          version: version
        ) { controlModelLoader in
          try! store.read(
            modelKey, model: unet.unwrapped, strict: version == .minimaxH3,
            codec: [.jit, .q6p, .q8p, .ezm7, .i8x, externalData]
          ) {
            name, _, _, _ in
            if let result = controlModelLoader.loadMergedWeight(name: name) {
              if case .continue(let name, _, let store) = result, shouldOffload(name: name) {
                return .continue(
                  name, codec: [.ezm7, .externalOnDemand, .q6p, .q8p, .jit], store: store)
              } else {
                return result
              }
            }
            guard !loadedFromWeightsCache else {
              return .fail
            }
            // Patch for bias weights which missing a 1/8 scale. Note that this is not needed if we merge this into the model import step like we do for Hunyuan.
            if version == .flux1
              && (name.hasSuffix("_out_proj-17-1]") || name.hasSuffix("_out_proj-18-1]")),
              let tensor = store.read(
                name,
                codec: [
                  .ezm7, .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap), .q6p,
                  .q8p, .i8x,
                ])
            {
              return .final(
                graph.withNoGrad {
                  let scaleFactor: Float = 8
                  return
                    ((1 / scaleFactor) * graph.variable(Tensor<FloatType>(from: tensor)).toGPU(0))
                    .rawValue.toCPU()
                })
            }
            if shouldOffload(name: name) {
              return .continue(name, codec: [.ezm7, .externalOnDemand, .q6p, .q8p, .jit])
            }
            return .continue(name)
          }
        }
      }
    }
    graph.openStore(
      filePath, flags: .readOnly, externalStore: TensorData.externalStore(filePath: filePath)
    ) { store in
      loadWeights(unet, from: store, loadedFromWeightsCache: loadedFromWeightsCache)
      teaCache?.loadModels(from: store) { model, store in
        loadWeights(model, from: store, loadedFromWeightsCache: false)
      }
      if let unconditionalUNet = unconditionalUNet {
        store.read(
          "unconditional_dit", model: unconditionalUNet.unwrapped,
          codec: [.jit, .q6p, .q8p, .ezm7, .i8x, externalData]
        ) { name, _, _, _ in
          if shouldOffload(name: name) {
            return .continue(name, codec: [.ezm7, .externalOnDemand, .q6p, .q8p, .jit])
          }
          return .continue(name)
        }
      }
      if let timeEmbed = timeEmbed {
        store.read(
          "time_embed", model: timeEmbed,
          codec: [
            .q6p, .q8p, .i8x, .ezm7,
            .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
          ])
      }
      if let previewer = previewer {
        previewer.maxConcurrency = .limit(4)
        previewer.compile(inputs: xT)
        store.read(
          "previewer", model: previewer,
          codec: [
            .q6p, .q8p, .i8x, .ezm7,
            .externalData(deviceProperties.isFreadPreferred ? .fread : .mmap),
          ])
      }
    }
    self.unet = unet
    self.unconditionalUNet = unconditionalUNet
    if startWidth > tiledWidth || startHeight > tiledHeight {
      (xTileWeightsAndIndexes, yTileWeightsAndIndexes) = xyTileWeightsAndIndexes(
        width: startWidth, height: startHeight, xTiles: xTiles, yTiles: yTiles,
        tileSize: (width: tiledWidth, height: tiledHeight), tileOverlap: tileOverlap)
    }
    return true
  }

  private func sliceInputs(
    _ inputs: [DynamicGraph.AnyTensor], originalShape: TensorShape, xyTiles: Int,
    index: Int, inputStartYPad: Int, inputEndYPad: Int, inputStartXPad: Int, inputEndXPad: Int,
    modifier: SamplerModifier, referenceImageCount: Int, referenceAudioCount: Int, tokenLength: Int
  ) -> [DynamicGraph.AnyTensor] {
    let count = inputs.count
    return inputs.enumerated().map {
      // For FLUX.1, if it is the first one, we need to handle its slicing (rotary encoding).
      switch version {
      case .minimaxH3:
        if $0.0 == 1 {
          let rotary = DynamicGraph.Tensor<FloatType>($0.1)
          let shape = rotary.shape
          let frames = originalShape[0] / shape[0]
          let height =
            originalShape[1]
            - MiniMaxH3AudioHeight(
              videoLatentFrames: frames, latentWidth: originalShape[2])
          let width = originalShape[2]
          let prefixLength = shape[1] - frames * (height / 2) * (width / 2)
          var videoRotary = rotary[0..<shape[0], prefixLength..<shape[1], 0..<1, 0..<128]
            .copied().reshaped(
              format: .NHWC, shape: [shape[0], frames, height / 2, width / 2, 128])
          videoRotary = videoRotary[
            0..<shape[0], 0..<frames, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<128
          ].copied()
          let videoLength =
            frames * ((inputEndYPad - inputStartYPad) / 2)
            * ((inputEndXPad - inputStartXPad) / 2)
          videoRotary = videoRotary.reshaped(.NHWC(shape[0], videoLength, 1, 128))
          if modifier == .fl2va && referenceImageCount > 0 {
            let frameLength = height / 2 * (width / 2)
            let conditionEnd = tokenLength + referenceImageCount * frameLength
            let conditionRotary = (0..<referenceImageCount).map { index in
              rotary[
                0..<shape[0],
                (tokenLength + index * frameLength)..<(tokenLength + (index + 1) * frameLength),
                0..<1, 0..<128
              ].copied().reshaped(.NHWC(shape[0], height / 2, width / 2, 128))[
                0..<shape[0], (inputStartYPad / 2)..<(inputEndYPad / 2),
                (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<128
              ].copied().reshaped(.NHWC(shape[0], videoLength / frames, 1, 128))
            }
            return Concat(axis: 1)(
              inputs: rotary[0..<shape[0], 0..<tokenLength, 0..<1, 0..<128].copied(),
              conditionRotary + [
                rotary[0..<shape[0], conditionEnd..<prefixLength, 0..<1, 0..<128].copied(),
                videoRotary,
              ])[0]
          }
          return Functional.concat(
            axis: 1, rotary[0..<shape[0], 0..<prefixLength, 0..<1, 0..<128].copied(),
            videoRotary)
        }
        if modifier == .fl2va && (2..<(2 + referenceImageCount)).contains($0.0) {
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<1, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<$0.1.shape[3]
          ].copied()
        }
        return $0.1
      case .flux1:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let referenceSequenceLength: Int
          let tokenLength: Int
          if referenceImageCount > 0 {
            tokenLength = inputs[2].shape[1]
            referenceSequenceLength = inputs[1].shape[1]
          } else {
            tokenLength = shape[1] - (originalShape[1] / 2) * (originalShape[2] / 2)
            referenceSequenceLength = 0
          }
          let graph = $0.1.graph
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<tokenLength, 0..<shape[2], 0..<shape[3]
          ].copied()
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], tokenLength..<(shape[1] - referenceSequenceLength), 0..<shape[2],
            0..<shape[3]
          ]
          .copied().reshaped(
            .NHWC(shape[0], originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let referenceEncoding: DynamicGraph.Tensor<FloatType>?
          if referenceSequenceLength > 0 {
            referenceEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], (shape[1] - referenceSequenceLength)..<shape[1], 0..<shape[2],
              0..<shape[3]
            ].copied()
          } else {
            referenceEncoding = nil
          }
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<shape[0], (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], h * w, 1, shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], h * w + tokenLength + referenceSequenceLength, 1, shape[3]),
            of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<tokenLength, 0..<1, 0..<shape[3]] = tokenEncoding
          finalEncoding[0..<shape[0], tokenLength..<(tokenLength + h * w), 0..<1, 0..<shape[3]] =
            sliceEncoding
          if let referenceEncoding = referenceEncoding {
            finalEncoding[
              0..<shape[0], (tokenLength + h * w)..<(tokenLength + h * w + referenceSequenceLength),
              0..<1, 0..<shape[3]] = referenceEncoding
          }
          return finalEncoding
        }
      case .hiDreamI1:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let originalWidth = modifier == .editing ? originalShape[2] * 2 : originalShape[2]
          let tokenLength = shape[1] - (originalShape[1] / 2) * (originalWidth / 2)
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<(shape[1] - tokenLength), 0..<shape[2], 0..<shape[3]
          ].copied().reshaped(
            .NHWC(shape[0], originalShape[1] / 2, originalWidth / 2, shape[3]))
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], (shape[1] - tokenLength)..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied()
          let h = inputEndYPad / 2 - inputStartYPad / 2
          var w = inputEndXPad / 2 - inputStartXPad / 2
          if modifier == .editing {
            w = w * 2
          }
          // We do a continuous slice for editing.
          let sliceEncoding = imageEncoding[
            0..<shape[0], (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputStartXPad / 2 + w), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], h * w, 1, shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], h * w + tokenLength, 1, shape[3]), of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<(h * w), 0..<1, 0..<shape[3]] =
            sliceEncoding
          finalEncoding[0..<shape[0], (h * w)..<(h * w + tokenLength), 0..<1, 0..<shape[3]] =
            tokenEncoding
          return finalEncoding
        }
      case .hiDreamO1:
        if $0.0 == 1 {
          let shape = $0.1.shape
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 1..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], originalShape[1], originalShape[2], shape[3]))
          let timeEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<1, 0..<shape[2], 0..<shape[3]
          ].copied()
          let h = inputEndYPad - inputStartYPad
          let w = inputEndXPad - inputStartXPad
          let sliceEncoding = imageEncoding[
            0..<shape[0], inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad,
            0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], h * w, shape[2], shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], 1 + h * w, shape[2], shape[3]), of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<1, 0..<shape[2], 0..<shape[3]] = timeEncoding
          finalEncoding[0..<shape[0], 1..<(1 + h * w), 0..<shape[2], 0..<shape[3]] =
            sliceEncoding
          return finalEncoding
        }
      case .hunyuanVideo:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let t = shape[1] / ((originalShape[1] / 2) * (originalShape[2] / 2))
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1).reshaped(
            .NHWC(t, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          return imageEncoding[
            0..<t, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], t * h * w, 1, shape[3]))
        } else if $0.0 == 1 {
          let shape = $0.1.shape
          let t = inputs[0].shape[1] / ((originalShape[1] / 2) * (originalShape[2] / 2))
          let tokenLength =
            shape[1] - t * (originalShape[1] / 2) * (originalShape[2] / 2)
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<(shape[1] - tokenLength), 0..<shape[2], 0..<shape[3]
          ].copied().reshaped(
            .NHWC(t, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], (shape[1] - tokenLength)..<shape[1], 0..<shape[2], 0..<shape[3]
          ]
          .copied()
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<t, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], t * h * w, 1, shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], t * h * w + tokenLength, 1, shape[3]), of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<(t * h * w), 0..<1, 0..<shape[3]] = sliceEncoding
          finalEncoding[
            0..<shape[0], (t * h * w)..<(tokenLength + t * h * w), 0..<1, 0..<shape[3]] =
            tokenEncoding
          return finalEncoding
        }
      case .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b,
        .svdI2v, .v1, .v2, .wurstchenStageB, .wurstchenStageC, .seedvr2_3b, .seedvr2_7b:
        break
      case .ideogram4:
        let imageLength = (originalShape[1] / 2) * (originalShape[2] / 2)
        let hasUnconditionalConditions = count > Ideogram4ConditionCount
        if $0.0 == 1 || (hasUnconditionalConditions && $0.0 == Ideogram4ConditionCount) {
          let shape = $0.1.shape
          guard shape.count == 2, shape[0] == imageLength else { break }
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1).reshaped(
            .HWC(originalShape[1] / 2, originalShape[2] / 2, shape[1]))
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          return imageEncoding[
            (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[1]
          ].copied().reshaped(.WC(h * w, shape[1]))
        } else if $0.0 == 2
          || (hasUnconditionalConditions && $0.0 == Ideogram4ConditionCount + 1)
        {
          let shape = $0.1.shape
          guard shape.count == 4, shape[0] == 1, shape[1] >= imageLength else { break }
          let tokenLength = shape[1] - imageLength
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<imageLength, 0..<shape[2], 0..<shape[3]
          ].copied().reshaped(
            .NHWC(shape[0], originalShape[1] / 2, originalShape[2] / 2, shape[2] * shape[3]))
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<shape[0], (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<(shape[2] * shape[3])
          ].copied().reshaped(.NHWC(shape[0], h * w, shape[2], shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], h * w + tokenLength, shape[2], shape[3]),
            of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<(h * w), 0..<shape[2], 0..<shape[3]] = sliceEncoding
          if tokenLength > 0 {
            finalEncoding[
              0..<shape[0], (h * w)..<(h * w + tokenLength), 0..<shape[2], 0..<shape[3]] =
              DynamicGraph.Tensor<FloatType>($0.1)[
                0..<shape[0], imageLength..<shape[1], 0..<shape[2], 0..<shape[3]
              ].copied()
          }
          return finalEncoding
        }
      case .krea2:
        if $0.0 == 1 {
          let imageLength = (originalShape[1] / 2) * (originalShape[2] / 2)
          let shape = $0.1.shape
          guard shape.count == 4, shape[0] == 1, shape[1] > imageLength else { break }
          let tokenLength = shape[1] - imageLength
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<imageLength, 0..<shape[2], 0..<shape[3]
          ].copied().reshaped(
            .NHWC(shape[0], originalShape[1] / 2, originalShape[2] / 2, shape[2] * shape[3]))
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], imageLength..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied()
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<shape[0], (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<(shape[2] * shape[3])
          ].copied().reshaped(.NHWC(shape[0], h * w, shape[2], shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], h * w + tokenLength, shape[2], shape[3]),
            of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<(h * w), 0..<shape[2], 0..<shape[3]] = sliceEncoding
          finalEncoding[
            0..<shape[0], (h * w)..<(h * w + tokenLength), 0..<shape[2], 0..<shape[3]] =
            tokenEncoding
          return finalEncoding
        }
      case .wan21_1_3b, .wan21_14b, .wan22_5b, .longcatVideoAvatar1_5:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let t = shape[1] / ((originalShape[1] / 2) * (originalShape[2] / 2))
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1).reshaped(
            .NHWC(t, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          return imageEncoding[
            0..<t, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], t * h * w, 1, shape[3]))
        }
      case .qwenImage2_1:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let rotary = DynamicGraph.Tensor<Float>($0.1).reshaped(
            .NHWC(shape[0], originalShape[1], originalShape[2], shape[3]))
          let h = inputEndYPad - inputStartYPad
          let w = inputEndXPad - inputStartXPad
          return rotary[
            0..<shape[0], inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad,
            0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], h * w, 1, shape[3]))
        }
      case .qwenImage:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let tokenLength = shape[1] - (originalShape[1] / 2) * (originalShape[2] / 2)
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<(shape[1] - tokenLength), 0..<shape[2], 0..<shape[3]
          ].copied().reshaped(
            .NHWC(1, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], (shape[1] - tokenLength)..<shape[1], 0..<shape[2], 0..<shape[3]
          ]
          .copied()
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<1, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], h * w, 1, shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], h * w + tokenLength, 1, shape[3]), of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<(h * w), 0..<1, 0..<shape[3]] = sliceEncoding
          finalEncoding[
            0..<shape[0], (h * w)..<(tokenLength + h * w), 0..<1, 0..<shape[3]] =
            tokenEncoding
          return finalEncoding
        }
      case .zImage:
        if $0.0 == count - 131 || $0.0 == count - 132 {
          let shape = $0.1.shape
          let tokenLength = shape[1] - (originalShape[1] / 2) * (originalShape[2] / 2)
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<(shape[1] - tokenLength), 0..<shape[2], 0..<shape[3]
          ].copied().reshaped(
            .NHWC(1, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], (shape[1] - tokenLength)..<shape[1], 0..<shape[2], 0..<shape[3]
          ]
          .copied()
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<1, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], h * w, 1, shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], h * w + tokenLength, 1, shape[3]), of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<(h * w), 0..<1, 0..<shape[3]] = sliceEncoding
          finalEncoding[
            0..<shape[0], (h * w)..<(tokenLength + h * w), 0..<1, 0..<shape[3]] =
            tokenEncoding
          return finalEncoding
        }
      case .ernieImage:
        let shape = $0.1.shape
        let imageLength = (originalShape[1] / 2) * (originalShape[2] / 2)
        guard shape.count == 4, shape[0] == 1, shape[1] > imageLength, shape[3] == 128 else {
          break
        }
        let tokenLength = shape[1] - imageLength
        let graph = $0.1.graph
        let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
          0..<shape[0], 0..<imageLength, 0..<shape[2], 0..<shape[3]
        ].copied().reshaped(
          .NHWC(shape[0], originalShape[1] / 2, originalShape[2] / 2, shape[2] * shape[3]))
        let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
          0..<shape[0], imageLength..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let h = inputEndYPad / 2 - inputStartYPad / 2
        let w = inputEndXPad / 2 - inputStartXPad / 2
        let sliceEncoding = imageEncoding[
          0..<shape[0], (inputStartYPad / 2)..<(inputEndYPad / 2),
          (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<(shape[2] * shape[3])
        ].copied().reshaped(.NHWC(shape[0], h * w, shape[2], shape[3]))
        var finalEncoding = graph.variable(
          $0.1.kind, .NHWC(shape[0], h * w + tokenLength, shape[2], shape[3]),
          of: FloatType.self)
        finalEncoding[0..<shape[0], 0..<(h * w), 0..<shape[2], 0..<shape[3]] = sliceEncoding
        finalEncoding[
          0..<shape[0], (h * w)..<(h * w + tokenLength), 0..<shape[2], 0..<shape[3]] =
          tokenEncoding
        return finalEncoding
      case .flux2, .flux2_9b, .flux2_4b:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let tokenLength = shape[1] - (originalShape[1] / 2) * (originalShape[2] / 2)
          let graph = $0.1.graph
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<(shape[1] - tokenLength), 0..<shape[2],
            0..<shape[3]
          ].copied().reshaped(
            .NHWC(1, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let tokenEncoding = DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], (shape[1] - tokenLength)..<shape[1], 0..<shape[2], 0..<shape[3]
          ]
          .copied()
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          let sliceEncoding = imageEncoding[
            0..<1, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(shape[0], h * w, 1, shape[3]))
          var finalEncoding = graph.variable(
            $0.1.kind, .NHWC(shape[0], h * w + tokenLength, 1, shape[3]), of: FloatType.self)
          finalEncoding[0..<shape[0], 0..<(h * w), 0..<1, 0..<shape[3]] = sliceEncoding
          finalEncoding[
            0..<shape[0],
            (h * w)..<(tokenLength + h * w),
            0..<1, 0..<shape[3]] =
            tokenEncoding
          return finalEncoding
        }
      case .cosmos2_5_2b:
        if $0.0 == 0 {
          let shape = $0.1.shape
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1).reshaped(
            .NHWC(1, originalShape[1] / 2, originalShape[2] / 2, shape[3]))
          let h = inputEndYPad / 2 - inputStartYPad / 2
          let w = inputEndXPad / 2 - inputStartXPad / 2
          return imageEncoding[
            0..<1, (inputStartYPad / 2)..<(inputEndYPad / 2),
            (inputStartXPad / 2)..<(inputEndXPad / 2), 0..<shape[3]
          ].copied().reshaped(.NHWC(1, h * w, 1, shape[3]))
        }
      case .ltx2, .ltx2_3:
        if $0.0 == 0 || $0.0 == 2 {
          let shape = $0.1.shape
          let imageEncoding = DynamicGraph.Tensor<FloatType>($0.1).reshaped(
            format: $0.1.format,
            shape: [originalShape[0], -1, originalShape[2], shape[2] * shape[3]])
          let h = inputEndYPad - inputStartYPad
          let w = inputEndXPad - inputStartXPad
          return imageEncoding[
            0..<originalShape[0], inputStartYPad..<inputEndYPad,
            inputStartXPad..<inputEndXPad, 0..<(shape[2] * shape[3])
          ].copied().reshaped(.NHWC(1, originalShape[0] * h * w, shape[2], shape[3]))
        }
      }
      let shape = $0.1.shape
      guard shape.count == 4 else { return $0.1 }
      if shape[0] == originalShape[0] {
        // This is likely a one with xT same shape, from Wurstchen B model.
        if version == .wurstchenStageB || version == .wurstchenStageC {
          if (originalShape[1] % shape[1]) == 0 && (originalShape[2] % shape[2]) == 0
            && ((originalShape[1] / shape[1]) == (originalShape[2] / shape[2]))
          {
            // This may have issues with 3x3 convolution downsampling with strides, but luckily in UNet we deal with, these don't exist.
            let scaleFactor = originalShape[1] / shape[1]
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], (inputStartYPad / scaleFactor)..<(inputEndYPad / scaleFactor),
              (inputStartXPad / scaleFactor)..<(inputEndXPad / scaleFactor), 0..<shape[3]
            ].copied()
          }
        }
      } else if originalShape[0] * xyTiles == shape[0] {
        return DynamicGraph.Tensor<FloatType>($0.1)[
          (index * originalShape[0])..<((index + 1) * originalShape[0]), 0..<shape[1], 0..<shape[2],
          0..<shape[3]
        ].copied()
      }
      return $0.1
    }
  }

  private func compile(
    _ unet: ModelBuilderOrModel, unconditionalUNet: ModelBuilderOrModel?, tokenLengthUncond: Int,
    tokenLengthCond: Int, isCfgEnabled: Bool, referenceImageCount: Int, referenceAudioCount: Int,
    inputs: [DynamicGraph.AnyTensor]
  ) {
    switch version {
    case .minimaxH3:
      let fixedConditionCount =
        50 * (18 + (referenceImageCount > 0 ? 6 : 0) + (referenceAudioCount > 0 ? 6 : 0)) + 4
      precondition(
        inputs.count == fixedConditionCount + 3 + referenceImageCount + referenceAudioCount)
      let firstInput = DynamicGraph.Tensor<FloatType>(inputs[0])
      let shape = firstInput.shape
      precondition(shape[3] == MiniMaxH3Configuration.videoChannels)
      let videoLatentFrames = shape[0] / (isCfgEnabled ? 2 : 1)
      let audioHeight = MiniMaxH3AudioHeight(
        videoLatentFrames: videoLatentFrames, latentWidth: shape[2])
      let frames =
        videoLatentFrames == 1 ? 1 : (videoLatentFrames - 2) / 5 * 17 + 5
      let audioRows =
        2
        * Int(
          (Double(frames) / Double(MiniMaxH3Configuration.framesPerSecond) * 40).rounded())
      let videoHeight = shape[1] - audioHeight
      let video = firstInput[
        0..<videoLatentFrames, 0..<videoHeight, 0..<shape[2],
        0..<MiniMaxH3Configuration.videoChannels
      ].copied()
      let audioCapacity =
        videoLatentFrames * audioHeight * shape[2] * MiniMaxH3Configuration.videoChannels
        / MiniMaxH3Configuration.audioChannels
      let audio = firstInput[
        0..<videoLatentFrames, videoHeight..<shape[1], 0..<shape[2],
        0..<MiniMaxH3Configuration.videoChannels
      ].copied().reshaped(.WC(audioCapacity, MiniMaxH3Configuration.audioChannels))[
        0..<audioRows, 0..<MiniMaxH3Configuration.audioChannels
      ].copied().reshaped(.HWC(1, audioRows, MiniMaxH3Configuration.audioChannels))
      let text = DynamicGraph.Tensor<Float>(inputs[1])
      let rotary = DynamicGraph.Tensor<FloatType>(inputs[2])
      let modelInputs: [DynamicGraph.AnyTensor] =
        [
          video, audio,
          text[0..<1, 0..<text.shape[1], 0..<text.shape[2]].copied(),
          rotary[0..<1, 0..<rotary.shape[1], 0..<1, 0..<128].copied(),
        ]
        + inputs[3...]
      if let teaCache {
        teaCache.compile(model: unet, inputs: modelInputs)
      } else {
        unet.compile(inputs: modelInputs)
      }
      return
    case .hunyuanVideo:
      guard isCfgEnabled else {
        unet.compile(inputs: inputs)
        teaCache?.compile(model: unet, inputs: inputs)
        return
      }
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().map {
        let shape = $0.1.shape
        switch $0.0 {
        case 0:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        case 1...2:
          return $0.1
        case 3:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<max(tokenLengthUncond, tokenLengthCond), 0..<shape[2]
          ]
          .copied()
        default:
          return DynamicGraph.Tensor<FloatType>($0.1)[0..<1, 0..<shape[1], 0..<shape[2]]
        }
      }
      unet.compile(inputs: inputs)
      teaCache?.compile(model: unet, inputs: inputs)
      return
    case .longcatVideoAvatar1_5:
      precondition(!isCfgEnabled, "LongCat-Video-Avatar requires guidance scale 1 (distilled).")
      // inputs layout:
      // [x, cleanCondLatents?, rot, tEmb, 48 x (4 text/audio KVs + optional 2 clean KVs)]. The
      // optional clean cond latents are sampler-side substitution inputs, not model inputs.
      let longCatLayerCount = 48
      let longCatPerLayerConditionCount = inputs.count > 240 ? 6 : 4
      let longCatModelInputCount =
        2 + longCatLayerCount * longCatPerLayerConditionCount
      unet.compile(inputs: [inputs[0]] + Array(inputs[(inputs.count - longCatModelInputCount)...]))
      return
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      guard isCfgEnabled else {
        unet.compile(inputs: inputs)
        teaCache?.compile(model: unet, inputs: inputs)
        return
      }
      let vaceContextExists = (inputs[8].shape.count == 1 && inputs[8].shape[0] == 1)
      let vaceLayers: Int
      let injectImage: Bool
      if version == .wan21_1_3b {
        vaceLayers = vaceContextExists ? 15 : 0
        injectImage =
          inputs.count > 10 + (isCfgEnabled ? 4 : 2) * 30
          + (vaceContextExists ? 15 * (isCfgEnabled ? 4 : 2) + 2 : 0)
      } else {
        vaceLayers = vaceContextExists ? 8 : 0
        injectImage =
          inputs.count > 10 + (isCfgEnabled ? 4 : 2) * 40
          + (vaceContextExists ? 8 * (isCfgEnabled ? 4 : 2) + 2 : 0)
      }
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().compactMap {
        let shape = $0.1.shape
        switch $0.0 {
        case 0:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        case 1...(vaceContextExists ? 9 : 7), (inputs.count - 2)..<inputs.count:
          return $0.1
        default:
          if injectImage {
            if vaceContextExists, $0.0 < 9 + vaceLayers * 4 {
              if $0.0 % 2 == 0 {
                return nil
              }
            } else {
              if ($0.0 - (vaceContextExists ? 9 + vaceLayers * 4 : 7)) % 6 == 1
                || ($0.0 - (vaceContextExists ? 9 + vaceLayers * 4 : 7)) % 6 == 3
              {
                return nil  // Remove positive ones.
              }
            }
            return $0.1
          } else {
            if $0.0 % 2 == 0 {
              return $0.1
            }
            return nil
          }
        }
      }
      unet.compile(inputs: inputs)
      teaCache?.compile(model: unet, inputs: inputs)
      return
    case .flux1:
      if let teaCache = teaCache {
        let inputs = inputs.map {
          var shape = $0.shape
          guard shape[0] > 1 else {
            return $0
          }
          shape[0] = 1
          return DynamicGraph.Tensor<FloatType>($0).reshaped(format: $0.format, shape: shape)
        }
        unet.compile(inputs: inputs)
        teaCache.compile(model: unet, inputs: inputs)
        return
      }
    case .seedvr2_3b, .seedvr2_7b:
      let x = DynamicGraph.Tensor<FloatType>(inputs[0])
      let shape = x.shape
      let batchSize = isCfgEnabled ? shape[0] / 2 : shape[0]
      precondition(batchSize == 1)
      let text = DynamicGraph.Tensor<FloatType>(inputs[2])
      let textShape = text.shape
      let useConditional = isCfgEnabled && tokenLengthCond > tokenLengthUncond
      let textIndex = isCfgEnabled ? (useConditional ? 1 : 0) : min(textShape[0] - 1, 1)
      let textLength = useConditional || !isCfgEnabled ? tokenLengthCond : tokenLengthUncond
      let textInput = text[
        textIndex..<(textIndex + 1), 0..<textLength, 0..<textShape[2]
      ].copied().reshaped(.WC(textLength, textShape[2]))
      let timestep = DynamicGraph.Tensor<FloatType>(inputs[1])
      let timestepInput = timestep[0..<1, 0..<timestep.shape[1]].copied()
      let xInput = x[
        0..<1, 0..<shape[1], 0..<shape[2], 0..<shape[3]
      ].copied().reshaped(.WC(shape[1] * shape[2], shape[3]))
      unet.compile(
        inputs: [
          xInput, timestepInput, textInput,
        ] + (useConditional ? Array(inputs[18..<33]) : Array(inputs[3..<18])))
      return
    case .ideogram4:
      let x = DynamicGraph.Tensor<FloatType>(inputs[0])
      let xShape = x.shape
      let xInput: DynamicGraph.Tensor<FloatType>
      if isCfgEnabled {
        xInput = x[0..<(xShape[0] / 2), 0..<xShape[1], 0..<xShape[2], 0..<xShape[3]].copied()
      } else {
        xInput = x
      }
      let text = DynamicGraph.Tensor<FloatType>(inputs[1])
      let textShape = text.shape
      precondition(textShape.count == 3)
      let batchSize = xInput.shape[0]
      let useUnconditionalDiT = unconditionalUNet != nil
      let useConditional =
        useUnconditionalDiT || (isCfgEnabled && tokenLengthCond > tokenLengthUncond)
      let textOffset =
        useUnconditionalDiT ? 0 : (isCfgEnabled && useConditional ? tokenLengthUncond : 0)
      let textLength = useConditional || !isCfgEnabled ? tokenLengthCond : tokenLengthUncond
      let textInput = text[
        0..<batchSize, textOffset..<(textOffset + textLength), 0..<textShape[2]
      ].copied()
      let conditionEnd =
        useUnconditionalDiT ? 1 + Ideogram4ConditionCount : inputs.count
      unet.compile(
        inputs: [xInput, textInput, inputs[2], inputs[3]] + Array(inputs[4..<conditionEnd]))
      if let unconditionalUNet = unconditionalUNet {
        precondition(inputs.count > conditionEnd)
        unconditionalUNet.compile(inputs: [xInput] + Array(inputs[conditionEnd..<inputs.count]))
      }
      return
    case .krea2:
      let x = DynamicGraph.Tensor<FloatType>(inputs[0])
      let xShape = x.shape
      let xInput: DynamicGraph.Tensor<FloatType>
      if isCfgEnabled {
        xInput = x[0..<(xShape[0] / 2), 0..<xShape[1], 0..<xShape[2], 0..<xShape[3]].copied()
      } else {
        xInput = x
      }
      let text = DynamicGraph.Tensor<FloatType>(inputs[1])
      let textShape = text.shape
      let useConditional = isCfgEnabled && tokenLengthCond > tokenLengthUncond
      let textOffset = isCfgEnabled && useConditional ? tokenLengthUncond : 0
      let textLength = useConditional || !isCfgEnabled ? tokenLengthCond : tokenLengthUncond
      let batchSize = xInput.shape[0]
      let textInput = text[
        0..<batchSize, textOffset..<(textOffset + textLength), 0..<textShape[2]
      ].copied()
      unet.compile(inputs: [xInput, textInput] + Array(inputs[2...]))
      return
    case .hiDreamO1:
      unet.compile(inputs: inputs)
      return
    case .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
      .ssd1b, .svdI2v, .v1, .v2, .wurstchenStageB, .wurstchenStageC:
      break
    case .hiDreamI1:
      var inputs = inputs.map {
        var shape = $0.shape
        guard shape[0] > 1 else {
          return $0
        }
        shape[0] = 1
        return DynamicGraph.Tensor<FloatType>($0).reshaped(format: $0.format, shape: shape)
      }
      if modifier == .editing {
        // When modifier is editing, we need to transform the input s.t. the 2 * channel becomes 2 * width.
        let shape = inputs[0].shape
        inputs[0] = inputs[0].as(of: FloatType.self).contiguous().reshaped(
          format: .NHWC, shape: [shape[0], shape[1], shape[2], 2, shape[3] / 2]
        ).transposed(2, 3).reshaped(.NHWC(shape[0], shape[1], 2 * shape[2], shape[3] / 2))
          .contiguous()
      }
      if let teaCache = teaCache {
        unet.compile(inputs: Array(inputs[0..<51] + inputs[52...]))
        teaCache.compile(model: unet, inputs: inputs)
      } else {
        unet.compile(inputs: inputs)
      }
      return
    case .qwenImage2_1:
      guard isCfgEnabled else {
        unet.compile(inputs: inputs)
        return
      }
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().map {
        let shape = $0.1.shape
        switch $0.0 {
        case 0, 7...:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied()
        case 1:
          return DynamicGraph.Tensor<Float>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied()
        default:
          return $0.1
        }
      }
      unet.compile(inputs: inputs)
      return
    case .qwenImage:
      guard isCfgEnabled else {
        unet.compile(inputs: inputs)
        // TODO: TeaCache insert here.
        return
      }
      let count = inputs.count
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().map {
        let shape = $0.1.shape
        switch $0.0 {
        case 0:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        case 1:
          return $0.1
        case count - 719:  // This is 2 when reference image not provided.
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<max(tokenLengthUncond, tokenLengthCond), 0..<shape[2]
          ]
          .copied()
        default:
          return $0.1
        }
      }
      unet.compile(inputs: inputs)
      // TODO: TeaCache insert here.
      return
    case .zImage:
      guard isCfgEnabled else {
        var inputs = inputs
        inputs.remove(at: inputs.count - 131)
        unet.compile(inputs: inputs)
        // TODO: TeaCache insert here.
        return
      }
      let count = inputs.count
      let roundUpTokenLengthUncond = (tokenLengthUncond + 31) / 32 * 32
      let roundUpTokenLengthCond = (tokenLengthCond + 31) / 32 * 32
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().compactMap {
        let shape = $0.1.shape
        switch $0.0 {
        case 0:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        case count - 130:
          return DynamicGraph.Tensor<Float>($0.1)[
            0..<shape[0], 0..<max(roundUpTokenLengthUncond, roundUpTokenLengthCond), 0..<shape[2]]
        case count - 131:
          return roundUpTokenLengthUncond > roundUpTokenLengthCond ? nil : $0.1
        case count - 132:
          return roundUpTokenLengthUncond > roundUpTokenLengthCond ? $0.1 : nil
        default:
          return $0.1
        }
      }
      unet.compile(inputs: inputs)
      // TODO: TeaCache insert here.
      return
    case .ernieImage:
      guard isCfgEnabled else {
        unet.compile(inputs: inputs)
        return
      }
      let useConditional = tokenLengthCond > tokenLengthUncond
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().compactMap {
        let shape = $0.1.shape
        switch $0.0 {
        case 0:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied()
        case 1, 2:
          return useConditional ? nil : $0.1
        case 3, 4:
          return useConditional ? $0.1 : nil
        case 5:
          let tokenLength = useConditional ? tokenLengthCond : tokenLengthUncond
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<shape[0], 0..<tokenLength, 0..<shape[2]
          ].copied()
        default:
          return $0.1
        }
      }
      unet.compile(inputs: inputs)
      return
    case .flux2, .flux2_9b, .flux2_4b:
      guard isCfgEnabled else {
        unet.compile(inputs: inputs)
        // TODO: TeaCache insert here.
        return
      }
      let cachedKVCount: Int
      if modifier == .kontextKv && referenceImageCount > 0 {
        if version == .flux2_9b {
          cachedKVCount = (8 + 24) * 2
        } else if version == .flux2_4b {
          cachedKVCount = (5 + 20) * 2
        } else {
          cachedKVCount = (8 + 48) * 2
        }
      } else {
        cachedKVCount = 0
      }
      let dynamicEndIndex = inputs.count - cachedKVCount
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().map {
        let shape = $0.1.shape
        switch $0.0 {
        case 0:
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        case dynamicEndIndex - 18:
          return DynamicGraph.Tensor<Float>($0.1)[
            0..<(shape[0] / 2), 0..<max(tokenLengthUncond, tokenLengthCond), 0..<shape[2]]
        default:
          return $0.1
        }
      }
      unet.compile(inputs: inputs)
      // TODO: TeaCache insert here.
      return
    case .cosmos2_5_2b:
      guard isCfgEnabled else {
        unet.compile(inputs: inputs)
        return
      }
      let kvStartIndex = 2 + CosmosFixedTimeConditionCount
      let inputs: [DynamicGraph.AnyTensor] = inputs.enumerated().compactMap {
        let shape = $0.1.shape
        if $0.0 == 0 {
          let value = DynamicGraph.Tensor<FloatType>($0.1)
          return value[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
        } else if $0.0 < kvStartIndex {
          return $0.1
        } else {
          return (($0.0 - kvStartIndex) % 2 == 0) ? $0.1 : nil
        }
      }
      unet.compile(inputs: inputs)
      return
    case .ltx2:
      let firstInput = DynamicGraph.Tensor<FloatType>(inputs[0])
      var shape = firstInput.shape
      if isCfgEnabled {
        shape[0] = shape[0] / 2
      }
      // Separate firstInput into video input and audio input.
      let batchSize = shape[0]
      let startWidth = shape[2]
      let (audioFrames, audioHeight) = LTX2ExtractAudioFramesAndHeight(shape)
      let startHeight = shape[1] - audioHeight
      let videoInput = firstInput[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<shape[3]]
        .contiguous()
      let audioInput = firstInput[
        0..<batchSize, startHeight..<shape[1], 0..<startWidth, 0..<shape[3]
      ].contiguous().reshaped(.HWC(1, audioFrames, shape[3]))
      let restInputs = Array(inputs[(inputs.count - 1255)...])
      guard isCfgEnabled else {
        unet.compile(inputs: [videoInput, audioInput] + restInputs)
        return
      }
      unet.compile(
        inputs: [videoInput, audioInput]
          + restInputs.enumerated().map {
            guard $0.0 >= 3, ($0.0 - 3) % 26 < 4, $0.0 - 3 < 48 * 26 else { return $0.1 }
            // For text conditioning, just extract the first half.
            let shape = $0.1.shape
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          })
      return
    case .ltx2_3:
      let firstInput = DynamicGraph.Tensor<FloatType>(inputs[0])
      var shape = firstInput.shape
      if isCfgEnabled {
        shape[0] = shape[0] / 2
      }
      let batchSize = shape[0]
      let startWidth = shape[2]
      let (audioFrames, audioHeight) = LTX2ExtractAudioFramesAndHeight(shape)
      let startHeight = shape[1] - audioHeight
      let videoInput = firstInput[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<shape[3]]
        .contiguous()
      let audioInput = firstInput[
        0..<batchSize, startHeight..<shape[1], 0..<startWidth, 0..<shape[3]
      ].contiguous().reshaped(.HWC(1, audioFrames, shape[3]))
      let restInputs = Array(inputs[(inputs.count - 1545)...])
      guard isCfgEnabled else {
        unet.compile(inputs: [videoInput, audioInput] + restInputs)
        return
      }
      unet.compile(
        inputs: [videoInput, audioInput]
          + restInputs.enumerated().map {
            guard $0.0 >= 3 + 48 * 32 + 4 else { return $0.1 }
            let shape = $0.1.shape
            if shape.count == 3 {
              return DynamicGraph.Tensor<FloatType>($0.1)[
                0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2]
              ].copied()
            } else {
              return DynamicGraph.Tensor<FloatType>($0.1)[
                0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
              ]
            }
          })
      return
    }
    unet.compile(inputs: inputs)
  }

  private func callAsFunction(
    referenceImageCount: Int, referenceAudioCount: Int, step: Int,
    timestep: (now: Float, next: Float),
    audioShiftRatio: Float, index: Int,
    tokenLengthUncond: Int, tokenLengthCond: Int, isCfgEnabled: Bool,
    inputs firstInput: DynamicGraph.Tensor<FloatType>,
    _ restInputs: [DynamicGraph.AnyTensor]
  ) -> DynamicGraph.Tensor<FloatType> {
    guard let unet = unet else { return firstInput }
    switch version {
    case .minimaxH3:
      let fixedConditionCount =
        50 * (18 + (referenceImageCount > 0 ? 6 : 0) + (referenceAudioCount > 0 ? 6 : 0)) + 4
      precondition(
        restInputs.count == fixedConditionCount + 2 + referenceImageCount + referenceAudioCount)
      let graph = firstInput.graph
      let eligibleForApproximation = usesSolAttention && step >= solAttentionStart
      // The RF schedule repeats alpha = 1 at the clean endpoint. The reference stops before this
      // zero-delta evaluation, and evaluating H3 at timestep zero can produce non-finite FP16 values.
      guard timestep.now > 0 else {
        let velocity = graph.variable(like: firstInput)
        velocity.full(0)
        return velocity
      }
      let shape = firstInput.shape
      precondition(shape[3] == MiniMaxH3Configuration.videoChannels)
      let videoLatentFrames = shape[0] / (isCfgEnabled ? 2 : 1)
      let audioHeight = MiniMaxH3AudioHeight(
        videoLatentFrames: videoLatentFrames, latentWidth: shape[2])
      let frames =
        videoLatentFrames == 1 ? 1 : (videoLatentFrames - 2) / 5 * 17 + 5
      let audioRows =
        2
        * Int(
          (Double(frames) / Double(MiniMaxH3Configuration.framesPerSecond) * 40).rounded())
      let videoHeight = shape[1] - audioHeight
      let audioCapacity =
        videoLatentFrames * audioHeight * shape[2] * MiniMaxH3Configuration.videoChannels
        / MiniMaxH3Configuration.audioChannels
      let videoSigma = timestep.now / 1_000
      let nextVideoSigma = timestep.next / 1_000
      let videoDelta = videoSigma - nextVideoSigma
      let audioScale =
        videoDelta > 0
        ? (MiniMaxH3AudioSigma(forVideoSigma: videoSigma, audioShiftRatio: audioShiftRatio)
          - MiniMaxH3AudioSigma(forVideoSigma: nextVideoSigma, audioShiftRatio: audioShiftRatio))
          / videoDelta : 0
      let text = DynamicGraph.Tensor<Float>(restInputs[0])
      let rotary = DynamicGraph.Tensor<FloatType>(restInputs[1])
      let textLength = isCfgEnabled ? max(tokenLengthUncond, tokenLengthCond) : tokenLengthCond
      var velocity = graph.variable(like: firstInput)
      // Compile and evaluate the longest text branch first, like the other video models.
      for branch in (isCfgEnabled ? (tokenLengthCond > tokenLengthUncond ? [1, 0] : [0, 1]) : [0]) {
        let frameRange = (branch * videoLatentFrames)..<((branch + 1) * videoLatentFrames)
        let video = firstInput[
          frameRange, 0..<videoHeight, 0..<shape[2], 0..<MiniMaxH3Configuration.videoChannels
        ].copied()
        let audio = firstInput[
          frameRange, videoHeight..<shape[1], 0..<shape[2],
          0..<MiniMaxH3Configuration.videoChannels
        ].copied().reshaped(.WC(audioCapacity, MiniMaxH3Configuration.audioChannels))[
          0..<audioRows, 0..<MiniMaxH3Configuration.audioChannels
        ].copied().reshaped(.HWC(1, audioRows, MiniMaxH3Configuration.audioChannels))
        let length = isCfgEnabled && branch == 0 ? tokenLengthUncond : tokenLengthCond
        let branchText: DynamicGraph.Tensor<Float>
        if length < textLength && text.shape[1] > textLength {
          branchText = Functional.concat(
            axis: 1, text[branch..<(branch + 1), 0..<length, 0..<text.shape[2]].copied(),
            text[branch..<(branch + 1), textLength..<text.shape[1], 0..<text.shape[2]].copied())
        } else {
          branchText = text[
            branch..<(branch + 1), 0..<(length + text.shape[1] - textLength), 0..<text.shape[2]
          ].copied()
        }
        let branchRotary = Functional.concat(
          axis: 1,
          rotary[branch..<(branch + 1), 0..<length, 0..<1, 0..<128].copied(),
          rotary[
            branch..<(branch + 1), textLength..<rotary.shape[1], 0..<1, 0..<128
          ].copied())
        let modelInputs: [DynamicGraph.AnyTensor] =
          [video, audio, branchText, branchRotary] + restInputs[2...]
        let result: [DynamicGraph.AnyTensor]
        if let teaCache {
          let firstInputs = Array(
            modelInputs.prefix(
              4 + referenceImageCount + referenceAudioCount + 18 + (referenceImageCount > 0 ? 6 : 0)
                + (referenceAudioCount > 0 ? 6 : 0)))
          let firstOutput = teaCache.infer(inputs: firstInputs)
          let hiddenState = firstOutput[0]
          let firstBlock = firstOutput[1].as(of: Float.self)
          let channels = firstBlock.shape[2]
          let videoLength = videoLatentFrames * videoHeight / 2 * (shape[2] / 2)
          let videoStart = branchRotary.shape[1] - videoLength
          let visionLength = text.shape[1] - textLength
          let audioStart = videoStart - visionLength - audioRows
          let signals: [DynamicGraph.AnyTensor] = [
            firstBlock[0..<1, audioStart..<(audioStart + audioRows), 0..<channels].copied(),
            firstBlock[0..<1, videoStart..<(videoStart + videoLength), 0..<channels].copied(),
          ]
          let marker = index * (isCfgEnabled ? 2 : 1) + branch
          let tailInputs = [modelInputs[3]] + modelInputs.dropFirst(firstInputs.count)
          if usesSolAttention && step == solAttentionStart {
            unet.compile(true, inputs: [hiddenState] + tailInputs)
          }
          let shouldUseCache = teaCache.shouldUseCacheForTimeEmbedding(
            signals, model: unet, step: step, marker: marker, of: Float.self)
          if shouldUseCache,
            let cached = teaCache(model: unet, inputs: hiddenState, tailInputs, marker: marker)
          {
            result = cached
          } else {
            result = unet(eligibleForApproximation, inputs: hiddenState, tailInputs)
            teaCache.cache(outputs: result, marker: marker)
          }
        } else {
          if usesSolAttention && step == solAttentionStart {
            unet.compile(true, inputs: modelInputs)
          }
          result = unet(eligibleForApproximation, inputs: video, Array(modelInputs.dropFirst()))
        }
        let videoVelocity = DynamicGraph.Tensor<FloatType>(from: result[0])
        let audioVelocity = DynamicGraph.Tensor<FloatType>(from: result[1]) * audioScale
        var packedAudio = graph.variable(
          .GPU(0), .WC(audioCapacity, MiniMaxH3Configuration.audioChannels), of: FloatType.self)
        packedAudio.full(0)
        packedAudio[0..<audioRows, 0..<MiniMaxH3Configuration.audioChannels] =
          audioVelocity.reshaped(.WC(audioRows, MiniMaxH3Configuration.audioChannels))
        let packedVelocity = Functional.concat(
          axis: 1, videoVelocity,
          packedAudio.reshaped(
            .NHWC(videoLatentFrames, audioHeight, shape[2], MiniMaxH3Configuration.videoChannels)))
        guard isCfgEnabled else { return packedVelocity }
        velocity[frameRange, 0..<shape[1], 0..<shape[2], 0..<shape[3]] = packedVelocity
        graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else { break }
      }
      return velocity
    case .hunyuanVideo:
      guard isCfgEnabled else {
        let shouldUseCache =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [firstInput] + restInputs, model: unet, step: step, marker: index, of: FloatType.self)
          ?? false
        let et: DynamicGraph.Tensor<FloatType>
        if shouldUseCache,
          let result = teaCache!(model: unet, inputs: firstInput, restInputs, marker: index)
        {
          et = result[0].as(of: FloatType.self)
        } else {
          let result = unet(
            inputs: firstInput, restInputs
          )
          et = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: index)
        }
        return et
      }
      let shape = firstInput.shape
      let tokenLength = max(tokenLengthUncond, tokenLengthCond)
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      if tokenLengthCond > tokenLengthUncond {
        // This if-clause is useful because we compiled the graph with longest token, so later we don't need to trigger the automatic re-compilation.
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let otherConds = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return $0.1
          case 1:
            let imageLength = shape[1] - tokenLength
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<(imageLength + tokenLengthCond), 0..<shape[2], 0..<shape[3]
            ].copied()
          case 2:
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
              0..<shape[2]
            ].copied()
          default:
            return DynamicGraph.Tensor<FloatType>($0.1)[1..<2, 0..<shape[1], 0..<shape[2]]
              .copied()
          }
        }
        // While xCond == xUncond, the pooled condition (used for adaptive layernorm) is different between cond / uncond branches, therefore, we need to check this for both cond / uncond branches.
        let shouldUseCacheCond =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [xCond] + otherConds, model: unet, step: step, marker: index * 2, of: FloatType.self)
          ?? false
        if shouldUseCacheCond,
          let result = teaCache!(model: unet, inputs: xCond, otherConds, marker: index * 2)
        {
          etCond = result[0].as(of: FloatType.self)
        } else {
          let result = unet(inputs: xCond, otherConds)
          etCond = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: index * 2)
        }
        etCond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return $0.1
          case 1:
            let imageLength = shape[1] - tokenLength
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<(imageLength + tokenLengthUncond), 0..<shape[2], 0..<shape[3]
            ].copied()
          case 2:
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return DynamicGraph.Tensor<FloatType>($0.1)[0..<1, 0..<shape[1], 0..<shape[2]]
              .copied()
          }
        }
        let shouldUseCacheUncond =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [xUncond] + otherUnconds, model: unet, step: step, marker: index * 2 + 1,
            of: FloatType.self)
          ?? false
        if shouldUseCacheUncond,
          let result = teaCache!(model: unet, inputs: xUncond, otherUnconds, marker: index * 2 + 1)
        {
          etUncond = result[0].as(of: FloatType.self)
        } else {
          let result = unet(inputs: xUncond, otherUnconds)
          etUncond = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: index * 2 + 1)
        }
      } else {
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return $0.1
          case 1:
            let imageLength = shape[1] - tokenLength
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<(imageLength + tokenLengthUncond), 0..<shape[2], 0..<shape[3]
            ].copied()
          case 2:
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return DynamicGraph.Tensor<FloatType>($0.1)[0..<1, 0..<shape[1], 0..<shape[2]]
              .copied()
          }
        }
        let shouldUseCacheUncond =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [xUncond] + otherUnconds, model: unet, step: step, marker: index * 2 + 1,
            of: FloatType.self)
          ?? false
        if shouldUseCacheUncond,
          let result = teaCache!(model: unet, inputs: xUncond, otherUnconds, marker: index * 2 + 1)
        {
          etUncond = result[0].as(of: FloatType.self)
        } else {
          let result = unet(inputs: xUncond, otherUnconds)
          etUncond = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: index * 2 + 1)
        }
        etUncond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let otherConds = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return $0.1
          case 1:
            let imageLength = shape[1] - tokenLength
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<(imageLength + tokenLengthCond), 0..<shape[2], 0..<shape[3]
            ].copied()
          case 2:
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
              0..<shape[2]
            ].copied()
          default:
            return DynamicGraph.Tensor<FloatType>($0.1)[1..<2, 0..<shape[1], 0..<shape[2]]
              .copied()
          }
        }
        let shouldUseCacheCond =
          teaCache?.shouldUseCacheForTimeEmbedding(
            [xCond] + otherConds, model: unet, step: step, marker: index * 2, of: FloatType.self)
          ?? false
        if shouldUseCacheCond,
          let result = teaCache!(model: unet, inputs: xCond, otherConds, marker: index * 2)
        {
          etCond = result[0].as(of: FloatType.self)
        } else {
          let result = unet(inputs: xCond, otherConds)
          etCond = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: index * 2)
        }
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .longcatVideoAvatar1_5:
      precondition(!isCfgEnabled, "LongCat-Video-Avatar requires guidance scale 1 (distilled).")
      let shape = firstInput.shape
      let longCatLayerCount = 48
      let longCatPerLayerConditionCount = restInputs.count > 240 ? 6 : 4
      let longCatKVStart = restInputs.count - longCatLayerCount * longCatPerLayerConditionCount
      let hasCleanCondLatents = longCatKVStart >= 3
      let longCatModelInputCount =
        2 + longCatLayerCount * longCatPerLayerConditionCount
      let kvCache = longCatPerLayerConditionCount == 6
      var xIn = firstInput
      var cleanCondLatents: DynamicGraph.Tensor<FloatType>? = nil
      var condFrames = 0
      if hasCleanCondLatents {
        let cleanCond = DynamicGraph.Tensor<FloatType>(restInputs[0])
        let cleanCondShape = cleanCond.shape
        condFrames = min(shape[0], cleanCondShape[0])
        xIn = firstInput.copied()
        // The conditioning prefix is substituted with clean latents at every step, matching
        // LongCat's diffusion-forcing style conditioning (cond latents carry t = 0).
        xIn[
          0..<condFrames, 0..<min(shape[1], cleanCondShape[1]),
          0..<min(shape[2], cleanCondShape[2]), 0..<shape[3]
        ] =
          cleanCond[
            0..<condFrames, 0..<min(shape[1], cleanCondShape[1]),
            0..<min(shape[2], cleanCondShape[2]), 0..<shape[3]
          ]
        cleanCondLatents = cleanCond
      }
      let modelInputs = Array(restInputs[(restInputs.count - longCatModelInputCount)...])
      let modelEt = unet(inputs: xIn, modelInputs)[0].as(of: FloatType.self)
      var et: DynamicGraph.Tensor<FloatType>
      if kvCache {
        et = firstInput.copied()
        et.full(0)
        let noiseFrames = min(shape[0] - condFrames, modelEt.shape[0])
        et[
          condFrames..<(condFrames + noiseFrames), 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ] = modelEt[0..<noiseFrames, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
      } else {
        et = modelEt
      }
      if let cleanCondLatents = cleanCondLatents, condFrames > 0, timestep.now > 0 {
        // Steer the sampler's cond prefix exactly onto clean latents: with rectified flow,
        // v = (x_t - x_0) / sigma is the straight-line velocity whose Euler updates land on x_0.
        let sigma = timestep.now / 1_000
        et[0..<condFrames, 0..<shape[1], 0..<shape[2], 0..<shape[3]] = Functional.add(
          left: firstInput[0..<condFrames, 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied(),
          right: cleanCondLatents[0..<condFrames, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
            .copied(),
          leftScalar: 1 / sigma, rightScalar: -1 / sigma)
      }
      return et
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      let shouldUseCache =
        teaCache?.shouldUseCacheForTimeEmbedding(
          Array(restInputs[1..<7]), model: unet, step: step, marker: index * 2, of: Float.self)
        ?? false
      guard isCfgEnabled else {
        let et: DynamicGraph.Tensor<FloatType>
        if shouldUseCache,
          let result = teaCache!(model: unet, inputs: firstInput, restInputs, marker: index * 2)
        {
          et = result[0].as(of: FloatType.self)
        } else {
          let result = unet(
            inputs: firstInput, restInputs
          )
          et = result[0].as(of: FloatType.self)
          teaCache?.cache(outputs: result, marker: index * 2)
        }
        return et
      }
      let vaceContextExists = (restInputs[7].shape.count == 1 && restInputs[7].shape[0] == 1)
      let vaceLayers: Int
      let injectImage: Bool
      if version == .wan21_1_3b {
        vaceLayers = vaceContextExists ? 15 : 0
        injectImage =
          restInputs.count > 9 + (isCfgEnabled ? 4 : 2) * 30
          + (vaceContextExists ? 15 * (isCfgEnabled ? 4 : 2) + 2 : 0)
      } else {
        vaceLayers = vaceContextExists ? 8 : 0
        injectImage =
          restInputs.count > 9 + (isCfgEnabled ? 4 : 2) * 40
          + (vaceContextExists ? 8 * (isCfgEnabled ? 4 : 2) + 2 : 0)
      }
      let shape = firstInput.shape
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        .copied()
      let restInputsUncond: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
        switch $0.0 {
        case 0..<(vaceContextExists ? 9 : 7), (restInputs.count - 2)..<restInputs.count:
          return $0.1
        default:
          if injectImage {
            if vaceContextExists, $0.0 < 9 + vaceLayers * 4 {
              if $0.0 % 2 == 0 {
                return nil
              }
            } else {
              if ($0.0 - (vaceContextExists ? 9 + vaceLayers * 4 : 7)) % 6 == 1
                || ($0.0 - (vaceContextExists ? 9 + vaceLayers * 4 : 7)) % 6 == 3
              {
                return nil  // Remove positive ones.
              }
            }
            return $0.1
          } else {
            if ($0.0 - (vaceContextExists ? 2 : 0)) % 2 == 1 {
              return $0.1
            }
            return nil
          }
        }
      }
      if shouldUseCache,
        let uncond = teaCache!(model: unet, inputs: xUncond, restInputsUncond, marker: index * 2)
      {
        etUncond = uncond[0].as(of: FloatType.self)
      } else {
        let result = unet(
          inputs: xUncond, restInputsUncond
        )
        etUncond = result[0].as(of: FloatType.self)
        teaCache?.cache(outputs: result, marker: index * 2)
      }
      etUncond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
      guard !isCancelled.load(ordering: .acquiring) else {
        return Functional.concat(axis: 0, etUncond, etUncond)
      }
      let xCond = firstInput[(shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        .copied()
      let restInputsCond: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
        switch $0.0 {
        case 0..<(vaceContextExists ? 9 : 7), (restInputs.count - 2)..<restInputs.count:
          return $0.1
        default:
          if injectImage {
            if vaceContextExists, $0.0 < 9 + vaceLayers * 4 {
              if $0.0 % 2 == 1 {
                return nil
              }
            } else {
              if ($0.0 - (vaceContextExists ? 9 + vaceLayers * 4 : 7)) % 6 == 0
                || ($0.0 - (vaceContextExists ? 9 + vaceLayers * 4 : 7)) % 6 == 2
              {
                return nil  // Remove negative ones.
              }
            }
            return $0.1
          } else {
            if ($0.0 - (vaceContextExists ? 2 : 0)) % 2 == 0 {
              return $0.1
            }
            return nil
          }
        }
      }
      if shouldUseCache,
        let cond = teaCache!(model: unet, inputs: xCond, restInputsCond, marker: index * 2 + 1)
      {
        etCond = cond[0].as(of: FloatType.self)
      } else {
        let result = unet(
          inputs: xCond, restInputsCond
        )
        etCond = result[0].as(of: FloatType.self)
        teaCache?.cache(outputs: result, marker: index * 2 + 1)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .ideogram4:
      if let unconditionalUNet = unconditionalUNet {
        precondition(restInputs.count > Ideogram4ConditionCount)
        let shape = firstInput.shape
        let batchSize = shape[0] / 2
        let xCond = firstInput[
          batchSize..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let text = DynamicGraph.Tensor<FloatType>(restInputs[0])
        let textShape = text.shape
        precondition(textShape.count == 3 && textShape[0] == batchSize)
        let textCond = text[
          0..<batchSize, 0..<tokenLengthCond, 0..<textShape[2]
        ].copied()
        let etCond = unet(
          inputs: xCond,
          [textCond] + Array(restInputs[1..<Ideogram4ConditionCount])
        )[0].as(of: FloatType.self)
        etCond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[
          0..<batchSize, 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let etUncond = unconditionalUNet(
          inputs: xUncond,
          Array(restInputs[Ideogram4ConditionCount..<restInputs.count])
        )[0].as(of: FloatType.self)
        return Functional.concat(axis: 0, etUncond, etCond)
      }
      let text = DynamicGraph.Tensor<FloatType>(restInputs[0])
      let textShape = text.shape
      precondition(textShape.count == 3)
      let imageIndicator = restInputs[1]
      let fixedSuffix = Array(restInputs[2...])
      let shape = firstInput.shape
      guard isCfgEnabled else {
        let textInput = text[
          0..<shape[0], 0..<tokenLengthCond, 0..<textShape[2]
        ].copied()
        let et = unet(
          inputs: firstInput, [textInput, imageIndicator] + fixedSuffix
        )[0].as(of: FloatType.self)
        return et
      }
      let batchSize = shape[0] / 2
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      if tokenLengthCond > tokenLengthUncond {
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let textCond = text[
          0..<batchSize, tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
          0..<textShape[2]
        ].copied()
        etCond = unet(
          inputs: xCond, [textCond, imageIndicator] + fixedSuffix
        )[0].as(of: FloatType.self)
        etCond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[
          0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let textUncond = text[
          0..<batchSize, 0..<tokenLengthUncond, 0..<textShape[2]
        ].copied()
        etUncond = unet(
          inputs: xUncond, [textUncond, imageIndicator] + fixedSuffix
        )[0].as(of: FloatType.self)
      } else {
        let xUncond = firstInput[
          0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let textUncond = text[
          0..<batchSize, 0..<tokenLengthUncond, 0..<textShape[2]
        ].copied()
        etUncond = unet(
          inputs: xUncond, [textUncond, imageIndicator] + fixedSuffix
        )[0].as(of: FloatType.self)
        etUncond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let textCond = text[
          0..<batchSize, tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
          0..<textShape[2]
        ].copied()
        etCond = unet(
          inputs: xCond, [textCond, imageIndicator] + fixedSuffix
        )[0].as(of: FloatType.self)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .krea2:
      let text = DynamicGraph.Tensor<FloatType>(restInputs[0])
      let textShape = text.shape
      let fixedSuffix = Array(restInputs[1...])
      let shape = firstInput.shape
      guard isCfgEnabled else {
        let textInput = text[
          0..<shape[0], 0..<tokenLengthCond, 0..<textShape[2]
        ].copied()
        return unet(inputs: firstInput, [textInput] + fixedSuffix)[0].as(of: FloatType.self)
      }
      let batchSize = shape[0] / 2
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      if tokenLengthCond > tokenLengthUncond {
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let textCond = text[
          0..<batchSize, tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
          0..<textShape[2]
        ].copied()
        etCond = unet(
          inputs: xCond, [textCond] + fixedSuffix
        )[0].as(of: FloatType.self)
        etCond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[
          0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let textUncond = text[
          0..<batchSize, 0..<tokenLengthUncond, 0..<textShape[2]
        ].copied()
        etUncond = unet(
          inputs: xUncond, [textUncond] + fixedSuffix
        )[0].as(of: FloatType.self)
      } else {
        let xUncond = firstInput[
          0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let textUncond = text[
          0..<batchSize, 0..<tokenLengthUncond, 0..<textShape[2]
        ].copied()
        etUncond = unet(
          inputs: xUncond, [textUncond] + fixedSuffix
        )[0].as(of: FloatType.self)
        etUncond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let textCond = text[
          0..<batchSize, tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
          0..<textShape[2]
        ].copied()
        etCond = unet(
          inputs: xCond, [textCond] + fixedSuffix
        )[0].as(of: FloatType.self)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .qwenImage2_1:
      guard isCfgEnabled else {
        return unet(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
      }
      let shape = firstInput.shape
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      if tokenLengthCond > tokenLengthUncond {
        // Evaluate the longest branch first to match the compiled graph.
        let xCond = firstInput[(shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return DynamicGraph.Tensor<Float>($0.1)[
              (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
            ].copied()
          case 6...:
            let length = shape[1] - max(tokenLengthUncond, tokenLengthCond) + tokenLengthCond
            return DynamicGraph.Tensor<FloatType>($0.1)[
              (shape[0] / 2)..<shape[0], 0..<length, 0..<shape[2], 0..<shape[3]
            ].copied()
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
        etCond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return DynamicGraph.Tensor<Float>($0.1)[
              0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
            ].copied()
          case 6...:
            let length = shape[1] - max(tokenLengthUncond, tokenLengthCond) + tokenLengthUncond
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<(shape[0] / 2), 0..<length, 0..<shape[2], 0..<shape[3]
            ].copied()
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
      } else {
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return DynamicGraph.Tensor<Float>($0.1)[
              0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
            ].copied()
          case 6...:
            let length = shape[1] - max(tokenLengthUncond, tokenLengthCond) + tokenLengthUncond
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<(shape[0] / 2), 0..<length, 0..<shape[2], 0..<shape[3]
            ].copied()
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
        etUncond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[(shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case 0:
            return DynamicGraph.Tensor<Float>($0.1)[
              (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
            ].copied()
          case 6...:
            let length = shape[1] - max(tokenLengthUncond, tokenLengthCond) + tokenLengthCond
            return DynamicGraph.Tensor<FloatType>($0.1)[
              (shape[0] / 2)..<shape[0], 0..<length, 0..<shape[2], 0..<shape[3]
            ].copied()
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .qwenImage:
      guard isCfgEnabled else {
        let et = unet(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
        return et
      }
      let shape = firstInput.shape
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      if tokenLengthCond > tokenLengthUncond {
        // This if-clause is useful because we compiled the graph with longest token, so later we don't need to trigger the automatic re-compilation.
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let count = restInputs.count
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case count - 719:  // Offset for reference image.
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
              0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
        etCond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case count - 719:  // Offset for reference image.
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
      } else {
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let count = restInputs.count
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case count - 719:  // Offset for reference image.
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
        etUncond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case count - 719:  // Offset for reference image.
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond),
              0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .flux1:
      if let teaCache = teaCache {
        let shape = firstInput.shape
        let batchSize = shape[0]
        guard batchSize > 1 else {
          let shouldUseCache = teaCache.shouldUseCacheForTimeEmbedding(
            [firstInput] + restInputs, model: unet, step: step, marker: index, of: FloatType.self)
          let et: DynamicGraph.Tensor<FloatType>
          if shouldUseCache,
            let result = teaCache(model: unet, inputs: firstInput, restInputs, marker: index)
          {
            et = result[0].as(of: FloatType.self)
          } else {
            let result = unet(
              inputs: firstInput, restInputs
            )
            et = result[0].as(of: FloatType.self)
            teaCache.cache(outputs: result, marker: index)
          }
          return et
        }
        let graph = firstInput.graph
        var et = graph.variable(like: firstInput)
        for i in 0..<batchSize {
          let x0 = firstInput[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
          let others = restInputs.map {
            var shape = $0.shape
            guard shape[0] > 1 else { return $0 }
            shape[0] = 1
            return DynamicGraph.Tensor<FloatType>($0).reshaped(
              format: $0.format, shape: shape, offset: [i]
            ).copied()
          }
          let shouldUseCache =
            teaCache.shouldUseCacheForTimeEmbedding(
              [x0] + others, model: unet, step: step, marker: index * batchSize + i,
              of: FloatType.self)
          let et0: DynamicGraph.Tensor<FloatType>
          if shouldUseCache,
            let result = teaCache(model: unet, inputs: x0, others, marker: index * batchSize + i)
          {
            et0 = result[0].as(of: FloatType.self)
          } else {
            let result = unet(
              inputs: x0, others
            )
            et0 = result[0].as(of: FloatType.self)
            teaCache.cache(outputs: result, marker: index * batchSize + i)
          }
          et[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] = et0
          guard !isCancelled.load(ordering: .acquiring) else {
            return et
          }
        }
        return et
      }
    case .zImage:
      guard isCfgEnabled else {
        var restInputs = restInputs
        restInputs.remove(at: restInputs.count - 131)
        let et = unet(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
        return et
      }
      let shape = firstInput.shape
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      let roundUpTokenLengthUncond = (tokenLengthUncond + 31) / 32 * 32
      let roundUpTokenLengthCond = (tokenLengthCond + 31) / 32 * 32
      let count = restInputs.count
      if roundUpTokenLengthCond > roundUpTokenLengthUncond {
        // This if-clause is useful because we compiled the graph with longest token, so later we don't need to trigger the automatic re-compilation.
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
          let shape = $0.1.shape
          switch $0.0 {
          case count - 130:  // Offset for text condition.
            return DynamicGraph.Tensor<Float>($0.1)[
              0..<shape[0],
              roundUpTokenLengthUncond..<(roundUpTokenLengthUncond + roundUpTokenLengthCond),
              0..<shape[2]
            ].copied()
          case count - 132:
            return nil
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
        etCond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
          let shape = $0.1.shape
          switch $0.0 {
          case count - 130:  // Offset for text condition.
            return DynamicGraph.Tensor<Float>($0.1)[
              0..<shape[0], 0..<roundUpTokenLengthUncond, 0..<shape[2]
            ].copied()
          case count - 131:
            return nil
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
      } else {
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
          let shape = $0.1.shape
          switch $0.0 {
          case count - 130:  // Offset for text condition.
            return DynamicGraph.Tensor<Float>($0.1)[
              0..<shape[0], 0..<roundUpTokenLengthUncond, 0..<shape[2]
            ].copied()
          case count - 131:
            return nil
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
        etUncond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
          let shape = $0.1.shape
          switch $0.0 {
          case count - 130:  // Offset for text condition.
            return DynamicGraph.Tensor<Float>($0.1)[
              0..<shape[0],
              roundUpTokenLengthUncond..<(roundUpTokenLengthUncond + roundUpTokenLengthCond),
              0..<shape[2]
            ].copied()
          case count - 132:
            return nil
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .hiDreamI1:
      var firstInput = firstInput
      var shape = firstInput.shape
      if modifier == .editing {
        // Move the conditioning to the right side.
        firstInput = firstInput.contiguous().reshaped(
          format: .NHWC, shape: [shape[0], shape[1], shape[2], 2, shape[3] / 2]
        ).transposed(2, 3).reshaped(.NHWC(shape[0], shape[1], 2 * shape[2], shape[3] / 2))
          .contiguous()
        shape = firstInput.shape
      }
      if let teaCache = teaCache {
        let shouldUseCache =
          teaCache.shouldUseCacheForTimeEmbedding(
            Array(restInputs[50..<51]), model: unet, step: step, marker: index, of: Float.self)
        let batchSize = shape[0]
        guard batchSize > 1 else {
          let et: DynamicGraph.Tensor<FloatType>
          if shouldUseCache,
            let result = teaCache(model: unet, inputs: firstInput, restInputs, marker: index)
          {
            et = result[0].as(of: FloatType.self)
          } else {
            let result = unet(
              inputs: firstInput, Array(restInputs[0..<50]) + Array(restInputs[51...]))
            et = result[0].as(of: FloatType.self)
            teaCache.cache(outputs: result, marker: index)
          }
          if modifier == .editing {
            // remove the conditioning.
            return et[0..<shape[0], 0..<shape[1], 0..<(shape[2] / 2), 0..<shape[3]].copied()
          } else {
            return et
          }
        }
        let graph = firstInput.graph
        var et = graph.variable(like: firstInput)
        for i in 0..<batchSize {
          let x0 = firstInput[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
          let others = restInputs.map {
            var shape = $0.shape
            guard shape[0] > 1 else { return $0 }
            shape[0] = 1
            return DynamicGraph.Tensor<FloatType>($0).reshaped(
              format: $0.format, shape: shape, offset: [i]
            ).copied()
          }
          let et0: DynamicGraph.Tensor<FloatType>
          if shouldUseCache,
            let result = teaCache(
              model: unet, inputs: firstInput, restInputs, marker: index * batchSize + i)
          {
            et0 = result[0].as(of: FloatType.self)
          } else {
            let result = unet(inputs: x0, Array(others[0..<50]) + Array(others[51...]))
            et0 = result[0].as(of: FloatType.self)
            teaCache.cache(outputs: result, marker: index * batchSize + i)
          }
          et[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] = et0
          guard !isCancelled.load(ordering: .acquiring) else {
            if modifier == .editing {
              // remove the conditioning.
              return et[0..<shape[0], 0..<shape[1], 0..<(shape[2] / 2), 0..<shape[3]].copied()
            } else {
              return et
            }
          }
        }
        if modifier == .editing {
          // remove the conditioning.
          return et[0..<shape[0], 0..<shape[1], 0..<(shape[2] / 2), 0..<shape[3]].copied()
        } else {
          return et
        }
      } else {
        let batchSize = shape[0]
        guard batchSize > 1 else {
          let et = unet(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
          if modifier == .editing {
            // remove the conditioning.
            return et[0..<shape[0], 0..<shape[1], 0..<(shape[2] / 2), 0..<shape[3]].copied()
          } else {
            return et
          }
        }
        let graph = firstInput.graph
        var et = graph.variable(like: firstInput)
        for i in 0..<batchSize {
          let x0 = firstInput[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]].copied()
          let others = restInputs.map {
            var shape = $0.shape
            guard shape[0] > 1 else { return $0 }
            shape[0] = 1
            return DynamicGraph.Tensor<FloatType>($0).reshaped(
              format: $0.format, shape: shape, offset: [i]
            ).copied()
          }
          et[i..<(i + 1), 0..<shape[1], 0..<shape[2], 0..<shape[3]] = unet(inputs: x0, others)[0]
            .as(
              of: FloatType.self)
          guard !isCancelled.load(ordering: .acquiring) else {
            if modifier == .editing {
              // remove the conditioning.
              return et[0..<shape[0], 0..<shape[1], 0..<(shape[2] / 2), 0..<shape[3]].copied()
            } else {
              return et
            }
          }
        }
        if modifier == .editing {
          // remove the conditioning.
          return et[0..<shape[0], 0..<shape[1], 0..<(shape[2] / 2), 0..<shape[3]].copied()
        } else {
          return et
        }
      }
    case .flux2, .flux2_9b, .flux2_4b:
      guard isCfgEnabled else {
        let et = unet(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
        return et
      }
      let shape = firstInput.shape
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      let cachedKVCount: Int
      if modifier == .kontextKv && referenceImageCount > 0 {
        if version == .flux2_9b {
          cachedKVCount = (8 + 24) * 2
        } else if version == .flux2_4b {
          cachedKVCount = (5 + 20) * 2
        } else {
          cachedKVCount = (8 + 48) * 2
        }
      } else {
        cachedKVCount = 0
      }
      let dynamicEndIndex = restInputs.count - cachedKVCount
      if tokenLengthCond > tokenLengthUncond {
        // This if-clause is useful because we compiled the graph with longest token, so later we don't need to trigger the automatic re-compilation.
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case dynamicEndIndex - 18:  // Offset for text condition.
            return DynamicGraph.Tensor<Float>($0.1)[
              (shape[0] / 2)..<shape[0], 0..<tokenLengthCond, 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
        etCond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case dynamicEndIndex - 18:  // Offset for text condition.
            return DynamicGraph.Tensor<Float>($0.1)[
              0..<(shape[0] / 2), 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
      } else {
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case dynamicEndIndex - 18:  // Offset for text condition.
            return DynamicGraph.Tensor<Float>($0.1)[
              0..<(shape[0] / 2), 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
        etUncond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ]
        .copied()
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().map {
          let shape = $0.1.shape
          switch $0.0 {
          case dynamicEndIndex - 18:  // Offset for text condition.
            return DynamicGraph.Tensor<Float>($0.1)[
              (shape[0] / 2)..<shape[0], 0..<tokenLengthCond, 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .ernieImage:
      guard isCfgEnabled else {
        return unet(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
      }
      let shape = firstInput.shape
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      if tokenLengthCond > tokenLengthUncond {
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
          switch $0.0 {
          case 0, 1:
            return nil
          case 4:
            let shape = $0.1.shape
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond), 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
        etCond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[
          0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
          switch $0.0 {
          case 2, 3:
            return nil
          case 4:
            let shape = $0.1.shape
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
      } else {
        let xUncond = firstInput[
          0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let otherUnconds: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
          switch $0.0 {
          case 2, 3:
            return nil
          case 4:
            let shape = $0.1.shape
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], 0..<tokenLengthUncond, 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etUncond = unet(inputs: xUncond, otherUnconds)[0].as(of: FloatType.self)
        etUncond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        let otherConds: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
          switch $0.0 {
          case 0, 1:
            return nil
          case 4:
            let shape = $0.1.shape
            return DynamicGraph.Tensor<FloatType>($0.1)[
              0..<shape[0], tokenLengthUncond..<(tokenLengthUncond + tokenLengthCond), 0..<shape[2]
            ].copied()
          default:
            return $0.1
          }
        }
        etCond = unet(inputs: xCond, otherConds)[0].as(of: FloatType.self)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .cosmos2_5_2b:
      guard isCfgEnabled else {
        let et = unet(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
        return et
      }
      let shape = firstInput.shape
      let kvStartIndex = 1 + CosmosFixedTimeConditionCount
      let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
        .copied()
      let uncondInputs: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
        if $0.0 < kvStartIndex {
          return $0.1
        } else {
          return (($0.0 - kvStartIndex) % 2 == 0) ? $0.1 : nil
        }
      }
      let etUncond = unet(inputs: xUncond, uncondInputs)[0].as(of: FloatType.self)
      etUncond.graph.joined()
      guard !isCancelled.load(ordering: .acquiring) else {
        return Functional.concat(axis: 0, etUncond, etUncond)
      }
      let xCond = firstInput[
        (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
      ].copied()
      let condInputs: [DynamicGraph.AnyTensor] = restInputs.enumerated().compactMap {
        if $0.0 < kvStartIndex {
          return $0.1
        } else {
          return (($0.0 - kvStartIndex) % 2 == 1) ? $0.1 : nil
        }
      }
      let etCond = unet(inputs: xCond, condInputs)[0].as(of: FloatType.self)
      return Functional.concat(axis: 0, etUncond, etCond)
    case .ltx2:
      let graph = firstInput.graph
      var shape = firstInput.shape
      // Separate firstInput into video input and audio input.
      if isCfgEnabled {
        shape[0] = shape[0] / 2
      }
      let batchSize = shape[0]
      let startWidth = shape[2]
      let (audioFrames, audioHeight) = LTX2ExtractAudioFramesAndHeight(shape)
      let startHeight = shape[1] - audioHeight
      guard isCfgEnabled else {
        var videoInput = firstInput[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<shape[3]]
          .copied()
        let audioInput = firstInput[
          0..<batchSize, startHeight..<shape[1], 0..<startWidth, 0..<shape[3]
        ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
        if restInputs.count > 1255 {
          let firstFrame = DynamicGraph.Tensor<FloatType>(restInputs[0])
          let firstFrameShape = firstFrame.shape
          videoInput[
            0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
            0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]] =
            firstFrame[
              0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
              0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]]
        }
        let output = unet(
          inputs: videoInput, [audioInput] + restInputs[(restInputs.count - 1255)...]
        )
        .map {
          $0.as(of: FloatType.self)
        }
        let videoOutput = output[0].reshaped(.NHWC(batchSize, startHeight, startWidth, shape[3]))
        var audioOutput = graph.variable(
          .GPU(0), .HWC(1, batchSize * startWidth * audioHeight, shape[3]), of: FloatType.self)
        audioOutput.full(0)
        audioOutput[0..<1, 0..<audioFrames, 0..<shape[3]] = output[1]
        return Functional.concat(
          axis: 1, videoOutput,
          audioOutput.reshaped(.NHWC(batchSize, audioHeight, startWidth, shape[3])))
      }
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      let xUncond = firstInput[0..<batchSize, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
      var videoInputUncond = xUncond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<shape[3]]
        .copied()
      let audioInputUncond = xUncond[
        0..<batchSize, startHeight..<shape[1], 0..<startWidth, 0..<shape[3]
      ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
      if restInputs.count > 1255 {
        let firstFrame = DynamicGraph.Tensor<FloatType>(restInputs[0])
        let firstFrameShape = firstFrame.shape
        videoInputUncond[
          0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
          0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]] =
          firstFrame[
            0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
            0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]]
      }
      let restInputsUncond = Array(restInputs[(restInputs.count - 1255)...]).enumerated().map {
        guard $0.0 >= 3, ($0.0 - 3) % 26 < 4, $0.0 - 3 < 48 * 26 else { return $0.1 }
        let shape = $0.1.shape
        return DynamicGraph.Tensor<FloatType>($0.1)[
          0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
      }
      let outputUncond = unet(inputs: videoInputUncond, [audioInputUncond] + restInputsUncond)
        .map {
          $0.as(of: FloatType.self)
        }
      let videoOutputUncond = outputUncond[0].reshaped(
        .NHWC(batchSize, startHeight, startWidth, shape[3]))
      var audioOutputUncond = graph.variable(
        .GPU(0), .HWC(1, batchSize * startWidth * audioHeight, shape[3]), of: FloatType.self)
      audioOutputUncond.full(0)
      audioOutputUncond[0..<1, 0..<audioFrames, 0..<shape[3]] = outputUncond[1]
      etUncond = Functional.concat(
        axis: 1, videoOutputUncond,
        audioOutputUncond.reshaped(.NHWC(batchSize, audioHeight, startWidth, shape[3])))
      etUncond.graph.joined()  // Wait for the result to be fully populated. Seems otherwise I can have Metal error for very large executions.
      guard !isCancelled.load(ordering: .acquiring) else {
        return Functional.concat(axis: 0, etUncond, etUncond)
      }
      let xCond = firstInput[batchSize..<(batchSize * 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
      var videoInputCond = xCond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<shape[3]]
        .copied()
      let audioInputCond = xCond[
        0..<batchSize, startHeight..<shape[1], 0..<startWidth, 0..<shape[3]
      ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
      if restInputs.count > 1255 {
        let firstFrame = DynamicGraph.Tensor<FloatType>(restInputs[0])
        let firstFrameShape = firstFrame.shape
        videoInputCond[
          0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
          0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]] =
          firstFrame[
            0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
            0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]]
      }
      let restInputsCond = Array(restInputs[(restInputs.count - 1255)...]).enumerated().map {
        guard $0.0 >= 3, ($0.0 - 3) % 26 < 4, $0.0 - 3 < 48 * 26 else { return $0.1 }
        let shape = $0.1.shape
        return DynamicGraph.Tensor<FloatType>($0.1)[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
      }
      let outputCond = unet(inputs: videoInputCond, [audioInputCond] + restInputsCond)
        .map {
          $0.as(of: FloatType.self)
        }
      let videoOutputCond = outputCond[0].reshaped(
        .NHWC(batchSize, startHeight, startWidth, shape[3]))
      var audioOutputCond = graph.variable(
        .GPU(0), .HWC(1, batchSize * startWidth * audioHeight, shape[3]), of: FloatType.self)
      audioOutputCond.full(0)
      audioOutputCond[0..<1, 0..<audioFrames, 0..<shape[3]] = outputCond[1]
      etCond = Functional.concat(
        axis: 1, videoOutputCond,
        audioOutputCond.reshaped(.NHWC(batchSize, audioHeight, startWidth, shape[3])))
      return Functional.concat(axis: 0, etUncond, etCond)
    case .ltx2_3:
      let graph = firstInput.graph
      var shape = firstInput.shape
      if isCfgEnabled {
        shape[0] = shape[0] / 2
      }
      let batchSize = shape[0]
      let startWidth = shape[2]
      let (audioFrames, audioHeight) = LTX2ExtractAudioFramesAndHeight(shape)
      let startHeight = shape[1] - audioHeight
      guard isCfgEnabled else {
        var videoInput = firstInput[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<shape[3]]
          .copied()
        let audioInput = firstInput[
          0..<batchSize, startHeight..<shape[1], 0..<startWidth, 0..<shape[3]
        ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
        if restInputs.count > 1545 {
          let firstFrame = DynamicGraph.Tensor<FloatType>(restInputs[0])
          let firstFrameShape = firstFrame.shape
          videoInput[
            0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
            0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]] =
            firstFrame[
              0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
              0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]]
        }
        let output = unet(
          inputs: videoInput, [audioInput] + restInputs[(restInputs.count - 1545)...]
        )
        .map {
          $0.as(of: FloatType.self)
        }
        let videoOutput = output[0].reshaped(.NHWC(batchSize, startHeight, startWidth, shape[3]))
        var audioOutput = graph.variable(
          .GPU(0), .HWC(1, batchSize * startWidth * audioHeight, shape[3]), of: FloatType.self)
        audioOutput.full(0)
        audioOutput[0..<1, 0..<audioFrames, 0..<shape[3]] = output[1]
        return Functional.concat(
          axis: 1, videoOutput,
          audioOutput.reshaped(.NHWC(batchSize, audioHeight, startWidth, shape[3])))
      }
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      let xUncond = firstInput[0..<batchSize, 0..<shape[1], 0..<shape[2], 0..<shape[3]]
      var videoInputUncond = xUncond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<shape[3]]
        .copied()
      let audioInputUncond = xUncond[
        0..<batchSize, startHeight..<shape[1], 0..<startWidth, 0..<shape[3]
      ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
      if restInputs.count > 1545 {
        let firstFrame = DynamicGraph.Tensor<FloatType>(restInputs[0])
        let firstFrameShape = firstFrame.shape
        videoInputUncond[
          0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
          0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]] =
          firstFrame[
            0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
            0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]]
      }
      let restInputsUncond = Array(restInputs[(restInputs.count - 1545)...]).enumerated().map {
        guard $0.0 >= 3 + 48 * 32 + 4 else { return $0.1 }
        let shape = $0.1.shape
        if shape.count == 3 {
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2]
          ].copied()
        } else {
          return DynamicGraph.Tensor<FloatType>($0.1)[
            0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied()
        }
      }
      let outputUncond = unet(inputs: videoInputUncond, [audioInputUncond] + restInputsUncond)
        .map {
          $0.as(of: FloatType.self)
        }
      let videoOutputUncond = outputUncond[0].reshaped(
        .NHWC(batchSize, startHeight, startWidth, shape[3]))
      var audioOutputUncond = graph.variable(
        .GPU(0), .HWC(1, batchSize * startWidth * audioHeight, shape[3]), of: FloatType.self)
      audioOutputUncond.full(0)
      audioOutputUncond[0..<1, 0..<audioFrames, 0..<shape[3]] = outputUncond[1]
      etUncond = Functional.concat(
        axis: 1, videoOutputUncond,
        audioOutputUncond.reshaped(.NHWC(batchSize, audioHeight, startWidth, shape[3])))
      etUncond.graph.joined()
      guard !isCancelled.load(ordering: .acquiring) else {
        return Functional.concat(axis: 0, etUncond, etUncond)
      }
      let xCond = firstInput[batchSize..<(batchSize * 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
      var videoInputCond = xCond[0..<batchSize, 0..<startHeight, 0..<startWidth, 0..<shape[3]]
        .copied()
      let audioInputCond = xCond[
        0..<batchSize, startHeight..<shape[1], 0..<startWidth, 0..<shape[3]
      ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
      if restInputs.count > 1545 {
        let firstFrame = DynamicGraph.Tensor<FloatType>(restInputs[0])
        let firstFrameShape = firstFrame.shape
        videoInputCond[
          0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
          0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]] =
          firstFrame[
            0..<min(batchSize, firstFrameShape[0]), 0..<min(startHeight, firstFrameShape[1]),
            0..<min(startWidth, firstFrameShape[2]), 0..<shape[3]]
      }
      let restInputsCond = Array(restInputs[(restInputs.count - 1545)...]).enumerated().map {
        guard $0.0 >= 3 + 48 * 32 + 4 else { return $0.1 }
        let shape = $0.1.shape
        if shape.count == 3 {
          return DynamicGraph.Tensor<FloatType>($0.1)[
            (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2]
          ].copied()
        } else {
          return DynamicGraph.Tensor<FloatType>($0.1)[
            (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
          ].copied()
        }
      }
      let outputCond = unet(inputs: videoInputCond, [audioInputCond] + restInputsCond)
        .map {
          $0.as(of: FloatType.self)
        }
      let videoOutputCond = outputCond[0].reshaped(
        .NHWC(batchSize, startHeight, startWidth, shape[3]))
      var audioOutputCond = graph.variable(
        .GPU(0), .HWC(1, batchSize * startWidth * audioHeight, shape[3]), of: FloatType.self)
      audioOutputCond.full(0)
      audioOutputCond[0..<1, 0..<audioFrames, 0..<shape[3]] = outputCond[1]
      etCond = Functional.concat(
        axis: 1, videoOutputCond,
        audioOutputCond.reshaped(.NHWC(batchSize, audioHeight, startWidth, shape[3])))
      return Functional.concat(axis: 0, etUncond, etCond)
    case .seedvr2_3b, .seedvr2_7b:
      let timestep = DynamicGraph.Tensor<FloatType>(restInputs[0])
      let text = DynamicGraph.Tensor<FloatType>(restInputs[1])
      guard isCfgEnabled else {
        let textIndex = min(text.shape[0] - 1, 1)
        return unet(
          inputs: firstInput,
          [
            timestep,
            text[
              textIndex..<(textIndex + 1), 0..<tokenLengthCond, 0..<text.shape[2]
            ].copied().reshaped(.WC(tokenLengthCond, text.shape[2])),
          ] + Array(restInputs[2..<17])
        )[0].as(of: FloatType.self)
      }
      let shape = firstInput.shape
      let batchSize = isCfgEnabled ? shape[0] / 2 : shape[0]
      precondition(batchSize == 1)
      let etUncond: DynamicGraph.Tensor<FloatType>
      let etCond: DynamicGraph.Tensor<FloatType>
      if tokenLengthCond > tokenLengthUncond {
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        etCond = unet(
          inputs: xCond,
          [
            timestep,
            text[
              1..<2, 0..<tokenLengthCond, 0..<text.shape[2]
            ].copied().reshaped(.WC(tokenLengthCond, text.shape[2])),
          ] + Array(restInputs[17..<32])
        )[0].as(of: FloatType.self)
        etCond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etCond, etCond)
        }
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        etUncond = unet(
          inputs: xUncond,
          [
            timestep,
            text[
              0..<1, 0..<tokenLengthUncond, 0..<text.shape[2]
            ].copied().reshaped(.WC(tokenLengthUncond, text.shape[2])),
          ] + Array(restInputs[2..<17])
        )[0].as(of: FloatType.self)
      } else {
        let xUncond = firstInput[0..<(shape[0] / 2), 0..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied()
        etUncond = unet(
          inputs: xUncond,
          [
            timestep,
            text[
              0..<1, 0..<tokenLengthUncond, 0..<text.shape[2]
            ].copied().reshaped(.WC(tokenLengthUncond, text.shape[2])),
          ] + Array(restInputs[2..<17])
        )[0].as(of: FloatType.self)
        etUncond.graph.joined()
        guard !isCancelled.load(ordering: .acquiring) else {
          return Functional.concat(axis: 0, etUncond, etUncond)
        }
        let xCond = firstInput[
          (shape[0] / 2)..<shape[0], 0..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied()
        etCond = unet(
          inputs: xCond,
          [
            timestep,
            text[
              1..<2, 0..<tokenLengthCond, 0..<text.shape[2]
            ].copied().reshaped(.WC(tokenLengthCond, text.shape[2])),
          ] + Array(restInputs[17..<32])
        )[0].as(of: FloatType.self)
      }
      return Functional.concat(axis: 0, etUncond, etCond)
    case .hiDreamO1:
      let predicted = unet(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
      let reciprocalSigma: Float = 1 / max(timestep.now / 1000, 1e-6)
      return (firstInput - predicted) * reciprocalSigma
    case .auraflow, .kandinsky21, .pixart, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner,
      .ssd1b, .svdI2v, .v1, .v2, .wurstchenStageB, .wurstchenStageC:
      break
    }
    return unet(inputs: firstInput, restInputs)[0].as(of: FloatType.self)
  }

  private static func audioHeight(
    _ shape: TensorShape, version: ModelVersion, isCfgEnabled: Bool
  ) -> (Int, Int) {
    var shape = shape
    if isCfgEnabled {
      shape[0] /= 2
    }
    switch version {
    case .minimaxH3:
      let height = MiniMaxH3AudioHeight(videoLatentFrames: shape[0], latentWidth: shape[2])
      let frames = shape[0] == 1 ? 1 : (shape[0] - 2) / 5 * 17 + 5
      let rows =
        2
        * Int(
          (Double(frames) / Double(MiniMaxH3Configuration.framesPerSecond) * 40).rounded())
      return ((rows * 32 + 23) / 24, height)
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .svdI2v, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .flux1, .sd3Large, .hunyuanVideo, .wan21_1_3b,
      .wan21_14b, .hiDreamI1, .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage,
      .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .seedvr2_3b, .seedvr2_7b,
      .ideogram4, .krea2, .longcatVideoAvatar1_5:
      return (0, 0)
    case .ltx2, .ltx2_3:
      return LTX2ExtractAudioFramesAndHeight(shape)
    }
  }

  private func internalDiffuse(
    xyTiles: Int, index: Int, inputStartYPad: Int, inputEndYPad: Int, inputStartXPad: Int,
    inputEndXPad: Int, xT: DynamicGraph.Tensor<FloatType>, inputs: [DynamicGraph.AnyTensor],
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ restInputs: [DynamicGraph.AnyTensor],
      _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
    ), referenceImageCount: Int, referenceAudioCount: Int, step: Int,
    timestep: (now: Float, next: Float),
    audioShiftRatio: Float,
    tokenLengthUncond: Int,
    tokenLengthCond: Int,
    isCfgEnabled: Bool, audioFrames: Int, audioHeight: Int,
    controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    let shape = xT.shape
    let graph = xT.graph
    var x = xT[
      0..<shape[0], inputStartYPad..<inputEndYPad, inputStartXPad..<inputEndXPad, 0..<shape[3]
    ].copied()
    if audioHeight > 0 {
      let w = inputEndXPad - inputStartXPad
      let h = Self.audioHeight(
        [shape[0], 1, w, shape[3]], version: version, isCfgEnabled: isCfgEnabled
      ).1
      var slicedAudio = graph.variable(
        .GPU(0), .HWC(1, shape[0] * h * w, shape[3]), of: FloatType.self)
      slicedAudio.full(0)
      if isCfgEnabled {
        let audioInputUncond = xT[
          0..<(shape[0] / 2), (shape[1] - audioHeight)..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
        let audioInputCond = xT[
          (shape[0] / 2)..<shape[0], (shape[1] - audioHeight)..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
        slicedAudio[0..<1, 0..<audioFrames, 0..<shape[3]] = audioInputUncond
        slicedAudio[
          0..<1, ((shape[0] / 2) * h * w)..<((shape[0] / 2) * h * w + audioFrames), 0..<shape[3]] =
          audioInputCond
      } else {
        let audioInput = xT[
          0..<shape[0], (shape[1] - audioHeight)..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
        slicedAudio[0..<1, 0..<audioFrames, 0..<shape[3]] = audioInput
      }
      x = Functional.concat(
        axis: 1, x, slicedAudio.reshaped(.NHWC(shape[0], h, w, shape[3])))
    }
    // Need to rework the shape. For Wurstchen B, we need to slice them up.
    // For ControlNet, we already sliced them up into batch dimension, now need to extract them out.
    let (injectedControls, injectedT2IAdapters, injectedAttentionKVs) = injectedControlsAndAdapters(
      x, inputs, inputStartYPad, inputEndYPad, inputStartXPad, inputEndXPad, &controlNets)
    let inputs = sliceInputs(
      inputs + injectedControls + injectedT2IAdapters, originalShape: shape, xyTiles: xyTiles,
      index: index, inputStartYPad: inputStartYPad,
      inputEndYPad: inputEndYPad, inputStartXPad: inputStartXPad, inputEndXPad: inputEndXPad,
      modifier: modifier, referenceImageCount: referenceImageCount,
      referenceAudioCount: referenceAudioCount,
      tokenLength: isCfgEnabled ? max(tokenLengthUncond, tokenLengthCond) : tokenLengthCond)
    return self(
      referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
      step: step, timestep: timestep, audioShiftRatio: audioShiftRatio, index: index,
      tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
      isCfgEnabled: isCfgEnabled, inputs: x, inputs + injectedAttentionKVs)
  }

  private func tiledDiffuse(
    tiledDiffusion: TiledConfiguration, xT: DynamicGraph.Tensor<FloatType>,
    inputs: [DynamicGraph.AnyTensor],
    injectedControlsAndAdapters: (
      _ xT: DynamicGraph.Tensor<FloatType>, _ restInputs: [DynamicGraph.AnyTensor],
      _ inputStartYPad: Int, _ inputEndYPad: Int,
      _ inputStartXPad: Int, _ inputEndXPad: Int, _ existingControlNets: inout [Model?]
    ) -> (
      injectedControls: [DynamicGraph.Tensor<FloatType>],
      injectedT2IAdapters: [DynamicGraph.Tensor<FloatType>],
      injectedAttentionKVs: [DynamicGraph.Tensor<FloatType>]
    ), referenceImageCount: Int, referenceAudioCount: Int, step: Int,
    timestep: (now: Float, next: Float),
    audioShiftRatio: Float,
    tokenLengthUncond: Int,
    tokenLengthCond: Int,
    isCfgEnabled: Bool,
    controlNets: inout [Model?]
  ) -> DynamicGraph.Tensor<FloatType> {
    guard let xTileWeightsAndIndexes = xTileWeightsAndIndexes,
      let yTileWeightsAndIndexes = yTileWeightsAndIndexes
    else {
      let (injectedControls, injectedT2IAdapters, injectedAttentionKVs) =
        injectedControlsAndAdapters(
          xT, inputs, 0, 0, 0, 0, &controlNets)
      return self(
        referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
        step: step, timestep: timestep, audioShiftRatio: audioShiftRatio, index: 0,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: isCfgEnabled,
        inputs: xT, inputs + injectedControls + injectedT2IAdapters + injectedAttentionKVs)
    }
    let shape = xT.shape
    let (audioFrames, audioHeight) = Self.audioHeight(
      shape, version: version, isCfgEnabled: isCfgEnabled)
    let startHeight = shape[1] - audioHeight
    let startWidth = shape[2]
    let tileScaleFactor: Int
    switch version {
    case .wurstchenStageB:
      tileScaleFactor = 16
    case .wan22_5b, .minimaxH3, .qwenImage2_1:
      tileScaleFactor = 4
    case .ltx2, .ltx2_3:
      tileScaleFactor = 2
    case .wurstchenStageC:
      tileScaleFactor = 1
    case .auraflow, .pixart, .flux1, .ernieImage, .flux2, .flux2_4b, .flux2_9b, .hunyuanVideo,
      .hiDreamI1,
      .kandinsky21, .qwenImage, .sd3, .sd3Large, .sdxlBase, .sdxlRefiner, .ssd1b,
      .svdI2v, .v1, .v2,
      .wan21_14b, .wan21_1_3b, .zImage, .cosmos2_5_2b, .seedvr2_3b, .seedvr2_7b, .ideogram4,
      .krea2, .longcatVideoAvatar1_5:
      tileScaleFactor = 8
    case .hiDreamO1:
      tileScaleFactor = 32
    }
    let tiledWidth =
      tiledDiffusion.isEnabled
      ? min(tiledDiffusion.tileSize.width * tileScaleFactor, startWidth) : startWidth
    let tiledHeight =
      tiledDiffusion.isEnabled
      ? min(tiledDiffusion.tileSize.height * tileScaleFactor, startHeight) : startHeight
    let tileOverlap = min(
      min(
        tiledDiffusion.tileOverlap * tileScaleFactor / 2,
        Int((Double(tiledHeight / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor),
      Int((Double(tiledWidth / 3) / Double(tileScaleFactor)).rounded(.down)) * tileScaleFactor)
    let yTiles =
      (startHeight - tileOverlap * 2 + (tiledHeight - tileOverlap * 2) - 1)
      / (tiledHeight - tileOverlap * 2)
    let xTiles =
      (startWidth - tileOverlap * 2 + (tiledWidth - tileOverlap * 2) - 1)
      / (tiledWidth - tileOverlap * 2)
    var et = [DynamicGraph.Tensor<FloatType>]()
    let graph = xT.graph
    guard !isCancelled.load(ordering: .acquiring) else {
      return graph.variable(
        Tensor<FloatType>(.GPU(0), .NHWC(shape[0], startHeight, startWidth, shape[3])))
    }
    for y in 0..<yTiles {
      let yOfs = y * (tiledHeight - tileOverlap * 2) + (y > 0 ? tileOverlap : 0)
      let (inputStartYPad, inputEndYPad) = paddedTileStartAndEnd(
        iOfs: yOfs, length: startHeight, tileSize: tiledHeight, tileOverlap: tileOverlap)
      for x in 0..<xTiles {
        let xOfs = x * (tiledWidth - tileOverlap * 2) + (x > 0 ? tileOverlap : 0)
        let (inputStartXPad, inputEndXPad) = paddedTileStartAndEnd(
          iOfs: xOfs, length: startWidth, tileSize: tiledWidth, tileOverlap: tileOverlap)
        et.append(
          internalDiffuse(
            xyTiles: xTiles * yTiles, index: y * xTiles + x, inputStartYPad: inputStartYPad,
            inputEndYPad: inputEndYPad, inputStartXPad: inputStartXPad, inputEndXPad: inputEndXPad,
            xT: xT, inputs: inputs, injectedControlsAndAdapters: injectedControlsAndAdapters,
            referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
            step: step, timestep: timestep, audioShiftRatio: audioShiftRatio,
            tokenLengthUncond: tokenLengthUncond,
            tokenLengthCond: tokenLengthCond,
            isCfgEnabled: isCfgEnabled, audioFrames: audioFrames, audioHeight: audioHeight,
            controlNets: &controlNets))
        guard !isCancelled.load(ordering: .acquiring) else {
          return graph.variable(
            Tensor<FloatType>(
              .GPU(0), .NHWC(shape[0], startHeight + audioHeight, startWidth, shape[3])))
        }
      }
    }
    graph.joined()
    let etRawValues: [Tensor<FloatType>]
    if audioHeight > 0 {
      etRawValues = et.map {
        let shape = $0.shape
        let audioHeight = Self.audioHeight(shape, version: version, isCfgEnabled: isCfgEnabled).1
        return $0[0..<shape[0], 0..<(shape[1] - audioHeight), 0..<shape[2], 0..<shape[3]].copied()
          .rawValue.toCPU()
      }
    } else {
      etRawValues = et.map { $0.rawValue.toCPU() }
    }
    let channels = etRawValues[0].shape[3]
    var etRaw = Tensor<FloatType>(.CPU, .NHWC(shape[0], startHeight, startWidth, channels))
    etRaw.withUnsafeMutableBytes {
      guard var fp = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
      for b in 0..<shape[0] {
        for j in 0..<startHeight {
          let yWeightAndIndex = yTileWeightsAndIndexes[j]
          for i in 0..<startWidth {
            let xWeightAndIndex = xTileWeightsAndIndexes[i]
            for k in 0..<channels {
              fp[k] = 0
            }
            for y in yWeightAndIndex {
              for x in xWeightAndIndex {
                let weight = FloatType(x.weight * y.weight)
                let index = y.index * xTiles + x.index
                let tensor = etRawValues[index]
                tensor.withUnsafeBytes {
                  guard var v = $0.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
                    return
                  }
                  // Note that while result is outputChannels, this is padded to 4 i.e. channels.
                  v =
                    v + b * tiledHeight * tiledWidth * channels + x.offset * channels + y.offset
                    * tiledWidth * channels
                  for k in 0..<channels {
                    fp[k] += v[k] * weight
                  }
                }
              }
            }
            fp += channels
          }
        }
      }
    }
    guard audioHeight > 0 else { return graph.variable(etRaw.toGPU(0)) }
    var finalEt = graph.variable(
      .GPU(0), .NHWC(shape[0], startHeight + audioHeight, startWidth, channels), of: FloatType.self)
    finalEt[0..<shape[0], 0..<startHeight, 0..<startWidth, 0..<channels] = graph.variable(
      etRaw.toGPU(0))
    // Extract and homonizing all audios to the same shape.
    if isCfgEnabled {
      let audioOutputs = et.map {
        let shape = $0.shape
        let audioHeight = Self.audioHeight(shape, version: version, isCfgEnabled: isCfgEnabled).1
        let audioOutputUncond = $0[
          0..<(shape[0] / 2), (shape[1] - audioHeight)..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
        let audioOutputCond = $0[
          (shape[0] / 2)..<shape[0], (shape[1] - audioHeight)..<shape[1], 0..<shape[2], 0..<shape[3]
        ].copied().reshaped(.HWC(1, audioFrames, shape[3]))
        return (audioOutputUncond, audioOutputCond)
      }
      var audioOutput = graph.variable(
        .GPU(0), .HWC(1, shape[0] * startWidth * audioHeight, shape[3]), of: FloatType.self)
      audioOutput.full(0)
      if audioOutputs.count > 1 {
        audioOutput[0..<1, 0..<audioFrames, 0..<shape[3]] =
          (1 / Float(audioOutputs.count))
          * (audioOutputs[1..<audioOutputs.count].reduce(audioOutputs[0].0) { $0 + $1.0 })
        audioOutput[
          0..<1,
          ((shape[0] / 2) * startWidth * audioHeight)..<((shape[0] / 2) * startWidth * audioHeight
            + audioFrames), 0..<shape[3]] =
          (1 / Float(audioOutputs.count))
          * (audioOutputs[1..<audioOutputs.count].reduce(audioOutputs[0].1) { $0 + $1.1 })
      } else {
        audioOutput[0..<1, 0..<audioFrames, 0..<shape[3]] = audioOutputs[0].0
        audioOutput[
          0..<1,
          ((shape[0] / 2) * startWidth * audioHeight)..<((shape[0] / 2) * startWidth * audioHeight
            + audioFrames), 0..<shape[3]] = audioOutputs[0].1
      }
      finalEt[
        0..<shape[0], startHeight..<(startHeight + audioHeight), 0..<startWidth, 0..<channels] =
        audioOutput.reshaped(.NHWC(shape[0], audioHeight, startWidth, channels))
    } else {
      let audioOutputs = et.map {
        let shape = $0.shape
        let audioHeight = Self.audioHeight(shape, version: version, isCfgEnabled: isCfgEnabled).1
        return $0[0..<shape[0], (shape[1] - audioHeight)..<shape[1], 0..<shape[2], 0..<shape[3]]
          .copied().reshaped(.HWC(1, audioFrames, shape[3]))
      }
      var audioOutput = graph.variable(
        .GPU(0), .HWC(1, shape[0] * startWidth * audioHeight, shape[3]), of: FloatType.self)
      audioOutput.full(0)
      if audioOutputs.count > 1 {
        audioOutput[0..<1, 0..<audioFrames, 0..<shape[3]] =
          (1 / Float(audioOutputs.count))
          * (audioOutputs[1..<audioOutputs.count].reduce(audioOutputs[0]) { $0 + $1 })
      } else {
        audioOutput[0..<1, 0..<audioFrames, 0..<shape[3]] = audioOutputs[0]
      }
      finalEt[
        0..<shape[0], startHeight..<(startHeight + audioHeight), 0..<startWidth, 0..<channels] =
        audioOutput.reshaped(.NHWC(shape[0], audioHeight, startWidth, channels))
    }
    return finalEt
  }

  public func callAsFunction(
    timestep timestepValue: (now: Float, next: Float), audioShiftRatio: Float,
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
    if let extraProjection = extraProjection, let timeEmbed = timeEmbed, let timestep = timestep {
      let batchSize = xT.shape[0]
      var embGPU = timeEmbed(inputs: timestep)[0].as(of: FloatType.self)
      embGPU = embGPU + extraProjection.reshaped(.NC(batchSize, 384 * 4))
      if tiledDiffusion.isEnabled {
        return tiledDiffuse(
          tiledDiffusion: tiledDiffusion, xT: xT, inputs: [embGPU, c[0]],
          injectedControlsAndAdapters: injectedControlsAndAdapters,
          referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
          step: step, timestep: timestepValue,
          audioShiftRatio: audioShiftRatio,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          isCfgEnabled: isCfgEnabled, controlNets: &controlNets)
      } else {
        return self(
          referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
          step: step, timestep: timestepValue,
          audioShiftRatio: audioShiftRatio, index: 0,
          tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
          isCfgEnabled: isCfgEnabled, inputs: xT, [embGPU, c[0]])
      }
    }
    // Interleaving injectedAdapters with c.
    var c = c
    if injectedIPAdapters.count > 0 {
      switch version {
      case .v1:
        let injectIPAdapters = injectedIPAdapters.count / 32
        var newC = [c[0]]
        for i in stride(from: 0, to: 32, by: 2) {
          for j in 0..<injectIPAdapters {
            newC.append(injectedIPAdapters[i + j * 32])  // ip_k
            newC.append(injectedIPAdapters[i + 1 + j * 32])  // ip_v
          }
        }
        c = newC
      case .sdxlBase, .sdxlRefiner, .ssd1b:
        precondition(injectedIPAdapters.count % (c.count - 1) == 0)
        precondition((c.count - 1) % 2 == 0)
        let injectIPAdapters = injectedIPAdapters.count / (c.count - 1)
        var newC = [c[0]]
        for i in stride(from: 0, to: c.count - 1, by: 2) {
          newC.append(c[i + 1])  // k
          newC.append(c[i + 2])  // v
          for j in 0..<injectIPAdapters {
            newC.append(injectedIPAdapters[i + j * (c.count - 1)])  // ip_k
            newC.append(injectedIPAdapters[i + 1 + j * (c.count - 1)])  // ip_v
          }
        }
        c = newC
      case .flux1:
        let injectIPAdapters = injectedIPAdapters.count / 40
        var newC = c
        for i in stride(from: 0, to: 40, by: 2) {
          for j in 0..<injectIPAdapters {
            newC.append(injectedIPAdapters[i + j * 40])  // ip_k
            newC.append(injectedIPAdapters[i + 1 + j * 40])  // ip_v
          }
        }
        c = newC
      case .v2, .sd3, .sd3Large, .pixart, .auraflow, .kandinsky21, .svdI2v, .wurstchenStageC,
        .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1, .hiDreamO1,
        .qwenImage, .qwenImage2_1, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b,
        .cosmos2_5_2b, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2,
        .longcatVideoAvatar1_5, .minimaxH3:
        fatalError()
      }
    }
    if tiledDiffusion.isEnabled {
      return tiledDiffuse(
        tiledDiffusion: tiledDiffusion, xT: xT, inputs: (timestep.map { [$0] } ?? []) + c,
        injectedControlsAndAdapters: injectedControlsAndAdapters,
        referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
        step: step, timestep: timestepValue,
        audioShiftRatio: audioShiftRatio,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: isCfgEnabled, controlNets: &controlNets)
    } else {
      let runtimeInputs = (timestep.map { [$0] } ?? []) + c
      let (injectedControls, injectedT2IAdapters, injectedAttentionKVs) =
        injectedControlsAndAdapters(
          xT, runtimeInputs, 0, 0, 0, 0, &controlNets)
      return self(
        referenceImageCount: referenceImageCount, referenceAudioCount: referenceAudioCount,
        step: step, timestep: timestepValue,
        audioShiftRatio: audioShiftRatio, index: 0,
        tokenLengthUncond: tokenLengthUncond, tokenLengthCond: tokenLengthCond,
        isCfgEnabled: isCfgEnabled, inputs: xT,
        runtimeInputs + injectedControls + injectedT2IAdapters + injectedAttentionKVs)
    }
  }

  public func decode(_ x: DynamicGraph.Tensor<FloatType>) -> DynamicGraph.Tensor<FloatType> {
    switch version {
    case .wurstchenStageC:
      if let previewer = previewer {
        return previewer(inputs: x)[0].as(of: FloatType.self)
      }
      return x
    case .v1, .v2, .sd3, .sd3Large, .pixart, .auraflow, .flux1, .sdxlBase, .sdxlRefiner, .ssd1b,
      .svdI2v, .kandinsky21, .wurstchenStageB, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .hiDreamI1,
      .hiDreamO1, .qwenImage, .qwenImage2_1, .wan22_5b, .zImage, .ernieImage, .flux2, .flux2_9b,
      .flux2_4b, .cosmos2_5_2b, .ideogram4, .krea2, .ltx2, .ltx2_3, .seedvr2_3b, .seedvr2_7b,
      .longcatVideoAvatar1_5, .minimaxH3:
      return x
    }
  }

  public mutating func cancel() {
    isCancelled.store(true, ordering: .releasing)
    unet?.cancel()
    unconditionalUNet?.cancel()
    teaCache?.cancel()
    unet = nil
    unconditionalUNet = nil
    teaCache = nil
  }
}
