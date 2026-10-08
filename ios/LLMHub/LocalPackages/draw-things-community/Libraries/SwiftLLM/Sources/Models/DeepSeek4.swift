import Foundation
import NNC

public enum DeepSeek4AttentionKind: Sendable, Equatable {
  case raw
  case compressed(compressionRatio: Int)
  case indexed(compressionRatio: Int)
}

public enum DeepSeek4RouterKind: Sendable, Equatable {
  case standard
  case tokenHash
}

public struct DeepSeek4ModelConfiguration: Sendable {
  public var vocabularySize: Int
  public var hiddenSize: Int
  public var layers: Int
  public var hcCount: Int
  public var attentionHeads: Int
  public var attentionHeadDim: Int
  public var rotaryDim: Int
  public var rawWindow: Int
  public var expertCount: Int
  public var routedExperts: Int
  public var expertIntermediateSize: Int
  public var sharedIntermediateSize: Int
  public var expertResidentSlots: [Int]
  public var attentionOutputGroups: Int
  public var attentionLowRank: Int
  public var queryLowRank: Int
  public var indexerHeads: Int
  public var indexerHeadDim: Int
  public var indexerTopK: Int
  public var ropeTheta: Double
  public var ropeScaleFactor: Double
  public var ropeOriginalContext: Int
  public var ropeYarnBetaFast: Double
  public var ropeYarnBetaSlow: Double
  public var layerAttentionKinds: [DeepSeek4AttentionKind]
  public var layerRouterKinds: [DeepSeek4RouterKind]

  public init(
    vocabularySize: Int, hiddenSize: Int, layers: Int, hcCount: Int,
    attentionHeads: Int, attentionHeadDim: Int, rotaryDim: Int, rawWindow: Int,
    expertCount: Int, routedExperts: Int, expertIntermediateSize: Int,
    sharedIntermediateSize: Int, attentionOutputGroups: Int, attentionLowRank: Int,
    queryLowRank: Int, indexerHeads: Int, indexerHeadDim: Int, indexerTopK: Int,
    ropeTheta: Double, ropeScaleFactor: Double, ropeOriginalContext: Int,
    ropeYarnBetaFast: Double, ropeYarnBetaSlow: Double,
    layerAttentionKinds: [DeepSeek4AttentionKind],
    layerRouterKinds: [DeepSeek4RouterKind], expertResidentSlots: [Int]
  ) {
    precondition(layerAttentionKinds.count == layers)
    precondition(layerRouterKinds.count == layers)
    precondition(
      expertResidentSlots.count == layers
        && expertResidentSlots.allSatisfy { $0 >= routedExperts && $0 <= expertCount })
    self.vocabularySize = vocabularySize
    self.hiddenSize = hiddenSize
    self.layers = layers
    self.hcCount = hcCount
    self.attentionHeads = attentionHeads
    self.attentionHeadDim = attentionHeadDim
    self.rotaryDim = rotaryDim
    self.rawWindow = rawWindow
    self.expertCount = expertCount
    self.routedExperts = routedExperts
    self.expertIntermediateSize = expertIntermediateSize
    self.sharedIntermediateSize = sharedIntermediateSize
    self.expertResidentSlots = expertResidentSlots
    self.attentionOutputGroups = attentionOutputGroups
    self.attentionLowRank = attentionLowRank
    self.queryLowRank = queryLowRank
    self.indexerHeads = indexerHeads
    self.indexerHeadDim = indexerHeadDim
    self.indexerTopK = indexerTopK
    self.ropeTheta = ropeTheta
    self.ropeScaleFactor = ropeScaleFactor
    self.ropeOriginalContext = ropeOriginalContext
    self.ropeYarnBetaFast = ropeYarnBetaFast
    self.ropeYarnBetaSlow = ropeYarnBetaSlow
    self.layerAttentionKinds = layerAttentionKinds
    self.layerRouterKinds = layerRouterKinds
  }

  public var attentionOutputLowDim: Int { attentionOutputGroups * attentionLowRank }
  public var hcMixDim: Int { 2 * hcCount + hcCount * hcCount }

  public func attentionKind(layerIndex: Int) -> DeepSeek4AttentionKind {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return layerAttentionKinds[layerIndex]
  }

  public func routerKind(layerIndex: Int) -> DeepSeek4RouterKind {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return layerRouterKinds[layerIndex]
  }

  /// Compression ratios used by the configured attention layers.
  public var compressionRatios: [Int] {
    var ratios = Set<Int>()
    for attentionKind in layerAttentionKinds {
      switch attentionKind {
      case .raw:
        break
      case .compressed(let compressionRatio), .indexed(let compressionRatio):
        ratios.insert(compressionRatio)
      }
    }
    return ratios.sorted()
  }

  /// Compression ratios used by indexed attention layers.
  public var indexerCompressionRatios: [Int] {
    var ratios = Set<Int>()
    for attentionKind in layerAttentionKinds {
      if case .indexed(let compressionRatio) = attentionKind {
        ratios.insert(compressionRatio)
      }
    }
    return ratios.sorted()
  }
}

extension DeepSeek4ModelConfiguration {
  public static let deepSeekV4Flash = DeepSeek4ModelConfiguration(
    vocabularySize: 129_280,
    hiddenSize: 4_096,
    layers: 43,
    hcCount: 4,
    attentionHeads: 64,
    attentionHeadDim: 512,
    rotaryDim: 64,
    rawWindow: 128,
    expertCount: 256,
    routedExperts: 6,
    expertIntermediateSize: 2_048,
    sharedIntermediateSize: 2_048,
    attentionOutputGroups: 8,
    attentionLowRank: 1_024,
    queryLowRank: 1_024,
    indexerHeads: 64,
    indexerHeadDim: 128,
    indexerTopK: 512,
    ropeTheta: 160_000,
    ropeScaleFactor: 16,
    ropeOriginalContext: 65_536,
    ropeYarnBetaFast: 32,
    ropeYarnBetaSlow: 1,
    layerAttentionKinds: [
      .raw, .raw, .indexed(compressionRatio: 4), .compressed(compressionRatio: 128),
    ]
      + (4..<43).map {
        $0.isMultiple(of: 2) ? .indexed(compressionRatio: 4) : .compressed(compressionRatio: 128)
      },
    layerRouterKinds: (0..<43).map { $0 < 3 ? .tokenHash : .standard },
    expertResidentSlots: Array(repeating: 256, count: 43))
}

/// Describes the compressor rows and retained state for one continuation step.
public struct DeepSeek4CompressionPlan: Sendable, Equatable {
  public let compressionRatio: Int
  public let existingRowCount: Int
  public let totalRowCount: Int
  public let emittedRowCount: Int
  public let compressorTokenOffset: Int
  public let compressorTokenCount: Int
  public let compressorOutputTokenOffset: Int
  public let stateCount: Int
  public let nextStateCount: Int
  public let nextStateOffset: Int

  public init(cachedTokenLength: Int, tokenLength: Int, compressionRatio: Int) {
    precondition(cachedTokenLength >= 0)
    precondition(tokenLength > 0)
    precondition(compressionRatio > 0)
    let existingRowCount = cachedTokenLength / compressionRatio
    let cachedRemainder = cachedTokenLength % compressionRatio
    let keepsPreviousWindow = compressionRatio == 4 && existingRowCount > 0
    let stateCount = cachedRemainder + (keepsPreviousWindow ? compressionRatio : 0)
    let compressorTokenOffset = cachedTokenLength - stateCount
    let totalTokenLength = cachedTokenLength + tokenLength
    let totalRowCount = totalTokenLength / compressionRatio
    let emittedRowCount = totalRowCount - existingRowCount
    let retainedWindowCount =
      compressionRatio == 4 && existingRowCount > 0 && emittedRowCount > 0 ? 1 : 0
    let compressorTokenCount =
      (retainedWindowCount + emittedRowCount) * compressionRatio
    let compressorOutputTokenOffset =
      compressorTokenOffset + retainedWindowCount * compressionRatio
    let nextRemainder = totalTokenLength % compressionRatio
    let nextStateCount =
      nextRemainder
      + (compressionRatio == 4 && totalRowCount > 0 ? compressionRatio : 0)
    let nextStateTokenOffset =
      compressionRatio == 4 && totalRowCount > 0
      ? (totalRowCount - 1) * compressionRatio
      : totalRowCount * compressionRatio
    let nextStateOffset = nextStateTokenOffset - compressorTokenOffset
    precondition(nextStateOffset >= 0)
    precondition(nextStateOffset + nextStateCount <= stateCount + tokenLength)
    self.compressionRatio = compressionRatio
    self.existingRowCount = existingRowCount
    self.totalRowCount = totalRowCount
    self.emittedRowCount = emittedRowCount
    self.compressorTokenOffset = compressorTokenOffset
    self.compressorTokenCount = compressorTokenCount
    self.compressorOutputTokenOffset = compressorOutputTokenOffset
    self.stateCount = stateCount
    self.nextStateCount = nextStateCount
    self.nextStateOffset = nextStateOffset
  }

  public var stateCapacity: Int {
    DeepSeek4CompressorStateCapacity(compressionRatio: compressionRatio)
  }
}

public func DeepSeek4CompressorStateCapacity(compressionRatio: Int) -> Int {
  precondition(compressionRatio > 0)
  return compressionRatio == 4 ? compressionRatio * 2 - 1 : compressionRatio - 1
}

private func DeepSeek4RopeYarnRamp(low: Double, high: Double, index: Int) -> Double {
  let y = (Double(index / 2) - low) / max(0.001, high - low)
  return 1.0 - min(1.0, max(0.0, y))
}

private func DeepSeek4RopeYarnCorrDim(
  nDims: Int, originalContext: Int, nRot: Double, base: Double
) -> Double {
  return Double(nDims) * log(Double(originalContext) / (nRot * 2.0 * Double.pi))
    / (2.0 * log(base))
}

private func DeepSeek4RopeYarnCorrDims(
  nDims: Int, originalContext: Int, base: Double, betaFast: Double, betaSlow: Double
) -> (Double, Double) {
  let start = floor(
    DeepSeek4RopeYarnCorrDim(
      nDims: nDims, originalContext: originalContext, nRot: betaFast, base: base))
  let end = ceil(
    DeepSeek4RopeYarnCorrDim(
      nDims: nDims, originalContext: originalContext, nRot: betaSlow, base: base))
  return (max(0.0, start), min(Double(nDims - 1), end))
}

public func DeepSeek4RotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, cachedTokenLength: Int = 0,
  positionStride: Int = 1, compressed: Bool = false, headDim: Int? = nil,
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash,
  of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  precondition(sequenceLength > 0)
  precondition(positionStride > 0)
  let headDim = headDim ?? configuration.attentionHeadDim
  let nRot = configuration.rotaryDim
  precondition(nRot > 0 && nRot.isMultiple(of: 2))
  precondition(headDim >= nRot && (headDim - nRot).isMultiple(of: 2))
  let nNope = headDim - nRot
  let freqBase = compressed ? configuration.ropeTheta : 10_000
  let freqScale = compressed ? 1.0 / configuration.ropeScaleFactor : 1.0
  let extFactor = compressed ? 1.0 : 0.0
  let attnFactor = extFactor != 0 ? 1.0 / (1.0 + 0.1 * log(1.0 / freqScale)) : 1.0
  let corr = DeepSeek4RopeYarnCorrDims(
    nDims: nRot, originalContext: configuration.ropeOriginalContext,
    base: freqBase, betaFast: configuration.ropeYarnBetaFast,
    betaSlow: configuration.ropeYarnBetaSlow)
  let rotaryComponents = stride(from: 0, to: nRot, by: 2).map { i in
    (
      frequency: pow(freqBase, -Double(i) / Double(nRot)),
      rampMix: DeepSeek4RopeYarnRamp(low: corr.0, high: corr.1, index: i) * extFactor
    )
  }
  let mscale =
    extFactor != 0
    ? attnFactor * (1.0 + 0.1 * log(1.0 / freqScale)) : attnFactor
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, headDim))
  rotary.withUnsafeMutableBytes { rotaryBuffer in
    guard let rotaryPointer = rotaryBuffer.baseAddress?.assumingMemoryBound(to: FloatType.self)
    else { preconditionFailure("A non-empty rotary tensor must have storage") }
    for row in 0..<sequenceLength {
      let rowPointer = rotaryPointer.advanced(by: row * headDim)
      for pairIndex in 0..<(nNope / 2) {
        rowPointer[pairIndex * 2] = 1
        rowPointer[pairIndex * 2 + 1] = 0
      }
      let position = Double(cachedTokenLength + row * positionStride)
      for componentIndex in 0..<(nRot / 2) {
        let component = rotaryComponents[componentIndex]
        let thetaExtrap = position * component.frequency
        let thetaInterp = freqScale * thetaExtrap
        let theta =
          thetaInterp * (1.0 - component.rampMix) + thetaExtrap * component.rampMix
        let offset = nNope + componentIndex * 2
        rowPointer[offset] = FloatType(cos(theta) * mscale)
        rowPointer[offset + 1] = FloatType(sin(theta) * mscale)
      }
    }
  }
  return rotary
}

private func DeepSeek4HCSplitWeightedSum(
  mix: Model.IO, scale: ModelIOConvertible, base: ModelIOConvertible,
  residualHC: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> (post: Model.IO, comb: Model.IO, weighted: Model.IO) {
  let hc = configuration.hcCount
  let outputs = HyperConnection(
    count: hc, sinkhornIterations: 20, epsilon: 1.0e-6, operation: .splitWeightedSum
  )(mix, scale, base, residualHC)
  return (
    outputs[0].reshaped([tokenLength, hc]),
    outputs[1].reshaped([tokenLength, hc, hc]),
    outputs[2].reshaped([tokenLength, configuration.hiddenSize])
  )
}

private func DeepSeek4HCExpand(
  block: Model.IO, residualHC: Model.IO, post: Model.IO, comb: Model.IO,
  tokenLength: Int, configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let hc = configuration.hcCount
  return HyperConnection(count: hc, operation: .expand)(
    block, residualHC,
    post, comb)[0]
}

private func DeepSeek4AttentionProjection(
  prefix: String, x: Model.IO, rotary: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> (query: Model.IO, keyValue: Model.IO, queryRank: Model.IO) {
  let headDim = configuration.attentionHeadDim
  let heads = configuration.attentionHeads
  let wqA = Dense(
    count: configuration.queryLowRank, noBias: true, flags: [.Float16],
    name: "\(prefix).wq_a")
  let wqB = Dense(
    count: heads * headDim, noBias: true, flags: [.Float16],
    name: "\(prefix).wq_b")
  let wkv = Dense(
    count: headDim, noBias: true, flags: [.Float16], name: "\(prefix).wkv")
  let q8Input = x

  let queryLowRankRaw = wqA(q8Input)
  let queryRank = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).q_norm")(
    queryLowRankRaw
  )
  let qDense = wqB(queryRank).reshaped([1, tokenLength, heads, headDim])
  let query = RMSNormCmul(epsilon: 1.0e-6, axis: [3], elementwiseAffine: false)(
    qDense,
    rotary.reshaped([1, tokenLength, 1, headDim])
  )

  let kvRaw = wkv(q8Input).reshaped([1, tokenLength, 1, headDim])
  let keyValue = RMSNormCmul(epsilon: 1.0e-6, axis: [3], name: "\(prefix).kv_norm")(
    kvRaw, rotary.reshaped([1, tokenLength, 1, headDim])
  )
  return (query, keyValue, queryRank)
}

private func DeepSeek4AttentionOutput<FloatType: TensorNumeric>(
  prefix: String, heads: Model.IO, rotary: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> Model.IO {
  let headDim = configuration.attentionHeadDim
  let outGroups = configuration.attentionOutputGroups
  let headsPerGroup = configuration.attentionHeads / outGroups
  let groupDim = headsPerGroup * headDim
  let headsBack = Functional.cmul(
    left: heads.reshaped([tokenLength, configuration.attentionHeads, headDim]),
    right: rotary.reshaped([tokenLength, 1, headDim]), conjugate: true
  ).reshaped([tokenLength, configuration.attentionHeads * headDim])
  let woA = Parameter<FloatType>(
    .GPU(0), .HWC(outGroups, configuration.attentionLowRank, groupDim),
    name: "\(prefix).wo_a")
  let groupedHeads =
    headsBack.reshaped([tokenLength, outGroups, 1, groupDim])
  let low = Matmul(transposeB: (1, 2))(groupedHeads, woA)
    .reshaped([tokenLength, configuration.attentionOutputLowDim])
  let woB = Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).wo_b")
  return woB(low).reshaped([tokenLength, configuration.hiddenSize])
}

private func DeepSeek4Ratio4RollingPool(
  kvProjected: Model.IO, scoreProjected: Model.IO, ape: ModelIOConvertible,
  zeroPad: Model.IO, negInfPad: Model.IO,
  sourceWindowCount: Int, outputWindowCount: Int, headDim: Int
) -> Model.IO {
  let compressionRatio = 4
  let width = 2 * headDim
  let rowWidth = width * compressionRatio
  let sourceWindowOffset = sourceWindowCount - outputWindowCount
  precondition(sourceWindowOffset == 0 || sourceWindowOffset == 1)
  let previousWindowCount = max(sourceWindowCount - 1, 0)
  let paddingWindowCount = outputWindowCount - previousWindowCount
  precondition(paddingWindowCount == 0 || paddingWindowCount == 1)
  let score =
    scoreProjected.reshaped([sourceWindowCount, compressionRatio, width])
    + ape
  let primaryKV = kvProjected.reshaped(
    [sourceWindowCount, compressionRatio, headDim], offset: [0, 0, 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let primaryScore = score.reshaped(
    [sourceWindowCount, compressionRatio, headDim], offset: [0, 0, 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let companionKV = kvProjected.reshaped(
    [outputWindowCount, compressionRatio, headDim],
    offset: [sourceWindowOffset, 0, outputWindowCount > 0 ? headDim : 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let companionScore = score.reshaped(
    [outputWindowCount, compressionRatio, headDim],
    offset: [sourceWindowOffset, 0, outputWindowCount > 0 ? headDim : 0],
    strides: [rowWidth, width, 1]
  ).transposed(1, 2).contiguous()
  let previousPrimaryKV = primaryKV.reshaped(
    [previousWindowCount, headDim, compressionRatio], offset: [0, 0, 0],
    strides: [headDim * compressionRatio, compressionRatio, 1])
  let previousPrimaryScore = primaryScore.reshaped(
    [previousWindowCount, headDim, compressionRatio], offset: [0, 0, 0],
    strides: [headDim * compressionRatio, compressionRatio, 1])
  let previousKV = Concat(axis: 0)([zeroPad, previousPrimaryKV])
  let previousScore = Concat(axis: 0)([negInfPad, previousPrimaryScore])
  let rows = outputWindowCount * headDim
  let paddedKV = Concat(axis: 1)([
    previousKV.reshaped([rows, compressionRatio]),
    companionKV.reshaped([rows, compressionRatio]),
  ]).reshaped([outputWindowCount, headDim, 2 * compressionRatio])
  let paddedScore = Concat(axis: 1)([
    previousScore.reshaped([rows, compressionRatio]),
    companionScore.reshaped([rows, compressionRatio]),
  ]).reshaped([outputWindowCount, headDim, 2 * compressionRatio])
  let weights = paddedScore.reshaped([rows, 2 * compressionRatio])
    .softmax()
    .reshaped([outputWindowCount, headDim, 2 * compressionRatio])
  return (weights .* paddedKV).reduced(.sum, axis: [2]).reshaped([
    outputWindowCount, headDim,
  ])
}

private func DeepSeek4Compressor<FloatType: TensorNumeric>(
  prefix: String, x: Model.IO, rotary: Model.IO, tokenLength: Int, compressionRatio: Int,
  outputRowCount: Int, headDim: Int, emitIndexerWHT: Bool,
  compressorInputCacheOut: Model.IO,
  zeroPad: Model.IO?, negInfPad: Model.IO?,
  configuration: DeepSeek4ModelConfiguration,
  of dataType: FloatType.Type
) -> Model.IO {
  let sourceWindowCount = tokenLength / compressionRatio
  let width = (compressionRatio == 4 ? 2 : 1) * headDim
  let kv = Dense(count: width, noBias: true, name: "\(prefix).wkv")
  let gate = Dense(count: width, noBias: true, name: "\(prefix).wgate")
  let ape = Parameter<FloatType>(
    .GPU(0), .HWC(1, compressionRatio, width), name: "\(prefix).ape")()
  let kvProjected = kv(x)
  kvProjected.add(dependencies: [compressorInputCacheOut])
  let scoreProjected = gate(x)
  scoreProjected.add(dependencies: [compressorInputCacheOut])
  let pooled: Model.IO
  if compressionRatio == 4 {
    guard let zeroPad, let negInfPad else {
      preconditionFailure("Ratio-4 compression requires explicit padding inputs.")
    }
    pooled = DeepSeek4Ratio4RollingPool(
      kvProjected: kvProjected, scoreProjected: scoreProjected, ape: ape,
      zeroPad: zeroPad, negInfPad: negInfPad,
      sourceWindowCount: sourceWindowCount, outputWindowCount: outputRowCount,
      headDim: headDim)
  } else {
    precondition(sourceWindowCount == outputRowCount)
    let kvRows = kvProjected.reshaped([outputRowCount, compressionRatio, headDim])
    let scores =
      scoreProjected.reshaped([outputRowCount, compressionRatio, headDim])
      + ape
    let weights = scores.transposed(1, 2)
      .reshaped([outputRowCount * headDim, compressionRatio])
      .softmax()
      .reshaped([outputRowCount, headDim, compressionRatio])
      .transposed(1, 2)
    pooled = (weights .* kvRows).reduced(.sum, axis: [1]).reshaped([
      outputRowCount, headDim,
    ])
  }
  let compressed = RMSNormCmul(epsilon: 1.0e-6, axis: [1], name: "\(prefix).norm")(
    pooled,
    rotary.reshaped(
      [outputRowCount, headDim], offset: [0, 0], strides: [headDim, 1]))
  if emitIndexerWHT {
    return WalshHadamardTransform(scale: 1.0 / Float(headDim).squareRoot())(compressed)
  } else {
    return compressed
  }
}

private func DeepSeek4IndexerSelection(
  prefix: String, queryRank: Model.IO, attnNorm: Model.IO, rotary: Model.IO,
  indexerKVIn: Model.IO, indexerKVOut: Model.IO, tokenLength: Int, queryOffset: Int,
  compressionRatio: Int,
  compressedRows: Int, configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let indexerWqB = Dense(
    count: configuration.indexerHeads * configuration.indexerHeadDim, noBias: true,
    name: "\(prefix).indexer.wq_b")
  let indexerWeightsProj = Dense(
    count: configuration.indexerHeads, noBias: true, name: "\(prefix).indexer.weights_proj")
  let indexDense = indexerWqB(queryRank)
    .reshaped([
      tokenLength, configuration.indexerHeads, configuration.indexerHeadDim,
    ])
  let indexRows = Functional.cmul(
    left: indexDense,
    right: rotary.reshaped([tokenLength, 1, configuration.indexerHeadDim])
  ).reshaped([
    tokenLength * configuration.indexerHeads, configuration.indexerHeadDim,
  ])
  let indexQ = WalshHadamardTransform(
    scale: 1.0 / Float(configuration.indexerHeadDim).squareRoot())(indexRows)
  let indexWeights =
    indexerWeightsProj(attnNorm)
    * (1.0
      / (Float(configuration.indexerHeadDim).squareRoot()
        * Float(configuration.indexerHeads).squareRoot()))
  let out = ScaledDotProductArgPartition(
    kth: configuration.indexerTopK,
    scale: 1,
    isCausal: true,
    compressionRatio: compressionRatio,
    queryOffset: queryOffset,
    name: "\(prefix).indexer.sdpap")(
      indexQ.reshaped([tokenLength, configuration.indexerHeads, configuration.indexerHeadDim]),
      indexerKVIn,
      indexWeights
    )
  out.add(dependencies: [indexerKVOut])
  return out
}

private struct DeepSeek4SWAttentionInputs {
  let rotary: Input
  let rawKeyValue: Input
}

private struct DeepSeek4CompressedSparseAttentionInputs {
  let rotary: Input
  let indexerRotary: Input
  let compressorRotary: Input
  let indexerCompressorRotary: Input
  let compressorZeroPad: Input
  let compressorNegInfPad: Input
  let indexerCompressorZeroPad: Input
  let indexerCompressorNegInfPad: Input
  let rawKeyValue: Input
  let compressedKeyValue: Input
  let compressorInputCache: Input
  let indexerKeyValue: Input
  let compressionRatio: Int
  let cachedCompressorInputLength: Int
}

private struct DeepSeek4HighlyCompressedAttentionInputs {
  let rotary: Input
  let compressorRotary: Input
  let causalCompressedIndices: Input
  let rawKeyValue: Input
  let compressedKeyValue: Input
  let compressorInputCache: Input
  let compressionRatio: Int
  let cachedCompressorInputLength: Int
}

private enum DeepSeek4LayerAttentionInputs {
  case swa(DeepSeek4SWAttentionInputs)
  case compressedSparse(DeepSeek4CompressedSparseAttentionInputs)
  case highlyCompressed(DeepSeek4HighlyCompressedAttentionInputs)
}

private func DeepSeek4SWAttention<FloatType: TensorNumeric>(
  prefix: String, tokenLength: Int, cachedRawTokenLength: Int,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> Model {
  let x = Input()
  let rotary = Input()
  let kvIn = Input()
  let headDim = configuration.attentionHeadDim
  let totalRawRows = cachedRawTokenLength + tokenLength
  let sinks = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, configuration.attentionHeads, 1), name: "\(prefix).attn_sink")
  let projection = DeepSeek4AttentionProjection(
    prefix: prefix, x: x, rotary: rotary, tokenLength: tokenLength,
    configuration: configuration)
  let kvOut = ConformDataFormat(
    .FP8E4M3, preservedTail: configuration.rotaryDim
  )(projection.keyValue).moved(
    to: kvIn.reshaped(
      [1, tokenLength, 1, headDim], offset: [0, cachedRawTokenLength, 0, 0],
      strides: [headDim * totalRawRows, headDim, headDim, 1]), flags: [.disableOpt])
  let heads = ScaledDotProductAttention(
    scale: 1.0 / Float(headDim).squareRoot(), isCausal: true,
    hasAttentionSinks: true, slidingWindow: configuration.rawWindow,
    name: "\(prefix).swa")(projection.query, kvIn, kvIn, sinks)
  heads.add(dependencies: [kvOut])
  let output = DeepSeek4AttentionOutput(
    prefix: prefix,
    heads: heads.reshaped([tokenLength, configuration.attentionHeads * headDim]),
    rotary: rotary, tokenLength: tokenLength, configuration: configuration, of: dataType)
  return Model([x, rotary, kvIn], [output])
}

private func DeepSeek4HighlyCompressedAttention<FloatType: TensorNumeric>(
  prefix: String, tokenLength: Int, cachedTokenLength: Int, cachedRawTokenLength: Int,
  cachedCompressorInputLength: Int, compressionRatio: Int,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> Model {
  let x = Input()
  let rotary = Input()
  let compressorRotary = Input()
  let causalCompressedIndices = Input()
  let kvIn = Input()
  let compressedKVIn = Input()
  let compressorInputCacheIn = Input()
  let headDim = configuration.attentionHeadDim
  let totalRawRows = cachedRawTokenLength + tokenLength
  let sinks = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, configuration.attentionHeads, 1), name: "\(prefix).attn_sink")
  let projection = DeepSeek4AttentionProjection(
    prefix: prefix, x: x, rotary: rotary, tokenLength: tokenLength,
    configuration: configuration)
  let kvOut = ConformDataFormat(
    .FP8E4M3, preservedTail: configuration.rotaryDim
  )(projection.keyValue).moved(
    to: kvIn.reshaped(
      [1, tokenLength, 1, headDim], offset: [0, cachedRawTokenLength, 0, 0],
      strides: [totalRawRows * headDim, headDim, headDim, 1]), flags: [.disableOpt])

  let plan = DeepSeek4CompressionPlan(
    cachedTokenLength: cachedTokenLength, tokenLength: tokenLength,
    compressionRatio: compressionRatio)
  let hidden = configuration.hiddenSize
  precondition(cachedCompressorInputLength >= plan.stateCount)
  let compressorInputCacheOut = x.moved(
    to: compressorInputCacheIn.reshaped(
      [tokenLength, hidden], offset: [cachedCompressorInputLength, 0],
      strides: [hidden, 1]), flags: [.disableOpt])
  let compressorInput =
    plan.compressorTokenCount > 0
    ? compressorInputCacheIn.reshaped(
      [plan.compressorTokenCount, hidden],
      offset: [cachedCompressorInputLength - plan.stateCount, 0],
      strides: [hidden, 1])
    : compressorInputCacheIn.reshaped([0])
  let emittedRows = DeepSeek4Compressor(
    prefix: "\(prefix).compressor", x: compressorInput,
    rotary: compressorRotary, tokenLength: plan.compressorTokenCount,
    compressionRatio: compressionRatio, outputRowCount: plan.emittedRowCount,
    headDim: headDim, emitIndexerWHT: false, compressorInputCacheOut: compressorInputCacheOut,
    zeroPad: nil, negInfPad: nil,
    configuration: configuration, of: dataType)
  let compressedKVOut = ConformDataFormat(
    .FP8E4M3, preservedTail: configuration.rotaryDim)(emittedRows).moved(
      to: plan.emittedRowCount > 0
        ? compressedKVIn.reshaped(
          [plan.emittedRowCount, headDim], offset: [plan.existingRowCount, 0],
          strides: [headDim, 1])
        : compressedKVIn.reshaped([0]), flags: [.disableOpt])
  let compressedAttentionKV =
    plan.totalRowCount == 0
    ? compressedKVIn.reshaped([0])
    : compressedKVIn.reshaped(
      [1, plan.totalRowCount, 1, headDim], format: .NHWC)
  let heads = SparseIndexedAttention(
    scale: 1.0 / Float(headDim).squareRoot(),
    isCausal: true, hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow)(
      projection.query, kvIn, kvIn,
      compressedAttentionKV, compressedAttentionKV,
      causalCompressedIndices.reshaped(
        [tokenLength, max(plan.totalRowCount, 1)], format: .NHWC),
      sinks
    )
  heads.add(dependencies: [kvOut, compressedKVOut])
  let output = DeepSeek4AttentionOutput(
    prefix: prefix,
    heads: heads.reshaped([tokenLength, configuration.attentionHeads * headDim]),
    rotary: rotary, tokenLength: tokenLength, configuration: configuration, of: dataType)
  return Model(
    [
      x, rotary, compressorRotary, causalCompressedIndices, kvIn,
      compressedKVIn, compressorInputCacheIn,
    ], [output])
}

private func DeepSeek4CompressedSparseAttention<FloatType: TensorNumeric>(
  prefix: String, tokenLength: Int, cachedTokenLength: Int, cachedRawTokenLength: Int,
  cachedCompressorInputLength: Int, compressionRatio: Int,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> Model {
  let x = Input()
  let rotary = Input()
  let indexerRotary = Input()
  let compressorRotary = Input()
  let indexerCompressorRotary = Input()
  let compressorZeroPad = Input()
  let compressorNegInfPad = Input()
  let indexerCompressorZeroPad = Input()
  let indexerCompressorNegInfPad = Input()
  let kvIn = Input()
  let compressedKVIn = Input()
  let compressorInputCacheIn = Input()
  let indexerKVIn = Input()
  let headDim = configuration.attentionHeadDim
  let totalRawRows = cachedRawTokenLength + tokenLength
  let sinks = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, configuration.attentionHeads, 1), name: "\(prefix).attn_sink")
  let projection = DeepSeek4AttentionProjection(
    prefix: prefix, x: x, rotary: rotary, tokenLength: tokenLength,
    configuration: configuration)
  let kvOut = ConformDataFormat(
    .FP8E4M3, preservedTail: configuration.rotaryDim
  )(projection.keyValue).moved(
    to: kvIn.reshaped(
      [1, tokenLength, 1, headDim], offset: [0, cachedRawTokenLength, 0, 0],
      strides: [totalRawRows * headDim, headDim, headDim, 1]), flags: [.disableOpt])

  let plan = DeepSeek4CompressionPlan(
    cachedTokenLength: cachedTokenLength, tokenLength: tokenLength,
    compressionRatio: compressionRatio)
  let hidden = configuration.hiddenSize
  precondition(cachedCompressorInputLength >= plan.stateCount)
  let compressorInputCacheOut = x.moved(
    to: compressorInputCacheIn.reshaped(
      [tokenLength, hidden], offset: [cachedCompressorInputLength, 0],
      strides: [hidden, 1]), flags: [.disableOpt])
  let compressorInput =
    plan.compressorTokenCount > 0
    ? compressorInputCacheIn.reshaped(
      [plan.compressorTokenCount, hidden],
      offset: [cachedCompressorInputLength - plan.stateCount, 0],
      strides: [hidden, 1])
    : compressorInputCacheIn.reshaped([0])
  let emittedRows = DeepSeek4Compressor(
    prefix: "\(prefix).compressor", x: compressorInput,
    rotary: compressorRotary, tokenLength: plan.compressorTokenCount,
    compressionRatio: compressionRatio, outputRowCount: plan.emittedRowCount,
    headDim: headDim, emitIndexerWHT: false,
    compressorInputCacheOut: compressorInputCacheOut,
    zeroPad: compressorZeroPad, negInfPad: compressorNegInfPad,
    configuration: configuration, of: dataType)
  let compressedKVOut = ConformDataFormat(
    .FP8E4M3, preservedTail: configuration.rotaryDim)(emittedRows).moved(
      to: plan.emittedRowCount > 0
        ? compressedKVIn.reshaped(
          [plan.emittedRowCount, headDim], offset: [plan.existingRowCount, 0],
          strides: [headDim, 1])
        : compressedKVIn.reshaped([0]), flags: [.disableOpt])

  let emittedIndexerRows = DeepSeek4Compressor(
    prefix: "\(prefix).indexer.compressor", x: compressorInput,
    rotary: indexerCompressorRotary, tokenLength: plan.compressorTokenCount,
    compressionRatio: compressionRatio, outputRowCount: plan.emittedRowCount,
    headDim: configuration.indexerHeadDim, emitIndexerWHT: true,
    compressorInputCacheOut: compressorInputCacheOut,
    zeroPad: indexerCompressorZeroPad, negInfPad: indexerCompressorNegInfPad,
    configuration: configuration, of: dataType)
  let indexerKVOut = emittedIndexerRows.moved(
    to: plan.emittedRowCount > 0
      ? indexerKVIn.reshaped(
        [plan.emittedRowCount, configuration.indexerHeadDim],
        offset: [plan.existingRowCount, 0], strides: [configuration.indexerHeadDim, 1])
      : indexerKVIn.reshaped([0]),
    flags: [.disableOpt])
  let selectedRows = DeepSeek4IndexerSelection(
    prefix: prefix, queryRank: projection.queryRank, attnNorm: x,
    rotary: indexerRotary, indexerKVIn: indexerKVIn, indexerKVOut: indexerKVOut,
    tokenLength: tokenLength,
    queryOffset: cachedTokenLength, compressionRatio: compressionRatio,
    compressedRows: plan.totalRowCount, configuration: configuration)
  let compressedAttentionKVInput =
    plan.totalRowCount == 0
    ? compressedKVIn.reshaped([0])
    : compressedKVIn.reshaped(
      [1, plan.totalRowCount, 1, headDim], format: .NHWC)
  let heads = SparseIndexedAttention(
    scale: 1.0 / Float(headDim).squareRoot(),
    isCausal: true, hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow)(
      projection.query, kvIn, kvIn,
      compressedAttentionKVInput, compressedAttentionKVInput,
      selectedRows.reshaped(
        [tokenLength, configuration.indexerTopK], format: .NHWC),
      sinks
    )
  heads.add(dependencies: [kvOut, compressedKVOut])
  let output = DeepSeek4AttentionOutput(
    prefix: prefix,
    heads: heads.reshaped([tokenLength, configuration.attentionHeads * headDim]),
    rotary: rotary, tokenLength: tokenLength, configuration: configuration, of: dataType)
  return Model(
    [
      x, rotary, indexerRotary, compressorRotary, indexerCompressorRotary,
      compressorZeroPad, compressorNegInfPad, indexerCompressorZeroPad,
      indexerCompressorNegInfPad, kvIn, compressedKVIn,
      compressorInputCacheIn, indexerKVIn,
    ], [output])
}

private func DeepSeek4SharedFFN(
  prefix: String, x: Model.IO, tokenLength: Int,
  dependencies: [Model.IO] = [], configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let swiglu = SwiGLU(
    count: configuration.sharedIntermediateSize, clamp: 10,
    name: "\(prefix).shared_experts")
  let down = Dense(
    count: configuration.hiddenSize, noBias: true, name: "\(prefix).shared_experts.w2")
  let hidden = swiglu(x)
  if !dependencies.isEmpty {
    hidden.add(dependencies: dependencies)
  }
  return down(hidden)
}

private func DeepSeek4RoutedMoE(
  prefix: String, x: Model.IO, routerInput: Model.IO, tokens: Model.IO,
  layerIndex: Int, tokenLength: Int, routerKind: DeepSeek4RouterKind,
  configuration: DeepSeek4ModelConfiguration
) -> (routed: Model.IO, shared: Model.IO) {
  let router = Dense(count: configuration.expertCount, noBias: true, name: "\(prefix).gate")
  let logits = router(routerInput)
  let route: Model.IO
  switch routerKind {
  case .standard:
    route = Parameter<Float>(
      .GPU(0), .C(configuration.expertCount), name: "\(prefix).gate.bias"
    ).reshaped([configuration.expertCount])
  case .tokenHash:
    let tokenHash = Embedding(
      Int32.self, vocabularySize: configuration.vocabularySize,
      embeddingSize: configuration.routedExperts,
      name: "\(prefix).gate.tid2eid")
    route = tokenHash(tokens).reshaped([
      tokenLength, configuration.routedExperts,
    ])
  }
  let pairs = tokenLength * configuration.routedExperts
  let prepared = MoERouting(
    kth: configuration.routedExperts, weightScale: 1.5,
    preselected: routerKind == .tokenHash, singleInputToken: true,
    name: "\(prefix).routing"
  )(logits.to(of: x), route, x)
  if configuration.expertResidentSlots[layerIndex] == configuration.expertCount {
    let hidden = SegmentedSwiGLU(
      segments: configuration.expertCount, count: configuration.expertIntermediateSize,
      clamp: 10, name: "\(prefix).experts"
    )([
      prepared[0], prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]),
    ])
    let down = SegmentedDense(
      segments: configuration.expertCount, count: configuration.hiddenSize,
      noBias: true, name: "\(prefix).experts.w2")
    let routed = down(hidden, prepared[3], prepared[4])
    let scattered = Functional.scatterAdd(
      count: tokenLength, countPerOutput: configuration.routedExperts,
      routed, index: prepared[2]
    ).reshaped([tokenLength, configuration.hiddenSize])
    let shared = DeepSeek4SharedFFN(
      prefix: prefix, x: x, tokenLength: tokenLength, configuration: configuration)
    return (scattered, shared)
  }
  let gateWeight = Parameter<Float>(
    .GPU(0),
    .HWC(
      configuration.expertCount, configuration.expertIntermediateSize,
      configuration.hiddenSize),
    name: "\(prefix).experts.streaming_gate"
  )()
  let upWeight = Parameter<Float>(
    .GPU(0),
    .HWC(
      configuration.expertCount, configuration.expertIntermediateSize,
      configuration.hiddenSize),
    name: "\(prefix).experts.streaming_up"
  )()
  let downWeight = Parameter<Float>(
    .GPU(0),
    .HWC(
      configuration.expertCount, configuration.hiddenSize,
      configuration.expertIntermediateSize),
    name: "\(prefix).experts.w2"
  )()
  let resident = MoEWeightStreaming(
    residentSlots: configuration.expertResidentSlots[layerIndex],
    routingWidth: configuration.routedExperts,
    name: "\(prefix).expert_streaming"
  )([
    prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]),
    gateWeight, upWeight, downWeight,
  ])
  // Keep these Extract IOs alive through graph materialization. Dependencies are
  // stored as raw model IOs by ccv, while the consumers retain the Swift wrappers.
  let residentIndices = resident[0]
  let residentCounts = resident[1]
  let residentScales = resident[2]
  let residentGate = resident[3]
  let residentUp = resident[4]
  let residentDown = resident[5]
  let shared = DeepSeek4SharedFFN(
    prefix: prefix, x: x, tokenLength: tokenLength,
    dependencies: [residentGate, residentUp, residentDown], configuration: configuration)
  let hidden = SegmentedSwiGLU(
    segments: 0, count: configuration.expertIntermediateSize, clamp: 10, functional: true,
    name: "\(prefix).experts.functional"
  )([prepared[0], residentIndices, residentCounts, residentGate, residentUp, residentScales])
  hidden.add(dependencies: [shared])
  let routed = SegmentedDense(
    segments: 0, count: configuration.hiddenSize, noBias: true, functional: true,
    name: "\(prefix).experts.w2.functional"
  )([hidden, residentIndices, residentCounts, residentDown])
  let scattered = Functional.scatterAdd(
    count: tokenLength, countPerOutput: configuration.routedExperts,
    routed, index: prepared[2]
  ).reshaped([tokenLength, configuration.hiddenSize])
  return (scattered, shared)
}

private func DeepSeek4Embedding<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokens: Model.IO, tokenLength: Int,
  configuration: DeepSeek4ModelConfiguration
) -> Model.IO {
  let embed = Embedding(
    FloatType.self, vocabularySize: configuration.vocabularySize,
    embeddingSize: configuration.hiddenSize,
    name: "embed")
  let tokenEmbedding = embed(tokens).reshaped([tokenLength, 1, configuration.hiddenSize])
  let hcBroadcast = Parameter<FloatType>(
    .GPU(0), .HWC(1, configuration.hcCount, 1),
    name: "embed.hc_broadcast")
  return (tokenEmbedding .* hcBroadcast.reshaped([1, configuration.hcCount, 1]))
    .reshaped([tokenLength, configuration.hcCount, configuration.hiddenSize])
}

private func DeepSeek4Layer<FloatType: TensorNumeric>(
  prefix: String, layerIndex: Int, tokens: Model.IO, residualHC: Model.IO,
  attentionInputs: DeepSeek4LayerAttentionInputs,
  tokenLength: Int, cachedTokenLength: Int, cachedRawTokenLength: Int,
  configuration: DeepSeek4ModelConfiguration, of dataType: FloatType.Type
) -> Model.IO {
  let routerKind = configuration.routerKind(layerIndex: layerIndex)
  let hc = configuration.hcCount
  let hcDim = hc * configuration.hiddenSize
  let mixDim = configuration.hcMixDim

  let attnHC = Dense(count: mixDim, noBias: true, name: "\(prefix).hc_attn_fn")
  let attnScale = Parameter<Float>(.GPU(0), .C(3), name: "\(prefix).hc_attn_scale")
  let attnBase = Parameter<Float>(.GPU(0), .C(mixDim), name: "\(prefix).hc_attn_base")

  let ffnHC = Dense(count: mixDim, noBias: true, name: "\(prefix).hc_ffn_fn")
  let ffnScale = Parameter<Float>(.GPU(0), .C(3), name: "\(prefix).hc_ffn_scale")
  let ffnBase = Parameter<Float>(.GPU(0), .C(mixDim), name: "\(prefix).hc_ffn_base")

  let attnFlat = RMSNorm(epsilon: 1.0e-6, axis: [1], elementwiseAffine: false)(
    residualHC.reshaped([tokenLength, hcDim])
  )
  let attnMix = attnHC(attnFlat)
  let attnParts = DeepSeek4HCSplitWeightedSum(
    mix: attnMix, scale: attnScale, base: attnBase, residualHC: residualHC,
    tokenLength: tokenLength, configuration: configuration)
  let attnNorm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).attn_norm")(
    attnParts.weighted
  )
  let attnBranch = attnNorm.to(FloatType.dataType)

  let attention: Model
  let inputs: [ModelIOConvertible]
  switch attentionInputs {
  case .swa(let attentionInputs):
    attention = DeepSeek4SWAttention(
      prefix: "\(prefix).attn", tokenLength: tokenLength,
      cachedRawTokenLength: cachedRawTokenLength,
      configuration: configuration, of: dataType)
    inputs = [attnBranch, attentionInputs.rotary, attentionInputs.rawKeyValue]
  case .compressedSparse(let attentionInputs):
    attention = DeepSeek4CompressedSparseAttention(
      prefix: "\(prefix).attn", tokenLength: tokenLength,
      cachedTokenLength: cachedTokenLength, cachedRawTokenLength: cachedRawTokenLength,
      cachedCompressorInputLength: attentionInputs.cachedCompressorInputLength,
      compressionRatio: attentionInputs.compressionRatio,
      configuration: configuration, of: dataType)
    inputs = [
      attnBranch, attentionInputs.rotary, attentionInputs.indexerRotary,
      attentionInputs.compressorRotary, attentionInputs.indexerCompressorRotary,
      attentionInputs.compressorZeroPad, attentionInputs.compressorNegInfPad,
      attentionInputs.indexerCompressorZeroPad,
      attentionInputs.indexerCompressorNegInfPad, attentionInputs.rawKeyValue,
      attentionInputs.compressedKeyValue, attentionInputs.compressorInputCache,
      attentionInputs.indexerKeyValue,
    ]
  case .highlyCompressed(let attentionInputs):
    attention = DeepSeek4HighlyCompressedAttention(
      prefix: "\(prefix).attn", tokenLength: tokenLength,
      cachedTokenLength: cachedTokenLength, cachedRawTokenLength: cachedRawTokenLength,
      cachedCompressorInputLength: attentionInputs.cachedCompressorInputLength,
      compressionRatio: attentionInputs.compressionRatio,
      configuration: configuration, of: dataType)
    inputs = [
      attnBranch, attentionInputs.rotary, attentionInputs.compressorRotary,
      attentionInputs.causalCompressedIndices, attentionInputs.rawKeyValue,
      attentionInputs.compressedKeyValue, attentionInputs.compressorInputCache,
    ]
  }
  let attnOutput = attention(inputs)
  let afterAttn = DeepSeek4HCExpand(
    block: attnOutput, residualHC: residualHC, post: attnParts.post, comb: attnParts.comb,
    tokenLength: tokenLength, configuration: configuration)

  let ffnFlat = RMSNorm(epsilon: 1.0e-6, axis: [1], elementwiseAffine: false)(
    afterAttn.reshaped([tokenLength, hcDim])
  )
  let ffnMix = ffnHC(ffnFlat)
  let ffnParts = DeepSeek4HCSplitWeightedSum(
    mix: ffnMix, scale: ffnScale, base: ffnBase, residualHC: afterAttn,
    tokenLength: tokenLength, configuration: configuration)
  let ffnNorm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "\(prefix).ffn_norm")(
    ffnParts.weighted
  )
  let ffnBranch = ffnNorm

  let moe = DeepSeek4RoutedMoE(
    prefix: "\(prefix).ffn", x: ffnBranch, routerInput: ffnNorm.to(FloatType.dataType),
    tokens: tokens, layerIndex: layerIndex, tokenLength: tokenLength, routerKind: routerKind,
    configuration: configuration)
  let ffnBlock = moe.routed + moe.shared
  return DeepSeek4HCExpand(
    block: ffnBlock, residualHC: afterAttn, post: ffnParts.post,
    comb: ffnParts.comb, tokenLength: tokenLength, configuration: configuration)
}

private func DeepSeek4OutputHead<FloatType: TensorNumeric>(
  x: Model.IO, tokenLength: Int, configuration: DeepSeek4ModelConfiguration,
  of dataType: FloatType.Type
) -> Model.IO {
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let hcDim = hc * hidden
  let outputFlatInput = x.reshaped([tokenLength, hcDim]).reshaped(
    [1, hcDim], offset: [tokenLength - 1, 0], strides: [hcDim, 1]
  ).copied()
  let outputInput = outputFlatInput.reshaped([1, hc, hidden])
  let hcFn = Dense(count: hc, noBias: true, name: "hc_head_fn")
  let hcScale = Parameter<Float>(.GPU(0), .WC(1, 1), name: "hc_head_scale")
  let hcBase = Parameter<Float>(.GPU(0), .WC(1, hc), name: "hc_head_base")
  let flat = RMSNorm(epsilon: 1.0e-6, axis: [1], elementwiseAffine: false)(outputFlatInput)
  let mix = hcFn(flat)
  let weights = (mix .* hcScale + hcBase).sigmoid() + 1.0e-6
  let hiddenState = (outputInput .* weights.reshaped([1, hc, 1]))
    .reduced(.sum, axis: [1])
    .reshaped([1, hidden])
  let norm = RMSNorm(epsilon: 1.0e-6, axis: [1], name: "norm")
  let head = Dense(count: configuration.vocabularySize, noBias: true, name: "head")
  return head(norm(hiddenState).to(FloatType.dataType))
}

private func DeepSeek4Prefix<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int,
  cachedRawTokenLength: Int, cachedCompressorInputLengths: [Int: Int],
  configuration: DeepSeek4ModelConfiguration
) -> (inputs: [Input], hidden: Model.IO) {
  precondition(tokenLength > 0)
  precondition(cachedRawTokenLength >= 0 && cachedRawTokenLength <= cachedTokenLength)
  let tokens = Input()
  let rawRotary = Input()
  let usesCompressedRotary = configuration.layerAttentionKinds.contains {
    if case .raw = $0 { return false }
    return true
  }
  let usesIndexerRotary = !configuration.indexerCompressionRatios.isEmpty
  let compressedRotary = usesCompressedRotary ? Input() : nil
  let indexerRotary = usesIndexerRotary ? Input() : nil
  var inputs: [Input] = [tokens, rawRotary]
  if let compressedRotary = compressedRotary {
    inputs.append(compressedRotary)
  }
  if let indexerRotary = indexerRotary {
    inputs.append(indexerRotary)
  }
  let compressorZeroPad: Input?
  let compressorNegInfPad: Input?
  if configuration.compressionRatios.contains(4) {
    let zeroPad = Input()
    let negInfPad = Input()
    compressorZeroPad = zeroPad
    compressorNegInfPad = negInfPad
    inputs.append(contentsOf: [zeroPad, negInfPad])
  } else {
    compressorZeroPad = nil
    compressorNegInfPad = nil
  }
  let indexerCompressorZeroPad: Input?
  let indexerCompressorNegInfPad: Input?
  if configuration.indexerCompressionRatios.contains(4) {
    let zeroPad = Input()
    let negInfPad = Input()
    indexerCompressorZeroPad = zeroPad
    indexerCompressorNegInfPad = negInfPad
    inputs.append(contentsOf: [zeroPad, negInfPad])
  } else {
    indexerCompressorZeroPad = nil
    indexerCompressorNegInfPad = nil
  }
  var compressorRotaries = [Int: Input]()
  for compressionRatio in configuration.compressionRatios {
    let compressorRotary = Input()
    compressorRotaries[compressionRatio] = compressorRotary
    inputs.append(compressorRotary)
  }
  var indexerCompressorRotaries = [Int: Input]()
  for compressionRatio in configuration.indexerCompressionRatios {
    let compressorRotary = Input()
    indexerCompressorRotaries[compressionRatio] = compressorRotary
    inputs.append(compressorRotary)
  }
  var out = DeepSeek4Embedding(
    dataType, tokens: tokens, tokenLength: tokenLength, configuration: configuration
  ).to(.Float32)
  var causalCompressedIndices = [Int: Input]()

  for layerIndex in 0..<configuration.layers {
    let prefix = "layers.\(layerIndex)"
    let attentionInputs: DeepSeek4LayerAttentionInputs
    switch configuration.attentionKind(layerIndex: layerIndex) {
    case .raw:
      let rawKeyValue = Input()
      inputs.append(rawKeyValue)
      attentionInputs = .swa(
        DeepSeek4SWAttentionInputs(rotary: rawRotary, rawKeyValue: rawKeyValue))
    case .compressed(let compressionRatio):
      guard let compressedRotary,
        let compressorRotary = compressorRotaries[compressionRatio]
      else {
        preconditionFailure("Highly compressed attention requires rotary inputs.")
      }
      let causalCompressedIndicesInput: Input
      if let existing = causalCompressedIndices[compressionRatio] {
        causalCompressedIndicesInput = existing
      } else {
        let input = Input()
        causalCompressedIndices[compressionRatio] = input
        inputs.append(input)
        causalCompressedIndicesInput = input
      }
      let rawKeyValue = Input()
      let compressedKeyValue = Input()
      let compressorInputCache = Input()
      inputs.append(contentsOf: [
        rawKeyValue, compressedKeyValue, compressorInputCache,
      ])
      attentionInputs = .highlyCompressed(
        DeepSeek4HighlyCompressedAttentionInputs(
          rotary: compressedRotary, compressorRotary: compressorRotary,
          causalCompressedIndices: causalCompressedIndicesInput,
          rawKeyValue: rawKeyValue, compressedKeyValue: compressedKeyValue,
          compressorInputCache: compressorInputCache,
          compressionRatio: compressionRatio,
          cachedCompressorInputLength: cachedCompressorInputLengths[compressionRatio]
            ?? cachedTokenLength))
    case .indexed(let compressionRatio):
      guard let compressedRotary, let indexerRotary,
        let compressorRotary = compressorRotaries[compressionRatio],
        let indexerCompressorRotary = indexerCompressorRotaries[compressionRatio],
        let compressorZeroPad, let compressorNegInfPad,
        let indexerCompressorZeroPad, let indexerCompressorNegInfPad
      else {
        preconditionFailure("Compressed sparse attention requires rotary and padding inputs.")
      }
      let rawKeyValue = Input()
      let compressedKeyValue = Input()
      let compressorInputCache = Input()
      let indexerKeyValue = Input()
      inputs.append(contentsOf: [
        rawKeyValue, compressedKeyValue, compressorInputCache, indexerKeyValue,
      ])
      attentionInputs = .compressedSparse(
        DeepSeek4CompressedSparseAttentionInputs(
          rotary: compressedRotary, indexerRotary: indexerRotary,
          compressorRotary: compressorRotary,
          indexerCompressorRotary: indexerCompressorRotary,
          compressorZeroPad: compressorZeroPad, compressorNegInfPad: compressorNegInfPad,
          indexerCompressorZeroPad: indexerCompressorZeroPad,
          indexerCompressorNegInfPad: indexerCompressorNegInfPad,
          rawKeyValue: rawKeyValue, compressedKeyValue: compressedKeyValue,
          compressorInputCache: compressorInputCache, indexerKeyValue: indexerKeyValue,
          compressionRatio: compressionRatio,
          cachedCompressorInputLength: cachedCompressorInputLengths[compressionRatio]
            ?? cachedTokenLength))
    }
    let layer = DeepSeek4Layer(
      prefix: prefix, layerIndex: layerIndex, tokens: tokens, residualHC: out,
      attentionInputs: attentionInputs,
      tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      cachedRawTokenLength: cachedRawTokenLength,
      configuration: configuration, of: dataType)
    out = layer
  }

  return (inputs, out)
}

public func DeepSeek4PrefixHiddenState<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  cachedRawTokenLength: Int = 0, cachedCompressorInputLengths: [Int: Int] = [:],
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash
) -> Model {
  let prefix = DeepSeek4Prefix(
    dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
    cachedRawTokenLength: cachedRawTokenLength,
    cachedCompressorInputLengths: cachedCompressorInputLengths,
    configuration: configuration)
  return Model(prefix.inputs, [prefix.hidden])
}

public func DeepSeek4CausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int,
  cachedRawTokenLength: Int,
  cachedCompressorInputLengths: [Int: Int],
  configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash
) -> Model {
  let prefix = DeepSeek4Prefix(
    dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
    cachedRawTokenLength: cachedRawTokenLength,
    cachedCompressorInputLengths: cachedCompressorInputLengths,
    configuration: configuration)
  let output = DeepSeek4OutputHead(
    x: prefix.hidden, tokenLength: tokenLength, configuration: configuration, of: dataType
  )
  return Model(prefix.inputs, [output])
}
