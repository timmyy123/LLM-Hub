import Foundation
import NNC

public struct DeepSeek4_1ModelConfiguration: Codable, Sendable {
  public var vocabularySize = 129_280
  public var hiddenSize = 5_120
  public var layers = 40
  public var hcCount = 4
  public var hcSinkhornIterations = 20
  public var hcEpsilon: Float = 1e-6
  public var normEpsilon: Float = 1e-20
  public var attentionHeads = 64
  public var attentionHeadDim = 512
  public var rotaryDim = 64
  public var rawWindow = 128
  public var expertCount = 384
  public var routedExperts = 6
  public var expertIntermediateSize = 2_304
  public var sharedIntermediateSize = 2_304
  public var expertResidentSlots = Array(repeating: 384, count: 40)
  public var attentionOutputGroups = 8
  public var attentionLowRank = 1_024
  public var queryLowRank = 1_280
  public var indexerHeads = 32
  public var indexerHeadDim = 128
  public var indexerTopK = 512
  public var ropeTheta: Double = 10_000
  public var compressedRopeTheta: Double = 160_000
  public var ropeScaleFactor: Double = 16
  public var ropeOriginalContext = 65_536
  public var ropeYarnBetaFast: Double = 32
  public var ropeYarnBetaSlow: Double = 1
  public var kvSourceLayers = [2, 8, 14, 20]
  public var indexSourceLayers = [2, 8, 14, 20, 24, 28, 32, 36]
  public var compressionRatios =
    [0, 0] + Array(repeating: 2, count: 18) + Array(repeating: 1, count: 20)
  public var candidateSourceLayer = 20
  public var candidateBlockSize = 8
  public var candidateTopKBlocks = 2_048
  public var engram = DeepSeek4_1EngramConfiguration.deepSeekV4_1Flash

  /// Includes reserved token spellings so Engram sees the complete vocabulary in ID order.
  public static let specialTokens: [String: Int32] = {
    var tokens: [String: Int32] = [
      "<｜begin▁of▁sentence｜>": 0,
      "<｜end▁of▁sentence｜>": 1,
      "<｜▁pad▁｜>": 2,
      "<｜System｜>": 128_799,
      "<｜fim▁hole｜>": 128_800,
      "<｜fim▁begin｜>": 128_801,
      "<｜fim▁end｜>": 128_802,
      "<｜User｜>": 128_803,
      "<｜Assistant｜>": 128_804,
      "<|EOT|>": 128_805,
      "<｜tool▁calls▁begin｜>": 128_806,
      "<｜tool▁calls▁end｜>": 128_807,
      "<｜tool▁call▁begin｜>": 128_808,
      "<｜tool▁call▁end｜>": 128_809,
      "<｜tool▁outputs▁begin｜>": 128_810,
      "<｜tool▁outputs▁end｜>": 128_811,
      "<｜tool▁output▁begin｜>": 128_812,
      "<｜tool▁output▁end｜>": 128_813,
      "<｜tool▁sep｜>": 128_814,
      "<｜begin▁of▁repo▁name｜>": 128_815,
      "<｜end▁of▁repo▁name｜>": 128_816,
      "<｜begin▁of▁file▁name｜>": 128_817,
      "<｜end▁of▁file▁name｜>": 128_818,
      "<｜begin▁of▁file｜>": 128_819,
      "<｜end▁of▁file｜>": 128_820,
      "<think>": 128_821,
      "</think>": 128_822,
      "<｜place▁holder▁for▁copy｜>": 128_823,
      "<｜place▁holder▁for▁pointer▁replace｜>": 128_824,
      "｜DSML｜": 128_825,
      "<｜begin▁sys｜>": 128_826,
      "<｜end▁sys｜>": 128_827,
      "<｜latest_reminder｜>": 128_828,
      "<｜action｜>": 128_829,
      "<｜query｜>": 128_830,
      "<｜authority｜>": 128_831,
      "<｜domain｜>": 128_832,
      "<｜task｜>": 128_833,
      "<｜political｜>": 128_834,
      "<｜entity｜>": 128_835,
      "<｜title｜>": 128_836,
      "<｜safety｜>": 128_837,
      "<｜answer｜>": 128_838,
      "<｜search｜>": 128_839,
      "<dsml:": 128_840,
      "</dsml:": 128_841,
      "<｜search▁begin｜>": 128_842,
      "<｜search▁end｜>": 128_843,
      "<｜extracted_url｜>": 128_844,
      "<｜read_url｜>": 128_845,
      "<｜end_of_query｜>": 128_846,
      "<｜rl_image_pad｜>": 129_262,
      "<｜rl_image_start｜>": 129_263,
      "<｜deepseek_image｜>": 129_264,
      "<｜/polygon｜>": 129_271,
      "<｜polygon｜>": 129_272,
      "<｜/point｜>": 129_273,
      "<｜point｜>": 129_274,
      "<｜/box｜>": 129_275,
      "<｜box｜>": 129_276,
      "<｜/ref｜>": 129_277,
      "<｜ref｜>": 129_278,
    ]
    for i in 0..<799 {
      tokens["<｜place▁holder▁no▁\(i)｜>"] = Int32(128_000 + i)
    }
    for i in 21...435 {
      tokens[String(format: "<|place_holder_mm_span_%04d|>", i)] = Int32(128_826 + i)
    }
    for i in 436...441 {
      tokens[String(format: "<|place_holder_mm_span_%04d|>", i)] = Int32(128_829 + i)
    }
    tokens["<|place_holder_mm_span_0442|>"] = 129_279
    return tokens
  }()

  public init() {}

  public static let deepSeekV4_1Flash = DeepSeek4_1ModelConfiguration()
  public var hcMixDim: Int { (2 + hcCount) * hcCount }
  public var attentionOutputLowDim: Int { attentionOutputGroups * attentionLowRank }

  public func kvSource(layerIndex: Int) -> Int? {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return kvSourceLayers.last { $0 <= layerIndex }
  }

  public func indexSource(layerIndex: Int) -> Int? {
    precondition(layerIndex >= 0 && layerIndex < layers)
    return indexSourceLayers.last { $0 <= layerIndex }
  }
}

// Follow V4 Flash: FP32 residuals, mHC and FFNs; FP16 attention and caches.

/// Adjacent-pair complex RoPE. Compressed-attention layers apply YaRN to both
/// raw and global KV; layers 0 and 1 use ordinary RoPE with theta 10,000.
public func DeepSeek4_1RotaryEmbedding<FloatType: TensorNumeric>(
  sequenceLength: Int, cachedTokenLength: Int = 0, positionStride: Int = 1,
  compressed: Bool, headDim: Int = 512,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash,
  of dataType: FloatType.Type = Float.self
) -> Tensor<FloatType> {
  precondition(sequenceLength > 0 && cachedTokenLength >= 0 && positionStride > 0)
  let rotaryDim = configuration.rotaryDim
  precondition(rotaryDim > 0 && rotaryDim.isMultiple(of: 2) && headDim >= rotaryDim)
  precondition((headDim - rotaryDim).isMultiple(of: 2))
  let base = compressed ? configuration.compressedRopeTheta : configuration.ropeTheta
  let low = max(
    floor(
      Double(rotaryDim)
        * log(
          Double(configuration.ropeOriginalContext) / (configuration.ropeYarnBetaFast * 2 * .pi))
        / (2 * log(base))), 0)
  let high = min(
    ceil(
      Double(rotaryDim)
        * log(
          Double(configuration.ropeOriginalContext) / (configuration.ropeYarnBetaSlow * 2 * .pi))
        / (2 * log(base))), Double(rotaryDim - 1))
  var frequencies = [Float]()
  for i in 0..<(rotaryDim / 2) {
    var frequency = 1 / pow(Float(base), Float(2 * i) / Float(rotaryDim))
    if compressed {
      let ramp = min(max((Float(i) - Float(low)) / Float(max(high - low, 1e-3)), 0), 1)
      let smooth = 1 - ramp
      frequency =
        frequency / Float(configuration.ropeScaleFactor) * (1 - smooth) + frequency * smooth
    }
    frequencies.append(frequency)
  }
  var values = Array(repeating: Float(0), count: sequenceLength * headDim)
  for row in 0..<sequenceLength {
    for pair in 0..<((headDim - rotaryDim) / 2) {
      values[row * headDim + pair * 2] = 1
    }
    for i in 0..<(rotaryDim / 2) {
      let angle = Float(cachedTokenLength + row * positionStride) * frequencies[i]
      values[row * headDim + headDim - rotaryDim + 2 * i] = cos(angle)
      values[row * headDim + headDim - rotaryDim + 2 * i + 1] = sin(angle)
    }
  }
  return Tensor<FloatType>(from: Tensor<Float>(values, .CPU, .NHWC(1, sequenceLength, 1, headDim)))
}

/// The Engram table is fetched separately. Inputs are the HC residual and the
/// concatenated, dequantized embedding rows for these tokens, in hash-column order.
public func DeepSeek4_1Engram<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, residual: Model.IO, embeddings: Model.IO,
  tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model.IO {
  precondition(tokenLength > 0)
  let hidden = configuration.hiddenSize
  let hc = configuration.hcCount
  let embedding = embeddings.reshaped([tokenLength, configuration.engram.embeddingSize])
    .to(FloatType.dataType)
  let keys = Dense(count: hc * hidden, noBias: true, name: "\(prefix).wk")(embedding)
    .to(.Float32).reshaped([tokenLength, hc, hidden])
  let value = Dense(count: hidden, noBias: true, name: "\(prefix).wv")(embedding)
    .to(.Float32).reshaped([tokenLength, 1, hidden])
  let qWeight = Parameter<Float>(.GPU(0), .WC(hc, hidden), name: "\(prefix).q_weight")
  let kWeight = Parameter<Float>(.GPU(0), .WC(hc, hidden), name: "\(prefix).k_weight")
  let q = RMSNorm(epsilon: configuration.normEpsilon, axis: [2], elementwiseAffine: false)(residual)
  let k = RMSNorm(epsilon: configuration.normEpsilon, axis: [2], elementwiseAffine: false)(keys)
  let weights = (qWeight .* kWeight).reshaped([1, hc, hidden])
  let dot = (q .* k .* weights).reduced(.sum, axis: [2]) * (1 / Float(hidden).squareRoot())
  let gate = SignedSquareRoot(minimumMagnitude: 1e-6)(dot).sigmoid()
  return residual + gate .* value
}

/// Inputs: FP32 HC residual and incoming pre-mix. Outputs: next pre-mix, post-mix,
/// combination matrix, and the residual collapsed with the *incoming* pre-mix.
public func DeepSeek4_1HCMix(
  prefix: String, residual: Model.IO, incomingPre: Model.IO, tokenLength: Int,
  mixTokenLength: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> (Model.IO, Model.IO, Model.IO, Model.IO) {
  precondition(tokenLength >= 0)
  let hc = configuration.hcCount
  let hidden = configuration.hiddenSize
  let mixTokenLength = mixTokenLength ?? tokenLength
  precondition((0...tokenLength).contains(mixTokenLength))
  let flat = residual.reshaped(
    mixTokenLength > 0 ? [mixTokenLength, hc * hidden] : [0],
    offset: [tokenLength - mixTokenLength, 0], strides: [hc * hidden, 1])
  let normalized = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], elementwiseAffine: false)(flat)
  let mix = Dense(count: configuration.hcMixDim, noBias: true, name: "\(prefix)_fn")(
    normalized)
  let scale = Parameter<Float>(.GPU(0), .C(3), name: "\(prefix)_scale")
  let base = Parameter<Float>(.GPU(0), .C(configuration.hcMixDim), name: "\(prefix)_base")
  let parts = HyperConnection(
    count: hc, sinkhornIterations: configuration.hcSinkhornIterations,
    epsilon: configuration.hcEpsilon, operation: .split)(mix, scale, base)
  let collapsed = (residual .* incomingPre.reshaped([tokenLength, hc, 1]))
    .reduced(.sum, axis: [1]).reshaped([tokenLength, hidden])
  return (parts[0], parts[1], parts[2], collapsed)
}

/// Inputs: normalized sublayer input and RoPE. Outputs: rotated query, raw
/// KV, and the normalized low-rank query used by index-source layers.
public func DeepSeek4_1AttentionProjection(
  prefix: String, x: Model.IO, rotary: Model.IO, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> (Model.IO, Model.IO, Model.IO) {
  let dim = configuration.attentionHeadDim
  let heads = configuration.attentionHeads
  let low = Dense(count: configuration.queryLowRank, noBias: true, name: "\(prefix).wq_a")(
    x
  )
  let rank = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).q_norm")(low)
  let query = Dense(count: heads * dim, noBias: true, name: "\(prefix).wq_b")(
    rank
  )
  .reshaped([1, tokenLength, heads, dim])
  let rotatedQuery = Functional.cmul(left: query, right: rotary.reshaped([1, tokenLength, 1, dim]))
  let raw = Dense(count: dim, noBias: true, name: "\(prefix).wkv")(x)
  let normalized = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).kv_norm")(raw)
    .reshaped([1, tokenLength, 1, dim])
  let rotatedKV = Functional.cmul(
    left: normalized, right: rotary.reshaped([1, tokenLength, 1, dim])
  )
  return (rotatedQuery, rotatedKV, rank)
}

/// Inputs: attention head outputs and RoPE. The grouped output projection is
/// stored as [groups, low rank, heads per group * head dimension], as in DeepSeek4.
public func DeepSeek4_1AttentionOutput<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, heads: Model.IO, rotary: Model.IO, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model.IO {
  let dim = configuration.attentionHeadDim
  let groups = configuration.attentionOutputGroups
  let groupDim = configuration.attentionHeads / groups * dim
  let unrotated = Functional.cmul(
    left: heads.reshaped([tokenLength, configuration.attentionHeads, dim]),
    right: rotary.reshaped([tokenLength, 1, dim]), conjugate: true
  )
  .reshaped([tokenLength, groups, 1, groupDim])
  let woA = Parameter<FloatType>(
    .GPU(0), .HWC(groups, configuration.attentionLowRank, groupDim), name: "\(prefix).wo_a")
  let low = Matmul(transposeB: (1, 2))(unrotated, woA)
    .reshaped([tokenLength, configuration.attentionOutputLowDim])
  let output = Dense(count: configuration.hiddenSize, noBias: true, name: "\(prefix).wo_b")(
    low
  )
  return output
}

/// Projects complete groups to pre-RoPE global latents. Retain an incomplete
/// group's input outside this model and prepend it on the next invocation.
public func DeepSeek4_1Compressor(
  prefix: String, x: Model.IO, tokenLength: Int, compressionRatio: Int,
  dependencies: [Model.IO] = [],
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model.IO {
  precondition(compressionRatio == 1 || compressionRatio == 2)
  precondition(tokenLength >= 0 && tokenLength.isMultiple(of: compressionRatio))
  let headDim = configuration.attentionHeadDim
  let rows = tokenLength / compressionRatio
  let projected = Dense(count: headDim, noBias: true, name: "\(prefix).wkv")(x)
  if !dependencies.isEmpty {
    projected.add(dependencies: dependencies)
  }
  let pooled: Model.IO
  if compressionRatio == 2 {
    let scores = Dense(count: headDim, noBias: true, name: "\(prefix).wgate")(x)
    if !dependencies.isEmpty {
      scores.add(dependencies: dependencies)
    }
    let weights = scores.reshaped([rows, compressionRatio, headDim]).transposed(1, 2).contiguous()
      .reshaped([rows * headDim, compressionRatio]).softmax()
      .reshaped([rows, headDim, compressionRatio])
    let values = projected.reshaped([rows, compressionRatio, headDim]).transposed(1, 2).contiguous()
    pooled = (weights .* values).reduced(.sum, axis: [2]).reshaped([rows, headDim])
  } else {
    pooled = projected
  }
  let output = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).norm")(
    pooled
  )
  return output
}

/// Inputs: normalized sublayer input, low-rank query, and indexer RoPE.
/// Outputs: rotated query heads and scaled head weights.
public func DeepSeek4_1IndexerQuery(
  prefix: String, x: Model.IO, rank: Model.IO, rotary: Model.IO, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> (Model.IO, Model.IO) {
  let heads = configuration.indexerHeads
  let dim = configuration.indexerHeadDim
  let query = Dense(count: heads * dim, noBias: true, name: "\(prefix).wq_b")(
    rank
  )
  .reshaped([tokenLength, heads, dim])
  let rotated = Functional.cmul(left: query, right: rotary.reshaped([tokenLength, 1, dim]))
  let weights =
    Dense(count: heads, noBias: true, name: "\(prefix).weights_proj")(x)
    * (1 / Float(dim * heads).squareRoot())
  return (rotated, weights)
}

/// Inputs: compressed latent before RoPE and the RoPE at each group's first token.
public func DeepSeek4_1IndexerKey(
  prefix: String, latent: Model.IO, rotary: Model.IO, rowCount: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model.IO {
  precondition(rowCount >= 0)
  let dim = configuration.indexerHeadDim
  let key = Dense(count: dim, noBias: true, name: "\(prefix).wk")(latent)
  let normalized = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).k_norm")(
    key
  )
  let rotated = Functional.cmul(left: normalized, right: rotary.reshaped([rowCount, dim]))
  return rotated
}

/// Inputs: index query [tokens, heads, dim], shared index keys [rows, dim],
/// head weights [tokens, heads], and (after the candidate source) its int32 block IDs.
/// Returns global row ids in position order with -1 for unreachable positions.
/// The candidate source returns [tokens, candidateTopKBlocks] IDs for later indexers;
/// its own selection still scores all reachable rows. Empty key tensors keep
/// the same graph when no compression group has completed.
public func DeepSeek4_1IndexerSelection(
  query: Model.IO, keys: Model.IO, weights: Model.IO, candidates: Model.IO? = nil,
  tokenLength: Int, keyLength: Int, cachedTokenLength: Int, layerIndex: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model.IO {
  precondition(tokenLength >= 0 && keyLength >= 0 && cachedTokenLength >= 0)
  precondition(configuration.indexSourceLayers.contains(layerIndex))
  let ratio = configuration.compressionRatios[layerIndex]
  precondition(ratio > 0 && keyLength == (cachedTokenLength + tokenLength) / ratio)
  precondition((candidates != nil) == (layerIndex > configuration.candidateSourceLayer))
  let isSource = layerIndex == configuration.candidateSourceLayer
  let candidatePool: ScaledDotProductArgPartition.CandidatePool?
  if isSource {
    candidatePool = .produce(
      size: configuration.candidateBlockSize, count: configuration.candidateTopKBlocks)
  } else if candidates != nil {
    candidatePool = .consume(
      size: configuration.candidateBlockSize, count: configuration.candidateTopKBlocks)
  } else {
    candidatePool = nil
  }
  let selection = ScaledDotProductArgPartition(
    kth: max(1, min(configuration.indexerTopK, keyLength)), scale: 1, isCausal: true,
    compressionRatio: ratio, queryOffset: cachedTokenLength,
    sortIndices: true, candidatePool: candidatePool)
  var arguments = [
    query.reshaped([tokenLength, configuration.indexerHeads, configuration.indexerHeadDim]),
    keys.reshaped([keyLength, configuration.indexerHeadDim]),
    weights.reshaped([tokenLength, configuration.indexerHeads]),
  ]
  if let candidates = candidates {
    arguments.append(candidates)
  }
  return selection(arguments)
}

/// Sliding-window attention. Inputs: hidden, rotary, raw KV. Cache buffers are
/// caller-owned; raw KV prepends the last min(window, cachedTokenLength) rows.
public func DeepSeek4_1SWAttention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  cachedRawTokenLength: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength >= 0 && cachedTokenLength >= 0)
  precondition(configuration.compressionRatios[layerIndex] == 0)
  let prefix = "layers.\(layerIndex).attn"
  let dim = configuration.attentionHeadDim
  let cachedRaw = cachedRawTokenLength ?? min(configuration.rawWindow, cachedTokenLength)
  let rawRows = cachedRaw + tokenLength
  let x = Input()
  let rotary = Input()
  let rawCache = Input()
  let (query, raw, _) = DeepSeek4_1AttentionProjection(
    prefix: prefix, x: x, rotary: rotary, tokenLength: tokenLength, configuration: configuration)
  let rawWrite = raw.moved(
    to: rawCache.reshaped(
      tokenLength > 0 ? [1, tokenLength, 1, dim] : [0], offset: [0, cachedRaw, 0, 0],
      strides: [rawRows * dim, dim, dim, 1]), flags: [.disableOpt])
  let sinks = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, configuration.attentionHeads, 1), name: "\(prefix).attn_sink")
  let heads = ScaledDotProductAttention(
    scale: 1 / Float(dim).squareRoot(), isCausal: true, hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow)(query, rawCache, rawCache, sinks)
  heads.add(dependencies: [rawWrite])
  let output = DeepSeek4_1AttentionOutput(
    dataType, prefix: prefix, heads: heads, rotary: rotary, tokenLength: tokenLength,
    configuration: configuration)
  return Model([x, rotary, rawCache], [output])
}

/// Compressed attention that computes this chunk's global-row indices.
/// Raw-window and selected global KV share one softmax with the attention sink.
/// Inputs: hidden, rotary, raw KV, global KV, index KV; KV-source layers then
/// take compressor input, compressor rotary, and indexer compressor rotary;
/// followed by indexer rotary, and candidates after the candidate-source layer.
/// Outputs: hidden, indices, and candidates on the candidate-source layer.
public func DeepSeek4_1CompressedSparseAttention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  cachedRawTokenLength: Int? = nil, boundedReplayTokens: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength >= 0 && cachedTokenLength >= 0)
  precondition(configuration.indexSourceLayers.contains(layerIndex))
  let prefix = "layers.\(layerIndex).attn"
  let dim = configuration.attentionHeadDim
  let hidden = configuration.hiddenSize
  let indexDim = configuration.indexerHeadDim
  let ratio = configuration.compressionRatios[layerIndex]
  precondition(ratio == 1 || ratio == 2)
  let cachedRaw = cachedRawTokenLength ?? min(configuration.rawWindow, cachedTokenLength)
  let queryTokenLength = boundedReplayTokens ?? tokenLength
  precondition((0...tokenLength).contains(queryTokenLength))
  let queryOffset = cachedTokenLength + tokenLength - queryTokenLength
  let rawRows = cachedRaw + queryTokenLength
  let globalRows = (cachedTokenLength + tokenLength) / ratio
  let x = Input()
  let rotary = Input()
  let rawCache = Input()
  let globalCache = Input()
  let indexCache = Input()
  var inputs = [x, rotary, rawCache, globalCache, indexCache]
  let queryInput: Model.IO
  let queryRotary: Model.IO
  if layerIndex == configuration.candidateSourceLayer {
    queryInput = x.reshaped(
      queryTokenLength > 0 ? [queryTokenLength, hidden] : [0],
      offset: [tokenLength - queryTokenLength, 0], strides: [hidden, 1]
    )
    queryRotary = rotary.reshaped(
      queryTokenLength > 0 ? [queryTokenLength, dim] : [0],
      offset: [tokenLength - queryTokenLength, 0], strides: [dim, 1]
    )
  } else {
    queryInput = x
    queryRotary = rotary
  }
  let (query, raw, rank) = DeepSeek4_1AttentionProjection(
    prefix: prefix, x: queryInput, rotary: queryRotary, tokenLength: queryTokenLength,
    configuration: configuration)
  let rawWrite = raw.moved(
    to: rawCache.reshaped(
      queryTokenLength > 0 ? [1, queryTokenLength, 1, dim] : [0], offset: [0, cachedRaw, 0, 0],
      strides: [rawRows * dim, dim, dim, 1]), flags: [.disableOpt])
  var cacheWrites = [rawWrite]
  var indexWrite: Model.IO?
  if configuration.kvSourceLayers.contains(layerIndex) {
    let previousRows = cachedTokenLength / ratio
    let emittedRows = globalRows - previousRows
    let remainder = cachedTokenLength % ratio
    let compressorCache = Input()
    let compressorRotary = Input()
    let indexerCompressorRotary = Input()
    inputs.append(contentsOf: [compressorCache, compressorRotary, indexerCompressorRotary])
    let write = x.moved(
      to: compressorCache.reshaped(
        [tokenLength, hidden], offset: [remainder, 0], strides: [hidden, 1]),
      flags: [.disableOpt])
    cacheWrites.append(write)
    // Keep compressor/key projections and cache writes in every graph. Empty
    // compression groups flow through as zero-sized tensors, as in V4 Flash.
    let completeInput = compressorCache.reshaped(
      emittedRows > 0 ? [emittedRows * ratio, hidden] : [0])
    let latent = DeepSeek4_1Compressor(
      prefix: "\(prefix).compressor", x: completeInput, tokenLength: emittedRows * ratio,
      compressionRatio: ratio, dependencies: [write], configuration: configuration)
    let key = DeepSeek4_1IndexerKey(
      prefix: "\(prefix).indexer", latent: latent, rotary: indexerCompressorRotary,
      rowCount: emittedRows, configuration: configuration)
    let keyWrite = key.moved(
      to: emittedRows > 0
        ? indexCache.reshaped(
          [emittedRows, indexDim], offset: [previousRows, 0], strides: [indexDim, 1])
        : indexCache.reshaped([0]), flags: [.disableOpt])
    indexWrite = keyWrite
    cacheWrites.append(keyWrite)
    // Both projections read the original latent. Global KV rotation must not
    // overwrite the indexer's pre-RoPE input.
    let rotated = Functional.cmul(
      left: latent, right: compressorRotary.reshaped([emittedRows, dim]))
    let globalWrite = rotated.moved(
      to: emittedRows > 0
        ? globalCache.reshaped(
          [emittedRows, dim], offset: [previousRows, 0], strides: [dim, 1])
        : globalCache.reshaped([0]), flags: [.disableOpt])
    cacheWrites.append(globalWrite)
  }
  let indexerRotary = Input()
  inputs.append(indexerRotary)
  let indexQueryRotary: Model.IO
  if layerIndex == configuration.candidateSourceLayer {
    indexQueryRotary = indexerRotary.reshaped(
      queryTokenLength > 0 ? [queryTokenLength, indexDim] : [0],
      offset: [tokenLength - queryTokenLength, 0], strides: [indexDim, 1]
    )
  } else {
    indexQueryRotary = indexerRotary
  }
  let (indexQuery, weights) = DeepSeek4_1IndexerQuery(
    prefix: "\(prefix).indexer", x: queryInput, rank: rank, rotary: indexQueryRotary,
    tokenLength: queryTokenLength, configuration: configuration)
  let candidates = layerIndex > configuration.candidateSourceLayer ? Input() : nil
  if let candidates {
    inputs.append(candidates)
  }
  let selected = DeepSeek4_1IndexerSelection(
    query: indexQuery, keys: indexCache, weights: weights, candidates: candidates,
    tokenLength: queryTokenLength, keyLength: globalRows, cachedTokenLength: queryOffset,
    layerIndex: layerIndex, configuration: configuration)
  if let indexWrite { selected.add(dependencies: [indexWrite]) }
  let global = globalCache.reshaped(globalRows > 0 ? [1, globalRows, 1, dim] : [0])
  let sinks = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, configuration.attentionHeads, 1), name: "\(prefix).attn_sink")
  let heads = SparseIndexedAttention(
    scale: 1 / Float(dim).squareRoot(), isCausal: true, hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow)(
      query, rawCache, rawCache, global, global, selected[0], sinks)
  heads.add(dependencies: cacheWrites)
  let output = DeepSeek4_1AttentionOutput(
    dataType, prefix: prefix, heads: heads, rotary: queryRotary, tokenLength: queryTokenLength,
    configuration: configuration)
  var outputs = [output, selected[0]]
  if layerIndex == configuration.candidateSourceLayer {
    outputs.append(selected[1])
  }
  return Model(inputs, outputs)
}

/// Reuses global KV and the latest index source's selection for this chunk.
/// Still attends to raw-window and selected global KV with one shared softmax.
/// Inputs: hidden, rotary, raw KV, global KV, indices. Output: hidden.
public func DeepSeek4_1SharedAttention<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  cachedRawTokenLength: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  precondition(tokenLength >= 0 && cachedTokenLength >= 0)
  precondition(!configuration.indexSourceLayers.contains(layerIndex))
  precondition(!configuration.kvSourceLayers.contains(layerIndex))
  let prefix = "layers.\(layerIndex).attn"
  let dim = configuration.attentionHeadDim
  let ratio = configuration.compressionRatios[layerIndex]
  precondition(ratio == 1 || ratio == 2)
  let cachedRaw = cachedRawTokenLength ?? min(configuration.rawWindow, cachedTokenLength)
  let rawRows = cachedRaw + tokenLength
  let globalRows = (cachedTokenLength + tokenLength) / ratio
  let x = Input()
  let rotary = Input()
  let rawCache = Input()
  let globalCache = Input()
  let indices = Input()
  let (query, raw, _) = DeepSeek4_1AttentionProjection(
    prefix: prefix, x: x, rotary: rotary, tokenLength: tokenLength, configuration: configuration)
  let rawWrite = raw.moved(
    to: rawCache.reshaped(
      tokenLength > 0 ? [1, tokenLength, 1, dim] : [0], offset: [0, cachedRaw, 0, 0],
      strides: [rawRows * dim, dim, dim, 1]), flags: [.disableOpt])
  let global = globalCache.reshaped(globalRows > 0 ? [1, globalRows, 1, dim] : [0])
  let sinks = Parameter<FloatType>(
    .GPU(0), .NHWC(1, 1, configuration.attentionHeads, 1), name: "\(prefix).attn_sink")
  let heads = SparseIndexedAttention(
    scale: 1 / Float(dim).squareRoot(), isCausal: true, hasAttentionSinks: true,
    slidingWindow: configuration.rawWindow)(
      query, rawCache, rawCache, global, global, indices, sinks)
  heads.add(dependencies: [rawWrite])
  let output = DeepSeek4_1AttentionOutput(
    dataType, prefix: prefix, heads: heads, rotary: rotary, tokenLength: tokenLength,
    configuration: configuration)
  return Model([x, rotary, rawCache, globalCache, indices], [output])
}

private func DeepSeek4_1SharedFFN(
  prefix: String, x: Model.IO, dependencies: [Model.IO] = [],
  configuration: DeepSeek4_1ModelConfiguration
) -> Model.IO {
  let hidden = SwiGLU(
    count: configuration.sharedIntermediateSize, clamp: 10, name: "\(prefix).shared_experts")(x)
  if !dependencies.isEmpty {
    hidden.add(dependencies: dependencies)
  }
  return Dense(
    count: configuration.hiddenSize, noBias: true, name: "\(prefix).shared_experts.w2")(hidden)
}

/// FP32 expert evaluation and accumulation with fused SwiGLU, as in V4 Flash.
/// When resident slots are fewer than experts, load the three expert banks with
/// `.jit` and `.externalOnDemand` so MoEWeightStreaming can read their source regions.
public func DeepSeek4_1MoE<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, prefix: String, tokenLength: Int, layerIndex: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> (model: Model, bias: Model.Parameters) {
  precondition(tokenLength >= 0 && layerIndex >= 0 && layerIndex < configuration.layers)
  let x = Input()
  let logits = Dense(
    count: configuration.expertCount, noBias: true, name: "\(prefix).gate")(
      x.to(FloatType.dataType))
  let bias = Parameter<Float>(.GPU(0), .C(configuration.expertCount), name: "\(prefix).gate.bias")
  let prepared = MoERouting(
    kth: configuration.routedExperts, weightScale: 1.5, normalizationEpsilon: 1e-20,
    singleInputToken: true, name: "\(prefix).routing")(logits.to(of: x), bias, x)
  let pairs = tokenLength * configuration.routedExperts
  let shared: Model.IO
  let routed: Model.IO
  if configuration.expertResidentSlots[layerIndex] == configuration.expertCount {
    shared = DeepSeek4_1SharedFFN(prefix: prefix, x: x, configuration: configuration)
    let hidden = SegmentedSwiGLU(
      segments: configuration.expertCount, count: configuration.expertIntermediateSize,
      clamp: 10, name: "\(prefix).experts")([
        prepared[0], prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]),
      ])
    routed = SegmentedDense(
      segments: configuration.expertCount, count: configuration.hiddenSize,
      noBias: true, name: "\(prefix).experts.w2")(hidden, prepared[3], prepared[4])
  } else {
    precondition(configuration.expertResidentSlots[layerIndex] >= configuration.routedExperts)
    let gateWeight = Parameter<Float>(
      .GPU(0),
      .HWC(
        configuration.expertCount, configuration.expertIntermediateSize, configuration.hiddenSize),
      name: "\(prefix).experts.streaming_gate")()
    let upWeight = Parameter<Float>(
      .GPU(0),
      .HWC(
        configuration.expertCount, configuration.expertIntermediateSize, configuration.hiddenSize),
      name: "\(prefix).experts.streaming_up")()
    let downWeight = Parameter<Float>(
      .GPU(0),
      .HWC(
        configuration.expertCount, configuration.hiddenSize, configuration.expertIntermediateSize),
      name: "\(prefix).experts.w2")()
    let resident = MoEWeightStreaming(
      residentSlots: configuration.expertResidentSlots[layerIndex],
      routingWidth: configuration.routedExperts, name: "\(prefix).expert_streaming")([
        prepared[3], prepared[4], prepared[1].reshaped([pairs, 1]), gateWeight, upWeight,
        downWeight,
      ])
    let residentIndices = resident[0]
    let residentCounts = resident[1]
    let residentScales = resident[2]
    let residentGate = resident[3]
    let residentUp = resident[4]
    let residentDown = resident[5]
    shared = DeepSeek4_1SharedFFN(
      prefix: prefix, x: x, dependencies: [residentGate, residentUp, residentDown],
      configuration: configuration)
    let hidden = SegmentedSwiGLU(
      segments: 0, count: configuration.expertIntermediateSize, clamp: 10, functional: true,
      name: "\(prefix).experts.functional")([
        prepared[0], residentIndices, residentCounts, residentGate, residentUp, residentScales,
      ])
    hidden.add(dependencies: [shared])
    routed = SegmentedDense(
      segments: 0, count: configuration.hiddenSize, noBias: true, functional: true,
      name: "\(prefix).experts.functional.w2")([
        hidden, residentIndices, residentCounts, residentDown,
      ])
  }
  let scattered = Functional.scatterAdd(
    count: tokenLength, countPerOutput: configuration.routedExperts,
    routed, index: prepared[2]
  ).reshaped([tokenLength, configuration.hiddenSize])
  return (Model([x], [scattered + shared]), bias.parameters)
}

/// Expands text or image embeddings into the initial HC residual streams. The caller
/// supplies the reference's initial one-hot pre-mix [1, 0, 0, 0].
public func DeepSeek4_1Embedding<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> Model {
  let tokens = Input()
  let injected = Input()
  let embed = Embedding(
    dataType, vocabularySize: configuration.vocabularySize,
    embeddingSize: configuration.hiddenSize, name: "embed")
  let embedding = Concat(axis: 0)(embed(tokens), injected)
    .reshaped([1, tokenLength, 1, configuration.hiddenSize])
  return Model(
    [tokens, injected],
    [
      Upsample(.nearest, widthScale: Float(configuration.hcCount), heightScale: 1)(embedding)
        .to(.Float32)
        .reshaped([tokenLength, configuration.hcCount, configuration.hiddenSize])
    ])
}

/// A text transformer layer, including its optional Engram contribution. The
/// caller supplies the previous FFN's pre-mix and the attention cache buffers.
/// This layer's attention pre-mix collapses its FFN input; its FFN pre-mix is
/// returned for the next layer (or the final output head).
/// outputTokenLength keeps the input attention/cache updates but evaluates the
/// FFN and returns residual/pre-mix for only that many trailing query tokens.
/// Inputs: residual, incoming pre-mix, optional Engram embeddings, then the
/// attention variant's inputs excluding hidden. Outputs: residual, next pre-mix,
/// followed by indices and optional candidates on index-source layers.
public func DeepSeek4_1Layer<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, layerIndex: Int,
  cachedRawTokenLength: Int? = nil, outputTokenLength: Int? = nil,
  boundedReplayTokens: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> (model: Model, bias: Model.Parameters) {
  let prefix = "layers.\(layerIndex)"
  let residual = Input()
  let incomingPre = Input()
  var inputs = [residual, incomingPre]
  let beforeAttention: Model.IO
  if configuration.engram.layers.contains(layerIndex) {
    let engramEmbeddings = Input()
    inputs.append(engramEmbeddings)
    beforeAttention = DeepSeek4_1Engram(
      dataType, prefix: "\(prefix).engram", residual: residual, embeddings: engramEmbeddings,
      tokenLength: tokenLength, configuration: configuration)
  } else {
    beforeAttention = residual
  }
  let attentionTokenLength = boundedReplayTokens ?? tokenLength
  precondition((0...tokenLength).contains(attentionTokenLength))
  let attentionResidual: Model.IO
  if layerIndex == configuration.candidateSourceLayer {
    let hc = configuration.hcCount
    let hidden = configuration.hiddenSize
    attentionResidual = beforeAttention.reshaped(
      attentionTokenLength > 0 ? [attentionTokenLength, hc, hidden] : [0],
      offset: [tokenLength - attentionTokenLength, 0, 0], strides: [hc * hidden, hidden, 1]
    )
  } else {
    attentionResidual = beforeAttention
  }
  // The decoder's global KV/index projections consume every encoder output.
  // Only its local residual mixing, attention, and FFN use the replay suffix.
  let (attnPre, attnPost, attnCombination, attnCollapsed) = DeepSeek4_1HCMix(
    prefix: "\(prefix).hc_attn", residual: beforeAttention, incomingPre: incomingPre,
    tokenLength: tokenLength, mixTokenLength: attentionTokenLength, configuration: configuration)
  let normalizedAttention = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).attn_norm")(
      attnCollapsed
    ).to(FloatType.dataType)
  let rotary = Input()
  let rawKeyValue = Input()
  inputs.append(contentsOf: [rotary, rawKeyValue])
  let attention: Model.IO
  var indices: Model.IO?
  var candidates: Model.IO?
  if configuration.compressionRatios[layerIndex] == 0 {
    attention = DeepSeek4_1SWAttention(
      dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      layerIndex: layerIndex, cachedRawTokenLength: cachedRawTokenLength,
      configuration: configuration)(
        normalizedAttention, rotary, rawKeyValue)
  } else if configuration.indexSourceLayers.contains(layerIndex) {
    let globalKeyValue = Input()
    let indexKeyValue = Input()
    inputs.append(contentsOf: [globalKeyValue, indexKeyValue])
    var args: [Model.IO] = [
      normalizedAttention, rotary, rawKeyValue, globalKeyValue, indexKeyValue,
    ]
    if configuration.kvSourceLayers.contains(layerIndex) {
      let compressorInput = Input()
      let compressorRotary = Input()
      let indexerCompressorRotary = Input()
      inputs.append(contentsOf: [compressorInput, compressorRotary, indexerCompressorRotary])
      args.append(contentsOf: [compressorInput, compressorRotary, indexerCompressorRotary])
    }
    let indexerRotary = Input()
    inputs.append(indexerRotary)
    args.append(indexerRotary)
    if layerIndex > configuration.candidateSourceLayer {
      let candidates = Input()
      inputs.append(candidates)
      args.append(candidates)
    }
    let result = DeepSeek4_1CompressedSparseAttention(
      dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      layerIndex: layerIndex, cachedRawTokenLength: cachedRawTokenLength,
      boundedReplayTokens: boundedReplayTokens, configuration: configuration)(args)
    attention = result[0]
    indices = result[1]
    if layerIndex == configuration.candidateSourceLayer { candidates = result[2] }
  } else {
    let globalKeyValue = Input()
    let indices = Input()
    inputs.append(contentsOf: [globalKeyValue, indices])
    attention = DeepSeek4_1SharedAttention(
      dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
      layerIndex: layerIndex, cachedRawTokenLength: cachedRawTokenLength,
      configuration: configuration)(
        normalizedAttention, rotary, rawKeyValue, globalKeyValue, indices)
  }
  let afterAttention = HyperConnection(count: configuration.hcCount, operation: .expand)(
    attention, attentionResidual, attnPost, attnCombination)[0]
  let ffnTokenLength = outputTokenLength ?? attentionTokenLength
  precondition(ffnTokenLength >= 0 && ffnTokenLength <= attentionTokenLength)
  let ffnResidual: Model.IO
  let incomingFFNPre: Model.IO
  if outputTokenLength != nil {
    let hc = configuration.hcCount
    let hidden = configuration.hiddenSize
    ffnResidual = afterAttention.reshaped(
      ffnTokenLength > 0 ? [ffnTokenLength, hc, hidden] : [0],
      offset: [attentionTokenLength - ffnTokenLength, 0, 0],
      strides: [hc * hidden, hidden, 1]
    )
    incomingFFNPre = attnPre.reshaped(
      ffnTokenLength > 0 ? [ffnTokenLength, hc] : [0],
      offset: [attentionTokenLength - ffnTokenLength, 0], strides: [hc, 1]
    )
  } else {
    ffnResidual = afterAttention
    incomingFFNPre = attnPre
  }
  let (ffnPre, ffnPost, ffnCombination, ffnCollapsed) = DeepSeek4_1HCMix(
    prefix: "\(prefix).hc_ffn", residual: ffnResidual, incomingPre: incomingFFNPre,
    tokenLength: ffnTokenLength, configuration: configuration)
  let normalizedFFN = RMSNorm(
    epsilon: configuration.normEpsilon, axis: [1], name: "\(prefix).ffn_norm")(
      ffnCollapsed
    )
  let moe = DeepSeek4_1MoE(
    dataType, prefix: "\(prefix).ffn", tokenLength: ffnTokenLength, layerIndex: layerIndex,
    configuration: configuration)
  let ffn = moe.model(normalizedFFN)
  let expanded = HyperConnection(count: configuration.hcCount, operation: .expand)(
    ffn, ffnResidual, ffnPost, ffnCombination)[0]
  var outputs = [expanded, ffnPre]
  if let indices { outputs.append(indices) }
  if let candidates { outputs.append(candidates) }
  return (Model(inputs, outputs), moe.bias)
}

/// Collapses the final residual with the last FFN's returned pre-mix. V4.1 has
/// no learned HC head projection. Normalization is FP32; the vocabulary
/// projection uses the activation datatype, as in V4 Flash.
public func DeepSeek4_1OutputHead<FloatType: TensorNumeric>(
  x: Model.IO, incomingPre: Model.IO, tokenLength: Int,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash,
  of dataType: FloatType.Type
) -> Model.IO {
  let collapsed =
    (x .* incomingPre.reshaped([tokenLength, configuration.hcCount, 1]))
    .reduced(.sum, axis: [1]).reshaped([tokenLength, configuration.hiddenSize])
  let norm = RMSNorm(epsilon: configuration.normEpsilon, axis: [1], name: "norm")
  let head = Dense(count: configuration.vocabularySize, noBias: true, name: "head")
  return head(norm(collapsed).to(FloatType.dataType))
}

/// Attention rows at each layer, followed by the output row count.
/// boundedReplay maps to "SWA Bounded Replay" in the paper: only the trailing
/// boundedReplayTokens traverse the decoder. Zero skips decoder work while
/// still publishing this chunk's encoder-derived global KV and index keys.
public func DeepSeek4_1CausalLMTokenLengths(
  tokenLength: Int, boundedReplayTokens: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> [Int] {
  precondition(tokenLength > 0 && configuration.layers > 0)
  let replayTokens = boundedReplayTokens ?? tokenLength
  precondition((0...tokenLength).contains(replayTokens))
  if boundedReplayTokens != nil {
    precondition(configuration.kvSourceLayers.last == configuration.candidateSourceLayer)
    precondition(configuration.engram.layers.allSatisfy { $0 < configuration.candidateSourceLayer })
    precondition(
      configuration.compressionRatios[configuration.candidateSourceLayer...].allSatisfy { $0 == 1 })
  }
  return (0..<configuration.layers).map {
    $0 < configuration.candidateSourceLayer ? tokenLength : replayTokens
  } + [min(1, replayTokens)]
}

private func DeepSeek4_1Prefix<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int, boundedReplayTokens: Int?,
  cachedRawTokenLengths: [Int],
  configuration: DeepSeek4_1ModelConfiguration
) -> (
  inputs: [Input], hidden: Model.IO, pre: Model.IO,
  biases: [(key: String, bias: Model.Parameters)]
) {
  precondition(tokenLength > 0 && cachedTokenLength >= 0)
  precondition(configuration.layers > 0)
  precondition(cachedRawTokenLengths.count == configuration.layers)
  precondition(cachedRawTokenLengths.allSatisfy { $0 >= 0 })
  let tokens = Input()
  let injected = Input()
  let incomingPre = Input()
  var inputs = [tokens, injected, incomingPre]
  let ratios = configuration.compressionRatios.prefix(configuration.layers)
  let rawRotary = ratios.contains(0) ? Input() : nil
  if let rawRotary { inputs.append(rawRotary) }
  let usesCompressedRotary = ratios.contains { $0 > 0 }
  let compressedRotary = usesCompressedRotary ? Input() : nil
  let indexerRotary = usesCompressedRotary ? Input() : nil
  if let compressedRotary, let indexerRotary {
    inputs.append(contentsOf: [compressedRotary, indexerRotary])
  }
  var compressorRotaries = [(ratio: Int, rotary: Input, indexerRotary: Input)]()
  for ratio in Set(ratios.filter { $0 > 0 }).sorted() {
    let rotary = Input()
    let indexerRotary = Input()
    compressorRotaries.append((ratio, rotary, indexerRotary))
    inputs.append(contentsOf: [rotary, indexerRotary])
  }
  var hidden = DeepSeek4_1Embedding(
    dataType, tokenLength: tokenLength,
    configuration: configuration)(tokens, injected)
  var pre: Model.IO = incomingPre
  var globalKeyValue: Input?
  var indexKeyValue: Input?
  var indices: Model.IO?
  var candidates: Model.IO?
  var biases = [(key: String, bias: Model.Parameters)]()
  let lengths = DeepSeek4_1CausalLMTokenLengths(
    tokenLength: tokenLength, boundedReplayTokens: boundedReplayTokens, configuration: configuration
  )
  for layer in 0..<configuration.layers {
    // The first decoder layer projects global KV/index keys from the complete
    // encoder chunk before slicing queries, raw KV, and the residual to replay.
    let layerTokenLength =
      layer == configuration.candidateSourceLayer ? tokenLength : lengths[layer]
    let layerCachedTokenLength = cachedTokenLength + tokenLength - layerTokenLength
    let ratio = configuration.compressionRatios[layer]
    let engramEmbeddings = configuration.engram.layers.contains(layer) ? Input() : nil
    if let engramEmbeddings { inputs.append(engramEmbeddings) }
    let rawKeyValue = Input()
    inputs.append(rawKeyValue)
    let compressorInput: Input?
    if configuration.kvSourceLayers.contains(layer) {
      let global = Input()
      let index = Input()
      let compressor = Input()
      inputs.append(contentsOf: [global, index, compressor])
      globalKeyValue = global
      indexKeyValue = index
      compressorInput = compressor
    } else {
      compressorInput = nil
    }
    // Reader layers reuse the latest source's cache and selection directly.
    let compressorRotary = compressorRotaries.first { $0.ratio == ratio }
    var args = [hidden, pre]
    if let engramEmbeddings { args.append(engramEmbeddings) }
    var rotary: Model.IO = (ratio > 0 ? compressedRotary : rawRotary)!
    if layer > configuration.candidateSourceLayer {
      let dim = configuration.attentionHeadDim
      rotary = rotary.reshaped(
        layerTokenLength > 0 ? [layerTokenLength, dim] : [0],
        offset: [tokenLength - layerTokenLength, 0], strides: [dim, 1]
      )
    }
    args.append(contentsOf: [rotary, rawKeyValue])
    if ratio > 0 {
      args.append(globalKeyValue!)
      if configuration.indexSourceLayers.contains(layer) {
        args.append(indexKeyValue!)
        if let compressorInput {
          args.append(contentsOf: [
            compressorInput, compressorRotary!.rotary, compressorRotary!.indexerRotary,
          ])
        }
        var queryRotary: Model.IO = indexerRotary!
        if layer > configuration.candidateSourceLayer {
          let dim = configuration.indexerHeadDim
          queryRotary = queryRotary.reshaped(
            layerTokenLength > 0 ? [layerTokenLength, dim] : [0],
            offset: [tokenLength - layerTokenLength, 0], strides: [dim, 1]
          )
        }
        args.append(queryRotary)
        if layer > configuration.candidateSourceLayer { args.append(candidates!) }
      } else {
        args.append(indices!)
      }
    }
    let (decoderLayer, bias) = DeepSeek4_1Layer(
      dataType, tokenLength: layerTokenLength, cachedTokenLength: layerCachedTokenLength,
      layerIndex: layer, cachedRawTokenLength: cachedRawTokenLengths[layer],
      outputTokenLength: layer == configuration.layers - 1 ? lengths.last! : nil,
      boundedReplayTokens: layer == configuration.candidateSourceLayer ? lengths[layer] : nil,
      configuration: configuration)
    biases.append(("layers.\(layer).ffn.gate.bias_vl", bias))
    let outputs = decoderLayer(args)
    hidden = outputs[0]
    pre = outputs[1]
    if configuration.indexSourceLayers.contains(layer) {
      indices = outputs[2]
      if layer == configuration.candidateSourceLayer { candidates = outputs[3] }
    }
  }
  return (inputs, hidden, pre, biases)
}

/// Complete text model, returning FP16/FP32 logits for the last query token.
/// Inputs: text token IDs, image embeddings (one input is empty), initial FP32
/// pre-mix [T, hc] with rows [1, 0, ...], raw rotary
/// (if used), compressed and indexer rotary (if used), then compressor/indexer
/// rotary pairs in ascending compression-ratio order. Each layer appends its
/// optional Engram embeddings and raw KV; KV-source layers also append global
/// KV, index KV, and compressor input. Rotary tensors and caches use the activation
/// dtype. Cache buffers are updated in place; callers retain the raw-window tail
/// and unfinished compressor input. Pass each layer's stored raw-row count when
/// binding a growing cache prefix. Input order depends only on the architecture.
public func DeepSeek4_1CausalLM<FloatType: TensorNumeric>(
  _ dataType: FloatType.Type, tokenLength: Int, cachedTokenLength: Int = 0,
  cachedRawTokenLengths: [Int]? = nil, boundedReplayTokens: Int? = nil,
  configuration: DeepSeek4_1ModelConfiguration = .deepSeekV4_1Flash
) -> (model: Model, biases: [(key: String, bias: Model.Parameters)]) {
  let prefix = DeepSeek4_1Prefix(
    dataType, tokenLength: tokenLength, cachedTokenLength: cachedTokenLength,
    boundedReplayTokens: boundedReplayTokens,
    cachedRawTokenLengths: cachedRawTokenLengths
      ?? Array(
        repeating: min(cachedTokenLength, configuration.rawWindow), count: configuration.layers),
    configuration: configuration)
  let output = DeepSeek4_1OutputHead(
    x: prefix.hidden, incomingPre: prefix.pre,
    tokenLength: min(1, boundedReplayTokens ?? tokenLength),
    configuration: configuration, of: dataType)
  return (Model(prefix.inputs, [output]), prefix.biases)
}
