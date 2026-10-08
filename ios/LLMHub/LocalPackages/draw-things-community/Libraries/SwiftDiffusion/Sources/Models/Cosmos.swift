import DiffusionMappings
import Foundation
import NNC

private let CosmosLayerCount = 28
public let CosmosFixedTimeConditionCount = CosmosLayerCount * 9 + 2
public let CosmosFixedKVConditionCount = CosmosLayerCount * 2

public func AnimaRotaryPositionEmbedding(
  sequenceLengths: (Int, Int), headDimension: Int = 64, theta: Double = 10_000
) -> Tensor<Float> {
  precondition(headDimension % 2 == 0)
  let half = headDimension / 2
  let totalSequenceLength = sequenceLengths.0 + sequenceLengths.1
  var rotary = Tensor<Float>(.CPU, .NHWC(1, totalSequenceLength, 1, headDimension))
  var offset = 0
  for sequenceLength in [sequenceLengths.0, sequenceLengths.1] {
    for i in 0..<sequenceLength {
      for k in 0..<half {
        let freq = Double(i) / pow(theta, Double(2 * k) / Double(headDimension))
        rotary[0, offset + i, 0, k * 2] = Float(cos(freq))
        rotary[0, offset + i, 0, k * 2 + 1] = Float(sin(freq))
      }
    }
    offset += sequenceLength
  }
  return rotary
}

public func CosmosRotaryPositionEmbedding(
  frames: Int = 1, height: Int, width: Int,
  headDimension: Int = 128, heads: Int = 1, theta: Double = 10_000,
  ropeScale: (Double, Double, Double) = (1.0, 4.0, 4.0)
) -> Tensor<Float> {
  precondition(headDimension % 2 == 0)
  let dimH = headDimension / 6 * 2
  let dimW = headDimension / 6 * 2
  let dimT = headDimension - dimH - dimW
  let half = headDimension / 2
  precondition(half == dimT / 2 + dimH / 2 + dimW / 2)
  let hNTKFactor = pow(ropeScale.1, Double(dimH) / Double(dimH - 2))
  let wNTKFactor = pow(ropeScale.2, Double(dimW) / Double(dimW - 2))
  let tNTKFactor = pow(ropeScale.0, Double(dimT) / Double(dimT - 2))
  let hTheta = theta * hNTKFactor
  let wTheta = theta * wNTKFactor
  let tTheta = theta * tNTKFactor
  let temporalFreqs = (0..<(dimT / 2)).map { 1.0 / pow(tTheta, Double(2 * $0) / Double(dimT)) }
  let heightFreqs = (0..<(dimH / 2)).map { 1.0 / pow(hTheta, Double(2 * $0) / Double(dimH)) }
  let widthFreqs = (0..<(dimW / 2)).map { 1.0 / pow(wTheta, Double(2 * $0) / Double(dimW)) }
  var rotary = Tensor<Float>(.CPU, .NHWC(1, frames * height * width, heads, headDimension))
  for t in 0..<frames {
    for y in 0..<height {
      for x in 0..<width {
        let token = t * height * width + y * width + x
        for head in 0..<heads {
          var i = 0
          for freq in temporalFreqs {
            let theta = Double(t) * freq
            rotary[0, token, head, i * 2] = Float(cos(theta))
            rotary[0, token, head, i * 2 + 1] = Float(sin(theta))
            i += 1
          }
          for freq in heightFreqs {
            let theta = Double(y) * freq
            rotary[0, token, head, i * 2] = Float(cos(theta))
            rotary[0, token, head, i * 2 + 1] = Float(sin(theta))
            i += 1
          }
          for freq in widthFreqs {
            let theta = Double(x) * freq
            rotary[0, token, head, i * 2] = Float(cos(theta))
            rotary[0, token, head, i * 2 + 1] = Float(sin(theta))
            i += 1
          }
        }
      }
    }
  }
  return rotary
}

private func AnimaSegmentedAttention(
  queries: Model.IO, keys: Model.IO, values: Model.IO, queryLength: (Int, Int),
  keyValueLength: (Int, Int), headDimension: Int, numberOfHeads: Int, usesFlashAttention: Bool
)
  -> Model.IO
{
  let totalQueryLength = queryLength.0 + queryLength.1
  let totalKeyValueLength = keyValueLength.0 + keyValueLength.1
  let segments = [
    (query: queryLength.0, keyValue: keyValueLength.0, queryOffset: 0, keyValueOffset: 0),
    (
      query: queryLength.1, keyValue: keyValueLength.1, queryOffset: queryLength.0,
      keyValueOffset: keyValueLength.0
    ),
  ]
  var outs = [Model.IO]()
  if usesFlashAttention {
    let attentionScale = 1.0 / Float(headDimension).squareRoot().squareRoot()
    let queries = attentionScale * queries
    let keys = attentionScale * keys
    for segment in segments {
      let query = queries.reshaped(
        [1, segment.query, numberOfHeads, headDimension],
        offset: [0, segment.queryOffset, 0, 0],
        strides: [
          totalQueryLength * numberOfHeads * headDimension, numberOfHeads * headDimension,
          headDimension, 1,
        ]
      ).contiguous()
      let key = keys.reshaped(
        [1, segment.keyValue, numberOfHeads, headDimension],
        offset: [0, segment.keyValueOffset, 0, 0],
        strides: [
          totalKeyValueLength * numberOfHeads * headDimension, numberOfHeads * headDimension,
          headDimension, 1,
        ]
      ).contiguous()
      let value = values.reshaped(
        [1, segment.keyValue, numberOfHeads, headDimension],
        offset: [0, segment.keyValueOffset, 0, 0],
        strides: [
          totalKeyValueLength * numberOfHeads * headDimension, numberOfHeads * headDimension,
          headDimension, 1,
        ]
      ).contiguous()
      let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
      let out = scaledDotProductAttention(query, key, value)
      if let last = outs.last {
        out.add(dependencies: [last])
      }
      outs.append(out)
    }
  } else {
    for segment in segments {
      let query = queries.reshaped(
        [1, segment.query, numberOfHeads, headDimension],
        offset: [0, segment.queryOffset, 0, 0],
        strides: [
          totalQueryLength * numberOfHeads * headDimension, numberOfHeads * headDimension,
          headDimension, 1,
        ]
      ).transposed(1, 2).contiguous()
      let key = keys.reshaped(
        [1, segment.keyValue, numberOfHeads, headDimension],
        offset: [0, segment.keyValueOffset, 0, 0],
        strides: [
          totalKeyValueLength * numberOfHeads * headDimension, numberOfHeads * headDimension,
          headDimension, 1,
        ]
      ).transposed(1, 2).contiguous()
      let value = values.reshaped(
        [1, segment.keyValue, numberOfHeads, headDimension],
        offset: [0, segment.keyValueOffset, 0, 0],
        strides: [
          totalKeyValueLength * numberOfHeads * headDimension, numberOfHeads * headDimension,
          headDimension, 1,
        ]
      ).transposed(1, 2).contiguous()
      var dot =
        Matmul(transposeB: (2, 3))(
          (1.0 / Float(headDimension).squareRoot()) * query, key)
      dot = dot.reshaped([numberOfHeads * segment.query, segment.keyValue]).softmax()
      dot = dot.reshaped([1, numberOfHeads, segment.query, segment.keyValue])
      let out = (dot * value).transposed(1, 2)
      if let last = outs.last {
        out.add(dependencies: [last])
      }
      outs.append(out)
    }
  }
  let concat = Concat(axis: 1)
  concat.flags = .disableOpt
  return concat(outs)
}

private func AnimaSelfAttention(
  prefix: String, width: Int, headDimension: Int, numberOfHeads: Int,
  tokenLength: (Int, Int), usesFlashAttention: Bool
)
  -> (ModelWeightMapper, Model)
{
  let totalTokenLength = tokenLength.0 + tokenLength.1
  let x = Input()
  let rot = Input()
  let toKeys = Dense(count: headDimension * numberOfHeads, noBias: true, name: "k_proj")
  let toQueries = Dense(count: headDimension * numberOfHeads, noBias: true, name: "q_proj")
  let toValues = Dense(count: headDimension * numberOfHeads, noBias: true, name: "v_proj")
  var keys = toKeys(x).reshaped(.NHWC(1, totalTokenLength, numberOfHeads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  let normedKeys = normK(keys)
  keys = Functional.cmul(left: normedKeys, right: rot)
  var queries = toQueries(x).reshaped(.NHWC(1, totalTokenLength, numberOfHeads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  let normedQueries = normQ(queries)
  queries = Functional.cmul(left: normedQueries, right: rot)
  let values = toValues(x).reshaped(.NHWC(1, totalTokenLength, numberOfHeads, headDimension))
  let out = AnimaSegmentedAttention(
    queries: queries, keys: keys, values: values, queryLength: tokenLength,
    keyValueLength: tokenLength, headDimension: headDimension, numberOfHeads: numberOfHeads,
    usesFlashAttention: usesFlashAttention)
  let flattenedOut = out.reshaped([totalTokenLength, width])
  let unifyHeads = Dense(count: width, noBias: true, name: "out_proj")
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    let normQElement = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    let normKElement = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    mapping["\(prefix).q_proj.weight"] = ModelWeightElement(
      [toQueries.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    mapping["\(prefix).q_norm.weight"] = normQElement
    mapping["\(prefix).k_proj.weight"] = ModelWeightElement(
      [toKeys.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    mapping["\(prefix).k_norm.weight"] = normKElement
    mapping["\(prefix).v_proj.weight"] = [toValues.weight.name]
    mapping["\(prefix).o_proj.weight"] = [unifyHeads.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot], [unifyHeads(flattenedOut).to(of: x)]))
}

private func AnimaCrossAttention(
  prefix: String, queryDim: Int, headDimension: Int, numberOfHeads: Int,
  tokenLength: (Int, Int), contextLength: (Int, Int), usesFlashAttention: Bool
) -> (ModelWeightMapper, Model) {
  let totalTokenLength = tokenLength.0 + tokenLength.1
  let totalContextLength = contextLength.0 + contextLength.1
  let x = Input()
  let context = Input()
  let queryRot = Input()
  let contextRot = Input()
  let toKeys = Dense(count: headDimension * numberOfHeads, noBias: true, name: "c_k_proj")
  let toQueries = Dense(count: headDimension * numberOfHeads, noBias: true, name: "c_q_proj")
  let toValues = Dense(count: headDimension * numberOfHeads, noBias: true, name: "c_v_proj")
  var keys = toKeys(context).reshaped(.NHWC(1, totalContextLength, numberOfHeads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  let normedKeys = normK(keys)
  keys = Functional.cmul(left: normedKeys, right: contextRot)
  var queries = toQueries(x).reshaped(.NHWC(1, totalTokenLength, numberOfHeads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  let normedQueries = normQ(queries)
  queries = Functional.cmul(left: normedQueries, right: queryRot)
  let values = toValues(context).reshaped(
    .NHWC(1, totalContextLength, numberOfHeads, headDimension))
  let out = AnimaSegmentedAttention(
    queries: queries, keys: keys, values: values, queryLength: tokenLength,
    keyValueLength: contextLength, headDimension: headDimension, numberOfHeads: numberOfHeads,
    usesFlashAttention: usesFlashAttention)
  let flattenedOut = out.reshaped([totalTokenLength, queryDim])
  let unifyHeads = Dense(count: queryDim, noBias: true, name: "c_out_proj")
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    let normQElement = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    let normKElement = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    mapping["\(prefix).q_proj.weight"] = ModelWeightElement(
      [toQueries.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    mapping["\(prefix).q_norm.weight"] = normQElement
    mapping["\(prefix).k_proj.weight"] = ModelWeightElement(
      [toKeys.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    mapping["\(prefix).k_norm.weight"] = normKElement
    mapping["\(prefix).v_proj.weight"] = [toValues.weight.name]
    mapping["\(prefix).o_proj.weight"] = [unifyHeads.weight.name]
    return mapping
  }
  return (
    mapper,
    Model([x, context, queryRot, contextRot], [unifyHeads(flattenedOut).to(of: x)])
  )
}

private func AnimaAdapterBlock(
  tokenLength: (Int, Int), contextLength: (Int, Int), width: Int, headDimension: Int,
  numberOfHeads: Int, intermediateSize: Int, prefix: String, usesFlashAttention: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let context = Input()
  let targetRot = Input()
  let sourceRot = Input()
  let normSelf = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_self")
  let (selfAttentionMapper, selfAttention) = AnimaSelfAttention(
    prefix: "\(prefix).self_attn", width: width, headDimension: headDimension,
    numberOfHeads: numberOfHeads, tokenLength: tokenLength, usesFlashAttention: usesFlashAttention)
  var out = x + selfAttention(normSelf(x), targetRot)
  let normCross = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_cross")
  let (crossAttentionMapper, crossAttention) = AnimaCrossAttention(
    prefix: "\(prefix).cross_attn", queryDim: width, headDimension: headDimension,
    numberOfHeads: numberOfHeads, tokenLength: tokenLength, contextLength: contextLength,
    usesFlashAttention: usesFlashAttention)
  out = out + crossAttention(normCross(out), context, targetRot, sourceRot)
  let normMlp = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_mlp")
  let fc1 = Dense(count: intermediateSize, name: "mlp_fc1")
  let fc2 = Dense(count: width, name: "mlp_fc2")
  out = out + fc2(fc1(normMlp(out)).GELU())
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(selfAttentionMapper(format)) { v, _ in v }
    mapping.merge(crossAttentionMapper(format)) { v, _ in v }
    mapping["\(prefix).norm_self_attn.weight"] = [normSelf.weight.name]
    mapping["\(prefix).norm_cross_attn.weight"] = [normCross.weight.name]
    mapping["\(prefix).norm_mlp.weight"] = [normMlp.weight.name]
    mapping["\(prefix).mlp.0.weight"] = [fc1.weight.name]
    mapping["\(prefix).mlp.0.bias"] = [fc1.bias.name]
    mapping["\(prefix).mlp.2.weight"] = [fc2.weight.name]
    mapping["\(prefix).mlp.2.bias"] = [fc2.bias.name]
    return mapping
  }
  return (mapper, Model([x, context, targetRot, sourceRot], [out]))
}

public func AnimaLLMAdapter(
  targetLength: (Int, Int), sourceLength: (Int, Int), vocabularySize: Int = 32_128,
  modelDimension: Int = 1_024, layers: Int = 6, headDimension: Int = 64, numberOfHeads: Int = 16,
  intermediateSize: Int = 4_096, usesFlashAttention: Bool = true
) -> (ModelWeightMapper, Model) {
  let sourceHiddenStates = Input()
  let targetInputIDs = Input()
  let targetRot = Input()
  let sourceRot = Input()
  let embed = Embedding(
    FloatType.self, vocabularySize: vocabularySize, embeddingSize: modelDimension,
    name: "token_embedding")
  var out = embed(targetInputIDs)
  var blockMappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = AnimaAdapterBlock(
      tokenLength: targetLength, contextLength: sourceLength, width: modelDimension,
      headDimension: headDimension, numberOfHeads: numberOfHeads,
      intermediateSize: intermediateSize,
      prefix: "blocks.\(i)", usesFlashAttention: usesFlashAttention)
    out = block(out, sourceHiddenStates, targetRot, sourceRot)
    blockMappers.append(mapper)
  }
  let outProj = Dense(count: modelDimension, name: "out_proj")
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_out")
  out = norm(outProj(out))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["embed.weight"] = [embed.weight.name]
    for blockMapper in blockMappers {
      mapping.merge(blockMapper(format)) { v, _ in v }
    }
    mapping["out_proj.weight"] = [outProj.weight.name]
    mapping["out_proj.bias"] = [outProj.bias.name]
    mapping["norm.weight"] = [norm.weight.name]
    return mapping
  }
  return (mapper, Model([sourceHiddenStates, targetInputIDs, targetRot, sourceRot], [out]))
}

private func LoRAAnimaSelfAttention(
  prefix: String, width: Int, headDimension: Int, numberOfHeads: Int,
  tokenLength: (Int, Int), usesFlashAttention: Bool, layerIndex: Int,
  configuration: LoRANetworkConfiguration
)
  -> (ModelWeightMapper, Model)
{
  let totalTokenLength = tokenLength.0 + tokenLength.1
  let x = Input()
  let rot = Input()
  let toKeys = LoRADense(
    count: headDimension * numberOfHeads, configuration: configuration, noBias: true,
    index: layerIndex, name: "k_proj")
  let toQueries = LoRADense(
    count: headDimension * numberOfHeads, configuration: configuration, noBias: true,
    index: layerIndex, name: "q_proj")
  let toValues = LoRADense(
    count: headDimension * numberOfHeads, configuration: configuration, noBias: true,
    index: layerIndex, name: "v_proj")
  var keys = toKeys(x).reshaped(.NHWC(1, totalTokenLength, numberOfHeads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  let normedKeys = normK(keys)
  keys = Functional.cmul(left: normedKeys, right: rot)
  var queries = toQueries(x).reshaped(.NHWC(1, totalTokenLength, numberOfHeads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  let normedQueries = normQ(queries)
  queries = Functional.cmul(left: normedQueries, right: rot)
  let values = toValues(x).reshaped(.NHWC(1, totalTokenLength, numberOfHeads, headDimension))
  let out = AnimaSegmentedAttention(
    queries: queries, keys: keys, values: values, queryLength: tokenLength,
    keyValueLength: tokenLength, headDimension: headDimension, numberOfHeads: numberOfHeads,
    usesFlashAttention: usesFlashAttention)
  let flattenedOut = out.reshaped([totalTokenLength, width])
  let unifyHeads = LoRADense(
    count: width, configuration: configuration, noBias: true, index: layerIndex, name: "out_proj")
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    let normQElement = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    let normKElement = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    mapping["\(prefix).q_proj.weight"] = ModelWeightElement(
      [toQueries.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    mapping["\(prefix).q_norm.weight"] = normQElement
    mapping["\(prefix).k_proj.weight"] = ModelWeightElement(
      [toKeys.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    mapping["\(prefix).k_norm.weight"] = normKElement
    mapping["\(prefix).v_proj.weight"] = [toValues.weight.name]
    mapping["\(prefix).o_proj.weight"] = [unifyHeads.weight.name]
    return mapping
  }
  return (mapper, Model([x, rot], [unifyHeads(flattenedOut).to(of: x)]))
}

private func LoRAAnimaCrossAttention(
  prefix: String, queryDim: Int, headDimension: Int, numberOfHeads: Int,
  tokenLength: (Int, Int), contextLength: (Int, Int), usesFlashAttention: Bool, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let totalTokenLength = tokenLength.0 + tokenLength.1
  let totalContextLength = contextLength.0 + contextLength.1
  let x = Input()
  let context = Input()
  let queryRot = Input()
  let contextRot = Input()
  let toKeys = LoRADense(
    count: headDimension * numberOfHeads, configuration: configuration, noBias: true,
    index: layerIndex, name: "c_k_proj")
  let toQueries = LoRADense(
    count: headDimension * numberOfHeads, configuration: configuration, noBias: true,
    index: layerIndex, name: "c_q_proj")
  let toValues = LoRADense(
    count: headDimension * numberOfHeads, configuration: configuration, noBias: true,
    index: layerIndex, name: "c_v_proj")
  var keys = toKeys(context).reshaped(.NHWC(1, totalContextLength, numberOfHeads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  let normedKeys = normK(keys)
  keys = Functional.cmul(left: normedKeys, right: contextRot)
  var queries = toQueries(x).reshaped(.NHWC(1, totalTokenLength, numberOfHeads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  let normedQueries = normQ(queries)
  queries = Functional.cmul(left: normedQueries, right: queryRot)
  let values = toValues(context).reshaped(
    .NHWC(1, totalContextLength, numberOfHeads, headDimension))
  let out = AnimaSegmentedAttention(
    queries: queries, keys: keys, values: values, queryLength: tokenLength,
    keyValueLength: contextLength, headDimension: headDimension, numberOfHeads: numberOfHeads,
    usesFlashAttention: usesFlashAttention)
  let flattenedOut = out.reshaped([totalTokenLength, queryDim])
  let unifyHeads = LoRADense(
    count: queryDim, configuration: configuration, noBias: true, index: layerIndex,
    name: "c_out_proj")
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    let normQElement = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    let normKElement = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    mapping["\(prefix).q_proj.weight"] = ModelWeightElement(
      [toQueries.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    mapping["\(prefix).q_norm.weight"] = normQElement
    mapping["\(prefix).k_proj.weight"] = ModelWeightElement(
      [toKeys.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    mapping["\(prefix).k_norm.weight"] = normKElement
    mapping["\(prefix).v_proj.weight"] = [toValues.weight.name]
    mapping["\(prefix).o_proj.weight"] = [unifyHeads.weight.name]
    return mapping
  }
  return (
    mapper,
    Model([x, context, queryRot, contextRot], [unifyHeads(flattenedOut).to(of: x)])
  )
}

private func LoRAAnimaAdapterBlock(
  tokenLength: (Int, Int), contextLength: (Int, Int), width: Int, headDimension: Int,
  numberOfHeads: Int, intermediateSize: Int, prefix: String, usesFlashAttention: Bool,
  layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let context = Input()
  let targetRot = Input()
  let sourceRot = Input()
  let normSelf = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_self")
  let (selfAttentionMapper, selfAttention) = LoRAAnimaSelfAttention(
    prefix: "\(prefix).self_attn", width: width, headDimension: headDimension,
    numberOfHeads: numberOfHeads, tokenLength: tokenLength, usesFlashAttention: usesFlashAttention,
    layerIndex: layerIndex, configuration: configuration)
  var out = x + selfAttention(normSelf(x), targetRot)
  let normCross = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_cross")
  let (crossAttentionMapper, crossAttention) = LoRAAnimaCrossAttention(
    prefix: "\(prefix).cross_attn", queryDim: width, headDimension: headDimension,
    numberOfHeads: numberOfHeads, tokenLength: tokenLength, contextLength: contextLength,
    usesFlashAttention: usesFlashAttention, layerIndex: layerIndex,
    configuration: configuration)
  out = out + crossAttention(normCross(out), context, targetRot, sourceRot)
  let normMlp = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_mlp")
  let fc1 = LoRADense(
    count: intermediateSize, configuration: configuration, index: layerIndex, name: "mlp_fc1")
  let fc2 = LoRADense(
    count: width, configuration: configuration, index: layerIndex, name: "mlp_fc2")
  out = out + fc2(fc1(normMlp(out)).GELU())
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(selfAttentionMapper(format)) { v, _ in v }
    mapping.merge(crossAttentionMapper(format)) { v, _ in v }
    mapping["\(prefix).norm_self_attn.weight"] = [normSelf.weight.name]
    mapping["\(prefix).norm_cross_attn.weight"] = [normCross.weight.name]
    mapping["\(prefix).norm_mlp.weight"] = [normMlp.weight.name]
    mapping["\(prefix).mlp.0.weight"] = [fc1.weight.name]
    mapping["\(prefix).mlp.0.bias"] = [fc1.bias.name]
    mapping["\(prefix).mlp.2.weight"] = [fc2.weight.name]
    mapping["\(prefix).mlp.2.bias"] = [fc2.bias.name]
    return mapping
  }
  return (mapper, Model([x, context, targetRot, sourceRot], [out]))
}

public func LoRAAnimaLLMAdapter(
  targetLength: (Int, Int), sourceLength: (Int, Int), vocabularySize: Int = 32_128,
  modelDimension: Int = 1_024, layers: Int = 6, headDimension: Int = 64, numberOfHeads: Int = 16,
  intermediateSize: Int = 4_096, usesFlashAttention: Bool = true,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let sourceHiddenStates = Input()
  let targetInputIDs = Input()
  let targetRot = Input()
  let sourceRot = Input()
  let embed = Embedding(
    FloatType.self, vocabularySize: vocabularySize, embeddingSize: modelDimension,
    name: "token_embedding")
  var out = embed(targetInputIDs)
  var blockMappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = LoRAAnimaAdapterBlock(
      tokenLength: targetLength, contextLength: sourceLength, width: modelDimension,
      headDimension: headDimension, numberOfHeads: numberOfHeads,
      intermediateSize: intermediateSize,
      prefix: "blocks.\(i)", usesFlashAttention: usesFlashAttention, layerIndex: i,
      configuration: LoRAConfiguration)
    out = block(out, sourceHiddenStates, targetRot, sourceRot)
    blockMappers.append(mapper)
  }
  let outProj = LoRADense(
    count: modelDimension, configuration: LoRAConfiguration, index: 0, name: "out_proj")
  let norm = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm_out")
  out = norm(outProj(out))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["embed.weight"] = [embed.weight.name]
    for blockMapper in blockMappers {
      mapping.merge(blockMapper(format)) { v, _ in v }
    }
    mapping["out_proj.weight"] = [outProj.weight.name]
    mapping["out_proj.bias"] = [outProj.bias.name]
    mapping["norm.weight"] = [norm.weight.name]
    return mapping
  }
  return (
    mapper,
    Model([sourceHiddenStates, targetInputIDs, targetRot, sourceRot], [out], trainable: false)
  )
}

private func CosmosSelfAttention(
  prefix: (String, String), batchSize: Int, tokenLength: Int, hiddenSize: Int, headDimension: Int,
  numberOfHeads: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let toKeys = Dense(count: hiddenSize, noBias: true, name: "k_proj")
  let toQueries = Dense(count: hiddenSize, noBias: true, name: "q_proj")
  let toValues = Dense(count: hiddenSize, noBias: true, name: "v_proj")
  var keys = toKeys(x).reshaped(
    .NHWC(batchSize, tokenLength, numberOfHeads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  let normedKeys = normK(keys)
  keys = Functional.cmul(left: normedKeys, right: rot)
  var queries = toQueries(x).reshaped(
    .NHWC(batchSize, tokenLength, numberOfHeads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  let normedQueries = normQ(queries)
  queries = Functional.cmul(left: normedQueries, right: rot)
  let values = toValues(x).reshaped(
    .NHWC(batchSize, tokenLength, numberOfHeads, headDimension))
  let out: Model.IO
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    let attentionScale = 1.0 / Float(headDimension).squareRoot().squareRoot()
    queries = attentionScale * queries
    keys = attentionScale * keys
  }
  switch usesFlashAttention {
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values)
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1 / Float(headDimension).squareRoot(), flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values)
  case .none:
    let query = queries.transposed(1, 2).contiguous()
    let key = keys.transposed(1, 2).contiguous()
    let value = values.transposed(1, 2).contiguous()
    var dot = Matmul(transposeB: (2, 3))(query, key)
    dot = dot.reshaped([batchSize * numberOfHeads * tokenLength, tokenLength]).softmax()
    dot = dot.reshaped([batchSize, numberOfHeads, tokenLength, tokenLength])
    out = (dot * value).transposed(1, 2)
  }
  let mergedOut = out.reshaped([batchSize, tokenLength, hiddenSize])
  let unifyHeads = Dense(count: hiddenSize, noBias: true, name: "out_proj")
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let toQ = ModelWeightElement(
      [toQueries.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    let toK = ModelWeightElement(
      [toKeys.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    let normQElement = ModelWeightElement(
      [normQ.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    let normKElement = ModelWeightElement(
      [normK.weight.name], interleavedIndices: [0], numberOfHeads: 1, headDimension: headDimension)
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).q_proj.weight"] = toQ
      mapping["\(prefix.0).q_norm.weight"] = normQElement
      mapping["\(prefix.0).k_proj.weight"] = toK
      mapping["\(prefix.0).k_norm.weight"] = normKElement
      mapping["\(prefix.0).v_proj.weight"] = [toValues.weight.name]
      mapping["\(prefix.0).output_proj.weight"] = [unifyHeads.weight.name]
    case .diffusers:
      mapping["\(prefix.1).to_q.weight"] = toQ
      mapping["\(prefix.1).norm_q.weight"] = normQElement
      mapping["\(prefix.1).to_k.weight"] = toK
      mapping["\(prefix.1).norm_k.weight"] = normKElement
      mapping["\(prefix.1).to_v.weight"] = [toValues.weight.name]
      mapping["\(prefix.1).to_out.0.weight"] = [unifyHeads.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot], [unifyHeads(mergedOut).to(of: x)]))
}

private func CosmosCrossAttention(
  prefix: (String, String), batchSize: Int, tokenLength: Int, contextLength: Int, hiddenSize: Int,
  headDimension: Int, numberOfHeads: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toQueries = Dense(count: hiddenSize, noBias: true, name: "c_q_proj")
  var queries = toQueries(x).reshaped(
    .NHWC(batchSize, tokenLength, numberOfHeads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  queries = normQ(queries)
  let attentionKeys = keys.to(of: queries)
  let attentionValues = values.to(of: queries)
  let out: Model.IO
  if usesFlashAttention == .scale1 {
    let attentionScale = 1.0 / Float(headDimension).squareRoot().squareRoot()
    queries = attentionScale * queries
  }
  switch usesFlashAttention {
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, attentionKeys, attentionValues)
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1 / Float(headDimension).squareRoot(), flags: [.Float16])
    out = scaledDotProductAttention(queries, attentionKeys, attentionValues)
  case .none:
    let query = queries.transposed(1, 2).contiguous()
    let key = attentionKeys.transposed(1, 2).contiguous()
    let value = attentionValues.transposed(1, 2).contiguous()
    var dot =
      Matmul(transposeB: (2, 3))(
        (1.0 / Float(headDimension).squareRoot()) * query, key)
    dot = dot.reshaped([batchSize * numberOfHeads * tokenLength, contextLength]).softmax()
    dot = dot.reshaped([batchSize, numberOfHeads, tokenLength, contextLength])
    out = (dot * value).transposed(1, 2)
  }
  let mergedOut = out.reshaped([batchSize, tokenLength, hiddenSize])
  let unifyHeads = Dense(count: hiddenSize, noBias: true, name: "c_out_proj")
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).q_proj.weight"] = [toQueries.weight.name]
      mapping["\(prefix.0).q_norm.weight"] = [normQ.weight.name]
      mapping["\(prefix.0).output_proj.weight"] = [unifyHeads.weight.name]
    case .diffusers:
      mapping["\(prefix.1).to_q.weight"] = [toQueries.weight.name]
      mapping["\(prefix.1).norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).to_out.0.weight"] = [unifyHeads.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x, keys, values], [unifyHeads(mergedOut).to(of: x)]))
}

private func CosmosFeedForward(prefix: (String, String), hiddenSize: Int, mlpRatio: Int)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let fc1 = Dense(count: hiddenSize * mlpRatio, noBias: true, name: "fc1")
  let fc2 = Dense(count: hiddenSize, noBias: true, name: "fc2")
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).layer1.weight"] = [fc1.weight.name]
      mapping["\(prefix.0).layer2.weight"] = [fc2.weight.name]
    case .diffusers:
      mapping["\(prefix.1).net.0.proj.weight"] = [fc1.weight.name]
      mapping["\(prefix.1).net.2.weight"] = [fc2.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x], [fc2(fc1(x).GELU())]))
}

private func CosmosTransformerBlock(
  prefix: (String, String), batchSize: Int, tokenLength: Int, contextLength: Int, hiddenSize: Int,
  hiddenFeatures: Int, headDimension: Int, numberOfHeads: Int, mlpRatio: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let norm1Shift = Input()
  let norm1Scale = Input()
  let norm1Gate = Input()
  let norm2Shift = Input()
  let norm2Scale = Input()
  let norm2Gate = Input()
  let norm3Shift = Input()
  let norm3Scale = Input()
  let norm3Gate = Input()
  let contextKeys = Input()
  let contextValues = Input()
  let rot = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let norm1Out = norm1Scale.to(of: x) .* norm1(x) + norm1Shift.to(of: x)
  let (selfAttentionMapper, selfAttention) = CosmosSelfAttention(
    prefix: ("\(prefix.0).self_attn", "\(prefix.1).attn1"), batchSize: batchSize,
    tokenLength: tokenLength,
    hiddenSize: hiddenSize, headDimension: headDimension, numberOfHeads: numberOfHeads,
    usesFlashAttention: usesFlashAttention)
  var out =
    x
    + norm1Gate.to(of: x) .* selfAttention(norm1Out.to(.Float16), rot).to(of: x)
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let norm2Out = norm2Scale.to(of: out) .* norm2(out) + norm2Shift.to(of: out)
  let (crossAttentionMapper, crossAttention) = CosmosCrossAttention(
    prefix: ("\(prefix.0).cross_attn", "\(prefix.1).attn2"), batchSize: batchSize,
    tokenLength: tokenLength,
    contextLength: contextLength, hiddenSize: hiddenSize, headDimension: headDimension,
    numberOfHeads: numberOfHeads, usesFlashAttention: usesFlashAttention)
  out =
    out
    + norm2Gate.to(of: out)
    .* crossAttention(norm2Out.to(.Float16), contextKeys, contextValues).to(of: out)
  let norm3 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let norm3Out = norm3Scale.to(of: out) .* norm3(out) + norm3Shift.to(of: out)
  let (feedForwardMapper, feedForward) = CosmosFeedForward(
    prefix: ("\(prefix.0).mlp", "\(prefix.1).ff"), hiddenSize: hiddenSize, mlpRatio: mlpRatio)
  out =
    out
    + norm3Gate.to(of: out) .* feedForward(norm3Out.to(.Float16)).to(of: out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(selfAttentionMapper(format)) { v, _ in v }
    mapping.merge(crossAttentionMapper(format)) { v, _ in v }
    mapping.merge(feedForwardMapper(format)) { v, _ in v }
    return mapping
  }
  return (
    mapper,
    Model(
      [
        x, norm1Shift, norm1Scale, norm1Gate, norm2Shift, norm2Scale, norm2Gate, norm3Shift,
        norm3Scale, norm3Gate, contextKeys, contextValues, rot,
      ], [out])
  )
}

public func Cosmos(
  batchSize: Int, height: Int, width: Int, textLength: Int, inChannels: Int = 16,
  outChannels: Int = 16, layers: Int = 28, numberOfHeads: Int = 16,
  headDimension: Int = 128,
  mlpRatio: Int = 4, adaLNDimension: Int = 256, patchSize: (height: Int, width: Int) = (2, 2),
  usesFlashAttention: FlashAttentionLevel = .scale1
) -> (ModelWeightMapper, Model) {
  let hiddenSize = numberOfHeads * headDimension
  let projOutChannels = patchSize.height * patchSize.width * outChannels
  let tokenHeight = height / patchSize.height
  let tokenWidth = width / patchSize.width
  let tokenLength = tokenHeight * tokenWidth
  let hiddenStates = Input()
  let rot = Input()
  let blockModulations = (0..<layers).map { _ in
    (
      shift1: Input(), scale1: Input(), gate1: Input(),
      shift2: Input(), scale2: Input(), gate2: Input(),
      shift3: Input(), scale3: Input(), gate3: Input()
    )
  }
  let normOutShift = Input()
  let normOutScale = Input()
  let crossAttentionKVs = (0..<layers).map { _ in (Input(), Input()) }
  let zeroMask =
    (0 * hiddenStates.reduced(.mean, axis: [3])).reshaped(
      .NHWC(batchSize, height, width, 1))
  let paddedHiddenStates = Functional.concat(axis: 3, hiddenStates, zeroMask)
  let patchEmbed = Convolution(
    groups: 1, filters: hiddenSize, filterSize: [patchSize.height, patchSize.width], noBias: true,
    hint: Hint(stride: [patchSize.height, patchSize.width]), format: .OIHW, name: "x_embedder")
  var out = patchEmbed(paddedHiddenStates).reshaped(
    .HWC(batchSize, tokenLength, hiddenSize)
  ).to(.Float32)
  var blockMappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = CosmosTransformerBlock(
      prefix: ("blocks.\(i)", "transformer_blocks.\(i)"), batchSize: batchSize,
      tokenLength: tokenLength,
      contextLength: textLength, hiddenSize: hiddenSize, hiddenFeatures: adaLNDimension,
      headDimension: headDimension, numberOfHeads: numberOfHeads, mlpRatio: mlpRatio,
      usesFlashAttention: usesFlashAttention)
    let modulation = blockModulations[i]
    let kv = crossAttentionKVs[i]
    out = block(
      out, modulation.shift1, modulation.scale1, modulation.gate1, modulation.shift2,
      modulation.scale2, modulation.gate2, modulation.shift3, modulation.scale3, modulation.gate3,
      kv.0, kv.1, rot)
    blockMappers.append(mapper)
  }
  let normOut = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let normOutInput = out.to(of: hiddenStates)
  out = normOutScale.to(of: normOutInput) .* normOut(normOutInput)
    + normOutShift.to(of: normOutInput)
  out = out.to(of: hiddenStates)
  let projOut = Dense(count: projOutChannels, noBias: true, name: "proj_out")
  out = projOut(out).reshaped(
    [
      batchSize, tokenHeight, tokenWidth, patchSize.height, patchSize.width, outChannels,
    ]).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped(
      [batchSize, height, width, outChannels])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["x_embedder.proj.1.weight"] = [patchEmbed.weight.name]
    case .diffusers:
      mapping["patch_embed.proj.weight"] = [patchEmbed.weight.name]
    }
    for blockMapper in blockMappers {
      mapping.merge(blockMapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["final_layer.linear.weight"] = [projOut.weight.name]
    case .diffusers:
      mapping["proj_out.weight"] = [projOut.weight.name]
    }
    return mapping
  }
  var inputs: [Input] = [hiddenStates, rot]
  for modulation in blockModulations {
    inputs.append(contentsOf: [
      modulation.shift1, modulation.scale1, modulation.gate1,
      modulation.shift2, modulation.scale2, modulation.gate2,
      modulation.shift3, modulation.scale3, modulation.gate3,
    ])
  }
  inputs.append(contentsOf: [normOutShift, normOutScale])
  for kv in crossAttentionKVs {
    inputs.append(contentsOf: [kv.0, kv.1])
  }
  return (mapper, Model(inputs, [out]))
}

private func CosmosAdaLayerNormZeroFixed(
  prefix: (String, String), timesteps: Int, hiddenSize: Int, hiddenFeatures: Int, name: String
) -> (ModelWeightMapper, Model) {
  let embeddedTimestep = Input()
  let tembShift = Input()
  let tembScale = Input()
  let tembGate = Input()
  let linear1 = Dense(count: hiddenFeatures, noBias: true, name: "\(name)_linear_1")
  let shift = Dense(count: hiddenSize, noBias: true, name: "\(name)_shift")
  let scale = Dense(count: hiddenSize, noBias: true, name: "\(name)_scale")
  let gate = Dense(count: hiddenSize, noBias: true, name: "\(name)_gate")
  let hidden = linear1(embeddedTimestep.swish())
  let shiftOut = shift(hidden) + tembShift
  let scaleOut = scale(hidden) + tembScale
  let gateOut = gate(hidden) + tembGate
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let linear2 = ModelWeightElement(
      [shift.weight.name, scale.weight.name, gate.weight.name],
      offsets: [0, hiddenSize, hiddenSize * 2])
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).1.weight"] = [linear1.weight.name]
      mapping["\(prefix.0).2.weight"] = linear2
    case .diffusers:
      mapping["\(prefix.1).linear_1.weight"] = [linear1.weight.name]
      mapping["\(prefix.1).linear_2.weight"] = linear2
    }
    return mapping
  }
  return (
    mapper,
    Model(
      [embeddedTimestep, tembShift, tembScale, tembGate],
      [
        shiftOut.reshaped([timesteps, 1, hiddenSize]),
        scaleOut.reshaped([timesteps, 1, hiddenSize]),
        gateOut.reshaped([timesteps, 1, hiddenSize]),
      ])
  )
}

private func CosmosAdaLayerNormFixed(
  prefix: (String, String), timesteps: Int, hiddenSize: Int, hiddenFeatures: Int, name: String
) -> (ModelWeightMapper, Model) {
  let embeddedTimestep = Input()
  let tembShift = Input()
  let tembScale = Input()
  let linear1 = Dense(count: hiddenFeatures, noBias: true, name: "\(name)_linear_1")
  let shift = Dense(count: hiddenSize, noBias: true, name: "\(name)_shift")
  let scale = Dense(count: hiddenSize, noBias: true, name: "\(name)_scale")
  let hidden = linear1(embeddedTimestep.swish())
  let shiftOut = shift(hidden) + tembShift
  let scaleOut = scale(hidden) + tembScale + 1
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let linear2 = ModelWeightElement(
      [shift.weight.name, scale.weight.name], offsets: [0, hiddenSize])
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).1.weight"] = [linear1.weight.name]
      mapping["\(prefix.0).2.weight"] = linear2
    case .diffusers:
      mapping["\(prefix.1).linear_1.weight"] = [linear1.weight.name]
      mapping["\(prefix.1).linear_2.weight"] = linear2
    }
    return mapping
  }
  return (
    mapper,
    Model(
      [embeddedTimestep, tembShift, tembScale],
      [
        shiftOut.reshaped([timesteps, 1, hiddenSize]),
        scaleOut.reshaped([timesteps, 1, hiddenSize]),
      ]
    )
  )
}

private func CosmosCrossAttentionFixed(
  prefix: (String, String), batchSize: Int, contextLength: Int, hiddenSize: Int, headDimension: Int,
  numberOfHeads: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let toKeys = Dense(count: hiddenSize, noBias: true, name: "c_k_proj")
  let toValues = Dense(count: hiddenSize, noBias: true, name: "c_v_proj")
  var keys = toKeys(context).reshaped(
    .NHWC(batchSize, contextLength, numberOfHeads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  keys = normK(keys)
  if usesFlashAttention == .scale1 {
    let attentionScale = 1.0 / Float(headDimension).squareRoot().squareRoot()
    keys = attentionScale * keys
  }
  let values = toValues(context).reshaped(
    .NHWC(batchSize, contextLength, numberOfHeads, headDimension))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).k_proj.weight"] = [toKeys.weight.name]
      mapping["\(prefix.0).k_norm.weight"] = [normK.weight.name]
      mapping["\(prefix.0).v_proj.weight"] = [toValues.weight.name]
    case .diffusers:
      mapping["\(prefix.1).to_k.weight"] = [toKeys.weight.name]
      mapping["\(prefix.1).norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).to_v.weight"] = [toValues.weight.name]
    }
    return mapping
  }
  return (mapper, Model([context], [keys, values]))
}

public func CosmosFixed(
  timesteps: Int, batchSize: Int, textLength: Int, layers: Int = 28,
  numberOfHeads: Int = 16, headDimension: Int = 128, adaLNDimension: Int = 256,
  usesFlashAttention: FlashAttentionLevel = .scale1
) -> (ModelWeightMapper, Model) {
  let hiddenSize = numberOfHeads * headDimension
  let context = Input()
  let timestepProjection = Input()
  let timeNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "time_norm")
  let timeLinear1 = Dense(count: hiddenSize, noBias: true, name: "time_linear_1")
  let timeShift = Dense(count: hiddenSize, noBias: true, name: "time_shift")
  let timeScale = Dense(count: hiddenSize, noBias: true, name: "time_scale")
  let timeGate = Dense(count: hiddenSize, noBias: true, name: "time_gate")
  let embeddedTimestep = timeNorm(timestepProjection)
  let timeHidden = timeLinear1(timestepProjection).swish()
  let tembShift = timeShift(timeHidden)
  let tembScale = timeScale(timeHidden) + 1
  let tembGate = timeGate(timeHidden)
  var outputs = [Model.IO]()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (norm1Mapper, norm1) = CosmosAdaLayerNormZeroFixed(
      prefix: ("blocks.\(i).adaln_modulation_self_attn", "transformer_blocks.\(i).norm1"),
      timesteps: timesteps, hiddenSize: hiddenSize,
      hiddenFeatures: adaLNDimension, name: "norm1")
    let norm1Out = norm1(embeddedTimestep, tembShift, tembScale, tembGate)
    outputs.append(norm1Out[0])
    outputs.append(norm1Out[1])
    outputs.append(norm1Out[2])
    mappers.append(norm1Mapper)
    let (norm2Mapper, norm2) = CosmosAdaLayerNormZeroFixed(
      prefix: ("blocks.\(i).adaln_modulation_cross_attn", "transformer_blocks.\(i).norm2"),
      timesteps: timesteps, hiddenSize: hiddenSize,
      hiddenFeatures: adaLNDimension, name: "norm2")
    let norm2Out = norm2(embeddedTimestep, tembShift, tembScale, tembGate)
    outputs.append(norm2Out[0])
    outputs.append(norm2Out[1])
    outputs.append(norm2Out[2])
    mappers.append(norm2Mapper)
    let (norm3Mapper, norm3) = CosmosAdaLayerNormZeroFixed(
      prefix: ("blocks.\(i).adaln_modulation_mlp", "transformer_blocks.\(i).norm3"),
      timesteps: timesteps, hiddenSize: hiddenSize,
      hiddenFeatures: adaLNDimension, name: "norm3")
    let norm3Out = norm3(embeddedTimestep, tembShift, tembScale, tembGate)
    outputs.append(norm3Out[0])
    outputs.append(norm3Out[1])
    outputs.append(norm3Out[2])
    mappers.append(norm3Mapper)
  }
  let (normOutMapper, normOut) = CosmosAdaLayerNormFixed(
    prefix: ("final_layer.adaln_modulation", "norm_out"), timesteps: timesteps,
    hiddenSize: hiddenSize,
    hiddenFeatures: adaLNDimension, name: "norm_out")
  let normOutOut = normOut(embeddedTimestep, tembShift, tembScale)
  outputs.append(normOutOut[0])
  outputs.append(normOutOut[1])
  mappers.append(normOutMapper)
  for i in 0..<layers {
    let (crossAttentionMapper, crossAttention) = CosmosCrossAttentionFixed(
      prefix: ("blocks.\(i).cross_attn", "transformer_blocks.\(i).attn2"),
      batchSize: batchSize, contextLength: textLength,
      hiddenSize: hiddenSize, headDimension: headDimension, numberOfHeads: numberOfHeads,
      usesFlashAttention: usesFlashAttention)
    let crossAttentionOut = crossAttention(context)
    outputs.append(crossAttentionOut[0])
    outputs.append(crossAttentionOut[1])
    mappers.append(crossAttentionMapper)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let timeLinear2 = ModelWeightElement(
      [timeShift.weight.name, timeScale.weight.name, timeGate.weight.name],
      offsets: [0, hiddenSize, hiddenSize * 2])
    switch format {
    case .generativeModels:
      mapping["t_embedding_norm.weight"] = [timeNorm.weight.name]
      mapping["t_embedder.1.linear_1.weight"] = [timeLinear1.weight.name]
      mapping["t_embedder.1.linear_2.weight"] = timeLinear2
    case .diffusers:
      mapping["time_embed.norm.weight"] = [timeNorm.weight.name]
      mapping["time_embed.t_embedder.linear_1.weight"] = [timeLinear1.weight.name]
      mapping["time_embed.t_embedder.linear_2.weight"] = timeLinear2
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([context, timestepProjection], outputs))
}

public func CosmosFixedOutputShapes(
  timesteps: Int, batchSize: Int, textLength: Int, layers: Int = 28,
  numberOfHeads: Int = 16, headDimension: Int = 128
) -> [TensorShape] {
  let hiddenSize = numberOfHeads * headDimension
  let timeConditions = Array(
    repeating: TensorShape([timesteps, 1, hiddenSize]), count: layers * 9 + 2)
  let kvConditions = Array(
    repeating: TensorShape([batchSize, textLength, numberOfHeads, headDimension]),
    count: layers * 2)
  return timeConditions + kvConditions
}

private func LoRACosmosSelfAttention(
  prefix: (String, String), batchSize: Int, tokenLength: Int, hiddenSize: Int, headDimension: Int,
  numberOfHeads: Int, usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let toKeys = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "k_proj")
  let toQueries = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "q_proj")
  let toValues = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "v_proj")
  var keys = toKeys(x).reshaped(.NHWC(batchSize, tokenLength, numberOfHeads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  let normedKeys = normK(keys)
  keys = Functional.cmul(left: normedKeys, right: rot)
  var queries = toQueries(x).reshaped(
    .NHWC(batchSize, tokenLength, numberOfHeads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  let normedQueries = normQ(queries)
  queries = Functional.cmul(left: normedQueries, right: rot)
  let values = toValues(x).reshaped(.NHWC(batchSize, tokenLength, numberOfHeads, headDimension))
  let out: Model.IO
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    let attentionScale = 1.0 / Float(headDimension).squareRoot().squareRoot()
    queries = attentionScale * queries
    keys = attentionScale * keys
  }
  switch usesFlashAttention {
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values)
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1 / Float(headDimension).squareRoot(), flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values)
  case .none:
    let query = queries.transposed(1, 2).contiguous()
    let key = keys.transposed(1, 2).contiguous()
    let value = values.transposed(1, 2).contiguous()
    var dot = Matmul(transposeB: (2, 3))(query, key)
    dot = dot.reshaped([batchSize * numberOfHeads * tokenLength, tokenLength]).softmax()
    dot = dot.reshaped([batchSize, numberOfHeads, tokenLength, tokenLength])
    out = (dot * value).transposed(1, 2)
  }
  let mergedOut = out.reshaped([batchSize, tokenLength, hiddenSize])
  let unifyHeads = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "out_proj")
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let toQ = ModelWeightElement(
      [toQueries.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    let toK = ModelWeightElement(
      [toKeys.weight.name], interleavedIndices: [0], numberOfHeads: numberOfHeads,
      headDimension: headDimension)
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).q_proj.weight"] = toQ
      mapping["\(prefix.0).q_norm.weight"] = [normQ.weight.name]
      mapping["\(prefix.0).k_proj.weight"] = toK
      mapping["\(prefix.0).k_norm.weight"] = [normK.weight.name]
      mapping["\(prefix.0).v_proj.weight"] = [toValues.weight.name]
      mapping["\(prefix.0).output_proj.weight"] = [unifyHeads.weight.name]
    case .diffusers:
      mapping["\(prefix.1).to_q.weight"] = toQ
      mapping["\(prefix.1).norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).to_k.weight"] = toK
      mapping["\(prefix.1).norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).to_v.weight"] = [toValues.weight.name]
      mapping["\(prefix.1).to_out.0.weight"] = [unifyHeads.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot], [unifyHeads(mergedOut).to(of: x)]))
}

private func LoRACosmosCrossAttention(
  prefix: (String, String), batchSize: Int, tokenLength: Int, contextLength: Int, hiddenSize: Int,
  headDimension: Int, numberOfHeads: Int, usesFlashAttention: FlashAttentionLevel,
  layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toQueries = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "c_q_proj")
  var queries = toQueries(x).reshaped(
    .NHWC(batchSize, tokenLength, numberOfHeads, headDimension))
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  queries = normQ(queries)
  let attentionKeys = keys.to(of: queries)
  let attentionValues = values.to(of: queries)
  let out: Model.IO
  if usesFlashAttention == .scale1 {
    let attentionScale = 1.0 / Float(headDimension).squareRoot().squareRoot()
    queries = attentionScale * queries
  }
  switch usesFlashAttention {
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, attentionKeys, attentionValues)
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1 / Float(headDimension).squareRoot(), flags: [.Float16])
    out = scaledDotProductAttention(queries, attentionKeys, attentionValues)
  case .none:
    let query = queries.transposed(1, 2).contiguous()
    let key = attentionKeys.transposed(1, 2).contiguous()
    let value = attentionValues.transposed(1, 2).contiguous()
    var dot =
      Matmul(transposeB: (2, 3))(
        (1.0 / Float(headDimension).squareRoot()) * query, key)
    dot = dot.reshaped([batchSize * numberOfHeads * tokenLength, contextLength]).softmax()
    dot = dot.reshaped([batchSize, numberOfHeads, tokenLength, contextLength])
    out = (dot * value).transposed(1, 2)
  }
  let mergedOut = out.reshaped([batchSize, tokenLength, hiddenSize])
  let unifyHeads = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "c_out_proj")
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).q_proj.weight"] = [toQueries.weight.name]
      mapping["\(prefix.0).q_norm.weight"] = [normQ.weight.name]
      mapping["\(prefix.0).output_proj.weight"] = [unifyHeads.weight.name]
    case .diffusers:
      mapping["\(prefix.1).to_q.weight"] = [toQueries.weight.name]
      mapping["\(prefix.1).norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).to_out.0.weight"] = [unifyHeads.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x, keys, values], [unifyHeads(mergedOut).to(of: x)]))
}

private func LoRACosmosFeedForward(
  prefix: (String, String), hiddenSize: Int, mlpRatio: Int, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let fc1 = LoRADense(
    count: hiddenSize * mlpRatio, configuration: configuration, noBias: true, index: layerIndex,
    name: "fc1")
  let fc2 = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex, name: "fc2")
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).layer1.weight"] = [fc1.weight.name]
      mapping["\(prefix.0).layer2.weight"] = [fc2.weight.name]
    case .diffusers:
      mapping["\(prefix.1).net.0.proj.weight"] = [fc1.weight.name]
      mapping["\(prefix.1).net.2.weight"] = [fc2.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x], [fc2(fc1(x).GELU())]))
}

private func LoRACosmosTransformerBlock(
  prefix: (String, String), batchSize: Int, tokenLength: Int, contextLength: Int, hiddenSize: Int,
  hiddenFeatures: Int, headDimension: Int, numberOfHeads: Int, mlpRatio: Int,
  usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let norm1Shift = Input()
  let norm1Scale = Input()
  let norm1Gate = Input()
  let norm2Shift = Input()
  let norm2Scale = Input()
  let norm2Gate = Input()
  let norm3Shift = Input()
  let norm3Scale = Input()
  let norm3Gate = Input()
  let contextKeys = Input()
  let contextValues = Input()
  let rot = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let norm1Out = norm1Scale.to(of: x) .* norm1(x) + norm1Shift.to(of: x)
  let (selfAttentionMapper, selfAttention) = LoRACosmosSelfAttention(
    prefix: ("\(prefix.0).self_attn", "\(prefix.1).attn1"), batchSize: batchSize,
    tokenLength: tokenLength,
    hiddenSize: hiddenSize, headDimension: headDimension, numberOfHeads: numberOfHeads,
    usesFlashAttention: usesFlashAttention, layerIndex: layerIndex, configuration: configuration)
  var out =
    x
    + norm1Gate.to(of: x) .* selfAttention(norm1Out.to(.Float16), rot).to(of: x)
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let norm2Out = norm2Scale.to(of: out) .* norm2(out) + norm2Shift.to(of: out)
  let (crossAttentionMapper, crossAttention) = LoRACosmosCrossAttention(
    prefix: ("\(prefix.0).cross_attn", "\(prefix.1).attn2"), batchSize: batchSize,
    tokenLength: tokenLength,
    contextLength: contextLength, hiddenSize: hiddenSize, headDimension: headDimension,
    numberOfHeads: numberOfHeads, usesFlashAttention: usesFlashAttention, layerIndex: layerIndex,
    configuration: configuration)
  out =
    out
    + norm2Gate.to(of: out)
    .* crossAttention(norm2Out.to(.Float16), contextKeys, contextValues).to(of: out)
  let norm3 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let norm3Out = norm3Scale.to(of: out) .* norm3(out) + norm3Shift.to(of: out)
  let (feedForwardMapper, feedForward) = LoRACosmosFeedForward(
    prefix: ("\(prefix.0).mlp", "\(prefix.1).ff"), hiddenSize: hiddenSize, mlpRatio: mlpRatio,
    layerIndex: layerIndex, configuration: configuration)
  out =
    out
    + norm3Gate.to(of: out) .* feedForward(norm3Out.to(.Float16)).to(of: out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(selfAttentionMapper(format)) { v, _ in v }
    mapping.merge(crossAttentionMapper(format)) { v, _ in v }
    mapping.merge(feedForwardMapper(format)) { v, _ in v }
    return mapping
  }
  return (
    mapper,
    Model(
      [
        x, norm1Shift, norm1Scale, norm1Gate, norm2Shift, norm2Scale, norm2Gate, norm3Shift,
        norm3Scale, norm3Gate, contextKeys, contextValues, rot,
      ], [out])
  )
}

public func LoRACosmos(
  batchSize: Int, height: Int, width: Int, textLength: Int, inChannels: Int = 16,
  outChannels: Int = 16, layers: Int = 28, numberOfHeads: Int = 16,
  headDimension: Int = 128,
  mlpRatio: Int = 4, adaLNDimension: Int = 256, patchSize: (height: Int, width: Int) = (2, 2),
  usesFlashAttention: FlashAttentionLevel = .scale1, LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, ModelWeightMapper) {
  let hiddenSize = numberOfHeads * headDimension
  let projOutChannels = patchSize.height * patchSize.width * outChannels
  let tokenHeight = height / patchSize.height
  let tokenWidth = width / patchSize.width
  let tokenLength = tokenHeight * tokenWidth
  let hiddenStates = Input()
  let rot = Input()
  let blockModulations = (0..<layers).map { _ in
    (
      shift1: Input(), scale1: Input(), gate1: Input(),
      shift2: Input(), scale2: Input(), gate2: Input(),
      shift3: Input(), scale3: Input(), gate3: Input()
    )
  }
  let normOutShift = Input()
  let normOutScale = Input()
  let crossAttentionKVs = (0..<layers).map { _ in (Input(), Input()) }
  let zeroMask =
    (0 * hiddenStates.reduced(.mean, axis: [3])).reshaped(
      .NHWC(batchSize, height, width, 1))
  let paddedHiddenStates = Functional.concat(axis: 3, hiddenStates, zeroMask)
  let patchEmbed = LoRAConvolution(
    groups: 1, filters: hiddenSize, filterSize: [patchSize.height, patchSize.width],
    configuration: LoRAConfiguration, noBias: true,
    hint: Hint(stride: [patchSize.height, patchSize.width]), format: .OIHW, name: "x_embedder")
  var out = patchEmbed(paddedHiddenStates).reshaped(
    .HWC(batchSize, tokenLength, hiddenSize)
  ).to(.Float32)
  var blockMappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = LoRACosmosTransformerBlock(
      prefix: ("blocks.\(i)", "transformer_blocks.\(i)"), batchSize: batchSize,
      tokenLength: tokenLength,
      contextLength: textLength, hiddenSize: hiddenSize, hiddenFeatures: adaLNDimension,
      headDimension: headDimension, numberOfHeads: numberOfHeads, mlpRatio: mlpRatio,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    let modulation = blockModulations[i]
    let kv = crossAttentionKVs[i]
    out = block(
      out, modulation.shift1, modulation.scale1, modulation.gate1, modulation.shift2,
      modulation.scale2, modulation.gate2, modulation.shift3, modulation.scale3, modulation.gate3,
      kv.0, kv.1, rot)
    blockMappers.append(mapper)
  }
  let normOut = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let normOutInput = out.to(of: hiddenStates)
  out = normOutScale.to(of: normOutInput) .* normOut(normOutInput)
    + normOutShift.to(of: normOutInput)
  out = out.to(of: hiddenStates)
  let projOut = LoRADense(
    count: projOutChannels, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "proj_out")
  out = projOut(out).reshaped(
    [
      batchSize, tokenHeight, tokenWidth, patchSize.height, patchSize.width, outChannels,
    ]).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped(
      [batchSize, height, width, outChannels])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["x_embedder.proj.1.weight"] = [patchEmbed.weight.name]
    case .diffusers:
      mapping["patch_embed.proj.weight"] = [patchEmbed.weight.name]
    }
    for blockMapper in blockMappers {
      mapping.merge(blockMapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["final_layer.linear.weight"] = [projOut.weight.name]
    case .diffusers:
      mapping["proj_out.weight"] = [projOut.weight.name]
    }
    return mapping
  }
  var inputs: [Input] = [hiddenStates, rot]
  for modulation in blockModulations {
    inputs.append(contentsOf: [
      modulation.shift1, modulation.scale1, modulation.gate1,
      modulation.shift2, modulation.scale2, modulation.gate2,
      modulation.shift3, modulation.scale3, modulation.gate3,
    ])
  }
  inputs.append(contentsOf: [normOutShift, normOutScale])
  for kv in crossAttentionKVs {
    inputs.append(contentsOf: [kv.0, kv.1])
  }
  return (Model(inputs, [out], trainable: false), mapper)
}

private func LoRACosmosAdaLayerNormZeroFixed(
  prefix: (String, String), timesteps: Int, hiddenSize: Int, hiddenFeatures: Int, name: String,
  layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let embeddedTimestep = Input()
  let tembShift = Input()
  let tembScale = Input()
  let tembGate = Input()
  let linear1 = LoRADense(
    count: hiddenFeatures, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_linear_1")
  let shift = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_shift")
  let scale = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_scale")
  let gate = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_gate")
  let hidden = linear1(embeddedTimestep.swish())
  let shiftOut = shift(hidden) + tembShift
  let scaleOut = scale(hidden) + tembScale
  let gateOut = gate(hidden) + tembGate
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let linear2 = ModelWeightElement(
      [shift.weight.name, scale.weight.name, gate.weight.name],
      offsets: [0, hiddenSize, hiddenSize * 2])
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).1.weight"] = [linear1.weight.name]
      mapping["\(prefix.0).2.weight"] = linear2
    case .diffusers:
      mapping["\(prefix.1).linear_1.weight"] = [linear1.weight.name]
      mapping["\(prefix.1).linear_2.weight"] = linear2
    }
    return mapping
  }
  return (
    mapper,
    Model(
      [embeddedTimestep, tembShift, tembScale, tembGate],
      [
        shiftOut.reshaped([timesteps, 1, hiddenSize]),
        scaleOut.reshaped([timesteps, 1, hiddenSize]),
        gateOut.reshaped([timesteps, 1, hiddenSize]),
      ])
  )
}

private func LoRACosmosAdaLayerNormFixed(
  prefix: (String, String), timesteps: Int, hiddenSize: Int, hiddenFeatures: Int, name: String,
  layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let embeddedTimestep = Input()
  let tembShift = Input()
  let tembScale = Input()
  let linear1 = LoRADense(
    count: hiddenFeatures, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_linear_1")
  let shift = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_shift")
  let scale = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_scale")
  let hidden = linear1(embeddedTimestep.swish())
  let shiftOut = shift(hidden) + tembShift
  let scaleOut = scale(hidden) + tembScale + 1
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let linear2 = ModelWeightElement(
      [shift.weight.name, scale.weight.name], offsets: [0, hiddenSize])
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).1.weight"] = [linear1.weight.name]
      mapping["\(prefix.0).2.weight"] = linear2
    case .diffusers:
      mapping["\(prefix.1).linear_1.weight"] = [linear1.weight.name]
      mapping["\(prefix.1).linear_2.weight"] = linear2
    }
    return mapping
  }
  return (
    mapper,
    Model(
      [embeddedTimestep, tembShift, tembScale],
      [
        shiftOut.reshaped([timesteps, 1, hiddenSize]),
        scaleOut.reshaped([timesteps, 1, hiddenSize]),
      ])
  )
}

private func LoRACosmosCrossAttentionFixed(
  prefix: (String, String), batchSize: Int, contextLength: Int, hiddenSize: Int, headDimension: Int,
  numberOfHeads: Int, usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let toKeys = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "c_k_proj")
  let toValues = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "c_v_proj")
  var keys = toKeys(context).reshaped(
    .NHWC(batchSize, contextLength, numberOfHeads, headDimension))
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  keys = normK(keys)
  if usesFlashAttention == .scale1 {
    let attentionScale = 1.0 / Float(headDimension).squareRoot().squareRoot()
    keys = attentionScale * keys
  }
  let values = toValues(context).reshaped(
    .NHWC(batchSize, contextLength, numberOfHeads, headDimension))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).k_proj.weight"] = [toKeys.weight.name]
      mapping["\(prefix.0).k_norm.weight"] = [normK.weight.name]
      mapping["\(prefix.0).v_proj.weight"] = [toValues.weight.name]
    case .diffusers:
      mapping["\(prefix.1).to_k.weight"] = [toKeys.weight.name]
      mapping["\(prefix.1).norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).to_v.weight"] = [toValues.weight.name]
    }
    return mapping
  }
  return (mapper, Model([context], [keys, values]))
}

public func LoRACosmosFixed(
  timesteps: Int, batchSize: Int, textLength: Int, layers: Int = 28,
  numberOfHeads: Int = 16, headDimension: Int = 128, adaLNDimension: Int = 256,
  usesFlashAttention: FlashAttentionLevel = .scale1, LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, ModelWeightMapper) {
  let hiddenSize = numberOfHeads * headDimension
  let context = Input()
  let timestepProjection = Input()
  let timeNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "time_norm")
  let timeLinear1 = LoRADense(
    count: hiddenSize, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "time_linear_1")
  let timeShift = LoRADense(
    count: hiddenSize, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "time_shift")
  let timeScale = LoRADense(
    count: hiddenSize, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "time_scale")
  let timeGate = LoRADense(
    count: hiddenSize, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "time_gate")
  let embeddedTimestep = timeNorm(timestepProjection)
  let timeHidden = timeLinear1(timestepProjection).swish()
  let tembShift = timeShift(timeHidden)
  let tembScale = timeScale(timeHidden) + 1
  let tembGate = timeGate(timeHidden)
  var outputs = [Model.IO]()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (norm1Mapper, norm1) = LoRACosmosAdaLayerNormZeroFixed(
      prefix: ("blocks.\(i).adaln_modulation_self_attn", "transformer_blocks.\(i).norm1"),
      timesteps: timesteps, hiddenSize: hiddenSize,
      hiddenFeatures: adaLNDimension, name: "norm1", layerIndex: i,
      configuration: LoRAConfiguration)
    let norm1Out = norm1(embeddedTimestep, tembShift, tembScale, tembGate)
    outputs.append(norm1Out[0])
    outputs.append(norm1Out[1])
    outputs.append(norm1Out[2])
    mappers.append(norm1Mapper)
    let (norm2Mapper, norm2) = LoRACosmosAdaLayerNormZeroFixed(
      prefix: ("blocks.\(i).adaln_modulation_cross_attn", "transformer_blocks.\(i).norm2"),
      timesteps: timesteps, hiddenSize: hiddenSize,
      hiddenFeatures: adaLNDimension, name: "norm2", layerIndex: i,
      configuration: LoRAConfiguration)
    let norm2Out = norm2(embeddedTimestep, tembShift, tembScale, tembGate)
    outputs.append(norm2Out[0])
    outputs.append(norm2Out[1])
    outputs.append(norm2Out[2])
    mappers.append(norm2Mapper)
    let (norm3Mapper, norm3) = LoRACosmosAdaLayerNormZeroFixed(
      prefix: ("blocks.\(i).adaln_modulation_mlp", "transformer_blocks.\(i).norm3"),
      timesteps: timesteps, hiddenSize: hiddenSize,
      hiddenFeatures: adaLNDimension, name: "norm3", layerIndex: i,
      configuration: LoRAConfiguration)
    let norm3Out = norm3(embeddedTimestep, tembShift, tembScale, tembGate)
    outputs.append(norm3Out[0])
    outputs.append(norm3Out[1])
    outputs.append(norm3Out[2])
    mappers.append(norm3Mapper)
  }
  let (normOutMapper, normOut) = LoRACosmosAdaLayerNormFixed(
    prefix: ("final_layer.adaln_modulation", "norm_out"), timesteps: timesteps,
    hiddenSize: hiddenSize,
    hiddenFeatures: adaLNDimension, name: "norm_out", layerIndex: 0,
    configuration: LoRAConfiguration)
  let normOutOut = normOut(embeddedTimestep, tembShift, tembScale)
  outputs.append(normOutOut[0])
  outputs.append(normOutOut[1])
  mappers.append(normOutMapper)
  for i in 0..<layers {
    let (crossAttentionMapper, crossAttention) = LoRACosmosCrossAttentionFixed(
      prefix: ("blocks.\(i).cross_attn", "transformer_blocks.\(i).attn2"),
      batchSize: batchSize, contextLength: textLength,
      hiddenSize: hiddenSize, headDimension: headDimension, numberOfHeads: numberOfHeads,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    let crossAttentionOut = crossAttention(context)
    outputs.append(crossAttentionOut[0])
    outputs.append(crossAttentionOut[1])
    mappers.append(crossAttentionMapper)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let timeLinear2 = ModelWeightElement(
      [timeShift.weight.name, timeScale.weight.name, timeGate.weight.name],
      offsets: [0, hiddenSize, hiddenSize * 2])
    switch format {
    case .generativeModels:
      mapping["t_embedding_norm.weight"] = [timeNorm.weight.name]
      mapping["t_embedder.1.linear_1.weight"] = [timeLinear1.weight.name]
      mapping["t_embedder.1.linear_2.weight"] = timeLinear2
    case .diffusers:
      mapping["time_embed.norm.weight"] = [timeNorm.weight.name]
      mapping["time_embed.t_embedder.linear_1.weight"] = [timeLinear1.weight.name]
      mapping["time_embed.t_embedder.linear_2.weight"] = timeLinear2
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (Model([context, timestepProjection], outputs, trainable: false), mapper)
}
