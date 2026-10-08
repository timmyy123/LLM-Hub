import DiffusionMappings
import Foundation
import NNC

public func GLMRotaryEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  sequenceLength: Int, of dataType: FloatType.Type = FloatType.self
) -> Tensor<FloatType> {
  var rotary = Tensor<FloatType>(.CPU, .NHWC(1, sequenceLength, 1, 128))
  for i in 0..<sequenceLength {
    for k in 0..<32 {
      let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 64)
      let sintheta = sin(theta)
      let costheta = cos(theta)
      rotary[0, i, 0, k * 2] = FloatType(costheta)
      rotary[0, i, 0, k * 2 + 1] = FloatType(sintheta)
    }
    for k in 32..<64 {
      rotary[0, i, 0, k * 2] = 1
      rotary[0, i, 0, k * 2 + 1] = 0
    }
  }
  return rotary
}

private func SelfAttention(
  prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, usesFlashAttention: Bool
) -> (
  Model, ModelWeightMapper
) {
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * hk, name: "v_proj")
  var keys = tokeys(x).reshaped([b, t, hk, k])
  var queries = toqueries(x).reshaped([b, t, h, k])
  var values = tovalues(x).reshaped([b, t, hk, k])
  var out: Model.IO
  if usesFlashAttention {
    queries = Functional.cmul(left: queries, right: rot)
    keys = Functional.cmul(left: keys, right: rot)
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), hasAttentionMask: true)
    out = scaledDotProductAttention(queries, keys, values, causalAttentionMask).reshaped([
      b * t, h * k,
    ])
  } else {
    if h > hk {
      keys = Concat(axis: 3)(Array(repeating: keys, count: h / hk))
      values = Concat(axis: 3)(Array(repeating: values, count: h / hk))
    }
    keys = keys.reshaped([b, t, h, k])
    queries = Functional.cmul(left: queries, right: rot)
    keys = Functional.cmul(left: keys, right: rot)
    keys = keys.transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
    values = values.reshaped([b, t, h, k]).transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
    dot = dot.reshaped([b * h * t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t, t])
    out = dot * values
    out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  }
  let unifyheads = Dense(count: k * h, noBias: true, name: "o")
  out = unifyheads(out)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).self_attention.query_key_value.weight"] = [
      toqueries.weight.name, tokeys.weight.name, tovalues.weight.name,
    ]
    mapping["\(prefix).self_attention.query_key_value.bias"] = [
      toqueries.bias.name, tokeys.bias.name, tovalues.bias.name,
    ]
    mapping["\(prefix).self_attention.dense.weight"] = [unifyheads.weight.name]
    return mapping
  }
  return (Model([x, rot, causalAttentionMask], [out]), mapper)
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true)
  let w3 = Dense(count: intermediateSize, noBias: true)
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true)
  out = w2(out)
  return (w1, w2, w3, Model([x], [out], name: name))
}

private func GLMTransformerBlock(
  prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int, MLP: Int, usesFlashAttention: Bool
)
  -> (
    Model, ModelWeightMapper
  )
{
  let x = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let norm1 = RMSNorm(epsilon: 1e-5, axis: [1], name: "attention_norm")
  var out = norm1(x)
  let (attention, attentionMapper) = SelfAttention(
    prefix: prefix, k: k, h: h, hk: hk, b: b, t: t, usesFlashAttention: usesFlashAttention)
  out = attention(out, rot, causalAttentionMask) + x
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-5, axis: [1], name: "ffn_norm")
  out = norm2(out)
  let (w1, w2, w3, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "ffn")
  out = residual + ffn(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).input_layernorm.weight"] = [norm1.weight.name]
    mapping.merge(attentionMapper(format)) { v, _ in v }
    mapping["\(prefix).post_attention_layernorm.weight"] = [norm2.weight.name]
    mapping["\(prefix).mlp.dense_h_to_4h.weight"] = [w1.weight.name, w3.weight.name]
    mapping["\(prefix).mlp.dense_4h_to_h.weight"] = [w2.weight.name]
    return mapping
  }
  return (Model([x, rot, causalAttentionMask], [out]), mapper)
}

private func TextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, embeddingSize: Int
) -> (Model, ModelWeightMapper) {
  let tokens = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "word_embeddings")
  let embedding = tokenEmbed(tokens)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["embedding.word_embeddings.weight"] = [tokenEmbed.weight.name]
    return mapping
  }
  return (Model([tokens], [embedding]), mapper)
}

public func GLMTransformer<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: Int,
  layers: Int, MLP: Int, heads: Int, batchSize: Int,
  outputPenultimate: Bool, applyFinalNorm: Bool, usesFlashAttention: Bool
) -> (Model, ModelWeightMapper) {
  let tokens = Input()
  let rot = Input()
  let causalAttentionMask = Input()
  let (embedding, embeddingMapper) = TextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, embeddingSize: width)
  var out = embedding(tokens)
  var mappers = [ModelWeightMapper]()
  var penultimate: Model.IO? = nil
  for i in 0..<layers {
    if i == layers - 1 && outputPenultimate {
      penultimate = out
    }
    let (layer, mapper) = GLMTransformerBlock(
      prefix: "encoder.layers.\(i)", k: width / heads, h: heads, hk: heads / 16, b: batchSize,
      t: tokenLength, MLP: MLP, usesFlashAttention: usesFlashAttention)
    out = layer(out, rot, causalAttentionMask)
    mappers.append(mapper)
  }
  let finalNorm: Model?
  if applyFinalNorm {
    let norm = RMSNorm(epsilon: 1e-5, axis: [1], name: "norm")
    out = norm(out)
    finalNorm = norm
  } else {
    finalNorm = nil
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(embeddingMapper(format)) { v, _ in v }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    if let finalNorm = finalNorm {
      mapping["encoder.final_layernorm.weight"] = [finalNorm.weight.name]
    }
    return mapping
  }
  if let penultimate = penultimate {
    return (Model([tokens, rot, causalAttentionMask], [penultimate, out]), mapper)
  } else {
    return (Model([tokens, rot, causalAttentionMask], [out]), mapper)
  }
}
