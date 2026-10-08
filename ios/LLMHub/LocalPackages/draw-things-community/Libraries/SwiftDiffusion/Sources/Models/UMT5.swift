import DiffusionMappings
import Foundation
import NNC

private func UMT5TextEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  vocabularySize: Int, embeddingSize: Int, name: String, of: FloatType.Type = FloatType.self
) -> Model {
  let tokenEmbed = Embedding(
    FloatType.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: name)
  return tokenEmbed
}

private func UMT5LayerSelfAttention(k: Int, h: Int, b: Int, t: Int, outFeatures: Int, upcast: Bool)
  -> (
    Model, Model, Model, Model, Model
  )
{
  let x = Input()
  let positionBias = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, noBias: true, name: "q")
  let tovalues = Dense(count: k * h, noBias: true, name: "v")
  let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
  // No scaling the queries.
  let queries = toqueries(x).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + positionBias
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: outFeatures, noBias: true, name: "o")
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  out = unifyheads(out)
  return (tokeys, toqueries, tovalues, unifyheads, Model([x, positionBias], [out]))
}

private func UMT5DenseGatedActDense(hiddenSize: Int, intermediateSize: Int) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let wi_0 = Dense(count: intermediateSize, noBias: true, name: "w0")
  let wi_1 = Dense(count: intermediateSize, noBias: true, name: "w1")
  var out = wi_1(x).to(.Float32) .* wi_0(x).GELU(approximate: .tanh).to(.Float32)
  let wo = Dense(count: hiddenSize, noBias: true, name: "wo")
  let scaleFactor: Float = 8
  out = scaleFactor * wo(((1 / scaleFactor) * out).to(of: x)).to(.Float32)
  return (wi_0, wi_1, wo, Model([x], [out]))
}

private func UMT5Block(
  prefix: String, k: Int, h: Int, b: Int, t: Int, outFeatures: Int, intermediateSize: Int,
  upcast: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let relativePositionEmbedding = Embedding(
    FloatType.self, vocabularySize: 32, embeddingSize: h, name: "relative_position_embedding")
  let positionBias =
    relativePositionEmbedding(relativePositionBuckets).reshaped([1, t, t, h])
    .permuted(0, 3, 1, 2) + attentionMask
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (tokeys, toqueries, tovalues, unifyheads, attention) = UMT5LayerSelfAttention(
    k: k, h: h, b: b, t: t, outFeatures: outFeatures, upcast: upcast)
  var out: Model.IO
  if upcast {
    let scaleFactor: Float = 8
    out = x + scaleFactor * attention(norm1(x).to(FloatType.dataType), positionBias).to(of: x)
  } else {
    out = x + attention(norm1(x).to(FloatType.dataType), positionBias).to(of: x)
  }
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [1], name: "norm1")
  let (wi_0, wi_1, wo, ff) = UMT5DenseGatedActDense(
    hiddenSize: outFeatures, intermediateSize: intermediateSize)
  out = out + ff(norm2(out).to(FloatType.dataType))
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).layer.0.layer_norm.weight"] = [norm1.weight.name]
    mapping["\(prefix).layer.0.SelfAttention.relative_attention_bias.weight"] = [
      relativePositionEmbedding.weight.name
    ]
    mapping["\(prefix).layer.0.SelfAttention.k.weight"] = [tokeys.weight.name]
    mapping["\(prefix).layer.0.SelfAttention.q.weight"] = [toqueries.weight.name]
    mapping["\(prefix).layer.0.SelfAttention.v.weight"] = [tovalues.weight.name]
    mapping["\(prefix).layer.0.SelfAttention.o.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).layer.1.layer_norm.weight"] = [norm2.weight.name]
    mapping["\(prefix).layer.1.DenseReluDense.wi_0.weight"] = [wi_0.weight.name]
    mapping["\(prefix).layer.1.DenseReluDense.wi_1.weight"] = [wi_1.weight.name]
    mapping["\(prefix).layer.1.DenseReluDense.wo.weight"] = [wo.weight.name]
    return mapping
  }
  return (mapper, Model([x, attentionMask, relativePositionBuckets], [out]))
}

func UMT5ForConditionalGeneration<FloatType: TensorNumeric & BinaryFloatingPoint>(
  b: Int, t: Int, vocabularySize: Int, channels: Int, intermediateSize: Int, upcast: Bool,
  of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let attentionMask = Input()
  let relativePositionBuckets = Input()
  let textEmbed = UMT5TextEmbedding(
    vocabularySize: vocabularySize, embeddingSize: channels, name: "shared", of: FloatType.self)
  var out = textEmbed(x).to(.Float32)
  var mappers = [ModelWeightMapper]()
  for i in 0..<24 {
    let (mapper, block) = UMT5Block(
      prefix: "encoder.block.\(i)", k: 64, h: channels / 64, b: b, t: t, outFeatures: channels,
      intermediateSize: intermediateSize, upcast: upcast ? i >= 12 : false)
    out = block(out, attentionMask, relativePositionBuckets)
    mappers.append(mapper)
  }
  let finalNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "final_norm")
  out = finalNorm(out).to(FloatType.dataType)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["shared.weight"] = [textEmbed.weight.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["encoder.final_layer_norm.weight"] = [finalNorm.weight.name]
    return mapping
  }
  return (mapper, Model([x, attentionMask, relativePositionBuckets], [out]))
}
