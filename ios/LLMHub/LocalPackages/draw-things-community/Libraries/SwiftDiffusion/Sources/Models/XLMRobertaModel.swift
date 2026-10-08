import NNC

public func XLMRobertaTextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, vocabularySize: Int, maxLength: Int, tokenTypes: Int,
  embeddingSize: Int
) -> Model {
  let tokens = Input()
  let tokenType = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize)
  let tokenTypeEmbed = Embedding(
    T.self, vocabularySize: tokenTypes, embeddingSize: embeddingSize)
  let positionEmbed = Embedding(
    T.self, vocabularySize: maxLength, embeddingSize: embeddingSize)
  let embedding = tokenEmbed(tokens) + tokenTypeEmbed(tokenType) + positionEmbed(positions)
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  let out = layerNorm(embedding)
  return Model([tokens, positions, tokenType], [out])
}

private func XLMRobertaSelfAttention(prefix: String, k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return Model([x, causalAttentionMask], [out])
}

private func XLMRobertaLayer(prefix: String, k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let causalAttentionMask = Input()
  let selfAttention = XLMRobertaSelfAttention(
    prefix: "\(prefix).attention", k: k, h: h, b: b, t: t)
  var out = selfAttention(x, causalAttentionMask)
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNorm(out + x)
  let intermediate = Dense(count: k * h * 4)
  let ff = out
  out = intermediate(out).GELU()
  let output = Dense(count: k * h)
  out = output(out)
  let layerNormFinal = LayerNorm(epsilon: 1e-5, axis: [1])
  out = layerNormFinal(out + ff)
  return Model([x, causalAttentionMask], [out])
}

public func XLMRobertaModel(numberOfLayers: Int, k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let causalAttentionMask = Input()
  var out: Model.IO = x
  for i in 0..<numberOfLayers {
    let layer = XLMRobertaLayer(
      prefix: "model.transformer.encoder.layer.\(i)", k: k, h: h, b: b, t: t)
    out = layer(out, causalAttentionMask)
  }
  return Model([x, causalAttentionMask], [out])
}
