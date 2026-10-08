import NNC

private func EvaSelfAttention(
  prefix: String, k: Int, h: Int, b: Int, t: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  if usesFlashAttention {
    let queries = toqueries(x).reshaped([b, t, h, k]).identity().identity()
    let keys = tokeys(x).reshaped([b, t, h, k]).identity()
    let values = tovalues(x).reshaped([b, t, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), multiHeadOutputProjectionFused: true)
    let out = scaledDotProductAttention(queries, keys, values).reshaped([
      b * t, h * k,
    ])
    return Model([x], [out])
  } else {
    let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
      .transposed(1, 2)
    let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys)
    dot = dot.reshaped([b * h * t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t, t])
    var out = dot * values
    out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
    let unifyheads = Dense(count: k * h)
    out = unifyheads(out)
    return Model([x], [out])
  }
}

private func EvaResidualAttentionBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, MLP: Int, usesFlashAttention: Bool
)
  -> Model
{
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-6, axis: [2])
  let attention = EvaSelfAttention(
    prefix: prefix, k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-6, axis: [1])
  let fc = Dense(count: MLP)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  return Model([x], [out])
}

func EvaVisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type,
  grid: Int, width: Int, MLP: Int, layers: Int, heads: Int, batchSize: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], hint: Hint(stride: [14, 14]), format: .OIHW)
  var out = conv1(x).reshaped([batchSize, grid * grid, width])
  let classEmbedding = Parameter<T>(.GPU(0), .HWC(1, 1, width))
  let positionalEmbedding = Parameter<T>(.GPU(0), .HWC(1, grid * grid + 1, width))
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  for i in 0..<layers {
    let block = EvaResidualAttentionBlock(
      prefix: "blocks.\(i)",
      k: width / heads, h: heads, b: batchSize, t: grid * grid + 1, MLP: MLP,
      usesFlashAttention: usesFlashAttention)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
  }
  return Model([x], [out])
}

private func BertSelfAttention(
  prefix: String, k: Int, h: Int, b: Int, t: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  if usesFlashAttention {
    let queries = toqueries(x).reshaped([b, t, h, k]).identity().identity()
    let keys = tokeys(x).reshaped([b, t, h, k]).identity()
    let values = tovalues(x).reshaped([b, t, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), multiHeadOutputProjectionFused: true)
    let out = scaledDotProductAttention(queries, keys, values).reshaped([
      b * t, h * k,
    ])
    return Model([x], [out])
  } else {
    let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
      .transposed(1, 2)
    let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys)
    dot = dot.reshaped([b * h * t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t, t])
    var out = dot * values
    out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
    let unifyheads = Dense(count: k * h)
    out = unifyheads(out)
    return Model([x], [out])
  }
}

private func BertCrossAttention(
  prefix: String, k: Int, h: Int, b: Int, queryEmbeddingLength: Int, imageEmbeddingLength: Int,
  usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let c = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  if usesFlashAttention {
    let queries = toqueries(x).reshaped([b, queryEmbeddingLength, h, k]).identity().identity()
    let keys = tokeys(c).reshaped([b, imageEmbeddingLength, h, k]).identity()
    let values = tovalues(c).reshaped([b, imageEmbeddingLength, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), multiHeadOutputProjectionFused: true)
    let out = scaledDotProductAttention(queries, keys, values).reshaped([
      b * queryEmbeddingLength, h * k,
    ])
    return Model([x, c], [out])
  } else {
    let keys = tokeys(c).reshaped([b, imageEmbeddingLength, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([
      b, queryEmbeddingLength, h, k,
    ])
    .transposed(1, 2)
    let values = tovalues(c).reshaped([b, imageEmbeddingLength, h, k]).transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys)
    dot = dot.reshaped([b * h * queryEmbeddingLength, imageEmbeddingLength])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, queryEmbeddingLength, imageEmbeddingLength])
    var out = dot * values
    out = out.reshaped([b, h, queryEmbeddingLength, k]).transposed(1, 2).reshaped([
      b * queryEmbeddingLength, h * k,
    ])
    let unifyheads = Dense(count: k * h)
    out = unifyheads(out)
    return Model([x, c], [out])
  }
}

private func BertLayer(
  prefix: String, k: Int, h: Int, b: Int, queryEmbeddingLength: Int, imageEmbeddingLength: Int,
  MLP: Int, hasCrossAttention: Bool, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let c: Model.IO?
  let attention1 = BertSelfAttention(
    prefix: prefix, k: k, h: h, b: b, t: queryEmbeddingLength,
    usesFlashAttention: usesFlashAttention)
  var out = x.reshaped([b * queryEmbeddingLength, h * k]) + attention1(x)
  let ln1 = LayerNorm(epsilon: 1e-12, axis: [1])
  out = ln1(out)
  if hasCrossAttention {
    let lc = Input()
    let attention2 = BertCrossAttention(
      prefix: prefix, k: k, h: h, b: b, queryEmbeddingLength: queryEmbeddingLength,
      imageEmbeddingLength: imageEmbeddingLength, usesFlashAttention: usesFlashAttention)
    out = out.reshaped([b * queryEmbeddingLength, h * k]) + attention2(out, lc)
    let ln2 = LayerNorm(epsilon: 1e-12, axis: [1])
    out = ln2(out)
    c = lc
  } else {
    c = nil
  }
  let fc = Dense(count: MLP)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  let ln3 = LayerNorm(epsilon: 1e-12, axis: [1])
  out = ln3(out + proj(gelu(fc(out))))
  if let c = c {
    return Model([x, c], [out])
  } else {
    return Model([x], [out])
  }
}

func BertModel(
  width: Int, queryEmbeddingLength: Int, imageEmbeddingLength: Int, MLP: Int, layers: Int,
  heads: Int, batchSize: Int, crossAttentionFreq: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let c = Input()
  let ln = LayerNorm(epsilon: 1e-12, axis: [1])
  var out = ln(x)
  for i in 0..<layers {
    let hasCrossAttention = (i % crossAttentionFreq) == 0
    let layer = BertLayer(
      prefix: "encoder.layer.\(i)", k: width / heads, h: heads, b: batchSize,
      queryEmbeddingLength: queryEmbeddingLength, imageEmbeddingLength: imageEmbeddingLength,
      MLP: MLP, hasCrossAttention: hasCrossAttention, usesFlashAttention: usesFlashAttention)
    if hasCrossAttention {
      out = layer(out, c)
    } else {
      out = layer(out)
    }
  }
  return Model([x, c], [out])
}

private func OPTSelfAttention(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), usesFlashAttention: Bool,
  injectKeysAndValues: Bool
) -> Model {
  let x = Input()
  let tokeys = Dense(count: k * h, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * h, name: "v_proj")
  let causalAttentionMask = Input()
  let kIn: Input?
  let kOut: Model.IO?
  let vIn: Input?
  let vOut: Model.IO?
  let keys: Model.IO
  let values: Model.IO
  if usesFlashAttention {
    if injectKeysAndValues {
      let kIn0 = Input()
      let vIn0 = Input()
      kIn = kIn0
      vIn = vIn0
      let kOut0 = Functional.concat(axis: 0, kIn0, tokeys(x))
      let vOut0 = Functional.concat(axis: 0, vIn0, tovalues(x))
      keys = kOut0.reshaped([b, t.0, h, k])
      values = vOut0.reshaped([b, t.0, h, k])
      kOut = kOut0
      vOut = vOut0
    } else {
      kIn = nil
      kOut = nil
      vIn = nil
      vOut = nil
      keys = tokeys(x).reshaped([b, t.0, h, k])
      values = tovalues(x).reshaped([b, t.0, h, k])
    }
    let queries = toqueries(x).reshaped([b, t.1, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), isCausal: true, hasAttentionMask: true,
      multiHeadOutputProjectionFused: true, name: "out_proj")
    let out = scaledDotProductAttention(queries, keys, values, causalAttentionMask).reshaped([
      b * t.1, h * k,
    ])
    if let kIn = kIn, let kOut = kOut, let vIn = vIn, let vOut = vOut {
      return Model([x, causalAttentionMask, kIn, vIn], [out, kOut, vOut])
    } else {
      return Model([x, causalAttentionMask], [out])
    }
  } else {
    if injectKeysAndValues {
      let kIn0 = Input()
      let vIn0 = Input()
      kIn = kIn0
      vIn = vIn0
      let kOut0 = Functional.concat(axis: 0, kIn0, tokeys(x))
      let vOut0 = Functional.concat(axis: 0, vIn0, tovalues(x))
      keys = kOut0.reshaped([b, t.0, h, k]).transposed(1, 2)
      values = vOut0.reshaped([b, t.0, h, k]).transposed(1, 2)
      kOut = kOut0
      vOut = vOut0
    } else {
      kIn = nil
      kOut = nil
      vIn = nil
      vOut = nil
      keys = tokeys(x).reshaped([b, t.0, h, k]).transposed(1, 2)
      values = tovalues(x).reshaped([b, t.0, h, k]).transposed(1, 2)
    }
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t.1, h, k])
      .transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
    dot = dot.reshaped([b * h * t.1, t.0])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t.1, t.0])
    var out = dot * values
    out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b * t.1, h * k])
    let unifyheads = Dense(count: k * h, name: "out_proj")
    out = unifyheads(out)
    if let kIn = kIn, let kOut = kOut, let vIn = vIn, let vOut = vOut {
      return Model([x, causalAttentionMask, kIn, vIn], [out, kOut, vOut])
    } else {
      return Model([x, causalAttentionMask], [out])
    }
  }
}

private func OPTDecodeLayer(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), MLP: Int, usesFlashAttention: Bool,
  injectKeysAndValues: Bool
) -> Model {
  let x = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1], name: "self_attn_layer_norm")
  var out = layerNorm1(x)
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1], name: "final_layer_norm")
  let fc = Dense(count: MLP, name: "fc1")
  let relu = ReLU()
  let proj = Dense(count: k * h, name: "fc2")
  let causalAttentionMask = Input()
  let attention = OPTSelfAttention(
    prefix: prefix, k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention,
    injectKeysAndValues: injectKeysAndValues)
  let kIn: Input?
  let kOut: Model.IO?
  let vIn: Input?
  let vOut: Model.IO?
  if injectKeysAndValues {
    let kIn0 = Input()
    let vIn0 = Input()
    let tuple = attention(out, causalAttentionMask, kIn0, vIn0)
    out = tuple[0] + x
    kIn = kIn0
    vIn = vIn0
    kOut = tuple[1]
    vOut = tuple[2]
  } else {
    out = attention(out, causalAttentionMask) + x
    kIn = nil
    vIn = nil
    kOut = nil
    vOut = nil
  }
  let residual = out
  out = layerNorm2(out)
  out = residual + proj(relu(fc(out)))
  if let kIn = kIn, let kOut = kOut, let vIn = vIn, let vOut = vOut {
    return Model([x, causalAttentionMask, kIn, vIn], [out, kOut, vOut])
  } else {
    return Model([x, causalAttentionMask], [out])
  }
}

private func OPTTextEmbedding<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, vocabularySize: Int, maxLength: Int, embeddingSize: Int
) -> Model {
  let queryEmbed = Input()
  let tokens = Input()
  let positions = Input()
  let tokenEmbed = Embedding(
    T.self, vocabularySize: vocabularySize, embeddingSize: embeddingSize, name: "embed_tokens")
  let positionEmbed = Embedding(
    T.self, vocabularySize: maxLength, embeddingSize: embeddingSize, name: "embed_positions")
  let embedding =
    Functional.concat(axis: 0, queryEmbed, tokenEmbed(tokens)) + positionEmbed(positions)
  return Model([queryEmbed, tokens, positions], [embedding])
}

func OPTDecoder<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, maxLength: Int, width: Int, tokenLength: Int,
  cachedTokenLength: Int, layers: Int, MLP: Int, heads: Int, batchSize: Int,
  usesFlashAttention: Bool, injectKeysAndValues: Bool
) -> Model {
  let queryEmbed = Input()
  let tokens = Input()
  let positions = Input()
  var kvs = [Input]()
  var kvOuts = [Model.IO]()
  let embedding = OPTTextEmbedding(
    T.self, batchSize: batchSize, vocabularySize: vocabularySize, maxLength: maxLength,
    embeddingSize: width)
  var out = embedding(queryEmbed, tokens, positions)
  let causalAttentionMask = Input()
  for i in 0..<layers {
    let layer = OPTDecodeLayer(
      prefix: "model.decoder.layers.\(i)", k: width / heads, h: heads, b: batchSize,
      t: (cachedTokenLength + tokenLength, tokenLength), MLP: MLP,
      usesFlashAttention: usesFlashAttention, injectKeysAndValues: injectKeysAndValues)
    if injectKeysAndValues {
      let kIn = Input()
      let vIn = Input()
      let tuple = layer(out, causalAttentionMask, kIn, vIn)
      out = tuple[0]
      kvs.append(kIn)
      kvs.append(vIn)
      kvOuts.append(tuple[1])
      kvOuts.append(tuple[2])
    } else {
      out = layer(out, causalAttentionMask)
    }
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1], name: "final_layer_norm")
  out = finalLayerNorm(out)
  return Model([queryEmbed, tokens, positions, causalAttentionMask] + kvs, [out] + kvOuts)
}
