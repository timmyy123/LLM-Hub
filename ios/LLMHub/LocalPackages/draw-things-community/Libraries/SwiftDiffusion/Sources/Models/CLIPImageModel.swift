import NNC

func CLIPSelfAttention(k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
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

private func OpenCLIPResidualAttentionBlock(k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let attention = CLIPSelfAttention(k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  return Model([x], [out])
}

public func VisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type,
  grid: Int, width: Int, layers: Int, heads: Int, batchSize: Int,
  noFinalLayerNorm: Bool = false
) -> Model {
  let x = Input()
  let classEmbedding = Parameter<T>(.GPU(0), .HWC(1, 1, width))
  let positionalEmbedding = Parameter<T>(.GPU(0), .HWC(1, grid * grid + 1, width))
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], noBias: true,
    hint: Hint(stride: [14, 14]), format: .OIHW)
  var out = conv1(x).reshaped([batchSize, grid * grid, width])
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  let lnPre = LayerNorm(epsilon: 1e-5, axis: [2])
  out = lnPre(out)
  for _ in 0..<layers {
    let block = OpenCLIPResidualAttentionBlock(
      k: width / heads, h: heads, b: batchSize, t: grid * grid + 1)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
  }
  if !noFinalLayerNorm {
    let lnPost = LayerNorm(epsilon: 1e-5, axis: [1], name: "post_layernorm")
    out = lnPost(out.reshaped([batchSize, width], strides: [width, 1]))
  }
  return Model([x], [out])
}

private func CLIPResidualAttentionBlock(k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let attention = CLIPSelfAttention(k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = QuickGELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  return Model([x], [out])
}

public func CLIPVisionTransformer<T: TensorNumeric>(
  _ dataType: T.Type,
  grid: Int, width: Int, layers: Int, heads: Int, batchSize: Int,
  noFinalLayerNorm: Bool = false
) -> Model {
  let x = Input()
  let classEmbedding = Parameter<T>(.GPU(0), .HWC(1, 1, width))
  let positionalEmbedding = Parameter<T>(.GPU(0), .HWC(1, grid * grid + 1, width))
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], noBias: true,
    hint: Hint(stride: [14, 14]), format: .OIHW)
  var out = conv1(x).reshaped([batchSize, grid * grid, width])
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  let lnPre = LayerNorm(epsilon: 1e-5, axis: [2])
  out = lnPre(out)
  for _ in 0..<layers {
    let block = CLIPResidualAttentionBlock(
      k: width / heads, h: heads, b: batchSize, t: grid * grid + 1)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
  }
  if !noFinalLayerNorm {
    let lnPost = LayerNorm(epsilon: 1e-5, axis: [1], name: "post_layernorm")
    out = lnPost(out.reshaped([batchSize, width], strides: [width, 1]))
  }
  return Model([x], [out])
}
