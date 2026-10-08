import NNC

private func SigLIPSelfAttention(k: Int, h: Int, b: Int, t: Int, usesFlashAttention: Bool) -> (
  Model, Model, Model, Model, Model
) {
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
      b, t, h * k,
    ])
    return (toqueries, tokeys, tovalues, scaledDotProductAttention, Model([x], [out]))
  } else {
    let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k]).transposed(
      1, 2)
    let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(queries, keys)
    dot = dot.reshaped([b * h * t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t, t])
    var out = dot * values
    out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    let unifyheads = Dense(count: k * h)
    out = unifyheads(out)
    return (toqueries, tokeys, tovalues, unifyheads, Model([x], [out]))
  }
}

private func SigLIPResidualAttentionBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, MLP: Int, usesFlashAttention: Bool,
  approximate: GELU.Approximate
) -> Model {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-6, axis: [2])
  let (_, _, _, _, attention) = SigLIPSelfAttention(
    k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention)
  var out = x + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-6, axis: [2])
  let fc = Dense(count: MLP)
  let gelu = GELU(approximate: approximate)
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  return Model([x], [out])
}

public func SigLIPVisionTransformer<T: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: T.Type, gridX: Int, gridY: Int, filterSize: Int, width: Int, layers: Int, heads: Int,
  MLP: Int, batchSize: Int, usesFlashAttention: Bool, approximate: GELU.Approximate
) -> Model {
  let x = Input()
  let posEmbed = Parameter<T>(
    .GPU(0), .HWC(1, gridX * gridY, width), initBound: 1, name: "pos_embed")
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [filterSize, filterSize],
    hint: Hint(stride: [filterSize, filterSize]), format: .OIHW)
  var out = conv1(x).reshaped([batchSize, gridX * gridY, width])
  out = out + posEmbed
  for i in 0..<layers {
    let block = SigLIPResidualAttentionBlock(
      prefix: "model.encoder.model.visual.blocks.\(i)", k: width / heads, h: heads, b: batchSize,
      t: gridX * gridY, MLP: MLP, usesFlashAttention: usesFlashAttention, approximate: approximate)
    out = block(out.reshaped([batchSize, gridX * gridY, width]))
  }
  let lnPost = LayerNorm(epsilon: 1e-6, axis: [2])
  out = lnPost(out)
  return Model([x], [out])
}

func MoondreamVisionProjection(layers: Int, approximate: GELU.Approximate) -> Model {
  let x = Input()
  let mlp1fc = Dense(count: 2048 * 4)
  let mlp1gelu = GELU(approximate: approximate)
  let mlp1proj = Dense(count: 2048)
  var out = mlp1proj(mlp1gelu(mlp1fc(x)))
  if layers > 1 {
    assert(layers == 2)
    let ln = LayerNorm(epsilon: 1e-5, axis: [1])
    out = ln(out)
    let mlp2fc = Dense(count: 2048 * 4)
    let mlp2gelu = GELU()
    let mlp2proj = Dense(count: 2048)
    out = out + mlp2proj(mlp2gelu(mlp2fc(out)))
  }
  return Model([x], [out])
}

func SelfAttention(
  prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int), rotaryDim: Int,
  usesFlashAttention: Bool
)
  -> Model
{
  let x = Input()
  let costheta = Input()
  let sintheta = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * hk, name: "k_proj")
  let toqueries = Dense(count: k * h, name: "q_proj")
  let tovalues = Dense(count: k * hk, name: "v_proj")
  let kIn = Input()
  let vIn = Input()
  var keys = tokeys(x).reshaped([b, t.1, hk, k])
  var queries = toqueries(x).reshaped([b, t.1, h, k])
  let values = tovalues(x).reshaped([b, t.1, hk, k])
  let keysRot0 = keys.reshaped(
    [b, t.1, hk, rotaryDim / 2], offset: [0, 0, 0, 0], strides: [t.1 * hk * k, hk * k, k, 1])
  let keysRot1 = keys.reshaped(
    [b, t.1, hk, rotaryDim / 2], offset: [0, 0, 0, rotaryDim / 2],
    strides: [t.1 * hk * k, hk * k, k, 1])
  let keysPass = keys.reshaped(
    [b, t.1, hk, k - rotaryDim], offset: [0, 0, 0, rotaryDim],
    strides: [t.1 * hk * k, hk * k, k, 1])
  let queriesRot0 = queries.reshaped(
    [b, t.1, h, rotaryDim / 2], offset: [0, 0, 0, 0], strides: [t.1 * h * k, h * k, k, 1])
  let queriesRot1 = queries.reshaped(
    [b, t.1, h, rotaryDim / 2], offset: [0, 0, 0, rotaryDim / 2],
    strides: [t.1 * h * k, h * k, k, 1])
  let queriesPass = queries.reshaped(
    [b, t.1, h, k - rotaryDim], offset: [0, 0, 0, rotaryDim], strides: [t.1 * h * k, h * k, k, 1])
  queries = Functional.concat(
    axis: 3, queriesRot0 .* costheta - queriesRot1 .* sintheta,
    queriesRot0 .* sintheta + queriesRot1 .* costheta, queriesPass)
  keys = Functional.concat(
    axis: 3, keysRot0 .* costheta - keysRot1 .* sintheta,
    keysRot0 .* sintheta + keysRot1 .* costheta, keysPass)
  let kOut = keys.moved(
    to: kIn.reshaped(
      [b, t.1, hk, k], offset: [0, t.0 - t.1, 0, 0], strides: [t.0 * hk * k, hk * k, k, 1]))
  let vOut = values.moved(
    to: vIn.reshaped(
      [b, t.1, hk, k], offset: [0, t.0 - t.1, 0, 0], strides: [t.0 * hk * k, hk * k, k, 1]))
  var out: Model.IO
  if usesFlashAttention {
    out = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), isCausal: true, hasAttentionMask: true)(
        queries, kIn, vIn, causalAttentionMask
      ).reshaped([b * t.1, h * k])
    out.add(dependencies: [kOut, vOut])
  } else {
    precondition(hk == h)
    let keys = kIn.transposed(1, 2)
    keys.add(dependencies: [kOut])
    let queries = ((1.0 / Float(k).squareRoot()) * queries).transposed(
      1, 2)
    let values = vIn.transposed(1, 2)
    values.add(dependencies: [vOut])
    var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
    dot = dot.reshaped([b * h * t.1, t.0])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t.1, t.0])
    out = dot * values
    out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b * t.1, h * k])
  }
  let unifyheads = Dense(count: k * h, name: "out_proj")
  out = unifyheads(out)
  return Model([x, costheta, sintheta, causalAttentionMask, kIn, vIn], [out])
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize)
  var out = GELU()(w1(x))
  let w2 = Dense(count: hiddenSize)
  out = w2(out)
  return (w1, w2, Model([x], [out], name: name))
}

func PhiTransformerBlock(
  prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int), MLP: Int, rotaryDim: Int,
  usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let costheta = Input()
  let sintheta = Input()
  let causalAttentionMask = Input()
  let kIn = Input()
  let vIn = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [1], name: "attention_norm")
  var out = norm1(x)
  let attention = SelfAttention(
    prefix: prefix, k: k, h: h, hk: hk, b: b, t: t, rotaryDim: rotaryDim,
    usesFlashAttention: usesFlashAttention)
  let residual = out
  out = attention(out, costheta, sintheta, causalAttentionMask, kIn, vIn) + x
  let (_, _, ffn) = FeedForward(hiddenSize: h * k, intermediateSize: MLP, name: "ffn")
  out = out + ffn(residual)
  return Model([x, costheta, sintheta, causalAttentionMask, kIn, vIn], [out])
}

func PhiDecoder<T: TensorNumeric>(
  _ dataType: T.Type, vocabularySize: Int, width: Int, tokenLength: Int, cachedTokenLength: Int,
  layers: Int, MLP: Int, rotaryDim: Int, heads: Int, batchSize: Int, usesFlashAttention: Bool
) -> Model {
  let textEmb = Input()
  let costheta = Input()
  let sintheta = Input()
  let causalAttentionMask = Input()
  var kvs = [Input]()
  var out: Model.IO = textEmb
  for i in 0..<layers {
    let layer = PhiTransformerBlock(
      prefix: "model.transformer.h.\(i)", k: width / heads, h: heads, hk: heads, b: batchSize,
      t: (cachedTokenLength + tokenLength, tokenLength), MLP: MLP, rotaryDim: rotaryDim,
      usesFlashAttention: usesFlashAttention)
    let kIn = Input()
    let vIn = Input()
    out = layer(out, costheta, sintheta, causalAttentionMask, kIn, vIn)
    kvs.append(kIn)
    kvs.append(vIn)
  }
  out = out.reshaped(.WC(1, width), offset: [tokenLength - 1, 0], strides: [width, 1])
  let norm = LayerNorm(epsilon: 1e-5, axis: [1], name: "norm")
  out = norm(out)
  let output = Dense(count: vocabularySize, name: "output")
  out = output(out)
  return Model([textEmb, costheta, sintheta, causalAttentionMask] + kvs, [out])
}
