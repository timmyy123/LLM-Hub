import Foundation
import NNC

public func HiDreamO1TimeEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  timestep: Float, batchSize: Int, of dataType: FloatType.Type
) -> Tensor<FloatType> {
  var embedding = Tensor<FloatType>(.CPU, .WC(batchSize, 256))
  let half = 128
  for i in 0..<half {
    let freq = timestep * 1000 * exp(-log(Float(10_000)) * Float(i) / Float(half))
    let c = FloatType(cos(freq))
    let s = FloatType(sin(freq))
    for b in 0..<batchSize {
      embedding[b, i] = c
      embedding[b, i + half] = s
    }
  }
  return embedding
}

public func HiDreamO1RotaryPositionEmbedding<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: Int, textLength: Int, height: Int, width: Int,
  of dataType: FloatType.Type
) -> Tensor<FloatType> {
  let tokenLength = textLength + 1 + height * width
  let half = 64
  let mropeSection = [24, 20, 20]
  var rotary = Tensor<FloatType>(.CPU, .NHWC(batchSize, tokenLength, 1, 128))
  for token in 0..<tokenLength {
    let imageIndex = token - textLength - 1
    let imageY = imageIndex >= 0 ? imageIndex / width : 0
    let imageX = imageIndex >= 0 ? imageIndex % width : 0
    for i in 0..<half {
      var axis = 0
      if i < mropeSection[1] * 3 && i % 3 == 1 {
        axis = 1
      } else if i < mropeSection[2] * 3 && i % 3 == 2 {
        axis = 2
      }
      let position: Int
      if imageIndex >= 0 {
        position =
          4_096
          + (axis == 0 ? 0 : (axis == 1 ? imageY : imageX))
      } else {
        position = token
      }
      let freq = Double(position) / pow(5_000_000, Double(i) / Double(half))
      let c = FloatType(cos(freq))
      let s = FloatType(sin(freq))
      for b in 0..<batchSize {
        rotary[b, token, 0, i * 2] = c
        rotary[b, token, 0, i * 2 + 1] = s
      }
    }
  }
  return rotary
}

private func HiDreamO1FeedForward(
  hiddenSize: Int, intermediateSize: Int, scaleFactor: Int
) -> Model {
  let x = Input()
  let gate = Dense(
    count: intermediateSize, noBias: true, flags: [.Float16],
    name: "mlp_gate_proj")
  let up = Dense(
    count: intermediateSize, noBias: true, flags: [.Float16], name: "mlp_up_proj")
  let x16 = x.to(.Float16)
  let scale = Float(scaleFactor)
  var out: Model.IO
  if scaleFactor > 1 {
    out = Functional.swishMul(value: up(x16), gate: (1 / scale) * gate(x16), beta: scale)
  } else {
    out = Functional.swishMul(value: up(x16), gate: gate(x16))
  }
  let down = Dense(count: hiddenSize, noBias: true, name: "mlp_down_proj")
  out = down(out).to(.Float32)
  if scaleFactor > 1 {
    out = out * scale
  }
  return Model([x], [out])
}

private func HiDreamO1TextTransformerBlock(
  batchSize: Int, textLength: Int, hiddenSize: Int, intermediateSize: Int, scaleFactor: Int
) -> Model {
  let x = Input()
  let rot = Input()
  let heads = 32
  let kvHeads = 8
  let headDimension = 128
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [2], name: "input_layernorm")
  let normalized = norm1(x).to(.Float16)
  let toKeys = Dense(
    count: headDimension * kvHeads, noBias: true, name: "k_proj")
  let toQueries = Dense(
    count: headDimension * heads, noBias: true, name: "q_proj")
  let toValues = Dense(
    count: headDimension * kvHeads, noBias: true, name: "v_proj")
  var keys = toKeys(normalized).reshaped([
    batchSize, textLength, kvHeads, headDimension,
  ])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries = toQueries(normalized).reshaped([
    batchSize, textLength, heads, headDimension,
  ])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = toValues(normalized).reshaped([
    batchSize, textLength, kvHeads, headDimension,
  ])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  let attention = ScaledDotProductAttention(
    scale: 1.0 / Float(headDimension).squareRoot(), isCausal: true,
    flags: [.Float16])(queries, keys, values)
    .reshaped([batchSize, textLength, hiddenSize])
  let outProj = Dense(count: hiddenSize, noBias: true, name: "out_proj")
  var out = x + outProj(attention).to(.Float32)
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [2], name: "post_attention_layernorm")
  let feedForward = HiDreamO1FeedForward(
    hiddenSize: hiddenSize, intermediateSize: intermediateSize, scaleFactor: scaleFactor)
  out = residual + feedForward(norm2(out))
  return Model([x, rot], [out, keys, values])
}

private func HiDreamO1TransformerBlock(
  batchSize: Int, textLength: Int, imageTokenCount: Int, hiddenSize: Int, intermediateSize: Int,
  scaleFactor: Int
) -> Model {
  let x = Input()
  let rot = Input()
  let textKeys = Input()
  let textValues = Input()
  let heads = 32
  let kvHeads = 8
  let headDimension = 128
  let dynamicTokenCount = imageTokenCount + 1
  let norm1 = RMSNorm(epsilon: 1e-6, axis: [2], name: "input_layernorm")
  let normalized = norm1(x).to(.Float16)
  let toKeys = Dense(
    count: headDimension * kvHeads, noBias: true, name: "k_proj")
  let toQueries = Dense(
    count: headDimension * heads, noBias: true, name: "q_proj")
  let toValues = Dense(
    count: headDimension * kvHeads, noBias: true, name: "v_proj")
  var keys = toKeys(normalized).reshaped([
    batchSize, dynamicTokenCount, kvHeads, headDimension,
  ])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries = toQueries(normalized).reshaped([
    batchSize, dynamicTokenCount, heads, headDimension,
  ])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  let values = toValues(normalized).reshaped([
    batchSize, dynamicTokenCount, kvHeads, headDimension,
  ])
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  let timeQueries = queries.reshaped(
    [batchSize, 1, heads, headDimension],
    strides: [
      dynamicTokenCount * heads * headDimension,
      heads * headDimension, headDimension, 1,
    ]
  ).contiguous()
  let imageQueries = queries.reshaped(
    [batchSize, imageTokenCount, heads, headDimension],
    offset: [0, 1, 0, 0],
    strides: [
      dynamicTokenCount * heads * headDimension,
      heads * headDimension, headDimension, 1,
    ]
  ).contiguous()
  let keyValuesTokenCount = textLength + dynamicTokenCount
  let keysWithText = Functional.concat(axis: 1, textKeys, keys)
  let valuesWithText = Functional.concat(axis: 1, textValues, values)
  let timeKeysWithText = keysWithText.reshaped(
    [batchSize, textLength + 1, kvHeads, headDimension],
    strides: [
      keyValuesTokenCount * kvHeads * headDimension,
      kvHeads * headDimension, headDimension, 1,
    ]
  ).contiguous()
  let timeValuesWithText = valuesWithText.reshaped(
    [batchSize, textLength + 1, kvHeads, headDimension],
    strides: [
      keyValuesTokenCount * kvHeads * headDimension,
      kvHeads * headDimension, headDimension, 1,
    ]
  ).contiguous()
  let scale = 1.0 / Float(headDimension).squareRoot()
  let timeOut = ScaledDotProductAttention(scale: scale, flags: [.Float16])(
    timeQueries, timeKeysWithText, timeValuesWithText)
  let imageOut = ScaledDotProductAttention(scale: scale, flags: [.Float16])(
    imageQueries, keysWithText, valuesWithText)
  let outProj = Dense(count: hiddenSize, noBias: true, name: "out_proj")
  var out =
    x
    + outProj(
      Functional.concat(axis: 1, timeOut, imageOut).reshaped([
        batchSize, dynamicTokenCount, hiddenSize,
      ])
    ).to(.Float32)
  let residual = out
  let norm2 = RMSNorm(epsilon: 1e-6, axis: [2], name: "post_attention_layernorm")
  let feedForward = HiDreamO1FeedForward(
    hiddenSize: hiddenSize, intermediateSize: intermediateSize, scaleFactor: scaleFactor)
  out = residual + feedForward(norm2(out))
  return Model([x, rot, textKeys, textValues], [out])
}

private func HiDreamO1TimestepEmbedder(hiddenSize: Int) -> Model {
  let x = Input()
  let fc0 = Dense(count: hiddenSize, name: "t_embedder_0")
  let fc1 = Dense(count: hiddenSize, name: "t_embedder_1")
  return Model([x], [fc1(fc0(x).swish())])
}

private func HiDreamO1PixelEmbedder(hiddenSize: Int) -> Model {
  let x = Input()
  let proj1 = Dense(
    count: 1_024, noBias: true, name: "x_embedder_0")
  let proj2 = Dense(count: hiddenSize, name: "x_embedder_1")
  return Model([x], [proj2(proj1(x))])
}

public func HiDreamO1TextFixed<FloatType: TensorNumeric & BinaryFloatingPoint>(
  _ dataType: FloatType.Type, batchSize: Int, textLength: Int, layers: Int, hiddenSize: Int,
  intermediateSize: Int,
  vocabularySize: Int
) -> Model {
  let tokens = Input()
  let rot = Input()
  let tokenEmbed = Embedding(
    FloatType.self, vocabularySize: vocabularySize, embeddingSize: hiddenSize,
    name: "tok_embeddings")
  var out = tokenEmbed(tokens).to(.Float32).reshaped([
    batchSize, textLength, hiddenSize,
  ])
  var outputs = [Model.IO]()
  for i in 0..<layers {
    let scaleFactor = i < 16 ? 2 : (i < 35 ? 4 : 64)
    let layer = HiDreamO1TextTransformerBlock(
      batchSize: batchSize, textLength: textLength, hiddenSize: hiddenSize,
      intermediateSize: intermediateSize, scaleFactor: scaleFactor)
    let layerOut = layer(out, rot)
    out = layerOut[0]
    outputs.append(layerOut[1])
    outputs.append(layerOut[2])
  }
  outputs.append(out)
  return Model([tokens, rot], outputs)
}

public func HiDreamO1(
  batchSize: Int, height: Int, width: Int, textLength: Int, layers: Int, hiddenSize: Int,
  intermediateSize: Int
) -> Model {
  let pixelPatches = Input()
  let timestep = Input()
  let rot = Input()
  let textKVs = (0..<(layers * 2)).map { _ in Input() }
  let imageTokenCount = height * width
  let pixelEmbedder = HiDreamO1PixelEmbedder(hiddenSize: hiddenSize)
  let timestepEmbedder = HiDreamO1TimestepEmbedder(hiddenSize: hiddenSize)
  let pixelEmbedding = pixelEmbedder(
    pixelPatches.reshaped([batchSize, imageTokenCount, 3 * 32 * 32]).to(.Float16)
  ).to(.Float32)
  let timeEmbedding = timestepEmbedder(timestep).to(.Float32).reshaped([
    batchSize, 1, hiddenSize,
  ])
  var out = Functional.concat(axis: 1, timeEmbedding, pixelEmbedding)
  for i in 0..<layers {
    let scaleFactor = i < 16 ? 2 : (i < 35 ? 4 : 64)
    let layer = HiDreamO1TransformerBlock(
      batchSize: batchSize, textLength: textLength, imageTokenCount: imageTokenCount,
      hiddenSize: hiddenSize, intermediateSize: intermediateSize, scaleFactor: scaleFactor)
    out = layer(out, rot, textKVs[i * 2], textKVs[i * 2 + 1])
  }
  let imageOut = out.reshaped(
    [batchSize, imageTokenCount, hiddenSize], offset: [0, 1, 0],
    strides: [(imageTokenCount + 1) * hiddenSize, hiddenSize, 1]
  ).contiguous()
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], name: "norm")
  let linear = Dense(count: 3 * 32 * 32, name: "linear")
  let predicted = linear(norm(imageOut).to(.Float16)).reshaped([
    batchSize, height, width, 3 * 32 * 32,
  ])
  return Model([pixelPatches, timestep, rot] + textKVs, [predicted])
}
