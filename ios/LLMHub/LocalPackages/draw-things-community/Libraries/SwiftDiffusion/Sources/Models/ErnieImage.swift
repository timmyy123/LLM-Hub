import DiffusionMappings
import Foundation
import NNC

public func ErnieImageTimeEmbedding(
  timestep: Float, batchSize: Int, embeddingSize: Int, maxPeriod: Int = 10_000
) -> Tensor<Float> {
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .WC(batchSize, embeddingSize))
  let half = embeddingSize / 2
  for i in 0..<half {
    let exponent = -log(Float(maxPeriod)) * Float(i) / Float(half)
    let freq = exp(exponent)
    let value = timestep * freq
    let sinValue = sin(value)
    let cosValue = cos(value)
    for j in 0..<batchSize {
      embedding[j, i] = sinValue
      embedding[j, i + half] = cosValue
    }
  }
  return embedding
}

public func ErnieImageRotaryPositionEmbedding(
  height: Int, width: Int, textLength: Int, heads: Int = 1
) -> (Tensor<Float>, Tensor<Float>) {
  let imageLength = height * width
  var cosTensor = Tensor<Float>(
    .CPU, .NHWC(1, imageLength + textLength, heads, 128))
  var sinTensor = Tensor<Float>(
    .CPU, .NHWC(1, imageLength + textLength, heads, 128))

  for token in 0..<(imageLength + textLength) {
    let positions: (Double, Double, Double)
    if token < imageLength {
      positions = (Double(textLength), Double(token / width), Double(token % width))
    } else {
      positions = (Double(token - imageLength), 0, 0)
    }
    for j in 0..<heads {
      for k in 0..<64 {
        let angle: Double
        if k < 16 {
          angle = positions.0 / pow(256.0, Double(k * 2) / 32.0)
        } else if k < 40 {
          angle = positions.1 / pow(256.0, Double((k - 16) * 2) / 48.0)
        } else {
          angle = positions.2 / pow(256.0, Double((k - 40) * 2) / 48.0)
        }
        let c = Float(cos(angle))
        let s = Float(sin(angle))
        cosTensor[0, token, j, k * 2] = c
        cosTensor[0, token, j, k * 2 + 1] = c
        // Bake the non-interleaved rotate_half sign into the swapped-half multiplier.
        let signedS = k < 32 ? -s : s
        sinTensor[0, token, j, k * 2] = signedS
        sinTensor[0, token, j, k * 2 + 1] = signedS
      }
    }
  }
  return (cosTensor, sinTensor)
}

public func ErnieImageFixed(timesteps: Int, channels: Int) -> (
  ModelWeightMapper, Model
) {
  let txt = Input()
  let t = Input()
  let textProj = Dense(count: channels, noBias: true, name: "c_embedder")
  let projectedText = textProj(txt)
  let timeMlp0 = Dense(count: channels, name: "time_embedder_0")
  let timeMlp2 = Dense(count: channels, name: "time_embedder_1")
  let timeCondition = timeMlp2(timeMlp0(t).swish())
  let timeModulation = timeCondition.swish()
  let adaLNs = (0..<6).map { Dense(count: channels, name: "ada_ln_\($0)") }
  let shiftMSA = adaLNs[0](timeModulation)
  let scaleMSA = 1 + adaLNs[1](timeModulation)
  let gateMSA = adaLNs[2](timeModulation).to(.Float32)
  let shiftMLP = adaLNs[3](timeModulation)
  let scaleMLP = 1 + adaLNs[4](timeModulation)
  let gateMLP = adaLNs[5](timeModulation).to(.Float32)
  let finalScale = Dense(count: channels, name: "scale")
  let finalShift = Dense(count: channels, name: "shift")
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["text_proj.weight"] = [textProj.weight.name]
    mapping["time_embedding.linear_1.weight"] = [timeMlp0.weight.name]
    mapping["time_embedding.linear_1.bias"] = [timeMlp0.bias.name]
    mapping["time_embedding.linear_2.weight"] = [timeMlp2.weight.name]
    mapping["time_embedding.linear_2.bias"] = [timeMlp2.bias.name]
    mapping["adaLN_modulation.1.weight"] = ModelWeightElement(adaLNs.map { $0.weight.name })
    mapping["adaLN_modulation.1.bias"] = ModelWeightElement(adaLNs.map { $0.bias.name })
    mapping["final_norm.linear.weight"] = [finalScale.weight.name, finalShift.weight.name]
    mapping["final_norm.linear.bias"] = [finalScale.bias.name, finalShift.bias.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [txt, t],
      [
        projectedText, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP,
        1 + finalScale(timeCondition), finalShift(timeCondition),
      ])
  )
}

public func ErnieImageFixedOutputShapes(tokenLength: Int, channels: Int) -> [TensorShape] {
  return [TensorShape([1, tokenLength, channels])]
    + Array(repeating: TensorShape([1, 1, channels]), count: 8)
}

private func ErnieImageFeedForward(
  hiddenSize: Int, intermediateSize: Int, scaleFactor: Int, name: String
) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let gate = Dense(count: intermediateSize, noBias: true, name: "\(name)_gate_proj")
  let up = Dense(count: intermediateSize, noBias: true, flags: [.Float16], name: "\(name)_up_proj")
  var out: Model.IO
  if scaleFactor > 1 {
    out = up((1.0 / Float(scaleFactor)) * x)
  } else {
    out = up(x)
  }
  out = out .* gate(x).GELU()
  let down = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = down(out).to(.Float32)
  return (gate, down, up, Model([x], [out]))
}

private func ErnieImageTransformerBlock(
  prefix: String, hiddenSize: Int, k: Int, h: Int, batchSize: Int, t: (keyValue: Int, query: Int),
  intermediateSize: Int, scaleFactor: (qk: Int, v: Int, projDown: Int),
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotCos = Input()
  let rotSin = Input()
  let shiftMSA = Input()
  let scaleMSA = Input()
  let gateMSA = Input()
  let shiftMLP = Input()
  let scaleMLP = Input()
  let gateMLP = Input()

  let attentionNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "attention_norm1")
  var out = attentionNorm(x)
  out = out.to(.Float16) .* scaleMSA + shiftMSA
  if scaleFactor.qk > 1 {
    out = (1 / Float(scaleFactor.qk)) * out
  }
  let tokeys = Dense(count: k * h, noBias: true, name: "k")
  let toqueries = Dense(count: k * h, noBias: true, flags: [.Float16], name: "q")
  let tovalues = Dense(count: k * h, noBias: true, name: "v")
  var keys = tokeys(out).reshaped([batchSize, t.keyValue, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries: Model.IO
  if t.keyValue != t.query {
    queries = toqueries(
      out.reshaped(
        [batchSize, t.query, k * h], offset: [0, 0, 0], strides: [t.keyValue * h * k, h * k, 1])
    ).reshaped([batchSize, t.query, h, k])
  } else {
    queries = toqueries(out).reshaped([batchSize, t.keyValue, h, k])
  }
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  if scaleFactor.v > 1 {
    out = (1 / Float(scaleFactor.v)) * out
  }
  let values = tovalues(out).reshaped([batchSize, t.keyValue, h, k])
  let queryCos: Model.IO
  let querySin: Model.IO
  if t.keyValue != t.query {
    queryCos = rotCos.reshaped(
      [1, t.query, 1, k], offset: [0, 0, 0, 0], strides: [t.keyValue * k, k, k, 1])
    querySin = rotSin.reshaped(
      [1, t.query, 1, k], offset: [0, 0, 0, 0], strides: [t.keyValue * k, k, k, 1])
  } else {
    queryCos = rotCos
    querySin = rotSin
  }
  queries =
    queries .* queryCos.to(of: queries) + Functional.rotateHalf(queries) .* querySin.to(of: queries)
  keys = keys .* rotCos.to(of: keys) + Functional.rotateHalf(keys) .* rotSin.to(of: keys)

  var attentionOut: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = keys.transposed(1, 2)
    let transposedQueries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
    let transposedValues = values.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
    dot = dot.reshaped([batchSize * h * t.query, t.keyValue])
    dot = dot.softmax()
    dot = dot.reshaped([batchSize, h, t.query, t.keyValue])
    attentionOut = (dot * transposedValues).transposed(1, 2).reshaped([
      batchSize, t.query, hiddenSize,
    ])
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    attentionOut = scaledDotProductAttention(
      (1.0 / Float(k).squareRoot()) * queries, keys, values
    ).reshaped([batchSize, t.query, hiddenSize])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    attentionOut = scaledDotProductAttention(
      queries, keys, values
    ).reshaped([batchSize, t.query, hiddenSize])
  }
  let xIn: Model.IO
  if t.keyValue > t.query {
    xIn = x.reshaped(
      [batchSize, t.query, h * k], offset: [0, 0, 0], strides: [t.keyValue * h * k, h * k, 1])
  } else {
    xIn = x
  }

  let attentionOutProj = Dense(count: hiddenSize, noBias: true, name: "o")
  attentionOut = attentionOutProj(attentionOut).to(of: xIn)
  if scaleFactor.v * scaleFactor.qk > 1 {
    attentionOut = Functional.mul(
      left: attentionOut, right: gateMSA,
      scalar: Float(scaleFactor.qk * scaleFactor.v))
  } else {
    attentionOut = attentionOut .* gateMSA
  }
  out = xIn + attentionOut

  let mlpNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "attention_norm2")
  var mlpIn = mlpNorm(out).to(.Float16)
  mlpIn = mlpIn .* scaleMLP + shiftMLP
  let (ffnGate, ffnDown, ffnUp, ffn) = ErnieImageFeedForward(
    hiddenSize: hiddenSize, intermediateSize: intermediateSize, scaleFactor: scaleFactor.projDown,
    name: "ffn")
  var mlpOut = ffn(mlpIn)
  if scaleFactor.projDown > 1 {
    mlpOut = Functional.mul(
      left: mlpOut, right: gateMLP, scalar: Float(scaleFactor.projDown))
  } else {
    mlpOut = mlpOut .* gateMLP
  }
  out = out + mlpOut

  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).adaLN_sa_ln.weight"] = [attentionNorm.weight.name]
    mapping["\(prefix).self_attention.to_q.weight"] = [toqueries.weight.name]
    mapping["\(prefix).self_attention.to_k.weight"] = [tokeys.weight.name]
    mapping["\(prefix).self_attention.to_v.weight"] = [tovalues.weight.name]
    mapping["\(prefix).self_attention.norm_k.weight"] = [normK.weight.name]
    mapping["\(prefix).self_attention.norm_q.weight"] = [normQ.weight.name]
    mapping["\(prefix).self_attention.to_out.0.weight"] = [attentionOutProj.weight.name]
    mapping["\(prefix).adaLN_mlp_ln.weight"] = [mlpNorm.weight.name]
    mapping["\(prefix).mlp.gate_proj.weight"] = [ffnGate.weight.name]
    mapping["\(prefix).mlp.up_proj.weight"] = [ffnUp.weight.name]
    mapping["\(prefix).mlp.linear_fc2.weight"] = [ffnDown.weight.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rotCos, rotSin, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP], [out])
  )
}

public func ErnieImage(
  batchSize: Int, height: Int, width: Int, textLength: Int,
  layers: Int, channels: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotCos = Input()
  let rotSin = Input()
  let txt = Input()
  let shiftMSA = Input()
  let scaleMSA = Input()
  let gateMSA = Input()
  let shiftMLP = Input()
  let scaleMLP = Input()
  let gateMLP = Input()
  let finalScale = Input()
  let finalShift = Input()
  let patchHeight = height / 2
  let patchWidth = width / 2
  let headDim = 128
  precondition(channels % headDim == 0)
  let heads = channels / headDim
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW,
    name: "x_embedder")
  let img = xEmbedder(x).reshaped([batchSize, patchHeight * patchWidth, channels]).to(
    .Float32)

  let sequenceLength = patchHeight * patchWidth + textLength
  let rotCosResized = rotCos.reshaped([1, sequenceLength, 1, 128])
  let rotSinResized = rotSin.reshaped([1, sequenceLength, 1, 128])
  var out = Functional.concat(axis: 1, img, txt.to(.Float32))
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (blockMapper, block) = ErnieImageTransformerBlock(
      prefix: "layers.\(i)", hiddenSize: channels, k: headDim, h: heads, batchSize: batchSize,
      t: (
        keyValue: sequenceLength, query: i == layers - 1 ? patchHeight * patchWidth : sequenceLength
      ),
      intermediateSize: channels * 3, scaleFactor: (qk: 1, v: 8, projDown: 8),
      usesFlashAttention: usesFlashAttention)
    out = block(
      out, rotCosResized, rotSinResized, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP,
      gateMLP)
    mappers.append(blockMapper)
  }

  let finalNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false, name: "final_norm")
  out = finalNorm(out).to(.Float16) .* finalScale + finalShift
  let finalLinear = Dense(count: 128, name: "linear")
  out = finalLinear(out)

  let unpatchified = out.reshaped(
    [batchSize, patchHeight, patchWidth, 32, 2, 2]
  ).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([batchSize, height, width, 32])

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["x_embedder.proj.weight"] = [xEmbedder.weight.name]
    mapping["x_embedder.proj.bias"] = [xEmbedder.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_linear.weight"] = [finalLinear.weight.name]
    mapping["final_linear.bias"] = [finalLinear.bias.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [
        x, rotCos, rotSin, txt, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP,
        finalScale, finalShift,
      ], [unpatchified])
  )
}

private func LoRAErnieImageFeedForward(
  hiddenSize: Int, intermediateSize: Int, scaleFactor: Int, configuration: LoRANetworkConfiguration,
  layerIndex: Int, name: String
) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let gate = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_gate_proj")
  let up = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, flags: [.Float16],
    index: layerIndex, name: "\(name)_up_proj")
  var out: Model.IO
  if scaleFactor > 1 {
    out = up((1.0 / Float(scaleFactor)) * x)
  } else {
    out = up(x)
  }
  out = out .* gate(x).GELU()
  let down = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex,
    name: "\(name)_down_proj")
  out = down(out).to(.Float32)
  return (gate, down, up, Model([x], [out]))
}

private func LoRAErnieImageTransformerBlock(
  prefix: String, hiddenSize: Int, k: Int, h: Int, batchSize: Int, t: (keyValue: Int, query: Int),
  intermediateSize: Int, scaleFactor: (qk: Int, v: Int, projDown: Int),
  usesFlashAttention: FlashAttentionLevel, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotCos = Input()
  let rotSin = Input()
  let shiftMSA = Input()
  let scaleMSA = Input()
  let gateMSA = Input()
  let shiftMLP = Input()
  let scaleMLP = Input()
  let gateMLP = Input()

  let attentionNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "attention_norm1")
  var out = attentionNorm(x)
  out = out.to(.Float16) .* scaleMSA + shiftMSA
  if scaleFactor.qk > 1 {
    out = (1 / Float(scaleFactor.qk)) * out
  }
  let tokeys = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "k")
  let toqueries = LoRADense(
    count: k * h, configuration: configuration, noBias: true, flags: [.Float16],
    index: layerIndex, name: "q")
  let tovalues = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "v")
  var keys = tokeys(out).reshaped([batchSize, t.keyValue, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_k")
  keys = normK(keys)
  var queries: Model.IO
  if t.keyValue != t.query {
    queries = toqueries(
      out.reshaped(
        [batchSize, t.query, k * h], offset: [0, 0, 0], strides: [t.keyValue * h * k, h * k, 1])
    ).reshaped([batchSize, t.query, h, k])
  } else {
    queries = toqueries(out).reshaped([batchSize, t.keyValue, h, k])
  }
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "norm_q")
  queries = normQ(queries)
  if scaleFactor.v > 1 {
    out = (1 / Float(scaleFactor.v)) * out
  }
  let values = tovalues(out).reshaped([batchSize, t.keyValue, h, k])
  let queryCos: Model.IO
  let querySin: Model.IO
  if t.keyValue != t.query {
    queryCos = rotCos.reshaped(
      [1, t.query, 1, k], offset: [0, 0, 0, 0], strides: [t.keyValue * k, k, k, 1])
    querySin = rotSin.reshaped(
      [1, t.query, 1, k], offset: [0, 0, 0, 0], strides: [t.keyValue * k, k, k, 1])
  } else {
    queryCos = rotCos
    querySin = rotSin
  }
  queries =
    queries .* queryCos.to(of: queries) + Functional.rotateHalf(queries) .* querySin.to(of: queries)
  keys = keys .* rotCos.to(of: keys) + Functional.rotateHalf(keys) .* rotSin.to(of: keys)

  var attentionOut: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = keys.transposed(1, 2)
    let transposedQueries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
    let transposedValues = values.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
    dot = dot.reshaped([batchSize * h * t.query, t.keyValue])
    dot = dot.softmax()
    dot = dot.reshaped([batchSize, h, t.query, t.keyValue])
    attentionOut = (dot * transposedValues).transposed(1, 2).reshaped([
      batchSize, t.query, hiddenSize,
    ])
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    attentionOut = scaledDotProductAttention(
      (1.0 / Float(k).squareRoot()) * queries, keys, values
    ).reshaped([batchSize, t.query, hiddenSize])
    scaledDotProductAttention.gradientCheckpointing = false
  case .quantized:
    let scaledDotProductAttention: Model
    if configuration.testing {
      scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(), flags: [.Int8, .Float16])
    } else {
      queries = (1.0 / Float(k).squareRoot()) * queries
      scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Int8, .Float16])
    }
    attentionOut = scaledDotProductAttention(queries, keys, values).reshaped([
      batchSize, t.query, hiddenSize,
    ])
    scaledDotProductAttention.gradientCheckpointing = false
  case .scaleMerged:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
    attentionOut = scaledDotProductAttention(queries, keys, values).reshaped([
      batchSize, t.query, hiddenSize,
    ])
    scaledDotProductAttention.gradientCheckpointing = false
  }
  let xIn: Model.IO
  if t.keyValue > t.query {
    xIn = x.reshaped(
      [batchSize, t.query, h * k], offset: [0, 0, 0], strides: [t.keyValue * h * k, h * k, 1])
  } else {
    xIn = x
  }

  let attentionOutProj = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: layerIndex, name: "o")
  attentionOut = attentionOutProj(attentionOut).to(of: xIn)
  if scaleFactor.v * scaleFactor.qk > 1 {
    attentionOut = Functional.mul(
      left: attentionOut, right: gateMSA,
      scalar: Float(scaleFactor.qk * scaleFactor.v))
  } else {
    attentionOut = attentionOut .* gateMSA
  }
  out = xIn + attentionOut

  let mlpNorm = RMSNorm(epsilon: 1e-6, axis: [2], name: "attention_norm2")
  var mlpIn = mlpNorm(out).to(.Float16)
  mlpIn = mlpIn .* scaleMLP + shiftMLP
  let (ffnGate, ffnDown, ffnUp, ffn) = LoRAErnieImageFeedForward(
    hiddenSize: hiddenSize, intermediateSize: intermediateSize, scaleFactor: scaleFactor.projDown,
    configuration: configuration, layerIndex: layerIndex, name: "ffn")
  if configuration.gradientCheckpointingFeedForward {
    ffn.gradientCheckpointing = true
  }
  var mlpOut = ffn(mlpIn)
  if scaleFactor.projDown > 1 {
    mlpOut = Functional.mul(
      left: mlpOut, right: gateMLP, scalar: Float(scaleFactor.projDown))
  } else {
    mlpOut = mlpOut .* gateMLP
  }
  out = out + mlpOut

  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).adaLN_sa_ln.weight"] = [attentionNorm.weight.name]
    mapping["\(prefix).self_attention.to_q.weight"] = [toqueries.weight.name]
    mapping["\(prefix).self_attention.to_k.weight"] = [tokeys.weight.name]
    mapping["\(prefix).self_attention.to_v.weight"] = [tovalues.weight.name]
    mapping["\(prefix).self_attention.norm_k.weight"] = [normK.weight.name]
    mapping["\(prefix).self_attention.norm_q.weight"] = [normQ.weight.name]
    mapping["\(prefix).self_attention.to_out.0.weight"] = [attentionOutProj.weight.name]
    mapping["\(prefix).adaLN_mlp_ln.weight"] = [mlpNorm.weight.name]
    mapping["\(prefix).mlp.gate_proj.weight"] = [ffnGate.weight.name]
    mapping["\(prefix).mlp.up_proj.weight"] = [ffnUp.weight.name]
    mapping["\(prefix).mlp.linear_fc2.weight"] = [ffnDown.weight.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rotCos, rotSin, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP], [out])
  )
}

public func LoRAErnieImage(
  batchSize: Int, height: Int, width: Int, textLength: Int, layers: Int, channels: Int,
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, ModelWeightMapper) {
  let x = Input()
  let rotCos = Input()
  let rotSin = Input()
  let txt = Input()
  let shiftMSA = Input()
  let scaleMSA = Input()
  let gateMSA = Input()
  let shiftMLP = Input()
  let scaleMLP = Input()
  let gateMLP = Input()
  let finalScale = Input()
  let finalShift = Input()
  let patchHeight = height / 2
  let patchWidth = width / 2
  let headDim = 128
  precondition(channels % headDim == 0)
  let heads = channels / headDim
  let xEmbedder = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW,
    index: 0, name: "x_embedder")
  let img = xEmbedder(x).reshaped([batchSize, patchHeight * patchWidth, channels]).to(.Float32)

  let sequenceLength = patchHeight * patchWidth + textLength
  let rotCosResized = rotCos.reshaped([1, sequenceLength, 1, 128])
  let rotSinResized = rotSin.reshaped([1, sequenceLength, 1, 128])
  var out = Functional.concat(axis: 1, img, txt.to(.Float32))
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (blockMapper, block) = LoRAErnieImageTransformerBlock(
      prefix: "layers.\(i)", hiddenSize: channels, k: headDim, h: heads, batchSize: batchSize,
      t: (
        keyValue: sequenceLength, query: i == layers - 1 ? patchHeight * patchWidth : sequenceLength
      ),
      intermediateSize: channels * 3, scaleFactor: (qk: 1, v: 8, projDown: 8),
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    out = block(
      out, rotCosResized, rotSinResized, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP,
      gateMLP)
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      block.gradientCheckpointing = true
    }
    mappers.append(blockMapper)
  }

  let finalNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false, name: "final_norm")
  out = finalNorm(out).to(.Float16) .* finalScale + finalShift
  let finalLinear = LoRADense(
    count: 128, configuration: LoRAConfiguration, index: 0, name: "linear")
  out = finalLinear(out)

  let unpatchified = out.reshaped(
    [batchSize, patchHeight, patchWidth, 32, 2, 2]
  ).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([batchSize, height, width, 32])

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["x_embedder.proj.weight"] = [xEmbedder.weight.name]
    mapping["x_embedder.proj.bias"] = [xEmbedder.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_linear.weight"] = [finalLinear.weight.name]
    mapping["final_linear.bias"] = [finalLinear.bias.name]
    return mapping
  }
  return (
    Model(
      [
        x, rotCos, rotSin, txt, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP,
        finalScale, finalShift,
      ], [unpatchified], trainable: false),
    mapper
  )
}

public func LoRAErnieImageFixed(timesteps: Int, channels: Int, LoRAConfiguration: LoRANetworkConfiguration
) -> (Model, ModelWeightMapper) {
  let txt = Input()
  let t = Input()
  let textProj = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: 0, name: "c_embedder")
  let projectedText = textProj(txt)
  let timeMlp0 = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "time_embedder_0")
  let timeMlp2 = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "time_embedder_1")
  let timeCondition = timeMlp2(timeMlp0(t).swish())
  let timeModulation = timeCondition.swish()
  let adaLNs = (0..<6).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, index: 0, name: "ada_ln_\($0)")
  }
  let shiftMSA = adaLNs[0](timeModulation)
  let scaleMSA = 1 + adaLNs[1](timeModulation)
  let gateMSA = adaLNs[2](timeModulation).to(.Float32)
  let shiftMLP = adaLNs[3](timeModulation)
  let scaleMLP = 1 + adaLNs[4](timeModulation)
  let gateMLP = adaLNs[5](timeModulation).to(.Float32)
  let finalScale = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "scale")
  let finalShift = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "shift")
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["text_proj.weight"] = [textProj.weight.name]
    mapping["time_embedding.linear_1.weight"] = [timeMlp0.weight.name]
    mapping["time_embedding.linear_1.bias"] = [timeMlp0.bias.name]
    mapping["time_embedding.linear_2.weight"] = [timeMlp2.weight.name]
    mapping["time_embedding.linear_2.bias"] = [timeMlp2.bias.name]
    mapping["adaLN_modulation.1.weight"] = ModelWeightElement(adaLNs.map { $0.weight.name })
    mapping["adaLN_modulation.1.bias"] = ModelWeightElement(adaLNs.map { $0.bias.name })
    mapping["final_norm.linear.weight"] = [finalScale.weight.name, finalShift.weight.name]
    mapping["final_norm.linear.bias"] = [finalScale.bias.name, finalShift.bias.name]
    return mapping
  }
  return (
    Model(
      [txt, t],
      [
        projectedText, shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP,
        1 + finalScale(timeCondition), finalShift(timeCondition),
      ]),
    mapper
  )
}
