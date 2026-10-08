import DiffusionMappings
import Foundation
import NNC

func sinCos2DPositionEmbedding(height: Int, width: Int, embeddingSize: Int) -> Tensor<Float> {
  precondition(embeddingSize % 4 == 0)
  var embedding = Tensor<Float>(.CPU, .HWC(height, width, embeddingSize))
  let halfOfHalf = embeddingSize / 4
  let omega: [Double] = (0..<halfOfHalf).map {
    pow(Double(1.0 / 10000), Double($0) / Double(halfOfHalf))
  }
  for i in 0..<height {
    let y = Double(i) / (Double(height) / 64) / 2
    for j in 0..<width {
      let x = Double(j) / (Double(width) / 64) / 2
      for k in 0..<halfOfHalf {
        let xFreq = x * omega[k]
        embedding[i, j, k] = Float(sin(xFreq))
        embedding[i, j, k + halfOfHalf] = Float(cos(xFreq))
        let yFreq = y * omega[k]
        embedding[i, j, k + 2 * halfOfHalf] = Float(sin(yFreq))
        embedding[i, j, k + 3 * halfOfHalf] = Float(cos(yFreq))
      }
    }
  }
  return embedding
}

private func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func MLP(hiddenSize: Int, intermediateSize: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize, name: "\(name)_fc1")
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = Dense(count: hiddenSize, flags: [.Float32], name: "\(name)_fc2")
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

private func SelfAttention(
  k: Int, h: Int, b: Int, t: Int, usesFlashAttention: FlashAttentionLevel
) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let tokeys = Dense(count: k * h, name: "k")
  let toqueries = Dense(count: k * h, name: "q")
  let tovalues = Dense(count: k * h, name: "v")
  if usesFlashAttention != .none {
    let keys = tokeys(x).reshaped([b, t, h, k])
    let queries: Model.IO
    let scale: Float
    if usesFlashAttention == .scale1 {
      queries = (1.0 / Float(k).squareRoot() * toqueries(x)).reshaped([b, t, h, k])
      scale = 1
    } else {
      queries = toqueries(x).reshaped([b, t, h, k])
      scale = 1.0 / Float(k).squareRoot()
    }
    let values = tovalues(x).reshaped([b, t, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: scale, flags: [.Float16],
      multiHeadOutputProjectionFused: true, name: "o")
    let out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
    return (tokeys, toqueries, tovalues, scaledDotProductAttention, Model([x], [out]))
  } else {
    let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
    // No scaling the queries.
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
      .transposed(1, 2)
    let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
    var out: Model.IO
    if b * h <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped([1, t, k], offset: [i, 0, 0], strides: [t * k, k, 1])
        let query = queries.reshaped([1, t, k], offset: [i, 0, 0], strides: [t * k, k, 1])
        let value = values.reshaped([1, t, k], offset: [i, 0, 0], strides: [t * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t, t])
        dot = dot.softmax()
        dot = dot.reshaped([1, t, t])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * t, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, t, t])
      out = dot * values
      out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    }
    let unifyheads = Dense(count: k * h, name: "o")
    out = unifyheads(out)
    return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
  }
}

private func CrossAttentionKeysAndValues(
  k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), usesFlashAttention: FlashAttentionLevel
) -> (
  Model, Model, Model
) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = Dense(count: k * h, name: "c_q")
  if t.0 == t.1 {
    let t = t.0
    if usesFlashAttention != .none {
      let queries: Model.IO
      let scale: Float
      if usesFlashAttention == .scale1 {
        queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
        scale = 1
      } else {
        queries = toqueries(x).reshaped([b, hw, h, k])
        scale = 1.0 / Float(k).squareRoot()
      }
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: scale, flags: [.Float16],
        multiHeadOutputProjectionFused: true, name: "c_o")
      let out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
      return (toqueries, scaledDotProductAttention, Model([x, keys, values], [out]))
    } else {
      let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      let transposedQueries = queries.transposed(1, 2)
      var dot = Matmul(transposeB: (2, 3))(transposedQueries, keys)
      dot = dot.reshaped([b * h * hw, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, hw, t])
      var out = dot * values
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      let unifyheads = Dense(count: k * h, name: "c_o")
      out = unifyheads(out)
      return (toqueries, unifyheads, Model([x, keys, values], [out]))
    }
  } else {
    let b0 = b / 2
    var out: Model.IO
    if usesFlashAttention != .none {
      let queries: Model.IO
      let scale: Float
      if usesFlashAttention == .scale1 {
        queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
        scale = 1
      } else {
        queries = toqueries(x).reshaped([b, hw, h, k])
        scale = 1.0 / Float(k).squareRoot()
      }
      let out0: Model.IO
      if b0 == 1 || t.0 >= t.1 {
        let keys0 = keys.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries0 = queries.reshaped([b0, hw, h, k])
        let values0 = values.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out0 = ScaledDotProductAttention(
          scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
        )(
          queries0, keys0, values0
        ).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys0 = keys.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries0 = queries.reshaped(
            [1, hw, h, k], offset: [i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values0 = values.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(
              scale: scale,
              flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
            )(
              queries0, keys0, values0
            ).reshaped([1, hw, h * k]))
        }
        out0 = Concat(axis: 0)(outs)
      }
      let out1: Model.IO
      if b0 == 1 || t.1 >= t.0 {
        let keys1 = keys.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries1 = queries.reshaped(
          [b0, hw, h, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
        let values1 = values.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out1 = ScaledDotProductAttention(
          scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
        )(
          queries1, keys1, values1
        ).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys1 = keys.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries1 = queries.reshaped(
            [1, hw, h, k], offset: [b0 + i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values1 = values.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(
              scale: scale,
              flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
            )(
              queries1, keys1, values1
            ).reshaped([1, hw, h * k]))
        }
        out1 = Concat(axis: 0)(outs)
      }
      out = Functional.concat(axis: 0, out0, out1)
    } else {
      let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      let transposedQueries = queries.transposed(1, 2)
      let keys0 = keys.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      let queries0 = transposedQueries.reshaped([b0, h, hw, k])
      let values0 = values.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot0 = Matmul(transposeB: (2, 3))(queries0, keys0)
      dot0 = dot0.reshaped([b0 * h * hw, t.0])
      dot0 = dot0.softmax()
      dot0 = dot0.reshaped([b0, h, hw, t.0])
      var out0 = dot0 * values0
      out0 = out0.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      let keys1 = keys.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      let queries1 = transposedQueries.reshaped(
        [b0, h, hw, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, hw * k, k, 1])
      let values1 = values.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot1 = Matmul(transposeB: (2, 3))(queries1, keys1)
      dot1.add(dependencies: [out0])
      dot1 = dot1.reshaped([b0 * h * hw, t.1])
      dot1 = dot1.softmax()
      dot1 = dot1.reshaped([b0, h, hw, t.1])
      var out1 = dot1 * values1
      out1 = out1.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      out = Functional.concat(axis: 0, out0, out1)
    }
    let unifyheads = Dense(count: k * h, name: "c_o")
    out = unifyheads(out)
    return (toqueries, unifyheads, Model([x, keys, values], [out]))
  }
}

private func PixArtMSBlock<FloatType: TensorNumeric & BinaryFloatingPoint>(
  prefix: (String, String), k: Int, h: Int, b: Int, hw: Int, t: (Int, Int),
  usesFlashAttention: FlashAttentionLevel,
  of: FloatType.Type = FloatType.self
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let shiftMsa = Input()
  let scaleMsa = Input()
  let gateMsa = Input()
  let keys = Input()
  let values = Input()
  let shiftMlp = Input()
  let scaleMlp = Input()
  let gateMlp = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn) = SelfAttention(
    k: k, h: h, b: b, t: hw, usesFlashAttention: usesFlashAttention)
  var out =
    x + gateMsa
    .* attn(norm1(x) .* scaleMsa + shiftMsa)
  let (toqueries2, unifyheads2, crossAttn) = CrossAttentionKeysAndValues(
    k: k, h: h, b: b, hw: hw, t: t, usesFlashAttention: usesFlashAttention)
  out = out + crossAttn(out, keys, values)
  let (fc1, fc2, mlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4, name: "mlp")
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = out + gateMlp
    .* mlp(norm2(out) .* scaleMlp + shiftMlp)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).attn.qkv.weight"] = [
        toqueries1.weight.name, tokeys1.weight.name, tovalues1.weight.name,
      ]
      mapping["\(prefix.0).attn.qkv.bias"] = [
        toqueries1.bias.name, tokeys1.bias.name, tovalues1.bias.name,
      ]
      mapping["\(prefix.0).attn.proj.weight"] = [unifyheads1.weight.name]
      mapping["\(prefix.0).attn.proj.bias"] = [unifyheads1.bias.name]
      mapping["\(prefix.0).cross_attn.q_linear.weight"] = [toqueries2.weight.name]
      mapping["\(prefix.0).cross_attn.q_linear.bias"] = [toqueries2.bias.name]
      mapping["\(prefix.0).cross_attn.proj.weight"] = [unifyheads2.weight.name]
      mapping["\(prefix.0).cross_attn.proj.bias"] = [unifyheads2.bias.name]
      mapping["\(prefix.0).mlp.fc1.weight"] = [fc1.weight.name]
      mapping["\(prefix.0).mlp.fc1.bias"] = [fc1.bias.name]
      mapping["\(prefix.0).mlp.fc2.weight"] = [fc2.weight.name]
      mapping["\(prefix.0).mlp.fc2.bias"] = [fc2.bias.name]
    case .diffusers:
      mapping["\(prefix.1).attn1.to_q.weight"] = [toqueries1.weight.name]
      mapping["\(prefix.1).attn1.to_q.bias"] = [toqueries1.bias.name]
      mapping["\(prefix.1).attn1.to_k.weight"] = [tokeys1.weight.name]
      mapping["\(prefix.1).attn1.to_k.bias"] = [tokeys1.bias.name]
      mapping["\(prefix.1).attn1.to_v.weight"] = [tovalues1.weight.name]
      mapping["\(prefix.1).attn1.to_v.bias"] = [tovalues1.bias.name]
      mapping["\(prefix.1).attn1.to_out.0.weight"] = [unifyheads1.weight.name]
      mapping["\(prefix.1).attn1.to_out.0.bias"] = [unifyheads1.bias.name]
      mapping["\(prefix.1).attn2.to_q.weight"] = [toqueries2.weight.name]
      mapping["\(prefix.1).attn2.to_q.bias"] = [toqueries2.bias.name]
      mapping["\(prefix.1).attn2.to_out.0.weight"] = [unifyheads2.weight.name]
      mapping["\(prefix.1).attn2.to_out.0.bias"] = [unifyheads2.bias.name]
      mapping["\(prefix.1).ff.net.0.proj.weight"] = [fc1.weight.name]
      mapping["\(prefix.1).ff.net.0.proj.bias"] = [fc1.bias.name]
      mapping["\(prefix.1).ff.net.2.weight"] = [fc2.weight.name]
      mapping["\(prefix.1).ff.net.2.bias"] = [fc2.bias.name]
    }
    return mapping
  }
  return (
    mapper,
    Model([x, shiftMsa, scaleMsa, gateMsa, keys, values, shiftMlp, scaleMlp, gateMlp], [out])
  )
}

public func PixArt<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: Int, height: Int, width: Int, channels: Int, layers: Int, tokenLength: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let posEmbed = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([batchSize, h * w, channels]) + posEmbed
  var mappers = [ModelWeightMapper]()
  var inputs = [Input]()
  for i in 0..<layers {
    let shiftMsa = Input()
    let scaleMsa = Input()
    let gateMsa = Input()
    let keys = Input()
    let values = Input()
    let shiftMlp = Input()
    let scaleMlp = Input()
    let gateMlp = Input()
    let (mapper, block) = PixArtMSBlock(
      prefix: ("blocks.\(i)", "transformer_blocks.\(i)"), k: channels / 16, h: 16, b: batchSize,
      hw: h * w, t: tokenLength,
      usesFlashAttention: usesFlashAttention, of: FloatType.self)
    out = block(out, shiftMsa, scaleMsa, gateMsa, keys, values, shiftMlp, scaleMlp, gateMlp)
    mappers.append(mapper)
    inputs.append(contentsOf: [
      shiftMsa, scaleMsa, gateMsa, keys, values, shiftMlp, scaleMlp, gateMlp,
    ])
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shift = Input()
  let scale = Input()
  inputs.append(contentsOf: [shift, scale])
  out = scale .* normFinal(out) + shift
  let linear = Dense(count: 2 * 2 * 8, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 2, 2, 8]).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    batchSize, h * 2, w * 2, 8,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping["pos_embed.proj.weight"] = [xEmbedder.weight.name]
      mapping["pos_embed.proj.bias"] = [xEmbedder.bias.name]
      mapping["proj_out.weight"] = [linear.weight.name]
      mapping["proj_out.bias"] = [linear.bias.name]
    case .generativeModels:
      mapping["x_embedder.proj.weight"] = [xEmbedder.weight.name]
      mapping["x_embedder.proj.bias"] = [xEmbedder.bias.name]
      mapping["final_layer.linear.weight"] = [linear.weight.name]
      mapping["final_layer.linear.bias"] = [linear.bias.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([x, posEmbed] + inputs, [out]))
}

private func LoRAMLP(
  hiddenSize: Int, intermediateSize: Int, configuration: LoRANetworkConfiguration, name: String
) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = LoRADense(count: intermediateSize, configuration: configuration, name: "\(name)_fc1")
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = LoRADense(
    count: hiddenSize, configuration: configuration, flags: [.Float32], name: "\(name)_fc2")
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

private func LoRASelfAttention(
  k: Int, h: Int, b: Int, t: Int, configuration: LoRANetworkConfiguration,
  usesFlashAttention: FlashAttentionLevel
) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let tokeys = LoRADense(count: k * h, configuration: configuration, name: "k")
  let toqueries = LoRADense(count: k * h, configuration: configuration, name: "q")
  let tovalues = LoRADense(count: k * h, configuration: configuration, name: "v")
  if usesFlashAttention != .none {
    let keys = tokeys(x).reshaped([b, t, h, k])
    let queries: Model.IO
    let scale: Float
    if usesFlashAttention == .scale1 {
      queries = (1.0 / Float(k).squareRoot() * toqueries(x)).reshaped([b, t, h, k])
      scale = 1
    } else {
      queries = toqueries(x).reshaped([b, t, h, k])
      scale = 1.0 / Float(k).squareRoot()
    }
    let values = tovalues(x).reshaped([b, t, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    var out = scaledDotProductAttention(queries, keys, values).reshaped([b, t, k * h])
    let unifyheads = LoRADense(count: k * h, configuration: configuration, name: "o")
    out = unifyheads(out)
    return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
  } else {
    let keys = tokeys(x).reshaped([b, t, h, k]).transposed(1, 2)
    // No scaling the queries.
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
      .transposed(1, 2)
    let values = tovalues(x).reshaped([b, t, h, k]).transposed(1, 2)
    var out: Model.IO
    if b * h <= 256 && configuration.testing {
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped([1, t, k], offset: [i, 0, 0], strides: [t * k, k, 1])
        let query = queries.reshaped([1, t, k], offset: [i, 0, 0], strides: [t * k, k, 1])
        let value = values.reshaped([1, t, k], offset: [i, 0, 0], strides: [t * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t, t])
        dot = dot.softmax()
        dot = dot.reshaped([1, t, t])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * t, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, t, t])
      out = dot * values
      out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
    }
    let unifyheads = LoRADense(count: k * h, configuration: configuration, name: "o")
    out = unifyheads(out)
    return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
  }
}

private func LoRACrossAttentionKeysAndValues(
  k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), configuration: LoRANetworkConfiguration,
  usesFlashAttention: FlashAttentionLevel
) -> (
  Model, Model, Model
) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = LoRADense(count: k * h, configuration: configuration, name: "c_q")
  if t.0 == t.1 {
    let t = t.0
    if usesFlashAttention != .none {
      let queries: Model.IO
      let scale: Float
      if usesFlashAttention == .scale1 {
        queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
        scale = 1
      } else {
        queries = toqueries(x).reshaped([b, hw, h, k])
        scale = 1.0 / Float(k).squareRoot()
      }
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
      let unifyheads = LoRADense(count: k * h, configuration: configuration, name: "c_o")
      out = unifyheads(out)
      return (toqueries, unifyheads, Model([x, keys, values], [out]))
    } else {
      let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      let transposedQueries = queries.transposed(1, 2)
      var dot = Matmul(transposeB: (2, 3))(transposedQueries, keys)
      dot = dot.reshaped([b * h * hw, t])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, hw, t])
      var out = dot * values
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      let unifyheads = LoRADense(count: k * h, configuration: configuration, name: "c_o")
      out = unifyheads(out)
      return (toqueries, unifyheads, Model([x, keys, values], [out]))
    }
  } else {
    let b0 = b / 2
    var out: Model.IO
    if usesFlashAttention != .none {
      let queries: Model.IO
      let scale: Float
      if usesFlashAttention == .scale1 {
        queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
        scale = 1
      } else {
        queries = toqueries(x).reshaped([b, hw, h, k])
        scale = 1.0 / Float(k).squareRoot()
      }
      let out0: Model.IO
      if b0 == 1 || t.0 >= t.1 {
        let keys0 = keys.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries0 = queries.reshaped([b0, hw, h, k])
        let values0 = values.reshaped(
          [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out0 = ScaledDotProductAttention(
          scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
        )(
          queries0, keys0, values0
        ).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys0 = keys.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries0 = queries.reshaped(
            [1, hw, h, k], offset: [i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values0 = values.reshaped(
            [1, t.0, h, k], offset: [i, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(
              scale: scale,
              flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
            )(
              queries0, keys0, values0
            ).reshaped([1, hw, h * k]))
        }
        out0 = Concat(axis: 0)(outs)
      }
      let out1: Model.IO
      if b0 == 1 || t.1 >= t.0 {
        let keys1 = keys.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        let queries1 = queries.reshaped(
          [b0, hw, h, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
        let values1 = values.reshaped(
          [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
        )
        out1 = ScaledDotProductAttention(
          scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
        )(
          queries1, keys1, values1
        ).reshaped([b0, hw, h * k])
      } else {
        var outs = [Model.IO]()
        for i in 0..<b0 {
          let keys1 = keys.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          let queries1 = queries.reshaped(
            [1, hw, h, k], offset: [b0 + i, 0, 0, 0], strides: [h * hw * k, h * k, k, 1])
          let values1 = values.reshaped(
            [1, t.1, h, k], offset: [b0 + i, 0, 0, 0],
            strides: [max(t.0, t.1) * h * k, h * k, k, 1]
          )
          outs.append(
            ScaledDotProductAttention(
              scale: scale,
              flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
            )(
              queries1, keys1, values1
            ).reshaped([1, hw, h * k]))
        }
        out1 = Concat(axis: 0)(outs)
      }
      out = Functional.concat(axis: 0, out0, out1)
    } else {
      let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      let transposedQueries = queries.transposed(1, 2)
      let keys0 = keys.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      let queries0 = transposedQueries.reshaped([b0, h, hw, k])
      let values0 = values.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot0 = Matmul(transposeB: (2, 3))(queries0, keys0)
      dot0 = dot0.reshaped([b0 * h * hw, t.0])
      dot0 = dot0.softmax()
      dot0 = dot0.reshaped([b0, h, hw, t.0])
      var out0 = dot0 * values0
      out0 = out0.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      let keys1 = keys.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      let queries1 = transposedQueries.reshaped(
        [b0, h, hw, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, hw * k, k, 1])
      let values1 = values.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot1 = Matmul(transposeB: (2, 3))(queries1, keys1)
      dot1.add(dependencies: [out0])
      dot1 = dot1.reshaped([b0 * h * hw, t.1])
      dot1 = dot1.softmax()
      dot1 = dot1.reshaped([b0, h, hw, t.1])
      var out1 = dot1 * values1
      out1 = out1.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      out = Functional.concat(axis: 0, out0, out1)
    }
    let unifyheads = LoRADense(count: k * h, configuration: configuration, name: "c_o")
    out = unifyheads(out)
    return (toqueries, unifyheads, Model([x, keys, values], [out]))
  }
}

private func LoRAPixArtMSBlock<FloatType: TensorNumeric & BinaryFloatingPoint>(
  prefix: (String, String), k: Int, h: Int, b: Int, hw: Int, t: (Int, Int),
  configuration: LoRANetworkConfiguration,
  usesFlashAttention: FlashAttentionLevel, of: FloatType.Type = FloatType.self
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let shiftMsa = Input()
  let scaleMsa = Input()
  let gateMsa = Input()
  let keys = Input()
  let values = Input()
  let shiftMlp = Input()
  let scaleMlp = Input()
  let gateMlp = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn) = LoRASelfAttention(
    k: k, h: h, b: b, t: hw, configuration: configuration, usesFlashAttention: usesFlashAttention)
  var out =
    x + gateMsa
    .* attn(norm1(x) .* scaleMsa + shiftMsa)
  let (toqueries2, unifyheads2, crossAttn) = LoRACrossAttentionKeysAndValues(
    k: k, h: h, b: b, hw: hw, t: t, configuration: configuration,
    usesFlashAttention: usesFlashAttention)
  out = out + crossAttn(out, keys, values)
  let (fc1, fc2, mlp) = LoRAMLP(
    hiddenSize: k * h, intermediateSize: k * h * 4, configuration: configuration, name: "mlp")
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = out + gateMlp
    .* mlp(norm2(out) .* scaleMlp + shiftMlp)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).attn.qkv.weight"] = [
        toqueries1.weight.name, tokeys1.weight.name, tovalues1.weight.name,
      ]
      mapping["\(prefix.0).attn.qkv.bias"] = [
        toqueries1.bias.name, tokeys1.bias.name, tovalues1.bias.name,
      ]
      mapping["\(prefix.0).attn.proj.weight"] = [unifyheads1.weight.name]
      mapping["\(prefix.0).attn.proj.bias"] = [unifyheads1.bias.name]
      mapping["\(prefix.0).cross_attn.q_linear.weight"] = [toqueries2.weight.name]
      mapping["\(prefix.0).cross_attn.q_linear.bias"] = [toqueries2.bias.name]
      mapping["\(prefix.0).cross_attn.proj.weight"] = [unifyheads2.weight.name]
      mapping["\(prefix.0).cross_attn.proj.bias"] = [unifyheads2.bias.name]
      mapping["\(prefix.0).mlp.fc1.weight"] = [fc1.weight.name]
      mapping["\(prefix.0).mlp.fc1.bias"] = [fc1.bias.name]
      mapping["\(prefix.0).mlp.fc2.weight"] = [fc2.weight.name]
      mapping["\(prefix.0).mlp.fc2.bias"] = [fc2.bias.name]
    case .diffusers:
      mapping["\(prefix.1).attn1.to_q.weight"] = [toqueries1.weight.name]
      mapping["\(prefix.1).attn1.to_q.bias"] = [toqueries1.bias.name]
      mapping["\(prefix.1).attn1.to_k.weight"] = [tokeys1.weight.name]
      mapping["\(prefix.1).attn1.to_k.bias"] = [tokeys1.bias.name]
      mapping["\(prefix.1).attn1.to_v.weight"] = [tovalues1.weight.name]
      mapping["\(prefix.1).attn1.to_v.bias"] = [tovalues1.bias.name]
      mapping["\(prefix.1).attn1.to_out.0.weight"] = [unifyheads1.weight.name]
      mapping["\(prefix.1).attn1.to_out.0.bias"] = [unifyheads1.bias.name]
      mapping["\(prefix.1).attn2.to_q.weight"] = [toqueries2.weight.name]
      mapping["\(prefix.1).attn2.to_q.bias"] = [toqueries2.bias.name]
      mapping["\(prefix.1).attn2.to_out.0.weight"] = [unifyheads2.weight.name]
      mapping["\(prefix.1).attn2.to_out.0.bias"] = [unifyheads2.bias.name]
      mapping["\(prefix.1).ff.net.0.proj.weight"] = [fc1.weight.name]
      mapping["\(prefix.1).ff.net.0.proj.bias"] = [fc1.bias.name]
      mapping["\(prefix.1).ff.net.2.weight"] = [fc2.weight.name]
      mapping["\(prefix.1).ff.net.2.bias"] = [fc2.bias.name]
    }
    return mapping
  }
  return (
    mapper,
    Model([x, shiftMsa, scaleMsa, gateMsa, keys, values, shiftMlp, scaleMlp, gateMlp], [out])
  )
}

public func LoRAPixArt<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: Int, height: Int, width: Int, channels: Int, layers: Int, tokenLength: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration,
  of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let posEmbed = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([batchSize, h * w, channels]) + posEmbed
  var mappers = [ModelWeightMapper]()
  var inputs = [Input]()
  for i in 0..<layers {
    let shiftMsa = Input()
    let scaleMsa = Input()
    let gateMsa = Input()
    let keys = Input()
    let values = Input()
    let shiftMlp = Input()
    let scaleMlp = Input()
    let gateMlp = Input()
    let (mapper, block) = LoRAPixArtMSBlock(
      prefix: ("blocks.\(i)", "transformer_blocks.\(i)"), k: channels / 16, h: 16, b: batchSize,
      hw: h * w, t: tokenLength, configuration: LoRAConfiguration,
      usesFlashAttention: usesFlashAttention, of: FloatType.self)
    out = block(out, shiftMsa, scaleMsa, gateMsa, keys, values, shiftMlp, scaleMlp, gateMlp)
    mappers.append(mapper)
    inputs.append(contentsOf: [
      shiftMsa, scaleMsa, gateMsa, keys, values, shiftMlp, scaleMlp, gateMlp,
    ])
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shift = Input()
  let scale = Input()
  inputs.append(contentsOf: [shift, scale])
  out = scale .* normFinal(out) + shift
  let linear = LoRADense(count: 2 * 2 * 8, configuration: LoRAConfiguration, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 2, 2, 8]).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    batchSize, h * 2, w * 2, 8,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping["pos_embed.proj.weight"] = [xEmbedder.weight.name]
      mapping["pos_embed.proj.bias"] = [xEmbedder.bias.name]
      mapping["proj_out.weight"] = [linear.weight.name]
      mapping["proj_out.bias"] = [linear.bias.name]
    case .generativeModels:
      mapping["x_embedder.proj.weight"] = [xEmbedder.weight.name]
      mapping["x_embedder.proj.bias"] = [xEmbedder.bias.name]
      mapping["final_layer.linear.weight"] = [linear.weight.name]
      mapping["final_layer.linear.bias"] = [linear.bias.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([x, posEmbed] + inputs, [out]))
}

private func CrossAttentionFixed(k: Int, h: Int, b: Int, t: (Int, Int), usesFlashAttention: Bool)
  -> (
    Model, Model, Model
  )
{
  let context = Input()
  let tokeys = Dense(count: k * h, name: "c_k")
  let tovalues = Dense(count: k * h, name: "c_v")
  // We shouldn't transpose if we are going to do that within the UNet.
  if t.0 == t.1 {
    var keys = tokeys(context).reshaped([b, t.0, h, k])
    var values = tovalues(context).reshaped([b, t.0, h, k])
    if !usesFlashAttention {
      keys = keys.transposed(1, 2)
      values = values.transposed(1, 2)
    }
    return (tokeys, tovalues, Model([context], [keys, values]))
  } else {
    let keys = tokeys(context)
    let values = tovalues(context)
    return (tokeys, tovalues, Model([context], [keys, values]))
  }
}

private func PixArtMSBlockFixed<FloatType: TensorNumeric & BinaryFloatingPoint>(
  prefix: (String, String), k: Int, h: Int, b: Int, t: (Int, Int), usesFlashAttention: Bool,
  of: FloatType.Type = FloatType.self
) -> (
  ModelWeightMapper, Model
) {
  let context = Input()
  let shiftMsa = Input()
  let scaleMsa = Input()
  let gateMsa = Input()
  let shiftMlp = Input()
  let scaleMlp = Input()
  let gateMlp = Input()
  let shiftMsaShift = Parameter<FloatType>(.GPU(0), .HWC(1, 1, k * h), name: "scale_shift_table_0")
  let scaleMsaShift = Parameter<FloatType>(.GPU(0), .HWC(1, 1, k * h), name: "scale_shift_table_1")
  let gateMsaShift = Parameter<FloatType>(.GPU(0), .HWC(1, 1, k * h), name: "scale_shift_table_2")
  var outs = [Model.IO]()
  outs.append(shiftMsa + shiftMsaShift)
  outs.append(scaleMsa + scaleMsaShift)
  outs.append(gateMsa + gateMsaShift)
  let (tokeys2, tovalues2, crossAttn) = CrossAttentionFixed(
    k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention)
  outs.append(crossAttn(context))
  let shiftMlpShift = Parameter<FloatType>(.GPU(0), .HWC(1, 1, k * h), name: "scale_shift_table_3")
  let scaleMlpShift = Parameter<FloatType>(.GPU(0), .HWC(1, 1, k * h), name: "scale_shift_table_4")
  let gateMlpShift = Parameter<FloatType>(.GPU(0), .HWC(1, 1, k * h), name: "scale_shift_table_5")
  outs.append(shiftMlp + shiftMlpShift)
  outs.append(scaleMlp + scaleMlpShift)
  outs.append(gateMlp + gateMlpShift)
  let mapper: ModelWeightMapper = { format in
    let formatPrefix: String
    switch format {
    case .generativeModels:
      formatPrefix = prefix.0
    case .diffusers:
      formatPrefix = prefix.1
    }
    var mapping = ModelWeightMapping()
    mapping["\(formatPrefix).scale_shift_table"] = [
      shiftMsaShift.weight.name, scaleMsaShift.weight.name, gateMsaShift.weight.name,
      shiftMlpShift.weight.name, scaleMlpShift.weight.name, gateMlpShift.weight.name,
    ]
    switch format {
    case .generativeModels:
      mapping["\(formatPrefix).cross_attn.kv_linear.weight"] = [
        tokeys2.weight.name, tovalues2.weight.name,
      ]
      mapping["\(formatPrefix).cross_attn.kv_linear.bias"] = [
        tokeys2.bias.name, tovalues2.bias.name,
      ]
    case .diffusers:
      mapping["\(formatPrefix).attn2.to_k.weight"] = [tokeys2.weight.name]
      mapping["\(formatPrefix).attn2.to_k.bias"] = [tokeys2.bias.name]
      mapping["\(formatPrefix).attn2.to_v.weight"] = [tovalues2.weight.name]
      mapping["\(formatPrefix).attn2.to_v.bias"] = [tovalues2.bias.name]
    }
    return mapping
  }
  return (
    mapper, Model([context, shiftMsa, scaleMsa, gateMsa, shiftMlp, scaleMlp, gateMlp], outs)
  )
}

public func PixArtFixed<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: Int, channels: Int, layers: Int, tokenLength: (Int, Int),
  usesFlashAttention: Bool, of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let t = Input()
  let y = Input()
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: channels)
  let t0 = tEmbedder(t)
  let t1 = t0.swish()
  let tBlock = (0..<6).map { Dense(count: channels, name: "t_block_\($0)") }
  var adaln = tBlock.map { $0(t1) }
  adaln[1] = 1 + adaln[1]
  adaln[4] = 1 + adaln[4]
  let (fc1, fc2, yEmbedder) = MLP(
    hiddenSize: channels, intermediateSize: channels, name: "y_embedder")
  let y0 = yEmbedder(y)
  var mappers = [ModelWeightMapper]()
  var outs = [Model.IO]()
  for i in 0..<layers {
    let (mapper, block) = PixArtMSBlockFixed(
      prefix: ("blocks.\(i)", "transformer_blocks.\(i)"), k: channels / 16, h: 16, b: batchSize,
      t: tokenLength,
      usesFlashAttention: usesFlashAttention, of: FloatType.self)
    let out = block(y0, adaln[0], adaln[1], adaln[2], adaln[3], adaln[4], adaln[5])
    mappers.append(mapper)
    outs.append(out)
  }
  let shiftShift = Parameter<FloatType>(
    .GPU(0), .HWC(1, 1, channels), name: "final_scale_shift_table_0")
  let scaleShift = Parameter<FloatType>(
    .GPU(0), .HWC(1, 1, channels), name: "final_scale_shift_table_1")
  outs.append(shiftShift + t0)
  outs.append(scaleShift + 1 + t0)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping["adaln_single.emb.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
      mapping["adaln_single.emb.timestep_embedder.linear_1.bias"] = [tMlp0.bias.name]
      mapping["adaln_single.emb.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
      mapping["adaln_single.emb.timestep_embedder.linear_2.bias"] = [tMlp2.bias.name]
      mapping["adaln_single.linear.weight"] = ModelWeightElement(tBlock.map { $0.weight.name })
      mapping["adaln_single.linear.bias"] = ModelWeightElement(tBlock.map { $0.bias.name })
      mapping["caption_projection.linear_1.weight"] = [fc1.weight.name]
      mapping["caption_projection.linear_1.bias"] = [fc1.bias.name]
      mapping["caption_projection.linear_2.weight"] = [fc2.weight.name]
      mapping["caption_projection.linear_2.bias"] = [fc2.bias.name]
      mapping["scale_shift_table"] = [shiftShift.weight.name, scaleShift.weight.name]
    case .generativeModels:
      mapping["t_embedder.mlp.0.weight"] = [tMlp0.weight.name]
      mapping["t_embedder.mlp.0.bias"] = [tMlp0.bias.name]
      mapping["t_embedder.mlp.2.weight"] = [tMlp2.weight.name]
      mapping["t_embedder.mlp.2.bias"] = [tMlp2.bias.name]
      mapping["t_block.1.weight"] = ModelWeightElement(tBlock.map { $0.weight.name })
      mapping["t_block.1.bias"] = ModelWeightElement(tBlock.map { $0.bias.name })
      mapping["y_embedder.y_proj.fc1.weight"] = [fc1.weight.name]
      mapping["y_embedder.y_proj.fc1.bias"] = [fc1.bias.name]
      mapping["y_embedder.y_proj.fc2.weight"] = [fc2.weight.name]
      mapping["y_embedder.y_proj.fc2.bias"] = [fc2.bias.name]
      mapping["final_layer.scale_shift_table"] = [shiftShift.weight.name, scaleShift.weight.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (mapper, Model([t, y], outs))
}
