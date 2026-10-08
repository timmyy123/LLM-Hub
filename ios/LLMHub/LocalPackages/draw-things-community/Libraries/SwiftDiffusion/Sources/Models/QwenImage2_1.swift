import DiffusionMappings
import Foundation
import NNC

struct QwenImage2_1Segment {
  let image: Bool
  let length: Int
  let sourceOffset: Int
  var height = 0
  var width = 0
}

func QwenImage2_1AttentionMask(_ segments: [QwenImage2_1Segment])
  -> Tensor<Float>
{
  let count = segments.reduce(0) { $0 + $1.length }
  var mask = Tensor<Float>(.CPU, .NHWC(1, 1, count, count))
  _ = mask.withUnsafeMutableBytes { $0.initializeMemory(as: UInt8.self, repeating: 0) }
  var start = 0
  for segment in segments {
    let end = start + segment.length
    for row in start..<end {
      let allowed = segment.image ? end : row + 1
      if allowed < count {
        for column in allowed..<count { mask[0, 0, row, column] = -.infinity }
      }
    }
    start = end
  }
  return mask
}

func QwenImage2_1RotaryEmbedding(_ segments: [QwenImage2_1Segment]) -> Tensor<Float> {
  let count = segments.reduce(0) { $0 + $1.length }
  var result = Tensor<Float>(.CPU, .NHWC(1, count, 1, 128))
  var token = 0
  var position = 0
  for segment in segments {
    for local in 0..<segment.length {
      let positions =
        segment.image
        ? [
          position, local / segment.width - (segment.height - segment.height / 2),
          local % segment.width - (segment.width - segment.width / 2),
        ]
        : [position + local, position + local, position + local]
      var offset = 0
      for (axis, dimension) in [16, 56, 56].enumerated() {
        for pair in 0..<(dimension / 2) {
          let frequency = pow(Float(10_000), -Float(2 * pair) / Float(dimension))
          let phase = Float(positions[axis]) * frequency
          result[0, token, 0, offset + 2 * pair] = cos(phase)
          result[0, token, 0, offset + 2 * pair + 1] = sin(phase)
        }
        offset += dimension
      }
      token += 1
    }
    position += segment.image ? max(segment.height, segment.width) : segment.length
  }
  return result
}

private func TransformerBlockFixed<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, batchSize: Int, channels: Int, count: Int,
  segments: [(length: Int, image: Bool)], outputKVOnly: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotary = Input()
  let mask = Input()
  let scale1 = Input()
  let gate1 = Input()
  let scale2 = Input()
  let gate2 = Input()
  let heads = channels / 128
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let attentionInput = norm1(x).to(T.dataType) .* scale1
  let toKeys = Dense(count: channels, noBias: true, name: "x_k")
  let toValues = Dense(count: channels, noBias: true, name: "x_v")
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  let keys = Functional.cmul(
    left: normK(toKeys(attentionInput).reshaped([batchSize, count, heads, 128])),
    right: rotary
  )
  let values = toValues(attentionInput).reshaped([batchSize, count, heads, 128])
  if outputKVOnly {
    let mapper: ModelWeightMapper = { _ in
      var mapping = ModelWeightMapping()
      mapping["\(prefix).attn.to_k.weight"] = [toKeys.weight.name]
      mapping["\(prefix).attn.to_v.weight"] = [toValues.weight.name]
      mapping["\(prefix).attn.norm_k.weight"] = [normK.weight.name]
      return mapping
    }
    return (mapper, Model([x, rotary, scale1], [keys, values]))
  }
  let toQueries = Dense(count: channels, noBias: true, name: "x_q")
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  var queries = Functional.cmul(
    left: normQ(toQueries(attentionInput).reshaped([batchSize, count, heads, 128])),
    right: rotary
  )
  let attended: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = keys.transposed(1, 2)
    queries = ((1 / Float(128).squareRoot()) * queries).transposed(1, 2)
    let transposedValues = values.transposed(1, 2)
    if batchSize * heads <= 256 {
      let causalAttentionMask = mask.reshaped([count, count])
      var outs = [Model.IO]()
      for i in 0..<(batchSize * heads) {
        let key = transposedKeys.reshaped(
          [1, count, 128], offset: [i, 0, 0], strides: [count * 128, 128, 1])
        let query = queries.reshaped(
          [1, count, 128], offset: [i, 0, 0], strides: [count * 128, 128, 1])
        let value = transposedValues.reshaped(
          [1, count, 128], offset: [i, 0, 0], strides: [count * 128, 128, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([count, count]) + causalAttentionMask
        dot = dot.softmax()
        dot = dot.reshaped([1, count, count])
        outs.append(dot * value)
      }
      attended = Concat(axis: 0)(outs).reshaped([batchSize, heads, count, 128])
        .transposed(1, 2).reshaped([batchSize * count, channels])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, transposedKeys) + mask
      dot = dot.reshaped([batchSize * heads * count, count]).softmax()
      dot = dot.reshaped([batchSize, heads, count, count])
      attended = (dot * transposedValues).transposed(1, 2)
        .reshaped([batchSize * count, channels])
    }
  case .scale1:
    queries = (1 / Float(128).squareRoot()) * queries
    var start = 0
    var parts = [Model.IO]()
    for segment in segments where segment.length > 0 {
      let end = start + segment.length
      let q = queries.reshaped(
        [batchSize, segment.length, heads, 128], offset: [0, start, 0, 0],
        strides: [count * channels, channels, 128, 1]
      ).contiguous()
      let k = keys.reshaped(
        [batchSize, end, heads, 128], strides: [count * channels, channels, 128, 1]
      )
      .contiguous()
      let v = values.reshaped(
        [batchSize, end, heads, 128], strides: [count * channels, channels, 128, 1]
      )
      .contiguous()
      // Causal attention is aligned to the end of the key sequence, including preceding blocks.
      let attention = ScaledDotProductAttention(
        scale: 1, isCausal: !segment.image, flags: [.Float16])
      parts.append(attention(q, k, v).reshaped([batchSize, segment.length, channels]))
      start = end
    }
    attended = (parts.count == 1 ? parts[0] : Concat(axis: 1)(parts)).reshaped([
      batchSize * count, channels,
    ])
  case .scaleMerged, .quantized:
    var start = 0
    var parts = [Model.IO]()
    for segment in segments where segment.length > 0 {
      let end = start + segment.length
      let q = queries.reshaped(
        [batchSize, segment.length, heads, 128], offset: [0, start, 0, 0],
        strides: [count * channels, channels, 128, 1]
      ).contiguous()
      let k = keys.reshaped(
        [batchSize, end, heads, 128], strides: [count * channels, channels, 128, 1]
      )
      .contiguous()
      let v = values.reshaped(
        [batchSize, end, heads, 128], strides: [count * channels, channels, 128, 1]
      )
      .contiguous()
      // Causal attention is aligned to the end of the key sequence, including preceding blocks.
      let attention = ScaledDotProductAttention(
        scale: 1 / Float(128).squareRoot(),
        isCausal: !segment.image,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      parts.append(attention(q, k, v).reshaped([batchSize, segment.length, channels]))
      start = end
    }
    attended = (parts.count == 1 ? parts[0] : Concat(axis: 1)(parts)).reshaped([
      batchSize * count, channels,
    ])
  }
  let unifyHeads = Dense(count: channels, noBias: true, name: "x_o")
  var out = x + (unifyHeads(attended) .* gate1).to(of: x)
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let ffnInput = norm2(out).to(T.dataType) .* scale2
  let up = Dense(count: channels * 3, noBias: true, name: "ffn_up_proj")
  let gate = Dense(count: channels * 3, noBias: true, name: "ffn_gate_proj")
  let down = Dense(count: channels, noBias: true, name: "ffn_down_proj")
  let product = Functional.swishMul(value: up(ffnInput), gate: gate(ffnInput))
  out = out + (down(product) .* gate2).to(of: out)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn.to_k.weight"] = [toKeys.weight.name]
    mapping["\(prefix).attn.to_v.weight"] = [toValues.weight.name]
    mapping["\(prefix).attn.to_q.weight"] = [toQueries.weight.name]
    mapping["\(prefix).attn.norm_k.weight"] = [normK.weight.name]
    mapping["\(prefix).attn.norm_q.weight"] = [normQ.weight.name]
    mapping["\(prefix).attn.to_out.0.weight"] = [unifyHeads.weight.name]
    mapping["\(prefix).img_mlp.proj.weight"] = [up.weight.name]
    mapping["\(prefix).img_mlp.gate_layer.weight"] = [gate.weight.name]
    mapping["\(prefix).img_mlp.out.weight"] = [down.weight.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rotary, scale1, gate1, scale2, gate2] + (usesFlashAttention != .none ? [] : [mask]),
      [out, keys, values])
  )
}

private func TransformerBlock<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, batchSize: Int, channels: Int, count: Int,
  prefixLength: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotary = Input()
  let scale1 = Input()
  let gate1 = Input()
  let scale2 = Input()
  let gate2 = Input()
  let cachedK = Input()
  let cachedV = Input()
  let heads = channels / 128
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let attentionInput = norm1(x).to(T.dataType) .* scale1
  let toKeys = Dense(count: channels, noBias: true, name: "x_k")
  let toValues = Dense(count: channels, noBias: true, name: "x_v")
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  let keys = Functional.cmul(
    left: normK(toKeys(attentionInput).reshaped([batchSize, count, heads, 128])),
    right: rotary
  )
  let values = toValues(attentionInput).reshaped([batchSize, count, heads, 128])
  let toQueries = Dense(count: channels, noBias: true, name: "x_q")
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  var queries = Functional.cmul(
    left: normQ(toQueries(attentionInput).reshaped([batchSize, count, heads, 128])),
    right: rotary
  )
  // Keep RoPE outputs contiguous when concatenating batched cached K/V, as in Flux2.
  let allKeys = Functional.concat(
    axis: 1, cachedK, keys, flags: batchSize > 1 ? [.disableOpt] : [])
  let allValues = Functional.concat(
    axis: 1, cachedV, values, flags: batchSize > 1 ? [.disableOpt] : [])
  let attended: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = allKeys.transposed(1, 2)
    queries = ((1 / Float(128).squareRoot()) * queries).transposed(1, 2)
    let transposedValues = allValues.transposed(1, 2)
    if batchSize * heads <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(batchSize * heads) {
        let key = transposedKeys.reshaped(
          [1, prefixLength + count, 128], offset: [i, 0, 0],
          strides: [(prefixLength + count) * 128, 128, 1])
        let query = queries.reshaped(
          [1, count, 128], offset: [i, 0, 0], strides: [count * 128, 128, 1])
        let value = transposedValues.reshaped(
          [1, prefixLength + count, 128], offset: [i, 0, 0],
          strides: [(prefixLength + count) * 128, 128, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([count, prefixLength + count])
        dot = dot.softmax()
        dot = dot.reshaped([1, count, prefixLength + count])
        outs.append(dot * value)
      }
      attended = Concat(axis: 0)(outs).reshaped([batchSize, heads, count, 128])
        .transposed(1, 2).reshaped([batchSize * count, channels])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, transposedKeys)
      dot = dot.reshaped([batchSize * heads * count, prefixLength + count]).softmax()
      dot = dot.reshaped([batchSize, heads, count, prefixLength + count])
      attended = (dot * transposedValues).transposed(1, 2)
        .reshaped([batchSize * count, channels])
    }
  case .scale1:
    queries = (1 / Float(128).squareRoot()) * queries
    let attention = ScaledDotProductAttention(
      scale: 1,
      isCausal: false,
      flags: [.Float16])
    attended = attention(queries, allKeys, allValues).reshaped([batchSize * count, channels])
  case .scaleMerged, .quantized:
    let attention = ScaledDotProductAttention(
      scale: 1 / Float(128).squareRoot(),
      isCausal: false,
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    attended = attention(queries, allKeys, allValues).reshaped([batchSize * count, channels])
  }
  let unifyHeads = Dense(count: channels, noBias: true, name: "x_o")
  var out = x + (unifyHeads(attended) .* gate1).to(of: x)
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let ffnInput = norm2(out).to(T.dataType) .* scale2
  let up = Dense(count: channels * 3, noBias: true, name: "ffn_up_proj")
  let gate = Dense(count: channels * 3, noBias: true, name: "ffn_gate_proj")
  let down = Dense(count: channels, noBias: true, name: "ffn_down_proj")
  let product = Functional.swishMul(value: up(ffnInput), gate: gate(ffnInput))
  out = out + (down(product) .* gate2).to(of: out)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn.to_k.weight"] = [toKeys.weight.name]
    mapping["\(prefix).attn.to_v.weight"] = [toValues.weight.name]
    mapping["\(prefix).attn.to_q.weight"] = [toQueries.weight.name]
    mapping["\(prefix).attn.norm_k.weight"] = [normK.weight.name]
    mapping["\(prefix).attn.norm_q.weight"] = [normQ.weight.name]
    mapping["\(prefix).attn.to_out.0.weight"] = [unifyHeads.weight.name]
    mapping["\(prefix).img_mlp.proj.weight"] = [up.weight.name]
    mapping["\(prefix).img_mlp.gate_layer.weight"] = [gate.weight.name]
    mapping["\(prefix).img_mlp.out.weight"] = [down.weight.name]
    return mapping
  }
  return (mapper, Model([x, rotary, scale1, gate1, scale2, gate2, cachedK, cachedV], [out]))
}

// Outputs: five timestep tables in the model dtype, then post-RoPE K and V for each block.
public func QwenImage2_1Fixed<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, textLength: Int, referenceLength: Int, timesteps: Int,
  channels: Int, layers: Int, segments: [(length: Int, image: Bool)],
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let count = textLength + referenceLength
  precondition(segments.reduce(0) { $0 + $1.length } == count)
  let context = Input()
  let time = Input()
  let rotary = Input()
  let mask = Input()
  let reference = Input()
  let textNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "context_norm")
  let textIn = Dense(count: channels, noBias: true, name: "context_embedder_0")
  let textOut = Dense(count: channels, noBias: true, name: "context_embedder_1")
  let xEmbedder =
    referenceLength > 0 ? Dense(count: channels, noBias: true, name: "x_embedder") : nil
  let time0 = Dense(count: channels, noBias: true, name: "t_embedder_0")
  let time2 = Dense(count: channels, noBias: true, name: "t_embedder_1")
  let temb = time2(time0(time).swish())
  let modulationInput = temb.swish()
  let modulation = (0..<4).map { Dense(count: channels, noBias: true, name: "x_ada_ln_\($0)") }
  let finalModulation = Dense(count: channels, noBias: true, name: "ada_ln_0")
  let tables = [
    1 + modulation[0](modulationInput),
    modulation[1](modulationInput).tanh(),
    1 + modulation[2](modulationInput),
    modulation[3](modulationInput).tanh(),
    1 + finalModulation(modulationInput),
  ]
  let outputs = tables.map {
    $0.reshaped([timesteps, channels], strides: [channels, 1]).contiguous()
  }
  let prefixModulation = tables.prefix(4).map {
    $0.reshaped([1, channels], offset: [timesteps, 0], strides: [channels, 1]).contiguous()
  }
  let encodedText = textOut(
    GELU(approximate: .tanh)(
      textIn(
        textNorm(context.reshaped([batchSize * textLength, 4096])))))
  let encodedReference = xEmbedder.map { $0(reference) }
  var prefixes = [Model.IO]()
  for batch in 0..<batchSize {
    var textOffset = batch * textLength
    var referenceOffset = 0
    for segment in segments where segment.length > 0 {
      if let encodedReference = encodedReference, segment.image {
        prefixes.append(
          encodedReference.reshaped(
            [segment.length, channels], offset: [referenceOffset, 0], strides: [channels, 1]
          ).contiguous())
        referenceOffset += segment.length
      } else {
        prefixes.append(
          encodedText.reshaped(
            [segment.length, channels], offset: [textOffset, 0], strides: [channels, 1]
          ).contiguous())
        textOffset += segment.length
      }
    }
  }
  var out = (prefixes.count == 1 ? prefixes[0] : Concat(axis: 0)(prefixes)).to(.Float32)
  var mappers = [ModelWeightMapper]()
  var cachedKVs = [Model.IO]()
  for i in 0..<layers {
    let (mapper, block) = TransformerBlockFixed(
      dataType, prefix: "transformer_blocks.\(i)", batchSize: batchSize, channels: channels,
      count: count,
      segments: segments, outputKVOnly: i == layers - 1, usesFlashAttention: usesFlashAttention)
    mappers.append(mapper)
    if i == layers - 1 {
      let kv = block(out, rotary, prefixModulation[0])
      cachedKVs += [kv[0], kv[1]]
    } else {
      let result = block(
        [out, rotary] + prefixModulation + (usesFlashAttention != .none ? [] : [mask]))
      out = result[0]
      cachedKVs += [result[1], result[2]]
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers { mapping.merge(mapper(format)) { v, _ in v } }
    mapping["txt_in.text_norm.weight"] = [textNorm.weight.name]
    mapping["txt_in.in_layer.weight"] = [textIn.weight.name]
    mapping["txt_in.out_layer.weight"] = [textOut.weight.name]
    if let xEmbedder = xEmbedder { mapping["img_in.weight"] = [xEmbedder.weight.name] }
    mapping["time_text_embed.timestep_embedder.linear_1.weight"] = [time0.weight.name]
    mapping["time_text_embed.timestep_embedder.linear_2.weight"] = [time2.weight.name]
    mapping["modulation.1.weight"] = ModelWeightElement(modulation.map { $0.weight.name })
    mapping["norm_out.linear.weight"] = [finalModulation.weight.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [context, time, rotary] + (usesFlashAttention != .none ? [] : [mask])
        + (referenceLength > 0 ? [reference] : []),
      outputs + cachedKVs)
  )
}

// Inputs: target, target RoPE, five current modulation rows, 32 K/V pairs.
public func QwenImage2_1<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, height: Int, width: Int, prefixLength: Int,
  channels: Int, layers: Int, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotary = Input()
  let modulation = (0..<5).map { _ in Input() }
  let cachedKVs = (0..<(2 * layers)).map { _ in Input() }
  let count = height * width
  let xEmbedder = Dense(count: channels, noBias: true, name: "x_embedder")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let projOut = Dense(count: 64, noBias: true, name: "linear")
  var out = xEmbedder(x.reshaped([batchSize * count, 64])).to(.Float32)
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = TransformerBlock(
      dataType, prefix: "transformer_blocks.\(i)", batchSize: batchSize, channels: channels,
      count: count,
      prefixLength: prefixLength, usesFlashAttention: usesFlashAttention)
    mappers.append(mapper)
    out = block(
      [out, rotary] + Array(modulation.prefix(4)) + [cachedKVs[2 * i], cachedKVs[2 * i + 1]])
  }
  out = projOut(normFinal(out).to(T.dataType) .* modulation[4])
    .reshaped([batchSize, height, width, 64])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers { mapping.merge(mapper(format)) { v, _ in v } }
    mapping["img_in.weight"] = [xEmbedder.weight.name]
    mapping["proj_out.weight"] = [projOut.weight.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rotary] + modulation + cachedKVs,
      [out])
  )
}

private func LoRATransformerBlockFixed<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, layerIndex: Int, batchSize: Int, channels: Int, count: Int,
  segments: [(length: Int, image: Bool)], outputKVOnly: Bool,
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotary = Input()
  let mask = Input()
  let scale1 = Input()
  let gate1 = Input()
  let scale2 = Input()
  let gate2 = Input()
  let heads = channels / 128
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let attentionInput = norm1(x).to(T.dataType) .* scale1
  let toKeys = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex, name: "x_k")
  let toValues = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex, name: "x_v")
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  let keys = Functional.cmul(
    left: normK(toKeys(attentionInput).reshaped([batchSize, count, heads, 128])),
    right: rotary
  )
  let values = toValues(attentionInput).reshaped([batchSize, count, heads, 128])
  if outputKVOnly {
    let mapper: ModelWeightMapper = { _ in
      var mapping = ModelWeightMapping()
      mapping["\(prefix).attn.to_k.weight"] = [toKeys.weight.name]
      mapping["\(prefix).attn.to_v.weight"] = [toValues.weight.name]
      mapping["\(prefix).attn.norm_k.weight"] = [normK.weight.name]
      return mapping
    }
    return (mapper, Model([x, rotary, scale1], [keys, values]))
  }
  let toQueries = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex, name: "x_q")
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  var queries = Functional.cmul(
    left: normQ(toQueries(attentionInput).reshaped([batchSize, count, heads, 128])),
    right: rotary
  )
  let attended: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = keys.transposed(1, 2)
    queries = ((1 / Float(128).squareRoot()) * queries).transposed(1, 2)
    let transposedValues = values.transposed(1, 2)
    if batchSize * heads <= 256 {
      let causalAttentionMask = mask.reshaped([count, count])
      var outs = [Model.IO]()
      for i in 0..<(batchSize * heads) {
        let key = transposedKeys.reshaped(
          [1, count, 128], offset: [i, 0, 0], strides: [count * 128, 128, 1])
        let query = queries.reshaped(
          [1, count, 128], offset: [i, 0, 0], strides: [count * 128, 128, 1])
        let value = transposedValues.reshaped(
          [1, count, 128], offset: [i, 0, 0], strides: [count * 128, 128, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([count, count]) + causalAttentionMask
        dot = dot.softmax()
        dot = dot.reshaped([1, count, count])
        outs.append(dot * value)
      }
      attended = Concat(axis: 0)(outs).reshaped([batchSize, heads, count, 128])
        .transposed(1, 2).reshaped([batchSize * count, channels])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, transposedKeys) + mask
      dot = dot.reshaped([batchSize * heads * count, count]).softmax()
      dot = dot.reshaped([batchSize, heads, count, count])
      attended = (dot * transposedValues).transposed(1, 2)
        .reshaped([batchSize * count, channels])
    }
  case .scale1:
    queries = (1 / Float(128).squareRoot()) * queries
    var start = 0
    var parts = [Model.IO]()
    for segment in segments where segment.length > 0 {
      let end = start + segment.length
      let q = queries.reshaped(
        [batchSize, segment.length, heads, 128], offset: [0, start, 0, 0],
        strides: [count * channels, channels, 128, 1]
      ).contiguous()
      let k = keys.reshaped(
        [batchSize, end, heads, 128], strides: [count * channels, channels, 128, 1]
      )
      .contiguous()
      let v = values.reshaped(
        [batchSize, end, heads, 128], strides: [count * channels, channels, 128, 1]
      )
      .contiguous()
      // Causal attention is aligned to the end of the key sequence, including preceding blocks.
      let attention = ScaledDotProductAttention(
        scale: 1, isCausal: !segment.image, flags: [.Float16])
      parts.append(attention(q, k, v).reshaped([batchSize, segment.length, channels]))
      start = end
    }
    attended = (parts.count == 1 ? parts[0] : Concat(axis: 1)(parts)).reshaped([
      batchSize * count, channels,
    ])
  case .scaleMerged, .quantized:
    var start = 0
    var parts = [Model.IO]()
    for segment in segments where segment.length > 0 {
      let end = start + segment.length
      let q = queries.reshaped(
        [batchSize, segment.length, heads, 128], offset: [0, start, 0, 0],
        strides: [count * channels, channels, 128, 1]
      ).contiguous()
      let k = keys.reshaped(
        [batchSize, end, heads, 128], strides: [count * channels, channels, 128, 1]
      )
      .contiguous()
      let v = values.reshaped(
        [batchSize, end, heads, 128], strides: [count * channels, channels, 128, 1]
      )
      .contiguous()
      // Causal attention is aligned to the end of the key sequence, including preceding blocks.
      let attention = ScaledDotProductAttention(
        scale: 1 / Float(128).squareRoot(),
        isCausal: !segment.image,
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      parts.append(attention(q, k, v).reshaped([batchSize, segment.length, channels]))
      start = end
    }
    attended = (parts.count == 1 ? parts[0] : Concat(axis: 1)(parts)).reshaped([
      batchSize * count, channels,
    ])
  }
  let unifyHeads = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex, name: "x_o")
  var out = x + (unifyHeads(attended) .* gate1).to(of: x)
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let ffnInput = norm2(out).to(T.dataType) .* scale2
  let up = LoRADense(
    count: channels * 3, configuration: LoRAConfiguration, noBias: true, index: layerIndex,
    name: "ffn_up_proj")
  let gate = LoRADense(
    count: channels * 3, configuration: LoRAConfiguration, noBias: true, index: layerIndex,
    name: "ffn_gate_proj")
  let down = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex,
    name: "ffn_down_proj")
  let product = Functional.swishMul(value: up(ffnInput), gate: gate(ffnInput))
  out = out + (down(product) .* gate2).to(of: out)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn.to_k.weight"] = [toKeys.weight.name]
    mapping["\(prefix).attn.to_v.weight"] = [toValues.weight.name]
    mapping["\(prefix).attn.to_q.weight"] = [toQueries.weight.name]
    mapping["\(prefix).attn.norm_k.weight"] = [normK.weight.name]
    mapping["\(prefix).attn.norm_q.weight"] = [normQ.weight.name]
    mapping["\(prefix).attn.to_out.0.weight"] = [unifyHeads.weight.name]
    mapping["\(prefix).img_mlp.proj.weight"] = [up.weight.name]
    mapping["\(prefix).img_mlp.gate_layer.weight"] = [gate.weight.name]
    mapping["\(prefix).img_mlp.out.weight"] = [down.weight.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rotary, scale1, gate1, scale2, gate2] + (usesFlashAttention != .none ? [] : [mask]),
      [out, keys, values])
  )
}

private func LoRATransformerBlock<T: TensorNumeric>(
  _ dataType: T.Type, prefix: String, layerIndex: Int, batchSize: Int, channels: Int, count: Int,
  prefixLength: Int, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotary = Input()
  let scale1 = Input()
  let gate1 = Input()
  let scale2 = Input()
  let gate2 = Input()
  let cachedK = Input()
  let cachedV = Input()
  let heads = channels / 128
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let attentionInput = norm1(x).to(T.dataType) .* scale1
  let toKeys = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex, name: "x_k")
  let toValues = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex, name: "x_v")
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  let keys = Functional.cmul(
    left: normK(toKeys(attentionInput).reshaped([batchSize, count, heads, 128])),
    right: rotary
  )
  let values = toValues(attentionInput).reshaped([batchSize, count, heads, 128])
  let toQueries = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex, name: "x_q")
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  var queries = Functional.cmul(
    left: normQ(toQueries(attentionInput).reshaped([batchSize, count, heads, 128])),
    right: rotary
  )
  // Keep RoPE outputs contiguous when concatenating batched cached K/V, as in Flux2.
  let allKeys = Functional.concat(
    axis: 1, cachedK, keys, flags: batchSize > 1 ? [.disableOpt] : [])
  let allValues = Functional.concat(
    axis: 1, cachedV, values, flags: batchSize > 1 ? [.disableOpt] : [])
  let attended: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = allKeys.transposed(1, 2)
    queries = ((1 / Float(128).squareRoot()) * queries).transposed(1, 2)
    let transposedValues = allValues.transposed(1, 2)
    if batchSize * heads <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(batchSize * heads) {
        let key = transposedKeys.reshaped(
          [1, prefixLength + count, 128], offset: [i, 0, 0],
          strides: [(prefixLength + count) * 128, 128, 1])
        let query = queries.reshaped(
          [1, count, 128], offset: [i, 0, 0], strides: [count * 128, 128, 1])
        let value = transposedValues.reshaped(
          [1, prefixLength + count, 128], offset: [i, 0, 0],
          strides: [(prefixLength + count) * 128, 128, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([count, prefixLength + count])
        dot = dot.softmax()
        dot = dot.reshaped([1, count, prefixLength + count])
        outs.append(dot * value)
      }
      attended = Concat(axis: 0)(outs).reshaped([batchSize, heads, count, 128])
        .transposed(1, 2).reshaped([batchSize * count, channels])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, transposedKeys)
      dot = dot.reshaped([batchSize * heads * count, prefixLength + count]).softmax()
      dot = dot.reshaped([batchSize, heads, count, prefixLength + count])
      attended = (dot * transposedValues).transposed(1, 2)
        .reshaped([batchSize * count, channels])
    }
  case .scale1:
    queries = (1 / Float(128).squareRoot()) * queries
    let attention = ScaledDotProductAttention(
      scale: 1,
      isCausal: false,
      flags: [.Float16])
    attended = attention(queries, allKeys, allValues).reshaped([batchSize * count, channels])
  case .scaleMerged, .quantized:
    let attention = ScaledDotProductAttention(
      scale: 1 / Float(128).squareRoot(),
      isCausal: false,
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    attended = attention(queries, allKeys, allValues).reshaped([batchSize * count, channels])
  }
  let unifyHeads = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex, name: "x_o")
  var out = x + (unifyHeads(attended) .* gate1).to(of: x)
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let ffnInput = norm2(out).to(T.dataType) .* scale2
  let up = LoRADense(
    count: channels * 3, configuration: LoRAConfiguration, noBias: true, index: layerIndex,
    name: "ffn_up_proj")
  let gate = LoRADense(
    count: channels * 3, configuration: LoRAConfiguration, noBias: true, index: layerIndex,
    name: "ffn_gate_proj")
  let down = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: layerIndex,
    name: "ffn_down_proj")
  let product = Functional.swishMul(value: up(ffnInput), gate: gate(ffnInput))
  out = out + (down(product) .* gate2).to(of: out)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn.to_k.weight"] = [toKeys.weight.name]
    mapping["\(prefix).attn.to_v.weight"] = [toValues.weight.name]
    mapping["\(prefix).attn.to_q.weight"] = [toQueries.weight.name]
    mapping["\(prefix).attn.norm_k.weight"] = [normK.weight.name]
    mapping["\(prefix).attn.norm_q.weight"] = [normQ.weight.name]
    mapping["\(prefix).attn.to_out.0.weight"] = [unifyHeads.weight.name]
    mapping["\(prefix).img_mlp.proj.weight"] = [up.weight.name]
    mapping["\(prefix).img_mlp.gate_layer.weight"] = [gate.weight.name]
    mapping["\(prefix).img_mlp.out.weight"] = [down.weight.name]
    return mapping
  }
  return (mapper, Model([x, rotary, scale1, gate1, scale2, gate2, cachedK, cachedV], [out]))
}

// Outputs: five timestep tables in the model dtype, then post-RoPE K and V for each block.
public func LoRAQwenImage2_1Fixed<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, textLength: Int, referenceLength: Int, timesteps: Int,
  channels: Int, layers: Int, segments: [(length: Int, image: Bool)],
  usesFlashAttention: FlashAttentionLevel, LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let count = textLength + referenceLength
  precondition(segments.reduce(0) { $0 + $1.length } == count)
  let context = Input()
  let time = Input()
  let rotary = Input()
  let mask = Input()
  let reference = Input()
  let textNorm = RMSNorm(epsilon: 1e-6, axis: [1], name: "context_norm")
  let textIn = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, name: "context_embedder_0")
  let textOut = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, name: "context_embedder_1")
  let xEmbedder =
    referenceLength > 0
    ? LoRADense(count: channels, configuration: LoRAConfiguration, noBias: true, name: "x_embedder")
    : nil
  let time0 = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, name: "t_embedder_0")
  let time2 = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, name: "t_embedder_1")
  let temb = time2(time0(time).swish())
  let modulationInput = temb.swish()
  let modulation = (0..<4).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, noBias: true, name: "x_ada_ln_\($0)")
  }
  let finalModulation = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, name: "ada_ln_0")
  let tables = [
    1 + modulation[0](modulationInput),
    modulation[1](modulationInput).tanh(),
    1 + modulation[2](modulationInput),
    modulation[3](modulationInput).tanh(),
    1 + finalModulation(modulationInput),
  ]
  let outputs = tables.map {
    $0.reshaped([timesteps, channels], strides: [channels, 1]).contiguous()
  }
  let prefixModulation = tables.prefix(4).map {
    $0.reshaped([1, channels], offset: [timesteps, 0], strides: [channels, 1]).contiguous()
  }
  let encodedText = textOut(
    GELU(approximate: .tanh)(
      textIn(
        textNorm(context.reshaped([batchSize * textLength, 4096])))))
  let encodedReference = xEmbedder.map { $0(reference) }
  var prefixes = [Model.IO]()
  for batch in 0..<batchSize {
    var textOffset = batch * textLength
    var referenceOffset = 0
    for segment in segments where segment.length > 0 {
      if let encodedReference = encodedReference, segment.image {
        prefixes.append(
          encodedReference.reshaped(
            [segment.length, channels], offset: [referenceOffset, 0], strides: [channels, 1]
          ).contiguous())
        referenceOffset += segment.length
      } else {
        prefixes.append(
          encodedText.reshaped(
            [segment.length, channels], offset: [textOffset, 0], strides: [channels, 1]
          ).contiguous())
        textOffset += segment.length
      }
    }
  }
  var out = (prefixes.count == 1 ? prefixes[0] : Concat(axis: 0)(prefixes)).to(.Float32)
  var mappers = [ModelWeightMapper]()
  var cachedKVs = [Model.IO]()
  for i in 0..<layers {
    let (mapper, block) = LoRATransformerBlockFixed(
      dataType, prefix: "transformer_blocks.\(i)", layerIndex: i, batchSize: batchSize,
      channels: channels,
      count: count,
      segments: segments, outputKVOnly: i == layers - 1, usesFlashAttention: usesFlashAttention,
      LoRAConfiguration: LoRAConfiguration)
    mappers.append(mapper)
    if i == layers - 1 {
      let kv = block(out, rotary, prefixModulation[0])
      cachedKVs += [kv[0], kv[1]]
    } else {
      let result = block(
        [out, rotary] + prefixModulation + (usesFlashAttention != .none ? [] : [mask]))
      out = result[0]
      cachedKVs += [result[1], result[2]]
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers { mapping.merge(mapper(format)) { v, _ in v } }
    mapping["txt_in.text_norm.weight"] = [textNorm.weight.name]
    mapping["txt_in.in_layer.weight"] = [textIn.weight.name]
    mapping["txt_in.out_layer.weight"] = [textOut.weight.name]
    if let xEmbedder = xEmbedder { mapping["img_in.weight"] = [xEmbedder.weight.name] }
    mapping["time_text_embed.timestep_embedder.linear_1.weight"] = [time0.weight.name]
    mapping["time_text_embed.timestep_embedder.linear_2.weight"] = [time2.weight.name]
    mapping["modulation.1.weight"] = ModelWeightElement(modulation.map { $0.weight.name })
    mapping["norm_out.linear.weight"] = [finalModulation.weight.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [context, time, rotary] + (usesFlashAttention != .none ? [] : [mask])
        + (referenceLength > 0 ? [reference] : []),
      outputs + cachedKVs)
  )
}

// Inputs: target, target RoPE, five current modulation rows, 32 K/V pairs.
public func LoRAQwenImage2_1<T: TensorNumeric>(
  _ dataType: T.Type, batchSize: Int, height: Int, width: Int, prefixLength: Int,
  channels: Int, layers: Int, usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rotary = Input()
  let modulation = (0..<5).map { _ in Input() }
  let cachedKVs = (0..<(2 * layers)).map { _ in Input() }
  let count = height * width
  let xEmbedder = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, name: "x_embedder")
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [1], elementwiseAffine: false)
  let projOut = LoRADense(count: 64, configuration: LoRAConfiguration, noBias: true, name: "linear")
  var out = xEmbedder(x.reshaped([batchSize * count, 64])).to(.Float32)
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = LoRATransformerBlock(
      dataType, prefix: "transformer_blocks.\(i)", layerIndex: i, batchSize: batchSize,
      channels: channels,
      count: count,
      prefixLength: prefixLength, usesFlashAttention: usesFlashAttention,
      LoRAConfiguration: LoRAConfiguration)
    mappers.append(mapper)
    out = block(
      [out, rotary] + Array(modulation.prefix(4)) + [cachedKVs[2 * i], cachedKVs[2 * i + 1]])
  }
  out = projOut(normFinal(out).to(T.dataType) .* modulation[4])
    .reshaped([batchSize, height, width, 64])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers { mapping.merge(mapper(format)) { v, _ in v } }
    mapping["img_in.weight"] = [xEmbedder.weight.name]
    mapping["proj_out.weight"] = [projOut.weight.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rotary] + modulation + cachedKVs,
      [out])
  )
}

// A common resize for vision and VAE keeps four latent tokens per VLM image slot.
public func QwenImage2_1ReferenceSize(height: Int, width: Int) -> (height: Int, width: Int) {
  let scale = sqrt(Double(1024 * 1024) / Double(height * width))
  return (
    max(32, Int((Double(height) * scale / 32).rounded()) * 32),
    max(32, Int((Double(width) * scale / 32).rounded()) * 32)
  )
}
