import DiffusionMappings
import Foundation
import NNC

private func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, noBias: true, name: "\(name)_linear1")
  let linear2 = Dense(count: intermediateSize, noBias: true, name: "\(name)_linear2")
  var out = linear1(x).swish() .* linear2(x)
  let outProjection = Dense(
    count: hiddenSize, noBias: true, flags: .Float32, name: "\(name)_out_proj")
  out = outProjection(out)
  return (linear1, linear2, outProjection, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let contextNorm1 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context) + contextChunks[0]
  let contextToKeys = Dense(count: k * h, noBias: true, name: "c_k")
  let contextToQueries = Dense(count: k * h, noBias: true, name: "c_q")
  let contextToValues = Dense(count: k * h, noBias: true, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  // Now run attention.
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    keys = keys.reshaped([b, t + hw, h, k]).transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
      .transposed(1, 2)
    values = values.reshaped([b, t + hw, h, k]).transposed(1, 2)
    if b * h <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped([1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        let query = queries.reshaped(
          [1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        let value = values.reshaped(
          [1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t + hw, t + hw])
        dot = dot.softmax()
        dot = dot.reshaped([1, t + hw, t + hw])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, t + hw, k]).transposed(1, 2).reshaped([b, t + hw, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * (t + hw), t + hw])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, t + hw, t + hw])
      out = dot * values
      out = out.reshaped([b, h, t + hw, k]).transposed(1, 2).reshaped([b, t + hw, h * k])
    }
  case .scale1:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged, .quantized:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = queries.reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1])
    let unifyheads = Dense(count: k * h, noBias: true, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + contextChunks[2] .* contextOut
  }
  xOut = x + xChunks[2] .* xOut
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextLinear2: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextLinear2, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 8 / 3, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
    contextOut = context + contextChunks[5]
      .* contextFF(contextNorm2(contextOut) .* contextChunks[4] + contextChunks[3])
  } else {
    contextLinear1 = nil
    contextLinear2 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xLinear2, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 8 / 3, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  xOut = x + xChunks[5] .* xFF(xNorm2(xOut) .* xChunks[4] + xChunks[3])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping["\(prefix.1).attn.add_q_proj.weight"] = [contextToQueries.weight.name]
      mapping["\(prefix.1).attn.add_k_proj.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix.1).attn.add_v_proj.weight"] = [contextToValues.weight.name]
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.1).attn.to_add_out.weight"] = [contextUnifyheads.weight.name]
      }
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      if let contextLinear1 = contextLinear1, let contextLinear2 = contextLinear2,
        let contextOutProjection = contextOutProjection
      {
        mapping["\(prefix.1).ff_context.linear_1.weight"] = [contextLinear1.weight.name]
        mapping["\(prefix.1).ff_context.linear_2.weight"] = [contextLinear2.weight.name]
        mapping[
          "\(prefix.1).ff_context.out_projection.weight"
        ] = [contextOutProjection.weight.name]
      }
      mapping["\(prefix.1).ff.linear_1.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.1).ff.linear_2.weight"] = [xLinear2.weight.name]
      mapping["\(prefix.1).ff.out_projection.weight"] = [xOutProjection.weight.name]
    case .generativeModels:
      mapping["\(prefix.0).attn.w1q.weight"] = [contextToQueries.weight.name]
      mapping["\(prefix.0).attn.w1k.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix.0).attn.w1v.weight"] = [contextToValues.weight.name]
      mapping["\(prefix.0).attn.w2q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.0).attn.w2k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.0).attn.w2v.weight"] = [xToValues.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.0).attn.w1o.weight"] = [contextUnifyheads.weight.name]
      }
      mapping["\(prefix.0).attn.w2o.weight"] = [xUnifyheads.weight.name]
      if let contextLinear1 = contextLinear1, let contextLinear2 = contextLinear2,
        let contextOutProjection = contextOutProjection
      {
        mapping["\(prefix.0).mlpC.c_fc1.weight"] = [contextLinear1.weight.name]
        mapping["\(prefix.0).mlpC.c_fc2.weight"] = [contextLinear2.weight.name]
        mapping[
          "\(prefix.0).mlpC.c_proj.weight"
        ] = [contextOutProjection.weight.name]
      }
      mapping["\(prefix.0).mlpX.c_fc1.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.0).mlpX.c_fc2.weight"] = [xLinear2.weight.name]
      mapping["\(prefix.0).mlpX.c_proj.weight"] = [xOutProjection.weight.name]
    }
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([context, x] + contextChunks + xChunks, [contextOut, xOut]))
  } else {
    return (mapper, Model([context, x] + contextChunks + xChunks, [xOut]))
  }
}

private func SingleTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  let xOut = xChunks[1] .* xNorm1(x) + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = LayerNorm(epsilon: 1e-5, axis: [3], elementwiseAffine: false)
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  var keys = xK
  var values = xV
  var queries = xQ
  // Now run attention.
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    keys = keys.reshaped([b, t + hw, h, k]).transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
      .transposed(1, 2)
    values = values.reshaped([b, t + hw, h, k]).transposed(1, 2)
    if b * h <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped([1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        let query = queries.reshaped(
          [1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        let value = values.reshaped(
          [1, t + hw, k], offset: [i, 0, 0], strides: [(t + hw) * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t + hw, t + hw])
        dot = dot.softmax()
        dot = dot.reshaped([1, t + hw, t + hw])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, t + hw, k]).transposed(1, 2).reshaped([b, t + hw, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * (t + hw), t + hw])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, t + hw, t + hw])
      out = dot * values
      out = out.reshaped([b, h, t + hw, k]).transposed(1, 2).reshaped([b, t + hw, h * k])
    }
  case .scale1:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged, .quantized:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = queries.reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
    xIn = x.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  out = xUnifyheads(out)
  out = xIn + xChunks[2] .* out
  // Attentions are now. Now run MLP.
  let (xLinear1, xLinear2, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 8 / 3, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-5, axis: [2], elementwiseAffine: false)
  out = xIn + xChunks[5] .* xFF(xNorm2(out) .* xChunks[4] + xChunks[3])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.1).ff.linear_1.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.1).ff.linear_2.weight"] = [xLinear2.weight.name]
      mapping["\(prefix.1).ff.out_projection.weight"] = [xOutProjection.weight.name]
    case .generativeModels:
      mapping["\(prefix.0).attn.w1q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.0).attn.w1k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.0).attn.w1v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.0).attn.w1o.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.0).mlp.c_fc1.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.0).mlp.c_fc2.weight"] = [xLinear2.weight.name]
      mapping["\(prefix.0).mlp.c_proj.weight"] = [xOutProjection.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x] + xChunks, [out]))
}

public func AuraFlow<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: Int, tokenLength: Int, height: Int, width: Int, maxSequence: Int, channels: Int,
  layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let contextIn = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([batchSize, h * w, channels])
  let posEmbed = Parameter<FloatType>(
    .GPU(0), .NHWC(1, maxSequence, maxSequence, channels), name: "pos_embed")
  let spatialPosEmbed: Model.IO
  let maxDim = max(h, w)
  if maxDim > maxSequence {
    spatialPosEmbed = Upsample(
      .bilinear, widthScale: Float(maxDim) / Float(maxSequence),
      heightScale: Float(maxDim) / Float(maxSequence))(posEmbed)
      .reshaped(
        [1, h, w, channels], offset: [0, (maxDim - h) / 2, (maxDim - w) / 2, 0],
        strides: [maxDim * maxDim * channels, maxDim * channels, channels, 1]
      ).contiguous().reshaped([1, h * w, channels])
  } else {
    spatialPosEmbed = posEmbed.reshaped(
      [1, h, w, channels], offset: [0, (maxSequence - h) / 2, (maxSequence - w) / 2, 0],
      strides: [maxSequence * maxSequence * channels, maxSequence * channels, channels, 1]
    ).contiguous().reshaped([1, h * w, channels])
  }
  out = spatialPosEmbed + out
  var adaLNChunks = [Input]()
  var mappers = [ModelWeightMapper]()
  var context: Model.IO = contextIn
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = JointTransformerBlock(
      prefix: ("double_layers.\(i)", "joint_transformer_blocks.\(i)"), k: 256, h: channels / 256,
      b: batchSize,
      t: tokenLength + 8,
      hw: h * w, contextBlockPreOnly: false, usesFlashAttention: usesFlashAttention)
    let blockOut = block([context, out] + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  out = Functional.concat(axis: 1, context, out)
  for i in 0..<layers.1 {
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = SingleTransformerBlock(
      prefix: ("single_layers.\(i)", "single_transformer_blocks.\(i)"), k: 256, h: channels / 256,
      b: batchSize,
      t: tokenLength + 8,
      hw: h * w,
      contextBlockPreOnly: i == layers.1 - 1, usesFlashAttention: usesFlashAttention)
    out = block([out] + xChunks)
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  out = scale .* out + shift
  let projOut = Dense(count: 2 * 2 * 4, noBias: true, name: "linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 2, 2, 4]).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    batchSize, h * 2, w * 2, 4,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .diffusers:
      mapping["pos_embed.proj.weight"] = [xEmbedder.weight.name]
      mapping["pos_embed.proj.bias"] = [xEmbedder.bias.name]
      mapping["pos_embed.pos_embed"] = [posEmbed.weight.name]
      mapping["proj_out.weight"] = [projOut.weight.name]
    case .generativeModels:
      mapping["init_x_linear.weight"] = [xEmbedder.weight.name]
      mapping["init_x_linear.bias"] = [xEmbedder.bias.name]
      mapping["positional_encoding"] = [posEmbed.weight.name]
      mapping["final_linear.weight"] = [projOut.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x, contextIn] + adaLNChunks, [out]))
}

private func JointTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int, contextBlockPreOnly: Bool
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, noBias: true, name: "context_ada_ln_\($0)")
  }
  var contextChunks = contextAdaLNs.map { $0(c) }
  contextChunks[1] = 1 + contextChunks[1]
  let xAdaLNs = (0..<6).map { Dense(count: k * h, noBias: true, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  if !contextBlockPreOnly {
    contextChunks[4] = 1 + contextChunks[4]
  }
  xChunks[4] = 1 + xChunks[4]
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping[
        "\(prefix.1).norm1_context.linear.weight"
      ] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].weight.name
        })
      mapping["\(prefix.1).norm1.linear.weight"] = ModelWeightElement(
        (0..<6).map {
          xAdaLNs[$0].weight.name
        })
    case .generativeModels:
      mapping[
        "\(prefix.0).modC.1.weight"
      ] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].weight.name
        })
      mapping["\(prefix.0).modX.1.weight"] = ModelWeightElement(
        (0..<6).map {
          xAdaLNs[$0].weight.name
        })
    }
    return mapping
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

private func SingleTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let xAdaLNs = (0..<6).map { Dense(count: k * h, noBias: true, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  xChunks[4] = 1 + xChunks[4]
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .diffusers:
      mapping["\(prefix.1).norm1.linear.weight"] = ModelWeightElement(
        (0..<6).map {
          xAdaLNs[$0].weight.name
        })
    case .generativeModels:
      mapping["\(prefix.0).modCX.1.weight"] = ModelWeightElement(
        (0..<6).map {
          xAdaLNs[$0].weight.name
        })
    }
    return mapping
  }
  return (mapper, Model([c], xChunks))
}

public func AuraFlowFixed<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: (Int, Int), channels: Int, layers: (Int, Int),
  of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let timestep = Input()
  let contextIn = Input()
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: channels)
  let c = tEmbedder(timestep).reshaped([batchSize.1, 1, channels]).swish()
  let contextEmbedder = Dense(count: channels, noBias: true, name: "context_embedder")
  var outs = [Model.IO]()
  var context = contextEmbedder(contextIn)
  let registerTokens = Parameter<FloatType>(.GPU(0), .HWC(1, 8, channels), name: "register_tokens")
  context = Functional.concat(
    axis: 1, Concat(axis: 0)(Array(repeating: registerTokens, count: batchSize.0)), context)
  outs.append(context)
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (mapper, block) = JointTransformerBlockFixed(
      prefix: ("double_layers.\(i)", "joint_transformer_blocks.\(i)"), k: 256, h: channels / 256,
      contextBlockPreOnly: false
    )
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlockFixed(
      prefix: ("single_layers.\(i)", "single_transformer_blocks.\(i)"), k: 256, h: channels / 256)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let scale = Dense(count: channels, noBias: true, name: "ada_ln_0")
  let shift = Dense(count: channels, noBias: true, name: "ada_ln_1")
  outs.append(contentsOf: [shift(c), 1 + scale(c)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .diffusers:
      mapping["time_step_proj.linear_1.weight"] = [tMlp0.weight.name]
      mapping["time_step_proj.linear_1.bias"] = [tMlp0.bias.name]
      mapping["time_step_proj.linear_2.weight"] = [tMlp2.weight.name]
      mapping["time_step_proj.linear_2.bias"] = [tMlp2.bias.name]
      mapping["context_embedder.weight"] = [contextEmbedder.weight.name]
      mapping["register_tokens"] = [registerTokens.weight.name]
      mapping["norm_out.linear.weight"] = [scale.weight.name, shift.weight.name]
    case .generativeModels:
      mapping["t_embedder.mlp.0.weight"] = [tMlp0.weight.name]
      mapping["t_embedder.mlp.0.bias"] = [tMlp0.bias.name]
      mapping["t_embedder.mlp.2.weight"] = [tMlp2.weight.name]
      mapping["t_embedder.mlp.2.bias"] = [tMlp2.bias.name]
      mapping["cond_seq_linear.weight"] = [contextEmbedder.weight.name]
      mapping["register_tokens"] = [registerTokens.weight.name]
      mapping["modF.1.weight"] = [shift.weight.name, scale.weight.name]
    }
    return mapping
  }
  return (mapper, Model([contextIn, timestep], outs))
}
