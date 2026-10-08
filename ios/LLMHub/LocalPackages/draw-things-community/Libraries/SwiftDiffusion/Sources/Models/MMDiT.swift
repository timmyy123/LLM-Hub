import DiffusionMappings
import NNC

private func TimeEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "t_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "t_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func VectorEmbedder(channels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "y_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "y_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func MLP(hiddenSize: Int, intermediateSize: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = Dense(count: intermediateSize, flags: [.Float16], name: "\(name)_fc1")
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = Dense(count: hiddenSize, flags: [.Float32], name: "\(name)_fc2")
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, qkNorm: Bool, useDualAttention: Bool, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut =
    contextChunks[1] .* (upcast ? contextNorm1(context).to(.Float16) : contextNorm1(context))
    + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, flags: [.Float16], name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut)
  var contextQ = contextToQueries(contextOut)
  let contextV = contextToValues(contextOut)
  let normAddedK: Model?
  let normAddedQ: Model?
  if qkNorm {
    let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
    contextK = normK(contextK.reshaped([b, t, h, k]))
    let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
    contextQ = normQ(contextQ.reshaped([b, t, h, k]))
    normAddedK = normK
    normAddedQ = normQ
  } else {
    normAddedK = nil
    normAddedQ = nil
  }
  let xChunks = (0..<(useDualAttention ? 9 : 6)).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xNorm1X = (upcast ? xNorm1(x).to(.Float16) : xNorm1(x))
  var xOut = xChunks[1] .* xNorm1X + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, flags: [.Float16], name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  var xQ = xToQueries(xOut)
  let xV = xToValues(xOut)
  let xToKeys2: Model?
  let xToQueries2: Model?
  let xToValues2: Model?
  var xK2: Model.IO?
  var xQ2: Model.IO?
  var xV2: Model.IO?
  if useDualAttention {
    xToKeys2 = Dense(count: k * h, name: "x_k_2")
    xToQueries2 = Dense(count: k * h, flags: [.Float16], name: "x_q_2")
    xToValues2 = Dense(count: k * h, name: "x_v_2")
    let out = xChunks[7] .* xNorm1X + xChunks[6]
    xK2 = xToKeys2?(out)
    xQ2 = xToQueries2?(out)
    xV2 = xToValues2?(out)
  } else {
    xToKeys2 = nil
    xToQueries2 = nil
    xToValues2 = nil
    xK2 = nil
    xQ2 = nil
    xV2 = nil
  }
  let normK: Model?
  let normQ: Model?
  let normK2: Model?
  let normQ2: Model?
  if qkNorm {
    let lnK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
    xK = lnK(xK.reshaped([b, hw, h, k]))
    let lnQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
    xQ = lnQ(xQ.reshaped([b, hw, h, k]))
    if useDualAttention {
      let lnK2 = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k_2")
      xK2 = xK2.map { lnK2($0.reshaped([b, hw, h, k])) }
      let lnQ2 = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q_2")
      xQ2 = xQ2.map { lnQ2($0.reshaped([b, hw, h, k])) }
      normK2 = lnK2
      normQ2 = lnQ2
    } else {
      normK2 = nil
      normQ2 = nil
    }
    normK = lnK
    normQ = lnQ
  } else {
    normK = nil
    normQ = nil
    normK2 = nil
    normQ2 = nil
  }
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  // Now run attention.
  var out: Model.IO
  var out2: Model.IO? = nil
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
    if var xK2 = xK2, var xQ2 = xQ2, var xV2 = xV2 {
      xK2 = xK2.reshaped([b, hw, h, k]).transposed(1, 2)
      xQ2 = ((1.0 / Float(k).squareRoot()) * xQ2).reshaped([b, hw, h, k])
        .transposed(1, 2)
      xV2 = xV2.reshaped([b, hw, h, k]).transposed(1, 2)
      if b * h <= 256 {
        var outs = [Model.IO]()
        for i in 0..<(b * h) {
          let key = xK2.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
          let query = xQ2.reshaped(
            [1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
          let value = xV2.reshaped(
            [1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
          var dot = Matmul(transposeB: (1, 2))(query, key)
          if let last = outs.last {
            dot.add(dependencies: [last])
          }
          dot = dot.reshaped([hw, hw])
          dot = dot.softmax()
          dot = dot.reshaped([1, hw, hw])
          outs.append(dot * value)
        }
        let out = Concat(axis: 0)(outs)
        out2 = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      } else {
        var dot = Matmul(transposeB: (2, 3))(xQ2, xK2)
        dot = dot.reshaped([b * h * hw, hw])
        dot = dot.softmax()
        dot = dot.reshaped([b, h, hw, hw])
        let out = dot * xV2
        out2 = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      }
    }
  case .scale1:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
    if var xK2 = xK2, var xQ2 = xQ2, var xV2 = xV2 {
      xK2 = xK2.reshaped([b, hw, h, k])
      xQ2 = ((1.0 / Float(k).squareRoot()) * xQ2).reshaped([b, hw, h, k])
      xV2 = xV2.reshaped([b, hw, h, k])
      let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
      out2 = scaledDotProductAttention(xQ2, xK2, xV2).reshaped([b, hw, k * h])
    }
  case .scaleMerged, .quantized:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = queries.reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
    if var xK2 = xK2, var xQ2 = xQ2, var xV2 = xV2 {
      xK2 = xK2.reshaped([b, hw, h, k])
      xQ2 = xQ2.reshaped([b, hw, h, k])
      xV2 = xV2.reshaped([b, hw, h, k])
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(),
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      out2 = scaledDotProductAttention(xQ2, xK2, xV2).reshaped([b, hw, k * h])
    }
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1])
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    if upcast {
      contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
    } else {
      contextOut = context + contextChunks[2] .* contextOut
    }
  }
  let xUnifyheads2: Model?
  if let out = out2 {
    xUnifyheads2 = Dense(count: k * h, name: "x_o_2")
    out2 = xUnifyheads2?(out)
  } else {
    xUnifyheads2 = nil
  }
  if upcast {
    xOut = x + (xChunks[2] .* xOut).to(of: x)
  } else {
    xOut = x + xChunks[2] .* xOut
  }
  if let out = out2 {
    if upcast {
      xOut = xOut + (xChunks[8] .* out).to(of: x)
    } else {
      xOut = xOut + xChunks[8] .* out
    }
  }
  // Attentions are now. Now run MLP.
  let contextFc1: Model?
  let contextFc2: Model?
  if !contextBlockPreOnly {
    let contextMlp: Model
    (contextFc1, contextFc2, contextMlp) = MLP(
      hiddenSize: k * h, intermediateSize: k * h * 4, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    if upcast {
      contextOut = contextOut + contextChunks[5].to(of: contextOut)
        .* contextMlp(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
        .to(of: contextOut)
    } else {
      contextOut = contextOut + contextChunks[5]
        .* contextMlp(contextNorm2(contextOut) .* contextChunks[4] + contextChunks[3])
    }
  } else {
    contextFc1 = nil
    contextFc2 = nil
  }
  let (xFc1, xFc2, xMlp) = MLP(hiddenSize: k * h, intermediateSize: k * h * 4, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if upcast {
    xOut = xOut + xChunks[5].to(of: xOut)
      .* xMlp(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]).to(of: xOut)
  } else {
    xOut = xOut + xChunks[5] .* xMlp(xNorm2(xOut) .* xChunks[4] + xChunks[3])
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).context_block.attn.qkv.weight"] = [
        contextToQueries.weight.name, contextToKeys.weight.name, contextToValues.weight.name,
      ]
      mapping["\(prefix.0).context_block.attn.qkv.bias"] = [
        contextToQueries.bias.name, contextToKeys.bias.name, contextToValues.bias.name,
      ]
      mapping["\(prefix.0).x_block.attn.qkv.weight"] = [
        xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
      ]
      mapping["\(prefix.0).x_block.attn.qkv.bias"] = [
        xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name,
      ]
      if let xToQueries2 = xToQueries2, let xToKeys2 = xToKeys2, let xToValues2 = xToValues2 {
        mapping["\(prefix.0).x_block.attn2.qkv.weight"] = [
          xToQueries2.weight.name, xToKeys2.weight.name, xToValues2.weight.name,
        ]
        mapping["\(prefix.0).x_block.attn2.qkv.bias"] = [
          xToQueries2.bias.name, xToKeys2.bias.name, xToValues2.bias.name,
        ]
      }
      if let normAddedK = normAddedK, let normAddedQ = normAddedQ, let normK = normK,
        let normQ = normQ
      {
        mapping["\(prefix.0).context_block.attn.ln_k.weight"] = [normAddedK.weight.name]
        mapping["\(prefix.0).context_block.attn.ln_q.weight"] = [normAddedQ.weight.name]
        mapping["\(prefix.0).x_block.attn.ln_k.weight"] = [normK.weight.name]
        mapping["\(prefix.0).x_block.attn.ln_q.weight"] = [normQ.weight.name]
        if let normK2 = normK2, let normQ2 = normQ2 {
          mapping["\(prefix.0).x_block.attn2.ln_k.weight"] = [normK2.weight.name]
          mapping["\(prefix.0).x_block.attn2.ln_q.weight"] = [normQ2.weight.name]
        }
      }
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.0).context_block.attn.proj.weight"] = [contextUnifyheads.weight.name]
        mapping["\(prefix.0).context_block.attn.proj.bias"] = [contextUnifyheads.bias.name]
      }
      mapping["\(prefix.0).x_block.attn.proj.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.0).x_block.attn.proj.bias"] = [xUnifyheads.bias.name]
      if let xUnifyheads2 = xUnifyheads2 {
        mapping["\(prefix.0).x_block.attn2.proj.weight"] = [xUnifyheads2.weight.name]
        mapping["\(prefix.0).x_block.attn2.proj.bias"] = [xUnifyheads2.bias.name]
      }
      if let contextFc1 = contextFc1, let contextFc2 = contextFc2 {
        mapping["\(prefix.0).context_block.mlp.fc1.weight"] = [contextFc1.weight.name]
        mapping["\(prefix.0).context_block.mlp.fc1.bias"] = [contextFc1.bias.name]
        mapping["\(prefix.0).context_block.mlp.fc2.weight"] = [contextFc2.weight.name]
        mapping["\(prefix.0).context_block.mlp.fc2.bias"] = [contextFc2.bias.name]
      }
      mapping["\(prefix.0).x_block.mlp.fc1.weight"] = [xFc1.weight.name]
      mapping["\(prefix.0).x_block.mlp.fc1.bias"] = [xFc1.bias.name]
      mapping["\(prefix.0).x_block.mlp.fc2.weight"] = [xFc2.weight.name]
      mapping["\(prefix.0).x_block.mlp.fc2.bias"] = [xFc2.bias.name]
    case .diffusers:
      mapping["\(prefix.1).attn.add_q_proj.weight"] = [contextToQueries.weight.name]
      mapping["\(prefix.1).attn.add_q_proj.bias"] = [contextToQueries.bias.name]
      mapping["\(prefix.1).attn.add_k_proj.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix.1).attn.add_k_proj.bias"] = [contextToKeys.bias.name]
      mapping["\(prefix.1).attn.add_v_proj.weight"] = [contextToValues.weight.name]
      mapping["\(prefix.1).attn.add_v_proj.bias"] = [contextToValues.bias.name]
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.to_v.bias"] = [xToValues.bias.name]
      if let xToQueries2 = xToQueries2, let xToKeys2 = xToKeys2, let xToValues2 = xToValues2 {
        mapping["\(prefix.1).attn2.to_q.weight"] = [xToQueries2.weight.name]
        mapping["\(prefix.1).attn2.to_q.bias"] = [xToQueries2.bias.name]
        mapping["\(prefix.1).attn2.to_k.weight"] = [xToKeys2.weight.name]
        mapping["\(prefix.1).attn2.to_k.bias"] = [xToKeys2.bias.name]
        mapping["\(prefix.1).attn2.to_v.weight"] = [xToValues2.weight.name]
        mapping["\(prefix.1).attn2.to_v.bias"] = [xToValues2.bias.name]
      }
      if let normAddedK = normAddedK, let normAddedQ = normAddedQ, let normK = normK,
        let normQ = normQ
      {
        mapping["\(prefix.1).attn.norm_added_k.weight"] = [normAddedK.weight.name]
        mapping["\(prefix.1).attn.norm_added_q.weight"] = [normAddedQ.weight.name]
        mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
        mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
        if let normK2 = normK2, let normQ2 = normQ2 {
          mapping["\(prefix.1).attn2.norm_k.weight"] = [normK2.weight.name]
          mapping["\(prefix.1).attn2.norm_q.weight"] = [normQ2.weight.name]
        }
      }
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.1).attn.to_add_out.weight"] = [contextUnifyheads.weight.name]
        mapping["\(prefix.1).attn.to_add_out.bias"] = [contextUnifyheads.bias.name]
      }
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.1).attn.to_out.0.bias"] = [xUnifyheads.bias.name]
      if let xUnifyheads2 = xUnifyheads2 {
        mapping["\(prefix.1).attn2.to_out.0.weight"] = [xUnifyheads2.weight.name]
        mapping["\(prefix.1).attn2.to_out.0.bias"] = [xUnifyheads2.bias.name]
      }
      if let contextFc1 = contextFc1, let contextFc2 = contextFc2 {
        mapping["\(prefix.1).ff_context.net.0.proj.weight"] = [contextFc1.weight.name]
        mapping["\(prefix.1).ff_context.net.0.proj.bias"] = [contextFc1.bias.name]
        mapping["\(prefix.1).ff_context.net.2.weight"] = [contextFc2.weight.name]
        mapping["\(prefix.1).ff_context.net.2.bias"] = [contextFc2.bias.name]
      }
      mapping["\(prefix.1).ff.net.0.proj.weight"] = [xFc1.weight.name]
      mapping["\(prefix.1).ff.net.0.proj.bias"] = [xFc1.bias.name]
      mapping["\(prefix.1).ff.net.2.weight"] = [xFc2.weight.name]
      mapping["\(prefix.1).ff.net.2.bias"] = [xFc2.bias.name]
    }
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([context, x] + contextChunks + xChunks, [contextOut, xOut]))
  } else {
    return (mapper, Model([context, x] + contextChunks + xChunks, [xOut]))
  }
}

public func MMDiT<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: Int, t: Int, height: Int, width: Int, channels: Int, layers: Int,
  upcast: Bool, qkNorm: Bool, dualAttentionLayers: [Int], posEmbedMaxSize: Int,
  usesFlashAttention: FlashAttentionLevel,
  of: FloatType.Type = FloatType.self
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let contextIn = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([batchSize, h * w, channels])
  let posEmbed = Parameter<FloatType>(
    .GPU(0), .NHWC(1, posEmbedMaxSize, posEmbedMaxSize, channels), name: "pos_embed")
  let spatialPosEmbed = posEmbed.reshaped(
    [1, h, w, channels], offset: [0, (posEmbedMaxSize - h) / 2, (posEmbedMaxSize - w) / 2, 0],
    strides: [
      posEmbedMaxSize * posEmbedMaxSize * channels, posEmbedMaxSize * channels, channels, 1,
    ]
  ).contiguous().reshaped([1, h * w, channels])
  out = spatialPosEmbed + out
  var adaLNChunks = [Input]()
  var mappers = [ModelWeightMapper]()
  var context: Model.IO = contextIn
  if upcast {
    out = out.to(.Float32)
    context = context.to(.Float32)
  }
  for i in 0..<layers {
    let contextBlockPreOnly = (i == layers - 1)
    let useDualAttention = dualAttentionLayers.contains(i)
    let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in Input() }
    let xChunks = (0..<(useDualAttention ? 9 : 6)).map { _ in Input() }
    let (mapper, block) = JointTransformerBlock(
      prefix: ("diffusion_model.joint_blocks.\(i)", "transformer_blocks.\(i)"), k: 64,
      h: channels / 64, b: batchSize, t: t, hw: h * w, contextBlockPreOnly: contextBlockPreOnly,
      upcast: upcast, qkNorm: qkNorm, useDualAttention: useDualAttention,
      usesFlashAttention: usesFlashAttention)
    let blockOut = block([context, out] + contextChunks + xChunks)
    if i == layers - 1 {
      out = blockOut
    } else {
      context = blockOut[0]
      out = blockOut[1]
    }
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  out = scale .* (upcast ? normFinal(out).to(.Float16) : normFinal(out)) + shift
  let linear = Dense(count: 2 * 2 * 16, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    batchSize, h * 2, w * 2, 16,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["diffusion_model.x_embedder.proj.weight"] = [xEmbedder.weight.name]
      mapping["diffusion_model.x_embedder.proj.bias"] = [xEmbedder.bias.name]
      mapping["diffusion_model.pos_embed"] = [posEmbed.weight.name]
      mapping["diffusion_model.final_layer.linear.weight"] = [linear.weight.name]
      mapping["diffusion_model.final_layer.linear.bias"] = [linear.bias.name]
    case .diffusers:
      mapping["pos_embed.proj.weight"] = [xEmbedder.weight.name]
      mapping["pos_embed.proj.bias"] = [xEmbedder.bias.name]
      mapping["pos_embed.pos_embed"] = [posEmbed.weight.name]
      mapping["proj_out.weight"] = [linear.weight.name]
      mapping["proj_out.bias"] = [linear.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x, contextIn] + adaLNChunks, [out]))
}

private func LoRAMLP(
  hiddenSize: Int, intermediateSize: Int, configuration: LoRANetworkConfiguration, index: Int,
  name: String
) -> (Model, Model, Model) {
  let x = Input()
  let fc1 = LoRADense(
    count: intermediateSize, configuration: configuration, flags: [.Float16], index: index,
    name: "\(name)_fc1")
  var out = GELU(approximate: .tanh)(fc1(x))
  let fc2 = LoRADense(
    count: hiddenSize, configuration: configuration, flags: [.Float32], index: index,
    name: "\(name)_fc2")
  out = fc2(out)
  return (fc1, fc2, Model([x], [out]))
}

private func LoRAJointTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, qkNorm: Bool, useDualAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut =
    contextChunks[1] .* (upcast ? contextNorm1(context).to(.Float16) : contextNorm1(context))
    + contextChunks[0]
  let contextToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_k")
  let contextToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "c_q")
  let contextToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_v")
  var contextK = contextToKeys(contextOut)
  var contextQ = contextToQueries(contextOut)
  let contextV = contextToValues(contextOut)
  let normAddedK: Model?
  let normAddedQ: Model?
  if qkNorm {
    let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
    contextK = normK(contextK.reshaped([b, t, h, k]))
    let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
    contextQ = normQ(contextQ.reshaped([b, t, h, k]))
    normAddedK = normK
    normAddedQ = normQ
  } else {
    normAddedK = nil
    normAddedQ = nil
  }
  let xChunks = (0..<(useDualAttention ? 9 : 6)).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xNorm1X = (upcast ? xNorm1(x).to(.Float16) : xNorm1(x))
  var xOut = xChunks[1] .* xNorm1X + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_v")
  var xK = xToKeys(xOut)
  var xQ = xToQueries(xOut)
  let xV = xToValues(xOut)
  let xToKeys2: Model?
  let xToQueries2: Model?
  let xToValues2: Model?
  var xK2: Model.IO?
  var xQ2: Model.IO?
  var xV2: Model.IO?
  if useDualAttention {
    xToKeys2 = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex,
      name: "x_k_2")
    xToQueries2 = LoRADense(
      count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex,
      name: "x_q_2")
    xToValues2 = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex,
      name: "x_v_2")
    let out = xChunks[7] .* xNorm1X + xChunks[6]
    xK2 = xToKeys2?(out)
    xQ2 = xToQueries2?(out)
    xV2 = xToValues2?(out)
  } else {
    xToKeys2 = nil
    xToQueries2 = nil
    xToValues2 = nil
    xK2 = nil
    xQ2 = nil
    xV2 = nil
  }
  let normK: Model?
  let normQ: Model?
  let normK2: Model?
  let normQ2: Model?
  if qkNorm {
    let lnK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
    xK = lnK(xK.reshaped([b, hw, h, k]))
    let lnQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
    xQ = lnQ(xQ.reshaped([b, hw, h, k]))
    if useDualAttention {
      let lnK2 = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k_2")
      xK2 = xK2.map { lnK2($0.reshaped([b, hw, h, k])) }
      let lnQ2 = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q_2")
      xQ2 = xQ2.map { lnQ2($0.reshaped([b, hw, h, k])) }
      normK2 = lnK2
      normQ2 = lnQ2
    } else {
      normK2 = nil
      normQ2 = nil
    }
    normK = lnK
    normQ = lnQ
  } else {
    normK = nil
    normQ = nil
    normK2 = nil
    normQ2 = nil
  }
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  // Now run attention.
  var out: Model.IO
  var out2: Model.IO? = nil
  switch usesFlashAttention {
  case .none:
    keys = keys.reshaped([b, t + hw, h, k]).transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
      .transposed(1, 2)
    values = values.reshaped([b, t + hw, h, k]).transposed(1, 2)
    // During training, we don't optimize this.
    if b * h <= 256 && configuration.testing {
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
    if var xK2 = xK2, var xQ2 = xQ2, var xV2 = xV2 {
      xK2 = xK2.reshaped([b, hw, h, k]).transposed(1, 2)
      xQ2 = ((1.0 / Float(k).squareRoot()) * xQ2).reshaped([b, hw, h, k])
        .transposed(1, 2)
      xV2 = xV2.reshaped([b, hw, h, k]).transposed(1, 2)
      if b * h <= 256 {
        var outs = [Model.IO]()
        for i in 0..<(b * h) {
          let key = xK2.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
          let query = xQ2.reshaped(
            [1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
          let value = xV2.reshaped(
            [1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
          var dot = Matmul(transposeB: (1, 2))(query, key)
          if let last = outs.last {
            dot.add(dependencies: [last])
          }
          dot = dot.reshaped([hw, hw])
          dot = dot.softmax()
          dot = dot.reshaped([1, hw, hw])
          outs.append(dot * value)
        }
        let out = Concat(axis: 0)(outs)
        out2 = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      } else {
        var dot = Matmul(transposeB: (2, 3))(xQ2, xK2)
        dot = dot.reshaped([b * h * hw, hw])
        dot = dot.softmax()
        dot = dot.reshaped([b, h, hw, hw])
        let out = dot * xV2
        out2 = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
      }
    }
  case .scale1:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = ((1.0 / Float(k).squareRoot()) * queries).reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
    if var xK2 = xK2, var xQ2 = xQ2, var xV2 = xV2 {
      xK2 = xK2.reshaped([b, hw, h, k])
      xQ2 = ((1.0 / Float(k).squareRoot()) * xQ2).reshaped([b, hw, h, k])
      xV2 = xV2.reshaped([b, hw, h, k])
      let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
      out2 = scaledDotProductAttention(xQ2, xK2, xV2).reshaped([b, hw, k * h])
    }
  case .scaleMerged, .quantized:
    keys = keys.reshaped([b, t + hw, h, k])
    queries = queries.reshaped([b, t + hw, h, k])
    values = values.reshaped([b, t + hw, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
    if var xK2 = xK2, var xQ2 = xQ2, var xV2 = xV2 {
      xK2 = xK2.reshaped([b, hw, h, k])
      xQ2 = xQ2.reshaped([b, hw, h, k])
      xV2 = xV2.reshaped([b, hw, h, k])
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(),
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      out2 = scaledDotProductAttention(xQ2, xK2, xV2).reshaped([b, hw, k * h])
    }
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped([b, t, h * k], strides: [(t + hw) * h * k, h * k, 1])
    let unifyheads = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], offset: [0, t, 0], strides: [(t + hw) * h * k, h * k, 1])
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    if upcast {
      contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
    } else {
      contextOut = context + contextChunks[2] .* contextOut
    }
  }
  let xUnifyheads2: Model?
  if let out = out2 {
    xUnifyheads2 = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "x_o_2")
    out2 = xUnifyheads2?(out)
  } else {
    xUnifyheads2 = nil
  }
  if upcast {
    xOut = x + (xChunks[2] .* xOut).to(of: context)
  } else {
    xOut = x + xChunks[2] .* xOut
  }
  if let out = out2 {
    if upcast {
      xOut = xOut + (xChunks[8] .* out).to(of: x)
    } else {
      xOut = xOut + xChunks[8] .* out
    }
  }
  // Attentions are now. Now run MLP.
  let contextFc1: Model?
  let contextFc2: Model?
  if !contextBlockPreOnly {
    let contextMlp: Model
    (contextFc1, contextFc2, contextMlp) = LoRAMLP(
      hiddenSize: k * h, intermediateSize: k * h * 4, configuration: configuration,
      index: layerIndex, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    if upcast {
      contextOut = contextOut + contextChunks[5].to(of: contextOut)
        .* contextMlp(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
        .to(of: contextOut)
    } else {
      contextOut = contextOut + contextChunks[5]
        .* contextMlp(contextNorm2(contextOut) .* contextChunks[4] + contextChunks[3])
    }
  } else {
    contextFc1 = nil
    contextFc2 = nil
  }
  let (xFc1, xFc2, xMlp) = LoRAMLP(
    hiddenSize: k * h, intermediateSize: k * h * 4, configuration: configuration, index: layerIndex,
    name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if upcast {
    xOut = xOut + xChunks[5].to(of: xOut)
      .* xMlp(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]).to(of: xOut)
  } else {
    xOut = xOut + xChunks[5] .* xMlp(xNorm2(xOut) .* xChunks[4] + xChunks[3])
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).context_block.attn.qkv.weight"] = [
        contextToQueries.weight.name, contextToKeys.weight.name, contextToValues.weight.name,
      ]
      mapping["\(prefix.0).context_block.attn.qkv.bias"] = [
        contextToQueries.bias.name, contextToKeys.bias.name, contextToValues.bias.name,
      ]
      mapping["\(prefix.0).x_block.attn.qkv.weight"] = [
        xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
      ]
      mapping["\(prefix.0).x_block.attn.qkv.bias"] = [
        xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name,
      ]
      if let xToQueries2 = xToQueries2, let xToKeys2 = xToKeys2, let xToValues2 = xToValues2 {
        mapping["\(prefix.0).x_block.attn2.qkv.weight"] = [
          xToQueries2.weight.name, xToKeys2.weight.name, xToValues2.weight.name,
        ]
        mapping["\(prefix.0).x_block.attn2.qkv.bias"] = [
          xToQueries2.bias.name, xToKeys2.bias.name, xToValues2.bias.name,
        ]
      }
      if let normAddedK = normAddedK, let normAddedQ = normAddedQ, let normK = normK,
        let normQ = normQ
      {
        mapping["\(prefix.0).context_block.attn.ln_k.weight"] = [normAddedK.weight.name]
        mapping["\(prefix.0).context_block.attn.ln_q.weight"] = [normAddedQ.weight.name]
        mapping["\(prefix.0).x_block.attn.ln_k.weight"] = [normK.weight.name]
        mapping["\(prefix.0).x_block.attn.ln_q.weight"] = [normQ.weight.name]
        if let normK2 = normK2, let normQ2 = normQ2 {
          mapping["\(prefix.0).x_block.attn2.ln_k.weight"] = [normK2.weight.name]
          mapping["\(prefix.0).x_block.attn2.ln_q.weight"] = [normQ2.weight.name]
        }
      }
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.0).context_block.attn.proj.weight"] = [contextUnifyheads.weight.name]
        mapping["\(prefix.0).context_block.attn.proj.bias"] = [contextUnifyheads.bias.name]
      }
      mapping["\(prefix.0).x_block.attn.proj.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.0).x_block.attn.proj.bias"] = [xUnifyheads.bias.name]
      if let xUnifyheads2 = xUnifyheads2 {
        mapping["\(prefix.0).x_block.attn2.proj.weight"] = [xUnifyheads2.weight.name]
        mapping["\(prefix.0).x_block.attn2.proj.bias"] = [xUnifyheads2.bias.name]
      }
      if let contextFc1 = contextFc1, let contextFc2 = contextFc2 {
        mapping["\(prefix.0).context_block.mlp.fc1.weight"] = [contextFc1.weight.name]
        mapping["\(prefix.0).context_block.mlp.fc1.bias"] = [contextFc1.bias.name]
        mapping["\(prefix.0).context_block.mlp.fc2.weight"] = [contextFc2.weight.name]
        mapping["\(prefix.0).context_block.mlp.fc2.bias"] = [contextFc2.bias.name]
      }
      mapping["\(prefix.0).x_block.mlp.fc1.weight"] = [xFc1.weight.name]
      mapping["\(prefix.0).x_block.mlp.fc1.bias"] = [xFc1.bias.name]
      mapping["\(prefix.0).x_block.mlp.fc2.weight"] = [xFc2.weight.name]
      mapping["\(prefix.0).x_block.mlp.fc2.bias"] = [xFc2.bias.name]
    case .diffusers:
      mapping["\(prefix.1).attn.add_q_proj.weight"] = [contextToQueries.weight.name]
      mapping["\(prefix.1).attn.add_q_proj.bias"] = [contextToQueries.bias.name]
      mapping["\(prefix.1).attn.add_k_proj.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix.1).attn.add_k_proj.bias"] = [contextToKeys.bias.name]
      mapping["\(prefix.1).attn.add_v_proj.weight"] = [contextToValues.weight.name]
      mapping["\(prefix.1).attn.add_v_proj.bias"] = [contextToValues.bias.name]
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.to_v.bias"] = [xToValues.bias.name]
      if let xToQueries2 = xToQueries2, let xToKeys2 = xToKeys2, let xToValues2 = xToValues2 {
        mapping["\(prefix.1).attn2.to_q.weight"] = [xToQueries2.weight.name]
        mapping["\(prefix.1).attn2.to_q.bias"] = [xToQueries2.bias.name]
        mapping["\(prefix.1).attn2.to_k.weight"] = [xToKeys2.weight.name]
        mapping["\(prefix.1).attn2.to_k.bias"] = [xToKeys2.bias.name]
        mapping["\(prefix.1).attn2.to_v.weight"] = [xToValues2.weight.name]
        mapping["\(prefix.1).attn2.to_v.bias"] = [xToValues2.bias.name]
      }
      if let normAddedK = normAddedK, let normAddedQ = normAddedQ, let normK = normK,
        let normQ = normQ
      {
        mapping["\(prefix.1).attn.norm_added_k.weight"] = [normAddedK.weight.name]
        mapping["\(prefix.1).attn.norm_added_q.weight"] = [normAddedQ.weight.name]
        mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
        mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
        if let normK2 = normK2, let normQ2 = normQ2 {
          mapping["\(prefix.1).attn2.norm_k.weight"] = [normK2.weight.name]
          mapping["\(prefix.1).attn2.norm_q.weight"] = [normQ2.weight.name]
        }
      }
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.1).attn.to_add_out.weight"] = [contextUnifyheads.weight.name]
        mapping["\(prefix.1).attn.to_add_out.bias"] = [contextUnifyheads.bias.name]
      }
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.1).attn.to_out.0.bias"] = [xUnifyheads.bias.name]
      if let xUnifyheads2 = xUnifyheads2 {
        mapping["\(prefix.1).attn2.to_out.0.weight"] = [xUnifyheads2.weight.name]
        mapping["\(prefix.1).attn2.to_out.0.bias"] = [xUnifyheads2.bias.name]
      }
      if let contextFc1 = contextFc1, let contextFc2 = contextFc2 {
        mapping["\(prefix.1).ff_context.net.0.proj.weight"] = [contextFc1.weight.name]
        mapping["\(prefix.1).ff_context.net.0.proj.bias"] = [contextFc1.bias.name]
        mapping["\(prefix.1).ff_context.net.2.weight"] = [contextFc2.weight.name]
        mapping["\(prefix.1).ff_context.net.2.bias"] = [contextFc2.bias.name]
      }
      mapping["\(prefix.1).ff.net.0.proj.weight"] = [xFc1.weight.name]
      mapping["\(prefix.1).ff.net.0.proj.bias"] = [xFc1.bias.name]
      mapping["\(prefix.1).ff.net.2.weight"] = [xFc2.weight.name]
      mapping["\(prefix.1).ff.net.2.bias"] = [xFc2.bias.name]
    }
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([context, x] + contextChunks + xChunks, [contextOut, xOut]))
  } else {
    return (mapper, Model([context, x] + contextChunks + xChunks, [xOut]))
  }
}

public func LoRAMMDiT<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: Int, t: Int, height: Int, width: Int, channels: Int, layers: Int,
  upcast: Bool, qkNorm: Bool, dualAttentionLayers: [Int], posEmbedMaxSize: Int,
  usesFlashAttention: FlashAttentionLevel,
  LoRAConfiguration: LoRANetworkConfiguration, of: FloatType.Type = FloatType.self
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let contextIn = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([batchSize, h * w, channels])
  let posEmbed = Parameter<FloatType>(
    .GPU(0), .NHWC(1, posEmbedMaxSize, posEmbedMaxSize, channels), name: "pos_embed")
  let spatialPosEmbed = posEmbed.reshaped(
    [1, h, w, channels], offset: [0, (posEmbedMaxSize - h) / 2, (posEmbedMaxSize - w) / 2, 0],
    strides: [
      posEmbedMaxSize * posEmbedMaxSize * channels, posEmbedMaxSize * channels, channels, 1,
    ]
  ).contiguous().reshaped([1, h * w, channels])
  out = spatialPosEmbed + out
  var adaLNChunks = [Input]()
  var mappers = [ModelWeightMapper]()
  var context: Model.IO = contextIn
  if upcast {
    out = out.to(.Float32)
    context = context.to(.Float32)
  }
  for i in 0..<layers {
    let contextBlockPreOnly = (i == layers - 1)
    let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in Input() }
    let useDualAttention = dualAttentionLayers.contains(i)
    let xChunks = (0..<(useDualAttention ? 9 : 6)).map { _ in Input() }
    let (mapper, block) = LoRAJointTransformerBlock(
      prefix: ("diffusion_model.joint_blocks.\(i)", "transformer_blocks.\(i)"), k: 64,
      h: channels / 64, b: batchSize, t: t, hw: h * w, contextBlockPreOnly: contextBlockPreOnly,
      upcast: upcast, qkNorm: qkNorm, useDualAttention: useDualAttention,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    let blockOut = block([context, out] + contextChunks + xChunks)
    if contextBlockPreOnly {
      out = blockOut
    } else {
      context = blockOut[0]
      out = blockOut[1]
    }
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  out = scale .* (upcast ? normFinal(out).to(.Float16) : normFinal(out)) + shift
  let linear = LoRADense(count: 2 * 2 * 16, configuration: LoRAConfiguration, name: "linear")
  out = linear(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous().reshaped([
    batchSize, h * 2, w * 2, 16,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["diffusion_model.x_embedder.proj.weight"] = [xEmbedder.weight.name]
      mapping["diffusion_model.x_embedder.proj.bias"] = [xEmbedder.bias.name]
      mapping["diffusion_model.pos_embed"] = [posEmbed.weight.name]
      mapping["diffusion_model.final_layer.linear.weight"] = [linear.weight.name]
      mapping["diffusion_model.final_layer.linear.bias"] = [linear.bias.name]
    case .diffusers:
      mapping["pos_embed.proj.weight"] = [xEmbedder.weight.name]
      mapping["pos_embed.proj.bias"] = [xEmbedder.bias.name]
      mapping["pos_embed.pos_embed"] = [posEmbed.weight.name]
      mapping["proj_out.weight"] = [linear.weight.name]
      mapping["proj_out.bias"] = [linear.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x, contextIn] + adaLNChunks, [out]))
}

private func JointTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int, b: Int, contextBlockPreOnly: Bool,
  useDualAttention: Bool
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  var contextChunks = contextAdaLNs.map { $0(c) }
  contextChunks[1] = 1 + contextChunks[1]
  let xAdaLNs = (0..<(useDualAttention ? 9 : 6)).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  if !contextBlockPreOnly {
    contextChunks[4] = 1 + contextChunks[4]
  }
  xChunks[4] = 1 + xChunks[4]
  if useDualAttention {
    xChunks[7] = 1 + xChunks[7]
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping[
        "\(prefix.0).context_block.adaLN_modulation.1.weight"
      ] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].weight.name })
      mapping[
        "\(prefix.0).context_block.adaLN_modulation.1.bias"
      ] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].bias.name })
      mapping["\(prefix.0).x_block.adaLN_modulation.1.weight"] = ModelWeightElement(
        (0..<(useDualAttention ? 9 : 6)).map {
          xAdaLNs[$0].weight.name
        })
      mapping["\(prefix.0).x_block.adaLN_modulation.1.bias"] = ModelWeightElement(
        (0..<(useDualAttention ? 9 : 6)).map { xAdaLNs[$0].bias.name })
    case .diffusers:
      if contextBlockPreOnly {
        mapping["\(prefix.1).norm1_context.linear.weight"] = [
          contextAdaLNs[1].weight.name, contextAdaLNs[0].weight.name,
        ]
        mapping["\(prefix.1).norm1_context.linear.bias"] = [
          contextAdaLNs[1].bias.name, contextAdaLNs[0].bias.name,
        ]
      } else {
        mapping[
          "\(prefix.1).norm1_context.linear.weight"
        ] = ModelWeightElement((0..<6).map { contextAdaLNs[$0].weight.name })
        mapping[
          "\(prefix.1).norm1_context.linear.bias"
        ] = ModelWeightElement((0..<6).map { contextAdaLNs[$0].bias.name })
      }
      mapping["\(prefix.1).norm1.linear.weight"] = ModelWeightElement(
        (0..<(useDualAttention ? 9 : 6)).map { xAdaLNs[$0].weight.name })
      mapping["\(prefix.1).norm1.linear.bias"] = ModelWeightElement(
        (0..<(useDualAttention ? 9 : 6)).map { xAdaLNs[$0].bias.name })
    }
    return mapping
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

public func MMDiTFixed(batchSize: Int, channels: Int, layers: Int, dualAttentionLayers: [Int]) -> (
  ModelWeightMapper, Model
) {
  let timestep = Input()
  let y = Input()
  let contextIn = Input()
  let (tMlp0, tMlp2, tEmbedder) = TimeEmbedder(channels: channels)
  let (yMlp0, yMlp2, yEmbedder) = VectorEmbedder(channels: channels)
  let c = (tEmbedder(timestep) + yEmbedder(y)).reshaped([batchSize, 1, channels]).swish()
  let contextEmbedder = Dense(count: channels, name: "context_embedder")
  var outs = [Model.IO]()
  let context = contextEmbedder(contextIn)
  outs.append(context)
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers {
    let (mapper, block) = JointTransformerBlockFixed(
      prefix: ("diffusion_model.joint_blocks.\(i)", "transformer_blocks.\(i)"), k: 64,
      h: channels / 64, b: batchSize, contextBlockPreOnly: i == layers - 1,
      useDualAttention: dualAttentionLayers.contains(i))
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let shift = Dense(count: channels, name: "ada_ln_0")
  let scale = Dense(count: channels, name: "ada_ln_1")
  outs.append(contentsOf: [shift(c), 1 + scale(c)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["diffusion_model.t_embedder.mlp.0.weight"] = [tMlp0.weight.name]
      mapping["diffusion_model.t_embedder.mlp.0.bias"] = [tMlp0.bias.name]
      mapping["diffusion_model.t_embedder.mlp.2.weight"] = [tMlp2.weight.name]
      mapping["diffusion_model.t_embedder.mlp.2.bias"] = [tMlp2.bias.name]
      mapping["diffusion_model.y_embedder.mlp.0.weight"] = [yMlp0.weight.name]
      mapping["diffusion_model.y_embedder.mlp.0.bias"] = [yMlp0.bias.name]
      mapping["diffusion_model.y_embedder.mlp.2.weight"] = [yMlp2.weight.name]
      mapping["diffusion_model.y_embedder.mlp.2.bias"] = [yMlp2.bias.name]
      mapping["diffusion_model.context_embedder.weight"] = [contextEmbedder.weight.name]
      mapping["diffusion_model.context_embedder.bias"] = [contextEmbedder.bias.name]
      mapping[
        "diffusion_model.final_layer.adaLN_modulation.1.weight"
      ] = [shift.weight.name, scale.weight.name]
      mapping[
        "diffusion_model.final_layer.adaLN_modulation.1.bias"
      ] = [shift.bias.name, scale.bias.name]
    case .diffusers:
      mapping["time_text_embed.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_1.bias"] = [tMlp0.bias.name]
      mapping["time_text_embed.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_2.bias"] = [tMlp2.bias.name]
      mapping["time_text_embed.text_embedder.linear_1.weight"] = [yMlp0.weight.name]
      mapping["time_text_embed.text_embedder.linear_1.bias"] = [yMlp0.bias.name]
      mapping["time_text_embed.text_embedder.linear_2.weight"] = [yMlp2.weight.name]
      mapping["time_text_embed.text_embedder.linear_2.bias"] = [yMlp2.bias.name]
      mapping["context_embedder.weight"] = [contextEmbedder.weight.name]
      mapping["context_embedder.bias"] = [contextEmbedder.bias.name]
      mapping[
        "norm_out.linear.weight"
      ] = [scale.weight.name, shift.weight.name]
      mapping[
        "norm_out.linear.bias"
      ] = [scale.bias.name, shift.bias.name]
    }
    return mapping
  }
  return (mapper, Model([contextIn, timestep, y], outs))
}
