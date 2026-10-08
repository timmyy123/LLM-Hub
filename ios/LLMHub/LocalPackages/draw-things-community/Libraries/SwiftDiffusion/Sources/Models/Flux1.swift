import DiffusionMappings
import Foundation
import NNC

public func Flux1RotaryPositionEmbedding(
  height: Int, width: Int, tokenLength: Int, referenceSizes: [(height: Int, width: Int)],
  channels: Int, heads: Int = 1
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(
    .CPU,
    .NHWC(
      1, height * width + tokenLength + referenceSizes.reduce(0) { $0 + $1.height * $1.width },
      heads, channels))
  let dim0 = channels / 8
  let dim1 = channels * 7 / 16
  let dim2 = dim1
  assert(channels % 16 == 0)
  for i in 0..<tokenLength {
    for j in 0..<heads {
      for k in 0..<(dim0 / 2) {
        let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, j, k * 2] = Float(costheta)
        rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
        rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim2 / 2) {
        let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
        rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
      }
    }
  }
  for y in 0..<height {
    for x in 0..<width {
      let i = y * width + x + tokenLength
      for j in 0..<heads {
        for k in 0..<(dim0 / 2) {
          let theta = 0 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, k * 2] = Float(costheta)
          rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim1 / 2) {
          let theta = Double(y) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim2 / 2) {
          let theta = Double(x) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  var index = width * height + tokenLength
  var h = 0
  var w = 0
  for referenceSize in referenceSizes {
    let height = referenceSize.height
    let width = referenceSize.width
    var hOffset = 0
    var wOffset = 0
    if height + h > width + w {
      wOffset = w
    } else {
      hOffset = h
    }
    for y in 0..<height {
      for x in 0..<width {
        let i = y * width + x + index
        for j in 0..<heads {
          for k in 0..<(dim0 / 2) {
            let theta = 1 * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))  // Use time index at 1.
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, k * 2] = Float(costheta)
            rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim1 / 2) {
            let theta = Double(y + hOffset) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim1))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim2 / 2) {
            let theta = Double(x + wOffset) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim2))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
          }
        }
      }
    }
    index += height * width
    h = max(h, height + hOffset)
    w = max(w, width + wOffset)
  }
  return rotTensor
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, flags: [.Float16], name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, flags: [.Float32], name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextNorm1(context).to(.Float16) .* contextChunks[1] + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, flags: [.Float16], name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, flags: [.Float16], name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    keys = keys.transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries)
      .transposed(1, 2)
    values = values.transposed(1, 2)
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
      dot = dot.reshaped([b, h, (t + hw), t + hw])
      out = dot * values
      out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
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
    contextOut = context + (contextOut .* contextChunks[2]).to(of: context)
  }
  xOut = x + (xOut .* xChunks[2]).to(of: x)
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    if upcast {
      contextOut =
        contextOut
        + contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
        .* contextChunks[5].to(of: contextOut)
    } else {
      contextOut =
        contextOut
        + (contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
        .* contextChunks[5])
        .to(of: contextOut)
    }
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if upcast {
    xOut = xOut + xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])
      .* xChunks[5].to(of: xOut)
  } else {
    xOut =
      xOut + (xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]) .* xChunks[5]).to(of: xOut)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping: [String: ModelWeightElement] = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).txt_attn.qkv.weight"] = [
        contextToQueries.weight.name, contextToKeys.weight.name, contextToValues.weight.name,
      ]
      mapping["\(prefix.0).txt_attn.qkv.bias"] = [
        contextToQueries.bias.name, contextToKeys.bias.name, contextToValues.bias.name,
      ]
      mapping["\(prefix.0).txt_attn.norm.key_norm.scale"] = [normAddedK.weight.name]
      mapping["\(prefix.0).txt_attn.norm.query_norm.scale"] = [normAddedQ.weight.name]
      mapping["\(prefix.0).img_attn.qkv.weight"] = [
        xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
      ]
      mapping["\(prefix.0).img_attn.qkv.bias"] = [
        xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name,
      ]
      mapping["\(prefix.0).img_attn.norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).img_attn.norm.query_norm.scale"] = [normQ.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.0).txt_attn.proj.weight"] = [contextUnifyheads.weight.name]
        mapping["\(prefix.0).txt_attn.proj.bias"] = [contextUnifyheads.bias.name]
      }
      mapping["\(prefix.0).img_attn.proj.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.0).img_attn.proj.bias"] = [xUnifyheads.bias.name]
      if let contextLinear1 = contextLinear1,
        let contextOutProjection = contextOutProjection
      {
        mapping["\(prefix.0).txt_mlp.0.weight"] = [contextLinear1.weight.name]
        mapping["\(prefix.0).txt_mlp.0.bias"] = [contextLinear1.bias.name]
        mapping["\(prefix.0).txt_mlp.2.weight"] = [contextOutProjection.weight.name]
        mapping["\(prefix.0).txt_mlp.2.bias"] = [contextOutProjection.bias.name]
      }
      mapping["\(prefix.0).img_mlp.0.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.0).img_mlp.0.bias"] = [xLinear1.bias.name]
      mapping["\(prefix.0).img_mlp.2.weight"] = [xOutProjection.weight.name]
      mapping["\(prefix.0).img_mlp.2.bias"] = [xOutProjection.bias.name]
    case .diffusers:
      mapping["\(prefix.1).attn.add_q_proj.weight"] = [contextToQueries.weight.name]
      mapping["\(prefix.1).attn.add_q_proj.bias"] = [contextToQueries.bias.name]
      mapping["\(prefix.1).attn.add_k_proj.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix.1).attn.add_k_proj.bias"] = [contextToKeys.bias.name]
      mapping["\(prefix.1).attn.add_v_proj.weight"] = [contextToValues.weight.name]
      mapping["\(prefix.1).attn.add_v_proj.bias"] = [contextToValues.bias.name]
      mapping["\(prefix.1).attn.norm_added_k.weight"] = [normAddedK.weight.name]
      mapping["\(prefix.1).attn.norm_added_q.weight"] = [normAddedQ.weight.name]
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.to_v.bias"] = [xToValues.bias.name]
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.1).attn.to_add_out.weight"] = [contextUnifyheads.weight.name]
        mapping["\(prefix.1).attn.to_add_out.bias"] = [contextUnifyheads.bias.name]
      }
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.1).attn.to_out.0.bias"] = [xUnifyheads.bias.name]
      if let contextLinear1 = contextLinear1,
        let contextOutProjection = contextOutProjection
      {
        mapping["\(prefix.1).ff_context.net.0.proj.weight"] = [contextLinear1.weight.name]
        mapping["\(prefix.1).ff_context.net.0.proj.bias"] = [contextLinear1.bias.name]
        mapping["\(prefix.1).ff_context.net.2.weight"] = [contextOutProjection.weight.name]
        mapping["\(prefix.1).ff_context.net.2.bias"] = [contextOutProjection.bias.name]
      }
      mapping["\(prefix.1).ff.net.0.proj.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.1).ff.net.0.proj.bias"] = [xLinear1.bias.name]
      mapping["\(prefix.1).ff.net.2.weight"] = [xOutProjection.weight.name]
      mapping["\(prefix.1).ff.net.2.bias"] = [xOutProjection.bias.name]
    }
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([context, x, rot] + contextChunks + xChunks, [contextOut, xOut]))
  } else {
    return (mapper, Model([context, x, rot] + contextChunks + xChunks, [xOut]))
  }
}

private func SingleTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, referenceSequenceLength: Int,
  contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, flags: [.Float16], name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  var keys = xK
  var values = xV
  var queries = xQ
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    keys = keys.transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries)
      .transposed(1, 2)
    values = values.transposed(1, 2)
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
      dot = dot.reshaped([b, h, (t + hw), t + hw])
      out = dot * values
      out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped(
      [b, hw - referenceSequenceLength, h * k], offset: [0, t, 0],
      strides: [(t + hw) * h * k, h * k, 1]
    )
    xIn = x.reshaped(
      [b, hw - referenceSequenceLength, h * k], offset: [0, t, 0],
      strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    xOut = xOut.reshaped(
      [b, hw - referenceSequenceLength, h * k], offset: [0, t, 0],
      strides: [(t + hw) * h * k, h * k, 1]
    )
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: false, name: "x")
  out = xUnifyheads(out) + xFF(xOut)
  out = xIn + (out .* xChunks[2]).to(of: xIn)
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).linear1.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xLinear1.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3])
      mapping["\(prefix.0).linear1.bias"] = ModelWeightElement(
        [
          xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name, xLinear1.bias.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3])
      mapping["\(prefix.0).norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).norm.query_norm.scale"] = [normQ.weight.name]
      mapping["\(prefix.0).linear2.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xOutProjection.weight.name], format: .I, offsets: [0, k * h])
      mapping["\(prefix.0).linear2.bias"] = [xOutProjection.bias.name]
    case .diffusers:
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.to_v.bias"] = [xToValues.bias.name]
      mapping["\(prefix.1).proj_mlp.weight"] = [xLinear1.weight.name]
      mapping["\(prefix.1).proj_mlp.bias"] = [xLinear1.bias.name]
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).proj_out.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xOutProjection.weight.name], format: .I, offsets: [0, k * h])
      mapping["\(prefix.1).proj_out.bias"] = [xOutProjection.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [out]))
}

public func Flux1Norm1(
  batchSize: Int, height: Int, width: Int, channels: Int
) -> Model {
  let x = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
  let xChunks = (0..<2).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = xChunks[1] .* xNorm1(out).to(.Float16) + xChunks[0]
  return Model([x] + xChunks, [out])
}

public func Flux1(
  batchSize: Int, tokenLength: Int, referenceSequenceLength: Int, height: Int, width: Int,
  channels: Int, layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, contextPreloaded: Bool, injectControls: Bool,
  injectIPAdapterLengths: [Int: [Int]], outputResidual: Bool, inputResidual: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out: Model.IO
  let referenceLatents: Input?
  let imgInX: Model.IO?
  if referenceSequenceLength > 0 && (layers.0 > 0 || layers.1 > 0) {
    let latents = Input()
    let imgIn = xEmbedder(x).reshaped([batchSize, h * w, channels])
    out = Functional.concat(axis: 1, imgIn, latents, flags: [.disableOpt]).to(.Float32)
    referenceLatents = latents
    if outputResidual {
      imgInX = imgIn.to(.Float32)
    } else {
      imgInX = nil
    }
  } else {
    out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
    referenceLatents = nil
    imgInX = out
  }
  var adaLNChunks = [Input]()
  var injectedControls = [Input]()
  var injectedIPAdapters = [Input]()
  let residualIn: Input?
  if inputResidual {
    let residual = Input()
    residualIn = residual
    out = out + residual
  } else {
    residualIn = nil
  }
  var mappers = [ModelWeightMapper]()
  let contextEmbedder: Model?
  var context: Model.IO?
  let rotAndContextIn: [Input]
  if layers.0 > 0 || layers.1 > 0 {
    let contextIn = Input()
    let rot = Input()
    if !contextPreloaded {
      let embedder = Dense(count: channels, name: "context_embedder")
      context = embedder(contextIn).to(.Float32)
      contextEmbedder = embedder
    } else {
      context = contextIn.to(.Float32)
      contextEmbedder = nil
    }
    rotAndContextIn = [rot] + (referenceLatents.map { [$0] } ?? []) + [contextIn]
  } else {
    context = nil
    contextEmbedder = nil
    rotAndContextIn = []
  }
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = JointTransformerBlock(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize, t: tokenLength,
      hw: h * w + referenceSequenceLength, contextBlockPreOnly: false, upcast: i > (layers.0 - 3),
      usesFlashAttention: usesFlashAttention
    )
    let blockOut = block([context!, out, rotAndContextIn[0]] + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    if injectControls {
      let injectedControl = Input()
      let injectedControlFP32 = injectedControl.to(.Float32)
      injectedControlFP32.add(dependencies: [blockOut])  // out has no associated nodes, use blockOut instead.
      let scaleFactor: Float = 8
      out = out + (injectedControlFP32 * scaleFactor)
      injectedControls.append(injectedControl)
    }
    if let injectIPAdapterLengths = injectIPAdapterLengths[i] {
      let image = out
      for (j, injectIPAdapterLength) in injectIPAdapterLengths.enumerated() {
        let ipKeys = Input()
        let ipValues = Input()
        let block = PuLIDCrossAttentionKeysAndValues(
          prefix: "", name: "\(j).double_\(i)", outputDim: channels, k: 2048 / 16, h: 16,
          b: batchSize, t: (injectIPAdapterLength, h * w + referenceSequenceLength),
          usesFlashAttention: usesFlashAttention == .scaleMerged ? .scale1 : usesFlashAttention)
        out = out + block(image, ipKeys, ipValues).to(of: out)
        injectedIPAdapters.append(contentsOf: [ipKeys, ipValues])
      }
    }
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  if let context = context {
    out = Functional.concat(axis: 1, context, out)
  }
  for i in 0..<layers.1 {
    let xChunks = (0..<3).map { _ in Input() }
    let (mapper, block) = SingleTransformerBlock(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize, t: tokenLength,
      hw: h * w + referenceSequenceLength, referenceSequenceLength: referenceSequenceLength,
      contextBlockPreOnly: i == layers.1 - 1, usesFlashAttention: usesFlashAttention)
    out = block([out, rotAndContextIn[0]] + xChunks)
    if injectControls {
      let injectedControl = Input()
      let injectedControlFP32 = injectedControl.to(.Float32)
      injectedControlFP32.add(dependencies: [out])
      let scaleFactor: Float = 8
      if i == layers.1 - 1 {
        out = out + (injectedControlFP32 * scaleFactor)
      } else {
        let encoderHiddenStates = out.reshaped(
          [batchSize, tokenLength, channels], offset: [0, 0, 0],
          strides: [(tokenLength + h * w + referenceSequenceLength) * channels, channels, 1])
        var hiddenStates = out.reshaped(
          [batchSize, h * w + referenceSequenceLength, channels], offset: [0, tokenLength, 0],
          strides: [(tokenLength + h * w + referenceSequenceLength) * channels, channels, 1])
        hiddenStates = hiddenStates + (injectedControlFP32 * scaleFactor)
        out = Functional.concat(axis: 1, encoderHiddenStates, hiddenStates)
      }
      injectedControls.append(injectedControl)
    }
    if let injectIPAdapterLengths = injectIPAdapterLengths[i + layers.0] {
      if i == layers.1 - 1 {
        let image = out
        for (j, injectIPAdapterLength) in injectIPAdapterLengths.enumerated() {
          let ipKeys = Input()
          let ipValues = Input()
          let block = PuLIDCrossAttentionKeysAndValues(
            prefix: "", name: "\(j).single_\(i)", outputDim: channels, k: 2048 / 16, h: 16,
            b: batchSize, t: (injectIPAdapterLength, h * w),
            usesFlashAttention: usesFlashAttention == .scaleMerged ? .scale1 : usesFlashAttention)
          out = out + block(image, ipKeys, ipValues).to(of: out)
          injectedIPAdapters.append(contentsOf: [ipKeys, ipValues])
        }
      } else {
        let encoderHiddenStates = out.reshaped(
          [batchSize, tokenLength, channels], offset: [0, 0, 0],
          strides: [(tokenLength + h * w + referenceSequenceLength) * channels, channels, 1])
        var hiddenStates = out.reshaped(
          [batchSize, h * w + referenceSequenceLength, channels], offset: [0, tokenLength, 0],
          strides: [(tokenLength + h * w + referenceSequenceLength) * channels, channels, 1])
        let image = hiddenStates
        for (j, injectIPAdapterLength) in injectIPAdapterLengths.enumerated() {
          let ipKeys = Input()
          let ipValues = Input()
          let block = PuLIDCrossAttentionKeysAndValues(
            prefix: "", name: "\(j).single_\(i)", outputDim: channels, k: 2048 / 16, h: 16,
            b: batchSize, t: (injectIPAdapterLength, h * w + referenceSequenceLength),
            usesFlashAttention: usesFlashAttention == .scaleMerged ? .scale1 : usesFlashAttention)
          hiddenStates = hiddenStates + block(image, ipKeys, ipValues).to(of: hiddenStates)
          injectedIPAdapters.append(contentsOf: [ipKeys, ipValues])
        }
        out = Functional.concat(axis: 1, encoderHiddenStates, hiddenStates)
      }
    }
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
  }
  let residualOut: Model.IO?
  if outputResidual, let imgInX = imgInX {
    residualOut = out - imgInX
  } else {
    residualOut = nil
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = normFinal(out).to(.Float16) .* scale + shift
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 16, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    batchSize, h * 2, w * 2, 16,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["img_in.weight"] = [xEmbedder.weight.name]
      mapping["img_in.bias"] = [xEmbedder.bias.name]
      if let contextEmbedder = contextEmbedder {
        mapping["txt_in.weight"] = [contextEmbedder.weight.name]
        mapping["txt_in.bias"] = [contextEmbedder.bias.name]
      }
      mapping["final_layer.linear.weight"] = [projOut.weight.name]
      mapping["final_layer.linear.bias"] = [projOut.bias.name]
    case .diffusers:
      mapping["x_embedder.weight"] = [xEmbedder.weight.name]
      mapping["x_embedder.bias"] = [xEmbedder.bias.name]
      if let contextEmbedder = contextEmbedder {
        mapping["context_embedder.weight"] = [contextEmbedder.weight.name]
        mapping["context_embedder.bias"] = [contextEmbedder.bias.name]
      }
      mapping["proj_out.weight"] = [projOut.weight.name]
      mapping["proj_out.bias"] = [projOut.bias.name]
    }
    return mapping
  }
  var inputs: [Input] = [x] + (residualIn.map { [$0] } ?? []) + rotAndContextIn
  inputs = inputs + adaLNChunks + injectedIPAdapters + injectedControls
  return (
    mapper, Model(inputs, [out] + (residualOut.map { [$0] } ?? []))
  )
}

private func LoRAMLPEmbedder(channels: Int, configuration: LoRANetworkConfiguration, name: String)
  -> (Model, Model, Model)
{
  let x = Input()
  let fc0 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LoRAFeedForward(
  hiddenSize: Int, intermediateSize: Int, upcast: Bool, configuration: LoRANetworkConfiguration,
  index: Int, name: String
) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = LoRADense(
    count: intermediateSize, configuration: configuration, flags: [.Float16], index: index,
    name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = LoRADense(
    count: hiddenSize, configuration: configuration, flags: [.Float32], index: index,
    name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func LoRAJointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextNorm1(context).to(.Float16) .* contextChunks[1] + contextChunks[0]
  let contextToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_k")
  let contextToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "c_q")
  let contextToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_v")
  var contextK = contextToKeys(contextOut).reshaped([b, t, h, k])
  let normAddedK = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_k")
  contextK = normAddedK(contextK)
  var contextQ = contextToQueries(contextOut).reshaped([b, t, h, k])
  let normAddedQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "c_norm_q")
  contextQ = normAddedQ(contextQ)
  let contextV = contextToValues(contextOut).reshaped([b, t, h, k])
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, contextK, xK)
  var values = Functional.concat(axis: 1, contextV, xV)
  var queries = Functional.concat(axis: 1, contextQ, xQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    keys = keys.transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries)
      .transposed(1, 2)
    values = values.transposed(1, 2)
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
      dot = dot.reshaped([b, h, (t + hw), t + hw])
      out = dot * values
      out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
    scaledDotProductAttention.gradientCheckpointing = false
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
    scaledDotProductAttention.gradientCheckpointing = false
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
    contextOut = context + (contextOut .* contextChunks[2]).to(of: context)
  }
  xOut = x + (xOut .* xChunks[2]).to(of: x)
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = LoRAFeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, configuration: configuration,
      index: layerIndex, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    if upcast {
      contextOut =
        contextOut
        + contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
        .* contextChunks[5].to(of: contextOut)
    } else {
      contextOut =
        contextOut
        + (contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
        .* contextChunks[5])
        .to(of: contextOut)
    }
    if configuration.gradientCheckpointingFeedForward {
      contextFF.gradientCheckpointing = true
    }
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, configuration: configuration,
    index: layerIndex, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if upcast {
    xOut = xOut + xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])
      .* xChunks[5].to(of: xOut)
  } else {
    xOut =
      xOut + (xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]) .* xChunks[5]).to(of: xOut)
  }
  if configuration.gradientCheckpointingFeedForward {
    xFF.gradientCheckpointing = true
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping: ModelWeightMapping = [:]
    mapping["\(prefix).txt_attn.qkv.weight"] = [
      contextToQueries.weight.name, contextToKeys.weight.name, contextToValues.weight.name,
    ]
    mapping["\(prefix).txt_attn.qkv.bias"] = [
      contextToQueries.bias.name, contextToKeys.bias.name, contextToValues.bias.name,
    ]
    mapping["\(prefix).txt_attn.norm.key_norm.scale"] = [normAddedK.weight.name]
    mapping["\(prefix).txt_attn.norm.query_norm.scale"] = [normAddedQ.weight.name]
    mapping["\(prefix).img_attn.qkv.weight"] = [
      xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
    ]
    mapping["\(prefix).img_attn.qkv.bias"] = [
      xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name,
    ]
    mapping["\(prefix).img_attn.norm.key_norm.scale"] = [normK.weight.name]
    mapping["\(prefix).img_attn.norm.query_norm.scale"] = [normQ.weight.name]
    if let contextUnifyheads = contextUnifyheads {
      mapping["\(prefix).txt_attn.proj.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).txt_attn.proj.bias"] = [contextUnifyheads.bias.name]
    }
    mapping["\(prefix).img_attn.proj.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).img_attn.proj.bias"] = [xUnifyheads.bias.name]
    if let contextLinear1 = contextLinear1,
      let contextOutProjection = contextOutProjection
    {
      mapping["\(prefix).txt_mlp.0.weight"] = [contextLinear1.weight.name]
      mapping["\(prefix).txt_mlp.0.bias"] = [contextLinear1.bias.name]
      mapping["\(prefix).txt_mlp.2.weight"] = [contextOutProjection.weight.name]
      mapping["\(prefix).txt_mlp.2.bias"] = [contextOutProjection.bias.name]
    }
    mapping["\(prefix).img_mlp.0.weight"] = [xLinear1.weight.name]
    mapping["\(prefix).img_mlp.0.bias"] = [xLinear1.bias.name]
    mapping["\(prefix).img_mlp.2.weight"] = [xOutProjection.weight.name]
    mapping["\(prefix).img_mlp.2.bias"] = [xOutProjection.bias.name]
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([context, x, rot] + contextChunks + xChunks, [contextOut, xOut]))
  } else {
    return (mapper, Model([context, x, rot] + contextChunks + xChunks, [xOut]))
  }
}

private func LoRASingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, referenceSequenceLength: Int,
  contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, t + hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, t + hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, t + hw, h, k])
  var keys = xK
  var values = xV
  var queries = xQ
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  // Now run attention.
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    keys = keys.transposed(1, 2)
    queries = ((1.0 / Float(k).squareRoot()) * queries)
      .transposed(1, 2)
    values = values.transposed(1, 2)
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
      dot = dot.reshaped([b, h, (t + hw), t + hw])
      out = dot * values
      out = out.reshaped([b, h, (t + hw), k]).transposed(1, 2).reshaped([b, (t + hw), h * k])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
    scaledDotProductAttention.gradientCheckpointing = false
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
    scaledDotProductAttention.gradientCheckpointing = false
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped(
      [b, hw - referenceSequenceLength, h * k], offset: [0, t, 0],
      strides: [(t + hw) * h * k, h * k, 1]
    )
    xIn = x.reshaped(
      [b, hw - referenceSequenceLength, h * k], offset: [0, t, 0],
      strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    xOut = xOut.reshaped(
      [b, hw - referenceSequenceLength, h * k], offset: [0, t, 0],
      strides: [(t + hw) * h * k, h * k, 1]
    )
  }
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_o")
  let (xLinear1, xOutProjection, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: false, configuration: configuration,
    index: layerIndex, name: "x")
  out = xUnifyheads(out) + xFF(xOut)
  if configuration.gradientCheckpointingFeedForward {
    xFF.gradientCheckpointing = true
  }
  out = xIn + (out .* xChunks[2]).to(of: xIn)
  let mapper: ModelWeightMapper = { _ in
    var mapping: ModelWeightMapping = [:]
    mapping["\(prefix).linear1.weight"] = [
      xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xLinear1.weight.name,
    ]
    mapping["\(prefix).linear1.bias"] = [
      xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name, xLinear1.bias.name,
    ]
    mapping["\(prefix).norm.key_norm.scale"] = [normK.weight.name]
    mapping["\(prefix).norm.query_norm.scale"] = [normQ.weight.name]
    mapping["\(prefix).linear2.weight"] = ModelWeightElement(
      [xUnifyheads.weight.name, xOutProjection.weight.name], format: .I)
    mapping["\(prefix).linear2.bias"] = [xOutProjection.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [out]))
}

public func LoRAFlux1Norm1(
  batchSize: Int, height: Int, width: Int, channels: Int,
  LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
  let xChunks = (0..<2).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = xChunks[1] .* xNorm1(out).to(.Float16) + xChunks[0]
  return Model([x] + xChunks, [out])
}

public func LoRAFlux1(
  batchSize: Int, tokenLength: Int, referenceSequenceLength: Int, height: Int, width: Int,
  channels: Int, layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, contextPreloaded: Bool, injectControls: Bool,
  injectIPAdapterLengths: [Int: [Int]], outputResidual: Bool, inputResidual: Bool,
  LoRAConfiguration: LoRANetworkConfiguration,
  useConvolutionForPatchify: Bool = true
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder: Model
  var out: Model.IO
  let imgInX: Model.IO?
  let referenceLatents: Input?
  if referenceSequenceLength > 0 && (layers.0 > 0 || layers.1 > 0) {
    let latents = Input()
    if useConvolutionForPatchify {
      xEmbedder = LoRAConvolution(
        groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
        hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
      let imgIn = xEmbedder(x).reshaped([batchSize, h * w, channels])
      out = Functional.concat(axis: 1, imgIn, latents, flags: [.disableOpt]).to(.Float32)
      if outputResidual {
        imgInX = imgIn.to(.Float32)
      } else {
        imgInX = nil
      }
    } else {
      xEmbedder = LoRADense(count: channels, configuration: LoRAConfiguration, name: "x_embedder")
      let imgIn = xEmbedder(x)
      out = Functional.concat(axis: 1, imgIn, latents, flags: [.disableOpt]).to(.Float32)
      if outputResidual {
        imgInX = imgIn.to(.Float32)
      } else {
        imgInX = nil
      }
    }
    referenceLatents = latents
  } else {
    if useConvolutionForPatchify {
      xEmbedder = LoRAConvolution(
        groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
        hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
      out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
    } else {
      xEmbedder = LoRADense(count: channels, configuration: LoRAConfiguration, name: "x_embedder")
      out = xEmbedder(x).to(.Float32)
    }
    referenceLatents = nil
    imgInX = out
  }
  var adaLNChunks = [Input]()
  var injectedControls = [Input]()
  var injectedIPAdapters = [Input]()
  let residualIn: Input?
  if inputResidual {
    let residual = Input()
    residualIn = residual
    out = out + residual
  } else {
    residualIn = nil
  }
  var mappers = [ModelWeightMapper]()
  let contextEmbedder: Model?
  var context: Model.IO?
  let rotAndContextIn: [Input]
  if layers.0 > 0 || layers.1 > 0 {
    let contextIn = Input()
    let rot = Input()
    if !contextPreloaded {
      let embedder = LoRADense(
        count: channels, configuration: LoRAConfiguration, name: "context_embedder")
      context = embedder(contextIn).to(.Float32)
      contextEmbedder = embedder
    } else {
      context = contextIn.to(.Float32)
      contextEmbedder = nil
    }
    rotAndContextIn = [rot] + (referenceLatents.map { [$0] } ?? []) + [contextIn]
  } else {
    context = nil
    contextEmbedder = nil
    rotAndContextIn = []
  }
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = LoRAJointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: tokenLength,
      hw: h * w + referenceSequenceLength, contextBlockPreOnly: false, upcast: i > 16,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration
    )
    let blockOut = block([context!, out, rotAndContextIn[0]] + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      block.gradientCheckpointing = true
    }
    if injectControls {
      let injectedControl = Input()
      let injectedControlFP32 = injectedControl.to(.Float32)
      injectedControlFP32.add(dependencies: [blockOut])  // out has no associated nodes, use blockOut instead.
      let scaleFactor: Float = 8
      out = out + (injectedControlFP32 * scaleFactor)
      injectedControls.append(injectedControl)
    }
    if let injectIPAdapterLengths = injectIPAdapterLengths[i] {
      let image = out
      for (j, injectIPAdapterLength) in injectIPAdapterLengths.enumerated() {
        let ipKeys = Input()
        let ipValues = Input()
        let block = PuLIDCrossAttentionKeysAndValues(
          prefix: "", name: "\(j).double_\(i)", outputDim: channels, k: 2048 / 16, h: 16,
          b: batchSize, t: (injectIPAdapterLength, h * w + referenceSequenceLength),
          usesFlashAttention: usesFlashAttention == .scaleMerged ? .scale1 : usesFlashAttention)
        out = out + block(image, ipKeys, ipValues).to(of: out)
        injectedIPAdapters.append(contentsOf: [ipKeys, ipValues])
      }
    }
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  if let context = context {
    out = Functional.concat(axis: 1, context, out)
  }
  for i in 0..<layers.1 {
    let xChunks = (0..<3).map { _ in Input() }
    let (mapper, block) = LoRASingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: tokenLength,
      hw: h * w + referenceSequenceLength, referenceSequenceLength: referenceSequenceLength,
      contextBlockPreOnly: i == layers.1 - 1, usesFlashAttention: usesFlashAttention,
      layerIndex: i + layers.0, configuration: LoRAConfiguration)
    out = block([out, rotAndContextIn[0]] + xChunks)
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      block.gradientCheckpointing = true
    }
    if injectControls {
      let injectedControl = Input()
      let injectedControlFP32 = injectedControl.to(.Float32)
      injectedControlFP32.add(dependencies: [out])
      let scaleFactor: Float = 8
      if i == layers.1 - 1 {
        out = out + (injectedControlFP32 * scaleFactor)
      } else {
        let encoderHiddenStates = out.reshaped(
          [batchSize, tokenLength, channels], offset: [0, 0, 0],
          strides: [(tokenLength + h * w + referenceSequenceLength) * channels, channels, 1])
        var hiddenStates = out.reshaped(
          [batchSize, h * w + referenceSequenceLength, channels], offset: [0, tokenLength, 0],
          strides: [(tokenLength + h * w + referenceSequenceLength) * channels, channels, 1])
        hiddenStates = hiddenStates + (injectedControlFP32 * scaleFactor)
        out = Functional.concat(axis: 1, encoderHiddenStates, hiddenStates)
      }
      injectedControls.append(injectedControl)
    }
    if let injectIPAdapterLengths = injectIPAdapterLengths[i + layers.0] {
      if i == layers.1 - 1 {
        let image = out
        for (j, injectIPAdapterLength) in injectIPAdapterLengths.enumerated() {
          let ipKeys = Input()
          let ipValues = Input()
          let block = PuLIDCrossAttentionKeysAndValues(
            prefix: "", name: "\(j).single_\(i)", outputDim: channels, k: 2048 / 16, h: 16,
            b: batchSize, t: (injectIPAdapterLength, h * w),
            usesFlashAttention: usesFlashAttention == .scaleMerged ? .scale1 : usesFlashAttention)
          out = out + block(image, ipKeys, ipValues).to(of: out)
          injectedIPAdapters.append(contentsOf: [ipKeys, ipValues])
        }
      } else {
        let encoderHiddenStates = out.reshaped(
          [batchSize, tokenLength, channels], offset: [0, 0, 0],
          strides: [(tokenLength + h * w + referenceSequenceLength) * channels, channels, 1])
        var hiddenStates = out.reshaped(
          [batchSize, h * w + referenceSequenceLength, channels], offset: [0, tokenLength, 0],
          strides: [(tokenLength + h * w + referenceSequenceLength) * channels, channels, 1])
        let image = hiddenStates
        for (j, injectIPAdapterLength) in injectIPAdapterLengths.enumerated() {
          let ipKeys = Input()
          let ipValues = Input()
          let block = PuLIDCrossAttentionKeysAndValues(
            prefix: "", name: "\(j).single_\(i)", outputDim: channels, k: 2048 / 16, h: 16,
            b: batchSize, t: (injectIPAdapterLength, h * w + referenceSequenceLength),
            usesFlashAttention: usesFlashAttention == .scaleMerged ? .scale1 : usesFlashAttention)
          hiddenStates = hiddenStates + block(image, ipKeys, ipValues).to(of: hiddenStates)
          injectedIPAdapters.append(contentsOf: [ipKeys, ipValues])
        }
        out = Functional.concat(axis: 1, encoderHiddenStates, hiddenStates)
      }
    }
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
  }
  let residualOut: Model.IO?
  if outputResidual, let imgInX = imgInX {
    residualOut = out - imgInX
  } else {
    residualOut = nil
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = normFinal(out).to(.Float16) .* scale + shift
  let projOut = LoRADense(
    count: 2 * 2 * 16, configuration: LoRAConfiguration, index: 0, name: "linear")
  out = projOut(out)
  // Unpatchify
  if useConvolutionForPatchify {
    out = out.reshaped([batchSize, h, w, 16, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous()
      .reshaped([
        batchSize, h * 2, w * 2, 16,
      ])
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["img_in.weight"] = [xEmbedder.weight.name]
    mapping["img_in.bias"] = [xEmbedder.bias.name]
    if let contextEmbedder = contextEmbedder {
      mapping["txt_in.weight"] = [contextEmbedder.weight.name]
      mapping["txt_in.bias"] = [contextEmbedder.bias.name]
    }
    mapping["final_layer.linear.weight"] = [projOut.weight.name]
    mapping["final_layer.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  var inputs: [Input] = [x] + (residualIn.map { [$0] } ?? []) + rotAndContextIn
  inputs = inputs + adaLNChunks + injectedIPAdapters + injectedControls
  return (
    mapper,
    Model(inputs, [out] + (residualOut.map { [$0] } ?? []), trainable: false)
  )
}

private func JointTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int, contextBlockPreOnly: Bool
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    Dense(count: k * h, name: "context_ada_ln_\($0)")
  }
  var contextChunks = contextAdaLNs.map { $0(c) }
  contextChunks[1] = 1 + contextChunks[1]
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  if !contextBlockPreOnly {
    contextChunks[4] = 1 + contextChunks[4]
  }
  xChunks[4] = 1 + xChunks[4]
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).txt_mod.lin.weight"] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].weight.name
        })
      mapping["\(prefix.0).txt_mod.lin.bias"] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].bias.name
        })
      mapping["\(prefix.0).img_mod.lin.weight"] = ModelWeightElement(
        (0..<6).map { xAdaLNs[$0].weight.name })
      mapping["\(prefix.0).img_mod.lin.bias"] = ModelWeightElement(
        (0..<6).map { xAdaLNs[$0].bias.name })
    case .diffusers:
      mapping["\(prefix.1).norm1_context.linear.weight"] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].weight.name
        })
      mapping["\(prefix.1).norm1_context.linear.bias"] = ModelWeightElement(
        (0..<(contextBlockPreOnly ? 2 : 6)).map {
          contextAdaLNs[$0].bias.name
        })
      mapping["\(prefix.1).norm1.linear.weight"] = ModelWeightElement(
        (0..<6).map { xAdaLNs[$0].weight.name })
      mapping["\(prefix.1).norm1.linear.bias"] = ModelWeightElement(
        (0..<6).map { xAdaLNs[$0].bias.name })
    }
    return mapping
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

private func SingleTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let xAdaLNs = (0..<3).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).modulation.lin.weight"] = ModelWeightElement(
        (0..<3).map { xAdaLNs[$0].weight.name })
      mapping["\(prefix.0).modulation.lin.bias"] = ModelWeightElement(
        (0..<3).map { xAdaLNs[$0].bias.name })
    case .diffusers:
      mapping["\(prefix.1).norm.linear.weight"] = ModelWeightElement(
        (0..<3).map { xAdaLNs[$0].weight.name })
      mapping["\(prefix.1).norm.linear.bias"] = ModelWeightElement(
        (0..<3).map { xAdaLNs[$0].bias.name })
    }
    return mapping
  }
  return (mapper, Model([c], xChunks))
}

public func Flux1Fixed(
  batchSize: (Int, Int), channels: Int, layers: (Int, Int),
  contextPreloaded: Bool, numberOfReferenceImages: Int, guidanceEmbed: Bool = false
) -> (ModelWeightMapper, Model) {
  let timestep = Input()
  let y = Input()
  var outs = [Model.IO]()
  var referenceImages = [Input]()
  if numberOfReferenceImages > 0 {
    let xEmbedder = Convolution(
      groups: 1, filters: channels, filterSize: [2, 2],
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
    for _ in 0..<numberOfReferenceImages {
      let x = Input()
      let out = xEmbedder(x)
      referenceImages.append(x)
      outs.append(out)
    }
  }
  let contextIn: Input?
  let guidance: Input?
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: "t")
  var vec = tEmbedder(timestep)
  let gMlp0: Model?
  let gMlp2: Model?
  if guidanceEmbed {
    let (mlp0, mlp2, gEmbedder) = MLPEmbedder(channels: channels, name: "guidance")
    let g = Input()
    vec = vec + gEmbedder(g)
    guidance = g
    gMlp0 = mlp0
    gMlp2 = mlp2
  } else {
    gMlp0 = nil
    gMlp2 = nil
    guidance = nil
  }
  let (yMlp0, yMlp2, yEmbedder) = MLPEmbedder(channels: channels, name: "vector")
  vec = vec + yEmbedder(y)
  let contextEmbedder: Model?
  if contextPreloaded {
    let cIn = Input()
    let embedder = Dense(count: channels, name: "context_embedder")
    let context = embedder(cIn)
    outs.append(context)
    contextIn = cIn
    contextEmbedder = embedder
  } else {
    contextIn = nil
    contextEmbedder = nil
  }
  let c = vec.reshaped([batchSize.1, 1, channels]).swish()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (mapper, block) = JointTransformerBlockFixed(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      contextBlockPreOnly: false)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlockFixed(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let scale = Dense(count: channels, name: "ada_ln_0")
  let shift = Dense(count: channels, name: "ada_ln_1")
  outs.append(contentsOf: [shift(c), 1 + scale(c)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["time_in.in_layer.weight"] = [tMlp0.weight.name]
      mapping["time_in.in_layer.bias"] = [tMlp0.bias.name]
      mapping["time_in.out_layer.weight"] = [tMlp2.weight.name]
      mapping["time_in.out_layer.bias"] = [tMlp2.bias.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["guidance_in.in_layer.weight"] = [gMlp0.weight.name]
        mapping["guidance_in.in_layer.bias"] = [gMlp0.bias.name]
        mapping["guidance_in.out_layer.weight"] = [gMlp2.weight.name]
        mapping["guidance_in.out_layer.bias"] = [gMlp2.bias.name]
      }
      mapping["vector_in.in_layer.weight"] = [yMlp0.weight.name]
      mapping["vector_in.in_layer.bias"] = [yMlp0.bias.name]
      mapping["vector_in.out_layer.weight"] = [yMlp2.weight.name]
      mapping["vector_in.out_layer.bias"] = [yMlp2.bias.name]
      if let contextEmbedder = contextEmbedder {
        mapping["txt_in.weight"] = [contextEmbedder.weight.name]
        mapping["txt_in.bias"] = [contextEmbedder.bias.name]
      }
      mapping["final_layer.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
      mapping["final_layer.adaLN_modulation.1.bias"] = [shift.bias.name, scale.bias.name]
    case .diffusers:
      mapping["time_text_embed.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_1.bias"] = [tMlp0.bias.name]
      mapping["time_text_embed.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_2.bias"] = [tMlp2.bias.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["time_text_embed.guidance_embedder.linear_1.weight"] = [gMlp0.weight.name]
        mapping["time_text_embed.guidance_embedder.linear_1.bias"] = [gMlp0.bias.name]
        mapping["time_text_embed.guidance_embedder.linear_2.weight"] = [gMlp2.weight.name]
        mapping["time_text_embed.guidance_embedder.linear_2.bias"] = [gMlp2.bias.name]
      }
      mapping["time_text_embed.text_embedder.linear_1.weight"] = [yMlp0.weight.name]
      mapping["time_text_embed.text_embedder.linear_1.bias"] = [yMlp0.bias.name]
      mapping["time_text_embed.text_embedder.linear_2.weight"] = [yMlp2.weight.name]
      mapping["time_text_embed.text_embedder.linear_2.bias"] = [yMlp2.bias.name]
      if let contextEmbedder = contextEmbedder {
        mapping["context_embedder.weight"] = [contextEmbedder.weight.name]
        mapping["context_embedder.bias"] = [contextEmbedder.bias.name]
      }
      mapping["norm_out.linear.weight"] = [scale.weight.name, shift.weight.name]
      mapping["norm_out.linear.bias"] = [scale.bias.name, shift.bias.name]
    }
    return mapping
  }
  return (
    mapper,
    Model(
      (contextIn.map { [$0] } ?? []) + referenceImages + [timestep, y]
        + (guidance.map { [$0] } ?? []), outs)
  )
}

private func ChromaDistillGuidanceLayer(prefix: String, layerIndex: Int) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let (mlp0, mlp2, mlp) = MLPEmbedder(channels: 5_120, name: "distilled_guidance")
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], name: "distilled_guidance_norm")
  let out = x + mlp(norm(x))
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).layers.\(layerIndex).in_layer.weight"] = [mlp0.weight.name]
    mapping["\(prefix).layers.\(layerIndex).in_layer.bias"] = [mlp0.bias.name]
    mapping["\(prefix).layers.\(layerIndex).out_layer.weight"] = [mlp2.weight.name]
    mapping["\(prefix).layers.\(layerIndex).out_layer.bias"] = [mlp2.bias.name]
    mapping["\(prefix).norms.\(layerIndex).scale"] = [norm.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

public func ChromaFixed(
  channels: Int, distilledGuidanceLayers: Int, layers: (Int, Int), contextPreloaded: Bool
) -> (
  ModelWeightMapper, Model
) {
  let timestep = Input()
  var outs = [Model.IO]()
  let contextIn: Input?
  let contextEmbedder: Model?
  if contextPreloaded {
    let cIn = Input()
    let embedder = Dense(count: channels, name: "context_embedder")
    let context = embedder(cIn)
    outs.append(context)
    contextIn = cIn
    contextEmbedder = embedder
  } else {
    contextIn = nil
    contextEmbedder = nil
  }
  let inProj = Dense(count: 5_120, name: "distilled_guidance_in_proj")
  var out = inProj(timestep)
  var mappers = [ModelWeightMapper]()
  for i in 0..<distilledGuidanceLayers {
    let (mapper, block) = ChromaDistillGuidanceLayer(
      prefix: "distilled_guidance_layer", layerIndex: i)
    out = block(out)
    mappers.append(mapper)
  }
  let outProj = Dense(count: channels, name: "distilled_guidance_out_proj")
  out = outProj(out)
  let mods = out.chunked(layers.0 * 12 + layers.1 * 3 + 2, axis: 1)
  for i in 0..<layers.0 {
    var contextMods = Array(
      mods[(layers.1 * 3 + layers.0 * 6 + i * 6)..<(layers.1 * 3 + layers.0 * 6 + (i + 1) * 6)])
    contextMods[1] = contextMods[1] + 1
    contextMods[4] = contextMods[4] + 1
    outs.append(contentsOf: contextMods)
    var xMods = Array(mods[(layers.1 * 3 + i * 6)..<(layers.1 * 3 + (i + 1) * 6)])
    xMods[1] = xMods[1] + 1
    xMods[4] = xMods[4] + 1
    outs.append(contentsOf: xMods)
  }
  for i in 0..<layers.1 {
    var xMods = Array(mods[(i * 3)..<((i + 1) * 3)])
    xMods[1] = xMods[1] + 1
    outs.append(contentsOf: xMods)
  }
  let shift = mods[layers.0 * 12 + layers.1 * 3]
  let scale = mods[layers.0 * 12 + layers.1 * 3 + 1]
  outs.append(contentsOf: [shift, 1 + scale])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    if let contextEmbedder = contextEmbedder {
      mapping["txt_in.weight"] = [contextEmbedder.weight.name]
      mapping["txt_in.bias"] = [contextEmbedder.bias.name]
    }
    mapping["distilled_guidance_layer.in_proj.weight"] = [inProj.weight.name]
    mapping["distilled_guidance_layer.in_proj.bias"] = [inProj.bias.name]
    mapping["distilled_guidance_layer.out_proj.weight"] = [outProj.weight.name]
    mapping["distilled_guidance_layer.out_proj.bias"] = [outProj.bias.name]
    return mapping
  }
  return (
    mapper,
    Model((contextIn.map { [$0] } ?? []) + [timestep], outs)
  )
}

private func JointTransformerBlockFixedOutputShapes(
  prefix: String, batchSize: Int, k: Int, h: Int, contextBlockPreOnly: Bool
) -> [TensorShape] {
  let contextOutputShapes = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    TensorShape([batchSize, 1, k * h])
  }
  let xOutputShapes = (0..<6).map { _ in TensorShape([batchSize, 1, k * h]) }
  return contextOutputShapes + xOutputShapes
}

private func SingleTransformerBlockFixedOutputShapes(
  prefix: String, batchSize: Int, k: Int, h: Int
) -> [TensorShape] {
  let xOutputShapes = (0..<3).map { _ in TensorShape([batchSize, 1, k * h]) }
  return xOutputShapes
}

public func Flux1FixedOutputShapes(
  batchSize: (Int, Int), tokenLength: Int, channels: Int, layers: (Int, Int), contextPreloaded: Bool
) -> [TensorShape] {
  var outs = [TensorShape]()
  if contextPreloaded {
    outs.append(TensorShape([batchSize.0, tokenLength, channels]))
  }
  for i in 0..<layers.0 {
    let contextBlockPreOnly = i == layers.0 - 1 && layers.1 == 0
    let outputShapes = JointTransformerBlockFixedOutputShapes(
      prefix: "double_blocks.\(i)", batchSize: batchSize.1, k: 128, h: channels / 128,
      contextBlockPreOnly: contextBlockPreOnly)
    outs.append(contentsOf: outputShapes)
  }
  for i in 0..<layers.1 {
    let outputShapes = SingleTransformerBlockFixedOutputShapes(
      prefix: "single_blocks.\(i)", batchSize: batchSize.1, k: 128, h: channels / 128)
    outs.append(contentsOf: outputShapes)
  }
  outs.append(contentsOf: [
    TensorShape([batchSize.1, 1, channels]), TensorShape([batchSize.1, 1, channels]),
  ])
  return outs
}

private func LoRAJointTransformerBlockFixed(
  prefix: String, k: Int, h: Int, contextBlockPreOnly: Bool, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let contextAdaLNs = (0..<(contextBlockPreOnly ? 2 : 6)).map {
    LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "context_ada_ln_\($0)")
  }
  var contextChunks = contextAdaLNs.map { $0(c) }
  contextChunks[1] = 1 + contextChunks[1]
  let xAdaLNs = (0..<6).map {
    LoRADense(count: k * h, configuration: configuration, index: layerIndex, name: "x_ada_ln_\($0)")
  }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  if !contextBlockPreOnly {
    contextChunks[4] = 1 + contextChunks[4]
  }
  xChunks[4] = 1 + xChunks[4]
  let mapper: ModelWeightMapper = { _ in
    var mapping: ModelWeightMapping = [:]
    mapping["\(prefix).txt_mod.lin.weight"] = ModelWeightElement(
      (0..<(contextBlockPreOnly ? 2 : 6)).map {
        contextAdaLNs[$0].weight.name
      })
    mapping["\(prefix).txt_mod.lin.bias"] = ModelWeightElement(
      (0..<(contextBlockPreOnly ? 2 : 6)).map {
        contextAdaLNs[$0].bias.name
      })
    mapping["\(prefix).img_mod.lin.weight"] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].weight.name })
    mapping["\(prefix).img_mod.lin.bias"] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].bias.name })
    return mapping
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

private func LoRASingleTransformerBlockFixed(
  prefix: String, k: Int, h: Int, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let xAdaLNs = (0..<3).map {
    LoRADense(count: k * h, configuration: configuration, index: layerIndex, name: "x_ada_ln_\($0)")
  }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  let mapper: ModelWeightMapper = { _ in
    var mapping: ModelWeightMapping = [:]
    mapping["\(prefix).modulation.lin.weight"] = ModelWeightElement(
      (0..<3).map { xAdaLNs[$0].weight.name })
    mapping["\(prefix).modulation.lin.bias"] = ModelWeightElement(
      (0..<3).map { xAdaLNs[$0].bias.name })
    return mapping
  }
  return (mapper, Model([c], xChunks))
}

public func LoRAFlux1Fixed(
  batchSize: (Int, Int), channels: Int, layers: (Int, Int),
  LoRAConfiguration: LoRANetworkConfiguration,
  contextPreloaded: Bool, numberOfReferenceImages: Int, guidanceEmbed: Bool = false
) -> (ModelWeightMapper, Model) {
  let timestep = Input()
  let y = Input()
  var outs = [Model.IO]()
  var referenceImages = [Input]()
  if numberOfReferenceImages > 0 {
    let xEmbedder = Convolution(
      groups: 1, filters: channels, filterSize: [2, 2],
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
    for _ in 0..<numberOfReferenceImages {
      let x = Input()
      let out = xEmbedder(x)
      referenceImages.append(x)
      outs.append(out)
    }
  }
  let contextIn: Input?
  let guidance: Input?
  let (tMlp0, tMlp2, tEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "t")
  var vec = tEmbedder(timestep)
  let gMlp0: Model?
  let gMlp2: Model?
  if guidanceEmbed {
    let (mlp0, mlp2, gEmbedder) = LoRAMLPEmbedder(
      channels: channels, configuration: LoRAConfiguration, name: "guidance")
    let g = Input()
    vec = vec + gEmbedder(g)
    guidance = g
    gMlp0 = mlp0
    gMlp2 = mlp2
  } else {
    gMlp0 = nil
    gMlp2 = nil
    guidance = nil
  }
  let (yMlp0, yMlp2, yEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "vector")
  vec = vec + yEmbedder(y)
  let contextEmbedder: Model?
  if contextPreloaded {
    let cIn = Input()
    let embedder = LoRADense(
      count: channels, configuration: LoRAConfiguration, name: "context_embedder")
    let context = embedder(cIn)
    outs.append(context)
    contextIn = cIn
    contextEmbedder = embedder
  } else {
    contextIn = nil
    contextEmbedder = nil
  }
  let c = vec.reshaped([batchSize.1, 1, channels]).swish()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (mapper, block) = LoRAJointTransformerBlockFixed(
      prefix: "double_blocks.\(i)", k: 128, h: channels / 128,
      contextBlockPreOnly: false, layerIndex: i, configuration: LoRAConfiguration)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = LoRASingleTransformerBlockFixed(
      prefix: "single_blocks.\(i)", k: 128, h: channels / 128, layerIndex: i + layers.0,
      configuration: LoRAConfiguration)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let scale = LoRADense(count: channels, configuration: LoRAConfiguration, name: "ada_ln_0")
  let shift = LoRADense(count: channels, configuration: LoRAConfiguration, name: "ada_ln_1")
  outs.append(contentsOf: [shift(c), 1 + scale(c)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["time_in.in_layer.weight"] = [tMlp0.weight.name]
    mapping["time_in.in_layer.bias"] = [tMlp0.bias.name]
    mapping["time_in.out_layer.weight"] = [tMlp2.weight.name]
    mapping["time_in.out_layer.bias"] = [tMlp2.bias.name]
    if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
      mapping["guidance_in.in_layer.weight"] = [gMlp0.weight.name]
      mapping["guidance_in.in_layer.bias"] = [gMlp0.bias.name]
      mapping["guidance_in.out_layer.weight"] = [gMlp2.weight.name]
      mapping["guidance_in.out_layer.bias"] = [gMlp2.bias.name]
    }
    mapping["vector_in.in_layer.weight"] = [yMlp0.weight.name]
    mapping["vector_in.in_layer.bias"] = [yMlp0.bias.name]
    mapping["vector_in.out_layer.weight"] = [yMlp2.weight.name]
    mapping["vector_in.out_layer.bias"] = [yMlp2.bias.name]
    if let contextEmbedder = contextEmbedder {
      mapping["txt_in.weight"] = [contextEmbedder.weight.name]
      mapping["txt_in.bias"] = [contextEmbedder.bias.name]
    }
    mapping["final_layer.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    mapping["final_layer.adaLN_modulation.1.bias"] = [shift.bias.name, scale.bias.name]
    return mapping
  }
  return (
    mapper,
    Model(
      (contextIn.map { [$0] } ?? []) + referenceImages + [timestep, y]
        + (guidance.map { [$0] } ?? []), outs)
  )
}

private func LoRAChromaDistillGuidanceLayer(
  prefix: String, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let (mlp0, mlp2, mlp) = LoRAMLPEmbedder(
    channels: 5_120, configuration: configuration, name: "distilled_guidance")
  let norm = RMSNorm(epsilon: 1e-6, axis: [2], name: "distilled_guidance_norm")
  let out = x + mlp(norm(x))
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).layers.\(layerIndex).in_layer.weight"] = [mlp0.weight.name]
    mapping["\(prefix).layers.\(layerIndex).in_layer.bias"] = [mlp0.bias.name]
    mapping["\(prefix).layers.\(layerIndex).out_layer.weight"] = [mlp2.weight.name]
    mapping["\(prefix).layers.\(layerIndex).out_layer.bias"] = [mlp2.bias.name]
    mapping["\(prefix).norms.\(layerIndex).scale"] = [norm.weight.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

public func LoRAChromaFixed(
  channels: Int, distilledGuidanceLayers: Int, layers: (Int, Int),
  LoRAConfiguration: LoRANetworkConfiguration, contextPreloaded: Bool
) -> (
  ModelWeightMapper, Model
) {
  let timestep = Input()
  var outs = [Model.IO]()
  let contextIn: Input?
  let contextEmbedder: Model?
  if contextPreloaded {
    let cIn = Input()
    let embedder = LoRADense(
      count: channels, configuration: LoRAConfiguration, name: "context_embedder")
    let context = embedder(cIn)
    outs.append(context)
    contextIn = cIn
    contextEmbedder = embedder
  } else {
    contextIn = nil
    contextEmbedder = nil
  }
  let inProj = LoRADense(
    count: 5_120, configuration: LoRAConfiguration, name: "distilled_guidance_in_proj")
  var out = inProj(timestep)
  var mappers = [ModelWeightMapper]()
  for i in 0..<distilledGuidanceLayers {
    let (mapper, block) = LoRAChromaDistillGuidanceLayer(
      prefix: "distilled_guidance_layer", layerIndex: i, configuration: LoRAConfiguration)
    out = block(out)
    mappers.append(mapper)
  }
  let outProj = LoRADense(
    count: channels, configuration: LoRAConfiguration, name: "distilled_guidance_out_proj")
  out = outProj(out)
  let mods = out.chunked(layers.0 * 12 + layers.1 * 3 + 2, axis: 1)
  for i in 0..<layers.0 {
    var contextMods = Array(
      mods[(layers.1 * 3 + layers.0 * 6 + i * 6)..<(layers.1 * 3 + layers.0 * 6 + (i + 1) * 6)])
    contextMods[1] = contextMods[1] + 1
    contextMods[4] = contextMods[4] + 1
    outs.append(contentsOf: contextMods)
    var xMods = Array(mods[(layers.1 * 3 + i * 6)..<(layers.1 * 3 + (i + 1) * 6)])
    xMods[1] = xMods[1] + 1
    xMods[4] = xMods[4] + 1
    outs.append(contentsOf: xMods)
  }
  for i in 0..<layers.1 {
    var xMods = Array(mods[(i * 3)..<((i + 1) * 3)])
    xMods[1] = xMods[1] + 1
    outs.append(contentsOf: xMods)
  }
  let shift = mods[layers.0 * 12 + layers.1 * 3]
  let scale = mods[layers.0 * 12 + layers.1 * 3 + 1]
  outs.append(contentsOf: [shift, 1 + scale])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    if let contextEmbedder = contextEmbedder {
      mapping["txt_in.weight"] = [contextEmbedder.weight.name]
      mapping["txt_in.bias"] = [contextEmbedder.bias.name]
    }
    mapping["distilled_guidance_layer.in_proj.weight"] = [inProj.weight.name]
    mapping["distilled_guidance_layer.in_proj.bias"] = [inProj.bias.name]
    mapping["distilled_guidance_layer.out_proj.weight"] = [outProj.weight.name]
    mapping["distilled_guidance_layer.out_proj.bias"] = [outProj.bias.name]
    return mapping
  }
  return (
    mapper,
    Model((contextIn.map { [$0] } ?? []) + [timestep], outs)
  )
}

public func ControlNetFlux1(
  union: Bool,
  batchSize: Int, tokenLength: Int, height: Int, width: Int, channels: Int, layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let tokenLength = union ? tokenLength + 1 : tokenLength
  let x = Input()
  let controlnetX = Input()
  let contextIn = Input()
  let rot = Input()
  let h = height / 2
  let w = width / 2
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  // Doing this here such that when we do tiled diffusion, we deal with smaller input even though there are some redundant compute.
  let controlnetXEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "controlnet_x_embedder")
  var out =
    xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
    + controlnetXEmbedder(controlnetX).reshaped([1, h * w, channels]).to(.Float32)
  var adaLNChunks = [Input]()
  var mappers = [ModelWeightMapper]()
  var context = contextIn.to(.Float32)
  var zeroConvs = [Dense]()
  var outs = [Model.IO]()
  for i in 0..<layers.0 {
    let contextBlockPreOnly = i == layers.0 - 1 && layers.1 == 0
    let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = JointTransformerBlock(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize, t: tokenLength,
      hw: h * w, contextBlockPreOnly: contextBlockPreOnly, upcast: i > (layers.0 - 3),
      usesFlashAttention: usesFlashAttention
    )
    let blockOut = block([context, out, rot] + contextChunks + xChunks)
    if contextBlockPreOnly {
      out = blockOut
    } else {
      context = blockOut[0]
      out = blockOut[1]
    }
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
    let zeroConv = Dense(count: channels, name: "zero_conv")
    if let last = outs.last {
      blockOut.add(dependencies: [last])  // out has no associated nodes, use blockOut instead.
    }
    let scaleFactor: Float = 1 / 8  // We already scaled bias for zero conv.
    outs.append(zeroConv((out * scaleFactor).to(of: controlnetX)))
    zeroConvs.append(zeroConv)
  }
  if layers.1 > 0 {
    out = Functional.concat(axis: 1, context, out)
  }
  for i in 0..<layers.1 {
    let xChunks = (0..<3).map { _ in Input() }
    let (mapper, block) = SingleTransformerBlock(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize, t: tokenLength, hw: h * w, referenceSequenceLength: 0,
      contextBlockPreOnly: i == layers.1 - 1, usesFlashAttention: usesFlashAttention)
    out = block([out, rot] + xChunks)
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
    let zeroConv = Dense(count: channels, name: "zero_conv")
    if let last = outs.last {
      out.add(dependencies: [last])
    }
    let scaleFactor: Float = 1 / 8
    if i == layers.1 - 1 {
      outs.append(zeroConv((out * scaleFactor).to(of: controlnetX)))
    } else {
      outs.append(
        zeroConv(
          (out.reshaped(
            [batchSize, h * w, channels], offset: [0, tokenLength, 0],
            strides: [(tokenLength + h * w) * channels, channels, 1]) * scaleFactor).to(
              of: controlnetX)))
    }
    zeroConvs.append(zeroConv)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["img_in.weight"] = [xEmbedder.weight.name]
      mapping["img_in.bias"] = [xEmbedder.bias.name]
      mapping["pos_embed_input.weight"] = [controlnetXEmbedder.weight.name]
      mapping["pos_embed_input.bias"] = [controlnetXEmbedder.bias.name]
      for i in 0..<layers.0 {
        mapping["controlnet_blocks.\(i).linear.weight"] = [zeroConvs[i].weight.name]
        mapping["controlnet_blocks.\(i).linear.bias"] = [zeroConvs[i].bias.name]
      }
      for i in 0..<layers.1 {
        mapping["single_controlnet_blocks.\(i).linear.weight"] = [
          zeroConvs[i + layers.0].weight.name
        ]
        mapping["single_controlnet_blocks.\(i).linear.bias"] = [zeroConvs[i + layers.0].bias.name]
      }
    case .diffusers:
      mapping["x_embedder.weight"] = [xEmbedder.weight.name]
      mapping["x_embedder.bias"] = [xEmbedder.bias.name]
      mapping["controlnet_x_embedder.weight"] = [controlnetXEmbedder.weight.name]
      mapping["controlnet_x_embedder.bias"] = [controlnetXEmbedder.bias.name]
      for i in 0..<layers.0 {
        mapping["controlnet_blocks.\(i).weight"] = [zeroConvs[i].weight.name]
        mapping["controlnet_blocks.\(i).bias"] = [zeroConvs[i].bias.name]
      }
      for i in 0..<layers.1 {
        mapping["controlnet_single_blocks.\(i).weight"] = [zeroConvs[i + layers.0].weight.name]
        mapping["controlnet_single_blocks.\(i).bias"] = [zeroConvs[i + layers.0].bias.name]
      }
    }
    return mapping
  }
  return (mapper, Model([x, controlnetX, rot, contextIn] + adaLNChunks, outs))
}

public func ControlNetFlux1Fixed<FloatType: TensorNumeric & BinaryFloatingPoint>(
  union: Bool,
  batchSize: (Int, Int), channels: Int, layers: (Int, Int), guidanceEmbed: Bool = false,
  of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let timestep = Input()
  let y = Input()
  let contextIn = Input()
  let guidance: Input?
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: "t")
  var vec = tEmbedder(timestep)
  let gMlp0: Model?
  let gMlp2: Model?
  if guidanceEmbed {
    let (mlp0, mlp2, gEmbedder) = MLPEmbedder(channels: channels, name: "guidance")
    let g = Input()
    vec = vec + gEmbedder(g)
    guidance = g
    gMlp0 = mlp0
    gMlp2 = mlp2
  } else {
    gMlp0 = nil
    gMlp2 = nil
    guidance = nil
  }
  let (yMlp0, yMlp2, yEmbedder) = MLPEmbedder(channels: channels, name: "vector")
  vec = vec + yEmbedder(y)
  var outs = [Model.IO]()
  let contextEmbedder = Dense(count: channels, name: "context_embedder")
  var context = contextEmbedder(contextIn)
  let controlnetMode: Input?
  let controlnetModeEmbedder: Embedding?
  if union {
    let modeEmbedder = Embedding(
      FloatType.self, vocabularySize: 10, embeddingSize: channels, name: "controlnet_mode_embedder")
    let mode = Input()
    var modeEmbed = modeEmbedder(mode).reshaped([1, 1, channels])
    if batchSize.0 > 1 {
      modeEmbed = Concat(axis: 0)(Array(repeating: modeEmbed, count: batchSize.0))
    }
    context = Functional.concat(axis: 1, modeEmbed, context)
    controlnetMode = mode
    controlnetModeEmbedder = modeEmbedder
  } else {
    controlnetMode = nil
    controlnetModeEmbedder = nil
  }
  outs.append(context)
  let c = vec.reshaped([batchSize.1, 1, channels]).swish()
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let contextBlockPreOnly = i == layers.0 - 1 && layers.1 == 0
    let (mapper, block) = JointTransformerBlockFixed(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      contextBlockPreOnly: contextBlockPreOnly)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlockFixed(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128)
    let blockOut = block(c)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["time_in.in_layer.weight"] = [tMlp0.weight.name]
      mapping["time_in.in_layer.bias"] = [tMlp0.bias.name]
      mapping["time_in.out_layer.weight"] = [tMlp2.weight.name]
      mapping["time_in.out_layer.bias"] = [tMlp2.bias.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["guidance_in.in_layer.weight"] = [gMlp0.weight.name]
        mapping["guidance_in.in_layer.bias"] = [gMlp0.bias.name]
        mapping["guidance_in.out_layer.weight"] = [gMlp2.weight.name]
        mapping["guidance_in.out_layer.bias"] = [gMlp2.bias.name]
      }
      mapping["vector_in.in_layer.weight"] = [yMlp0.weight.name]
      mapping["vector_in.in_layer.bias"] = [yMlp0.bias.name]
      mapping["vector_in.out_layer.weight"] = [yMlp2.weight.name]
      mapping["vector_in.out_layer.bias"] = [yMlp2.bias.name]
      mapping["txt_in.weight"] = [contextEmbedder.weight.name]
      mapping["txt_in.bias"] = [contextEmbedder.bias.name]
    case .diffusers:
      mapping["time_text_embed.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_1.bias"] = [tMlp0.bias.name]
      mapping["time_text_embed.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
      mapping["time_text_embed.timestep_embedder.linear_2.bias"] = [tMlp2.bias.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["time_text_embed.guidance_embedder.linear_1.weight"] = [gMlp0.weight.name]
        mapping["time_text_embed.guidance_embedder.linear_1.bias"] = [gMlp0.bias.name]
        mapping["time_text_embed.guidance_embedder.linear_2.weight"] = [gMlp2.weight.name]
        mapping["time_text_embed.guidance_embedder.linear_2.bias"] = [gMlp2.bias.name]
      }
      mapping["time_text_embed.text_embedder.linear_1.weight"] = [yMlp0.weight.name]
      mapping["time_text_embed.text_embedder.linear_1.bias"] = [yMlp0.bias.name]
      mapping["time_text_embed.text_embedder.linear_2.weight"] = [yMlp2.weight.name]
      mapping["time_text_embed.text_embedder.linear_2.bias"] = [yMlp2.bias.name]
      mapping["context_embedder.weight"] = [contextEmbedder.weight.name]
      mapping["context_embedder.bias"] = [contextEmbedder.bias.name]
      if let controlnetModeEmbedder = controlnetModeEmbedder {
        mapping["controlnet_mode_embedder.weight"] = [controlnetModeEmbedder.weight.name]
      }
    }
    return mapping
  }
  return (
    mapper,
    Model(
      (controlnetMode.map { [$0] } ?? []) + [contextIn, timestep, y]
        + (guidance.map { [$0] } ?? []), outs)
  )
}
