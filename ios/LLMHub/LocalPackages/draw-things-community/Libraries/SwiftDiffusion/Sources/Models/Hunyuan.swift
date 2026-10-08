import DiffusionMappings
import Foundation
import NNC

public func HunyuanRotaryPositionEmbedding(
  height: Int, width: Int, time: Int, tokenLength: Int, channels: Int, heads: Int = 1
)
  -> (Tensor<Float>, Tensor<Float>)
{
  var rotNdTensor0 = Tensor<Float>(.CPU, .NHWC(1, time * height * width, heads, channels))
  var rotNdTensor1 = Tensor<Float>(
    .CPU, .NHWC(1, time * height * width + tokenLength, heads, channels))
  let dim0 = channels / 8
  let dim1 = channels * 7 / 16
  let dim2 = dim1
  assert(channels % 16 == 0)
  for t in 0..<time {
    for y in 0..<height {
      for x in 0..<width {
        let i = t * height * width + y * width + x
        for j in 0..<heads {
          for k in 0..<(dim0 / 2) {
            let theta = Double(t) * 1.0 / pow(256, Double(k) * 2 / Double(dim0))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotNdTensor0[0, i, j, k * 2] = Float(costheta)
            rotNdTensor0[0, i, j, k * 2 + 1] = Float(sintheta)
            rotNdTensor1[0, i, j, k * 2] = Float(costheta)
            rotNdTensor1[0, i, j, k * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim1 / 2) {
            let theta = Double(y) * 1.0 / pow(256, Double(k) * 2 / Double(dim1))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotNdTensor0[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
            rotNdTensor0[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
            rotNdTensor1[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
            rotNdTensor1[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim2 / 2) {
            let theta = Double(x) * 1.0 / pow(256, Double(k) * 2 / Double(dim2))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotNdTensor0[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
            rotNdTensor0[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
            rotNdTensor1[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
            rotNdTensor1[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
          }
        }
      }
    }
  }
  for i in (time * height * width)..<(time * height * width + tokenLength) {
    for j in 0..<heads {
      for k in 0..<(channels / 2) {
        rotNdTensor1[0, i, j, k * 2] = 1
        rotNdTensor1[0, i, j, k * 2 + 1] = 0
      }
    }
  }
  return (rotNdTensor0, rotNdTensor1)
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func RefinerSelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int))
  -> (
    Model, ModelWeightMapper
  )
{
  let x = Input()
  let tokeys = Dense(count: k * hk, name: "refiner_k_proj")
  let toqueries = Dense(count: k * h, name: "refiner_q_proj")
  let tovalues = Dense(count: k * hk, name: "refiner_v_proj")
  var out: Model.IO
  if t.0 > 0 {
    let keys = tokeys(x)
    let queries = toqueries(x)
    let values = tovalues(x)
    let out0 = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
      queries.reshaped([b, t.0, h, k]), keys.reshaped([b, t.0, hk, k]),
      values.reshaped([b, t.0, hk, k])
    ).reshaped([b * t.0, h * k])
    let out1 = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
      queries.reshaped(
        [b, t.1, h, k], offset: [0, b * t.0, 0, 0], strides: [t.1 * h * k, h * k, k, 1]),
      keys.reshaped(
        [b, t.1, hk, k], offset: [0, b * t.0, 0, 0], strides: [t.1 * hk * k, hk * k, k, 1]),
      values.reshaped(
        [b, t.1, hk, k], offset: [0, b * t.0, 0, 0], strides: [t.1 * hk * k, hk * k, k, 1])
    ).reshaped([b * t.1, h * k])
    out = Functional.concat(axis: 0, out0, out1)
  } else {
    let keys = tokeys(x).reshaped([b, t.1, hk, k])
    let queries = toqueries(x).reshaped([b, t.1, h, k])
    let values = tovalues(x).reshaped([b, t.1, hk, k])
    out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
      queries, keys, values
    ).reshaped([b * t.1, h * k])
  }
  let unifyheads = Dense(count: k * h, name: "refiner_out_proj")
  out = unifyheads(out)
  let mapper: ModelWeightMapper = { _ in
    // The rotary in Llama is first half and second half, we can be clever and do the extra transpose here to use with cmul.
    var mapping = ModelWeightMapping()
    mapping["\(prefix).self_attn_qkv.weight"] = [
      toqueries.weight.name, tokeys.weight.name, tovalues.weight.name,
    ]
    mapping["\(prefix).self_attn.qkv.weight"] = mapping["\(prefix).self_attn_qkv.weight"]
    mapping["\(prefix).self_attn_qkv.bias"] = [
      toqueries.bias.name, tokeys.bias.name, tovalues.bias.name,
    ]
    mapping["\(prefix).self_attn.qkv.bias"] = mapping["\(prefix).self_attn_qkv.bias"]
    mapping["\(prefix).self_attn_proj.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).self_attn.proj.weight"] = mapping["\(prefix).self_attn_proj.weight"]
    mapping["\(prefix).self_attn_proj.bias"] = [unifyheads.bias.name]
    mapping["\(prefix).self_attn.proj.bias"] = mapping["\(prefix).self_attn_proj.bias"]
    return mapping
  }
  return (Model([x], [out]), mapper)
}

private func IndividualRefinerBlock(prefix: String, b: Int, t: (Int, Int)) -> (
  Model, ModelWeightMapper
) {
  let x = Input()
  let c = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [1], name: "refiner_norm1")
  let gateMsa = Dense(count: 3_072, name: "refiner_ada_ln_msa")
  let (attention, attentionMapper) = RefinerSelfAttention(
    prefix: prefix, k: 128, h: 24, hk: 24, b: b, t: t)
  let gateMsaC = gateMsa(c)
  let attentionX = attention(norm1(x))
  let gatedX: Model.IO
  if t.0 > 0 {
    gatedX = Functional.concat(
      axis: 0,
      (gateMsaC.reshaped([b, 1, 3_072])
        .* attentionX.reshaped([b, t.0, 3_072], strides: [t.0 * 3_072, 3_072, 1])).reshaped([
          b * t.0, 3_072,
        ]),
      (gateMsaC.reshaped([b, 1, 3_072], offset: [b, 0, 0], strides: [3_072, 3_072, 1])
        .* attentionX.reshaped(
          [b, t.1, 3_072], offset: [0, b * t.0, 0], strides: [t.1 * 3_072, 3_072, 1])).reshaped([
          b * t.1, 3_072,
        ]))
  } else {
    gatedX = (gateMsaC.reshaped([b, 1, 3_072]) .* attentionX.reshaped([b, t.1, 3_072])).reshaped([
      b * t.1, 3_072,
    ])
  }
  var out = x + gatedX
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [1], name: "refiner_norm2")
  let mlp0 = Dense(count: 3_072 * 4, name: "refiner_mlp_0")
  let mlp1 = Dense(count: 3_072, name: "refiner_mlp_1")
  let gateMlp = Dense(count: 3_072, name: "refiner_ada_ln_mlp")
  let mlpOut = mlp1(mlp0(norm2(out)).swish())
  let gateMlpC = gateMlp(c)
  let gatedMlpOut: Model.IO
  if t.0 > 0 {
    gatedMlpOut = Functional.concat(
      axis: 0,
      (gateMlpC.reshaped([b, 1, 3_072])
        .* mlpOut.reshaped([b, t.0, 3_072], strides: [t.0 * 3_072, 3_072, 1])).reshaped([
          b * t.0, 3_072,
        ]),
      (gateMlpC.reshaped([b, 1, 3_072], offset: [b, 0, 0], strides: [3_072, 3_072, 1])
        .* mlpOut.reshaped(
          [b, t.1, 3_072], offset: [0, b * t.0, 0], strides: [t.1 * 3_072, 3_072, 1])).reshaped([
          b * t.1, 3_072,
        ]))
  } else {
    gatedMlpOut = (gateMlpC.reshaped([b, 1, 3_072]) .* mlpOut.reshaped([b, t.1, 3_072])).reshaped([
      b * t.1, 3_072,
    ])
  }
  out = out + gatedMlpOut
  let mapper: ModelWeightMapper = { format in
    var mapping = attentionMapper(format)
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm1.bias"] = [norm1.bias.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).norm2.bias"] = [norm2.bias.name]
    mapping["\(prefix).mlp.fc1.weight"] = [mlp0.weight.name]
    mapping["\(prefix).mlp.0.weight"] = mapping["\(prefix).mlp.fc1.weight"]
    mapping["\(prefix).mlp.fc1.bias"] = [mlp0.bias.name]
    mapping["\(prefix).mlp.0.bias"] = mapping["\(prefix).mlp.fc1.bias"]
    mapping["\(prefix).mlp.fc2.weight"] = [mlp1.weight.name]
    mapping["\(prefix).mlp.2.weight"] = mapping["\(prefix).mlp.fc2.weight"]
    mapping["\(prefix).mlp.fc2.bias"] = [mlp1.bias.name]
    mapping["\(prefix).mlp.2.bias"] = mapping["\(prefix).mlp.fc2.bias"]
    mapping["\(prefix).adaLN_modulation.1.weight"] = [gateMsa.weight.name, gateMlp.weight.name]
    mapping["\(prefix).adaLN_modulation.1.bias"] = [gateMsa.bias.name, gateMlp.bias.name]
    return mapping
  }
  return (Model([x, c], [out]), mapper)
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, upcast: Bool, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let linear1 = Dense(count: intermediateSize, flags: [.Float16], name: "\(name)_linear1")
  var out = linear1(x).GELU(approximate: .tanh)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = Dense(count: hiddenSize, name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool, upcast: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let xChunks = (0..<6).map { _ in Input() }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context).to(.Float16) + contextChunks[0]
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
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
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
  xQ = Functional.cmul(left: xQ, right: rot)
  xK = Functional.cmul(left: xK, right: rot)
  var keys = Functional.concat(axis: 1, xK, contextK)
  var values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
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
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged, .quantized:
    out = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])(
        queries, keys, values
      ).reshaped([b, t + hw, h * k])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], offset: [0, hw, 0], strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: x)
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + ((upcast ? contextChunks[5].to(of: contextOut) : contextChunks[5])
      .* contextFF(
        contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + ((upcast ? xChunks[5].to(of: xOut) : xChunks[5])
    .* xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])).to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).txt_attn_qkv.weight"] = [
      contextToQueries.weight.name, contextToKeys.weight.name, contextToValues.weight.name,
    ]
    mapping["\(prefix).txt_attn.qkv.weight"] = mapping["\(prefix).txt_attn_qkv.weight"]
    mapping["\(prefix).txt_attn_qkv.bias"] = [
      contextToQueries.bias.name, contextToKeys.bias.name, contextToValues.bias.name,
    ]
    mapping["\(prefix).txt_attn.qkv.bias"] = mapping["\(prefix).txt_attn_qkv.bias"]
    mapping["\(prefix).txt_attn_k_norm.weight"] = [normAddedK.weight.name]
    mapping["\(prefix).txt_attn_q_norm.weight"] = [normAddedQ.weight.name]
    mapping["\(prefix).img_attn_qkv.weight"] = [
      xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
    ]
    mapping["\(prefix).img_attn.qkv.weight"] = mapping["\(prefix).img_attn_qkv.weight"]
    mapping["\(prefix).img_attn_qkv.bias"] = [
      xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name,
    ]
    mapping["\(prefix).img_attn.qkv.bias"] = mapping["\(prefix).img_attn_qkv.bias"]
    mapping["\(prefix).img_attn_k_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).img_attn_q_norm.weight"] = [normQ.weight.name]
    if let contextUnifyheads = contextUnifyheads {
      mapping["\(prefix).txt_attn_proj.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).txt_attn.proj.weight"] = mapping["\(prefix).txt_attn_proj.weight"]
      mapping["\(prefix).txt_attn_proj.bias"] = [contextUnifyheads.bias.name]
      mapping["\(prefix).txt_attn.proj.bias"] = mapping["\(prefix).txt_attn_proj.bias"]
    }
    mapping["\(prefix).img_attn_proj.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).img_attn.proj.weight"] = mapping["\(prefix).img_attn_proj.weight"]
    mapping["\(prefix).img_attn_proj.bias"] = [xUnifyheads.bias.name]
    mapping["\(prefix).img_attn.proj.bias"] = mapping["\(prefix).img_attn_proj.bias"]
    let scaleFactor: Float = upcast ? 8 : 1
    if let contextLinear1 = contextLinear1,
      let contextOutProjection = contextOutProjection
    {
      mapping["\(prefix).txt_mlp.fc1.weight"] = [contextLinear1.weight.name]
      mapping["\(prefix).txt_mlp.0.weight"] = mapping["\(prefix).txt_mlp.fc1.weight"]
      mapping["\(prefix).txt_mlp.fc1.bias"] = [contextLinear1.bias.name]
      mapping["\(prefix).txt_mlp.0.bias"] = mapping["\(prefix).txt_mlp.fc1.bias"]
      mapping["\(prefix).txt_mlp.fc2.weight"] = [contextOutProjection.weight.name]
      mapping["\(prefix).txt_mlp.2.weight"] = mapping["\(prefix).txt_mlp.fc2.weight"]
      mapping["\(prefix).txt_mlp.fc2.bias"] = ModelWeightElement(
        [contextOutProjection.bias.name], scale: (1 / scaleFactor))
      mapping["\(prefix).txt_mlp.2.bias"] = mapping["\(prefix).txt_mlp.fc2.bias"]
    }
    mapping["\(prefix).img_mlp.fc1.weight"] = [xLinear1.weight.name]
    mapping["\(prefix).img_mlp.0.weight"] = mapping["\(prefix).img_mlp.fc1.weight"]
    mapping["\(prefix).img_mlp.fc1.bias"] = [xLinear1.bias.name]
    mapping["\(prefix).img_mlp.0.bias"] = mapping["\(prefix).img_mlp.fc1.bias"]
    mapping["\(prefix).img_mlp.fc2.weight"] = [xOutProjection.weight.name]
    mapping["\(prefix).img_mlp.2.weight"] = mapping["\(prefix).img_mlp.fc2.weight"]
    mapping["\(prefix).img_mlp.fc2.bias"] = ModelWeightElement(
      [xOutProjection.bias.name], scale: (1 / scaleFactor))
    mapping["\(prefix).img_mlp.2.bias"] = mapping["\(prefix).img_mlp.fc2.bias"]
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut, contextOut]))
  } else {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut]))
  }
}

private func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
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
  var queries = Functional.cmul(left: xQ, right: rot)
  var keys = Functional.cmul(left: xK, right: rot)
  var values = xV
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
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xLinear1 = Dense(count: k * h * 4, name: "x_linear1")
  let xOutProjection = Dense(count: k * h, name: "x_out_proj")
  out = xUnifyheads(out) + xOutProjection(xLinear1(xOut).GELU(approximate: .tanh))
  out = xIn + xChunks[2].to(of: xIn) .* out.to(of: xIn)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).linear1.weight"] = ModelWeightElement(
      [xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xLinear1.weight.name],
      offsets: [0, k * h, k * h * 2, k * h * 3])
    mapping["\(prefix).linear1.bias"] = ModelWeightElement(
      [xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name, xLinear1.bias.name],
      offsets: [0, k * h, k * h * 2, k * h * 3])
    mapping["\(prefix).k_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).q_norm.weight"] = [normQ.weight.name]
    mapping["\(prefix).linear2.weight"] = ModelWeightElement(
      [xUnifyheads.weight.name, xOutProjection.weight.name], format: .I, offsets: [0, k * h])
    mapping["\(prefix).linear2.bias"] = [xOutProjection.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [out]))
}

public func HunyuanNorm1(time: Int, height: Int, width: Int, channels: Int) -> Model {
  let x = Input()
  let h = height / 2
  let w = width / 2
  let imgIn = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = imgIn(x).reshaped([1, time * h * w, channels]).to(.Float32)
  let xChunks = (0..<2).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = xChunks[1] .* xNorm1(out).to(.Float16) + xChunks[0]
  return Model([x] + xChunks, [out])
}

public func Hunyuan(
  time: Int, height: Int, width: Int, textLength: Int, channels: Int, layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, outputResidual: Bool, inputResidual: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let imgIn = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var adaLNChunks = [Input]()
  var mappers = [ModelWeightMapper]()
  var context: Model.IO?
  var rotAndContextIn: [Input]
  if layers.0 > 0 || layers.1 > 0 {
    let rot = Input()
    let contextIn = Input()
    context = contextIn.to(.Float32)
    rotAndContextIn = [rot, contextIn]
  } else {
    context = nil
    rotAndContextIn = []
  }
  let h = height / 2
  let w = width / 2
  var out = imgIn(x).reshaped([1, time * h * w, channels]).to(.Float32)
  let imgInX = out
  let residualIn: Input?
  if inputResidual {
    let residual = Input()
    residualIn = residual
    out = out + residual
  } else {
    residualIn = nil
  }
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = JointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: channels / 128, b: 1, t: textLength,
      hw: time * h * w,
      contextBlockPreOnly: false, upcast: true, usesFlashAttention: usesFlashAttention)
    let blockOut = block([out, context!, rotAndContextIn[0]] + contextChunks + xChunks)
    out = blockOut[0]
    context = blockOut[1]
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  if let context = context {
    rotAndContextIn.insert(Input(), at: 1)
    out = Functional.concat(axis: 1, out, context)
  }
  for i in 0..<layers.1 {
    let xChunks = (0..<3).map { _ in Input() }
    let (mapper, block) = SingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: channels / 128, b: 1, t: textLength,
      hw: time * h * w,
      contextBlockPreOnly: i == layers.1 - 1, usesFlashAttention: usesFlashAttention)
    out = block([out, rotAndContextIn[1]] + xChunks)
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
  }
  let residualOut: Model.IO?
  if outputResidual {
    residualOut = out - imgInX
  } else {
    residualOut = nil
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = scale .* normFinal(out).to(.Float16) + shift
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = projOut(out).reshaped([time, h, w, 16, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous()
    .reshaped([
      time, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["img_in.proj.weight"] = [imgIn.weight.name]
    mapping["img_in.proj.bias"] = [imgIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.linear.weight"] = [projOut.weight.name]
    mapping["final_layer.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x] + (residualIn.map { [$0] } ?? []) + rotAndContextIn + adaLNChunks,
      [out] + (residualOut.map { [$0] } ?? []))
  )
}

private func JointTransformerBlockFixed(
  prefix: String, k: Int, h: Int, contextBlockPreOnly: Bool
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
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).txt_mod.linear.weight"] = ModelWeightElement(
      (0..<(contextBlockPreOnly ? 2 : 6)).map {
        contextAdaLNs[$0].weight.name
      })
    mapping["\(prefix).txt_mod.lin.weight"] = mapping["\(prefix).txt_mod.linear.weight"]
    mapping["\(prefix).txt_mod.linear.bias"] = ModelWeightElement(
      (0..<(contextBlockPreOnly ? 2 : 6)).map {
        contextAdaLNs[$0].bias.name
      })
    mapping["\(prefix).txt_mod.lin.bias"] = mapping["\(prefix).txt_mod.linear.bias"]
    mapping["\(prefix).img_mod.linear.weight"] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].weight.name })
    mapping["\(prefix).img_mod.lin.weight"] = mapping["\(prefix).img_mod.linear.weight"]
    mapping["\(prefix).img_mod.linear.bias"] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].bias.name })
    mapping["\(prefix).img_mod.lin.bias"] = mapping["\(prefix).img_mod.linear.bias"]
    return mapping
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

private func SingleTransformerBlockFixed(
  prefix: String, k: Int, h: Int
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let xAdaLNs = (0..<3).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).modulation.linear.weight"] = ModelWeightElement(
      (0..<3).map { xAdaLNs[$0].weight.name })
    mapping["\(prefix).modulation.lin.weight"] = mapping["\(prefix).modulation.linear.weight"]
    mapping["\(prefix).modulation.linear.bias"] = ModelWeightElement(
      (0..<3).map { xAdaLNs[$0].bias.name })
    mapping["\(prefix).modulation.lin.bias"] = mapping["\(prefix).modulation.linear.bias"]
    return mapping
  }
  return (mapper, Model([c], xChunks))
}

public func HunyuanFixed(timesteps: Int, channels: Int, layers: (Int, Int), textLength: (Int, Int))
  -> (
    ModelWeightMapper, Model
  )
{
  let txt = Input()
  let t = Input()
  let vector = Input()
  let guidanceEmbed = Input()
  let (tMlp0, tMlp2, timeEmbedder) = MLPEmbedder(channels: channels, name: "txt_in_t")
  var c: Model.IO
  if textLength.0 > 0 {
    c = Functional.concat(
      axis: 0, txt.reshaped([timesteps, textLength.0, 4096]).reduced(.mean, axis: [1]),
      txt.reshaped(
        [timesteps, textLength.1, 4096], offset: [0, timesteps * textLength.0, 0],
        strides: [textLength.1 * 4096, 4096, 1]
      ).reduced(.mean, axis: [1])
    ).reshaped([timesteps * 2, 4096])
  } else {
    c = txt.reshaped([timesteps, textLength.1, 4096]).reduced(.mean, axis: [1]).reshaped([
      timesteps, 4096,
    ])
  }
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: channels, name: "c")
  c = timeEmbedder(t) + contextEmbedder(c)
  c = c.swish()
  let inputEmbedder = Dense(count: channels, name: "input_embedder")
  var context = inputEmbedder(txt)
  var mappers = [ModelWeightMapper]()
  for i in 0..<2 {
    let (block, mapper) = IndividualRefinerBlock(
      prefix: "txt_in.individual_token_refiner.blocks.\(i)", b: timesteps, t: textLength)
    context = block(context, c)
    mappers.append(mapper)
  }
  var outs = [Model.IO]()
  outs.append(context)
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(channels: channels, name: "t")
  let (vMlp0, vMlp2, vectorIn) = MLPEmbedder(channels: channels, name: "vector")
  let (gMlp0, gMlp2, guidanceIn) = MLPEmbedder(channels: channels, name: "guidance")
  var vec = timeIn(t) + vectorIn(vector) + guidanceIn(guidanceEmbed)
  vec = vec.reshaped([textLength.0 > 0 ? timesteps * 2 : timesteps, 1, channels]).swish()
  for i in 0..<layers.0 {
    let (mapper, block) = JointTransformerBlockFixed(
      prefix: "double_blocks.\(i)", k: 128, h: channels / 128, contextBlockPreOnly: false)
    let blockOut = block(vec)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlockFixed(
      prefix: "single_blocks.\(i)", k: 128, h: channels / 128)
    let blockOut = block(vec)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let scale = Dense(count: channels, name: "ada_ln_0")
  let shift = Dense(count: channels, name: "ada_ln_1")
  outs.append(contentsOf: [shift(vec), 1 + scale(vec)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["txt_in.input_embedder.weight"] = [inputEmbedder.weight.name]
    mapping["txt_in.input_embedder.bias"] = [inputEmbedder.bias.name]
    mapping["txt_in.t_embedder.mlp.0.weight"] = [tMlp0.weight.name]
    mapping["txt_in.t_embedder.in_layer.weight"] = mapping["txt_in.t_embedder.mlp.0.weight"]
    mapping["txt_in.t_embedder.mlp.0.bias"] = [tMlp0.bias.name]
    mapping["txt_in.t_embedder.in_layer.bias"] = mapping["txt_in.t_embedder.mlp.0.bias"]
    mapping["txt_in.t_embedder.mlp.2.weight"] = [tMlp2.weight.name]
    mapping["txt_in.t_embedder.out_layer.weight"] = mapping["txt_in.t_embedder.mlp.2.weight"]
    mapping["txt_in.t_embedder.mlp.2.bias"] = [tMlp2.bias.name]
    mapping["txt_in.t_embedder.out_layer.bias"] = mapping["txt_in.t_embedder.mlp.2.bias"]
    mapping["txt_in.c_embedder.linear_1.weight"] = [cLinear1.weight.name]
    mapping["txt_in.c_embedder.in_layer.weight"] = mapping["txt_in.c_embedder.linear_1.weight"]
    mapping["txt_in.c_embedder.linear_1.bias"] = [cLinear1.bias.name]
    mapping["txt_in.c_embedder.in_layer.bias"] = mapping["txt_in.c_embedder.linear_1.bias"]
    mapping["txt_in.c_embedder.linear_2.weight"] = [cLinear2.weight.name]
    mapping["txt_in.c_embedder.out_layer.weight"] = mapping["txt_in.c_embedder.linear_2.weight"]
    mapping["txt_in.c_embedder.linear_2.bias"] = [cLinear2.bias.name]
    mapping["txt_in.c_embedder.out_layer.bias"] = mapping["txt_in.c_embedder.linear_2.bias"]
    mapping["time_in.mlp.0.weight"] = [timeInMlp0.weight.name]
    mapping["time_in.in_layer.weight"] = mapping["time_in.mlp.0.weight"]
    mapping["time_in.mlp.0.bias"] = [timeInMlp0.bias.name]
    mapping["time_in.in_layer.bias"] = mapping["time_in.mlp.0.bias"]
    mapping["time_in.mlp.2.weight"] = [timeInMlp2.weight.name]
    mapping["time_in.out_layer.weight"] = mapping["time_in.mlp.2.weight"]
    mapping["time_in.mlp.2.bias"] = [timeInMlp2.bias.name]
    mapping["time_in.out_layer.bias"] = mapping["time_in.mlp.2.bias"]
    mapping["vector_in.in_layer.weight"] = [vMlp0.weight.name]
    mapping["vector_in.in_layer.bias"] = [vMlp0.bias.name]
    mapping["vector_in.out_layer.weight"] = [vMlp2.weight.name]
    mapping["vector_in.out_layer.bias"] = [vMlp2.bias.name]
    mapping["guidance_in.mlp.0.weight"] = [gMlp0.weight.name]
    mapping["guidance_in.in_layer.weight"] = mapping["guidance_in.mlp.0.weight"]
    mapping["guidance_in.mlp.0.bias"] = [gMlp0.bias.name]
    mapping["guidance_in.in_layer.bias"] = mapping["guidance_in.mlp.0.bias"]
    mapping["guidance_in.mlp.2.weight"] = [gMlp2.weight.name]
    mapping["guidance_in.out_layer.weight"] = mapping["guidance_in.mlp.2.weight"]
    mapping["guidance_in.mlp.2.bias"] = [gMlp2.bias.name]
    mapping["guidance_in.out_layer.bias"] = mapping["guidance_in.mlp.2.bias"]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    mapping["final_layer.adaLN_modulation.1.bias"] = [shift.bias.name, scale.bias.name]
    return mapping
  }
  return (mapper, Model([txt, t, vector, guidanceEmbed], outs))
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

public func HunyuanFixedOutputShapes(
  batchSize: Int, channels: Int, layers: (Int, Int), textLength: Int
) -> [TensorShape] {
  var outs = [TensorShape]()
  outs.append(TensorShape([batchSize, textLength, channels]))
  for i in 0..<layers.0 {
    let contextBlockPreOnly = i == layers.0 - 1 && layers.1 == 0
    let outputShapes = JointTransformerBlockFixedOutputShapes(
      prefix: "double_blocks.\(i)", batchSize: batchSize, k: 128, h: channels / 128,
      contextBlockPreOnly: contextBlockPreOnly)
    outs.append(contentsOf: outputShapes)
  }
  for i in 0..<layers.1 {
    let outputShapes = SingleTransformerBlockFixedOutputShapes(
      prefix: "single_blocks.\(i)", batchSize: batchSize, k: 128, h: channels / 128)
    outs.append(contentsOf: outputShapes)
  }
  outs.append(contentsOf: [
    TensorShape([batchSize, 1, channels]), TensorShape([batchSize, 1, channels]),
  ])
  return outs
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

private func LoRARefinerSelfAttention(
  prefix: String, k: Int, h: Int, hk: Int, b: Int, t: (Int, Int),
  configuration: LoRANetworkConfiguration
) -> (
  Model, ModelWeightMapper
) {
  let x = Input()
  let tokeys = LoRADense(count: k * hk, configuration: configuration, name: "refiner_k_proj")
  let toqueries = LoRADense(count: k * h, configuration: configuration, name: "refiner_q_proj")
  let tovalues = LoRADense(count: k * hk, configuration: configuration, name: "refiner_v_proj")
  var out: Model.IO
  if t.0 > 0 {
    let keys = tokeys(x)
    let queries = toqueries(x)
    let values = tovalues(x)
    let out0 = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
      queries.reshaped([b, t.0, h, k]), keys.reshaped([b, t.0, hk, k]),
      values.reshaped([b, t.0, hk, k])
    ).reshaped([b * t.0, h * k])
    let out1 = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
      queries.reshaped(
        [b, t.1, h, k], offset: [0, b * t.0, 0, 0], strides: [t.1 * h * k, h * k, k, 1]),
      keys.reshaped(
        [b, t.1, hk, k], offset: [0, b * t.0, 0, 0], strides: [t.1 * hk * k, hk * k, k, 1]),
      values.reshaped(
        [b, t.1, hk, k], offset: [0, b * t.0, 0, 0], strides: [t.1 * hk * k, hk * k, k, 1])
    ).reshaped([b * t.1, h * k])
    out = Functional.concat(axis: 0, out0, out1)
  } else {
    let keys = tokeys(x).reshaped([b, t.1, hk, k])
    let queries = toqueries(x).reshaped([b, t.1, h, k])
    let values = tovalues(x).reshaped([b, t.1, hk, k])
    out = ScaledDotProductAttention(scale: 1.0 / Float(k).squareRoot())(
      queries, keys, values
    ).reshaped([b * t.1, h * k])
  }
  let unifyheads = LoRADense(count: k * h, configuration: configuration, name: "refiner_out_proj")
  out = unifyheads(out)
  let mapper: ModelWeightMapper = { _ in
    // The rotary in Llama is first half and second half, we can be clever and do the extra transpose here to use with cmul.
    var mapping = ModelWeightMapping()
    mapping["\(prefix).self_attn_qkv.weight"] = [
      toqueries.weight.name, tokeys.weight.name, tovalues.weight.name,
    ]
    mapping["\(prefix).self_attn_qkv.bias"] = [
      toqueries.bias.name, tokeys.bias.name, tovalues.bias.name,
    ]
    mapping["\(prefix).self_attn_proj.weight"] = [unifyheads.weight.name]
    mapping["\(prefix).self_attn.proj.weight"] = mapping["\(prefix).self_attn_proj.weight"]
    mapping["\(prefix).self_attn_proj.bias"] = [unifyheads.bias.name]
    mapping["\(prefix).self_attn.proj.bias"] = mapping["\(prefix).self_attn_proj.bias"]
    return mapping
  }
  return (Model([x], [out]), mapper)
}

private func LoRAIndividualRefinerBlock(
  prefix: String, b: Int, t: (Int, Int), configuration: LoRANetworkConfiguration
) -> (Model, ModelWeightMapper) {
  let x = Input()
  let c = Input()
  let norm1 = LayerNorm(epsilon: 1e-6, axis: [1], name: "refiner_norm1")
  let gateMsa = LoRADense(count: 3_072, configuration: configuration, name: "refiner_ada_ln_msa")
  let (attention, attentionMapper) = LoRARefinerSelfAttention(
    prefix: prefix, k: 128, h: 24, hk: 24, b: b, t: t, configuration: configuration)
  let gateMsaC = gateMsa(c)
  let attentionX = attention(norm1(x))
  let gatedX: Model.IO
  if t.0 > 0 {
    gatedX = Functional.concat(
      axis: 0,
      (gateMsaC.reshaped([b, 1, 3_072])
        .* attentionX.reshaped([b, t.0, 3_072], strides: [t.0 * 3_072, 3_072, 1])).reshaped([
          b * t.0, 3_072,
        ]),
      (gateMsaC.reshaped([b, 1, 3_072], offset: [b, 0, 0], strides: [3_072, 3_072, 1])
        .* attentionX.reshaped(
          [b, t.1, 3_072], offset: [0, b * t.0, 0], strides: [t.1 * 3_072, 3_072, 1])).reshaped([
          b * t.1, 3_072,
        ]))
  } else {
    gatedX = (gateMsaC.reshaped([b, 1, 3_072]) .* attentionX.reshaped([b, t.1, 3_072])).reshaped([
      b * t.1, 3_072,
    ])
  }
  var out = x + gatedX
  let norm2 = LayerNorm(epsilon: 1e-6, axis: [1], name: "refiner_norm2")
  let mlp0 = LoRADense(count: 3_072 * 4, configuration: configuration, name: "refiner_mlp_0")
  let mlp1 = LoRADense(count: 3_072, configuration: configuration, name: "refiner_mlp_1")
  let gateMlp = LoRADense(count: 3_072, configuration: configuration, name: "refiner_ada_ln_mlp")
  let mlpOut = mlp1(mlp0(norm2(out)).swish())
  let gateMlpC = gateMlp(c)
  let gatedMlpOut: Model.IO
  if t.0 > 0 {
    gatedMlpOut = Functional.concat(
      axis: 0,
      (gateMlpC.reshaped([b, 1, 3_072])
        .* mlpOut.reshaped([b, t.0, 3_072], strides: [t.0 * 3_072, 3_072, 1])).reshaped([
          b * t.0, 3_072,
        ]),
      (gateMlpC.reshaped([b, 1, 3_072], offset: [b, 0, 0], strides: [3_072, 3_072, 1])
        .* mlpOut.reshaped(
          [b, t.1, 3_072], offset: [0, b * t.0, 0], strides: [t.1 * 3_072, 3_072, 1])).reshaped([
          b * t.1, 3_072,
        ]))
  } else {
    gatedMlpOut = (gateMlpC.reshaped([b, 1, 3_072]) .* mlpOut.reshaped([b, t.1, 3_072])).reshaped([
      b * t.1, 3_072,
    ])
  }
  out = out + gatedMlpOut
  let mapper: ModelWeightMapper = { format in
    var mapping = attentionMapper(format)
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm1.bias"] = [norm1.bias.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).norm2.bias"] = [norm2.bias.name]
    mapping["\(prefix).mlp.fc1.weight"] = [mlp0.weight.name]
    mapping["\(prefix).mlp.0.weight"] = mapping["\(prefix).mlp.fc1.weight"]
    mapping["\(prefix).mlp.fc1.bias"] = [mlp0.bias.name]
    mapping["\(prefix).mlp.0.bias"] = mapping["\(prefix).mlp.fc1.bias"]
    mapping["\(prefix).mlp.fc2.weight"] = [mlp1.weight.name]
    mapping["\(prefix).mlp.2.weight"] = mapping["\(prefix).mlp.fc2.weight"]
    mapping["\(prefix).mlp.fc2.bias"] = [mlp1.bias.name]
    mapping["\(prefix).mlp.2.bias"] = mapping["\(prefix).mlp.fc2.bias"]
    mapping["\(prefix).adaLN_modulation.1.weight"] = [gateMsa.weight.name, gateMlp.weight.name]
    mapping["\(prefix).adaLN_modulation.1.bias"] = [gateMsa.bias.name, gateMlp.bias.name]
    return mapping
  }
  return (Model([x, c], [out]), mapper)
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
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 8
    out = (1 / scaleFactor) * out
  }
  let outProjection = LoRADense(
    count: hiddenSize, configuration: configuration, index: index, name: "\(name)_out_proj")
  out = outProjection(out)
  if upcast {
    let scaleFactor: Float = 8
    out = out.to(.Float32) * scaleFactor
  }
  return (linear1, outProjection, Model([x], [out]))
}

private func LoRAJointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool, upcast: Bool,
  usesFlashAttention: FlashAttentionLevel, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let xChunks = (0..<6).map { _ in Input() }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextChunks[1] .* contextNorm1(context).to(.Float16) + contextChunks[0]
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
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
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
  xQ = Functional.cmul(left: xQ, right: rot)
  xK = Functional.cmul(left: xK, right: rot)
  var keys = Functional.concat(axis: 1, xK, contextK)
  var values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
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
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged, .quantized:
    out = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])(
        queries, keys, values
      ).reshaped([b, t + hw, h * k])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], offset: [0, hw, 0], strides: [(t + hw) * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]).contiguous()
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + (contextChunks[2] .* contextOut).to(of: context)
  }
  xOut = x + (xChunks[2] .* xOut).to(of: x)
  // Attentions are now. Now run MLP.
  let contextLinear1: Model?
  let contextOutProjection: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextLinear1, contextOutProjection, contextFF) = LoRAFeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, configuration: configuration,
      index: layerIndex, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + ((upcast ? contextChunks[5].to(of: contextOut) : contextChunks[5])
      .* contextFF(
        contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])).to(
        of: contextOut)
  } else {
    contextLinear1 = nil
    contextOutProjection = nil
  }
  let (xLinear1, xOutProjection, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 4, upcast: upcast, configuration: configuration,
    index: layerIndex, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  xOut =
    xOut
    + ((upcast ? xChunks[5].to(of: xOut) : xChunks[5])
    .* xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3])).to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).txt_attn_qkv.weight"] = [
      contextToQueries.weight.name, contextToKeys.weight.name, contextToValues.weight.name,
    ]
    mapping["\(prefix).txt_attn.qkv.weight"] = mapping["\(prefix).txt_attn_qkv.weight"]
    mapping["\(prefix).txt_attn_qkv.bias"] = [
      contextToQueries.bias.name, contextToKeys.bias.name, contextToValues.bias.name,
    ]
    mapping["\(prefix).txt_attn.qkv.bias"] = mapping["\(prefix).txt_attn_qkv.bias"]
    mapping["\(prefix).txt_attn_k_norm.weight"] = [normAddedK.weight.name]
    mapping["\(prefix).txt_attn_q_norm.weight"] = [normAddedQ.weight.name]
    mapping["\(prefix).img_attn_qkv.weight"] = [
      xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
    ]
    mapping["\(prefix).img_attn_qkv.weight"] = mapping["\(prefix).img_attn.qkv.weight"]
    mapping["\(prefix).img_attn_qkv.bias"] = [
      xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name,
    ]
    mapping["\(prefix).img_attn.qkv.bias"] = mapping["\(prefix).img_attn_qkv.bias"]
    mapping["\(prefix).img_attn_k_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).img_attn_q_norm.weight"] = [normQ.weight.name]
    if let contextUnifyheads = contextUnifyheads {
      mapping["\(prefix).txt_attn_proj.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).txt_attn.proj.weight"] = mapping["\(prefix).txt_attn_proj.weight"]
      mapping["\(prefix).txt_attn_proj.bias"] = [contextUnifyheads.bias.name]
      mapping["\(prefix).txt_attn.proj.bias"] = mapping["\(prefix).txt_attn_proj.bias"]
    }
    mapping["\(prefix).img_attn_proj.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).img_attn.proj.weight"] = mapping["\(prefix).img_attn_proj.weight"]
    mapping["\(prefix).img_attn_proj.bias"] = [xUnifyheads.bias.name]
    mapping["\(prefix).img_attn.proj.bias"] = mapping["\(prefix).img_attn_proj.bias"]
    let scaleFactor: Float = upcast ? 8 : 1
    if let contextLinear1 = contextLinear1,
      let contextOutProjection = contextOutProjection
    {
      mapping["\(prefix).txt_mlp.fc1.weight"] = [contextLinear1.weight.name]
      mapping["\(prefix).txt_mlp.0.weight"] = mapping["\(prefix).txt_mlp.fc1.weight"]
      mapping["\(prefix).txt_mlp.fc1.bias"] = [contextLinear1.bias.name]
      mapping["\(prefix).txt_mlp.0.bias"] = mapping["\(prefix).txt_mlp.fc1.bias"]
      mapping["\(prefix).txt_mlp.fc2.weight"] = [contextOutProjection.weight.name]
      mapping["\(prefix).txt_mlp.2.weight"] = mapping["\(prefix).txt_mlp.fc2.weight"]
      mapping["\(prefix).txt_mlp.fc2.bias"] = ModelWeightElement(
        [contextOutProjection.bias.name], scale: (1 / scaleFactor))
      mapping["\(prefix).txt_mlp.2.bias"] = mapping["\(prefix).txt_mlp.fc2.bias"]
    }
    mapping["\(prefix).img_mlp.fc1.weight"] = [xLinear1.weight.name]
    mapping["\(prefix).img_mlp.0.weight"] = mapping["\(prefix).img_mlp.fc1.weight"]
    mapping["\(prefix).img_mlp.fc1.bias"] = [xLinear1.bias.name]
    mapping["\(prefix).img_mlp.0.bias"] = mapping["\(prefix).img_mlp.fc1.bias"]
    mapping["\(prefix).img_mlp.fc2.weight"] = [xOutProjection.weight.name]
    mapping["\(prefix).img_mlp.2.weight"] = mapping["\(prefix).img_mlp.fc2.weight"]
    mapping["\(prefix).img_mlp.fc2.bias"] = ModelWeightElement(
      [xOutProjection.bias.name], scale: (1 / scaleFactor))
    mapping["\(prefix).img_mlp.2.bias"] = mapping["\(prefix).img_mlp.fc2.bias"]
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut, contextOut]))
  } else {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut]))
  }
}

private func LoRASingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xChunks[1] .* xNorm1(x).to(.Float16) + xChunks[0]
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
  var queries = Functional.cmul(left: xQ, right: rot)
  var keys = Functional.cmul(left: xK, right: rot)
  var values = xV
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
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, (t + hw), k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xIn = x.reshaped([b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1])
      .contiguous()
    xOut = xOut.reshaped(
      [b, hw, h * k], strides: [(t + hw) * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_o")
  let xLinear1 = LoRADense(
    count: k * h * 4, configuration: configuration, index: layerIndex, name: "x_linear1")
  let xOutProjection = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_out_proj")
  out = xUnifyheads(out) + xOutProjection(xLinear1(xOut).GELU(approximate: .tanh))
  out = xIn + xChunks[2].to(of: xIn) .* out.to(of: xIn)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).linear1.weight"] = ModelWeightElement(
      [xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xLinear1.weight.name],
      offsets: [0, k * h, k * h * 2, k * h * 3])
    mapping["\(prefix).linear1.bias"] = ModelWeightElement(
      [xToQueries.bias.name, xToKeys.bias.name, xToValues.bias.name, xLinear1.bias.name],
      offsets: [0, k * h, k * h * 2, k * h * 3])
    mapping["\(prefix).k_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).q_norm.weight"] = [normQ.weight.name]
    mapping["\(prefix).linear2.weight"] = ModelWeightElement(
      [xUnifyheads.weight.name, xOutProjection.weight.name], format: .I, offsets: [0, k * h])
    mapping["\(prefix).linear2.bias"] = [xOutProjection.bias.name]
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [out]))
}

public func LoRAHunyuanNorm1(
  time: Int, height: Int, width: Int, channels: Int, LoRAConfiguration: LoRANetworkConfiguration
) -> Model {
  let x = Input()
  let h = height / 2
  let w = width / 2
  let imgIn = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out = imgIn(x).reshaped([1, time * h * w, channels]).to(.Float32)
  let xChunks = (0..<2).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = xChunks[1] .* xNorm1(out).to(.Float16) + xChunks[0]
  return Model([x] + xChunks, [out])
}

func LoRAHunyuan(
  time: Int, height: Int, width: Int, textLength: Int, channels: Int, layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, outputResidual: Bool, inputResidual: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let imgIn = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var adaLNChunks = [Input]()
  var mappers = [ModelWeightMapper]()
  var context: Model.IO?
  var rotAndContextIn: [Input]
  if layers.0 > 0 || layers.1 > 0 {
    let rot = Input()
    let contextIn = Input()
    context = contextIn.to(.Float32)
    rotAndContextIn = [rot, contextIn]
  } else {
    context = nil
    rotAndContextIn = []
  }
  let h = height / 2
  let w = width / 2
  var out = imgIn(x).reshaped([1, time * h * w, channels]).to(.Float32)
  let imgInX = out
  let residualIn: Input?
  if inputResidual {
    let residual = Input()
    residualIn = residual
    out = out + residual
  } else {
    residualIn = nil
  }
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let (mapper, block) = LoRAJointTransformerBlock(
      prefix: "double_blocks.\(i)", k: 128, h: channels / 128, b: 1, t: textLength,
      hw: time * h * w, contextBlockPreOnly: false, upcast: true,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    let blockOut = block([out, context!, rotAndContextIn[0]] + contextChunks + xChunks)
    out = blockOut[0]
    context = blockOut[1]
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  if let context = context {
    rotAndContextIn.insert(Input(), at: 1)
    out = Functional.concat(axis: 1, out, context)
  }
  for i in 0..<layers.1 {
    let xChunks = (0..<3).map { _ in Input() }
    let (mapper, block) = LoRASingleTransformerBlock(
      prefix: "single_blocks.\(i)", k: 128, h: channels / 128, b: 1, t: textLength,
      hw: time * h * w, contextBlockPreOnly: i == layers.1 - 1,
      usesFlashAttention: usesFlashAttention, layerIndex: i + layers.0,
      configuration: LoRAConfiguration)
    out = block([out, rotAndContextIn[1]] + xChunks)
    adaLNChunks.append(contentsOf: xChunks)
    mappers.append(mapper)
  }
  let residualOut: Model.IO?
  if outputResidual {
    residualOut = out - imgInX
  } else {
    residualOut = nil
  }
  let shift = Input()
  let scale = Input()
  adaLNChunks.append(contentsOf: [shift, scale])
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = scale .* normFinal(out).to(.Float16) + shift
  let projOut = LoRADense(
    count: 2 * 2 * 16, configuration: LoRAConfiguration, index: 0, name: "linear")
  out = projOut(out).reshaped([time, h, w, 16, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous()
    .reshaped([
      time, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["img_in.proj.weight"] = [imgIn.weight.name]
    mapping["img_in.proj.bias"] = [imgIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.linear.weight"] = [projOut.weight.name]
    mapping["final_layer.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  return (
    mapper,
    Model(
      [x] + (residualIn.map { [$0] } ?? []) + rotAndContextIn + adaLNChunks,
      [out] + (residualOut.map { [$0] } ?? []))
  )
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
    var mapping = ModelWeightMapping()
    mapping["\(prefix).txt_mod.linear.weight"] = ModelWeightElement(
      (0..<(contextBlockPreOnly ? 2 : 6)).map {
        contextAdaLNs[$0].weight.name
      })
    mapping["\(prefix).txt_mod.lin.weight"] = mapping["\(prefix).txt_mod.linear.weight"]
    mapping["\(prefix).txt_mod.linear.bias"] = ModelWeightElement(
      (0..<(contextBlockPreOnly ? 2 : 6)).map {
        contextAdaLNs[$0].bias.name
      })
    mapping["\(prefix).txt_mod.lin.bias"] = mapping["\(prefix).txt_mod.linear.bias"]
    mapping["\(prefix).img_mod.linear.weight"] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].weight.name })
    mapping["\(prefix).img_mod.lin.weight"] = mapping["\(prefix).img_mod.linear.weight"]
    mapping["\(prefix).img_mod.linear.bias"] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].bias.name })
    mapping["\(prefix).img_mod.lin.bias"] = mapping["\(prefix).img_mod.linear.bias"]
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
    var mapping = ModelWeightMapping()
    mapping["\(prefix).modulation.linear.weight"] = ModelWeightElement(
      (0..<3).map { xAdaLNs[$0].weight.name })
    mapping["\(prefix).modulation.lin.weight"] = mapping["\(prefix).modulation.linear.weight"]
    mapping["\(prefix).modulation.linear.bias"] = ModelWeightElement(
      (0..<3).map { xAdaLNs[$0].bias.name })
    mapping["\(prefix).modulation.lin.bias"] = mapping["\(prefix).modulation.linear.bias"]
    return mapping
  }
  return (mapper, Model([c], xChunks))
}

func LoRAHunyuanFixed(
  timesteps: Int, channels: Int, layers: (Int, Int), textLength: (Int, Int),
  LoRAConfiguration: LoRANetworkConfiguration
) -> (
  ModelWeightMapper, Model
) {
  let txt = Input()
  let t = Input()
  let vector = Input()
  let guidanceEmbed = Input()
  let (tMlp0, tMlp2, timeEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "txt_in_t")
  var c: Model.IO
  if textLength.0 > 0 {
    c = Functional.concat(
      axis: 0, txt.reshaped([timesteps, textLength.0, 4096]).reduced(.mean, axis: [1]),
      txt.reshaped(
        [timesteps, textLength.1, 4096], offset: [0, timesteps * textLength.0, 0],
        strides: [textLength.1 * 4096, 4096, 1]
      ).reduced(.mean, axis: [1])
    ).reshaped([timesteps * 2, 4096])
  } else {
    c = txt.reshaped([timesteps, textLength.1, 4096]).reduced(.mean, axis: [1]).reshaped([
      timesteps, 4096,
    ])
  }
  let (cLinear1, cLinear2, contextEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "c")
  c = timeEmbedder(t) + contextEmbedder(c)
  c = c.swish()
  let inputEmbedder = LoRADense(
    count: channels, configuration: LoRAConfiguration, name: "input_embedder")
  var context = inputEmbedder(txt)
  var mappers = [ModelWeightMapper]()
  for i in 0..<2 {
    let (block, mapper) = LoRAIndividualRefinerBlock(
      prefix: "txt_in.individual_token_refiner.blocks.\(i)", b: timesteps, t: textLength,
      configuration: LoRAConfiguration)
    context = block(context, c)
    mappers.append(mapper)
  }
  var outs = [Model.IO]()
  outs.append(context)
  let (timeInMlp0, timeInMlp2, timeIn) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "t")
  let (vMlp0, vMlp2, vectorIn) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "vector")
  let (gMlp0, gMlp2, guidanceIn) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "guidance")
  var vec = timeIn(t) + vectorIn(vector) + guidanceIn(guidanceEmbed)
  vec = vec.reshaped([textLength.0 > 0 ? timesteps * 2 : timesteps, 1, channels]).swish()
  for i in 0..<layers.0 {
    let (mapper, block) = LoRAJointTransformerBlockFixed(
      prefix: "double_blocks.\(i)", k: 128, h: channels / 128, contextBlockPreOnly: false,
      layerIndex: i, configuration: LoRAConfiguration)
    let blockOut = block(vec)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = LoRASingleTransformerBlockFixed(
      prefix: "single_blocks.\(i)", k: 128, h: channels / 128, layerIndex: i + layers.0,
      configuration: LoRAConfiguration)
    let blockOut = block(vec)
    mappers.append(mapper)
    outs.append(blockOut)
  }
  let scale = LoRADense(count: channels, configuration: LoRAConfiguration, name: "ada_ln_0")
  let shift = LoRADense(count: channels, configuration: LoRAConfiguration, name: "ada_ln_1")
  outs.append(contentsOf: [shift(vec), 1 + scale(vec)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["txt_in.input_embedder.weight"] = [inputEmbedder.weight.name]
    mapping["txt_in.input_embedder.bias"] = [inputEmbedder.bias.name]
    mapping["txt_in.t_embedder.mlp.0.weight"] = [tMlp0.weight.name]
    mapping["txt_in.t_embedder.in_layer.weight"] = mapping["txt_in.t_embedder.mlp.0.weight"]
    mapping["txt_in.t_embedder.mlp.0.bias"] = [tMlp0.bias.name]
    mapping["txt_in.t_embedder.in_layer.bias"] = mapping["txt_in.t_embedder.mlp.0.bias"]
    mapping["txt_in.t_embedder.mlp.2.weight"] = [tMlp2.weight.name]
    mapping["txt_in.t_embedder.out_layer.weight"] = mapping["txt_in.t_embedder.mlp.2.weight"]
    mapping["txt_in.t_embedder.mlp.2.bias"] = [tMlp2.bias.name]
    mapping["txt_in.t_embedder.out_layer.bias"] = mapping["txt_in.t_embedder.mlp.2.bias"]
    mapping["txt_in.c_embedder.linear_1.weight"] = [cLinear1.weight.name]
    mapping["txt_in.c_embedder.in_layer.weight"] = mapping["txt_in.c_embedder.linear_1.weight"]
    mapping["txt_in.c_embedder.linear_1.bias"] = [cLinear1.bias.name]
    mapping["txt_in.c_embedder.in_layer.bias"] = mapping["txt_in.c_embedder.linear_1.bias"]
    mapping["txt_in.c_embedder.linear_2.weight"] = [cLinear2.weight.name]
    mapping["txt_in.c_embedder.out_layer.weight"] = mapping["txt_in.c_embedder.linear_2.weight"]
    mapping["txt_in.c_embedder.linear_2.bias"] = [cLinear2.bias.name]
    mapping["txt_in.c_embedder.out_layer.bias"] = mapping["txt_in.c_embedder.linear_2.bias"]
    mapping["time_in.mlp.0.weight"] = [timeInMlp0.weight.name]
    mapping["time_in.in_layer.weight"] = mapping["time_in.mlp.0.weight"]
    mapping["time_in.mlp.0.bias"] = [timeInMlp0.bias.name]
    mapping["time_in.in_layer.bias"] = mapping["time_in.mlp.0.bias"]
    mapping["time_in.mlp.2.weight"] = [timeInMlp2.weight.name]
    mapping["time_in.out_layer.weight"] = mapping["time_in.mlp.2.weight"]
    mapping["time_in.mlp.2.bias"] = [timeInMlp2.bias.name]
    mapping["time_in.out_layer.bias"] = mapping["time_in.mlp.2.bias"]
    mapping["vector_in.in_layer.weight"] = [vMlp0.weight.name]
    mapping["vector_in.in_layer.bias"] = [vMlp0.bias.name]
    mapping["vector_in.out_layer.weight"] = [vMlp2.weight.name]
    mapping["vector_in.out_layer.bias"] = [vMlp2.bias.name]
    mapping["guidance_in.mlp.0.weight"] = [gMlp0.weight.name]
    mapping["guidance_in.in_layer.weight"] = mapping["guidance_in.mlp.0.weight"]
    mapping["guidance_in.mlp.0.bias"] = [gMlp0.bias.name]
    mapping["guidance_in.in_layer.bias"] = mapping["guidance_in.mlp.0.bias"]
    mapping["guidance_in.mlp.2.weight"] = [gMlp2.weight.name]
    mapping["guidance_in.out_layer.weight"] = mapping["guidance_in.mlp.2.weight"]
    mapping["guidance_in.mlp.2.bias"] = [gMlp2.bias.name]
    mapping["guidance_in.out_layer.bias"] = mapping["guidance_in.mlp.2.bias"]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    mapping["final_layer.adaLN_modulation.1.bias"] = [shift.bias.name, scale.bias.name]
    return mapping
  }
  return (mapper, Model([txt, t, vector, guidanceEmbed], outs))
}
