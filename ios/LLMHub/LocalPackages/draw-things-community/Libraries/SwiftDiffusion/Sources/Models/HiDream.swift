import DiffusionMappings
import Foundation
import NNC

public func HiDreamRotaryPositionEmbedding(
  height: Int, width: Int, tokenLength: Int, channels: Int, heads: Int = 1
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, tokenLength + height * width, heads, channels))
  let dim1 = (channels / 8) * 2
  let dim2 = dim1
  let dim0 = channels - dim1 - dim2
  assert(channels % 16 == 0)
  for y in 0..<height {
    for x in 0..<width {
      let i = y * width + x
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
  for i in (height * width)..<(height * width + tokenLength) {
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
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, flags: [.Float16], name: "\(name)_w1")
  let w3 = Dense(count: intermediateSize, noBias: true, flags: [.Float16], name: "\(name)_w3")
  var out = Functional.swishMul(value: w3(x), gate: w1(x))
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 4
    out = (1 / scaleFactor) * out
  }
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_w2")
  out = w2(out)
  if upcast {
    let scaleFactor: Float = 4
    out = out.to(.Float32) * scaleFactor
  } else {
    out = out.to(.Float32)
  }
  return (w1, w2, w3, Model([x], [out]))
}

private func MoEFeedForward(
  segments: Int, tokenLength: Int, hiddenSize: Int, intermediateSize: Int, upcast: Bool,
  name: String
) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let gate = Dense(count: segments, noBias: true, name: "\(name)_gate")
  let route = gate(x).reshaped([tokenLength, 4]).softmax().partitioned(
    kth: 2, axis: 1, descending: true)
  var weights = route[0].reshaped([tokenLength * 2])
  let experts = route[1].reshaped([tokenLength * 2])  // This is to select into experts.
  let sort = experts.sorted(axis: 0, descending: false)
  let sortIndices = sort[1]
  weights = IndexSelect()(weights, sortIndices)  // Reorder the weights by the sorting order.
  let expertIds = sort[0].uniqueConsecutive(count: segments)
  let indices = 0.5 * sortIndices  // Scale it to 0..<tokenLength.
  let gathered = IndexSelect()(x.reshaped([tokenLength, hiddenSize]), indices)
  let w1 = SegmentedDense(
    segments: segments, count: intermediateSize, noBias: true, flags: [.Float16], name: "\(name)_w1"
  )
  let w3 = SegmentedDense(
    segments: segments, count: intermediateSize, noBias: true, flags: [.Float16], name: "\(name)_w3"
  )
  var out = w1(gathered, expertIds).swish() .* w3(gathered, expertIds)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 4
    out = (1 / scaleFactor) * out
  }
  let w2 = SegmentedDense(segments: segments, count: hiddenSize, noBias: true, name: "\(name)_w2")
  out = w2(out, expertIds)
  // Out is tokenLength * 2, now multiply weights and scale back.
  out = out .* weights.reshaped([tokenLength * 2, 1])
  out = Functional.scatterAdd(count: tokenLength, out, index: indices)
  if upcast {
    let scaleFactor: Float = 4
    out = out.to(.Float32) * scaleFactor
  } else {
    out = out.to(.Float32)
  }
  return (gate, w1, w2, w3, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let xChunks = (0..<6).map { _ in Input() }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextNorm1(context).to(.Float16) .* contextChunks[1] + contextChunks[0]
  let contextToKeys = Dense(count: k * h, name: "c_k")
  let contextToQueries = Dense(count: k * h, flags: [.Float16], name: "c_q")
  let contextToValues = Dense(count: k * h, name: "c_v")
  var contextK = contextToKeys(contextOut)
  let normAddedK = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_k")
  contextK = normAddedK(contextK).reshaped([b, t.1, h, k])
  var contextQ = contextToQueries(contextOut)
  let normAddedQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_q")
  contextQ = normAddedQ(contextQ).reshaped([b, t.1, h, k])
  let contextV = contextToValues(contextOut).reshaped([b, t.1, h, k])
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, flags: [.Float16], name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  var values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  // Reshape queries because llama3 encoder doesn't participate query, just serve as kv.
  queries = Functional.cmul(left: queries, right: rot).reshaped(
    [b, t.0 + hw, h, k], strides: [(t.1 + hw) * h * k, h * k, k, 1]
  ).contiguous()
  keys = Functional.cmul(left: keys, right: rot)
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1.0 / Float(k).squareRoot().squareRoot()) * keys
  }
  // Now run attention.
  let out: Model.IO
  if usesFlashAttention != .none {
    out = ScaledDotProductAttention(
      scale: usesFlashAttention == .scale1 ? 1 : (1.0 / Float(k).squareRoot()),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
    )(
      queries, keys, values
    ).reshaped([b, t.0 + hw, h * k])
  } else {
    keys = keys.transposed(1, 2)
    queries = queries.transposed(1, 2)
    values = values.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<(b * h) {
      let key = keys.reshaped(
        [1, t.1 + hw, k], offset: [i, 0, 0], strides: [(t.1 + hw) * k, k, 1])
      let query = queries.reshaped(
        [1, t.0 + hw, k], offset: [i, 0, 0], strides: [(t.0 + hw) * k, k, 1])
      let value = values.reshaped(
        [1, t.1 + hw, k], offset: [i, 0, 0], strides: [(t.1 + hw) * k, k, 1])
      var dot = Matmul(transposeB: (1, 2))(query, key)
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot.reshaped([t.0 + hw, t.1 + hw])
      dot = dot.softmax()
      dot = dot.reshaped([1, t.0 + hw, t.1 + hw])
      outs.append(dot * value)
    }
    out = Concat(axis: 0)(outs).reshaped([b, h, t.0 + hw, k]).transposed(1, 2).reshaped([
      b, t.0 + hw, h * k,
    ])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t.0, h * k], offset: [0, hw, 0], strides: [(t.0 + hw) * h * k, h * k, 1]
    )
    let unifyheads = Dense(count: k * h, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t.0 + hw) * h * k, h * k, 1])
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut =
      context.reshaped([b, t.0, h * k], strides: [t.1 * h * k, h * k, 1]).contiguous()
      + (contextOut .* contextChunks[2]).to(of: context)
  }
  xOut = x + (xOut .* xChunks[2]).to(of: x)
  // Attentions are now. Now run MLP.
  let contextW1: Model?
  let contextW2: Model?
  let contextW3: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextW1, contextW2, contextW3, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: 6_912, upcast: upcast, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + contextFF(
        contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
      .* contextChunks[5].to(of: contextOut)
  } else {
    contextW1 = nil
    contextW2 = nil
    contextW3 = nil
  }
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: 3_584, upcast: upcast, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = MoEFeedForward(
    segments: 4, tokenLength: b * hw, hiddenSize: k * h, intermediateSize: 6_912, upcast: upcast,
    name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xIn = xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]
  xOut =
    xOut
    + (xSharedFF(xIn) + xMoEFF(xIn).reshaped([b, hw, h * k])) .* xChunks[5].to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn1.to_q_t.weight"] = [contextToQueries.weight.name]
    mapping["\(prefix).attn1.to_q_t.bias"] = [contextToQueries.bias.name]
    mapping["\(prefix).attn1.to_k_t.weight"] = [contextToKeys.weight.name]
    mapping["\(prefix).attn1.to_k_t.bias"] = [contextToKeys.bias.name]
    mapping["\(prefix).attn1.to_v_t.weight"] = [contextToValues.weight.name]
    mapping["\(prefix).attn1.to_v_t.bias"] = [contextToValues.bias.name]
    mapping["\(prefix).attn1.k_rms_norm_t.weight"] = [normAddedK.weight.name]
    mapping["\(prefix).attn1.q_rms_norm_t.weight"] = [normAddedQ.weight.name]
    mapping["\(prefix).attn1.to_q.weight"] = [xToQueries.weight.name]
    mapping["\(prefix).attn1.to_q.bias"] = [xToQueries.bias.name]
    mapping["\(prefix).attn1.to_k.weight"] = [xToKeys.weight.name]
    mapping["\(prefix).attn1.to_k.bias"] = [xToKeys.bias.name]
    mapping["\(prefix).attn1.to_v.weight"] = [xToValues.weight.name]
    mapping["\(prefix).attn1.to_v.bias"] = [xToValues.bias.name]
    mapping["\(prefix).attn1.k_rms_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).attn1.q_rms_norm.weight"] = [normQ.weight.name]
    if let contextUnifyheads = contextUnifyheads {
      mapping["\(prefix).attn1.to_out_t.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).attn1.to_out_t.bias"] = [contextUnifyheads.bias.name]
    }
    mapping["\(prefix).attn1.to_out.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).attn1.to_out.bias"] = [xUnifyheads.bias.name]
    if let contextW1 = contextW1, let contextW2 = contextW2, let contextW3 = contextW3 {
      mapping["\(prefix).ff_t.w1.weight"] = [contextW1.weight.name]
      mapping["\(prefix).ff_t.w2.weight"] = [contextW2.weight.name]
      mapping["\(prefix).ff_t.w3.weight"] = [contextW3.weight.name]
    }
    mapping["\(prefix).ff_i.shared_experts.w1.weight"] = [xSharedW1.weight.name]
    mapping["\(prefix).ff_i.shared_experts.w2.weight"] = [xSharedW2.weight.name]
    mapping["\(prefix).ff_i.shared_experts.w3.weight"] = [xSharedW3.weight.name]
    mapping["\(prefix).ff_i.gate.weight"] = [xMoEGate.weight.name]
    // We will build reverse look-up durign importing for this to work, hence the index is required to recover the order.
    mapping["\(prefix).ff_i.experts.0.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 3)
    mapping["\(prefix).ff_i.experts.0.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 3)
    mapping["\(prefix).ff_i.experts.0.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 3)
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut, contextOut]))
  } else {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut]))
  }
}

private func SingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = Dense(count: k * h, name: "x_k")
  let xToQueries = Dense(count: k * h, flags: [.Float16], name: "x_q")
  let xToValues = Dense(count: k * h, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw + t.1, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw + t.1, h, k])
  var xV = xToValues(xOut).reshaped([b, hw + t.1, h, k])
  // Reshape queries because llama3 encoder doesn't participate query, just serve as kv.
  let xLength = contextBlockPreOnly ? hw : hw + t.0
  xQ = Functional.cmul(left: xQ, right: rot).reshaped(
    [b, xLength, h, k], strides: [(t.1 + hw) * h * k, h * k, k, 1]
  ).contiguous()
  xK = Functional.cmul(left: xK, right: rot)
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    xQ = (1.0 / Float(k).squareRoot().squareRoot()) * xQ
    xK = (1.0 / Float(k).squareRoot().squareRoot()) * xK
  }
  // Now run attention.
  if usesFlashAttention != .none {
    xOut = ScaledDotProductAttention(
      scale: usesFlashAttention == .scale1 ? 1 : (1.0 / Float(k).squareRoot()),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
    )(
      xQ, xK, xV
    ).reshaped([b, xLength, h * k])
  } else {
    xK = xK.transposed(1, 2)
    xQ = xQ.transposed(1, 2)
    xV = xV.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<(b * h) {
      let key = xK.reshaped([1, t.1 + hw, k], offset: [i, 0, 0], strides: [(t.1 + hw) * k, k, 1])
      let query = xQ.reshaped(
        [1, xLength, k], offset: [i, 0, 0], strides: [xLength * k, k, 1])
      let value = xV.reshaped(
        [1, t.1 + hw, k], offset: [i, 0, 0], strides: [(t.1 + hw) * k, k, 1])
      var dot = Matmul(transposeB: (1, 2))(query, key)
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot.reshaped([xLength, t.1 + hw])
      dot = dot.softmax()
      dot = dot.reshaped([1, xLength, t.1 + hw])
      outs.append(dot * value)
    }
    xOut = Concat(axis: 0)(outs).reshaped([b, h, xLength, k]).transposed(1, 2).reshaped([
      b, xLength, h * k,
    ])
  }
  let xIn = x.reshaped([b, xLength, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
  let xUnifyheads = Dense(count: k * h, name: "x_o")
  xOut = xUnifyheads(xOut)
  xOut = xIn + (xOut .* xChunks[2]).to(of: xIn)
  // Attentions are now. Now run MLP.
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: 3_584, upcast: upcast, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = MoEFeedForward(
    segments: 4, tokenLength: b * xLength, hiddenSize: k * h, intermediateSize: 6_912,
    upcast: upcast,
    name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xFFIn = xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]
  xOut =
    xOut
    + (xSharedFF(xFFIn) + xMoEFF(xFFIn).reshaped([b, xLength, h * k])) .* xChunks[5].to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn1.to_q.weight"] = [xToQueries.weight.name]
    mapping["\(prefix).attn1.to_q.bias"] = [xToQueries.bias.name]
    mapping["\(prefix).attn1.to_k.weight"] = [xToKeys.weight.name]
    mapping["\(prefix).attn1.to_k.bias"] = [xToKeys.bias.name]
    mapping["\(prefix).attn1.to_v.weight"] = [xToValues.weight.name]
    mapping["\(prefix).attn1.to_v.bias"] = [xToValues.bias.name]
    mapping["\(prefix).attn1.k_rms_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).attn1.q_rms_norm.weight"] = [normQ.weight.name]
    mapping["\(prefix).attn1.to_out.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).attn1.to_out.bias"] = [xUnifyheads.bias.name]
    mapping["\(prefix).ff_i.shared_experts.w1.weight"] = [xSharedW1.weight.name]
    mapping["\(prefix).ff_i.shared_experts.w2.weight"] = [xSharedW2.weight.name]
    mapping["\(prefix).ff_i.shared_experts.w3.weight"] = [xSharedW3.weight.name]
    mapping["\(prefix).ff_i.gate.weight"] = [xMoEGate.weight.name]
    // We will build reverse look-up durign importing for this to work, hence the index is required to recover the order.
    mapping["\(prefix).ff_i.experts.0.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 3)
    mapping["\(prefix).ff_i.experts.0.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 3)
    mapping["\(prefix).ff_i.experts.0.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 3)
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [xOut]))
}

public func HiDream(
  batchSize: Int, height: Int, width: Int, textLength: (Int, Int), layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, outputResidual: Bool, inputResidual: Bool
)
  -> (
    Model, ModelWeightMapper
  )
{
  let x = Input()
  let h = height / 2
  let w = width / 2
  let imgIn = Dense(count: 2_560, name: "x_embedder")
  var out = imgIn(
    x.reshaped([batchSize, h, 2, w, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous()
      .reshaped([batchSize, h * w, 2 * 2 * 16])
  ).to(.Float32)
  let imgInX = out
  let residualIn: Input?
  if inputResidual {
    let residual = Input()
    residualIn = residual
    out = out + residual
  } else {
    residualIn = nil
  }
  let rot: [Input]
  let encoderHiddenStates: [Input]
  var context: Model.IO?
  if layers.0 > 0 || layers.1 > 0 {
    rot = [Input()]
    encoderHiddenStates = (0..<(layers.0 + layers.1 + 1)).map { _ in Input() }
    context = encoderHiddenStates[encoderHiddenStates.count - 1].to(.Float32)
  } else {
    rot = []
    encoderHiddenStates = []
    context = nil
  }
  var mappers = [ModelWeightMapper]()
  var adaLNChunks = [Input]()
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let contextIn = Functional.concat(
      axis: 1, context!, encoderHiddenStates[i].to(.Float32), flags: [.disableOpt])
    let (mapper, block) = JointTransformerBlock(
      prefix: "double_stream_blocks.\(i).block", k: 128, h: 20, b: batchSize,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: false, upcast: i > 12, usesFlashAttention: usesFlashAttention)
    let blockOut = block([out, contextIn, rot[0]] + contextChunks + xChunks)
    out = blockOut[0]
    context = blockOut[1]
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  if let context = context {
    out = Functional.concat(axis: 1, out, context)
  }
  for i in 0..<layers.1 {
    let xChunks = (0..<6).map { _ in Input() }
    let xIn = Functional.concat(
      axis: 1, out, encoderHiddenStates[layers.0 + i].to(.Float32), flags: [.disableOpt])
    let (mapper, block) = SingleTransformerBlock(
      prefix: "single_stream_blocks.\(i).block", k: 128, h: 20, b: batchSize,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: i == layers.1 - 1, upcast: false, usesFlashAttention: usesFlashAttention)
    out = block([xIn, rot[0]] + xChunks)
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
  out = normFinal(out).to(.Float16) .* scale + shift
  let projOut = Dense(count: 2 * 2 * 16, name: "linear")
  out = (-projOut(out)).reshaped([batchSize, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous()
    .reshaped([
      batchSize, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["x_embedder.proj.weight"] = [imgIn.weight.name]
    mapping["x_embedder.proj.bias"] = [imgIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.linear.weight"] = [projOut.weight.name]
    mapping["final_layer.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  var inputs: [Input] = [x] + (residualIn.map { [$0] } ?? []) + rot
  inputs = inputs + encoderHiddenStates + adaLNChunks
  return (
    Model(inputs, [out] + (residualOut.map { [$0] } ?? [])), mapper
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
    mapping[
      "\(prefix).adaLN_modulation.1.weight"
    ] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].weight.name }
        + (0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].weight.name })
    mapping[
      "\(prefix).adaLN_modulation.1.bias"
    ] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].bias.name }
        + (0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].bias.name })
    return mapping
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

private func SingleTransformerBlockFixed(
  prefix: String, k: Int, h: Int
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let xAdaLNs = (0..<6).map { Dense(count: k * h, name: "x_ada_ln_\($0)") }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  xChunks[4] = 1 + xChunks[4]
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping[
      "\(prefix).adaLN_modulation.1.weight"
    ] = ModelWeightElement((0..<6).map { xAdaLNs[$0].weight.name })
    mapping[
      "\(prefix).adaLN_modulation.1.bias"
    ] = ModelWeightElement((0..<6).map { xAdaLNs[$0].bias.name })
    return mapping
  }
  return (mapper, Model([c], xChunks))
}

public func HiDreamFixed(timesteps: Int, layers: (Int, Int), outputTimesteps: Bool) -> (
  Model, ModelWeightMapper
) {
  let t = Input()
  let vector = Input()
  let (tMlp0, tMlp2, timeEmbedder) = MLPEmbedder(channels: 2_560, name: "t")
  let (pMlp0, pMlp2, pooledEmbedder) = MLPEmbedder(channels: 2_560, name: "p")
  let tEmb = timeEmbedder(t)
  var vec = tEmb + pooledEmbedder(vector)
  let t5EncoderHiddenStates = Input()
  let llamaEncoderHiddenStates = (0..<32).map { _ in Input() }
  let captionProjections = (0..<(layers.0 + layers.1 + 1)).map { _ in
    Dense(count: 2_560, noBias: true, name: "caption_projection")
  }
  var encoderHiddenStates = [Model.IO]()
  for i in 0..<(layers.0 + layers.1) {
    encoderHiddenStates.append(
      captionProjections[i](llamaEncoderHiddenStates[min(i, llamaEncoderHiddenStates.count - 1)]))
  }
  let t5HiddenStates = captionProjections[layers.0 + layers.1](t5EncoderHiddenStates)
  encoderHiddenStates.append(
    Functional.concat(
      axis: 1, t5HiddenStates, encoderHiddenStates[encoderHiddenStates.count - 1]
    ))
  vec = vec.reshaped([timesteps, 1, 2_560]).swish()
  var mappers = [ModelWeightMapper]()
  var outs = [Model.IO]()
  for i in 0..<layers.0 {
    let (mapper, block) = JointTransformerBlockFixed(
      prefix: "double_stream_blocks.\(i).block", k: 128, h: 20, contextBlockPreOnly: false)
    mappers.append(mapper)
    let blockOut = block(vec)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlockFixed(
      prefix: "single_stream_blocks.\(i).block", k: 128, h: 20)
    mappers.append(mapper)
    let blockOut = block(vec)
    outs.append(blockOut)
  }
  let scale = Dense(count: 2_560, name: "ada_ln_0")
  let shift = Dense(count: 2_560, name: "ada_ln_1")
  outs.append(contentsOf: [shift(vec), 1 + scale(vec)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["t_embedder.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
    mapping["t_embedder.timestep_embedder.linear_1.bias"] = [tMlp0.bias.name]
    mapping["t_embedder.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
    mapping["t_embedder.timestep_embedder.linear_2.bias"] = [tMlp2.bias.name]
    mapping["p_embedder.pooled_embedder.linear_1.weight"] = [pMlp0.weight.name]
    mapping["p_embedder.pooled_embedder.linear_1.bias"] = [pMlp0.bias.name]
    mapping["p_embedder.pooled_embedder.linear_2.weight"] = [pMlp2.weight.name]
    mapping["p_embedder.pooled_embedder.linear_2.bias"] = [pMlp2.bias.name]
    for i in 0..<(layers.0 + layers.1 + 1) {
      mapping["caption_projection.\(i).linear.weight"] = [captionProjections[i].weight.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    mapping["final_layer.adaLN_modulation.1.bias"] = [shift.bias.name, scale.bias.name]
    return mapping
  }
  return (
    Model(
      [t, vector, t5EncoderHiddenStates] + llamaEncoderHiddenStates,
      encoderHiddenStates + (outputTimesteps ? [tEmb.to(.Float32)] : []) + outs),
    mapper
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
  let xOutputShapes = (0..<6).map { _ in TensorShape([batchSize, 1, k * h]) }
  return xOutputShapes
}

public func HiDreamFixedOutputShapes(timesteps: Int, layers: (Int, Int), textLength: Int)
  -> [TensorShape]
{
  var outs = [TensorShape]()
  var encoderHiddenStates = [TensorShape]()
  for _ in 0..<(layers.0 + layers.1) {
    encoderHiddenStates.append(TensorShape([timesteps, textLength, 2_560]))
  }
  encoderHiddenStates.append(TensorShape([timesteps, textLength * 2, 2_560]))
  for i in 0..<layers.0 {
    let contextBlockPreOnly = i == layers.0 - 1 && layers.1 == 0
    let outputShapes = JointTransformerBlockFixedOutputShapes(
      prefix: "double_stream_blocks.\(i).block", batchSize: timesteps, k: 128, h: 2_560 / 128,
      contextBlockPreOnly: contextBlockPreOnly)
    outs.append(contentsOf: outputShapes)
  }
  for i in 0..<layers.1 {
    let outputShapes = SingleTransformerBlockFixedOutputShapes(
      prefix: "single_stream_blocks.\(i).block", batchSize: timesteps, k: 128, h: 2_560 / 128)
    outs.append(contentsOf: outputShapes)
  }
  outs.append(contentsOf: [
    TensorShape([timesteps, 1, 2_560]), TensorShape([timesteps, 1, 2_560]),
  ])
  return encoderHiddenStates + outs
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
  Model, Model, Model, Model
) {
  let x = Input()
  let w1 = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, flags: [.Float16],
    index: index,
    name: "\(name)_w1")
  let w3 = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, flags: [.Float16],
    index: index,
    name: "\(name)_w3")
  var out = Functional.swishMul(value: w3(x), gate: w1(x))
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 4
    out = (1 / scaleFactor) * out
  }
  let w2 = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: index, name: "\(name)_w2")
  out = w2(out)
  if upcast {
    let scaleFactor: Float = 4
    out = out.to(.Float32) * scaleFactor
  } else {
    out = out.to(.Float32)
  }
  return (w1, w2, w3, Model([x], [out]))
}

private func LoRAMoEFeedForward(
  segments: Int, tokenLength: Int, hiddenSize: Int, intermediateSize: Int, upcast: Bool,
  configuration: LoRANetworkConfiguration, index: Int, name: String
) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let gate = LoRADense(
    count: segments, configuration: configuration, noBias: true, index: index, name: "\(name)_gate")
  let route = gate(x).reshaped([tokenLength, 4]).softmax().partitioned(
    kth: 2, axis: 1, descending: true)
  var weights = route[0].reshaped([tokenLength * 2])
  let experts = route[1].reshaped([tokenLength * 2])  // This is to select into experts.
  let sort = experts.sorted(axis: 0, descending: false)
  let sortIndices = sort[1]
  weights = IndexSelect()(weights, sortIndices)  // Reorder the weights by the sorting order.
  let expertIds = sort[0].uniqueConsecutive(count: segments)
  let indices = 0.5 * sortIndices  // Scale it to 0..<tokenLength.
  let gathered = IndexSelect()(x.reshaped([tokenLength, hiddenSize]), indices)
  let w1 = LoRASegmentedDense(
    segments: segments, count: intermediateSize, configuration: configuration, noBias: true,
    flags: [.Float16],
    index: index, name: "\(name)_w1")
  let w3 = LoRASegmentedDense(
    segments: segments, count: intermediateSize, configuration: configuration, noBias: true,
    flags: [.Float16],
    index: index, name: "\(name)_w3")
  var out = w1(gathered, expertIds).swish() .* w3(gathered, expertIds)
  // The scale down is integrated into out proj bias.
  if upcast {
    let scaleFactor: Float = 4
    out = (1 / scaleFactor) * out
  }
  let w2 = LoRASegmentedDense(
    segments: segments, count: hiddenSize, configuration: configuration, noBias: true, index: index,
    name: "\(name)_w2")
  out = w2(out, expertIds)
  // Out is tokenLength * 2, now multiply weights and scale back.
  out = out .* weights.reshaped([tokenLength * 2, 1])
  out = Functional.scatterAdd(count: tokenLength, out, index: indices)
  if upcast {
    let scaleFactor: Float = 4
    out = out.to(.Float32) * scaleFactor
  } else {
    out = out.to(.Float32)
  }
  return (gate, w1, w2, w3, Model([x], [out]))
}

private func LoRAJointTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in
    Input()
  }
  let xChunks = (0..<6).map { _ in Input() }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextNorm1(context).to(.Float16) .* contextChunks[1] + contextChunks[0]
  let contextToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_k")
  let contextToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "c_q")
  let contextToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "c_v")
  var contextK = contextToKeys(contextOut)
  let normAddedK = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_k")
  contextK = normAddedK(contextK).reshaped([b, t.1, h, k])
  var contextQ = contextToQueries(contextOut)
  let normAddedQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "c_norm_q")
  contextQ = normAddedQ(contextQ).reshaped([b, t.1, h, k])
  let contextV = contextToValues(contextOut).reshaped([b, t.1, h, k])
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  var values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  // Reshape queries because llama3 encoder doesn't participate query, just serve as kv.
  queries = Functional.cmul(left: queries, right: rot).reshaped(
    [b, t.0 + hw, h, k], strides: [(t.1 + hw) * h * k, h * k, k, 1]
  ).contiguous()
  keys = Functional.cmul(left: keys, right: rot)
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1.0 / Float(k).squareRoot().squareRoot()) * keys
  }
  // Now run attention.
  let out: Model.IO
  if usesFlashAttention != .none {
    out = ScaledDotProductAttention(
      scale: usesFlashAttention == .scale1 ? 1 : (1.0 / Float(k).squareRoot()),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
    )(
      queries, keys, values
    ).reshaped([b, t.0 + hw, h * k])
  } else {
    keys = keys.transposed(1, 2)
    queries = queries.transposed(1, 2)
    values = values.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<(b * h) {
      let key = keys.reshaped(
        [1, t.1 + hw, k], offset: [i, 0, 0], strides: [(t.1 + hw) * k, k, 1])
      let query = queries.reshaped(
        [1, t.0 + hw, k], offset: [i, 0, 0], strides: [(t.0 + hw) * k, k, 1])
      let value = values.reshaped(
        [1, t.1 + hw, k], offset: [i, 0, 0], strides: [(t.1 + hw) * k, k, 1])
      var dot = Matmul(transposeB: (1, 2))(query, key)
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot.reshaped([t.0 + hw, t.1 + hw])
      dot = dot.softmax()
      dot = dot.reshaped([1, t.0 + hw, t.1 + hw])
      outs.append(dot * value)
    }
    out = Concat(axis: 0)(outs).reshaped([b, h, t.0 + hw, k]).transposed(1, 2).reshaped([
      b, t.0 + hw, h * k,
    ])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t.0, h * k], offset: [0, hw, 0], strides: [(t.0 + hw) * h * k, h * k, 1]
    )
    let unifyheads = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [(t.0 + hw) * h * k, h * k, 1])
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut =
      context.reshaped([b, t.0, h * k], strides: [t.1 * h * k, h * k, 1]).contiguous()
      + (contextOut .* contextChunks[2]).to(of: context)
  }
  xOut = x + (xOut .* xChunks[2]).to(of: x)
  // Attentions are now. Now run MLP.
  let contextW1: Model?
  let contextW2: Model?
  let contextW3: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextW1, contextW2, contextW3, contextFF) = LoRAFeedForward(
      hiddenSize: k * h, intermediateSize: 6_912, upcast: upcast, configuration: configuration,
      index: layerIndex, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    contextOut =
      contextOut
      + contextFF(
        contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
      .* contextChunks[5].to(of: contextOut)
  } else {
    contextW1 = nil
    contextW2 = nil
    contextW3 = nil
  }
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: 3_584, upcast: upcast, configuration: configuration,
    index: layerIndex, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = LoRAMoEFeedForward(
    segments: 4, tokenLength: b * hw, hiddenSize: k * h, intermediateSize: 6_912, upcast: upcast,
    configuration: configuration, index: layerIndex, name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xIn = xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]
  xOut =
    xOut
    + (xSharedFF(xIn) + xMoEFF(xIn).reshaped([b, hw, h * k])) .* xChunks[5].to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn1.to_q_t.weight"] = [contextToQueries.weight.name]
    mapping["\(prefix).attn1.to_q_t.bias"] = [contextToQueries.bias.name]
    mapping["\(prefix).attn1.to_k_t.weight"] = [contextToKeys.weight.name]
    mapping["\(prefix).attn1.to_k_t.bias"] = [contextToKeys.bias.name]
    mapping["\(prefix).attn1.to_v_t.weight"] = [contextToValues.weight.name]
    mapping["\(prefix).attn1.to_v_t.bias"] = [contextToValues.bias.name]
    mapping["\(prefix).attn1.k_rms_norm_t.weight"] = [normAddedK.weight.name]
    mapping["\(prefix).attn1.q_rms_norm_t.weight"] = [normAddedQ.weight.name]
    mapping["\(prefix).attn1.to_q.weight"] = [xToQueries.weight.name]
    mapping["\(prefix).attn1.to_q.bias"] = [xToQueries.bias.name]
    mapping["\(prefix).attn1.to_k.weight"] = [xToKeys.weight.name]
    mapping["\(prefix).attn1.to_k.bias"] = [xToKeys.bias.name]
    mapping["\(prefix).attn1.to_v.weight"] = [xToValues.weight.name]
    mapping["\(prefix).attn1.to_v.bias"] = [xToValues.bias.name]
    mapping["\(prefix).attn1.k_rms_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).attn1.q_rms_norm.weight"] = [normQ.weight.name]
    if let contextUnifyheads = contextUnifyheads {
      mapping["\(prefix).attn1.to_out_t.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).attn1.to_out_t.bias"] = [contextUnifyheads.bias.name]
    }
    mapping["\(prefix).attn1.to_out.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).attn1.to_out.bias"] = [xUnifyheads.bias.name]
    if let contextW1 = contextW1, let contextW2 = contextW2, let contextW3 = contextW3 {
      mapping["\(prefix).ff_t.w1.weight"] = [contextW1.weight.name]
      mapping["\(prefix).ff_t.w2.weight"] = [contextW2.weight.name]
      mapping["\(prefix).ff_t.w3.weight"] = [contextW3.weight.name]
    }
    mapping["\(prefix).ff_i.shared_experts.w1.weight"] = [xSharedW1.weight.name]
    mapping["\(prefix).ff_i.shared_experts.w2.weight"] = [xSharedW2.weight.name]
    mapping["\(prefix).ff_i.shared_experts.w3.weight"] = [xSharedW3.weight.name]
    mapping["\(prefix).ff_i.gate.weight"] = [xMoEGate.weight.name]
    // We will build reverse look-up durign importing for this to work, hence the index is required to recover the order.
    mapping["\(prefix).ff_i.experts.0.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 3)
    mapping["\(prefix).ff_i.experts.0.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 3)
    mapping["\(prefix).ff_i.experts.0.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 3)
    return mapping
  }
  if !contextBlockPreOnly {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut, contextOut]))
  } else {
    return (mapper, Model([x, context, rot] + contextChunks + xChunks, [xOut]))
  }
}

private func LoRASingleTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, contextBlockPreOnly: Bool,
  upcast: Bool, usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex, name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_k")
  xK = normK(xK).reshaped([b, hw + t.1, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-5, axis: [2], name: "x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw + t.1, h, k])
  var xV = xToValues(xOut).reshaped([b, hw + t.1, h, k])
  // Reshape queries because llama3 encoder doesn't participate query, just serve as kv.
  let xLength = contextBlockPreOnly ? hw : hw + t.0
  xQ = Functional.cmul(left: xQ, right: rot).reshaped(
    [b, xLength, h, k], strides: [(t.1 + hw) * h * k, h * k, k, 1]
  ).contiguous()
  xK = Functional.cmul(left: xK, right: rot)
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    xQ = (1.0 / Float(k).squareRoot().squareRoot()) * xQ
    xK = (1.0 / Float(k).squareRoot().squareRoot()) * xK
  }
  // Now run attention.
  if usesFlashAttention != .none {
    xOut = ScaledDotProductAttention(
      scale: usesFlashAttention == .scale1 ? 1 : (1.0 / Float(k).squareRoot()),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]
    )(
      xQ, xK, xV
    ).reshaped([b, xLength, h * k])
  } else {
    xK = xK.transposed(1, 2)
    xQ = xQ.transposed(1, 2)
    xV = xV.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<(b * h) {
      let key = xK.reshaped([1, t.1 + hw, k], offset: [i, 0, 0], strides: [(t.1 + hw) * k, k, 1])
      let query = xQ.reshaped(
        [1, xLength, k], offset: [i, 0, 0], strides: [xLength * k, k, 1])
      let value = xV.reshaped(
        [1, t.1 + hw, k], offset: [i, 0, 0], strides: [(t.1 + hw) * k, k, 1])
      var dot = Matmul(transposeB: (1, 2))(query, key)
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot.reshaped([xLength, t.1 + hw])
      dot = dot.softmax()
      dot = dot.reshaped([1, xLength, t.1 + hw])
      outs.append(dot * value)
    }
    xOut = Concat(axis: 0)(outs).reshaped([b, h, xLength, k]).transposed(1, 2).reshaped([
      b, xLength, h * k,
    ])
  }
  let xIn = x.reshaped([b, xLength, h * k], strides: [(t.1 + hw) * h * k, h * k, 1]).contiguous()
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "x_o")
  xOut = xUnifyheads(xOut)
  xOut = xIn + (xOut .* xChunks[2]).to(of: xIn)
  // Attentions are now. Now run MLP.
  let (xSharedW1, xSharedW2, xSharedW3, xSharedFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: 3_584, upcast: upcast, configuration: configuration,
    index: layerIndex, name: "x_shared")
  let (xMoEGate, xMoEW1, xMoEW2, xMoEW3, xMoEFF) = LoRAMoEFeedForward(
    segments: 4, tokenLength: b * xLength, hiddenSize: k * h, intermediateSize: 6_912,
    upcast: upcast, configuration: configuration, index: layerIndex, name: "x_moe")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xFFIn = xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]
  xOut =
    xOut
    + (xSharedFF(xFFIn) + xMoEFF(xFFIn).reshaped([b, xLength, h * k])) .* xChunks[5].to(of: xOut)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).attn1.to_q.weight"] = [xToQueries.weight.name]
    mapping["\(prefix).attn1.to_q.bias"] = [xToQueries.bias.name]
    mapping["\(prefix).attn1.to_k.weight"] = [xToKeys.weight.name]
    mapping["\(prefix).attn1.to_k.bias"] = [xToKeys.bias.name]
    mapping["\(prefix).attn1.to_v.weight"] = [xToValues.weight.name]
    mapping["\(prefix).attn1.to_v.bias"] = [xToValues.bias.name]
    mapping["\(prefix).attn1.k_rms_norm.weight"] = [normK.weight.name]
    mapping["\(prefix).attn1.q_rms_norm.weight"] = [normQ.weight.name]
    mapping["\(prefix).attn1.to_out.weight"] = [xUnifyheads.weight.name]
    mapping["\(prefix).attn1.to_out.bias"] = [xUnifyheads.bias.name]
    mapping["\(prefix).ff_i.shared_experts.w1.weight"] = [xSharedW1.weight.name]
    mapping["\(prefix).ff_i.shared_experts.w2.weight"] = [xSharedW2.weight.name]
    mapping["\(prefix).ff_i.shared_experts.w3.weight"] = [xSharedW3.weight.name]
    mapping["\(prefix).ff_i.gate.weight"] = [xMoEGate.weight.name]
    // We will build reverse look-up durign importing for this to work, hence the index is required to recover the order.
    mapping["\(prefix).ff_i.experts.0.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w1.weight"] = ModelWeightElement(
      [xMoEW1.weight.name], index: 3)
    mapping["\(prefix).ff_i.experts.0.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w2.weight"] = ModelWeightElement(
      [xMoEW2.weight.name], index: 3)
    mapping["\(prefix).ff_i.experts.0.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 0)
    mapping["\(prefix).ff_i.experts.1.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 1)
    mapping["\(prefix).ff_i.experts.2.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 2)
    mapping["\(prefix).ff_i.experts.3.w3.weight"] = ModelWeightElement(
      [xMoEW3.weight.name], index: 3)
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [xOut]))
}

public func LoRAHiDream(
  batchSize: Int, height: Int, width: Int, textLength: (Int, Int), layers: (Int, Int),
  usesFlashAttention: FlashAttentionLevel, outputResidual: Bool, inputResidual: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
)
  -> (
    Model, ModelWeightMapper
  )
{
  let x = Input()
  let h = height / 2
  let w = width / 2
  let imgIn = LoRADense(count: 2_560, configuration: LoRAConfiguration, name: "x_embedder")
  var out = imgIn(
    x.reshaped([batchSize, h, 2, w, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous()
      .reshaped([batchSize, h * w, 2 * 2 * 16])
  ).to(.Float32)
  let imgInX = out
  let residualIn: Input?
  if inputResidual {
    let residual = Input()
    residualIn = residual
    out = out + residual
  } else {
    residualIn = nil
  }
  let rot: [Input]
  let encoderHiddenStates: [Input]
  var context: Model.IO?
  if layers.0 > 0 || layers.1 > 0 {
    rot = [Input()]
    encoderHiddenStates = (0..<(layers.0 + layers.1 + 1)).map { _ in Input() }
    context = encoderHiddenStates[encoderHiddenStates.count - 1].to(.Float32)
  } else {
    rot = []
    encoderHiddenStates = []
    context = nil
  }
  var mappers = [ModelWeightMapper]()
  var adaLNChunks = [Input]()
  for i in 0..<layers.0 {
    let contextChunks = (0..<6).map { _ in Input() }
    let xChunks = (0..<6).map { _ in Input() }
    let contextIn = Functional.concat(
      axis: 1, context!, encoderHiddenStates[i].to(.Float32), flags: [.disableOpt])
    let (mapper, block) = LoRAJointTransformerBlock(
      prefix: "double_stream_blocks.\(i).block", k: 128, h: 20, b: batchSize,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: false, upcast: i > 12, usesFlashAttention: usesFlashAttention,
      layerIndex: i, configuration: LoRAConfiguration)
    let blockOut = block([out, contextIn, rot[0]] + contextChunks + xChunks)
    out = blockOut[0]
    context = blockOut[1]
    adaLNChunks.append(contentsOf: contextChunks + xChunks)
    mappers.append(mapper)
  }
  if let context = context {
    out = Functional.concat(axis: 1, out, context)
  }
  for i in 0..<layers.1 {
    let xChunks = (0..<6).map { _ in Input() }
    let xIn = Functional.concat(
      axis: 1, out, encoderHiddenStates[layers.0 + i].to(.Float32), flags: [.disableOpt])
    let (mapper, block) = LoRASingleTransformerBlock(
      prefix: "single_stream_blocks.\(i).block", k: 128, h: 20, b: batchSize,
      t: (textLength.0 + textLength.1, textLength.0 + textLength.1 * 2), hw: h * w,
      contextBlockPreOnly: i == layers.1 - 1, upcast: false, usesFlashAttention: usesFlashAttention,
      layerIndex: i + layers.0, configuration: LoRAConfiguration)
    out = block([xIn, rot[0]] + xChunks)
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
  out = normFinal(out).to(.Float16) .* scale + shift
  let projOut = LoRADense(
    count: 2 * 2 * 16, configuration: LoRAConfiguration, index: 0, name: "linear")
  out = (-projOut(out)).reshaped([batchSize, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous()
    .reshaped([
      batchSize, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["x_embedder.proj.weight"] = [imgIn.weight.name]
    mapping["x_embedder.proj.bias"] = [imgIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.linear.weight"] = [projOut.weight.name]
    mapping["final_layer.linear.bias"] = [projOut.bias.name]
    return mapping
  }
  var inputs: [Input] = [x] + (residualIn.map { [$0] } ?? []) + rot
  inputs = inputs + encoderHiddenStates + adaLNChunks
  return (
    Model(inputs, [out] + (residualOut.map { [$0] } ?? [])), mapper
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
    mapping[
      "\(prefix).adaLN_modulation.1.weight"
    ] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].weight.name }
        + (0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].weight.name })
    mapping[
      "\(prefix).adaLN_modulation.1.bias"
    ] = ModelWeightElement(
      (0..<6).map { xAdaLNs[$0].bias.name }
        + (0..<(contextBlockPreOnly ? 2 : 6)).map { contextAdaLNs[$0].bias.name })
    return mapping
  }
  return (mapper, Model([c], contextChunks + xChunks))
}

private func SingleTransformerBlockFixed(
  prefix: String, k: Int, h: Int, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let c = Input()
  let xAdaLNs = (0..<6).map {
    LoRADense(count: k * h, configuration: configuration, index: layerIndex, name: "x_ada_ln_\($0)")
  }
  var xChunks = xAdaLNs.map { $0(c) }
  xChunks[1] = 1 + xChunks[1]
  xChunks[4] = 1 + xChunks[4]
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping[
      "\(prefix).adaLN_modulation.1.weight"
    ] = ModelWeightElement((0..<6).map { xAdaLNs[$0].weight.name })
    mapping[
      "\(prefix).adaLN_modulation.1.bias"
    ] = ModelWeightElement((0..<6).map { xAdaLNs[$0].bias.name })
    return mapping
  }
  return (mapper, Model([c], xChunks))
}

public func LoRAHiDreamFixed(
  timesteps: Int, layers: (Int, Int), outputTimesteps: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (
  Model, ModelWeightMapper
) {
  let t = Input()
  let vector = Input()
  let (tMlp0, tMlp2, timeEmbedder) = LoRAMLPEmbedder(
    channels: 2_560, configuration: LoRAConfiguration, name: "t")
  let (pMlp0, pMlp2, pooledEmbedder) = LoRAMLPEmbedder(
    channels: 2_560, configuration: LoRAConfiguration, name: "p")
  let tEmb = timeEmbedder(t)
  var vec = tEmb + pooledEmbedder(vector)
  let t5EncoderHiddenStates = Input()
  let llamaEncoderHiddenStates = (0..<32).map { _ in Input() }
  let captionProjections = (0..<(layers.0 + layers.1 + 1)).map {
    LoRADense(
      count: 2_560, configuration: LoRAConfiguration, noBias: true, index: $0,
      name: "caption_projection")
  }
  var encoderHiddenStates = [Model.IO]()
  for i in 0..<(layers.0 + layers.1) {
    encoderHiddenStates.append(
      captionProjections[i](llamaEncoderHiddenStates[min(i, llamaEncoderHiddenStates.count - 1)]))
  }
  let t5HiddenStates = captionProjections[layers.0 + layers.1](t5EncoderHiddenStates)
  encoderHiddenStates.append(
    Functional.concat(
      axis: 1, t5HiddenStates, encoderHiddenStates[encoderHiddenStates.count - 1]
    ))
  vec = vec.reshaped([timesteps, 1, 2_560]).swish()
  var mappers = [ModelWeightMapper]()
  var outs = [Model.IO]()
  for i in 0..<layers.0 {
    let (mapper, block) = LoRAJointTransformerBlockFixed(
      prefix: "double_stream_blocks.\(i).block", k: 128, h: 20, contextBlockPreOnly: false,
      layerIndex: i, configuration: LoRAConfiguration)
    mappers.append(mapper)
    let blockOut = block(vec)
    outs.append(blockOut)
  }
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlockFixed(
      prefix: "single_stream_blocks.\(i).block", k: 128, h: 20, layerIndex: i + layers.0,
      configuration: LoRAConfiguration)
    mappers.append(mapper)
    let blockOut = block(vec)
    outs.append(blockOut)
  }
  let scale = LoRADense(count: 2_560, configuration: LoRAConfiguration, name: "ada_ln_0")
  let shift = LoRADense(count: 2_560, configuration: LoRAConfiguration, name: "ada_ln_1")
  outs.append(contentsOf: [shift(vec), 1 + scale(vec)])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["t_embedder.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
    mapping["t_embedder.timestep_embedder.linear_1.bias"] = [tMlp0.bias.name]
    mapping["t_embedder.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
    mapping["t_embedder.timestep_embedder.linear_2.bias"] = [tMlp2.bias.name]
    mapping["p_embedder.pooled_embedder.linear_1.weight"] = [pMlp0.weight.name]
    mapping["p_embedder.pooled_embedder.linear_1.bias"] = [pMlp0.bias.name]
    mapping["p_embedder.pooled_embedder.linear_2.weight"] = [pMlp2.weight.name]
    mapping["p_embedder.pooled_embedder.linear_2.bias"] = [pMlp2.bias.name]
    for i in 0..<(layers.0 + layers.1 + 1) {
      mapping["caption_projection.\(i).linear.weight"] = [captionProjections[i].weight.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["final_layer.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    mapping["final_layer.adaLN_modulation.1.bias"] = [shift.bias.name, scale.bias.name]
    return mapping
  }
  return (
    Model(
      [t, vector, t5EncoderHiddenStates] + llamaEncoderHiddenStates,
      encoderHiddenStates + (outputTimesteps ? [tEmb.to(.Float32)] : []) + outs),
    mapper
  )
}
