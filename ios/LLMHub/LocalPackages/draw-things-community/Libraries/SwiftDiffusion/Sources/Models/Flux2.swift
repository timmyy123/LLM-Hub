import DiffusionMappings
import Foundation
import NNC

public func Flux2RotaryPositionEmbedding(
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
  let dim0 = channels / 4
  let dim1 = dim0
  let dim2 = dim1
  let dim3 = dim2
  assert(channels % 16 == 0)
  for y in 0..<height {
    for x in 0..<width {
      let i = y * width + x
      for j in 0..<heads {
        for k in 0..<(dim0 / 2) {
          let theta = 0 * 1.0 / pow(2_000, Double(k) * 2 / Double(dim0))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, k * 2] = Float(costheta)
          rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim1 / 2) {
          let theta = Double(y) * 1.0 / pow(2_000, Double(k) * 2 / Double(dim1))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim2 / 2) {
          let theta = Double(x) * 1.0 / pow(2_000, Double(k) * 2 / Double(dim2))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim3 / 2) {
          let theta = 0 * 1.0 / pow(2_000, Double(k) * 2 / Double(dim3))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2) + (dim2 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2) + (dim2 / 2)) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  var index = width * height
  for (n, referenceSize) in referenceSizes.enumerated() {
    let height = referenceSize.height
    let width = referenceSize.width
    for y in 0..<height {
      for x in 0..<width {
        let i = y * width + x + index
        for j in 0..<heads {
          for k in 0..<(dim0 / 2) {
            let theta = Double(10 + n * 10) * 1.0 / pow(2_000, Double(k) * 2 / Double(dim0))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, k * 2] = Float(costheta)
            rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim1 / 2) {
            let theta = Double(y) * 1.0 / pow(2_000, Double(k) * 2 / Double(dim1))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim2 / 2) {
            let theta = Double(x) * 1.0 / pow(2_000, Double(k) * 2 / Double(dim2))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
          }
          for k in 0..<(dim3 / 2) {
            let theta = 0 * 1.0 / pow(2_000, Double(k) * 2 / Double(dim3))
            let sintheta = sin(theta)
            let costheta = cos(theta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2) + (dim2 / 2)) * 2] = Float(costheta)
            rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2) + (dim2 / 2)) * 2 + 1] = Float(sintheta)
          }
        }
      }
    }
    index += height * width
  }
  let tokenOffset = index
  for i in 0..<tokenLength {
    for j in 0..<heads {
      for k in 0..<(dim0 / 2) {
        let theta = 0 * 1.0 / pow(2_000, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i + tokenOffset, j, k * 2] = Float(costheta)
        rotTensor[0, i + tokenOffset, j, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = 0 * 1.0 / pow(2_000, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i + tokenOffset, j, (k + (dim0 / 2)) * 2] = Float(costheta)
        rotTensor[0, i + tokenOffset, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim2 / 2) {
        let theta = 0 * 1.0 / pow(2_000, Double(k) * 2 / Double(dim2))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i + tokenOffset, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
        rotTensor[0, i + tokenOffset, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim3 / 2) {
        let theta = Double(i) * 1.0 / pow(2_000, Double(k) * 2 / Double(dim3))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, i + tokenOffset, j, (k + (dim0 / 2) + (dim1 / 2) + (dim2 / 2)) * 2] = Float(
          costheta)
        rotTensor[0, i + tokenOffset, j, (k + (dim0 / 2) + (dim1 / 2) + (dim2 / 2)) * 2 + 1] =
          Float(sintheta)
      }
    }
  }
  return rotTensor
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, noBias: true, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, noBias: true, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func FeedForward(hiddenSize: Int, intermediateSize: Int, scaleFactor: Float?, name: String)
  -> (
    Model, Model, Model, Model
  )
{
  let x = Input()
  let w1 = Dense(
    count: intermediateSize, noBias: true, flags: [.Float16], name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  var out: Model.IO
  if let scaleFactor = scaleFactor {
    out = w3((1 / scaleFactor) * x)  // x usually is smaller (hiddenSize < intermediateSize).
  } else {
    out = w3(x)
  }
  out = Functional.swishMul(value: out, gate: w1(x))
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out)
  if let _ = scaleFactor {
    out = out.to(.Float32)  // scaleFactor is fused in the other add kernel.
  }
  return (w1, w2, w3, Model([x], [out]))
}

private func JointTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  scaleFactor: Float?, usesFlashAttention: FlashAttentionLevel, cachedSequenceLength: Int = 0,
  kvCache: Bool = false
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let cachedKeys = kvCache ? Input() : nil
  let cachedValues = kvCache ? Input() : nil
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in Input() }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextNorm1(context).to(.Float16) .* contextChunks[1] + contextChunks[0]
  let contextToKeys = Dense(count: k * h, noBias: true, name: "c_k")
  let contextToQueries = Dense(count: k * h, noBias: true, flags: [.Float16], name: "c_q")
  let contextToValues = Dense(count: k * h, noBias: true, name: "c_v")
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
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, flags: [.Float16], name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  var values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  if var cachedKeys: Model.IO = cachedKeys, var cachedValues: Model.IO = cachedValues {
    if b > 1 {
      cachedKeys = Concat(axis: 0)(Array(repeating: cachedKeys, count: b))
      cachedKeys.add(dependencies: [xOut])
      cachedValues = Concat(axis: 0)(Array(repeating: cachedValues, count: b))
      cachedValues.add(dependencies: [xOut])
      keys = Functional.concat(axis: 1, keys, cachedKeys, flags: [.disableOpt])
      values = Functional.concat(axis: 1, values, cachedValues, flags: [.disableOpt])
    } else {
      keys = Functional.concat(axis: 1, keys, cachedKeys)
      values = Functional.concat(axis: 1, values, cachedValues)
    }
  }
  let querySequenceLength = t + hw
  let keyValueSequenceLength = querySequenceLength + cachedSequenceLength
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
        let key = keys.reshaped(
          [1, keyValueSequenceLength, k], offset: [i, 0, 0],
          strides: [keyValueSequenceLength * k, k, 1])
        let query = queries.reshaped(
          [1, querySequenceLength, k], offset: [i, 0, 0],
          strides: [querySequenceLength * k, k, 1])
        let value = values.reshaped(
          [1, keyValueSequenceLength, k], offset: [i, 0, 0],
          strides: [keyValueSequenceLength * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([querySequenceLength, keyValueSequenceLength])
        dot = dot.softmax()
        dot = dot.reshaped([1, querySequenceLength, keyValueSequenceLength])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, querySequenceLength, k]).transposed(1, 2).reshaped([
        b, querySequenceLength, h * k,
      ])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * querySequenceLength, keyValueSequenceLength])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, querySequenceLength, keyValueSequenceLength])
      out = dot * values
      out = out.reshaped([b, h, querySequenceLength, k]).transposed(1, 2).reshaped([
        b, querySequenceLength, h * k,
      ])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, querySequenceLength, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, querySequenceLength, k * h])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], offset: [0, hw, 0], strides: [querySequenceLength * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = Dense(count: k * h, noBias: true, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [querySequenceLength * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + (contextOut .* contextChunks[2]).to(of: context)
  }
  xOut = x + (xOut .* xChunks[2]).to(of: x)
  // Attentions are now. Now run MLP.
  let contextW1: Model?
  let contextW2: Model?
  let contextW3: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextW1, contextW2, contextW3, contextFF) = FeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 3, scaleFactor: scaleFactor, name: "c")
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    if let scaleFactor = scaleFactor {
      contextOut = Add(rightScalar: scaleFactor)(
        contextOut,
        contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
          .* contextChunks[5].to(of: contextOut))
    } else {
      contextOut =
        contextOut
        + (contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
        .* contextChunks[5])
        .to(of: contextOut)
    }
  } else {
    contextW1 = nil
    contextW2 = nil
    contextW3 = nil
  }
  let (xW1, xW2, xW3, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 3, scaleFactor: scaleFactor, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if let scaleFactor = scaleFactor {
    xOut = Add(rightScalar: scaleFactor)(
      xOut, xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]) .* xChunks[5].to(of: xOut))
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
      mapping["\(prefix.0).txt_attn.norm.key_norm.scale"] = [normAddedK.weight.name]
      mapping["\(prefix.0).txt_attn.norm.query_norm.scale"] = [normAddedQ.weight.name]
      mapping["\(prefix.0).img_attn.qkv.weight"] = [
        xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
      ]
      mapping["\(prefix.0).img_attn.norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).img_attn.norm.query_norm.scale"] = [normQ.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.0).txt_attn.proj.weight"] = [contextUnifyheads.weight.name]
      }
      mapping["\(prefix.0).img_attn.proj.weight"] = [xUnifyheads.weight.name]
      if let contextW1 = contextW1, let contextW2 = contextW2, let contextW3 = contextW3 {
        mapping["\(prefix.0).txt_mlp.0.weight"] = [contextW1.weight.name, contextW3.weight.name]
        mapping["\(prefix.0).txt_mlp.2.weight"] = [contextW2.weight.name]
      }
      mapping["\(prefix.0).img_mlp.0.weight"] = [xW1.weight.name, xW3.weight.name]
      mapping["\(prefix.0).img_mlp.2.weight"] = [xW2.weight.name]
    case .diffusers:
      mapping["\(prefix.1).attn.add_q_proj.weight"] = [contextToQueries.weight.name]
      mapping["\(prefix.1).attn.add_k_proj.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix.1).attn.add_v_proj.weight"] = [contextToValues.weight.name]
      mapping["\(prefix.1).attn.norm_added_k.weight"] = [normAddedK.weight.name]
      mapping["\(prefix.1).attn.norm_added_q.weight"] = [normAddedQ.weight.name]
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.1).attn.to_add_out.weight"] = [contextUnifyheads.weight.name]
      }
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      if let contextW1 = contextW1, let contextW2 = contextW2, let contextW3 = contextW3 {
        mapping["\(prefix.1).ff_context.linear_in.weight"] = [
          contextW1.weight.name, contextW3.weight.name,
        ]
        mapping["\(prefix.1).ff_context.linear_out.weight"] = [contextW2.weight.name]
      }
      mapping["\(prefix.1).ff.linear_in.weight"] = [xW1.weight.name, xW3.weight.name]
      mapping["\(prefix.1).ff.linear_out.weight"] = [xW2.weight.name]
    }
    return mapping
  }
  if !contextBlockPreOnly {
    return (
      mapper,
      Model(
        [context, x, rot] + (cachedKeys.map { [$0] } ?? []) + (cachedValues.map { [$0] } ?? [])
          + contextChunks + xChunks, [contextOut, xOut])
    )
  } else {
    return (
      mapper,
      Model(
        [context, x, rot] + (cachedKeys.map { [$0] } ?? []) + (cachedValues.map { [$0] } ?? [])
          + contextChunks + xChunks, [xOut])
    )
  }
}

private func SingleTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, referenceSequenceLength: Int,
  contextBlockPreOnly: Bool, usesFlashAttention: FlashAttentionLevel,
  cachedSequenceLength: Int = 0, kvCache: Bool = false
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let cachedKeys = kvCache ? Input() : nil
  let cachedValues = kvCache ? Input() : nil
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, flags: [.Float16], name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
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
  if var cachedKeys: Model.IO = cachedKeys, var cachedValues: Model.IO = cachedValues {
    if b > 1 {
      cachedKeys = Concat(axis: 0)(Array(repeating: cachedKeys, count: b))
      cachedKeys.add(dependencies: [xOut])
      cachedValues = Concat(axis: 0)(Array(repeating: cachedValues, count: b))
      cachedValues.add(dependencies: [xOut])
      keys = Functional.concat(axis: 1, keys, cachedKeys, flags: [.disableOpt])
      values = Functional.concat(axis: 1, values, cachedValues, flags: [.disableOpt])
    } else {
      keys = Functional.concat(axis: 1, keys, cachedKeys)
      values = Functional.concat(axis: 1, values, cachedValues)
    }
  }
  let querySequenceLength = t + hw
  let keyValueSequenceLength = querySequenceLength + cachedSequenceLength
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
        let key = keys.reshaped(
          [1, keyValueSequenceLength, k], offset: [i, 0, 0],
          strides: [keyValueSequenceLength * k, k, 1])
        let query = queries.reshaped(
          [1, querySequenceLength, k], offset: [i, 0, 0],
          strides: [querySequenceLength * k, k, 1])
        let value = values.reshaped(
          [1, keyValueSequenceLength, k], offset: [i, 0, 0],
          strides: [keyValueSequenceLength * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([querySequenceLength, keyValueSequenceLength])
        dot = dot.softmax()
        dot = dot.reshaped([1, querySequenceLength, keyValueSequenceLength])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, querySequenceLength, k]).transposed(1, 2).reshaped([
        b, querySequenceLength, h * k,
      ])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * querySequenceLength, keyValueSequenceLength])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, querySequenceLength, keyValueSequenceLength])
      out = dot * values
      out = out.reshaped([b, h, querySequenceLength, k]).transposed(1, 2).reshaped([
        b, querySequenceLength, h * k,
      ])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, querySequenceLength, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, querySequenceLength, k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped(
      [b, hw - referenceSequenceLength, h * k], strides: [querySequenceLength * h * k, h * k, 1]
    )
    .contiguous()
    xIn = x.reshaped(
      [b, hw - referenceSequenceLength, h * k], strides: [querySequenceLength * h * k, h * k, 1]
    )
    .contiguous()
    xOut = xOut.reshaped(
      [b, hw - referenceSequenceLength, h * k], strides: [querySequenceLength * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  let xW1 = Dense(count: k * h * 3, noBias: true, flags: [.Float16], name: "x_w1")
  let xW3 = Dense(count: k * h * 3, noBias: true, name: "x_w3")
  let xW2 = Dense(count: k * h, noBias: true, name: "x_w2")
  out = xUnifyheads(out) + xW2(xW3(xOut) .* xW1(xOut).swish())
  out = xIn + (out .* xChunks[2]).to(of: xIn)
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).linear1.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xW1.weight.name,
          xW3.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3, k * h * 6])
      mapping["\(prefix.0).norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).norm.query_norm.scale"] = [normQ.weight.name]
      mapping["\(prefix.0).linear2.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xW2.weight.name], format: .I, offsets: [0, k * h])
    case .diffusers:
      mapping["\(prefix.1).attn.to_qkv_mlp_proj.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xW1.weight.name,
          xW3.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3, k * h * 6])
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).attn.to_out.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xW2.weight.name], format: .I, offsets: [0, k * h])
    }
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rot] + (cachedKeys.map { [$0] } ?? []) + (cachedValues.map { [$0] } ?? []) + xChunks,
      [
        out
      ])
  )
}

public func Flux2(
  batchSize: Int, tokenLength: Int, referenceSequenceLength: Int, height: Int, width: Int,
  channels: Int, layers: (Int, Int), usesFlashAttention: FlashAttentionLevel,
  kvCache: Bool
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let contextIn = Input()
  let rot = Input()
  let h = height / 2
  let w = width / 2
  let kvCache = kvCache && referenceSequenceLength > 0
  let xEmbedder = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2], noBias: true,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out: Model.IO
  let referenceLatents: Input?
  if kvCache {
    out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
    referenceLatents = nil
  } else if referenceSequenceLength > 0 && (layers.0 > 0 || layers.1 > 0) {
    let latents = Input()
    let imgIn = xEmbedder(x).reshaped([batchSize, h * w, channels])
    out = Functional.concat(axis: 1, imgIn, latents, flags: [.disableOpt]).to(.Float32)
    referenceLatents = latents
  } else {
    out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
    referenceLatents = nil
  }
  var context = contextIn.to(.Float32)
  let xChunks = (0..<6).map { _ in Input() }
  let contextChunks = (0..<6).map { _ in Input() }
  let cachedAttentionKVs =
    kvCache ? (0..<(layers.0 + layers.1)).map { _ in (Input(), Input()) } : []
  let sequenceLength = h * w + tokenLength + (kvCache ? 0 : referenceSequenceLength)
  let rotResized = rot.reshaped(.NHWC(1, sequenceLength, 1, 128))
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (mapper, block) = JointTransformerBlock(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize,
      t: tokenLength, hw: h * w + (kvCache ? 0 : referenceSequenceLength),
      contextBlockPreOnly: false,
      scaleFactor: i > layers.0 - 3 ? 8 : nil, usesFlashAttention: usesFlashAttention,
      cachedSequenceLength: kvCache ? referenceSequenceLength : 0, kvCache: kvCache)
    var blockInputs: [Model.IO] = [context, out, rotResized]
    if kvCache {
      blockInputs.append(cachedAttentionKVs[i].0)
      blockInputs.append(cachedAttentionKVs[i].1)
    }
    let blockOut = block(blockInputs + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    mappers.append(mapper)
  }
  let singleChunks = (0..<3).map { _ in Input() }
  out = Functional.concat(axis: 1, out, context)
  for i in 0..<layers.1 {
    let (mapper, block) = SingleTransformerBlock(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize,
      t: tokenLength, hw: h * w + (kvCache ? 0 : referenceSequenceLength),
      referenceSequenceLength: kvCache ? 0 : referenceSequenceLength,
      contextBlockPreOnly: i == layers.1 - 1, usesFlashAttention: usesFlashAttention,
      cachedSequenceLength: kvCache ? referenceSequenceLength : 0, kvCache: kvCache)
    var blockInputs: [Model.IO] = [out, rotResized]
    if kvCache {
      let cachedAttentionKV = cachedAttentionKVs[layers.0 + i]
      blockInputs.append(cachedAttentionKV.0)
      blockInputs.append(cachedAttentionKV.1)
    }
    out = block(blockInputs + singleChunks)
    mappers.append(mapper)
  }
  let scale = Input()
  let shift = Input()
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = normFinal(out).to(.Float16) .* scale + shift
  let projOut = Dense(count: 2 * 2 * 32, noBias: true, name: "linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 32, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    batchSize, h * 2, w * 2, 32,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["img_in.weight"] = [xEmbedder.weight.name]
      mapping["final_layer.linear.weight"] = [projOut.weight.name]
    case .diffusers:
      mapping["x_embedder.weight"] = [xEmbedder.weight.name]
      mapping["proj_out.weight"] = [projOut.weight.name]
    }
    return mapping
  }
  var inputs: [Input] = [x, rot] + (referenceLatents.map { [$0] } ?? []) + [contextIn]
  inputs.append(contentsOf: xChunks + contextChunks + singleChunks)
  inputs.append(contentsOf: [scale, shift])
  inputs.append(contentsOf: cachedAttentionKVs.flatMap { [$0.0, $0.1] })
  return (mapper, Model(inputs, [out]))
}

private func JointTransformerBlockImageOnly(
  prefix: (String, String), k: Int, h: Int, b: Int, scaleFactor: Float?,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToQueries = Dense(count: k * h, noBias: true, flags: [.Float16], name: "x_q")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var keys = xToKeys(xOut).reshaped([b, -1, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  keys = normK(keys)
  var queries = xToQueries(xOut).reshaped([b, -1, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  queries = normQ(queries)
  let values = xToValues(xOut).reshaped([b, -1, h, k])
  let rotatedKeys = Functional.cmul(left: keys, right: rot)
  queries = Functional.cmul(left: queries, right: rot)
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = rotatedKeys.transposed(1, 2)
    let transposedQueries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
    let transposedValues = values.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
    dot = dot.softmax()
    out = dot * transposedValues
    out = out.transposed(1, 2).reshaped([b, -1, h * k])
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention((1.0 / Float(k).squareRoot()) * queries, rotatedKeys, values)
      .reshaped([b, -1, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, rotatedKeys, values).reshaped([b, -1, k * h])
  }
  let xUnifyheads = Dense(count: k * h, noBias: true, name: "x_o")
  xOut = x + (xUnifyheads(out) .* xChunks[2]).to(of: x)
  let (xW1, xW2, xW3, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 3, scaleFactor: scaleFactor, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if let scaleFactor = scaleFactor {
    xOut = Add(rightScalar: scaleFactor)(
      xOut, xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]) .* xChunks[5].to(of: xOut))
  } else {
    xOut =
      xOut + (xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]) .* xChunks[5]).to(of: xOut)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping: [String: ModelWeightElement] = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).img_attn.qkv.weight"] = [
        xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
      ]
      mapping["\(prefix.0).img_attn.norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).img_attn.norm.query_norm.scale"] = [normQ.weight.name]
      mapping["\(prefix.0).img_attn.proj.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.0).img_mlp.0.weight"] = [xW1.weight.name, xW3.weight.name]
      mapping["\(prefix.0).img_mlp.2.weight"] = [xW2.weight.name]
    case .diffusers:
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.1).ff.linear_in.weight"] = [xW1.weight.name, xW3.weight.name]
      mapping["\(prefix.1).ff.linear_out.weight"] = [xW2.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [xOut, rotatedKeys, values]))
}

private func SingleTransformerBlockImageOnly(
  prefix: (String, String), k: Int, h: Int, b: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  var xChunks = (0..<2).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = Dense(count: k * h, noBias: true, name: "x_k")
  let xToValues = Dense(count: k * h, noBias: true, name: "x_v")
  var keys = xToKeys(xOut).reshaped([b, -1, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  keys = normK(keys)
  let values = xToValues(xOut).reshaped([b, -1, h, k])
  let rotatedKeys = Functional.cmul(left: keys, right: rot)
  let xToQueries: Model?
  let normQ: Model?
  let xUnifyheads: Model?
  let xW1: Model?
  let xW3: Model?
  let xW2: Model?
  let refOut: Model.IO?
  if !contextBlockPreOnly {
    xChunks.append(Input())
    let xToQueriesLocal = Dense(count: k * h, noBias: true, flags: [.Float16], name: "x_q")
    var queries = xToQueriesLocal(xOut).reshaped([b, -1, h, k])
    let normQLocal = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
    queries = normQLocal(queries)
    queries = Functional.cmul(left: queries, right: rot)
    var out: Model.IO
    switch usesFlashAttention {
    case .none:
      let transposedKeys = rotatedKeys.transposed(1, 2)
      let transposedQueries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
      let transposedValues = values.transposed(1, 2)
      var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
      dot = dot.softmax()
      out = dot * transposedValues
      out = out.transposed(1, 2).reshaped([b, -1, h * k])
    case .scale1:
      let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
      out = scaledDotProductAttention((1.0 / Float(k).squareRoot()) * queries, rotatedKeys, values)
        .reshaped([b, -1, k * h])
    case .scaleMerged, .quantized:
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(),
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      out = scaledDotProductAttention(queries, rotatedKeys, values).reshaped([b, -1, k * h])
    }
    let xUnifyheadsLocal = Dense(count: k * h, noBias: true, name: "x_o")
    let xW1Local = Dense(count: k * h * 3, noBias: true, flags: [.Float16], name: "x_w1")
    let xW3Local = Dense(count: k * h * 3, noBias: true, name: "x_w3")
    let xW2Local = Dense(count: k * h, noBias: true, name: "x_w2")
    out = xUnifyheadsLocal(out) + xW2Local(xW3Local(xOut) .* xW1Local(xOut).swish())
    out = x + (out .* xChunks[2]).to(of: x)
    xToQueries = xToQueriesLocal
    normQ = normQLocal
    xUnifyheads = xUnifyheadsLocal
    xW1 = xW1Local
    xW3 = xW3Local
    xW2 = xW2Local
    refOut = out
  } else {
    xToQueries = nil
    normQ = nil
    xUnifyheads = nil
    xW1 = nil
    xW3 = nil
    xW2 = nil
    refOut = nil
  }
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    guard let xToQueries = xToQueries, let normQ = normQ, let xUnifyheads = xUnifyheads,
      let xW1 = xW1, let xW3 = xW3, let xW2 = xW2
    else {
      return mapping
    }
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).linear1.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xW1.weight.name,
          xW3.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3, k * h * 6])
      mapping["\(prefix.0).norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).norm.query_norm.scale"] = [normQ.weight.name]
      mapping["\(prefix.0).linear2.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xW2.weight.name], format: .I, offsets: [0, k * h])
    case .diffusers:
      mapping["\(prefix.1).attn.to_qkv_mlp_proj.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xW1.weight.name,
          xW3.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3, k * h * 6])
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).attn.to_out.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xW2.weight.name], format: .I, offsets: [0, k * h])
    }
    return mapping
  }
  if let refOut = refOut {
    return (mapper, Model([x, rot] + xChunks, [refOut, rotatedKeys, values]))
  } else {
    return (mapper, Model([x, rot] + xChunks, [rotatedKeys, values]))
  }
}

public func Flux2Fixed(
  timesteps: Int,
  channels: Int, layers: (Int, Int), numberOfReferenceImages: Int, guidanceEmbed: Bool,
  usesFlashAttention: FlashAttentionLevel, kvCache: Bool
) -> (
  ModelWeightMapper, Model
) {
  let kvCache = kvCache && numberOfReferenceImages > 0
  let contextIn = Input()
  let t = Input()
  let referenceRot = kvCache ? Input() : nil
  var referenceImages = [Input]()
  var referenceOutputs = [Model.IO]()
  let xEmbedder: Convolution?
  if numberOfReferenceImages > 0 {
    let embedder = Convolution(
      groups: 1, filters: channels, filterSize: [2, 2], noBias: true,
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
    xEmbedder = embedder
    for _ in 0..<numberOfReferenceImages {
      let x = Input()
      let out = embedder(x)
      referenceImages.append(x)
      referenceOutputs.append(out)
    }
  } else {
    xEmbedder = nil
  }
  let contextEmbedder = Dense(count: channels, noBias: true, name: "context_embedder")
  let context = contextEmbedder(contextIn)
  let (tMlp0, tMlp2, tEmbedder) = MLPEmbedder(channels: channels, name: "t")
  var vec = tEmbedder(t)
  let g: Input?
  let gMlp0: Model?
  let gMlp2: Model?
  if guidanceEmbed {
    let (mlp0, mlp2, gEmbedder) = MLPEmbedder(channels: channels, name: "guidance")
    let input = Input()
    vec = vec + gEmbedder(input)
    g = input
    gMlp0 = mlp0
    gMlp2 = mlp2
  } else {
    g = nil
    gMlp0 = nil
    gMlp2 = nil
  }
  vec = vec.swish()
  let xAdaLNs = (0..<6).map { Dense(count: channels, noBias: true, name: "x_ada_ln_\($0)") }
  let contextAdaLNs = (0..<6).map {
    Dense(count: channels, noBias: true, name: "context_ada_ln_\($0)")
  }
  var xChunks = xAdaLNs.map { $0(vec) }
  var contextChunks = contextAdaLNs.map { $0(vec) }
  xChunks[1] = 1 + xChunks[1]
  xChunks[4] = 1 + xChunks[4]
  contextChunks[1] = 1 + contextChunks[1]
  contextChunks[4] = 1 + contextChunks[4]
  let singleAdaLNs = (0..<3).map {
    Dense(count: channels, noBias: true, name: "single_ada_ln_\($0)")
  }
  var singleChunks = singleAdaLNs.map { $0(vec) }
  singleChunks[1] = 1 + singleChunks[1]
  let scale = Dense(count: channels, noBias: true, name: "ada_ln_0")
  let shift = Dense(count: channels, noBias: true, name: "ada_ln_1")
  let finalChunks = [(1 + scale(vec)), shift(vec)]
  var cachedReferenceKVs = [Model.IO]()
  var mappers = [ModelWeightMapper]()
  if let referenceRot = referenceRot {
    let fixedXChunks = xChunks.map {
      $0.reshaped(
        [1, 1, channels], offset: [timesteps - 1, 0, 0], strides: [channels, channels, 1]
      ).contiguous()
    }
    let fixedSingleChunks = singleChunks.map {
      $0.reshaped(
        [1, 1, channels], offset: [timesteps - 1, 0, 0], strides: [channels, channels, 1]
      ).contiguous()
    }
    let referenceLatents = referenceOutputs.map { $0.reshaped([1, -1, channels]).to(.Float32) }
    var referenceOut = referenceLatents[0]
    if referenceLatents.count > 1 {
      referenceOut = Concat(axis: 1)(referenceLatents)
    }
    let referenceRotResized = referenceRot.reshaped([1, -1, 1, 128])
    for i in 0..<layers.0 {
      let (mapper, block) = JointTransformerBlockImageOnly(
        prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
        b: 1, scaleFactor: i > layers.0 - 3 ? 8 : nil, usesFlashAttention: usesFlashAttention)
      let blockOut = block([referenceOut, referenceRotResized] + fixedXChunks)
      referenceOut = blockOut[0]
      cachedReferenceKVs.append(blockOut[1])
      cachedReferenceKVs.append(blockOut[2])
      mappers.append(mapper)
    }
    for i in 0..<layers.1 {
      let (mapper, block) = SingleTransformerBlockImageOnly(
        prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128,
        h: channels / 128, b: 1, contextBlockPreOnly: i == layers.1 - 1,
        usesFlashAttention: usesFlashAttention)
      if i == layers.1 - 1 {
        let blockOut = block([
          referenceOut, referenceRotResized, fixedSingleChunks[0], fixedSingleChunks[1],
        ])
        cachedReferenceKVs.append(blockOut[0])
        cachedReferenceKVs.append(blockOut[1])
      } else {
        let blockOut = block([referenceOut, referenceRotResized] + fixedSingleChunks)
        referenceOut = blockOut[0]
        cachedReferenceKVs.append(blockOut[1])
        cachedReferenceKVs.append(blockOut[2])
        mappers.append(mapper)
      }
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      if let xEmbedder = xEmbedder {
        mapping["img_in.weight"] = [xEmbedder.weight.name]
      }
      mapping["txt_in.weight"] = [contextEmbedder.weight.name]
      mapping["time_in.in_layer.weight"] = [tMlp0.weight.name]
      mapping["time_in.out_layer.weight"] = [tMlp2.weight.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["guidance_in.in_layer.weight"] = [gMlp0.weight.name]
        mapping["guidance_in.out_layer.weight"] = [gMlp2.weight.name]
      }
      mapping[
        "double_stream_modulation_img.lin.weight"
      ] = ModelWeightElement(xAdaLNs.map { $0.weight.name })
      mapping[
        "double_stream_modulation_txt.lin.weight"
      ] = ModelWeightElement(contextAdaLNs.map { $0.weight.name })
      mapping[
        "single_stream_modulation.lin.weight"
      ] = ModelWeightElement(singleAdaLNs.map { $0.weight.name })
      mapping["final_layer.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    case .diffusers:
      if let xEmbedder = xEmbedder {
        mapping["x_embedder.weight"] = [xEmbedder.weight.name]
      }
      mapping["context_embedder.weight"] = [contextEmbedder.weight.name]
      mapping["time_guidance_embed.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
      mapping["time_guidance_embed.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["time_guidance_embed.guidance_embedder.linear_1.weight"] = [gMlp0.weight.name]
        mapping["time_guidance_embed.guidance_embedder.linear_2.weight"] = [gMlp2.weight.name]
      }
      mapping[
        "double_stream_modulation_img.linear.weight"
      ] = ModelWeightElement(xAdaLNs.map { $0.weight.name })
      mapping[
        "double_stream_modulation_txt.linear.weight"
      ] = ModelWeightElement(contextAdaLNs.map { $0.weight.name })
      mapping[
        "single_stream_modulation.linear.weight"
      ] = ModelWeightElement(singleAdaLNs.map { $0.weight.name })
      mapping["norm_out.linear.weight"] = [scale.weight.name, shift.weight.name]
    }
    return mapping
  }
  var inputs: [Input] = [contextIn] + referenceImages + [t]
  inputs.append(contentsOf: g.map { [$0] } ?? [])
  inputs.append(contentsOf: referenceRot.map { [$0] } ?? [])
  var outputs = [Model.IO]()
  if !kvCache {
    outputs.append(contentsOf: referenceOutputs)
  }
  outputs.append(context)
  outputs.append(contentsOf: xChunks)
  outputs.append(contentsOf: contextChunks)
  outputs.append(contentsOf: singleChunks)
  outputs.append(contentsOf: finalChunks)
  outputs.append(contentsOf: cachedReferenceKVs)
  let model = Model(inputs, outputs)
  return (mapper, model)
}

public func Flux2FixedOutputShapes(tokenLength: Int, channels: Int) -> [TensorShape] {
  var outs = [TensorShape]()
  outs.append(TensorShape([1, tokenLength, channels]))
  for _ in 0..<(6 + 6 + 3 + 2) {
    outs.append(TensorShape([1, 1, channels]))
  }
  return outs
}

private func LoRAMLPEmbedder(
  channels: Int, configuration: LoRANetworkConfiguration, index: Int, name: String
) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = LoRADense(
    count: channels, configuration: configuration, noBias: true, index: index,
    name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = LoRADense(
    count: channels, configuration: configuration, noBias: true, index: index,
    name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LoRAFeedForward(
  hiddenSize: Int, intermediateSize: Int, scaleFactor: Float?,
  configuration: LoRANetworkConfiguration, index: Int, name: String
)
  -> (
    Model, Model, Model, Model
  )
{
  let x = Input()
  let w1 = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, flags: [.Float16],
    index: index, name: "\(name)_gate_proj")
  let w3 = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, index: index,
    name: "\(name)_up_proj")
  var out: Model.IO
  if let scaleFactor = scaleFactor {
    out = w3((1 / scaleFactor) * x)  // x usually is smaller (hiddenSize < intermediateSize).
  } else {
    out = w3(x)
  }
  out = Functional.swishMul(value: out, gate: w1(x))
  let w2 = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, index: index,
    name: "\(name)_down_proj")
  out = w2(out)
  if let _ = scaleFactor {
    out = out.to(.Float32)  // scaleFactor is fused in the other add kernel.
  }
  return (w1, w2, w3, Model([x], [out]))
}

private func LoRAJointTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, contextBlockPreOnly: Bool,
  scaleFactor: Float?, usesFlashAttention: FlashAttentionLevel, cachedSequenceLength: Int = 0,
  kvCache: Bool = false, layerIndex: Int, configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  let x = Input()
  let rot = Input()
  let cachedKeys = kvCache ? Input() : nil
  let cachedValues = kvCache ? Input() : nil
  let contextChunks = (0..<(contextBlockPreOnly ? 2 : 6)).map { _ in Input() }
  let contextNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var contextOut = contextNorm1(context).to(.Float16) .* contextChunks[1] + contextChunks[0]
  let contextToKeys = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "c_k")
  let contextToQueries = LoRADense(
    count: k * h, configuration: configuration, noBias: true, flags: [.Float16], index: layerIndex,
    name: "c_q")
  let contextToValues = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "c_v")
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
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, noBias: true, flags: [.Float16], index: layerIndex,
    name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_v")
  var xK = xToKeys(xOut).reshaped([b, hw, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  xK = normK(xK)
  var xQ = xToQueries(xOut).reshaped([b, hw, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  xQ = normQ(xQ)
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var keys = Functional.concat(axis: 1, xK, contextK)
  var values = Functional.concat(axis: 1, xV, contextV)
  var queries = Functional.concat(axis: 1, xQ, contextQ)
  queries = Functional.cmul(left: queries, right: rot)
  keys = Functional.cmul(left: keys, right: rot)
  if var cachedKeys: Model.IO = cachedKeys, var cachedValues: Model.IO = cachedValues {
    if b > 1 {
      cachedKeys = Concat(axis: 0)(Array(repeating: cachedKeys, count: b))
      cachedKeys.add(dependencies: [xOut])
      cachedValues = Concat(axis: 0)(Array(repeating: cachedValues, count: b))
      cachedValues.add(dependencies: [xOut])
      keys = Functional.concat(axis: 1, keys, cachedKeys, flags: [.disableOpt])
      values = Functional.concat(axis: 1, values, cachedValues, flags: [.disableOpt])
    } else {
      keys = Functional.concat(axis: 1, keys, cachedKeys)
      values = Functional.concat(axis: 1, values, cachedValues)
    }
  }
  let querySequenceLength = t + hw
  let keyValueSequenceLength = querySequenceLength + cachedSequenceLength
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
        let key = keys.reshaped(
          [1, keyValueSequenceLength, k], offset: [i, 0, 0],
          strides: [keyValueSequenceLength * k, k, 1])
        let query = queries.reshaped(
          [1, querySequenceLength, k], offset: [i, 0, 0],
          strides: [querySequenceLength * k, k, 1])
        let value = values.reshaped(
          [1, keyValueSequenceLength, k], offset: [i, 0, 0],
          strides: [keyValueSequenceLength * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([querySequenceLength, keyValueSequenceLength])
        dot = dot.softmax()
        dot = dot.reshaped([1, querySequenceLength, keyValueSequenceLength])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, querySequenceLength, k]).transposed(1, 2).reshaped([
        b, querySequenceLength, h * k,
      ])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * querySequenceLength, keyValueSequenceLength])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, querySequenceLength, keyValueSequenceLength])
      out = dot * values
      out = out.reshaped([b, h, querySequenceLength, k]).transposed(1, 2).reshaped([
        b, querySequenceLength, h * k,
      ])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, querySequenceLength, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, querySequenceLength, k * h])
  }
  let contextUnifyheads: Model?
  if !contextBlockPreOnly {
    contextOut = out.reshaped(
      [b, t, h * k], offset: [0, hw, 0], strides: [querySequenceLength * h * k, h * k, 1]
    ).contiguous()
    let unifyheads = LoRADense(
      count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "c_o")
    contextOut = unifyheads(contextOut)
    contextUnifyheads = unifyheads
  } else {
    contextUnifyheads = nil
  }
  xOut = out.reshaped([b, hw, h * k], strides: [querySequenceLength * h * k, h * k, 1])
    .contiguous()
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_o")
  xOut = xUnifyheads(xOut)
  if !contextBlockPreOnly {
    contextOut = context + (contextOut .* contextChunks[2]).to(of: context)
  }
  xOut = x + (xOut .* xChunks[2]).to(of: x)
  // Attentions are now. Now run MLP.
  let contextW1: Model?
  let contextW2: Model?
  let contextW3: Model?
  if !contextBlockPreOnly {
    let contextFF: Model
    (contextW1, contextW2, contextW3, contextFF) = LoRAFeedForward(
      hiddenSize: k * h, intermediateSize: k * h * 3, scaleFactor: scaleFactor,
      configuration: configuration, index: layerIndex, name: "c")
    if configuration.gradientCheckpointingFeedForward {
      contextFF.gradientCheckpointing = true
    }
    let contextNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
    if let scaleFactor = scaleFactor {
      contextOut =
        Add(rightScalar: scaleFactor)(
          contextOut,
          contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
            .* contextChunks[5].to(of: contextOut))
    } else {
      contextOut =
        contextOut
        + (contextFF(contextNorm2(contextOut).to(.Float16) .* contextChunks[4] + contextChunks[3])
        .* contextChunks[5])
        .to(of: contextOut)
    }
  } else {
    contextW1 = nil
    contextW2 = nil
    contextW3 = nil
  }
  let (xW1, xW2, xW3, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 3, scaleFactor: scaleFactor,
    configuration: configuration, index: layerIndex, name: "x")
  if configuration.gradientCheckpointingFeedForward {
    xFF.gradientCheckpointing = true
  }
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if let scaleFactor = scaleFactor {
    xOut = Add(rightScalar: scaleFactor)(
      xOut, (xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]) .* xChunks[5].to(of: xOut)))
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
      mapping["\(prefix.0).txt_attn.norm.key_norm.scale"] = [normAddedK.weight.name]
      mapping["\(prefix.0).txt_attn.norm.query_norm.scale"] = [normAddedQ.weight.name]
      mapping["\(prefix.0).img_attn.qkv.weight"] = [
        xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
      ]
      mapping["\(prefix.0).img_attn.norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).img_attn.norm.query_norm.scale"] = [normQ.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.0).txt_attn.proj.weight"] = [contextUnifyheads.weight.name]
      }
      mapping["\(prefix.0).img_attn.proj.weight"] = [xUnifyheads.weight.name]
      if let contextW1 = contextW1, let contextW2 = contextW2, let contextW3 = contextW3 {
        mapping["\(prefix.0).txt_mlp.0.weight"] = [contextW1.weight.name, contextW3.weight.name]
        mapping["\(prefix.0).txt_mlp.2.weight"] = [contextW2.weight.name]
      }
      mapping["\(prefix.0).img_mlp.0.weight"] = [xW1.weight.name, xW3.weight.name]
      mapping["\(prefix.0).img_mlp.2.weight"] = [xW2.weight.name]
    case .diffusers:
      mapping["\(prefix.1).attn.add_q_proj.weight"] = [contextToQueries.weight.name]
      mapping["\(prefix.1).attn.add_k_proj.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix.1).attn.add_v_proj.weight"] = [contextToValues.weight.name]
      mapping["\(prefix.1).attn.norm_added_k.weight"] = [normAddedK.weight.name]
      mapping["\(prefix.1).attn.norm_added_q.weight"] = [normAddedQ.weight.name]
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      if let contextUnifyheads = contextUnifyheads {
        mapping["\(prefix.1).attn.to_add_out.weight"] = [contextUnifyheads.weight.name]
      }
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      if let contextW1 = contextW1, let contextW2 = contextW2, let contextW3 = contextW3 {
        mapping["\(prefix.1).ff_context.linear_in.weight"] = [
          contextW1.weight.name, contextW3.weight.name,
        ]
        mapping["\(prefix.1).ff_context.linear_out.weight"] = [contextW2.weight.name]
      }
      mapping["\(prefix.1).ff.linear_in.weight"] = [xW1.weight.name, xW3.weight.name]
      mapping["\(prefix.1).ff.linear_out.weight"] = [xW2.weight.name]
    }
    return mapping
  }
  if !contextBlockPreOnly {
    return (
      mapper,
      Model(
        [context, x, rot] + (cachedKeys.map { [$0] } ?? []) + (cachedValues.map { [$0] } ?? [])
          + contextChunks + xChunks, [contextOut, xOut])
    )
  } else {
    return (
      mapper,
      Model(
        [context, x, rot] + (cachedKeys.map { [$0] } ?? []) + (cachedValues.map { [$0] } ?? [])
          + contextChunks + xChunks, [xOut])
    )
  }
}

private func LoRASingleTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, t: Int, hw: Int, referenceSequenceLength: Int,
  contextBlockPreOnly: Bool, usesFlashAttention: FlashAttentionLevel,
  cachedSequenceLength: Int = 0, kvCache: Bool = false, layerIndex: Int, MLPLayerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let cachedKeys = kvCache ? Input() : nil
  let cachedValues = kvCache ? Input() : nil
  let xChunks = (0..<3).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, noBias: true, flags: [.Float16], index: layerIndex,
    name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_v")
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
  if var cachedKeys: Model.IO = cachedKeys, var cachedValues: Model.IO = cachedValues {
    if b > 1 {
      cachedKeys = Concat(axis: 0)(Array(repeating: cachedKeys, count: b))
      cachedKeys.add(dependencies: [xOut])
      cachedValues = Concat(axis: 0)(Array(repeating: cachedValues, count: b))
      cachedValues.add(dependencies: [xOut])
      keys = Functional.concat(axis: 1, keys, cachedKeys, flags: [.disableOpt])
      values = Functional.concat(axis: 1, values, cachedValues, flags: [.disableOpt])
    } else {
      keys = Functional.concat(axis: 1, keys, cachedKeys)
      values = Functional.concat(axis: 1, values, cachedValues)
    }
  }
  let querySequenceLength = t + hw
  let keyValueSequenceLength = querySequenceLength + cachedSequenceLength
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
        let key = keys.reshaped(
          [1, keyValueSequenceLength, k], offset: [i, 0, 0],
          strides: [keyValueSequenceLength * k, k, 1])
        let query = queries.reshaped(
          [1, querySequenceLength, k], offset: [i, 0, 0],
          strides: [querySequenceLength * k, k, 1])
        let value = values.reshaped(
          [1, keyValueSequenceLength, k], offset: [i, 0, 0],
          strides: [keyValueSequenceLength * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([querySequenceLength, keyValueSequenceLength])
        dot = dot.softmax()
        dot = dot.reshaped([1, querySequenceLength, keyValueSequenceLength])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, querySequenceLength, k]).transposed(1, 2).reshaped([
        b, querySequenceLength, h * k,
      ])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * querySequenceLength, keyValueSequenceLength])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, querySequenceLength, keyValueSequenceLength])
      out = dot * values
      out = out.reshaped([b, h, querySequenceLength, k]).transposed(1, 2).reshaped([
        b, querySequenceLength, h * k,
      ])
    }
  case .scale1:
    queries = (1.0 / Float(k).squareRoot()) * queries
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, querySequenceLength, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, querySequenceLength, k * h])
  }
  var xIn: Model.IO = x
  if contextBlockPreOnly {
    out = out.reshaped(
      [b, hw - referenceSequenceLength, h * k], strides: [querySequenceLength * h * k, h * k, 1]
    )
    .contiguous()
    xIn = x.reshaped(
      [b, hw - referenceSequenceLength, h * k], strides: [querySequenceLength * h * k, h * k, 1]
    )
    .contiguous()
    xOut = xOut.reshaped(
      [b, hw - referenceSequenceLength, h * k], strides: [querySequenceLength * h * k, h * k, 1]
    )
    .contiguous()
  }
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_o")
  let xW1 = LoRADense(
    count: k * h * 3, configuration: configuration, noBias: true, flags: [.Float16],
    index: MLPLayerIndex, name: "x_w1")
  let xW3 = LoRADense(
    count: k * h * 3, configuration: configuration, noBias: true, index: MLPLayerIndex, name: "x_w3"
  )
  let xW2 = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: MLPLayerIndex, name: "x_w2")
  out = xUnifyheads(out) + xW2(xW3(xOut) .* xW1(xOut).swish())
  out = xIn + (out .* xChunks[2]).to(of: xIn)
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).linear1.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xW1.weight.name,
          xW3.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3, k * h * 6])
      mapping["\(prefix.0).norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).norm.query_norm.scale"] = [normQ.weight.name]
      mapping["\(prefix.0).linear2.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xW2.weight.name], format: .I, offsets: [0, k * h])
    case .diffusers:
      mapping["\(prefix.1).attn.to_qkv_mlp_proj.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xW1.weight.name,
          xW3.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3, k * h * 6])
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).attn.to_out.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xW2.weight.name], format: .I, offsets: [0, k * h])
    }
    return mapping
  }
  return (
    mapper,
    Model(
      [x, rot] + (cachedKeys.map { [$0] } ?? []) + (cachedValues.map { [$0] } ?? []) + xChunks,
      [out])
  )
}

private func LoRAJointTransformerBlockImageOnly(
  prefix: (String, String), k: Int, h: Int, b: Int, scaleFactor: Float?,
  usesFlashAttention: FlashAttentionLevel, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let xChunks = (0..<6).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, noBias: true, flags: [.Float16], index: layerIndex,
    name: "x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_v")
  var keys = xToKeys(xOut).reshaped([b, -1, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  keys = normK(keys)
  var queries = xToQueries(xOut).reshaped([b, -1, h, k])
  let normQ = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
  queries = normQ(queries)
  let values = xToValues(xOut).reshaped([b, -1, h, k])
  let rotatedKeys = Functional.cmul(left: keys, right: rot)
  queries = Functional.cmul(left: queries, right: rot)
  var out: Model.IO
  switch usesFlashAttention {
  case .none:
    let transposedKeys = rotatedKeys.transposed(1, 2)
    let transposedQueries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
    let transposedValues = values.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
    dot = dot.softmax()
    out = dot * transposedValues
    out = out.transposed(1, 2).reshaped([b, -1, h * k])
  case .scale1:
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    out = scaledDotProductAttention((1.0 / Float(k).squareRoot()) * queries, rotatedKeys, values)
      .reshaped([b, -1, k * h])
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    out = scaledDotProductAttention(queries, rotatedKeys, values).reshaped([b, -1, k * h])
  }
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_o")
  xOut = x + (xUnifyheads(out) .* xChunks[2]).to(of: x)
  let (xW1, xW2, xW3, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: k * h * 3, scaleFactor: scaleFactor,
    configuration: configuration, index: layerIndex, name: "x")
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  if let scaleFactor = scaleFactor {
    xOut = Add(rightScalar: scaleFactor)(
      xOut, xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]) .* xChunks[5].to(of: xOut))
  } else {
    xOut =
      xOut + (xFF(xNorm2(xOut).to(.Float16) .* xChunks[4] + xChunks[3]) .* xChunks[5]).to(of: xOut)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping: [String: ModelWeightElement] = [:]
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).img_attn.qkv.weight"] = [
        xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name,
      ]
      mapping["\(prefix.0).img_attn.norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).img_attn.norm.query_norm.scale"] = [normQ.weight.name]
      mapping["\(prefix.0).img_attn.proj.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.0).img_mlp.0.weight"] = [xW1.weight.name, xW3.weight.name]
      mapping["\(prefix.0).img_mlp.2.weight"] = [xW2.weight.name]
    case .diffusers:
      mapping["\(prefix.1).attn.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix.1).attn.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix.1).attn.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).attn.to_out.0.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix.1).ff.linear_in.weight"] = [xW1.weight.name, xW3.weight.name]
      mapping["\(prefix.1).ff.linear_out.weight"] = [xW2.weight.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot] + xChunks, [xOut, rotatedKeys, values]))
}

private func LoRASingleTransformerBlockImageOnly(
  prefix: (String, String), k: Int, h: Int, b: Int, contextBlockPreOnly: Bool,
  usesFlashAttention: FlashAttentionLevel, layerIndex: Int, MLPLayerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  var xChunks = (0..<2).map { _ in Input() }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let xOut = xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_k")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_v")
  var keys = xToKeys(xOut).reshaped([b, -1, h, k])
  let normK = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_k")
  keys = normK(keys)
  let values = xToValues(xOut).reshaped([b, -1, h, k])
  let rotatedKeys = Functional.cmul(left: keys, right: rot)
  let xToQueries: Model?
  let normQ: Model?
  let xUnifyheads: Model?
  let xW1: Model?
  let xW3: Model?
  let xW2: Model?
  let refOut: Model.IO?
  if !contextBlockPreOnly {
    xChunks.append(Input())
    let xToQueriesLocal = LoRADense(
      count: k * h, configuration: configuration, noBias: true, flags: [.Float16],
      index: layerIndex, name: "x_q")
    var queries = xToQueriesLocal(xOut).reshaped([b, -1, h, k])
    let normQLocal = RMSNorm(epsilon: 1e-6, axis: [3], name: "x_norm_q")
    queries = normQLocal(queries)
    queries = Functional.cmul(left: queries, right: rot)
    var out: Model.IO
    switch usesFlashAttention {
    case .none:
      let transposedKeys = rotatedKeys.transposed(1, 2)
      let transposedQueries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
      let transposedValues = values.transposed(1, 2)
      var dot = Matmul(transposeB: (2, 3))(transposedQueries, transposedKeys)
      dot = dot.softmax()
      out = dot * transposedValues
      out = out.transposed(1, 2).reshaped([b, -1, h * k])
    case .scale1:
      let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
      out = scaledDotProductAttention((1.0 / Float(k).squareRoot()) * queries, rotatedKeys, values)
        .reshaped([b, -1, k * h])
    case .scaleMerged, .quantized:
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(),
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      out = scaledDotProductAttention(queries, rotatedKeys, values).reshaped([b, -1, k * h])
    }
    let xUnifyheadsLocal = LoRADense(
      count: k * h, configuration: configuration, noBias: true, index: layerIndex, name: "x_o")
    let xW1Local = LoRADense(
      count: k * h * 3, configuration: configuration, noBias: true, flags: [.Float16],
      index: MLPLayerIndex, name: "x_w1")
    let xW3Local = LoRADense(
      count: k * h * 3, configuration: configuration, noBias: true, index: MLPLayerIndex,
      name: "x_w3")
    let xW2Local = LoRADense(
      count: k * h, configuration: configuration, noBias: true, index: MLPLayerIndex, name: "x_w2")
    out = xUnifyheadsLocal(out) + xW2Local(xW3Local(xOut) .* xW1Local(xOut).swish())
    out = x + (out .* xChunks[2]).to(of: x)
    xToQueries = xToQueriesLocal
    normQ = normQLocal
    xUnifyheads = xUnifyheadsLocal
    xW1 = xW1Local
    xW3 = xW3Local
    xW2 = xW2Local
    refOut = out
  } else {
    xToQueries = nil
    normQ = nil
    xUnifyheads = nil
    xW1 = nil
    xW3 = nil
    xW2 = nil
    refOut = nil
  }
  let mapper: ModelWeightMapper = { format in
    var mapping: ModelWeightMapping = [:]
    guard let xToQueries = xToQueries, let normQ = normQ, let xUnifyheads = xUnifyheads,
      let xW1 = xW1, let xW3 = xW3, let xW2 = xW2
    else {
      return mapping
    }
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).linear1.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xW1.weight.name,
          xW3.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3, k * h * 6])
      mapping["\(prefix.0).norm.key_norm.scale"] = [normK.weight.name]
      mapping["\(prefix.0).norm.query_norm.scale"] = [normQ.weight.name]
      mapping["\(prefix.0).linear2.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xW2.weight.name], format: .I, offsets: [0, k * h])
    case .diffusers:
      mapping["\(prefix.1).attn.to_qkv_mlp_proj.weight"] = ModelWeightElement(
        [
          xToQueries.weight.name, xToKeys.weight.name, xToValues.weight.name, xW1.weight.name,
          xW3.weight.name,
        ], offsets: [0, k * h, k * h * 2, k * h * 3, k * h * 6])
      mapping["\(prefix.1).attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix.1).attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix.1).attn.to_out.weight"] = ModelWeightElement(
        [xUnifyheads.weight.name, xW2.weight.name], format: .I, offsets: [0, k * h])
    }
    return mapping
  }
  if let refOut = refOut {
    return (mapper, Model([x, rot] + xChunks, [refOut, rotatedKeys, values]))
  } else {
    return (mapper, Model([x, rot] + xChunks, [rotatedKeys, values]))
  }
}

public func LoRAFlux2(
  batchSize: Int, tokenLength: Int, referenceSequenceLength: Int, height: Int, width: Int,
  channels: Int, layers: (Int, Int), usesFlashAttention: FlashAttentionLevel, kvCache: Bool,
  rotaryEmbeddingHeads: Int = 1,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let contextIn = Input()
  let rot = Input()
  let h = height / 2
  let w = width / 2
  let kvCache = kvCache && referenceSequenceLength > 0
  let xEmbedder = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    noBias: true, hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  var out: Model.IO
  let referenceLatents: Input?
  if kvCache {
    out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
    referenceLatents = nil
  } else if referenceSequenceLength > 0 && (layers.0 > 0 || layers.1 > 0) {
    let latents = Input()
    let imgIn = xEmbedder(x).reshaped([batchSize, h * w, channels])
    out = Functional.concat(axis: 1, imgIn, latents, flags: [.disableOpt]).to(.Float32)
    referenceLatents = latents
  } else {
    out = xEmbedder(x).reshaped([batchSize, h * w, channels]).to(.Float32)
    referenceLatents = nil
  }
  var context = contextIn.to(.Float32)
  let xChunks = (0..<6).map { _ in Input() }
  let contextChunks = (0..<6).map { _ in Input() }
  let cachedAttentionKVs =
    kvCache ? (0..<(layers.0 + layers.1)).map { _ in (Input(), Input()) } : []
  let sequenceLength = h * w + tokenLength + (kvCache ? 0 : referenceSequenceLength)
  let rotResized = rot.reshaped(.NHWC(1, sequenceLength, rotaryEmbeddingHeads, 128))
  var mappers = [ModelWeightMapper]()
  for i in 0..<layers.0 {
    let (mapper, block) = LoRAJointTransformerBlock(
      prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize,
      t: tokenLength, hw: h * w + (kvCache ? 0 : referenceSequenceLength),
      contextBlockPreOnly: false,
      scaleFactor: i > layers.0 - 3 ? 8 : nil, usesFlashAttention: usesFlashAttention,
      cachedSequenceLength: kvCache ? referenceSequenceLength : 0, kvCache: kvCache, layerIndex: i,
      configuration: LoRAConfiguration)
    var blockInputs: [Model.IO] = [context, out, rotResized]
    if kvCache {
      blockInputs.append(cachedAttentionKVs[i].0)
      blockInputs.append(cachedAttentionKVs[i].1)
    }
    let blockOut = block(blockInputs + contextChunks + xChunks)
    context = blockOut[0]
    out = blockOut[1]
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      block.gradientCheckpointing = true
    }
    mappers.append(mapper)
  }
  let singleChunks = (0..<3).map { _ in Input() }
  out = Functional.concat(axis: 1, out, context)
  for i in 0..<layers.1 {
    let (mapper, block) = LoRASingleTransformerBlock(
      prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128, h: channels / 128,
      b: batchSize,
      t: tokenLength, hw: h * w + (kvCache ? 0 : referenceSequenceLength),
      referenceSequenceLength: kvCache ? 0 : referenceSequenceLength,
      contextBlockPreOnly: i == layers.1 - 1, usesFlashAttention: usesFlashAttention,
      cachedSequenceLength: kvCache ? referenceSequenceLength : 0, kvCache: kvCache,
      layerIndex: i + layers.0, MLPLayerIndex: i, configuration: LoRAConfiguration)
    var blockInputs: [Model.IO] = [out, rotResized]
    if kvCache {
      let cachedAttentionKV = cachedAttentionKVs[layers.0 + i]
      blockInputs.append(cachedAttentionKV.0)
      blockInputs.append(cachedAttentionKV.1)
    }
    out = block(blockInputs + singleChunks)
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      block.gradientCheckpointing = true
    }
    mappers.append(mapper)
  }
  let scale = Input()
  let shift = Input()
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = normFinal(out).to(.Float16) .* scale + shift
  let projOut = LoRADense(
    count: 2 * 2 * 32, configuration: LoRAConfiguration, noBias: true, index: 0, name: "linear")
  out = projOut(out)
  // Unpatchify
  out = out.reshaped([batchSize, h, w, 32, 2, 2]).permuted(0, 1, 4, 2, 5, 3).contiguous().reshaped([
    batchSize, h * 2, w * 2, 32,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["img_in.weight"] = [xEmbedder.weight.name]
      mapping["final_layer.linear.weight"] = [projOut.weight.name]
    case .diffusers:
      mapping["x_embedder.weight"] = [xEmbedder.weight.name]
      mapping["proj_out.weight"] = [projOut.weight.name]
    }
    return mapping
  }
  var inputs: [Input] = [x, rot] + (referenceLatents.map { [$0] } ?? []) + [contextIn]
  inputs.append(contentsOf: xChunks + contextChunks + singleChunks)
  inputs.append(contentsOf: [scale, shift])
  inputs.append(contentsOf: cachedAttentionKVs.flatMap { [$0.0, $0.1] })
  return (mapper, Model(inputs, [out], trainable: false))
}

public func LoRAFlux2Fixed(
  timesteps: Int, channels: Int, layers: (Int, Int), numberOfReferenceImages: Int,
  guidanceEmbed: Bool, usesFlashAttention: FlashAttentionLevel, kvCache: Bool,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let kvCache = kvCache && numberOfReferenceImages > 0
  let contextIn = Input()
  let t = Input()
  let referenceRot = kvCache ? Input() : nil
  var referenceImages = [Input]()
  var referenceOutputs = [Model.IO]()
  let xEmbedder: Model?
  if numberOfReferenceImages > 0 {
    let embedder = LoRAConvolution(
      groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
      noBias: true, hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
    xEmbedder = embedder
    for _ in 0..<numberOfReferenceImages {
      let x = Input()
      let out = embedder(x)
      referenceImages.append(x)
      referenceOutputs.append(out)
    }
  } else {
    xEmbedder = nil
  }
  let contextEmbedder = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
    name: "context_embedder")
  let context = contextEmbedder(contextIn)
  let (tMlp0, tMlp2, tEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, index: 0, name: "t")
  var vec = tEmbedder(t)
  let g: Input?
  let gMlp0: Model?
  let gMlp2: Model?
  if guidanceEmbed {
    let (mlp0, mlp2, gEmbedder) = LoRAMLPEmbedder(
      channels: channels, configuration: LoRAConfiguration, index: 0, name: "guidance")
    let input = Input()
    vec = vec + gEmbedder(input)
    g = input
    gMlp0 = mlp0
    gMlp2 = mlp2
  } else {
    g = nil
    gMlp0 = nil
    gMlp2 = nil
  }
  vec = vec.swish()
  let xAdaLNs = (0..<6).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
      name: "x_ada_ln_\($0)")
  }
  let contextAdaLNs = (0..<6).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
      name: "context_ada_ln_\($0)")
  }
  var xChunks = xAdaLNs.map { $0(vec) }
  var contextChunks = contextAdaLNs.map { $0(vec) }
  xChunks[1] = 1 + xChunks[1]
  xChunks[4] = 1 + xChunks[4]
  contextChunks[1] = 1 + contextChunks[1]
  contextChunks[4] = 1 + contextChunks[4]
  let singleAdaLNs = (0..<3).map {
    LoRADense(
      count: channels, configuration: LoRAConfiguration, noBias: true, index: 0,
      name: "single_ada_ln_\($0)")
  }
  var singleChunks = singleAdaLNs.map { $0(vec) }
  singleChunks[1] = 1 + singleChunks[1]
  let scale = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: 0, name: "ada_ln_0")
  let shift = LoRADense(
    count: channels, configuration: LoRAConfiguration, noBias: true, index: 0, name: "ada_ln_1")
  let finalChunks = [(1 + scale(vec)), shift(vec)]
  var cachedReferenceKVs = [Model.IO]()
  var mappers = [ModelWeightMapper]()
  if let referenceRot = referenceRot {
    let fixedXChunks = xChunks.map {
      $0.reshaped(
        [1, 1, channels], offset: [timesteps - 1, 0, 0], strides: [channels, channels, 1]
      ).contiguous()
    }
    let fixedSingleChunks = singleChunks.map {
      $0.reshaped(
        [1, 1, channels], offset: [timesteps - 1, 0, 0], strides: [channels, channels, 1]
      ).contiguous()
    }
    let referenceLatents = referenceOutputs.map { $0.reshaped([1, -1, channels]).to(.Float32) }
    var referenceOut = referenceLatents[0]
    if referenceLatents.count > 1 {
      referenceOut = Concat(axis: 1)(referenceLatents)
    }
    let referenceRotResized = referenceRot.reshaped([1, -1, 1, 128])
    for i in 0..<layers.0 {
      let (mapper, block) = LoRAJointTransformerBlockImageOnly(
        prefix: ("double_blocks.\(i)", "transformer_blocks.\(i)"), k: 128, h: channels / 128,
        b: 1, scaleFactor: i > layers.0 - 3 ? 8 : nil, usesFlashAttention: usesFlashAttention,
        layerIndex: i, configuration: LoRAConfiguration)
      let blockOut = block([referenceOut, referenceRotResized] + fixedXChunks)
      referenceOut = blockOut[0]
      cachedReferenceKVs.append(blockOut[1])
      cachedReferenceKVs.append(blockOut[2])
      mappers.append(mapper)
    }
    for i in 0..<layers.1 {
      let (mapper, block) = LoRASingleTransformerBlockImageOnly(
        prefix: ("single_blocks.\(i)", "single_transformer_blocks.\(i)"), k: 128,
        h: channels / 128, b: 1, contextBlockPreOnly: i == layers.1 - 1,
        usesFlashAttention: usesFlashAttention, layerIndex: i + layers.0, MLPLayerIndex: i,
        configuration: LoRAConfiguration)
      if i == layers.1 - 1 {
        let blockOut = block([
          referenceOut, referenceRotResized, fixedSingleChunks[0], fixedSingleChunks[1],
        ])
        cachedReferenceKVs.append(blockOut[0])
        cachedReferenceKVs.append(blockOut[1])
      } else {
        let blockOut = block([referenceOut, referenceRotResized] + fixedSingleChunks)
        referenceOut = blockOut[0]
        cachedReferenceKVs.append(blockOut[1])
        cachedReferenceKVs.append(blockOut[2])
        mappers.append(mapper)
      }
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      if let xEmbedder = xEmbedder {
        mapping["img_in.weight"] = [xEmbedder.weight.name]
      }
      mapping["txt_in.weight"] = [contextEmbedder.weight.name]
      mapping["time_in.in_layer.weight"] = [tMlp0.weight.name]
      mapping["time_in.out_layer.weight"] = [tMlp2.weight.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["guidance_in.in_layer.weight"] = [gMlp0.weight.name]
        mapping["guidance_in.out_layer.weight"] = [gMlp2.weight.name]
      }
      mapping[
        "double_stream_modulation_img.lin.weight"
      ] = ModelWeightElement(xAdaLNs.map { $0.weight.name })
      mapping[
        "double_stream_modulation_txt.lin.weight"
      ] = ModelWeightElement(contextAdaLNs.map { $0.weight.name })
      mapping[
        "single_stream_modulation.lin.weight"
      ] = ModelWeightElement(singleAdaLNs.map { $0.weight.name })
      mapping["final_layer.adaLN_modulation.1.weight"] = [shift.weight.name, scale.weight.name]
    case .diffusers:
      if let xEmbedder = xEmbedder {
        mapping["x_embedder.weight"] = [xEmbedder.weight.name]
      }
      mapping["context_embedder.weight"] = [contextEmbedder.weight.name]
      mapping["time_guidance_embed.timestep_embedder.linear_1.weight"] = [tMlp0.weight.name]
      mapping["time_guidance_embed.timestep_embedder.linear_2.weight"] = [tMlp2.weight.name]
      if let gMlp0 = gMlp0, let gMlp2 = gMlp2 {
        mapping["time_guidance_embed.guidance_embedder.linear_1.weight"] = [gMlp0.weight.name]
        mapping["time_guidance_embed.guidance_embedder.linear_2.weight"] = [gMlp2.weight.name]
      }
      mapping[
        "double_stream_modulation_img.linear.weight"
      ] = ModelWeightElement(xAdaLNs.map { $0.weight.name })
      mapping[
        "double_stream_modulation_txt.linear.weight"
      ] = ModelWeightElement(contextAdaLNs.map { $0.weight.name })
      mapping[
        "single_stream_modulation.linear.weight"
      ] = ModelWeightElement(singleAdaLNs.map { $0.weight.name })
      mapping["norm_out.linear.weight"] = [scale.weight.name, shift.weight.name]
    }
    return mapping
  }
  var inputs: [Input] = [contextIn] + referenceImages + [t]
  inputs.append(contentsOf: g.map { [$0] } ?? [])
  inputs.append(contentsOf: referenceRot.map { [$0] } ?? [])
  var outputs = [Model.IO]()
  if !kvCache {
    outputs.append(contentsOf: referenceOutputs)
  }
  outputs.append(context)
  outputs.append(contentsOf: xChunks)
  outputs.append(contentsOf: contextChunks)
  outputs.append(contentsOf: singleChunks)
  outputs.append(contentsOf: finalChunks)
  outputs.append(contentsOf: cachedReferenceKVs)
  return (mapper, Model(inputs, outputs, trainable: false))
}
