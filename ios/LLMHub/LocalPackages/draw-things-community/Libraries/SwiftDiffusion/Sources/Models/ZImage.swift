import DiffusionMappings
import Foundation
import NNC

public func ZImageRotaryPositionEmbedding(
  height: Int, width: Int, tokenLength: Int, imagePaddedLength: Int, heads: Int = 1
) -> Tensor<Float> {
  var rotTensor = Tensor<Float>(
    .CPU, .NHWC(1, height * width + imagePaddedLength + tokenLength, heads, 128))
  let dim0 = 32
  let dim1 = 48
  let dim2 = 48
  for y in 0..<height {
    for x in 0..<width {
      let i = y * width + x
      for j in 0..<heads {
        for k in 0..<(dim0 / 2) {
          let theta = Double(tokenLength + 1) * 1.0 / pow(256, Double(k) * 2 / Double(dim0))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, k * 2] = Float(costheta)
          rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim1 / 2) {
          let theta =
            Double(y) * 1.0 / pow(256, Double(k) * 2 / Double(dim1))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim2 / 2) {
          let theta =
            Double(x) * 1.0 / pow(256, Double(k) * 2 / Double(dim2))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  if imagePaddedLength > 0 {
    for i in (height * width)..<(height * width + imagePaddedLength) {
      for j in 0..<heads {
        for k in 0..<(dim0 / 2) {
          let theta = Double(0) * 1.0 / pow(256, Double(k) * 2 / Double(dim0))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, k * 2] = Float(costheta)
          rotTensor[0, i, j, k * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim1 / 2) {
          let theta =
            Double(0) * 1.0 / pow(256, Double(k) * 2 / Double(dim1))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
        }
        for k in 0..<(dim2 / 2) {
          let theta =
            Double(0) * 1.0 / pow(256, Double(k) * 2 / Double(dim2))
          let sintheta = sin(theta)
          let costheta = cos(theta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
          rotTensor[0, i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
        }
      }
    }
  }
  let offset = height * width + imagePaddedLength
  for i in 0..<tokenLength {
    for j in 0..<heads {
      for k in 0..<(dim0 / 2) {
        let theta = Double(i + 1) * 1.0 / pow(256, Double(k) * 2 / Double(dim0))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, offset + i, j, k * 2] = Float(costheta)
        rotTensor[0, offset + i, j, k * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim1 / 2) {
        let theta = Double(0) * 1.0 / pow(256, Double(k) * 2 / Double(dim1))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, offset + i, j, (k + (dim0 / 2)) * 2] = Float(costheta)
        rotTensor[0, offset + i, j, (k + (dim0 / 2)) * 2 + 1] = Float(sintheta)
      }
      for k in 0..<(dim2 / 2) {
        let theta = Double(0) * 1.0 / pow(256, Double(k) * 2 / Double(dim2))
        let sintheta = sin(theta)
        let costheta = cos(theta)
        rotTensor[0, offset + i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2] = Float(costheta)
        rotTensor[0, offset + i, j, (k + (dim0 / 2) + (dim1 / 2)) * 2 + 1] = Float(sintheta)
      }
    }
  }
  return rotTensor
}

private func MLPEmbedder(channels: Int, intermediateSize: Int, name: String) -> (
  Model, Model, Model
) {
  let x = Input()
  let fc0 = Dense(count: intermediateSize, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func FeedForward(
  hiddenSize: Int, intermediateSize: Int, scaleFactor: (projUp: Int, projDown: Int),
  isBF16: Bool, name: String = ""
)
  -> (
    Model, Model, Model, Model
  )
{
  let x = Input()
  let w1 = Dense(
    count: intermediateSize, noBias: true, flags: [.Float16], name: "\(name)_gate_proj")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "\(name)_up_proj")
  let gate = w1(x)
  var out = w3(x)
  if scaleFactor.projUp > 1 {
    if scaleFactor.projDown != scaleFactor.projUp {
      out = Functional.swishMul(value: out, gate: gate, beta: Float(scaleFactor.projUp), scale: Float(scaleFactor.projUp) / Float(scaleFactor.projDown))
    } else {
      out = Functional.swishMul(value: out, gate: gate, beta: Float(scaleFactor.projUp))
    }
  } else {
    if scaleFactor.projDown > 1 {
      out = Functional.swishMul(value: out, gate: gate, scale: 1 / Float(scaleFactor.projDown))
    } else {
      out = Functional.swishMul(value: out, gate: gate)
    }
  }
  let w2 = Dense(count: hiddenSize, noBias: true, name: "\(name)_down_proj")
  out = w2(out).to(.Float32)
  return (w1, w2, w3, Model([x], [out], name: name))
}

private func ZImageTransformerBlock(
  prefix: String, name: String, k: Int, h: Int, b: Int, t: (keyValue: Int, query: Int),
  segments: [Int], scaleFactor: ((qk: Int, proj: Int), (projUp: Int, projDown: Int)),
  modulation: Bool, usesFlashAttention: FlashAttentionLevel, isBF16: Bool
) -> (Model, ModelWeightMapper) {
  let x = Input()
  let chunks: [Input]
  if modulation {
    chunks = (0..<4).map { _ in Input() }
  } else {
    chunks = []
  }
  let rot = Input()
  let tokeys = Dense(count: k * h, noBias: true, name: name.isEmpty ? "k" : "\(name)_k")
  let toqueries = Dense(
    count: k * h, noBias: true, flags: [.Float16], name: name.isEmpty ? "q" : "\(name)_q")
  let tovalues = Dense(count: k * h, noBias: true, name: name.isEmpty ? "v" : "\(name)_v")
  let attentionNorm1 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "attention_norm1" : "\(name)_attention_norm_1")
  var out = attentionNorm1(x)
  if modulation {
    out = out .* chunks[0]
  } else if scaleFactor.0.qk > 1 {
    out = (1 / Float(scaleFactor.0.qk)) * out
  }
  if isBF16 {
    out = out.to(.BFloat16)
  } else {
    out = out.to(.Float16)
  }
  var keys = tokeys(out)
  keys = (isBF16 ? keys.to(.Float32) : keys).reshaped([
    b, t.keyValue, h, k,
  ])
  var queries: Model.IO
  if t.keyValue != t.query {
    queries = toqueries(
      out.reshaped(
        [b, t.query, k * h], offset: [0, 0, 0], strides: [t.0 * h * k, h * k, 1])
    )
    queries = (isBF16 ? queries.to(.Float32) : queries).reshaped([
      b, t.query, h, k,
    ])
  } else {
    queries = toqueries(out)
    queries = (isBF16 ? queries.to(.Float32) : queries).reshaped([b, t.0, h, k])
  }
  let normK = RMSNorm(
    epsilon: 1e-5 / Float(scaleFactor.0.qk * scaleFactor.0.qk), axis: [3],
    name: name.isEmpty ? "norm_k" : "\(name)_norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(
    epsilon: 1e-5 / Float(scaleFactor.0.qk * scaleFactor.0.qk), axis: [3],
    name: name.isEmpty ? "norm_q" : "\(name)_norm_q")
  queries = normQ(queries)
  if scaleFactor.0.proj > 1 {
    out = (1 / Float(scaleFactor.0.proj)) * out
  }
  var values = tovalues(out)
  values = values.reshaped([b, t.0, h, k])
  let rotaryEmbedding = isBF16 ? rot.to(.Float32) : rot
  if t.keyValue != t.query {
    queries = Functional.cmul(
      left: queries,
      right: rotaryEmbedding.reshaped(
        [1, t.query, 1, k], offset: [0, 0, 0, 0], strides: [t.keyValue * k, k, k, 1]))
  } else {
    queries = Functional.cmul(left: queries, right: rotaryEmbedding)
  }
  keys = Functional.cmul(left: keys, right: rotaryEmbedding)
  if isBF16 {
    queries = queries.to(.BFloat16)
    keys = keys.to(.BFloat16)
  }
  switch usesFlashAttention {
  case .scale1:
    queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1.0 / Float(k).squareRoot().squareRoot()) * keys
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1]
        ).contiguous()
        let key = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        let value = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        outs.append(scaledDotProductAttention(query, key, value))
        offset += segment
      }
      let concat = Concat(axis: 1)
      concat.flags = .disableOpt
      out = concat(outs)
    } else {
      out = scaledDotProductAttention(
        queries, keys, values
      )
    }
  case .scaleMerged, .quantized:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1]
        ).contiguous()
        let key = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        let value = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        outs.append(scaledDotProductAttention(query, key, value))
        offset += segment
      }
      let concat = Concat(axis: 1)
      concat.flags = .disableOpt
      out = concat(outs)
    } else {
      out = scaledDotProductAttention(
        queries, keys, values
      )
    }
  case .none:
    queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1.0 / Float(k).squareRoot().squareRoot()) * keys
    if segments.count > 1 {
      var offset = 0
      var finalOuts = [Model.IO]()
      for segment in segments {
        var subQueries = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1]
        ).contiguous()
        var subKeys = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        var subValues = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        subKeys = subKeys.transposed(1, 2)
        subQueries = subQueries.transposed(1, 2)
        subValues = subValues.transposed(1, 2)
        var outs = [Model.IO]()
        for i in 0..<(b * h) {
          let key = subKeys.reshaped(
            [1, segment, k], offset: [i, 0, 0], strides: [segment * k, k, 1])
          let query = subQueries.reshaped(
            [1, segment, k], offset: [i, 0, 0], strides: [segment * k, k, 1])
          let value = subValues.reshaped(
            [1, segment, k], offset: [i, 0, 0], strides: [segment * k, k, 1])
          var dot = Matmul(transposeB: (1, 2))(query, key)
          if let last = outs.last {
            dot.add(dependencies: [last])
          }
          dot = dot.reshaped([segment, segment])
          dot = dot.softmax()
          dot = dot.reshaped([1, segment, segment])
          outs.append(dot * value)
        }
        finalOuts.append(Concat(axis: 0)(outs).reshaped([b, h, segment, k]).transposed(1, 2))
        offset += segment
      }
      out = Concat(axis: 1)(finalOuts)
    } else {
      keys = keys.transposed(1, 2)
      queries = queries.transposed(1, 2)
      values = values.transposed(1, 2)
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped(
          [1, t.keyValue, k], offset: [i, 0, 0], strides: [t.keyValue * k, k, 1])
        let query = queries.reshaped(
          [1, t.query, k], offset: [i, 0, 0], strides: [t.query * k, k, 1])
        let value = values.reshaped(
          [1, t.keyValue, k], offset: [i, 0, 0], strides: [t.keyValue * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t.query, t.keyValue])
        dot = dot.softmax()
        dot = dot.reshaped([1, t.query, t.keyValue])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs).reshaped([b, h, t.query, k]).transposed(1, 2)
    }
  }
  let xIn: Model.IO
  if t.keyValue > t.query {
    xIn = x.reshaped(
      [b, t.query, h * k], offset: [0, 0, 0], strides: [t.keyValue * h * k, h * k, 1])
    out = out.reshaped([b, t.query, h * k])
  } else {
    xIn = x
    out = out.reshaped([b, t.0, h * k])
  }
  let unifyheads = Dense(count: k * h, noBias: true, name: name.isEmpty ? "o" : "\(name)_o")
  out = unifyheads(out).to(of: xIn)
  var attentionNorm2Epsilon: Float = 1e-5
  if scaleFactor.0.proj * scaleFactor.0.qk > 1 {
    attentionNorm2Epsilon =
      attentionNorm2Epsilon
      / Float(scaleFactor.0.proj * scaleFactor.0.qk * scaleFactor.0.proj * scaleFactor.0.qk)
  }
  let attentionNorm2 = RMSNorm(
    epsilon: attentionNorm2Epsilon, axis: [2],
    name: name.isEmpty ? "attention_norm2" : "\(name)_attention_norm2")
  out = attentionNorm2(out)
  if modulation {
    out = out .* chunks[1]
  }
  out = xIn + out
  let (w1, w2, w3, ffn) = FeedForward(
    hiddenSize: h * k, intermediateSize: 10_240, scaleFactor: scaleFactor.1,
    isBF16: isBF16, name: name.isEmpty ? "ffn" : "\(name)_ffn")
  let feedForwardNorm1 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "ffn_norm1" : "\(name)_ffn_norm1")
  var feedForwardNorm2Epsilon: Float = 1e-5
  if scaleFactor.1.projUp * scaleFactor.1.projDown > 1 {
    feedForwardNorm2Epsilon =
      feedForwardNorm2Epsilon
      / Float(
        scaleFactor.1.projUp * scaleFactor.1.projDown * scaleFactor.1.projUp
          * scaleFactor.1.projDown)
  }
  let feedForwardNorm2 = RMSNorm(
    epsilon: feedForwardNorm2Epsilon, axis: [2],
    name: name.isEmpty ? "ffn_norm2" : "\(name)_ffn_norm2")
  let residual = out
  out = feedForwardNorm1(out)
  if modulation {
    out = out .* chunks[2]
  } else if scaleFactor.1.projUp > 1 {
    out = (1.0 / Float(scaleFactor.1.projUp)) * out
  }
  out = isBF16 ? out.to(.BFloat16) : out.to(.Float16)
  out = feedForwardNorm2(ffn(out))  // Already converted to Float32.
  if modulation {
    out = out .* chunks[3]
  }
  out = residual + out
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      if isBF16 {
        mapping["\(prefix).attention.qkv.weight"] = ModelWeightElement(
          [toqueries.weight.name, tokeys.weight.name, tovalues.weight.name], isBF16: true)
      } else {
        mapping["\(prefix).attention.qkv.weight"] = [
          toqueries.weight.name, tokeys.weight.name, tovalues.weight.name,
        ]
      }
      mapping["\(prefix).attention.q_norm.weight"] = [normQ.weight.name]
      mapping["\(prefix).attention.k_norm.weight"] = [normK.weight.name]
      mapping["\(prefix).attention.out.weight"] =
        isBF16
        ? ModelWeightElement([unifyheads.weight.name], isBF16: true)
        : [
          unifyheads.weight.name
        ]
      mapping["\(prefix).attention_norm1.weight"] = [attentionNorm1.weight.name]
      mapping["\(prefix).attention_norm2.weight"] = [attentionNorm2.weight.name]
      mapping["\(prefix).feed_forward.w1.weight"] =
        isBF16 ? ModelWeightElement([w1.weight.name], isBF16: true) : [w1.weight.name]
      mapping["\(prefix).feed_forward.w2.weight"] =
        isBF16 ? ModelWeightElement([w2.weight.name], isBF16: true) : [w2.weight.name]
      mapping["\(prefix).feed_forward.w3.weight"] =
        isBF16 ? ModelWeightElement([w3.weight.name], isBF16: true) : [w3.weight.name]
      mapping["\(prefix).ffn_norm1.weight"] = [feedForwardNorm1.weight.name]
      mapping["\(prefix).ffn_norm2.weight"] = [feedForwardNorm2.weight.name]
    case .diffusers:
      mapping["\(prefix).attention.to_q.weight"] =
        isBF16
        ? ModelWeightElement([toqueries.weight.name], isBF16: true)
        : [
          toqueries.weight.name
        ]
      mapping["\(prefix).attention.to_k.weight"] =
        isBF16
        ? ModelWeightElement([tokeys.weight.name], isBF16: true)
        : [
          tokeys.weight.name
        ]
      mapping["\(prefix).attention.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix).attention.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix).attention.to_v.weight"] =
        isBF16
        ? ModelWeightElement([tovalues.weight.name], isBF16: true)
        : [
          tovalues.weight.name
        ]
      mapping["\(prefix).attention.to_out.0.weight"] =
        isBF16
        ? ModelWeightElement([unifyheads.weight.name], isBF16: true)
        : [
          unifyheads.weight.name
        ]
      mapping["\(prefix).attention_norm1.weight"] = [attentionNorm1.weight.name]
      mapping["\(prefix).attention_norm2.weight"] = [attentionNorm2.weight.name]
      mapping["\(prefix).feed_forward.w1.weight"] =
        isBF16 ? ModelWeightElement([w1.weight.name], isBF16: true) : [w1.weight.name]
      mapping["\(prefix).feed_forward.w2.weight"] =
        isBF16 ? ModelWeightElement([w2.weight.name], isBF16: true) : [w2.weight.name]
      mapping["\(prefix).feed_forward.w3.weight"] =
        isBF16 ? ModelWeightElement([w3.weight.name], isBF16: true) : [w3.weight.name]
      mapping["\(prefix).ffn_norm1.weight"] = [feedForwardNorm1.weight.name]
      mapping["\(prefix).ffn_norm2.weight"] = [feedForwardNorm2.weight.name]
    }
    return mapping
  }
  return (Model([x, rot] + chunks, [out]), mapper)
}

public func ZImage(
  batchSize: Int, height: Int, width: Int, textLength: Int, channels: Int, layers: Int,
  activationQkScaling: [Int: Int], activationProjScaling: [Int: Int],
  activationFfnProjUpScaling: [Int: Int], activationFfnScaling: [Int: Int],
  usesFlashAttention: FlashAttentionLevel, isBF16: Bool
) -> (
  Model, ModelWeightMapper
) {
  let x = Input()
  let rot = Input()
  let txtIn = Input()
  let imgIn = Dense(count: channels, name: "x_embedder")
  let h = height / 2
  let w = width / 2
  var xOut = imgIn(
    x.reshaped([batchSize, h, 2, w, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous()
      .reshaped([batchSize, h * w, 2 * 2 * 16], format: .NHWC)
  )
  var mappers = [ModelWeightMapper]()
  var adaLNChunks = [Input]()
  let roundUpHW = (h * w + 31) / 32 * 32
  let xPadTokens: Input?
  if roundUpHW != h * w {
    let padTokens = Input()
    xOut = Functional.concat(axis: 1, xOut, padTokens, flags: .disableOpt)
    xPadTokens = padTokens
  } else {
    xPadTokens = nil
  }
  xOut = xOut.to(.Float32)
  let xRot = rot.reshaped([1, roundUpHW, 1, 128])
  for i in 0..<2 {
    let qkScaling = activationQkScaling[i + 2] ?? 1
    var projScaling = activationProjScaling[i + 2] ?? 1
    if projScaling < 0 {
      projScaling = 4 / -projScaling
    } else {
      projScaling = 4 * projScaling
    }
    let ffnProjUpScaling = activationFfnProjUpScaling[i + 2] ?? 1
    var ffnScaling = activationFfnScaling[i + 2] ?? 1
    if ffnScaling < 0 {
      ffnScaling = 32 / -ffnScaling
    } else {
      ffnScaling = 32 * ffnScaling
    }
    let chunks = (0..<4).map { _ in Input() }
    let (block, mapper) = ZImageTransformerBlock(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", k: 128, h: channels / 128, b: batchSize,
      t: (keyValue: roundUpHW, query: roundUpHW), segments: [],
      scaleFactor: (
        (qk: qkScaling, proj: projScaling), (projUp: ffnProjUpScaling, projDown: ffnScaling)
      ),
      modulation: true, usesFlashAttention: usesFlashAttention, isBF16: isBF16)
    xOut = block([xOut, xRot] + chunks)
    adaLNChunks.append(contentsOf: chunks)
    mappers.append(mapper)
  }
  var out = Functional.concat(axis: 1, xOut, txtIn)
  let rotResized = rot.reshaped(.NHWC(1, roundUpHW + textLength, 1, 128))
  for i in 0..<layers {
    let qkScaling = activationQkScaling[i + 4] ?? 1
    var projScaling = activationProjScaling[i + 4] ?? 1
    if !isBF16 {  // Only have default value for non-BF16 path.
      if projScaling < 0 {
        projScaling = 4 / -projScaling
      } else {
        projScaling = 4 * projScaling
      }
    }
    let ffnProjUpScaling = activationFfnProjUpScaling[i + 4] ?? 1
    var ffnScaling = activationFfnScaling[i + 4] ?? 1
    if !isBF16 {  // Only have default value for non-BF16 path.
      if ffnScaling < 0 {
        ffnScaling = 32 / -ffnScaling
      } else {
        ffnScaling = 32 * ffnScaling
      }
    }
    let chunks = (0..<4).map { _ in Input() }
    let (block, mapper) = ZImageTransformerBlock(
      prefix: "layers.\(i)", name: "", k: 128, h: channels / 128, b: batchSize,
      t: (
        keyValue: roundUpHW + textLength, query: i == layers - 1 ? h * w : roundUpHW + textLength
      ),
      segments: [],
      scaleFactor: (
        (qk: qkScaling, proj: projScaling), (projUp: ffnProjUpScaling, projDown: ffnScaling)
      ), modulation: true, usesFlashAttention: usesFlashAttention, isBF16: isBF16)
    out = block([out, rotResized] + chunks)
    adaLNChunks.append(contentsOf: chunks)
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let scale = Input()
  adaLNChunks.append(scale)
  let projOut = Dense(count: 2 * 2 * 16, name: "linear_final")
  out = normFinal(out).to(.Float16) .* scale
  out = (-projOut(out)).reshaped([batchSize, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous()
    .reshaped([
      batchSize, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["x_embedder.weight"] = [imgIn.weight.name]
      mapping["x_embedder.bias"] = [imgIn.bias.name]
      mapping["final_layer.linear.weight"] = [projOut.weight.name]
      mapping["final_layer.linear.bias"] = [projOut.bias.name]
    case .diffusers:
      mapping["all_x_embedder.2-1.weight"] = [imgIn.weight.name]
      mapping["all_x_embedder.2-1.bias"] = [imgIn.bias.name]
      mapping["all_final_layer.2-1.linear.weight"] = [projOut.weight.name]
      mapping["all_final_layer.2-1.linear.bias"] = [projOut.bias.name]
    }
    return mapping
  }
  return (Model([x] + (xPadTokens.map { [$0] } ?? []) + [rot, txtIn] + adaLNChunks, [out]), mapper)
}

private func ZImageTransformerBlockFixed(
  prefix: String, name: String, channels: Int,
  scaleFactor: ((qk: Int, proj: Int), (projUp: Int, projDown: Int))
) -> (Model, ModelWeightMapper) {
  let tEmbed = Input()
  let adaLNs = (0..<4).map {
    Dense(count: channels, name: name.isEmpty ? "ada_ln_\($0)" : "\(name)_ada_ln_\($0)")
  }
  var chunks = adaLNs.map { $0(tEmbed) }
  chunks[0] = (1 + chunks[0]).to(.Float32)
  if scaleFactor.0.qk > 1 {
    chunks[0] = (1.0 / Float(scaleFactor.0.qk)) * chunks[0]
  }
  chunks[1] = chunks[1].tanh().to(.Float32)
  chunks[2] = (1 + chunks[2]).to(.Float32)
  if scaleFactor.1.projUp > 1 {
    chunks[2] = (1.0 / Float(scaleFactor.1.projUp)) * chunks[2]
  }
  chunks[3] = chunks[3].tanh().to(.Float32)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).adaLN_modulation.0.weight"] = ModelWeightElement(
      (0..<4).map { adaLNs[$0].weight.name })
    mapping["\(prefix).adaLN_modulation.0.bias"] = ModelWeightElement(
      (0..<4).map { adaLNs[$0].bias.name })
    return mapping
  }
  return (Model([tEmbed], chunks), mapper)
}

public func ZImageFixed(
  batchSize: Int, tokenLength: (Int, Int), channels: Int, layers: Int,
  activationQkScaling: [Int: Int], activationProjScaling: [Int: Int],
  activationFfnProjUpScaling: [Int: Int], activationFfnScaling: [Int: Int],
  usesFlashAttention: FlashAttentionLevel, isBF16: Bool
) -> (
  Model, ModelWeightMapper
) {
  let txt = Input()
  let txtRot = Input()
  let t = Input()
  let txtNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "cap_norm")
  let txtIn = Dense(count: channels, name: "cap_embedder")
  var txtOut = txtIn(txtNorm(txt))
  let roundUpTokenLength = ((tokenLength.0 + 31) / 32 * 32, (tokenLength.1 + 31) / 32 * 32)
  let capPadTokens: Input?
  if roundUpTokenLength.0 != tokenLength.0 || roundUpTokenLength.1 != tokenLength.1 {
    let padTokens = Input()
    if tokenLength.0 > 0 {
      var txtOut0 = txtOut.reshaped(
        [batchSize, tokenLength.0, channels], offset: [0, 0, 0],
        strides: [(tokenLength.0 + tokenLength.1) * channels, channels, 1]
      ).contiguous()
      if roundUpTokenLength.0 != tokenLength.0 {
        let padTokens0 = padTokens.reshaped(
          [batchSize, roundUpTokenLength.0 - tokenLength.0, channels], offset: [0, 0, 0],
          strides: [
            (roundUpTokenLength.0 - tokenLength.0 + roundUpTokenLength.1 - tokenLength.1)
              * channels, channels, 1,
          ]
        ).contiguous()
        txtOut0 = Functional.concat(axis: 1, txtOut0, padTokens0, flags: .disableOpt)
      }
      var txtOut1 = txtOut.reshaped(
        [batchSize, tokenLength.1, channels], offset: [0, tokenLength.0, 0],
        strides: [(tokenLength.0 + tokenLength.1) * channels, channels, 1]
      ).contiguous()
      if roundUpTokenLength.1 != tokenLength.1 {
        let padTokens1 = padTokens.reshaped(
          [batchSize, roundUpTokenLength.1 - tokenLength.1, channels],
          offset: [0, roundUpTokenLength.0 - tokenLength.0, 0],
          strides: [
            (roundUpTokenLength.0 - tokenLength.0 + roundUpTokenLength.1 - tokenLength.1)
              * channels, channels, 1,
          ]
        ).contiguous()
        txtOut1 = Functional.concat(axis: 1, txtOut1, padTokens1, flags: .disableOpt)
      }
      txtOut = Functional.concat(axis: 1, txtOut0, txtOut1, flags: .disableOpt)
    } else {
      if roundUpTokenLength.1 != tokenLength.1 {
        txtOut = Functional.concat(axis: 1, txtOut, padTokens, flags: .disableOpt)
      }
    }
    capPadTokens = padTokens
  } else {
    capPadTokens = nil
  }
  txtOut = txtOut.to(.Float32)
  let (timeInMlp0, timeInMlp2, timeIn) = MLPEmbedder(
    channels: 256, intermediateSize: 1024, name: "t")
  let tOut = timeIn(t)
  var mappers = [ModelWeightMapper]()
  let segments: [Int]
  if roundUpTokenLength.0 > 0 {
    segments = [roundUpTokenLength.0, roundUpTokenLength.1]
  } else {
    segments = []
  }
  for i in 0..<2 {
    let qkScaling = activationQkScaling[i] ?? 1
    let projScaling = activationProjScaling[i] ?? 1
    let ffnProjUpScaling = activationFfnProjUpScaling[i] ?? 1
    let ffnScaling = activationFfnScaling[i] ?? 1
    let (block, mapper) = ZImageTransformerBlock(
      prefix: "context_refiner.\(i)", name: "context_refiner", k: 128, h: channels / 128,
      b: batchSize,
      t: (
        keyValue: roundUpTokenLength.0 + roundUpTokenLength.1,
        query: roundUpTokenLength.0 + roundUpTokenLength.1
      ),
      segments: segments,
      scaleFactor: (
        (qk: qkScaling, proj: 2 * projScaling), (projUp: ffnProjUpScaling, projDown: 2 * ffnScaling)
      ), modulation: false,
      usesFlashAttention: usesFlashAttention, isBF16: isBF16)
    txtOut = block(txtOut, txtRot)
    mappers.append(mapper)
  }
  var outs = [txtOut]
  for i in 0..<2 {
    let qkScaling = activationQkScaling[i + 2] ?? 1
    var projScaling = activationProjScaling[i + 2] ?? 1
    if projScaling < 0 {
      projScaling = 4 / -projScaling
    } else {
      projScaling = 4 * projScaling
    }
    let ffnProjUpScaling = activationFfnProjUpScaling[i + 2] ?? 1
    var ffnScaling = activationFfnScaling[i + 2] ?? 1
    if ffnScaling < 0 {
      ffnScaling = 32 / -ffnScaling
    } else {
      ffnScaling = 32 * ffnScaling
    }
    let (block, mapper) = ZImageTransformerBlockFixed(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", channels: channels,
      scaleFactor: (
        (qk: qkScaling, proj: projScaling), (projUp: ffnProjUpScaling, projDown: ffnScaling)
      ))
    outs.append(block(tOut))
    mappers.append(mapper)
  }
  for i in 0..<layers {
    let qkScaling = activationQkScaling[i + 4] ?? 1
    var projScaling = activationProjScaling[i + 4] ?? 1
    if projScaling < 0 {
      projScaling = 4 / -projScaling
    } else {
      projScaling = 4 * projScaling
    }
    let ffnProjUpScaling = activationFfnProjUpScaling[i + 4] ?? 1
    var ffnScaling = activationFfnScaling[i + 4] ?? 1
    if ffnScaling < 0 {
      ffnScaling = 32 / -ffnScaling
    } else {
      ffnScaling = 32 * ffnScaling
    }
    let (block, mapper) = ZImageTransformerBlockFixed(
      prefix: "layers.\(i)", name: "", channels: channels,
      scaleFactor: (
        (qk: qkScaling, proj: projScaling), (projUp: ffnProjUpScaling, projDown: ffnScaling)
      ))
    outs.append(block(tOut))
    mappers.append(mapper)
  }
  let scale = Dense(count: channels, name: "ada_ln_final")
  outs.append(1 + scale(tOut.swish()))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["cap_embedder.0.weight"] = [txtNorm.weight.name]
    mapping["cap_embedder.1.weight"] = [txtIn.weight.name]
    mapping["cap_embedder.1.bias"] = [txtIn.bias.name]
    mapping["t_embedder.mlp.0.weight"] = [timeInMlp0.weight.name]
    mapping["t_embedder.mlp.0.bias"] = [timeInMlp0.bias.name]
    mapping["t_embedder.mlp.2.weight"] = [timeInMlp2.weight.name]
    mapping["t_embedder.mlp.2.bias"] = [timeInMlp2.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["final_layer.adaLN_modulation.1.weight"] = [scale.weight.name]
      mapping["final_layer.adaLN_modulation.1.bias"] = [scale.bias.name]
    case .diffusers:
      mapping["all_final_layer.2-1.adaLN_modulation.1.weight"] = [scale.weight.name]
      mapping["all_final_layer.2-1.adaLN_modulation.1.bias"] = [scale.bias.name]
    }
    return mapping
  }
  return (Model([txt, txtRot, t] + (capPadTokens.map { [$0] } ?? []), outs), mapper)
}

private func ZImageTransformerBlockFixedOutputShapes(
  prefix: String, name: String, batchSize: Int, channels: Int
) -> [TensorShape] {
  return (0..<4).map { _ in
    TensorShape([batchSize, 1, channels])
  }
}

public func ZImageFixedOutputShapes(
  batchSize: Int, tokenLength: Int, channels: Int, layers: Int
) -> [TensorShape] {
  var outs = [TensorShape]()
  outs.append(TensorShape([batchSize, tokenLength, channels]))
  for i in 0..<2 {
    let outputShapes = ZImageTransformerBlockFixedOutputShapes(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", batchSize: batchSize, channels: channels)
    outs.append(contentsOf: outputShapes)
  }
  for i in 0..<layers {
    let outputShapes = ZImageTransformerBlockFixedOutputShapes(
      prefix: "layers.\(i)", name: "", batchSize: batchSize, channels: channels)
    outs.append(contentsOf: outputShapes)
  }
  outs.append(contentsOf: [
    TensorShape([batchSize, 1, channels])
  ])
  return outs
}

private func LoRAMLPEmbedder(
  channels: Int, intermediateSize: Int, configuration: LoRANetworkConfiguration, index: Int,
  name: String
) -> (
  Model, Model, Model
) {
  let x = Input()
  let fc0 = LoRADense(
    count: intermediateSize, configuration: configuration, index: index, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = LoRADense(
    count: channels, configuration: configuration, index: index, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LoRAFeedForward(
  hiddenSize: Int, intermediateSize: Int, scaleFactor: (projUp: Int, projDown: Int),
  isBF16: Bool, configuration: LoRANetworkConfiguration, index: Int, name: String = ""
)
  -> (
    Model, Model, Model, Model
  )
{
  let x = Input()
  let w1 = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, flags: [.Float16],
    prefix: name, index: index,
    name: "\(name)_gate_proj")
  let w3 = LoRADense(
    count: intermediateSize, configuration: configuration, noBias: true, prefix: name, index: index,
    name: "\(name)_up_proj")
  let gate = w1(x)
  var out = w3(x)
  if scaleFactor.projUp > 1 {
    if scaleFactor.projDown != scaleFactor.projUp {
      out = Functional.swishMul(value: out, gate: gate, beta: Float(scaleFactor.projUp), scale: Float(scaleFactor.projUp) / Float(scaleFactor.projDown))
    } else {
      out = Functional.swishMul(value: out, gate: gate, beta: Float(scaleFactor.projUp))
    }
  } else {
    if scaleFactor.projDown > 1 {
      out = Functional.swishMul(value: out, gate: gate, scale: 1 / Float(scaleFactor.projDown))
    } else {
      out = Functional.swishMul(value: out, gate: gate)
    }
  }
  let w2 = LoRADense(
    count: hiddenSize, configuration: configuration, noBias: true, prefix: name, index: index,
    name: "\(name)_down_proj")
  out = w2(out).to(.Float32)
  return (w1, w2, w3, Model([x], [out], name: name))
}

private func LoRAZImageTransformerBlock(
  prefix: String, name: String, k: Int, h: Int, b: Int, t: (keyValue: Int, query: Int),
  segments: [Int], scaleFactor: ((qk: Int, proj: Int), (projUp: Int, projDown: Int)),
  modulation: Bool, usesFlashAttention: FlashAttentionLevel, isBF16: Bool, layerIndex: Int,
  configuration: LoRANetworkConfiguration, rotaryEmbeddingHeads: Int = 1
) -> (Model, ModelWeightMapper) {
  let x = Input()
  let chunks: [Input]
  if modulation {
    chunks = (0..<4).map { _ in Input() }
  } else {
    chunks = []
  }
  let rot = Input()
  let tokeys = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex,
    name: name.isEmpty ? "k" : "\(name)_k")
  let toqueries = LoRADense(
    count: k * h, configuration: configuration, noBias: true, flags: [.Float16], index: layerIndex,
    name: name.isEmpty ? "q" : "\(name)_q")
  let tovalues = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex,
    name: name.isEmpty ? "v" : "\(name)_v")
  let attentionNorm1 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "attention_norm1" : "\(name)_attention_norm_1")
  var out = attentionNorm1(x)
  if modulation {
    out = out .* chunks[0]
  } else if scaleFactor.0.qk > 1 {
    out = (1 / Float(scaleFactor.0.qk)) * out
  }
  if isBF16 {
    out = out.to(.BFloat16)
  } else {
    out = out.to(.Float16)
  }
  var keys = tokeys(out)
  keys = (isBF16 ? keys.to(.Float32) : keys).reshaped([
    b, t.keyValue, h, k,
  ])
  var queries: Model.IO
  if t.keyValue != t.query {
    let queryIn = out.reshaped(
      [b, t.query, k * h], offset: [0, 0, 0], strides: [t.0 * h * k, h * k, 1]
    ).contiguous()
    queries = toqueries(queryIn)
    queries = (isBF16 ? queries.to(.Float32) : queries).reshaped([
      b, t.query, h, k,
    ])
  } else {
    queries = toqueries(out)
    queries = (isBF16 ? queries.to(.Float32) : queries).reshaped([b, t.0, h, k])
  }
  let normK = RMSNorm(
    epsilon: 1e-5 / Float(scaleFactor.0.qk * scaleFactor.0.qk), axis: [3],
    name: name.isEmpty ? "norm_k" : "\(name)_norm_k")
  keys = normK(keys)
  let normQ = RMSNorm(
    epsilon: 1e-5 / Float(scaleFactor.0.qk * scaleFactor.0.qk), axis: [3],
    name: name.isEmpty ? "norm_q" : "\(name)_norm_q")
  queries = normQ(queries)
  if scaleFactor.0.proj > 1 {
    out = (1 / Float(scaleFactor.0.proj)) * out
  }
  var values = tovalues(out)
  values = values.reshaped([b, t.0, h, k])
  let rotaryEmbedding = isBF16 ? rot.to(.Float32) : rot
  if t.keyValue != t.query {
    if rotaryEmbeddingHeads > 1 {
      let queryRot = rotaryEmbedding.reshaped(
        [1, t.query, h, k], offset: [0, 0, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
      ).contiguous()
      queries = Functional.cmul(left: queries, right: queryRot)
    } else {
      let queryRot = rotaryEmbedding.reshaped(
        [1, t.query, 1, k], offset: [0, 0, 0, 0], strides: [t.keyValue * k, k, k, 1]
      )
      queries = Functional.cmul(left: queries, right: queryRot)
    }
  } else {
    queries = Functional.cmul(left: queries, right: rotaryEmbedding)
  }
  keys = Functional.cmul(left: keys, right: rotaryEmbedding)
  if isBF16 {
    queries = queries.to(.BFloat16)
    keys = keys.to(.BFloat16)
  }
  switch usesFlashAttention {
  case .scale1:
    queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1.0 / Float(k).squareRoot().squareRoot()) * keys
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Float16])
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1]
        ).contiguous()
        let key = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        let value = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        outs.append(scaledDotProductAttention(query, key, value))
        offset += segment
      }
      let concat = Concat(axis: 1)
      concat.flags = .disableOpt
      out = concat(outs)
    } else {
      out = scaledDotProductAttention(
        queries, keys, values
      )
    }
    scaledDotProductAttention.gradientCheckpointing = false
  case .quantized:
    let scaledDotProductAttention: Model
    if configuration.testing {
      scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(), flags: [.Int8, .Float16])
    } else {
      queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
      keys = (1.0 / Float(k).squareRoot().squareRoot()) * keys
      scaledDotProductAttention = ScaledDotProductAttention(scale: 1, flags: [.Int8, .Float16])
    }
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1]
        ).contiguous()
        let key = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        let value = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        outs.append(scaledDotProductAttention(query, key, value))
        offset += segment
      }
      let concat = Concat(axis: 1)
      concat.flags = .disableOpt
      out = concat(outs)
    } else {
      out = scaledDotProductAttention(
        queries, keys, values
      )
    }
    scaledDotProductAttention.gradientCheckpointing = false
  case .scaleMerged:
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), flags: [.Float16])
    if segments.count > 1 {
      var offset = 0
      var outs = [Model.IO]()
      for segment in segments {
        let query = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1]
        ).contiguous()
        let key = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        let value = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        outs.append(scaledDotProductAttention(query, key, value))
        offset += segment
      }
      let concat = Concat(axis: 1)
      concat.flags = .disableOpt
      out = concat(outs)
    } else {
      out = scaledDotProductAttention(
        queries, keys, values
      )
    }
    scaledDotProductAttention.gradientCheckpointing = false
  case .none:
    queries = (1.0 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1.0 / Float(k).squareRoot().squareRoot()) * keys
    if segments.count > 1 {
      var offset = 0
      var finalOuts = [Model.IO]()
      for segment in segments {
        var subQueries = queries.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.query * h * k, h * k, k, 1]
        ).contiguous()
        var subKeys = keys.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        var subValues = values.reshaped(
          [b, segment, h, k], offset: [0, offset, 0, 0], strides: [t.keyValue * h * k, h * k, k, 1]
        ).contiguous()
        subKeys = subKeys.transposed(1, 2)
        subQueries = subQueries.transposed(1, 2)
        subValues = subValues.transposed(1, 2)
        var outs = [Model.IO]()
        for i in 0..<(b * h) {
          let key = subKeys.reshaped(
            [1, segment, k], offset: [i, 0, 0], strides: [segment * k, k, 1])
          let query = subQueries.reshaped(
            [1, segment, k], offset: [i, 0, 0], strides: [segment * k, k, 1])
          let value = subValues.reshaped(
            [1, segment, k], offset: [i, 0, 0], strides: [segment * k, k, 1])
          var dot = Matmul(transposeB: (1, 2))(query, key)
          if let last = outs.last {
            dot.add(dependencies: [last])
          }
          dot = dot.reshaped([segment, segment])
          dot = dot.softmax()
          dot = dot.reshaped([1, segment, segment])
          outs.append(dot * value)
        }
        finalOuts.append(Concat(axis: 0)(outs).reshaped([b, h, segment, k]).transposed(1, 2))
        offset += segment
      }
      out = Concat(axis: 1)(finalOuts)
    } else {
      keys = keys.transposed(1, 2)
      queries = queries.transposed(1, 2)
      values = values.transposed(1, 2)
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped(
          [1, t.keyValue, k], offset: [i, 0, 0], strides: [t.keyValue * k, k, 1])
        let query = queries.reshaped(
          [1, t.query, k], offset: [i, 0, 0], strides: [t.query * k, k, 1])
        let value = values.reshaped(
          [1, t.keyValue, k], offset: [i, 0, 0], strides: [t.keyValue * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([t.query, t.keyValue])
        dot = dot.softmax()
        dot = dot.reshaped([1, t.query, t.keyValue])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs).reshaped([b, h, t.query, k]).transposed(1, 2)
    }
  }
  let xIn: Model.IO
  if t.keyValue > t.query {
    xIn = x.reshaped(
      [b, t.query, h * k], offset: [0, 0, 0], strides: [t.keyValue * h * k, h * k, 1])
    out = out.reshaped([b, t.query, h * k])
  } else {
    xIn = x
    out = out.reshaped([b, t.0, h * k])
  }
  let unifyheads = LoRADense(
    count: k * h, configuration: configuration, noBias: true, index: layerIndex,
    name: name.isEmpty ? "o" : "\(name)_o")
  out = unifyheads(out).to(of: xIn)
  var attentionNorm2Epsilon: Float = 1e-5
  if scaleFactor.0.proj * scaleFactor.0.qk > 1 {
    attentionNorm2Epsilon =
      attentionNorm2Epsilon
      / Float(scaleFactor.0.proj * scaleFactor.0.qk * scaleFactor.0.proj * scaleFactor.0.qk)
  }
  let attentionNorm2 = RMSNorm(
    epsilon: attentionNorm2Epsilon, axis: [2],
    name: name.isEmpty ? "attention_norm2" : "\(name)_attention_norm2")
  out = attentionNorm2(out)
  if modulation {
    out = out .* chunks[1]
  } else if scaleFactor.1.projUp > 1 {
    out = (1.0 / Float(scaleFactor.1.projUp)) * out
  }
  out = xIn + out
  let (w1, w2, w3, ffn) = LoRAFeedForward(
    hiddenSize: h * k, intermediateSize: 10_240, scaleFactor: scaleFactor.1,
    isBF16: isBF16, configuration: configuration, index: layerIndex,
    name: name.isEmpty ? "ffn" : "\(name)_ffn")
  if configuration.gradientCheckpointingFeedForward {
    ffn.gradientCheckpointing = true
  }
  let feedForwardNorm1 = RMSNorm(
    epsilon: 1e-5, axis: [2], name: name.isEmpty ? "ffn_norm1" : "\(name)_ffn_norm1")
  var feedForwardNorm2Epsilon: Float = 1e-5
  if scaleFactor.1.projUp * scaleFactor.1.projDown > 1 {
    feedForwardNorm2Epsilon =
      feedForwardNorm2Epsilon
      / Float(
        scaleFactor.1.projUp * scaleFactor.1.projDown * scaleFactor.1.projUp
          * scaleFactor.1.projDown)
  }
  let feedForwardNorm2 = RMSNorm(
    epsilon: feedForwardNorm2Epsilon, axis: [2],
    name: name.isEmpty ? "ffn_norm2" : "\(name)_ffn_norm2")
  let residual = out
  out = feedForwardNorm1(out)
  if modulation {
    out = out .* chunks[2]
  }
  out = isBF16 ? out.to(.BFloat16) : out.to(.Float16)
  out = feedForwardNorm2(ffn(out))  // Already converted to Float32.
  if modulation {
    out = out .* chunks[3]
  }
  out = residual + out
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      if isBF16 {
        mapping["\(prefix).attention.qkv.weight"] = ModelWeightElement(
          [toqueries.weight.name, tokeys.weight.name, tovalues.weight.name], isBF16: true)
      } else {
        mapping["\(prefix).attention.qkv.weight"] = [
          toqueries.weight.name, tokeys.weight.name, tovalues.weight.name,
        ]
      }
      mapping["\(prefix).attention.q_norm.weight"] = [normQ.weight.name]
      mapping["\(prefix).attention.k_norm.weight"] = [normK.weight.name]
      mapping["\(prefix).attention.out.weight"] =
        isBF16
        ? ModelWeightElement([unifyheads.weight.name], isBF16: true)
        : [
          unifyheads.weight.name
        ]
      mapping["\(prefix).attention_norm1.weight"] = [attentionNorm1.weight.name]
      mapping["\(prefix).attention_norm2.weight"] = [attentionNorm2.weight.name]
      mapping["\(prefix).feed_forward.w1.weight"] =
        isBF16 ? ModelWeightElement([w1.weight.name], isBF16: true) : [w1.weight.name]
      mapping["\(prefix).feed_forward.w2.weight"] =
        isBF16 ? ModelWeightElement([w2.weight.name], isBF16: true) : [w2.weight.name]
      mapping["\(prefix).feed_forward.w3.weight"] =
        isBF16 ? ModelWeightElement([w3.weight.name], isBF16: true) : [w3.weight.name]
      mapping["\(prefix).ffn_norm1.weight"] = [feedForwardNorm1.weight.name]
      mapping["\(prefix).ffn_norm2.weight"] = [feedForwardNorm2.weight.name]
    case .diffusers:
      mapping["\(prefix).attention.to_q.weight"] =
        isBF16
        ? ModelWeightElement([toqueries.weight.name], isBF16: true)
        : [
          toqueries.weight.name
        ]
      mapping["\(prefix).attention.to_k.weight"] =
        isBF16
        ? ModelWeightElement([tokeys.weight.name], isBF16: true)
        : [
          tokeys.weight.name
        ]
      mapping["\(prefix).attention.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix).attention.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix).attention.to_v.weight"] =
        isBF16
        ? ModelWeightElement([tovalues.weight.name], isBF16: true)
        : [
          tovalues.weight.name
        ]
      mapping["\(prefix).attention.to_out.0.weight"] =
        isBF16
        ? ModelWeightElement([unifyheads.weight.name], isBF16: true)
        : [
          unifyheads.weight.name
        ]
      mapping["\(prefix).attention_norm1.weight"] = [attentionNorm1.weight.name]
      mapping["\(prefix).attention_norm2.weight"] = [attentionNorm2.weight.name]
      mapping["\(prefix).feed_forward.w1.weight"] =
        isBF16 ? ModelWeightElement([w1.weight.name], isBF16: true) : [w1.weight.name]
      mapping["\(prefix).feed_forward.w2.weight"] =
        isBF16 ? ModelWeightElement([w2.weight.name], isBF16: true) : [w2.weight.name]
      mapping["\(prefix).feed_forward.w3.weight"] =
        isBF16 ? ModelWeightElement([w3.weight.name], isBF16: true) : [w3.weight.name]
      mapping["\(prefix).ffn_norm1.weight"] = [feedForwardNorm1.weight.name]
      mapping["\(prefix).ffn_norm2.weight"] = [feedForwardNorm2.weight.name]
    }
    return mapping
  }
  return (Model([x, rot] + chunks, [out]), mapper)
}

public func LoRAZImage(
  batchSize: Int, height: Int, width: Int, textLength: Int, channels: Int, layers: Int,
  activationQkScaling: [Int: Int], activationProjScaling: [Int: Int],
  activationFfnProjUpScaling: [Int: Int], activationFfnScaling: [Int: Int],
  usesFlashAttention: FlashAttentionLevel, isBF16: Bool, rotaryEmbeddingHeads: Int = 1,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (
  Model, ModelWeightMapper
) {
  let x = Input()
  let rot = Input()
  let txtIn = Input()
  let imgIn = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "x_embedder")
  let h = height / 2
  let w = width / 2
  var xOut = imgIn(
    x.reshaped([batchSize, h, 2, w, 2, 16]).permuted(0, 1, 3, 2, 4, 5).contiguous()
      .reshaped([batchSize, h * w, 2 * 2 * 16], format: .NHWC)
  )
  var mappers = [ModelWeightMapper]()
  var adaLNChunks = [Input]()
  let roundUpHW = (h * w + 31) / 32 * 32
  let xPadTokens: Input?
  if roundUpHW != h * w {
    let padTokens = Input()
    xOut = Functional.concat(axis: 1, xOut, padTokens, flags: .disableOpt)
    xPadTokens = padTokens
  } else {
    xPadTokens = nil
  }
  xOut = xOut.to(.Float32)
  let xRot = rot.reshaped(
    [1, roundUpHW, rotaryEmbeddingHeads, 128], offset: [0, 0, 0, 0],
    strides: [
      (roundUpHW + textLength) * rotaryEmbeddingHeads * 128, rotaryEmbeddingHeads * 128, 128, 1,
    ]
  ).contiguous()
  for i in 0..<2 {
    let qkScaling = activationQkScaling[i + 2] ?? 1
    var projScaling = activationProjScaling[i + 2] ?? 1
    if projScaling < 0 {
      projScaling = 4 / -projScaling
    } else {
      projScaling = 4 * projScaling
    }
    let ffnProjUpScaling = activationFfnProjUpScaling[i + 2] ?? 1
    var ffnScaling = activationFfnScaling[i + 2] ?? 1
    if ffnScaling < 0 {
      ffnScaling = 32 / -ffnScaling
    } else {
      ffnScaling = 32 * ffnScaling
    }
    let chunks = (0..<4).map { _ in Input() }
    let (block, mapper) = LoRAZImageTransformerBlock(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", k: 128, h: channels / 128, b: batchSize,
      t: (keyValue: roundUpHW, query: roundUpHW), segments: [],
      scaleFactor: (
        (qk: qkScaling, proj: projScaling), (projUp: ffnProjUpScaling, projDown: ffnScaling)
      ),
      modulation: true, usesFlashAttention: usesFlashAttention, isBF16: isBF16, layerIndex: i,
      configuration: LoRAConfiguration, rotaryEmbeddingHeads: rotaryEmbeddingHeads)
    xOut = block([xOut, xRot] + chunks)
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      block.gradientCheckpointing = true
    }
    adaLNChunks.append(contentsOf: chunks)
    mappers.append(mapper)
  }
  var out = Functional.concat(axis: 1, xOut, txtIn)
  let rotResized = rot.reshaped(.NHWC(1, roundUpHW + textLength, rotaryEmbeddingHeads, 128))
  for i in 0..<layers {
    let qkScaling = activationQkScaling[i + 4] ?? 1
    var projScaling = activationProjScaling[i + 4] ?? 1
    if !isBF16 {  // Only have default value for non-BF16 path.
      if projScaling < 0 {
        projScaling = 4 / -projScaling
      } else {
        projScaling = 4 * projScaling
      }
    }
    let ffnProjUpScaling = activationFfnProjUpScaling[i + 4] ?? 1
    var ffnScaling = activationFfnScaling[i + 4] ?? 1
    if !isBF16 {  // Only have default value for non-BF16 path.
      if ffnScaling < 0 {
        ffnScaling = 32 / -ffnScaling
      } else {
        ffnScaling = 32 * ffnScaling
      }
    }
    let chunks = (0..<4).map { _ in Input() }
    let (block, mapper) = LoRAZImageTransformerBlock(
      prefix: "layers.\(i)", name: "", k: 128, h: channels / 128, b: batchSize,
      t: (
        keyValue: roundUpHW + textLength, query: i == layers - 1 ? h * w : roundUpHW + textLength
      ),
      segments: [],
      scaleFactor: (
        (qk: qkScaling, proj: projScaling), (projUp: ffnProjUpScaling, projDown: ffnScaling)
      ), modulation: true, usesFlashAttention: usesFlashAttention, isBF16: isBF16,
      layerIndex: i, configuration: LoRAConfiguration,
      rotaryEmbeddingHeads: rotaryEmbeddingHeads)
    out = block([out, rotResized] + chunks)
    if LoRAConfiguration.gradientCheckpointingTransformerLayer {
      block.gradientCheckpointing = true
    }
    adaLNChunks.append(contentsOf: chunks)
    mappers.append(mapper)
  }
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let scale = Input()
  adaLNChunks.append(scale)
  let projOut = LoRADense(
    count: 2 * 2 * 16, configuration: LoRAConfiguration, index: 0, name: "linear_final")
  out = normFinal(out).to(.Float16) .* scale
  out = (-projOut(out)).reshaped([batchSize, h, w, 2, 2, 16]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous()
    .reshaped([
      batchSize, h * 2, w * 2, 16,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["x_embedder.weight"] = [imgIn.weight.name]
      mapping["x_embedder.bias"] = [imgIn.bias.name]
      mapping["final_layer.linear.weight"] = [projOut.weight.name]
      mapping["final_layer.linear.bias"] = [projOut.bias.name]
    case .diffusers:
      mapping["all_x_embedder.2-1.weight"] = [imgIn.weight.name]
      mapping["all_x_embedder.2-1.bias"] = [imgIn.bias.name]
      mapping["all_final_layer.2-1.linear.weight"] = [projOut.weight.name]
      mapping["all_final_layer.2-1.linear.bias"] = [projOut.bias.name]
    }
    return mapping
  }
  var inputs = [x]
  if let xPadTokens {
    inputs.append(xPadTokens)
  }
  inputs.append(rot)
  inputs.append(txtIn)
  inputs.append(contentsOf: adaLNChunks)
  return (
    Model(inputs, [out], trainable: false), mapper
  )
}

private func LoRAZImageTransformerBlockFixed(
  prefix: String, name: String, channels: Int, layerIndex: Int,
  scaleFactor: ((qk: Int, proj: Int), (projUp: Int, projDown: Int)),
  configuration: LoRANetworkConfiguration
) -> (Model, ModelWeightMapper) {
  let tEmbed = Input()
  let adaLNs = (0..<4).map {
    LoRADense(
      count: channels, configuration: configuration, index: layerIndex,
      name: name.isEmpty ? "ada_ln_\($0)" : "\(name)_ada_ln_\($0)")
  }
  var chunks = adaLNs.map { $0(tEmbed) }
  chunks[0] = (1 + chunks[0]).to(.Float32)
  if scaleFactor.0.qk > 1 {
    chunks[0] = (1.0 / Float(scaleFactor.0.qk)) * chunks[0]
  }
  chunks[1] = chunks[1].tanh().to(.Float32)
  chunks[2] = (1 + chunks[2]).to(.Float32)
  if scaleFactor.1.projUp > 1 {
    chunks[2] = (1.0 / Float(scaleFactor.1.projUp)) * chunks[2]
  }
  chunks[3] = chunks[3].tanh().to(.Float32)
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).adaLN_modulation.0.weight"] = ModelWeightElement(
      (0..<4).map { adaLNs[$0].weight.name })
    mapping["\(prefix).adaLN_modulation.0.bias"] = ModelWeightElement(
      (0..<4).map { adaLNs[$0].bias.name })
    return mapping
  }
  return (Model([tEmbed], chunks), mapper)
}

public func LoRAZImageFixed(
  batchSize: Int, tokenLength: (Int, Int), channels: Int, layers: Int,
  activationQkScaling: [Int: Int], activationProjScaling: [Int: Int],
  activationFfnProjUpScaling: [Int: Int], activationFfnScaling: [Int: Int],
  usesFlashAttention: FlashAttentionLevel, isBF16: Bool, LoRAConfiguration: LoRANetworkConfiguration
) -> (
  Model, ModelWeightMapper
) {
  let txt = Input()
  let txtRot = Input()
  let t = Input()
  let txtNorm = RMSNorm(epsilon: 1e-5, axis: [2], name: "cap_norm")
  let txtIn = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "cap_embedder")
  var txtOut = txtIn(txtNorm(txt))
  let roundUpTokenLength = ((tokenLength.0 + 31) / 32 * 32, (tokenLength.1 + 31) / 32 * 32)
  let capPadTokens: Input?
  if roundUpTokenLength.0 != tokenLength.0 || roundUpTokenLength.1 != tokenLength.1 {
    let padTokens = Input()
    if tokenLength.0 > 0 {
      var txtOut0 = txtOut.reshaped(
        [batchSize, tokenLength.0, channels], offset: [0, 0, 0],
        strides: [(tokenLength.0 + tokenLength.1) * channels, channels, 1]
      ).contiguous()
      if roundUpTokenLength.0 != tokenLength.0 {
        let padTokens0 = padTokens.reshaped(
          [batchSize, roundUpTokenLength.0 - tokenLength.0, channels], offset: [0, 0, 0],
          strides: [
            (roundUpTokenLength.0 - tokenLength.0 + roundUpTokenLength.1 - tokenLength.1)
              * channels, channels, 1,
          ]
        ).contiguous()
        txtOut0 = Functional.concat(axis: 1, txtOut0, padTokens0, flags: .disableOpt)
      }
      var txtOut1 = txtOut.reshaped(
        [batchSize, tokenLength.1, channels], offset: [0, tokenLength.0, 0],
        strides: [(tokenLength.0 + tokenLength.1) * channels, channels, 1]
      ).contiguous()
      if roundUpTokenLength.1 != tokenLength.1 {
        let padTokens1 = padTokens.reshaped(
          [batchSize, roundUpTokenLength.1 - tokenLength.1, channels],
          offset: [0, roundUpTokenLength.0 - tokenLength.0, 0],
          strides: [
            (roundUpTokenLength.0 - tokenLength.0 + roundUpTokenLength.1 - tokenLength.1)
              * channels, channels, 1,
          ]
        ).contiguous()
        txtOut1 = Functional.concat(axis: 1, txtOut1, padTokens1, flags: .disableOpt)
      }
      txtOut = Functional.concat(axis: 1, txtOut0, txtOut1, flags: .disableOpt)
    } else {
      if roundUpTokenLength.1 != tokenLength.1 {
        txtOut = Functional.concat(axis: 1, txtOut, padTokens, flags: .disableOpt)
      }
    }
    capPadTokens = padTokens
  } else {
    capPadTokens = nil
  }
  txtOut = txtOut.to(.Float32)
  let (timeInMlp0, timeInMlp2, timeIn) = LoRAMLPEmbedder(
    channels: 256, intermediateSize: 1024, configuration: LoRAConfiguration, index: 0, name: "t")
  let tOut = timeIn(t)
  var mappers = [ModelWeightMapper]()
  let segments: [Int]
  if roundUpTokenLength.0 > 0 {
    segments = [roundUpTokenLength.0, roundUpTokenLength.1]
  } else {
    segments = []
  }
  for i in 0..<2 {
    let qkScaling = activationQkScaling[i] ?? 1
    let projScaling = activationProjScaling[i] ?? 1
    let ffnProjUpScaling = activationFfnProjUpScaling[i] ?? 1
    let ffnScaling = activationFfnScaling[i] ?? 1
    let (block, mapper) = LoRAZImageTransformerBlock(
      prefix: "context_refiner.\(i)", name: "context_refiner", k: 128, h: channels / 128,
      b: batchSize,
      t: (
        keyValue: roundUpTokenLength.0 + roundUpTokenLength.1,
        query: roundUpTokenLength.0 + roundUpTokenLength.1
      ),
      segments: segments,
      scaleFactor: (
        (qk: qkScaling, proj: 2 * projScaling), (projUp: ffnProjUpScaling, projDown: 2 * ffnScaling)
      ), modulation: false,
      usesFlashAttention: usesFlashAttention, isBF16: isBF16, layerIndex: i,
      configuration: LoRAConfiguration)
    txtOut = block(txtOut, txtRot)
    mappers.append(mapper)
  }
  var outs = [txtOut]
  for i in 0..<2 {
    let qkScaling = activationQkScaling[i + 2] ?? 1
    var projScaling = activationProjScaling[i + 2] ?? 1
    if projScaling < 0 {
      projScaling = 4 / -projScaling
    } else {
      projScaling = 4 * projScaling
    }
    let ffnProjUpScaling = activationFfnProjUpScaling[i + 2] ?? 1
    var ffnScaling = activationFfnScaling[i + 2] ?? 1
    if ffnScaling < 0 {
      ffnScaling = 32 / -ffnScaling
    } else {
      ffnScaling = 32 * ffnScaling
    }
    let (block, mapper) = LoRAZImageTransformerBlockFixed(
      prefix: "noise_refiner.\(i)", name: "noise_refiner", channels: channels, layerIndex: i,
      scaleFactor: (
        (qk: qkScaling, proj: projScaling), (projUp: ffnProjUpScaling, projDown: ffnScaling)
      ), configuration: LoRAConfiguration)
    outs.append(block(tOut))
    mappers.append(mapper)
  }
  for i in 0..<layers {
    let qkScaling = activationQkScaling[i + 4] ?? 1
    var projScaling = activationProjScaling[i + 4] ?? 1
    if projScaling < 0 {
      projScaling = 4 / -projScaling
    } else {
      projScaling = 4 * projScaling
    }
    let ffnProjUpScaling = activationFfnProjUpScaling[i + 4] ?? 1
    var ffnScaling = activationFfnScaling[i + 4] ?? 1
    if ffnScaling < 0 {
      ffnScaling = 32 / -ffnScaling
    } else {
      ffnScaling = 32 * ffnScaling
    }
    let (block, mapper) = LoRAZImageTransformerBlockFixed(
      prefix: "layers.\(i)", name: "", channels: channels, layerIndex: i,
      scaleFactor: (
        (qk: qkScaling, proj: projScaling), (projUp: ffnProjUpScaling, projDown: ffnScaling)
      ), configuration: LoRAConfiguration)
    outs.append(block(tOut))
    mappers.append(mapper)
  }
  let scale = LoRADense(
    count: channels, configuration: LoRAConfiguration, index: 0, name: "ada_ln_final")
  outs.append(1 + scale(tOut.swish()))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["cap_embedder.0.weight"] = [txtNorm.weight.name]
    mapping["cap_embedder.1.weight"] = [txtIn.weight.name]
    mapping["cap_embedder.1.bias"] = [txtIn.bias.name]
    mapping["t_embedder.mlp.0.weight"] = [timeInMlp0.weight.name]
    mapping["t_embedder.mlp.0.bias"] = [timeInMlp0.bias.name]
    mapping["t_embedder.mlp.2.weight"] = [timeInMlp2.weight.name]
    mapping["t_embedder.mlp.2.bias"] = [timeInMlp2.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["final_layer.adaLN_modulation.1.weight"] = [scale.weight.name]
      mapping["final_layer.adaLN_modulation.1.bias"] = [scale.bias.name]
    case .diffusers:
      mapping["all_final_layer.2-1.adaLN_modulation.1.weight"] = [scale.weight.name]
      mapping["all_final_layer.2-1.adaLN_modulation.1.bias"] = [scale.bias.name]
    }
    return mapping
  }
  return (
    Model([txt, txtRot, t] + (capPadTokens.map { [$0] } ?? []), outs), mapper
  )
}
