import DiffusionMappings
import Foundation
import NNC

public func WanRotaryPositionEmbedding(
  height: Int, width: Int, time: Int, channels: Int, heads: Int = 1
)
  -> Tensor<Float>
{
  var rotTensor = Tensor<Float>(.CPU, .NHWC(1, time * height * width, heads, channels))
  let dim1 = (channels / 6) * 2
  let dim2 = dim1
  let dim0 = channels - dim1 - dim2
  assert(channels % 16 == 0)
  for t in 0..<time {
    for y in 0..<height {
      for x in 0..<width {
        let i = t * height * width + y * width + x
        for j in 0..<heads {
          for k in 0..<(dim0 / 2) {
            let theta = Double(t) * 1.0 / pow(10_000, Double(k) * 2 / Double(dim0))
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
  }
  return rotTensor
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

private func WanAttentionBlock<FloatType: TensorNumeric & BinaryFloatingPoint>(
  prefix: String, weightsPrefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, time: Int,
  causalInference: (Int, pad: Int), intermediateSize: Int, injectImage: Bool,
  usesFlashAttention: FlashAttentionLevel,
  of: FloatType.Type = FloatType.self
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let c = (0..<6).map { _ in Input() }
  let modulations = (0..<6).map {
    Parameter<FloatType>(.GPU(0), .HWC(1, 1, k * h), name: "\(weightsPrefix)attn_ada_ln_\($0)")
  }
  let chunks = zip(c, modulations).map { $0 + $1 }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (xNorm1(x) .* (chunks[1] + 1) + chunks[0]).to(.Float16)
  let xToKeys = Dense(count: k * h, name: "\(weightsPrefix)x_k")
  let xToQueries = Dense(count: k * h, flags: [.Float16], name: "\(weightsPrefix)x_q")
  let xToValues = Dense(count: k * h, name: "\(weightsPrefix)x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var queries = Functional.cmul(left: xQ, right: rot)
  var keys = Functional.cmul(left: xK, right: rot)
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    queries = (1 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1 / Float(k).squareRoot().squareRoot()) * keys
  }
  var values = xV
  // Now run attention.
  var out: Model.IO
  if usesFlashAttention != .none {
    let scale = usesFlashAttention == .scale1 ? 1 : 1 / Float(k).squareRoot()
    if causalInference.0 > 0 && causalInference.0 + causalInference.pad < time {
      var outs = [Model.IO]()
      precondition(b == 1)
      let frames = hw / time * causalInference.0
      let padFrames = hw / time * causalInference.pad
      for i in 0..<((time + causalInference.0 - 1) / causalInference.0) {
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        let query = queries.reshaped(
          [b, min(hw - i * frames, frames), h, k], offset: [0, i * frames, 0, 0],
          strides: [hw * h * k, h * k, k, 1])
        let key = keys.reshaped(
          [b, min(hw, (i + 1) * frames + padFrames), h, k], offset: [0, 0, 0, 0],
          strides: [hw * h * k, h * k, k, 1])
        let value = values.reshaped(
          [b, min(hw, (i + 1) * frames + padFrames), h, k], offset: [0, 0, 0, 0],
          strides: [hw * h * k, h * k, k, 1])
        let out = scaledDotProductAttention(query, key, value)
        if let last = outs.last {
          out.add(dependencies: [last])
        }
        outs.append(out)
      }
      out = Concat(axis: 1)(outs).reshaped([b, hw, k * h])
    } else {
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
    }
  } else {
    keys = keys.transposed(1, 2)
    queries = queries.transposed(1, 2)
    values = values.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<(b * h) {
      let key = keys.reshaped(
        [1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
      let query = queries.reshaped(
        [1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
      let value = values.reshaped(
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
    out = Concat(axis: 0)(outs).reshaped([b, h, hw, k]).transposed(1, 2).reshaped([
      b, hw, h * k,
    ])
  }
  let xUnifyheads = Dense(count: k * h, name: "\(weightsPrefix)x_o")
  out = xUnifyheads(out)
  out = x + out.to(of: x) .* chunks[2]
  let xNorm3 = LayerNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_norm_3")
  xOut = xNorm3(out).to(.Float16)
  let xToContextQueries = Dense(count: k * h, name: "\(weightsPrefix)x_c_q")
  let cK = Input()
  var cQ = xToContextQueries(xOut)
  let contextNormQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_c_norm_q")
  cQ = contextNormQ(cQ).reshaped([b, hw, h, k])
  let cV = Input()
  var crossOut: Model.IO
  if usesFlashAttention != .none {
    let crossAttention = ScaledDotProductAttention(
      scale: 1 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    crossOut = crossAttention(cQ, cK, cV).reshaped([b, hw, k * h])
  } else {
    let cK = cK.transposed(1, 2)
    cQ = (1 / Float(k).squareRoot() * cQ).transposed(1, 2)
    let cV = cV.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(cQ, cK)
    dot = dot.reshaped([b * h, hw, t.0])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, hw, t.0])
    crossOut = (dot * cV).reshaped([b, h, hw, k]).transposed(1, 2).reshaped([
      b, hw, h * k,
    ])
  }
  var injectedImageKVs: [Model.IO]
  if injectImage {
    let cImgK = Input()
    let cImgV = Input()
    let crossOutImg: Model.IO
    if usesFlashAttention != .none {
      let crossAttentionImg = ScaledDotProductAttention(
        scale: 1 / Float(k).squareRoot(),
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      crossOutImg = crossAttentionImg(cQ, cImgK, cImgV).reshaped([b, hw, k * h])
      crossOutImg.add(dependencies: [crossOut])
    } else {
      let cImgK = cImgK.transposed(1, 2)
      let cImgV = cImgV.transposed(1, 2)
      var dot = Matmul(transposeB: (2, 3))(cQ, cImgK)
      dot.add(dependencies: [crossOut])
      dot = dot.reshaped([b * h, hw, t.1])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, hw, t.1])
      crossOutImg = (dot * cImgV).reshaped([b, h, hw, k]).transposed(1, 2).reshaped([
        b, hw, h * k,
      ])
    }
    crossOut = crossOut + crossOutImg
    injectedImageKVs = [cImgK, cImgV]
  } else {
    injectedImageKVs = []
  }
  let contextUnifyheads = Dense(count: k * h, name: "\(weightsPrefix)c_o")
  out = out + contextUnifyheads(crossOut).to(of: out)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (xLinear1, xOutProjection, xFF) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, upcast: false, name: "\(weightsPrefix)x")
  out =
    out + xFF((xNorm2(out) .* (chunks[4] + 1) + chunks[3]).to(.Float16)).to(of: out) .* chunks[5]
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix).modulation"] = ModelWeightElement(
        (0..<6).map { modulations[$0].weight.name })
      mapping["\(prefix).self_attn.q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix).self_attn.q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix).self_attn.k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix).self_attn.k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix).self_attn.v.weight"] = [xToValues.weight.name]
      mapping["\(prefix).self_attn.v.bias"] = [xToValues.bias.name]
      mapping["\(prefix).self_attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix).self_attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix).self_attn.o.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix).self_attn.o.bias"] = [xUnifyheads.bias.name]
      mapping["\(prefix).cross_attn.q.weight"] = [xToContextQueries.weight.name]
      mapping["\(prefix).cross_attn.q.bias"] = [xToContextQueries.bias.name]
      mapping["\(prefix).cross_attn.norm_q.weight"] = [contextNormQ.weight.name]
      mapping["\(prefix).cross_attn.o.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).cross_attn.o.bias"] = [contextUnifyheads.bias.name]
      mapping["\(prefix).norm3.weight"] = [xNorm3.weight.name]
      mapping["\(prefix).norm3.bias"] = [xNorm3.bias.name]
      mapping["\(prefix).ffn.0.weight"] = [xLinear1.weight.name]
      mapping["\(prefix).ffn.0.bias"] = [xLinear1.bias.name]
      mapping["\(prefix).ffn.2.weight"] = [xOutProjection.weight.name]
      mapping["\(prefix).ffn.2.bias"] = [xOutProjection.bias.name]
    case .diffusers:
      mapping["\(prefix).scale_shift_table"] = ModelWeightElement(
        (0..<6).map { modulations[$0].weight.name })
      mapping["\(prefix).attn1.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix).attn1.to_q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix).attn1.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix).attn1.to_k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix).attn1.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix).attn1.to_v.bias"] = [xToValues.bias.name]
      mapping["\(prefix).attn1.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix).attn1.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix).attn1.to_out.0.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix).attn1.to_out.0.bias"] = [xUnifyheads.bias.name]
      mapping["\(prefix).attn2.to_q.weight"] = [xToContextQueries.weight.name]
      mapping["\(prefix).attn2.to_q.bias"] = [xToContextQueries.bias.name]
      mapping["\(prefix).attn2.norm_q.weight"] = [contextNormQ.weight.name]
      mapping["\(prefix).attn2.to_out.0.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).attn2.to_out.0.bias"] = [contextUnifyheads.bias.name]
      mapping["\(prefix).norm2.weight"] = [xNorm3.weight.name]
      mapping["\(prefix).norm2.bias"] = [xNorm3.bias.name]
      mapping["\(prefix).ffn.net.0.proj.weight"] = [xLinear1.weight.name]
      mapping["\(prefix).ffn.net.0.proj.bias"] = [xLinear1.bias.name]
      mapping["\(prefix).ffn.net.2.weight"] = [xOutProjection.weight.name]
      mapping["\(prefix).ffn.net.2.bias"] = [xOutProjection.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot] + c + [cK, cV] + injectedImageKVs, [out]))
}

private func TimeEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func MLPEmbedder(channels: Int, name: String) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: channels, name: "\(name)_embedder_0")
  var out = fc0(x).GELU(approximate: .tanh)
  let fc2 = Dense(count: channels, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func MLPProj(inChannels: Int, outChannels: Int, name: String) -> (
  Model, Model, Model, Model, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_0")
  let fc0 = Dense(count: inChannels, name: "\(name)_embedder_0")
  var out = fc0(ln1(x)).GELU()
  let fc2 = Dense(count: outChannels, name: "\(name)_embedder_1")
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_1")
  out = ln2(fc2(out))
  return (ln1, ln2, fc0, fc2, Model([x], [out]))
}

public func Wan(
  channels: Int, layers: Int, vaceLayers: [Int], intermediateSize: Int, time: Int, height: Int,
  width: Int, textLength: Int, causalInference: (Int, pad: Int), injectImage: Bool,
  usesFlashAttention: FlashAttentionLevel, outputResidual: Bool, inputResidual: Bool,
  outputChannels: Int
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let imgIn = Convolution(
    groups: 1, filters: channels, filterSize: [2, 2],
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  let h = height / 2
  let w = width / 2
  var mappers = [ModelWeightMapper]()
  let xIn16 = imgIn(x).reshaped([1, time * h * w, channels])
  let xIn = xIn16.to(.Float32)
  var out = xIn
  let imgInX = out
  let residualIn: Input?
  if inputResidual {
    let residual = Input()
    residualIn = residual
    out = out + residual
  } else {
    residualIn = nil
  }
  var rotAndtOut = [Input]()
  if layers > 0 {
    let rot = Input()
    let tOut = (0..<6).map { _ in Input() }
    rotAndtOut = [rot] + tOut
  }
  var contextIn = [Input]()
  var hints = [Int: Model.IO]()
  var hintAttentions = [Int: Model.IO]()
  var lastHint: Model.IO? = nil
  if !vaceLayers.isEmpty {
    let scale = Input()
    let vIn = Input()
    contextIn.append(contentsOf: [scale, vIn])
    out = xIn16 + vIn.reshaped([1, time * h * w, channels])
    let rotAndtOut: [Model.IO] = [rotAndtOut[0]] + rotAndtOut[1..<7].map { $0.to(of: out) }
    for (i, n) in vaceLayers.enumerated() {
      let (mapper, block) = WanAttentionBlock(
        prefix: "vace_blocks.\(i)", weightsPrefix: "vace_", k: 128, h: channels / 128, b: 1,
        t: (textLength, 257),
        hw: time * h * w, time: time, causalInference: causalInference,
        intermediateSize: intermediateSize, injectImage: false,
        usesFlashAttention: usesFlashAttention, of: FloatType.self)
      let contextK = Input()
      let contextV = Input()
      contextIn.append(contentsOf: [contextK, contextV])
      out = block([out] + rotAndtOut + [contextK, contextV])
      if let lastHint = lastHint {
        out.add(dependencies: [lastHint])
      }
      hintAttentions[n] = out
      mappers.append(mapper)
      let afterProj = Dense(count: channels, name: "vace_after_proj")
      lastHint = scale .* afterProj(out)
      hints[n] = lastHint
    }
    out = xIn
  }
  for i in 0..<layers {
    let (mapper, block) = WanAttentionBlock(
      prefix: "blocks.\(i)", weightsPrefix: "", k: 128, h: channels / 128, b: 1,
      t: (textLength, 257),
      hw: time * h * w, time: time, causalInference: causalInference,
      intermediateSize: intermediateSize, injectImage: injectImage,
      usesFlashAttention: usesFlashAttention, of: Float.self)
    let contextK = Input()
    let contextV = Input()
    contextIn.append(contentsOf: [contextK, contextV])
    if injectImage {
      let contextImgK = Input()
      let contextImgV = Input()
      out = block([out] + rotAndtOut + [contextK, contextV, contextImgK, contextImgV])
      contextIn.append(contentsOf: [contextImgK, contextImgV])
    } else {
      out = block([out] + rotAndtOut + [contextK, contextV])
    }
    mappers.append(mapper)
    if let hint = hints[i] {
      if let hintAttention = hintAttentions[i] {
        hintAttention.add(dependencies: [out])
        hintAttentions[i] = nil
      }
      let hintAsOut = hint.to(of: out)
      hintAsOut.add(dependencies: [out])
      out = out + hintAsOut
    }
  }
  let residualOut: Model.IO?
  if outputResidual {
    residualOut = out - imgInX
  } else {
    residualOut = nil
  }
  let scale = Input()
  let shift = Input()
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (normFinal(out) .* scale + shift).to(.Float16)
  let projOut = Dense(count: 2 * 2 * outputChannels, name: "linear")
  out = projOut(out).reshaped([time, h, w, 2, 2, outputChannels]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous()
    .reshaped([
      time, h * 2, w * 2, outputChannels,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["patch_embedding.weight"] = [imgIn.weight.name]
      mapping["patch_embedding.bias"] = [imgIn.bias.name]
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      mapping["head.head.weight"] = [projOut.weight.name]
      mapping["head.head.bias"] = [projOut.bias.name]
    case .diffusers:
      mapping["patch_embedding.weight"] = [imgIn.weight.name]
      mapping["patch_embedding.bias"] = [imgIn.bias.name]
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      mapping["proj_out.weight"] = [projOut.weight.name]
      mapping["proj_out.bias"] = [projOut.bias.name]
    }
    return mapping
  }
  return (
    mapper,
    Model(
      [x] + (residualIn.map { [$0] } ?? []) + rotAndtOut + contextIn + [scale, shift],
      [out] + (residualOut.map { [$0] } ?? []))
  )
}

private func WanAttentionBlockFixed(
  prefix: String, weightsPrefix: String, k: Int, h: Int, b: (Int, Int), t: (Int, Int),
  injectImage: Bool
) -> (ModelWeightMapper, Model) {
  let context = Input()
  // Now run attention.
  let contextToKeys = Dense(count: k * h, name: "\(weightsPrefix)c_k")
  let contextToValues = Dense(count: k * h, name: "\(weightsPrefix)c_v")
  var cK = contextToKeys(context)
  let contextNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)c_norm_k")
  cK = contextNormK(cK).reshaped([b.0, t.0, h, k])
  let cV = contextToValues(context).reshaped([b.0, t.0, h, k])
  var ins = [context]
  var outs = [cK, cV]
  let cImgToKeys: Model?
  let cImgToValues: Model?
  let cImgNormK: Model?
  if injectImage {
    let contextImg = Input()
    let contextImgToKeys = Dense(count: k * h, name: "\(weightsPrefix)c_img_k")
    let contextImgToValues = Dense(count: k * h, name: "\(weightsPrefix)c_img_v")
    var cImgK = contextImgToKeys(contextImg)
    let contextImgNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)c_img_norm_k")
    cImgK = contextImgNormK(cImgK).reshaped([b.1, t.1, h, k])
    let cImgV = contextImgToValues(contextImg).reshaped([b.1, t.1, h, k])
    ins.append(contextImg)
    outs.append(contentsOf: [cImgK, cImgV])
    cImgToKeys = contextImgToKeys
    cImgToValues = contextImgToValues
    cImgNormK = contextImgNormK
  } else {
    cImgToKeys = nil
    cImgToValues = nil
    cImgNormK = nil
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix).cross_attn.k.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix).cross_attn.k.bias"] = [contextToKeys.bias.name]
      mapping["\(prefix).cross_attn.v.weight"] = [contextToValues.weight.name]
      mapping["\(prefix).cross_attn.v.bias"] = [contextToValues.bias.name]
      mapping["\(prefix).cross_attn.norm_k.weight"] = [contextNormK.weight.name]
      if let cImgToKeys = cImgToKeys, let cImgToValues = cImgToValues, let cImgNormK = cImgNormK {
        mapping["\(prefix).cross_attn.k_img.weight"] = [cImgToKeys.weight.name]
        mapping["\(prefix).cross_attn.k_img.bias"] = [cImgToKeys.bias.name]
        mapping["\(prefix).cross_attn.v_img.weight"] = [cImgToValues.weight.name]
        mapping["\(prefix).cross_attn.v_img.bias"] = [cImgToValues.bias.name]
        mapping["\(prefix).cross_attn.norm_k_img.weight"] = [cImgNormK.weight.name]
      }
    case .diffusers:
      mapping["\(prefix).attn2.to_k.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix).attn2.to_k.bias"] = [contextToKeys.bias.name]
      mapping["\(prefix).attn2.to_v.weight"] = [contextToValues.weight.name]
      mapping["\(prefix).attn2.to_v.bias"] = [contextToValues.bias.name]
      mapping["\(prefix).attn2.norm_k.weight"] = [contextNormK.weight.name]
      if let cImgToKeys = cImgToKeys, let cImgToValues = cImgToValues, let cImgNormK = cImgNormK {
        mapping["\(prefix).attn2.add_k_proj.weight"] = [cImgToKeys.weight.name]
        mapping["\(prefix).attn2.add_k_proj.bias"] = [cImgToKeys.bias.name]
        mapping["\(prefix).attn2.add_v_proj.weight"] = [cImgToValues.weight.name]
        mapping["\(prefix).attn2.add_v_proj.bias"] = [cImgToValues.bias.name]
        mapping["\(prefix).attn2.norm_added_k.weight"] = [cImgNormK.weight.name]
      }
    }
    return mapping
  }
  return (mapper, Model(ins, outs))
}

public func WanFixed(
  timesteps: Int, batchSize: (Int, Int), channels: Int, layers: Int, vaceLayers: [Int],
  textLength: Int, injectImage: Bool
) -> (
  ModelWeightMapper, Model
) {
  let txt = Input()
  let (cLinear1, cLinear2, contextEmbedder) = MLPEmbedder(channels: channels, name: "c")
  let context = contextEmbedder(txt)
  let t = Input()
  let (timeInMlp0, timeInMlp2, timeIn) = TimeEmbedder(channels: channels, name: "t")
  let vector = timeIn(t).reshaped([timesteps, 1, channels])
  let vectorIn = vector.swish()
  let timeProjections = (0..<6).map { Dense(count: channels, name: "ada_ln_\($0)") }
  var outs = timeProjections.map { $0(vectorIn).identity().identity().identity() }  // Have duplicate name ada_ln_0 / ada_ln_1, now have to push the order to make sure the proper weights are loaded :(.
  var ins = [txt, t]
  if !vaceLayers.isEmpty {
    let vaceContext = Input()
    let vaceIn = Convolution(
      groups: 1, filters: channels, filterSize: [2, 2],
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "vace_x_embedder")
    let beforeProj = Convolution(
      groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
      name: "vace_before_proj")
    outs.append(beforeProj(vaceIn(vaceContext)))
    ins.append(vaceContext)
  }
  let contextImg: Model.IO?
  let clipLn1: Model?
  let clipLn2: Model?
  let clipMlp0: Model?
  let clipMlp2: Model?
  if injectImage {
    let img = Input()
    let clipIn: Model
    (clipLn1, clipLn2, clipMlp0, clipMlp2, clipIn) = MLPProj(
      inChannels: 1_280, outChannels: channels, name: "clip")
    contextImg = clipIn(img)
    ins.append(img)
  } else {
    clipLn1 = nil
    clipLn2 = nil
    clipMlp0 = nil
    clipMlp2 = nil
    contextImg = nil
  }
  var mappers = [ModelWeightMapper]()
  for (i, _) in vaceLayers.enumerated() {
    let (mapper, block) = WanAttentionBlockFixed(
      prefix: "vace_blocks.\(i)", weightsPrefix: "vace_", k: 128, h: channels / 128, b: batchSize,
      t: (textLength, 257), injectImage: false)
    outs.append(block(context))
    mappers.append(mapper)
  }
  for i in 0..<layers {
    let (mapper, block) = WanAttentionBlockFixed(
      prefix: "blocks.\(i)", weightsPrefix: "", k: 128, h: channels / 128, b: batchSize,
      t: (textLength, 257), injectImage: injectImage)
    if let contextImg = contextImg {
      outs.append(block(context, contextImg))
    } else {
      outs.append(block(context))
    }
    mappers.append(mapper)
  }
  let scale = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_0")
  let shift = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_1")
  outs.append(1 + scale + vector)
  outs.append(vector + shift)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["text_embedding.0.weight"] = [cLinear1.weight.name]
      mapping["text_embedding.0.bias"] = [cLinear1.bias.name]
      mapping["text_embedding.2.weight"] = [cLinear2.weight.name]
      mapping["text_embedding.2.bias"] = [cLinear2.bias.name]
      mapping["time_embedding.0.weight"] = [timeInMlp0.weight.name]
      mapping["time_embedding.0.bias"] = [timeInMlp0.bias.name]
      mapping["time_embedding.2.weight"] = [timeInMlp2.weight.name]
      mapping["time_embedding.2.bias"] = [timeInMlp2.bias.name]
      if let clipLn1 = clipLn1, let clipLn2 = clipLn2, let clipMlp0 = clipMlp0,
        let clipMlp2 = clipMlp2
      {
        mapping["img_emb.proj.0.weight"] = [clipLn1.weight.name]
        mapping["img_emb.proj.0.bias"] = [clipLn1.bias.name]
        mapping["img_emb.proj.1.weight"] = [clipMlp0.weight.name]
        mapping["img_emb.proj.1.bias"] = [clipMlp0.bias.name]
        mapping["img_emb.proj.3.weight"] = [clipMlp2.weight.name]
        mapping["img_emb.proj.3.bias"] = [clipMlp2.bias.name]
        mapping["img_emb.proj.4.weight"] = [clipLn2.weight.name]
        mapping["img_emb.proj.4.bias"] = [clipLn2.bias.name]
      }
      mapping["time_projection.1.weight"] = ModelWeightElement(
        (0..<6).map { timeProjections[$0].weight.name })
      mapping["time_projection.1.bias"] = ModelWeightElement(
        (0..<6).map { timeProjections[$0].bias.name })
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      mapping["head.modulation"] = [shift.weight.name, scale.weight.name]
    case .diffusers:
      mapping["condition_embedder.text_embedder.linear_1.weight"] = [cLinear1.weight.name]
      mapping["condition_embedder.text_embedder.linear_1.bias"] = [cLinear1.bias.name]
      mapping["condition_embedder.text_embedder.linear_2.weight"] = [cLinear2.weight.name]
      mapping["condition_embedder.text_embedder.linear_2.bias"] = [cLinear2.bias.name]
      mapping["condition_embedder.time_embedder.linear_1.weight"] = [timeInMlp0.weight.name]
      mapping["condition_embedder.time_embedder.linear_1.bias"] = [timeInMlp0.bias.name]
      mapping["condition_embedder.time_embedder.linear_2.weight"] = [timeInMlp2.weight.name]
      mapping["condition_embedder.time_embedder.linear_2.bias"] = [timeInMlp2.bias.name]
      if let clipLn1 = clipLn1, let clipLn2 = clipLn2, let clipMlp0 = clipMlp0,
        let clipMlp2 = clipMlp2
      {
        mapping["condition_embedder.image_embedder.norm1.weight"] = [clipLn1.weight.name]
        mapping["condition_embedder.image_embedder.norm1.bias"] = [clipLn1.bias.name]
        mapping["condition_embedder.image_embedder.ff.net.0.proj.weight"] = [clipMlp0.weight.name]
        mapping["condition_embedder.image_embedder.ff.net.0.proj.bias"] = [clipMlp0.bias.name]
        mapping["condition_embedder.image_embedder.ff.net.2.weight"] = [clipMlp2.weight.name]
        mapping["condition_embedder.image_embedder.ff.net.2.bias"] = [clipMlp2.bias.name]
        mapping["condition_embedder.image_embedder.norm2.weight"] = [clipLn2.weight.name]
        mapping["condition_embedder.image_embedder.norm2.bias"] = [clipLn2.bias.name]
      }
      mapping["condition_embedder.time_proj.weight"] = ModelWeightElement(
        (0..<6).map { timeProjections[$0].weight.name })
      mapping["condition_embedder.time_proj.bias"] = ModelWeightElement(
        (0..<6).map { timeProjections[$0].bias.name })
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      mapping["scale_shift_table"] = [shift.weight.name, scale.weight.name]
    }
    return mapping
  }
  return (mapper, Model(ins, outs))
}

private func WanAttentionBlockFixedOutputShapes(
  prefix: String, k: Int, h: Int, b: (Int, Int), t: (Int, Int), injectImage: Bool
) -> [TensorShape] {
  var xOutputShapes = (0..<2).map { _ in TensorShape([b.0, t.0, h, k]) }
  if injectImage {
    for _ in 0..<2 {
      xOutputShapes.append(TensorShape([b.1, t.1, h, k]))
    }
  }
  return xOutputShapes
}

public func WanFixedOutputShapes(
  timesteps: Int, batchSize: (Int, Int), channels: Int, layers: Int, textLength: Int,
  injectImage: Bool
) -> [TensorShape] {
  var outs = [TensorShape]()
  for _ in 0..<6 {
    outs.append(TensorShape([timesteps, 1, channels]))
  }
  for i in 0..<layers {
    let outputShapes = WanAttentionBlockFixedOutputShapes(
      prefix: "blocks.\(i)", k: 128, h: channels / 128, b: batchSize, t: (textLength, 257),
      injectImage: injectImage)
    outs.append(contentsOf: outputShapes)
  }
  outs.append(contentsOf: [
    TensorShape([1, 1, channels]), TensorShape([1, 1, channels]),
  ])
  return outs
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

private func LoRAWanAttentionBlock(
  prefix: String, weightsPrefix: String, k: Int, h: Int, b: Int, t: (Int, Int), hw: Int, time: Int,
  causalInference: (Int, pad: Int),
  intermediateSize: Int, injectImage: Bool, usesFlashAttention: FlashAttentionLevel,
  layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let rot = Input()
  let c = (0..<6).map { _ in Input() }
  let modulations = (0..<6).map {
    Parameter<Float>(.GPU(0), .HWC(1, 1, k * h), name: "\(weightsPrefix)attn_ada_ln_\($0)")
  }
  let chunks = zip(c, modulations).map { $0 + $1 }
  let xNorm1 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var xOut = (xNorm1(x) .* (chunks[1] + 1) + chunks[0]).to(.Float16)
  let xToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "\(weightsPrefix)x_k")
  let xToQueries = LoRADense(
    count: k * h, configuration: configuration, flags: [.Float16], index: layerIndex,
    name: "\(weightsPrefix)x_q")
  let xToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "\(weightsPrefix)x_v")
  var xK = xToKeys(xOut)
  let normK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_norm_k")
  xK = normK(xK).reshaped([b, hw, h, k])
  var xQ = xToQueries(xOut)
  let normQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_norm_q")
  xQ = normQ(xQ).reshaped([b, hw, h, k])
  let xV = xToValues(xOut).reshaped([b, hw, h, k])
  var queries = Functional.cmul(left: xQ, right: rot)
  var keys = Functional.cmul(left: xK, right: rot)
  if usesFlashAttention != .quantized && usesFlashAttention != .scaleMerged {
    queries = (1 / Float(k).squareRoot().squareRoot()) * queries
    keys = (1 / Float(k).squareRoot().squareRoot()) * keys
  }
  var values = xV
  // Now run attention.
  var out: Model.IO
  if usesFlashAttention != .none {
    let scale = usesFlashAttention == .scale1 ? 1 : 1 / Float(k).squareRoot()
    if causalInference.0 > 0 && causalInference.0 + causalInference.pad < time {
      var outs = [Model.IO]()
      precondition(b == 1)
      let frames = hw / time * causalInference.0
      let padFrames = hw / time * causalInference.pad
      for i in 0..<((time + causalInference.0 - 1) / causalInference.0) {
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        let query = queries.reshaped(
          [b, min(hw - i * frames, frames), h, k], offset: [0, i * frames, 0, 0],
          strides: [hw * h * k, h * k, k, 1])
        let key = keys.reshaped(
          [b, min(hw, (i + 1) * frames + padFrames), h, k], offset: [0, 0, 0, 0],
          strides: [hw * h * k, h * k, k, 1])
        let value = values.reshaped(
          [b, min(hw, (i + 1) * frames + padFrames), h, k], offset: [0, 0, 0, 0],
          strides: [hw * h * k, h * k, k, 1])
        let out = scaledDotProductAttention(query, key, value)
        if let last = outs.last {
          out.add(dependencies: [last])
        }
        outs.append(out)
      }
      out = Concat(axis: 1)(outs).reshaped([b, hw, k * h])
    } else {
      let scaledDotProductAttention = ScaledDotProductAttention(
        scale: scale, flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
    }
  } else {
    keys = keys.transposed(1, 2)
    queries = queries.transposed(1, 2)
    values = values.transposed(1, 2)
    var outs = [Model.IO]()
    for i in 0..<(b * h) {
      let key = keys.reshaped(
        [1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
      let query = queries.reshaped(
        [1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
      let value = values.reshaped(
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
    out = Concat(axis: 0)(outs).reshaped([b, h, hw, k]).transposed(1, 2).reshaped([
      b, hw, h * k,
    ])
  }
  let xUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "\(weightsPrefix)x_o")
  out = xUnifyheads(out)
  out = x + out.to(of: x) .* chunks[2]
  let xNorm3 = LayerNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_norm_3")
  xOut = xNorm3(out).to(.Float16)
  let xToContextQueries = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "\(weightsPrefix)x_c_q")
  let cK = Input()
  var cQ = xToContextQueries(xOut)
  let contextNormQ = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)x_c_norm_q")
  cQ = contextNormQ(cQ).reshaped([b, hw, h, k])
  let cV = Input()
  var crossOut: Model.IO
  if usesFlashAttention != .none {
    let crossAttention = ScaledDotProductAttention(
      scale: 1 / Float(k).squareRoot(),
      flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
    crossOut = crossAttention(cQ, cK, cV).reshaped([b, hw, k * h])
  } else {
    let cK = cK.transposed(1, 2)
    cQ = (1 / Float(k).squareRoot() * cQ).transposed(1, 2)
    let cV = cV.transposed(1, 2)
    var dot = Matmul(transposeB: (2, 3))(cQ, cK)
    dot = dot.reshaped([b * h, hw, t.0])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, hw, t.0])
    crossOut = (dot * cV).reshaped([b, h, hw, k]).transposed(1, 2).reshaped([
      b, hw, h * k,
    ])
  }
  var injectedImageKVs: [Model.IO]
  if injectImage {
    let cImgK = Input()
    let cImgV = Input()
    let crossOutImg: Model.IO
    if usesFlashAttention != .none {
      let crossAttentionImg = ScaledDotProductAttention(
        scale: 1 / Float(k).squareRoot(),
        flags: usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
      crossOutImg = crossAttentionImg(cQ, cImgK, cImgV).reshaped([b, hw, k * h])
      crossOutImg.add(dependencies: [crossOut])
    } else {
      let cImgK = cImgK.transposed(1, 2)
      let cImgV = cImgV.transposed(1, 2)
      var dot = Matmul(transposeB: (2, 3))(cQ, cImgK)
      dot.add(dependencies: [crossOut])
      dot = dot.reshaped([b * h, hw, t.1])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, hw, t.1])
      crossOutImg = (dot * cImgV).reshaped([b, h, hw, k]).transposed(1, 2).reshaped([
        b, hw, h * k,
      ])
    }
    crossOut = crossOut + crossOutImg
    injectedImageKVs = [cImgK, cImgV]
  } else {
    injectedImageKVs = []
  }
  let contextUnifyheads = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "\(weightsPrefix)c_o")
  out = out + contextUnifyheads(crossOut).to(of: out)
  let xNorm2 = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  let (xLinear1, xOutProjection, xFF) = LoRAFeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, upcast: false,
    configuration: configuration, index: layerIndex, name: "\(weightsPrefix)x")
  out =
    out + xFF((xNorm2(out) .* (chunks[4] + 1) + chunks[3]).to(.Float16)).to(of: out) .* chunks[5]
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix).modulation"] = ModelWeightElement(
        (0..<6).map { modulations[$0].weight.name })
      mapping["\(prefix).self_attn.q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix).self_attn.q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix).self_attn.k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix).self_attn.k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix).self_attn.v.weight"] = [xToValues.weight.name]
      mapping["\(prefix).self_attn.v.bias"] = [xToValues.bias.name]
      mapping["\(prefix).self_attn.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix).self_attn.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix).self_attn.o.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix).self_attn.o.bias"] = [xUnifyheads.bias.name]
      mapping["\(prefix).cross_attn.q.weight"] = [xToContextQueries.weight.name]
      mapping["\(prefix).cross_attn.q.bias"] = [xToContextQueries.bias.name]
      mapping["\(prefix).cross_attn.norm_q.weight"] = [contextNormQ.weight.name]
      mapping["\(prefix).cross_attn.o.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).cross_attn.o.bias"] = [contextUnifyheads.bias.name]
      mapping["\(prefix).norm3.weight"] = [xNorm3.weight.name]
      mapping["\(prefix).norm3.bias"] = [xNorm3.bias.name]
      mapping["\(prefix).ffn.0.weight"] = [xLinear1.weight.name]
      mapping["\(prefix).ffn.0.bias"] = [xLinear1.bias.name]
      mapping["\(prefix).ffn.2.weight"] = [xOutProjection.weight.name]
      mapping["\(prefix).ffn.2.bias"] = [xOutProjection.bias.name]
    case .diffusers:
      mapping["\(prefix).scale_shift_table"] = ModelWeightElement(
        (0..<6).map { modulations[$0].weight.name })
      mapping["\(prefix).attn1.to_q.weight"] = [xToQueries.weight.name]
      mapping["\(prefix).attn1.to_q.bias"] = [xToQueries.bias.name]
      mapping["\(prefix).attn1.to_k.weight"] = [xToKeys.weight.name]
      mapping["\(prefix).attn1.to_k.bias"] = [xToKeys.bias.name]
      mapping["\(prefix).attn1.to_v.weight"] = [xToValues.weight.name]
      mapping["\(prefix).attn1.to_v.bias"] = [xToValues.bias.name]
      mapping["\(prefix).attn1.norm_k.weight"] = [normK.weight.name]
      mapping["\(prefix).attn1.norm_q.weight"] = [normQ.weight.name]
      mapping["\(prefix).attn1.to_out.0.weight"] = [xUnifyheads.weight.name]
      mapping["\(prefix).attn1.to_out.0.bias"] = [xUnifyheads.bias.name]
      mapping["\(prefix).attn2.to_q.weight"] = [xToContextQueries.weight.name]
      mapping["\(prefix).attn2.to_q.bias"] = [xToContextQueries.bias.name]
      mapping["\(prefix).attn2.norm_q.weight"] = [contextNormQ.weight.name]
      mapping["\(prefix).attn2.to_out.0.weight"] = [contextUnifyheads.weight.name]
      mapping["\(prefix).attn2.to_out.0.bias"] = [contextUnifyheads.bias.name]
      mapping["\(prefix).norm2.weight"] = [xNorm3.weight.name]
      mapping["\(prefix).norm2.bias"] = [xNorm3.bias.name]
      mapping["\(prefix).ffn.net.0.proj.weight"] = [xLinear1.weight.name]
      mapping["\(prefix).ffn.net.0.proj.bias"] = [xLinear1.bias.name]
      mapping["\(prefix).ffn.net.2.weight"] = [xOutProjection.weight.name]
      mapping["\(prefix).ffn.net.2.bias"] = [xOutProjection.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x, rot] + c + [cK, cV] + injectedImageKVs, [out]))
}

private func LoRATimeEmbedder(channels: Int, configuration: LoRANetworkConfiguration, name: String)
  -> (Model, Model, Model)
{
  let x = Input()
  let fc0 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_0")
  var out = fc0(x).swish()
  let fc2 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LoRAMLPEmbedder(channels: Int, configuration: LoRANetworkConfiguration, name: String)
  -> (Model, Model, Model)
{
  let x = Input()
  let fc0 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_0")
  var out = fc0(x).GELU(approximate: .tanh)
  let fc2 = LoRADense(count: channels, configuration: configuration, name: "\(name)_embedder_1")
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

private func LoRAMLPProj(
  inChannels: Int, outChannels: Int, configuration: LoRANetworkConfiguration, name: String
) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_0")
  let fc0 = LoRADense(count: inChannels, configuration: configuration, name: "\(name)_embedder_0")
  var out = fc0(ln1(x)).GELU()
  let fc2 = LoRADense(count: outChannels, configuration: configuration, name: "\(name)_embedder_1")
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [2], name: "\(name)_layer_norm_1")
  out = ln2(fc2(out))
  return (ln1, ln2, fc0, fc2, Model([x], [out]))
}

func LoRAWan(
  channels: Int, layers: Int, vaceLayers: [Int], intermediateSize: Int, time: Int, height: Int,
  width: Int, textLength: Int, causalInference: (Int, pad: Int), injectImage: Bool,
  usesFlashAttention: FlashAttentionLevel, outputResidual: Bool, inputResidual: Bool,
  outputChannels: Int,
  LoRAConfiguration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let imgIn = LoRAConvolution(
    groups: 1, filters: channels, filterSize: [2, 2], configuration: LoRAConfiguration,
    hint: Hint(stride: [2, 2]), format: .OIHW, name: "x_embedder")
  let h = height / 2
  let w = width / 2
  var mappers = [ModelWeightMapper]()
  let xIn16 = imgIn(x).reshaped([1, time * h * w, channels])
  let xIn = xIn16.to(.Float32)
  var out = xIn
  let imgInX = out
  let residualIn: Input?
  if inputResidual {
    let residual = Input()
    residualIn = residual
    out = out + residual
  } else {
    residualIn = nil
  }
  var rotAndtOut = [Input]()
  if layers > 0 {
    let rot = Input()
    let tOut = (0..<6).map { _ in Input() }
    rotAndtOut = [rot] + tOut
  }
  var contextIn = [Input]()
  var hints = [Int: Model.IO]()
  var hintAttentions = [Int: Model.IO]()
  var lastHint: Model.IO? = nil
  if !vaceLayers.isEmpty {
    let scale = Input()
    let vIn = Input()
    contextIn.append(contentsOf: [scale, vIn])
    out = xIn16 + vIn.reshaped([1, time * h * w, channels])
    let rotAndtOut: [Model.IO] = [rotAndtOut[0]] + rotAndtOut[1..<7].map { $0.to(of: out) }
    for (i, n) in vaceLayers.enumerated() {
      let (mapper, block) = WanAttentionBlock(
        prefix: "vace_blocks.\(i)", weightsPrefix: "vace_", k: 128, h: channels / 128, b: 1,
        t: (textLength, 257),
        hw: time * h * w, time: time, causalInference: causalInference,
        intermediateSize: intermediateSize, injectImage: false,
        usesFlashAttention: usesFlashAttention, of: FloatType.self)
      let contextK = Input()
      let contextV = Input()
      contextIn.append(contentsOf: [contextK, contextV])
      out = block([out] + rotAndtOut + [contextK, contextV])
      if let lastHint = lastHint {
        out.add(dependencies: [lastHint])
      }
      hintAttentions[n] = out
      mappers.append(mapper)
      let afterProj = Dense(count: channels, name: "vace_after_proj")
      lastHint = scale .* afterProj(out)
      hints[n] = lastHint
    }
    out = xIn
  }
  for i in 0..<layers {
    let (mapper, block) = LoRAWanAttentionBlock(
      prefix: "blocks.\(i)", weightsPrefix: "", k: 128, h: channels / 128, b: 1,
      t: (textLength, 257),
      hw: time * h * w, time: time, causalInference: causalInference,
      intermediateSize: intermediateSize, injectImage: injectImage,
      usesFlashAttention: usesFlashAttention, layerIndex: i, configuration: LoRAConfiguration)
    let contextK = Input()
    let contextV = Input()
    contextIn.append(contentsOf: [contextK, contextV])
    if injectImage {
      let contextImgK = Input()
      let contextImgV = Input()
      out = block([out] + rotAndtOut + [contextK, contextV, contextImgK, contextImgV])
      contextIn.append(contentsOf: [contextImgK, contextImgV])
    } else {
      out = block([out] + rotAndtOut + [contextK, contextV])
    }
    mappers.append(mapper)
    if let hint = hints[i] {
      if let hintAttention = hintAttentions[i] {
        hintAttention.add(dependencies: [out])
        hintAttentions[i] = nil
      }
      let hintAsOut = hint.to(of: out)
      hintAsOut.add(dependencies: [out])
      out = out + hintAsOut
    }
  }
  let residualOut: Model.IO?
  if outputResidual {
    residualOut = out - imgInX
  } else {
    residualOut = nil
  }
  let scale = Input()
  let shift = Input()
  let normFinal = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = (normFinal(out) .* scale + shift).to(.Float16)
  let projOut = LoRADense(
    count: 2 * 2 * outputChannels, configuration: LoRAConfiguration, index: 0, name: "linear")
  out = projOut(out).reshaped([time, h, w, 2, 2, outputChannels]).permuted(0, 1, 3, 2, 4, 5)
    .contiguous()
    .reshaped([
      time, h * 2, w * 2, outputChannels,
    ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["patch_embedding.weight"] = [imgIn.weight.name]
      mapping["patch_embedding.bias"] = [imgIn.bias.name]
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      mapping["head.head.weight"] = [projOut.weight.name]
      mapping["head.head.bias"] = [projOut.bias.name]
    case .diffusers:
      mapping["patch_embedding.weight"] = [imgIn.weight.name]
      mapping["patch_embedding.bias"] = [imgIn.bias.name]
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      mapping["proj_out.weight"] = [projOut.weight.name]
      mapping["proj_out.bias"] = [projOut.bias.name]
    }
    return mapping
  }
  return (
    mapper,
    Model(
      [x] + (residualIn.map { [$0] } ?? []) + rotAndtOut + contextIn + [scale, shift],
      [out] + (residualOut.map { [$0] } ?? []))
  )
}

private func LoRAWanAttentionBlockFixed(
  prefix: String, weightsPrefix: String, k: Int, h: Int, b: (Int, Int), t: (Int, Int),
  injectImage: Bool, layerIndex: Int,
  configuration: LoRANetworkConfiguration
) -> (ModelWeightMapper, Model) {
  let context = Input()
  // Now run attention.
  let contextToKeys = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "\(weightsPrefix)c_k")
  let contextToValues = LoRADense(
    count: k * h, configuration: configuration, index: layerIndex, name: "\(weightsPrefix)c_v")
  var cK = contextToKeys(context)
  let contextNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)c_norm_k")
  cK = contextNormK(cK).reshaped([b.0, t.0, h, k])
  let cV = contextToValues(context).reshaped([b.0, t.0, h, k])
  var ins = [context]
  var outs = [cK, cV]
  let cImgToKeys: Model?
  let cImgToValues: Model?
  let cImgNormK: Model?
  if injectImage {
    let contextImg = Input()
    let contextImgToKeys = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "\(weightsPrefix)c_img_k"
    )
    let contextImgToValues = LoRADense(
      count: k * h, configuration: configuration, index: layerIndex, name: "\(weightsPrefix)c_img_v"
    )
    var cImgK = contextImgToKeys(contextImg)
    let contextImgNormK = RMSNorm(epsilon: 1e-6, axis: [2], name: "\(weightsPrefix)c_img_norm_k")
    cImgK = contextImgNormK(cImgK).reshaped([b.1, t.1, h, k])
    let cImgV = contextImgToValues(contextImg).reshaped([b.1, t.1, h, k])
    ins.append(contextImg)
    outs.append(contentsOf: [cImgK, cImgV])
    cImgToKeys = contextImgToKeys
    cImgToValues = contextImgToValues
    cImgNormK = contextImgNormK
  } else {
    cImgToKeys = nil
    cImgToValues = nil
    cImgNormK = nil
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix).cross_attn.k.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix).cross_attn.k.bias"] = [contextToKeys.bias.name]
      mapping["\(prefix).cross_attn.v.weight"] = [contextToValues.weight.name]
      mapping["\(prefix).cross_attn.v.bias"] = [contextToValues.bias.name]
      mapping["\(prefix).cross_attn.norm_k.weight"] = [contextNormK.weight.name]
      if let cImgToKeys = cImgToKeys, let cImgToValues = cImgToValues, let cImgNormK = cImgNormK {
        mapping["\(prefix).cross_attn.k_img.weight"] = [cImgToKeys.weight.name]
        mapping["\(prefix).cross_attn.k_img.bias"] = [cImgToKeys.bias.name]
        mapping["\(prefix).cross_attn.v_img.weight"] = [cImgToValues.weight.name]
        mapping["\(prefix).cross_attn.v_img.bias"] = [cImgToValues.bias.name]
        mapping["\(prefix).cross_attn.norm_k_img.weight"] = [cImgNormK.weight.name]
      }
    case .diffusers:
      mapping["\(prefix).attn2.to_k.weight"] = [contextToKeys.weight.name]
      mapping["\(prefix).attn2.to_k.bias"] = [contextToKeys.bias.name]
      mapping["\(prefix).attn2.to_v.weight"] = [contextToValues.weight.name]
      mapping["\(prefix).attn2.to_v.bias"] = [contextToValues.bias.name]
      mapping["\(prefix).attn2.norm_k.weight"] = [contextNormK.weight.name]
      if let cImgToKeys = cImgToKeys, let cImgToValues = cImgToValues, let cImgNormK = cImgNormK {
        mapping["\(prefix).attn2.add_k_proj.weight"] = [cImgToKeys.weight.name]
        mapping["\(prefix).attn2.add_k_proj.bias"] = [cImgToKeys.bias.name]
        mapping["\(prefix).attn2.add_v_proj.weight"] = [cImgToValues.weight.name]
        mapping["\(prefix).attn2.add_v_proj.bias"] = [cImgToValues.bias.name]
        mapping["\(prefix).attn2.norm_added_k.weight"] = [cImgNormK.weight.name]
      }
    }
    return mapping
  }
  return (mapper, Model(ins, outs))
}

func LoRAWanFixed(
  timesteps: Int, batchSize: (Int, Int), channels: Int, layers: Int, vaceLayers: [Int],
  textLength: Int, injectImage: Bool, LoRAConfiguration: LoRANetworkConfiguration
) -> (
  ModelWeightMapper, Model
) {
  let txt = Input()
  let (cLinear1, cLinear2, contextEmbedder) = LoRAMLPEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "c")
  let context = contextEmbedder(txt)
  let t = Input()
  let (timeInMlp0, timeInMlp2, timeIn) = LoRATimeEmbedder(
    channels: channels, configuration: LoRAConfiguration, name: "t")
  let vector = timeIn(t).reshaped([timesteps, 1, channels])
  let vectorIn = vector.swish()
  let timeProjections = (0..<6).map {
    LoRADense(count: channels, configuration: LoRAConfiguration, name: "ada_ln_\($0)")
  }
  var outs = timeProjections.map { $0(vectorIn).identity().identity().identity() }  // Have duplicate name ada_ln_0 / ada_ln_1, now have to push the order to make sure the proper weights are loaded :(.
  var ins = [txt, t]
  if !vaceLayers.isEmpty {
    let vaceContext = Input()
    let vaceIn = Convolution(
      groups: 1, filters: channels, filterSize: [2, 2],
      hint: Hint(stride: [2, 2]), format: .OIHW, name: "vace_x_embedder")
    let beforeProj = Convolution(
      groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
      name: "vace_before_proj")
    outs.append(beforeProj(vaceIn(vaceContext)))
    ins.append(vaceContext)
  }
  let contextImg: Model.IO?
  let clipLn1: Model?
  let clipLn2: Model?
  let clipMlp0: Model?
  let clipMlp2: Model?
  if injectImage {
    let img = Input()
    let clipIn: Model
    (clipLn1, clipLn2, clipMlp0, clipMlp2, clipIn) = LoRAMLPProj(
      inChannels: 1_280, outChannels: channels, configuration: LoRAConfiguration, name: "clip")
    contextImg = clipIn(img)
    ins.append(img)
  } else {
    clipLn1 = nil
    clipLn2 = nil
    clipMlp0 = nil
    clipMlp2 = nil
    contextImg = nil
  }
  var mappers = [ModelWeightMapper]()
  for (i, _) in vaceLayers.enumerated() {
    let (mapper, block) = WanAttentionBlockFixed(
      prefix: "vace_blocks.\(i)", weightsPrefix: "vace_", k: 128, h: channels / 128, b: batchSize,
      t: (textLength, 257), injectImage: false)
    outs.append(block(context))
    mappers.append(mapper)
  }
  for i in 0..<layers {
    let (mapper, block) = LoRAWanAttentionBlockFixed(
      prefix: "blocks.\(i)", weightsPrefix: "", k: 128, h: channels / 128, b: batchSize,
      t: (textLength, 257),
      injectImage: injectImage, layerIndex: i, configuration: LoRAConfiguration)
    if let contextImg = contextImg {
      outs.append(block(context, contextImg))
    } else {
      outs.append(block(context))
    }
    mappers.append(mapper)
  }
  let scale = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_0")
  let shift = Parameter<Float>(.GPU(0), .HWC(1, 1, channels), name: "ada_ln_1")
  outs.append(1 + scale + vector)
  outs.append(vector + shift)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["text_embedding.0.weight"] = [cLinear1.weight.name]
      mapping["text_embedding.0.bias"] = [cLinear1.bias.name]
      mapping["text_embedding.2.weight"] = [cLinear2.weight.name]
      mapping["text_embedding.2.bias"] = [cLinear2.bias.name]
      mapping["time_embedding.0.weight"] = [timeInMlp0.weight.name]
      mapping["time_embedding.0.bias"] = [timeInMlp0.bias.name]
      mapping["time_embedding.2.weight"] = [timeInMlp2.weight.name]
      mapping["time_embedding.2.bias"] = [timeInMlp2.bias.name]
      if let clipLn1 = clipLn1, let clipLn2 = clipLn2, let clipMlp0 = clipMlp0,
        let clipMlp2 = clipMlp2
      {
        mapping["img_emb.proj.0.weight"] = [clipLn1.weight.name]
        mapping["img_emb.proj.0.bias"] = [clipLn1.bias.name]
        mapping["img_emb.proj.1.weight"] = [clipMlp0.weight.name]
        mapping["img_emb.proj.1.bias"] = [clipMlp0.bias.name]
        mapping["img_emb.proj.3.weight"] = [clipMlp2.weight.name]
        mapping["img_emb.proj.3.bias"] = [clipMlp2.bias.name]
        mapping["img_emb.proj.4.weight"] = [clipLn2.weight.name]
        mapping["img_emb.proj.4.bias"] = [clipLn2.bias.name]
      }
      mapping["time_projection.1.weight"] = ModelWeightElement(
        (0..<6).map { timeProjections[$0].weight.name })
      mapping["time_projection.1.bias"] = ModelWeightElement(
        (0..<6).map { timeProjections[$0].bias.name })
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      mapping["head.modulation"] = [shift.weight.name, scale.weight.name]
    case .diffusers:
      mapping["condition_embedder.text_embedder.linear_1.weight"] = [cLinear1.weight.name]
      mapping["condition_embedder.text_embedder.linear_1.bias"] = [cLinear1.bias.name]
      mapping["condition_embedder.text_embedder.linear_2.weight"] = [cLinear2.weight.name]
      mapping["condition_embedder.text_embedder.linear_2.bias"] = [cLinear2.bias.name]
      mapping["condition_embedder.time_embedder.linear_1.weight"] = [timeInMlp0.weight.name]
      mapping["condition_embedder.time_embedder.linear_1.bias"] = [timeInMlp0.bias.name]
      mapping["condition_embedder.time_embedder.linear_2.weight"] = [timeInMlp2.weight.name]
      mapping["condition_embedder.time_embedder.linear_2.bias"] = [timeInMlp2.bias.name]
      if let clipLn1 = clipLn1, let clipLn2 = clipLn2, let clipMlp0 = clipMlp0,
        let clipMlp2 = clipMlp2
      {
        mapping["condition_embedder.image_embedder.norm1.weight"] = [clipLn1.weight.name]
        mapping["condition_embedder.image_embedder.norm1.bias"] = [clipLn1.bias.name]
        mapping["condition_embedder.image_embedder.ff.net.0.proj.weight"] = [clipMlp0.weight.name]
        mapping["condition_embedder.image_embedder.ff.net.0.proj.bias"] = [clipMlp0.bias.name]
        mapping["condition_embedder.image_embedder.ff.net.2.weight"] = [clipMlp2.weight.name]
        mapping["condition_embedder.image_embedder.ff.net.2.bias"] = [clipMlp2.bias.name]
        mapping["condition_embedder.image_embedder.norm2.weight"] = [clipLn2.weight.name]
        mapping["condition_embedder.image_embedder.norm2.bias"] = [clipLn2.bias.name]
      }
      mapping["condition_embedder.time_proj.weight"] = ModelWeightElement(
        (0..<6).map { timeProjections[$0].weight.name })
      mapping["condition_embedder.time_proj.bias"] = ModelWeightElement(
        (0..<6).map { timeProjections[$0].bias.name })
      for mapper in mappers {
        mapping.merge(mapper(format)) { v, _ in v }
      }
      mapping["scale_shift_table"] = [shift.weight.name, scale.weight.name]
    }
    return mapping
  }
  return (mapper, Model(ins, outs))
}
