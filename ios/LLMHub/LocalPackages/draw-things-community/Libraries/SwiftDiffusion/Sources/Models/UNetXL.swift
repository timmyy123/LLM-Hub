import DiffusionMappings
import Foundation
import NNC

/// UNet for Stable Diffusion XL

func LabelEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out]))
}

func TimePosEmbed(modelChannels: Int) -> (Model, Model, Model) {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels)
  out = fc2(out)
  return (fc0, fc2, Model([x], [out], name: "time_pos_embed"))
}

func TimeResBlock(b: Int, h: Int, w: Int, channels: Int, flags: Functional.GEMMFlag) -> (
  Model, Model, Model, Model, Model, Model
) {
  let x = Input()
  let emb = Input()
  let y = x.reshaped([1, b, h * w, channels])  // [b, h, w, c] -> [1, b, h * w, c]
  let inLayerNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  var out = inLayerNorm(y)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: channels, filterSize: [3, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 0], end: [1, 0])), format: .OIHW)
  out = inLayerConv2d(out)
  let embLayer = Dense(count: channels, flags: flags)
  var embOut = Swish()(emb)
  embOut = embLayer(embOut).reshaped([1, 1, 1, channels])
  out = out + embOut
  let outLayerNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: channels, filterSize: [3, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 0], end: [1, 0])), format: .OIHW)
  out = y + outLayerConv2d(out)  // This layer should be zero init if training.
  out = out.reshaped([b, h, w, channels])
  return (
    inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d,
    Model([x, emb], [out], name: "time_stack")
  )
}

func CrossAttentionKeysAndValues(
  k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), upcastAttention: Bool,
  injectIPAdapterLengths: [Int], usesFlashAttention: FlashAttentionLevel, flags: Functional.GEMMFlag
) -> (
  Model, Model, Model
) {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = Dense(count: k * h, noBias: true, flags: flags)
  if usesFlashAttention != .none {
    if t.0 == t.1 {
      let queries = toqueries(x).reshaped([b, hw, h, k])
      if injectIPAdapterLengths.count > 0 {
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: upcastAttention
            ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))
        var out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, h * k])
        var ipKVs = [Input]()
        for _ in injectIPAdapterLengths {
          let ipKeys = Input()
          let ipValues = Input()
          let scaledDotProductAttention = ScaledDotProductAttention(
            scale: 1.0 / Float(k).squareRoot(),
            flags: upcastAttention
              ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))
          out = out + scaledDotProductAttention(queries, ipKeys, ipValues).reshaped([b, hw, h * k])
          ipKVs.append(contentsOf: [ipKeys, ipValues])
        }
        let unifyheads = Dense(count: k * h, flags: flags)
        out = unifyheads(out)
        return (toqueries, unifyheads, Model([x, keys, values] + ipKVs, [out]))
      } else {
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(),
          flags: upcastAttention ? [] : [.Float16],
          multiHeadOutputProjectionFused: true)
        let out = scaledDotProductAttention(queries, keys, values)
        return (toqueries, scaledDotProductAttention, Model([x, keys, values], [out]))
      }
    } else {
      let queries = toqueries(x).reshaped([b, hw, h, k])
      let b0 = b / 2
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
          scale: 1.0 / Float(k).squareRoot(),
          flags: upcastAttention
            ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        )(queries0, keys0, values0).reshaped([b0, hw, h * k])
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
              scale: 1.0 / Float(k).squareRoot(),
              flags: upcastAttention
                ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
            )(queries0, keys0, values0).reshaped([1, hw, h * k]))
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
          scale: 1.0 / Float(k).squareRoot(),
          flags: upcastAttention
            ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
        )(queries1, keys1, values1).reshaped([b0, hw, h * k])
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
              scale: 1.0 / Float(k).squareRoot(),
              flags: upcastAttention
                ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16])
            )(queries1, keys1, values1).reshaped([1, hw, h * k]))
        }
        out1 = Concat(axis: 0)(outs)
      }
      var out = Functional.concat(axis: 0, out0, out1)
      var ipKVs = [Input]()
      for _ in injectIPAdapterLengths {
        let ipKeys = Input()
        let ipValues = Input()
        let scaledDotProductAttention = ScaledDotProductAttention(
          scale: 1.0 / Float(k).squareRoot(), flags: upcastAttention ? [] : (usesFlashAttention == .quantized ? [.Int8, .Float16] : [.Float16]))
        out = out + scaledDotProductAttention(queries, ipKeys, ipValues).reshaped([b, hw, h * k])
        ipKVs.append(contentsOf: [ipKeys, ipValues])
      }
      let unifyheads = Dense(count: k * h, flags: flags)
      out = unifyheads(out)
      return (toqueries, unifyheads, Model([x, keys, values] + ipKVs, [out]))
    }
  } else {
    var out: Model.IO
    var queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      .transposed(1, 2)
    if t.0 == t.1 {
      let t = t.0
      var keys: Model.IO = keys
      if upcastAttention {
        keys = keys.to(.Float32)
        queries = queries.to(.Float32)
      }
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * hw, t])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: values)
      }
      dot = dot.reshaped([b, h, hw, t])
      out = dot * values
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b * hw, h * k])
    } else {
      let b0 = b / 2
      var keys0 = keys.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var queries0 = queries.reshaped([b0, h, hw, k])
      if upcastAttention {
        keys0 = keys0.to(.Float32)
        queries0 = queries0.to(.Float32)
      }
      let values0 = values.reshaped(
        [b0, t.0, h, k], offset: [0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot0 = Matmul(transposeB: (2, 3))(queries0, keys0)
      dot0 = dot0.reshaped([b0 * h * hw, t.0])
      dot0 = dot0.softmax()
      if upcastAttention {
        dot0 = dot0.to(of: values)
      }
      dot0 = dot0.reshaped([b0, h, hw, t.0])
      var out0 = dot0 * values0
      out0 = out0.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      var keys1 = keys.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var queries1 = queries.reshaped(
        [b0, h, hw, k], offset: [b0, 0, 0, 0], strides: [h * hw * k, hw * k, k, 1])
      if upcastAttention {
        keys1 = keys1.to(.Float32)
        queries1 = queries1.to(.Float32)
      }
      let values1 = values.reshaped(
        [b0, t.1, h, k], offset: [b0, 0, 0, 0], strides: [max(t.0, t.1) * h * k, h * k, k, 1]
      ).transposed(1, 2)
      var dot1 = Matmul(transposeB: (2, 3))(queries1, keys1)
      dot1.add(dependencies: [out0])
      dot1 = dot1.reshaped([b0 * h * hw, t.1])
      dot1 = dot1.softmax()
      if upcastAttention {
        dot1 = dot1.to(of: values)
      }
      dot1 = dot1.reshaped([b0, h, hw, t.1])
      var out1 = dot1 * values1
      out1 = out1.reshaped([b0, h, hw, k]).transposed(1, 2).reshaped([b0, hw, h * k])
      out = Functional.concat(axis: 0, out0, out1)
    }
    var ipKVs = [Input]()
    for injectIPAdapterLength in injectIPAdapterLengths {
      let ipKeys = Input()
      let ipValues = Input()
      var dot = Matmul(transposeB: (2, 3))(queries, ipKeys)
      dot = dot.reshaped([b * h * hw, injectIPAdapterLength])
      dot = dot.softmax()
      if upcastAttention {
        dot = dot.to(of: values)
      }
      dot = dot.reshaped([b, h, hw, injectIPAdapterLength])
      out = out + (dot * ipValues).transposed(1, 2).reshaped([b, hw, h * k])
      ipKVs.append(contentsOf: [ipKeys, ipValues])
    }
    let unifyheads = Dense(count: k * h, flags: flags)
    out = unifyheads(out).reshaped([b, hw, h * k])
    return (toqueries, unifyheads, Model([x, keys, values] + ipKVs, [out]))
  }
}

private func BasicTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  flags: Functional.GEMMFlag, isTemporalMixEnabled: Bool
) -> (PythonReader, ModelWeightMapper, Model) {
  let x = Input()
  let values = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(
    k: k, h: h, b: b, hw: hw, upcastAttention: upcastAttention,
    usesFlashAttention: usesFlashAttention, flags: flags, injectedAttentionKV: false)
  out = attn1(out) + x
  var residual = out
  let keys: Input?
  let layerNorm2: Model?
  let toqueries2: Model?
  let unifyheads2: Model?
  let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
  if t.0 == 1 && t.1 == 1 {
    out = values + residual
    keys = nil
    layerNorm2 = nil
    toqueries2 = nil
    unifyheads2 = nil
  } else {
    let keys2 = Input()
    let layerNorm = LayerNorm(epsilon: 1e-5, axis: [2])
    out = layerNorm(out)
    let (toqueries, unifyheads, attn2) = CrossAttentionKeysAndValues(
      k: k, h: h, b: b, hw: hw, t: t, upcastAttention: upcastAttention,
      injectIPAdapterLengths: injectIPAdapterLengths,
      usesFlashAttention: isTemporalMixEnabled ? .none : usesFlashAttention, flags: flags)
    out = attn2([out, keys2, values] + ipKVs) + residual
    keys = keys2
    layerNorm2 = layerNorm
    toqueries2 = toqueries
    unifyheads2 = unifyheads
  }
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let (fc10, fc11, fc2, ff) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, flags: flags)
  out = ff(out) + residual
  let reader: PythonReader = { stateDict, archive in
    guard
      let attn1_to_k_weight = stateDict[
        "\(prefix.0).attn1.to_k.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tokeys1.weight.copy(from: attn1_to_k_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_q_weight = stateDict[
        "\(prefix.0).attn1.to_q.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try toqueries1.weight.copy(from: attn1_to_q_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_v_weight = stateDict[
        "\(prefix.0).attn1.to_v.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tovalues1.weight.copy(from: attn1_to_v_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_out_weight = stateDict[
        "\(prefix.0).attn1.to_out.0.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let attn1_to_out_bias = stateDict[
        "\(prefix.0).attn1.to_out.0.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try unifyheads1.weight.copy(from: attn1_to_out_weight, zip: archive, of: FloatType.self)
    try unifyheads1.bias.copy(from: attn1_to_out_bias, zip: archive, of: FloatType.self)
    guard
      let ff_net_0_proj_weight = try stateDict[
        "\(prefix.0).ff.net.0.proj.weight"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_net_0_proj_bias = try stateDict[
        "\(prefix.0).ff.net.0.proj.bias"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    fc10.weight.copy(
      from: ff_net_0_proj_weight[0..<intermediateSize, 0..<ff_net_0_proj_weight.shape[1]])
    fc10.bias.copy(from: ff_net_0_proj_bias[0..<intermediateSize])
    fc11.weight.copy(
      from: ff_net_0_proj_weight[
        intermediateSize..<ff_net_0_proj_weight.shape[0], 0..<ff_net_0_proj_weight.shape[1]])
    fc11.bias.copy(from: ff_net_0_proj_bias[intermediateSize..<ff_net_0_proj_bias.shape[0]])
    guard
      let ff_net_2_weight = stateDict[
        "\(prefix.0).ff.net.2.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_net_2_bias = stateDict[
        "\(prefix.0).ff.net.2.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try fc2.weight.copy(from: ff_net_2_weight, zip: archive, of: FloatType.self)
    try fc2.bias.copy(from: ff_net_2_bias, zip: archive, of: FloatType.self)
    if let layerNorm2 = layerNorm2, let toqueries2 = toqueries2, let unifyheads2 = unifyheads2 {
      guard
        let norm2_weight = stateDict[
          "\(prefix.0).norm2.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let norm2_bias = stateDict[
          "\(prefix.0).norm2.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try layerNorm2.weight.copy(from: norm2_weight, zip: archive, of: FloatType.self)
      try layerNorm2.bias.copy(from: norm2_bias, zip: archive, of: FloatType.self)
      guard
        let attn2_to_q_weight = stateDict[
          "\(prefix.0).attn2.to_q.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try toqueries2.weight.copy(from: attn2_to_q_weight, zip: archive, of: FloatType.self)
      guard
        let attn2_to_out_weight = stateDict[
          "\(prefix.0).attn2.to_out.0.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let attn2_to_out_bias = stateDict[
          "\(prefix.0).attn2.to_out.0.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try unifyheads2.weight.copy(from: attn2_to_out_weight, zip: archive, of: FloatType.self)
      try unifyheads2.bias.copy(from: attn2_to_out_bias, zip: archive, of: FloatType.self)
    }
    guard
      let norm1_weight = stateDict[
        "\(prefix.0).norm1.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm1_bias = stateDict[
        "\(prefix.0).norm1.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm1.weight.copy(from: norm1_weight, zip: archive, of: FloatType.self)
    try layerNorm1.bias.copy(from: norm1_bias, zip: archive, of: FloatType.self)
    guard
      let norm3_weight = stateDict[
        "\(prefix.0).norm3.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm3_bias = stateDict[
        "\(prefix.0).norm3.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm3.weight.copy(from: norm3_weight, zip: archive, of: FloatType.self)
    try layerNorm3.bias.copy(from: norm3_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let formatPrefix: String
    switch format {
    case .generativeModels:
      formatPrefix = prefix.0
    case .diffusers:
      formatPrefix = prefix.1
    }
    mapping["\(formatPrefix).attn1.to_k.weight"] = [tokeys1.weight.name]
    mapping["\(formatPrefix).attn1.to_q.weight"] = [toqueries1.weight.name]
    mapping["\(formatPrefix).attn1.to_v.weight"] = [tovalues1.weight.name]
    mapping["\(formatPrefix).attn1.to_out.0.weight"] = [unifyheads1.weight.name]
    mapping["\(formatPrefix).attn1.to_out.0.bias"] = [unifyheads1.bias.name]
    mapping["\(formatPrefix).ff.net.0.proj.weight"] = [fc10.weight.name, fc11.weight.name]
    mapping["\(formatPrefix).ff.net.0.proj.bias"] = [fc10.bias.name, fc11.bias.name]
    mapping["\(formatPrefix).ff.net.2.weight"] = [fc2.weight.name]
    mapping["\(formatPrefix).ff.net.2.bias"] = [fc2.bias.name]
    if let layerNorm2 = layerNorm2, let toqueries2 = toqueries2, let unifyheads2 = unifyheads2 {
      mapping["\(formatPrefix).norm2.weight"] = [layerNorm2.weight.name]
      mapping["\(formatPrefix).norm2.bias"] = [layerNorm2.bias.name]
      mapping["\(formatPrefix).attn2.to_q.weight"] = [toqueries2.weight.name]
      mapping["\(formatPrefix).attn2.to_out.0.weight"] = [unifyheads2.weight.name]
      mapping["\(formatPrefix).attn2.to_out.0.bias"] = [unifyheads2.bias.name]
    }
    mapping["\(formatPrefix).norm1.weight"] = [layerNorm1.weight.name]
    mapping["\(formatPrefix).norm1.bias"] = [layerNorm1.bias.name]
    mapping["\(formatPrefix).norm3.weight"] = [layerNorm3.weight.name]
    mapping["\(formatPrefix).norm3.bias"] = [layerNorm3.bias.name]
    return mapping
  }
  if let keys = keys {
    return (reader, mapper, Model([x, keys, values] + ipKVs, [out]))
  } else {
    return (reader, mapper, Model([x, values] + ipKVs, [out]))
  }
}

func BasicTimeTransformerBlock(
  prefix: (String, String), k: Int, h: Int, b: Int, hw: Int, t: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], usesFlashAttention: FlashAttentionLevel, flags: Functional.GEMMFlag
) -> (PythonReader, ModelWeightMapper, Model) {
  let x = Input()
  let timeEmb = Input()
  let values = Input()
  var out = x.transposed(0, 1) + timeEmb.reshaped([1, b, k * h])
  let normIn = LayerNorm(epsilon: 1e-5, axis: [2])
  let (ffIn10, ffIn11, ffIn2, ffIn) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, flags: flags)
  out = ffIn(normIn(out).reshaped([hw * b, k * h])).reshaped([hw, b, k * h]) + out
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let (tokeys1, toqueries1, tovalues1, unifyheads1, attn1) = SelfAttention(
    k: k, h: h, b: hw, hw: b, upcastAttention: false, usesFlashAttention: usesFlashAttention,
    flags: flags, injectedAttentionKV: false)
  out = attn1(layerNorm1(out).reshaped([hw * b, k * h])) + out
  var residual = out
  let keys: Input?
  let layerNorm2: Model?
  let toqueries2: Model?
  let unifyheads2: Model?
  if t.0 == 1 && t.1 == 1 {
    out = values + residual
    keys = nil
    layerNorm2 = nil
    toqueries2 = nil
    unifyheads2 = nil
  } else {
    let keys2 = Input()
    let layerNorm = LayerNorm(epsilon: 1e-5, axis: [2])
    out = layerNorm(out).reshaped([hw * b, k * h])
    let (toqueries, unifyheads, attn2) = CrossAttentionKeysAndValues(
      k: k, h: h, b: hw, hw: b, t: t, upcastAttention: false,
      injectIPAdapterLengths: injectIPAdapterLengths, usesFlashAttention: .none, flags: flags)
    out = attn2(out, keys2, values) + residual
    keys = keys2
    layerNorm2 = layerNorm
    toqueries2 = toqueries
    unifyheads2 = unifyheads
  }
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out).reshaped([hw * b, k * h])
  let (fc10, fc11, fc2, ff) = FeedForward(
    hiddenSize: k * h, intermediateSize: intermediateSize, flags: flags)
  out = ff(out).reshaped([hw, b, k * h]) + residual
  out = out.transposed(0, 1)
  let reader: PythonReader = { stateDict, archive in
    guard
      let norm_in_weight = stateDict[
        "\(prefix.0).norm_in.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm_in_bias = stateDict[
        "\(prefix.0).norm_in.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try normIn.weight.copy(from: norm_in_weight, zip: archive, of: FloatType.self)
    try normIn.bias.copy(from: norm_in_bias, zip: archive, of: FloatType.self)
    guard
      let ff_in_net_0_proj_weight = try stateDict[
        "\(prefix.0).ff_in.net.0.proj.weight"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_in_net_0_proj_bias = try stateDict[
        "\(prefix.0).ff_in.net.0.proj.bias"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    ffIn10.weight.copy(
      from: ff_in_net_0_proj_weight[0..<intermediateSize, 0..<ff_in_net_0_proj_weight.shape[1]])
    ffIn10.bias.copy(
      from: ff_in_net_0_proj_bias[0..<intermediateSize])
    ffIn11.weight.copy(
      from: ff_in_net_0_proj_weight[
        intermediateSize..<ff_in_net_0_proj_weight.shape[0], 0..<ff_in_net_0_proj_weight.shape[1]])
    ffIn11.bias.copy(
      from: ff_in_net_0_proj_bias[intermediateSize..<ff_in_net_0_proj_bias.shape[0]])
    guard
      let ff_in_net_2_weight = stateDict[
        "\(prefix.0).ff_in.net.2.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_in_net_2_bias = stateDict[
        "\(prefix.0).ff_in.net.2.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try ffIn2.weight.copy(from: ff_in_net_2_weight, zip: archive, of: FloatType.self)
    try ffIn2.bias.copy(from: ff_in_net_2_bias, zip: archive, of: FloatType.self)
    guard
      let attn1_to_k_weight = stateDict[
        "\(prefix.0).attn1.to_k.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tokeys1.weight.copy(from: attn1_to_k_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_q_weight = stateDict[
        "\(prefix.0).attn1.to_q.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try toqueries1.weight.copy(from: attn1_to_q_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_v_weight = stateDict[
        "\(prefix.0).attn1.to_v.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try tovalues1.weight.copy(from: attn1_to_v_weight, zip: archive, of: FloatType.self)
    guard
      let attn1_to_out_weight = stateDict[
        "\(prefix.0).attn1.to_out.0.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let attn1_to_out_bias = stateDict[
        "\(prefix.0).attn1.to_out.0.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try unifyheads1.weight.copy(
      from: attn1_to_out_weight, zip: archive, of: FloatType.self)
    try unifyheads1.bias.copy(from: attn1_to_out_bias, zip: archive, of: FloatType.self)
    guard
      let ff_net_0_proj_weight = try stateDict[
        "\(prefix.0).ff.net.0.proj.weight"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_net_0_proj_bias = try stateDict[
        "\(prefix.0).ff.net.0.proj.bias"
      ]?.inflate(from: archive, of: FloatType.self)
    else {
      throw UnpickleError.tensorNotFound
    }
    fc10.weight.copy(
      from: ff_net_0_proj_weight[0..<intermediateSize, 0..<ff_net_0_proj_weight.shape[1]])
    fc10.bias.copy(
      from: ff_net_0_proj_bias[0..<intermediateSize])
    fc11.weight.copy(
      from: ff_net_0_proj_weight[
        intermediateSize..<ff_net_0_proj_weight.shape[0], 0..<ff_net_0_proj_weight.shape[1]])
    fc11.bias.copy(
      from: ff_net_0_proj_bias[intermediateSize..<ff_net_0_proj_bias.shape[0]])
    guard
      let ff_net_2_weight = stateDict[
        "\(prefix.0).ff.net.2.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let ff_net_2_bias = stateDict[
        "\(prefix.0).ff.net.2.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try fc2.weight.copy(from: ff_net_2_weight, zip: archive, of: FloatType.self)
    try fc2.bias.copy(from: ff_net_2_bias, zip: archive, of: FloatType.self)
    if let layerNorm2 = layerNorm2, let toqueries2 = toqueries2, let unifyheads2 = unifyheads2 {
      guard
        let norm2_weight = stateDict[
          "\(prefix.0).norm2.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let norm2_bias = stateDict[
          "\(prefix.0).norm2.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try layerNorm2.weight.copy(from: norm2_weight, zip: archive, of: FloatType.self)
      try layerNorm2.bias.copy(from: norm2_bias, zip: archive, of: FloatType.self)
      guard
        let attn2_to_q_weight = stateDict[
          "\(prefix.0).attn2.to_q.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try toqueries2.weight.copy(from: attn2_to_q_weight, zip: archive, of: FloatType.self)
      guard
        let attn2_to_out_weight = stateDict[
          "\(prefix.0).attn2.to_out.0.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let attn2_to_out_bias = stateDict[
          "\(prefix.0).attn2.to_out.0.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try unifyheads2.weight.copy(from: attn2_to_out_weight, zip: archive, of: FloatType.self)
      try unifyheads2.bias.copy(from: attn2_to_out_bias, zip: archive, of: FloatType.self)
    }
    guard
      let norm1_weight = stateDict[
        "\(prefix.0).norm1.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm1_bias = stateDict[
        "\(prefix.0).norm1.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm1.weight.copy(from: norm1_weight, zip: archive, of: FloatType.self)
    try layerNorm1.bias.copy(from: norm1_bias, zip: archive, of: FloatType.self)
    guard
      let norm3_weight = stateDict[
        "\(prefix.0).norm3.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let norm3_bias = stateDict[
        "\(prefix.0).norm3.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try layerNorm3.weight.copy(from: norm3_weight, zip: archive, of: FloatType.self)
    try layerNorm3.bias.copy(from: norm3_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    let formatPrefix: String
    switch format {
    case .generativeModels:
      formatPrefix = prefix.0
    case .diffusers:
      formatPrefix = prefix.1
    }
    mapping["\(formatPrefix).norm_in.weight"] = [normIn.weight.name]
    mapping["\(formatPrefix).norm_in.bias"] = [normIn.bias.name]
    mapping["\(formatPrefix).ff_in.net.0.proj.weight"] = [ffIn10.weight.name, ffIn11.weight.name]
    mapping["\(formatPrefix).ff_in.net.0.proj.bias"] = [ffIn10.bias.name, ffIn11.bias.name]
    mapping["\(formatPrefix).ff_in.net.2.weight"] = [ffIn2.weight.name]
    mapping["\(formatPrefix).ff_in.net.2.bias"] = [ffIn2.bias.name]
    mapping["\(formatPrefix).attn1.to_k.weight"] = [tokeys1.weight.name]
    mapping["\(formatPrefix).attn1.to_q.weight"] = [toqueries1.weight.name]
    mapping["\(formatPrefix).attn1.to_v.weight"] = [tovalues1.weight.name]
    mapping["\(formatPrefix).attn1.to_out.0.weight"] = [unifyheads1.weight.name]
    mapping["\(formatPrefix).attn1.to_out.0.bias"] = [unifyheads1.bias.name]
    mapping["\(formatPrefix).ff.net.0.proj.weight"] = [fc10.weight.name, fc11.weight.name]
    mapping["\(formatPrefix).ff.net.0.proj.bias"] = [fc10.bias.name, fc11.bias.name]
    mapping["\(formatPrefix).ff.net.2.weight"] = [fc2.weight.name]
    mapping["\(formatPrefix).ff.net.2.bias"] = [fc2.bias.name]
    if let layerNorm2 = layerNorm2, let toqueries2 = toqueries2, let unifyheads2 = unifyheads2 {
      mapping["\(formatPrefix).norm2.weight"] = [layerNorm2.weight.name]
      mapping["\(formatPrefix).norm2.bias"] = [layerNorm2.bias.name]
      mapping["\(formatPrefix).attn2.to_q.weight"] = [toqueries2.weight.name]
      mapping["\(formatPrefix).attn2.to_out.0.weight"] = [unifyheads2.weight.name]
      mapping["\(formatPrefix).attn2.to_out.0.bias"] = [unifyheads2.bias.name]
    }
    mapping["\(formatPrefix).norm1.weight"] = [layerNorm1.weight.name]
    mapping["\(formatPrefix).norm1.bias"] = [layerNorm1.bias.name]
    mapping["\(formatPrefix).norm3.weight"] = [layerNorm3.weight.name]
    mapping["\(formatPrefix).norm3.bias"] = [layerNorm3.bias.name]
    return mapping
  }
  if let keys = keys {
    return (reader, mapper, Model([x, timeEmb, keys, values], [out], name: "time_stack"))
  } else {
    return (reader, mapper, Model([x, timeEmb, values], [out], name: "time_stack"))
  }
}

private func SpatialTransformer<FloatType: TensorNumeric & BinaryFloatingPoint>(
  prefix: (String, String),
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: (Int, Int),
  intermediateSize: Int, injectIPAdapterLengths: [Int], upcastAttention: Bool,
  usesFlashAttention: FlashAttentionLevel, flags: Functional.GEMMFlag, isTemporalMixEnabled: Bool,
  of: FloatType.Type = FloatType.self
) -> (PythonReader, ModelWeightMapper, Model) {
  let x = Input()
  var kvs = [Model.IO]()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1], format: .OIHW)
  let hw = height * width
  out = projIn(out).reshaped([b, hw, k * h])
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  let timeEmb: Input?
  let mixFactor: Parameter<FloatType>?
  if depth > 0 && isTemporalMixEnabled {
    let emb = Input()
    kvs.append(emb)
    timeEmb = emb
    mixFactor = Parameter<FloatType>(.GPU(0), format: .NHWC, shape: [1], name: "time_mixer")
  } else {
    timeEmb = nil
    mixFactor = nil
  }
  for i in 0..<depth {
    if t.0 == 1 && t.1 == 1 {
      let values = Input()
      kvs.append(values)
      let (reader, mapper, block) = BasicTransformerBlock(
        prefix: ("\(prefix.0).transformer_blocks.\(i)", "\(prefix.1).transformer_blocks.\(i)"),
        k: k,
        h: h, b: b, hw: hw, t: t,
        intermediateSize: intermediateSize, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, flags: flags,
        isTemporalMixEnabled: isTemporalMixEnabled)
      let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
      kvs.append(contentsOf: ipKVs)
      out = block([out, values] + ipKVs)
      readers.append(reader)
      mappers.append(mapper)
      if let timeEmb = timeEmb, let mixFactor = mixFactor {
        let values = Input()
        kvs.append(values)
        let (reader, mapper, block) = BasicTimeTransformerBlock(
          prefix: ("\(prefix.0).time_stack.\(i)", "\(prefix.1).temporal_transformer_blocks.\(i)"),
          k: k, h: h, b: b,
          hw: hw, t: t, intermediateSize: intermediateSize,
          injectIPAdapterLengths: injectIPAdapterLengths, usesFlashAttention: usesFlashAttention,
          flags: flags)
        out = mixFactor .* out + (1 - mixFactor) .* block(out, timeEmb, values)
        readers.append(reader)
        mappers.append(mapper)
      }
    } else {
      let keys = Input()
      kvs.append(keys)
      let values = Input()
      kvs.append(values)
      let (reader, mapper, block) = BasicTransformerBlock(
        prefix: ("\(prefix.0).transformer_blocks.\(i)", "\(prefix.1).transformer_blocks.\(i)"),
        k: k,
        h: h, b: b, hw: hw, t: t,
        intermediateSize: intermediateSize, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention, flags: flags,
        isTemporalMixEnabled: isTemporalMixEnabled)
      let ipKVs = (0..<(injectIPAdapterLengths.count * 2)).map { _ in Input() }
      kvs.append(contentsOf: ipKVs)
      out = block([out, keys, values] + ipKVs)
      readers.append(reader)
      mappers.append(mapper)
      if let timeEmb = timeEmb, let mixFactor = mixFactor {
        let keys = Input()
        kvs.append(keys)
        let values = Input()
        kvs.append(values)
        let (reader, mapper, block) = BasicTimeTransformerBlock(
          prefix: ("\(prefix.0).time_stack.\(i)", "\(prefix.1).temporal_transformer_blocks.\(i)"),
          k: k, h: h, b: b,
          hw: hw, t: t, intermediateSize: intermediateSize,
          injectIPAdapterLengths: injectIPAdapterLengths, usesFlashAttention: usesFlashAttention,
          flags: flags)
        out = mixFactor .* out + (1 - mixFactor) .* block(out, timeEmb, keys, values)
        readers.append(reader)
        mappers.append(mapper)
      }
    }
  }
  out = out.reshaped([b, height, width, k * h])
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1], format: .OIHW)
  out = projOut(out) + x
  let reader: PythonReader = { stateDict, archive in
    guard let norm_weight = stateDict["\(prefix.0).norm.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_bias = stateDict["\(prefix.0).norm.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm.weight.copy(from: norm_weight, zip: archive, of: FloatType.self)
    try norm.bias.copy(from: norm_bias, zip: archive, of: FloatType.self)
    guard let proj_in_weight = stateDict["\(prefix.0).proj_in.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let proj_in_bias = stateDict["\(prefix.0).proj_in.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try projIn.weight.copy(from: proj_in_weight, zip: archive, of: FloatType.self)
    try projIn.bias.copy(from: proj_in_bias, zip: archive, of: FloatType.self)
    for reader in readers {
      try reader(stateDict, archive)
    }
    guard let proj_out_weight = stateDict["\(prefix.0).proj_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let proj_out_bias = stateDict["\(prefix.0).proj_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try projOut.weight.copy(from: proj_out_weight, zip: archive, of: FloatType.self)
    try projOut.bias.copy(from: proj_out_bias, zip: archive, of: FloatType.self)
    if let mixFactor = mixFactor {
      guard let mix_factor = stateDict["\(prefix.0).time_mixer.mix_factor"] else {
        throw UnpickleError.tensorNotFound
      }
      // Apply sigmod when loading the mix factor.
      try archive.with(mix_factor) {
        var tensor = Tensor<Float>(from: $0).toCPU()
        // Sigmoid.
        tensor[0] = 1.0 / (1.0 + expf(-tensor[0]))
        mixFactor.weight.copy(from: Tensor<FloatType>(from: tensor))
      }
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    let formatPrefix: String
    switch format {
    case .generativeModels:
      formatPrefix = prefix.0
    case .diffusers:
      formatPrefix = prefix.1
    }
    mapping["\(formatPrefix).norm.weight"] = [norm.weight.name]
    mapping["\(formatPrefix).norm.bias"] = [norm.bias.name]
    mapping["\(formatPrefix).proj_in.weight"] = [projIn.weight.name]
    mapping["\(formatPrefix).proj_in.bias"] = [projIn.bias.name]
    mapping["\(formatPrefix).proj_out.weight"] = [projOut.weight.name]
    mapping["\(formatPrefix).proj_out.bias"] = [projOut.bias.name]
    if let mixFactor = mixFactor {
      mapping["\(formatPrefix).time_mixer.mix_factor"] = [mixFactor.weight.name]
    }
    return mapping
  }
  return (reader, mapper, Model([x] + kvs, [out]))
}

func BlockLayer<FloatType: TensorNumeric & BinaryFloatingPoint>(
  prefix: (String, String),
  repeatStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingLength: (Int, Int), intermediateSize: Int,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  flags: Functional.GEMMFlag, isTemporalMixEnabled: Bool, of: FloatType.Type = FloatType.self
) -> (PythonReader, ModelWeightMapper, Model) {
  let x = Input()
  let emb = Input()
  var kvs = [Model.IO]()
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm, inLayerConv2d, embLayer, outLayerNorm, outLayerConv2d, skipModel, resBlock) =
    ResBlock(
      b: batchSize, outChannels: channels, skipConnection: skipConnection, flags: flags)
  var out = resBlock(x, emb)
  let timeInLayerNorm: Model?
  let timeInLayerConv2d: Model?
  let timeEmbLayer: Model?
  let timeOutLayerNorm: Model?
  let timeOutLayerConv2d: Model?
  let mixFactor: Parameter<FloatType>?
  if isTemporalMixEnabled {
    let timeResBlock: Model
    (
      timeInLayerNorm, timeInLayerConv2d, timeEmbLayer, timeOutLayerNorm, timeOutLayerConv2d,
      timeResBlock
    ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels, flags: flags)
    let mix = Parameter<FloatType>(.GPU(0), format: .NHWC, shape: [1], name: "time_mixer")
    out = mix .* out + (1 - mix) .* timeResBlock(out, emb)
    mixFactor = mix
  } else {
    timeInLayerNorm = nil
    timeInLayerConv2d = nil
    timeEmbLayer = nil
    timeOutLayerNorm = nil
    timeOutLayerConv2d = nil
    mixFactor = nil
  }
  var transformerReader: PythonReader? = nil
  var transformerMapper: ModelWeightMapper? = nil
  if attentionBlock > 0 {
    var c: [Input]
    if embeddingLength.0 == 1 && embeddingLength.1 == 1 {
      c = (0..<(attentionBlock * (injectIPAdapterLengths.count + 1))).map { _ in Input() }
      if isTemporalMixEnabled {
        c.append(contentsOf: (0..<(attentionBlock + 1)).map { _ in Input() })
      }
    } else {
      c = (0..<(attentionBlock * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input() }
      if isTemporalMixEnabled {
        c.append(contentsOf: (0..<(attentionBlock * 2 + 1)).map { _ in Input() })
      }
    }
    let transformer: Model
    (transformerReader, transformerMapper, transformer) = SpatialTransformer(
      prefix: ("\(prefix.0).1", "\(prefix.1).attentions.\(repeatStart)"),
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      depth: attentionBlock, t: embeddingLength,
      intermediateSize: channels * 4, injectIPAdapterLengths: injectIPAdapterLengths,
      upcastAttention: upcastAttention, usesFlashAttention: usesFlashAttention,
      flags: flags, isTemporalMixEnabled: isTemporalMixEnabled, of: FloatType.self)
    out = transformer([out] + c)
    kvs.append(contentsOf: c)
  }
  let reader: PythonReader = { stateDict, archive in
    guard
      let in_layers_0_weight = stateDict[
        "\(prefix.0).0.in_layers.0.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_0_bias = stateDict[
        "\(prefix.0).0.in_layers.0.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerNorm.weight.copy(from: in_layers_0_weight, zip: archive, of: FloatType.self)
    try inLayerNorm.bias.copy(from: in_layers_0_bias, zip: archive, of: FloatType.self)
    guard
      let in_layers_2_weight = stateDict[
        "\(prefix.0).0.in_layers.2.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_2_bias = stateDict[
        "\(prefix.0).0.in_layers.2.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerConv2d.weight.copy(from: in_layers_2_weight, zip: archive, of: FloatType.self)
    try inLayerConv2d.bias.copy(from: in_layers_2_bias, zip: archive, of: FloatType.self)
    guard
      let emb_layers_1_weight = stateDict[
        "\(prefix.0).0.emb_layers.1.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let emb_layers_1_bias = stateDict[
        "\(prefix.0).0.emb_layers.1.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try embLayer.weight.copy(from: emb_layers_1_weight, zip: archive, of: FloatType.self)
    try embLayer.bias.copy(from: emb_layers_1_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_0_weight = stateDict[
        "\(prefix.0).0.out_layers.0.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_0_bias = stateDict[
        "\(prefix.0).0.out_layers.0.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerNorm.weight.copy(from: out_layers_0_weight, zip: archive, of: FloatType.self)
    try outLayerNorm.bias.copy(from: out_layers_0_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_3_weight = stateDict[
        "\(prefix.0).0.out_layers.3.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_3_bias = stateDict[
        "\(prefix.0).0.out_layers.3.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerConv2d.weight.copy(from: out_layers_3_weight, zip: archive, of: FloatType.self)
    try outLayerConv2d.bias.copy(from: out_layers_3_bias, zip: archive, of: FloatType.self)
    if let skipModel = skipModel {
      guard
        let skip_connection_weight = stateDict[
          "\(prefix.0).0.skip_connection.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let skip_connection_bias = stateDict[
          "\(prefix.0).0.skip_connection.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try skipModel.weight.copy(from: skip_connection_weight, zip: archive, of: FloatType.self)
      try skipModel.bias.copy(from: skip_connection_bias, zip: archive, of: FloatType.self)
    }
    if let transformerReader = transformerReader {
      try transformerReader(stateDict, archive)
    }
    if let timeInLayerNorm = timeInLayerNorm, let timeInLayerConv2d = timeInLayerConv2d,
      let timeEmbLayer = timeEmbLayer, let timeOutLayerNorm = timeOutLayerNorm,
      let timeOutLayerConv2d = timeOutLayerConv2d, let mixFactor = mixFactor
    {
      guard
        let time_stack_in_layers_0_weight = stateDict[
          "\(prefix.0).0.time_stack.in_layers.0.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_in_layers_0_bias = stateDict[
          "\(prefix.0).0.time_stack.in_layers.0.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeInLayerNorm.weight.copy(
        from: time_stack_in_layers_0_weight, zip: archive, of: FloatType.self)
      try timeInLayerNorm.bias.copy(
        from: time_stack_in_layers_0_bias, zip: archive, of: FloatType.self)
      guard
        let time_stack_in_layers_2_weight = stateDict[
          "\(prefix.0).0.time_stack.in_layers.2.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_in_layers_2_bias = stateDict[
          "\(prefix.0).0.time_stack.in_layers.2.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeInLayerConv2d.weight.copy(
        from: time_stack_in_layers_2_weight, zip: archive, of: FloatType.self)
      try timeInLayerConv2d.bias.copy(
        from: time_stack_in_layers_2_bias, zip: archive, of: FloatType.self)
      guard
        let time_stack_emb_layers_1_weight = stateDict[
          "\(prefix.0).0.time_stack.emb_layers.1.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_emb_layers_1_bias = stateDict[
          "\(prefix.0).0.time_stack.emb_layers.1.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeEmbLayer.weight.copy(
        from: time_stack_emb_layers_1_weight, zip: archive, of: FloatType.self)
      try timeEmbLayer.bias.copy(
        from: time_stack_emb_layers_1_bias, zip: archive, of: FloatType.self)
      guard
        let time_stack_out_layers_0_weight = stateDict[
          "\(prefix.0).0.time_stack.out_layers.0.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_out_layers_0_bias = stateDict[
          "\(prefix.0).0.time_stack.out_layers.0.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeOutLayerNorm.weight.copy(
        from: time_stack_out_layers_0_weight, zip: archive, of: FloatType.self)
      try timeOutLayerNorm.bias.copy(
        from: time_stack_out_layers_0_bias, zip: archive, of: FloatType.self)
      guard
        let time_stack_out_layers_3_weight = stateDict[
          "\(prefix.0).0.time_stack.out_layers.3.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_out_layers_3_bias = stateDict[
          "\(prefix.0).0.time_stack.out_layers.3.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeOutLayerConv2d.weight.copy(
        from: time_stack_out_layers_3_weight, zip: archive, of: FloatType.self)
      try timeOutLayerConv2d.bias.copy(
        from: time_stack_out_layers_3_bias, zip: archive, of: FloatType.self)
      guard let mix_factor = stateDict["\(prefix.0).0.time_mixer.mix_factor"] else {
        throw UnpickleError.tensorNotFound
      }
      // Apply sigmod when loading the mix factor.
      try archive.with(mix_factor) {
        var tensor = Tensor<Float>(from: $0).toCPU()
        // Sigmoid.
        tensor[0] = 1.0 / (1.0 + expf(-tensor[0]))
        mixFactor.weight.copy(from: Tensor<FloatType>(from: tensor))
      }
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    if let transformerMapper = transformerMapper {
      mapping.merge(transformerMapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).0.in_layers.0.weight"] = [inLayerNorm.weight.name]
      mapping["\(prefix.0).0.in_layers.0.bias"] = [inLayerNorm.bias.name]
      mapping["\(prefix.0).0.in_layers.2.weight"] = [inLayerConv2d.weight.name]
      mapping["\(prefix.0).0.in_layers.2.bias"] = [inLayerConv2d.bias.name]
      mapping["\(prefix.0).0.emb_layers.1.weight"] = [embLayer.weight.name]
      mapping["\(prefix.0).0.emb_layers.1.bias"] = [embLayer.bias.name]
      mapping["\(prefix.0).0.out_layers.0.weight"] = [outLayerNorm.weight.name]
      mapping["\(prefix.0).0.out_layers.0.bias"] = [outLayerNorm.bias.name]
      mapping["\(prefix.0).0.out_layers.3.weight"] = [outLayerConv2d.weight.name]
      mapping["\(prefix.0).0.out_layers.3.bias"] = [outLayerConv2d.bias.name]
      if let skipModel = skipModel {
        mapping["\(prefix.0).0.skip_connection.weight"] = [skipModel.weight.name]
        mapping["\(prefix.0).0.skip_connection.bias"] = [skipModel.bias.name]
      }
      if let timeInLayerNorm = timeInLayerNorm, let timeInLayerConv2d = timeInLayerConv2d,
        let timeEmbLayer = timeEmbLayer, let timeOutLayerNorm = timeOutLayerNorm,
        let timeOutLayerConv2d = timeOutLayerConv2d, let mixFactor = mixFactor
      {
        mapping["\(prefix.0).0.time_stack.in_layers.0.weight"] = [timeInLayerNorm.weight.name]
        mapping["\(prefix.0).0.time_stack.in_layers.0.bias"] = [timeInLayerNorm.bias.name]
        mapping["\(prefix.0).0.time_stack.in_layers.2.weight"] = [timeInLayerConv2d.weight.name]
        mapping["\(prefix.0).0.time_stack.in_layers.2.bias"] = [timeInLayerConv2d.bias.name]
        mapping["\(prefix.0).0.time_stack.emb_layers.1.weight"] = [timeEmbLayer.weight.name]
        mapping["\(prefix.0).0.time_stack.emb_layers.1.bias"] = [timeEmbLayer.bias.name]
        mapping["\(prefix.0).0.time_stack.out_layers.0.weight"] = [timeOutLayerNorm.weight.name]
        mapping["\(prefix.0).0.time_stack.out_layers.0.bias"] = [timeOutLayerNorm.bias.name]
        mapping["\(prefix.0).0.time_stack.out_layers.3.weight"] = [timeOutLayerConv2d.weight.name]
        mapping["\(prefix.0).0.time_stack.out_layers.3.bias"] = [timeOutLayerConv2d.bias.name]
        mapping["\(prefix.0).0.time_mixer.mix_factor"] = [mixFactor.weight.name]
      }
    case .diffusers:
      if let timeInLayerNorm = timeInLayerNorm, let timeInLayerConv2d = timeInLayerConv2d,
        let timeEmbLayer = timeEmbLayer, let timeOutLayerNorm = timeOutLayerNorm,
        let timeOutLayerConv2d = timeOutLayerConv2d, let mixFactor = mixFactor
      {
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.norm1.weight"] = [
          inLayerNorm.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.norm1.bias"] = [
          inLayerNorm.bias.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.conv1.weight"] = [
          inLayerConv2d.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.conv1.bias"] = [
          inLayerConv2d.bias.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.time_emb_proj.weight"] = [
          embLayer.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.time_emb_proj.bias"] = [
          embLayer.bias.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.norm2.weight"] = [
          outLayerNorm.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.norm2.bias"] = [
          outLayerNorm.bias.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.conv2.weight"] = [
          outLayerConv2d.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.conv2.bias"] = [
          outLayerConv2d.bias.name
        ]
        if let skipModel = skipModel {
          mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.conv_shortcut.weight"] = [
            skipModel.weight.name
          ]
          mapping["\(prefix.1).resnets.\(repeatStart).spatial_res_block.conv_shortcut.bias"] = [
            skipModel.bias.name
          ]
        }
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.norm1.weight"] = [
          timeInLayerNorm.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.norm1.bias"] = [
          timeInLayerNorm.bias.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.conv1.weight"] = [
          timeInLayerConv2d.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.conv1.bias"] = [
          timeInLayerConv2d.bias.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.time_emb_proj.weight"] = [
          timeEmbLayer.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.time_emb_proj.bias"] = [
          timeEmbLayer.bias.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.norm2.weight"] = [
          timeOutLayerNorm.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.norm2.bias"] = [
          timeOutLayerNorm.bias.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.conv2.weight"] = [
          timeOutLayerConv2d.weight.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).temporal_res_block.conv2.bias"] = [
          timeOutLayerConv2d.bias.name
        ]
        mapping["\(prefix.1).resnets.\(repeatStart).time_mixer.mix_factor"] = [
          mixFactor.weight.name
        ]
      } else {
        mapping["\(prefix.1).resnets.\(repeatStart).norm1.weight"] = [inLayerNorm.weight.name]
        mapping["\(prefix.1).resnets.\(repeatStart).norm1.bias"] = [inLayerNorm.bias.name]
        mapping["\(prefix.1).resnets.\(repeatStart).conv1.weight"] = [inLayerConv2d.weight.name]
        mapping["\(prefix.1).resnets.\(repeatStart).conv1.bias"] = [inLayerConv2d.bias.name]
        mapping["\(prefix.1).resnets.\(repeatStart).time_emb_proj.weight"] = [embLayer.weight.name]
        mapping["\(prefix.1).resnets.\(repeatStart).time_emb_proj.bias"] = [embLayer.bias.name]
        mapping["\(prefix.1).resnets.\(repeatStart).norm2.weight"] = [outLayerNorm.weight.name]
        mapping["\(prefix.1).resnets.\(repeatStart).norm2.bias"] = [outLayerNorm.bias.name]
        mapping["\(prefix.1).resnets.\(repeatStart).conv2.weight"] = [outLayerConv2d.weight.name]
        mapping["\(prefix.1).resnets.\(repeatStart).conv2.bias"] = [outLayerConv2d.bias.name]
        if let skipModel = skipModel {
          mapping["\(prefix.1).resnets.\(repeatStart).conv_shortcut.weight"] = [
            skipModel.weight.name
          ]
          mapping["\(prefix.1).resnets.\(repeatStart).conv_shortcut.bias"] = [skipModel.bias.name]
        }
      }
    }
    return mapping
  }
  return (reader, mapper, Model([x, emb] + kvs, [out]))
}

func MiddleBlock<FloatType: TensorNumeric & BinaryFloatingPoint>(
  prefix: String,
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: (Int, Int), attentionBlock: Int,
  retainingNormProjInProjOutAndSecondResNetWhenNoAttentionBlocks: Bool,
  injectIPAdapterLengths: [Int], upcastAttention: Bool, usesFlashAttention: FlashAttentionLevel,
  isTemporalMixEnabled: Bool,
  x: Model.IO, emb: Model.IO, of: FloatType.Type = FloatType.self
) -> (PythonReader, ModelWeightMapper, Model.IO, [Input]) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (inLayerNorm1, inLayerConv2d1, embLayer1, outLayerNorm1, outLayerConv2d1, _, resBlock1) =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false, flags: [])
  var out = resBlock1(x, emb)
  let timeInLayerNorm1: Model?
  let timeInLayerConv2d1: Model?
  let timeEmbLayer1: Model?
  let timeOutLayerNorm1: Model?
  let timeOutLayerConv2d1: Model?
  let mixFactor1: Model?
  if isTemporalMixEnabled {
    let timeResBlock1: Model
    (
      timeInLayerNorm1, timeInLayerConv2d1, timeEmbLayer1, timeOutLayerNorm1, timeOutLayerConv2d1,
      timeResBlock1
    ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels, flags: [])
    let mix = Parameter<FloatType>(.GPU(0), format: .NHWC, shape: [1], name: "time_mixer")
    out = mix .* out + (1 - mix) .* timeResBlock1(out, emb)
    mixFactor1 = mix
  } else {
    timeInLayerNorm1 = nil
    timeInLayerConv2d1 = nil
    timeEmbLayer1 = nil
    timeOutLayerNorm1 = nil
    timeOutLayerConv2d1 = nil
    mixFactor1 = nil
  }
  var kvs: [Input]
  if embeddingLength.0 == 1 && embeddingLength.1 == 1 {
    kvs = (0..<(attentionBlock * (injectIPAdapterLengths.count + 1))).map { _ in Input() }
  } else {
    kvs = (0..<(attentionBlock * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input() }
  }
  let transformerReader: PythonReader?
  let transformerMapper: ModelWeightMapper?
  let inLayerNorm2: Model?
  let inLayerConv2d2: Model?
  let embLayer2: Model?
  let outLayerNorm2: Model?
  let outLayerConv2d2: Model?
  let timeInLayerNorm2: Model?
  let timeInLayerConv2d2: Model?
  let timeEmbLayer2: Model?
  let timeOutLayerNorm2: Model?
  let timeOutLayerConv2d2: Model?
  let mixFactor2: Model?
  if attentionBlock > 0 {
    if isTemporalMixEnabled {
      if embeddingLength.0 == 1 && embeddingLength.1 == 1 {
        kvs.append(contentsOf: (0..<(attentionBlock + 1)).map { _ in Input() })
      } else {
        kvs.append(contentsOf: (0..<(attentionBlock * 2 + 1)).map { _ in Input() })
      }
    }
    let (reader, mapper, transformer) = SpatialTransformer(
      prefix: ("\(prefix).middle_block.1", "mid_block.attentions.0"), ch: channels, k: k,
      h: numHeads, b: batchSize, height: height, width: width, depth: attentionBlock,
      t: embeddingLength, intermediateSize: channels * 4,
      injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention,
      usesFlashAttention: usesFlashAttention, flags: [], isTemporalMixEnabled: isTemporalMixEnabled,
      of: FloatType.self)
    transformerReader = reader
    transformerMapper = mapper
    out = transformer([out] + kvs)
    let resBlock2: Model
    (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
      ResBlock(b: batchSize, outChannels: channels, skipConnection: false, flags: [])
    out = resBlock2(out, emb)
    if isTemporalMixEnabled {
      let timeResBlock2: Model
      (
        timeInLayerNorm2, timeInLayerConv2d2, timeEmbLayer2, timeOutLayerNorm2, timeOutLayerConv2d2,
        timeResBlock2
      ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels, flags: [])
      let mix = Parameter<FloatType>(.GPU(0), format: .NHWC, shape: [1], name: "time_mixer")
      out = mix .* out + (1 - mix) .* timeResBlock2(out, emb)
      mixFactor2 = mix
    } else {
      timeInLayerNorm2 = nil
      timeInLayerConv2d2 = nil
      timeEmbLayer2 = nil
      timeOutLayerNorm2 = nil
      timeOutLayerConv2d2 = nil
      mixFactor2 = nil
    }
  } else {
    if retainingNormProjInProjOutAndSecondResNetWhenNoAttentionBlocks {
      let (reader, mapper, transformer) = SpatialTransformer(
        prefix: ("\(prefix).middle_block.1", "mid_block.attentions.0"), ch: channels, k: k,
        h: numHeads, b: batchSize, height: height, width: width, depth: 0,  // Zero depth so we still apply norm, proj_in, proj_out.
        t: embeddingLength, intermediateSize: channels * 4,
        injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention,
        usesFlashAttention: usesFlashAttention, flags: [],
        isTemporalMixEnabled: isTemporalMixEnabled, of: FloatType.self)
      transformerReader = reader
      transformerMapper = mapper
      out = transformer(out)
      let resBlock2: Model
      (inLayerNorm2, inLayerConv2d2, embLayer2, outLayerNorm2, outLayerConv2d2, _, resBlock2) =
        ResBlock(b: batchSize, outChannels: channels, skipConnection: false, flags: [])
      out = resBlock2(out, emb)
      if isTemporalMixEnabled {
        let timeResBlock2: Model
        (
          timeInLayerNorm2, timeInLayerConv2d2, timeEmbLayer2, timeOutLayerNorm2,
          timeOutLayerConv2d2, timeResBlock2
        ) = TimeResBlock(b: batchSize, h: height, w: width, channels: channels, flags: [])
        let mix = Parameter<FloatType>(
          .GPU(0), format: .NHWC, shape: [1], name: "time_mixer")
        out = mix .* out + (1 - mix) .* timeResBlock2(out, emb)
        mixFactor2 = mix
      } else {
        timeInLayerNorm2 = nil
        timeInLayerConv2d2 = nil
        timeEmbLayer2 = nil
        timeOutLayerNorm2 = nil
        timeOutLayerConv2d2 = nil
        mixFactor2 = nil
      }
    } else {
      transformerReader = nil
      transformerMapper = nil
      inLayerNorm2 = nil
      inLayerConv2d2 = nil
      embLayer2 = nil
      outLayerNorm2 = nil
      outLayerConv2d2 = nil
      timeInLayerNorm2 = nil
      timeInLayerConv2d2 = nil
      timeEmbLayer2 = nil
      timeOutLayerNorm2 = nil
      timeOutLayerConv2d2 = nil
      mixFactor2 = nil
    }
  }
  let reader: PythonReader = { stateDict, archive in
    guard
      let in_layers_0_0_weight = stateDict[
        "\(prefix).middle_block.0.in_layers.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_0_0_bias = stateDict["\(prefix).middle_block.0.in_layers.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerNorm1.weight.copy(from: in_layers_0_0_weight, zip: archive, of: FloatType.self)
    try inLayerNorm1.bias.copy(from: in_layers_0_0_bias, zip: archive, of: FloatType.self)
    guard
      let in_layers_0_2_weight = stateDict[
        "\(prefix).middle_block.0.in_layers.2.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let in_layers_0_2_bias = stateDict["\(prefix).middle_block.0.in_layers.2.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try inLayerConv2d1.weight.copy(from: in_layers_0_2_weight, zip: archive, of: FloatType.self)
    try inLayerConv2d1.bias.copy(from: in_layers_0_2_bias, zip: archive, of: FloatType.self)
    guard
      let emb_layers_0_1_weight = stateDict[
        "\(prefix).middle_block.0.emb_layers.1.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let emb_layers_0_1_bias = stateDict["\(prefix).middle_block.0.emb_layers.1.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try embLayer1.weight.copy(from: emb_layers_0_1_weight, zip: archive, of: FloatType.self)
    try embLayer1.bias.copy(from: emb_layers_0_1_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_0_0_weight = stateDict[
        "\(prefix).middle_block.0.out_layers.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_0_0_bias = stateDict[
        "\(prefix).middle_block.0.out_layers.0.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerNorm1.weight.copy(from: out_layers_0_0_weight, zip: archive, of: FloatType.self)
    try outLayerNorm1.bias.copy(from: out_layers_0_0_bias, zip: archive, of: FloatType.self)
    guard
      let out_layers_0_3_weight = stateDict[
        "\(prefix).middle_block.0.out_layers.3.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let out_layers_0_3_bias = stateDict["\(prefix).middle_block.0.out_layers.3.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try outLayerConv2d1.weight.copy(from: out_layers_0_3_weight, zip: archive, of: FloatType.self)
    try outLayerConv2d1.bias.copy(from: out_layers_0_3_bias, zip: archive, of: FloatType.self)
    if let timeInLayerNorm1 = timeInLayerNorm1, let timeInLayerConv2d1 = timeInLayerConv2d1,
      let timeEmbLayer1 = timeEmbLayer1, let timeOutLayerNorm1 = timeOutLayerNorm1,
      let timeOutLayerConv2d1 = timeOutLayerConv2d1, let mixFactor1 = mixFactor1
    {
      guard
        let time_stack_in_layers_0_0_weight = stateDict[
          "\(prefix).middle_block.0.time_stack.in_layers.0.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_in_layers_0_0_bias = stateDict[
          "\(prefix).middle_block.0.time_stack.in_layers.0.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeInLayerNorm1.weight.copy(
        from: time_stack_in_layers_0_0_weight, zip: archive, of: FloatType.self)
      try timeInLayerNorm1.bias.copy(
        from: time_stack_in_layers_0_0_bias, zip: archive, of: FloatType.self)
      guard
        let time_stack_in_layers_0_2_weight = stateDict[
          "\(prefix).middle_block.0.time_stack.in_layers.2.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_in_layers_0_2_bias = stateDict[
          "\(prefix).middle_block.0.time_stack.in_layers.2.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeInLayerConv2d1.weight.copy(
        from: time_stack_in_layers_0_2_weight, zip: archive, of: FloatType.self)
      try timeInLayerConv2d1.bias.copy(
        from: time_stack_in_layers_0_2_bias, zip: archive, of: FloatType.self)
      guard
        let time_stack_emb_layers_0_1_weight = stateDict[
          "\(prefix).middle_block.0.time_stack.emb_layers.1.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_emb_layers_0_1_bias = stateDict[
          "\(prefix).middle_block.0.time_stack.emb_layers.1.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeEmbLayer1.weight.copy(
        from: time_stack_emb_layers_0_1_weight, zip: archive, of: FloatType.self)
      try timeEmbLayer1.bias.copy(
        from: time_stack_emb_layers_0_1_bias, zip: archive, of: FloatType.self)
      guard
        let time_stack_out_layers_0_0_weight = stateDict[
          "\(prefix).middle_block.0.time_stack.out_layers.0.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_out_layers_0_0_bias = stateDict[
          "\(prefix).middle_block.0.time_stack.out_layers.0.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeOutLayerNorm1.weight.copy(
        from: time_stack_out_layers_0_0_weight, zip: archive, of: FloatType.self)
      try timeOutLayerNorm1.bias.copy(
        from: time_stack_out_layers_0_0_bias, zip: archive, of: FloatType.self)
      guard
        let time_stack_out_layers_0_3_weight = stateDict[
          "\(prefix).middle_block.0.time_stack.out_layers.3.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let time_stack_out_layers_0_3_bias = stateDict[
          "\(prefix).middle_block.0.time_stack.out_layers.3.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try timeOutLayerConv2d1.weight.copy(
        from: time_stack_out_layers_0_3_weight, zip: archive, of: FloatType.self)
      try timeOutLayerConv2d1.bias.copy(
        from: time_stack_out_layers_0_3_bias, zip: archive, of: FloatType.self)
      guard let mix_factor = stateDict["\(prefix).middle_block.0.time_mixer.mix_factor"] else {
        throw UnpickleError.tensorNotFound
      }
      // Apply sigmod when loading the mix factor.
      try archive.with(mix_factor) {
        var tensor = Tensor<Float>(from: $0).toCPU()
        // Sigmoid.
        tensor[0] = 1.0 / (1.0 + expf(-tensor[0]))
        mixFactor1.weight.copy(from: Tensor<FloatType>(from: tensor))
      }
    }
    try transformerReader?(stateDict, archive)
    if let inLayerNorm2 = inLayerNorm2, let inLayerConv2d2 = inLayerConv2d2,
      let embLayer2 = embLayer2, let outLayerNorm2 = outLayerNorm2,
      let outLayerConv2d2 = outLayerConv2d2
    {
      guard
        let in_layers_2_0_weight = stateDict[
          "\(prefix).middle_block.2.in_layers.0.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let in_layers_2_0_bias = stateDict["\(prefix).middle_block.2.in_layers.0.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try inLayerNorm2.weight.copy(from: in_layers_2_0_weight, zip: archive, of: FloatType.self)
      try inLayerNorm2.bias.copy(from: in_layers_2_0_bias, zip: archive, of: FloatType.self)
      guard
        let in_layers_2_2_weight = stateDict[
          "\(prefix).middle_block.2.in_layers.2.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let in_layers_2_2_bias = stateDict["\(prefix).middle_block.2.in_layers.2.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try inLayerConv2d2.weight.copy(from: in_layers_2_2_weight, zip: archive, of: FloatType.self)
      try inLayerConv2d2.bias.copy(from: in_layers_2_2_bias, zip: archive, of: FloatType.self)
      guard
        let emb_layers_2_1_weight = stateDict[
          "\(prefix).middle_block.2.emb_layers.1.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let emb_layers_2_1_bias = stateDict["\(prefix).middle_block.2.emb_layers.1.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try embLayer2.weight.copy(from: emb_layers_2_1_weight, zip: archive, of: FloatType.self)
      try embLayer2.bias.copy(from: emb_layers_2_1_bias, zip: archive, of: FloatType.self)
      guard
        let out_layers_2_0_weight = stateDict[
          "\(prefix).middle_block.2.out_layers.0.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let out_layers_2_0_bias = stateDict[
          "\(prefix).middle_block.2.out_layers.0.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try outLayerNorm2.weight.copy(from: out_layers_2_0_weight, zip: archive, of: FloatType.self)
      try outLayerNorm2.bias.copy(from: out_layers_2_0_bias, zip: archive, of: FloatType.self)
      guard
        let out_layers_2_3_weight = stateDict[
          "\(prefix).middle_block.2.out_layers.3.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let out_layers_2_3_bias = stateDict["\(prefix).middle_block.2.out_layers.3.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try outLayerConv2d2.weight.copy(from: out_layers_2_3_weight, zip: archive, of: FloatType.self)
      try outLayerConv2d2.bias.copy(from: out_layers_2_3_bias, zip: archive, of: FloatType.self)
      if let timeInLayerNorm2 = timeInLayerNorm2, let timeInLayerConv2d2 = timeInLayerConv2d2,
        let timeEmbLayer2 = timeEmbLayer2, let timeOutLayerNorm2 = timeOutLayerNorm2,
        let timeOutLayerConv2d2 = timeOutLayerConv2d2, let mixFactor2 = mixFactor2
      {
        guard
          let time_stack_in_layers_2_0_weight = stateDict[
            "\(prefix).middle_block.2.time_stack.in_layers.0.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let time_stack_in_layers_2_0_bias = stateDict[
            "\(prefix).middle_block.2.time_stack.in_layers.0.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try timeInLayerNorm2.weight.copy(
          from: time_stack_in_layers_2_0_weight, zip: archive, of: FloatType.self)
        try timeInLayerNorm2.bias.copy(
          from: time_stack_in_layers_2_0_bias, zip: archive, of: FloatType.self)
        guard
          let time_stack_in_layers_2_2_weight = stateDict[
            "\(prefix).middle_block.2.time_stack.in_layers.2.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let time_stack_in_layers_2_2_bias = stateDict[
            "\(prefix).middle_block.2.time_stack.in_layers.2.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try timeInLayerConv2d2.weight.copy(
          from: time_stack_in_layers_2_2_weight, zip: archive, of: FloatType.self)
        try timeInLayerConv2d2.bias.copy(
          from: time_stack_in_layers_2_2_bias, zip: archive, of: FloatType.self)
        guard
          let time_stack_emb_layers_2_1_weight = stateDict[
            "\(prefix).middle_block.2.time_stack.emb_layers.1.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let time_stack_emb_layers_2_1_bias = stateDict[
            "\(prefix).middle_block.2.time_stack.emb_layers.1.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try timeEmbLayer2.weight.copy(
          from: time_stack_emb_layers_2_1_weight, zip: archive, of: FloatType.self)
        try timeEmbLayer2.bias.copy(
          from: time_stack_emb_layers_2_1_bias, zip: archive, of: FloatType.self)
        guard
          let time_stack_out_layers_2_0_weight = stateDict[
            "\(prefix).middle_block.2.time_stack.out_layers.0.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let time_stack_out_layers_2_0_bias = stateDict[
            "\(prefix).middle_block.2.time_stack.out_layers.0.bias"
          ]
        else {
          throw UnpickleError.tensorNotFound
        }
        try timeOutLayerNorm2.weight.copy(
          from: time_stack_out_layers_2_0_weight, zip: archive, of: FloatType.self)
        try timeOutLayerNorm2.bias.copy(
          from: time_stack_out_layers_2_0_bias, zip: archive, of: FloatType.self)
        guard
          let time_stack_out_layers_2_3_weight = stateDict[
            "\(prefix).middle_block.2.time_stack.out_layers.3.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let time_stack_out_layers_2_3_bias = stateDict[
            "\(prefix).middle_block.2.time_stack.out_layers.3.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try timeOutLayerConv2d2.weight.copy(
          from: time_stack_out_layers_2_3_weight, zip: archive, of: FloatType.self)
        try timeOutLayerConv2d2.bias.copy(
          from: time_stack_out_layers_2_3_bias, zip: archive, of: FloatType.self)
        guard let mix_factor = stateDict["\(prefix).middle_block.2.time_mixer.mix_factor"] else {
          throw UnpickleError.tensorNotFound
        }
        // Apply sigmod when loading the mix factor.
        try archive.with(mix_factor) {
          var tensor = Tensor<Float>(from: $0).toCPU()
          // Sigmoid.
          tensor[0] = 1.0 / (1.0 + expf(-tensor[0]))
          mixFactor2.weight.copy(from: Tensor<FloatType>(from: tensor))
        }
      }
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    if let transformerMapper = transformerMapper {
      mapping.merge(transformerMapper(format)) { v, _ in v }
    }
    switch format {
    case .generativeModels:
      mapping["\(prefix).middle_block.0.in_layers.0.weight"] = [inLayerNorm1.weight.name]
      mapping["\(prefix).middle_block.0.in_layers.0.bias"] = [inLayerNorm1.bias.name]
      mapping["\(prefix).middle_block.0.in_layers.2.weight"] = [inLayerConv2d1.weight.name]
      mapping["\(prefix).middle_block.0.in_layers.2.bias"] = [inLayerConv2d1.bias.name]
      mapping["\(prefix).middle_block.0.emb_layers.1.weight"] = [embLayer1.weight.name]
      mapping["\(prefix).middle_block.0.emb_layers.1.bias"] = [embLayer1.bias.name]
      mapping["\(prefix).middle_block.0.out_layers.0.weight"] = [outLayerNorm1.weight.name]
      mapping["\(prefix).middle_block.0.out_layers.0.bias"] = [outLayerNorm1.bias.name]
      mapping["\(prefix).middle_block.0.out_layers.3.weight"] = [outLayerConv2d1.weight.name]
      mapping["\(prefix).middle_block.0.out_layers.3.bias"] = [outLayerConv2d1.bias.name]
      if let timeInLayerNorm1 = timeInLayerNorm1, let timeInLayerConv2d1 = timeInLayerConv2d1,
        let timeEmbLayer1 = timeEmbLayer1, let timeOutLayerNorm1 = timeOutLayerNorm1,
        let timeOutLayerConv2d1 = timeOutLayerConv2d1, let mixFactor1 = mixFactor1
      {
        mapping["\(prefix).middle_block.0.time_stack.in_layers.0.weight"] = [
          timeInLayerNorm1.weight.name
        ]
        mapping["\(prefix).middle_block.0.time_stack.in_layers.0.bias"] = [
          timeInLayerNorm1.bias.name
        ]
        mapping["\(prefix).middle_block.0.time_stack.in_layers.2.weight"] = [
          timeInLayerConv2d1.weight.name
        ]
        mapping["\(prefix).middle_block.0.time_stack.in_layers.2.bias"] = [
          timeInLayerConv2d1.bias.name
        ]
        mapping["\(prefix).middle_block.0.time_stack.emb_layers.1.weight"] = [
          timeEmbLayer1.weight.name
        ]
        mapping["\(prefix).middle_block.0.time_stack.emb_layers.1.bias"] = [timeEmbLayer1.bias.name]
        mapping["\(prefix).middle_block.0.time_stack.out_layers.0.weight"] = [
          timeOutLayerNorm1.weight.name
        ]
        mapping["\(prefix).middle_block.0.time_stack.out_layers.0.bias"] = [
          timeOutLayerNorm1.bias.name
        ]
        mapping["\(prefix).middle_block.0.time_stack.out_layers.3.weight"] = [
          timeOutLayerConv2d1.weight.name
        ]
        mapping["\(prefix).middle_block.0.time_stack.out_layers.3.bias"] = [
          timeOutLayerConv2d1.bias.name
        ]
        mapping["\(prefix).middle_block.0.time_mixer.mix_factor"] = [mixFactor1.weight.name]
      }
      if let inLayerNorm2 = inLayerNorm2, let inLayerConv2d2 = inLayerConv2d2,
        let embLayer2 = embLayer2, let outLayerNorm2 = outLayerNorm2,
        let outLayerConv2d2 = outLayerConv2d2
      {
        mapping["\(prefix).middle_block.2.in_layers.0.weight"] = [inLayerNorm2.weight.name]
        mapping["\(prefix).middle_block.2.in_layers.0.bias"] = [inLayerNorm2.bias.name]
        mapping["\(prefix).middle_block.2.in_layers.2.weight"] = [inLayerConv2d2.weight.name]
        mapping["\(prefix).middle_block.2.in_layers.2.bias"] = [inLayerConv2d2.bias.name]
        mapping["\(prefix).middle_block.2.emb_layers.1.weight"] = [embLayer2.weight.name]
        mapping["\(prefix).middle_block.2.emb_layers.1.bias"] = [embLayer2.bias.name]
        mapping["\(prefix).middle_block.2.out_layers.0.weight"] = [outLayerNorm2.weight.name]
        mapping["\(prefix).middle_block.2.out_layers.0.bias"] = [outLayerNorm2.bias.name]
        mapping["\(prefix).middle_block.2.out_layers.3.weight"] = [outLayerConv2d2.weight.name]
        mapping["\(prefix).middle_block.2.out_layers.3.bias"] = [outLayerConv2d2.bias.name]
        if let timeInLayerNorm2 = timeInLayerNorm2, let timeInLayerConv2d2 = timeInLayerConv2d2,
          let timeEmbLayer2 = timeEmbLayer2, let timeOutLayerNorm2 = timeOutLayerNorm2,
          let timeOutLayerConv2d2 = timeOutLayerConv2d2, let mixFactor2 = mixFactor2
        {
          mapping["\(prefix).middle_block.2.time_stack.in_layers.0.weight"] = [
            timeInLayerNorm2.weight.name
          ]
          mapping["\(prefix).middle_block.2.time_stack.in_layers.0.bias"] = [
            timeInLayerNorm2.bias.name
          ]
          mapping["\(prefix).middle_block.2.time_stack.in_layers.2.weight"] = [
            timeInLayerConv2d2.weight.name
          ]
          mapping["\(prefix).middle_block.2.time_stack.in_layers.2.bias"] = [
            timeInLayerConv2d2.bias.name
          ]
          mapping["\(prefix).middle_block.2.time_stack.emb_layers.1.weight"] = [
            timeEmbLayer2.weight.name
          ]
          mapping["\(prefix).middle_block.2.time_stack.emb_layers.1.bias"] = [
            timeEmbLayer2.bias.name
          ]
          mapping["\(prefix).middle_block.2.time_stack.out_layers.0.weight"] = [
            timeOutLayerNorm2.weight.name
          ]
          mapping["\(prefix).middle_block.2.time_stack.out_layers.0.bias"] = [
            timeOutLayerNorm2.bias.name
          ]
          mapping["\(prefix).middle_block.2.time_stack.out_layers.3.weight"] = [
            timeOutLayerConv2d2.weight.name
          ]
          mapping["\(prefix).middle_block.2.time_stack.out_layers.3.bias"] = [
            timeOutLayerConv2d2.bias.name
          ]
          mapping["\(prefix).middle_block.2.time_mixer.mix_factor"] = [mixFactor2.weight.name]
        }
      }
    case .diffusers:
      if let timeInLayerNorm1 = timeInLayerNorm1, let timeInLayerConv2d1 = timeInLayerConv2d1,
        let timeEmbLayer1 = timeEmbLayer1, let timeOutLayerNorm1 = timeOutLayerNorm1,
        let timeOutLayerConv2d1 = timeOutLayerConv2d1, let mixFactor1 = mixFactor1
      {
        mapping["mid_block.resnets.0.spatial_res_block.norm1.weight"] = [inLayerNorm1.weight.name]
        mapping["mid_block.resnets.0.spatial_res_block.norm1.bias"] = [inLayerNorm1.bias.name]
        mapping["mid_block.resnets.0.spatial_res_block.conv1.weight"] = [inLayerConv2d1.weight.name]
        mapping["mid_block.resnets.0.spatial_res_block.conv1.bias"] = [inLayerConv2d1.bias.name]
        mapping["mid_block.resnets.0.spatial_res_block.time_emb_proj.weight"] = [
          embLayer1.weight.name
        ]
        mapping["mid_block.resnets.0.spatial_res_block.time_emb_proj.bias"] = [embLayer1.bias.name]
        mapping["mid_block.resnets.0.spatial_res_block.norm2.weight"] = [outLayerNorm1.weight.name]
        mapping["mid_block.resnets.0.spatial_res_block.norm2.bias"] = [outLayerNorm1.bias.name]
        mapping["mid_block.resnets.0.spatial_res_block.conv2.weight"] = [
          outLayerConv2d1.weight.name
        ]
        mapping["mid_block.resnets.0.spatial_res_block.conv2.bias"] = [outLayerConv2d1.bias.name]
        mapping["mid_block.resnets.0.temporal_res_block.norm1.weight"] = [
          timeInLayerNorm1.weight.name
        ]
        mapping["mid_block.resnets.0.temporal_res_block.norm1.bias"] = [timeInLayerNorm1.bias.name]
        mapping["mid_block.resnets.0.temporal_res_block.conv1.weight"] = [
          timeInLayerConv2d1.weight.name
        ]
        mapping["mid_block.resnets.0.temporal_res_block.conv1.bias"] = [
          timeInLayerConv2d1.bias.name
        ]
        mapping["mid_block.resnets.0.temporal_res_block.time_emb_proj.weight"] = [
          timeEmbLayer1.weight.name
        ]
        mapping["mid_block.resnets.0.temporal_res_block.time_emb_proj.bias"] = [
          timeEmbLayer1.bias.name
        ]
        mapping["mid_block.resnets.0.temporal_res_block.norm2.weight"] = [
          timeOutLayerNorm1.weight.name
        ]
        mapping["mid_block.resnets.0.temporal_res_block.norm2.bias"] = [timeOutLayerNorm1.bias.name]
        mapping["mid_block.resnets.0.temporal_res_block.conv2.weight"] = [
          timeOutLayerConv2d1.weight.name
        ]
        mapping["mid_block.resnets.0.temporal_res_block.conv2.bias"] = [
          timeOutLayerConv2d1.bias.name
        ]
        mapping["mid_block.resnets.0.time_mixer.mix_factor"] = [mixFactor1.weight.name]
      } else {
        mapping["mid_block.resnets.0.norm1.weight"] = [inLayerNorm1.weight.name]
        mapping["mid_block.resnets.0.norm1.bias"] = [inLayerNorm1.bias.name]
        mapping["mid_block.resnets.0.conv1.weight"] = [inLayerConv2d1.weight.name]
        mapping["mid_block.resnets.0.conv1.bias"] = [inLayerConv2d1.bias.name]
        mapping["mid_block.resnets.0.time_emb_proj.weight"] = [embLayer1.weight.name]
        mapping["mid_block.resnets.0.time_emb_proj.bias"] = [embLayer1.bias.name]
        mapping["mid_block.resnets.0.norm2.weight"] = [outLayerNorm1.weight.name]
        mapping["mid_block.resnets.0.norm2.bias"] = [outLayerNorm1.bias.name]
        mapping["mid_block.resnets.0.conv2.weight"] = [outLayerConv2d1.weight.name]
        mapping["mid_block.resnets.0.conv2.bias"] = [outLayerConv2d1.bias.name]
      }
      if let inLayerNorm2 = inLayerNorm2, let inLayerConv2d2 = inLayerConv2d2,
        let embLayer2 = embLayer2, let outLayerNorm2 = outLayerNorm2,
        let outLayerConv2d2 = outLayerConv2d2
      {
        mapping["mid_block.resnets.1.norm1.weight"] = [inLayerNorm2.weight.name]
        mapping["mid_block.resnets.1.norm1.bias"] = [inLayerNorm2.bias.name]
        mapping["mid_block.resnets.1.conv1.weight"] = [inLayerConv2d2.weight.name]
        mapping["mid_block.resnets.1.conv1.bias"] = [inLayerConv2d2.bias.name]
        mapping["mid_block.resnets.1.time_emb_proj.weight"] = [embLayer2.weight.name]
        mapping["mid_block.resnets.1.time_emb_proj.bias"] = [embLayer2.bias.name]
        mapping["mid_block.resnets.1.norm2.weight"] = [outLayerNorm2.weight.name]
        mapping["mid_block.resnets.1.norm2.bias"] = [outLayerNorm2.bias.name]
        mapping["mid_block.resnets.1.conv2.weight"] = [outLayerConv2d2.weight.name]
        mapping["mid_block.resnets.1.conv2.bias"] = [outLayerConv2d2.bias.name]
        if let timeInLayerNorm2 = timeInLayerNorm2, let timeInLayerConv2d2 = timeInLayerConv2d2,
          let timeEmbLayer2 = timeEmbLayer2, let timeOutLayerNorm2 = timeOutLayerNorm2,
          let timeOutLayerConv2d2 = timeOutLayerConv2d2, let mixFactor2 = mixFactor2
        {
          mapping["mid_block.resnets.1.spatial_res_block.norm1.weight"] = [inLayerNorm2.weight.name]
          mapping["mid_block.resnets.1.spatial_res_block.norm1.bias"] = [inLayerNorm2.bias.name]
          mapping["mid_block.resnets.1.spatial_res_block.conv1.weight"] = [
            inLayerConv2d2.weight.name
          ]
          mapping["mid_block.resnets.1.spatial_res_block.conv1.bias"] = [inLayerConv2d2.bias.name]
          mapping["mid_block.resnets.1.spatial_res_block.time_emb_proj.weight"] = [
            embLayer2.weight.name
          ]
          mapping["mid_block.resnets.1.spatial_res_block.time_emb_proj.bias"] = [
            embLayer2.bias.name
          ]
          mapping["mid_block.resnets.1.spatial_res_block.norm2.weight"] = [
            outLayerNorm2.weight.name
          ]
          mapping["mid_block.resnets.1.spatial_res_block.norm2.bias"] = [outLayerNorm2.bias.name]
          mapping["mid_block.resnets.1.spatial_res_block.conv2.weight"] = [
            outLayerConv2d2.weight.name
          ]
          mapping["mid_block.resnets.1.spatial_res_block.conv2.bias"] = [outLayerConv2d2.bias.name]
          mapping["mid_block.resnets.1.temporal_res_block.norm1.weight"] = [
            timeInLayerNorm2.weight.name
          ]
          mapping["mid_block.resnets.1.temporal_res_block.norm1.bias"] = [
            timeInLayerNorm2.bias.name
          ]
          mapping["mid_block.resnets.1.temporal_res_block.conv1.weight"] = [
            timeInLayerConv2d2.weight.name
          ]
          mapping["mid_block.resnets.1.temporal_res_block.conv1.bias"] = [
            timeInLayerConv2d2.bias.name
          ]
          mapping["mid_block.resnets.1.temporal_res_block.time_emb_proj.weight"] = [
            timeEmbLayer2.weight.name
          ]
          mapping["mid_block.resnets.1.temporal_res_block.time_emb_proj.bias"] = [
            timeEmbLayer2.bias.name
          ]
          mapping["mid_block.resnets.1.temporal_res_block.norm2.weight"] = [
            timeOutLayerNorm2.weight.name
          ]
          mapping["mid_block.resnets.1.temporal_res_block.norm2.bias"] = [
            timeOutLayerNorm2.bias.name
          ]
          mapping["mid_block.resnets.1.temporal_res_block.conv2.weight"] = [
            timeOutLayerConv2d2.weight.name
          ]
          mapping["mid_block.resnets.1.temporal_res_block.conv2.bias"] = [
            timeOutLayerConv2d2.bias.name
          ]
          mapping["mid_block.resnets.1.time_mixer.mix_factor"] = [mixFactor2.weight.name]
        } else {
          mapping["mid_block.resnets.1.norm1.weight"] = [inLayerNorm2.weight.name]
          mapping["mid_block.resnets.1.norm1.bias"] = [inLayerNorm2.bias.name]
          mapping["mid_block.resnets.1.conv1.weight"] = [inLayerConv2d2.weight.name]
          mapping["mid_block.resnets.1.conv1.bias"] = [inLayerConv2d2.bias.name]
          mapping["mid_block.resnets.1.time_emb_proj.weight"] = [embLayer2.weight.name]
          mapping["mid_block.resnets.1.time_emb_proj.bias"] = [embLayer2.bias.name]
          mapping["mid_block.resnets.1.norm2.weight"] = [outLayerNorm2.weight.name]
          mapping["mid_block.resnets.1.norm2.bias"] = [outLayerNorm2.bias.name]
          mapping["mid_block.resnets.1.conv2.weight"] = [outLayerConv2d2.weight.name]
          mapping["mid_block.resnets.1.conv2.bias"] = [outLayerConv2d2.bias.name]
        }
      }
    }
    return mapping
  }
  return (reader, mapper, out, kvs)
}

func InputBlocks<FloatType: TensorNumeric & BinaryFloatingPoint>(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  injectIPAdapterLengths: [Int], upcastAttention: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel,
  isTemporalMixEnabled: Bool, x: Model.IO, emb: Model.IO, of: FloatType.Type = FloatType.self
) -> (PythonReader, ModelWeightMapper, [Model.IO], Model.IO, [Input]) {
  let conv2d = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    let upcastAttention = Set(upcastAttention[ds, default: []])
    for j in 0..<numRepeat {
      let (reader, mapper, inputLayer) = BlockLayer(
        prefix: ("model.diffusion_model.input_blocks.\(layerStart)", "down_blocks.\(i)"),
        repeatStart: j, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention.contains(j), usesFlashAttention: usesFlashAttention,
        flags: .Float16, isTemporalMixEnabled: isTemporalMixEnabled, of: FloatType.self)
      previousChannel = channel
      var c: [Input]
      if embeddingLength.0 == 1 && embeddingLength.1 == 1 {
        c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1))).map { _ in Input() }
        if isTemporalMixEnabled && attentionBlock[j] > 0 {
          c.append(contentsOf: (0..<(attentionBlock[j] + 1)).map { _ in Input() })
        }
      } else {
        c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input() }
        if isTemporalMixEnabled && attentionBlock[j] > 0 {
          c.append(contentsOf: (0..<(attentionBlock[j] * 2 + 1)).map { _ in Input() })
        }
      }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      passLayers.append(out)
      readers.append(reader)
      mappers.append(mapper)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
      out = downsample(out)
      passLayers.append(out)
      let downLayer = layerStart
      let reader: PythonReader = { stateDict, archive in
        guard
          let op_weight = stateDict["model.diffusion_model.input_blocks.\(downLayer).0.op.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard let op_bias = stateDict["model.diffusion_model.input_blocks.\(downLayer).0.op.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try downsample.weight.copy(from: op_weight, zip: archive, of: FloatType.self)
        try downsample.bias.copy(from: op_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      let mapper: ModelWeightMapper = { format in
        switch format {
        case .generativeModels:
          return [
            "model.diffusion_model.input_blocks.\(downLayer).0.op.weight": [downsample.weight.name],
            "model.diffusion_model.input_blocks.\(downLayer).0.op.bias": [downsample.bias.name],
          ]
        case .diffusers:
          return [
            "down_blocks.\(i).downsamplers.0.conv.weight": [downsample.weight.name],
            "down_blocks.\(i).downsamplers.0.conv.bias": [downsample.bias.name],
          ]
        }
      }
      mappers.append(mapper)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: PythonReader = { stateDict, archive in
    guard let input_blocks_0_0_weight = stateDict["model.diffusion_model.input_blocks.0.0.weight"]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard let input_blocks_0_0_bias = stateDict["model.diffusion_model.input_blocks.0.0.bias"]
    else {
      throw UnpickleError.tensorNotFound
    }
    try conv2d.weight.copy(from: input_blocks_0_0_weight, zip: archive, of: FloatType.self)
    try conv2d.bias.copy(from: input_blocks_0_0_bias, zip: archive, of: FloatType.self)
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["model.diffusion_model.input_blocks.0.0.weight"] = [conv2d.weight.name]
      mapping["model.diffusion_model.input_blocks.0.0.bias"] = [conv2d.bias.name]
    case .diffusers:
      mapping["conv_in.weight"] = [conv2d.weight.name]
      mapping["conv_in.bias"] = [conv2d.bias.name]
    }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (reader, mapper, passLayers, out, kvs)
}

func OutputBlocks<FloatType: TensorNumeric & BinaryFloatingPoint>(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  injectIPAdapterLengths: [Int], upcastAttention: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel,
  isTemporalMixEnabled: Bool, x: Model.IO, emb: Model.IO, inputs: [Model.IO],
  of: FloatType.Type = FloatType.self
) -> (PythonReader, ModelWeightMapper, Model.IO, [Input]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var kvs = [Input]()
  var inputIdx = inputs.count - 1
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat + 1)]
    let upcastAttention = Set(upcastAttention[ds, default: []])
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 3)(out, inputs[inputIdx])
      inputIdx -= 1
      let (reader, mapper, outputLayer) = BlockLayer(
        prefix: (
          "model.diffusion_model.output_blocks.\(layerStart)", "up_blocks.\(channels.count - 1 - i)"
        ),
        repeatStart: j, skipConnection: true,
        attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
        intermediateSize: channel * 4, injectIPAdapterLengths: injectIPAdapterLengths,
        upcastAttention: upcastAttention.contains(j), usesFlashAttention: usesFlashAttention,
        flags: .Float16, isTemporalMixEnabled: isTemporalMixEnabled, of: FloatType.self)
      var c: [Input]
      if embeddingLength.0 == 1 && embeddingLength.1 == 1 {
        c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1))).map { _ in Input() }
        if isTemporalMixEnabled && attentionBlock[j] > 0 {
          c.append(contentsOf: (0..<(attentionBlock[j] + 1)).map { _ in Input() })
        }
      } else {
        c = (0..<(attentionBlock[j] * (injectIPAdapterLengths.count + 1) * 2)).map { _ in Input() }
        if isTemporalMixEnabled && attentionBlock[j] > 0 {
          c.append(contentsOf: (0..<(attentionBlock[j] * 2 + 1)).map { _ in Input() })
        }
      }
      out = outputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      readers.append(reader)
      mappers.append(mapper)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW
        )
        out = conv2d(out)
        let upLayer = layerStart
        let convIdx = attentionBlock[j] > 0 ? 2 : 1
        let reader: PythonReader = { stateDict, archive in
          guard
            let op_weight = stateDict[
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight"
            ]
          else {
            throw UnpickleError.tensorNotFound
          }
          guard
            let op_bias = stateDict[
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias"]
          else {
            throw UnpickleError.tensorNotFound
          }
          try conv2d.weight.copy(from: op_weight, zip: archive, of: FloatType.self)
          try conv2d.bias.copy(from: op_bias, zip: archive, of: FloatType.self)
        }
        readers.append(reader)
        let reverseUpLayer = channels.count - 1 - i
        let mapper: ModelWeightMapper = { format in
          switch format {
          case .generativeModels:
            return [
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.weight": [
                conv2d.weight.name
              ],
              "model.diffusion_model.output_blocks.\(upLayer).\(convIdx).conv.bias": [
                conv2d.bias.name
              ],
            ]
          case .diffusers:
            return [
              "up_blocks.\(reverseUpLayer).upsamplers.0.conv.weight": [conv2d.weight.name],
              "up_blocks.\(reverseUpLayer).upsamplers.0.conv.bias": [conv2d.bias.name],
            ]
          }
        }
        mappers.append(mapper)
      }
      layerStart += 1
    }
  }
  let reader: PythonReader = { stateDict, archive in
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (reader, mapper, out, kvs)
}

public func UNetXL<FloatType: TensorNumeric & BinaryFloatingPoint>(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  inputAttentionRes: KeyValuePairs<Int, [Int]>,
  middleAttentionBlocks: Int,
  outputAttentionRes: KeyValuePairs<Int, [Int]>,
  embeddingLength: (Int, Int), injectIPAdapterLengths: [Int],
  upcastAttention: (KeyValuePairs<Int, [Int]>, Bool, KeyValuePairs<Int, [Int]>),
  usesFlashAttention: FlashAttentionLevel, injectControls: Bool, isTemporalMixEnabled: Bool,
  trainable: Bool? = nil, of: FloatType.Type = FloatType.self
) -> (Model, PythonReader, ModelWeightMapper) {
  let x = Input()
  let t_emb = Input()
  let y = Input()
  var injectedControls = [Model.IO]()
  if injectControls {
    injectedControls = (0..<(channels.count * 3 + 1)).map { _ in Input() }
  }
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let inputUpcastAttention = [Int: [Int]](
    uniqueKeysWithValues: upcastAttention.0.map { ($0.key, $0.value) })
  let outputUpcastAttention = [Int: [Int]](
    uniqueKeysWithValues: upcastAttention.2.map { ($0.key, $0.value) })
  let (timeFc0, timeFc2, timeEmbed) = TimeEmbed(modelChannels: channels[0])
  let (labelFc0, labelFc2, labelEmbed) = LabelEmbed(modelChannels: channels[0])
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  var (inputReader, inputMapper, inputs, inputBlocks, inputKVs) = InputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: inputAttentionRes, injectIPAdapterLengths: injectIPAdapterLengths,
    upcastAttention: inputUpcastAttention, usesFlashAttention: usesFlashAttention,
    isTemporalMixEnabled: isTemporalMixEnabled, x: x, emb: emb, of: FloatType.self)
  var out = inputBlocks
  let (middleReader, middleMapper, middleBlock, middleKVs) = MiddleBlock(
    prefix: "model.diffusion_model",
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingLength: embeddingLength, attentionBlock: middleAttentionBlocks,
    retainingNormProjInProjOutAndSecondResNetWhenNoAttentionBlocks: false,  // This is particular to SSD-1B. If SSD-1B starts to have ControlNet, we might want to expose this.
    injectIPAdapterLengths: injectIPAdapterLengths, upcastAttention: upcastAttention.1,
    usesFlashAttention: usesFlashAttention, isTemporalMixEnabled: isTemporalMixEnabled, x: out,
    emb: emb, of: FloatType.self)
  out = middleBlock
  if injectControls {
    out = out + injectedControls[injectedControls.count - 1]
    precondition(inputs.count + 1 == injectedControls.count)
    for i in 0..<inputs.count {
      inputs[i] = inputs[i] + injectedControls[i]
    }
  }
  let (outputReader, outputMapper, outputBlocks, outputKVs) = OutputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: outputAttentionRes, injectIPAdapterLengths: injectIPAdapterLengths,
    upcastAttention: outputUpcastAttention, usesFlashAttention: usesFlashAttention,
    isTemporalMixEnabled: isTemporalMixEnabled, x: out, emb: emb, inputs: inputs, of: FloatType.self
  )
  out = outputBlocks
  let outNorm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = outConv2d(out)
  let reader: PythonReader = { stateDict, archive in
    guard let time_embed_0_weight = stateDict["model.diffusion_model.time_embed.0.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let time_embed_0_bias = stateDict["model.diffusion_model.time_embed.0.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let time_embed_2_weight = stateDict["model.diffusion_model.time_embed.2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let time_embed_2_bias = stateDict["model.diffusion_model.time_embed.2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try timeFc0.weight.copy(from: time_embed_0_weight, zip: archive, of: FloatType.self)
    try timeFc0.bias.copy(from: time_embed_0_bias, zip: archive, of: FloatType.self)
    try timeFc2.weight.copy(from: time_embed_2_weight, zip: archive, of: FloatType.self)
    try timeFc2.bias.copy(from: time_embed_2_bias, zip: archive, of: FloatType.self)
    guard let label_emb_0_0_weight = stateDict["model.diffusion_model.label_emb.0.0.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let label_emb_0_0_bias = stateDict["model.diffusion_model.label_emb.0.0.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let label_emb_0_2_weight = stateDict["model.diffusion_model.label_emb.0.2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let label_emb_0_2_bias = stateDict["model.diffusion_model.label_emb.0.2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try labelFc0.weight.copy(from: label_emb_0_0_weight, zip: archive, of: FloatType.self)
    try labelFc0.bias.copy(from: label_emb_0_0_bias, zip: archive, of: FloatType.self)
    try labelFc2.weight.copy(from: label_emb_0_2_weight, zip: archive, of: FloatType.self)
    try labelFc2.bias.copy(from: label_emb_0_2_bias, zip: archive, of: FloatType.self)
    try inputReader(stateDict, archive)
    try middleReader(stateDict, archive)
    try outputReader(stateDict, archive)
    guard let out_0_weight = stateDict["model.diffusion_model.out.0.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let out_0_bias = stateDict["model.diffusion_model.out.0.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try outNorm.weight.copy(from: out_0_weight, zip: archive, of: FloatType.self)
    try outNorm.bias.copy(from: out_0_bias, zip: archive, of: FloatType.self)
    guard let out_2_weight = stateDict["model.diffusion_model.out.2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let out_2_bias = stateDict["model.diffusion_model.out.2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try outConv2d.weight.copy(from: out_2_weight, zip: archive, of: FloatType.self)
    try outConv2d.bias.copy(from: out_2_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(inputMapper(format)) { v, _ in v }
    mapping.merge(middleMapper(format)) { v, _ in v }
    mapping.merge(outputMapper(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      mapping["model.diffusion_model.time_embed.0.weight"] = [timeFc0.weight.name]
      mapping["model.diffusion_model.time_embed.0.bias"] = [timeFc0.bias.name]
      mapping["model.diffusion_model.time_embed.2.weight"] = [timeFc2.weight.name]
      mapping["model.diffusion_model.time_embed.2.bias"] = [timeFc2.bias.name]
      mapping["model.diffusion_model.label_emb.0.0.weight"] = [labelFc0.weight.name]
      mapping["model.diffusion_model.label_emb.0.0.bias"] = [labelFc0.bias.name]
      mapping["model.diffusion_model.label_emb.0.2.weight"] = [labelFc2.weight.name]
      mapping["model.diffusion_model.label_emb.0.2.bias"] = [labelFc2.bias.name]
      mapping["model.diffusion_model.out.0.weight"] = [outNorm.weight.name]
      mapping["model.diffusion_model.out.0.bias"] = [outNorm.bias.name]
      mapping["model.diffusion_model.out.2.weight"] = [outConv2d.weight.name]
      mapping["model.diffusion_model.out.2.bias"] = [outConv2d.bias.name]
    case .diffusers:
      mapping["time_embedding.linear_1.weight"] = [timeFc0.weight.name]
      mapping["time_embedding.linear_1.bias"] = [timeFc0.bias.name]
      mapping["time_embedding.linear_2.weight"] = [timeFc2.weight.name]
      mapping["time_embedding.linear_2.bias"] = [timeFc2.bias.name]
      mapping["add_embedding.linear_1.weight"] = [labelFc0.weight.name]
      mapping["add_embedding.linear_1.bias"] = [labelFc0.bias.name]
      mapping["add_embedding.linear_2.weight"] = [labelFc2.weight.name]
      mapping["add_embedding.linear_2.bias"] = [labelFc2.bias.name]
      mapping["conv_norm_out.weight"] = [outNorm.weight.name]
      mapping["conv_norm_out.bias"] = [outNorm.bias.name]
      mapping["conv_out.weight"] = [outConv2d.weight.name]
      mapping["conv_out.bias"] = [outConv2d.bias.name]
    }
    return mapping
  }
  return (
    Model(
      [x, t_emb, y] + inputKVs + middleKVs + outputKVs + injectedControls, [out],
      trainable: trainable), reader,
    mapper
  )
}

func CrossAttentionFixed(
  k: Int, h: Int, b: Int, t: (Int, Int), usesFlashAttention: FlashAttentionLevel, name: String
)
  -> (Model, Model, Model)
{
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  // We shouldn't transpose if we are going to do that within the UNet.
  if t.0 == t.1 {
    var keys = tokeys(c).reshaped([b, t.0, h, k])
    var values = tovalues(c).reshaped([b, t.0, h, k])
    if usesFlashAttention == .none {
      keys = keys.transposed(1, 2)
      values = values.transposed(1, 2)
    }
    return (tokeys, tovalues, Model([c], [keys, values], name: name))
  } else {
    let keys = tokeys(c)
    let values = tovalues(c)
    return (tokeys, tovalues, Model([c], [keys, values], name: name))
  }
}

func Attention1Fixed(k: Int, h: Int, b: Int, t: (Int, Int), name: String) -> (Model, Model, Model) {
  let c = Input()
  let tovalues = Dense(count: k * h, noBias: true)
  let unifyheads = Dense(count: k * h)
  let values = unifyheads(tovalues(c))
  return (tovalues, unifyheads, Model([c], [values], name: name))
}

func BasicTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int, b: Int, t: (Int, Int), intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (PythonReader, ModelWeightMapper, Model) {
  if t.0 == 1 && t.1 == 1 {
    let (tovalues2, unifyheads2, attn2) = Attention1Fixed(
      k: k, h: h, b: b, t: t, name: "")
    let reader: PythonReader = { stateDict, archive in
      guard
        let attn2_to_v_weight = stateDict[
          "\(prefix.0).attn2.to_v.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tovalues2.weight.copy(from: attn2_to_v_weight, zip: archive, of: FloatType.self)
      guard
        let attn2_to_out_weight = stateDict[
          "\(prefix.0).attn2.to_out.0.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let attn2_to_out_bias = stateDict[
          "\(prefix.0).attn2.to_out.0.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try unifyheads2.weight.copy(from: attn2_to_out_weight, zip: archive, of: FloatType.self)
      try unifyheads2.bias.copy(from: attn2_to_out_bias, zip: archive, of: FloatType.self)
    }
    let mapper: ModelWeightMapper = { format in
      var mapping = ModelWeightMapping()
      switch format {
      case .generativeModels:
        mapping["\(prefix.0).attn2.to_v.weight"] = [tovalues2.weight.name]
        mapping["\(prefix.0).attn2.to_out.0.weight"] = [
          unifyheads2.weight.name
        ]
        mapping["\(prefix.0).attn2.to_out.0.bias"] = [unifyheads2.bias.name]
      case .diffusers:
        mapping["\(prefix.1).attn2.to_v.weight"] = [tovalues2.weight.name]
        mapping["\(prefix.1).attn2.to_out.0.weight"] = [unifyheads2.weight.name]
        mapping["\(prefix.1).attn2.to_out.0.bias"] = [unifyheads2.bias.name]
      }
      return mapping
    }
    return (reader, mapper, attn2)
  } else {
    let (tokeys2, tovalues2, attn2) = CrossAttentionFixed(
      k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention, name: "")
    let reader: PythonReader = { stateDict, archive in
      guard
        let attn2_to_k_weight = stateDict[
          "\(prefix.0).attn2.to_k.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tokeys2.weight.copy(from: attn2_to_k_weight, zip: archive, of: FloatType.self)
      guard
        let attn2_to_v_weight = stateDict[
          "\(prefix.0).attn2.to_v.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tovalues2.weight.copy(from: attn2_to_v_weight, zip: archive, of: FloatType.self)
    }
    let mapper: ModelWeightMapper = { format in
      var mapping = ModelWeightMapping()
      switch format {
      case .generativeModels:
        mapping["\(prefix.0).attn2.to_k.weight"] = [tokeys2.weight.name]
        mapping["\(prefix.0).attn2.to_v.weight"] = [tovalues2.weight.name]
      case .diffusers:
        mapping["\(prefix.1).attn2.to_k.weight"] = [tokeys2.weight.name]
        mapping["\(prefix.1).attn2.to_v.weight"] = [tovalues2.weight.name]
      }
      return mapping
    }
    return (reader, mapper, attn2)
  }
}

func TimePosEmbedTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int
) -> (PythonReader, ModelWeightMapper, Model) {
  let (timePosFc0, timePosFc2, timePosEmbed) = TimePosEmbed(modelChannels: k * h)
  let reader: PythonReader = { stateDict, archive in
    guard
      let time_pos_embed_0_weight = stateDict[
        "\(prefix.0).time_pos_embed.0.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_pos_embed_0_bias = stateDict[
        "\(prefix.0).time_pos_embed.0.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try timePosFc0.weight.copy(from: time_pos_embed_0_weight, zip: archive, of: FloatType.self)
    try timePosFc0.bias.copy(from: time_pos_embed_0_bias, zip: archive, of: FloatType.self)
    guard
      let time_pos_embed_2_weight = stateDict[
        "\(prefix.0).time_pos_embed.2.weight"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    guard
      let time_pos_embed_2_bias = stateDict[
        "\(prefix.0).time_pos_embed.2.bias"
      ]
    else {
      throw UnpickleError.tensorNotFound
    }
    try timePosFc2.weight.copy(from: time_pos_embed_2_weight, zip: archive, of: FloatType.self)
    try timePosFc2.bias.copy(from: time_pos_embed_2_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["\(prefix.0).time_pos_embed.0.weight"] = [timePosFc0.weight.name]
      mapping["\(prefix.0).time_pos_embed.0.bias"] = [timePosFc0.bias.name]
      mapping["\(prefix.0).time_pos_embed.2.weight"] = [timePosFc2.weight.name]
      mapping["\(prefix.0).time_pos_embed.2.bias"] = [timePosFc2.bias.name]
    case .diffusers:
      mapping["\(prefix.1).time_pos_embed.linear_1.weight"] = [timePosFc0.weight.name]
      mapping["\(prefix.1).time_pos_embed.linear_1.bias"] = [timePosFc0.bias.name]
      mapping["\(prefix.1).time_pos_embed.linear_2.weight"] = [timePosFc2.weight.name]
      mapping["\(prefix.1).time_pos_embed.linear_2.bias"] = [timePosFc2.bias.name]
    }
    return mapping
  }
  return (reader, mapper, timePosEmbed)
}

func BasicTimeTransformerBlockFixed(
  prefix: (String, String), k: Int, h: Int, b: Int, t: (Int, Int), intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel
) -> (PythonReader, ModelWeightMapper, Model) {
  if t.0 == 1 && t.1 == 1 {
    let (tovalues2, unifyheads2, attn2) = Attention1Fixed(
      k: k, h: h, b: b, t: t, name: "time_stack")
    let reader: PythonReader = { stateDict, archive in
      guard
        let attn2_to_v_weight = stateDict[
          "\(prefix.0).attn2.to_v.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tovalues2.weight.copy(from: attn2_to_v_weight, zip: archive, of: FloatType.self)
      guard
        let attn2_to_out_weight = stateDict[
          "\(prefix.0).attn2.to_out.0.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard
        let attn2_to_out_bias = stateDict[
          "\(prefix.0).attn2.to_out.0.bias"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try unifyheads2.weight.copy(from: attn2_to_out_weight, zip: archive, of: FloatType.self)
      try unifyheads2.bias.copy(from: attn2_to_out_bias, zip: archive, of: FloatType.self)
    }
    let mapper: ModelWeightMapper = { format in
      var mapping = ModelWeightMapping()
      switch format {
      case .generativeModels:
        mapping["\(prefix.0).attn2.to_v.weight"] = [tovalues2.weight.name]
        mapping["\(prefix.0).attn2.to_out.0.weight"] = [unifyheads2.weight.name]
        mapping["\(prefix.0).attn2.to_out.0.bias"] = [unifyheads2.bias.name]
      case .diffusers:
        mapping["\(prefix.1).attn2.to_v.weight"] = [tovalues2.weight.name]
        mapping["\(prefix.1).attn2.to_out.0.weight"] = [unifyheads2.weight.name]
        mapping["\(prefix.1).attn2.to_out.0.bias"] = [unifyheads2.bias.name]
      }
      return mapping
    }
    return (reader, mapper, attn2)
  } else {
    let (tokeys2, tovalues2, attn2) = CrossAttentionFixed(
      k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention, name: "time_stack")
    let reader: PythonReader = { stateDict, archive in
      guard
        let attn2_to_k_weight = stateDict[
          "\(prefix.0).attn2.to_k.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tokeys2.weight.copy(from: attn2_to_k_weight, zip: archive, of: FloatType.self)
      guard
        let attn2_to_v_weight = stateDict[
          "\(prefix.0).attn2.to_v.weight"
        ]
      else {
        throw UnpickleError.tensorNotFound
      }
      try tovalues2.weight.copy(from: attn2_to_v_weight, zip: archive, of: FloatType.self)
    }
    let mapper: ModelWeightMapper = { format in
      var mapping = ModelWeightMapping()
      switch format {
      case .generativeModels:
        mapping["\(prefix.0).attn2.to_k.weight"] = [tokeys2.weight.name]
        mapping["\(prefix.0).attn2.to_v.weight"] = [tovalues2.weight.name]
      case .diffusers:
        mapping["\(prefix.1).attn2.to_k.weight"] = [tokeys2.weight.name]
        mapping["\(prefix.1).attn2.to_v.weight"] = [tovalues2.weight.name]
      }
      return mapping
    }
    return (reader, mapper, attn2)
  }
}

func SpatialTransformerFixed(
  prefix: (String, String),
  ch: Int, k: Int, h: Int, b: Int, depth: Int, t: (Int, Int),
  intermediateSize: Int, usesFlashAttention: FlashAttentionLevel, isTemporalMixEnabled: Bool
) -> (PythonReader, ModelWeightMapper, Model) {
  let c = Input()
  let numFrames: Input?
  var outs = [Model.IO]()
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  if isTemporalMixEnabled {
    let frames = Input()
    let (reader, mapper, timePosEmbed) = TimePosEmbedTransformerBlockFixed(
      prefix: prefix, k: k, h: h)
    outs.append(timePosEmbed(frames))
    readers.append(reader)
    mappers.append(mapper)
    numFrames = frames
  } else {
    numFrames = nil
  }
  for i in 0..<depth {
    let (reader, mapper, block) = BasicTransformerBlockFixed(
      prefix: ("\(prefix.0).transformer_blocks.\(i)", "\(prefix.1).transformer_blocks.\(i)"), k: k,
      h: h, b: b, t: t,
      intermediateSize: intermediateSize, usesFlashAttention: usesFlashAttention)
    outs.append(block(c))
    readers.append(reader)
    mappers.append(mapper)
    if isTemporalMixEnabled {
      let (timeReader, timeMapper, timeBlock) = BasicTimeTransformerBlockFixed(
        prefix: ("\(prefix.0).time_stack.\(i)", "\(prefix.1).temporal_transformer_blocks.\(i)"),
        k: k, h: h, b: b,
        t: t,
        intermediateSize: intermediateSize, usesFlashAttention: usesFlashAttention)
      outs.append(timeBlock(c))
      readers.append(timeReader)
      mappers.append(timeMapper)
    }
  }
  let reader: PythonReader = { stateDict, archive in
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  if let numFrames = numFrames {
    return (reader, mapper, Model([c, numFrames], outs))
  } else {
    return (reader, mapper, Model([c], outs))
  }
}

func BlockLayerFixed(
  prefix: (String, String),
  repeatStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingLength: (Int, Int), intermediateSize: Int,
  usesFlashAttention: FlashAttentionLevel, isTemporalMixEnabled: Bool
) -> (PythonReader, ModelWeightMapper, Model) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (transformerReader, transformerMapper, transformer) = SpatialTransformerFixed(
    prefix: ("\(prefix.0).1", "\(prefix.1).attentions.\(repeatStart)"),
    ch: channels, k: k, h: numHeads, b: batchSize,
    depth: attentionBlock, t: embeddingLength,
    intermediateSize: channels * 4, usesFlashAttention: usesFlashAttention,
    isTemporalMixEnabled: isTemporalMixEnabled)
  return (transformerReader, transformerMapper, transformer)
}

func MiddleBlockFixed(
  prefix: String,
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int,
  embeddingLength: (Int, Int), attentionBlock: Int, usesFlashAttention: FlashAttentionLevel,
  isTemporalMixEnabled: Bool, c: Model.IO, numFrames: [Model.IO]
) -> (PythonReader, ModelWeightMapper, Model.IO) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let (
    transformerReader, transformerMapper, transformer
  ) = SpatialTransformerFixed(
    prefix: ("\(prefix).middle_block.1", "mid_block.attentions.0"), ch: channels, k: k, h: numHeads,
    b: batchSize, depth: attentionBlock, t: embeddingLength, intermediateSize: channels * 4,
    usesFlashAttention: usesFlashAttention, isTemporalMixEnabled: isTemporalMixEnabled)
  let out = numFrames.isEmpty ? transformer(c) : transformer(c, numFrames[numFrames.count - 1])
  return (transformerReader, transformerMapper, out)
}

func InputBlocksFixed(
  prefix: String,
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel, isTemporalMixEnabled: Bool,
  c: Model.IO, numFrames: [Model.IO]
) -> (PythonReader, ModelWeightMapper, [Model.IO]) {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var outs = [Model.IO]()
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat)]
    for j in 0..<numRepeat {
      if attentionBlock[j] > 0 {
        let (reader, mapper, inputLayer) = BlockLayerFixed(
          prefix: ("\(prefix).input_blocks.\(layerStart)", "down_blocks.\(i)"),
          repeatStart: j, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
          intermediateSize: channel * 4, usesFlashAttention: usesFlashAttention,
          isTemporalMixEnabled: isTemporalMixEnabled)
        previousChannel = channel
        if numFrames.isEmpty {
          outs.append(inputLayer(c))
        } else {
          outs.append(inputLayer(c, numFrames[i]))
        }
        readers.append(reader)
        mappers.append(mapper)
      }
      layerStart += 1
    }
    if i != channels.count - 1 {
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  let reader: PythonReader = { stateDict, archive in
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (reader, mapper, outs)
}

func OutputBlocksFixed(
  prefix: String,
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingLength: (Int, Int), attentionRes: [Int: [Int]],
  usesFlashAttention: FlashAttentionLevel, isTemporalMixEnabled: Bool,
  c: Model.IO, numFrames: [Model.IO]
) -> (PythonReader, ModelWeightMapper, [Model.IO]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var outs = [Model.IO]()
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: Array(repeating: 0, count: numRepeat + 1)]
    for j in 0..<(numRepeat + 1) {
      if attentionBlock[j] > 0 {
        let (reader, mapper, outputLayer) = BlockLayerFixed(
          prefix: ("\(prefix).output_blocks.\(layerStart)", "up_blocks.\(channels.count - 1 - i)"),
          repeatStart: j, skipConnection: true,
          attentionBlock: attentionBlock[j], channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize, height: height, width: width, embeddingLength: embeddingLength,
          intermediateSize: channel * 4, usesFlashAttention: usesFlashAttention,
          isTemporalMixEnabled: isTemporalMixEnabled)
        if numFrames.isEmpty {
          outs.append(outputLayer(c))
        } else {
          outs.append(outputLayer(c, numFrames[i]))
        }
        readers.append(reader)
        mappers.append(mapper)
      }
      layerStart += 1
    }
  }
  let reader: PythonReader = { stateDict, archive in
    for reader in readers {
      try reader(stateDict, archive)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    return mapping
  }
  return (reader, mapper, outs)
}

public func UNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  embeddingLength: (Int, Int),
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int,
  outputAttentionRes: KeyValuePairs<Int, [Int]>, usesFlashAttention: FlashAttentionLevel,
  isTemporalMixEnabled: Bool, trainable: Bool? = nil
) -> (Model, PythonReader, ModelWeightMapper) {
  let c = Input()
  let numFrames: [Input]
  if isTemporalMixEnabled {
    numFrames = (0..<channels.count).map { _ in Input() }
  } else {
    numFrames = []
  }
  let inputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttentionRes = [Int: [Int]](
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let (inputReader, inputMapper, inputBlocks) = InputBlocksFixed(
    prefix: "model.diffusion_model",
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: inputAttentionRes, usesFlashAttention: usesFlashAttention,
    isTemporalMixEnabled: isTemporalMixEnabled,
    c: c, numFrames: numFrames)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let middleReader: PythonReader?
  let middleMapper: ModelWeightMapper?
  if middleAttentionBlocks > 0 {
    let (reader, mapper, middleBlock) = MiddleBlockFixed(
      prefix: "model.diffusion_model",
      channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
      height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
      embeddingLength: embeddingLength, attentionBlock: middleAttentionBlocks,
      usesFlashAttention: usesFlashAttention, isTemporalMixEnabled: isTemporalMixEnabled, c: c,
      numFrames: numFrames)
    out.append(middleBlock)
    middleReader = reader
    middleMapper = mapper
  } else {
    middleReader = nil
    middleMapper = nil
  }
  let (outputReader, outputMapper, outputBlocks) = OutputBlocksFixed(
    prefix: "model.diffusion_model",
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingLength: embeddingLength,
    attentionRes: outputAttentionRes, usesFlashAttention: usesFlashAttention,
    isTemporalMixEnabled: isTemporalMixEnabled,
    c: c, numFrames: numFrames)
  out.append(contentsOf: outputBlocks)
  let reader: PythonReader = { stateDict, archive in
    try inputReader(stateDict, archive)
    try middleReader?(stateDict, archive)
    try outputReader(stateDict, archive)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping.merge(inputMapper(format)) { v, _ in v }
    if let middleMapper = middleMapper {
      mapping.merge(middleMapper(format)) { v, _ in v }
    }
    mapping.merge(outputMapper(format)) { v, _ in v }
    return mapping
  }
  return (Model([c] + numFrames, out, trainable: trainable), reader, mapper)
}
