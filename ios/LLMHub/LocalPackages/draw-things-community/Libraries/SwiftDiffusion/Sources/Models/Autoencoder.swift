import DiffusionMappings
import NNC

/// Autoencoder

private func NHWCResnetBlock(
  prefix: (String, String), outChannels: Int, shortcut: Bool, specializingNames: Bool
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  let x = Input()
  let norm1 = GroupNorm(
    axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2],
    name: specializingNames ? "resnet_norm1" : "")
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: specializingNames ? "resnet_conv1" : "")
  out = conv1(out)
  let norm2 = GroupNorm(
    axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2],
    name: specializingNames ? "resnet_norm2" : "")
  out = norm2(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: specializingNames ? "resnet_conv2" : "")
  out = conv2(out)
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: specializingNames ? "resnet_shortcut" : ""
    )
    out = nin(x) + out
    ninShortcut = nin
  } else {
    out = x + out
    ninShortcut = nil
  }
  let reader: PythonReader = { stateDict, archive in
    guard let norm1_weight = stateDict["first_stage_model.\(prefix.0).norm1.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm1_bias = stateDict["first_stage_model.\(prefix.0).norm1.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm1.parameters(for: .weight).copy(from: norm1_weight, zip: archive, of: FloatType.self)
    try norm1.parameters(for: .bias).copy(from: norm1_bias, zip: archive, of: FloatType.self)
    guard let conv1_weight = stateDict["first_stage_model.\(prefix.0).conv1.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv1_bias = stateDict["first_stage_model.\(prefix.0).conv1.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try conv1.parameters(for: .weight).copy(from: conv1_weight, zip: archive, of: FloatType.self)
    try conv1.parameters(for: .bias).copy(from: conv1_bias, zip: archive, of: FloatType.self)
    guard let norm2_weight = stateDict["first_stage_model.\(prefix.0).norm2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm2_bias = stateDict["first_stage_model.\(prefix.0).norm2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm2.parameters(for: .weight).copy(from: norm2_weight, zip: archive, of: FloatType.self)
    try norm2.parameters(for: .bias).copy(from: norm2_bias, zip: archive, of: FloatType.self)
    guard let conv2_weight = stateDict["first_stage_model.\(prefix.0).conv2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv2_bias = stateDict["first_stage_model.\(prefix.0).conv2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try conv2.parameters(for: .weight).copy(from: conv2_weight, zip: archive, of: FloatType.self)
    try conv2.parameters(for: .bias).copy(from: conv2_bias, zip: archive, of: FloatType.self)
    if let ninShortcut = ninShortcut {
      guard let nin_shortcut_weight = stateDict["first_stage_model.\(prefix.0).nin_shortcut.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard let nin_shortcut_bias = stateDict["first_stage_model.\(prefix.0).nin_shortcut.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try ninShortcut.parameters(for: .weight).copy(
        from: nin_shortcut_weight, zip: archive, of: FloatType.self)
      try ninShortcut.parameters(for: .bias).copy(
        from: nin_shortcut_bias, zip: archive, of: FloatType.self)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["first_stage_model.\(prefix.0).norm1.weight"] = [norm1.weight.name]
      mapping["first_stage_model.\(prefix.0).norm1.bias"] = [norm1.bias.name]
      mapping["first_stage_model.\(prefix.0).conv1.weight"] = [conv1.weight.name]
      mapping["first_stage_model.\(prefix.0).conv1.bias"] = [conv1.bias.name]
      mapping["first_stage_model.\(prefix.0).norm2.weight"] = [norm2.weight.name]
      mapping["first_stage_model.\(prefix.0).norm2.bias"] = [norm2.bias.name]
      mapping["first_stage_model.\(prefix.0).conv2.weight"] = [conv2.weight.name]
      mapping["first_stage_model.\(prefix.0).conv2.bias"] = [conv2.bias.name]
      if let ninShortcut = ninShortcut {
        mapping["first_stage_model.\(prefix.0).nin_shortcut.weight"] = [ninShortcut.weight.name]
        mapping["first_stage_model.\(prefix.0).nin_shortcut.bias"] = [ninShortcut.bias.name]
      }
    case .diffusers:
      mapping["\(prefix.1).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.1).norm1.bias"] = [norm1.bias.name]
      mapping["\(prefix.1).conv1.weight"] = [conv1.weight.name]
      mapping["\(prefix.1).conv1.bias"] = [conv1.bias.name]
      mapping["\(prefix.1).norm2.weight"] = [norm2.weight.name]
      mapping["\(prefix.1).norm2.bias"] = [norm2.bias.name]
      mapping["\(prefix.1).conv2.weight"] = [conv2.weight.name]
      mapping["\(prefix.1).conv2.bias"] = [conv2.bias.name]
      if let ninShortcut = ninShortcut {
        mapping["\(prefix.1).conv_shortcut.weight"] = [ninShortcut.weight.name]
        mapping["\(prefix.1).conv_shortcut.bias"] = [ninShortcut.bias.name]
      }
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

private func NHWCAttnBlock(
  prefix: (String, String), inChannels: Int, batchSize: Int, width: Int, height: Int,
  highPrecisionKeysAndValues: Bool, usesFlashAttention: Bool, specializingNames: Bool
) -> (
  Model, PythonReader, ModelWeightMapper
) {
  let x = Input()
  let norm = GroupNorm(
    axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: specializingNames ? "attn_norm" : "")
  var out = norm(x)
  let hw = width * height
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: specializingNames ? "attn_to_q" : "")
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: specializingNames ? "attn_to_k" : "")
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: specializingNames ? "attn_to_v" : "")
  let projOut: Model
  if usesFlashAttention {
    if highPrecisionKeysAndValues {
      out = out.to(.Float32)
    }
    let q = toqueries(out).reshaped([batchSize, hw, inChannels]).identity().identity()
    let k = tokeys(out).reshaped([batchSize, hw, inChannels]).identity()
    let v = tovalues(out).reshaped([batchSize, hw, inChannels])
    projOut = ScaledDotProductAttention(
      scale: 1.0 / Float(inChannels).squareRoot(), multiHeadOutputProjectionFused: true,
      name: specializingNames ? "attn_out" : "")
    out = projOut(q, k, v).reshaped([batchSize, height, width, inChannels])
    out = x + out.to(of: x)
  } else {
    let original = out
    if highPrecisionKeysAndValues {
      out = out.to(.Float32)
    }
    let k = tokeys(out).reshaped([batchSize, hw, inChannels])
    let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
      batchSize, hw, inChannels,
    ])
    let v = tovalues(original).reshaped([batchSize, hw, inChannels])
    var dot = Matmul(transposeB: (1, 2))(q, k)
    dot = dot.reshaped([batchSize * hw, hw])
    dot = dot.softmax()
    dot = dot.reshaped([batchSize, hw, hw])
    out = dot.to(of: v) * v
    projOut = Convolution(
      groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
      name: specializingNames ? "attn_out" : "")
    out = x + projOut(out.reshaped([batchSize, height, width, inChannels]))
  }
  let reader: PythonReader = { stateDict, archive in
    guard let norm_weight = stateDict["first_stage_model.\(prefix.0).norm.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_bias = stateDict["first_stage_model.\(prefix.0).norm.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm.parameters(for: .weight).copy(from: norm_weight, zip: archive, of: FloatType.self)
    try norm.parameters(for: .bias).copy(from: norm_bias, zip: archive, of: FloatType.self)
    guard let k_weight = stateDict["first_stage_model.\(prefix.0).k.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let k_bias = stateDict["first_stage_model.\(prefix.0).k.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try tokeys.parameters(for: .weight).copy(from: k_weight, zip: archive, of: FloatType.self)
    try tokeys.parameters(for: .bias).copy(from: k_bias, zip: archive, of: FloatType.self)
    guard let q_weight = stateDict["first_stage_model.\(prefix.0).q.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let q_bias = stateDict["first_stage_model.\(prefix.0).q.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try toqueries.parameters(for: .weight).copy(from: q_weight, zip: archive, of: FloatType.self)
    try toqueries.parameters(for: .bias).copy(from: q_bias, zip: archive, of: FloatType.self)
    guard let v_weight = stateDict["first_stage_model.\(prefix.0).v.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let v_bias = stateDict["first_stage_model.\(prefix.0).v.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try tovalues.parameters(for: .weight).copy(from: v_weight, zip: archive, of: FloatType.self)
    try tovalues.parameters(for: .bias).copy(from: v_bias, zip: archive, of: FloatType.self)
    guard let proj_out_weight = stateDict["first_stage_model.\(prefix.0).proj_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let proj_out_bias = stateDict["first_stage_model.\(prefix.0).proj_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try projOut.parameters(for: .weight).copy(
      from: proj_out_weight, zip: archive, of: FloatType.self)
    try projOut.parameters(for: .bias).copy(from: proj_out_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["first_stage_model.\(prefix.0).norm.weight"] = [norm.weight.name]
      mapping["first_stage_model.\(prefix.0).norm.bias"] = [norm.bias.name]
      mapping["first_stage_model.\(prefix.0).q.weight"] = [toqueries.weight.name]
      mapping["first_stage_model.\(prefix.0).q.bias"] = [toqueries.bias.name]
      mapping["first_stage_model.\(prefix.0).k.weight"] = [tokeys.weight.name]
      mapping["first_stage_model.\(prefix.0).k.bias"] = [tokeys.bias.name]
      mapping["first_stage_model.\(prefix.0).v.weight"] = [tovalues.weight.name]
      mapping["first_stage_model.\(prefix.0).v.bias"] = [tovalues.bias.name]
      mapping["first_stage_model.\(prefix.0).proj_out.weight"] = [projOut.weight.name]
      mapping["first_stage_model.\(prefix.0).proj_out.bias"] = [projOut.bias.name]
    case .diffusers:
      mapping["\(prefix.1).group_norm.weight"] = [norm.weight.name]
      mapping["\(prefix.1).group_norm.bias"] = [norm.bias.name]
      mapping["\(prefix.1).to_q.weight"] = [toqueries.weight.name]
      mapping["\(prefix.1).to_q.bias"] = [toqueries.bias.name]
      mapping["\(prefix.1).to_k.weight"] = [tokeys.weight.name]
      mapping["\(prefix.1).to_k.bias"] = [tokeys.bias.name]
      mapping["\(prefix.1).to_v.weight"] = [tovalues.weight.name]
      mapping["\(prefix.1).to_v.bias"] = [tovalues.bias.name]
      mapping["\(prefix.1).to_out.0.weight"] = [projOut.weight.name]
      mapping["\(prefix.1).to_out.0.bias"] = [projOut.bias.name]
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

private func NHWCEncoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int,
  usesFlashAttention: Bool, quantLayer: Bool, outputChannels: Int
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = convIn(x)
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  var height = startHeight
  var width = startWidth
  for _ in 1..<channels.count {
    height *= 2
    width *= 2
  }
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (block, reader, mapper) = NHWCResnetBlock(
        prefix: ("encoder.down.\(i).block.\(j)", "encoder.down_blocks.\(i).resnets.\(j)"),
        outChannels: channel,
        shortcut: previousChannel != channel, specializingNames: false)
      readers.append(reader)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      // Conv always pad left first, then right, and pad top first then bottom.
      // Thus, we cannot have (0, 1, 0, 1) (left 0, right 1, top 0, bottom 1) padding as in
      // Stable Diffusion. Instead, we pad to (2, 1, 2, 1) and simply discard the first row and first column.
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])), format: .OIHW)
      out = conv2d(out).reshaped(
        [batchSize, height, width, channel], offset: [0, 1, 1, 0],
        strides: [channel * (height + 1) * (width + 1), (width + 1) * channel, channel, 1])
      let downLayer = i
      let reader: PythonReader = { stateDict, archive in
        guard
          let conv_weight = stateDict[
            "first_stage_model.encoder.down.\(downLayer).downsample.conv.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let conv_bias = stateDict[
            "first_stage_model.encoder.down.\(downLayer).downsample.conv.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try conv2d.parameters(for: .weight).copy(
          from: conv_weight, zip: archive, of: FloatType.self)
        try conv2d.parameters(for: .bias).copy(from: conv_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      let mapper: ModelWeightMapper = { format in
        var mapping = ModelWeightMapping()
        switch format {
        case .generativeModels:
          mapping["first_stage_model.encoder.down.\(downLayer).downsample.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["first_stage_model.encoder.down.\(downLayer).downsample.conv.bias"] = [
            conv2d.bias.name
          ]
        case .diffusers:
          mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.bias"] = [conv2d.bias.name]
        }
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let (midBlock1, midBlockReader1, midBlockMapper1) = NHWCResnetBlock(
    prefix: ("encoder.mid.block_1", "encoder.mid_block.resnets.0"), outChannels: previousChannel,
    shortcut: false, specializingNames: false)
  out = midBlock1(out)
  let (midAttn1, midAttnReader1, midAttnMapper1) = NHWCAttnBlock(
    prefix: ("encoder.mid.attn_1", "encoder.mid_block.attentions.0"), inChannels: previousChannel,
    batchSize: batchSize, width: startWidth, height: startHeight, highPrecisionKeysAndValues: false,
    usesFlashAttention: usesFlashAttention, specializingNames: false)
  out = midAttn1(out)
  let (midBlock2, midBlockReader2, midBlockMapper2) = NHWCResnetBlock(
    prefix: ("encoder.mid.block_2", "encoder.mid_block.resnets.1"), outChannels: previousChannel,
    shortcut: false, specializingNames: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: outputChannels * 2, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convOut(out)
  let quantConv2d: Model?
  if quantLayer {
    let quantConv = Convolution(
      groups: 1, filters: outputChannels * 2, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW)
    out = quantConv(out)
    quantConv2d = quantConv
  } else {
    quantConv2d = nil
  }
  let reader: PythonReader = { stateDict, archive in
    guard let conv_in_weight = stateDict["first_stage_model.encoder.conv_in.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_in_bias = stateDict["first_stage_model.encoder.conv_in.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convIn.parameters(for: .weight).copy(from: conv_in_weight, zip: archive, of: FloatType.self)
    try convIn.parameters(for: .bias).copy(from: conv_in_bias, zip: archive, of: FloatType.self)
    for reader in readers {
      try reader(stateDict, archive)
    }
    try midBlockReader1(stateDict, archive)
    try midAttnReader1(stateDict, archive)
    try midBlockReader2(stateDict, archive)
    guard let norm_out_weight = stateDict["first_stage_model.encoder.norm_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_out_bias = stateDict["first_stage_model.encoder.norm_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try normOut.parameters(for: .weight).copy(
      from: norm_out_weight, zip: archive, of: FloatType.self)
    try normOut.parameters(for: .bias).copy(from: norm_out_bias, zip: archive, of: FloatType.self)
    guard let conv_out_weight = stateDict["first_stage_model.encoder.conv_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_out_bias = stateDict["first_stage_model.encoder.conv_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convOut.parameters(for: .weight).copy(
      from: conv_out_weight, zip: archive, of: FloatType.self)
    try convOut.parameters(for: .bias).copy(from: conv_out_bias, zip: archive, of: FloatType.self)
    if let quantConv2d = quantConv2d {
      guard let quant_conv_weight = stateDict["first_stage_model.quant_conv.weight"] else {
        throw UnpickleError.tensorNotFound
      }
      guard let quant_conv_bias = stateDict["first_stage_model.quant_conv.bias"] else {
        throw UnpickleError.tensorNotFound
      }
      try quantConv2d.parameters(for: .weight).copy(
        from: quant_conv_weight, zip: archive, of: FloatType.self)
      try quantConv2d.parameters(for: .bias).copy(
        from: quant_conv_bias, zip: archive, of: FloatType.self)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      mapping["first_stage_model.encoder.conv_in.weight"] = [convIn.weight.name]
      mapping["first_stage_model.encoder.conv_in.bias"] = [convIn.bias.name]
      mapping["first_stage_model.encoder.norm_out.weight"] = [normOut.weight.name]
      mapping["first_stage_model.encoder.norm_out.bias"] = [normOut.bias.name]
      mapping["first_stage_model.encoder.conv_out.weight"] = [convOut.weight.name]
      mapping["first_stage_model.encoder.conv_out.bias"] = [convOut.bias.name]
      if let quantConv2d = quantConv2d {
        mapping["first_stage_model.quant_conv.weight"] = [quantConv2d.weight.name]
        mapping["first_stage_model.quant_conv.bias"] = [quantConv2d.bias.name]
      }
    case .diffusers:
      mapping["encoder.conv_in.weight"] = [convIn.weight.name]
      mapping["encoder.conv_in.bias"] = [convIn.bias.name]
      mapping["encoder.conv_norm_out.weight"] = [normOut.weight.name]
      mapping["encoder.conv_norm_out.bias"] = [normOut.bias.name]
      mapping["encoder.conv_out.weight"] = [convOut.weight.name]
      mapping["encoder.conv_out.bias"] = [convOut.bias.name]
      if let quantConv2d = quantConv2d {
        mapping["quant_conv.weight"] = [quantConv2d.weight.name]
        mapping["quant_conv.bias"] = [quantConv2d.bias.name]
      }
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

private func NHWCDecoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int,
  inputChannels: Int, highPrecisionKeysAndValues: Bool, usesFlashAttention: Bool,
  paddingFinalConvLayer: Bool, quantLayer: Bool, specializingNames: Bool
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  let x = Input()
  var out: Model.IO
  let postQuantConv2d: Model?
  if quantLayer {
    let postQuantConv = Convolution(
      groups: 1, filters: inputChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: specializingNames ? "post_quant_conv" : "")
    out = postQuantConv(x)
    postQuantConv2d = postQuantConv
  } else {
    out = x
    postQuantConv2d = nil
  }
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: specializingNames ? "conv_in" : "")
  out = convIn(out)
  let (midBlock1, midBlockReader1, midBlockMapper1) = NHWCResnetBlock(
    prefix: ("decoder.mid.block_1", "decoder.mid_block.resnets.0"), outChannels: previousChannel,
    shortcut: false, specializingNames: specializingNames)
  out = midBlock1(out)
  let (midAttn1, midAttnReader1, midAttnMapper1) = NHWCAttnBlock(
    prefix: ("decoder.mid.attn_1", "decoder.mid_block.attentions.0"), inChannels: previousChannel,
    batchSize: batchSize, width: startWidth, height: startHeight,
    highPrecisionKeysAndValues: highPrecisionKeysAndValues, usesFlashAttention: usesFlashAttention,
    specializingNames: specializingNames)
  out = midAttn1(out)
  let (midBlock2, midBlockReader2, midBlockMapper2) = NHWCResnetBlock(
    prefix: ("decoder.mid.block_2", "decoder.mid_block.resnets.1"), outChannels: previousChannel,
    shortcut: false, specializingNames: specializingNames)
  out = midBlock2(out)
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (block, reader, mapper) = NHWCResnetBlock(
        prefix: (
          "decoder.up.\(i).block.\(j)", "decoder.up_blocks.\(channels.count - 1 - i).resnets.\(j)"
        ), outChannels: channel,
        shortcut: previousChannel != channel, specializingNames: specializingNames)
      readers.append(reader)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
        name: specializingNames ? "upsample" : "")
      out = conv2d(out)
      let upLayer = i
      let reader: PythonReader = { stateDict, archive in
        guard
          let conv_weight = stateDict[
            "first_stage_model.decoder.up.\(upLayer).upsample.conv.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let conv_bias = stateDict["first_stage_model.decoder.up.\(upLayer).upsample.conv.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try conv2d.parameters(for: .weight).copy(
          from: conv_weight, zip: archive, of: FloatType.self)
        try conv2d.parameters(for: .bias).copy(from: conv_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      let mapper: ModelWeightMapper = { format in
        var mapping = ModelWeightMapping()
        switch format {
        case .generativeModels:
          mapping["first_stage_model.decoder.up.\(upLayer).upsample.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["first_stage_model.decoder.up.\(upLayer).upsample.conv.bias"] = [conv2d.bias.name]
        case .diffusers:
          mapping["decoder.up_blocks.\(channels.count - 1 - upLayer).upsamplers.0.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["decoder.up_blocks.\(channels.count - 1 - upLayer).upsamplers.0.conv.bias"] = [
            conv2d.bias.name
          ]
        }
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let normOut = GroupNorm(
    axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: specializingNames ? "norm_out" : "")
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 4 : 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: specializingNames ? "conv_out" : "")
  out = convOut(out)
  let reader: PythonReader = { stateDict, archive in
    if let postQuantConv2d = postQuantConv2d {
      guard let post_quant_conv_weight = stateDict["first_stage_model.post_quant_conv.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard let post_quant_conv_bias = stateDict["first_stage_model.post_quant_conv.bias"] else {
        throw UnpickleError.tensorNotFound
      }
      try postQuantConv2d.parameters(for: .weight).copy(
        from: post_quant_conv_weight, zip: archive, of: FloatType.self)
      try postQuantConv2d.parameters(for: .bias).copy(
        from: post_quant_conv_bias, zip: archive, of: FloatType.self)
    }
    guard let conv_in_weight = stateDict["first_stage_model.decoder.conv_in.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_in_bias = stateDict["first_stage_model.decoder.conv_in.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convIn.parameters(for: .weight).copy(from: conv_in_weight, zip: archive, of: FloatType.self)
    try convIn.parameters(for: .bias).copy(from: conv_in_bias, zip: archive, of: FloatType.self)
    try midBlockReader1(stateDict, archive)
    try midAttnReader1(stateDict, archive)
    try midBlockReader2(stateDict, archive)
    for reader in readers {
      try reader(stateDict, archive)
    }
    guard let norm_out_weight = stateDict["first_stage_model.decoder.norm_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_out_bias = stateDict["first_stage_model.decoder.norm_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try normOut.parameters(for: .weight).copy(
      from: norm_out_weight, zip: archive, of: FloatType.self)
    try normOut.parameters(for: .bias).copy(from: norm_out_bias, zip: archive, of: FloatType.self)
    guard let conv_out_weight = stateDict["first_stage_model.decoder.conv_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_out_bias = stateDict["first_stage_model.decoder.conv_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convOut.parameters(for: .weight).copy(
      from: conv_out_weight, zip: archive, of: FloatType.self)
    try convOut.parameters(for: .bias).copy(from: conv_out_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      if let postQuantConv2d = postQuantConv2d {
        mapping["first_stage_model.post_quant_conv.weight"] = [postQuantConv2d.weight.name]
        mapping["first_stage_model.post_quant_conv.bias"] = [postQuantConv2d.bias.name]
      }
      mapping["first_stage_model.decoder.conv_in.weight"] = [convIn.weight.name]
      mapping["first_stage_model.decoder.conv_in.bias"] = [convIn.bias.name]
      mapping["first_stage_model.decoder.norm_out.weight"] = [normOut.weight.name]
      mapping["first_stage_model.decoder.norm_out.bias"] = [normOut.bias.name]
      mapping["first_stage_model.decoder.conv_out.weight"] = [convOut.weight.name]
      mapping["first_stage_model.decoder.conv_out.bias"] = [convOut.bias.name]
    case .diffusers:
      if let postQuantConv2d = postQuantConv2d {
        mapping["post_quant_conv.weight"] = [postQuantConv2d.weight.name]
        mapping["post_quant_conv.bias"] = [postQuantConv2d.bias.name]
      }
      mapping["decoder.conv_in.weight"] = [convIn.weight.name]
      mapping["decoder.conv_in.bias"] = [convIn.bias.name]
      mapping["decoder.conv_norm_out.weight"] = [normOut.weight.name]
      mapping["decoder.conv_norm_out.bias"] = [normOut.bias.name]
      mapping["decoder.conv_out.weight"] = [convOut.weight.name]
      mapping["decoder.conv_out.bias"] = [convOut.bias.name]
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

private func NCHWResnetBlock(
  prefix: (String, String), outChannels: Int, shortcut: Bool, specializingNames: Bool
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  let x = Input()
  let norm1 = GroupNorm(
    axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3],
    name: specializingNames ? "resnet_norm1" : "")
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: specializingNames ? "resnet_conv1" : "")
  out = conv1(out)
  let norm2 = GroupNorm(
    axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3],
    name: specializingNames ? "resnet_norm2" : "")
  out = norm2(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: specializingNames ? "resnet_conv2" : "")
  out = conv2(out)
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: specializingNames ? "resnet_shortcut" : ""
    )
    out = nin(x) + out
    ninShortcut = nin
  } else {
    out = x + out
    ninShortcut = nil
  }
  let reader: PythonReader = { stateDict, archive in
    guard let norm1_weight = stateDict["first_stage_model.\(prefix.0).norm1.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm1_bias = stateDict["first_stage_model.\(prefix.0).norm1.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm1.parameters(for: .weight).copy(from: norm1_weight, zip: archive, of: FloatType.self)
    try norm1.parameters(for: .bias).copy(from: norm1_bias, zip: archive, of: FloatType.self)
    guard let conv1_weight = stateDict["first_stage_model.\(prefix.0).conv1.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv1_bias = stateDict["first_stage_model.\(prefix.0).conv1.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try conv1.parameters(for: .weight).copy(from: conv1_weight, zip: archive, of: FloatType.self)
    try conv1.parameters(for: .bias).copy(from: conv1_bias, zip: archive, of: FloatType.self)
    guard let norm2_weight = stateDict["first_stage_model.\(prefix.0).norm2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm2_bias = stateDict["first_stage_model.\(prefix.0).norm2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm2.parameters(for: .weight).copy(from: norm2_weight, zip: archive, of: FloatType.self)
    try norm2.parameters(for: .bias).copy(from: norm2_bias, zip: archive, of: FloatType.self)
    guard let conv2_weight = stateDict["first_stage_model.\(prefix.0).conv2.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv2_bias = stateDict["first_stage_model.\(prefix.0).conv2.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try conv2.parameters(for: .weight).copy(from: conv2_weight, zip: archive, of: FloatType.self)
    try conv2.parameters(for: .bias).copy(from: conv2_bias, zip: archive, of: FloatType.self)
    if let ninShortcut = ninShortcut {
      guard let nin_shortcut_weight = stateDict["first_stage_model.\(prefix.0).nin_shortcut.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard let nin_shortcut_bias = stateDict["first_stage_model.\(prefix.0).nin_shortcut.bias"]
      else {
        throw UnpickleError.tensorNotFound
      }
      try ninShortcut.parameters(for: .weight).copy(
        from: nin_shortcut_weight, zip: archive, of: FloatType.self)
      try ninShortcut.parameters(for: .bias).copy(
        from: nin_shortcut_bias, zip: archive, of: FloatType.self)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["first_stage_model.\(prefix.0).norm1.weight"] = [norm1.weight.name]
      mapping["first_stage_model.\(prefix.0).norm1.bias"] = [norm1.bias.name]
      mapping["first_stage_model.\(prefix.0).conv1.weight"] = [conv1.weight.name]
      mapping["first_stage_model.\(prefix.0).conv1.bias"] = [conv1.bias.name]
      mapping["first_stage_model.\(prefix.0).norm2.weight"] = [norm2.weight.name]
      mapping["first_stage_model.\(prefix.0).norm2.bias"] = [norm2.bias.name]
      mapping["first_stage_model.\(prefix.0).conv2.weight"] = [conv2.weight.name]
      mapping["first_stage_model.\(prefix.0).conv2.bias"] = [conv2.bias.name]
      if let ninShortcut = ninShortcut {
        mapping["first_stage_model.\(prefix.0).nin_shortcut.weight"] = [ninShortcut.weight.name]
        mapping["first_stage_model.\(prefix.0).nin_shortcut.bias"] = [ninShortcut.bias.name]
      }
    case .diffusers:
      mapping["\(prefix.1).norm1.weight"] = [norm1.weight.name]
      mapping["\(prefix.1).norm1.bias"] = [norm1.bias.name]
      mapping["\(prefix.1).conv1.weight"] = [conv1.weight.name]
      mapping["\(prefix.1).conv1.bias"] = [conv1.bias.name]
      mapping["\(prefix.1).norm2.weight"] = [norm2.weight.name]
      mapping["\(prefix.1).norm2.bias"] = [norm2.bias.name]
      mapping["\(prefix.1).conv2.weight"] = [conv2.weight.name]
      mapping["\(prefix.1).conv2.bias"] = [conv2.bias.name]
      if let ninShortcut = ninShortcut {
        mapping["\(prefix.1).conv_shortcut.weight"] = [ninShortcut.weight.name]
        mapping["\(prefix.1).conv_shortcut.bias"] = [ninShortcut.bias.name]
      }
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

private func NCHWAttnBlock(
  prefix: (String, String), inChannels: Int, batchSize: Int, width: Int, height: Int,
  highPrecisionKeysAndValues: Bool, usesFlashAttention: Bool, specializingNames: Bool
) -> (
  Model, PythonReader, ModelWeightMapper
) {
  let x = Input()
  let norm = GroupNorm(
    axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: specializingNames ? "attn_norm" : "")
  var out = norm(x)
  let hw = width * height
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: specializingNames ? "attn_to_q" : "")
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: specializingNames ? "attn_to_k" : "")
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: specializingNames ? "attn_to_v" : "")
  let original = out
  if highPrecisionKeysAndValues {
    out = out.to(.Float32)
  }
  let k = tokeys(out).reshaped([batchSize, inChannels, hw])
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    batchSize, inChannels, hw,
  ])
  let v = tovalues(original).reshaped([batchSize, inChannels, hw])
  var dot = Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize, hw, hw])
  out = Matmul(transposeB: (1, 2))(v, dot.to(of: v))
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: specializingNames ? "attn_out" : "")
  out = x + projOut(out.reshaped([batchSize, inChannels, height, width]))
  let reader: PythonReader = { stateDict, archive in
    guard let norm_weight = stateDict["first_stage_model.\(prefix.0).norm.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_bias = stateDict["first_stage_model.\(prefix.0).norm.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try norm.parameters(for: .weight).copy(from: norm_weight, zip: archive, of: FloatType.self)
    try norm.parameters(for: .bias).copy(from: norm_bias, zip: archive, of: FloatType.self)
    guard let k_weight = stateDict["first_stage_model.\(prefix.0).k.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let k_bias = stateDict["first_stage_model.\(prefix.0).k.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try tokeys.parameters(for: .weight).copy(from: k_weight, zip: archive, of: FloatType.self)
    try tokeys.parameters(for: .bias).copy(from: k_bias, zip: archive, of: FloatType.self)
    guard let q_weight = stateDict["first_stage_model.\(prefix.0).q.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let q_bias = stateDict["first_stage_model.\(prefix.0).q.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try toqueries.parameters(for: .weight).copy(from: q_weight, zip: archive, of: FloatType.self)
    try toqueries.parameters(for: .bias).copy(from: q_bias, zip: archive, of: FloatType.self)
    guard let v_weight = stateDict["first_stage_model.\(prefix.0).v.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let v_bias = stateDict["first_stage_model.\(prefix.0).v.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try tovalues.parameters(for: .weight).copy(from: v_weight, zip: archive, of: FloatType.self)
    try tovalues.parameters(for: .bias).copy(from: v_bias, zip: archive, of: FloatType.self)
    guard let proj_out_weight = stateDict["first_stage_model.\(prefix.0).proj_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let proj_out_bias = stateDict["first_stage_model.\(prefix.0).proj_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try projOut.parameters(for: .weight).copy(
      from: proj_out_weight, zip: archive, of: FloatType.self)
    try projOut.parameters(for: .bias).copy(from: proj_out_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    switch format {
    case .generativeModels:
      mapping["first_stage_model.\(prefix.0).norm.weight"] = [norm.weight.name]
      mapping["first_stage_model.\(prefix.0).norm.bias"] = [norm.bias.name]
      mapping["first_stage_model.\(prefix.0).q.weight"] = [toqueries.weight.name]
      mapping["first_stage_model.\(prefix.0).q.bias"] = [toqueries.bias.name]
      mapping["first_stage_model.\(prefix.0).k.weight"] = [tokeys.weight.name]
      mapping["first_stage_model.\(prefix.0).k.bias"] = [tokeys.bias.name]
      mapping["first_stage_model.\(prefix.0).v.weight"] = [tovalues.weight.name]
      mapping["first_stage_model.\(prefix.0).v.bias"] = [tovalues.bias.name]
      mapping["first_stage_model.\(prefix.0).proj_out.weight"] = [projOut.weight.name]
      mapping["first_stage_model.\(prefix.0).proj_out.bias"] = [projOut.bias.name]
    case .diffusers:
      mapping["\(prefix.1).group_norm.weight"] = [norm.weight.name]
      mapping["\(prefix.1).group_norm.bias"] = [norm.bias.name]
      mapping["\(prefix.1).to_q.weight"] = [toqueries.weight.name]
      mapping["\(prefix.1).to_q.bias"] = [toqueries.bias.name]
      mapping["\(prefix.1).to_k.weight"] = [tokeys.weight.name]
      mapping["\(prefix.1).to_k.bias"] = [tokeys.bias.name]
      mapping["\(prefix.1).to_v.weight"] = [tovalues.weight.name]
      mapping["\(prefix.1).to_v.bias"] = [tovalues.bias.name]
      mapping["\(prefix.1).to_out.0.weight"] = [projOut.weight.name]
      mapping["\(prefix.1).to_out.0.bias"] = [projOut.bias.name]
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

private func NCHWEncoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int,
  usesFlashAttention: Bool, quantLayer: Bool, outputChannels: Int
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  var height = startHeight
  var width = startWidth
  for _ in 1..<channels.count {
    height *= 2
    width *= 2
  }
  var out = convIn(x.permuted(0, 3, 1, 2).contiguous().reshaped(.NCHW(batchSize, 3, height, width)))
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (block, reader, mapper) = NCHWResnetBlock(
        prefix: ("encoder.down.\(i).block.\(j)", "encoder.down_blocks.\(i).resnets.\(j)"),
        outChannels: channel,
        shortcut: previousChannel != channel, specializingNames: false)
      readers.append(reader)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      // Conv always pad left first, then right, and pad top first then bottom.
      // Thus, we cannot have (0, 1, 0, 1) (left 0, right 1, top 0, bottom 1) padding as in
      // Stable Diffusion. Instead, we pad to (2, 1, 2, 1) and simply discard the first row and first column.
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])), format: .OIHW)
      out = conv2d(out).reshaped(
        [batchSize, channel, height, width], offset: [0, 0, 1, 1],
        strides: [channel * (height + 1) * (width + 1), (height + 1) * (width + 1), width + 1, 1])
      let downLayer = i
      let reader: PythonReader = { stateDict, archive in
        guard
          let conv_weight = stateDict[
            "first_stage_model.encoder.down.\(downLayer).downsample.conv.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let conv_bias = stateDict[
            "first_stage_model.encoder.down.\(downLayer).downsample.conv.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try conv2d.parameters(for: .weight).copy(
          from: conv_weight, zip: archive, of: FloatType.self)
        try conv2d.parameters(for: .bias).copy(from: conv_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      let mapper: ModelWeightMapper = { format in
        var mapping = ModelWeightMapping()
        switch format {
        case .generativeModels:
          mapping["first_stage_model.encoder.down.\(downLayer).downsample.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["first_stage_model.encoder.down.\(downLayer).downsample.conv.bias"] = [
            conv2d.bias.name
          ]
        case .diffusers:
          mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.bias"] = [conv2d.bias.name]
        }
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let (midBlock1, midBlockReader1, midBlockMapper1) = NCHWResnetBlock(
    prefix: ("encoder.mid.block_1", "encoder.mid_block.resnets.0"), outChannels: previousChannel,
    shortcut: false, specializingNames: false)
  out = midBlock1(out)
  let (midAttn1, midAttnReader1, midAttnMapper1) = NCHWAttnBlock(
    prefix: ("encoder.mid.attn_1", "encoder.mid_block.attentions.0"), inChannels: previousChannel,
    batchSize: batchSize, width: startWidth, height: startHeight, highPrecisionKeysAndValues: false,
    usesFlashAttention: usesFlashAttention, specializingNames: false)
  out = midAttn1(out)
  let (midBlock2, midBlockReader2, midBlockMapper2) = NCHWResnetBlock(
    prefix: ("encoder.mid.block_2", "encoder.mid_block.resnets.1"), outChannels: previousChannel,
    shortcut: false, specializingNames: false)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: outputChannels * 2, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convOut(out)
  let quantConv2d: Model?
  if quantLayer {
    let quantConv = Convolution(
      groups: 1, filters: outputChannels * 2, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW)
    out = quantConv(out).permuted(0, 2, 3, 1).contiguous().reshaped(
      .NHWC(batchSize, startHeight, startWidth, outputChannels * 2))
    quantConv2d = quantConv
  } else {
    out = out.permuted(0, 2, 3, 1).contiguous().reshaped(
      .NHWC(batchSize, startHeight, startWidth, outputChannels * 2))
    quantConv2d = nil
  }
  let reader: PythonReader = { stateDict, archive in
    guard let conv_in_weight = stateDict["first_stage_model.encoder.conv_in.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_in_bias = stateDict["first_stage_model.encoder.conv_in.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convIn.parameters(for: .weight).copy(from: conv_in_weight, zip: archive, of: FloatType.self)
    try convIn.parameters(for: .bias).copy(from: conv_in_bias, zip: archive, of: FloatType.self)
    for reader in readers {
      try reader(stateDict, archive)
    }
    try midBlockReader1(stateDict, archive)
    try midAttnReader1(stateDict, archive)
    try midBlockReader2(stateDict, archive)
    guard let norm_out_weight = stateDict["first_stage_model.encoder.norm_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_out_bias = stateDict["first_stage_model.encoder.norm_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try normOut.parameters(for: .weight).copy(
      from: norm_out_weight, zip: archive, of: FloatType.self)
    try normOut.parameters(for: .bias).copy(from: norm_out_bias, zip: archive, of: FloatType.self)
    guard let conv_out_weight = stateDict["first_stage_model.encoder.conv_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_out_bias = stateDict["first_stage_model.encoder.conv_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convOut.parameters(for: .weight).copy(
      from: conv_out_weight, zip: archive, of: FloatType.self)
    try convOut.parameters(for: .bias).copy(from: conv_out_bias, zip: archive, of: FloatType.self)
    if let quantConv2d = quantConv2d {
      guard let quant_conv_weight = stateDict["first_stage_model.quant_conv.weight"] else {
        throw UnpickleError.tensorNotFound
      }
      guard let quant_conv_bias = stateDict["first_stage_model.quant_conv.bias"] else {
        throw UnpickleError.tensorNotFound
      }
      try quantConv2d.parameters(for: .weight).copy(
        from: quant_conv_weight, zip: archive, of: FloatType.self)
      try quantConv2d.parameters(for: .bias).copy(
        from: quant_conv_bias, zip: archive, of: FloatType.self)
    }
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      mapping["first_stage_model.encoder.conv_in.weight"] = [convIn.weight.name]
      mapping["first_stage_model.encoder.conv_in.bias"] = [convIn.bias.name]
      mapping["first_stage_model.encoder.norm_out.weight"] = [normOut.weight.name]
      mapping["first_stage_model.encoder.norm_out.bias"] = [normOut.bias.name]
      mapping["first_stage_model.encoder.conv_out.weight"] = [convOut.weight.name]
      mapping["first_stage_model.encoder.conv_out.bias"] = [convOut.bias.name]
      if let quantConv2d = quantConv2d {
        mapping["first_stage_model.quant_conv.weight"] = [quantConv2d.weight.name]
        mapping["first_stage_model.quant_conv.bias"] = [quantConv2d.bias.name]
      }
    case .diffusers:
      mapping["encoder.conv_in.weight"] = [convIn.weight.name]
      mapping["encoder.conv_in.bias"] = [convIn.bias.name]
      mapping["encoder.conv_norm_out.weight"] = [normOut.weight.name]
      mapping["encoder.conv_norm_out.bias"] = [normOut.bias.name]
      mapping["encoder.conv_out.weight"] = [convOut.weight.name]
      mapping["encoder.conv_out.bias"] = [convOut.bias.name]
      if let quantConv2d = quantConv2d {
        mapping["quant_conv.weight"] = [quantConv2d.weight.name]
        mapping["quant_conv.bias"] = [quantConv2d.bias.name]
      }
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

private func NCHWDecoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int,
  inputChannels: Int, highPrecisionKeysAndValues: Bool, usesFlashAttention: Bool,
  paddingFinalConvLayer: Bool, quantLayer: Bool, specializingNames: Bool
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  let x = Input()
  var out: Model.IO
  let postQuantConv2d: Model?
  if quantLayer {
    let postQuantConv = Convolution(
      groups: 1, filters: inputChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: specializingNames ? "post_quant_conv" : "")
    out = postQuantConv(
      x.permuted(0, 3, 1, 2).contiguous().reshaped(
        .NCHW(batchSize, inputChannels, startHeight, startWidth)))
    postQuantConv2d = postQuantConv
  } else {
    out = x.permuted(0, 3, 1, 2).contiguous().reshaped(
      .NCHW(batchSize, inputChannels, startHeight, startWidth))
    postQuantConv2d = nil
  }
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: specializingNames ? "conv_in" : "")
  out = convIn(out)
  let (midBlock1, midBlockReader1, midBlockMapper1) = NCHWResnetBlock(
    prefix: ("decoder.mid.block_1", "decoder.mid_block.resnets.0"), outChannels: previousChannel,
    shortcut: false, specializingNames: specializingNames)
  out = midBlock1(out)
  let (midAttn1, midAttnReader1, midAttnMapper1) = NCHWAttnBlock(
    prefix: ("decoder.mid.attn_1", "decoder.mid_block.attentions.0"), inChannels: previousChannel,
    batchSize: batchSize, width: startWidth, height: startHeight,
    highPrecisionKeysAndValues: highPrecisionKeysAndValues, usesFlashAttention: usesFlashAttention,
    specializingNames: specializingNames)
  out = midAttn1(out)
  let (midBlock2, midBlockReader2, midBlockMapper2) = NCHWResnetBlock(
    prefix: ("decoder.mid.block_2", "decoder.mid_block.resnets.1"), outChannels: previousChannel,
    shortcut: false, specializingNames: specializingNames)
  out = midBlock2(out)
  var readers = [PythonReader]()
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (block, reader, mapper) = NCHWResnetBlock(
        prefix: (
          "decoder.up.\(i).block.\(j)", "decoder.up_blocks.\(channels.count - 1 - i).resnets.\(j)"
        ), outChannels: channel,
        shortcut: previousChannel != channel, specializingNames: specializingNames)
      readers.append(reader)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
        name: specializingNames ? "upsample" : "")
      out = conv2d(out)
      let upLayer = i
      let reader: PythonReader = { stateDict, archive in
        guard
          let conv_weight = stateDict[
            "first_stage_model.decoder.up.\(upLayer).upsample.conv.weight"]
        else {
          throw UnpickleError.tensorNotFound
        }
        guard
          let conv_bias = stateDict["first_stage_model.decoder.up.\(upLayer).upsample.conv.bias"]
        else {
          throw UnpickleError.tensorNotFound
        }
        try conv2d.parameters(for: .weight).copy(
          from: conv_weight, zip: archive, of: FloatType.self)
        try conv2d.parameters(for: .bias).copy(from: conv_bias, zip: archive, of: FloatType.self)
      }
      readers.append(reader)
      let mapper: ModelWeightMapper = { format in
        var mapping = ModelWeightMapping()
        switch format {
        case .generativeModels:
          mapping["first_stage_model.decoder.up.\(upLayer).upsample.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["first_stage_model.decoder.up.\(upLayer).upsample.conv.bias"] = [conv2d.bias.name]
        case .diffusers:
          mapping["decoder.up_blocks.\(channels.count - 1 - upLayer).upsamplers.0.conv.weight"] = [
            conv2d.weight.name
          ]
          mapping["decoder.up_blocks.\(channels.count - 1 - upLayer).upsamplers.0.conv.bias"] = [
            conv2d.bias.name
          ]
        }
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let normOut = GroupNorm(
    axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3], name: specializingNames ? "norm_out" : "")
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 4 : 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: specializingNames ? "conv_out" : "")
  out = convOut(out).permuted(0, 2, 3, 1).contiguous().reshaped(
    .NHWC(batchSize, startHeight * 8, startWidth * 8, paddingFinalConvLayer ? 4 : 3))
  let reader: PythonReader = { stateDict, archive in
    if let postQuantConv2d = postQuantConv2d {
      guard let post_quant_conv_weight = stateDict["first_stage_model.post_quant_conv.weight"]
      else {
        throw UnpickleError.tensorNotFound
      }
      guard let post_quant_conv_bias = stateDict["first_stage_model.post_quant_conv.bias"] else {
        throw UnpickleError.tensorNotFound
      }
      try postQuantConv2d.parameters(for: .weight).copy(
        from: post_quant_conv_weight, zip: archive, of: FloatType.self)
      try postQuantConv2d.parameters(for: .bias).copy(
        from: post_quant_conv_bias, zip: archive, of: FloatType.self)
    }
    guard let conv_in_weight = stateDict["first_stage_model.decoder.conv_in.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_in_bias = stateDict["first_stage_model.decoder.conv_in.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convIn.parameters(for: .weight).copy(from: conv_in_weight, zip: archive, of: FloatType.self)
    try convIn.parameters(for: .bias).copy(from: conv_in_bias, zip: archive, of: FloatType.self)
    try midBlockReader1(stateDict, archive)
    try midAttnReader1(stateDict, archive)
    try midBlockReader2(stateDict, archive)
    for reader in readers {
      try reader(stateDict, archive)
    }
    guard let norm_out_weight = stateDict["first_stage_model.decoder.norm_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let norm_out_bias = stateDict["first_stage_model.decoder.norm_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try normOut.parameters(for: .weight).copy(
      from: norm_out_weight, zip: archive, of: FloatType.self)
    try normOut.parameters(for: .bias).copy(from: norm_out_bias, zip: archive, of: FloatType.self)
    guard let conv_out_weight = stateDict["first_stage_model.decoder.conv_out.weight"] else {
      throw UnpickleError.tensorNotFound
    }
    guard let conv_out_bias = stateDict["first_stage_model.decoder.conv_out.bias"] else {
      throw UnpickleError.tensorNotFound
    }
    try convOut.parameters(for: .weight).copy(
      from: conv_out_weight, zip: archive, of: FloatType.self)
    try convOut.parameters(for: .bias).copy(from: conv_out_bias, zip: archive, of: FloatType.self)
  }
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    switch format {
    case .generativeModels:
      if let postQuantConv2d = postQuantConv2d {
        mapping["first_stage_model.post_quant_conv.weight"] = [postQuantConv2d.weight.name]
        mapping["first_stage_model.post_quant_conv.bias"] = [postQuantConv2d.bias.name]
      }
      mapping["first_stage_model.decoder.conv_in.weight"] = [convIn.weight.name]
      mapping["first_stage_model.decoder.conv_in.bias"] = [convIn.bias.name]
      mapping["first_stage_model.decoder.norm_out.weight"] = [normOut.weight.name]
      mapping["first_stage_model.decoder.norm_out.bias"] = [normOut.bias.name]
      mapping["first_stage_model.decoder.conv_out.weight"] = [convOut.weight.name]
      mapping["first_stage_model.decoder.conv_out.bias"] = [convOut.bias.name]
    case .diffusers:
      if let postQuantConv2d = postQuantConv2d {
        mapping["post_quant_conv.weight"] = [postQuantConv2d.weight.name]
        mapping["post_quant_conv.bias"] = [postQuantConv2d.bias.name]
      }
      mapping["decoder.conv_in.weight"] = [convIn.weight.name]
      mapping["decoder.conv_in.bias"] = [convIn.bias.name]
      mapping["decoder.conv_norm_out.weight"] = [normOut.weight.name]
      mapping["decoder.conv_norm_out.bias"] = [normOut.bias.name]
      mapping["decoder.conv_out.weight"] = [convOut.weight.name]
      mapping["decoder.conv_out.bias"] = [convOut.bias.name]
    }
    return mapping

  }
  return (Model([x], [out]), reader, mapper)
}

public func Encoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int,
  usesFlashAttention: Bool, format: TensorFormat, quantLayer: Bool = true, outputChannels: Int = 4
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  switch format {
  case .NHWC:
    return NHWCEncoder(
      channels: channels, numRepeat: numRepeat, batchSize: batchSize, startWidth: startWidth,
      startHeight: startHeight, usesFlashAttention: usesFlashAttention, quantLayer: quantLayer,
      outputChannels: outputChannels)
  case .NCHW:
    return NCHWEncoder(
      channels: channels, numRepeat: numRepeat, batchSize: batchSize, startWidth: startWidth,
      startHeight: startHeight, usesFlashAttention: usesFlashAttention, quantLayer: quantLayer,
      outputChannels: outputChannels)
  case .CHWN:
    fatalError()
  }
}

public func Decoder(
  channels: [Int], numRepeat: Int, batchSize: Int, startWidth: Int, startHeight: Int,
  inputChannels: Int, highPrecisionKeysAndValues: Bool, usesFlashAttention: Bool,
  paddingFinalConvLayer: Bool, format: TensorFormat, quantLayer: Bool = true,
  specializingNames: Bool = false
)
  -> (Model, PythonReader, ModelWeightMapper)
{
  switch format {
  case .NHWC:
    return NHWCDecoder(
      channels: channels, numRepeat: numRepeat, batchSize: batchSize, startWidth: startWidth,
      startHeight: startHeight, inputChannels: inputChannels,
      highPrecisionKeysAndValues: highPrecisionKeysAndValues,
      usesFlashAttention: usesFlashAttention, paddingFinalConvLayer: paddingFinalConvLayer,
      quantLayer: quantLayer, specializingNames: specializingNames)
  case .NCHW:
    return NCHWDecoder(
      channels: channels, numRepeat: numRepeat, batchSize: batchSize, startWidth: startWidth,
      startHeight: startHeight, inputChannels: inputChannels,
      highPrecisionKeysAndValues: highPrecisionKeysAndValues,
      usesFlashAttention: usesFlashAttention, paddingFinalConvLayer: paddingFinalConvLayer,
      quantLayer: quantLayer, specializingNames: specializingNames)
  case .CHWN:
    fatalError()
  }
}

public let DecoderSpecializingNamesMapping: [String: String] = [
  "t-post_quant_conv-0-0": "t-0-0",
  "t-post_quant_conv-0-1": "t-0-1",
  "t-conv_in-0-0": "t-1-0",
  "t-conv_in-0-1": "t-1-1",
  "t-attn_out-0-0": "t-10-0",
  "t-attn_out-0-1": "t-10-1",
  "t-resnet_norm1-1-0": "t-11-0",
  "t-resnet_norm1-1-1": "t-11-1",
  "t-resnet_conv1-1-0": "t-12-0",
  "t-resnet_conv1-1-1": "t-12-1",
  "t-resnet_norm2-1-0": "t-13-0",
  "t-resnet_norm2-1-1": "t-13-1",
  "t-resnet_conv2-1-0": "t-14-0",
  "t-resnet_conv2-1-1": "t-14-1",
  "t-resnet_norm1-2-0": "t-15-0",
  "t-resnet_norm1-2-1": "t-15-1",
  "t-resnet_conv1-2-0": "t-16-0",
  "t-resnet_conv1-2-1": "t-16-1",
  "t-resnet_norm2-2-0": "t-17-0",
  "t-resnet_norm2-2-1": "t-17-1",
  "t-resnet_conv2-2-0": "t-18-0",
  "t-resnet_conv2-2-1": "t-18-1",
  "t-resnet_norm1-3-0": "t-19-0",
  "t-resnet_norm1-3-1": "t-19-1",
  "t-resnet_norm1-0-0": "t-2-0",
  "t-resnet_norm1-0-1": "t-2-1",
  "t-resnet_conv1-3-0": "t-20-0",
  "t-resnet_conv1-3-1": "t-20-1",
  "t-resnet_norm2-3-0": "t-21-0",
  "t-resnet_norm2-3-1": "t-21-1",
  "t-resnet_conv2-3-0": "t-22-0",
  "t-resnet_conv2-3-1": "t-22-1",
  "t-resnet_norm1-4-0": "t-23-0",
  "t-resnet_norm1-4-1": "t-23-1",
  "t-resnet_conv1-4-0": "t-24-0",
  "t-resnet_conv1-4-1": "t-24-1",
  "t-resnet_norm2-4-0": "t-25-0",
  "t-resnet_norm2-4-1": "t-25-1",
  "t-resnet_conv2-4-0": "t-26-0",
  "t-resnet_conv2-4-1": "t-26-1",
  "t-upsample-0-0": "t-27-0",
  "t-upsample-0-1": "t-27-1",
  "t-resnet_norm1-5-0": "t-28-0",
  "t-resnet_norm1-5-1": "t-28-1",
  "t-resnet_conv1-5-0": "t-29-0",
  "t-resnet_conv1-5-1": "t-29-1",
  "t-resnet_conv1-0-0": "t-3-0",
  "t-resnet_conv1-0-1": "t-3-1",
  "t-resnet_norm2-5-0": "t-30-0",
  "t-resnet_norm2-5-1": "t-30-1",
  "t-resnet_conv2-5-0": "t-31-0",
  "t-resnet_conv2-5-1": "t-31-1",
  "t-resnet_norm1-6-0": "t-32-0",
  "t-resnet_norm1-6-1": "t-32-1",
  "t-resnet_conv1-6-0": "t-33-0",
  "t-resnet_conv1-6-1": "t-33-1",
  "t-resnet_norm2-6-0": "t-34-0",
  "t-resnet_norm2-6-1": "t-34-1",
  "t-resnet_conv2-6-0": "t-35-0",
  "t-resnet_conv2-6-1": "t-35-1",
  "t-resnet_norm1-7-0": "t-36-0",
  "t-resnet_norm1-7-1": "t-36-1",
  "t-resnet_conv1-7-0": "t-37-0",
  "t-resnet_conv1-7-1": "t-37-1",
  "t-resnet_norm2-7-0": "t-38-0",
  "t-resnet_norm2-7-1": "t-38-1",
  "t-resnet_conv2-7-0": "t-39-0",
  "t-resnet_conv2-7-1": "t-39-1",
  "t-resnet_norm2-0-0": "t-4-0",
  "t-resnet_norm2-0-1": "t-4-1",
  "t-upsample-1-0": "t-40-0",
  "t-upsample-1-1": "t-40-1",
  "t-resnet_norm1-8-0": "t-41-0",
  "t-resnet_norm1-8-1": "t-41-1",
  "t-resnet_conv1-8-0": "t-42-0",
  "t-resnet_conv1-8-1": "t-42-1",
  "t-resnet_norm2-8-0": "t-43-0",
  "t-resnet_norm2-8-1": "t-43-1",
  "t-resnet_conv2-8-0": "t-44-0",
  "t-resnet_conv2-8-1": "t-44-1",
  "t-resnet_shortcut-0-0": "t-45-0",
  "t-resnet_shortcut-0-1": "t-45-1",
  "t-resnet_norm1-9-0": "t-46-0",
  "t-resnet_norm1-9-1": "t-46-1",
  "t-resnet_conv1-9-0": "t-47-0",
  "t-resnet_conv1-9-1": "t-47-1",
  "t-resnet_norm2-9-0": "t-48-0",
  "t-resnet_norm2-9-1": "t-48-1",
  "t-resnet_conv2-9-0": "t-49-0",
  "t-resnet_conv2-9-1": "t-49-1",
  "t-resnet_conv2-0-0": "t-5-0",
  "t-resnet_conv2-0-1": "t-5-1",
  "t-resnet_norm1-10-0": "t-50-0",
  "t-resnet_norm1-10-1": "t-50-1",
  "t-resnet_conv1-10-0": "t-51-0",
  "t-resnet_conv1-10-1": "t-51-1",
  "t-resnet_norm2-10-0": "t-52-0",
  "t-resnet_norm2-10-1": "t-52-1",
  "t-resnet_conv2-10-0": "t-53-0",
  "t-resnet_conv2-10-1": "t-53-1",
  "t-upsample-2-0": "t-54-0",
  "t-upsample-2-1": "t-54-1",
  "t-resnet_norm1-11-0": "t-55-0",
  "t-resnet_norm1-11-1": "t-55-1",
  "t-resnet_conv1-11-0": "t-56-0",
  "t-resnet_conv1-11-1": "t-56-1",
  "t-resnet_norm2-11-0": "t-57-0",
  "t-resnet_norm2-11-1": "t-57-1",
  "t-resnet_conv2-11-0": "t-58-0",
  "t-resnet_conv2-11-1": "t-58-1",
  "t-resnet_shortcut-1-0": "t-59-0",
  "t-resnet_shortcut-1-1": "t-59-1",
  "t-attn_norm-0-0": "t-6-0",
  "t-attn_norm-0-1": "t-6-1",
  "t-resnet_norm1-12-0": "t-60-0",
  "t-resnet_norm1-12-1": "t-60-1",
  "t-resnet_conv1-12-0": "t-61-0",
  "t-resnet_conv1-12-1": "t-61-1",
  "t-resnet_norm2-12-0": "t-62-0",
  "t-resnet_norm2-12-1": "t-62-1",
  "t-resnet_conv2-12-0": "t-63-0",
  "t-resnet_conv2-12-1": "t-63-1",
  "t-resnet_norm1-13-0": "t-64-0",
  "t-resnet_norm1-13-1": "t-64-1",
  "t-resnet_conv1-13-0": "t-65-0",
  "t-resnet_conv1-13-1": "t-65-1",
  "t-resnet_norm2-13-0": "t-66-0",
  "t-resnet_norm2-13-1": "t-66-1",
  "t-resnet_conv2-13-0": "t-67-0",
  "t-resnet_conv2-13-1": "t-67-1",
  "t-norm_out-0-0": "t-68-0",
  "t-norm_out-0-1": "t-68-1",
  "t-conv_out-0-0": "t-69-0",
  "t-conv_out-0-1": "t-69-1",
  "t-attn_to_q-0-0": "t-7-0",
  "t-attn_to_q-0-1": "t-7-1",
  "t-attn_to_k-0-0": "t-8-0",
  "t-attn_to_k-0-1": "t-8-1",
  "t-attn_to_v-0-0": "t-9-0",
  "t-attn_to_v-0-1": "t-9-1",
]
