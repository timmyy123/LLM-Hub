import DiffusionMappings
import NNC

private func NHWCResnetBlockCausal3D(
  prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool, depth: Int, height: Int,
  width: Int
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let norm1 = GroupNorm(
    axis: 3, groups: 32, epsilon: 1e-6, reduce: [0, 1, 2], name: "resnet_norm1")
  let norm1X = norm1(x)
  var out = norm1X
  out = Swish()(out)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "resnet_conv1")
  out = conv1(
    out.padded(
      .replicate, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]
    ).reshaped([1, depth + 2, height + 2, width + 2, inChannels])
  ).reshaped([depth, height, width, outChannels])
  let norm2 = GroupNorm(
    axis: 3, groups: 32, epsilon: 1e-6, reduce: [0, 1, 2], name: "resnet_norm2")
  out = norm2(out)
  out = Swish()(out)
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "resnet_conv2")
  out = conv2(
    out.padded(
      .replicate, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]
    ).reshaped([1, depth + 2, height + 2, width + 2, outChannels])
  ).reshaped([depth, height, width, outChannels])
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "resnet_shortcut")
    let ninX = nin(x)
    out = ninX + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm1.bias"] = [norm1.bias.name]
    mapping["\(prefix).conv1.conv.weight"] = [conv1.weight.name]
    mapping["\(prefix).conv1.conv.bias"] = [conv1.bias.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).norm2.bias"] = [norm2.bias.name]
    mapping["\(prefix).conv2.conv.weight"] = [conv2.weight.name]
    mapping["\(prefix).conv2.conv.bias"] = [conv2.bias.name]
    if let ninShortcut = ninShortcut {
      mapping["\(prefix).conv_shortcut.conv.weight"] = [ninShortcut.weight.name]
      mapping["\(prefix).conv_shortcut.conv.bias"] = [ninShortcut.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func NHWCAttnBlockCausal3D(
  prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let causalAttentionMask = Input()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [0, 1, 2], name: "attn_norm")
  var out = norm(x)
  let hw = width * height * depth
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW, name: "to_k")
  let k = tokeys(out).reshaped([1, hw, inChannels])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW, name: "to_q")
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    1, hw, inChannels,
  ])
  var dot =
    Matmul(transposeB: (1, 2))(q, k).reshaped([
      depth, height * width, depth, height * width,
    ]) + causalAttentionMask
  dot = dot.reshaped([hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([1, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW, name: "to_v")
  let v = tovalues(out).reshaped([1, hw, inChannels])
  out = dot * v
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW, name: "proj_out")
  out = x + projOut(out.reshaped([depth, height, width, inChannels]))
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).group_norm.weight"] = [norm.weight.name]
    mapping["\(prefix).group_norm.bias"] = [norm.bias.name]
    mapping["\(prefix).to_k.weight"] = [tokeys.weight.name]
    mapping["\(prefix).to_k.bias"] = [tokeys.bias.name]
    mapping["\(prefix).to_q.weight"] = [toqueries.weight.name]
    mapping["\(prefix).to_q.bias"] = [toqueries.bias.name]
    mapping["\(prefix).to_v.weight"] = [tovalues.weight.name]
    mapping["\(prefix).to_v.bias"] = [tovalues.bias.name]
    mapping["\(prefix).to_out.0.weight"] = [projOut.weight.name]
    mapping["\(prefix).to_out.0.bias"] = [projOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, causalAttentionMask], [out]))
}

private func NHWCEncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let causalAttentionMask = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_in")
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for i in 1..<channels.count {
    height *= 2
    width *= 2
    if i > 1 {
      depth = (depth - 1) * 2 + 1
    }
  }
  var out = convIn(
    x.padded(
      .replicate, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]
    ).reshaped([1, depth + 2, height + 2, width + 2, 3])
  ).reshaped([depth, height, width, previousChannel])
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (mapper, block) = NHWCResnetBlockCausal3D(
        prefix: "encoder.down_blocks.\(i).resnets.\(j)", inChannels: previousChannel,
        outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      // Conv always pad left first, then right, and pad top first then bottom.
      // Thus, we cannot have (0, 1, 0, 1) (left 0, right 1, top 0, bottom 1) padding as in
      // Stable Diffusion. Instead, we pad to (2, 1, 2, 1) and simply discard the first row and first column.
      let oldHeight = height
      height /= 2
      let oldWidth = width
      width /= 2
      let oldDepth = depth
      let strideZ: Int
      if i > 0 {
        depth = (depth - 1) / 2 + 1
        strideZ = 2
      } else {
        strideZ = 1
      }
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3, 3],
        hint: Hint(
          stride: [strideZ, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        format: .OIHW, name: "downsample")
      out = conv2d(
        out.padded(.replicate, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
          1, oldDepth + 2, oldHeight + 2, oldWidth + 2, channel,
        ])
      ).reshaped([depth, height, width, channel])
      let downLayer = i
      let mapper: ModelWeightMapper = { _ in
        var mapping = ModelWeightMapping()
        mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.conv.weight"] = [
          conv2d.weight.name
        ]
        mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.conv.bias"] = [
          conv2d.bias.name
        ]
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let (midBlockMapper1, midBlock1) = NHWCResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock1(out)
  let (midAttnMapper1, midAttn1) = NHWCAttnBlockCausal3D(
    prefix: "encoder.mid_block.attentions.0", inChannels: previousChannel, depth: depth,
    height: height, width: width)
  out = midAttn1(out, causalAttentionMask)
  let (midBlockMapper2, midBlock2) = NHWCResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [0, 1, 2], name: "norm_out")
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_out")
  out = convOut(
    out.padded(.replicate, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]).reshaped([
      1, depth + 2, height + 2, width + 2, previousChannel,
    ])
  ).reshaped([depth, height, width, 32])
  let quantConv = Convolution(
    groups: 1, filters: 32, filterSize: [1, 1], format: .OIHW, name: "quant_conv")
  out = quantConv(out)
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["encoder.conv_in.conv.weight"] = [convIn.weight.name]
    mapping["encoder.conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    mapping["encoder.conv_norm_out.weight"] = [normOut.weight.name]
    mapping["encoder.conv_norm_out.bias"] = [normOut.bias.name]
    mapping["encoder.conv_out.conv.weight"] = [convOut.weight.name]
    mapping["encoder.conv_out.conv.bias"] = [convOut.bias.name]
    mapping["quant_conv.weight"] = [quantConv.weight.name]
    mapping["quant_conv.bias"] = [quantConv.bias.name]
    return mapping
  }
  return (mapper, Model([x, causalAttentionMask], [out]))
}

private func NHWCDecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int, paddingFinalConvLayer: Bool
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let causalAttentionMask = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1], format: .OIHW, name: "post_quant_conv")
  var out = postQuantConv(x)
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_in")
  out = convIn(
    out.padded(
      .replicate, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]
    ).reshaped([1, startDepth + 2, startHeight + 2, startWidth + 2, 16])
  ).reshaped([startDepth, startHeight, startWidth, previousChannel])
  let (midBlockMapper1, midBlock1) = NHWCResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock1(out)
  let (midAttnMapper1, midAttn1) = NHWCAttnBlockCausal3D(
    prefix: "decoder.mid_block.attentions.0", inChannels: previousChannel, depth: startDepth,
    height: startHeight, width: startWidth)
  out = midAttn1(out, causalAttentionMask)
  let (midBlockMapper2, midBlock2) = NHWCResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock2(out)
  var width = startWidth
  var height = startHeight
  var depth = startDepth
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (mapper, block) = NHWCResnetBlockCausal3D(
        prefix: "decoder.up_blocks.\(channels.count - 1 - i).resnets.\(j)",
        inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      width *= 2
      height *= 2
      if i < channels.count - 1 && depth > 1 {  // Scale time too.
        let first = out.reshaped(
          [1, height * width, channel], strides: [height * width * channel, channel, 1]
        ).contiguous()
        let more = out.reshaped(
          [(depth - 1), 1, height * width, channel], offset: [1, 0, 0, 0],
          strides: [height * width * channel, height * width * channel, channel, 1]
        ).contiguous()
        out = Functional.concat(
          axis: 0, first,
          Functional.concat(axis: 1, more, more).reshaped([
            (depth - 1) * 2, height * width, channel,
          ]))
        depth = 1 + (depth - 1) * 2
        out = out.reshaped([depth, height, width, channel])
      }
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        format: .OIHW, name: "upsample")
      out = conv2d(
        out.padded(
          .replicate, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]
        ).reshaped([1, depth + 2, height + 2, width + 2, channel])
      ).reshaped([depth, height, width, channel])
      let upLayer = channels.count - 1 - i
      let mapper: ModelWeightMapper = { _ in
        var mapping = ModelWeightMapping()
        mapping["decoder.up_blocks.\(upLayer).upsamplers.0.conv.conv.weight"] = [conv2d.weight.name]
        mapping["decoder.up_blocks.\(upLayer).upsamplers.0.conv.conv.bias"] = [conv2d.bias.name]
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let normOut = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [0, 1, 2], name: "norm_out")
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 4 : 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_out")
  out = convOut(
    out.padded(
      .replicate, begin: [2, 1, 1, 0], end: [0, 1, 1, 0]
    ).reshaped([1, depth + 2, height + 2, width + 2, channels[0]])
  ).reshaped([
    depth, height, width, paddingFinalConvLayer ? 4 : 3,
  ])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["post_quant_conv.weight"] = [postQuantConv.weight.name]
    mapping["post_quant_conv.bias"] = [postQuantConv.bias.name]
    mapping["decoder.conv_in.conv.weight"] = [convIn.weight.name]
    mapping["decoder.conv_in.conv.bias"] = [convIn.bias.name]
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["decoder.conv_norm_out.weight"] = [normOut.weight.name]
    mapping["decoder.conv_norm_out.bias"] = [normOut.bias.name]
    mapping["decoder.conv_out.conv.weight"] = [convOut.weight.name]
    mapping["decoder.conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, causalAttentionMask], [out]))
}

private func NCHWResnetBlockCausal3D(
  prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool, depth: Int, height: Int,
  width: Int
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let norm1 = GroupNorm(
    axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3], name: "resnet_norm1")
  let norm1X = norm1(x)
  var out = norm1X
  out = Swish()(out)
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "resnet_conv1")
  out = conv1(
    out.padded(
      .replicate, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]
    ).reshaped([1, inChannels, depth + 2, height + 2, width + 2])
  ).reshaped([outChannels, depth, height, width])
  let norm2 = GroupNorm(
    axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3], name: "resnet_norm2")
  out = norm2(out)
  out = Swish()(out)
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "resnet_conv2")
  out = conv2(
    out.padded(
      .replicate, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]
    ).reshaped([1, outChannels, depth + 2, height + 2, width + 2])
  ).reshaped([outChannels, depth, height, width])
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
      format: .OIHW, name: "resnet_shortcut")
    let ninX = nin(x.reshaped([1, inChannels, depth, height, width])).reshaped([
      outChannels, depth, height, width,
    ])
    out = ninX + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm1.bias"] = [norm1.bias.name]
    mapping["\(prefix).conv1.conv.weight"] = [conv1.weight.name]
    mapping["\(prefix).conv1.conv.bias"] = [conv1.bias.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).norm2.bias"] = [norm2.bias.name]
    mapping["\(prefix).conv2.conv.weight"] = [conv2.weight.name]
    mapping["\(prefix).conv2.conv.bias"] = [conv2.bias.name]
    if let ninShortcut = ninShortcut {
      mapping["\(prefix).conv_shortcut.conv.weight"] = [ninShortcut.weight.name]
      mapping["\(prefix).conv_shortcut.conv.bias"] = [ninShortcut.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func NCHWAttnBlockCausal3D(
  prefix: String, inChannels: Int, depth: Int, height: Int, width: Int
) -> (
  ModelWeightMapper, Model
) {
  let x = Input()
  let causalAttentionMask = Input()
  let norm = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3], name: "attn_norm")
  var out = norm(x).reshaped([1, inChannels, depth, height, width])
  let hw = width * height * depth
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    format: .OIHW, name: "to_k")
  let k = tokeys(out).reshaped([1, inChannels, hw])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    format: .OIHW, name: "to_q")
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    1, inChannels, hw,
  ])
  var dot =
    Matmul(transposeA: (1, 2))(q, k).reshaped([
      depth, height * width, depth, height * width,
    ]) + causalAttentionMask
  dot = dot.reshaped([hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([1, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    format: .OIHW, name: "to_v")
  let v = tovalues(out).reshaped([1, inChannels, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
    format: .OIHW, name: "proj_out")
  out =
    x
    + projOut(out.reshaped([1, inChannels, depth, height, width])).reshaped([
      inChannels, depth, height, width,
    ])
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).group_norm.weight"] = [norm.weight.name]
    mapping["\(prefix).group_norm.bias"] = [norm.bias.name]
    mapping["\(prefix).to_k.weight"] = [tokeys.weight.name]
    mapping["\(prefix).to_k.bias"] = [tokeys.bias.name]
    mapping["\(prefix).to_q.weight"] = [toqueries.weight.name]
    mapping["\(prefix).to_q.bias"] = [toqueries.bias.name]
    mapping["\(prefix).to_v.weight"] = [tovalues.weight.name]
    mapping["\(prefix).to_v.bias"] = [tovalues.bias.name]
    mapping["\(prefix).to_out.0.weight"] = [projOut.weight.name]
    mapping["\(prefix).to_out.0.bias"] = [projOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, causalAttentionMask], [out]))
}

private func NCHWEncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let causalAttentionMask = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_in")
  var height = startHeight
  var width = startWidth
  var depth = startDepth
  for i in 1..<channels.count {
    height *= 2
    width *= 2
    if i > 1 {
      depth = (depth - 1) * 2 + 1
    }
  }
  var out = convIn(
    x.permuted(3, 0, 1, 2).contiguous().reshaped(.NCHW(3, depth, height, width)).padded(
      .replicate, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]
    ).reshaped([1, 3, depth + 2, height + 2, width + 2])
  ).reshaped([previousChannel, depth, height, width])
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (mapper, block) = NCHWResnetBlockCausal3D(
        prefix: "encoder.down_blocks.\(i).resnets.\(j)", inChannels: previousChannel,
        outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      // Conv always pad left first, then right, and pad top first then bottom.
      // Thus, we cannot have (0, 1, 0, 1) (left 0, right 1, top 0, bottom 1) padding as in
      // Stable Diffusion. Instead, we pad to (2, 1, 2, 1) and simply discard the first row and first column.
      let oldHeight = height
      height /= 2
      let oldWidth = width
      width /= 2
      let oldDepth = depth
      let strideZ: Int
      if i > 0 {
        depth = (depth - 1) / 2 + 1
        strideZ = 2
      } else {
        strideZ = 1
      }
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3, 3],
        hint: Hint(
          stride: [strideZ, 2, 2], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        format: .OIHW, name: "downsample")
      out = conv2d(
        out.padded(.replicate, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
          1, channel, oldDepth + 2, oldHeight + 2, oldWidth + 2,
        ])
      ).reshaped([channel, depth, height, width])
      let downLayer = i
      let mapper: ModelWeightMapper = { _ in
        var mapping = ModelWeightMapping()
        mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.conv.weight"] = [
          conv2d.weight.name
        ]
        mapping["encoder.down_blocks.\(downLayer).downsamplers.0.conv.conv.bias"] = [
          conv2d.bias.name
        ]
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let (midBlockMapper1, midBlock1) = NCHWResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock1(out)
  let (midAttnMapper1, midAttn1) = NCHWAttnBlockCausal3D(
    prefix: "encoder.mid_block.attentions.0", inChannels: previousChannel, depth: depth,
    height: height, width: width)
  out = midAttn1(out, causalAttentionMask)
  let (midBlockMapper2, midBlock2) = NCHWResnetBlockCausal3D(
    prefix: "encoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel,
    shortcut: false, depth: depth, height: height, width: width)
  out = midBlock2(out)
  let normOut = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3], name: "norm_out")
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_out")
  out = convOut(
    out.padded(.replicate, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]).reshaped([
      1, previousChannel, depth + 2, height + 2, width + 2,
    ])
  ).reshaped([1, 32, depth, height, width])
  let quantConv = Convolution(
    groups: 1, filters: 32, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]), format: .OIHW,
    name: "quant_conv")
  out = quantConv(out).permuted(0, 2, 3, 4, 1).contiguous().reshaped(
    .NHWC(depth, height, width, 32))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["encoder.conv_in.conv.weight"] = [convIn.weight.name]
    mapping["encoder.conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    mapping["encoder.conv_norm_out.weight"] = [normOut.weight.name]
    mapping["encoder.conv_norm_out.bias"] = [normOut.bias.name]
    mapping["encoder.conv_out.conv.weight"] = [convOut.weight.name]
    mapping["encoder.conv_out.conv.bias"] = [convOut.bias.name]
    mapping["quant_conv.weight"] = [quantConv.weight.name]
    mapping["quant_conv.bias"] = [quantConv.bias.name]
    return mapping
  }
  return (mapper, Model([x, causalAttentionMask], [out]))
}

private func NCHWDecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int, paddingFinalConvLayer: Bool
)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  let causalAttentionMask = Input()
  var previousChannel = channels[channels.count - 1]
  let postQuantConv = Convolution(
    groups: 1, filters: 16, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]), format: .OIHW,
    name: "post_quant_conv")
  var out = postQuantConv(
    x.permuted(3, 0, 1, 2).contiguous().reshaped(
      [1, 16, startDepth, startHeight, startWidth], format: .NCHW)
  ).reshaped([16, startDepth, startHeight, startWidth])
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_in")
  out = convIn(
    out.padded(
      .replicate, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]
    ).reshaped([1, 16, startDepth + 2, startHeight + 2, startWidth + 2])
  ).reshaped([previousChannel, startDepth, startHeight, startWidth])
  let (midBlockMapper1, midBlock1) = NCHWResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.0", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock1(out)
  let (midAttnMapper1, midAttn1) = NCHWAttnBlockCausal3D(
    prefix: "decoder.mid_block.attentions.0", inChannels: previousChannel, depth: startDepth,
    height: startHeight, width: startWidth)
  out = midAttn1(out, causalAttentionMask)
  let (midBlockMapper2, midBlock2) = NCHWResnetBlockCausal3D(
    prefix: "decoder.mid_block.resnets.1", inChannels: previousChannel,
    outChannels: previousChannel, shortcut: false, depth: startDepth, height: startHeight,
    width: startWidth)
  out = midBlock2(out)
  var width = startWidth
  var height = startHeight
  var depth = startDepth
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat + 1 {
      let (mapper, block) = NCHWResnetBlockCausal3D(
        prefix: "decoder.up_blocks.\(channels.count - 1 - i).resnets.\(j)",
        inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel, depth: depth, height: height, width: width)
      mappers.append(mapper)
      out = block(out)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      width *= 2
      height *= 2
      if i < channels.count - 1 && depth > 1 {  // Scale time too.
        let first = out.reshaped(
          [channel, 1, height * width], strides: [depth * height * width, height * width, 1]
        ).contiguous()
        let more = out.reshaped(
          [channel, (depth - 1), 1, height * width], offset: [0, 1, 0, 0],
          strides: [depth * height * width, height * width, height * width, 1]
        ).contiguous()
        out = Functional.concat(
          axis: 1, first,
          Functional.concat(axis: 2, more, more).reshaped([
            channel, (depth - 1) * 2, height * width,
          ]))
        depth = 1 + (depth - 1) * 2
        out = out.reshaped([channel, depth, height, width])
      }
      let conv2d = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3, 3],
        hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
        format: .OIHW, name: "upsample")
      out = conv2d(
        out.padded(
          .replicate, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]
        ).reshaped([1, channel, depth + 2, height + 2, width + 2])
      ).reshaped([channel, depth, height, width])
      let upLayer = channels.count - 1 - i
      let mapper: ModelWeightMapper = { _ in
        var mapping = ModelWeightMapping()
        mapping["decoder.up_blocks.\(upLayer).upsamplers.0.conv.conv.weight"] = [conv2d.weight.name]
        mapping["decoder.up_blocks.\(upLayer).upsamplers.0.conv.conv.bias"] = [conv2d.bias.name]
        return mapping
      }
      mappers.append(mapper)
    }
  }
  let normOut = GroupNorm(axis: 0, groups: 32, epsilon: 1e-6, reduce: [1, 2, 3], name: "norm_out")
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: paddingFinalConvLayer ? 4 : 3, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 0, 0], end: [0, 0, 0])),
    format: .OIHW, name: "conv_out")
  out = convOut(
    out.padded(
      .replicate, begin: [0, 2, 1, 1], end: [0, 0, 1, 1]
    ).reshaped([1, channels[0], depth + 2, height + 2, width + 2])
  ).permuted(0, 2, 3, 4, 1).contiguous().reshaped(
    .NHWC(depth, height, width, paddingFinalConvLayer ? 4 : 3)
  )
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["post_quant_conv.weight"] = [postQuantConv.weight.name]
    mapping["post_quant_conv.bias"] = [postQuantConv.bias.name]
    mapping["decoder.conv_in.conv.weight"] = [convIn.weight.name]
    mapping["decoder.conv_in.conv.bias"] = [convIn.bias.name]
    mapping.merge(midBlockMapper1(format)) { v, _ in v }
    mapping.merge(midAttnMapper1(format)) { v, _ in v }
    mapping.merge(midBlockMapper2(format)) { v, _ in v }
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["decoder.conv_norm_out.weight"] = [normOut.weight.name]
    mapping["decoder.conv_norm_out.bias"] = [normOut.bias.name]
    mapping["decoder.conv_out.conv.weight"] = [convOut.weight.name]
    mapping["decoder.conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x, causalAttentionMask], [out]))
}

func EncoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int, startDepth: Int,
  format: TensorFormat
)
  -> (ModelWeightMapper, Model)
{
  switch format {
  case .NHWC:
    NHWCEncoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth)
  case .NCHW:
    NCHWEncoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth)
  case .CHWN:
    fatalError()
  }
}

func DecoderCausal3D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int,
  startDepth: Int, paddingFinalConvLayer: Bool, format: TensorFormat
)
  -> (ModelWeightMapper, Model)
{
  switch format {
  case .NHWC:
    NHWCDecoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth, paddingFinalConvLayer: paddingFinalConvLayer)
  case .NCHW:
    NCHWDecoderCausal3D(
      channels: channels, numRepeat: numRepeat, startWidth: startWidth, startHeight: startHeight,
      startDepth: startDepth, paddingFinalConvLayer: paddingFinalConvLayer)
  case .CHWN:
    fatalError()
  }
}
