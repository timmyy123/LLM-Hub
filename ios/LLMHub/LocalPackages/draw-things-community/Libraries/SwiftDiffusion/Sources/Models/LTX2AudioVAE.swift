import DiffusionMappings
import NNC

private func ResnetBlockCausal2D(prefix: String, inChannels: Int, outChannels: Int, shortcut: Bool)
  -> (
    ModelWeightMapper, Model
  )
{
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-8, axis: [1], elementwiseAffine: false, name: "resnet_norm1")
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    format: .OIHW, name: "resnet_conv1")
  out = conv1(out.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  let norm2 = RMSNorm(epsilon: 1e-8, axis: [1], elementwiseAffine: false, name: "resnet_norm2")
  out = norm2(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    format: .OIHW, name: "resnet_conv2")
  out = conv2(out.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  let ninShortcut: Model?
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "resnet_shortcut")
    out = nin(x) + out
    ninShortcut = nin
  } else {
    ninShortcut = nil
    out = x + out
  }
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).conv1.conv.weight"] = [conv1.weight.name]
    mapping["\(prefix).conv1.conv.bias"] = [conv1.bias.name]
    mapping["\(prefix).conv2.conv.weight"] = [conv2.weight.name]
    mapping["\(prefix).conv2.conv.bias"] = [conv2.bias.name]
    if let ninShortcut = ninShortcut {
      mapping["\(prefix).nin_shortcut.conv.weight"] = [ninShortcut.weight.name]
      mapping["\(prefix).nin_shortcut.conv.bias"] = [ninShortcut.bias.name]
    }
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func LTX2AudioEncoderCausal2D(
  channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int
) -> (ModelWeightMapper, Model) {
  let x = Input()
  var previousChannel = channels[0]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    format: .OIHW, name: "conv_in")
  var out = convIn(x.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  var mappers = [ModelWeightMapper]()
  for (i, channel) in channels.enumerated() {
    for j in 0..<numRepeat {
      let (mapper, block) = ResnetBlockCausal2D(
        prefix: "down.\(i).block.\(j)", inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel)
      out = block(out)
      mappers.append(mapper)
      previousChannel = channel
    }
    if i < channels.count - 1 {
      let conv = Convolution(
        groups: 1, filters: previousChannel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [0, 0], end: [0, 0])),
        format: .OIHW, name: "upsample")
      out = conv(out.padded(.zero, begin: [0, 0, 2, 0], end: [0, 0, 0, 1]))
      let downBlocks = i
      mappers.append { _ in
        var mapping = ModelWeightMapping()
        mapping["down.\(downBlocks).downsample.conv.weight"] = [conv.weight.name]
        mapping["down.\(downBlocks).downsample.conv.bias"] = [conv.bias.name]
        return mapping
      }
    }
  }
  let (midResnetBlock1Mapper, midResnetBlock1) = ResnetBlockCausal2D(
    prefix: "mid.block_1", inChannels: previousChannel, outChannels: previousChannel,
    shortcut: false)
  out = midResnetBlock1(out)
  mappers.append(midResnetBlock1Mapper)
  let (midResnetBlock2Mapper, midResnetBlock2) = ResnetBlockCausal2D(
    prefix: "mid.block_2", inChannels: previousChannel, outChannels: previousChannel,
    shortcut: false)
  out = midResnetBlock2(out)
  mappers.append(midResnetBlock2Mapper)
  let normOut = RMSNorm(epsilon: 1e-8, axis: [1], elementwiseAffine: false, name: "norm_out")
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: 16, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW,
    name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_in.conv.weight"] = [convIn.weight.name]
    mapping["conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_out.conv.weight"] = [convOut.weight.name]
    mapping["conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

func LTX2AudioDecoderCausal2D(channels: [Int], numRepeat: Int, startWidth: Int, startHeight: Int)
  -> (ModelWeightMapper, Model)
{
  let x = Input()
  var previousChannel = channels[channels.count - 1]
  let convIn = Convolution(
    groups: 1, filters: previousChannel, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
    format: .OIHW, name: "conv_in")
  var out = convIn(
    x.reshaped(.NCHW(1, startHeight, 8, startWidth)).transposed(1, 2).padded(
      .zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1]))
  var mappers = [ModelWeightMapper]()
  let (midResnetBlock1Mapper, midResnetBlock1) = ResnetBlockCausal2D(
    prefix: "mid.block_1", inChannels: previousChannel, outChannels: previousChannel,
    shortcut: false)
  out = midResnetBlock1(out)
  mappers.append(midResnetBlock1Mapper)
  let (midResnetBlock2Mapper, midResnetBlock2) = ResnetBlockCausal2D(
    prefix: "mid.block_2", inChannels: previousChannel, outChannels: previousChannel,
    shortcut: false)
  out = midResnetBlock2(out)
  mappers.append(midResnetBlock2Mapper)
  for (i, channel) in channels.enumerated().reversed() {
    for j in 0..<numRepeat {
      let (mapper, block) = ResnetBlockCausal2D(
        prefix: "up.\(i).block.\(j)", inChannels: previousChannel, outChannels: channel,
        shortcut: previousChannel != channel)
      out = block(out)
      mappers.append(mapper)
      previousChannel = channel
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv = Convolution(
        groups: 1, filters: previousChannel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])),
        format: .OIHW, name: "upsample")
      out = conv(out.padded(.zero, begin: [0, 0, 1, 1], end: [0, 0, 0, 1]))
      let upBlocks = i
      mappers.append { _ in
        var mapping = ModelWeightMapping()
        mapping["up.\(upBlocks).upsample.conv.conv.weight"] = [conv.weight.name]
        mapping["up.\(upBlocks).upsample.conv.conv.bias"] = [conv.bias.name]
        return mapping
      }
    }
  }
  let normOut = RMSNorm(epsilon: 1e-8, axis: [1], elementwiseAffine: false, name: "norm_out")
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: 2, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW,
    name: "conv_out")
  out = convOut(out.padded(.zero, begin: [0, 0, 2, 1], end: [0, 0, 0, 1])).transposed(2, 3)
    .reshaped([1, 128, 1, -1])
  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["conv_in.conv.weight"] = [convIn.weight.name]
    mapping["conv_in.conv.bias"] = [convIn.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping["conv_out.conv.weight"] = [convOut.weight.name]
    mapping["conv_out.conv.bias"] = [convOut.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}
