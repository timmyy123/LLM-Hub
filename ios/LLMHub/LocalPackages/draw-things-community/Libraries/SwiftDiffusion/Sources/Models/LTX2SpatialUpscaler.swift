import DiffusionMappings
import NNC

public enum LTX2SpatialUpscalerMode {
  case x2
  case x1_5
}

private func NCHWLTX2SpatialResBlock3D(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    name: "conv1")
  var out = conv1(x.reshaped([1, channels, depth, height, width])).reshaped([
    channels, depth, height, width,
  ])
  let norm1 = GroupNorm(axis: 0, groups: 32, epsilon: 1e-5, reduce: [1, 2, 3], name: "norm1")
  out = norm1(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    name: "conv2")
  out = conv2(out.reshaped([1, channels, depth, height, width])).reshaped([
    channels, depth, height, width,
  ])
  let norm2 = GroupNorm(axis: 0, groups: 32, epsilon: 1e-5, reduce: [1, 2, 3], name: "norm2")
  out = norm2(out)
  out = (out + x).swish()
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).conv1.weight"] = [conv1.weight.name]
    mapping["\(prefix).conv1.bias"] = [conv1.bias.name]
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm1.bias"] = [norm1.bias.name]
    mapping["\(prefix).conv2.weight"] = [conv2.weight.name]
    mapping["\(prefix).conv2.bias"] = [conv2.bias.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).norm2.bias"] = [norm2.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func NHWCLTX2SpatialResBlock3D(
  prefix: String, channels: Int, depth: Int, height: Int, width: Int
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let conv1 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    format: .OIHW, name: "conv1")
  var out = conv1(x.reshaped([1, depth, height, width, channels])).reshaped([
    depth, height, width, channels,
  ])
  let norm1 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [0, 1, 2], name: "norm1")
  out = norm1(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    format: .OIHW, name: "conv2")
  out = conv2(out.reshaped([1, depth, height, width, channels])).reshaped([
    depth, height, width, channels,
  ])
  let norm2 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [0, 1, 2], name: "norm2")
  out = (norm2(out) + x).swish()
  let mapper: ModelWeightMapper = { _ in
    var mapping = ModelWeightMapping()
    mapping["\(prefix).conv1.weight"] = [conv1.weight.name]
    mapping["\(prefix).conv1.bias"] = [conv1.bias.name]
    mapping["\(prefix).norm1.weight"] = [norm1.weight.name]
    mapping["\(prefix).norm1.bias"] = [norm1.bias.name]
    mapping["\(prefix).conv2.weight"] = [conv2.weight.name]
    mapping["\(prefix).conv2.bias"] = [conv2.bias.name]
    mapping["\(prefix).norm2.weight"] = [norm2.weight.name]
    mapping["\(prefix).norm2.bias"] = [norm2.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func NCHWLTX2SpatialUpscaler3D(
  inChannels: Int, midChannels: Int, numBlocks: Int, depth: Int, height: Int, width: Int,
  mode: LTX2SpatialUpscalerMode
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let initialConv = Convolution(
    groups: 1, filters: midChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    name: "initial_conv")
  var out = initialConv(
    x.permuted(3, 0, 1, 2).contiguous().reshaped(
      [1, inChannels, depth, height, width], format: .NCHW)
  ).reshaped([
    midChannels, depth, height, width,
  ])
  let initialNorm = GroupNorm(
    axis: 0, groups: 32, epsilon: 1e-5, reduce: [1, 2, 3], name: "initial_norm")
  out = initialNorm(out)
  out = out.swish()
  var mappers = [ModelWeightMapper]()
  for i in 0..<numBlocks {
    let (mapper, block) = NCHWLTX2SpatialResBlock3D(
      prefix: "res_blocks.\(i)", channels: midChannels, depth: depth, height: height,
      width: width)
    out = block(out)
    mappers.append(mapper)
  }

  let upsampleConv: Convolution
  let upsampleWeightKey: String
  let upsampleBiasKey: String
  let postHeight: Int
  let postWidth: Int
  let blurDown: Convolution?
  switch mode {
  case .x2:
    upsampleConv = Convolution(
      groups: 1, filters: midChannels * 4, filterSize: [1, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      name: "upsampler_conv")
    out = upsampleConv(out.reshaped([
      1, midChannels, depth, height, width,
    ]))
    out = out.reshaped([midChannels, 2, 2, depth, height, width]).permuted(
      0, 3, 4, 1, 5, 2
    ).contiguous()
    out = out.reshaped([midChannels, depth, height * 2, width * 2])
    upsampleWeightKey = "upsampler.0.weight"
    upsampleBiasKey = "upsampler.0.bias"
    postHeight = height * 2
    postWidth = width * 2
    blurDown = nil
  case .x1_5:
    precondition(height % 2 == 0 && width % 2 == 0)
    upsampleConv = Convolution(
      groups: 1, filters: midChannels * 9, filterSize: [1, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      name: "upsampler_conv")
    out = upsampleConv(out.reshaped([
      1, midChannels, depth, height, width,
    ]))
    out = out.reshaped([midChannels, 3, 3, depth, height, width]).permuted(
      0, 3, 4, 1, 5, 2
    ).contiguous()
    out = out.reshaped([midChannels, depth, height * 3, width * 3])
    let blur = Convolution(
      groups: midChannels, filters: midChannels, filterSize: [1, 5, 5], noBias: true,
      hint: Hint(stride: [1, 2, 2], border: Hint.Border(begin: [0, 2, 2], end: [0, 2, 2])),
      name: "upsampler_blur_down")
    out = blur(out.reshaped([
      1, midChannels, depth, height * 3, width * 3,
    ])).reshaped([
      midChannels, depth, height * 3 / 2, width * 3 / 2,
    ])
    upsampleWeightKey = "upsampler.conv.weight"
    upsampleBiasKey = "upsampler.conv.bias"
    postHeight = height * 3 / 2
    postWidth = width * 3 / 2
    blurDown = blur
  }

  for i in 0..<numBlocks {
    let (mapper, block) = NCHWLTX2SpatialResBlock3D(
      prefix: "post_upsample_res_blocks.\(i)", channels: midChannels, depth: depth,
      height: postHeight, width: postWidth)
    out = block(out)
    mappers.append(mapper)
  }
  let finalConv = Convolution(
    groups: 1, filters: inChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    name: "final_conv")
  out = finalConv(out.reshaped([1, midChannels, depth, postHeight, postWidth])).reshaped([
    inChannels, depth, postHeight, postWidth,
  ]).permuted(1, 2, 3, 0).contiguous().reshaped(.NHWC(depth, postHeight, postWidth, inChannels))

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["initial_conv.weight"] = [initialConv.weight.name]
    mapping["initial_conv.bias"] = [initialConv.bias.name]
    mapping["initial_norm.weight"] = [initialNorm.weight.name]
    mapping["initial_norm.bias"] = [initialNorm.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping[upsampleWeightKey] = [upsampleConv.weight.name]
    mapping[upsampleBiasKey] = [upsampleConv.bias.name]
    if let blurDown = blurDown {
      mapping["upsampler.blur_down.kernel"] = [blurDown.weight.name]
    }
    mapping["final_conv.weight"] = [finalConv.weight.name]
    mapping["final_conv.bias"] = [finalConv.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

private func NHWCLTX2SpatialUpscaler3D(
  inChannels: Int, midChannels: Int, numBlocks: Int, depth: Int, height: Int, width: Int,
  mode: LTX2SpatialUpscalerMode
) -> (ModelWeightMapper, Model) {
  let x = Input()
  let initialConv = Convolution(
    groups: 1, filters: midChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    format: .OIHW, name: "initial_conv")
  var out = initialConv(x.reshaped([1, depth, height, width, inChannels])).reshaped([
    depth, height, width, midChannels,
  ])
  let initialNorm = GroupNorm(
    axis: 3, groups: 32, epsilon: 1e-5, reduce: [0, 1, 2], name: "initial_norm")
  out = initialNorm(out)
  out = out.swish()
  var mappers = [ModelWeightMapper]()
  for i in 0..<numBlocks {
    let (mapper, block) = NHWCLTX2SpatialResBlock3D(
      prefix: "res_blocks.\(i)", channels: midChannels, depth: depth, height: height,
      width: width)
    out = block(out)
    mappers.append(mapper)
  }

  let upsampleConv: Convolution
  let upsampleWeightKey: String
  let upsampleBiasKey: String
  let postHeight: Int
  let postWidth: Int
  let blurDown: Convolution?
  switch mode {
  case .x2:
    upsampleConv = Convolution(
      groups: 1, filters: midChannels * 4, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "upsampler_conv")
    out = upsampleConv(out).reshaped([
      depth, height, width, midChannels, 2, 2,
    ]).permuted(
      0, 1, 4, 2, 5, 3
    ).contiguous()
    out = out.reshaped([depth, height * 2, width * 2, midChannels])
    upsampleWeightKey = "upsampler.0.weight"
    upsampleBiasKey = "upsampler.0.bias"
    postHeight = height * 2
    postWidth = width * 2
    blurDown = nil
  case .x1_5:
    precondition(height % 2 == 0 && width % 2 == 0)
    upsampleConv = Convolution(
      groups: 1, filters: midChannels * 9, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "upsampler_conv")
    out = upsampleConv(out).reshaped([
      depth, height, width, midChannels, 3, 3,
    ]).permuted(
      0, 1, 4, 2, 5, 3
    ).contiguous()
    out = out.reshaped([depth, height * 3, width * 3, midChannels])
    let blur = Convolution(
      groups: midChannels, filters: midChannels, filterSize: [5, 5], noBias: true,
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [2, 2])),
      format: .OIHW, name: "upsampler_blur_down")
    out = blur(out).reshaped([
      depth, height * 3 / 2, width * 3 / 2, midChannels,
    ])
    upsampleWeightKey = "upsampler.conv.weight"
    upsampleBiasKey = "upsampler.conv.bias"
    postHeight = height * 3 / 2
    postWidth = width * 3 / 2
    blurDown = blur
  }

  for i in 0..<numBlocks {
    let (mapper, block) = NHWCLTX2SpatialResBlock3D(
      prefix: "post_upsample_res_blocks.\(i)", channels: midChannels, depth: depth,
      height: postHeight, width: postWidth)
    out = block(out)
    mappers.append(mapper)
  }
  let finalConv = Convolution(
    groups: 1, filters: inChannels, filterSize: [3, 3, 3],
    hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [1, 1, 1], end: [1, 1, 1])),
    format: .OIHW, name: "final_conv")
  out = finalConv(out.reshaped([1, depth, postHeight, postWidth, midChannels])).reshaped([
    depth, postHeight, postWidth, inChannels,
  ])

  let mapper: ModelWeightMapper = { format in
    var mapping = ModelWeightMapping()
    mapping["initial_conv.weight"] = [initialConv.weight.name]
    mapping["initial_conv.bias"] = [initialConv.bias.name]
    mapping["initial_norm.weight"] = [initialNorm.weight.name]
    mapping["initial_norm.bias"] = [initialNorm.bias.name]
    for mapper in mappers {
      mapping.merge(mapper(format)) { v, _ in v }
    }
    mapping[upsampleWeightKey] = [upsampleConv.weight.name]
    mapping[upsampleBiasKey] = [upsampleConv.bias.name]
    if let blurDown = blurDown {
      mapping["upsampler.blur_down.kernel"] = [blurDown.weight.name]
    }
    mapping["final_conv.weight"] = [finalConv.weight.name]
    mapping["final_conv.bias"] = [finalConv.bias.name]
    return mapping
  }
  return (mapper, Model([x], [out]))
}

public func LTX2SpatialUpscaler3D(
  inChannels: Int, midChannels: Int, numBlocks: Int, depth: Int, height: Int, width: Int,
  mode: LTX2SpatialUpscalerMode, format: TensorFormat
) -> (ModelWeightMapper, Model) {
  switch format {
  case .NHWC:
    return NHWCLTX2SpatialUpscaler3D(
      inChannels: inChannels, midChannels: midChannels, numBlocks: numBlocks, depth: depth,
      height: height, width: width, mode: mode)
  case .NCHW:
    return NCHWLTX2SpatialUpscaler3D(
      inChannels: inChannels, midChannels: midChannels, numBlocks: numBlocks, depth: depth,
      height: height, width: width, mode: mode)
  case .CHWN:
    fatalError()
  }
}
