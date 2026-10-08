import NNC

func SeedVR2Conv3DToConv2D<FloatType: TensorNumeric & BinaryFloatingPoint>(
  graph: DynamicGraph, name: String, dataType: DataType, shape: TensorShape,
  store: DynamicGraph.Store, externalData: DynamicGraph.Store.Codec, modelPrefix: String,
  of _: FloatType.Type
) -> DynamicGraph.Store.ModelReaderResult? {
  guard
    name.hasSuffix("-0]") || name.hasSuffix("-1]"),
    name.hasPrefix("__\(modelPrefix)__[t-")
  else {
    return nil
  }
  let convPrefixes = [
    "conv_in", "conv_out", "conv1", "conv2", "conv_shortcut", "conv_up1", "conv_up2",
    "conv_down",
  ]
  guard convPrefixes.contains(where: { name.hasPrefix("__\(modelPrefix)__[t-\($0)-") }) else {
    return nil
  }
  guard let originalTensor = store.read(name, kind: .CPU, codec: [.jit, externalData]) else {
    return .continue(name)
  }
  let tensor = Tensor<FloatType>(from: originalTensor)
  let targetCount = shape.reduce(1, *)
  let sourceCount = tensor.shape.reduce(1, *)

  if shape.count == 4 {
    let kernelHeight = shape[2]
    let kernelWidth = shape[3]
    guard targetCount > 0, kernelHeight > 0, kernelWidth > 0, tensor.shape.count == 5,
      tensor.shape[3] == kernelHeight, tensor.shape[4] == kernelWidth
    else {
      return .continue(name)
    }
    let sourceOutputChannels = tensor.shape[0]
    let sourceInputChannels = tensor.shape[1]
    let targetOutputChannels = shape[0]
    let targetInputChannels = shape[1]
    guard sourceOutputChannels <= targetOutputChannels,
      sourceInputChannels <= targetInputChannels
    else {
      return .continue(name)
    }
    let reduced = graph.variable(tensor.toGPU(0)).reduced(.sum, axis: [2])
    let outTensor: Tensor<FloatType>
    if sourceOutputChannels == targetOutputChannels && sourceInputChannels == targetInputChannels {
      outTensor = reduced.rawValue.toCPU()
    } else {
      var out = graph.variable(
        Tensor<FloatType>(
          Array(repeating: FloatType.zero, count: targetCount), .CPU,
          .NCHW(targetOutputChannels, targetInputChannels, kernelHeight, kernelWidth)
        ).toGPU(0))
      out[
        0..<sourceOutputChannels, 0..<sourceInputChannels, 0..<kernelHeight, 0..<kernelWidth
      ] = reduced
      outTensor = out.rawValue.toCPU()
    }
    if dataType == .Float16 {
      return .final(outTensor)
    } else {
      return .final(Tensor<Float>(from: outTensor))
    }
  }

  if shape.count == 1 && sourceCount < targetCount {
    var out = graph.variable(
      Tensor<FloatType>(Array(repeating: FloatType.zero, count: targetCount), .CPU, .C(targetCount))
        .toGPU(0))
    out[0..<sourceCount] = graph.variable(tensor.toGPU(0))
    let outTensor = out.rawValue.toCPU()
    if dataType == .Float16 {
      return .final(outTensor)
    } else {
      return .final(Tensor<Float>(from: outTensor))
    }
  }

  return .continue(name)
}

private func SeedVR2ResnetBlock3D(
  inChannels: Int, outChannels: Int, depth: Int, height: Int, width: Int
) -> Model {
  let x = Input()
  let norm1 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: "norm1")
  var out = norm1(x)
  out = out.swish()
  let conv1: Model
  if depth == 1 {
    conv1 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv1")
    out = conv1(out)
  } else {
    conv1 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      format: .OIHW, name: "conv1")
    out = conv1(
      out.padded(.replicate, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
        .reshaped([1, depth + 2, height, width, inChannels])
    ).reshaped([depth, height, width, outChannels])
  }
  let norm2 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: "norm2")
  out = norm2(out)
  out = out.swish()
  let conv2: Model
  if depth == 1 {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv2")
    out = conv2(out)
  } else {
    conv2 = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      format: .OIHW, name: "conv2")
    out = conv2(
      out.padded(.replicate, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
        .reshaped([1, depth + 2, height, width, outChannels])
    ).reshaped([depth, height, width, outChannels])
  }
  if inChannels != outChannels {
    let shortcut: Model
    if depth == 1 {
      shortcut = Convolution(
        groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
        format: .OIHW, name: "conv_shortcut")
      out = shortcut(x) + out
    } else {
      shortcut = Convolution(
        groups: 1, filters: outChannels, filterSize: [1, 1, 1], hint: Hint(stride: [1, 1, 1]),
        format: .OIHW, name: "conv_shortcut")
      out =
        shortcut(x.reshaped([1, depth, height, width, inChannels])).reshaped([
          depth, height, width, outChannels,
        ]) + out
    }
  } else {
    out = x + out
  }
  return Model([x], [out])
}

private func SeedVR2AttentionBlock2D(
  channels: Int, batchSize: Int, height: Int, width: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: "group_norm")
  var out = norm(x)
  let hw = height * width
  let toQueries = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: "to_q")
  let toKeys = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: "to_k")
  let toValues = Convolution(
    groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
    name: "to_v")
  if usesFlashAttention {
    let q = toQueries(out).reshaped([batchSize, hw, channels])
    let k = toKeys(out).reshaped([batchSize, hw, channels])
    let v = toValues(out).reshaped([batchSize, hw, channels])
    let projOut = ScaledDotProductAttention(
      scale: 1.0 / Float(channels).squareRoot(), multiHeadOutputProjectionFused: true,
      name: "to_out")
    out = projOut(q, k, v).reshaped([batchSize, height, width, channels])
    out = x + out
  } else {
    let q = ((1.0 / Float(channels).squareRoot()) * toQueries(out)).reshaped([
      batchSize, hw, channels,
    ])
    let k = toKeys(out).reshaped([batchSize, hw, channels])
    let v = toValues(out).reshaped([batchSize, hw, channels])
    var dot = Matmul(transposeB: (1, 2))(q, k)
    dot = dot.reshaped([batchSize * hw, hw])
    dot = dot.softmax()
    dot = dot.reshaped([batchSize, hw, hw])
    out = dot * v
    let toOut = Convolution(
      groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW,
      name: "to_out")
    out = x + toOut(out.reshaped([batchSize, height, width, channels]))
  }
  return Model([x], [out])
}

private func SeedVR2DecoderMidBlock3D(depth: Int, height: Int, width: Int, usesFlashAttention: Bool)
  -> Model
{
  let x = Input()
  let resnet0 = SeedVR2ResnetBlock3D(
    inChannels: 512, outChannels: 512, depth: depth, height: height, width: width)
  var out = resnet0(x)
  let attn = SeedVR2AttentionBlock2D(
    channels: 512, batchSize: depth, height: height, width: width,
    usesFlashAttention: usesFlashAttention)
  out = attn(out)
  let resnet1 = SeedVR2ResnetBlock3D(
    inChannels: 512, outChannels: 512, depth: depth, height: height, width: width)
  out = resnet1(out)
  return Model([x], [out])
}

private func SeedVR2Upsample3D(
  channels: Int, depth: Int, height: Int, width: Int, temporalUp: Bool, spatialUp: Bool
) -> Model {
  let x = Input()
  let temporalRatio = temporalUp ? 2 : 1
  let spatialRatio = spatialUp ? 2 : 1
  let upscaleRatio = temporalRatio * spatialRatio * spatialRatio
  let upscaleConv: Model
  if depth == 1 {
    upscaleConv = Convolution(
      groups: 1, filters: channels * upscaleRatio, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]), format: .OIHW, name: "conv_up1")
  } else {
    upscaleConv = Convolution(
      groups: 1, filters: channels * upscaleRatio, filterSize: [1, 1, 1],
      hint: Hint(stride: [1, 1, 1]), format: .OIHW, name: "conv_up1")
  }
  var out =
    (depth == 1 ? upscaleConv(x) : upscaleConv(x.reshaped([1, depth, height, width, channels])))
    .reshaped([depth, height, width, spatialRatio, spatialRatio, temporalRatio, channels])
    .permuted(0, 5, 1, 3, 2, 4, 6).contiguous()
  let upDepthRaw = depth * temporalRatio
  let upHeight = height * spatialRatio
  let upWidth = width * spatialRatio
  out = out.reshaped([upDepthRaw, upHeight, upWidth, channels])
  if temporalUp {
    let first = out.reshaped(
      [1, upHeight, upWidth, channels],
      strides: [upHeight * upWidth * channels, upWidth * channels, channels, 1]
    ).contiguous()
    if depth == 1 {
      out = first
    } else {
      let rest = out.reshaped(
        [upDepthRaw - 2, upHeight, upWidth, channels], offset: [2, 0, 0, 0],
        strides: [upHeight * upWidth * channels, upWidth * channels, channels, 1]
      )
      .contiguous()
      out = Functional.concat(axis: 0, first, rest)
    }
  }
  let upDepth = temporalUp ? 1 + (depth - 1) * temporalRatio : upDepthRaw
  let conv: Model
  if upDepth == 1 {
    conv = Convolution(
      groups: 1, filters: channels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_up2")
    out = conv(out)
  } else {
    conv = Convolution(
      groups: 1, filters: channels, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      format: .OIHW, name: "conv_up2")
    out = conv(
      out.padded(.replicate, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
        .reshaped([1, upDepth + 2, upHeight, upWidth, channels])
    ).reshaped([upDepth, upHeight, upWidth, channels])
  }
  return Model([x], [out])
}

private func SeedVR2UpDecoderBlock3D(
  inChannels: Int, outChannels: Int, depth: Int, height: Int, width: Int, addUpsample: Bool,
  temporalUp: Bool, spatialUp: Bool
) -> Model {
  let x = Input()
  let resnet0 = SeedVR2ResnetBlock3D(
    inChannels: inChannels, outChannels: outChannels, depth: depth, height: height, width: width)
  var out = resnet0(x)
  let resnet1 = SeedVR2ResnetBlock3D(
    inChannels: outChannels, outChannels: outChannels, depth: depth, height: height, width: width)
  out = resnet1(out)
  let resnet2 = SeedVR2ResnetBlock3D(
    inChannels: outChannels, outChannels: outChannels, depth: depth, height: height, width: width)
  out = resnet2(out)
  if addUpsample {
    let upsample = SeedVR2Upsample3D(
      channels: outChannels, depth: depth, height: height, width: width, temporalUp: temporalUp,
      spatialUp: spatialUp)
    out = upsample(out)
  }
  return Model([x], [out])
}

public func SeedVR2Decoder3D(
  startDepth: Int, startHeight: Int, startWidth: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let convIn: Model
  var out: Model.IO
  if startDepth == 1 {
    convIn = Convolution(
      groups: 1, filters: 512, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_in")
    out = convIn(x)
  } else {
    convIn = Convolution(
      groups: 1, filters: 512, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      format: .OIHW, name: "conv_in")
    out = convIn(
      x.padded(.replicate, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
        .reshaped([1, startDepth + 2, startHeight, startWidth, 16])
    ).reshaped([startDepth, startHeight, startWidth, 512])
  }

  let midBlock = SeedVR2DecoderMidBlock3D(
    depth: startDepth, height: startHeight, width: startWidth,
    usesFlashAttention: usesFlashAttention)
  out = midBlock(out)

  let upBlock0 = SeedVR2UpDecoderBlock3D(
    inChannels: 512, outChannels: 512, depth: startDepth, height: startHeight, width: startWidth,
    addUpsample: true, temporalUp: true, spatialUp: true)
  out = upBlock0(out)
  let depth1 = 1 + (startDepth - 1) * 2
  let height1 = startHeight * 2
  let width1 = startWidth * 2

  let upBlock1 = SeedVR2UpDecoderBlock3D(
    inChannels: 512, outChannels: 512, depth: depth1, height: height1, width: width1,
    addUpsample: true, temporalUp: true, spatialUp: true)
  out = upBlock1(out)
  let depth2 = 1 + (depth1 - 1) * 2
  let height2 = height1 * 2
  let width2 = width1 * 2

  let upBlock2 = SeedVR2UpDecoderBlock3D(
    inChannels: 512, outChannels: 256, depth: depth2, height: height2, width: width2,
    addUpsample: true, temporalUp: false, spatialUp: true)
  out = upBlock2(out)
  let depth3 = depth2
  let height3 = height2 * 2
  let width3 = width2 * 2

  let upBlock3 = SeedVR2UpDecoderBlock3D(
    inChannels: 256, outChannels: 128, depth: depth3, height: height3, width: width3,
    addUpsample: false, temporalUp: false, spatialUp: false)
  out = upBlock3(out)

  let convNormOut = GroupNorm(
    axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: "conv_norm_out")
  out = convNormOut(out).swish()
  let convOut: Model
  if depth3 == 1 {
    convOut = Convolution(
      groups: 1, filters: 4, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_out")
    out = convOut(out)
  } else {
    convOut = Convolution(
      groups: 1, filters: 4, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      format: .OIHW, name: "conv_out")
    out = convOut(
      out.padded(.replicate, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
        .reshaped([1, depth3 + 2, height3, width3, 128])
    ).reshaped([depth3, height3, width3, 4])
  }

  return Model([x], [out])
}

private func SeedVR2Downsample3D(
  channels: Int, temporalDown: Bool, spatialDown: Bool, depth: Int, height: Int, width: Int
) -> Model {
  let x = Input()
  let temporalRatio = temporalDown ? 2 : 1
  let spatialRatio = spatialDown ? 2 : 1
  let temporalKernel = temporalDown ? 3 : 1
  let spatialKernel = spatialDown ? 3 : 1
  var out: Model.IO = x
  var inputHeight = height
  var inputWidth = width
  if spatialDown {
    out = out.padded(.zero, begin: [0, 0, 0, 0], end: [0, 1, 1, 0])
    inputHeight += 1
    inputWidth += 1
  }
  let conv: Model
  if depth == 1 {
    conv = Convolution(
      groups: 1, filters: channels, filterSize: [spatialKernel, spatialKernel],
      hint: Hint(stride: [spatialRatio, spatialRatio]), format: .OIHW, name: "conv_down")
    out = conv(out)
  } else if temporalDown {
    conv = Convolution(
      groups: 1, filters: channels, filterSize: [temporalKernel, spatialKernel, spatialKernel],
      hint: Hint(stride: [temporalRatio, spatialRatio, spatialRatio]), format: .OIHW,
      name: "conv_down")
    out = conv(
      out.padded(.replicate, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
        .reshaped([1, depth + 2, inputHeight, inputWidth, channels]))
  } else {
    conv = Convolution(
      groups: 1, filters: channels, filterSize: [temporalKernel, spatialKernel, spatialKernel],
      hint: Hint(stride: [temporalRatio, spatialRatio, spatialRatio]), format: .OIHW,
      name: "conv_down")
    out = conv(out.reshaped([1, depth, inputHeight, inputWidth, channels]))
  }
  let outputDepth = temporalDown ? (depth + 1) / 2 : depth
  let outputHeight = spatialDown ? height / 2 : height
  let outputWidth = spatialDown ? width / 2 : width
  return Model([x], [out.reshaped([outputDepth, outputHeight, outputWidth, channels])])
}

private func SeedVR2DownEncoderBlock3D(
  inChannels: Int, outChannels: Int, addDownsample: Bool, temporalDown: Bool, spatialDown: Bool,
  depth: Int, height: Int, width: Int
) -> Model {
  let x = Input()
  let resnet0 = SeedVR2ResnetBlock3D(
    inChannels: inChannels, outChannels: outChannels, depth: depth, height: height, width: width)
  var out = resnet0(x)
  let resnet1 = SeedVR2ResnetBlock3D(
    inChannels: outChannels, outChannels: outChannels, depth: depth, height: height, width: width)
  out = resnet1(out)
  if addDownsample {
    let downsample = SeedVR2Downsample3D(
      channels: outChannels, temporalDown: temporalDown, spatialDown: spatialDown, depth: depth,
      height: height, width: width)
    out = downsample(out)
  }
  return Model([x], [out])
}

private func SeedVR2EncoderMidBlock3D(depth: Int, height: Int, width: Int, usesFlashAttention: Bool)
  -> Model
{
  let x = Input()
  let resnet0 = SeedVR2ResnetBlock3D(
    inChannels: 512, outChannels: 512, depth: depth, height: height, width: width)
  var out = resnet0(x)
  let attn = SeedVR2AttentionBlock2D(
    channels: 512, batchSize: depth, height: height, width: width,
    usesFlashAttention: usesFlashAttention)
  out = attn(out)
  let resnet1 = SeedVR2ResnetBlock3D(
    inChannels: 512, outChannels: 512, depth: depth, height: height, width: width)
  out = resnet1(out)
  return Model([x], [out])
}

public func SeedVR2Encoder3D(
  startDepth: Int, startHeight: Int, startWidth: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let convIn: Model
  var out: Model.IO
  if startDepth == 1 {
    convIn = Convolution(
      groups: 1, filters: 128, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_in")
    out = convIn(x)
  } else {
    convIn = Convolution(
      groups: 1, filters: 128, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      format: .OIHW, name: "conv_in")
    out = convIn(
      x.padded(.replicate, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
        .reshaped([1, startDepth + 2, startHeight, startWidth, 3])
    ).reshaped([startDepth, startHeight, startWidth, 128])
  }

  let downBlock0 = SeedVR2DownEncoderBlock3D(
    inChannels: 128, outChannels: 128, addDownsample: true, temporalDown: false,
    spatialDown: true, depth: startDepth, height: startHeight, width: startWidth)
  out = downBlock0(out)
  let depth1 = startDepth
  let height1 = startHeight / 2
  let width1 = startWidth / 2

  let downBlock1 = SeedVR2DownEncoderBlock3D(
    inChannels: 128, outChannels: 256, addDownsample: true, temporalDown: true,
    spatialDown: true, depth: depth1, height: height1, width: width1)
  out = downBlock1(out)
  let depth2 = (depth1 + 1) / 2
  let height2 = height1 / 2
  let width2 = width1 / 2

  let downBlock2 = SeedVR2DownEncoderBlock3D(
    inChannels: 256, outChannels: 512, addDownsample: true, temporalDown: true,
    spatialDown: true, depth: depth2, height: height2, width: width2)
  out = downBlock2(out)
  let depth3 = (depth2 + 1) / 2
  let height3 = height2 / 2
  let width3 = width2 / 2

  let downBlock3 = SeedVR2DownEncoderBlock3D(
    inChannels: 512, outChannels: 512, addDownsample: false, temporalDown: false,
    spatialDown: false, depth: depth3, height: height3, width: width3)
  out = downBlock3(out)

  let midBlock = SeedVR2EncoderMidBlock3D(
    depth: depth3, height: height3, width: width3, usesFlashAttention: usesFlashAttention)
  out = midBlock(out)

  let convNormOut = GroupNorm(
    axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2], name: "conv_norm_out")
  out = convNormOut(out).swish()
  let convOut: Model
  if depth3 == 1 {
    convOut = Convolution(
      groups: 1, filters: 32, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
      format: .OIHW, name: "conv_out")
    out = convOut(out)
  } else {
    convOut = Convolution(
      groups: 1, filters: 32, filterSize: [3, 3, 3],
      hint: Hint(stride: [1, 1, 1], border: Hint.Border(begin: [0, 1, 1], end: [0, 1, 1])),
      format: .OIHW, name: "conv_out")
    out = convOut(
      out.padded(.replicate, begin: [2, 0, 0, 0], end: [0, 0, 0, 0])
        .reshaped([1, depth3 + 2, height3, width3, 512])
    ).reshaped([depth3, height3, width3, 32])
  }

  return Model([x], [out])
}
