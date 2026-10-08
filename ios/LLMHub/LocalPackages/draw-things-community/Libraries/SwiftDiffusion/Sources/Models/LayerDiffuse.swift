import NNC

private func ResBlock(b: Int, groups: Int, outChannels: Int, skipConnection: Bool) -> (
  Model, Model, Model, Model, Model?, Model
) {
  let x = Input()
  let inLayerNorm = GroupNorm(axis: 3, groups: groups, epsilon: 1e-5, reduce: [1, 2])
  var out = inLayerNorm(x)
  out = Swish()(out)
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = inLayerConv2d(out)
  let outLayerNorm = GroupNorm(axis: 3, groups: groups, epsilon: 1e-5, reduce: [1, 2])
  out = outLayerNorm(out)
  out = Swish()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var skipModel: Model? = nil
  if skipConnection {
    let skip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]), format: .OIHW)
    out = skip(x) + outLayerConv2d(out)  // This layer should be zero init if training.
    skipModel = skip
  } else {
    out = x + outLayerConv2d(out)  // This layer should be zero init if training.
  }
  return (
    inLayerNorm, inLayerConv2d, outLayerNorm, outLayerConv2d, skipModel, Model([x], [out])
  )
}

private func SelfAttention(
  k: Int, h: Int, b: Int, hw: Int, usesFlashAttention: FlashAttentionLevel
)
  -> (
    Model, Model, Model, Model, Model
  )
{
  let x = Input()
  let tokeys = Dense(count: k * h, name: "to_k")
  let toqueries = Dense(count: k * h, name: "to_q")
  let tovalues = Dense(count: k * h, name: "to_v")
  if usesFlashAttention != .none {
    var queries: Model.IO
    if usesFlashAttention == .scale1 {
      queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k]).identity()
    } else {
      queries = toqueries(x).reshaped([b, hw, h, k]).identity().identity()
    }
    let keys = tokeys(x).reshaped([b, hw, h, k]).identity()
    let values = tovalues(x).reshaped([b, hw, h, k])
    let scaledDotProductAttention: ScaledDotProductAttention
    if usesFlashAttention == .scale1 {
      scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1, flags: [.Float16],
        multiHeadOutputProjectionFused: true, name: "to_out")
    } else {
      scaledDotProductAttention = ScaledDotProductAttention(
        scale: 1.0 / Float(k).squareRoot(),
        flags: [.Float16],
        multiHeadOutputProjectionFused: true, name: "to_out")
    }
    let out = scaledDotProductAttention(queries, keys, values).reshaped([b, hw, k * h])
    return (tokeys, toqueries, tovalues, scaledDotProductAttention, Model([x], [out]))
  } else {
    let keys = tokeys(x).reshaped([b, hw, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
      .transposed(1, 2)
    let values = tovalues(x).reshaped([b, hw, h, k]).transposed(1, 2)
    var out: Model.IO
    if b * h <= 256 {
      var outs = [Model.IO]()
      for i in 0..<(b * h) {
        let key = keys.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
        let query = queries.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
        let value = values.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
        var dot = Matmul(transposeB: (1, 2))(query, key)
        if let last = outs.last {
          dot.add(dependencies: [last])
        }
        dot = dot.reshaped([hw, hw])
        dot = dot.softmax()
        dot = dot.reshaped([1, hw, hw])
        outs.append(dot * value)
      }
      out = Concat(axis: 0)(outs)
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
    } else {
      var dot = Matmul(transposeB: (2, 3))(queries, keys)
      dot = dot.reshaped([b * h * hw, hw])
      dot = dot.softmax()
      dot = dot.reshaped([b, h, hw, hw])
      out = dot * values
      out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
    }
    let unifyheads = Dense(count: k * h, name: "to_out")
    out = unifyheads(out)
    return (tokeys, toqueries, tovalues, unifyheads, Model([x], [out]))
  }
}

private func DownBlock2D(
  prefix: String, inChannels: Int, outChannels: Int, numRepeat: Int, batchSize: Int, x: Model.IO
) -> (Model.IO, [Model.IO]) {
  var out: Model.IO = x
  var inChannels = inChannels
  var hiddenStates = [Model.IO]()
  for _ in 0..<numRepeat {
    let (_, _, _, _, _, resBlock) = ResBlock(
      b: batchSize, groups: 4, outChannels: outChannels, skipConnection: inChannels != outChannels)
    out = resBlock(out)
    hiddenStates.append(out)
    inChannels = outChannels
  }
  let downsample = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
  out = downsample(out)
  hiddenStates.append(out)
  return (out, hiddenStates)
}

private func AttnDownBlock2D(
  prefix: String, inChannels: Int, outChannels: Int, numRepeat: Int, batchSize: Int, numHeads: Int,
  startHeight: Int, startWidth: Int, downsample: Bool, usesFlashAttention: FlashAttentionLevel,
  x: Model.IO
) -> (Model.IO, [Model.IO]) {
  var out: Model.IO = x
  var inChannels = inChannels
  var hiddenStates = [Model.IO]()
  for _ in 0..<numRepeat {
    let (_, _, _, _, _, resBlock) = ResBlock(
      b: batchSize, groups: 4, outChannels: outChannels, skipConnection: inChannels != outChannels)
    out = resBlock(out)
    let norm = GroupNorm(axis: 3, groups: 4, epsilon: 1e-5, reduce: [1, 2])
    let residual = out
    out = norm(out)
    let (_, _, _, _, attn) = SelfAttention(
      k: outChannels / numHeads, h: numHeads, b: batchSize, hw: startHeight * startWidth,
      usesFlashAttention: usesFlashAttention)
    out =
      attn(out.reshaped([batchSize, startHeight * startWidth, outChannels])).reshaped([
        batchSize, startHeight, startWidth, outChannels,
      ]) + residual
    hiddenStates.append(out)
    inChannels = outChannels
  }
  if downsample {
    let downsample = Convolution(
      groups: 1, filters: outChannels, filterSize: [3, 3],
      hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])), format: .OIHW)
    out = downsample(out)
    hiddenStates.append(out)
  }
  return (out, hiddenStates)
}

private func AttnUpBlock2D(
  prefix: String, channels: Int, numRepeat: Int, batchSize: Int, numHeads: Int, startHeight: Int,
  startWidth: Int, usesFlashAttention: FlashAttentionLevel, x: Model.IO, hiddenStates: [Model.IO]
) -> Model.IO {
  var out: Model.IO = x
  for i in 0..<numRepeat {
    let (_, _, _, _, _, resBlock) = ResBlock(
      b: batchSize, groups: 4, outChannels: channels, skipConnection: true)
    out = Functional.concat(axis: 3, out, hiddenStates[i])
    out = resBlock(out)
    let norm = GroupNorm(axis: 3, groups: 4, epsilon: 1e-5, reduce: [1, 2])
    let residual = out
    out = norm(out)
    let (_, _, _, _, attn) = SelfAttention(
      k: channels / numHeads, h: numHeads, b: batchSize, hw: startHeight * startWidth,
      usesFlashAttention: usesFlashAttention)
    out =
      attn(out.reshaped([batchSize, startHeight * startWidth, channels])).reshaped([
        batchSize, startHeight, startWidth, channels,
      ]) + residual
  }
  out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
  let conv = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv(out)
  return out
}

private func UpBlock2D(
  prefix: String, channels: Int, numRepeat: Int, batchSize: Int, upsample: Bool, x: Model.IO,
  hiddenStates: [Model.IO]
) -> Model.IO {
  var out: Model.IO = x
  for i in 0..<numRepeat {
    let (_, _, _, _, _, resBlock) = ResBlock(
      b: batchSize, groups: 4, outChannels: channels, skipConnection: true)
    out = Functional.concat(axis: 3, out, hiddenStates[i])
    out = resBlock(out)
  }
  if upsample {
    out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
    let conv = Convolution(
      groups: 1, filters: channels, filterSize: [3, 3],
      hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
    out = conv(out)
  }
  return out
}

private func MidBlock2D(
  prefix: String, channels: Int, batchSize: Int, numHeads: Int, startHeight: Int, startWidth: Int,
  usesFlashAttention: FlashAttentionLevel, x: Model.IO
) -> Model.IO {
  var out: Model.IO = x
  let (_, _, _, _, _, resBlock1) = ResBlock(
    b: batchSize, groups: 4, outChannels: channels, skipConnection: false)
  out = resBlock1(out)
  let norm = GroupNorm(axis: 3, groups: 4, epsilon: 1e-5, reduce: [1, 2])
  let residual = out
  out = norm(out)
  let (_, _, _, _, attn) = SelfAttention(
    k: channels / numHeads, h: numHeads, b: batchSize, hw: startHeight * startWidth,
    usesFlashAttention: usesFlashAttention)
  out =
    attn(out.reshaped([batchSize, startHeight * startWidth, channels])).reshaped([
      batchSize, startHeight, startWidth, channels,
    ]) + residual
  let (_, _, _, _, _, resBlock2) = ResBlock(
    b: batchSize, groups: 4, outChannels: channels, skipConnection: false)
  out = resBlock2(out)
  return out
}

func TransparentVAEDecoder(
  startHeight: Int, startWidth: Int, usesFlashAttention: FlashAttentionLevel
) -> Model {
  let pixel = Input()
  let latent = Input()
  precondition(startHeight % 64 == 0)
  precondition(startWidth % 64 == 0)
  let convIn = Convolution(
    groups: 1, filters: 32, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "conv_in")
  let latentConvIn = Convolution(
    groups: 1, filters: 64, filterSize: [1, 1],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [0, 0], end: [0, 0])), format: .OIHW,
    name: "latent_conv_in")
  var out = convIn(pixel)
  var hiddenStates: [Model.IO] = [out]
  let channels = [32, 32, 64, 128, 256, 512, 512]
  for i in 0..<4 {
    if i == 3 {
      out = out + latentConvIn(latent)
    }
    let (downOut, downHiddenStates) = DownBlock2D(
      prefix: "down_blocks.\(i)", inChannels: i > 0 ? channels[i - 1] : 32,
      outChannels: channels[i], numRepeat: 2, batchSize: 1, x: out)
    hiddenStates.append(contentsOf: downHiddenStates)
    out = downOut
  }
  var startHeight = startHeight / 16
  var startWidth = startWidth / 16
  for i in 4..<7 {
    let (downOut, downHiddenStates) = AttnDownBlock2D(
      prefix: "down_blocks.\(i)", inChannels: channels[i - 1], outChannels: channels[i],
      numRepeat: 2, batchSize: 1, numHeads: channels[i] / 8, startHeight: startHeight,
      startWidth: startWidth, downsample: (i < channels.count - 1),
      usesFlashAttention: usesFlashAttention, x: out)
    hiddenStates.append(contentsOf: downHiddenStates)
    out = downOut
    if i < channels.count - 1 {
      startHeight /= 2
      startWidth /= 2
    }
  }
  let midOut = MidBlock2D(
    prefix: "mid_block", channels: 512, batchSize: 1, numHeads: 512 / 8, startHeight: startHeight,
    startWidth: startWidth, usesFlashAttention: usesFlashAttention, x: out)
  out = midOut
  for i in 0..<3 {
    let upOut = AttnUpBlock2D(
      prefix: "up_blocks.\(i)", channels: channels[channels.count - 1 - i], numRepeat: 3,
      batchSize: 1, numHeads: channels[channels.count - 1 - i] / 8, startHeight: startHeight,
      startWidth: startWidth, usesFlashAttention: usesFlashAttention, x: out,
      hiddenStates: hiddenStates[(hiddenStates.count - 3 * (i + 1))..<(hiddenStates.count - 3 * i)]
        .reversed())
    out = upOut
    startHeight *= 2
    startWidth *= 2
  }
  for i in 3..<7 {
    let upOut = UpBlock2D(
      prefix: "up_blocks.\(i)", channels: channels[channels.count - 1 - i], numRepeat: 3,
      batchSize: 1, upsample: (i < channels.count - 1), x: out,
      hiddenStates: hiddenStates[(hiddenStates.count - 3 * (i + 1))..<(hiddenStates.count - 3 * i)]
        .reversed())
    out = upOut
  }
  let normOut = GroupNorm(axis: 3, groups: 4, epsilon: 1e-5, reduce: [1, 2], name: "norm_out")
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW,
    name: "conv_out")
  out = convOut(out)

  return Model([pixel, latent], [out])
}
