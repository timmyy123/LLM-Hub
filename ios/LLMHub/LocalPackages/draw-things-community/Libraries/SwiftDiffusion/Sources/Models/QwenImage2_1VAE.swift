import NNC

private func NHWCResnetBlock(inChannels: Int, outChannels: Int, prefix: String) -> Model {
  let x = Input()
  let norm1 = RMSNorm(epsilon: 1e-12, axis: [3], name: "\(prefix)_norm1")
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
    format: .OIHW, name: "\(prefix)_conv1")
  let norm2 = RMSNorm(epsilon: 1e-12, axis: [3], name: "\(prefix)_norm2")
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
    format: .OIHW, name: "\(prefix)_conv2")
  var out = conv2(norm2(conv1(norm1(x).swish())).swish())
  if inChannels != outChannels {
    let shortcut = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "\(prefix)_conv_shortcut")
    out = out + shortcut(x)
  } else {
    out = out + x
  }
  return Model([x], [out])
}

private func NHWCAttnBlock(
  channels: Int, height: Int, width: Int, prefix: String, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let norm = RMSNorm(epsilon: 1e-12, axis: [3], name: "\(prefix)_norm")
  let toQKV = Convolution(
    groups: 1, filters: channels * 3, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW, name: "\(prefix)_to_qkv")
  let qkv = toQKV(norm(x)).reshaped([height * width, channels * 3])
  let q = qkv.reshaped([height * width, channels], strides: [channels * 3, 1]).contiguous()
  let k = qkv.reshaped(
    [height * width, channels], offset: [0, channels], strides: [channels * 3, 1]
  ).contiguous()
  let v = qkv.reshaped(
    [height * width, channels], offset: [0, channels * 2], strides: [channels * 3, 1]
  ).contiguous()
  let out: Model.IO
  if usesFlashAttention {
    let projOut = ScaledDotProductAttention(
      scale: 1 / Float(channels).squareRoot(), multiHeadOutputProjectionFused: true,
      name: "\(prefix)_proj")
    out = projOut(
      q.reshaped([1, height * width, channels]), k.reshaped([1, height * width, channels]),
      v.reshaped([1, height * width, channels])
    ).reshaped([1, height, width, channels])
  } else {
    let scores = Matmul(transposeB: (0, 1))(q, k) * (1 / Float(channels).squareRoot())
    let attended = Matmul()(scores.softmax(), v).reshaped([1, height, width, channels])
    let projOut = Convolution(
      groups: 1, filters: channels, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
      format: .OIHW, name: "\(prefix)_proj")
    out = projOut(attended)
  }
  return Model([x], [x + out])
}

private func NHWCDownsampleShortcut(
  inChannels: Int, outChannels: Int, height: Int, width: Int, temporal: Int, spatial: Int
) -> Model {
  let x = Input()
  if temporal == 1 && inChannels == outChannels {
    let out = AveragePool(filterSize: [spatial, spatial], hint: Hint(stride: [spatial, spatial]))(x)
    return Model([x], [out])
  }
  let h = height / spatial
  let w = width / spatial
  var out: Model.IO = x
  if temporal == 2 {
    out = out.padded(.zero, begin: [1, 0, 0, 0], end: [0, 0, 0, 0])
  }
  out = out.reshaped([1, temporal, h, spatial, w, spatial, inChannels])
    .permuted(0, 2, 4, 6, 1, 3, 5).copied()
  let group = inChannels * temporal * spatial * spatial / outChannels
  out = out.reshaped([1, h, w, outChannels, group]).reduced(.mean, axis: [4])
    .reshaped([1, h, w, outChannels])
  return Model([x], [out])
}

private func NHWCUpsampleShortcut(
  inChannels: Int, outChannels: Int, height: Int, width: Int, temporal: Int
) -> Model {
  let x = Input()
  let out: Model.IO
  if inChannels == outChannels {
    out = Upsample(.nearest, widthScale: 2, heightScale: 2)(x)
  } else if temporal == 2 {
    // The retained temporal slice contains the odd input channels.
    let selected = x.reshaped([height * width, outChannels, 2]).reshaped(
      [height * width, outChannels, 1], offset: [0, 0, 1], strides: [inChannels, 2, 1]
    ).contiguous().reshaped([1, height, width, outChannels])
    out = Upsample(.nearest, widthScale: 2, heightScale: 2)(selected)
  } else {
    // Each channel pair supplies the top and bottom rows of a spatial 2x2 block.
    let rows = x.reshaped([height, width, outChannels, 2]).permuted(0, 3, 1, 2)
      .contiguous().reshaped([1, height * 2, width, outChannels])
    out = Upsample(.nearest, widthScale: 2, heightScale: 1)(rows)
  }
  return Model([x], [out])
}

// Single-image specialization: temporal resampling convolutions are inactive for the first frame.
public func QwenImage2_1Encoder(
  channels: [Int], height: Int, width: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
    format: .OIHW, name: "encoder_conv_in")
  var out = convIn(x)
  var previousChannel = channels[0]
  var h = height
  var w = width
  for (i, channel) in channels.enumerated() {
    let downsample = i < channels.count - 1
    let shortcut = NHWCDownsampleShortcut(
      inChannels: previousChannel, outChannels: channel, height: h, width: w,
      temporal: i > 0 && downsample ? 2 : 1, spatial: downsample ? 2 : 1)(out)
    for j in 0..<2 {
      let block = NHWCResnetBlock(
        inChannels: j == 0 ? previousChannel : channel, outChannels: channel,
        prefix: "encoder_down_blocks_\(i)_resnets_\(j)")
      out = block(out)
    }
    if downsample {
      let conv = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3], hint: Hint(stride: [2, 2]),
        format: .OIHW, name: "encoder_down_blocks_\(i)_downsampler_resample_1")
      out = conv(out.padded(.zero, begin: [0, 0, 0, 0], end: [0, 1, 1, 0]))
      h /= 2
      w /= 2
    }
    out = out + shortcut
    previousChannel = channel
  }
  let midBlock1 = NHWCResnetBlock(
    inChannels: previousChannel, outChannels: previousChannel, prefix: "encoder_mid_block_resnets_0"
  )
  let midAttention = NHWCAttnBlock(
    channels: previousChannel, height: h, width: w, prefix: "encoder_mid_block_attentions_0",
    usesFlashAttention: usesFlashAttention)
  let midBlock2 = NHWCResnetBlock(
    inChannels: previousChannel, outChannels: previousChannel, prefix: "encoder_mid_block_resnets_1"
  )
  out = midBlock2(midAttention(midBlock1(out)))
  let normOut = RMSNorm(epsilon: 1e-12, axis: [3], name: "encoder_norm_out")
  let convOut = Convolution(
    groups: 1, filters: 128, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
    format: .OIHW, name: "encoder_conv_out")
  let quantConv = Convolution(
    groups: 1, filters: 128, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW, name: "quant_conv")
  out = quantConv(convOut(normOut(out).swish()))
  return Model([x], [out])
}

public func QwenImage2_1Decoder(
  channels: [Int], height: Int, width: Int, usesFlashAttention: Bool
) -> Model {
  let x = Input()
  let postQuantConv = Convolution(
    groups: 1, filters: 64, filterSize: [1, 1], hint: Hint(stride: [1, 1]),
    format: .OIHW, name: "post_quant_conv")
  let convIn = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
    format: .OIHW, name: "decoder_conv_in")
  var out = convIn(postQuantConv(x))
  let midBlock1 = NHWCResnetBlock(
    inChannels: channels[0], outChannels: channels[0], prefix: "decoder_mid_block_resnets_0")
  let midAttention = NHWCAttnBlock(
    channels: channels[0], height: height, width: width, prefix: "decoder_mid_block_attentions_0",
    usesFlashAttention: usesFlashAttention)
  let midBlock2 = NHWCResnetBlock(
    inChannels: channels[0], outChannels: channels[0], prefix: "decoder_mid_block_resnets_1")
  out = midBlock2(midAttention(midBlock1(out)))
  var previousChannel = channels[0]
  var h = height
  var w = width
  for (i, channel) in channels.enumerated() {
    let upsample = i < channels.count - 1
    let shortcut =
      upsample
      ? NHWCUpsampleShortcut(
        inChannels: previousChannel, outChannels: channel, height: h, width: w,
        temporal: i < 3 ? 2 : 1)(out) : nil
    for j in 0..<3 {
      let block = NHWCResnetBlock(
        inChannels: j == 0 ? previousChannel : channel, outChannels: channel,
        prefix: "decoder_up_blocks_\(i)_resnets_\(j)")
      out = block(out)
    }
    if let shortcut = shortcut {
      let conv = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
        format: .OIHW, name: "decoder_up_blocks_\(i)_upsampler_resample_1")
      // The final two upsamplers and residual stages can exceed FP16 range.
      // Return to input precision after the final normalization.
      out = conv(Upsample(.nearest, widthScale: 2, heightScale: 2)(out)) + shortcut
      if i == channels.count - 4 {
        out = out.to(.Float32)
      }
      h *= 2
      w *= 2
    }
    previousChannel = channel
  }
  let normOut = RMSNorm(epsilon: 1e-12, axis: [3], name: "decoder_norm_out")
  let convOut = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])),
    format: .OIHW, name: "decoder_conv_out")
  out = convOut(normOut(out).to(of: x).swish())
  return Model([x], [out])
}
