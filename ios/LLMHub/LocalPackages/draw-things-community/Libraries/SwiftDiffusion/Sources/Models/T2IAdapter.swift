import NNC

private func ResnetBlock(outChannels: Int, inConv: Bool) -> Model {
  let x = Input()
  let outX: Model.IO
  if inConv {
    let skip = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1],
      hint: Hint(stride: [1, 1]), format: .OIHW)
    outX = skip(x)
  } else {
    outX = x
  }
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = inLayerConv2d(outX)
  out = ReLU()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [1, 1],
    hint: Hint(stride: [1, 1]), format: .OIHW)
  out = outLayerConv2d(out) + outX
  return Model([x], [out])
}

func Adapter(channels: [Int], numRepeat: Int) -> Model {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = convIn(x)
  var previousChannel = channels[0]
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    for _ in 0..<numRepeat {
      let resnetBlock = ResnetBlock(
        outChannels: channel, inConv: previousChannel != channel)
      previousChannel = channel
      out = resnetBlock(out)
    }
    outs.append(out)
    if i != channels.count - 1 {
      let downsample = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
      out = downsample(out)
    }
  }
  return Model([x], outs)
}

private func ResnetBlockLight(outChannels: Int) -> Model {
  let x = Input()
  let inLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = inLayerConv2d(x)
  out = ReLU()(out)
  // Dropout if needed in the future (for training).
  let outLayerConv2d = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = outLayerConv2d(out) + x
  return Model(
    [x], [out]
  )
}

private func Extractor(
  prefix: String, channel: Int, innerChannel: Int, numRepeat: Int, downsample: Bool
)
  -> Model
{
  let x = Input()
  var out: Model.IO = x
  if downsample {
    let downsample = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
    out = downsample(out)
  }
  let inConv = Convolution(
    groups: 1, filters: innerChannel, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = inConv(out)
  for _ in 0..<numRepeat {
    let resnetBlock = ResnetBlockLight(outChannels: innerChannel)
    out = resnetBlock(out)
  }
  let outConv = Convolution(
    groups: 1, filters: channel, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = outConv(out)
  return Model([x], [out])
}

func AdapterLight(channels: [Int], numRepeat: Int) -> Model {
  let x = Input()
  var out: Model.IO = x
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let extractor = Extractor(
      prefix: "\(i)", channel: channel, innerChannel: channel / 4, numRepeat: numRepeat,
      downsample: i != 0)
    out = extractor(out)
    outs.append(out)
  }
  return Model([x], outs)
}

private func CLIPResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let attention = CLIPSelfAttention(
    k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = QuickGELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  return Model([x], [out])
}

func StyleAdapter(
  width: Int, outputDim: Int, layers: Int, heads: Int, tokens: Int, batchSize: Int
) -> Model {
  let x = Input()
  let lnPre = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = lnPre(x)
  for i in 0..<layers {
    let block = CLIPResidualAttentionBlock(
      prefix: "transformer_layes.\(i)", k: width / heads, h: heads, b: batchSize, t: 257 + tokens)
    out = block(out.reshaped([batchSize, 257 + tokens, width]))
  }
  let lnPost = LayerNorm(epsilon: 1e-5, axis: [2])
  out = lnPost(out.reshaped([batchSize, 257 + tokens, width]))
  let proj = Dense(count: outputDim, noBias: true)
  out = proj(out)
  return Model([x], [out])
}
