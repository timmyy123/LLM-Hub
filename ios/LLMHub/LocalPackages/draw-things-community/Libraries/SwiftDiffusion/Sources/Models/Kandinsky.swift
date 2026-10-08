import NNC

private func SelfAttention(
  prefix: String, k: Int, h: Int, b: Int, t: Int, usesFlashAttention: FlashAttentionLevel
)
  -> Model
{
  let x = Input()
  let causalAttentionMask = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  if usesFlashAttention != .none {
    let queries = toqueries(x).reshaped([b, t, h, k]).identity().identity()
    let keys = tokeys(x).reshaped([b, t, h, k]).identity()
    let values = tovalues(x).reshaped([b, t, h, k])
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(), isCausal: true, hasAttentionMask: true,
      flags: [.Float16],
      multiHeadOutputProjectionFused: true)
    let out = scaledDotProductAttention(queries, keys, values, causalAttentionMask).reshaped([
      b * t, h * k,
    ])
    return Model([x, causalAttentionMask], [out])
  } else {
    let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
      .permuted(0, 2, 1, 3)
    let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
    var dot = Matmul(transposeB: (2, 3))(queries, keys) + causalAttentionMask
    dot = dot.reshaped([b * h * t, t])
    dot = dot.softmax()
    dot = dot.reshaped([b, h, t, t])
    var out = dot * values
    out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
    let unifyheads = Dense(count: k * h)
    out = unifyheads(out)
    return Model([x, causalAttentionMask], [out])
  }
}

private func ResidualAttentionBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, usesFlashAttention: FlashAttentionLevel
) -> Model {
  let x = Input()
  let causalAttentionMask = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [1])
  let selfAttention = SelfAttention(
    prefix: "\(prefix).attn", k: k, h: h, b: b, t: t, usesFlashAttention: usesFlashAttention)
  var out = x + selfAttention(layerNorm1(x), causalAttentionMask)
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let intermediate = Dense(count: k * h * 4)
  let output = Dense(count: k * h)
  out = out + output(intermediate(layerNorm2(out)).GELU())
  return Model([x, causalAttentionMask], [out])
}

func timestepEmbedding(prefix: String, channels: Int) -> Model {
  let x = Input()
  let dense1 = Dense(count: channels)
  var out = dense1(x).swish()
  let dense2 = Dense(count: channels)
  out = dense2(out)
  return Model([x], [out])
}

public func DiffusionMappingModel(
  numberOfLayers: Int, k: Int, h: Int, b: Int, t: Int, outChannels: Int,
  usesFlashAttention: FlashAttentionLevel
)
  -> Model
{
  let x = Input()
  let causalAttentionMask = Input()
  var out: Model.IO = x
  for i in 0..<numberOfLayers {
    let layer = ResidualAttentionBlock(
      prefix: "model.transformer.resblocks.\(i)", k: k, h: h, b: b, t: t,
      usesFlashAttention: usesFlashAttention)
    out = layer(out, causalAttentionMask)
  }
  let finalLn = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLn(out)
  let outProj = Dense(count: outChannels)
  out = outProj(
    out.reshaped([b, 1, k * h], offset: [0, t - 1, 0], strides: [t * k * h, k * h, 1]))
  return Model([x, causalAttentionMask], [out])
}

private func ResBlock(
  prefix: String, batchSize: Int, outChannels: Int, up: Bool, down: Bool, skipConnection: Bool
) -> Model {
  let x = Input()
  let emb = Input()
  let norm1 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  var out = norm1(x).swish()
  var xhd: Model.IO = x
  if up {
    let hup = Upsample(.nearest, widthScale: 2, heightScale: 2)
    out = hup(out)
    let xup = Upsample(.nearest, widthScale: 2, heightScale: 2)
    xhd = xup(x)
  } else if down {
    let hdown = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
    out = hdown(out)
    let xdown = AveragePool(filterSize: [2, 2], hint: Hint(stride: [2, 2]))
    xhd = xdown(x)
  }
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv1(out)
  let embLayer = Dense(count: 2 * outChannels)
  let embOut = embLayer(emb.swish())
  let embScale = embOut.reshaped(
    [batchSize, 1, 1, outChannels], offset: [0, 0, 0, 0],
    strides: [outChannels * 2, outChannels * 2, outChannels * 2, 1])
  let embShift = embOut.reshaped(
    [batchSize, 1, 1, outChannels], offset: [0, 0, 0, outChannels],
    strides: [outChannels * 2, outChannels * 2, outChannels * 2, 1])
  let norm2 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = norm2(out) .* (1 + embScale) + embShift
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2(out.swish())
  if skipConnection {
    let conv = Convolution(groups: 1, filters: outChannels, filterSize: [1, 1], format: .OIHW)
    xhd = conv(xhd)
  }
  out = xhd + out
  return Model([x, emb], [out])
}

private func AttentionBlock(
  prefix: String, k: Int, h: Int, b: Int, t: Int, height: Int, width: Int,
  usesFlashAttention: FlashAttentionLevel
)
  -> Model
{
  let hw = height * width
  let x = Input()
  let encoderOut = Input()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  var out = norm(x)
  let toencoderkeys = Dense(count: k * h)
  let toencodervalues = Dense(count: k * h)
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let encoderIn = encoderOut
  if usesFlashAttention != .none {
    let encoderkeys = toencoderkeys(encoderIn).reshaped([b, t, h, k])
    let encodervalues = toencodervalues(encoderIn).reshaped([b, t, h, k])
    var keys = tokeys(out).reshaped([b, hw, h, k])
    let queries = toqueries(out).reshaped([b, hw, h, k]).identity().identity()
    var values = tovalues(out).reshaped([b, hw, h, k])
    keys = Functional.concat(axis: 1, encoderkeys, keys).identity()
    values = Functional.concat(axis: 1, encodervalues, values)
    let scaledDotProductAttention = ScaledDotProductAttention(
      scale: 1.0 / Float(k).squareRoot(),
      flags: [.Float16],
      multiHeadOutputProjectionFused: true)
    out = scaledDotProductAttention(queries, keys, values).reshaped([b, height, width, k * h]) + x
  } else {
    let encoderkeys = toencoderkeys(encoderIn).reshaped([b, t, h, k]).transposed(1, 2)
    let encodervalues = toencodervalues(encoderIn).reshaped([b, t, h, k]).transposed(1, 2)
    var keys = tokeys(out).reshaped([b, hw, h, k]).transposed(1, 2)
    let queries = ((1.0 / Float(k).squareRoot()) * toqueries(out)).reshaped([b, hw, h, k])
      .transposed(1, 2)
    var values = tovalues(out).reshaped([b, hw, h, k]).transposed(1, 2)
    keys = Functional.concat(axis: 2, encoderkeys, keys)
    values = Functional.concat(axis: 2, encodervalues, values)
    var outs = [Model.IO]()
    for i in 0..<(b * h) {
      let key = keys.reshaped([1, (hw + t), k], offset: [i, 0, 0], strides: [(hw + t) * k, k, 1])
      let query = queries.reshaped([1, hw, k], offset: [i, 0, 0], strides: [hw * k, k, 1])
      let value = values.reshaped(
        [1, (hw + t), k], offset: [i, 0, 0], strides: [(hw + t) * k, k, 1])
      var dot = Matmul(transposeB: (1, 2))(query, key)
      if let last = outs.last {
        dot.add(dependencies: [last])
      }
      dot = dot.reshaped([hw, t + hw])
      dot = dot.softmax()
      dot = dot.reshaped([1, hw, t + hw])
      outs.append(dot * value)
    }
    out = Concat(axis: 0)(outs)
    out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, height, width, k * h])
    let projOut = Convolution(groups: 1, filters: k * h, filterSize: [1, 1], format: .OIHW)
    out = projOut(out) + x
  }
  return Model([x, encoderOut], [out])
}

private func InputBlocks(
  prefix: String, batchSize: Int, channels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>,
  usesFlashAttention: FlashAttentionLevel, x: Model.IO, emb: Model.IO, xfOut: Model.IO
) -> (Model.IO, [Model.IO]) {
  let convIn = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = convIn(x)
  var i = 1
  var lastCh = channels
  var ds = 1
  var height = startHeight
  var width = startWidth
  var hs = [Model.IO]()
  hs.append(out)
  for (level, mult) in channelMult.enumerated() {
    let ch = channels * mult
    for _ in 0..<numResBlocks {
      let resBlock = ResBlock(
        prefix: "\(prefix).\(i).0", batchSize: batchSize, outChannels: ch, up: false, down: false,
        skipConnection: ch != lastCh)
      out = resBlock(out, emb)
      lastCh = ch
      if attentionResolutions.contains(ds) {
        let attentionBlock = AttentionBlock(
          prefix: "\(prefix).\(i).1", k: numHeadChannels, h: ch / numHeadChannels, b: batchSize,
          t: t, height: height, width: width, usesFlashAttention: usesFlashAttention)
        out = attentionBlock(out, xfOut)
      }
      hs.append(out)
      i += 1
    }
    if level != channelMult.count - 1 {
      let resBlock = ResBlock(
        prefix: "\(prefix).\(i).0", batchSize: batchSize, outChannels: ch, up: false, down: true,
        skipConnection: false)
      out = resBlock(out, emb)
      hs.append(out)
      i += 1
      ds *= 2
      height /= 2
      width /= 2
    }
  }
  return (out, hs)
}

private func OutputBlocks(
  prefix: String, batchSize: Int, channels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>,
  usesFlashAttention: FlashAttentionLevel, x: Model.IO, emb: Model.IO, xfOut: Model.IO,
  hs: [Model.IO]
) -> Model.IO {
  var out: Model.IO = x
  var i = 0
  var ds = 1
  var height = startHeight
  var width = startWidth
  for _ in 1..<channelMult.count {
    ds *= 2
    height /= 2
    width /= 2
  }
  for (level, mult) in channelMult.enumerated().reversed() {
    let ch = channels * mult
    for j in 0..<(numResBlocks + 1) {
      out = Functional.concat(axis: 3, out, hs[hs.count - 1 - i])
      let resBlock = ResBlock(
        prefix: "\(prefix).\(i).0", batchSize: batchSize, outChannels: ch, up: false, down: false,
        skipConnection: true)
      out = resBlock(out, emb)
      if attentionResolutions.contains(ds) {
        let attentionBlock = AttentionBlock(
          prefix: "\(prefix).\(i).1", k: numHeadChannels, h: ch / numHeadChannels, b: batchSize,
          t: t, height: height, width: width, usesFlashAttention: usesFlashAttention)
        out = attentionBlock(out, xfOut)
      }
      if level > 0 && j == numResBlocks {
        let resBlock = ResBlock(
          prefix: "\(prefix).\(i).2", batchSize: batchSize, outChannels: ch, up: true, down: false,
          skipConnection: false)
        out = resBlock(out, emb)
        ds /= 2
        height *= 2
        width *= 2
      }
      i += 1
    }
  }
  return out
}

public func ImageAndTextEmbedding(batchSize: Int) -> Model {
  let imageEmb = Input()
  let poolEmb = Input()
  let fullEmb = Input()
  let clipToSeq = Dense(count: 10 * 768)
  let projN = Dense(count: 384 * 4)
  let lnModelN = LayerNorm(epsilon: 1e-5, axis: [2])
  let imgLayer = Dense(count: 384 * 4)
  let toModelDimN = Dense(count: 768)
  let clipSeq = clipToSeq(imageEmb).reshaped([batchSize, 10, 768])
  let xfProj = lnModelN(projN(poolEmb)) + imgLayer(imageEmb)
  let textEmb = toModelDimN(fullEmb)
  let xfOut = Functional.concat(axis: 1, clipSeq, textEmb)
  return Model([poolEmb, fullEmb, imageEmb], [xfProj, xfOut])
}

public func UNetKandinsky(
  batchSize: Int, channels: Int, outChannels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>,
  usesFlashAttention: FlashAttentionLevel
) -> Model {
  let x = Input()
  let emb = Input()
  let xfOut = Input()
  let (inputBlocksOut, hs) = InputBlocks(
    prefix: "input_blocks", batchSize: batchSize, channels: channels, channelMult: channelMult,
    numResBlocks: numResBlocks, numHeadChannels: numHeadChannels, t: t, startHeight: startHeight,
    startWidth: startWidth, attentionResolutions: attentionResolutions,
    usesFlashAttention: usesFlashAttention, x: x, emb: emb, xfOut: xfOut
  )
  let ch = channelMult[channelMult.count - 1] * channels
  var out = inputBlocksOut
  let middleResBlock1 = ResBlock(
    prefix: "middle_block.0", batchSize: batchSize, outChannels: ch, up: false, down: false,
    skipConnection: false)
  out = middleResBlock1(out, emb)
  var height = startHeight
  var width = startWidth
  for _ in 1..<channelMult.count {
    height /= 2
    width /= 2
  }
  let middleAttentionBlock2 = AttentionBlock(
    prefix: "middle_block.1", k: numHeadChannels, h: ch / numHeadChannels, b: batchSize, t: t,
    height: height, width: width, usesFlashAttention: usesFlashAttention)
  out = middleAttentionBlock2(out, xfOut)
  let middleResBlock3 = ResBlock(
    prefix: "middle_block.2", batchSize: batchSize, outChannels: ch, up: false, down: false,
    skipConnection: false)
  out = middleResBlock3(out, emb)
  let outputBlocksOut = OutputBlocks(
    prefix: "output_blocks", batchSize: batchSize, channels: channels, channelMult: channelMult,
    numResBlocks: numResBlocks, numHeadChannels: numHeadChannels, t: t, startHeight: startHeight,
    startWidth: startWidth, attentionResolutions: attentionResolutions,
    usesFlashAttention: usesFlashAttention, x: out, emb: emb, xfOut: xfOut, hs: hs)
  out = outputBlocksOut
  let normOut = GroupNorm(axis: 3, groups: 32, epsilon: 1e-5, reduce: [1, 2])
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convOut(out)
  return Model([x, emb, xfOut], [out])
}

private func ResnetBlock(prefix: String, inChannels: Int, outChannels: Int) -> Model {
  let x = Input()
  let norm1 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = norm1(x).swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv1(out)
  let norm2 = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  out = norm2(out).swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2(out)
  if inChannels != outChannels {
    let shortcut = Convolution(groups: 1, filters: outChannels, filterSize: [1, 1], format: .OIHW)
    out = shortcut(x) + out
  } else {
    out = x + out
  }
  return Model([x], [out])
}

private func AttnBlock(
  prefix: String, inChannels: Int, batchSize: Int, height: Int, width: Int
) -> Model {
  let x = Input()
  let norm = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = norm(x)
  let hw = width * height
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let k = tokeys(out).reshaped([batchSize, hw, inChannels])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    batchSize, hw, inChannels,
  ])
  var dot = Matmul(transposeB: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let v = tovalues(out).reshaped([batchSize, hw, inChannels])
  out = dot * v
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = x + projOut(out.reshaped([batchSize, height, width, inChannels]))
  return Model([x], [out])
}

public func EncoderKandinsky(
  zChannels: Int, channels: Int, channelMult: [Int], numResBlocks: Int, startHeight: Int,
  startWidth: Int, attnResolutions: Set<Int>
) -> Model {
  let x = Input()
  let convIn = Convolution(
    groups: 1, filters: channels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = convIn(x)
  var lastCh = channels
  var currentRes = 256
  var height = startHeight
  var width = startWidth
  for (i, mult) in channelMult.enumerated() {
    let ch = channels * mult
    for j in 0..<numResBlocks {
      let resnetBlock = ResnetBlock(
        prefix: "encoder.down.\(i).block.\(j)", inChannels: lastCh, outChannels: ch)
      out = resnetBlock(out)
      lastCh = ch
      if attnResolutions.contains(currentRes) {
        let attnBlock = AttnBlock(
          prefix: "encoder.down.\(i).attn.\(j)", inChannels: ch, batchSize: 1, height: height,
          width: width)
        out = attnBlock(out)
      }
    }
    if i != channelMult.count - 1 {
      currentRes /= 2
      height /= 2
      width /= 2
      let conv2d = Convolution(
        groups: 1, filters: ch, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])), format: .OIHW)
      out = conv2d(out).reshaped(
        [1, height, width, ch], offset: [0, 1, 1, 0],
        strides: [(height + 1) * (width + 1) * ch, (width + 1) * ch, ch, 1])
    }
  }
  let midResnetBlock1 = ResnetBlock(
    prefix: "encoder.mid.block_1", inChannels: lastCh, outChannels: lastCh)
  out = midResnetBlock1(out)
  let midAttnBlock1 = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: lastCh, batchSize: 1, height: height, width: width)
  out = midAttnBlock1(out)
  let midResnetBlock2 = ResnetBlock(
    prefix: "encoder.mid.block_2", inChannels: lastCh, outChannels: lastCh)
  out = midResnetBlock2(out)
  let normOut = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  out = normOut(out).swish()
  let convOut = Convolution(
    groups: 1, filters: zChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convOut(out)
  let quantConv = Convolution(groups: 1, filters: zChannels, filterSize: [1, 1], format: .OIHW)
  out = quantConv(out)
  return Model([x], [out])
}

private func SpatialNorm(prefix: String, channels: Int, heightScale: Float, widthScale: Float)
  -> Model
{
  let x = Input()
  let zq = Input()
  let normLayer = GroupNorm(axis: 3, groups: 32, epsilon: 1e-6, reduce: [1, 2])
  var out = normLayer(x)
  let zqOut = Upsample(.nearest, widthScale: widthScale, heightScale: heightScale)(zq)
  // Helps to reduce RAM usage, otherwise all upsampling done from the input, therefore, would be
  // interpreted as can be computed at the same time.
  zqOut.add(dependencies: [out])
  let convY = Convolution(groups: 1, filters: channels, filterSize: [1, 1], format: .OIHW)
  let convB = Convolution(groups: 1, filters: channels, filterSize: [1, 1], format: .OIHW)
  out = out .* convY(zqOut)
  let bias = convB(zqOut)
  bias.add(dependencies: [out])
  out = out + bias
  return Model([x, zq], [out])
}

private func MOVQResnetBlock(prefix: String, inChannels: Int, outChannels: Int, scale: Float)
  -> Model
{
  let x = Input()
  let zq = Input()
  let norm1 = SpatialNorm(
    prefix: "\(prefix).norm1", channels: inChannels, heightScale: scale, widthScale: scale)
  var out = norm1(x, zq).swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv1(out)
  let norm2 = SpatialNorm(
    prefix: "\(prefix).norm2", channels: outChannels, heightScale: scale, widthScale: scale)
  out = norm2(out, zq).swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = conv2(out)
  if inChannels != outChannels {
    let shortcut = Convolution(groups: 1, filters: outChannels, filterSize: [1, 1], format: .OIHW)
    out = shortcut(x) + out
  } else {
    out = x + out
  }
  return Model([x, zq], [out])
}

private func MOVQAttnBlock(
  prefix: String, inChannels: Int, batchSize: Int, height: Int, width: Int, scale: Float
) -> Model {
  let x = Input()
  let zq = Input()
  let norm = SpatialNorm(
    prefix: "\(prefix).norm", channels: inChannels, heightScale: scale, widthScale: scale)
  var out = norm(x, zq)
  let hw = width * height
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let k = tokeys(out).reshaped([batchSize, hw, inChannels])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let q = ((1.0 / Float(inChannels).squareRoot()) * toqueries(out)).reshaped([
    batchSize, hw, inChannels,
  ])
  var dot = Matmul(transposeB: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  let v = tovalues(out).reshaped([batchSize, hw, inChannels])
  out = dot * v
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]), format: .OIHW)
  out = x + projOut(out.reshaped([batchSize, height, width, inChannels]))
  return Model([x, zq], [out])
}

public func MOVQDecoderKandinsky(
  zChannels: Int, channels: Int, channelMult: [Int], numResBlocks: Int, startHeight: Int,
  startWidth: Int, attnResolutions: Set<Int>
) -> Model {
  let x = Input()
  let postQuantConv = Convolution(groups: 1, filters: zChannels, filterSize: [1, 1], format: .OIHW)
  let z = postQuantConv(x)
  var blockIn = channels * channelMult[channelMult.count - 1]
  let convIn = Convolution(
    groups: 1, filters: blockIn, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  var out = convIn(z)
  let midBlock1 = MOVQResnetBlock(
    prefix: "decoder.mid.block_1", inChannels: blockIn, outChannels: blockIn, scale: 1)
  out = midBlock1(out, x)
  let midAttn1 = MOVQAttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: blockIn, batchSize: 1, height: startHeight,
    width: startWidth, scale: 1)
  out = midAttn1(out, x)
  let midBlock2 = MOVQResnetBlock(
    prefix: "decoder.mid.block_2", inChannels: blockIn, outChannels: blockIn, scale: 1)
  out = midBlock2(out, x)
  var ds = 1
  var currentRes = 32
  var height = startHeight
  var width = startWidth
  for (i, mult) in channelMult.enumerated().reversed() {
    let blockOut = channels * mult
    for j in 0..<(numResBlocks + 1) {
      let resnetBlock = MOVQResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", inChannels: blockIn, outChannels: blockOut,
        scale: Float(ds))
      out = resnetBlock(out, x)
      blockIn = blockOut
      if attnResolutions.contains(currentRes) {
        let attn = MOVQAttnBlock(
          prefix: "decoder.up.\(i).attn.\(j)", inChannels: blockIn, batchSize: 1, height: height,
          width: width, scale: Float(ds))
        out = attn(out, x)
      }
    }
    if i > 0 {
      out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
      let conv = Convolution(
        groups: 1, filters: blockIn, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
      out = conv(out)
      ds *= 2
      currentRes *= 2
      height *= 2
      width *= 2
    }
  }
  let normOut = SpatialNorm(
    prefix: "decoder.norm_out", channels: blockIn, heightScale: Float(ds), widthScale: Float(ds))
  out = normOut(out, x).swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])), format: .OIHW)
  out = convOut(out)
  return Model([x], [out])
}
