import NNC

private func ResnetBlock(prefix: String, outChannels: Int, shortcut: Bool) -> Model {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x)
  out = out.swish()
  let conv1 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv1(out)
  let norm2 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = norm2(out)
  out = out.swish()
  let conv2 = Convolution(
    groups: 1, filters: outChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = conv2(out)
  if shortcut {
    let nin = Convolution(
      groups: 1, filters: outChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
    out = nin(x) + out
  } else {
    out = x + out
  }
  return Model([x], [out])
}

private func AttnBlock(
  prefix: String, inChannels: Int, batchSize: Int, width: Int, height: Int, numHeads: Int,
  crossAttention: Bool
) -> Model {
  let x = Input()
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm1(x)
  let y: Model.IO?
  let y_: Model.IO
  if crossAttention {
    let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
    let y0 = Input()
    y_ = norm(y0)
    y = y0
  } else {
    y_ = out
    y = nil
  }
  let hw = width * height
  let attSize = inChannels / numHeads
  let tokeys = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let k = tokeys(out).reshaped([batchSize * numHeads, attSize, hw])
  let toqueries = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let q = ((1.0 / Float(attSize).squareRoot()) * toqueries(y_)).reshaped([
    batchSize * numHeads, attSize, hw,
  ])
  var dot = Matmul(transposeA: (1, 2))(q, k)
  dot = dot.reshaped([batchSize * numHeads * hw, hw])
  dot = dot.softmax()
  dot = dot.reshaped([batchSize * numHeads, hw, hw])
  let tovalues = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  let v = tovalues(out).reshaped([batchSize * numHeads, attSize, hw])
  out = Matmul(transposeB: (1, 2))(v, dot)
  let projOut = Convolution(
    groups: 1, filters: inChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = x + projOut(out.reshaped([batchSize, inChannels, height, width]))
  if let y = y {
    return Model([x, y], [out])
  } else {
    return Model([x], [out])
  }
}

private func MultiHeadEncoder(
  ch: Int, chMult: [Int], zChannels: Int, numHeads: Int, numResBlocks: Int, x: Model.IO
) -> [String: Model.IO] {
  let convIn = Convolution(
    groups: 1, filters: ch, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  var lastCh = ch
  var resolution = 512
  var outs = [String: Model.IO]()
  for (i, chM) in chMult.enumerated() {
    for j in 0..<numResBlocks {
      let block = ResnetBlock(
        prefix: "encoder.down.\(i).block.\(j)", outChannels: ch * chM, shortcut: lastCh != ch * chM)
      lastCh = ch * chM
      out = block(out)
      if i == chMult.count - 1 {
        let attnBlock = AttnBlock(
          prefix: "encoder.down.\(i).attn.\(j)", inChannels: lastCh, batchSize: 1,
          width: resolution, height: resolution, numHeads: numHeads, crossAttention: false)
        out = attnBlock(out)
      }
    }
    if i != chMult.count - 1 {
      outs["block_\(i)"] = out
      let downsample = Convolution(
        groups: 1, filters: lastCh, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [2, 2], end: [1, 1])))
      resolution = resolution / 2
      out = downsample(out).reshaped(
        [1, lastCh, resolution, resolution], offset: [0, 0, 1, 1],
        strides: [
          lastCh * (resolution + 1) * (resolution + 1), (resolution + 1) * (resolution + 1),
          resolution + 1, 1,
        ])
    }
  }
  let midBlock1 = ResnetBlock(
    prefix: "encoder.mid.block_1", outChannels: lastCh, shortcut: false)
  out = midBlock1(out)
  outs["block_\(chMult.count - 1)_attn"] = out
  let midAttn1 = AttnBlock(
    prefix: "encoder.mid.attn_1", inChannels: lastCh, batchSize: 1, width: resolution,
    height: resolution, numHeads: numHeads, crossAttention: false)
  out = midAttn1(out)
  let midBlock2 = ResnetBlock(
    prefix: "encoder.mid.block_2", outChannels: lastCh, shortcut: false)
  out = midBlock2(out)
  outs["mid_attn"] = out
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: zChannels, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  outs["out"] = out
  return outs
}

private func VectorQuantizer(nE: Int, eDim: Int, height: Int, width: Int) -> Model {
  let x = Input()
  let embedding = Input()
  var out = x.permuted(0, 2, 3, 1).reshaped([1 * height * width, eDim])
  let sum1 = (out .* out).reduced(.sum, axis: [1])
  let sum2 = (embedding .* embedding).reduced(.sum, axis: [1]).reshaped([1, nE])
  out = sum1 + (sum2 - 2 * Matmul(transposeB: (0, 1))(out, embedding))
  out = out.argmin(axis: 1)
  out = IndexSelect()(embedding, out.reshaped([height * width]))
  return Model([x, embedding], [out])
}

private func MultiHeadDecoderTransformer(
  ch: Int, chMult: [Int], zChannels: Int, numHeads: Int, numResBlocks: Int, x: Model.IO,
  hs: [String: Model.IO]
) -> Model.IO {
  var lastCh = ch * chMult[chMult.count - 1]
  var resolution = 512
  for _ in 0..<chMult.count - 1 {
    resolution = resolution / 2
  }
  let convIn = Convolution(
    groups: 1, filters: lastCh, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = convIn(x)
  let midBlock1 = ResnetBlock(
    prefix: "decoder.mid.block_1", outChannels: lastCh, shortcut: false)
  out = midBlock1(out)
  let midAttn1 = AttnBlock(
    prefix: "decoder.mid.attn_1", inChannels: lastCh, batchSize: 1, width: resolution,
    height: resolution, numHeads: numHeads, crossAttention: true)
  out = midAttn1(out, hs["mid_attn"]!)
  let midBlock2 = ResnetBlock(
    prefix: "decoder.mid.block_2", outChannels: lastCh, shortcut: false)
  out = midBlock2(out)
  for (i, chM) in chMult.enumerated().reversed() {
    for j in 0..<numResBlocks + 1 {
      let block = ResnetBlock(
        prefix: "decoder.up.\(i).block.\(j)", outChannels: ch * chM, shortcut: lastCh != ch * chM)
      lastCh = ch * chM
      out = block(out)
      if i == chMult.count - 1 {
        let attnBlock = AttnBlock(
          prefix: "decoder.up.\(i).attn.\(j)", inChannels: lastCh, batchSize: 1, width: resolution,
          height: resolution, numHeads: numHeads, crossAttention: true)
        out = attnBlock(out, hs["block_\(i)_attn"]!)
      }
    }
    if i != 0 {
      let upsample = Upsample(.nearest, widthScale: 2, heightScale: 2)
      out = upsample(out)
      let conv = Convolution(
        groups: 1, filters: lastCh, filterSize: [3, 3],
        hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
      out = conv(out)
      resolution = resolution * 2
    }
  }
  let normOut = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  out = normOut(out)
  out = out.swish()
  let convOut = Convolution(
    groups: 1, filters: 3, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = convOut(out)
  return out
}

func RestoreFormer(
  nEmbed: Int, embedDim: Int, ch: Int, chMult: [Int], zChannels: Int, numHeads: Int,
  numResBlocks: Int
) -> Model {
  let x = Input()
  let embedding = Input()
  let encoderOuts = MultiHeadEncoder(
    ch: ch, chMult: chMult, zChannels: zChannels, numHeads: numHeads, numResBlocks: numResBlocks,
    x: x)
  var out = encoderOuts["out"]!
  let quantConv = Convolution(
    groups: 1, filters: embedDim, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = quantConv(out)
  let vq = VectorQuantizer(nE: nEmbed, eDim: embedDim, height: 16, width: 16)
  out = vq(out, embedding)
  out = out.transposed(0, 1).reshaped([1, embedDim, 16, 16])
  let postQuantConv = Convolution(
    groups: 1, filters: zChannels, filterSize: [1, 1], hint: Hint(stride: [1, 1]))
  out = postQuantConv(out)
  let decoderOut = MultiHeadDecoderTransformer(
    ch: ch, chMult: chMult, zChannels: zChannels, numHeads: numHeads, numResBlocks: numResBlocks,
    x: out, hs: encoderOuts)
  out = decoderOut
  return Model([x, embedding], [out])
}
