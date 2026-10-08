import Foundation

private struct UNetFeatureMapSpec {
  let channels: Int
  let height: Int
  let width: Int
}

private func UNetSplitAttentionInstructionCount(
  batchSize: Int, heads: Int, headDimension: Int, queryLength: Int, keyLengths: (Int, Int)
) -> Int {
  if keyLengths.0 == keyLengths.1 {
    return ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: queryLength,
      sequenceDimensionB: keyLengths.0)
  }
  precondition(batchSize % 2 == 0)
  let halfBatch = batchSize / 2
  return ScaledDotProductAttentionInstructionCount(
    batchSize: halfBatch, heads: heads, headDimension: headDimension,
    sequenceDimensionA: queryLength,
    sequenceDimensionB: keyLengths.0)
    + ScaledDotProductAttentionInstructionCount(
      batchSize: halfBatch, heads: heads, headDimension: headDimension,
      sequenceDimensionA: queryLength,
      sequenceDimensionB: keyLengths.1)
}

private func UNetResBlockInstructionCount(
  batchSize: Int, height: Int, width: Int, inputChannels: Int, outChannels: Int,
  embChannels: Int = 1280,
  skipConnection: Bool
) -> Int {
  var total = 0
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: outChannels,
    kernelHeight: 3,
    kernelWidth: 3, inputChannels: inputChannels)
  total += DenseInstructionCount(rows: batchSize, input: embChannels, output: outChannels)
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: outChannels,
    kernelHeight: 3,
    kernelWidth: 3, inputChannels: outChannels)
  if skipConnection {
    total += ConvolutionInstructionCount(
      batchSize: batchSize, outHeight: height, outWidth: width, outChannels: outChannels,
      kernelHeight: 1,
      kernelWidth: 1, inputChannels: inputChannels)
  }
  return total
}

private func UNetTransformerBlockInstructionCount(
  batchSize: Int, hiddenSize: Int, heads: Int, headDimension: Int, hw: Int,
  embeddingLength: (Int, Int),
  injectIPAdapterLengths: [Int], injectedAttentionKV: Bool
) -> Int {
  let rowsQuery = batchSize * hw
  let contextRows = batchSize * max(embeddingLength.0, embeddingLength.1)
  let intermediateSize = hiddenSize * 4
  var total = 0

  // Self-attention q/k/v/o.
  if injectedAttentionKV {
    total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // q
    total +=
      2 * DenseInstructionCount(rows: batchSize * 2 * hw, input: hiddenSize, output: hiddenSize)  // k,v
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension, sequenceDimensionA: hw,
      sequenceDimensionB: 2 * hw)
  } else {
    total += 3 * DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // q,k,v
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension, sequenceDimensionA: hw,
      sequenceDimensionB: hw)
  }
  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // self o

  // Cross-attention q/k/v/o.
  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // q
  total += 2 * DenseInstructionCount(rows: contextRows, input: hiddenSize, output: hiddenSize)  // k,v
  total += UNetSplitAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: headDimension, queryLength: hw,
    keyLengths: embeddingLength)
  for injectIPAdapterLength in injectIPAdapterLengths {
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension, sequenceDimensionA: hw,
      sequenceDimensionB: injectIPAdapterLength)
  }
  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // cross o

  // GEGLU feed-forward: two up projections + one down.
  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: intermediateSize)
  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: intermediateSize)
  total += DenseInstructionCount(rows: rowsQuery, input: intermediateSize, output: hiddenSize)

  return total
}

private func UNetSpatialTransformerInstructionCount(
  batchSize: Int, channels: Int, heads: Int, height: Int, width: Int, embeddingLength: (Int, Int),
  injectIPAdapterLengths: [Int], injectedAttentionKV: Bool
) -> Int {
  let hw = height * width
  var total = 0
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: channels,
    kernelHeight: 1,
    kernelWidth: 1, inputChannels: channels)  // proj_in
  total += UNetTransformerBlockInstructionCount(
    batchSize: batchSize, hiddenSize: channels, heads: heads, headDimension: channels / heads,
    hw: hw,
    embeddingLength: embeddingLength, injectIPAdapterLengths: injectIPAdapterLengths,
    injectedAttentionKV: injectedAttentionKV)
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: channels,
    kernelHeight: 1,
    kernelWidth: 1, inputChannels: channels)  // proj_out
  return total
}

private func UNetBlockLayerInstructionCount(
  batchSize: Int, inputChannels: Int, channels: Int, heads: Int, height: Int, width: Int,
  embeddingLength: (Int, Int), attentionBlock: Bool, injectIPAdapterLengths: [Int],
  injectedAttentionKV: Bool
) -> Int {
  var total = UNetResBlockInstructionCount(
    batchSize: batchSize, height: height, width: width, inputChannels: inputChannels,
    outChannels: channels,
    skipConnection: inputChannels != channels)
  if attentionBlock {
    total += UNetSpatialTransformerInstructionCount(
      batchSize: batchSize, channels: channels, heads: heads, height: height, width: width,
      embeddingLength: embeddingLength, injectIPAdapterLengths: injectIPAdapterLengths,
      injectedAttentionKV: injectedAttentionKV)
  }
  return total
}

private func UNetCoreInstructionCount(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  channels: [Int], numRepeat: Int, headsForChannel: (Int) -> Int, injectIPAdapterLengths: [Int],
  injectAttentionKV: Bool
) -> Int {
  precondition(startWidth > 0 && startHeight > 0)
  let attentionRes: Set<Int> = [1, 2, 4]
  let embChannels = channels[0] * 4
  var total = 0

  // time_embed MLP (modelChannels=320 in current v1/v2 builders).
  total += DenseInstructionCount(rows: batchSize, input: channels[0], output: embChannels)
  total += DenseInstructionCount(rows: batchSize, input: embChannels, output: embChannels)

  // Input conv.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: startHeight, outWidth: startWidth, outChannels: channels[0],
    kernelHeight: 3, kernelWidth: 3, inputChannels: 4)

  // Input path.
  var passLayers: [UNetFeatureMapSpec] = [
    UNetFeatureMapSpec(channels: channels[0], height: startHeight, width: startWidth)
  ]
  var out = passLayers[0]
  var previousChannel = channels[0]
  var ds = 1
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes.contains(ds)
    for _ in 0..<numRepeat {
      total += UNetBlockLayerInstructionCount(
        batchSize: batchSize, inputChannels: previousChannel, channels: channel,
        heads: headsForChannel(channel),
        height: out.height, width: out.width, embeddingLength: embeddingLength,
        attentionBlock: attentionBlock,
        injectIPAdapterLengths: injectIPAdapterLengths,
        injectedAttentionKV: injectAttentionKV && attentionBlock)
      previousChannel = channel
      out = UNetFeatureMapSpec(channels: channel, height: out.height, width: out.width)
      passLayers.append(out)
    }
    if i != channels.count - 1 {
      total += ConvolutionInstructionCount(
        batchSize: batchSize, outHeight: out.height / 2, outWidth: out.width / 2,
        outChannels: channel,
        kernelHeight: 3, kernelWidth: 3, inputChannels: channel)
      out = UNetFeatureMapSpec(channels: channel, height: out.height / 2, width: out.width / 2)
      passLayers.append(out)
      ds *= 2
    }
  }

  // Middle block: Res + SpatialTransformer + Res at deepest resolution.
  let deepest = out
  let middleHeads = headsForChannel(deepest.channels)
  total += UNetResBlockInstructionCount(
    batchSize: batchSize, height: deepest.height, width: deepest.width,
    inputChannels: deepest.channels,
    outChannels: deepest.channels, skipConnection: false)
  total += UNetSpatialTransformerInstructionCount(
    batchSize: batchSize, channels: deepest.channels, heads: middleHeads, height: deepest.height,
    width: deepest.width,
    embeddingLength: embeddingLength, injectIPAdapterLengths: injectIPAdapterLengths,
    injectedAttentionKV: injectAttentionKV)
  total += UNetResBlockInstructionCount(
    batchSize: batchSize, height: deepest.height, width: deepest.width,
    inputChannels: deepest.channels,
    outChannels: deepest.channels, skipConnection: false)

  // Output path.
  var heights = [Int]()
  var widths = [Int]()
  var dss = [Int]()
  var h = startHeight
  var w = startWidth
  ds = 1
  heights.append(h)
  widths.append(w)
  dss.append(ds)
  for _ in 0..<(channels.count - 1) {
    h /= 2
    w /= 2
    ds *= 2
    heights.append(h)
    widths.append(w)
    dss.append(ds)
  }

  out = deepest
  var inputIdx = passLayers.count - 1
  for i in stride(from: channels.count - 1, through: 0, by: -1) {
    let channel = channels[i]
    let blockHeight = heights[i]
    let blockWidth = widths[i]
    let blockDs = dss[i]
    let attentionBlock = attentionRes.contains(blockDs)
    for j in 0..<(numRepeat + 1) {
      let skip = passLayers[inputIdx]
      inputIdx -= 1
      let concatChannels = out.channels + skip.channels
      total += UNetBlockLayerInstructionCount(
        batchSize: batchSize, inputChannels: concatChannels, channels: channel,
        heads: headsForChannel(channel),
        height: blockHeight, width: blockWidth, embeddingLength: embeddingLength,
        attentionBlock: attentionBlock,
        injectIPAdapterLengths: injectIPAdapterLengths,
        injectedAttentionKV: injectAttentionKV && attentionBlock)
      out = UNetFeatureMapSpec(channels: channel, height: blockHeight, width: blockWidth)
      if i > 0 && j == numRepeat {
        total += ConvolutionInstructionCount(
          batchSize: batchSize, outHeight: blockHeight * 2, outWidth: blockWidth * 2,
          outChannels: channel,
          kernelHeight: 3, kernelWidth: 3, inputChannels: channel)
        out = UNetFeatureMapSpec(channels: channel, height: blockHeight * 2, width: blockWidth * 2)
      }
    }
  }

  // Final out conv.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: startHeight, outWidth: startWidth, outChannels: 4,
    kernelHeight: 3,
    kernelWidth: 3, inputChannels: channels[0])

  return total
}

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// Ignores injected controls / T2I adapters because they only add tensors.
public func UNetInstructionCount(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int,
  injectIPAdapterLengths: [Int], injectAttentionKV: Bool
) -> Int {
  UNetCoreInstructionCount(
    batchSize: batchSize, embeddingLength: embeddingLength, startWidth: startWidth,
    startHeight: startHeight,
    channels: [320, 640, 1280, 1280], numRepeat: 2, headsForChannel: { _ in 8 },
    injectIPAdapterLengths: injectIPAdapterLengths, injectAttentionKV: injectAttentionKV)
}

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
public func UNetv2InstructionCount(
  batchSize: Int, embeddingLength: (Int, Int), startWidth: Int, startHeight: Int
) -> Int {
  UNetCoreInstructionCount(
    batchSize: batchSize, embeddingLength: embeddingLength, startWidth: startWidth,
    startHeight: startHeight,
    channels: [320, 640, 1280, 1280], numRepeat: 2, headsForChannel: { $0 / 64 },
    injectIPAdapterLengths: [], injectAttentionKV: false)
}
