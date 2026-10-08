import Foundation

private struct UNetXLFeatureMapSpec {
  let channels: Int
  let height: Int
  let width: Int
}

private func UNetXLResBlockInstructionCount(
  batchSize: Int, height: Int, width: Int, inputChannels: Int, outChannels: Int, embChannels: Int,
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

private func UNetXLTimeResBlockInstructionCount(
  batchSize: Int, height: Int, width: Int, channels: Int, embChannels: Int
) -> Int {
  let hw = height * width
  var total = 0
  // TimeResBlock reshapes to [1, b, hw, c] and uses 3x1 convs.
  total += ConvolutionInstructionCount(
    batchSize: 1, outHeight: batchSize, outWidth: hw, outChannels: channels, kernelHeight: 3,
    kernelWidth: 1, inputChannels: channels)
  total += DenseInstructionCount(rows: batchSize, input: embChannels, output: channels)
  total += ConvolutionInstructionCount(
    batchSize: 1, outHeight: batchSize, outWidth: hw, outChannels: channels, kernelHeight: 3,
    kernelWidth: 1, inputChannels: channels)
  return total
}

private func UNetXLSelfAttentionInstructionCount(
  batchSize: Int, sequenceLength: Int, hiddenSize: Int, heads: Int
) -> Int {
  let headDimension = hiddenSize / heads
  let rows = batchSize * sequenceLength
  var total = 0
  total += 4 * DenseInstructionCount(rows: rows, input: hiddenSize, output: hiddenSize)  // q,k,v,o
  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: headDimension,
    sequenceDimensionA: sequenceLength,
    sequenceDimensionB: sequenceLength)
  return total
}

private func UNetXLSplitCrossAttentionInstructionCount(
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

private func UNetXLCrossAttentionQOInstructionCount(
  batchSize: Int, queryLength: Int, hiddenSize: Int, heads: Int, embeddingLength: (Int, Int),
  injectIPAdapterLengths: [Int]
) -> Int {
  let headDimension = hiddenSize / heads
  let rowsQuery = batchSize * queryLength
  var total = 0
  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // q
  total += UNetXLSplitCrossAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: headDimension, queryLength: queryLength,
    keyLengths: embeddingLength)
  for ipLen in injectIPAdapterLengths {
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: queryLength,
      sequenceDimensionB: ipLen)
  }
  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // o
  return total
}

private func UNetXLFeedForwardGEGLUInstructionCount(
  batchSize: Int, sequenceLength: Int, hiddenSize: Int, intermediateSize: Int
) -> Int {
  let rows = batchSize * sequenceLength
  return DenseInstructionCount(rows: rows, input: hiddenSize, output: intermediateSize)
    + DenseInstructionCount(rows: rows, input: hiddenSize, output: intermediateSize)
    + DenseInstructionCount(rows: rows, input: intermediateSize, output: hiddenSize)
}

private func UNetXLBasicTransformerBlockInstructionCount(
  batchSize: Int, hw: Int, hiddenSize: Int, heads: Int, embeddingLength: (Int, Int),
  injectIPAdapterLengths: [Int]
) -> Int {
  var total = 0
  total += UNetXLSelfAttentionInstructionCount(
    batchSize: batchSize, sequenceLength: hw, hiddenSize: hiddenSize, heads: heads)
  if !(embeddingLength.0 == 1 && embeddingLength.1 == 1) {
    total += UNetXLCrossAttentionQOInstructionCount(
      batchSize: batchSize, queryLength: hw, hiddenSize: hiddenSize, heads: heads,
      embeddingLength: embeddingLength, injectIPAdapterLengths: injectIPAdapterLengths)
  }
  total += UNetXLFeedForwardGEGLUInstructionCount(
    batchSize: batchSize, sequenceLength: hw, hiddenSize: hiddenSize,
    intermediateSize: hiddenSize * 4)
  return total
}

private func UNetXLBasicTimeTransformerBlockInstructionCount(
  batchSize: Int, hw: Int, hiddenSize: Int, heads: Int, embeddingLength: (Int, Int),
  injectIPAdapterLengths: [Int]
) -> Int {
  // In time transformer, attention is over temporal length `batchSize`, batched across `hw`.
  var total = 0
  total += UNetXLFeedForwardGEGLUInstructionCount(
    batchSize: hw, sequenceLength: batchSize, hiddenSize: hiddenSize,
    intermediateSize: hiddenSize * 4)  // ff_in
  total += UNetXLSelfAttentionInstructionCount(
    batchSize: hw, sequenceLength: batchSize, hiddenSize: hiddenSize, heads: heads)
  if !(embeddingLength.0 == 1 && embeddingLength.1 == 1) {
    total += UNetXLCrossAttentionQOInstructionCount(
      batchSize: hw, queryLength: batchSize, hiddenSize: hiddenSize, heads: heads,
      embeddingLength: embeddingLength, injectIPAdapterLengths: injectIPAdapterLengths)
  }
  total += UNetXLFeedForwardGEGLUInstructionCount(
    batchSize: hw, sequenceLength: batchSize, hiddenSize: hiddenSize,
    intermediateSize: hiddenSize * 4)
  return total
}

private func UNetXLTimePosEmbedInstructionCount(frameRows: Int, hiddenSize: Int) -> Int {
  DenseInstructionCount(rows: frameRows, input: hiddenSize, output: hiddenSize * 4)
    + DenseInstructionCount(rows: frameRows, input: hiddenSize * 4, output: hiddenSize)
}

private func UNetXLSpatialTransformerInstructionCount(
  batchSize: Int, frameRowsForTemporalEmbed: Int, channels: Int, height: Int, width: Int,
  depth: Int,
  embeddingLength: (Int, Int), injectIPAdapterLengths: [Int], isTemporalMixEnabled: Bool
) -> Int {
  let hw = height * width
  let heads = channels / 64
  var total = 0
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: channels,
    kernelHeight: 1,
    kernelWidth: 1, inputChannels: channels)  // proj_in
  if depth > 0 {
    if isTemporalMixEnabled {
      total += UNetXLTimePosEmbedInstructionCount(
        frameRows: frameRowsForTemporalEmbed, hiddenSize: channels)
    }
    for _ in 0..<depth {
      total += UNetXLBasicTransformerBlockInstructionCount(
        batchSize: batchSize, hw: hw, hiddenSize: channels, heads: heads,
        embeddingLength: embeddingLength,
        injectIPAdapterLengths: injectIPAdapterLengths)
      if isTemporalMixEnabled {
        total += UNetXLBasicTimeTransformerBlockInstructionCount(
          batchSize: batchSize, hw: hw, hiddenSize: channels, heads: heads,
          embeddingLength: embeddingLength,
          injectIPAdapterLengths: injectIPAdapterLengths)
      }
    }
  }
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: channels,
    kernelHeight: 1,
    kernelWidth: 1, inputChannels: channels)  // proj_out
  return total
}

private func UNetXLBlockLayerInstructionCount(
  batchSize: Int, frameRowsForTemporalEmbed: Int, inputChannels: Int, channels: Int, height: Int,
  width: Int,
  attentionDepth: Int, embeddingLength: (Int, Int), injectIPAdapterLengths: [Int], embChannels: Int,
  isTemporalMixEnabled: Bool
) -> Int {
  var total = 0
  total += UNetXLResBlockInstructionCount(
    batchSize: batchSize, height: height, width: width, inputChannels: inputChannels,
    outChannels: channels,
    embChannels: embChannels, skipConnection: inputChannels != channels)
  if isTemporalMixEnabled {
    total += UNetXLTimeResBlockInstructionCount(
      batchSize: batchSize, height: height, width: width, channels: channels,
      embChannels: embChannels)
  }
  if attentionDepth > 0 {
    total += UNetXLSpatialTransformerInstructionCount(
      batchSize: batchSize, frameRowsForTemporalEmbed: frameRowsForTemporalEmbed,
      channels: channels,
      height: height, width: width, depth: attentionDepth, embeddingLength: embeddingLength,
      injectIPAdapterLengths: injectIPAdapterLengths, isTemporalMixEnabled: isTemporalMixEnabled)
  }
  return total
}

private func UNetXLBasicTransformerBlockFixedInstructionCount(
  batchSize: Int, hiddenSize: Int, embeddingLength: (Int, Int)
) -> Int {
  if embeddingLength.0 == 1 && embeddingLength.1 == 1 {
    // Attention1Fixed: to_v + out_proj on one token per batch.
    return DenseInstructionCount(rows: batchSize, input: hiddenSize, output: hiddenSize)
      + DenseInstructionCount(rows: batchSize, input: hiddenSize, output: hiddenSize)
  }
  let rows = batchSize * max(embeddingLength.0, embeddingLength.1)
  return 2 * DenseInstructionCount(rows: rows, input: hiddenSize, output: hiddenSize)  // k,v
}

private func UNetXLSpatialTransformerFixedInstructionCount(
  batchSize: Int, frameRowsForTemporalEmbed: Int, channels: Int, depth: Int,
  embeddingLength: (Int, Int),
  isTemporalMixEnabled: Bool
) -> Int {
  var total = 0
  if depth > 0 && isTemporalMixEnabled {
    total += UNetXLTimePosEmbedInstructionCount(
      frameRows: frameRowsForTemporalEmbed, hiddenSize: channels)
  }
  for _ in 0..<depth {
    total += UNetXLBasicTransformerBlockFixedInstructionCount(
      batchSize: batchSize, hiddenSize: channels, embeddingLength: embeddingLength)
    if isTemporalMixEnabled {
      total += UNetXLBasicTransformerBlockFixedInstructionCount(
        batchSize: batchSize, hiddenSize: channels, embeddingLength: embeddingLength)
    }
  }
  return total
}

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// Extra `labelInputChannels` matches the `y` input width used by the caller (e.g. SDXL pooled+time ids).
public func UNetXLInstructionCount(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int,
  outputAttentionRes: KeyValuePairs<Int, [Int]>, embeddingLength: (Int, Int),
  injectIPAdapterLengths: [Int], isTemporalMixEnabled: Bool,
  labelInputChannels: Int = 2816
) -> Int {
  precondition(!channels.isEmpty)
  let embChannels = channels[0] * 4
  let inputAttention = Dictionary(
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttention = Dictionary(
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let frameRowsForTemporalEmbed = batchSize
  var total = 0

  // Time and label embedders.
  total += DenseInstructionCount(rows: batchSize, input: channels[0], output: embChannels)
  total += DenseInstructionCount(rows: batchSize, input: embChannels, output: embChannels)
  total += DenseInstructionCount(rows: batchSize, input: labelInputChannels, output: embChannels)
  total += DenseInstructionCount(rows: batchSize, input: embChannels, output: embChannels)

  // Input conv.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: startHeight, outWidth: startWidth, outChannels: channels[0],
    kernelHeight: 3, kernelWidth: 3, inputChannels: 4)

  // Input blocks.
  var passLayers: [UNetXLFeatureMapSpec] = [
    UNetXLFeatureMapSpec(channels: channels[0], height: startHeight, width: startWidth)
  ]
  var out = passLayers[0]
  var previousChannel = channels[0]
  var ds = 1
  for (i, channel) in channels.enumerated() {
    let attentionDepths = inputAttention[ds, default: Array(repeating: 0, count: 2)]
    for j in 0..<2 {
      total += UNetXLBlockLayerInstructionCount(
        batchSize: batchSize, frameRowsForTemporalEmbed: frameRowsForTemporalEmbed,
        inputChannels: previousChannel,
        channels: channel, height: out.height, width: out.width, attentionDepth: attentionDepths[j],
        embeddingLength: embeddingLength, injectIPAdapterLengths: injectIPAdapterLengths,
        embChannels: embChannels,
        isTemporalMixEnabled: isTemporalMixEnabled)
      previousChannel = channel
      out = UNetXLFeatureMapSpec(channels: channel, height: out.height, width: out.width)
      passLayers.append(out)
    }
    if i != channels.count - 1 {
      total += ConvolutionInstructionCount(
        batchSize: batchSize, outHeight: out.height / 2, outWidth: out.width / 2,
        outChannels: channel,
        kernelHeight: 3, kernelWidth: 3, inputChannels: channel)
      out = UNetXLFeatureMapSpec(channels: channel, height: out.height / 2, width: out.width / 2)
      passLayers.append(out)
      ds *= 2
    }
  }

  // Middle block.
  let deepest = out
  total += UNetXLResBlockInstructionCount(
    batchSize: batchSize, height: deepest.height, width: deepest.width,
    inputChannels: deepest.channels,
    outChannels: deepest.channels, embChannels: embChannels, skipConnection: false)
  if isTemporalMixEnabled {
    total += UNetXLTimeResBlockInstructionCount(
      batchSize: batchSize, height: deepest.height, width: deepest.width,
      channels: deepest.channels,
      embChannels: embChannels)
  }
  if middleAttentionBlocks > 0 {
    total += UNetXLSpatialTransformerInstructionCount(
      batchSize: batchSize, frameRowsForTemporalEmbed: frameRowsForTemporalEmbed,
      channels: deepest.channels,
      height: deepest.height, width: deepest.width, depth: middleAttentionBlocks,
      embeddingLength: embeddingLength,
      injectIPAdapterLengths: injectIPAdapterLengths, isTemporalMixEnabled: isTemporalMixEnabled)
    total += UNetXLResBlockInstructionCount(
      batchSize: batchSize, height: deepest.height, width: deepest.width,
      inputChannels: deepest.channels,
      outChannels: deepest.channels, embChannels: embChannels, skipConnection: false)
    if isTemporalMixEnabled {
      total += UNetXLTimeResBlockInstructionCount(
        batchSize: batchSize, height: deepest.height, width: deepest.width,
        channels: deepest.channels,
        embChannels: embChannels)
    }
  }

  // Output blocks.
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
    let attentionDepths = outputAttention[blockDs, default: Array(repeating: 0, count: 3)]
    for j in 0..<3 {
      let skip = passLayers[inputIdx]
      inputIdx -= 1
      total += UNetXLBlockLayerInstructionCount(
        batchSize: batchSize, frameRowsForTemporalEmbed: frameRowsForTemporalEmbed,
        inputChannels: out.channels + skip.channels, channels: channel, height: blockHeight,
        width: blockWidth,
        attentionDepth: attentionDepths[j], embeddingLength: embeddingLength,
        injectIPAdapterLengths: injectIPAdapterLengths, embChannels: embChannels,
        isTemporalMixEnabled: isTemporalMixEnabled)
      out = UNetXLFeatureMapSpec(channels: channel, height: blockHeight, width: blockWidth)
      if i > 0 && j == 2 {
        total += ConvolutionInstructionCount(
          batchSize: batchSize, outHeight: blockHeight * 2, outWidth: blockWidth * 2,
          outChannels: channel,
          kernelHeight: 3, kernelWidth: 3, inputChannels: channel)
        out = UNetXLFeatureMapSpec(
          channels: channel, height: blockHeight * 2, width: blockWidth * 2)
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

// Counts only Dense (GEMM) instructions for `UNetXLFixed(...)`.
// `temporalFrameEmbeddingRows` lets SVD callers pass the number of frame embeddings used for
// `time_pos_embed`; defaults to `batchSize` which matches non-temporal usage.
public func UNetXLFixedInstructionCount(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int], embeddingLength: (Int, Int),
  inputAttentionRes: KeyValuePairs<Int, [Int]>, middleAttentionBlocks: Int,
  outputAttentionRes: KeyValuePairs<Int, [Int]>, isTemporalMixEnabled: Bool,
  temporalFrameEmbeddingRows: Int? = nil
) -> Int {
  let inputAttention = Dictionary(
    uniqueKeysWithValues: inputAttentionRes.map { ($0.key, $0.value) })
  let outputAttention = Dictionary(
    uniqueKeysWithValues: outputAttentionRes.map { ($0.key, $0.value) })
  let frameRows = temporalFrameEmbeddingRows ?? batchSize
  var total = 0

  // Input attention blocks.
  var ds = 1
  for channel in channels {
    let attentionDepths = inputAttention[ds, default: [0, 0]]
    for depth in attentionDepths where depth > 0 {
      total += UNetXLSpatialTransformerFixedInstructionCount(
        batchSize: batchSize, frameRowsForTemporalEmbed: frameRows, channels: channel, depth: depth,
        embeddingLength: embeddingLength, isTemporalMixEnabled: isTemporalMixEnabled)
    }
    ds *= 2
  }

  // Middle attention block.
  if middleAttentionBlocks > 0, let channel = channels.last {
    total += UNetXLSpatialTransformerFixedInstructionCount(
      batchSize: batchSize, frameRowsForTemporalEmbed: frameRows, channels: channel,
      depth: middleAttentionBlocks,
      embeddingLength: embeddingLength, isTemporalMixEnabled: isTemporalMixEnabled)
  }

  // Output attention blocks.
  ds = 1
  var dss = [Int]()
  for _ in channels.indices {
    dss.append(ds)
    ds *= 2
  }
  for (i, channel) in channels.enumerated().reversed() {
    let attentionDepths = outputAttention[dss[i], default: [0, 0, 0]]
    for depth in attentionDepths where depth > 0 {
      total += UNetXLSpatialTransformerFixedInstructionCount(
        batchSize: batchSize, frameRowsForTemporalEmbed: frameRows, channels: channel, depth: depth,
        embeddingLength: embeddingLength, isTemporalMixEnabled: isTemporalMixEnabled)
    }
  }

  _ = startHeight
  _ = startWidth
  return total
}
