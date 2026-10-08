import Foundation

private struct KandinskyFeatureMapSpec {
  let channels: Int
  let height: Int
  let width: Int
}

private func KandinskyResBlockInstructionCount(
  batchSize: Int, height: Int, width: Int, inputChannels: Int, outChannels: Int,
  embeddingChannels: Int,
  up: Bool, down: Bool, skipConnection: Bool
) -> Int {
  let outHeight = up ? (height * 2) : (down ? (height / 2) : height)
  let outWidth = up ? (width * 2) : (down ? (width / 2) : width)
  var total = 0
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: outHeight, outWidth: outWidth, outChannels: outChannels,
    kernelHeight: 3,
    kernelWidth: 3, inputChannels: inputChannels)
  total += DenseInstructionCount(rows: batchSize, input: embeddingChannels, output: 2 * outChannels)
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: outHeight, outWidth: outWidth, outChannels: outChannels,
    kernelHeight: 3,
    kernelWidth: 3, inputChannels: outChannels)
  if skipConnection {
    total += ConvolutionInstructionCount(
      batchSize: batchSize, outHeight: outHeight, outWidth: outWidth, outChannels: outChannels,
      kernelHeight: 1,
      kernelWidth: 1, inputChannels: inputChannels)
  }
  return total
}

private func KandinskyAttentionBlockInstructionCount(
  batchSize: Int, channels: Int, numHeadChannels: Int, t: Int, height: Int, width: Int
) -> Int {
  precondition(channels % numHeadChannels == 0)
  let heads = channels / numHeadChannels
  let hw = height * width
  let rowsLocal = batchSize * hw
  let rowsEncoder = batchSize * t
  var total = 0

  // Encoder K/V + local Q/K/V + output projection.
  total += 2 * DenseInstructionCount(rows: rowsEncoder, input: channels, output: channels)
  total += 3 * DenseInstructionCount(rows: rowsLocal, input: channels, output: channels)
  total += DenseInstructionCount(rows: rowsLocal, input: channels, output: channels)

  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: numHeadChannels, sequenceDimensionA: hw,
    sequenceDimensionB: t + hw)
  return total
}

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// `inputChannels` is extra because `UNetKandinsky(...)` builder signature does not expose it.
public func UNetKandinskyInstructionCount(
  batchSize: Int, channels: Int, outChannels: Int, channelMult: [Int], numResBlocks: Int,
  numHeadChannels: Int, t: Int, startHeight: Int, startWidth: Int, attentionResolutions: Set<Int>,
  inputChannels: Int = 4
) -> Int {
  precondition(startHeight > 0 && startWidth > 0)
  let embeddingChannels = channels * 4
  var total = 0

  // Simulate input blocks and capture skip tensors (`hs`) shapes/channels.
  var hs: [KandinskyFeatureMapSpec] = []
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: startHeight, outWidth: startWidth, outChannels: channels,
    kernelHeight: 3,
    kernelWidth: 3, inputChannels: inputChannels)
  var out = KandinskyFeatureMapSpec(channels: channels, height: startHeight, width: startWidth)
  hs.append(out)

  var lastCh = channels
  var ds = 1
  for (level, mult) in channelMult.enumerated() {
    let ch = channels * mult
    for _ in 0..<numResBlocks {
      total += KandinskyResBlockInstructionCount(
        batchSize: batchSize, height: out.height, width: out.width, inputChannels: lastCh,
        outChannels: ch,
        embeddingChannels: embeddingChannels, up: false, down: false, skipConnection: ch != lastCh)
      out = KandinskyFeatureMapSpec(channels: ch, height: out.height, width: out.width)
      lastCh = ch
      if attentionResolutions.contains(ds) {
        total += KandinskyAttentionBlockInstructionCount(
          batchSize: batchSize, channels: ch, numHeadChannels: numHeadChannels, t: t,
          height: out.height,
          width: out.width)
      }
      hs.append(out)
    }
    if level != channelMult.count - 1 {
      total += KandinskyResBlockInstructionCount(
        batchSize: batchSize, height: out.height, width: out.width, inputChannels: ch,
        outChannels: ch,
        embeddingChannels: embeddingChannels, up: false, down: true, skipConnection: false)
      out = KandinskyFeatureMapSpec(channels: ch, height: out.height / 2, width: out.width / 2)
      hs.append(out)
      ds *= 2
    }
  }

  // Middle blocks.
  let ch = channelMult.last! * channels
  total += KandinskyResBlockInstructionCount(
    batchSize: batchSize, height: out.height, width: out.width, inputChannels: ch, outChannels: ch,
    embeddingChannels: embeddingChannels, up: false, down: false, skipConnection: false)
  total += KandinskyAttentionBlockInstructionCount(
    batchSize: batchSize, channels: ch, numHeadChannels: numHeadChannels, t: t, height: out.height,
    width: out.width)
  total += KandinskyResBlockInstructionCount(
    batchSize: batchSize, height: out.height, width: out.width, inputChannels: ch, outChannels: ch,
    embeddingChannels: embeddingChannels, up: false, down: false, skipConnection: false)

  // Output blocks.
  var outputDs = 1
  var outputHeight = startHeight
  var outputWidth = startWidth
  for _ in 1..<channelMult.count {
    outputDs *= 2
    outputHeight /= 2
    outputWidth /= 2
  }
  var hsIndex = hs.count - 1
  for (level, mult) in channelMult.enumerated().reversed() {
    let chLevel = channels * mult
    for j in 0..<(numResBlocks + 1) {
      let skip = hs[hsIndex]
      hsIndex -= 1
      total += KandinskyResBlockInstructionCount(
        batchSize: batchSize, height: outputHeight, width: outputWidth,
        inputChannels: out.channels + skip.channels,
        outChannels: chLevel, embeddingChannels: embeddingChannels, up: false, down: false,
        skipConnection: true)
      out = KandinskyFeatureMapSpec(channels: chLevel, height: outputHeight, width: outputWidth)
      if attentionResolutions.contains(outputDs) {
        total += KandinskyAttentionBlockInstructionCount(
          batchSize: batchSize, channels: chLevel, numHeadChannels: numHeadChannels, t: t,
          height: outputHeight,
          width: outputWidth)
      }
      if level > 0 && j == numResBlocks {
        total += KandinskyResBlockInstructionCount(
          batchSize: batchSize, height: outputHeight, width: outputWidth, inputChannels: chLevel,
          outChannels: chLevel,
          embeddingChannels: embeddingChannels, up: true, down: false, skipConnection: false)
        outputDs /= 2
        outputHeight *= 2
        outputWidth *= 2
        out = KandinskyFeatureMapSpec(channels: chLevel, height: outputHeight, width: outputWidth)
      }
    }
  }

  // Final out conv.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: startHeight, outWidth: startWidth, outChannels: outChannels,
    kernelHeight: 3,
    kernelWidth: 3, inputChannels: channels)

  return total
}
