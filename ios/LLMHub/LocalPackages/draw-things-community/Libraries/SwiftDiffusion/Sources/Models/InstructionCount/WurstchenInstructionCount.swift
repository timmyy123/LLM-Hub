import Foundation

private func WurstchenResBlockInstructionCount(
  batchSize: Int, height: Int, width: Int, channels: Int, skip: Bool
) -> Int {
  var total = 0
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: channels,
    kernelHeight: 3,
    kernelWidth: 3, inputChannels: channels, groups: channels)
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: channels * 4,
    kernelHeight: 1, kernelWidth: 1, inputChannels: skip ? (channels * 2) : channels)
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: channels,
    kernelHeight: 1,
    kernelWidth: 1, inputChannels: channels * 4)
  return total
}

private func WurstchenTimestepBlockInstructionCount(
  batchSize: Int, timeEmbedSize: Int, channels: Int, tCondCount: Int
) -> Int {
  (1 + tCondCount)
    * DenseInstructionCount(rows: batchSize, input: timeEmbedSize, output: channels * 2)
}

private func WurstchenAttentionInstructionCount(
  batchSize: Int, height: Int, width: Int, channels: Int, nHead: Int, t: (Int, Int)
) -> Int {
  precondition(channels % nHead == 0)
  let hw = height * width
  let rows = batchSize * hw
  let headDimension = channels / nHead
  var total = 0

  // q/k/v projections on local feature map + output projection (counted even when fused with SDPA).
  total += 4 * DenseInstructionCount(rows: rows, input: channels, output: channels)

  if t.0 == t.1 || batchSize == 1 {
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: nHead, headDimension: headDimension, sequenceDimensionA: hw,
      sequenceDimensionB: hw + t.0)
  } else {
    precondition(batchSize % 2 == 0)
    let b0 = batchSize / 2
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: b0, heads: nHead, headDimension: headDimension, sequenceDimensionA: hw,
      sequenceDimensionB: hw + t.0)
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: b0, heads: nHead, headDimension: headDimension, sequenceDimensionA: hw,
      sequenceDimensionB: hw + t.1)
  }

  return total
}

private func WurstchenAttnBlockFixedInstructionCount(
  batchSize: Int, kvLength: Int, kvInputChannels: Int, channels: Int
) -> Int {
  let rows = batchSize * kvLength
  var total = 0
  total += DenseInstructionCount(rows: rows, input: kvInputChannels, output: channels)  // kv_mapper
  total += DenseInstructionCount(rows: rows, input: channels, output: channels)  // keys
  total += DenseInstructionCount(rows: rows, input: channels, output: channels)  // values
  return total
}

private func WurstchenSpatialMapperInstructionCount(
  batchSize: Int, height: Int, width: Int, inputChannels: Int, cHidden: Int
) -> Int {
  var total = 0
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: cHidden * 4,
    kernelHeight: 1, kernelWidth: 1, inputChannels: inputChannels)
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: cHidden, kernelHeight: 1,
    kernelWidth: 1, inputChannels: cHidden * 4)
  return total
}

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// Includes depthwise/pointwise and transposed convolutions as convolution counts.
public func WurstchenStageCInstructionCount(
  batchSize: Int, height: Int, width: Int, t: (Int, Int), inputChannels: Int = 16
) -> Int {
  let blocks: [[Int]] = [[8, 24], [24, 8]]
  var total = 0

  // embedding conv: cIn -> 2048 (1x1).
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: 2048, kernelHeight: 1,
    kernelWidth: 1, inputChannels: inputChannels)

  // Down path (32 blocks total), with one extra 1x1 downscaler between levels.
  for i in 0..<2 {
    if i > 0 {
      total += ConvolutionInstructionCount(
        batchSize: batchSize, outHeight: height, outWidth: width, outChannels: 2048,
        kernelHeight: 1,
        kernelWidth: 1, inputChannels: 2048)
    }
    for _ in 0..<blocks[0][i] {
      total += WurstchenResBlockInstructionCount(
        batchSize: batchSize, height: height, width: width, channels: 2048, skip: false)
      total += WurstchenTimestepBlockInstructionCount(
        batchSize: batchSize, timeEmbedSize: 64, channels: 2048, tCondCount: 2)
      total += WurstchenAttentionInstructionCount(
        batchSize: batchSize, height: height, width: width, channels: 2048, nHead: 32, t: t)
    }
  }

  // Up path (32 blocks total), with one skip-enabled ResBlock and one extra 1x1 upscaler.
  var consumedSkip = false
  for i in 0..<2 {
    for j in 0..<blocks[1][i] {
      let useSkip = !consumedSkip && i == 1 && j == 0
      if useSkip {
        consumedSkip = true
      }
      total += WurstchenResBlockInstructionCount(
        batchSize: batchSize, height: height, width: width, channels: 2048, skip: useSkip)
      total += WurstchenTimestepBlockInstructionCount(
        batchSize: batchSize, timeEmbedSize: 64, channels: 2048, tCondCount: 2)
      total += WurstchenAttentionInstructionCount(
        batchSize: batchSize, height: height, width: width, channels: 2048, nHead: 32, t: t)
    }
    if i < 1 {
      total += ConvolutionInstructionCount(
        batchSize: batchSize, outHeight: height, outWidth: width, outChannels: 2048,
        kernelHeight: 1,
        kernelWidth: 1, inputChannels: 2048)
    }
  }

  // Output conv: 2048 -> 16 (1x1).
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: height, outWidth: width, outChannels: 16, kernelHeight: 1,
    kernelWidth: 1, inputChannels: 2048)

  return total
}

// `t` here is the final clip sequence lengths passed to `WurstchenStageCFixed(...)`, which includes
// the +8 pooled/image tokens used by the builder call in `UNetFixedEncoder`.
public func WurstchenStageCFixedInstructionCount(
  batchSize: Int, t: (Int, Int), clipTextInputChannels: Int = 1280,
  clipPooledInputChannels: Int = 1280,
  clipImageInputChannels: Int = 1280
) -> Int {
  precondition(t.0 >= 8 && t.1 >= 8)
  let clipTextLength = max(t.0, t.1) - 8
  let clipLength = max(t.0, t.1)
  let attentionBlocks = 8 + 24 + 24 + 8
  var total = 0

  // CLIP/text/image mappers.
  total += DenseInstructionCount(
    rows: batchSize * clipTextLength, input: clipTextInputChannels, output: 2048)
  total += DenseInstructionCount(
    rows: batchSize, input: clipPooledInputChannels, output: 2048 * 4)
  total += DenseInstructionCount(
    rows: batchSize, input: clipImageInputChannels, output: 2048 * 4)

  // Per-attention-block fixed KV emitters on normalized clip sequence.
  for _ in 0..<attentionBlocks {
    total += WurstchenAttnBlockFixedInstructionCount(
      batchSize: batchSize, kvLength: clipLength, kvInputChannels: 2048, channels: 2048)
  }

  return total
}

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
public func WurstchenStageBInstructionCount(
  batchSize: Int, cIn: Int, height: Int, width: Int
) -> Int {
  precondition(height % 16 == 0 && width % 16 == 0)

  let cHidden = [320, 640, 1280, 1280]
  let blocks: [[Int]] = [[2, 6, 28, 6], [6, 28, 6, 2]]
  let attentions: [[Bool]] = [[false, false, true, true], [true, true, false, false]]
  let blockRepeat = [3, 3, 2, 2]
  var total = 0

  var h = height / 2
  var w = width / 2

  // embedding conv: cIn -> 320, 2x2 stride 2.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: h, outWidth: w, outChannels: cHidden[0], kernelHeight: 2,
    kernelWidth: 2, inputChannels: cIn)

  // Down path.
  for i in 0..<4 {
    if i > 0 {
      let nextH = h / 2
      let nextW = w / 2
      total += ConvolutionInstructionCount(
        batchSize: batchSize, outHeight: nextH, outWidth: nextW, outChannels: cHidden[i],
        kernelHeight: 2, kernelWidth: 2, inputChannels: cHidden[i - 1])
      h = nextH
      w = nextW
    }
    let attention = attentions[0][i]
    for _ in 0..<blocks[0][i] {
      total += WurstchenResBlockInstructionCount(
        batchSize: batchSize, height: h, width: w, channels: cHidden[i], skip: false)
      total += WurstchenTimestepBlockInstructionCount(
        batchSize: batchSize, timeEmbedSize: 64, channels: cHidden[i], tCondCount: 1)
      if attention {
        total += WurstchenAttentionInstructionCount(
          batchSize: batchSize, height: h, width: w, channels: cHidden[i], nHead: 20, t: (4, 4))
      }
    }
  }

  // Up path with repeated execution.
  for i in 0..<4 {
    let channels = cHidden[3 - i]
    let attention = attentions[1][i]
    let repeats = blockRepeat[i]
    let blockCount = blocks[1][i]
    let stageHasSkip = (i > 0)

    for _ in 0..<repeats {
      total += WurstchenResBlockInstructionCount(
        batchSize: batchSize, height: h, width: w, channels: channels, skip: stageHasSkip)
      if blockCount > 1 {
        total +=
          (blockCount - 1)
          * WurstchenResBlockInstructionCount(
            batchSize: batchSize, height: h, width: w, channels: channels, skip: false)
      }
      total +=
        blockCount
        * WurstchenTimestepBlockInstructionCount(
          batchSize: batchSize, timeEmbedSize: 64, channels: channels, tCondCount: 1)
      if attention {
        total +=
          blockCount
          * WurstchenAttentionInstructionCount(
            batchSize: batchSize, height: h, width: w, channels: channels, nHead: 20, t: (4, 4))
      }
    }

    if repeats > 1 {
      total +=
        (repeats - 1)
        * ConvolutionInstructionCount(
          batchSize: batchSize, outHeight: h, outWidth: w, outChannels: channels, kernelHeight: 1,
          kernelWidth: 1, inputChannels: channels)
    }

    if i < 3 {
      let nextH = h * 2
      let nextW = w * 2
      total += ConvolutionInstructionCount(
        batchSize: batchSize, outHeight: nextH, outWidth: nextW, outChannels: cHidden[2 - i],
        kernelHeight: 2, kernelWidth: 2, inputChannels: channels)
      h = nextH
      w = nextW
    }
  }

  // Final 1x1 conv before unpatchify: 320 -> 16.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: h, outWidth: w, outChannels: 16, kernelHeight: 1,
    kernelWidth: 1, inputChannels: cHidden[0])

  return total
}

public func WurstchenStageBFixedInstructionCount(
  batchSize: Int, height: Int, width: Int, effnetHeight: Int, effnetWidth: Int,
  effnetChannels: Int = 16,
  pixelsChannels: Int = 3, clipInputChannels: Int = 1280
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  let cHidden = [320, 640, 1280, 1280]
  let blocks: [[Int]] = [[2, 6, 28, 6], [6, 28, 6, 2]]
  let attentions: [[Bool]] = [[false, false, true, true], [true, true, false, false]]
  let halfH = height / 2
  let halfW = width / 2
  var total = 0

  // effnet_mapper and pixels_mapper.
  total += WurstchenSpatialMapperInstructionCount(
    batchSize: batchSize, height: halfH, width: halfW, inputChannels: effnetChannels,
    cHidden: cHidden[0])
  total += WurstchenSpatialMapperInstructionCount(
    batchSize: batchSize, height: 8, width: 8, inputChannels: pixelsChannels, cHidden: cHidden[0])

  // clip_mapper: [B,1,1280] -> [B,1,5120]
  total += DenseInstructionCount(rows: batchSize, input: clipInputChannels, output: 1280 * 4)

  // AttnBlockFixed only on attention-enabled blocks; all Stage B attentions use 1280 channels and 4 clip tokens.
  let attentionBlockCount =
    (attentions[0][0] ? blocks[0][0] : 0) + (attentions[0][1] ? blocks[0][1] : 0)
    + (attentions[0][2] ? blocks[0][2] : 0) + (attentions[0][3] ? blocks[0][3] : 0)
    + (attentions[1][0] ? blocks[1][0] : 0) + (attentions[1][1] ? blocks[1][1] : 0)
    + (attentions[1][2] ? blocks[1][2] : 0) + (attentions[1][3] ? blocks[1][3] : 0)
  for _ in 0..<attentionBlockCount {
    total += WurstchenAttnBlockFixedInstructionCount(
      batchSize: batchSize, kvLength: 4, kvInputChannels: 1280, channels: 1280)
  }

  // `effnetHeight`/`effnetWidth` are part of the builder shape contract but do not affect the counted
  // ops directly because the mapper conv runs after explicit upsampling to `height/2 x width/2`.
  _ = effnetHeight
  _ = effnetWidth

  return total
}
