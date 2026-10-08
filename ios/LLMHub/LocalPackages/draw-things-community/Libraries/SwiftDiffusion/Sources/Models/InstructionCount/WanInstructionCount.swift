import Foundation

private func WanSelfAttentionInstructionCount(
  heads: Int, headDimension: Int, batchSize: Int, hw: Int, time: Int,
  causalInference: (Int, pad: Int)
) -> Int {
  if causalInference.0 > 0 && causalInference.0 + causalInference.pad < time {
    precondition(batchSize == 1)
    let frames = hw / time * causalInference.0
    let padFrames = hw / time * causalInference.pad
    var total = 0
    for i in 0..<((time + causalInference.0 - 1) / causalInference.0) {
      let queryLength = min(hw - i * frames, frames)
      let keyLength = min(hw, (i + 1) * frames + padFrames)
      total += ScaledDotProductAttentionInstructionCount(
        batchSize: batchSize, heads: heads, headDimension: headDimension,
        sequenceDimensionA: queryLength, sequenceDimensionB: keyLength)
    }
    return total
  }
  return ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: headDimension,
    sequenceDimensionA: hw, sequenceDimensionB: hw)
}

private func WanAttentionBlockInstructionCount(
  channels: Int, intermediateSize: Int, batchSize: Int, textLength: Int, imageContextLength: Int,
  hw: Int, time: Int, causalInference: (Int, pad: Int), injectImage: Bool
) -> Int {
  let heads = channels / 128
  let headDimension = 128
  let rows = batchSize * hw
  var total = 0

  // Self-attention qkv + o.
  total += 4 * DenseInstructionCount(rows: rows, input: channels, output: channels)
  total += WanSelfAttentionInstructionCount(
    heads: heads, headDimension: headDimension, batchSize: batchSize, hw: hw, time: time,
    causalInference: causalInference)

  // Cross-attention q + o (k/v are precomputed in WanFixed and passed in).
  total += DenseInstructionCount(rows: rows, input: channels, output: channels)  // x_c_q
  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: headDimension,
    sequenceDimensionA: hw, sequenceDimensionB: textLength)
  if injectImage {
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: hw, sequenceDimensionB: imageContextLength)
  }
  total += DenseInstructionCount(rows: rows, input: channels, output: channels)  // c_o

  // FFN: hidden -> intermediate -> hidden.
  total += DenseInstructionCount(rows: rows, input: channels, output: intermediateSize)
  total += DenseInstructionCount(rows: rows, input: intermediateSize, output: channels)

  return total
}

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// Ignores LoRA, norms, elementwise ops, reshapes/concat, and control-path bookkeeping.
public func WanInstructionCount(
  channels: Int, layers: Int, vaceLayers: [Int], intermediateSize: Int, time: Int, height: Int,
  width: Int, textLength: Int, causalInference: (Int, pad: Int), injectImage: Bool,
  outputChannels: Int
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  precondition(channels % 128 == 0)

  let h = height / 2
  let w = width / 2
  let hw = time * h * w
  let imageContextLength = 257
  var total = 0

  // Patch embedding conv on latent input (`outputChannels` matches latent channels in current Wan variants).
  total += ConvolutionInstructionCount(
    batchSize: time, outHeight: h, outWidth: w, outChannels: channels,
    kernelHeight: 2, kernelWidth: 2, inputChannels: outputChannels)

  for _ in vaceLayers {
    total += WanAttentionBlockInstructionCount(
      channels: channels, intermediateSize: intermediateSize, batchSize: 1,
      textLength: textLength, imageContextLength: imageContextLength, hw: hw, time: time,
      causalInference: causalInference, injectImage: false)
    total += DenseInstructionCount(rows: hw, input: channels, output: channels)  // vace_after_proj
  }

  for _ in 0..<layers {
    total += WanAttentionBlockInstructionCount(
      channels: channels, intermediateSize: intermediateSize, batchSize: 1,
      textLength: textLength, imageContextLength: imageContextLength, hw: hw, time: time,
      causalInference: causalInference, injectImage: injectImage)
  }

  total += DenseInstructionCount(rows: hw, input: channels, output: 2 * 2 * outputChannels)
  return total
}

// Optional `vaceContextConvolutionInput` is needed only to count the two VACE convs exactly,
// because `WanFixed(...)` itself does not carry the VACE hint spatial shape.
public func WanFixedInstructionCount(
  timesteps: Int, batchSize: (Int, Int), channels: Int, layers: Int, vaceLayers: [Int],
  textLength: Int, injectImage: Bool,
  vaceContextConvolutionInput: (batchSize: Int, height: Int, width: Int, channels: Int)? = nil
) -> Int {
  precondition(channels % 128 == 0)
  let imageContextLength = 257
  var total = 0

  // Text condition embedder: 4096 -> channels -> channels.
  total += DenseInstructionCount(rows: batchSize.0 * textLength, input: 4096, output: channels)
  total += DenseInstructionCount(rows: batchSize.0 * textLength, input: channels, output: channels)

  // Time embedder: 256 -> channels -> channels.
  total += DenseInstructionCount(rows: timesteps, input: 256, output: channels)
  total += DenseInstructionCount(rows: timesteps, input: channels, output: channels)

  // 6 time projections from vector.
  total += 6 * DenseInstructionCount(rows: timesteps, input: channels, output: channels)

  if !vaceLayers.isEmpty, let vace = vaceContextConvolutionInput {
    precondition(vace.height % 2 == 0 && vace.width % 2 == 0)
    total += ConvolutionInstructionCount(
      batchSize: vace.batchSize, outHeight: vace.height / 2, outWidth: vace.width / 2,
      outChannels: channels, kernelHeight: 2, kernelWidth: 2, inputChannels: vace.channels)
    total += ConvolutionInstructionCount(
      batchSize: vace.batchSize, outHeight: vace.height / 2, outWidth: vace.width / 2,
      outChannels: channels, kernelHeight: 1, kernelWidth: 1, inputChannels: channels)
  }

  if injectImage {
    // MLPProj for CLIP image embedding: 1280 -> 1280 -> channels.
    let rows = batchSize.1 * imageContextLength
    total += DenseInstructionCount(rows: rows, input: 1280, output: 1280)
    total += DenseInstructionCount(rows: rows, input: 1280, output: channels)
  }

  let rowsText = batchSize.0 * textLength
  let rowsImage = batchSize.1 * imageContextLength
  for _ in vaceLayers {
    total += 2 * DenseInstructionCount(rows: rowsText, input: channels, output: channels)  // c_k/c_v
  }
  for _ in 0..<layers {
    total += 2 * DenseInstructionCount(rows: rowsText, input: channels, output: channels)  // c_k/c_v
    if injectImage {
      total += 2 * DenseInstructionCount(rows: rowsImage, input: channels, output: channels)  // c_img_k/c_img_v
    }
  }

  return total
}
