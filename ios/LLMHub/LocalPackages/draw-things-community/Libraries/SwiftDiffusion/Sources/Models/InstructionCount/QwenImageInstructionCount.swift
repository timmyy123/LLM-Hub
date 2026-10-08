import Foundation

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// Ignores norms, elementwise ops, RoPE cmul, and parameter-only tables/embeddings.
public func QwenImageInstructionCount(
  batchSize: Int, height: Int, width: Int, textLength: Int, referenceSequenceLength: Int,
  channels: Int, layers: Int, isQwenImageLayered: Bool
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  precondition(channels % 128 == 0)

  let h = height / 2
  let w = width / 2
  let imageSequenceLengthPerImage = h * w
  let b = isQwenImageLayered ? 1 : batchSize
  let imageSequenceLength = imageSequenceLengthPerImage * (isQwenImageLayered ? batchSize : 1)
  let xSequenceLength = imageSequenceLength + referenceSequenceLength
  let totalSequenceLength = xSequenceLength + textLength
  let heads = channels / 128
  let headDimension = 128
  var total = 0

  // Patchify conv on image latents only (reference latents are passed in already patchified).
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: h, outWidth: w, outChannels: channels, kernelHeight: 2,
    kernelWidth: 2, inputChannels: 16)

  for i in 0..<layers {
    let contextBlockPreOnly = (i == layers - 1)
    let rowsText = b * textLength
    let rowsXKV = b * xSequenceLength
    let rowsXOut =
      b * (contextBlockPreOnly ? (xSequenceLength - referenceSequenceLength) : xSequenceLength)

    // Joint attention q/k/v on text + image/reference streams.
    total += 3 * DenseInstructionCount(rows: rowsText, input: channels, output: channels)
    total += 3 * DenseInstructionCount(rows: rowsXKV, input: channels, output: channels)
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: b, heads: heads, headDimension: headDimension,
      sequenceDimensionA: totalSequenceLength,
      sequenceDimensionB: totalSequenceLength)

    if !contextBlockPreOnly {
      total += DenseInstructionCount(rows: rowsText, input: channels, output: channels)  // c_o
      total += DenseInstructionCount(rows: rowsText, input: channels, output: channels * 4)  // c mlp up
      total += DenseInstructionCount(rows: rowsText, input: channels * 4, output: channels)  // c mlp down
    }

    total += DenseInstructionCount(rows: rowsXOut, input: channels, output: channels)  // x_o
    total += DenseInstructionCount(rows: rowsXOut, input: channels, output: channels * 4)  // x mlp up
    total += DenseInstructionCount(rows: rowsXOut, input: channels * 4, output: channels)  // x mlp down
  }

  total += DenseInstructionCount(
    rows: batchSize * imageSequenceLengthPerImage, input: channels, output: 2 * 2 * 16)

  return total
}

// Extra params are required because `QwenImageFixed(...)` does not include text shape or total
// reference token count, but they affect exact GEMM/conv counts.
public func QwenImageFixedInstructionCount(
  timesteps: Int, batchSize: Int, textLength: Int, channels: Int, layers: Int,
  referenceSequenceLength: Int = 0, textInputChannels: Int = 3584
) -> Int {
  precondition(channels % 128 == 0)
  var total = 0

  // Optional reference-image patchify conv(s), aggregated by total post-patchify token count.
  if referenceSequenceLength > 0 {
    total += 64 * channels * referenceSequenceLength
  }

  // context_norm is ignored (RMSNorm); context_embedder is counted.
  total += DenseInstructionCount(
    rows: batchSize * textLength, input: textInputChannels, output: channels)

  if layers > 0 {
    // t embedder MLP: 256 -> channels -> channels.
    total += DenseInstructionCount(rows: timesteps, input: 256, output: channels)
    total += DenseInstructionCount(rows: timesteps, input: channels, output: channels)

    for i in 0..<layers {
      let contextBlockPreOnly = (i == layers - 1)
      let perLayerDenseCount = (contextBlockPreOnly ? 2 : 6) + 6
      total +=
        perLayerDenseCount
        * DenseInstructionCount(rows: timesteps, input: channels, output: channels)
    }

    // Final norm_out linear emits shift + scale.
    total += 2 * DenseInstructionCount(rows: timesteps, input: channels, output: channels)
  }

  return total
}
