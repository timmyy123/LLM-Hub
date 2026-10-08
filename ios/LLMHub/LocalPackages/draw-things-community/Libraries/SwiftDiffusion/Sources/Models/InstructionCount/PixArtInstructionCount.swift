import Foundation

private func PixArtCrossAttentionInstructionCount(
  batchSize: Int, heads: Int, headDimension: Int, queryLength: Int, tokenLength: (Int, Int)
) -> Int {
  if tokenLength.0 == tokenLength.1 {
    return ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: queryLength,
      sequenceDimensionB: tokenLength.0)
  }
  precondition(batchSize % 2 == 0)
  let splitBatch = batchSize / 2
  return ScaledDotProductAttentionInstructionCount(
    batchSize: splitBatch, heads: heads, headDimension: headDimension,
    sequenceDimensionA: queryLength,
    sequenceDimensionB: tokenLength.0)
    + ScaledDotProductAttentionInstructionCount(
      batchSize: splitBatch, heads: heads, headDimension: headDimension,
      sequenceDimensionA: queryLength,
      sequenceDimensionB: tokenLength.1)
}

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
public func PixArtInstructionCount(
  batchSize: Int, height: Int, width: Int, channels: Int, layers: Int, tokenLength: (Int, Int)
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  precondition(channels % 16 == 0)

  let h = height / 2
  let w = width / 2
  let hw = h * w
  let rows = batchSize * hw
  let heads = 16
  let headDimension = channels / heads
  var total = 0

  // x_embedder patchify conv: 8 -> channels, kernel 2x2 stride 2.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: h, outWidth: w, outChannels: channels, kernelHeight: 2,
    kernelWidth: 2, inputChannels: 8)

  for _ in 0..<layers {
    // Self-attention qkv + o.
    total += 4 * DenseInstructionCount(rows: rows, input: channels, output: channels)
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension, sequenceDimensionA: hw,
      sequenceDimensionB: hw)

    // Cross-attention (q + o here; k/v are precomputed in PixArtFixed).
    total += 2 * DenseInstructionCount(rows: rows, input: channels, output: channels)
    total += PixArtCrossAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension, queryLength: hw,
      tokenLength: tokenLength)

    // MLP hidden -> 4x -> hidden.
    total += DenseInstructionCount(rows: rows, input: channels, output: channels * 4)
    total += DenseInstructionCount(rows: rows, input: channels * 4, output: channels)
  }

  total += DenseInstructionCount(rows: rows, input: channels, output: 2 * 2 * 8)
  return total
}

// Extra caption input width is needed for exact caption_projection GEMM count.
public func PixArtFixedInstructionCount(
  batchSize: Int, channels: Int, layers: Int, tokenLength: (Int, Int),
  captionInputChannels: Int = 4096
) -> Int {
  precondition(channels % 16 == 0)
  let captionLength = max(tokenLength.0, tokenLength.1)
  let captionRows = batchSize * captionLength
  var total = 0

  // Time embedder + 6 adaln projections.
  total += DenseInstructionCount(rows: batchSize, input: 256, output: channels)
  total += DenseInstructionCount(rows: batchSize, input: channels, output: channels)
  total += 6 * DenseInstructionCount(rows: batchSize, input: channels, output: channels)

  // caption_projection MLP on caption tokens.
  total += DenseInstructionCount(rows: captionRows, input: captionInputChannels, output: channels)
  total += DenseInstructionCount(rows: captionRows, input: channels, output: channels)

  // Per-layer fixed cross-attn k/v projections.
  total += layers * 2 * DenseInstructionCount(rows: captionRows, input: channels, output: channels)

  return total
}
