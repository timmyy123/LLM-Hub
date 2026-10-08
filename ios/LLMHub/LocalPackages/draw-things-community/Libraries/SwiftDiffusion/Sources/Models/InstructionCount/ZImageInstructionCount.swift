import Foundation

private func roundUpTo32(_ value: Int) -> Int {
  (value + 31) / 32 * 32
}

private func ZImageTransformerBlockInstructionCount(
  batchSize: Int, keyValueLength: Int, queryLength: Int, segments: [Int], channels: Int
) -> Int {
  precondition(channels % 128 == 0)
  let heads = channels / 128
  let headDimension = 128
  let rowsKeyValue = batchSize * keyValueLength
  let rowsQuery = batchSize * queryLength
  var total = 0

  // q/k/v/o projections.
  total += DenseInstructionCount(rows: rowsKeyValue, input: channels, output: channels)  // k
  total += DenseInstructionCount(rows: rowsQuery, input: channels, output: channels)  // q
  total += DenseInstructionCount(rows: rowsKeyValue, input: channels, output: channels)  // v
  total += DenseInstructionCount(rows: rowsQuery, input: channels, output: channels)  // o

  // FFN: SwiGLU with intermediate 10_240.
  total += DenseInstructionCount(rows: rowsQuery, input: channels, output: 10_240)  // w1
  total += DenseInstructionCount(rows: rowsQuery, input: channels, output: 10_240)  // w3
  total += DenseInstructionCount(rows: rowsQuery, input: 10_240, output: channels)  // w2

  if segments.count > 1 && keyValueLength == queryLength {
    for segment in segments {
      total += ScaledDotProductAttentionInstructionCount(
        batchSize: batchSize, heads: heads, headDimension: headDimension,
        sequenceDimensionA: segment, sequenceDimensionB: segment)
    }
  } else {
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: queryLength, sequenceDimensionB: keyValueLength)
  }

  return total
}

// Counts only Dense (GEMM) and SDPA-equivalent attention instructions.
// `ZImage` uses a Dense patch embedding (not convolution) for image tokens.
public func ZImageInstructionCount(
  batchSize: Int, height: Int, width: Int, textLength: Int, channels: Int, layers: Int
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  let h = height / 2
  let w = width / 2
  let imageLength = h * w
  let roundUpHW = roundUpTo32(imageLength)
  var total = 0

  // x_embedder patchify Dense: 2*2*16 -> channels.
  total += DenseInstructionCount(rows: batchSize * imageLength, input: 2 * 2 * 16, output: channels)

  // Two noise refiners on image tokens only.
  for _ in 0..<2 {
    total += ZImageTransformerBlockInstructionCount(
      batchSize: batchSize, keyValueLength: roundUpHW, queryLength: roundUpHW, segments: [],
      channels: channels)
  }

  let totalLength = roundUpHW + textLength
  for i in 0..<layers {
    let queryLength = (i == layers - 1) ? imageLength : totalLength
    total += ZImageTransformerBlockInstructionCount(
      batchSize: batchSize, keyValueLength: totalLength, queryLength: queryLength, segments: [],
      channels: channels)
  }

  total += DenseInstructionCount(rows: batchSize * imageLength, input: channels, output: 2 * 2 * 16)
  return total
}

// `timesteps` is required because `ZImageFixed(...)` emits timestep-conditioned modulation outputs.
public func ZImageFixedInstructionCount(
  batchSize: Int, timesteps: Int, tokenLength: (Int, Int), channels: Int, layers: Int,
  textInputChannels: Int = 2560
) -> Int {
  precondition(channels % 128 == 0)
  let roundedTokenLength = (roundUpTo32(tokenLength.0), roundUpTo32(tokenLength.1))
  let totalRoundedTokenLength = roundedTokenLength.0 + roundedTokenLength.1
  let totalTokenLength = tokenLength.0 + tokenLength.1
  let segments = roundedTokenLength.0 > 0 ? [roundedTokenLength.0, roundedTokenLength.1] : []
  var total = 0

  // cap_embedder path: RMSNorm + Dense(count: channels) on all text tokens.
  total += DenseInstructionCount(
    rows: batchSize * totalTokenLength, input: textInputChannels, output: channels)

  // t_embedder MLP in ZImageFixed is 256 -> 1024 -> 256.
  total += DenseInstructionCount(rows: timesteps, input: 256, output: 1024)
  total += DenseInstructionCount(rows: timesteps, input: 1024, output: 256)

  // Two context refiners (full transformer blocks, no modulation) on padded text tokens.
  for _ in 0..<2 {
    total += ZImageTransformerBlockInstructionCount(
      batchSize: batchSize, keyValueLength: totalRoundedTokenLength,
      queryLength: totalRoundedTokenLength,
      segments: segments, channels: channels)
  }

  // Noise refiner fixed blocks and main-layer fixed blocks: each emits 4 modulation Dense(count: channels).
  total += 2 * 4 * DenseInstructionCount(rows: timesteps, input: 256, output: channels)
  total += layers * 4 * DenseInstructionCount(rows: timesteps, input: 256, output: channels)

  // Final modulation scale head.
  total += DenseInstructionCount(rows: timesteps, input: 256, output: channels)

  return total
}
