import Foundation

private func HiDreamMoEFeedForwardInstructionCount(
  tokenLength: Int, hiddenSize: Int, intermediateSize: Int, experts: Int = 4, topK: Int = 2
) -> Int {
  var total = 0
  total += DenseInstructionCount(rows: tokenLength, input: hiddenSize, output: experts)  // gate
  let routedRows = tokenLength * topK
  total += DenseInstructionCount(rows: routedRows, input: hiddenSize, output: intermediateSize)  // w1
  total += DenseInstructionCount(rows: routedRows, input: hiddenSize, output: intermediateSize)  // w3
  total += DenseInstructionCount(rows: routedRows, input: intermediateSize, output: hiddenSize)  // w2
  return total
}

private func HiDreamJointTransformerBlockInstructionCount(
  batchSize: Int, imageLength: Int, textQueryLength: Int, textKeyValueLength: Int
) -> Int {
  let channels = 2_560
  let heads = 20
  let headDimension = 128
  let rowsImage = batchSize * imageLength
  let rowsTextQuery = batchSize * textQueryLength
  let rowsTextKeyValue = batchSize * textKeyValueLength
  let attnQueryLength = imageLength + textQueryLength
  let attnKeyLength = imageLength + textKeyValueLength
  var total = 0

  // Context stream qkv + output projection.
  total += 3 * DenseInstructionCount(rows: rowsTextKeyValue, input: channels, output: channels)
  total += DenseInstructionCount(rows: rowsTextQuery, input: channels, output: channels)

  // Image stream qkv + output projection.
  total += 3 * DenseInstructionCount(rows: rowsImage, input: channels, output: channels)
  total += DenseInstructionCount(rows: rowsImage, input: channels, output: channels)

  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: headDimension,
    sequenceDimensionA: attnQueryLength, sequenceDimensionB: attnKeyLength)

  // Context FFN: intermediate 6912.
  total += DenseInstructionCount(rows: rowsTextQuery, input: channels, output: 6_912)
  total += DenseInstructionCount(rows: rowsTextQuery, input: channels, output: 6_912)
  total += DenseInstructionCount(rows: rowsTextQuery, input: 6_912, output: channels)

  // Image shared FFN + image MoE FFN.
  total += DenseInstructionCount(rows: rowsImage, input: channels, output: 3_584)
  total += DenseInstructionCount(rows: rowsImage, input: channels, output: 3_584)
  total += DenseInstructionCount(rows: rowsImage, input: 3_584, output: channels)
  total += HiDreamMoEFeedForwardInstructionCount(
    tokenLength: rowsImage, hiddenSize: channels, intermediateSize: 6_912)

  return total
}

private func HiDreamSingleTransformerBlockInstructionCount(
  batchSize: Int, imageLength: Int, textQueryLength: Int, textKeyValueLength: Int,
  contextBlockPreOnly: Bool
) -> Int {
  let channels = 2_560
  let heads = 20
  let headDimension = 128
  let keyValueLength = imageLength + textKeyValueLength
  let queryLength = contextBlockPreOnly ? imageLength : imageLength + textQueryLength
  let rowsKeyValue = batchSize * keyValueLength
  let rowsQuery = batchSize * queryLength
  var total = 0

  // qkv + o
  total += 3 * DenseInstructionCount(rows: rowsKeyValue, input: channels, output: channels)
  total += DenseInstructionCount(rows: rowsQuery, input: channels, output: channels)
  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: heads, headDimension: headDimension,
    sequenceDimensionA: queryLength, sequenceDimensionB: keyValueLength)

  // Shared FFN + MoE FFN on query tokens.
  total += DenseInstructionCount(rows: rowsQuery, input: channels, output: 3_584)
  total += DenseInstructionCount(rows: rowsQuery, input: channels, output: 3_584)
  total += DenseInstructionCount(rows: rowsQuery, input: 3_584, output: channels)
  total += HiDreamMoEFeedForwardInstructionCount(
    tokenLength: rowsQuery, hiddenSize: channels, intermediateSize: 6_912)

  return total
}

// Counts only Dense (GEMM) and SDPA-equivalent attention instructions.
public func HiDreamInstructionCount(
  batchSize: Int, height: Int, width: Int, textLength: (Int, Int), layers: (Int, Int)
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  let imageLength = (height / 2) * (width / 2)
  let textQueryLength = textLength.0 + textLength.1
  let textKeyValueLength = textLength.0 + textLength.1 * 2
  var total = 0

  // Patchify dense: 2*2*16 -> 2560.
  total += DenseInstructionCount(rows: batchSize * imageLength, input: 2 * 2 * 16, output: 2_560)

  for _ in 0..<layers.0 {
    total += HiDreamJointTransformerBlockInstructionCount(
      batchSize: batchSize, imageLength: imageLength, textQueryLength: textQueryLength,
      textKeyValueLength: textKeyValueLength)
  }
  for i in 0..<layers.1 {
    total += HiDreamSingleTransformerBlockInstructionCount(
      batchSize: batchSize, imageLength: imageLength, textQueryLength: textQueryLength,
      textKeyValueLength: textKeyValueLength, contextBlockPreOnly: i == layers.1 - 1)
  }

  total += DenseInstructionCount(rows: batchSize * imageLength, input: 2_560, output: 2 * 2 * 16)
  return total
}

// Extra shape params are required because `HiDreamFixed(...)` builder does not include text lengths
// or caption hidden sizes, but they are needed for exact GEMM counts.
public func HiDreamFixedInstructionCount(
  timesteps: Int, layers: (Int, Int), t5TextLength: Int, llamaTextLength: Int,
  tInputChannels: Int = 256, pooledInputChannels: Int = 2048,
  t5InputChannels: Int = 4096, llamaInputChannels: Int = 4096
) -> Int {
  let channels = 2_560
  let totalCaptionProjections = layers.0 + layers.1 + 1
  var total = 0

  // t and pooled embedders.
  total += DenseInstructionCount(rows: timesteps, input: tInputChannels, output: channels)
  total += DenseInstructionCount(rows: timesteps, input: channels, output: channels)
  total += DenseInstructionCount(rows: timesteps, input: pooledInputChannels, output: channels)
  total += DenseInstructionCount(rows: timesteps, input: channels, output: channels)

  // Caption projections: first (layers.0 + layers.1) consume llama states, last consumes t5 states.
  if totalCaptionProjections > 1 {
    total +=
      (totalCaptionProjections - 1)
      * DenseInstructionCount(
        rows: timesteps * llamaTextLength, input: llamaInputChannels, output: channels)
  }
  total += DenseInstructionCount(
    rows: timesteps * t5TextLength, input: t5InputChannels, output: channels)

  // Fixed modulation blocks.
  total += layers.0 * 12 * DenseInstructionCount(rows: timesteps, input: channels, output: channels)
  total += layers.1 * 6 * DenseInstructionCount(rows: timesteps, input: channels, output: channels)

  // Final shift/scale heads.
  total += 2 * DenseInstructionCount(rows: timesteps, input: channels, output: channels)

  return total
}
