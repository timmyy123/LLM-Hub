import Foundation

private func Krea2AttentionInstructionCount(
  batchSize: Int, tokenLength: Int, queryLength: Int, hiddenSize: Int, heads: Int,
  keyValueHeads: Int, segments: [Int] = []
) -> Int {
  precondition(hiddenSize % heads == 0)
  precondition(heads % keyValueHeads == 0)
  precondition(queryLength <= tokenLength)
  precondition(
    segments.isEmpty || (queryLength == tokenLength && segments.reduce(0, +) == tokenLength))
  let headDim = hiddenSize / heads
  let keyValueSize = headDim * keyValueHeads
  let rowsQuery = batchSize * queryLength
  let rowsKeyValue = batchSize * tokenLength
  var total = 0

  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // q
  total += DenseInstructionCount(rows: rowsKeyValue, input: hiddenSize, output: keyValueSize)  // k
  total += DenseInstructionCount(rows: rowsKeyValue, input: hiddenSize, output: keyValueSize)  // v
  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // gate
  total += DenseInstructionCount(rows: rowsQuery, input: hiddenSize, output: hiddenSize)  // o

  if segments.count > 1 {
    for segment in segments {
      total += ScaledDotProductAttentionInstructionCount(
        batchSize: batchSize, heads: heads, headDimension: headDim,
        sequenceDimensionA: segment, sequenceDimensionB: segment)
    }
  } else {
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDim,
      sequenceDimensionA: queryLength, sequenceDimensionB: tokenLength)
  }

  return total
}

private func Krea2SwiGLUInstructionCount(
  rows: Int, hiddenSize: Int, intermediateSize: Int
) -> Int {
  var total = 0
  total += DenseInstructionCount(rows: rows, input: hiddenSize, output: intermediateSize)  // gate
  total += DenseInstructionCount(rows: rows, input: hiddenSize, output: intermediateSize)  // up
  total += DenseInstructionCount(rows: rows, input: intermediateSize, output: hiddenSize)  // down
  return total
}

private func Krea2TransformerBlockInstructionCount(
  batchSize: Int, tokenLength: Int, queryLength: Int, hiddenSize: Int = 6_144, heads: Int = 48,
  keyValueHeads: Int = 12, intermediateSize: Int = 16_384
) -> Int {
  let rowsQuery = batchSize * queryLength
  var total = 0
  total += Krea2AttentionInstructionCount(
    batchSize: batchSize, tokenLength: tokenLength, queryLength: queryLength,
    hiddenSize: hiddenSize, heads: heads, keyValueHeads: keyValueHeads)
  total += Krea2SwiGLUInstructionCount(
    rows: rowsQuery, hiddenSize: hiddenSize, intermediateSize: intermediateSize)
  return total
}

private func Krea2TextFusionBlockInstructionCount(
  batchSize: Int, tokenLength: Int, segments: [Int] = [], hiddenSize: Int = 2_560,
  heads: Int = 20, intermediateSize: Int = 6_912
) -> Int {
  let rows = batchSize * tokenLength
  var total = 0
  total += Krea2AttentionInstructionCount(
    batchSize: batchSize, tokenLength: tokenLength, queryLength: tokenLength,
    hiddenSize: hiddenSize, heads: heads, keyValueHeads: heads, segments: segments)
  total += Krea2SwiGLUInstructionCount(
    rows: rows, hiddenSize: hiddenSize, intermediateSize: intermediateSize)
  return total
}

public func Krea2InstructionCount(
  batchSize: Int, height: Int, width: Int, textLength: Int, hiddenSize: Int = 6_144,
  layers: Int = 28, intermediateSize: Int = 16_384
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  let patchHeight = height / 2
  let patchWidth = width / 2
  let imageLength = patchHeight * patchWidth
  let tokenLength = imageLength + textLength
  var total = 0

  total += DenseInstructionCount(
    rows: batchSize * imageLength, input: 2 * 2 * 16, output: hiddenSize)

  for i in 0..<layers {
    let queryLength = i == layers - 1 ? imageLength : tokenLength
    total += Krea2TransformerBlockInstructionCount(
      batchSize: batchSize, tokenLength: tokenLength, queryLength: queryLength,
      hiddenSize: hiddenSize, intermediateSize: intermediateSize)
  }

  total += DenseInstructionCount(
    rows: batchSize * imageLength, input: hiddenSize, output: 2 * 2 * 16)
  return total
}

public func Krea2TextFusionAdapterInstructionCount(
  batchSize: Int, textLength: (Int, Int), hiddenSize: Int = 2_560,
  layerwiseTokenLength: Int = 12, layers: Int = 2, intermediateSize: Int = 6_912
) -> Int {
  let totalTextLength = textLength.0 + textLength.1
  let segments = textLength.0 > 0 ? [textLength.0, textLength.1] : []
  var total = 0

  for _ in 0..<layers {
    total += Krea2TextFusionBlockInstructionCount(
      batchSize: batchSize * totalTextLength, tokenLength: layerwiseTokenLength,
      hiddenSize: hiddenSize, intermediateSize: intermediateSize)
  }

  total += DenseInstructionCount(
    rows: batchSize * totalTextLength * hiddenSize, input: layerwiseTokenLength, output: 1)

  for _ in 0..<layers {
    total += Krea2TextFusionBlockInstructionCount(
      batchSize: batchSize, tokenLength: totalTextLength, segments: segments,
      hiddenSize: hiddenSize, intermediateSize: intermediateSize)
  }

  return total
}

public func Krea2FixedInstructionCount(
  batchSize: Int, timesteps: Int, textLength: Int, textInputChannels: Int = 2_560,
  hiddenSize: Int = 6_144
) -> Int {
  precondition(timesteps > 0)
  var total = 0

  total += DenseInstructionCount(
    rows: batchSize * textLength, input: textInputChannels, output: hiddenSize)
  total += DenseInstructionCount(
    rows: batchSize * textLength, input: hiddenSize, output: hiddenSize)

  total += DenseInstructionCount(rows: timesteps, input: 256, output: hiddenSize)
  total += DenseInstructionCount(rows: timesteps, input: hiddenSize, output: hiddenSize)
  total += 6 * DenseInstructionCount(rows: timesteps, input: hiddenSize, output: hiddenSize)

  return total
}
