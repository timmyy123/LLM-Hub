import Foundation

private func HiDreamO1FeedForwardInstructionCount(
  rows: Int, hiddenSize: Int = 4_096, intermediateSize: Int = 12_288
) -> Int {
  var total = 0
  total += DenseInstructionCount(rows: rows, input: hiddenSize, output: intermediateSize)
  total += DenseInstructionCount(rows: rows, input: hiddenSize, output: intermediateSize)
  total += DenseInstructionCount(rows: rows, input: intermediateSize, output: hiddenSize)
  return total
}

private func HiDreamO1TransformerLayerInstructionCount(
  batchSize: Int, queryLength: Int, keyValueLength: Int,
  hiddenSize: Int = 4_096, keyValueSize: Int = 8 * 128, intermediateSize: Int = 12_288
) -> Int {
  let rows = batchSize * queryLength
  var total = 0

  total += DenseInstructionCount(rows: rows, input: hiddenSize, output: hiddenSize)
  total += DenseInstructionCount(rows: rows, input: hiddenSize, output: keyValueSize)
  total += DenseInstructionCount(rows: rows, input: hiddenSize, output: keyValueSize)
  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: 32,
    headDimension: 128, sequenceDimensionA: queryLength,
    sequenceDimensionB: keyValueLength)
  total += DenseInstructionCount(rows: rows, input: hiddenSize, output: hiddenSize)
  total += HiDreamO1FeedForwardInstructionCount(
    rows: rows, hiddenSize: hiddenSize, intermediateSize: intermediateSize)

  return total
}

public func HiDreamO1InstructionCount(
  batchSize: Int, height: Int, width: Int, textLength: Int,
  layers: Int = 36
) -> Int {
  let imageLength = height * width
  let dynamicLength = imageLength + 1
  var total = 0

  total += DenseInstructionCount(
    rows: batchSize * imageLength, input: 3 * 32 * 32,
    output: 1_024)
  total += DenseInstructionCount(
    rows: batchSize * imageLength, input: 1_024,
    output: 4_096)
  total += DenseInstructionCount(
    rows: batchSize, input: 256,
    output: 4_096)
  total += DenseInstructionCount(
    rows: batchSize, input: 4_096, output: 4_096)

  for _ in 0..<layers {
    total += HiDreamO1TransformerLayerInstructionCount(
      batchSize: batchSize, queryLength: dynamicLength, keyValueLength: textLength + dynamicLength)
  }

  total += DenseInstructionCount(
    rows: batchSize * imageLength, input: 4_096,
    output: 3 * 32 * 32)
  return total
}

public func HiDreamO1FixedInstructionCount(
  batchSize: Int, textLength: Int, layers: Int = 36
) -> Int {
  var total = 0

  for _ in 0..<layers {
    total += HiDreamO1TransformerLayerInstructionCount(
      batchSize: batchSize, queryLength: textLength, keyValueLength: textLength)
  }

  return total
}
