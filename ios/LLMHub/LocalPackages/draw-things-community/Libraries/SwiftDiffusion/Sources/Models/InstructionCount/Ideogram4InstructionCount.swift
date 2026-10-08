import Foundation

private func Ideogram4TransformerBlockInstructionCount(
  batchSize: Int, queryLength: Int, keyValueLength: Int, channels: Int = 4_608,
  headDimension: Int = 256, intermediateSize: Int = 12_288
) -> Int {
  precondition(channels % headDimension == 0)
  let rowsQuery = batchSize * queryLength
  let rowsKeyValue = batchSize * keyValueLength
  var total = 0

  total += DenseInstructionCount(rows: rowsQuery, input: channels, output: channels)
  total += 2 * DenseInstructionCount(rows: rowsKeyValue, input: channels, output: channels)
  total += DenseInstructionCount(rows: rowsQuery, input: channels, output: channels)
  total += ScaledDotProductAttentionInstructionCount(
    batchSize: batchSize, heads: channels / headDimension, headDimension: headDimension,
    sequenceDimensionA: queryLength, sequenceDimensionB: keyValueLength)
  total += 2 * DenseInstructionCount(rows: rowsQuery, input: channels, output: intermediateSize)
  total += DenseInstructionCount(rows: rowsQuery, input: intermediateSize, output: channels)

  return total
}

public func Ideogram4InstructionCount(
  batchSize: Int, height: Int, width: Int, textLength: Int,
  channels: Int = 4_608, layers: Int = 34, intermediateSize: Int = 12_288
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  let patchHeight = height / 2
  let patchWidth = width / 2
  let imageLength = patchHeight * patchWidth
  let totalLength = imageLength + textLength
  var total = 0

  total += DenseInstructionCount(rows: batchSize * imageLength, input: 128, output: channels)

  for i in 0..<layers {
    let queryLength = i == layers - 1 ? imageLength : totalLength
    total += Ideogram4TransformerBlockInstructionCount(
      batchSize: batchSize, queryLength: queryLength, keyValueLength: totalLength,
      channels: channels,
      intermediateSize: intermediateSize)
  }

  total += DenseInstructionCount(rows: batchSize * imageLength, input: channels, output: 128)
  return total
}

public func Ideogram4FixedInstructionCount(
  timesteps: Int, batchSize: Int, textLength: Int, textInputChannels: Int = 4_096 * 13,
  channels: Int = 4_608, layers: Int = 34, modulationSize: Int = 512
) -> Int {
  var total = 0

  total += DenseInstructionCount(
    rows: batchSize * textLength, input: textInputChannels, output: channels)
  total += DenseInstructionCount(rows: timesteps, input: channels, output: channels)
  total += DenseInstructionCount(rows: timesteps, input: channels, output: channels)
  total += DenseInstructionCount(rows: timesteps, input: channels, output: modulationSize)

  for _ in 0..<layers {
    total += 4 * DenseInstructionCount(rows: timesteps, input: modulationSize, output: channels)
  }

  total += DenseInstructionCount(rows: timesteps, input: modulationSize, output: channels)
  return total
}
