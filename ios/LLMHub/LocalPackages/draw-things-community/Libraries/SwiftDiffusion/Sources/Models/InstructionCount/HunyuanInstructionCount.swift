import Foundation

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// Ignores LoRA, norms, elementwise ops, reshapes/concat, and tea-cache auxiliary models.
public func HunyuanInstructionCount(
  time: Int, height: Int, width: Int, textLength: Int, channels: Int, layers: (Int, Int)
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  precondition(channels % 128 == 0)

  let imageHeight = height / 2
  let imageWidth = width / 2
  let imageSequenceLength = time * imageHeight * imageWidth
  let totalSequenceLength = textLength + imageSequenceLength
  let heads = channels / 128
  let headDimension = 128
  var total = 0

  // x_embedder patchify conv runs on `time` frames independently: 16ch -> channels.
  total += ConvolutionInstructionCount(
    batchSize: time, outHeight: imageHeight, outWidth: imageWidth, outChannels: channels,
    kernelHeight: 2, kernelWidth: 2, inputChannels: 16)

  let rowsText = textLength
  let rowsImage = imageSequenceLength

  for _ in 0..<layers.0 {
    // JointTransformerBlock: text + image streams, FFN intermediate=4*channels.
    total += 4 * DenseInstructionCount(rows: rowsText, input: channels, output: channels)
    total += DenseInstructionCount(rows: rowsText, input: channels, output: channels * 4)
    total += DenseInstructionCount(rows: rowsText, input: channels * 4, output: channels)

    total += 4 * DenseInstructionCount(rows: rowsImage, input: channels, output: channels)
    total += DenseInstructionCount(rows: rowsImage, input: channels, output: channels * 4)
    total += DenseInstructionCount(rows: rowsImage, input: channels * 4, output: channels)

    total += ScaledDotProductAttentionInstructionCount(
      batchSize: 1, heads: heads, headDimension: headDimension,
      sequenceDimensionA: totalSequenceLength, sequenceDimensionB: totalSequenceLength)
  }

  for i in 0..<layers.1 {
    let isLast = (i == layers.1 - 1)
    let projectedSequenceLength = isLast ? imageSequenceLength : totalSequenceLength

    total += 3 * DenseInstructionCount(rows: totalSequenceLength, input: channels, output: channels)
    total += DenseInstructionCount(rows: projectedSequenceLength, input: channels, output: channels)  // x_o
    total += DenseInstructionCount(
      rows: projectedSequenceLength, input: channels, output: channels * 4)
    total += DenseInstructionCount(
      rows: projectedSequenceLength, input: channels * 4, output: channels)
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: 1, heads: heads, headDimension: headDimension,
      sequenceDimensionA: totalSequenceLength, sequenceDimensionB: totalSequenceLength)
  }

  total += DenseInstructionCount(rows: imageSequenceLength, input: channels, output: 2 * 2 * 16)

  return total
}

public func HunyuanFixedInstructionCount(
  timesteps: Int, channels: Int, layers: (Int, Int), textLength: (Int, Int)
) -> Int {
  precondition(channels % 128 == 0)
  let cfgBatch = textLength.0 > 0 ? 2 : 1
  let vecRows = timesteps * cfgBatch
  let textTokenRows = timesteps * (textLength.0 + textLength.1)
  var total = 0

  // txt_in_t embedder (t -> channels) and txt_in.c_embedder on pooled text.
  total += DenseInstructionCount(rows: vecRows, input: 256, output: channels)
  total += DenseInstructionCount(rows: vecRows, input: channels, output: channels)
  total += DenseInstructionCount(rows: vecRows, input: 4096, output: channels)
  total += DenseInstructionCount(rows: vecRows, input: channels, output: channels)

  // txt_in.input_embedder on all text tokens.
  total += DenseInstructionCount(rows: textTokenRows, input: 4096, output: channels)

  // Two individual token refiner blocks.
  for _ in 0..<2 {
    // Refiner self attention (hidden = 3072, heads = 24, headDim = 128).
    total += 4 * DenseInstructionCount(rows: textTokenRows, input: 3072, output: 3072)  // q,k,v,o
    if textLength.0 > 0 {
      total += ScaledDotProductAttentionInstructionCount(
        batchSize: timesteps, heads: 24, headDimension: 128,
        sequenceDimensionA: textLength.0, sequenceDimensionB: textLength.0)
      total += ScaledDotProductAttentionInstructionCount(
        batchSize: timesteps, heads: 24, headDimension: 128,
        sequenceDimensionA: textLength.1, sequenceDimensionB: textLength.1)
    } else {
      total += ScaledDotProductAttentionInstructionCount(
        batchSize: timesteps, heads: 24, headDimension: 128,
        sequenceDimensionA: textLength.1, sequenceDimensionB: textLength.1)
    }

    // Refiner AdaLN gates + MLP.
    total += DenseInstructionCount(rows: vecRows, input: channels, output: 3072)  // gateMsa
    total += DenseInstructionCount(rows: textTokenRows, input: 3072, output: 3072 * 4)  // mlp0
    total += DenseInstructionCount(rows: textTokenRows, input: 3072 * 4, output: 3072)  // mlp1
    total += DenseInstructionCount(rows: vecRows, input: channels, output: 3072)  // gateMlp
  }

  // time_in, vector_in, guidance_in MLP embedders.
  total += DenseInstructionCount(rows: vecRows, input: 256, output: channels)
  total += DenseInstructionCount(rows: vecRows, input: channels, output: channels)
  total += DenseInstructionCount(rows: vecRows, input: 768, output: channels)
  total += DenseInstructionCount(rows: vecRows, input: channels, output: channels)
  total += DenseInstructionCount(rows: vecRows, input: 256, output: channels)
  total += DenseInstructionCount(rows: vecRows, input: channels, output: channels)

  // Fixed modulation heads emitted for denoiser blocks.
  total += layers.0 * 12 * DenseInstructionCount(rows: vecRows, input: channels, output: channels)
  total += layers.1 * 3 * DenseInstructionCount(rows: vecRows, input: channels, output: channels)

  // Final fixed shift/scale heads.
  total += 2 * DenseInstructionCount(rows: vecRows, input: channels, output: channels)

  return total
}
