import Foundation

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
// Ignores LoRA, norms, elementwise ops, reshapes, concat/slice, ControlNet, and IP-Adapter paths.
public func Flux1InstructionCount(
  batchSize: Int, tokenLength: Int, referenceSequenceLength: Int, height: Int, width: Int,
  channels: Int, layers: (Int, Int), contextPreloaded: Bool
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  precondition(channels % 128 == 0)

  let imageHeight = height / 2
  let imageWidth = width / 2
  let imageSequenceLength = imageHeight * imageWidth
  let imageAndReferenceSequenceLength = imageSequenceLength + referenceSequenceLength
  let totalSequenceLength = tokenLength + imageAndReferenceSequenceLength
  let heads = channels / 128
  let headDimension = 128
  var total = 0

  // x_embedder patchify conv: 16 -> channels, kernel 2x2 stride 2.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: imageHeight, outWidth: imageWidth, outChannels: channels,
    kernelHeight: 2, kernelWidth: 2, inputChannels: 16)

  if !contextPreloaded && (layers.0 > 0 || layers.1 > 0) {
    total += DenseInstructionCount(
      rows: batchSize * tokenLength, input: 4096, output: channels)
  }

  if layers.0 > 0 {
    let rows = batchSize * totalSequenceLength
    let perBlockDense = 12 * DenseInstructionCount(rows: rows, input: channels, output: channels)
    let perBlockSDPA = ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: totalSequenceLength, sequenceDimensionB: totalSequenceLength)
    total += layers.0 * (perBlockDense + perBlockSDPA)
  }

  for i in 0..<layers.1 {
    let isLast = (i == layers.1 - 1)
    let projectedSequenceLength = isLast ? imageSequenceLength : totalSequenceLength
    total +=
      3
      * DenseInstructionCount(
        rows: batchSize * totalSequenceLength, input: channels, output: channels)
    total += DenseInstructionCount(
      rows: batchSize * projectedSequenceLength, input: channels, output: channels)
    total += DenseInstructionCount(
      rows: batchSize * projectedSequenceLength, input: channels, output: channels * 4)
    total += DenseInstructionCount(
      rows: batchSize * projectedSequenceLength, input: channels * 4, output: channels)
    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: totalSequenceLength, sequenceDimensionB: totalSequenceLength)
  }

  let finalSequenceLength: Int
  if layers.1 > 0 {
    finalSequenceLength = imageSequenceLength
  } else if layers.0 == 0 {
    finalSequenceLength = imageSequenceLength
  } else {
    finalSequenceLength = totalSequenceLength
  }
  total += DenseInstructionCount(
    rows: batchSize * finalSequenceLength, input: channels, output: 2 * 2 * 16)

  return total
}

// `tokenLength` and `referenceSequenceLength` are included because they affect fixed-path shapes,
// even though they are not directly in the `Flux1Fixed(...)` builder signature.
public func Flux1FixedInstructionCount(
  batchSize: (Int, Int), tokenLength: Int, referenceSequenceLength: Int, channels: Int,
  layers: (Int, Int), contextPreloaded: Bool, guidanceEmbed: Bool
) -> Int {
  precondition(channels % 128 == 0)
  var total = 0

  // Reference image patchify convs (if any). Reference sequence length is post-patchify tokens.
  if referenceSequenceLength > 0 {
    total += 64 * channels * referenceSequenceLength
  }

  // t MLP embedder: 256 -> channels -> channels
  total += DenseInstructionCount(rows: batchSize.1, input: 256, output: channels)
  total += DenseInstructionCount(rows: batchSize.1, input: channels, output: channels)

  if guidanceEmbed {
    total += DenseInstructionCount(rows: batchSize.1, input: 256, output: channels)
    total += DenseInstructionCount(rows: batchSize.1, input: channels, output: channels)
  }

  // y/vector MLP embedder: 768 -> channels -> channels
  total += DenseInstructionCount(rows: batchSize.1, input: 768, output: channels)
  total += DenseInstructionCount(rows: batchSize.1, input: channels, output: channels)

  // Fixed path precomputes context embedding when contextPreloaded is true.
  if contextPreloaded {
    total += DenseInstructionCount(
      rows: batchSize.0 * tokenLength, input: 4096, output: channels)
  }

  total +=
    layers.0 * 12
    * DenseInstructionCount(rows: batchSize.1, input: channels, output: channels)
  total +=
    layers.1 * 3
    * DenseInstructionCount(rows: batchSize.1, input: channels, output: channels)

  // Final fixed shift/scale heads.
  total += 2 * DenseInstructionCount(rows: batchSize.1, input: channels, output: channels)

  return total
}
