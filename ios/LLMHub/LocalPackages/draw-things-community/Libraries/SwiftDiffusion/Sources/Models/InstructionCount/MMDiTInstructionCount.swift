import Foundation

// Counts only Dense (GEMM), SDPA-equivalent attention, and Convolution instructions.
public func MMDiTInstructionCount(
  batchSize: Int, t: Int, height: Int, width: Int, channels: Int, layers: Int,
  dualAttentionLayers: [Int]
) -> Int {
  precondition(height % 2 == 0 && width % 2 == 0)
  precondition(channels % 64 == 0)

  let h = height / 2
  let w = width / 2
  let hw = h * w
  let heads = channels / 64
  let headDimension = 64
  let totalLength = t + hw
  let rowsContext = batchSize * t
  let rowsImage = batchSize * hw
  var total = 0

  // x_embedder patchify conv: 16 -> channels.
  total += ConvolutionInstructionCount(
    batchSize: batchSize, outHeight: h, outWidth: w, outChannels: channels, kernelHeight: 2,
    kernelWidth: 2, inputChannels: 16)

  for i in 0..<layers {
    let contextBlockPreOnly = (i == layers - 1)
    let useDualAttention = dualAttentionLayers.contains(i)

    // Joint attention qkv projections.
    total += 3 * DenseInstructionCount(rows: rowsContext, input: channels, output: channels)
    total += 3 * DenseInstructionCount(rows: rowsImage, input: channels, output: channels)
    if useDualAttention {
      total += 3 * DenseInstructionCount(rows: rowsImage, input: channels, output: channels)
    }

    total += ScaledDotProductAttentionInstructionCount(
      batchSize: batchSize, heads: heads, headDimension: headDimension,
      sequenceDimensionA: totalLength,
      sequenceDimensionB: totalLength)
    if useDualAttention {
      total += ScaledDotProductAttentionInstructionCount(
        batchSize: batchSize, heads: heads, headDimension: headDimension, sequenceDimensionA: hw,
        sequenceDimensionB: hw)
    }

    if !contextBlockPreOnly {
      total += DenseInstructionCount(rows: rowsContext, input: channels, output: channels)  // c_o
      total += DenseInstructionCount(rows: rowsContext, input: channels, output: channels * 4)  // c mlp up
      total += DenseInstructionCount(rows: rowsContext, input: channels * 4, output: channels)  // c mlp down
    }

    total += DenseInstructionCount(rows: rowsImage, input: channels, output: channels)  // x_o
    if useDualAttention {
      total += DenseInstructionCount(rows: rowsImage, input: channels, output: channels)  // x_o_2
    }

    total += DenseInstructionCount(rows: rowsImage, input: channels, output: channels * 4)  // x mlp up
    total += DenseInstructionCount(rows: rowsImage, input: channels * 4, output: channels)  // x mlp down
  }

  total += DenseInstructionCount(rows: rowsImage, input: channels, output: 2 * 2 * 16)
  return total
}

// Extra shape params are needed because `MMDiTFixed(...)` does not include context token length
// or input feature widths, but they affect exact GEMM counts.
public func MMDiTFixedInstructionCount(
  batchSize: Int, contextLength: Int, channels: Int, layers: Int, dualAttentionLayers: [Int],
  contextInputChannels: Int = 4096, pooledInputChannels: Int = 2048
) -> Int {
  precondition(channels % 64 == 0)
  var total = 0

  // t_embedder and y_embedder MLPs.
  total += DenseInstructionCount(rows: batchSize, input: 256, output: channels)
  total += DenseInstructionCount(rows: batchSize, input: channels, output: channels)
  total += DenseInstructionCount(rows: batchSize, input: pooledInputChannels, output: channels)
  total += DenseInstructionCount(rows: batchSize, input: channels, output: channels)

  // context_embedder on text tokens.
  total += DenseInstructionCount(
    rows: batchSize * contextLength, input: contextInputChannels, output: channels)

  for i in 0..<layers {
    let contextBlockPreOnly = (i == layers - 1)
    let useDualAttention = dualAttentionLayers.contains(i)
    let contextAdaLNCount = contextBlockPreOnly ? 2 : 6
    let xAdaLNCount = useDualAttention ? 9 : 6
    total +=
      (contextAdaLNCount + xAdaLNCount)
      * DenseInstructionCount(rows: batchSize, input: channels, output: channels)
  }

  // Final shift + scale.
  total += 2 * DenseInstructionCount(rows: batchSize, input: channels, output: channels)

  return total
}
