import Foundation

@inline(__always)
public func DenseInstructionCount(rows: Int, input: Int, output: Int) -> Int {
  precondition(rows >= 0 && input >= 0 && output >= 0)
  return rows * output * input  // M * N * K (FMA)
}

@inline(__always)
public func ScaledDotProductAttentionInstructionCount(
  batchSize: Int, heads: Int, headDimension: Int, sequenceDimensionA: Int, sequenceDimensionB: Int
) -> Int {
  precondition(
    batchSize >= 0 && heads >= 0 && headDimension >= 0 && sequenceDimensionA >= 0
      && sequenceDimensionB >= 0)
  return batchSize * heads * (2 * headDimension + 5) * sequenceDimensionA * sequenceDimensionB
}

@inline(__always)
public func ConvolutionInstructionCount(
  batchSize: Int, outHeight: Int, outWidth: Int, outChannels: Int, kernelHeight: Int,
  kernelWidth: Int, inputChannels: Int, groups: Int = 1
) -> Int {
  precondition(
    batchSize >= 0 && outHeight >= 0 && outWidth >= 0 && outChannels >= 0 && kernelHeight >= 0
      && kernelWidth >= 0 && inputChannels >= 0 && groups > 0)
  precondition(inputChannels % groups == 0)
  let kernelK = kernelHeight * kernelWidth * (inputChannels / groups)
  let outputElements = batchSize * outHeight * outWidth * outChannels
  return outputElements * kernelK
}
