import Darwin
import Diffusion
import Foundation
import FrameCompression
import NNC

private enum CompressionTestError: Swift.Error, CustomStringConvertible {
  case shapeMismatch(String, String)
  case expectedMoreArtifacts(Double, Double)
  case expectedSomeArtifacts(Double)
  case decodedRangeOutOfBounds(Float, Float)

  var description: String {
    switch self {
    case .shapeMismatch(let lhs, let rhs):
      return "Decoded tensor shape mismatch: \(lhs) vs \(rhs)"
    case .expectedMoreArtifacts(let highQualityMAE, let lowQualityMAE):
      return
        "Expected low-quality run to have higher MAE. high=\(highQualityMAE), low=\(lowQualityMAE)"
    case .expectedSomeArtifacts(let mae):
      return "Expected non-zero compression artifacts, got MAE=\(mae)"
    case .decodedRangeOutOfBounds(let minValue, let maxValue):
      return "Decoded values should stay in [-1, 1], got min=\(minValue), max=\(maxValue)"
    }
  }
}

private func shapeString(_ tensor: Tensor<FloatType>) -> String {
  tensor.shape.map(String.init).joined(separator: "x")
}

private func buildPatternTensor(width: Int, height: Int) -> Tensor<FloatType> {
  var tensor = Tensor<FloatType>(.CPU, .NHWC(1, height, width, 3))
  tensor.withUnsafeMutableBytes { buffer in
    guard let ptr = buffer.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
    for y in 0..<height {
      for x in 0..<width {
        let fx = Float(x) / Float(max(width - 1, 1))
        let fy = Float(y) / Float(max(height - 1, 1))
        let checker = ((x / 8 + y / 8) & 1) == 0 ? Float(1) : Float(-1)
        let r = max(-1, min(1, sin(fx * .pi * 10) * 0.8 + cos(fy * .pi * 6) * 0.2))
        let g = checker * 0.95
        let b = max(-1, min(1, (fx * 2 - 1) * 0.6 + (fy * 2 - 1) * 0.4))
        let offset = y * width * 3 + x * 3
        ptr[offset] = FloatType(r)
        ptr[offset + 1] = FloatType(g)
        ptr[offset + 2] = FloatType(b)
      }
    }
  }
  return tensor
}

private func meanAbsoluteError(_ lhs: Tensor<FloatType>, _ rhs: Tensor<FloatType>) -> Double {
  precondition(lhs.shape == rhs.shape)
  let count = lhs.shape[0] * lhs.shape[1] * lhs.shape[2] * lhs.shape[3]
  var sum = 0.0
  lhs.withUnsafeBytes { lhsBuffer in
    guard let lhsPtr = lhsBuffer.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
      return
    }
    rhs.withUnsafeBytes { rhsBuffer in
      guard let rhsPtr = rhsBuffer.baseAddress?.assumingMemoryBound(to: FloatType.self) else {
        return
      }
      for i in 0..<count {
        sum += Double(abs(Float(lhsPtr[i]) - Float(rhsPtr[i])))
      }
    }
  }
  return sum / Double(count)
}

private func minMax(_ tensor: Tensor<FloatType>) -> (Float, Float) {
  let count = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3]
  var minValue = Float.greatestFiniteMagnitude
  var maxValue = -Float.greatestFiniteMagnitude
  tensor.withUnsafeBytes { buffer in
    guard let ptr = buffer.baseAddress?.assumingMemoryBound(to: FloatType.self) else { return }
    for i in 0..<count {
      let value = Float(ptr[i])
      minValue = min(minValue, value)
      maxValue = max(maxValue, value)
    }
  }
  return (minValue, maxValue)
}

private func runCodecTest(_ codec: CompressionCodec, input: Tensor<FloatType>) throws {
  let highQuality = try FrameCompression.applyCompressionArtifacts(
    to: input, codec: codec, quality: 95)
  let lowQuality = try FrameCompression.applyCompressionArtifacts(
    to: input, codec: codec, quality: 10)

  guard input.shape == highQuality.shape else {
    throw CompressionTestError.shapeMismatch(shapeString(input), shapeString(highQuality))
  }
  guard input.shape == lowQuality.shape else {
    throw CompressionTestError.shapeMismatch(shapeString(input), shapeString(lowQuality))
  }

  let highQualityMAE = meanAbsoluteError(input, highQuality)
  let lowQualityMAE = meanAbsoluteError(input, lowQuality)
  let (highMin, highMax) = minMax(highQuality)
  let (lowMin, lowMax) = minMax(lowQuality)

  guard highQualityMAE > 0.0001 else {
    throw CompressionTestError.expectedSomeArtifacts(highQualityMAE)
  }
  guard lowQualityMAE > highQualityMAE + 0.001 else {
    throw CompressionTestError.expectedMoreArtifacts(highQualityMAE, lowQualityMAE)
  }
  guard highMin >= -1.0001, highMax <= 1.0001 else {
    throw CompressionTestError.decodedRangeOutOfBounds(highMin, highMax)
  }
  guard lowMin >= -1.0001, lowMax <= 1.0001 else {
    throw CompressionTestError.decodedRangeOutOfBounds(lowMin, lowMax)
  }

  print("Codec \(codec.rawValue) passed.")
  print("  High quality MAE: \(highQualityMAE)")
  print("  Low quality MAE:  \(lowQualityMAE)")
  print("  High quality range: [\(highMin), \(highMax)]")
  print("  Low quality range:  [\(lowMin), \(lowMax)]")
}

private func run() throws {
  let input = buildPatternTensor(width: 256, height: 256)
  try runCodecTest(.jpeg, input: input)
  try runCodecTest(.h264, input: input)

  do {
    try runCodecTest(.h265, input: input)
  } catch let error as FrameCompression.Error {
    switch error {
    case .codecUnavailable(.h265, _):
      print("Codec h265 unavailable on this machine, skipping.")
    default:
      throw error
    }
  }
}

do {
  try run()
} catch {
  fputs("FrameCompression test failed: \(error)\n", stderr)
  exit(1)
}
