import LLM
import NNC
import XCTest

final class DeepSeek4RotaryEmbeddingTests: XCTestCase {
  func testRawFloat32MatchesScalarReference() {
    assertMatchesScalarReference(
      sequenceLength: 5, cachedTokenLength: 17, positionStride: 1,
      compressed: false, headDim: nil, of: Float.self)
  }

  func testCompressedStridedFloat32MatchesScalarReference() {
    let configuration = DeepSeek4ModelConfiguration.deepSeekV4Flash
    assertMatchesScalarReference(
      sequenceLength: 7, cachedTokenLength: 65_531, positionStride: 4,
      compressed: true, headDim: configuration.indexerHeadDim,
      configuration: configuration, of: Float.self)
  }

  func testCompressedRotaryOnlyFloat16MatchesScalarReference() {
    let configuration = DeepSeek4ModelConfiguration.deepSeekV4Flash
    assertMatchesScalarReference(
      sequenceLength: 4, cachedTokenLength: 1_024, positionStride: 128,
      compressed: true, headDim: configuration.rotaryDim,
      configuration: configuration, of: Float16.self)
  }

  private func assertMatchesScalarReference<Scalar: TensorNumeric & BinaryFloatingPoint>(
    sequenceLength: Int, cachedTokenLength: Int, positionStride: Int,
    compressed: Bool, headDim: Int?,
    configuration: DeepSeek4ModelConfiguration = .deepSeekV4Flash,
    of dataType: Scalar.Type, file: StaticString = #filePath, line: UInt = #line
  ) {
    let rotary = DeepSeek4RotaryEmbedding(
      sequenceLength: sequenceLength, cachedTokenLength: cachedTokenLength,
      positionStride: positionStride, compressed: compressed, headDim: headDim,
      configuration: configuration, of: dataType)
    let resolvedHeadDim = headDim ?? configuration.attentionHeadDim
    XCTAssertEqual(
      rotary.shape, [1, sequenceLength, 1, resolvedHeadDim], file: file, line: line)
    let actual = values(of: rotary)
    let expected = scalarReference(
      sequenceLength: sequenceLength, cachedTokenLength: cachedTokenLength,
      positionStride: positionStride, compressed: compressed, headDim: resolvedHeadDim,
      configuration: configuration, of: dataType)
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for index in expected.indices {
      XCTAssertEqual(
        actual[index], expected[index], "Mismatch at flat index \(index)", file: file,
        line: line)
    }
  }

  private func values<Scalar: TensorNumeric>(of tensor: Tensor<Scalar>) -> [Scalar] {
    var values = [Scalar]()
    tensor.withUnsafeBytes { buffer in
      guard let pointer = buffer.baseAddress?.assumingMemoryBound(to: Scalar.self) else { return }
      values = Array(
        UnsafeBufferPointer(start: pointer, count: tensor.shape.reduce(1, *)))
    }
    return values
  }

  private func scalarReference<Scalar: TensorNumeric & BinaryFloatingPoint>(
    sequenceLength: Int, cachedTokenLength: Int, positionStride: Int,
    compressed: Bool, headDim: Int, configuration: DeepSeek4ModelConfiguration,
    of dataType: Scalar.Type
  ) -> [Scalar] {
    let nRot = configuration.rotaryDim
    let nNope = headDim - nRot
    let freqBase = compressed ? configuration.ropeTheta : 10_000
    let freqScale = compressed ? 1.0 / configuration.ropeScaleFactor : 1.0
    let extFactor = compressed ? 1.0 : 0.0
    let attnFactor = extFactor != 0 ? 1.0 / (1.0 + 0.1 * log(1.0 / freqScale)) : 1.0
    let corrStart = floor(
      Double(nRot)
        * log(
          Double(configuration.ropeOriginalContext)
            / (configuration.ropeYarnBetaFast * 2.0 * Double.pi))
        / (2.0 * log(freqBase)))
    let corrEnd = ceil(
      Double(nRot)
        * log(
          Double(configuration.ropeOriginalContext)
            / (configuration.ropeYarnBetaSlow * 2.0 * Double.pi))
        / (2.0 * log(freqBase)))
    let corr = (max(0.0, corrStart), min(Double(nRot - 1), corrEnd))
    var rotary = Array(repeating: Scalar.zero, count: sequenceLength * headDim)
    for row in 0..<sequenceLength {
      let rowOffset = row * headDim
      for i in stride(from: 0, to: nNope, by: 2) {
        rotary[rowOffset + i] = 1
        rotary[rowOffset + i + 1] = 0
      }
      let position = Double(cachedTokenLength + row * positionStride)
      for i in stride(from: 0, to: nRot, by: 2) {
        let frequency = pow(freqBase, -Double(i) / Double(nRot))
        let thetaExtrap = position * frequency
        let thetaInterp = freqScale * thetaExtrap
        let ramp =
          1.0
          - min(
            1.0,
            max(
              0.0,
              (Double(i / 2) - corr.0) / max(0.001, corr.1 - corr.0)))
        let rampMix = ramp * extFactor
        let theta = thetaInterp * (1.0 - rampMix) + thetaExtrap * rampMix
        let mscale =
          extFactor != 0
          ? attnFactor * (1.0 + 0.1 * log(1.0 / freqScale)) : attnFactor
        let offset = rowOffset + nNope + i
        rotary[offset] = Scalar(cos(theta) * mscale)
        rotary[offset + 1] = Scalar(sin(theta) * mscale)
      }
    }
    return rotary
  }
}
