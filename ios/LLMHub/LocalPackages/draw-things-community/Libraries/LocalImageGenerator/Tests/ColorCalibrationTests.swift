import DataModels
import Diffusion
import NNC
import XCTest

@testable import LocalImageGenerator

final class ColorCalibrationTests: XCTestCase {
  func testNoneIsNoOp() {
    let image = makeTensor(width: 4, height: 4) { x, y, c in
      Float(x + y + c) / 10 - 0.5
    }
    let result = ColorCalibrationProcessor.apply(
      to: [image], reference: image, mask: nil, colorCalibration: .disabled,
      isTransparentDecoder: false)
    XCTAssertEqual(result.count, 1)
    XCTAssertTensorEqual(result[0], image)
  }

  func testLABPreservesShapeAndRange() {
    let image = makeTensor(width: 8, height: 8) { x, y, c in
      Float((x * 3 + y * 5 + c * 7) % 17) / 8.5 - 1
    }
    let reference = makeTensor(width: 8, height: 8) { x, y, c in
      Float((x * 11 + y * 2 + c * 13) % 19) / 9.5 - 1
    }
    let result = ColorCalibrationProcessor.apply(
      to: [image], reference: reference, mask: nil, colorCalibration: .lab,
      isTransparentDecoder: false)[0]
    XCTAssertEqual(result.shape, image.shape)
    for value in tensorValues(result) {
      XCTAssertGreaterThanOrEqual(value, -1.0001)
      XCTAssertLessThanOrEqual(value, 1.0001)
    }
  }

  func testLABMovesGeneratedColorsTowardReference() {
    let image = makeTensor(width: 8, height: 8) { _, _, c in
      c == 0 ? 0.8 : -0.8
    }
    let reference = makeTensor(width: 8, height: 8) { _, _, c in
      c == 2 ? 0.8 : -0.8
    }
    let result = ColorCalibrationProcessor.apply(
      to: [image], reference: reference, mask: nil, colorCalibration: .lab,
      isTransparentDecoder: false)[0]
    let before = averageRGB(image)
    let after = averageRGB(result)
    XCTAssertGreaterThan(after.b, before.b)
    XCTAssertLessThan(after.r, before.r)
  }

  func testLABPreservesExtraChannels() {
    let image = makeTensor(width: 6, height: 6, channels: 4) { _, _, c in
      c == 0 ? 0.25 : (c == 1 ? 0.8 : -0.8)
    }
    let reference = makeTensor(width: 6, height: 6) { _, _, c in
      c == 2 ? 0.8 : -0.8
    }
    let result = ColorCalibrationProcessor.apply(
      to: [image], reference: reference, mask: nil, colorCalibration: .lab,
      isTransparentDecoder: false)[0]
    for y in 0..<6 {
      for x in 0..<6 {
        XCTAssertEqual(Float(result[0, y, x, 0]), 0.25, accuracy: 0.0001)
      }
    }
  }

  func testLABResizesReferenceToOutputShape() {
    let image = makeTensor(width: 8, height: 8) { _, _, c in
      c == 0 ? 0.8 : -0.8
    }
    let reference = makeTensor(width: 2, height: 2) { x, y, c in
      c == ((x + y) % 3) ? 0.8 : -0.8
    }
    let result = ColorCalibrationProcessor.apply(
      to: [image], reference: reference, mask: nil, colorCalibration: .lab,
      isTransparentDecoder: false)[0]
    XCTAssertEqual(result.shape, image.shape)
  }

  func testMaskAwareLABLeavesRetainedPixelsUnchanged() {
    let image = makeTensor(width: 4, height: 1) { _, _, c in
      c == 0 ? 0.8 : -0.8
    }
    let reference = makeTensor(width: 4, height: 1) { _, _, c in
      c == 2 ? 0.8 : -0.8
    }
    var mask = Tensor<UInt8>(.CPU, .NC(1, 4))
    mask[0, 0] = 0
    mask[0, 1] = 0
    mask[0, 2] = 2
    mask[0, 3] = 2
    let result = ColorCalibrationProcessor.apply(
      to: [image], reference: reference, mask: mask, colorCalibration: .lab,
      isTransparentDecoder: false)[0]
    for x in 0..<2 {
      XCTAssertEqual(Float(result[0, 0, x, 0]), Float(image[0, 0, x, 0]), accuracy: 0.0001)
      XCTAssertEqual(Float(result[0, 0, x, 1]), Float(image[0, 0, x, 1]), accuracy: 0.0001)
      XCTAssertEqual(Float(result[0, 0, x, 2]), Float(image[0, 0, x, 2]), accuracy: 0.0001)
    }
    XCTAssertGreaterThan(Float(result[0, 0, 2, 2]), Float(image[0, 0, 2, 2]))
    XCTAssertGreaterThan(Float(result[0, 0, 3, 2]), Float(image[0, 0, 3, 2]))
  }

  private func makeTensor(
    width: Int, height: Int, channels: Int = 3, value: (Int, Int, Int) -> Float
  ) -> Tensor<FloatType> {
    var tensor = Tensor<FloatType>(.CPU, .NHWC(1, height, width, channels))
    for y in 0..<height {
      for x in 0..<width {
        for c in 0..<channels {
          tensor[0, y, x, c] = FloatType(value(x, y, c))
        }
      }
    }
    return tensor
  }

  private func tensorValues(_ tensor: Tensor<FloatType>) -> [Float] {
    let shape = tensor.shape
    var values = [Float]()
    values.reserveCapacity(shape.reduce(1, *))
    for n in 0..<shape[0] {
      for y in 0..<shape[1] {
        for x in 0..<shape[2] {
          for c in 0..<shape[3] {
            values.append(Float(tensor[n, y, x, c]))
          }
        }
      }
    }
    return values
  }

  private func averageRGB(_ tensor: Tensor<FloatType>) -> (r: Float, g: Float, b: Float) {
    let shape = tensor.shape
    let channelStart = shape[3] - 3
    var r: Float = 0
    var g: Float = 0
    var b: Float = 0
    let count = Float(shape[0] * shape[1] * shape[2])
    for n in 0..<shape[0] {
      for y in 0..<shape[1] {
        for x in 0..<shape[2] {
          r += Float(tensor[n, y, x, channelStart])
          g += Float(tensor[n, y, x, channelStart + 1])
          b += Float(tensor[n, y, x, channelStart + 2])
        }
      }
    }
    return (r / count, g / count, b / count)
  }

  private func XCTAssertTensorEqual(
    _ lhs: Tensor<FloatType>, _ rhs: Tensor<FloatType>, file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(lhs.shape, rhs.shape, file: file, line: line)
    let lhsValues = tensorValues(lhs)
    let rhsValues = tensorValues(rhs)
    XCTAssertEqual(lhsValues.count, rhsValues.count, file: file, line: line)
    for i in 0..<lhsValues.count {
      XCTAssertEqual(lhsValues[i], rhsValues[i], accuracy: 0.0001, file: file, line: line)
    }
  }
}
