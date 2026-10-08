import Diffusion
import NNC
import XCTest

@testable import LocalImageGenerator

final class LongCatAvatarTests: XCTestCase {
  func testAVCCoversAudioWithoutAddingAnExtraSegmentAtExactBoundaries() throws {
    for (target, count, generated) in [
      (1, 1, 93), (93, 1, 93), (94, 2, 173), (173, 2, 173), (200, 3, 253),
    ] {
      let plan = try LongCatAvatarAVCPlan(
        segmentFrames: 93, conditionFrames: 13, targetVideoFrames: target)
      XCTAssertEqual(plan.segmentCount, count)
      XCTAssertEqual(plan.generatedVideoFrames, generated)
      XCTAssertEqual(plan.stride, 80)
    }
  }

  func testAVCRejectsInvalidTemporalAlignmentAndOverlap() {
    for (segment, condition) in [(92, 13), (93, 12), (93, 93), (93, 0)] {
      XCTAssertThrowsError(
        try LongCatAvatarAVCPlan(
          segmentFrames: segment, conditionFrames: condition, targetVideoFrames: 200))
    }
  }

  func testContinuationKeepsReferenceThenLastFramesInOrder() throws {
    let reference = Tensor<FloatType>(
      Array(repeating: FloatType(-1), count: 12), .CPU, .NHWC(1, 2, 2, 3))
    let frames = (0..<7).map {
      Tensor<FloatType>(Array(repeating: FloatType($0), count: 12), .CPU, .NHWC(1, 2, 2, 3))
    }
    let input = try longCatImageInput(
      referenceImage: reference, continuationFrames: frames.suffix(5), conditionFrames: 5)
    XCTAssertEqual(input.shape, [6, 2, 2, 3])
    XCTAssertEqual(input[0, 0, 0, 0], -1)
    for frame in 1..<6 {
      XCTAssertEqual(input[frame, 1, 1, 2], FloatType(frame + 1))
    }
    XCTAssertThrowsError(
      try longCatImageInput(
        referenceImage: reference, continuationFrames: frames.suffix(3), conditionFrames: 5))
    let wrongSize = Tensor<FloatType>(.CPU, .NHWC(1, 1, 1, 3))
    XCTAssertThrowsError(
      try longCatImageInput(
        referenceImage: reference, continuationFrames: [wrongSize][...], conditionFrames: 1))
  }
}
