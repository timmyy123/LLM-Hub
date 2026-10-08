import Foundation
import NNC
import XCTest

@testable import Diffusion

final class QwenImage2_1ContractTests: XCTestCase {
  func testBlockCausalMaskKeepsAdjacentImagesSeparate() {
    let segments: [QwenImage2_1Segment] = [
      .init(image: false, length: 2, sourceOffset: 0),
      .init(image: true, length: 4, sourceOffset: 0, height: 2, width: 2),
      .init(image: true, length: 2, sourceOffset: 4, height: 1, width: 2),
      .init(image: false, length: 2, sourceOffset: 2),
      .init(image: true, length: 4, sourceOffset: 6, height: 2, width: 2),
    ]
    let imageIDs = [-1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 2, 2, 2, 2]
    let mask = QwenImage2_1AttentionMask(segments)
    // Independent predicate from Diffusers' block-causal attention reference.
    for query in imageIDs.indices {
      for key in imageIDs.indices {
        let allowed = key <= query || (imageIDs[query] >= 0 && imageIDs[query] == imageIDs[key])
        XCTAssertEqual(mask[0, 0, query, key].isFinite, allowed, "q=\(query), k=\(key)")
      }
    }
  }

  func testImagePositionsAndFollowingText() {
    let rotary = QwenImage2_1RotaryEmbedding([
      .init(image: false, length: 2, sourceOffset: 0),
      .init(image: true, length: 8, sourceOffset: 0, height: 2, width: 4),
      .init(image: false, length: 1, sourceOffset: 2),
    ])
    // First frequency of each axis has unit angular frequency.
    XCTAssertEqual(rotary[0, 2, 0, 0], cos(Float(2)), accuracy: 1e-6)
    XCTAssertEqual(rotary[0, 2, 0, 16], cos(Float(-1)), accuracy: 1e-6)
    XCTAssertEqual(rotary[0, 2, 0, 72], cos(Float(-2)), accuracy: 1e-6)
    XCTAssertEqual(rotary[0, 9, 0, 0], cos(Float(2)), accuracy: 1e-6)
    XCTAssertEqual(rotary[0, 10, 0, 0], cos(Float(6)), accuracy: 1e-6)
  }

  func testFlowScheduleMatchesReferenceShiftAndStretch() {
    let shift = exp(0.5 + Double(1024 - 256) * 0.4 / Double(8192 - 256))
    let discretization = Denoiser.LinearDiscretization(
      .rf(.init(shiftTerminal: 0.02)), objective: .u(conditionScale: 1000),
      timestepSpacing: .trailing)
    let actual = discretization.alphasCumprod(steps: 40, shift: shift).map { 1 - $0 }
    let shifted = (0..<40).map { i -> Double in
      let sigma = 1 - Double(i) / 40
      return shift / (shift + (1 / sigma - 1))
    }
    let scale = (1 - shifted.last!) / 0.98
    for i in 0..<40 {
      XCTAssertEqual(actual[i], 1 - (1 - shifted[i]) / scale, accuracy: 1e-12)
    }
    XCTAssertEqual(actual[39], 0.02, accuracy: 1e-12)
    XCTAssertEqual(actual[40], 0)
    let original = Denoiser.LinearDiscretization(
      .rf(.init()), objective: .u(conditionScale: 1000), timestepSpacing: .trailing)
    XCTAssertEqual(original.alphasCumprod(steps: 4, shift: 1), [0, 0.25, 0.5, 0.75, 1])
    XCTAssertEqual(discretization.alphasCumprod(steps: 1, shift: shift), [0, 1])
  }
}
