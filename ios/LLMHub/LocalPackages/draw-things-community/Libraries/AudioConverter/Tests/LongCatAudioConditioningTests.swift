import NNC
import XCTest

@testable import AudioConverter

final class LongCatAudioConditioningTests: XCTestCase {
  func testRejectsFeatureTensorsAsWaveforms() {
    let encoder = LongCatAudioConditioningEncoder(filePath: "/missing-whisper.ckpt")
    let features = Tensor<Float>(.CPU, .HWC(1, 1, 32_000))
    XCTAssertThrowsError(try encoder.encode(features, videoFrames: 93, framesPerSecond: 25)) {
      guard case LongCatAudioConditioningEncoderError.invalidWaveform = $0 else {
        return XCTFail("Expected waveform validation before checkpoint loading: \($0)")
      }
    }
  }

  func testCancellationPrecedesCheckpointLoading() {
    let encoder = LongCatAudioConditioningEncoder(filePath: "/missing-whisper.ckpt")
    let waveform = AudioInput(samples: [0, 0.5, 0], sampleRate: 16_000).waveform
    XCTAssertThrowsError(
      try encoder.encode(waveform, videoFrames: 93, framesPerSecond: 25, shouldContinue: { false })
    ) {
      guard case LongCatAudioConditioningEncoderError.cancelled = $0 else {
        return XCTFail("Expected cancellation before checkpoint loading: \($0)")
      }
    }
  }

  func testAVCWindowsKeepFullAudioContext() {
    let blockSize = 5 * 1_280
    let features = LongCatAudioFeatures(
      values: (0..<253).flatMap { Array(repeating: Float($0), count: blockSize) },
      videoFrames: 253, framesPerSecond: 25)
    let first = features.conditioning(videoFrames: 93)
    let second = features.conditioning(startFrame: 80, videoFrames: 93)
    let last = features.conditioning(startFrame: 160, videoFrames: 93)

    XCTAssertEqual(first.audioFirst.shape, [1, 1, 32_000])
    XCTAssertEqual(first.audioLatter.shape, [1, 23, 51_200])
    for (window, frame) in [0, 0, 0, 1, 2].enumerated() {
      XCTAssertEqual(Float(first.audioFirst[0, 0, window * blockSize]), Float(frame))
    }
    for window in 0..<5 {
      XCTAssertEqual(Float(second.audioFirst[0, 0, window * blockSize]), Float(78 + window))
    }
    // The final window of segment 2 reaches past its end (frame 172) into the full audio.
    XCTAssertEqual(Float(second.audioLatter[0, 22, 7 * blockSize]), 174)
    // Only the end of the entire encoded span clamps the window.
    XCTAssertEqual(Float(last.audioLatter[0, 22, 7 * blockSize]), 252)
  }

  func testPackedAVCSegmentsMatchFullAudioWindows() {
    let blockSize = 5 * 1_280
    let features = LongCatAudioFeatures(
      values: (0..<(253 * blockSize)).map { Float($0 % 1_021) / 1_021 },
      videoFrames: 253, framesPerSecond: 25)
    let encoded = features.conditioning()
    for (start, frames) in [(0, 253), (0, 93), (4, 93), (80, 93), (160, 93)] {
      let expected = features.conditioning(startFrame: start, videoFrames: frames)
      let actual = encoded.segment(startFrame: start, videoFrames: frames)
      XCTAssertEqual(actual.audioFirst.shape, expected.audioFirst.shape)
      XCTAssertEqual(actual.audioLatter.shape, expected.audioLatter.shape)
      XCTAssertEqual(
        Array(actual.audioFirst), Array(expected.audioFirst), "First frame at \(start)")
      XCTAssertEqual(
        Array(actual.audioLatter), Array(expected.audioLatter), "Later frames at \(start)")
    }
  }
}
