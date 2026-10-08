import AudioConverter
import XCTest

#if canImport(AVFoundation)
  import AVFoundation
#endif

final class AudioInputTests: XCTestCase {
  func testVideoFrameCountRoundsUp() {
    let input = AudioInput(samples: [0, 0, 0], sampleRate: 2)

    XCTAssertEqual(input.videoFrameCount(framesPerSecond: 1), 2)
  }

  func testWaveformTensorPadsAndDuplicatesMono() {
    let input = AudioInput(samples: [0.25, -0.5], sampleRate: 4)

    let waveform = input.waveformTensor(videoFrames: 1, framesPerSecond: 1)

    XCTAssertEqual(waveform.shape, [2, 4])
    XCTAssertEqual(waveform[0, 0], 0.25)
    XCTAssertEqual(waveform[0, 1], -0.5)
    XCTAssertEqual(waveform[0, 2], 0)
    XCTAssertEqual(waveform[0, 3], 0)
    for i in 0..<4 {
      XCTAssertEqual(waveform[0, i], waveform[1, i])
    }
  }

  func testConditioningWaveformPreservesUntrimmedSamples() {
    let input = AudioInput(samples: [0.25, -0.5, 0.75], sampleRate: 4)
    let waveform = input.waveform

    XCTAssertEqual(waveform.shape, [1, 3])
    XCTAssertEqual(waveform[0, 0], 0.25)
    XCTAssertEqual(waveform[0, 1], -0.5)
    XCTAssertEqual(waveform[0, 2], 0.75)
    // Export padding must not change the waveform supplied for loudness normalization.
    _ = input.waveformTensor(videoFrames: 1, framesPerSecond: 1)
    XCTAssertEqual(input.waveform.shape, [1, 3])
  }

  func testRejectsInvalidSampleRate() {
    XCTAssertThrowsError(try AudioInput(contentsOf: "/tmp/missing.caf", sampleRate: 0))
  }

  #if canImport(AVFoundation)
    func testResamplesStereoWithoutMixingChannels() throws {
      let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "AudioConverterTests-\(UUID().uuidString).caf")
      defer { try? FileManager.default.removeItem(at: url) }

      try autoreleasepool {
        let sourceFrames: AVAudioFrameCount = 800
        let sourceFormat = try XCTUnwrap(
          AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 8_000, channels: 2,
            interleaved: false))
        let file = try AVAudioFile(
          forWriting: url, settings: sourceFormat.settings, commonFormat: .pcmFormatFloat32,
          interleaved: false)
        let buffer = try XCTUnwrap(
          AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: sourceFrames))
        let channels = try XCTUnwrap(buffer.floatChannelData)
        buffer.frameLength = sourceFrames
        for i in 0..<Int(sourceFrames) {
          channels[0][i] = 0.25
          channels[1][i] = 0.5
        }
        try file.write(from: buffer)
      }

      let input = try AudioInput(contentsOf: url.path, sampleRate: 16_000)

      XCTAssertEqual(input.sampleRate, 16_000)
      XCTAssertGreaterThanOrEqual(input.waveform.shape[1], 1_590)
      XCTAssertLessThanOrEqual(input.waveform.shape[1], 1_610)
      XCTAssertEqual(input.waveform.shape[0], 2)
      XCTAssertEqual(input.waveform[0, 800], 0.25, accuracy: 0.005)
      XCTAssertEqual(input.waveform[1, 800], 0.5, accuracy: 0.005)
      let exported = input.waveformTensor(videoFrames: 1, framesPerSecond: 10)
      XCTAssertEqual(exported[0, 800], 0.25, accuracy: 0.005)
      XCTAssertEqual(exported[1, 800], 0.5, accuracy: 0.005)
    }
  #endif
}
