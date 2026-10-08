import Diffusion
import Foundation
import NNC

public enum LongCatAudioConditioningEncoderError: Swift.Error, LocalizedError {
  case missingAudioEncoder(String)
  case invalidWaveform
  case invalidFrameCount
  case cancelled

  public var errorDescription: String? {
    switch self {
    case .missingAudioEncoder(let path):
      return
        "Missing Whisper audio encoder at \(path). Convert it with ModelConverter --audio-encoder."
    case .invalidWaveform:
      return "LongCat audio input must be a nonempty CPU waveform in [channels, samples] order."
    case .invalidFrameCount:
      return "LongCat audio conditioning requires a positive frame count and frame rate."
    case .cancelled:
      return "Audio encoding cancelled."
    }
  }
}

public struct LongCatAudioConditioning {
  public let audioFirst: Tensor<FloatType>
  public let audioLatter: Tensor<FloatType>

  public init(audioFirst: Tensor<FloatType>, audioLatter: Tensor<FloatType>) {
    self.audioFirst = audioFirst
    self.audioLatter = audioLatter
  }

  public var tensors: [Tensor<FloatType>] {
    [audioFirst, audioLatter]
  }

  /// Slices packed full-video conditioning at a VAE-aligned AVC segment boundary.
  public func segment(startFrame: Int, videoFrames: Int) -> LongCatAudioConditioning {
    let scale = LongCatAudioConditioningEncoder.vaeScale
    precondition(startFrame >= 0 && startFrame % scale == 0)
    precondition(videoFrames > 0 && videoFrames % scale == 1)
    let start = startFrame / scale
    let count = (videoFrames - 1) / scale
    precondition(start + count <= audioLatter.shape[1])
    if start == 0 && count == audioLatter.shape[1] {
      return self
    }
    let first: Tensor<FloatType>
    if start == 0 {
      first = audioFirst
    } else {
      // The previous latent group's final five windows are centered on this segment's
      // first frame. Reuse them to preserve context from both sides of the boundary.
      let width = audioLatter.shape[2]
      first = audioLatter[
        0..<1, (start - 1)..<start, (width - audioFirst.shape[2])..<width
      ].copied()
    }
    let latter = audioLatter[0..<1, start..<(start + count), 0..<audioLatter.shape[2]].copied()
    return LongCatAudioConditioning(audioFirst: first, audioLatter: latter)
  }
}

public struct LongCatAudioFeatures {
  fileprivate let values: [Float]
  public let videoFrames: Int
  public let framesPerSecond: Int

  init(values: [Float], videoFrames: Int, framesPerSecond: Int) {
    precondition(
      values.count
        == videoFrames * LongCatAudioConditioningEncoder.audioBlocks
        * LongCatAudioConditioningEncoder.audioChannels)
    self.values = values
    self.videoFrames = videoFrames
    self.framesPerSecond = framesPerSecond
  }

  public static func zero(videoFrames: Int, framesPerSecond: Int) -> LongCatAudioFeatures {
    LongCatAudioFeatures(
      values: [Float](
        repeating: 0,
        count: videoFrames * LongCatAudioConditioningEncoder.audioBlocks
          * LongCatAudioConditioningEncoder.audioChannels),
      videoFrames: videoFrames, framesPerSecond: framesPerSecond)
  }
}

public struct LongCatAudioConditioningEncoder {
  public static let sampleRate = 16_000
  public static let audioWindow = 5
  public static let audioBlocks = 5
  public static let audioChannels = 1_280
  public static let vaeScale = 4
  public let filePath: String

  public init(filePath: String) {
    self.filePath = filePath
  }

  // Approximate the pipeline's -23 LUFS loudness normalization with an RMS-based gain. The
  // reference implementation uses ITU-R BS.1770 gating; RMS tracks it closely for speech.
  private static func loudnessNormalize(_ samples: [Float], targetLUFS: Float = -23) -> [Float] {
    var sumOfSquares: Double = 0
    for sample in samples {
      sumOfSquares += Double(sample) * Double(sample)
    }
    let rms = (sumOfSquares / Double(samples.count)).squareRoot()
    guard rms > 1e-8 else { return samples }
    let loudness = Float(20 * log10(rms))
    guard abs(loudness) < 100 else { return samples }
    let gain = pow(10, (targetLUFS - loudness) / 20)
    return samples.map { $0 * gain }
  }

  /// The waveform must be resampled to 16 kHz. Downmix channels before Whisper normalization.
  public func encode(
    _ waveform: Tensor<Float>, videoFrames: Int, framesPerSecond: Int,
    shouldContinue: () -> Bool = { true }
  ) throws -> LongCatAudioFeatures {
    guard waveform.kind == .CPU, waveform.shape.count == 2,
      waveform.shape[0] > 0, waveform.shape[1] > 0
    else {
      throw LongCatAudioConditioningEncoderError.invalidWaveform
    }
    guard videoFrames > 0, framesPerSecond > 0 else {
      throw LongCatAudioConditioningEncoderError.invalidFrameCount
    }
    guard shouldContinue() else {
      throw LongCatAudioConditioningEncoderError.cancelled
    }
    var samples = [Float](repeating: 0, count: waveform.shape[1])
    for channel in 0..<waveform.shape[0] {
      for i in 0..<samples.count {
        samples[i] += waveform[channel, i] / Float(waveform.shape[0])
      }
    }
    let values = try whisperFeatures(
      samples: samples, videoFrames: videoFrames, framesPerSecond: framesPerSecond,
      shouldContinue: shouldContinue)
    return LongCatAudioFeatures(
      values: values, videoFrames: videoFrames, framesPerSecond: framesPerSecond)
  }

  /// Runs Whisper over the padded speech and returns per-video-frame features [frames, 5, 1280].
  private func whisperFeatures(
    samples: [Float], videoFrames: Int, framesPerSecond: Int, shouldContinue: () -> Bool
  ) throws -> [Float] {
    guard FileManager.default.fileExists(atPath: filePath) else {
      throw LongCatAudioConditioningEncoderError.missingAudioEncoder(filePath)
    }
    precondition(videoFrames > 0)
    precondition(framesPerSecond > 0)
    // Pad audio to the generated duration.
    let targetSamples = (videoFrames * Self.sampleRate + framesPerSecond - 1) / framesPerSecond
    var samples = Self.loudnessNormalize(samples)
    if samples.count < targetSamples {
      samples.append(contentsOf: [Float](repeating: 0, count: targetSamples - samples.count))
    }
    let encoderFPS = 50
    let chunkSamples = 30 * Self.sampleRate
    let encoderFramesPerChunk = 1_500
    let totalEncoderFrames = videoFrames * encoderFPS / framesPerSecond
    let chunks = (max(totalEncoderFrames, 1) + encoderFramesPerChunk - 1) / encoderFramesPerChunk
    let graph = DynamicGraph()
    // groupedFeatures[g][frame * channels + c]
    var groupedFeatures = [[Float]](
      repeating: [Float](
        repeating: 0, count: chunks * encoderFramesPerChunk * Self.audioChannels),
      count: Self.audioBlocks)
    try graph.withNoGrad {
      let (_, encoder) = WhisperEncoder(
        width: 1_280, layers: 32, heads: 20, melBins: 128, frames: 3_000,
        intermediateSize: 5_120, usesFlashAttention: true)
      var loaded = false
      for chunk in 0..<chunks {
        guard shouldContinue() else {
          throw LongCatAudioConditioningEncoderError.cancelled
        }
        let start = chunk * chunkSamples
        let chunkArray = Array(
          samples[min(start, samples.count)..<min(start + chunkSamples, samples.count)])
        let mel = WhisperMelSpectrogram(samples: chunkArray)
        let melGPU = graph.variable(
          Tensor<FloatType>(from: mel).reshaped(.NCHW(1, 128, 1, 3_000)).toGPU(0))
        if !loaded {
          encoder.compile(inputs: melGPU)
          graph.openStore(
            filePath, flags: .readOnly,
            externalStore: TensorData.externalStore(filePath: filePath)
          ) { store in
            store.read(
              "audio_encoder", model: encoder, codec: [.jit, .q6p, .q8p, .ezm7, .externalData])
          }
          loaded = true
        }
        let outputs = encoder(inputs: melGPU).map { $0.as(of: FloatType.self) }
        precondition(outputs.count == Self.audioBlocks)
        for (g, output) in outputs.enumerated() {
          let cpuTensor = output.rawValue.toCPU()
          for f in 0..<encoderFramesPerChunk {
            let frameIndex = chunk * encoderFramesPerChunk + f
            for c in 0..<Self.audioChannels {
              groupedFeatures[g][frameIndex * Self.audioChannels + c] = Float(cpuTensor[0, f, c])
            }
          }
        }
      }
    }
    // Linear interpolation from 50fps encoder frames to the video fps (align_corners = true).
    let sourceFrames = max(totalEncoderFrames, 2)
    var features = [Float](
      repeating: 0, count: videoFrames * Self.audioBlocks * Self.audioChannels)
    for g in 0..<Self.audioBlocks {
      for f in 0..<videoFrames {
        let position =
          videoFrames > 1
          ? Double(f) * Double(sourceFrames - 1) / Double(videoFrames - 1) : 0
        let lower = min(Int(position), sourceFrames - 1)
        let upper = min(lower + 1, sourceFrames - 1)
        let fraction = Float(position - Double(lower))
        for c in 0..<Self.audioChannels {
          let lowerValue = groupedFeatures[g][lower * Self.audioChannels + c]
          let upperValue = groupedFeatures[g][upper * Self.audioChannels + c]
          features[(f * Self.audioBlocks + g) * Self.audioChannels + c] =
            lowerValue + (upperValue - lowerValue) * fraction
        }
      }
    }
    return features
  }
}

extension LongCatAudioFeatures {
  /// Packs a segment from full-video per-frame features. Windows are clamped globally so segment
  /// boundaries match the reference AVC pipeline's audio_start_idx slicing.
  public func conditioning(startFrame: Int = 0, videoFrames segmentVideoFrames: Int? = nil)
    -> LongCatAudioConditioning
  {
    let segmentVideoFrames = segmentVideoFrames ?? videoFrames
    precondition(startFrame >= 0)
    precondition(segmentVideoFrames > 0)
    precondition(segmentVideoFrames % LongCatAudioConditioningEncoder.vaeScale == 1)
    precondition(startFrame + segmentVideoFrames <= videoFrames)
    let latentFrames =
      (segmentVideoFrames - 1) / LongCatAudioConditioningEncoder.vaeScale + 1
    let middle = LongCatAudioConditioningEncoder.audioWindow / 2
    let blockSize =
      LongCatAudioConditioningEncoder.audioBlocks * LongCatAudioConditioningEncoder.audioChannels
    func windowFeature(frame: Int, window: Int) -> ArraySlice<Float> {
      let clamped = min(max(startFrame + frame + window - middle, 0), videoFrames - 1)
      return values[(clamped * blockSize)..<((clamped + 1) * blockSize)]
    }
    var first = Tensor<FloatType>(
      .CPU, .HWC(1, 1, LongCatAudioConditioningEncoder.audioWindow * blockSize))
    for w in 0..<LongCatAudioConditioningEncoder.audioWindow {
      let slice = windowFeature(frame: 0, window: w)
      for (i, value) in slice.enumerated() {
        first[0, 0, w * blockSize + i] = FloatType(value)
      }
    }
    // Latent groups cover frames [1 + t*4, 1 + t*4 + 3]; each group contributes 8 (frame, window)
    // pairs: group frame 0 windows 0...middle, middle window of frames 1..<n-1, and group frame
    // n-1 windows middle...(window-1).
    var latter = Tensor<FloatType>(
      .CPU,
      .HWC(
        1, latentFrames - 1,
        (LongCatAudioConditioningEncoder.audioWindow
          + LongCatAudioConditioningEncoder.vaeScale - 1) * blockSize))
    for t in 0..<(latentFrames - 1) {
      var pairs = [(Int, Int)]()
      for w in 0...middle {
        pairs.append((0, w))
      }
      for n in 1..<(LongCatAudioConditioningEncoder.vaeScale - 1) {
        pairs.append((n, middle))
      }
      for w in middle..<LongCatAudioConditioningEncoder.audioWindow {
        pairs.append((LongCatAudioConditioningEncoder.vaeScale - 1, w))
      }
      precondition(
        pairs.count
          == LongCatAudioConditioningEncoder.audioWindow
          + LongCatAudioConditioningEncoder.vaeScale - 1)
      for (j, pair) in pairs.enumerated() {
        let frame = 1 + t * LongCatAudioConditioningEncoder.vaeScale + pair.0
        let slice = windowFeature(frame: frame, window: pair.1)
        for (i, value) in slice.enumerated() {
          latter[0, t, j * blockSize + i] = FloatType(value)
        }
      }
    }
    return LongCatAudioConditioning(audioFirst: first, audioLatter: latter)
  }
}
