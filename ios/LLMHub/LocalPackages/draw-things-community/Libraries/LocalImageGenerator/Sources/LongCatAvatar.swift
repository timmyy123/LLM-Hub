import DataModels
import Diffusion
import Foundation
import ImageGenerator
import ModelZoo
import NNC

public enum LongCatAvatarError: Error, LocalizedError {
  case invalidConfiguration
  case invalidImage
  case invalidAudio
  case generationFailed
  case cancelled

  public var errorDescription: String? {
    switch self {
    case .invalidConfiguration:
      return
        "LongCat AVC requires a LongCat model, guidance scale 1, and 4k + 1 segment and condition frame counts, with fewer condition frames than segment frames."
    case .invalidImage:
      return
        "LongCat AVC requires a reference image and continuation frames with matching dimensions."
    case .invalidAudio:
      return
        "LongCat AVC requires a nonempty CPU audio waveform in [channels, samples] order at 16000 Hz."
    case .generationFailed:
      return "LongCat AVC segment generation failed."
    case .cancelled:
      return "LongCat AVC cancelled."
    }
  }
}

struct LongCatAvatarAVCPlan {
  let segmentFrames: Int
  let conditionFrames: Int
  let targetVideoFrames: Int
  let stride: Int
  let segmentCount: Int
  let generatedVideoFrames: Int

  init(segmentFrames: Int, conditionFrames: Int, targetVideoFrames: Int) throws {
    guard targetVideoFrames > 0, conditionFrames > 0, segmentFrames > conditionFrames,
      segmentFrames % 4 == 1, conditionFrames % 4 == 1
    else { throw LongCatAvatarError.invalidConfiguration }
    self.segmentFrames = segmentFrames
    self.conditionFrames = conditionFrames
    self.targetVideoFrames = targetVideoFrames
    stride = segmentFrames - conditionFrames
    segmentCount =
      targetVideoFrames <= segmentFrames
      ? 1 : ((targetVideoFrames - segmentFrames + stride - 1) / stride) + 1
    generatedVideoFrames = segmentFrames + (segmentCount - 1) * stride
  }
}

extension LocalImageGenerator {
  /// Runs LongCat's segmented continuation on the caller's generation queue. The full 16 kHz
  /// waveform is encoded once so feature interpolation and audio windows span segment boundaries.
  /// `segmentStart` receives a zero-based index and the total segment count.
  public func generateLongCatAvatarAVC(
    trace: ImageGeneratorTrace, image: Tensor<FloatType>, audio: Tensor<Float>,
    text: String, negativeText: String, configuration: GenerationConfiguration,
    conditionFrames: Int = 13, fileMapping: [String: String] = [:],
    zeroAudioFeatures: Bool = false,
    cancellation: (@escaping () -> Void) -> Void,
    segmentStart: (Int, Int) -> Bool,
    feedback:
      @escaping (ImageGeneratorSignpost, Set<ImageGeneratorSignpost>, Tensor<FloatType>?) -> Bool
  ) throws -> [Tensor<FloatType>] {
    dispatchPrecondition(condition: .onQueue(queue))
    guard let model = configuration.model,
      ModelZoo.versionForModel(model) == .longcatVideoAvatar1_5,
      configuration.guidanceScale == 1
    else { throw LongCatAvatarError.invalidConfiguration }
    guard audio.kind == .CPU, audio.shape.count == 2, audio.shape[0] > 0, audio.shape[1] > 0
    else { throw LongCatAvatarError.invalidAudio }
    guard image.kind == .CPU, image.shape.count == 4, image.shape[0] == 1,
      image.shape[1] > 0, image.shape[2] > 0, image.shape[3] >= 3
    else { throw LongCatAvatarError.invalidImage }
    let fps = max(Int(ModelZoo.framesPerSecondForModel(model).rounded()), 1)
    let sampleRate = ModelZoo.audioSampleRateForModel(model)
    let plan = try LongCatAvatarAVCPlan(
      segmentFrames: Int(configuration.numFrames), conditionFrames: conditionFrames,
      targetVideoFrames: (audio.shape[1] * fps + sampleRate - 1) / sampleRate)
    let encoderFilePath = ModelZoo.audioEncoderForModel(model).map {
      fileMapping[$0] ?? ModelZoo.filePathForModelDownloaded($0)
    }
    // The first segment encodes the full audio inside encodeAudioCond. Later segments slice
    // that encoded conditioning so interpolation and audio windows remain continuous.
    let audioContext = AudioConditioningContext(
      waveform: audio, encoderFilePath: encoderFilePath, videoFrames: plan.generatedVideoFrames,
      zeroAudioFeatures: zeroAudioFeatures)
    var allFrames = [Tensor<FloatType>]()
    var currentSegmentFrames = [Tensor<FloatType>]()
    for segmentIndex in 0..<plan.segmentCount {
      guard segmentStart(segmentIndex, plan.segmentCount) else {
        throw LongCatAvatarError.cancelled
      }
      audioContext.startFrame = segmentIndex * plan.stride
      let inputImage =
        segmentIndex == 0
        ? image
        : try longCatImageInput(
          referenceImage: image,
          continuationFrames: currentSegmentFrames.suffix(conditionFrames),
          conditionFrames: conditionFrames)
      var builder = GenerationConfigurationBuilder(from: configuration)
      builder.seed = configuration.seed &+ UInt32(segmentIndex)
      let result = generate(
        trace: trace, image: inputImage, scaleFactor: 1, mask: nil, hints: [],
        text: text, negativeText: negativeText, configuration: builder.build(),
        fileMapping: fileMapping, keywords: [], audioContext: audioContext,
        cancellation: cancellation, feedback: feedback)
      guard let frames = result.0, frames.count == plan.segmentFrames else {
        throw LongCatAvatarError.generationFailed
      }
      currentSegmentFrames = frames
      allFrames.append(
        contentsOf: segmentIndex == 0 ? frames[...] : frames.dropFirst(conditionFrames))
      if allFrames.count > plan.targetVideoFrames {
        allFrames.removeLast(allFrames.count - plan.targetVideoFrames)
      }
    }
    return allFrames
  }
}

func longCatImageInput(
  referenceImage: Tensor<FloatType>, continuationFrames: ArraySlice<Tensor<FloatType>>,
  conditionFrames: Int
) throws -> Tensor<FloatType> {
  let shape = referenceImage.shape
  guard continuationFrames.count == conditionFrames, referenceImage.kind == .CPU,
    shape.count == 4, shape[0] == 1, shape[3] >= 3
  else { throw LongCatAvatarError.invalidImage }
  let height = shape[1]
  let width = shape[2]
  let channels = shape[3]
  var tensor = Tensor<FloatType>(.CPU, .NHWC(conditionFrames + 1, height, width, channels))
  tensor[0..<1, 0..<height, 0..<width, 0..<channels] = referenceImage
  for (index, frame) in continuationFrames.enumerated() {
    guard frame.kind == .CPU, frame.shape == shape else { throw LongCatAvatarError.invalidImage }
    tensor[(index + 1)..<(index + 2), 0..<height, 0..<width, 0..<channels] = frame
  }
  return tensor
}
