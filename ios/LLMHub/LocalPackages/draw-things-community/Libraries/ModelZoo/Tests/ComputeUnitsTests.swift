import DataModels
import Diffusion
import Foundation
import XCTest

@testable import ModelZoo

final class ComputeUnitsTests: XCTestCase {
  private struct TestCase {
    let version: ModelVersion
    let width: UInt16
    let height: UInt16
    let frames: UInt32
  }

  private static let testCases: [TestCase] = [
    .init(version: .v1, width: 8, height: 8, frames: 1),
    .init(version: .v2, width: 8, height: 8, frames: 1),
    .init(version: .svdI2v, width: 8, height: 8, frames: 121),
    .init(version: .kandinsky21, width: 8, height: 8, frames: 1),
    .init(version: .sdxlBase, width: 8, height: 8, frames: 1),
    .init(version: .sdxlRefiner, width: 8, height: 8, frames: 1),
    .init(version: .ssd1b, width: 8, height: 8, frames: 1),
    .init(version: .wurstchenStageC, width: 8, height: 8, frames: 1),
    .init(version: .wurstchenStageB, width: 16, height: 16, frames: 1),
    .init(version: .sd3, width: 8, height: 8, frames: 1),
    .init(version: .sd3Large, width: 8, height: 8, frames: 1),
    .init(version: .pixart, width: 8, height: 8, frames: 1),
    .init(version: .auraflow, width: 8, height: 8, frames: 1),
    .init(version: .flux1, width: 8, height: 8, frames: 1),
    .init(version: .hunyuanVideo, width: 8, height: 8, frames: 121),
    .init(version: .wan21_1_3b, width: 8, height: 8, frames: 121),
    .init(version: .wan21_14b, width: 8, height: 8, frames: 121),
    .init(version: .wan22_5b, width: 8, height: 8, frames: 121),
    .init(version: .hiDreamI1, width: 8, height: 8, frames: 1),
    .init(version: .hiDreamO1, width: 8, height: 8, frames: 1),
    .init(version: .qwenImage, width: 8, height: 8, frames: 1),
    .init(version: .zImage, width: 8, height: 8, frames: 1),
    .init(version: .ernieImage, width: 8, height: 8, frames: 1),
    .init(version: .flux2, width: 8, height: 8, frames: 1),
    .init(version: .flux2_9b, width: 8, height: 8, frames: 1),
    .init(version: .flux2_4b, width: 8, height: 8, frames: 1),
    .init(version: .cosmos2_5_2b, width: 8, height: 8, frames: 1),
    .init(version: .ideogram4, width: 8, height: 8, frames: 1),
    .init(version: .krea2, width: 8, height: 8, frames: 1),
    .init(version: .ltx2, width: 8, height: 8, frames: 121),
    .init(version: .ltx2_3, width: 8, height: 8, frames: 121),
    .init(version: .seedvr2_3b, width: 8, height: 8, frames: 1),
    .init(version: .seedvr2_7b, width: 8, height: 8, frames: 1),
  ]

  private func configuration(
    modelName: String, width: UInt16, height: UInt16, steps: UInt32 = 20, batchSize: UInt32 = 1,
    numFrames: UInt32 = 1
  ) -> GenerationConfiguration {
    GenerationConfiguration(
      id: 1, startWidth: width, startHeight: height, steps: steps, guidanceScale: 7.5,
      strength: 1.0, model: modelName, batchSize: batchSize, numFrames: numFrames)
  }

  private func mapping(
    modelName: String, version: ModelVersion, paddedTextEncodingLength: Int? = nil
  ) -> (model: [String: ModelZoo.Specification], lora: [String: LoRAZoo.Specification]) {
    let specification = ModelZoo.Specification(
      name: modelName, file: "test.ckpt", prefix: "test", version: version,
      paddedTextEncodingLength: paddedTextEncodingLength)
    return (model: [modelName: specification], lora: [:])
  }

  func testFilePathForModelDownloadedUsesExternalFolderForNewFiles() throws {
    let previousExternalUrls = ModelZoo.externalUrls
    let previousPreference = ModelZoo.isExternalUrlsPreferred
    defer {
      ModelZoo.externalUrls = previousExternalUrls
      ModelZoo.isExternalUrlsPreferred = previousPreference
    }

    let fileManager = FileManager.default
    let fileName = "model-zoo-path-selection-\(UUID().uuidString).ckpt"
    let internalPath = ModelZoo.internalFilePathForModelDownloaded(fileName)
    let externalDirectory = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(
      UUID().uuidString, isDirectory: true)
    let externalPath = externalDirectory.appendingPathComponent(fileName).path

    try fileManager.createDirectory(at: externalDirectory, withIntermediateDirectories: true)
    defer {
      try? fileManager.removeItem(at: externalDirectory)
      try? fileManager.removeItem(atPath: internalPath)
    }
    try? fileManager.removeItem(atPath: internalPath)

    ModelZoo.isExternalUrlsPreferred = false
    ModelZoo.externalUrls = [externalDirectory]

    XCTAssertEqual(ModelZoo.filePathForModelDownloaded(fileName), externalPath)
  }

  private func calibrationModelCoefficient(_ modelVersion: ModelVersion) -> Double {
    switch modelVersion {
    case .v1:
      return 0.5
    case .v2:
      return 0.5294117647
    case .sdxlBase:
      return 0.5882352941
    case .kandinsky21:
      return 0.7647058824
    case .sdxlRefiner:
      return 0.5588235294
    case .ssd1b:
      return 0.4117647059
    case .sd3:
      return 0.5294117647
    case .pixart:
      return 0.4117647059
    case .auraflow:
      return 1.029411765
    case .flux1:
      return 2.588235294
    case .sd3Large:
      return 1.176470588
    case .svdI2v:
      return 0.88 * 0.8
    case .wurstchenStageC:
      return 1.18
    case .wurstchenStageB:
      return 1.18
    case .hunyuanVideo:
      return 3.529411765 * 0.8
    case .wan21_1_3b:
      return 1.176470588 * 0.8
    case .wan21_14b:
      return 2.823529412 * 0.8
    case .wan22_5b:
      return 1.176470588 * 0.8
    case .hiDreamI1:
      return 2.84465488969
    case .hiDreamO1:
      return 2.84465488969
    case .qwenImage, .qwenImage2_1:
      return 2.84465488969
    case .zImage:
      return 1.176470588
    case .ernieImage:
      return 2.588235294
    case .flux2:
      return 2.588235294 * 2
    case .flux2_9b:
      return 2.588235294
    case .flux2_4b:
      return 1.176470588 * 0.8
    case .cosmos2_5_2b:
      return 1.176470588 * 0.8
    case .ideogram4:
      return 2.588235294
    case .krea2:
      return 2.588235294
    case .ltx2, .ltx2_3:
      return 1.176470588 * 0.8
    case .seedvr2_3b:
      return 1.176470588 * 0.8
    case .seedvr2_7b:
      return 1.176470588
    case .longcatVideoAvatar1_5, .minimaxH3:
      fatalError()
    }
  }

  private func legacyEstimate(
    _ configuration: GenerationConfiguration, version: ModelVersion,
    modifier: SamplerModifier = .none
  ) -> Int {
    let isCfg =
      isCfgEnabled(
        textGuidanceScale: configuration.guidanceScale,
        imageGuidanceScale: configuration.imageGuidanceScale,
        startFrameCfg: configuration.startFrameCfg, version: version, modifier: modifier
      )
    let (cfgChannels, _) = cfgChannelsAndInputChannels(
      channels: 0, conditionShape: nil, isCfgEnabled: isCfg,
      textGuidanceScale: configuration.guidanceScale,
      imageGuidanceScale: configuration.imageGuidanceScale, version: version,
      modifier: modifier)
    let batchSize: Int
    let numFrames: Int
    var modelCoefficient = calibrationModelCoefficient(version)
    var root = Double(Int(configuration.startWidth) * 64 * Int(configuration.startHeight) * 64)
    switch version {
    case .v1, .v2, .kandinsky21, .sdxlBase, .sdxlRefiner, .ssd1b, .wurstchenStageC,
      .wurstchenStageB, .sd3, .pixart, .auraflow, .sd3Large, .flux1, .qwenImage, .qwenImage2_1,
      .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .cosmos2_5_2b, .hiDreamI1, .hiDreamO1,
      .seedvr2_3b, .seedvr2_7b, .ideogram4, .krea2:
      batchSize = max(1, Int(configuration.batchSize)) * cfgChannels
      numFrames = 1
    case .svdI2v:
      batchSize = cfgChannels
      numFrames = Int(configuration.numFrames)
    case .hunyuanVideo:
      batchSize = cfgChannels
      numFrames = (Int(configuration.numFrames) - 1) / 4 + 1
    case .wan21_1_3b, .wan21_14b, .wan22_5b:
      batchSize = cfgChannels
      numFrames = (Int(configuration.numFrames) - 1) / 4 + 1
    case .ltx2, .ltx2_3:
      batchSize = cfgChannels
      numFrames = (Int(configuration.numFrames) - 1) / 8 + 1
      if configuration.causalInferenceEnabled && configuration.causalInference > 0
        && configuration.causalInference + max(0, configuration.causalInferencePad) < numFrames
      {
        let sequenceLength = root * Double(numFrames)
        let lowerTriangle = sequenceLength * sequenceLength * 0.5
        let upperRidgeLength = root * Double(configuration.causalInference)
        let upperRidgePad = upperRidgeLength * root * Double(configuration.causalInferencePad)
        let upperRidge = upperRidgeLength * upperRidgeLength * 0.5 + upperRidgePad
        let totalArea =
          lowerTriangle + upperRidge * Double(configuration.causalInference) / Double(numFrames)
        modelCoefficient = modelCoefficient * (totalArea / (sequenceLength * sequenceLength))
      }
    case .longcatVideoAvatar1_5, .minimaxH3:
      fatalError()
    }
    root = root * Double(numFrames)
    let scalingFactor: Double = 0.00000922917
    return Int(
      (modelCoefficient * pow(root * scalingFactor, 1.9) * Double(configuration.steps)
        * Double(max(configuration.strength, 0.05)) * Double(batchSize)).rounded(.up))
  }

  private static let instructionCalibrationScale: Double = 5.14816e-12

  private func cfgChannels(
    _ configuration: GenerationConfiguration, version: ModelVersion,
    modifier: SamplerModifier = .none
  ) -> Int {
    let isCfg =
      isCfgEnabled(
        textGuidanceScale: configuration.guidanceScale,
        imageGuidanceScale: configuration.imageGuidanceScale,
        startFrameCfg: configuration.startFrameCfg, version: version, modifier: modifier
      )
    let (cfgChannels, _) = cfgChannelsAndInputChannels(
      channels: 0, conditionShape: nil, isCfgEnabled: isCfg,
      textGuidanceScale: configuration.guidanceScale,
      imageGuidanceScale: configuration.imageGuidanceScale, version: version, modifier: modifier)
    return cfgChannels
  }

  private func computeUnitsTokenLength(modelName: String, version: ModelVersion) -> Int {
    let paddedLength = ModelZoo.paddedTextEncodingLengthForModel(modelName)
    if paddedLength > 0 {
      return paddedLength
    }
    switch version {
    case .flux1:
      return 512
    case .hunyuanVideo:
      return 256
    case .qwenImage, .qwenImage2_1, .zImage, .ernieImage, .flux2, .flux2_9b, .flux2_4b, .ideogram4:
      return 512
    case .krea2:
      return 256
    case .ltx2, .ltx2_3:
      return 128
    case .hiDreamI1:
      return 128
    default:
      return 77
    }
  }

  private func modelStartSizeFromConfiguration(
    _ configuration: GenerationConfiguration, version: ModelVersion
  ) -> (width: Int, height: Int) {
    let rawWidth = max(1, Int(configuration.startWidth))
    let rawHeight = max(1, Int(configuration.startHeight))
    switch version {
    case .wurstchenStageC:
      return (
        Int((Double(rawWidth) * 3.0 / 2.0).rounded(.up)),
        Int((Double(rawHeight) * 3.0 / 2.0).rounded(.up))
      )
    case .wan22_5b:
      return (rawWidth * 4, rawHeight * 4)
    case .hiDreamO1, .ltx2, .ltx2_3:
      return (rawWidth * 2, rawHeight * 2)
    default:
      return (rawWidth * 8, rawHeight * 8)
    }
  }

  private func rawInstructionCountCurrentPath(
    _ configuration: GenerationConfiguration, modelName: String, version: ModelVersion
  ) -> Int {
    let modelStartSize = modelStartSizeFromConfiguration(configuration, version: version)
    let startWidth = (modelStartSize.width + 1) / 2 * 2
    let startHeight = (modelStartSize.height + 1) / 2 * 2
    let cfg = cfgChannels(configuration, version: version)
    let imageBatch = max(1, Int(configuration.batchSize)) * cfg
    let tokenLength = max(1, computeUnitsTokenLength(modelName: modelName, version: version))
    let numFrames = max(1, Int(configuration.numFrames))
    switch version {
    case .flux1:
      let count = Flux1InstructionCount(
        batchSize: 1, tokenLength: tokenLength, referenceSequenceLength: 0,
        height: startHeight, width: startWidth, channels: 3072, layers: (19, 38),
        contextPreloaded: true)
      return count * imageBatch
    case .flux2_9b:
      let count = Flux2InstructionCount(
        batchSize: 1, tokenLength: tokenLength, referenceSequenceLength: 0,
        height: startHeight, width: startWidth, channels: 4096, layers: (8, 24))
      return count * imageBatch
    case .ltx2:
      let ltx2VideoFrames = max(1, (numFrames - 1) / 8 + 1)
      let audioFrames = (ltx2VideoFrames - 1) * 8 + 1
      let paddedTokenLength = max(tokenLength, 1024)
      let count = LTX2InstructionCount(
        time: ltx2VideoFrames, h: startHeight, w: startWidth, textLength: paddedTokenLength,
        audioFrames: audioFrames, channels: (4096, 2048), layers: 48, tokenModulation: false,
        KV: true, useGatedAttention: false, textCrossAttentionAdaLN: false)
      return count * imageBatch
    case .ltx2_3:
      let ltx2VideoFrames = max(1, (numFrames - 1) / 8 + 1)
      let audioFrames = (ltx2VideoFrames - 1) * 8 + 1
      let paddedTokenLength = max(tokenLength, 1024)
      let count = LTX2InstructionCount(
        time: ltx2VideoFrames, h: startHeight, w: startWidth, textLength: paddedTokenLength,
        audioFrames: audioFrames, channels: (4096, 2048), layers: 48, tokenModulation: false,
        KV: false, useGatedAttention: true, textCrossAttentionAdaLN: true)
      return count * imageBatch
    default:
      fatalError("unsupported version")
    }
  }

  private func rawInstructionCountLTX2OldPath(
    _ configuration: GenerationConfiguration, modelName: String
  ) -> Int {
    let rawStartWidth = max(1, Int(configuration.startWidth))
    let rawStartHeight = max(1, Int(configuration.startHeight))
    let startWidth = (rawStartWidth + 1) / 2 * 2
    let startHeight = (rawStartHeight + 1) / 2 * 2
    let cfg = cfgChannels(configuration, version: .ltx2)
    let imageBatch = max(1, Int(configuration.batchSize)) * cfg
    let tokenLength = max(
      1, max(computeUnitsTokenLength(modelName: modelName, version: .ltx2), 1024))
    let numFrames = max(1, Int(configuration.numFrames))
    let chunkedVideoFrames = max(1, (numFrames - 1) / 4 + 1)
    let audioFrames = (chunkedVideoFrames - 1) * 8 + 1
    let count = LTX2InstructionCount(
      time: chunkedVideoFrames, h: startHeight, w: startWidth, textLength: tokenLength,
      audioFrames: audioFrames, channels: (4096, 2048), layers: 48, tokenModulation: false,
      KV: true, useGatedAttention: false, textCrossAttentionAdaLN: false)
    return count * imageBatch
  }

  private func instructionEstimateFromRaw(
    _ rawCount: Int, configuration: GenerationConfiguration
  ) -> Int {
    let estimate =
      Double(rawCount) * Self.instructionCalibrationScale
      * Double(configuration.steps) * Double(max(configuration.strength, 0.05))
    return Int(estimate.rounded(.up))
  }

  func testInstructionEstimateIsInLegacyOrderOfMagnitude() throws {
    for testCase in Self.testCases {
      let modelName = "test-\(testCase.version.rawValue)"
      let configuration = configuration(
        modelName: modelName, width: testCase.width, height: testCase.height,
        numFrames: testCase.frames)
      let overrideMapping = mapping(modelName: modelName, version: testCase.version)
      let legacy = legacyEstimate(configuration, version: testCase.version)
      let estimate = try XCTUnwrap(
        ComputeUnits.from(
          configuration, hasImage: false, shuffleCount: 0, overrideMapping: overrideMapping))
      XCTAssertGreaterThan(
        legacy, 0, "Baseline estimate should be positive for \(testCase.version)")
      XCTAssertGreaterThan(estimate, 0, "Estimate should be positive for \(testCase.version)")
      let ratio = Double(estimate) / Double(legacy)
      XCTAssertGreaterThanOrEqual(
        ratio, 0.01, "Estimate ratio too small for \(testCase.version): \(ratio)")
      XCTAssertLessThanOrEqual(
        ratio, 10.0, "Estimate ratio too large for \(testCase.version): \(ratio)")
    }
  }

  func testInstructionEstimateHandlesAllVersions() throws {
    for testCase in Self.testCases {
      let modelName = "test-\(testCase.version.rawValue)-close"
      let configuration = configuration(
        modelName: modelName, width: testCase.width, height: testCase.height,
        numFrames: testCase.frames)
      let overrideMapping = mapping(modelName: modelName, version: testCase.version)
      let estimate = try XCTUnwrap(
        ComputeUnits.from(
          configuration, hasImage: false, shuffleCount: 0, overrideMapping: overrideMapping))
      XCTAssertGreaterThan(estimate, 0)
    }
  }

  func testMiniMaxH3FrameAlignmentMatchesGeneration() throws {
    let modelName = "test-minimax-h3-frame-alignment"
    let overrideMapping = mapping(modelName: modelName, version: .minimaxH3)
    for frames in 1...400 {
      var alignedFrames = frames
      if frames != 1 {
        alignedFrames = max(frames, 5)
        while alignedFrames % 17 != 5 {
          alignedFrames += 1
        }
      }
      let requested = configuration(
        modelName: modelName, width: 12, height: 12, numFrames: UInt32(frames))
      let aligned = configuration(
        modelName: modelName, width: 12, height: 12, numFrames: UInt32(alignedFrames))
      let estimate = try XCTUnwrap(
        ComputeUnits.from(
          requested, hasImage: false, shuffleCount: 0, overrideMapping: overrideMapping))
      let alignedEstimate = try XCTUnwrap(
        ComputeUnits.from(
          aligned, hasImage: false, shuffleCount: 0, overrideMapping: overrideMapping))
      XCTAssertEqual(estimate, alignedEstimate, "Frame count: \(frames)")
    }
  }

  func testMiniMaxH3MainInstructionCountExcludesRefiner() {
    let videoRows = 24 * 24
    let audioRows = 4
    let rows = videoRows + audioRows + 45
    let expected =
      (videoRows * 96 + audioRows * 32) * 5_376 * 2
      + 50
      * (4 * rows * 5_376 * 7_168 + 3 * rows * 5_376 * 14_336
        + 56 * (2 * 128 + 5) * rows * rows)
    XCTAssertEqual(
      MiniMaxH3InstructionCount(
        videoTime: 1, videoHeight: 48, videoWidth: 48, audioLength: audioRows, textLength: 45),
      expected)
    let referenceRows = 2 * 24 * 24
    let conditionedRows = rows + referenceRows
    XCTAssertEqual(
      MiniMaxH3InstructionCount(
        videoTime: 1, videoHeight: 48, videoWidth: 48, audioLength: audioRows, textLength: 45,
        referenceSequenceLength: referenceRows),
      expected + 50
        * (4 * referenceRows * 5_376 * 7_168 + 3 * referenceRows * 5_376 * 14_336
          + 56 * (2 * 128 + 5) * (conditionedRows * conditionedRows - rows * rows)))
  }

  func testMiniMaxH3FixedInstructionCountIncludesRefinerOnce() {
    for timesteps in [1, 21, 50] {
      for layers in [1, 50] {
        let textLength = 45
        let refiner =
          textLength * 5_120 * 5_376
          + 2
          * (4 * textLength * 5_376 * 7_168 + 3 * textLength * 5_376 * 14_336
            + 56 * (2 * 128 + 5) * textLength * textLength)
        let modulation = 2 * timesteps * (256 * 5_376 + 5_376 * 2_688 * (1 + 18 * layers + 2))
        XCTAssertEqual(
          MiniMaxH3FixedInstructionCount(
            timesteps: timesteps, hiddenSize: 5_376, layers: layers, textLength: (0, textLength)),
          modulation + refiner)
        XCTAssertEqual(
          MiniMaxH3FixedInstructionCount(
            timesteps: timesteps, hiddenSize: 5_376, layers: layers,
            textLength: (textLength, textLength)),
          modulation + 2 * refiner)
        XCTAssertEqual(
          MiniMaxH3FixedInstructionCount(
            timesteps: timesteps, hiddenSize: 5_376, layers: layers, textLength: (0, textLength),
            referenceSequenceLength: 2 * 24 * 24),
          modulation / 2 * 3 + refiner + 2 * 24 * 24 * 96 * 5_376)
      }
    }
  }

  func testMiniMaxH3EstimateSharesFixedWorkAcrossCFG() throws {
    let modelName = "test-minimax-h3-fixed-cfg"
    let overrideMapping = mapping(modelName: modelName, version: .minimaxH3)
    let mainCount = MiniMaxH3InstructionCount(
      videoTime: 1, videoHeight: 48, videoWidth: 48, audioLength: 4, textLength: 512)
    for steps: UInt32 in [1, 21, 50] {
      for guidanceScale: Float in [1, 2] {
        let fixedCount = MiniMaxH3FixedInstructionCount(
          timesteps: Int(steps), hiddenSize: 5_376, layers: 50,
          textLength: (guidanceScale == 1 ? 0 : 512, 512))
        let cfg = GenerationConfiguration(
          id: 1, startWidth: 12, startHeight: 12, steps: steps, guidanceScale: guidanceScale,
          strength: 1, model: modelName, batchSize: 4, numFrames: 1)
        let estimate = try XCTUnwrap(
          ComputeUnits.from(
            cfg, hasImage: false, shuffleCount: 0, overrideMapping: overrideMapping))
        let mainEstimate =
          Double(mainCount * (guidanceScale == 1 ? 1 : 2)) * Self.instructionCalibrationScale
          * Double(steps)
        let fixedEstimate = Double(fixedCount) * Self.instructionCalibrationScale
        XCTAssertEqual(estimate, Int((mainEstimate + fixedEstimate).rounded(.up)))
      }
    }
  }

  func testQwenImage2_1PrefixWorkIsSharedAcrossImageBatch() throws {
    let modelName = "test-qwen-image-2.1-fixed"
    let overrideMapping = mapping(modelName: modelName, version: .qwenImage2_1)
    for steps: UInt32 in [1, 40] {
      let config = configuration(modelName: modelName, width: 8, height: 8, steps: steps)
      let single = try XCTUnwrap(
        ComputeUnits.from(
          config, hasImage: true, shuffleCount: 0,
          overrideMapping: overrideMapping))
      let batchConfig = configuration(
        modelName: modelName, width: 8, height: 8, steps: steps, batchSize: 2)
      let batch = try XCTUnwrap(
        ComputeUnits.from(
          batchConfig, hasImage: true, shuffleCount: 0,
          overrideMapping: overrideMapping))
      XCTAssertGreaterThan(batch, single)
      XCTAssertLessThan(batch, 2 * single)
    }
  }

  func testFlux2KVCacheInstructionCountMovesReferenceWorkToFixedPath() throws {
    let legacyMain = Flux2InstructionCount(
      batchSize: 1, tokenLength: 512, referenceSequenceLength: 1024, height: 64, width: 64,
      channels: 4096, layers: (8, 24))
    let kvMain = Flux2InstructionCount(
      batchSize: 1, tokenLength: 512, referenceSequenceLength: 0,
      cachedReferenceSequenceLength: 1024, height: 64, width: 64, channels: 4096,
      layers: (8, 24))
    XCTAssertLessThan(kvMain, legacyMain)

    let legacyFixed = Flux2FixedInstructionCount(
      contextBatchSize: 1, tokenLength: 512, timestepCount: 1, referenceSequenceLength: 1024,
      channels: 4096, layers: (8, 24), guidanceEmbed: false)
    let kvFixed = Flux2FixedInstructionCount(
      contextBatchSize: 1, tokenLength: 512, timestepCount: 2, referenceSequenceLength: 1024,
      channels: 4096, layers: (8, 24), guidanceEmbed: false, kvCache: true)
    XCTAssertGreaterThan(kvFixed, legacyFixed)
  }

  func testDiagnosticsFlux2_9BVsFlux1AndLTX2Shape() throws {
    let sizes: [(pixel: Int, latent: UInt16)] = [(768, 12), (1024, 16), (1280, 20)]

    print("==== Flux1 vs Flux2_9B (current ComputeUnits path) ====")
    for size in sizes {
      let flux1Model = "diag-flux1-missing-\(size.pixel)"
      let flux2Model = "diag-flux2_9b-missing-\(size.pixel)"
      let flux1Cfg = configuration(modelName: flux1Model, width: size.latent, height: size.latent)
      let flux2Cfg = configuration(modelName: flux2Model, width: size.latent, height: size.latent)
      let flux1Legacy = legacyEstimate(flux1Cfg, version: .flux1)
      let flux2Legacy = legacyEstimate(flux2Cfg, version: .flux2_9b)
      let flux1Raw = rawInstructionCountCurrentPath(
        flux1Cfg, modelName: flux1Model, version: .flux1)
      let flux2Raw = rawInstructionCountCurrentPath(
        flux2Cfg, modelName: flux2Model, version: .flux2_9b)
      let flux1Estimate = instructionEstimateFromRaw(
        flux1Raw, configuration: flux1Cfg)
      let flux2Estimate = instructionEstimateFromRaw(
        flux2Raw, configuration: flux2Cfg)
      print(
        "\(size.pixel)x\(size.pixel) legacy flux1=\(flux1Legacy) flux2_9b=\(flux2Legacy) | estimate flux1=\(flux1Estimate) flux2_9b=\(flux2Estimate)"
      )
    }

    print("==== LTX2 Old vs Current Latent Shape Path ====")
    for size in sizes {
      let modelName = "diag-ltx2-\(size.pixel)"
      let cfg = configuration(
        modelName: modelName, width: size.latent, height: size.latent, numFrames: 49)
      let legacy = legacyEstimate(cfg, version: .ltx2)
      let rawCurrent = rawInstructionCountCurrentPath(cfg, modelName: modelName, version: .ltx2)
      let rawOld = rawInstructionCountLTX2OldPath(cfg, modelName: modelName)
      let estimateCurrent = instructionEstimateFromRaw(
        rawCurrent, configuration: cfg)
      let estimateOld = instructionEstimateFromRaw(
        rawOld, configuration: cfg)
      print(
        "\(size.pixel)x\(size.pixel) legacy=\(legacy) | old estimate=\(estimateOld) | current estimate=\(estimateCurrent)"
      )
    }
  }

  func testDiagnosticsAllModelsAcrossCommonResolutions() throws {
    let resolutions: [(pixel: Int, latent: UInt16)] = [(768, 12), (1024, 16), (1280, 20)]
    let versions = Self.testCases.map(\.version)
    let videoVersions: Set<ModelVersion> = [
      .svdI2v, .hunyuanVideo, .wan21_1_3b, .wan21_14b, .wan22_5b, .ltx2, .ltx2_3,
    ]

    print("==== Full Model Comparison (legacy vs instruction-estimate) ====")
    print("instructionCalibrationScale=\(Self.instructionCalibrationScale)")
    print("model,frames,resolution,legacy,estimate,ratio")
    for version in versions {
      for resolution in resolutions {
        let frames: UInt32 = videoVersions.contains(version) ? 121 : 1
        let modelName = "diag-all-\(version.rawValue)-\(resolution.pixel)"
        let cfg = configuration(
          modelName: modelName, width: resolution.latent, height: resolution.latent,
          numFrames: frames)
        let overrideMapping = mapping(modelName: modelName, version: version)
        let legacy = legacyEstimate(cfg, version: version)
        let estimate = try XCTUnwrap(
          ComputeUnits.from(
            cfg, hasImage: false, shuffleCount: 0, overrideMapping: overrideMapping))
        let ratio = Double(estimate) / Double(legacy)
        print(
          "\(version.rawValue),\(frames),\(resolution.pixel)x\(resolution.pixel),\(legacy),\(estimate),\(ratio)"
        )
      }
    }
  }
}
