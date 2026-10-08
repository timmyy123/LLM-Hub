import Darwin
import Diffusion
import Foundation
import ImageGenerator
import Logging
import ModelZoo
import NNC
import UniformTypeIdentifiers
import XCTest

@testable import MediaGenerationKit

final class MediaGenerationKitTests: XCTestCase {
  func testMiniMaxH3TokenizerDoesNotChangeQwen3() {
    let resources = MediaGenerationExecutionUtilities.createLocalResources()
    defer { try? FileManager.default.removeItem(atPath: resources.tempDir) }
    let qwen3 = resources.generator.tokenizerQwen3
    let h3 = resources.generator.tokenizerQwen3MinimaxH3
    let h3Tokens: [String: Int32] = [
      "<d>": 151_669, "</d>": 151_670, "<|cutoff|>": 151_671,
      "<|lyrics_start|>": 151_672, "<|lyrics_end|>": 151_673,
      "<|caption_start|>": 151_674, "<|caption_end|>": 151_675,
    ]
    for (token, id) in h3Tokens {
      XCTAssertEqual(h3.tokenize(text: token).0, [id], token)
      XCTAssertEqual(h3.decode([id]), token)
      XCTAssertGreaterThan(qwen3.tokenize(text: token).0.count, 1, token)
    }
    XCTAssertEqual(qwen3.tokenize(text: "<d>").0, [90_707, 29])
    XCTAssertEqual(qwen3.tokenize(text: "</d>").0, [522, 67, 29])
    let imageTokens: [String: Int32] = [
      "<|boi_token|>": 151_669, "<|bor_token|>": 151_670,
      "<|eor_token|>": 151_671, "<|bot_token|>": 151_672, "<|tms_token|>": 151_673,
    ]
    for (token, id) in imageTokens {
      XCTAssertEqual(qwen3.tokenize(text: token).0, [id], token)
      XCTAssertEqual(qwen3.decode([id]), token)
      XCTAssertGreaterThan(h3.tokenize(text: token).0.count, 1, token)
    }
    let text = "[English] Hello there."
    XCTAssertEqual(
      h3.tokenize(text: "<d>\(text)</d>").0,
      [151_669] + qwen3.tokenize(text: text).0 + [151_670])
    XCTAssertEqual(
      h3.tokenize(text: "<d></d><d></d>").0, [151_669, 151_670, 151_669, 151_670])
    for prompt in [
      "", " ", "A dog in a meadow.", "你好，世界！",
      "<Picture 1>: <|vision_start|><|image_pad|><|vision_end|>Hello",
    ] {
      XCTAssertEqual(h3.tokenize(text: prompt).0, qwen3.tokenize(text: prompt).0)
    }
  }

  func testPipelineCopyDoesNotShareConfiguration() async throws {
    let modelsDirectory = try makeTemporaryDirectory()
    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      "flux_2_klein_4b_q8p.ckpt",
      backend: .local(directory: modelsDirectory.path)
    )
    pipeline.logger = Logger(label: "com.draw-things.tests.pipeline")

    var copy = pipeline
    copy.configuration.width = 768
    copy.configuration.height = 768
    copy.configuration.steps = 30

    XCTAssertEqual(pipeline.configuration.model, copy.configuration.model)
    XCTAssertNotEqual(pipeline.configuration.width, copy.configuration.width)
    XCTAssertNotEqual(pipeline.configuration.height, copy.configuration.height)
    XCTAssertNotEqual(pipeline.configuration.steps, copy.configuration.steps)
  }

  func testPipelinePublishesResolvedModelFile() async throws {
    let modelsDirectory = try makeTemporaryDirectory()
    let pipeline = try await MediaGenerationPipeline.fromPretrained(
      "FLUX.2 [klein] 4B",
      backend: .local(directory: modelsDirectory.path)
    )

    XCTAssertEqual(pipeline.model, "flux_2_klein_4b_q8p.ckpt")
    XCTAssertEqual(pipeline.configuration.model, "flux_2_klein_4b_q8p.ckpt")
  }

  func testAsyncFromPretrainedPublishesResolvedModelFile() async throws {
    let modelsDirectory = try makeTemporaryDirectory()
    let pipeline = try await MediaGenerationPipeline.fromPretrained(
      "FLUX.2 [klein] 4B",
      backend: .local(directory: modelsDirectory.path)
    )

    XCTAssertEqual(pipeline.model, "flux_2_klein_4b_q8p.ckpt")
    XCTAssertEqual(pipeline.configuration.model, "flux_2_klein_4b_q8p.ckpt")
  }

  func testEnvironmentResolvesKnownModelReference() throws {
    let resolved = try MediaGenerationEnvironment.default.resolveModel("flux_2_klein_4b_q8p.ckpt")

    XCTAssertEqual(resolved?.file, "flux_2_klein_4b_q8p.ckpt")
    XCTAssertEqual(resolved?.name, "FLUX.2 [klein] 4B")
  }

  func testEnvironmentSuggestsClosestModels() throws {
    let suggestions = try MediaGenerationEnvironment.default.suggestedModels(
      for: "flux_2_klein_4b"
    )

    XCTAssertFalse(suggestions.isEmpty)
    XCTAssertTrue(suggestions.contains { $0.file == "flux_2_klein_4b_q8p.ckpt" })
  }

  func testEnvironmentCanInspectKnownModel() throws {
    let inspection = try MediaGenerationEnvironment.default.inspectModel("flux_2_klein_4b_q8p.ckpt")

    XCTAssertEqual(inspection.file, "flux_2_klein_4b_q8p.ckpt")
    XCTAssertEqual(inspection.name, "FLUX.2 [klein] 4B")
    XCTAssertNotNil(inspection.version)
    XCTAssertEqual(inspection.version, "FLUX.2 4B")
  }

  func testEnvironmentListsDownloadableModels() throws {
    let models = try MediaGenerationEnvironment.default.downloadableModels()

    XCTAssertFalse(models.isEmpty)
    XCTAssertTrue(models.contains { $0.file == "flux_2_klein_4b_q8p.ckpt" })
  }

  func testEnvironmentResolveModelSyncThrowsOnlyWhenRemoteFetchWouldBeNeeded() throws {
    XCTAssertNoThrow(
      try MediaGenerationEnvironment.default.resolveModel(
        "flux_2_klein_4b_q8p.ckpt", offline: false)
    )

    XCTAssertThrowsError(
      try MediaGenerationEnvironment.default.resolveModel(
        "model-that-requires-remote-catalog",
        offline: false
      )
    ) { error in
      guard case MediaGenerationKitError.asyncOperationRequired(let operation) = error else {
        return XCTFail("unexpected error: \(error)")
      }
      XCTAssertEqual(operation, "resolveModel")
    }
  }

  func testEnvironmentSuggestedModelsSyncThrowsWhenRemoteCatalogIsNotCached() {
    XCTAssertThrowsError(
      try MediaGenerationEnvironment.default.suggestedModels(
        for: "flux_2_klein_4b",
        offline: false
      )
    ) { error in
      guard case MediaGenerationKitError.asyncOperationRequired(let operation) = error else {
        return XCTFail("unexpected error: \(error)")
      }
      XCTAssertEqual(operation, "suggestedModels")
    }
  }

  func testEnvironmentInspectModelSyncThrowsOnlyWhenRemoteFetchWouldBeNeeded() throws {
    XCTAssertNoThrow(
      try MediaGenerationEnvironment.default.inspectModel(
        "flux_2_klein_4b_q8p.ckpt", offline: false)
    )

    XCTAssertThrowsError(
      try MediaGenerationEnvironment.default.inspectModel(
        "model-that-requires-remote-catalog",
        offline: false
      )
    ) { error in
      guard case MediaGenerationKitError.asyncOperationRequired(let operation) = error else {
        return XCTFail("unexpected error: \(error)")
      }
      XCTAssertEqual(operation, "inspectModel")
    }
  }

  func testEnvironmentDownloadableModelsSyncThrowsWhenRemoteCatalogIsNotCached() {
    XCTAssertThrowsError(
      try MediaGenerationEnvironment.default.downloadableModels(offline: false)
    ) { error in
      guard case MediaGenerationKitError.asyncOperationRequired(let operation) = error else {
        return XCTFail("unexpected error: \(error)")
      }
      XCTAssertEqual(operation, "downloadableModels")
    }
  }

  func testEnvironmentMaxTotalWeightsCacheSizeRoundTrips() {
    let original = MediaGenerationEnvironment.default.maxTotalWeightsCacheSize
    addTeardownBlock {
      MediaGenerationEnvironment.default.maxTotalWeightsCacheSize = original
    }

    let expected = UInt64(6) * 1_024 * 1_024 * 1_024
    MediaGenerationEnvironment.default.maxTotalWeightsCacheSize = expected

    XCTAssertEqual(MediaGenerationEnvironment.default.maxTotalWeightsCacheSize, expected)
  }

  func testEnvironmentExternalURLsCanBeCleared() throws {
    let modelsDirectory = try makeTemporaryDirectory()
    let originalExternalURLs = MediaGenerationEnvironment.default.externalUrls
    let originalModelZooExternalURLs = ModelZoo.externalUrls
    addTeardownBlock {
      MediaGenerationEnvironment.default.externalUrls = originalExternalURLs
      ModelZoo.externalUrls = originalModelZooExternalURLs
    }

    MediaGenerationEnvironment.default.externalUrls = [modelsDirectory]
    XCTAssertEqual(ModelZoo.externalUrls, [modelsDirectory])

    MediaGenerationEnvironment.default.externalUrls = []
    XCTAssertEqual(ModelZoo.externalUrls, [])
  }

  func testLocalBackendPrefersEnvironmentModelsDirectory() async throws {
    let invalidFile = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mediagenerationkit-env-\(UUID().uuidString).txt"
    )
    try Data("invalid".utf8).write(to: invalidFile)
    addTeardownBlock {
      try? FileManager.default.removeItem(at: invalidFile)
    }

    let previous = ProcessInfo.processInfo.environment["DRAWTHINGS_MODELS_DIR"]
    setenv("DRAWTHINGS_MODELS_DIR", invalidFile.path, 1)
    addTeardownBlock {
      if let previous {
        setenv("DRAWTHINGS_MODELS_DIR", previous, 1)
      } else {
        unsetenv("DRAWTHINGS_MODELS_DIR")
      }
    }

    MediaGenerationEnvironment.default = MediaGenerationEnvironment(
      storage: MediaGenerationEnvironment.Storage(
        externalUrls: MediaGenerationDefaults.defaultExternalURLs()
      )
    )

    do {
      _ = try await MediaGenerationPipeline.fromPretrained(
        "flux_2_klein_4b_q8p.ckpt",
        backend: .local
      )
      XCTFail("expected invalid models directory error")
    } catch {
      guard case MediaGenerationKitError.invalidModelsDirectory = error else {
        return XCTFail("unexpected error: \(error)")
      }
    }
  }

  func testExecutionInputsMapSemanticRoles() throws {
    let data = Data(base64Encoded: Self.onePixelPNGBase64)!
    let executionInputs = try MediaGenerationPipeline.executionInputs(from: [
      MediaGenerationPipeline.data(data),
      MediaGenerationPipeline.data(data).mask(),
      MediaGenerationPipeline.data(data).moodboard(),
      MediaGenerationPipeline.data(data).depth(),
    ])

    XCTAssertNotNil(executionInputs.image)
    XCTAssertNotNil(executionInputs.mask)
    XCTAssertEqual(executionInputs.hints.count, 2)
    XCTAssertEqual(executionInputs.hints[0].type, .shuffle)
    XCTAssertEqual(executionInputs.hints[0].images.count, 1)
    XCTAssertEqual(executionInputs.hints[1].type, .depth)
    XCTAssertEqual(executionInputs.hints[1].images.count, 1)
  }

  func testExecutionInputsRejectDuplicatePrimaryImages() {
    let data = Data(base64Encoded: Self.onePixelPNGBase64)!
    XCTAssertThrowsError(
      try MediaGenerationPipeline.executionInputs(from: [
        MediaGenerationPipeline.data(data),
        MediaGenerationPipeline.data(data),
      ])
    )
  }

  func testPreviewCollectionExposesImageBatchCount() {
    let previewTensor = Tensor<FloatType>(
      [
        0, 1, 2,
        3, 4, 5,
      ],
      .CPU,
      .NHWC(2, 1, 1, 3)
    )
    let preview = MediaGenerationPipeline.Preview(tensor: previewTensor, version: .flux2)

    XCTAssertEqual(preview.count, 2)
    XCTAssertEqual(preview.startIndex, 0)
    XCTAssertEqual(preview.endIndex, 2)
  }

  func testConfigurationValidationRejectsInvalidDimensions() async throws {
    let modelsDirectory = try makeTemporaryDirectory()
    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      "flux_2_klein_4b_q8p.ckpt",
      backend: .local(directory: modelsDirectory.path)
    )
    pipeline.configuration.width = 513

    XCTAssertThrowsError(
      try pipeline.configuration.runtimeConfiguration(template: .default)
    ) { error in
      guard case MediaGenerationKitError.generationFailed(let message) = error else {
        return XCTFail("unexpected error: \(error)")
      }
      XCTAssertTrue(message.contains("multiples of 64"))
    }
  }

  func testConfigurationRoundTripsExtendedJSConfigurationFields() async throws {
    let modelsDirectory = try makeTemporaryDirectory()
    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      "flux_2_klein_4b_q8p.ckpt",
      backend: .local(directory: modelsDirectory.path)
    )
    pipeline.configuration.seedMode = .scaleAlike
    pipeline.configuration.clipSkip = 2
    pipeline.configuration.hiresFix = true
    pipeline.configuration.hiresFixWidth = 1024
    pipeline.configuration.hiresFixHeight = 768
    pipeline.configuration.hiresFixStrength = 0.35
    pipeline.configuration.tiledDecoding = true
    pipeline.configuration.decodingTileWidth = 128
    pipeline.configuration.decodingTileHeight = 192
    pipeline.configuration.decodingTileOverlap = 64
    pipeline.configuration.tiledDiffusion = true
    pipeline.configuration.diffusionTileWidth = 192
    pipeline.configuration.diffusionTileHeight = 128
    pipeline.configuration.diffusionTileOverlap = 64
    pipeline.configuration.upscaler = "4x_ultrasharp_f16.ckpt"
    pipeline.configuration.upscalerScaleFactor = 4
    pipeline.configuration.imageGuidanceScale = 1.5
    pipeline.configuration.maskBlur = 4
    pipeline.configuration.maskBlurOutset = 12
    pipeline.configuration.sharpness = 0.2
    pipeline.configuration.faceRestoration = "gfpgan_1.4_f16.ckpt"
    pipeline.configuration.clipWeight = 0.8
    pipeline.configuration.negativePromptForImagePrior = true
    pipeline.configuration.imagePriorSteps = 9
    pipeline.configuration.refinerModel = "sdxl_refiner_1.0_f16.ckpt"
    pipeline.configuration.originalImageHeight = 1024
    pipeline.configuration.originalImageWidth = 768
    pipeline.configuration.cropTop = 16
    pipeline.configuration.cropLeft = 8
    pipeline.configuration.targetImageHeight = 512
    pipeline.configuration.targetImageWidth = 640
    pipeline.configuration.aestheticScore = 7
    pipeline.configuration.negativeAestheticScore = 1.5
    pipeline.configuration.zeroNegativePrompt = true
    pipeline.configuration.refinerStart = 0.8
    pipeline.configuration.negativeOriginalImageHeight = 1536
    pipeline.configuration.negativeOriginalImageWidth = 1024
    pipeline.configuration.numFrames = 13
    pipeline.configuration.fps = 12
    pipeline.configuration.motionScale = 127
    pipeline.configuration.guidingFrameNoise = 0.15
    pipeline.configuration.startFrameGuidance = 1.1
    pipeline.configuration.shift = 1.2
    pipeline.configuration.stage2Steps = 11
    pipeline.configuration.stage2Guidance = 3.5
    pipeline.configuration.stage2Shift = 0.7
    pipeline.configuration.stochasticSamplingGamma = 0.05
    pipeline.configuration.preserveOriginalAfterInpaint = true
    pipeline.configuration.t5TextEncoder = true
    pipeline.configuration.separateClipL = true
    pipeline.configuration.clipLText = "clip-l"
    pipeline.configuration.separateOpenClipG = true
    pipeline.configuration.openClipGText = "open-clip-g"
    pipeline.configuration.speedUpWithGuidanceEmbed = true
    pipeline.configuration.guidanceEmbed = 2.25
    pipeline.configuration.resolutionDependentShift = true
    pipeline.configuration.teaCache = true
    pipeline.configuration.teaCacheStart = 2
    pipeline.configuration.teaCacheEnd = 8
    pipeline.configuration.teaCacheThreshold = 0.11
    pipeline.configuration.teaCacheMaxSkipSteps = 3
    pipeline.configuration.separateT5 = true
    pipeline.configuration.t5Text = "t5"
    pipeline.configuration.causalInference = 5
    pipeline.configuration.causalInferencePad = 2
    pipeline.configuration.cfgZeroStar = true
    pipeline.configuration.cfgZeroInitSteps = 4
    pipeline.configuration.compressionArtifacts = .H264
    pipeline.configuration.compressionArtifactsQuality = 55
    pipeline.configuration.colorCalibration = .lab
    pipeline.configuration.expandPromptToJson = true

    let runtime = try pipeline.configuration.runtimeConfiguration(template: .default)

    XCTAssertEqual(runtime.seedMode, .scaleAlike)
    XCTAssertEqual(runtime.clipSkip, 2)
    XCTAssertTrue(runtime.hiresFix)
    XCTAssertEqual(runtime.hiresFixStartWidth, 16)
    XCTAssertEqual(runtime.hiresFixStartHeight, 12)
    XCTAssertEqual(runtime.hiresFixStrength, 0.35, accuracy: 0.0001)
    XCTAssertTrue(runtime.tiledDecoding)
    XCTAssertEqual(runtime.decodingTileWidth, 2)
    XCTAssertEqual(runtime.decodingTileHeight, 3)
    XCTAssertEqual(runtime.decodingTileOverlap, 1)
    XCTAssertTrue(runtime.tiledDiffusion)
    XCTAssertEqual(runtime.diffusionTileWidth, 3)
    XCTAssertEqual(runtime.diffusionTileHeight, 2)
    XCTAssertEqual(runtime.diffusionTileOverlap, 1)
    XCTAssertEqual(runtime.upscaler, "4x_ultrasharp_f16.ckpt")
    XCTAssertEqual(runtime.upscalerScaleFactor, 4)
    XCTAssertEqual(runtime.imageGuidanceScale, 1.5, accuracy: 0.0001)
    XCTAssertEqual(runtime.maskBlur, 4, accuracy: 0.0001)
    XCTAssertEqual(runtime.maskBlurOutset, 12)
    XCTAssertEqual(runtime.sharpness, 0.2, accuracy: 0.0001)
    XCTAssertEqual(runtime.faceRestoration, "gfpgan_1.4_f16.ckpt")
    XCTAssertEqual(runtime.clipWeight, 0.8, accuracy: 0.0001)
    XCTAssertTrue(runtime.negativePromptForImagePrior)
    XCTAssertEqual(runtime.imagePriorSteps, 9)
    XCTAssertEqual(runtime.refinerModel, "sdxl_refiner_1.0_f16.ckpt")
    XCTAssertEqual(runtime.originalImageHeight, 1024)
    XCTAssertEqual(runtime.originalImageWidth, 768)
    XCTAssertEqual(runtime.cropTop, 16)
    XCTAssertEqual(runtime.cropLeft, 8)
    XCTAssertEqual(runtime.targetImageHeight, 512)
    XCTAssertEqual(runtime.targetImageWidth, 640)
    XCTAssertEqual(runtime.aestheticScore, 7, accuracy: 0.0001)
    XCTAssertEqual(runtime.negativeAestheticScore, 1.5, accuracy: 0.0001)
    XCTAssertTrue(runtime.zeroNegativePrompt)
    XCTAssertEqual(runtime.refinerStart, 0.8, accuracy: 0.0001)
    XCTAssertEqual(runtime.negativeOriginalImageHeight, 1536)
    XCTAssertEqual(runtime.negativeOriginalImageWidth, 1024)
    XCTAssertEqual(runtime.numFrames, 13)
    XCTAssertEqual(runtime.fpsId, 12)
    XCTAssertEqual(runtime.motionBucketId, 127)
    XCTAssertEqual(runtime.condAug, 0.15, accuracy: 0.0001)
    XCTAssertEqual(runtime.startFrameCfg, 1.1, accuracy: 0.0001)
    XCTAssertEqual(runtime.shift, 1.2, accuracy: 0.0001)
    XCTAssertEqual(runtime.stage2Steps, 11)
    XCTAssertEqual(runtime.stage2Cfg, 3.5, accuracy: 0.0001)
    XCTAssertEqual(runtime.stage2Shift, 0.7, accuracy: 0.0001)
    XCTAssertEqual(runtime.stochasticSamplingGamma, 0.05, accuracy: 0.0001)
    XCTAssertTrue(runtime.preserveOriginalAfterInpaint)
    XCTAssertTrue(runtime.t5TextEncoder)
    XCTAssertTrue(runtime.separateClipL)
    XCTAssertEqual(runtime.clipLText, "clip-l")
    XCTAssertTrue(runtime.separateOpenClipG)
    XCTAssertEqual(runtime.openClipGText, "open-clip-g")
    XCTAssertTrue(runtime.speedUpWithGuidanceEmbed)
    XCTAssertEqual(runtime.guidanceEmbed, 2.25, accuracy: 0.0001)
    XCTAssertTrue(runtime.resolutionDependentShift)
    XCTAssertTrue(runtime.teaCache)
    XCTAssertEqual(runtime.teaCacheStart, 2)
    XCTAssertEqual(runtime.teaCacheEnd, 8)
    XCTAssertEqual(runtime.teaCacheThreshold, 0.11, accuracy: 0.0001)
    XCTAssertEqual(runtime.teaCacheMaxSkipSteps, 3)
    XCTAssertTrue(runtime.separateT5)
    XCTAssertEqual(runtime.t5Text, "t5")
    XCTAssertTrue(runtime.causalInferenceEnabled)
    XCTAssertEqual(runtime.causalInference, 5)
    XCTAssertEqual(runtime.causalInferencePad, 2)
    XCTAssertTrue(runtime.cfgZeroStar)
    XCTAssertEqual(runtime.cfgZeroInitSteps, 4)
    XCTAssertEqual(runtime.compressionArtifacts, .H264)
    XCTAssertEqual(runtime.compressionArtifactsQuality, 55, accuracy: 0.0001)
    XCTAssertEqual(runtime.colorCalibration, .lab)
    XCTAssertTrue(runtime.expandPromptToJson)
  }

  func testStateMapsSamplingSignpost() {
    let state = MediaGenerationPipeline.State(signpost: .sampling(2), totalSteps: 20)
    guard case .generating(let step, let totalSteps) = state else {
      return XCTFail("unexpected state: \(state)")
    }
    XCTAssertEqual(step, 3)
    XCTAssertEqual(totalSteps, 20)
  }

  func testLoRAStoreRejectsNonCloudBackend() throws {
    let modelsDirectory = try makeTemporaryDirectory()
    XCTAssertThrowsError(try LoRAStore(backend: .local(directory: modelsDirectory.path)))
  }

  func testLoRAStoreAcceptsCloudBackend() throws {
    XCTAssertNoThrow(try LoRAStore(backend: .cloudCompute(apiKey: "test-api-key")))
  }

  func testLoRAStoreDeleteRejectsEmptyKeys() async throws {
    let store = try LoRAStore(backend: .cloudCompute(apiKey: "test-api-key"))

    await XCTAssertThrowsErrorAsync(try await store.delete(keys: [""]))
  }

  func testCloudAuthenticatorRegistryReusesAuthenticatorForSameKey() {
    let first = CloudAuthenticatorRegistry.shared.authenticator(
      apiKey: "test-api-key",
      baseURL: CloudConfiguration.defaultBaseURL
    )
    let second = CloudAuthenticatorRegistry.shared.authenticator(
      apiKey: "test-api-key",
      baseURL: CloudConfiguration.defaultBaseURL
    )
    let different = CloudAuthenticatorRegistry.shared.authenticator(
      apiKey: "different-api-key",
      baseURL: CloudConfiguration.defaultBaseURL
    )

    XCTAssertTrue(first === second)
    XCTAssertFalse(first === different)
  }

  func testCloudAuthenticatorRegistrySharesCachedRemoteModels() {
    let apiKey = "test-api-key-\(UUID().uuidString)"
    let first = CloudAuthenticatorRegistry.shared.authenticator(
      apiKey: apiKey,
      baseURL: CloudConfiguration.defaultBaseURL
    )
    let second = CloudAuthenticatorRegistry.shared.authenticator(
      apiKey: apiKey,
      baseURL: CloudConfiguration.defaultBaseURL
    )

    first.updateRemoteModelsFromHandshake([
      "flux_2_klein_4b_i8x.ckpt",
      "qwen_image_2512_i8x.ckpt",
    ])

    XCTAssertEqual(
      second.remoteModels(),
      Set(["flux_2_klein_4b_i8x.ckpt", "qwen_image_2512_i8x.ckpt"])
    )
  }

  func testCloudAuthenticatorRemoteModelsDefaultsToEmptySet() {
    let authenticator = CloudAuthenticatorRegistry.shared.authenticator(
      apiKey: "test-api-key-\(UUID().uuidString)",
      baseURL: CloudConfiguration.defaultBaseURL
    )

    XCTAssertEqual(authenticator.remoteModels(), Set<String>())
  }

  func testLoRAImporterMissingFileThrowsFileNotFound() {
    let missingFile = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mediagenerationkit-missing-\(UUID().uuidString).safetensors"
    )
    var importer = LoRAImporter(file: missingFile)

    XCTAssertThrowsError(try importer.inspect()) { error in
      guard case LoRAConvertError.fileNotFound(let path) = error else {
        return XCTFail("unexpected error: \(error)")
      }
      XCTAssertEqual(path, missingFile.path)
    }
  }

  func testInputRoleWrappersPreserveRoles() {
    let input = MediaGenerationPipeline.data(Data([0x89, 0x50, 0x4e, 0x47]))

    XCTAssertEqual(input.role, .image)
    XCTAssertEqual((input.mask() as? MediaGenerationPipeline.MaskInput)?.role, .mask)
    XCTAssertEqual((input.moodboard() as? MediaGenerationPipeline.MoodboardInput)?.role, .moodboard)
    XCTAssertEqual((input.depth() as? MediaGenerationPipeline.DepthInput)?.role, .depth)
  }

  func testResultCanWritePNG() throws {
    let data = Data(base64Encoded: Self.onePixelPNGBase64)!
    let result = try MediaGenerationPipeline.Result(encodedData: data)
    XCTAssertEqual(result.width, 1)
    XCTAssertEqual(result.height, 1)

    let outputDirectory = try makeTemporaryDirectory()
    let outputFile = outputDirectory.appendingPathComponent("result.png")
    try result.write(to: outputFile, type: .png)

    let writtenData = try Data(contentsOf: outputFile)
    XCTAssertFalse(writtenData.isEmpty)
  }

  private func makeTemporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mediagenerationkit-tests-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock {
      try? FileManager.default.removeItem(at: directory)
    }
    return directory
  }

  private static let onePixelPNGBase64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2p2ioAAAAASUVORK5CYII="
}

private func XCTAssertThrowsErrorAsync<T>(
  _ expression: @autoclosure () async throws -> T,
  file: StaticString = #filePath,
  line: UInt = #line
) async {
  do {
    _ = try await expression()
    XCTFail("expected error", file: file, line: line)
  } catch {
  }
}
