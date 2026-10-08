import ArgumentParser
import CLICloudAuth
import Diffusion
import Downloader
import Foundation
import ImageGenerator
import ModelZoo
import NNC
import XCTest

@testable import DrawThingsCLILib

private final class TestToolAccountProvider: ToolAccountProvider {
  let resolve: (Bool, @escaping (Result<String?, Error>) -> Void) -> Void

  init(
    resolve: @escaping (Bool, @escaping (Result<String?, Error>) -> Void) -> Void = {
      _, completion in completion(.success(nil))
    }
  ) {
    self.resolve = resolve
  }

  func resolveDrawThingsCredential(
    prepareIfNeeded: Bool, completion: @escaping (Result<String?, Error>) -> Void
  ) {
    resolve(prepareIfNeeded, completion)
  }

  func drawThingsInsufficientFunds(apiKey: String) {
    XCTFail("These invocations should not open a purchase flow")
  }
}

final class DrawThingsCLIInvocationTests: XCTestCase {

  private func withContext(
    accountProvider: ToolAccountProvider? = nil,
    resolveModelsDirectory: ((URL?, @escaping (Result<URL, Error>) -> Void) -> Void)? = nil,
    unloadTextGenerator: (() -> Void)? = nil,
    modelDownloadEvent: ((ModelDownloadEvent) -> Void)? = nil,
    imageGenerationEvent: ((ImageGenerationEvent) -> Void)? = nil,
    _ body: (DrawThingsCLIContext, URL, () -> (String, String)) throws -> Void
  ) throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let input = try XCTUnwrap(tmpfile())
    let output = try XCTUnwrap(tmpfile())
    let error = try XCTUnwrap(tmpfile())
    defer {
      fclose(input)
      fclose(output)
      fclose(error)
      try? FileManager.default.removeItem(at: directory)
    }
    let context = DrawThingsCLIContext(
      input: input, output: output, error: error,
      environment: ["DRAWTHINGS_MODELS_DIR": "Models"], isStandardOutputTTY: false,
      resolvePath: { URL(fileURLWithPath: $0, relativeTo: directory).standardizedFileURL.path },
      accountProvider: accountProvider,
      resolveModelsDirectory: resolveModelsDirectory, unloadTextGenerator: unloadTextGenerator,
      modelDownloadEvent: modelDownloadEvent, imageGenerationEvent: imageGenerationEvent)
    func contents(_ stream: UnsafeMutablePointer<FILE>) -> String {
      fflush(stream)
      rewind(stream)
      var bytes = [UInt8](repeating: 0, count: 4096)
      var data = Data()
      while true {
        let count = fread(&bytes, 1, bytes.count, stream)
        if count == 0 { break }
        data.append(contentsOf: bytes.prefix(count))
      }
      return String(decoding: data, as: UTF8.self)
    }
    try body(context, directory, { (contents(output), contents(error)) })
  }

  func testHelpAndErrorsReturnWithoutExiting() throws {
    try withContext { context, _, contents in
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["generate", "--help"], context: context), 0)
      XCTAssertTrue(contents().0.contains("--model"))
      XCTAssertTrue(contents().1.isEmpty)
      XCTAssertNotEqual(DrawThingsCLI.run(arguments: ["--not-a-flag"], context: context), 0)
      XCTAssertTrue(contents().1.contains("--not-a-flag"))
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["models", "--help"], context: context), 0)
    }
  }

  func testHostStatusReportsDefaultWithoutProvisioningOrRevealingKey() throws {
    for key: String? in [nil, "dk_managed_test"] {
      try withContext(
        accountProvider: TestToolAccountProvider(resolve: { prepare, completion in
          XCTAssertFalse(prepare)
          completion(.success(key))
        })
      ) { context, _, contents in
        XCTAssertEqual(DrawThingsCLI.run(arguments: ["auth", "status"], context: context), 0)
        let value = try XCTUnwrap(
          JSONSerialization.jsonObject(with: Data(contents().0.utf8)) as? [String: Any])
        XCTAssertEqual(value["defaultBackend"] as? String, key == nil ? "local" : "cloud")
        XCTAssertEqual(value["cloudCredentialAvailable"] as? Bool, key != nil)
        XCTAssertFalse(contents().0.contains("dk_"))
      }
    }
    try withContext { context, _, contents in
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["auth", "status"], context: context), 0)
      XCTAssertTrue(contents().0.contains("\"local\""))
    }
  }

  func testHostKeyUsesDrawThingsEndpointAndExplicitOverridesStayIndependent() throws {
    var preparationCount = 0
    try withContext(
      accountProvider: TestToolAccountProvider(resolve: { prepare, completion in
        XCTAssertTrue(prepare)
        preparationCount += 1
        completion(.success("dk_host"))
      })
    ) { context, _, _ in
      let stored = CLICloudCredentials(
        provider: "google", apiKey: "dk_standalone", apiBaseURL: "https://other.test",
        savedAt: Date())
      let hosted = try context.cloudAuthentication(
        explicitAPIKey: nil, explicitBaseURL: nil, storedCredentials: stored)
      XCTAssertEqual(hosted.apiKey, "dk_host")
      XCTAssertEqual(hosted.baseURL, CLICloudDefaultAPIBaseURL)
      XCTAssertEqual(hosted.hostAPIKey, "dk_host")
      let overridden = try context.cloudAuthentication(
        explicitAPIKey: "dk_explicit", explicitBaseURL: "https://custom.test",
        storedCredentials: nil)
      XCTAssertEqual(overridden.apiKey, "dk_explicit")
      XCTAssertEqual(overridden.baseURL.absoluteString, "https://custom.test")
      XCTAssertNil(overridden.hostAPIKey)
      XCTAssertThrowsError(
        try context.cloudAuthentication(
          explicitAPIKey: nil, explicitBaseURL: "https://custom.test", storedCredentials: nil))
      XCTAssertEqual(preparationCount, 1)
    }
  }

  func testSignedOutHostDoesNotFallBackToAnotherSavedCLIAccount() throws {
    for prepareIfNeeded in [false, true] {
      try withContext(
        accountProvider: TestToolAccountProvider(resolve: { prepare, completion in
          XCTAssertEqual(prepare, prepareIfNeeded)
          completion(.success(nil))
        })
      ) { context, _, _ in
        let stored = CLICloudCredentials(
          provider: "google", apiKey: "dk_other_user", apiBaseURL: nil, savedAt: Date())
        XCTAssertThrowsError(
          try context.cloudAuthentication(
            explicitAPIKey: nil, explicitBaseURL: nil, storedCredentials: stored,
            prepareIfNeeded: prepareIfNeeded))
      }
    }
  }

  func testGenerationDefaultRechecksHostCredentialBeforeDownloadingWeights() throws {
    var savedKey: String?
    var lookups = 0
    let provider = TestToolAccountProvider(resolve: { prepare, completion in
      XCTAssertFalse(prepare)
      lookups += 1
      completion(.success(savedKey))
    })
    for key: String? in [nil, "dk_managed_test", nil] {
      savedKey = key
      let previousLookups = lookups
      var started = false
      try withContext(
        accountProvider: provider,
        unloadTextGenerator: { XCTFail("This test must not load local weights") },
        modelDownloadEvent: { _ in XCTFail("This test must not download weights") },
        imageGenerationEvent: { event in
          if case .started(_, _, _, _, _, let cancel) = event {
            started = true
            cancel()  // Verify cloud routing without connecting or spending credits.
          }
        }
      ) { context, _, contents in
        XCTAssertEqual(
          DrawThingsCLI.run(
            arguments: [
              "generate", "--no-download-missing", "--model", "flux_2_klein_4b_q6p.ckpt",
              "--prompt", "a cube", "--output", "cube.png",
            ], context: context), key == nil ? 1 : 130)
        XCTAssertEqual(lookups, previousLookups + 1)
        XCTAssertEqual(started, key != nil)
        let (output, error) = contents()
        XCTAssertTrue(output.contains(key == nil ? "Backend: local" : "Backend: cloud-compute"))
        XCTAssertEqual(error.contains("Missing model files:"), key == nil)
        XCTAssertFalse(output.contains("dk_managed_test"))
      }
    }
  }

  func testLocalOnlyRequestsDoNotConsultHostCredential() throws {
    let customModel = "test-local-model-\(UUID().uuidString).ckpt"
    let originalOverrides = ModelZoo.overrideMapping
    defer { ModelZoo.overrideMapping = originalOverrides }
    ModelZoo.overrideMapping[customModel] = ModelZoo.Specification(
      name: "Local custom model", file: customModel, prefix: "", version: .v1)
    let cases: [(String, [String])] = [
      ("flux_2_klein_4b_q6p.ckpt", ["--local"]),
      ("flux_2_klein_4b_q6p.ckpt", ["--offline"]),
      ("flux_2_klein_4b_q6p.ckpt", ["--output", "cube.mp4"]),
      ("flux_2_klein_4b_q6p.ckpt", ["--audio", "voice.wav"]),
      (
        "flux_2_klein_4b_q6p.ckpt",
        ["--config-json", #"{"loras":[{"file":"custom_lora.ckpt","weight":1}]}"#]
      ),
      ("ltx_2_19b_dev_q8p.ckpt", []),
      (customModel, []),
    ]
    for (model, options) in cases {
      let outputOptions = options.contains("--output") ? [] : ["--output", "cube.png"]
      try withContext(
        accountProvider: TestToolAccountProvider(resolve: { _, completion in
          XCTFail("Local-only request accessed the account: \(model) \(options)")
          completion(.success("dk_managed_test"))
        })
      ) { context, _, contents in
        XCTAssertNotEqual(
          DrawThingsCLI.run(
            arguments: [
              "generate", "--no-download-missing", "--model", model, "--prompt", "a cube",
            ]
              + options + outputOptions, context: context), 0)
        let (output, error) = contents()
        XCTAssertTrue(output.contains("Backend: local"), "\(model) \(options): \(error)")
      }
    }
  }

  func testExplicitBackendsOverrideHostDefault() throws {
    for flag in ["--cloud-compute", "--remote"] {
      try withContext(
        accountProvider: TestToolAccountProvider(resolve: { _, completion in
          XCTFail("Explicit backend must not look up the automatic default")
          completion(.success(nil))
        }),
        imageGenerationEvent: { event in
          if case .started(_, _, _, _, _, let cancel) = event { cancel() }
        }
      ) { context, _, contents in
        XCTAssertEqual(
          DrawThingsCLI.run(
            arguments: [
              "generate", flag, "--model", "flux_2_klein_4b_q6p.ckpt", "--prompt", "a cube",
              "--output", "cube.png",
            ], context: context), 130)
        XCTAssertTrue(contents().0.contains("Backend: \(flag.dropFirst(2))"))
      }
      XCTAssertThrowsError(try GenerateBackendOptions.parse(["--local", flag]).validate())
    }
  }

  func testCloudCredentialWaitCanBeCancelled() throws {
    let requested = expectation(description: "credential requested")
    let finished = expectation(description: "cancelled")
    try withContext(
      accountProvider: TestToolAccountProvider(resolve: { _, _ in requested.fulfill() })
    ) { context, _, _ in
      DispatchQueue.global().async {
        do {
          _ = try context.cloudAPIKey(prepareIfNeeded: true)
          XCTFail("Expected cancellation")
        } catch {
          XCTAssertTrue(error is DrawThingsCLIInvocationError)
        }
        finished.fulfill()
      }
      wait(for: [requested], timeout: 1)
      context.cancel()
      wait(for: [finished], timeout: 1)
    }
  }

  func testRepeatedImagesPreserveOrderAndDuplicates() throws {
    let command = try XCTUnwrap(
      DrawThingsCLI.parseAsRoot([
        "generate", "--image", "first.png", "--prompt", "combine the references",
        "--image", "second.png", "--image", "first.png", "--audio", "audio.wav",
      ]) as? DrawThingsCLI.Generate)
    XCTAssertEqual(command.imageInput.image, ["first.png", "second.png", "first.png"])
    XCTAssertEqual(command.imageInput.audio, "audio.wav")
    XCTAssertEqual(try GenerateImageInputOptions.parse([]).image, [])
    XCTAssertEqual(
      try GenerateImageInputOptions.parse(["--image", "first.png"]).image, ["first.png"])
    XCTAssertThrowsError(try GenerateImageInputOptions.parse(["--image", "first.png", "--image"]))
  }

  func testImageAliasesResolveToPrimaryImage() throws {
    for alias in ["--init-image", "--input-image"] {
      try withContext { context, _, contents in
        XCTAssertNotEqual(
          DrawThingsCLI.run(
            arguments: [
              "generate", "--offline", "--avc", "--image", "first.png", "--image", "second.png",
              alias, "second.png",
            ], context: context), 0)
        XCTAssertTrue(contents().1.contains("Use only one of --image or \(alias)"))
      }
      try withContext { context, _, contents in
        XCTAssertNotEqual(
          DrawThingsCLI.run(
            arguments: [
              "generate", "--offline", "--avc", "--image", "first.png", "--image", "second.png",
              alias, "./first.png",
            ], context: context), 0)
        XCTAssertTrue(contents().1.contains("--avc accepts exactly one --image."))
      }
      try withContext { context, _, contents in
        XCTAssertNotEqual(
          DrawThingsCLI.run(
            arguments: ["generate", "--offline", "--avc", alias, "first.png"], context: context), 0)
        XCTAssertTrue(contents().1.contains("--output is required with --avc."))
      }
    }
  }

  func testAVCRequiresExactlyOneImageBeforeModelResolution() throws {
    for (images, message) in [
      ([String](), "--image is required with --avc."),
      (["--image", "first.png", "--image", "second.png"], "--avc accepts exactly one --image."),
    ] {
      try withContext(
        resolveModelsDirectory: { _, _ in
          XCTFail("Invalid AVC images must fail before model access")
        }
      ) { context, _, contents in
        XCTAssertNotEqual(
          DrawThingsCLI.run(
            arguments: ["generate", "--offline", "--avc"] + images, context: context), 0)
        XCTAssertTrue(contents().1.contains(message))
      }
    }
  }

  func testCompletionUsesInvocationDirectoryAndMatchesStdout() throws {
    XCTAssertEqual(
      DrawThingsCLIContext.process().path("~/completion.sh"),
      URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("completion.sh")
        .standardizedFileURL.path)
    try withContext { context, directory, contents in
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["completion", "bash"], context: context), 0)
      let expected = contents().0
      XCTAssertFalse(expected.isEmpty)
      XCTAssertEqual(
        DrawThingsCLI.run(
          arguments: ["completion", "bash", "--output", "completion.sh"], context: context), 0)
      XCTAssertEqual(
        try String(contentsOf: directory.appendingPathComponent("completion.sh"), encoding: .utf8),
        expected)
    }
  }

  func testEnvironmentAndModelSettingsAreInvocationScoped() throws {
    try withContext { context, directory, contents in
      let originalURLs = ModelZoo.externalUrls
      let originalPreference = ModelZoo.isExternalUrlsPreferred
      XCTAssertEqual(
        DrawThingsCLI.run(arguments: ["models", "list", "--offline"], context: context), 0)
      XCTAssertTrue(
        FileManager.default.fileExists(atPath: directory.appendingPathComponent("Models").path))
      XCTAssertTrue(contents().0.contains("Models directory:"))
      XCTAssertEqual(ModelZoo.externalUrls, originalURLs)
      XCTAssertEqual(ModelZoo.isExternalUrlsPreferred, originalPreference)
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["--help"], context: context), 0)
      XCTAssertFalse(context.offline)
    }
  }

  func testCommunityModelListRefreshesCachedCatalog() throws {
    let stale = ModelZoo.Specification(
      name: "Stale community model", file: "test-stale-community.ckpt", prefix: "", version: .v1)
    let h3 = ModelZoo.Specification(
      name: "MiniMax H3", file: "minimax_h3_fl2va_q8p.ckpt", prefix: "", version: .minimaxH3)
    let official = try XCTUnwrap(
      ModelZoo.availableSpecifications.first { $0.remoteApiModelConfig == nil })
    var fetches = 0
    let resolver = CommunityModelResolver(
      cachedSpecifications: { [stale] },
      fetchSpecifications: {
        fetches += 1
        return [official, h3]
      })

    let files = resolver.allSpecifications().map(\.file)
    XCTAssertEqual(fetches, 1)
    XCTAssertTrue(files.contains(h3.file))
    XCTAssertFalse(files.contains(stale.file))
    XCTAssertEqual(files.first, official.file)
    XCTAssertEqual(files.filter { $0 == official.file }.count, 1)
  }

  func testCommunityModelListFallsBackToCacheWhenRefreshUnavailable() {
    let cached = ModelZoo.Specification(
      name: "Cached community model", file: "test-cached-community.ckpt", prefix: "", version: .v1)
    var fetches = 0
    let resolver = CommunityModelResolver(
      cachedSpecifications: { [cached] },
      fetchSpecifications: {
        fetches += 1
        return []
      })

    XCTAssertTrue(resolver.allSpecifications().contains { $0.file == cached.file })
    XCTAssertEqual(fetches, 1)
  }

  func testOfflineCommunityCatalogDoesNotFetch() {
    let cached = ModelZoo.Specification(
      name: "Cached community model", file: "test-cached-community.ckpt", prefix: "", version: .v1)
    let resolver = CommunityModelResolver(
      cachedSpecifications: { [cached] },
      fetchSpecifications: {
        XCTFail("Offline catalog access must not fetch")
        return []
      })

    XCTAssertTrue(
      resolver.allSpecifications(allowNetwork: false).contains { $0.file == cached.file })
    XCTAssertNil(resolver.resolve("minimax_h3_ref2va_i8x.ckpt", allowNetwork: false))
  }

  func testDownloadedCheckpointCanResolveFreshCommunityMetadata() throws {
    try withContext { _, directory, _ in
      let h3 = ModelZoo.Specification(
        name: "MiniMax H3 Ref2VA", file: "minimax_h3_ref2va_i8x.ckpt", prefix: "",
        version: .minimaxH3,
        huggingFaceLink: "MiniMaxAI/MiniMax-H3")
      try Data().write(to: directory.appendingPathComponent(h3.file))
      let originalURLs = ModelZoo.externalUrls
      let originalPreference = ModelZoo.isExternalUrlsPreferred
      let originalOverrides = ModelZoo.overrideMapping
      let originalRepoOverrides = ModelZoo.huggingFaceRepoOverrideMapping
      defer {
        ModelZoo.externalUrls = originalURLs
        ModelZoo.isExternalUrlsPreferred = originalPreference
        ModelZoo.overrideMapping = originalOverrides
        ModelZoo.huggingFaceRepoOverrideMapping = originalRepoOverrides
      }
      ModelZoo.externalUrls = [directory]
      ModelZoo.isExternalUrlsPreferred = true
      XCTAssertTrue(ModelZoo.isModelDownloaded(h3.file))
      var fetches = 0
      let resolver = CommunityModelResolver(
        cachedSpecifications: { [] },
        fetchSpecifications: {
          fetches += 1
          return [h3]
        })

      XCTAssertEqual(resolver.resolve(h3.file)?.file, h3.file)
      XCTAssertEqual(fetches, 1)
      XCTAssertEqual(ModelZoo.overrideMapping[h3.file]?.file, h3.file)
      XCTAssertEqual(
        ModelZoo.specificationForHuggingFaceRepo("MiniMaxAI/MiniMax-H3")?.file, h3.file)
    }
  }

  func testKnownCommunityModelUsesCacheWithoutFetching() {
    let cached = ModelZoo.Specification(
      name: "Cached community model", file: "test-cached-community.ckpt", prefix: "", version: .v1)
    let originalOverrides = ModelZoo.overrideMapping
    defer { ModelZoo.overrideMapping = originalOverrides }
    let resolver = CommunityModelResolver(
      cachedSpecifications: { [cached] },
      fetchSpecifications: {
        XCTFail("Resolving known metadata should use the cache")
        return []
      })

    XCTAssertEqual(resolver.resolve(cached.file)?.file, cached.file)
    XCTAssertEqual(resolver.resolve(cached.name)?.file, cached.file)
  }

  func testCapturedInputAndCancellation() throws {
    try withContext { context, _, _ in
      fputs("a prompt with 🐈\n", context.input)
      fflush(context.input)
      rewind(context.input)
      XCTAssertEqual(String(decoding: try context.readInput(), as: UTF8.self), "a prompt with 🐈\n")
      context.cancel()
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["--help"], context: context), 130)
      XCTAssertThrowsError(try context.readInput())
    }
  }

  func testHostCallbacksAreLazyAndInvocationScoped() throws {
    var resolutions = 0
    var unloads = 0
    try withContext(
      resolveModelsDirectory: { requested, completion in
        resolutions += 1
        completion(.success(requested!.appendingPathComponent("Authorized")))
      }, unloadTextGenerator: { unloads += 1 }
    ) { context, directory, contents in
      XCTAssertEqual(DrawThingsCLI.run(arguments: ["--help"], context: context), 0)
      XCTAssertEqual(resolutions, 0)
      XCTAssertEqual(unloads, 0)
      for invocation in 1...2 {
        XCTAssertEqual(
          DrawThingsCLI.run(arguments: ["models", "list", "--offline"], context: context), 0)
        XCTAssertEqual(resolutions, invocation)
        XCTAssertEqual(unloads, 0)
      }
      XCTAssertTrue(
        contents().0.contains(directory.appendingPathComponent("Models/Authorized").path))
      try context.prepareForLocalModelExecution()
      try context.prepareForLocalModelExecution()
      XCTAssertEqual(unloads, 1)
      context.finishInvocation()
      try context.prepareForLocalModelExecution()
      XCTAssertEqual(unloads, 2)
      context.finishInvocation()
    }
  }

  func testPermissionCancellationDoesNotUnload() throws {
    enum PermissionError: Error { case denied }
    try withContext(
      resolveModelsDirectory: { _, completion in completion(.failure(PermissionError.denied)) },
      unloadTextGenerator: { XCTFail("Permission failure must not unload the text generator") }
    ) { context, _, _ in
      XCTAssertNotEqual(
        DrawThingsCLI.run(arguments: ["models", "list", "--offline"], context: context), 0)
    }
    var pendingCompletion: ((Result<URL, Error>) -> Void)?
    try withContext(resolveModelsDirectory: { _, completion in pendingCompletion = completion }) {
      context, directory, _ in
      DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { context.cancel() }
      XCTAssertEqual(
        DrawThingsCLI.run(arguments: ["models", "list", "--offline"], context: context), 130)
      // A dismissed command may still receive the picker result; it owns no
      // folder access and must not resume execution or touch closed streams.
      pendingCompletion?(.success(directory))
    }
  }

  func testModelDownloadCardCancellationReturnsCommandOutput() throws {
    let file = "test-download-\(UUID().uuidString).ckpt"
    let originalOverrides = ModelZoo.overrideMapping
    defer { ModelZoo.overrideMapping = originalOverrides }
    ModelZoo.overrideMapping[file] = ModelZoo.Specification(
      name: "Test download", file: file, prefix: "", version: .v1)
    var started: UUID?
    var finished: UUID?
    try withContext(modelDownloadEvent: { event in
      switch event {
      case .started(let id, let name, _, let files, let cancel):
        started = id
        XCTAssertEqual(name, "Test download")
        XCTAssertEqual(files, [file])
        cancel()
      case .progress:
        XCTFail("Cancellation before transfer must not download data")
      case .finished(let id):
        finished = id
      }
    }) { context, _, contents in
      XCTAssertEqual(
        DrawThingsCLI.run(
          arguments: ["models", "ensure", "--model", file, "--no-include-dependencies"],
          context: context), 130)
      XCTAssertNotNil(started)
      XCTAssertEqual(started, finished)
      XCTAssertEqual(contents().1, "Download model cancelled by user\n")
      XCTAssertFalse(contents().0.contains("Model ready:"))
    }
  }

  func testDownloadEventsRespectMissingFilesAndOfflineOptions() throws {
    let first = "test-first-\(UUID().uuidString).ckpt"
    let second = "test-second-\(UUID().uuidString).ckpt"
    var starts = 0
    var finishes = 0
    try withContext(modelDownloadEvent: { event in
      switch event {
      case .started(_, _, _, let files, let cancel):
        starts += 1
        XCTAssertEqual(files, [first, second])
        cancel()
      case .finished: finishes += 1
      case .progress: XCTFail("This test cancels before network access")
      }
    }) { context, directory, _ in
      let previousURLs = ModelZoo.externalUrls
      let previousPreference = ModelZoo.isExternalUrlsPreferred
      defer {
        ModelZoo.externalUrls = previousURLs
        ModelZoo.isExternalUrlsPreferred = previousPreference
      }
      ModelZoo.externalUrls = [directory]
      ModelZoo.isExternalUrlsPreferred = true
      let existing = "test-existing-\(UUID().uuidString).ckpt"
      try Data().write(to: directory.appendingPathComponent(existing))
      try ModelDownloader.ensureFiles(
        context: context, [existing], modelsDirectory: directory, downloadMissing: true)
      XCTAssertEqual(starts, 0)
      XCTAssertThrowsError(
        try ModelDownloader.ensureFiles(
          context: context, [first], modelsDirectory: directory, downloadMissing: false))
      context.offline = true
      XCTAssertThrowsError(
        try ModelDownloader.ensureFiles(
          context: context, [first], modelsDirectory: directory, downloadMissing: true))
      XCTAssertEqual(starts, 0)
      context.offline = false
      XCTAssertThrowsError(
        try ModelDownloader.ensureFiles(
          context: context, [existing, first, first, second], modelsDirectory: directory,
          downloadMissing: true)
      ) { error in
        XCTAssertEqual(error.localizedDescription, "Download model cancelled by user")
      }
      XCTAssertEqual(starts, 1)
      XCTAssertEqual(finishes, 1)
    }
  }

  func testFinishedDownloadCannotCancelGenerationOrAnotherDownload() throws {
    var cancellations = [() -> Void]()
    try withContext(modelDownloadEvent: { event in
      if case .started(_, _, _, _, let cancel) = event { cancellations.append(cancel) }
    }) { context, _, _ in
      let first = context.beginModelDownload(name: "First", subtitle: "", files: ["first.ckpt"])
      context.finishModelDownload(first)
      cancellations[0]()
      XCTAssertFalse(context.isCancelled)
      let second = context.beginModelDownload(name: "Second", subtitle: "", files: ["second.ckpt"])
      cancellations[0]()
      XCTAssertFalse(context.isCancelled)
      var stopped = 0
      context.setCancellation { stopped += 1 }
      cancellations[1]()
      cancellations[1]()
      XCTAssertEqual(stopped, 1)
      XCTAssertThrowsError(try context.checkCancellation()) { error in
        XCTAssertEqual(error.localizedDescription, "Download model cancelled by user")
      }
      context.setCancellation(nil)
      context.finishModelDownload(second)
    }
  }

  func testGenerationCardCancellationReturnsToolResponse() throws {
    var started: UUID?
    var finished: UUID?
    let prompt = "A quiet mountain lake.\nSoft morning light."
    try withContext(
      unloadTextGenerator: { XCTFail("Cancelled generation must not load a local model") },
      imageGenerationEvent: { event in
        switch event {
        case .started(let id, let name, let version, let receivedPrompt, let signposts, let cancel):
          started = id
          XCTAssertFalse(name.isEmpty)
          XCTAssertEqual(receivedPrompt, prompt)
          XCTAssertEqual(version, .flux2_4b)
          XCTAssertTrue(signposts.contains(.sampling(20)))
          cancel()
        case .finished(let id): finished = id
        case .progress, .segmentStarted:
          XCTFail("Cancellation must happen before starting inference or connecting")
        }
      }
    ) { context, _, contents in
      XCTAssertEqual(
        DrawThingsCLI.run(
          arguments: [
            "generate", "--remote", "--remote-url", "127.0.0.1", "--remote-port", "1", "--model",
            "flux_2_klein_4b_q6p.ckpt",
            "--prompt", prompt, "--steps", "20", "--output", "result.png",
          ], context: context), 130)
      XCTAssertNotNil(started)
      XCTAssertEqual(started, finished)
      XCTAssertEqual(contents().1, "Generation aborted by user.\n")
      XCTAssertFalse(contents().0.contains("Wrote:"))
    }
  }

  func testGenerationEventsCarryPreviewAndIgnoreUpdatesAfterFinish() throws {
    var events = [ImageGenerationEvent]()
    try withContext(imageGenerationEvent: { events.append($0) }) { context, _, _ in
      let signposts: Set<ImageGeneratorSignpost> = [.textEncoded, .sampling(20), .imageDecoded]
      let id = context.beginImageGeneration(
        name: "Model", version: .v1, prompt: "Prompt", signposts: signposts)
      let preview = Tensor<FloatType>([-1, -0.5, 0.5, 1], .CPU, .NHWC(1, 1, 1, 4))
      context.updateImageGeneration(signpost: .sampling(3), signposts: signposts, preview: preview)
      context.updateImageGeneration(signpost: .sampling(4), signposts: signposts, preview: nil)
      context.beginImageGenerationSegment()
      context.finishImageGeneration(id)
      context.updateImageGeneration(signpost: .imageDecoded, signposts: signposts, preview: nil)
      XCTAssertEqual(events.count, 5)
      guard
        case .progress(let receivedID, let signpost, let receivedSignposts, let receivedPreview) =
          events[1]
      else { return XCTFail("Missing progress event") }
      XCTAssertEqual(receivedID, id)
      XCTAssertEqual(signpost, .sampling(3))
      XCTAssertEqual(receivedSignposts, signposts)
      let tensor = try XCTUnwrap(receivedPreview)
      XCTAssertEqual(tensor.shape, preview.shape)
      for channel in 0..<4 {
        XCTAssertEqual(tensor[0, 0, 0, channel], preview[0, 0, 0, channel])
      }
      guard case .progress(_, _, _, nil) = events[2] else {
        return XCTFail("Missing progress without a preview")
      }
      guard case .segmentStarted(let segmentID) = events[3] else {
        return XCTFail("Missing segment event")
      }
      XCTAssertEqual(segmentID, id)
      guard case .finished(let finishedID) = events[4] else {
        return XCTFail("Missing finish event")
      }
      XCTAssertEqual(finishedID, id)
    }
  }

  func testFinishedGenerationCannotCancelDownloadOrAnotherGeneration() throws {
    var cancellations = [() -> Void]()
    try withContext(imageGenerationEvent: { event in
      if case .started(_, _, _, _, _, let cancel) = event { cancellations.append(cancel) }
    }) { context, _, _ in
      let first = context.beginImageGeneration(
        name: "First", version: .v1, prompt: "", signposts: [])
      context.finishImageGeneration(first)
      cancellations[0]()
      XCTAssertFalse(context.isCancelled)
      let download = context.beginModelDownload(name: "Download", subtitle: "", files: [])
      cancellations[0]()
      XCTAssertFalse(context.isCancelled)
      context.finishModelDownload(download)
      let second = context.beginImageGeneration(
        name: "Second", version: .v1, prompt: "", signposts: [])
      cancellations[0]()
      XCTAssertFalse(context.isCancelled)
      var stopped = 0
      context.setCancellation { stopped += 1 }
      cancellations[1]()
      cancellations[1]()
      XCTAssertEqual(stopped, 1)
      XCTAssertThrowsError(try context.checkCancellation()) { error in
        XCTAssertEqual(error.localizedDescription, "Generation aborted by user.")
      }
      context.setCancellation(nil)
      context.finishImageGeneration(second)
    }
  }

}
