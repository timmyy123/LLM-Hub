#if os(macOS) || os(Linux)
import ArgumentParser
import CLICloudAuth
import Foundation
import MediaGenerationKit
import UniformTypeIdentifiers

private enum MediaGenerationKitCLICredentialsStore {
  private static let store = CLICloudCredentialsStore(
    applicationName: "MediaGenerationKitCLI")

  static func load() -> CLICloudCredentials? {
    store.load()
  }

  static func save(_ credentials: CLICloudCredentials) throws {
    try store.save(credentials)
  }

  static func remove() throws {
    try store.remove()
  }

  static func description() -> String {
    store.credentialsPath
  }
}

private struct AuthCommandOutput: Encodable {
  enum Action: String, Encodable {
    case login
    case logout
  }

  let action: Action
  let provider: String?
  let credentialsPath: String
  let cloudAPIBaseURL: String?

  static func login(
    credentials: CLICloudCredentials,
    credentialsPath: String,
    apiBaseURL: URL
  ) -> AuthCommandOutput {
    AuthCommandOutput(
      action: .login,
      provider: credentials.provider,
      credentialsPath: credentialsPath,
      cloudAPIBaseURL: apiBaseURL.absoluteString
    )
  }

  static func logout(credentialsPath: String) -> AuthCommandOutput {
    AuthCommandOutput(
      action: .logout,
      provider: nil,
      credentialsPath: credentialsPath,
      cloudAPIBaseURL: nil
    )
  }
}

private let defaultCloudAPIBaseURL = CLICloudDefaultAPIBaseURL

struct MediaGenerationKitCLIRunner {
  fileprivate static let defaultPromptValue = "a beautiful landscape with mountains and a lake"

  var modelsDirectory: String? = nil
  var prompt: String = Self.defaultPromptValue
  var promptFile: String? = nil
  var negativePrompt: String = ""
  var model: String = "flux_2_klein_4b_q8p.ckpt"
  var steps: Int? = nil
  var seed: UInt32 = 0
  var width: Int = 1024
  var height: Int = 1024
  var guidanceScale: Float? = nil
  var output: String = "output.png"
  var moodboard: [String] = []
  var json: Bool = false
  var jsonlProgress: Bool = false
  var dryRun: Bool = false
  var printResolvedConfig: Bool = false
  var saveManifest: String? = nil
  var verbose: Bool = false
  var weightsCache: Int = 8

  var useRemote: Bool = false
  var useCloudCompute: Bool = false
  var remoteUrl: String? = nil
  var remotePort: Int = 7859
  var remoteTls: Bool = false
  var listRemoteModels: Bool = false
  var ensureModel: String? = nil
  var convertLora: String? = nil
  var loraOutput: String? = nil
  var loraOutputDirectory: String? = nil
  var loraScale: Double = 1.0
  var uploadLora: String? = nil
  var listDownloadableModels: Bool = false
  var inspectModel: String? = nil
  var includeDownloadedModels: Bool = false
  var storageInfo: Bool = false

  var apiKey: String? = nil
  var cloudAPIBaseURL: String? = nil
  var testAuth: Bool = false
  var printAuthState: Bool = false
  var testAuthState: Bool = false

  var inputStrength: Float? = nil
  var inputImagePath: String? = nil
  var requireInputImage: Bool = false
  var logModelsDirectory: Bool = true

  /// Convert structured SDK auth state into a stable one-line string for CLI logs.
  /// We keep this mapping explicit so integration tests can assert exact state transitions.
  private enum CLIErrorCode: String, Encodable {
    case invalidArgument = "invalid_argument"
    case modelNotFound = "model_not_found"
    case modelFilesMissing = "model_files_missing"
    case downloadFailed = "download_failed"
    case generationFailed = "generation_failed"
    case outputWriteFailed = "output_write_failed"
    case cancelled = "cancelled"
    case authFailed = "auth_failed"
    case networkFailed = "network_failed"
    case timeout = "timeout"
    case internalError = "internal_error"
  }

  private enum MediaGenerationKitCLIExecutionError: LocalizedError {
    case invalidArgument(String)
    case modelNotFound(String)
    case modelFilesMissing(String)
    case outputWriteFailed(String)
    case timeout(String)
    case authFailed(String)
    case networkFailed(String)
    case generationFailed(String)

    var errorDescription: String? {
      switch self {
      case .invalidArgument(let message):
        return message
      case .modelNotFound(let message):
        return message
      case .modelFilesMissing(let message):
        return message
      case .outputWriteFailed(let message):
        return message
      case .timeout(let message):
        return message
      case .authFailed(let message):
        return message
      case .networkFailed(let message):
        return message
      case .generationFailed(let message):
        return message
      }
    }
  }

  private struct CLIErrorDescriptor {
    let code: CLIErrorCode
    let message: String
    let exitCode: Int32
  }

  private struct JSONErrorInfo: Encodable {
    let code: String
    let message: String
  }

  private struct JSONProgressEvent: Encodable {
    let event: String
    let elapsedSeconds: Double?
    let step: Int?
    let totalSteps: Int?
    let bytes: Int?
    let totalBytes: Int?
    let signpost: String?
    let error: JSONErrorInfo?
    let exitCode: Int?
  }

  private struct JSONGenerationOutput: Encodable {
    let ok: Bool
    let output: String
    let imageCount: Int
    let model: String
    let seed: UInt32
    let elapsedSeconds: Double
    let manifestPath: String?
  }

  private struct JSONDryRunOutput: Encodable {
    let ok: Bool
    let dryRun: Bool
    let model: String
    let seed: UInt32
    let manifestPath: String?
    let resolvedConfiguration: JSONResolvedConfiguration
  }

  private struct JSONResolvedConfiguration: Encodable {
    let prompt: String
    let negativePrompt: String
    let model: String
    let seed: UInt32
    let numInferenceSteps: Int
    let guidanceScale: Float
    let width: Int
    let height: Int
    let sampler: String
    let batchCount: Int
    let batchSize: Int
    let strength: Float
    let backend: String
    let output: String
    let inputImage: String?
    let moodboard: [String]
  }

  private struct JSONRunManifest: Encodable {
    let dryRun: Bool
    let output: String
    let imageCount: Int
    let model: String
    let seed: UInt32
    let elapsedSeconds: Double?
    let resolvedConfiguration: JSONResolvedConfiguration
  }

  private struct JSONModelInspectOutput: Encodable {
    let ok: Bool
    let inputModel: String
    let resolvedModel: String
    let isModelDownloaded: Bool
    let info: CLIModelInspectInfo
  }

  private struct CLIModelInspectInfo: Encodable {
    let name: String?
    let version: String?
    let description: String?
    let huggingFaceLink: String?
  }

  private struct JSONErrorOutput: Encodable {
    let ok: Bool
    let error: JSONErrorInfo
    let exitCode: Int
  }

  private final class GenerationRunState: @unchecked Sendable {
    struct Snapshot {
      let textEncodedTime: TimeInterval?
      let firstSamplingTime: TimeInterval?
      let lastSamplingTime: TimeInterval?
      let imageDecodedTime: TimeInterval?
      let generationResult: Result<[MediaGenerationPipeline.Result], Error>?
    }

    private let lock = NSLock()
    private var textEncodedTime: TimeInterval?
    private var firstSamplingTime: TimeInterval?
    private var lastSamplingTime: TimeInterval?
    private var imageDecodedTime: TimeInterval?
    private var generationResult: Result<[MediaGenerationPipeline.Result], Error>?

    func recordTextEncoded(_ elapsed: TimeInterval) {
      lock.lock()
      textEncodedTime = elapsed
      lock.unlock()
    }

    func recordSampling(step: Int, elapsed: TimeInterval) {
      lock.lock()
      if step == 1, firstSamplingTime == nil {
        firstSamplingTime = elapsed
      }
      lastSamplingTime = elapsed
      lock.unlock()
    }

    func recordImageDecoded(_ elapsed: TimeInterval) {
      lock.lock()
      imageDecodedTime = elapsed
      lock.unlock()
    }

    func setResult(_ result: Result<[MediaGenerationPipeline.Result], Error>) {
      lock.lock()
      generationResult = result
      lock.unlock()
    }

    func snapshot() -> Snapshot {
      lock.lock()
      defer { lock.unlock() }
      return Snapshot(
        textEncodedTime: textEncodedTime,
        firstSamplingTime: firstSamplingTime,
        lastSamplingTime: lastSamplingTime,
        imageDecodedTime: imageDecodedTime,
        generationResult: generationResult
      )
    }
  }

  private final class AsyncResultBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<Value, Error>?

    func store(_ result: Result<Value, Error>) {
      lock.lock()
      defer { lock.unlock() }
      self.result = result
    }

    func load() -> Result<Value, Error>? {
      lock.lock()
      defer { lock.unlock() }
      return result
    }
  }

  private var machineReadableOutputEnabled: Bool {
    json || jsonlProgress
  }

  private func resolvedCloudAPIBaseURL(using storedCredentials: CLICloudCredentials?) throws -> URL {
    do {
      return try CLICloudAuthClient.resolvedAPIBaseURL(
        explicit: cloudAPIBaseURL,
        storedCredentials: storedCredentials
      )
    } catch let error as CLICloudAuthError {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(error.localizedDescription)
    }
  }

  private func textLog(_ text: String, terminator: String = "\n") {
    guard !machineReadableOutputEnabled else { return }
    Swift.print(text, terminator: terminator)
  }

  private func emitJSON<T: Encodable>(_ value: T) {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    guard let data = try? encoder.encode(value),
      let text = String(data: data, encoding: .utf8)
    else {
      return
    }
    Swift.print(text)
  }

  private func renderJSON<T: Encodable>(_ value: T, pretty: Bool) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = pretty ? [.sortedKeys, .prettyPrinted] : [.sortedKeys]
    let data = try encoder.encode(value)
    return String(decoding: data, as: UTF8.self)
  }

  private func writeJSONFile<T: Encodable>(_ value: T, to path: String) throws {
    let text = try renderJSON(value, pretty: true)
    do {
      try text.write(to: URL(fileURLWithPath: path), atomically: true, encoding: .utf8)
    } catch {
      throw MediaGenerationKitCLIExecutionError.outputWriteFailed(
        "Failed to write JSON to '\(path)': \(error.localizedDescription)")
    }
  }

  private func formatBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
  }

  private func readPromptFromStdin() throws -> String {
    let data = FileHandle.standardInput.readDataToEndOfFile()
    guard var text = String(data: data, encoding: .utf8) else {
      throw MediaGenerationKitCLIExecutionError.invalidArgument("Failed to decode stdin prompt as UTF-8.")
    }
    while text.hasSuffix("\n") || text.hasSuffix("\r") {
      text.removeLast()
    }
    return text
  }

  private func resolvePromptText() throws -> String {
    if let promptFile = promptFile {
      if prompt == "-" {
        throw MediaGenerationKitCLIExecutionError.invalidArgument(
          "Do not combine --prompt-file with --prompt '-'. Use one prompt input source.")
      }
      if prompt != Self.defaultPromptValue {
        throw MediaGenerationKitCLIExecutionError.invalidArgument(
          "Do not combine --prompt-file with an explicit --prompt value.")
      }
      if promptFile == "-" {
        return try readPromptFromStdin()
      }
      do {
        var text = try String(contentsOfFile: promptFile, encoding: .utf8)
        while text.hasSuffix("\n") || text.hasSuffix("\r") {
          text.removeLast()
        }
        return text
      } catch {
        throw MediaGenerationKitCLIExecutionError.invalidArgument(
          "Failed to read --prompt-file '\(promptFile)': \(error.localizedDescription)")
      }
    }
    if prompt == "-" {
      return try readPromptFromStdin()
    }
    return prompt
  }

  fileprivate func mergedAlias(
    primary: String?, alias: String?, primaryFlag: String, aliasFlag: String
  ) throws -> String? {
    if let primary, let alias, primary != alias {
      throw MediaGenerationKitCLIExecutionError.invalidArgument("Use only one of \(primaryFlag) or \(aliasFlag).")
    }
    return primary ?? alias
  }

  private func loadImageData(path: String, label: String) throws -> Data {
    let filePath = URL(fileURLWithPath: path).standardizedFileURL.path
    guard FileManager.default.fileExists(atPath: filePath) else {
      throw MediaGenerationKitCLIExecutionError.invalidArgument("\(label) path does not exist: \(filePath)")
    }
    do {
      return try Data(contentsOf: URL(fileURLWithPath: filePath))
    } catch {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "Failed to read \(label) '\(filePath)': \(error.localizedDescription)")
    }
  }

  /// Wait while allowing main-queue callbacks (for auth-state observer printing).
  private func waitForSemaphore(_ semaphore: DispatchSemaphore, timeout: TimeInterval = 300) -> Bool
  {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
      if semaphore.wait(timeout: .now()) == .success {
        return true
      }
      RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.01))
    }
    return false
  }

  private func runAsync<T>(
    timeout: TimeInterval = 300,
    operation: @escaping @Sendable () async throws -> T
  ) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let resultBox = AsyncResultBox<T>()
    Task {
      do {
        let value = try await operation()
        resultBox.store(.success(value))
      } catch {
        resultBox.store(.failure(error))
      }
      semaphore.signal()
    }
    guard waitForSemaphore(semaphore, timeout: timeout) else {
      throw MediaGenerationKitCLIExecutionError.timeout("Operation timed out.")
    }
    guard let result = resultBox.load() else {
      throw MediaGenerationKitCLIExecutionError.generationFailed(
        "Operation completed without a result.")
    }
    return try result.get()
  }

  private func outputType(for path: String) -> UTType {
    let url = URL(fileURLWithPath: path)
    if let type = UTType(filenameExtension: url.pathExtension), type.conforms(to: .image) {
      return type
    }
    return .png
  }

  private func fetchShortTermToken(
    apiKey: String,
    baseURL: URL,
    emitStates: Bool
  ) throws -> String {
    try CLICloudAuthClient(baseURL: baseURL).fetchShortTermToken(
      apiKey: apiKey,
      emitStates: emitStates,
      stateHandler: { state in
        textLog("  [AuthState] \(describeCLICloudAuthState(state))")
      }
    ).value
  }

  private func inspectInfo(for resolvedModel: String) async -> CLIModelInspectInfo {
    guard let inspection = try? await MediaGenerationEnvironment.default.inspectModel(
      resolvedModel,
      offline: false
    ) else {
      return CLIModelInspectInfo(
        name: nil,
        version: nil,
        description: nil,
        huggingFaceLink: nil
      )
    }
    return CLIModelInspectInfo(
      name: inspection.name,
      version: inspection.version,
      description: inspection.description,
      huggingFaceLink: inspection.huggingFaceLink
    )
  }

  private func defaultModelsDirectoryURL() throws -> URL {
    let url: URL
    if let modelsDirectory, !modelsDirectory.isEmpty {
      url = URL(fileURLWithPath: modelsDirectory, isDirectory: true).standardizedFileURL
    } else if let externalURL = MediaGenerationEnvironment.default.externalUrls.first {
      url = externalURL.standardizedFileURL
    } else {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "Could not resolve a models directory. Pass --models-dir or configure DRAWTHINGS_MODELS_DIR.")
    }
    var isDirectory: ObjCBool = false
    let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
    if exists && !isDirectory.boolValue {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "Models directory path is not a directory: \(url.path)")
    }
    if !exists {
      try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }
    return url
  }

  private func prepareDefaultEnvironmentForLocalModels() throws -> URL {
    let directoryURL = try defaultModelsDirectoryURL()
    MediaGenerationEnvironment.default.externalUrls = [directoryURL]
    return directoryURL
  }

  private func resolvedRemoteEndpoint() throws -> (endpoint: MediaGenerationPipeline.Endpoint, useTLS: Bool) {
    guard let remoteUrl, !remoteUrl.isEmpty else {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--remote-url is required.")
    }
    let trimmed = remoteUrl.trimmingCharacters(in: .whitespacesAndNewlines)
    let candidate = trimmed.contains("://") ? trimmed : "http://\(trimmed)"
    let components = URLComponents(string: candidate)
    guard let components, let host = components.host, !host.isEmpty else {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "Failed to parse --remote-url '\(remoteUrl)'.")
    }
    let port = components.port ?? remotePort
    return (
      MediaGenerationPipeline.Endpoint(host: host, port: port),
      remoteTls || components.scheme?.lowercased() == "https"
    )
  }

  private func outputURL(
    for basePath: String,
    index: Int,
    totalCount: Int,
    type: UTType
  ) -> URL {
    let baseURL = URL(fileURLWithPath: basePath)
    let preferredExtension = type.preferredFilenameExtension
    if totalCount == 1 {
      if baseURL.pathExtension.isEmpty, let preferredExtension {
        return baseURL.appendingPathExtension(preferredExtension)
      }
      return baseURL
    }
    let directory = baseURL.deletingLastPathComponent()
    let stem =
      baseURL.pathExtension.isEmpty
      ? baseURL.lastPathComponent
      : baseURL.deletingPathExtension().lastPathComponent
    let pathExtension = baseURL.pathExtension.isEmpty ? preferredExtension ?? "" : baseURL.pathExtension
    let filename = pathExtension.isEmpty ? "\(stem)-\(index)" : "\(stem)-\(index).\(pathExtension)"
    return directory.appendingPathComponent(filename)
  }

  private func sanitizedFilenameStem(_ value: String) -> String {
    let scalars = value.unicodeScalars.map { scalar -> Character in
      if CharacterSet.alphanumerics.contains(scalar) {
        return Character(scalar)
      }
      return "_"
    }
    let raw = String(scalars)
    let collapsed = raw.replacingOccurrences(
      of: "_+",
      with: "_",
      options: .regularExpression
    )
    let trimmed = collapsed.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
    return trimmed.isEmpty ? "lora" : trimmed.lowercased()
  }

  private func resolvedLoRAOutputURL(inputURL: URL) throws -> URL {
    if loraOutput != nil, loraOutputDirectory != nil {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "Use only one of --output or --output-dir for LoRA conversion.")
    }

    if let loraOutput {
      return URL(fileURLWithPath: loraOutput).standardizedFileURL
    }

    let directoryURL: URL
    if let loraOutputDirectory {
      directoryURL = URL(fileURLWithPath: loraOutputDirectory, isDirectory: true).standardizedFileURL
    } else {
      directoryURL = inputURL.deletingLastPathComponent()
    }
    let stem = sanitizedFilenameStem(inputURL.deletingPathExtension().lastPathComponent)
    return directoryURL.appendingPathComponent("\(stem)_lora_f16.ckpt")
  }

  private func resolveModelReferenceOrThrow(_ input: String, flag: String = "--model") async throws
    -> String
  {
    if let resolved = await MediaGenerationEnvironment.default.resolveModel(input, offline: false)?.file {
      return resolved
    }
    let suggestions = await MediaGenerationEnvironment.default.suggestedModels(
      for: input,
      offline: false
    )
    guard !suggestions.isEmpty else {
      throw MediaGenerationKitCLIExecutionError.modelNotFound("Could not resolve \(flag) '\(input)'.")
    }
    let lines = suggestions.map { "  - \($0.file) (\($0.name))" }.joined(separator: "\n")
    throw MediaGenerationKitCLIExecutionError.modelNotFound(
      "Could not resolve \(flag) '\(input)'.\nClosest matches:\n\(lines)")
  }

  private func resolveGenerationModelOrThrow(
    _ input: String,
    backend: MediaGenerationPipeline.Backend,
    flag: String = "--model"
  ) async throws -> String {
    switch backend {
    case .local:
      return try await resolveModelReferenceOrThrow(input, flag: flag)
    case .remote, .cloudCompute:
      return await MediaGenerationEnvironment.default.resolveModel(input, offline: false)?.file ?? input
    }
  }

  private func resolveModelReferenceError(
    _ input: String,
    flag: String
  ) async throws -> MediaGenerationKitCLIExecutionError {
    let localSuggestions = await MediaGenerationEnvironment.default.suggestedModels(
      for: input,
      offline: false
    ).map {
      "\($0.file) (\($0.name))"
    }
    let combinedSuggestions = Array(localSuggestions.prefix(5))
    guard !combinedSuggestions.isEmpty else {
      return .modelNotFound("Could not resolve \(flag) '\(input)'.")
    }
    let lines = combinedSuggestions.map { "  - \($0)" }.joined(separator: "\n")
    return .modelNotFound(
      "Could not resolve \(flag) '\(input)'.\nClosest matches:\n\(lines)")
  }

  private func resolvedConfiguration(
    prompt: String,
    negativePrompt: String,
    configuration: MediaGenerationPipeline.Configuration,
    backend: String,
    inputImage: String?,
    moodboard: [String]
  ) -> JSONResolvedConfiguration {
    JSONResolvedConfiguration(
      prompt: prompt,
      negativePrompt: negativePrompt,
      model: configuration.model,
      seed: configuration.seed,
      numInferenceSteps: configuration.steps,
      guidanceScale: configuration.guidanceScale,
      width: configuration.width,
      height: configuration.height,
      sampler: String(describing: configuration.sampler),
      batchCount: configuration.batchCount,
      batchSize: configuration.batchSize,
      strength: configuration.strength,
      backend: backend,
      output: output,
      inputImage: inputImage,
      moodboard: moodboard
    )
  }

  private func describeError(_ error: Error) -> CLIErrorDescriptor {
    if let commandError = error as? MediaGenerationKitCLIExecutionError {
      switch commandError {
      case .invalidArgument(let message):
        return CLIErrorDescriptor(code: .invalidArgument, message: message, exitCode: 2)
      case .modelNotFound(let message):
        return CLIErrorDescriptor(code: .modelNotFound, message: message, exitCode: 3)
      case .modelFilesMissing(let message):
        return CLIErrorDescriptor(code: .modelFilesMissing, message: message, exitCode: 4)
      case .outputWriteFailed(let message):
        return CLIErrorDescriptor(code: .outputWriteFailed, message: message, exitCode: 7)
      case .timeout(let message):
        return CLIErrorDescriptor(code: .timeout, message: message, exitCode: 12)
      case .authFailed(let message):
        return CLIErrorDescriptor(code: .authFailed, message: message, exitCode: 9)
      case .networkFailed(let message):
        return CLIErrorDescriptor(code: .networkFailed, message: message, exitCode: 10)
      case .generationFailed(let message):
        return CLIErrorDescriptor(code: .generationFailed, message: message, exitCode: 6)
      }
    }

    if let validation = error as? ValidationError {
      return CLIErrorDescriptor(
        code: .invalidArgument, message: validation.message, exitCode: 2)
    }

    if let sdkError = error as? MediaGenerationKitError {
      switch sdkError {
      case .invalidModelsDirectory:
        return CLIErrorDescriptor(
          code: .modelFilesMissing,
          message: "Models directory is invalid or inaccessible.",
          exitCode: 4)
      case .generationFailed(let message):
        let normalizedMessage = message.isEmpty ? "Generation failed." : message
        if normalizedMessage.hasPrefix("invalid request:") {
          return CLIErrorDescriptor(code: .invalidArgument, message: normalizedMessage, exitCode: 2)
        }
        return CLIErrorDescriptor(code: .generationFailed, message: normalizedMessage, exitCode: 6)
      case .cancelled:
        return CLIErrorDescriptor(code: .cancelled, message: "Generation cancelled.", exitCode: 8)
      case .remoteNotConfigured, .notConfigured, .localNotConfigured:
        return CLIErrorDescriptor(
          code: .invalidArgument, message: sdkError.localizedDescription, exitCode: 2)
      case .unresolvedModelReference:
        return CLIErrorDescriptor(
          code: .modelNotFound, message: sdkError.localizedDescription, exitCode: 3)
      case .modelNotFoundOnRemote(let model):
        return CLIErrorDescriptor(
          code: .modelNotFound, message: "Model not found on remote: \(model)", exitCode: 3)
      case .modelNotFoundInCatalog(let model):
        return CLIErrorDescriptor(
          code: .modelNotFound, message: "Model not found in catalog: \(model)", exitCode: 3)
      case .downloadFailed(let message):
        return CLIErrorDescriptor(code: .downloadFailed, message: message, exitCode: 5)
      case .hashMismatch(let file):
        return CLIErrorDescriptor(
          code: .downloadFailed, message: "Checksum mismatch: \(file)", exitCode: 5)
      case .insufficientStorage:
        return CLIErrorDescriptor(
          code: .modelFilesMissing, message: "Insufficient storage space.", exitCode: 4)
      case .asyncOperationRequired:
        return CLIErrorDescriptor(
          code: .invalidArgument, message: sdkError.localizedDescription, exitCode: 2)
      }
    }

    if error is URLError {
      return CLIErrorDescriptor(
        code: .networkFailed, message: error.localizedDescription, exitCode: 10)
    }

    return CLIErrorDescriptor(
      code: .internalError, message: error.localizedDescription, exitCode: 1)
  }

  func run() async throws {
    do {
      try await runImpl()
    } catch {
      let mapped = describeError(error)
      let errorPayload = JSONErrorInfo(code: mapped.code.rawValue, message: mapped.message)
      if machineReadableOutputEnabled {
        if jsonlProgress {
          emitJSON(
            JSONProgressEvent(
              event: "error",
              elapsedSeconds: nil,
              step: nil,
              totalSteps: nil,
              bytes: nil,
              totalBytes: nil,
              signpost: nil,
              error: errorPayload,
              exitCode: Int(mapped.exitCode)
            ))
        }
        if json || !jsonlProgress {
          emitJSON(
            JSONErrorOutput(ok: false, error: errorPayload, exitCode: Int(mapped.exitCode)))
        }
      } else {
        Swift.print("Error [\(mapped.code.rawValue)]: \(mapped.message)")
      }
      throw ExitCode(rawValue: mapped.exitCode)
    }
  }

  private func runImpl() async throws {
    let storedCredentials = MediaGenerationKitCLICredentialsStore.load()
    let baseURL = try resolvedCloudAPIBaseURL(using: storedCredentials)
    let effectiveAPIKey = CLICloudAuthClient.effectiveAPIKey(
      explicit: apiKey,
      storedCredentials: storedCredentials
    )

    if testAuthState {
      guard effectiveAPIKey != nil else {
        throw MediaGenerationKitCLIExecutionError.invalidArgument(
          "--api-key is required with --test-auth-state unless you already signed in with `auth login`.")
      }
      if apiKey == nil, let storedCredentials {
        textLog("Using saved cloud credentials (\(storedCredentials.provider)) from:")
        textLog("  \(MediaGenerationKitCLICredentialsStore.description())")
      }

      textLog("Cloud authentication configured")
      textLog("  API base URL: \(baseURL)")
      textLog("  Testing authentication...")
      _ = try fetchShortTermToken(apiKey: effectiveAPIKey!, baseURL: baseURL, emitStates: true)
      textLog("Auth state validation complete.")
      return
    }

    if testAuth {
      guard effectiveAPIKey != nil else {
        throw MediaGenerationKitCLIExecutionError.invalidArgument(
          "--api-key is required with --test-auth unless you already signed in with `auth login`.")
      }
      if apiKey == nil, let storedCredentials {
        textLog("Using saved cloud credentials (\(storedCredentials.provider)) from:")
        textLog("  \(MediaGenerationKitCLICredentialsStore.description())")
      }
      _ = try fetchShortTermToken(
        apiKey: effectiveAPIKey!,
        baseURL: baseURL,
        emitStates: printAuthState || testAuth
      )
      textLog("Short-term token fetched successfully.")
      textLog("Authentication test complete.")
      return
    }

    if useRemote && useCloudCompute {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--remote and --cloud-compute cannot be combined.")
    }

    if listRemoteModels && !useRemote {
      throw MediaGenerationKitCLIExecutionError.invalidArgument("--list-remote-models requires --remote.")
    }
    if storageInfo && useRemote {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--storage-info is local-only and cannot be combined with --remote.")
    }
    if ensureModel != nil && useRemote {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--model is local-only and cannot be combined with remote ensure.")
    }
    if convertLora != nil && useRemote {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--convert-lora is local-only and cannot be combined with --remote.")
    }
    if uploadLora != nil && useRemote {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--upload-lora is upload-only and cannot be combined with --remote.")
    }
    if uploadLora != nil && effectiveAPIKey == nil {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--upload-lora requires --api-key unless you already signed in with `auth login`.")
    }
    if listDownloadableModels && useRemote {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--list-downloadable-models is local-only and cannot be combined with --remote.")
    }
    if inputStrength != nil && inputImagePath == nil {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--strength requires --image.")
    }
    if useCloudCompute && effectiveAPIKey == nil {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--cloud-compute requires --api-key unless you already signed in with `auth login`.")
    }

    let hasUtilityAction =
      listRemoteModels || storageInfo || ensureModel != nil || convertLora != nil
      || uploadLora != nil || listDownloadableModels || inspectModel != nil
    if dryRun && hasUtilityAction {
      throw MediaGenerationKitCLIExecutionError.invalidArgument("--dry-run is only supported for generation flow.")
    }
    if printResolvedConfig && hasUtilityAction {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--print-resolved-config is only supported for generation flow.")
    }
    if saveManifest != nil && hasUtilityAction {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--save-manifest is only supported for generation flow.")
    }

    if listRemoteModels {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--list-remote-models is not supported by the current MediaGenerationKit public API.")
    }

    if storageInfo {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--storage-info is not supported by the current MediaGenerationKit public API.")
    }

    if let ensureModel = ensureModel {
      _ = try prepareDefaultEnvironmentForLocalModels()
      textLog("Ensuring model files...")
      textLog("  Model: \(ensureModel)")
      let resolved = try runAsync(timeout: 600) {
        try await MediaGenerationEnvironment.default.ensure(ensureModel) { state in
          switch state {
          case .resolving:
            self.textLog("  Resolving model...")
          case .verifying(let file, let fileIndex, let totalFiles):
            self.textLog("  [Verify] \(fileIndex)/\(totalFiles) \(file)")
          case .downloading(
            let file,
            let fileIndex,
            let totalFiles,
            let bytesWritten,
            let totalBytesExpected
          ):
            let percent =
              totalBytesExpected > 0
              ? (Double(bytesWritten) / Double(totalBytesExpected)) * 100
              : 0
            let line = String(
              format: "  [Download] %d/%d %@: %lld/%lld bytes (%.1f%%)",
              fileIndex,
              totalFiles,
              file,
              bytesWritten,
              totalBytesExpected,
              percent
            )
            self.textLog("\r\(line)", terminator: "")
            fflush(stdout)
          }
        }
      }
      textLog("")
      if resolved.file != ensureModel {
        textLog("  Resolved model: \(resolved.file)")
      }
      textLog("Model ensure complete.")
      return
    }

    if let convertLora = convertLora {
      let inputURL = URL(fileURLWithPath: convertLora).standardizedFileURL
      let outputURL = try resolvedLoRAOutputURL(inputURL: inputURL)
      var importer = LoRAImporter(file: inputURL)
      textLog("Converting LoRA...")
      textLog("  Source: \(inputURL.path)")
      textLog("  Output: \(outputURL.path)")
      if loraScale != 1.0 {
        textLog("  Scale:  \(loraScale)")
      }
      var lastStep: Float = -1
      try importer.inspect()
      try importer.import(to: outputURL, scaleFactor: loraScale) { step in
        let progress = Float(step)
        if progress != lastStep {
          self.textLog(String(format: "\r  Progress: %.0f%%", progress * 100), terminator: "")
          fflush(stdout)
          lastStep = progress
        }
      }
      textLog("")
      textLog("LoRA conversion complete.")
      textLog("  Output path: \(outputURL.path)")
      if let version = importer.version {
        textLog("  Version:     \(version)")
      }
      return
    }

    if let uploadLora = uploadLora {
      guard let effectiveAPIKey else {
        throw MediaGenerationKitCLIExecutionError.invalidArgument(
          "--upload-lora requires --api-key unless you already signed in with `auth login`.")
      }
      if apiKey == nil, let storedCredentials {
        textLog("Using saved cloud credentials (\(storedCredentials.provider)) from:")
        textLog("  \(MediaGenerationKitCLICredentialsStore.description())")
      }
      let fileURL = URL(fileURLWithPath: uploadLora).standardizedFileURL
      let fileData = try Data(contentsOf: fileURL)
      let backend = MediaGenerationPipeline.Backend.cloudCompute(
        apiKey: effectiveAPIKey,
        options: .init(baseURL: baseURL)
      )
      let store = try LoRAStore(backend: backend)
      textLog("Uploading LoRA to DrawThings cloud...")
      textLog("  File: \(fileURL.path)")
      let uploaded = try runAsync(timeout: 300) {
        try await store.upload(fileData, file: fileURL.lastPathComponent)
      }
      textLog("LoRA upload complete.")
      textLog("  File:   \(uploaded.file)")
      textLog("  SHA256: \(uploaded.sha256)")
      return
    }

    if listDownloadableModels {
      _ = try prepareDefaultEnvironmentForLocalModels()
      let models = await MediaGenerationEnvironment.default.downloadableModels(
        includeDownloaded: includeDownloadedModels,
        offline: false
      )
      textLog("  Downloadable models: \(models.count)")
      for model in models {
        textLog(
          "    - [official] \(model.file) | \(model.name) | \(model.version ?? "n/a") | downloaded=\(model.isDownloaded)"
        )
      }
      textLog("Model list complete.")
      return
    }

    if let inspectModel = inspectModel {
      _ = try? prepareDefaultEnvironmentForLocalModels()
      let backend: MediaGenerationPipeline.Backend
      if useRemote {
        let remote = try resolvedRemoteEndpoint()
        backend = .remote(remote.endpoint, options: .init(useTLS: remote.useTLS))
      } else {
        if let modelsDirectory {
          backend = .local(directory: modelsDirectory)
        } else {
          backend = .local
        }
      }
      let resolvedModel = try await resolveGenerationModelOrThrow(
        inspectModel,
        backend: backend,
        flag: "--model"
      )
      let info = await inspectInfo(for: resolvedModel)
      let isModelDownloaded =
        (try? await MediaGenerationEnvironment.default.inspectModel(
          resolvedModel,
          offline: false
        ).isDownloaded) ?? false
      if json {
        emitJSON(
          JSONModelInspectOutput(
            ok: true,
            inputModel: inspectModel,
            resolvedModel: resolvedModel,
            isModelDownloaded: isModelDownloaded,
            info: info
          ))
      } else {
        textLog("Model inspect:")
        textLog("  Input model: \(inspectModel)")
        textLog("  Resolved model: \(resolvedModel)")
        textLog("  Name: \(info.name ?? "n/a")")
        textLog("  Version: \(info.version ?? "n/a")")
        textLog("  Description: \(info.description ?? "n/a")")
        textLog("  Hugging Face: \(info.huggingFaceLink ?? "n/a")")
        textLog("  Downloaded: \(isModelDownloaded)")
      }
      return
    }

    let effectivePrompt = try resolvePromptText()
    if effectivePrompt.isEmpty {
      throw MediaGenerationKitCLIExecutionError.invalidArgument("Prompt is empty.")
    }
    if requireInputImage && inputImagePath == nil {
      throw MediaGenerationKitCLIExecutionError.invalidArgument("--image is required.")
    }

    MediaGenerationEnvironment.default.maxTotalWeightsCacheSize =
      UInt64(max(0, weightsCache)) * 1_024 * 1_024 * 1_024

    guard width > 0, width % 64 == 0, height > 0, height % 64 == 0 else {
      throw MediaGenerationKitCLIExecutionError.invalidArgument(
        "--width and --height must be positive multiples of 64.")
    }
    if let steps, steps <= 0 {
      throw MediaGenerationKitCLIExecutionError.invalidArgument("--num-inference-steps must be > 0.")
    }
    let effectiveStrength = inputStrength ?? (inputImagePath != nil ? 1.0 : nil)
    if let effectiveStrength, !(0...1).contains(effectiveStrength) {
      throw MediaGenerationKitCLIExecutionError.invalidArgument("--strength must be in [0, 1].")
    }

    let generationBackend: MediaGenerationPipeline.Backend
    let backendDescription: String
    if useCloudCompute {
      generationBackend = .cloudCompute(
        apiKey: effectiveAPIKey,
        options: .init(baseURL: baseURL)
      )
      backendDescription = "cloud-compute"
      if apiKey == nil, let storedCredentials {
        textLog("Using saved cloud credentials (\(storedCredentials.provider)) from:")
        textLog("  \(MediaGenerationKitCLICredentialsStore.description())")
      }
    } else if useRemote {
      let remote = try resolvedRemoteEndpoint()
      generationBackend = .remote(remote.endpoint, options: .init(useTLS: remote.useTLS))
      backendDescription = "remote"
    } else {
      if let modelsDirectory {
        generationBackend = .local(directory: modelsDirectory)
      } else {
        generationBackend = .local
      }
      backendDescription = "local"
    }
    let resolvedModel = try await resolveGenerationModelOrThrow(model, backend: generationBackend)
    if resolvedModel != model {
      textLog("Resolved model: \(model) -> \(resolvedModel)")
    }
    if logModelsDirectory, !useRemote, !useCloudCompute,
      let modelsDirectoryURL = try? defaultModelsDirectoryURL()
    {
      textLog("Models directory: \(modelsDirectoryURL.path)")
    }
    if verbose {
      textLog("Weights cache: \(weightsCache) GiB")
      textLog("Backend: \(backendDescription)")
    }

    var pipeline = try await MediaGenerationPipeline.fromPretrained(
      resolvedModel,
      backend: generationBackend
    )
    pipeline.configuration.width = width
    pipeline.configuration.height = height
    pipeline.configuration.seed = seed
    if let steps {
      pipeline.configuration.steps = steps
    }
    if let guidanceScale {
      pipeline.configuration.guidanceScale = guidanceScale
    }
    if let effectiveStrength {
      pipeline.configuration.strength = effectiveStrength
    }
    let finalNegativePrompt = negativePrompt
    var inputs: [MediaGenerationPipeline.Input] = []
    if let inputImagePath {
      _ = try loadImageData(path: inputImagePath, label: "input image")
      inputs.append(MediaGenerationPipeline.file(inputImagePath))
    }
    for moodboardPath in moodboard {
      _ = try loadImageData(path: moodboardPath, label: "moodboard image")
      inputs.append(MediaGenerationPipeline.file(moodboardPath).moodboard())
    }
    let generationPipeline = pipeline
    let generationInputs = inputs
    let estimatedComputeUnits =
      useCloudCompute ? try generationPipeline.estimatedComputeUnits(inputs: generationInputs) : nil

    let resolvedConfigurationPayload = resolvedConfiguration(
      prompt: effectivePrompt,
      negativePrompt: finalNegativePrompt,
      configuration: generationPipeline.configuration,
      backend: backendDescription,
      inputImage: inputImagePath,
      moodboard: moodboard
    )

    if printResolvedConfig {
      if machineReadableOutputEnabled {
        emitJSON(resolvedConfigurationPayload)
      } else {
        let pretty = try renderJSON(resolvedConfigurationPayload, pretty: true)
        textLog("Resolved configuration:")
        textLog(pretty)
      }
    }

    if let estimatedComputeUnits {
      textLog("Current request compute units: \(estimatedComputeUnits.formatted())")
    }

    var manifestPath: String?
    if dryRun {
      if let saveManifest = saveManifest {
        let manifest = JSONRunManifest(
          dryRun: true,
          output: output,
          imageCount: 0,
          model: resolvedConfigurationPayload.model,
          seed: resolvedConfigurationPayload.seed,
          elapsedSeconds: nil,
          resolvedConfiguration: resolvedConfigurationPayload
        )
        try writeJSONFile(manifest, to: saveManifest)
        manifestPath = saveManifest
        textLog("Saved manifest: \(saveManifest)")
      }
      textLog("Dry run completed. Use the same command without --dry-run to start generation.")
      if json {
        emitJSON(
          JSONDryRunOutput(
            ok: true,
            dryRun: true,
            model: resolvedConfigurationPayload.model,
            seed: resolvedConfigurationPayload.seed,
            manifestPath: manifestPath,
            resolvedConfiguration: resolvedConfigurationPayload
          ))
      }
      return
    }

    textLog("Starting generation...")
    let startTime = Date()
    let runState = GenerationRunState()

    if jsonlProgress {
      emitJSON(
          JSONProgressEvent(
          event: "started",
          elapsedSeconds: 0,
          step: nil,
          totalSteps: generationPipeline.configuration.steps,
          bytes: nil,
          totalBytes: nil,
          signpost: nil,
          error: nil,
          exitCode: nil
        ))
    }

    let generatedResults = try runAsync(timeout: 1800) {
      try await generationPipeline.generate(
        prompt: effectivePrompt,
        negativePrompt: finalNegativePrompt,
        inputs: generationInputs
      ) { state, _ in
        switch state {
        case .uploading(let uploaded, let total):
          self.textLog("  [Upload] \(uploaded)/\(total) bytes")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "upload_progress",
                elapsedSeconds: nil,
                step: nil,
                totalSteps: nil,
                bytes: uploaded,
                totalBytes: total,
                signpost: nil,
                error: nil,
                exitCode: nil
                ))
          }
        case .downloading(let downloaded, let total):
          self.textLog("  [Download] \(downloaded)/\(total) bytes")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "download_progress",
                elapsedSeconds: nil,
                step: nil,
                totalSteps: nil,
                bytes: downloaded,
                totalBytes: total,
                signpost: nil,
                error: nil,
                exitCode: nil
                ))
          }
        case .encodingText:
          let elapsed = Date().timeIntervalSince(startTime)
          runState.recordTextEncoded(elapsed)
          self.textLog("  [Progress] Text encoded (\(String(format: "%.2f", elapsed))s)")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "text_encoded",
                elapsedSeconds: elapsed,
                step: nil,
                totalSteps: nil,
                bytes: nil,
                totalBytes: nil,
                signpost: "textEncoded",
                error: nil,
                exitCode: nil
              ))
          }
        case .encodingInputs:
          let elapsed = Date().timeIntervalSince(startTime)
          self.textLog("  [Progress] Inputs encoded (\(String(format: "%.2f", elapsed))s)")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "inputs_encoded",
                elapsedSeconds: elapsed,
                step: nil,
                totalSteps: nil,
                bytes: nil,
                totalBytes: nil,
                signpost: "encodingInputs",
                error: nil,
                exitCode: nil
              ))
          }
        case .generating(let step, let totalSteps):
          let elapsed = Date().timeIntervalSince(startTime)
          runState.recordSampling(step: step, elapsed: elapsed)
          self.textLog(
            "  [Progress] Sampling step \(step)/\(totalSteps) (\(String(format: "%.2f", elapsed))s)"
          )
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "sampling",
                elapsedSeconds: elapsed,
                step: step,
                totalSteps: totalSteps,
                bytes: nil,
                totalBytes: nil,
                signpost: "sampling",
                error: nil,
                exitCode: nil
              ))
          }
        case .decoding:
          let elapsed = Date().timeIntervalSince(startTime)
          runState.recordImageDecoded(elapsed)
          self.textLog("  [Progress] Image decoded (\(String(format: "%.2f", elapsed))s)")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "image_decoded",
                elapsedSeconds: elapsed,
                step: nil,
                totalSteps: nil,
                bytes: nil,
                totalBytes: nil,
                signpost: "imageDecoded",
                error: nil,
                exitCode: nil
              ))
          }
        case .postprocessing:
          let elapsed = Date().timeIntervalSince(startTime)
          self.textLog("  [Progress] Postprocessing (\(String(format: "%.2f", elapsed))s)")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "postprocessing",
                elapsedSeconds: elapsed,
                step: nil,
                totalSteps: nil,
                bytes: nil,
                totalBytes: nil,
                signpost: "postprocessing",
                error: nil,
                exitCode: nil
              ))
          }
        case .preparing:
          let elapsed = Date().timeIntervalSince(startTime)
          self.textLog("  [Progress] Preparing (\(String(format: "%.2f", elapsed))s)")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "preparing",
                elapsedSeconds: elapsed,
                step: nil,
                totalSteps: nil,
                bytes: nil,
                totalBytes: nil,
                signpost: "preparing",
                error: nil,
                exitCode: nil
              ))
          }
        case .ensuringResources:
          let elapsed = Date().timeIntervalSince(startTime)
          self.textLog("  [Progress] Ensuring resources (\(String(format: "%.2f", elapsed))s)")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "ensuring_resources",
                elapsedSeconds: elapsed,
                step: nil,
                totalSteps: nil,
                bytes: nil,
                totalBytes: nil,
                signpost: "ensuringResources",
                error: nil,
                exitCode: nil
              ))
          }
        case .cancelling:
          let elapsed = Date().timeIntervalSince(startTime)
          self.textLog("  [Progress] Cancelling (\(String(format: "%.2f", elapsed))s)")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "cancelling",
                elapsedSeconds: elapsed,
                step: nil,
                totalSteps: nil,
                bytes: nil,
                totalBytes: nil,
                signpost: "cancelling",
                error: nil,
                exitCode: nil
              ))
          }
        case .completed:
          let elapsed = Date().timeIntervalSince(startTime)
          self.textLog("  [Progress] Completed (\(String(format: "%.2f", elapsed))s)")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "completed",
                elapsedSeconds: elapsed,
                step: nil,
                totalSteps: nil,
                bytes: nil,
                totalBytes: nil,
                signpost: "completed",
                error: nil,
                exitCode: nil
              ))
          }
        case .cancelled:
          let elapsed = Date().timeIntervalSince(startTime)
          self.textLog("  [Progress] Cancelled (\(String(format: "%.2f", elapsed))s)")
          if jsonlProgress {
            self.emitJSON(
              JSONProgressEvent(
                event: "cancelled",
                elapsedSeconds: elapsed,
                step: nil,
                totalSteps: nil,
                bytes: nil,
                totalBytes: nil,
                signpost: "cancelled",
                error: nil,
                exitCode: nil
              ))
          }
        case .resolvingBackend(let backend):
          if verbose {
            self.textLog("  [Progress] Resolving backend \(String(describing: backend))")
          }
        case .resolvingModel(let model):
          if verbose {
            self.textLog("  [Progress] Resolving model \(model)")
          }
        }
    }
    }
    runState.setResult(.success(generatedResults))

    let elapsed = Date().timeIntervalSince(startTime)
    let runSnapshot = runState.snapshot()

    switch runSnapshot.generationResult {
    case .success(let results):
      textLog("")
      textLog("Generation completed in \(String(format: "%.2f", elapsed)) seconds")
      textLog("Generated \(results.count) image(s)")
      textLog("")
      textLog("Timing breakdown:")
      if let textEncodedTime = runSnapshot.textEncodedTime {
        textLog("  Text encoding: \(String(format: "%.2f", textEncodedTime))s")
      }
      if let firstSamplingTime = runSnapshot.firstSamplingTime,
        let lastSamplingTime = runSnapshot.lastSamplingTime
      {
        let samplingDuration = lastSamplingTime - firstSamplingTime
        textLog("  Sampling: \(String(format: "%.2f", samplingDuration))s")
      }
      if let imageDecodedTime = runSnapshot.imageDecodedTime,
        let lastSamplingTime = runSnapshot.lastSamplingTime
      {
        let decodingDuration = imageDecodedTime - lastSamplingTime
        textLog("  Image decoding: \(String(format: "%.2f", decodingDuration))s")
      }

      let type = outputType(for: output)
      for (index, result) in results.enumerated() {
        let outputURL = outputURL(for: output, index: index, totalCount: results.count, type: type)
        do {
          try result.write(to: outputURL, type: type)
        } catch {
          throw MediaGenerationKitCLIExecutionError.outputWriteFailed(
            "Failed to write output image to '\(outputURL.path)': \(error.localizedDescription)")
        }
        textLog("Saved to: \(outputURL.path)")
      }
      textLog("")

      if let saveManifest = saveManifest {
        let manifest = JSONRunManifest(
          dryRun: false,
          output: output,
          imageCount: results.count,
          model: resolvedConfigurationPayload.model,
          seed: resolvedConfigurationPayload.seed,
          elapsedSeconds: elapsed,
          resolvedConfiguration: resolvedConfigurationPayload
        )
        try writeJSONFile(manifest, to: saveManifest)
        manifestPath = saveManifest
        textLog("Saved manifest: \(saveManifest)")
      }

      if json {
        emitJSON(
          JSONGenerationOutput(
            ok: true,
            output: output,
            imageCount: results.count,
            model: resolvedConfigurationPayload.model,
            seed: resolvedConfigurationPayload.seed,
            elapsedSeconds: elapsed,
            manifestPath: manifestPath
          ))
      }

    case .failure(let error):
      throw error

    case .none:
      throw MediaGenerationKitCLIExecutionError.generationFailed("Generation completed without a result.")
    }
  }
}

@main
struct MediaGenerationKitCLI: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "media-generation-kit-cli",
    abstract: "MediaGenerationKit CLI",
    subcommands: [
      Generate.self,
      Auth.self,
      Models.self,
      Lora.self,
      Storage.self,
    ],
    defaultSubcommand: Generate.self
  )

  private static let modelsDirectoryHelp = ArgumentHelp(
    "Models directory.",
    discussion:
      "Resolution order: --models-dir, DRAWTHINGS_MODELS_DIR, then Info.plist defaults and ~/Documents/Models via MediaGenerationEnvironment.default."
  )

  fileprivate static func makeRunner(modelsDirectory: String? = nil) -> MediaGenerationKitCLIRunner {
    MediaGenerationKitCLIRunner(modelsDirectory: modelsDirectory)
  }

  struct ModelsDirectoryOptions: ParsableArguments {
    @Option(name: .customLong("models-dir"), help: MediaGenerationKitCLI.modelsDirectoryHelp)
    var modelsDir: String?
  }

  struct SharedGenerationOptions: ParsableArguments {
    @Option(name: .shortAndLong, help: "The text prompt for image generation.")
    var prompt: String = MediaGenerationKitCLIRunner.defaultPromptValue

    @Option(
      name: .customLong("prompt-file"),
      help: "Read prompt from a file path. Use '-' to read from stdin."
    )
    var promptFile: String?

    @Option(name: .shortAndLong, help: "Negative prompt describing what to avoid.")
    var negativePrompt: String = ""

    @Option(name: .shortAndLong, help: "The model file name to use.")
    var model: String = "flux_2_klein_4b_q8p.ckpt"

    @Option(
      name: .customLong("num-inference-steps"),
      help: "Number of inference steps. Uses the model recommended value when omitted."
    )
    var steps: Int?

    @Option(
      name: [.customShort("S"), .long],
      help: "Random seed for reproducibility (0 for random)."
    )
    var seed: UInt32 = 0

    @Option(name: .shortAndLong, help: "Image width in pixels (must be multiple of 64).")
    var width: Int = 1024

    @Option(name: .shortAndLong, help: "Image height in pixels (must be multiple of 64).")
    var height: Int = 1024

    @Option(
      name: .shortAndLong,
      help: "Guidance scale for classifier-free guidance. Uses the model recommended value when omitted."
    )
    var guidanceScale: Float?

    @Option(name: .shortAndLong, help: "Output file path for the generated image.")
    var output: String = "output.png"

    @Option(
      name: .customLong("moodboard"),
      help:
        "Moodboard reference image path. Repeat to add multiple moodboard images. MediaGenerationKitCLI maps these onto shuffle hints."
    )
    var moodboard: [String] = []

    @Flag(name: .long, help: "Emit final generation result as one JSON object.")
    var json: Bool = false

    @Flag(
      name: [.customLong("jsonl"), .customLong("jsonl-progress")],
      help: "Emit generation progress as JSON Lines."
    )
    var jsonlProgress: Bool = false

    @Flag(name: .customLong("dry-run"), help: "Validate and resolve config without generation.")
    var dryRun: Bool = false

    @Flag(name: .customLong("print-resolved-config"), help: "Print resolved generation config.")
    var printResolvedConfig: Bool = false

    @Option(name: .customLong("save-manifest"), help: "Write run manifest JSON to this path.")
    var saveManifest: String?

    @Flag(help: "Enable verbose output.")
    var verbose: Bool = false

    @Option(name: .long, help: "The weights cache size in GiB (default: 8).")
    var weightsCache: Int = 8

    fileprivate func apply(to runner: inout MediaGenerationKitCLIRunner) {
      runner.prompt = prompt
      runner.promptFile = promptFile
      runner.negativePrompt = negativePrompt
      runner.model = model
      runner.steps = steps
      runner.seed = seed
      runner.width = width
      runner.height = height
      runner.guidanceScale = guidanceScale
      runner.output = output
      runner.moodboard = moodboard
      runner.json = json
      runner.jsonlProgress = jsonlProgress
      runner.dryRun = dryRun
      runner.printResolvedConfig = printResolvedConfig
      runner.saveManifest = saveManifest
      runner.verbose = verbose
      runner.weightsCache = weightsCache
    }
  }

  struct ImageInputOptions: ParsableArguments {
    @Option(name: .long, help: "Denoising strength for img2img (0...1).")
    var strength: Float?

    @Option(name: .long, help: "Input image path for img2img.")
    var image: String?

    @Option(name: .customLong("init-image"), help: "Alias of --image.")
    var initImage: String?

    @Option(name: .customLong("input-image"), help: "Alias of --image.")
    var inputImage: String?

    fileprivate func resolvedPath(using runner: MediaGenerationKitCLIRunner) throws -> String? {
      try runner.mergedAlias(
        primary: try runner.mergedAlias(
          primary: image, alias: initImage, primaryFlag: "--image", aliasFlag: "--init-image"),
        alias: inputImage, primaryFlag: "--image", aliasFlag: "--input-image")
    }
  }

  struct RemoteEndpointOptions: ParsableArguments {
    @Flag(name: .long, help: "Generate on a remote server instead of local models.")
    var remote: Bool = false

    @Option(name: .customLong("remote-url"), help: "Remote server URL.")
    var remoteUrl: String?

    @Option(name: .customLong("remote-port"), help: "Remote server port (default: 7859).")
    var remotePort: Int = 7859

    @Flag(name: .customLong("remote-tls"), help: "Use TLS for remote connection.")
    var remoteTls: Bool = false

    fileprivate func apply(to runner: inout MediaGenerationKitCLIRunner) throws {
      if remote {
        guard let remoteUrl, !remoteUrl.isEmpty else {
          throw ValidationError("--remote-url is required with --remote.")
        }
        runner.useRemote = true
        runner.remoteUrl = remoteUrl
        runner.remotePort = remotePort
        runner.remoteTls = remoteTls
        runner.logModelsDirectory = false
        return
      }
      if remoteUrl != nil || remotePort != 7859 || remoteTls {
        throw ValidationError("Remote endpoint flags require --remote.")
      }
    }
  }

  struct CloudComputeOptions: ParsableArguments {
    @Flag(name: .customLong("cloud-compute"), help: "Generate on Draw Things cloud compute.")
    var cloudCompute: Bool = false

    @Option(help: "DrawThings API key for cloud compute. Uses saved credentials if omitted.")
    var apiKey: String?

    @Option(name: .customLong("cloud-api-base-url"), help: "Cloud API base URL (default: https://api.drawthings.ai).")
    var cloudAPIBaseURL: String?

    fileprivate func apply(to runner: inout MediaGenerationKitCLIRunner) throws {
      if cloudCompute {
        runner.useCloudCompute = true
        runner.apiKey = apiKey
        runner.cloudAPIBaseURL = cloudAPIBaseURL
        runner.logModelsDirectory = false
        return
      }
      if apiKey != nil || cloudAPIBaseURL != nil {
        throw ValidationError("Cloud compute flags require --cloud-compute.")
      }
    }
  }

  struct RemoteServerOptions: ParsableArguments {
    @Option(name: .customLong("remote-url"), help: "Remote server URL.")
    var remoteUrl: String

    @Option(name: .customLong("remote-port"), help: "Remote server port (default: 7859).")
    var remotePort: Int = 7859

    @Flag(name: .customLong("remote-tls"), help: "Use TLS for remote connection.")
    var remoteTls: Bool = false
  }

  struct CloudAuthOptions: ParsableArguments {
    @Option(help: "DrawThings API key for cloud authentication. Uses saved credentials if omitted.")
    var apiKey: String?

    @Option(name: .customLong("cloud-api-base-url"), help: "Cloud API base URL (default: https://api.drawthings.ai).")
    var cloudAPIBaseURL: String?

    fileprivate func apply(to runner: inout MediaGenerationKitCLIRunner) {
      runner.apiKey = apiKey
      runner.cloudAPIBaseURL = cloudAPIBaseURL
    }
  }

  struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "generate",
      abstract: "Generate images."
    )

    @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions
    @OptionGroup var options: SharedGenerationOptions
    @OptionGroup var imageInput: ImageInputOptions
    @OptionGroup var remoteOptions: RemoteEndpointOptions
    @OptionGroup var cloudOptions: CloudComputeOptions

    func run() async throws {
      var runner = MediaGenerationKitCLI.makeRunner(modelsDirectory: modelsDirectoryOptions.modelsDir)
      options.apply(to: &runner)
      runner.inputStrength = imageInput.strength
      runner.inputImagePath = try imageInput.resolvedPath(using: runner)
      try remoteOptions.apply(to: &runner)
      try cloudOptions.apply(to: &runner)
      try await runner.run()
    }
  }

  struct Auth: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "auth",
      abstract: "Authentication helpers.",
      subcommands: [State.self, Token.self, Login.self, Logout.self]
    )

    struct State: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "state",
        abstract: "Validate auth-state progression."
      )

      @OptionGroup var auth: CloudAuthOptions

      func run() async throws {
        var runner = MediaGenerationKitCLI.makeRunner()
        auth.apply(to: &runner)
        runner.printAuthState = true
        runner.testAuthState = true
        try await runner.run()
      }
    }

    struct Token: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "token",
        abstract: "Fetch short-term token and exit."
      )

      @OptionGroup var auth: CloudAuthOptions

      func run() async throws {
        var runner = MediaGenerationKitCLI.makeRunner()
        auth.apply(to: &runner)
        runner.testAuth = true
        try await runner.run()
      }
    }

    struct Login: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "login",
        abstract: "Sign in with Google in your browser and save the returned Draw Things API key."
      )

      @Option(name: .customLong("cloud-api-base-url"), help: "Cloud API base URL (default: https://api.drawthings.ai).")
      var cloudAPIBaseURL: String?

      @Flag(name: .long, help: "Emit login result as JSON.")
      var json: Bool = false

      func run() async throws {
        let baseURL: URL
        if let cloudAPIBaseURL {
          guard let parsedURL = URL(string: cloudAPIBaseURL) else {
            throw ValidationError("Invalid --cloud-api-base-url: \(cloudAPIBaseURL)")
          }
          baseURL = parsedURL
        } else {
          baseURL = defaultCloudAPIBaseURL
        }

        if !json {
          Swift.print("Starting Google sign-in in your browser...")
          Swift.print("Credentials will be saved to: \(MediaGenerationKitCLICredentialsStore.description())")
        }

        let credentials = try CLICloudGoogleOAuthDesktopFlow.signIn(
          apiBaseURL: baseURL,
          callbackApplicationName: "MediaGenerationKitCLI",
          listenerQueueLabel: "ai.drawthings.mediagenerationkitcli.google-oauth"
        )
        try MediaGenerationKitCLICredentialsStore.save(credentials)

        let output = AuthCommandOutput.login(
          credentials: credentials,
          credentialsPath: MediaGenerationKitCLICredentialsStore.description(),
          apiBaseURL: baseURL
        )
        if json {
          let encoder = JSONEncoder()
          encoder.outputFormatting = [.sortedKeys]
          let data = try encoder.encode(output)
          Swift.print(String(decoding: data, as: UTF8.self))
        } else {
          Swift.print("Google sign-in complete.")
          Swift.print("Saved API key to: \(MediaGenerationKitCLICredentialsStore.description())")
        }
      }
    }

    struct Logout: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "logout",
        abstract: "Remove saved cloud credentials."
      )

      @Flag(name: .long, help: "Emit logout result as JSON.")
      var json: Bool = false

      func run() async throws {
        try MediaGenerationKitCLICredentialsStore.remove()
        let output = AuthCommandOutput.logout(
          credentialsPath: MediaGenerationKitCLICredentialsStore.description())
        if json {
          let encoder = JSONEncoder()
          encoder.outputFormatting = [.sortedKeys]
          let data = try encoder.encode(output)
          Swift.print(String(decoding: data, as: UTF8.self))
        } else {
          Swift.print("Removed saved cloud credentials from: \(MediaGenerationKitCLICredentialsStore.description())")
        }
      }
    }
  }

  struct Models: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "models",
      abstract: "Model catalog operations.",
      subcommands: [List.self, Ensure.self, ListRemote.self, Inspect.self]
    )

    struct List: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List downloadable models from catalog."
      )

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      @Flag(help: "Include already-downloaded models.")
      var includeDownloaded: Bool = false

      func run() async throws {
        var runner = MediaGenerationKitCLI.makeRunner(modelsDirectory: modelsDirectoryOptions.modelsDir)
        runner.listDownloadableModels = true
        runner.includeDownloadedModels = includeDownloaded
        try await runner.run()
      }
    }

    struct Ensure: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "ensure",
        abstract: "Ensure model files exist locally (download if missing)."
      )

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      @Option(name: .long, help: "Model file or reference.")
      var model: String

      func run() async throws {
        var runner = MediaGenerationKitCLI.makeRunner(modelsDirectory: modelsDirectoryOptions.modelsDir)
        runner.ensureModel = model
        try await runner.run()
      }
    }

    struct ListRemote: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "list-remote",
        abstract: "List remote server models/files."
      )

      @OptionGroup var remoteOptions: RemoteServerOptions

      func run() async throws {
        var runner = MediaGenerationKitCLI.makeRunner()
        runner.useRemote = true
        runner.remoteUrl = remoteOptions.remoteUrl
        runner.remotePort = remoteOptions.remotePort
        runner.remoteTls = remoteOptions.remoteTls
        runner.logModelsDirectory = false
        runner.listRemoteModels = true
        try await runner.run()
      }
    }

    struct Inspect: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "inspect",
        abstract: "Inspect resolved model metadata, using the online catalog when needed."
      )

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      @Option(name: .long, help: "Model file, name, hf:// repo, or Hugging Face URL.")
      var model: String

      @Flag(name: .long, help: "Emit inspect result as JSON.")
      var json: Bool = false

      func run() async throws {
        var runner = MediaGenerationKitCLI.makeRunner(modelsDirectory: modelsDirectoryOptions.modelsDir)
        runner.inspectModel = model
        runner.json = json
        try await runner.run()
      }
    }
  }

  struct Lora: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "lora",
      abstract: "LoRA conversion and upload operations.",
      subcommands: [Convert.self, Upload.self]
    )

    struct Convert: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "convert",
        abstract: "Convert safetensors LoRA to DrawThings ckpt."
      )

      @Option(name: .long, help: "Input LoRA path (.safetensors).")
      var input: String

      @Option(name: .shortAndLong, help: "Output file path for converted LoRA.")
      var output: String?

      @Option(name: .customLong("output-dir"), help: "Directory for derived converted LoRA output filename.")
      var outputDir: String?

      @Option(name: .long, help: "Scale factor.")
      var scale: Double = 1.0

      func run() async throws {
        var runner = MediaGenerationKitCLI.makeRunner()
        runner.convertLora = input
        runner.loraOutput = output
        runner.loraOutputDirectory = outputDir
        runner.loraScale = scale
        try await runner.run()
      }
    }

    struct Upload: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "upload",
        abstract: "Upload converted LoRA to DrawThings cloud."
      )

      @Option(name: .long, help: "Input LoRA ckpt path.")
      var input: String

      @OptionGroup var auth: CloudAuthOptions

      func run() async throws {
        var runner = MediaGenerationKitCLI.makeRunner()
        runner.uploadLora = input
        auth.apply(to: &runner)
        try await runner.run()
      }
    }
  }

  struct Storage: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "storage",
      abstract: "Storage info commands.",
      subcommands: [Info.self],
      defaultSubcommand: Info.self
    )

    struct Info: AsyncParsableCommand {
      static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Print storage usage."
      )

      @OptionGroup var modelsDirectoryOptions: ModelsDirectoryOptions

      func run() async throws {
        var runner = MediaGenerationKitCLI.makeRunner(modelsDirectory: modelsDirectoryOptions.modelsDir)
        runner.storageInfo = true
        try await runner.run()
      }
    }
  }
}
#else
import ArgumentParser

@main
struct MediaGenerationKitCLI: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "media-generation-kit-cli",
    abstract: "MediaGenerationKit CLI is only supported on macOS and Linux."
  )

  mutating func run() async throws {
    throw CleanExit.message("media-generation-kit-cli is only supported on macOS and Linux.")
  }
}
#endif
