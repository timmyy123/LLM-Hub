import CLICloudAuth
import Diffusion
import Downloader
import Foundation
import ImageGenerator
import NNC

#if canImport(Darwin)
  import Darwin
#else
  import Glibc
#endif

/// The host owns these streams until `DrawThingsCLI.run` returns. Callbacks use
/// the captured streams, including when generation runs on another thread.
public final class DrawThingsCLIContext {
  public enum Output {
    case standardOutput
    case standardError
  }

  public let input: UnsafeMutablePointer<FILE>
  public let output: UnsafeMutablePointer<FILE>
  public let error: UnsafeMutablePointer<FILE>
  public let environment: [String: String]
  public let executablePath: String?
  public let isStandardOutputTTY: Bool
  private let resolvePath: (String) -> String
  private let cancellationRequested: () -> Bool
  private let accountProvider: ToolAccountProvider?
  private let resolveModelsDirectory: ((URL?, @escaping (Result<URL, Error>) -> Void) -> Void)?
  private let unloadTextGenerator: (() -> Void)?
  private var didRequestTopUp = false
  var hasCloudAccountHost: Bool { accountProvider != nil }
  private var modelsDirectory: URL?
  private var isAccessingModelsDirectory = false
  private var didUnloadTextGenerator = false
  private let cancellationLock = NSLock()
  private var cancellation: (() -> Void)?
  private var cancelled = false
  private var cancellationError = DrawThingsCLIInvocationError.cancelled
  private var modelDownloadID: UUID?
  private var imageGenerationID: UUID?
  let modelDownloadEvent: ((ModelDownloadEvent) -> Void)?
  let imageGenerationEvent: ((ImageGenerationEvent) -> Void)?
  var offline = false

  public init(
    input: UnsafeMutablePointer<FILE>, output: UnsafeMutablePointer<FILE>,
    error: UnsafeMutablePointer<FILE>, environment: [String: String],
    executablePath: String? = nil, isStandardOutputTTY: Bool,
    resolvePath: @escaping (String) -> String,
    cancellationRequested: @escaping () -> Bool = { false },
    accountProvider: ToolAccountProvider? = nil,
    resolveModelsDirectory: ((URL?, @escaping (Result<URL, Error>) -> Void) -> Void)? = nil,
    unloadTextGenerator: (() -> Void)? = nil,
    modelDownloadEvent: ((ModelDownloadEvent) -> Void)? = nil,
    imageGenerationEvent: ((ImageGenerationEvent) -> Void)? = nil
  ) {
    self.input = input
    self.output = output
    self.error = error
    self.environment = environment
    self.executablePath = executablePath
    self.isStandardOutputTTY = isStandardOutputTTY
    self.resolvePath = resolvePath
    self.cancellationRequested = cancellationRequested
    self.accountProvider = accountProvider
    self.resolveModelsDirectory = resolveModelsDirectory
    self.unloadTextGenerator = unloadTextGenerator
    self.modelDownloadEvent = modelDownloadEvent
    self.imageGenerationEvent = imageGenerationEvent
  }

  public static func process() -> DrawThingsCLIContext {
    let directory = URL(
      fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    return DrawThingsCLIContext(
      input: stdin, output: stdout, error: stderr,
      environment: ProcessInfo.processInfo.environment,
      executablePath: CommandLine.arguments.first,
      isStandardOutputTTY: isatty(STDOUT_FILENO) != 0,
      resolvePath: {
        URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, relativeTo: directory)
          .standardizedFileURL.path
      })
  }

  var isCancelled: Bool {
    cancellationLock.lock()
    let value = cancelled
    cancellationLock.unlock()
    return value || cancellationRequested()
  }

  public func cancel() {
    cancel(nil)
  }

  private enum Activity {
    case modelDownload(UUID)
    case imageGeneration(UUID)
  }

  private func cancel(_ activity: Activity?) {
    cancellationLock.lock()
    guard !cancelled else {
      cancellationLock.unlock()
      return
    }
    switch activity {
    case .modelDownload(let id):
      guard modelDownloadID == id else {
        cancellationLock.unlock()
        return
      }
      cancellationError = .modelDownloadCancelled
    case .imageGeneration(let id):
      guard imageGenerationID == id else {
        cancellationLock.unlock()
        return
      }
      cancellationError = .generationAborted
    case nil:
      cancellationError = .cancelled
    }
    cancelled = true
    let cancellation = self.cancellation
    cancellationLock.unlock()
    cancellation?()
  }

  func setCancellation(_ cancellation: (() -> Void)?) {
    cancellationLock.lock()
    self.cancellation = cancellation
    let cancelled = self.cancelled
    cancellationLock.unlock()
    if cancelled { cancellation?() }
  }

  func checkCancellation() throws {
    cancellationLock.lock()
    let error = cancelled ? cancellationError : nil
    cancellationLock.unlock()
    if let error { throw error }
    if cancellationRequested() { throw DrawThingsCLIInvocationError.cancelled }
  }

  func beginModelDownload(name: String, subtitle: String, files: [String]) -> UUID {
    let id = UUID()
    cancellationLock.lock()
    modelDownloadID = id
    cancellationLock.unlock()
    modelDownloadEvent?(
      .started(
        id: id, name: name, subtitle: subtitle, files: files,
        cancel: { [weak self] in self?.cancel(.modelDownload(id)) }))
    return id
  }

  func finishModelDownload(_ id: UUID) {
    cancellationLock.lock()
    if modelDownloadID == id { modelDownloadID = nil }
    cancellationLock.unlock()
    modelDownloadEvent?(.finished(id: id))
  }

  func beginImageGeneration(
    name: String, version: ModelVersion, prompt: String, signposts: Set<ImageGeneratorSignpost>
  ) -> UUID {
    let id = UUID()
    cancellationLock.lock()
    imageGenerationID = id
    cancellationLock.unlock()
    imageGenerationEvent?(
      .started(
        id: id, name: name, version: version, prompt: prompt, signposts: signposts,
        cancel: { [weak self] in self?.cancel(.imageGeneration(id)) }))
    return id
  }

  func updateImageGeneration(
    signpost: ImageGeneratorSignpost, signposts: Set<ImageGeneratorSignpost>,
    preview: Tensor<FloatType>?
  ) {
    guard let imageGenerationEvent else { return }
    cancellationLock.lock()
    let id = imageGenerationID
    cancellationLock.unlock()
    guard let id else { return }
    imageGenerationEvent(
      .progress(id: id, signpost: signpost, signposts: signposts, preview: preview))
  }

  func beginImageGenerationSegment() {
    cancellationLock.lock()
    let id = imageGenerationID
    cancellationLock.unlock()
    guard let id else { return }
    imageGenerationEvent?(.segmentStarted(id: id))
  }

  func finishImageGeneration(_ id: UUID) {
    cancellationLock.lock()
    if imageGenerationID == id { imageGenerationID = nil }
    cancellationLock.unlock()
    imageGenerationEvent?(.finished(id: id))
  }

  func path(_ value: String) -> String { resolvePath(value) }

  func resolveModelsDirectoryAccess(_ requested: URL?) throws -> URL? {
    guard let resolveModelsDirectory else { return requested }
    if let modelsDirectory { return modelsDirectory }
    try checkCancellation()
    // Folder permission is asynchronous UI. Wait only on the command thread;
    // cancellation can return even if the picker is still being presented.
    let condition = NSCondition()
    var result: Result<URL, Error>?
    resolveModelsDirectory(requested) { value in
      condition.lock()
      result = value
      condition.broadcast()
      condition.unlock()
    }
    condition.lock()
    while result == nil && !isCancelled {
      condition.wait(until: Date().addingTimeInterval(0.1))
    }
    let resolved = result
    condition.unlock()
    try checkCancellation()
    guard let resolved else { throw DrawThingsCLIInvocationError.cancelled }
    let url = try resolved.get()
    #if canImport(Darwin)
      isAccessingModelsDirectory = url.startAccessingSecurityScopedResource()
    #endif
    modelsDirectory = url
    return url
  }

  func prepareForLocalModelExecution() throws {
    try checkCancellation()
    guard !didUnloadTextGenerator else { return }
    unloadTextGenerator?()
    didUnloadTextGenerator = true
    try checkCancellation()
  }

  func cloudAPIKey(prepareIfNeeded: Bool) throws -> String? {
    guard let accountProvider else { return nil }
    try checkCancellation()
    // Like the model-folder picker, wait on the command thread, never the UI thread.
    let condition = NSCondition()
    var result: Result<String?, Error>?
    accountProvider.resolveDrawThingsCredential(prepareIfNeeded: prepareIfNeeded) { value in
      condition.lock()
      result = value
      condition.broadcast()
      condition.unlock()
    }
    condition.lock()
    while result == nil && !isCancelled {
      condition.wait(until: Date().addingTimeInterval(0.1))
    }
    let resolved = result
    condition.unlock()
    try checkCancellation()
    guard let resolved else { throw DrawThingsCLIInvocationError.cancelled }
    return try resolved.get()
  }

  func handleCloudAuthenticationError(_ error: Error, hostAPIKey: String?) {
    if case CLICloudAuthError.insufficientFunds = error {
      print("Draw Things has insufficient funds. Complete the top-up, then retry generation.")
      if let hostAPIKey, !didRequestTopUp, !isCancelled {
        didRequestTopUp = true
        accountProvider?.drawThingsInsufficientFunds(apiKey: hostAPIKey)
      }
    } else {
      let message =
        hostAPIKey.map {
          error.localizedDescription.replacingOccurrences(of: $0, with: "<redacted>")
        } ?? error.localizedDescription
      print("[CloudAuth] \(message)")
    }
  }

  func cloudAuthentication(
    explicitAPIKey: String?, explicitBaseURL: String?, storedCredentials: CLICloudCredentials?,
    prepareIfNeeded: Bool = true
  ) throws -> (apiKey: String, baseURL: URL, hostAPIKey: String?) {
    // A host-owned key must only go to Draw Things, never an endpoint overridden
    // by shell arguments or old standalone CLI credentials.
    if hasCloudAccountHost && explicitAPIKey == nil && explicitBaseURL == nil {
      guard let key = try cloudAPIKey(prepareIfNeeded: prepareIfNeeded) else {
        throw CLICloudAuthError.authenticationFailed(
          "No Local Code Draw Things credential is available.")
      }
      return (key, CLICloudDefaultAPIBaseURL, key)
    }
    guard
      let key = CLICloudAuthClient.effectiveAPIKey(
        explicit: explicitAPIKey, storedCredentials: storedCredentials, environment: environment)
    else {
      throw CLICloudAuthError.authenticationFailed(
        "--cloud-compute requires --api-key, DRAWTHINGS_API_KEY, or saved credentials from `auth login`."
      )
    }
    let baseURL = try CLICloudAuthClient.resolvedAPIBaseURL(
      explicit: explicitBaseURL, storedCredentials: storedCredentials)
    return (key, baseURL, nil)
  }

  func finishInvocation() {
    #if canImport(Darwin)
      if isAccessingModelsDirectory { modelsDirectory?.stopAccessingSecurityScopedResource() }
    #endif
    modelsDirectory = nil
    isAccessingModelsDirectory = false
    didUnloadTextGenerator = false
    didRequestTopUp = false
  }

  func write(_ value: String, to destination: Output = .standardOutput) {
    let stream = destination == .standardOutput ? output : error
    let data = Data(value.utf8)
    data.withUnsafeBytes { bytes in
      if let address = bytes.baseAddress { _ = fwrite(address, 1, bytes.count, stream) }
    }
    fflush(stream)
  }

  func print(_ items: Any..., separator: String = " ", terminator: String = "\n") {
    write(items.map { String(describing: $0) }.joined(separator: separator) + terminator)
  }

  func readInput() throws -> Data {
    var data = Data()
    var bytes = [UInt8](repeating: 0, count: 4096)
    while true {
      try checkCancellation()
      var descriptor = pollfd(fd: fileno(input), events: Int16(POLLIN), revents: 0)
      let ready = poll(&descriptor, 1, 100)
      if ready == 0 { continue }
      if ready < 0 {
        if errno == EINTR { continue }
        throw DrawThingsCLIInvocationError.inputFailed
      }
      let count = read(fileno(input), &bytes, bytes.count)
      if count == 0 { return data }
      if count < 0 {
        if errno == EINTR { continue }
        throw DrawThingsCLIInvocationError.inputFailed
      }
      data.append(contentsOf: bytes.prefix(count))
    }
  }
}

enum DrawThingsCLIInvocationError: Error, LocalizedError {
  case cancelled
  case modelDownloadCancelled
  case generationAborted
  case inputFailed

  var errorDescription: String? {
    switch self {
    case .cancelled: return "Cancelled."
    case .modelDownloadCancelled: return "Download model cancelled by user"
    case .generationAborted: return "Generation aborted by user."
    case .inputFailed: return "Unable to read standard input."
    }
  }
}
