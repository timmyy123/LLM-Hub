import DataModels
import Foundation
import ModelZoo

/// Process-scoped environment helpers for model resolution, downloads, and local execution settings.
///
/// Most applications use ``default`` and only override `externalUrls` or
/// `maxTotalWeightsCacheSize` when they need custom model search paths or cache sizing.
public struct MediaGenerationEnvironment: Sendable {
  /// Progress states emitted by ``ensure(_:offline:stateHandler:)``.
  public enum EnsureState: Sendable, Equatable {
    case resolving
    case verifying(file: String, fileIndex: Int, totalFiles: Int)
    case downloading(
      file: String,
      fileIndex: Int,
      totalFiles: Int,
      bytesWritten: Int64,
      totalBytesExpected: Int64
    )
  }

  internal final class Storage: @unchecked Sendable {
    private let lock = NSLock()
    private var currentExternalURLs: [URL]
    private var cachedLocalResources: MediaGenerationLocalResources?

    init(externalUrls: [URL]) {
      self.currentExternalURLs = externalUrls
      ModelZoo.externalUrls = externalUrls
    }

    var externalUrls: [URL] {
      get {
        lock.lock()
        defer { lock.unlock() }
        return currentExternalURLs
      }
      set {
        lock.lock()
        currentExternalURLs = newValue
        lock.unlock()
        ModelZoo.externalUrls = newValue
      }
    }

    func localResources() -> MediaGenerationLocalResources {
      lock.lock()
      if let cached = cachedLocalResources, FileManager.default.fileExists(atPath: cached.tempDir) {
        lock.unlock()
        DeviceCapability.cacheUri = URL(fileURLWithPath: cached.tempDir)
        return cached
      }
      let localResources = MediaGenerationExecutionUtilities.createLocalResources()
      self.cachedLocalResources = localResources
      lock.unlock()

      DeviceCapability.cacheUri = URL(fileURLWithPath: localResources.tempDir)
      return localResources
    }

    var maxTotalWeightsCacheSize: UInt64 {
      get {
        DeviceCapability.maxTotalWeightsCacheSize
      }
      set {
        DeviceCapability.maxTotalWeightsCacheSize = newValue
      }
    }

    func ensure(
      _ model: String,
      offline: Bool,
      stateHandler: (@Sendable (EnsureState) -> Void)?
    ) async throws -> ModelResolver.Model {
      guard let resolvedModel = try await ModelResolver.resolve(model, offline: offline) else {
        throw MediaGenerationKitError.unresolvedModelReference(
          query: model,
          suggestions: await ModelResolver.suggestions(model, limit: 5, offline: offline).map(\.file)
        )
      }

      try Task.checkCancellation()
      stateHandler?(.resolving)
      try await ensureModelReady(
        file: resolvedModel.file,
        includeDependencies: true,
        verification: { file, fileIndex, totalFiles in
          stateHandler?(
            .verifying(
              file: file,
              fileIndex: fileIndex,
              totalFiles: totalFiles
            )
          )
        },
        progress: { progress in
          stateHandler?(
            .downloading(
              file: progress.file,
              fileIndex: progress.fileIndex,
              totalFiles: progress.totalFiles,
              bytesWritten: progress.bytesWritten,
              totalBytesExpected: progress.totalBytesExpected
            )
          )
        }
      )
      return resolvedModel
    }

    func modelsDirectoryURL() throws -> URL {
      let externalUrls = self.externalUrls
      guard let primaryURL = externalUrls.first else {
        throw MediaGenerationKitError.invalidModelsDirectory
      }
      return try MediaGenerationDefaults.ensureDirectory(primaryURL)
    }

    func resolveModel(_ model: String, offline: Bool) throws -> MediaGenerationResolvedModel? {
      try ModelResolver.resolve(model, offline: offline, operation: "resolveModel")
    }

    func resolveModel(_ model: String, offline: Bool) async -> MediaGenerationResolvedModel? {
      try? await ModelResolver.resolve(model, offline: offline)
    }

    func suggestions(for model: String, limit: Int, offline: Bool) throws
      -> [MediaGenerationResolvedModel]
    {
      try ModelResolver.suggestions(model, limit: limit, offline: offline)
    }

    func suggestions(for model: String, limit: Int, offline: Bool) async -> [MediaGenerationResolvedModel]
    {
      await ModelResolver.suggestions(model, limit: limit, offline: offline)
    }

    func inspect(_ model: String, offline: Bool) throws -> MediaGenerationResolvedModel {
      guard
        let specification = try ModelResolver.specification(
          for: model,
          offline: offline,
          operation: "inspectModel"
        )
      else {
        throw MediaGenerationKitError.unresolvedModelReference(
          query: model,
          suggestions: (try? ModelResolver.suggestions(model, limit: 5, offline: offline).map(\.file))
            ?? []
        )
      }

      return ModelResolver.model(from: specification)
    }

    func inspect(_ model: String, offline: Bool) async throws -> MediaGenerationResolvedModel {
      guard let specification = await ModelResolver.specification(for: model, offline: offline) else {
        throw MediaGenerationKitError.unresolvedModelReference(
          query: model,
          suggestions: await ModelResolver.suggestions(model, limit: 5, offline: offline).map(\.file)
        )
      }

      return ModelResolver.model(from: specification)
    }

    func downloadableModels(includeDownloaded: Bool, offline: Bool) throws
      -> [MediaGenerationResolvedModel]
    {
      try ModelResolver.catalogModelsSynchronouslyIfAvailable(
        includeDownloaded: includeDownloaded,
        offline: offline
      )
    }

    func downloadableModels(includeDownloaded: Bool, offline: Bool) async -> [MediaGenerationResolvedModel]
    {
      await ModelResolver.catalogModels(includeDownloaded: includeDownloaded, offline: offline)
    }
  }

  /// Shared process-scoped environment.
  public static var `default` = MediaGenerationEnvironment(
    storage: Storage(externalUrls: MediaGenerationDefaults.defaultExternalURLs())
  )

  internal let storage: Storage

  /// Model search roots used for local model resolution.
  public var externalUrls: [URL] {
    get {
      storage.externalUrls
    }
    nonmutating set {
      storage.externalUrls = newValue
    }
  }

  /// Maximum total weights-cache size in bytes.
  public var maxTotalWeightsCacheSize: UInt64 {
    get {
      storage.maxTotalWeightsCacheSize
    }
    nonmutating set {
      storage.maxTotalWeightsCacheSize = newValue
    }
  }

  /// Resolves a model reference and downloads the files needed for local execution.
  public func ensure(
    _ model: String,
    offline: Bool = false,
    stateHandler: (@Sendable (EnsureState) -> Void)? = nil
  ) async throws -> MediaGenerationResolvedModel {
    try await storage.ensure(model, offline: offline, stateHandler: stateHandler)
  }

  /// Resolves a model reference synchronously using offline or already-cached catalog data.
  public func resolveModel(
    _ model: String,
    offline: Bool = true
  ) throws -> MediaGenerationResolvedModel? {
    try storage.resolveModel(model, offline: offline)
  }

  /// Resolves a model reference, allowing remote catalog fetches when `offline` is `false`.
  public func resolveModel(
    _ model: String,
    offline: Bool = false
  ) async -> MediaGenerationResolvedModel? {
    await storage.resolveModel(model, offline: offline)
  }

  /// Returns close model matches synchronously using offline or already-cached catalog data.
  public func suggestedModels(
    for model: String,
    limit: Int = 5,
    offline: Bool = true
  ) throws -> [MediaGenerationResolvedModel] {
    try storage.suggestions(for: model, limit: limit, offline: offline)
  }

  /// Returns close model matches, allowing remote catalog fetches when `offline` is `false`.
  public func suggestedModels(
    for model: String,
    limit: Int = 5,
    offline: Bool = false
  ) async -> [MediaGenerationResolvedModel] {
    await storage.suggestions(for: model, limit: limit, offline: offline)
  }

  /// Returns detailed information for a resolved model using offline or already-cached catalog data.
  public func inspectModel(
    _ model: String,
    offline: Bool = true
  ) throws -> MediaGenerationResolvedModel {
    return try storage.inspect(model, offline: offline)
  }

  /// Returns detailed information for a resolved model, allowing remote catalog fetches when `offline` is `false`.
  public func inspectModel(
    _ model: String,
    offline: Bool = false
  ) async throws -> MediaGenerationResolvedModel {
    try await storage.inspect(model, offline: offline)
  }

  /// Lists downloadable models synchronously using offline or already-cached catalog data.
  public func downloadableModels(
    includeDownloaded: Bool = true,
    offline: Bool = true
  ) throws -> [MediaGenerationResolvedModel] {
    try storage.downloadableModels(includeDownloaded: includeDownloaded, offline: offline)
  }

  /// Lists downloadable models, allowing remote catalog fetches when `offline` is `false`.
  public func downloadableModels(
    includeDownloaded: Bool = true,
    offline: Bool = false
  ) async -> [MediaGenerationResolvedModel] {
    await storage.downloadableModels(includeDownloaded: includeDownloaded, offline: offline)
  }

  internal init(storage: Storage) {
    self.storage = storage
  }

  internal static func local(_ directory: String?) throws -> MediaGenerationEnvironment {
    if let directory, !directory.isEmpty {
      let url = try MediaGenerationDefaults.ensureDirectory(
        URL(fileURLWithPath: directory, isDirectory: true)
      )
      return MediaGenerationEnvironment(storage: Storage(externalUrls: [url]))
    }
    return .default
  }
}

internal enum MediaGenerationDefaults {
  private static let modelsDirectoryInfoKeys = [
    "MediaGenerationKitModelsDirectory",
    "DrawThingsModelsDirectory",
  ]
  private static let apiKeyInfoKeys = [
    "MediaGenerationKitAPIKey",
    "DrawThingsAPIKey",
  ]

  static func defaultExternalURLs() -> [URL] {
    if let envPath = ProcessInfo.processInfo.environment["DRAWTHINGS_MODELS_DIR"], !envPath.isEmpty
    {
      return [URL(fileURLWithPath: envPath, isDirectory: true)]
    }
    if let bundlePath = infoPlistValue(for: modelsDirectoryInfoKeys) {
      return [URL(fileURLWithPath: bundlePath, isDirectory: true)]
    }
    if let documentsURL = FileManager.default.urls(
      for: .documentDirectory, in: .userDomainMask
    ).first
    {
      return [documentsURL.appendingPathComponent("Models", isDirectory: true)]
    }
    return []
  }

  static func resolveAPIKey(explicitAPIKey: String?) throws -> String {
    if let explicitAPIKey, !explicitAPIKey.isEmpty {
      return explicitAPIKey
    }
    if let envValue = ProcessInfo.processInfo.environment["DRAWTHINGS_API_KEY"], !envValue.isEmpty {
      return envValue
    }
    if let bundleValue = infoPlistValue(for: apiKeyInfoKeys) {
      return bundleValue
    }
    throw MediaGenerationKitError.notConfigured
  }

  static func ensureDirectory(_ url: URL) throws -> URL {
    let normalized = url.standardizedFileURL
    var isDirectory: ObjCBool = false
    let exists = FileManager.default.fileExists(atPath: normalized.path, isDirectory: &isDirectory)
    if exists && !isDirectory.boolValue {
      throw MediaGenerationKitError.invalidModelsDirectory
    }
    if !exists {
      try FileManager.default.createDirectory(
        at: normalized,
        withIntermediateDirectories: true
      )
    }
    return normalized
  }

  private static func infoPlistValue(for keys: [String]) -> String? {
    guard let dictionary = Bundle.main.infoDictionary else {
      return nil
    }
    for key in keys {
      if let value = dictionary[key] as? String, !value.isEmpty {
        return value
      }
    }
    return nil
  }
}
