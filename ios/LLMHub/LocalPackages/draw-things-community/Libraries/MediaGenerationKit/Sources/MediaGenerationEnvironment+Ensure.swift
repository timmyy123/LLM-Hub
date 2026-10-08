import CryptoKit
import Downloader
import Foundation
import GRPCServer
import ModelZoo

private final class MediaGenerationVerifiedSHACache {
  static let shared = MediaGenerationVerifiedSHACache()

  private struct State {
    var cache: [String: String] = [:]
    var cacheURL: URL?
  }

  private var state = ProtectedValue(State())

  private init() {}

  func configure(directory: URL) {
    let url = directory.appendingPathComponent(".verified_sha256.json")
    var shouldReload = false
    state.modify { state in
      guard state.cacheURL != url else { return }
      state.cacheURL = url
      state.cache = [:]
      shouldReload = true
    }
    guard shouldReload else { return }
    if let data = try? Data(contentsOf: url),
      let decoded = try? JSONDecoder().decode([String: String].self, from: data)
    {
      state.modify { state in
        if state.cacheURL == url {
          state.cache = decoded
        }
      }
    } else {
      state.modify { state in
        if state.cacheURL == url {
          state.cache = [:]
        }
      }
    }
  }

  func cached(for file: String) -> String? {
    var sha: String?
    state.modify { state in
      sha = state.cache[file]
    }
    return sha
  }

  func set(_ sha: String, for file: String) {
    var snapshot: [String: String] = [:]
    var cacheURL: URL?
    state.modify { state in
      state.cache[file] = sha
      snapshot = state.cache
      cacheURL = state.cacheURL
    }
    persist(cache: snapshot, cacheURL: cacheURL)
  }

  func remove(file: String) {
    var snapshot: [String: String] = [:]
    var cacheURL: URL?
    state.modify { state in
      state.cache.removeValue(forKey: file)
      snapshot = state.cache
      cacheURL = state.cacheURL
    }
    persist(cache: snapshot, cacheURL: cacheURL)
  }

  private func persist(cache: [String: String], cacheURL: URL?) {
    guard let cacheURL, let data = try? JSONEncoder().encode(cache) else { return }
    try? data.write(to: cacheURL, options: .atomic)
  }
}

struct MediaGenerationModelDownloadProgress {
  let file: String
  let fileIndex: Int
  let totalFiles: Int
  let bytesWritten: Int64
  let totalBytesExpected: Int64
}

extension MediaGenerationEnvironment.Storage {
  func ensureModelReady(
    file: String,
    includeDependencies: Bool = true,
    verification: ((String, Int, Int) -> Void)? = nil,
    progress: ((MediaGenerationModelDownloadProgress) -> Void)? = nil
  ) async throws {
    let modelsDirectory = try modelsDirectoryURL()
    let externalUrls = self.externalUrls
    ModelZoo.externalUrls = externalUrls

    let allFiles: [String]
    if let specification = ModelZoo.specificationForModel(file) {
      if specification.remoteApiModelConfig != nil {
        throw MediaGenerationKitError.modelNotFoundInCatalog(file)
      }
      if includeDependencies {
        var seen = Set<String>()
        allFiles = ModelZoo.filesToDownload(specification).map(\.file).filter { seen.insert($0).inserted }
      } else {
        allFiles = [specification.file]
      }
    } else {
      allFiles = [file]
    }

    guard !allFiles.isEmpty else { return }

    MediaGenerationVerifiedSHACache.shared.configure(directory: modelsDirectory)

    let largeSHAThresholdBytes: Int64 = 300 * 1024 * 1024
    var filesToDownload: [String] = []

    for (index, fileName) in allFiles.enumerated() {
      try Task.checkCancellation()
      verification?(fileName, index + 1, allFiles.count)
      if try Self.fileNeedsDownload(
        fileName,
        largeSHAThresholdBytes: largeSHAThresholdBytes
      ) {
        filesToDownload.append(fileName)
      }
    }

    for (index, fileName) in filesToDownload.enumerated() {
      try Task.checkCancellation()
      try await Self.downloadFile(
        fileName: fileName,
        fileIndex: index,
        totalFiles: filesToDownload.count,
        progress: progress
      )
      if let expectedSHA = ModelZoo.fileSHA256ForModelDownloaded(fileName) {
        MediaGenerationVerifiedSHACache.shared.set(expectedSHA, for: fileName)
      }
    }
  }

  private static func fileNeedsDownload(
    _ fileName: String,
    largeSHAThresholdBytes: Int64
  ) throws -> Bool {
    guard ModelZoo.isModelDownloaded(fileName) else {
      return true
    }

    let expectedSHA = ModelZoo.fileSHA256ForModelDownloaded(fileName)
    let filePath = ModelZoo.filePathForModelDownloaded(fileName)
    let fileURL = URL(fileURLWithPath: filePath)

    if let cachedSHA = MediaGenerationVerifiedSHACache.shared.cached(for: fileName) {
      if let expectedSHA {
        if cachedSHA == expectedSHA {
          return false
        }
      } else {
        return false
      }
    }

    let fileSizeBytes =
      (try? FileManager.default.attributesOfItem(atPath: filePath))?[.size] as? Int64 ?? 0
    if fileSizeBytes > largeSHAThresholdBytes {
      if let expectedSHA {
        MediaGenerationVerifiedSHACache.shared.set(expectedSHA, for: fileName)
      }
      return false
    }

    guard let actualSHA = try? sha256Hex(fileURL: fileURL) else {
      return false
    }

    if let expectedSHA {
      if actualSHA == expectedSHA {
        MediaGenerationVerifiedSHACache.shared.set(actualSHA, for: fileName)
        return false
      }
      try? FileManager.default.removeItem(at: fileURL)
      MediaGenerationVerifiedSHACache.shared.remove(file: fileName)
      return true
    }

    MediaGenerationVerifiedSHACache.shared.set(actualSHA, for: fileName)
    return false
  }

  private static func downloadFile(
    fileName: String,
    fileIndex: Int,
    totalFiles: Int,
    progress: ((MediaGenerationModelDownloadProgress) -> Void)?
  ) async throws {
    let localURL = URL(fileURLWithPath: ModelZoo.filePathForModelDownloaded(fileName))
    let remoteURL = URL(string: "https://static.libnnc.org/\(fileName)")!
    let expectedSHA = ModelZoo.fileSHA256ForModelDownloaded(fileName)
    try FileManager.default.createDirectory(
      at: localURL.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )

    let downloader = ResumableDownloader(
      remoteUrl: remoteURL,
      localUrl: localURL,
      sha256: expectedSHA
    )
    let bridge = MediaGenerationAsyncResultBridge<Void>()

    do {
      try await withTaskCancellationHandler(operation: {
        try await withCheckedThrowingContinuation { continuation in
          guard bridge.install(continuation) else { return }
          downloader.resume { totalBytesWritten, totalBytesExpectedToWrite, isComplete, error in
            if let error {
              bridge.resume(throwing: error)
              return
            }
            progress?(
              MediaGenerationModelDownloadProgress(
                file: fileName,
                fileIndex: fileIndex + 1,
                totalFiles: totalFiles,
                bytesWritten: totalBytesWritten,
                totalBytesExpected: totalBytesExpectedToWrite
              )
            )
            if isComplete {
              bridge.resume(returning: ())
            }
          }
        }
      }, onCancel: {
        bridge.cancel()
        DispatchQueue.global(qos: .userInitiated).async {
          downloader.cancel()
        }
      })
    } catch is CancellationError {
      throw CancellationError()
    } catch {
      if let downloaderError = error as? ResumableDownloader.DownloadError,
        case .fileMismatch = downloaderError
      {
        throw MediaGenerationKitError.hashMismatch(fileName)
      }
      let nsError = error as NSError
      if nsError.domain == NSCocoaErrorDomain && nsError.code == NSFileWriteOutOfSpaceError {
        throw MediaGenerationKitError.insufficientStorage
      }
      throw MediaGenerationKitError.downloadFailed(error.localizedDescription)
    }
  }

  private static func sha256Hex(fileURL: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: fileURL)
    defer { try? handle.close() }

    var hasher = SHA256()
    while true {
      let data = try handle.read(upToCount: 1024 * 1024) ?? Data()
      if data.isEmpty {
        break
      }
      hasher.update(data: data)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
