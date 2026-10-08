import Crypto
import Foundation

/// Result returned after a successful LoRA upload.
public struct LoRAUploadResult {
  public let file: String
  public let sha256: String
}

public enum LoRAUploadError: Error, LocalizedError {
  case storageLimitExceeded
  case serverError(Int)
  case networkError(Error)
  case uploadFailed
  case invalidRequest

  public var errorDescription: String? {
    switch self {
    case .storageLimitExceeded:
      return "DrawThings cloud storage limit exceeded"
    case .serverError(let code):
      return "Server error (HTTP \(code))"
    case .networkError(let error):
      return "Network error: \(error.localizedDescription)"
    case .uploadFailed:
      return "PUT to signed URL failed"
    case .invalidRequest:
      return "Failed to encode LoRA upload request"
    }
  }
}

/// Manages LoRA files stored on Draw Things cloud compute.
///
/// This is separate from ``LoRAImporter``: importing converts a local file
/// into Draw Things format, while `LoRAStore` manages remote stored LoRA files.
public struct LoRAStore: Sendable {
  /// Remote LoRA file metadata.
  public struct File: Sendable, Hashable {
    public let key: String
    public let file: String
    public let sha256: String
    public let size: Int64

    public init(key: String, file: String, sha256: String, size: Int64) {
      self.key = key
      self.file = file
      self.sha256 = sha256
      self.size = size
    }
  }

  private final class Storage: @unchecked Sendable {
    let authenticator: CloudAuthenticator
    let baseURL: URL
    let appCheck: AppCheckConfiguration

    init(authenticator: CloudAuthenticator, baseURL: URL, appCheck: AppCheckConfiguration) {
      self.authenticator = authenticator
      self.baseURL = baseURL
      self.appCheck = appCheck
    }
  }

  private let storage: Storage

  /// Initializes remote LoRA storage access from a cloud-capable backend.
  /// Non-cloud backends should fail during initialization.
  public init(backend: MediaGenerationPipeline.Backend) throws {
    guard case .cloudCompute(let apiKey, let options) = backend else {
      throw MediaGenerationKitError.generationFailed(
        "LoRAStore requires a cloudCompute backend")
    }
    let resolvedAPIKey = try MediaGenerationDefaults.resolveAPIKey(explicitAPIKey: apiKey)
    let baseURL = options.baseURL ?? CloudConfiguration.defaultBaseURL
    self.storage = Storage(
      authenticator: CloudAuthenticatorRegistry.shared.authenticator(
        apiKey: resolvedAPIKey,
        baseURL: baseURL
      ),
      baseURL: baseURL,
      appCheck: options.appCheck
    )
  }

  /// Lists remote LoRA files.
  ///
  /// The current public cloud API does not expose a list endpoint, so this method throws.
  public func list() async throws -> [File] {
    throw MediaGenerationKitError.generationFailed(
      "LoRAStore.list() is not implemented because the current cloud API does not expose a list endpoint")
  }

  /// Upload raw LoRA bytes to the store.
  ///
  /// `Data` is the primary API rather than `URL` so the store does not own
  /// filesystem reads. The current cloud upload endpoint returns the stored
  /// file name and SHA256, but not a delete key, so the returned `File.key`
  /// is empty until the server surface is expanded.
  public func upload(_ data: Data, file: String) async throws -> File {
    let uploadDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mediagenerationkit-upload-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: uploadDirectory, withIntermediateDirectories: true)
    let fileURL = uploadDirectory.appendingPathComponent(file)
    try data.write(to: fileURL, options: .atomic)
    defer {
      try? FileManager.default.removeItem(at: uploadDirectory)
    }
    let uploaded = try await uploadFile(at: fileURL)
    return File(
      key: "",
      file: uploaded.file,
      sha256: uploaded.sha256,
      size: Int64(data.count)
    )
  }

  /// Deletes remote LoRA files using keys returned by the cloud API.
  public func delete(_ files: [File]) async throws {
    try await delete(keys: files.map(\.key))
  }

  /// Deletes remote LoRA files by key.
  public func delete(keys: [String]) async throws {
    let filteredKeys = keys.filter { !$0.isEmpty }
    guard filteredKeys.count == keys.count else {
      throw MediaGenerationKitError.generationFailed(
        "LoRAStore.delete(keys:) requires cloud file keys from a listing response")
    }
    guard !filteredKeys.isEmpty else {
      return
    }

    struct DeleteFilesRequest: Codable {
      let keys: [String]
    }

    let token = try await shortTermToken()
    var request = URLRequest(
      url: storage.baseURL.appendingPathComponent("/delete_uploaded_files")
    )
    request.httpMethod = "POST"
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")
    request.addValue(token, forHTTPHeaderField: "Authorization")
    request.httpBody = try JSONEncoder().encode(DeleteFilesRequest(keys: filteredKeys))

    let (_, response) = try await URLSession.shared.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse else {
      throw MediaGenerationKitError.generationFailed("LoRA delete failed: missing HTTP response")
    }
    guard httpResponse.statusCode == 200 else {
      throw MediaGenerationKitError.generationFailed(
        "LoRA delete failed with status code \(httpResponse.statusCode)")
    }
  }

  private func shortTermToken() async throws -> String {
    try await withCheckedThrowingContinuation { continuation in
      storage.authenticator.getShortTermToken(appCheck: storage.appCheck) { result in
        continuation.resume(with: result)
      }
    }
  }

  private func uploadFile(at fileURL: URL) async throws -> LoRAUploadResult {
    let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
    let sha256 = SHA256.hash(data: data).compactMap { String(format: "%02x", $0) }.joined()
    let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
    let fileSize = (attrs[.size] as? Int64) ?? Int64(data.count)
    let filename = fileURL.lastPathComponent
    let token = try await shortTermToken()
    return try await withCheckedThrowingContinuation { continuation in
      postGetSignedURLs(
        baseURL: storage.baseURL,
        filename: filename,
        sha256: sha256,
        size: fileSize,
        fileURL: fileURL,
        token: token
      ) { result in
        continuation.resume(with: result)
      }
    }
  }
}

private struct LoRAUploadRequest: Encodable {
  struct FileMetadata: Encodable {
    var file: String
    var sha256: String
    var size: Int64
  }

  var files: [FileMetadata]
}

private struct LoRAUploadResponse: Decodable {
  struct SignedURLsPayload: Decodable {
    struct FileInfo: Decodable {
      var sha256: String
      var url: String
    }

    var urls: [String: FileInfo]
  }

  struct StorageLimitPayload: Decodable {
    var freeSpace: Int64
  }

  struct DataPayload: Decodable {
    var signedUrls: SignedURLsPayload?
    var alreadyExists: [String: String]?
    var storageLimitExceeded: StorageLimitPayload?
  }

  var code: UInt
  var data: DataPayload
}

private func postGetSignedURLs(
  baseURL: URL,
  filename: String,
  sha256: String,
  size: Int64,
  fileURL: URL,
  token: String,
  completion: @escaping (Result<LoRAUploadResult, LoRAUploadError>) -> Void
) {
  let endpoint = baseURL.appendingPathComponent("/get_byom_urls")
  var request = URLRequest(url: endpoint)
  request.httpMethod = "POST"
  request.addValue("application/json", forHTTPHeaderField: "Content-Type")
  request.addValue(token, forHTTPHeaderField: "Authorization")

  let body = LoRAUploadRequest(files: [.init(file: filename, sha256: sha256, size: size)])
  guard let jsonData = try? JSONEncoder().encode(body) else {
    completion(.failure(.invalidRequest))
    return
  }
  request.httpBody = jsonData

  URLSession.shared.dataTask(with: request) { data, response, error in
    if let error = error {
      completion(.failure(.networkError(error)))
      return
    }
    guard let http = response as? HTTPURLResponse else {
      completion(.failure(.serverError(0)))
      return
    }
    guard http.statusCode == 200, let data = data else {
      completion(.failure(.serverError(http.statusCode)))
      return
    }
    guard let decoded = try? JSONDecoder().decode(LoRAUploadResponse.self, from: data) else {
      completion(.failure(.serverError(http.statusCode)))
      return
    }

    if let signedURLsPayload = decoded.data.signedUrls {
      guard let fileInfo = signedURLsPayload.urls[filename],
        let signedURL = URL(string: fileInfo.url)
      else {
        completion(.failure(.uploadFailed))
        return
      }
      putFile(
        fileURL: fileURL,
        to: signedURL,
        filename: filename,
        serverSha256: fileInfo.sha256,
        completion: completion
      )
    } else if let alreadyExists = decoded.data.alreadyExists {
      let serverSha256 = alreadyExists[sha256] ?? sha256
      completion(.success(LoRAUploadResult(file: filename, sha256: serverSha256)))
    } else {
      completion(.failure(.storageLimitExceeded))
    }
  }.resume()
}

private func putFile(
  fileURL: URL,
  to signedURL: URL,
  filename: String,
  serverSha256: String,
  completion: @escaping (Result<LoRAUploadResult, LoRAUploadError>) -> Void
) {
  var request = URLRequest(url: signedURL)
  request.httpMethod = "PUT"
  URLSession.shared.uploadTask(with: request, fromFile: fileURL) { _, response, error in
    if let error = error {
      completion(.failure(.networkError(error)))
      return
    }
    guard let http = response as? HTTPURLResponse else {
      completion(.failure(.serverError(0)))
      return
    }
    guard (200...299).contains(http.statusCode) else {
      completion(.failure(.uploadFailed))
      return
    }
    completion(.success(LoRAUploadResult(file: filename, sha256: serverSha256)))
  }.resume()
}
