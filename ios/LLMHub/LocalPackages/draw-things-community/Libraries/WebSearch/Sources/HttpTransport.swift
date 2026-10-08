import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Minimal HTTP transport used by search and fetch tools.
public protocol HttpTransport {
  /// Performs a request and calls `completion` with response body data plus the HTTP response.
  func data(
    for request: URLRequest,
    completion: @escaping (Result<(Data, HTTPURLResponse), Error>) -> Void)
}

extension HttpTransport {
  /// Performs a request with async/await by wrapping the completion-handler API.
  public func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
    try await withCheckedThrowingContinuation { continuation in
      data(for: request) { result in
        continuation.resume(with: result)
      }
    }
  }
}

/// `URLSession`-backed HTTP transport.
public struct URLSessionHttpTransport: HttpTransport {
  private static let defaultSession: URLSession = {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.urlCache = nil
    configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
    configuration.httpCookieStorage = nil
    configuration.httpShouldSetCookies = false
    configuration.urlCredentialStorage = nil
    return URLSession(configuration: configuration)
  }()

  private let session: URLSession

  /// Creates a transport backed by a shared, uncached ephemeral session.
  public init() {
    session = Self.defaultSession
  }

  /// Creates a transport backed by `session`.
  public init(session: URLSession) {
    self.session = session
  }

  /// Performs a request with `URLSession`.
  public func data(
    for request: URLRequest,
    completion: @escaping (Result<(Data, HTTPURLResponse), Error>) -> Void
  ) {
    session.dataTask(with: request) { data, response, error in
      if let error {
        completion(.failure(error))
        return
      }
      guard let data, let httpResponse = response as? HTTPURLResponse else {
        completion(.failure(WebSearchError.invalidResponse))
        return
      }
      completion(.success((data, httpResponse)))
    }
    .resume()
  }
}
