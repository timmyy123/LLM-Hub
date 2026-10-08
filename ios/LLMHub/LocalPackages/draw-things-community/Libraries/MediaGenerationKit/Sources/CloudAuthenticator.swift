import Foundation

// MARK: - Cloud Authenticator

/// Shared short-term token cache keyed externally by `CloudAuthenticatorRegistry`.
///
/// The new MediaGenerationKit cloud path only needs `/sdk/token`; per-request
/// authorization happens in `MediaGenerationCloudAuthentication`.
internal final class CloudAuthenticator {
  internal enum State {
    case idle
    case fetchingToken
    case authenticated(expiresAt: Date?)
    case failed(Error)
  }

  private let apiKey: String
  private let endpoint: URL
  private let tokenRefreshThreshold: TimeInterval
  private let stateHandler: ((State) -> Void)?
  private let requestTimeout: TimeInterval

  private var appCheck: AppCheckConfiguration
  private var appCheckRevision: UInt64 = 0
  private var cachedShortTermToken: String?
  private var tokenExpiry: Date?
  private var cachedRemoteModels = Set<String>()
  private var inFlightTokenRequests: [UInt64: InFlightTokenRequest] = [:]
  private let tokenQueue = DispatchQueue(label: "com.drawthings.mediagenerationkit.cloud-auth")

  private struct InFlightTokenRequest {
    let appCheck: AppCheckConfiguration
    var waiters: [(Result<String, Error>) -> Void]
  }

  private var sdkTokenEndpoint: URL {
    endpoint.appendingPathComponent("/sdk/token")
  }

  init(
    configuration: CloudConfiguration,
    stateHandler: ((State) -> Void)? = nil
  ) {
    self.apiKey = configuration.apiKey
    self.endpoint = configuration.baseURL
    self.tokenRefreshThreshold = configuration.tokenRefreshThreshold
    self.requestTimeout = configuration.requestTimeout
    self.appCheck = configuration.appCheck
    self.stateHandler = stateHandler
  }

  func getShortTermToken(
    appCheck: AppCheckConfiguration,
    completion: @escaping (Result<String, Error>) -> Void
  ) {
    tokenQueue.async { [weak self] in
      guard let self = self else {
        completion(.failure(CloudAuthenticatorError.invalidState))
        return
      }
      if self.appCheck != appCheck {
        self.appCheck = appCheck
        self.cachedShortTermToken = nil
        self.tokenExpiry = nil
        self.appCheckRevision &+= 1
        self.stateHandler?(.idle)
      }
      self.getShortTermTokenLocked(
        appCheck: appCheck,
        revision: self.appCheckRevision,
        completion: completion
      )
    }
  }

  func shortTermToken(appCheck: AppCheckConfiguration) async throws -> String {
    let bridge = MediaGenerationAsyncResultBridge<String>()
    return try await withTaskCancellationHandler(operation: {
      try await withCheckedThrowingContinuation { continuation in
        guard bridge.install(continuation) else { return }
        getShortTermToken(appCheck: appCheck) { result in
          bridge.resume(with: result)
        }
      }
    }, onCancel: {
      bridge.cancel()
    })
  }

  func updateRemoteModelsFromHandshake(_ models: some Sequence<String>) {
    tokenQueue.sync {
      cacheRemoteModels(models)
    }
  }

  func remoteModels() -> Set<String> {
    tokenQueue.sync {
      cachedRemoteModels
    }
  }

  private func cacheRemoteModels(_ models: some Sequence<String>) {
    cachedRemoteModels = Set(models)
  }

  private func getShortTermTokenLocked(
    appCheck: AppCheckConfiguration,
    revision: UInt64,
    completion: @escaping (Result<String, Error>) -> Void
  ) {
    if let token = cachedShortTermToken,
      let expiry = tokenExpiry,
      expiry.timeIntervalSinceNow > tokenRefreshThreshold
    {
      stateHandler?(.authenticated(expiresAt: expiry))
      completion(.success(token))
      return
    }

    if var inFlightRequest = inFlightTokenRequests[revision], inFlightRequest.appCheck == appCheck {
      inFlightRequest.waiters.append(completion)
      inFlightTokenRequests[revision] = inFlightRequest
      return
    }

    inFlightTokenRequests[revision] = InFlightTokenRequest(
      appCheck: appCheck,
      waiters: [completion]
    )
    stateHandler?(.fetchingToken)
    fetchShortTermToken(appCheck: appCheck, revision: revision)
  }

  private func fetchShortTermToken(
    appCheck: AppCheckConfiguration,
    revision: UInt64
  ) {
    var request = URLRequest(url: sdkTokenEndpoint)
    request.httpMethod = "POST"
    request.timeoutInterval = requestTimeout
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    struct TokenRequest: Codable {
      let apiKey: String
      let appCheckType: String
      let appCheckToken: String?
    }

    struct TokenResponse: Codable {
      let shortTermToken: String
      let expiresIn: Int
    }

    let requestBody = TokenRequest(
      apiKey: apiKey,
      appCheckType: appCheck.type,
      appCheckToken: appCheck.token
    )

    guard let body = try? JSONEncoder().encode(requestBody) else {
      let error = CloudAuthenticatorError.encodingFailed
      finishTokenRequest(revision: revision, appCheck: appCheck, result: .failure(error))
      return
    }
    request.httpBody = body

    URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
      guard let self = self else {
        return
      }

      let result: Result<String, Error>
      let expiresIn: Int?
      if let error = error {
        result = .failure(error)
        expiresIn = nil
      } else if let httpResponse = response as? HTTPURLResponse {
        guard httpResponse.statusCode == 200, let data = data else {
          result = .failure(CloudAuthenticatorError.serverError(httpResponse.statusCode))
          expiresIn = nil
          self.tokenQueue.async {
            self.finishTokenRequest(
              revision: revision,
              appCheck: appCheck,
              result: result,
              expiresIn: expiresIn
            )
          }
          return
        }
        do {
          let tokenResponse = try JSONDecoder().decode(TokenResponse.self, from: data)
          result = .success(tokenResponse.shortTermToken)
          expiresIn = tokenResponse.expiresIn
        } catch {
          result = .failure(CloudAuthenticatorError.decodingFailed)
          expiresIn = nil
        }
      } else {
        result = .failure(CloudAuthenticatorError.invalidResponse)
        expiresIn = nil
      }

      self.tokenQueue.async {
        self.finishTokenRequest(
          revision: revision,
          appCheck: appCheck,
          result: result,
          expiresIn: expiresIn
        )
      }
    }.resume()
  }

  private func finishTokenRequest(
    revision: UInt64,
    appCheck: AppCheckConfiguration,
    result: Result<String, Error>,
    expiresIn: Int? = nil
  ) {
    let waiters = inFlightTokenRequests.removeValue(forKey: revision)?.waiters ?? []

    switch result {
    case .success(let token):
      let expiresAt = expiresIn.map { Date().addingTimeInterval(TimeInterval($0)) }
      if appCheckRevision == revision && self.appCheck == appCheck {
        cachedShortTermToken = token
        tokenExpiry = expiresAt
        stateHandler?(.authenticated(expiresAt: expiresAt))
      }
    case .failure(let error):
      if appCheckRevision == revision && self.appCheck == appCheck {
        stateHandler?(.failed(error))
      }
    }

    waiters.forEach { $0(result) }
  }
}

// MARK: - Errors

enum CloudAuthenticatorError: Error, LocalizedError {
  case invalidState
  case encodingFailed
  case decodingFailed
  case invalidResponse
  case serverError(Int)

  var errorDescription: String? {
    switch self {
    case .invalidState:
      return "Authenticator is in an invalid state"
    case .encodingFailed:
      return "Failed to encode request"
    case .decodingFailed:
      return "Failed to decode response"
    case .invalidResponse:
      return "Invalid server response"
    case .serverError(let code):
      return "Server error: \(code)"
    }
  }
}
