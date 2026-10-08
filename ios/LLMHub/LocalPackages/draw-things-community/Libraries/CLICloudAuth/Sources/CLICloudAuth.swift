import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif
#if canImport(Network)
  import Network
#endif

public let CLICloudDefaultAPIBaseURL = URL(string: "https://api.drawthings.ai")!

public struct CLICloudCredentials: Codable {
  public let provider: String
  public let apiKey: String
  public let apiBaseURL: String?
  public let savedAt: Date

  public init(provider: String, apiKey: String, apiBaseURL: String?, savedAt: Date) {
    self.provider = provider
    self.apiKey = apiKey
    self.apiBaseURL = apiBaseURL
    self.savedAt = savedAt
  }
}

public struct CLICloudCredentialsStore {
  private let applicationName: String

  public init(applicationName: String) {
    self.applicationName = applicationName
  }

  public var credentialsURL: URL {
    credentialsDirectoryURL.appendingPathComponent("cloud-credentials.json", isDirectory: false)
  }

  public var credentialsPath: String {
    credentialsURL.path
  }

  public func load() -> CLICloudCredentials? {
    let url = credentialsURL
    guard FileManager.default.fileExists(atPath: url.path),
      let data = try? Data(contentsOf: url),
      let credentials = try? JSONDecoder().decode(CLICloudCredentials.self, from: data)
    else {
      return nil
    }
    return credentials
  }

  public func save(_ credentials: CLICloudCredentials) throws {
    let url = credentialsURL
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(credentials)
    try data.write(to: url, options: .atomic)
    try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
  }

  public func remove() throws {
    let url = credentialsURL
    guard FileManager.default.fileExists(atPath: url.path) else { return }
    try FileManager.default.removeItem(at: url)
  }

  private var credentialsDirectoryURL: URL {
    let home = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
    #if os(macOS)
      return
        home
        .appendingPathComponent("Library", isDirectory: true)
        .appendingPathComponent("Application Support", isDirectory: true)
        .appendingPathComponent(applicationName, isDirectory: true)
    #else
      return
        home
        .appendingPathComponent(".config", isDirectory: true)
        .appendingPathComponent(applicationName, isDirectory: true)
    #endif
  }
}

public enum CLICloudAuthError: LocalizedError {
  case invalidAPIBaseURL(String)
  case savedInvalidAPIBaseURL(String)
  case authenticationFailed(String)
  case insufficientFunds
  case operationTimedOut(String)
  case unsupportedGoogleLoginPlatform
  case listenerFailed(String)
  case listenerTimedOut
  case callbackFailed(String)
  case startFailed(String)
  case browserLaunchFailed(String)

  public var errorDescription: String? {
    switch self {
    case .invalidAPIBaseURL(let value):
      return "Invalid cloud API base URL: \(value)"
    case .savedInvalidAPIBaseURL(let value):
      return "Saved credentials contain an invalid cloud API base URL: \(value)"
    case .authenticationFailed(let message):
      return message
    case .insufficientFunds:
      return "Insufficient Draw Things funds. Add PAYG credit and retry generation."
    case .operationTimedOut(let message):
      return message
    case .unsupportedGoogleLoginPlatform:
      return "Google browser login is not supported on this platform."
    case .listenerFailed(let message):
      return "Failed to start local OAuth callback listener: \(message)"
    case .listenerTimedOut:
      return "Timed out waiting for Google OAuth callback."
    case .callbackFailed(let message):
      return "Google OAuth callback failed: \(message)"
    case .startFailed(let message):
      return "Failed to start Google login: \(message)"
    case .browserLaunchFailed(let message):
      return "Failed to launch browser: \(message)"
    }
  }
}

public enum CLICloudAuthState {
  case idle
  case fetchingToken
  case authenticated(expiresAt: Date?)
  case failed(Error)
}

public func describeCLICloudAuthState(_ state: CLICloudAuthState) -> String {
  switch state {
  case .idle:
    return "idle"
  case .fetchingToken:
    return "fetchingToken"
  case .authenticated(let expiresAt):
    if let expiresAt {
      return "authenticated(expiresAt: \(expiresAt))"
    }
    return "authenticated(expiresAt: nil)"
  case .failed(let error):
    return "failed(\(error))"
  }
}

public struct CLICloudShortTermToken {
  public let value: String
  public let expiresIn: Int

  public init(value: String, expiresIn: Int) {
    self.value = value
    self.expiresIn = expiresIn
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

private func waitForSemaphore(_ semaphore: DispatchSemaphore, timeout: TimeInterval) -> Bool {
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
    throw CLICloudAuthError.operationTimedOut("Operation timed out.")
  }
  guard let result = resultBox.load() else {
    throw CLICloudAuthError.authenticationFailed("Operation completed without a result.")
  }
  return try result.get()
}

public struct CLICloudAuthClient {
  public let baseURL: URL

  private let session: URLSession

  public init(baseURL: URL = CLICloudDefaultAPIBaseURL, session: URLSession = .shared) {
    self.baseURL = baseURL
    self.session = session
  }

  public static func apiURL(baseURL: URL, path: String) -> URL {
    var url = baseURL
    for component in path.split(separator: "/") {
      url.appendPathComponent(String(component))
    }
    return url
  }

  public static func resolvedAPIBaseURL(
    explicit: String?,
    storedCredentials: CLICloudCredentials?
  ) throws -> URL {
    if let explicit, !explicit.isEmpty {
      guard let parsedURL = URL(string: explicit) else {
        throw CLICloudAuthError.invalidAPIBaseURL(explicit)
      }
      return parsedURL
    }
    if let storedAPIBaseURL = storedCredentials?.apiBaseURL, !storedAPIBaseURL.isEmpty {
      guard let parsedURL = URL(string: storedAPIBaseURL) else {
        throw CLICloudAuthError.savedInvalidAPIBaseURL(storedAPIBaseURL)
      }
      return parsedURL
    }
    return CLICloudDefaultAPIBaseURL
  }

  public static func effectiveAPIKey(
    explicit: String?,
    storedCredentials: CLICloudCredentials?,
    environment: [String: String] = ProcessInfo.processInfo.environment
  ) -> String? {
    if let explicit, !explicit.isEmpty {
      return explicit
    }
    if let storedKey = storedCredentials?.apiKey, !storedKey.isEmpty {
      return storedKey
    }
    let env = environment["DRAWTHINGS_API_KEY"]
    return env?.isEmpty == false ? env : nil
  }

  public func fetchShortTermToken(
    apiKey: String,
    emitStates: Bool = false,
    stateHandler: ((CLICloudAuthState) -> Void)? = nil
  ) throws -> CLICloudShortTermToken {
    if emitStates {
      stateHandler?(.idle)
      stateHandler?(.fetchingToken)
    }

    struct TokenRequest: Codable {
      let apiKey: String
      let appCheckType: String
      let appCheckToken: String?
    }

    struct TokenResponse: Codable {
      let shortTermToken: String
      let expiresIn: Int
    }

    let requestBody = try JSONEncoder().encode(
      TokenRequest(apiKey: apiKey, appCheckType: "none", appCheckToken: nil)
    )
    let request: URLRequest = {
      var request = URLRequest(url: Self.apiURL(baseURL: baseURL, path: "/sdk/token"))
      request.httpMethod = "POST"
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
      request.httpBody = requestBody
      return request
    }()

    do {
      let tokenResponse = try runAsync(timeout: 30) {
        let (data, response) = try await self.session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
          throw CLICloudAuthError.authenticationFailed(
            "Authentication failed: missing HTTP response.")
        }
        if httpResponse.statusCode == 402 { throw CLICloudAuthError.insufficientFunds }
        guard httpResponse.statusCode == 200 else {
          throw CLICloudAuthError.authenticationFailed(
            "Authentication failed with status code \(httpResponse.statusCode).")
        }
        return try JSONDecoder().decode(TokenResponse.self, from: data)
      }
      if emitStates {
        let expiresAt = Date().addingTimeInterval(TimeInterval(tokenResponse.expiresIn))
        stateHandler?(.authenticated(expiresAt: expiresAt))
      }
      return CLICloudShortTermToken(
        value: tokenResponse.shortTermToken,
        expiresIn: tokenResponse.expiresIn
      )
    } catch {
      if emitStates {
        stateHandler?(.failed(error))
      }
      throw error
    }
  }

  public func authenticateGenerationRequest(
    shortTermToken: String,
    encodedBlob: Data,
    fromBridge: Bool,
    estimatedComputeUnits: Double?,
    cancellation: (@escaping () -> Void) -> Void
  ) throws -> String? {
    struct AuthenticationRequest: Codable {
      let blob: String
      let fromBridge: Bool
      let attestationSupported: Bool
      let assertionPayload: String?
      let originalTransactionId: String?
      let isSandbox: Bool
      let consumableType: String?
      let amount: Double?
    }

    struct AuthenticationResponse: Codable {
      let gRPCToken: String
    }

    let amount = estimatedComputeUnits.flatMap { $0 > 0 ? $0 : nil }
    let paygEnabled =
      amount != nil
      ? fetchPaygEnabled(shortTermToken: shortTermToken)
      : false

    var request = URLRequest(url: Self.apiURL(baseURL: baseURL, path: "/authenticate"))
    request.httpMethod = "POST"
    request.timeoutInterval = 30
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")
    request.addValue(shortTermToken, forHTTPHeaderField: "Authorization")
    request.httpBody = try JSONEncoder().encode(
      AuthenticationRequest(
        blob: encodedBlob.base64EncodedString(),
        fromBridge: fromBridge,
        attestationSupported: false,
        assertionPayload: nil,
        originalTransactionId: nil,
        isSandbox: false,
        consumableType: paygEnabled ? "payg" : nil,
        amount: amount
      ))

    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<String?, Error> = .success(nil)
    let task = session.dataTask(with: request) { data, response, error in
      defer { semaphore.signal() }
      if let error {
        result = .failure(error)
        return
      }
      guard let httpResponse = response as? HTTPURLResponse else {
        result = .failure(
          CLICloudAuthError.authenticationFailed(
            "Cloud authentication failed: missing HTTP response."))
        return
      }
      if httpResponse.statusCode == 402 {
        result = .failure(CLICloudAuthError.insufficientFunds)
        return
      }
      guard httpResponse.statusCode == 200, let data else {
        let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
        result = .failure(
          CLICloudAuthError.authenticationFailed(
            "Cloud authentication failed with status \(httpResponse.statusCode). \(body)")
        )
        return
      }
      do {
        result = .success(
          try JSONDecoder().decode(AuthenticationResponse.self, from: data).gRPCToken)
      } catch {
        result = .failure(error)
      }
    }
    cancellation {
      task.cancel()
    }
    task.resume()
    guard semaphore.wait(timeout: .now() + 30) == .success else {
      task.cancel()
      throw CLICloudAuthError.operationTimedOut("Cloud authentication timed out.")
    }
    return try result.get()
  }

  public func fetchPaygEnabled(shortTermToken: String) -> Bool {
    struct PaygStatusResponse: Codable {
      let paygEnabled: Bool
      let paygEligible: Bool
    }

    var request = URLRequest(url: Self.apiURL(baseURL: baseURL, path: "/billing/stripe/payg"))
    request.httpMethod = "GET"
    request.timeoutInterval = 10
    request.addValue(shortTermToken, forHTTPHeaderField: "Authorization")

    let semaphore = DispatchSemaphore(value: 0)
    var isEnabled = false
    let task = session.dataTask(with: request) { data, response, _ in
      defer { semaphore.signal() }
      guard
        let httpResponse = response as? HTTPURLResponse,
        httpResponse.statusCode == 200,
        let data,
        let status = try? JSONDecoder().decode(PaygStatusResponse.self, from: data)
      else {
        return
      }
      isEnabled = status.paygEnabled && status.paygEligible
    }
    task.resume()
    guard semaphore.wait(timeout: .now() + 10) == .success else {
      task.cancel()
      return false
    }
    return isEnabled
  }
}

public enum CLICloudGoogleOAuthDesktopFlow {
  private static let callbackPath = "/oauth2callback"

  private struct AuthorizationCallback {
    let apiKey: String?
    let provider: String?
    let error: String?
    let errorDescription: String?
  }

  #if canImport(Network)
    private final class LoopbackServer {
      private static let maxRequestSize = 64 * 1024
      private let callbackApplicationName: String
      private let queue: DispatchQueue
      private let readySemaphore = DispatchSemaphore(value: 0)
      private let callbackSemaphore = DispatchSemaphore(value: 0)
      private var portValue: UInt16?
      private var callbackResult: Result<AuthorizationCallback, Error>?
      private let listener: NWListener

      init(callbackApplicationName: String, queueLabel: String) throws {
        self.callbackApplicationName = callbackApplicationName
        queue = DispatchQueue(label: queueLabel)
        do {
          let parameters = NWParameters.tcp
          parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
          listener = try NWListener(using: parameters)
        } catch {
          throw CLICloudAuthError.listenerFailed(error.localizedDescription)
        }

        listener.stateUpdateHandler = { [weak self] state in
          guard let self else { return }
          switch state {
          case .ready:
            self.portValue = self.listener.port?.rawValue
            self.readySemaphore.signal()
          case .failed(let error):
            self.callbackResult = .failure(error)
            self.readySemaphore.signal()
            self.callbackSemaphore.signal()
          default:
            break
          }
        }

        listener.newConnectionHandler = { [weak self] connection in
          self?.handle(connection: connection)
        }
        listener.start(queue: queue)
      }

      var redirectURL: URL {
        get throws {
          let waitResult = readySemaphore.wait(timeout: .now() + 10)
          guard waitResult == .success, let portValue else {
            throw CLICloudAuthError.listenerFailed("Listener did not become ready.")
          }
          return URL(string: "http://127.0.0.1:\(portValue)\(callbackPath)")!
        }
      }

      func waitForCallback(timeout: TimeInterval = 180) throws -> AuthorizationCallback {
        let waitResult = callbackSemaphore.wait(timeout: .now() + timeout)
        guard waitResult == .success else {
          listener.cancel()
          throw CLICloudAuthError.listenerTimedOut
        }
        switch callbackResult {
        case .success(let callback):
          return callback
        case .failure(let error):
          throw CLICloudAuthError.callbackFailed(error.localizedDescription)
        case .none:
          throw CLICloudAuthError.callbackFailed("No callback received.")
        }
      }

      private func handle(connection: NWConnection) {
        connection.start(queue: queue)
        receiveRequest(on: connection, accumulatedData: Data())
      }

      private func receiveRequest(on connection: NWConnection, accumulatedData: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) {
          [weak self] data, _, isComplete, error in
          guard let self else { return }
          if let error {
            self.finish(connection: connection, result: .failure(error))
            return
          }
          var accumulatedData = accumulatedData
          if let data {
            accumulatedData.append(data)
          }
          do {
            if try self.processRequestIfReady(accumulatedData, on: connection) {
              return
            }
          } catch {
            self.finish(connection: connection, result: .failure(error))
            return
          }
          if isComplete {
            self.finish(
              connection: connection,
              result: .failure(
                CLICloudAuthError.callbackFailed("Malformed callback request."))
            )
            return
          }
          guard accumulatedData.count < Self.maxRequestSize else {
            self.finish(
              connection: connection,
              result: .failure(
                CLICloudAuthError.callbackFailed(
                  "OAuth callback request exceeded size limit."))
            )
            return
          }
          self.receiveRequest(on: connection, accumulatedData: accumulatedData)
        }
      }

      private func processRequestIfReady(_ requestData: Data, on connection: NWConnection) throws
        -> Bool
      {
        guard let headerText = completeHeaderText(from: requestData) else {
          return false
        }
        let firstLine =
          headerText.components(separatedBy: "\r\n").first
          ?? headerText.components(separatedBy: "\n").first
        guard let firstLine else {
          throw CLICloudAuthError.callbackFailed("Unexpected HTTP request line.")
        }
        let requestParts = firstLine.split(separator: " ")
        guard requestParts.count >= 2 else {
          throw CLICloudAuthError.callbackFailed("Unexpected HTTP request line.")
        }
        let requestTarget = String(requestParts[1])
        guard
          let components = URLComponents(string: "http://127.0.0.1\(requestTarget)"),
          components.path == callbackPath
        else {
          writeHTTPResponse(
            connection: connection,
            statusLine: "HTTP/1.1 404 Not Found",
            body: "<html><body><h1>Not Found</h1></body></html>"
          ) { _ in
            self.finish(
              connection: connection,
              result: .failure(CLICloudAuthError.callbackFailed("Unexpected callback path."))
            )
          }
          return true
        }

        let queryItems = Dictionary(
          uniqueKeysWithValues: (components.queryItems ?? []).map { ($0.name, $0.value ?? "") })
        let callback = AuthorizationCallback(
          apiKey: queryItems["api_key"],
          provider: queryItems["provider"],
          error: queryItems["error"],
          errorDescription: queryItems["error_description"]
        )
        writeHTTPResponse(
          connection: connection,
          statusLine: "HTTP/1.1 200 OK",
          body: authorizationResultPage(callback)
        ) { sendError in
          if let sendError {
            self.finish(connection: connection, result: .failure(sendError))
          } else {
            self.finish(connection: connection, result: .success(callback))
          }
        }
        return true
      }

      private func authorizationResultPage(_ callback: AuthorizationCallback) -> String {
        let success = callback.error == nil
        let title = success ? "Signed in to Draw Things CLI" : "Sign-in failed"
        let escapedTitle = htmlEscaped(title)
        let escapedApplicationName = htmlEscaped(callbackApplicationName)
        let detail =
          success
          ? "You may now close this page"
          : "Return to \(escapedApplicationName) and try signing in again."
        let iconBase64 = """
          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
          Rvzwx44qe6j2b3ScPhiUg1z1ct3NeCvTjrl02dpkosKYPDOoVW4dChCxr8KiSb6V2LZJeKPsDTrlW0/PJLQQ4/cfnbJ+/2dj
          65npVrdPQo0PD6pFFa6z97O8V2YXk5kivTa5V3DuUozCio1JSrcgJ+PpvAlNE5uULhtprb2s64tl1TT1krjbKaI2YMqjJ+9U
          j998/OSxEtzF8F6s38j5fYMsX87g2btPz8531XS7gLqVpQcv8tJzO/03PnXjaCfTqJ3qOT1wami5Y40AFIZHhe06jUyltQdl
          roi7xkSB/cGgjrCsKoiglKqCny+XIBJWK4l+tNXnnZ/5c6VjdZ6/uj08WVYtEoB3xjrrgnUrk14d9C9br0IdBF/a32qbunXG
          WM2amFVbx1kU1rqKItqAUhPA+5fT8wRZkbluZ2DU6+PjjfG7IdSZ4RTEh9b1hydPzvJ2efnkoh6fjhfzejL53EtxNpN7E/AJ
          RpuD9/3MX5tZNczYssoY4unp1tamkAwyN3Cmy6lQODBqJ9f7pSmNaqLkej3HlZfLee198NWqqatmtYzVSkAAoY2R7KDMjza2
          724/qJoRaoa0UeYv7vTv3tz4gZ/0WN46P838crObdQ3dPzutMGz1srgWa1z3rCBtITdonetkDfFBaYPWtzdHkDnvlEzO7Ltf
          jwBbu0fD7SNXdlZtaqrqpRfuhlR6H7DcPru4eFCZh2NOgCHBoKde/0m/tV8O+0XeyYq9jpt+8ytf/8//vaNMafODsnP/f/3P
          jdzudbOdXt7vlA2bWcAIBMg+Ykxote51iiRonBMRZEKtyGpi4uzTPxFiPMzyLMRzloHhVaj/+9PzNqX9YfcDO8Pj6XSnzFhx
          12lirqMs6zYiWaPKLGuQiVVEVWwNXtsauDI7IZ2smQskJsXUl1Xn+D4AauOcVbBW3bTeX5487PfMw3efbm3vzWcXqakfTGmy
          9KPN7Of91t/9wc/8JLbAGqxmWUx/y8/95V/7/Jd+yk/8yaPRxv/9V//qm//vC+/5kZ9C1ky6iViHFCMwkY+pDpG9P7mcNXXT
          1qu2WogPyiibWaXYOMOf/UW/7KJqNhHNdTldQAzNarpqfdNYy1VTVRia1vc72bRunGZXZFlmWbNhNEZ7YtFaFfZ2N3830XkQ
          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
          """.replacingOccurrences(of: "\n", with: "")
        let iconSource = "data:image/png;base64,\(iconBase64)"
        return """
          <!doctype html>
          <html>
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>\(escapedTitle)</title>
            <style>
              :root {
                color-scheme: light dark;
                --background: #fffdf8;
                --text: #111111;
                --muted: #5f6368;
                --border: #e7e2d8;
                --icon-surface: #ffffff;
              }
              @media (prefers-color-scheme: dark) {
                :root {
                  --background: #111111;
                  --text: #f7f7f2;
                  --muted: #b6b6b2;
                  --border: #343434;
                  --icon-surface: #ffffff;
                }
              }
              * {
                box-sizing: border-box;
              }
              body {
                min-height: 100vh;
                margin: 0;
                display: grid;
                place-items: center;
                padding: 24px;
                background: var(--background);
                color: var(--text);
                font: 15px/1.5 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              }
              main {
                width: min(100%, 430px);
                text-align: center;
              }
              .icon {
                width: 64px;
                height: 64px;
                margin: 0 auto 26px;
                display: grid;
                place-items: center;
                border: 1px solid var(--border);
                border-radius: 16px;
                background: var(--icon-surface);
                box-shadow: 0 12px 32px rgba(17, 17, 17, 0.08);
              }
              .app-icon {
                width: 42px;
                height: 42px;
                display: block;
              }
              h1 {
                margin: 0;
                font-size: 32px;
                line-height: 1.2;
                letter-spacing: 0;
                font-weight: 650;
              }
              p {
                margin: 18px 0 0;
                color: var(--muted);
                font-size: 16px;
              }
              @media (max-width: 420px) {
                main {
                  width: 100%;
                }
                h1 {
                  font-size: 29px;
                }
              }
            </style>
          </head>
          <body>
            <main>
              <div class="icon">
                <img class="app-icon" src="\(iconSource)" alt="">
              </div>
              <h1>\(escapedTitle)</h1>
              <p>\(detail)</p>
            </main>
            <script>
              function closeAuthWindow() {
                window.open("", "_self");
                window.close();
              }
              window.setTimeout(closeAuthWindow, 700);
            </script>
          </body>
          </html>
          """
      }

      private func htmlEscaped(_ value: String) -> String {
        value
          .replacingOccurrences(of: "&", with: "&amp;")
          .replacingOccurrences(of: "<", with: "&lt;")
          .replacingOccurrences(of: ">", with: "&gt;")
          .replacingOccurrences(of: "\"", with: "&quot;")
          .replacingOccurrences(of: "'", with: "&#39;")
      }

      private func completeHeaderText(from requestData: Data) -> String? {
        let delimiters = [
          requestData.range(of: Data("\r\n\r\n".utf8)),
          requestData.range(of: Data("\n\n".utf8)),
        ]
        guard let delimiter = delimiters.compactMap({ $0 }).first else {
          return nil
        }
        let headerData = Data(requestData[..<delimiter.lowerBound])
        return String(data: headerData, encoding: .utf8)
      }

      private func writeHTTPResponse(
        connection: NWConnection,
        statusLine: String,
        body: String,
        completion: @escaping (NWError?) -> Void
      ) {
        let bodyData = Data(body.utf8)
        let responseText = """
          \(statusLine)\r
          Content-Type: text/html; charset=utf-8\r
          Content-Length: \(bodyData.count)\r
          Connection: close\r
          \r
          \(body)
          """
        connection.send(content: Data(responseText.utf8), completion: .contentProcessed(completion))
      }

      private func finish(connection: NWConnection, result: Result<AuthorizationCallback, Error>) {
        callbackResult = result
        connection.cancel()
        listener.cancel()
        callbackSemaphore.signal()
      }
    }
  #endif

  public static func signIn(
    apiBaseURL: URL = CLICloudDefaultAPIBaseURL,
    callbackApplicationName: String,
    listenerQueueLabel: String = "ai.drawthings.cloud-auth.google-oauth"
  ) throws -> CLICloudCredentials {
    #if canImport(Network)
      let callbackServer = try LoopbackServer(
        callbackApplicationName: callbackApplicationName,
        queueLabel: listenerQueueLabel
      )
      let redirectURL = try callbackServer.redirectURL
      let authorizationURL = try startGoogleLogin(apiBaseURL: apiBaseURL, redirectURL: redirectURL)
      try openBrowser(authorizationURL)
      let callback = try callbackServer.waitForCallback()
      if let error = callback.error {
        throw CLICloudAuthError.callbackFailed(callback.errorDescription ?? error)
      }
      guard let apiKey = callback.apiKey, !apiKey.isEmpty else {
        throw CLICloudAuthError.callbackFailed("API key missing from callback.")
      }
      return CLICloudCredentials(
        provider: callback.provider ?? "google",
        apiKey: apiKey,
        apiBaseURL: apiBaseURL.absoluteString,
        savedAt: Date()
      )
    #else
      throw CLICloudAuthError.unsupportedGoogleLoginPlatform
    #endif
  }

  private static func startGoogleLogin(apiBaseURL: URL, redirectURL: URL) throws -> URL {
    var request = URLRequest(
      url: CLICloudAuthClient.apiURL(baseURL: apiBaseURL, path: "/auth/google/login"))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONEncoder().encode(["redirect_uri": redirectURL.absoluteString])

    let semaphore = DispatchSemaphore(value: 0)
    var responseData: Data?
    var response: URLResponse?
    var responseError: Error?
    let task = URLSession.shared.dataTask(with: request) { data, urlResponse, error in
      responseData = data
      response = urlResponse
      responseError = error
      semaphore.signal()
    }
    task.resume()
    semaphore.wait()
    if let responseError {
      throw CLICloudAuthError.startFailed(responseError.localizedDescription)
    }
    guard let httpResponse = response as? HTTPURLResponse else {
      throw CLICloudAuthError.startFailed("missing HTTP response.")
    }
    guard let responseData else {
      throw CLICloudAuthError.startFailed("missing response body.")
    }
    guard (200..<300).contains(httpResponse.statusCode) else {
      let message = String(data: responseData, encoding: .utf8) ?? "HTTP \(httpResponse.statusCode)"
      throw CLICloudAuthError.startFailed(message)
    }
    let payload = try JSONDecoder().decode(GoogleLoginStartResponse.self, from: responseData)
    guard let authorizationURL = URL(string: payload.authorizationURL) else {
      throw CLICloudAuthError.startFailed("invalid authorization URL.")
    }
    return authorizationURL
  }

  private struct GoogleLoginStartResponse: Decodable {
    let authorizationURL: String

    private enum CodingKeys: String, CodingKey {
      case authorizationURL = "authorization_url"
    }
  }

  private static func openBrowser(_ url: URL) throws {
    #if os(macOS)
      try runBrowserLauncher("/usr/bin/open", argument: url.absoluteString)
    #elseif os(Linux)
      try runBrowserLauncher("/usr/bin/xdg-open", argument: url.absoluteString)
    #else
      throw CLICloudAuthError.browserLaunchFailed("Unsupported platform.")
    #endif
  }

  #if os(macOS) || os(Linux)
    private static func runBrowserLauncher(_ tool: String, argument: String) throws {
      let process = Process()
      process.executableURL = URL(fileURLWithPath: tool)
      process.arguments = [argument]
      let errorPipe = Pipe()
      process.standardError = errorPipe
      do {
        try process.run()
        process.waitUntilExit()
      } catch {
        throw CLICloudAuthError.browserLaunchFailed(error.localizedDescription)
      }
      guard process.terminationStatus == 0 else {
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        let errorMessage = String(data: errorData, encoding: .utf8)?
          .trimmingCharacters(in: .whitespacesAndNewlines)
        let detail =
          errorMessage?.isEmpty == false
          ? errorMessage! : "exit status \(process.terminationStatus)"
        throw CLICloudAuthError.browserLaunchFailed(detail)
      }
    }
  #endif
}
