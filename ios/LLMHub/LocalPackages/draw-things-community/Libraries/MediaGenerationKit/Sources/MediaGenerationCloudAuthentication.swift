import Foundation
import GRPCServer

internal enum MediaGenerationRemoteAuthenticationMode {
  /// No authentication (open server)
  case none
  /// Shared secret password authentication
  case sharedSecret(String)
  /// DrawThings Cloud Compute authentication backed by a shared authenticator registry.
  case cloudCompute(apiKey: String, baseURL: URL, appCheck: AppCheckConfiguration)
}

internal enum MediaGenerationCloudAuthentication {
  private static let paygStatusCacheTTL: TimeInterval = 30
  private static let paygStatusCacheQueue = DispatchQueue(
    label: "com.drawthings.sdk.cloud-auth.payg-cache")
  private static var paygStatusCache: [PaygCacheKey: (enabled: Bool, cachedAt: Date)] = [:]

  private struct PaygCacheKey: Hashable {
    let shortTermToken: String
    let baseURL: URL
  }

  private struct AuthenticationRequest: Codable {
    let blob: String
    let fromBridge: Bool
    let attestationSupported: Bool
    let assertionPayload: String?
    let originalTransactionId: String?
    let isSandbox: Bool
    let consumableType: String?
    let amount: Double?
  }

  private struct AuthenticationResponse: Codable {
    let gRPCToken: String
  }

  private struct PaygStatusResponse: Codable {
    let paygEnabled: Bool
    let paygEligible: Bool
  }

  private final class TaskCancellationBag: @unchecked Sendable {
    private struct State {
      var cancelled = false
      var tasks: [URLSessionTask] = []
    }

    private var state = ProtectedValue(State())

    func register(_ task: URLSessionTask) {
      var shouldCancelImmediately = false
      state.modify { state in
        shouldCancelImmediately = state.cancelled
        if !state.cancelled {
          state.tasks.append(task)
        }
      }
      if shouldCancelImmediately {
        task.cancel()
      }
    }

    func cancelAll() {
      var tasks: [URLSessionTask] = []
      var shouldCancel = false
      state.modify { state in
        if state.cancelled {
          return
        }
        state.cancelled = true
        tasks = state.tasks
        state.tasks.removeAll()
        shouldCancel = true
      }
      guard shouldCancel else { return }
      tasks.forEach { $0.cancel() }
    }
  }

  static func authenticate(
    shortTermToken: String,
    encodedBlob: String,
    fromBridge: Bool,
    estimatedComputeUnits: Double?,
    baseURL: URL = CloudConfiguration.defaultBaseURL,
    timeout: TimeInterval = 30,
    cancellation: (@escaping () -> Void) -> Void,
    onFailure: ((String) -> Void)? = nil
  ) -> String? {
    let taskCancellationBag = TaskCancellationBag()
    cancellation {
      taskCancellationBag.cancelAll()
    }

    let paygEnabled =
      cachedPaygEnabled(shortTermToken: shortTermToken, baseURL: baseURL)
      ?? fetchPaygEnabled(
        shortTermToken: shortTermToken,
        baseURL: baseURL,
        timeout: min(timeout, 10),
        taskCancellationBag: taskCancellationBag
      )
    let isPositiveAmount = (estimatedComputeUnits ?? 0) > 0

    var request = URLRequest(url: baseURL.appendingPathComponent("/authenticate"))
    request.httpMethod = "POST"
    request.timeoutInterval = timeout
    request.addValue("application/json", forHTTPHeaderField: "Content-Type")
    request.addValue(shortTermToken, forHTTPHeaderField: "Authorization")

    let requestBody = AuthenticationRequest(
      blob: encodedBlob,
      fromBridge: fromBridge,
      attestationSupported: false,
      assertionPayload: nil,
      originalTransactionId: nil,
      isSandbox: false,
      consumableType: paygEnabled && isPositiveAmount ? "payg" : nil,
      amount: isPositiveAmount ? estimatedComputeUnits : nil
    )

    guard let body = try? JSONEncoder().encode(requestBody) else {
      return nil
    }
    request.httpBody = body

    let group = DispatchGroup()
    group.enter()
    var bearerToken: String?

    let task = URLSession.shared.dataTask(with: request) { data, response, error in
      defer { group.leave() }

      if let error = error {
        onFailure?("network error: \(error.localizedDescription)")
        return
      }
      guard let httpResponse = response as? HTTPURLResponse else {
        onFailure?("no HTTP response")
        return
      }
      guard httpResponse.statusCode == 200, let data = data else {
        let body = data.flatMap { String(data: $0, encoding: .utf8) } ?? "(no body)"
        onFailure?("HTTP \(httpResponse.statusCode): \(body)")
        return
      }
      guard let authResponse = try? JSONDecoder().decode(AuthenticationResponse.self, from: data)
      else {
        let raw = String(data: data, encoding: .utf8) ?? "(non-UTF8 body)"
        onFailure?("decode failed, body: \(raw)")
        return
      }

      bearerToken = authResponse.gRPCToken
    }

    taskCancellationBag.register(task)
    task.resume()

    guard group.wait(timeout: .now() + timeout) != .timedOut else {
      taskCancellationBag.cancelAll()
      return nil
    }
    return bearerToken
  }

  static func prefetchPaygEnabled(
    shortTermToken: String,
    baseURL: URL
  ) async -> Bool {
    if let cached = cachedPaygEnabled(shortTermToken: shortTermToken, baseURL: baseURL) {
      return cached
    }

    var request = URLRequest(url: baseURL.appendingPathComponent("/billing/stripe/payg"))
    request.httpMethod = "GET"
    request.timeoutInterval = 10
    request.addValue(shortTermToken, forHTTPHeaderField: "Authorization")

    do {
      let (data, response) = try await URLSession.shared.data(for: request)
      guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
        cachePaygEnabled(false, shortTermToken: shortTermToken, baseURL: baseURL)
        return false
      }
      let status = try JSONDecoder().decode(PaygStatusResponse.self, from: data)
      let isEnabled = status.paygEnabled && status.paygEligible
      cachePaygEnabled(isEnabled, shortTermToken: shortTermToken, baseURL: baseURL)
      return isEnabled
    } catch {
      return false
    }
  }

  private static func fetchPaygEnabled(
    shortTermToken: String,
    baseURL: URL,
    timeout: TimeInterval,
    taskCancellationBag: TaskCancellationBag
  ) -> Bool {
    if let cached = cachedPaygEnabled(shortTermToken: shortTermToken, baseURL: baseURL) {
      return cached
    }

    let group = DispatchGroup()
    group.enter()
    var isEnabled = false

    var request = URLRequest(url: baseURL.appendingPathComponent("/billing/stripe/payg"))
    request.httpMethod = "GET"
    request.timeoutInterval = timeout
    request.addValue(shortTermToken, forHTTPHeaderField: "Authorization")

    let task = URLSession.shared.dataTask(with: request) { data, response, _ in
      defer { group.leave() }
      guard let httpResponse = response as? HTTPURLResponse,
        httpResponse.statusCode == 200,
        let data = data,
        let status = try? JSONDecoder().decode(PaygStatusResponse.self, from: data)
      else {
        return
      }
      isEnabled = status.paygEnabled && status.paygEligible
    }

    taskCancellationBag.register(task)
    task.resume()

    guard group.wait(timeout: .now() + timeout) != .timedOut else {
      task.cancel()
      cachePaygEnabled(false, shortTermToken: shortTermToken, baseURL: baseURL)
      return false
    }

    cachePaygEnabled(isEnabled, shortTermToken: shortTermToken, baseURL: baseURL)
    return isEnabled
  }

  private static func cachedPaygEnabled(shortTermToken: String, baseURL: URL) -> Bool? {
    let now = Date()
    let key = PaygCacheKey(shortTermToken: shortTermToken, baseURL: baseURL)
    return paygStatusCacheQueue.sync {
      guard let cached = paygStatusCache[key],
        now.timeIntervalSince(cached.cachedAt) <= paygStatusCacheTTL
      else {
        return nil
      }
      return cached.enabled
    }
  }

  private static func cachePaygEnabled(
    _ enabled: Bool,
    shortTermToken: String,
    baseURL: URL
  ) {
    let key = PaygCacheKey(shortTermToken: shortTermToken, baseURL: baseURL)
    let now = Date()
    paygStatusCacheQueue.sync {
      paygStatusCache[key] = (enabled: enabled, cachedAt: now)
    }
  }
}
