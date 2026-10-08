import Foundation
import GRPCServer

internal final class CloudAuthenticatorRegistry {
  static let shared = CloudAuthenticatorRegistry()

  private struct Key: Hashable {
    let apiKey: String
    let baseURL: URL
  }

  private var authenticators = ProtectedValue([Key: CloudAuthenticator]())

  private init() {}

  func authenticator(
    apiKey: String,
    baseURL: URL = CloudConfiguration.defaultBaseURL
  ) -> CloudAuthenticator {
    let key = Key(apiKey: apiKey, baseURL: baseURL)
    var authenticator: CloudAuthenticator?
    authenticators.modify { authenticators in
      if let cached = authenticators[key] {
        authenticator = cached
        return
      }
      let created = CloudAuthenticator(
        configuration: CloudConfiguration(
          apiKey: apiKey,
          appCheck: .none,
          baseURL: baseURL,
          tokenRefreshThreshold: 300
        )
      )
      authenticators[key] = created
      authenticator = created
    }
    return authenticator!
  }
}
