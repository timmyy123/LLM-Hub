import Foundation

// MARK: - App Check Configuration

/// Configuration for app verification services.
///
/// App Check provides an additional layer of security by verifying that cloud
/// requests come from your legitimate app. You can use Firebase App Check,
/// Supabase, or rely on Native App Attest (automatic on iOS real devices).
///
/// ## Usage with Firebase App Check
///
/// 1. Set up Firebase App Check in your Firebase project:
///    - Go to Firebase Console > App Check
///    - Register your app with App Attest (iOS) or Play Integrity (Android)
///
/// 2. Add Firebase SDK to your app and get an App Check token:
///    ```swift
///    import FirebaseAppCheck
///    import MediaGenerationKit
///
///    // Get token from Firebase SDK
///    let tokenResult = try await AppCheck.appCheck().token(forcingRefresh: false)
///
///    let pipeline = try await MediaGenerationPipeline.fromPretrained(
///      "hf://black-forest-labs/FLUX.2-klein-4B",
///      backend: .cloudCompute(
///        apiKey: "dk_xxx",
///        options: .init(appCheck: .firebase(token: tokenResult.token))
///      )
///    )
///    ```
///
/// ## Usage with Supabase
///
/// 1. Set up authentication in your Supabase project
///
/// 2. Get a JWT token from Supabase:
///    ```swift
///    import Supabase
///    import MediaGenerationKit
///
///    let client = SupabaseClient(supabaseURL: url, supabaseKey: anonKey)
///    let authSession = try await client.auth.signInAnonymously()
///
///    let pipeline = try await MediaGenerationPipeline.fromPretrained(
///      "hf://black-forest-labs/FLUX.2-klein-4B",
///      backend: .cloudCompute(
///        apiKey: "dk_xxx",
///        options: .init(appCheck: .supabase(token: authSession.accessToken))
///      )
///    )
///    ```
///
/// ## Without App Check
///
/// On iOS real devices, Native App Attest is used automatically.
/// On simulators or macOS, raw API key is used (less secure).
///
public enum AppCheckConfiguration: Sendable, Equatable {
  /// No app check - relies on Native App Attest on iOS real devices, or raw API key
  case none

  /// Firebase App Check token
  /// - Parameter token: The App Check token from Firebase SDK (`AppCheck.appCheck().token()`)
  case firebase(token: String)

  /// Supabase authentication token
  /// - Parameter token: The access token from Supabase SDK (`session.accessToken`)
  case supabase(token: String)

  /// Internal type identifier for API requests
  internal var type: String {
    switch self {
    case .none: return "none"
    case .firebase: return "firebase"
    case .supabase: return "supabase"
    }
  }

  /// Internal token getter
  internal var token: String? {
    switch self {
    case .none: return nil
    case .firebase(let token): return token
    case .supabase(let token): return token
    }
  }
}

// MARK: - Cloud Configuration

/// Internal configuration for MediaGenerationKit cloud services.
internal struct CloudConfiguration {
  /// API key for DrawThings Cloud
  public let apiKey: String

  /// App Check configuration for additional security
  public let appCheck: AppCheckConfiguration

  /// Base URL for the API server
  /// Default: https://api.drawthings.ai
  public let baseURL: URL

  /// Token refresh threshold in seconds (refresh before expiry)
  /// Default: 300 seconds (5 minutes)
  public let tokenRefreshThreshold: TimeInterval

  /// Network timeout for short-term token requests.
  public let requestTimeout: TimeInterval

  /// Default API base URL
  public static let defaultBaseURL = URL(string: "https://api.drawthings.ai")!

  /// Initialize with just an API key (simplest setup)
  /// - Parameter apiKey: DrawThings API key
  public init(apiKey: String) {
    self.apiKey = apiKey
    self.appCheck = .none
    self.baseURL = Self.defaultBaseURL
    self.tokenRefreshThreshold = 300
    self.requestTimeout = 30
  }

  /// Initialize with API key and App Check
  /// - Parameters:
  ///   - apiKey: DrawThings API key
  ///   - appCheck: App Check configuration
  public init(apiKey: String, appCheck: AppCheckConfiguration) {
    self.apiKey = apiKey
    self.appCheck = appCheck
    self.baseURL = Self.defaultBaseURL
    self.tokenRefreshThreshold = 300
    self.requestTimeout = 30
  }

  /// Full initializer with all options
  /// - Parameters:
  ///   - apiKey: DrawThings API key
  ///   - appCheck: App Check configuration
  ///   - baseURL: Custom API base URL (for testing)
  ///   - tokenRefreshThreshold: Token refresh threshold in seconds
  ///   - requestTimeout: Timeout for token requests.
  public init(
    apiKey: String,
    appCheck: AppCheckConfiguration,
    baseURL: URL? = nil,
    tokenRefreshThreshold: TimeInterval = 300,
    requestTimeout: TimeInterval = 30
  ) {
    self.apiKey = apiKey
    self.appCheck = appCheck
    self.baseURL = baseURL ?? Self.defaultBaseURL
    self.tokenRefreshThreshold = tokenRefreshThreshold
    self.requestTimeout = requestTimeout
  }
}
