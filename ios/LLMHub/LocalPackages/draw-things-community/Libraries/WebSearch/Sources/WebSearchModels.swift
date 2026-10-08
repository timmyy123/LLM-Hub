import Foundation

/// Browser-style user agent used by default for web search and fetch requests.
public let WebSearchDefaultUserAgent =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/537.36 "
  + "(KHTML, like Gecko) Chrome/135.0.7049.95 Safari/537.36"

/// Search provider used for web search requests.
public enum WebSearchProvider: CaseIterable, Equatable {
  /// DuckDuckGo HTML search.
  case duckDuckGo
  /// Sogou web search.
  case sogou
  /// Kagi Search API.
  case kagi(apiKey: String)
  /// Brave Search API.
  case brave(apiKey: String)
  /// Web search disabled.
  case disabled

  public static let allCases: [WebSearchProvider] = [
    .duckDuckGo, .sogou, .kagi(apiKey: ""), .brave(apiKey: ""), .disabled,
  ]

  /// Creates a provider from its persisted identifier and API key.
  public init?(identifier: String, apiKey: String) {
    switch identifier {
    case "duckduckgo":
      self = .duckDuckGo
    case "sogou":
      self = .sogou
    case "kagi":
      self = .kagi(apiKey: apiKey)
    case "brave":
      self = .brave(apiKey: apiKey)
    case "disabled":
      self = .disabled
    default:
      return nil
    }
  }

  /// Identifier used by settings persistence and command-line arguments.
  public var identifier: String {
    switch self {
    case .duckDuckGo:
      return "duckduckgo"
    case .sogou:
      return "sogou"
    case .kagi:
      return "kagi"
    case .brave:
      return "brave"
    case .disabled:
      return "disabled"
    }
  }
}

/// Safe-search setting passed through to DuckDuckGo's HTML search endpoint.
public enum DuckDuckGoSafeSearch: String, Codable {
  /// Strict filtering.
  case strict
  /// DuckDuckGo's moderate filtering.
  case moderate
  /// Safe-search disabled.
  case off

  var duckDuckGoValue: String {
    switch self {
    case .strict:
      return "1"
    case .moderate:
      return "-1"
    case .off:
      return "-2"
    }
  }
}

/// Freshness filter passed through to web search providers.
public enum WebSearchTimeFilter: String, Codable {
  /// Results from the past day.
  case day
  /// Results from the past week.
  case week
  /// Results from the past month.
  case month
  /// Results from the past year.
  case year

  var duckDuckGoValue: String {
    switch self {
    case .day:
      return "d"
    case .week:
      return "w"
    case .month:
      return "m"
    case .year:
      return "y"
    }
  }

  var sogouValue: String {
    switch self {
    case .day:
      return "1"
    case .week:
      return "2"
    case .month:
      return "3"
    case .year:
      return "4"
    }
  }
}

/// Options for a DuckDuckGo HTML search request.
public struct DuckDuckGoSearchOptions: Codable {
  /// DuckDuckGo region code, such as `us-en` or `wt-wt`.
  public var region: String
  /// Safe-search filtering mode.
  public var safeSearch: DuckDuckGoSafeSearch
  /// Optional freshness filter.
  public var timeFilter: WebSearchTimeFilter?
  /// Maximum number of unique results to return.
  public var maxResults: Int
  /// Maximum number of DuckDuckGo result pages to request.
  public var pages: Int
  /// Request timeout in seconds.
  public var timeout: TimeInterval
  /// User agent used for DuckDuckGo requests.
  public var userAgent: String

  /// Creates DuckDuckGo search options, clamping counts to at least one.
  public init(
    region: String = "us-en",
    safeSearch: DuckDuckGoSafeSearch = .moderate,
    timeFilter: WebSearchTimeFilter? = nil,
    maxResults: Int = 10,
    pages: Int = 1,
    timeout: TimeInterval = 15,
    userAgent: String = WebSearchDefaultUserAgent
  ) {
    self.region = region
    self.safeSearch = safeSearch
    self.timeFilter = timeFilter
    self.maxResults = max(1, maxResults)
    self.pages = max(1, pages)
    self.timeout = timeout
    self.userAgent = userAgent
  }
}

/// Options for a Sogou HTML search request.
public struct SogouSearchOptions: Codable {
  /// Optional freshness filter.
  public var timeFilter: WebSearchTimeFilter?
  /// Maximum number of unique results to return.
  public var maxResults: Int
  /// Maximum number of Sogou result pages to request.
  public var pages: Int
  /// Request timeout in seconds.
  public var timeout: TimeInterval
  /// User agent used for Sogou requests.
  public var userAgent: String

  /// Creates Sogou search options, clamping counts to at least one.
  public init(
    timeFilter: WebSearchTimeFilter? = nil,
    maxResults: Int = 10,
    pages: Int = 1,
    timeout: TimeInterval = 15,
    userAgent: String = WebSearchDefaultUserAgent
  ) {
    self.timeFilter = timeFilter
    self.maxResults = max(1, maxResults)
    self.pages = max(1, pages)
    self.timeout = timeout
    self.userAgent = userAgent
  }
}

/// Options for a Kagi Search API request.
public struct KagiSearchOptions: Codable {
  /// Optional freshness filter.
  public var timeFilter: WebSearchTimeFilter?
  /// Whether safe search is enabled.
  public var safeSearch: Bool
  /// Maximum number of unique results to return.
  public var maxResults: Int
  /// Maximum number of Kagi result pages to request.
  public var pages: Int
  /// Request timeout in seconds.
  public var timeout: TimeInterval
  /// User agent used for Kagi requests.
  public var userAgent: String

  /// Creates Kagi search options using limits supported by the v1 API.
  public init(
    timeFilter: WebSearchTimeFilter? = nil,
    safeSearch: Bool = true,
    maxResults: Int = 10,
    pages: Int = 1,
    timeout: TimeInterval = 15,
    userAgent: String = WebSearchDefaultUserAgent
  ) {
    self.timeFilter = timeFilter
    self.safeSearch = safeSearch
    self.maxResults = min(max(1, maxResults), 1_024)
    self.pages = min(max(1, pages), 10)
    self.timeout = timeout
    self.userAgent = userAgent
  }
}

/// Options for a Brave Search API request.
public struct BraveSearchOptions: Codable {
  /// Optional freshness filter.
  public var timeFilter: WebSearchTimeFilter?
  /// Whether safe search is enabled.
  public var safeSearch: Bool
  /// Maximum number of unique results to return.
  public var maxResults: Int
  /// Maximum number of Brave result pages to request.
  public var pages: Int
  /// Request timeout in seconds.
  public var timeout: TimeInterval
  /// User agent used for Brave requests.
  public var userAgent: String

  /// Creates Brave search options using limits supported by the v1 API.
  public init(
    timeFilter: WebSearchTimeFilter? = nil,
    safeSearch: Bool = true,
    maxResults: Int = 10,
    pages: Int = 1,
    timeout: TimeInterval = 15,
    userAgent: String = WebSearchDefaultUserAgent
  ) {
    self.timeFilter = timeFilter
    self.safeSearch = safeSearch
    self.maxResults = min(max(1, maxResults), 200)
    self.pages = min(max(1, pages), 10)
    self.timeout = timeout
    self.userAgent = userAgent
  }
}

/// Output format for a direct web fetch.
public enum WebFetchFormat: String, Codable {
  /// Convert HTML to Markdown when possible.
  case markdown
  /// Extract visible text when possible.
  case text
  /// Return decoded response text without conversion.
  case html

  var acceptHeader: String {
    switch self {
    case .markdown, .text:
      return "text/html,application/xhtml+xml,text/plain,*/*;q=0.8"
    case .html:
      return "text/html,application/xhtml+xml,*/*;q=0.8"
    }
  }
}

/// Options for fetching one URL directly.
public struct WebFetchOptions: Codable {
  /// Output conversion format.
  public var format: WebFetchFormat
  /// Request timeout in seconds.
  public var timeout: TimeInterval
  /// Maximum response body size accepted by the fetcher.
  public var maxBytes: Int
  /// User agent used for the request.
  public var userAgent: String

  /// Creates direct fetch options, clamping `maxBytes` to at least one.
  public init(
    format: WebFetchFormat = .markdown,
    timeout: TimeInterval = 30,
    maxBytes: Int = 5 * 1024 * 1024,
    userAgent: String = WebSearchDefaultUserAgent
  ) {
    self.format = format
    self.timeout = timeout
    self.maxBytes = max(1, maxBytes)
    self.userAgent = userAgent
  }
}

/// One normalized web search result.
public struct SearchResult: Codable, Equatable {
  /// One-based rank after de-duplication.
  public var rank: Int
  /// Result title.
  public var title: String
  /// Destination URL after decoding search-provider redirect wrappers when possible.
  public var url: URL
  /// Display URL shown by the search provider.
  public var displayURL: String
  /// Result snippet text.
  public var snippet: String
  /// Source engine identifier.
  public var source: String

  /// Creates a normalized search result.
  public init(
    rank: Int,
    title: String,
    url: URL,
    displayURL: String,
    snippet: String,
    source: String
  ) {
    self.rank = rank
    self.title = title
    self.url = url
    self.displayURL = displayURL
    self.snippet = snippet
    self.source = source
  }
}

/// Metadata for a direct fetch result.
public struct WebFetchMetadata: Codable, Equatable {
  /// Requested URL.
  public var url: URL
  /// Final URL reported by the HTTP response after redirects.
  public var finalURL: URL
  /// Response content type, when provided.
  public var contentType: String?
  /// HTTP status code.
  public var statusCode: Int
  /// Request duration in seconds.
  public var elapsedSeconds: Double
  /// Response body byte count.
  public var byteCount: Int
  /// Output conversion format.
  public var format: WebFetchFormat

  /// Creates fetch metadata.
  public init(
    url: URL,
    finalURL: URL,
    contentType: String?,
    statusCode: Int,
    elapsedSeconds: Double,
    byteCount: Int,
    format: WebFetchFormat
  ) {
    self.url = url
    self.finalURL = finalURL
    self.contentType = contentType
    self.statusCode = statusCode
    self.elapsedSeconds = elapsedSeconds
    self.byteCount = byteCount
    self.format = format
  }
}

/// OpenCode-style result for a direct web fetch.
public struct WebFetchResult: Codable, Equatable {
  /// Human-readable title, usually `URL (content-type)`.
  public var title: String
  /// Fetch diagnostics and conversion metadata.
  public var metadata: WebFetchMetadata
  /// Converted or raw response text.
  public var output: String

  /// Creates a web fetch result.
  public init(title: String, metadata: WebFetchMetadata, output: String) {
    self.title = title
    self.metadata = metadata
    self.output = output
  }
}

/// One search result with an optional direct fetch result.
public struct SearchResultBundle: Codable {
  /// Search result metadata.
  public var result: SearchResult
  /// Direct fetch result for `result.url`, when requested and successful.
  public var fetchResult: WebFetchResult?
  /// Fetch failure message, when fetching was requested and failed.
  public var fetchError: String?

  /// Creates a search result bundle.
  public init(result: SearchResult, fetchResult: WebFetchResult?, fetchError: String?) {
    self.result = result
    self.fetchResult = fetchResult
    self.fetchError = fetchError
  }
}

/// Errors produced by the WebSearch library.
public enum WebSearchError: LocalizedError {
  /// A URL string could not be parsed.
  case invalidURL(String)
  /// The transport returned a non-HTTP response.
  case invalidResponse
  /// A search page contained neither recognizable results nor an explicit no-results message.
  case unexpectedSearchResponse(URL?)
  /// The HTTP response status was not successful.
  case httpStatus(Int, URL?, body: String?, headers: [String: String], byteCount: Int)
  /// The response body could not be decoded as text.
  case bodyDecodingFailed(URL?)
  /// The URL scheme is not supported.
  case unsupportedScheme(String)
  /// The response content type is not supported by the requested operation.
  case unsupportedContentType(String?)
  /// The response body exceeded the configured byte limit.
  case responseTooLarge(Int, Int)
  /// The search provider returned an interstitial or access challenge instead of results.
  case searchBlocked(String, URL?)

  /// Human-readable error message.
  public var errorDescription: String? {
    switch self {
    case .invalidURL(let value):
      return "Invalid URL: \(value)"
    case .invalidResponse:
      return "Expected an HTTP response."
    case .unexpectedSearchResponse(let url):
      return
        "The search provider returned an unrecognized page\(url.map { " from \($0.absoluteString)" } ?? "")."
    case .httpStatus(let status, let url, _, _, _):
      if let url {
        return "HTTP \(status) from \(url.absoluteString)"
      }
      return "HTTP \(status)"
    case .bodyDecodingFailed(let url):
      if let url {
        return "Unable to decode response body from \(url.absoluteString)"
      }
      return "Unable to decode response body."
    case .unsupportedScheme(let value):
      return "Unsupported URL scheme: \(value)"
    case .unsupportedContentType(let contentType):
      return "Unsupported content type: \(contentType ?? "unknown")"
    case .responseTooLarge(let actual, let maximum):
      return "Response too large: \(actual) bytes exceeds limit of \(maximum) bytes"
    case .searchBlocked(let message, let url):
      if let url {
        return "\(message) from \(url.absoluteString)"
      }
      return message
    }
  }
}
