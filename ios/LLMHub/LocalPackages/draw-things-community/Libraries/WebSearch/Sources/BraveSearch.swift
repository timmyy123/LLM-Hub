import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Errors specific to Brave search configuration.
public enum BraveSearchError: LocalizedError {
  /// Brave was selected without an API key.
  case missingAPIKey

  public var errorDescription: String? {
    switch self {
    case .missingAPIKey:
      return "Brave requires an API key. Add one in Search Provider settings."
    }
  }
}

/// Searches Brave's v1 Search API.
public struct BraveSearch {
  private struct ResponseBody: Decodable {
    struct Query: Decodable {
      var moreResultsAvailable: Bool?

      private enum CodingKeys: String, CodingKey {
        case moreResultsAvailable = "more_results_available"
      }
    }

    struct Web: Decodable {
      struct Item: Decodable {
        var url: String?
        var title: String?
        var description: String?
      }

      var results: [Item]?
    }

    var query: Query?
    var web: Web?
  }

  private let apiKey: String
  private let httpTransport: HttpTransport
  private static let endpoint = URL(string: "https://api.search.brave.com/res/v1/web/search")!

  /// Creates a Brave search tool with an explicit API key.
  public init(apiKey: String, httpTransport: HttpTransport = URLSessionHttpTransport()) {
    self.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
    self.httpTransport = httpTransport
  }

  /// Searches Brave and calls `completion` with normalized, de-duplicated results.
  public func search(
    query: String,
    options: BraveSearchOptions = BraveSearchOptions(),
    completion: @escaping (Result<[SearchResult], Swift.Error>) -> Void
  ) {
    let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !apiKey.isEmpty else {
      completion(.failure(BraveSearchError.missingAPIKey))
      return
    }
    guard !normalizedQuery.isEmpty, options.pages > 0, options.maxResults > 0 else {
      completion(.success([]))
      return
    }
    searchPage(
      query: normalizedQuery, options: options, page: 1, results: [], seenURLs: [],
      completion: completion)
  }

  private func searchPage(
    query: String,
    options: BraveSearchOptions,
    page: Int,
    results: [SearchResult],
    seenURLs: Set<String>,
    completion: @escaping (Result<[SearchResult], Swift.Error>) -> Void
  ) {
    let request: URLRequest
    do {
      request = try Self.makeRequest(
        endpoint: Self.endpoint, apiKey: apiKey, query: query, page: page,
        options: options)
    } catch {
      completion(.failure(error))
      return
    }

    httpTransport.data(for: request) { result in
      do {
        let (data, response) = try result.get()
        let decodedBody = DuckDuckGoSearch.decodeBody(data)
        guard (200..<300).contains(response.statusCode) else {
          throw WebSearchError.httpStatus(
            response.statusCode, response.url, body: decodedBody,
            headers: response.allHeaderFields.reduce(into: [String: String]()) {
              guard let key = $1.key as? String else { return }
              $0[key] = String(describing: $1.value)
            }, byteCount: data.count)
        }
        let responseBody = try JSONDecoder().decode(ResponseBody.self, from: data)
        let pageResults = responseBody.web?.results ?? []
        var updatedResults = results
        var updatedSeenURLs = seenURLs
        for result in pageResults {
          guard let rawURL = result.url, let url = URL(string: rawURL),
            let rawTitle = result.title
          else { continue }
          let key = url.absoluteString
          guard !updatedSeenURLs.contains(key) else { continue }
          let title = normalizeWhitespace(rawTitle)
          guard !title.isEmpty else { continue }
          updatedSeenURLs.insert(key)
          updatedResults.append(
            SearchResult(
              rank: updatedResults.count + 1, title: title, url: url,
              displayURL: url.host ?? url.absoluteString,
              snippet: normalizeWhitespace(result.description ?? ""), source: "brave"))
          if updatedResults.count >= options.maxResults { break }
        }

        guard page < min(options.pages, 10), updatedResults.count < options.maxResults,
          !pageResults.isEmpty, responseBody.query?.moreResultsAvailable != false
        else {
          completion(.success(updatedResults))
          return
        }
        self.searchPage(
          query: query, options: options, page: page + 1, results: updatedResults,
          seenURLs: updatedSeenURLs, completion: completion)
      } catch {
        completion(.failure(error))
      }
    }
  }

  static func makeRequest(
    endpoint: URL,
    apiKey: String,
    query: String,
    page: Int,
    options: BraveSearchOptions
  ) throws -> URLRequest {
    guard var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
      throw WebSearchError.invalidURL(endpoint.absoluteString)
    }
    // Brave's offset is measured in pages of `count` results, so keep count fixed across pages.
    var queryItems = [
      URLQueryItem(name: "q", value: query),
      URLQueryItem(name: "count", value: String(min(max(1, options.maxResults), 20))),
      URLQueryItem(name: "offset", value: String(min(max(0, page - 1), 9))),
      URLQueryItem(name: "safesearch", value: options.safeSearch ? "moderate" : "off"),
      URLQueryItem(name: "text_decorations", value: "false"),
      URLQueryItem(name: "result_filter", value: "web"),
    ]
    if let timeFilter = options.timeFilter {
      let freshness: String
      switch timeFilter {
      case .day:
        freshness = "pd"
      case .week:
        freshness = "pw"
      case .month:
        freshness = "pm"
      case .year:
        freshness = "py"
      }
      queryItems.append(URLQueryItem(name: "freshness", value: freshness))
    }
    components.queryItems = queryItems
    components.percentEncodedQuery = components.percentEncodedQuery?.replacingOccurrences(
      of: "+", with: "%2B")
    guard let url = components.url else {
      throw WebSearchError.invalidURL(endpoint.absoluteString)
    }
    var request = URLRequest(url: url)
    request.httpMethod = "GET"
    request.timeoutInterval = options.timeout
    request.setValue(apiKey, forHTTPHeaderField: "X-Subscription-Token")
    request.setValue(options.userAgent, forHTTPHeaderField: "User-Agent")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    return request
  }
}
