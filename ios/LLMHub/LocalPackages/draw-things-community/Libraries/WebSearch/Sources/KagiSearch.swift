import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Errors specific to Kagi search configuration.
public enum KagiSearchError: LocalizedError {
  /// Kagi was selected without an API key.
  case missingAPIKey

  public var errorDescription: String? {
    switch self {
    case .missingAPIKey:
      return "Kagi requires an API key. Add one in Search Provider settings."
    }
  }
}

/// Searches Kagi's v1 Search API.
public struct KagiSearch {
  private struct RequestBody: Encodable {
    struct Filters: Encodable {
      var after: String
    }

    var query: String
    var workflow = "search"
    var format = "json"
    var timeout: TimeInterval
    var page: Int
    var limit: Int
    var filters: Filters?
    var safeSearch: Bool
  }

  private struct ResponseBody: Decodable {
    struct ResponseData: Decodable {
      struct Item: Decodable {
        var type: Int?
        var url: String?
        var title: String?
        var snippet: String?

        private enum CodingKeys: String, CodingKey {
          case type = "t"
          case url
          case title
          case snippet
        }
      }

      var search: [Item]?
    }

    var data: ResponseData?
  }

  private let apiKey: String
  private let httpTransport: HttpTransport
  private static let endpoint = URL(string: "https://kagi.com/api/v1/search")!

  /// Creates a Kagi search tool with an explicit API key.
  public init(apiKey: String, httpTransport: HttpTransport = URLSessionHttpTransport()) {
    self.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
    self.httpTransport = httpTransport
  }

  /// Searches Kagi and calls `completion` with normalized, de-duplicated results.
  public func search(
    query: String,
    options: KagiSearchOptions = KagiSearchOptions(),
    completion: @escaping (Result<[SearchResult], Swift.Error>) -> Void
  ) {
    let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !apiKey.isEmpty else {
      completion(.failure(KagiSearchError.missingAPIKey))
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
    options: KagiSearchOptions,
    page: Int,
    results: [SearchResult],
    seenURLs: Set<String>,
    completion: @escaping (Result<[SearchResult], Swift.Error>) -> Void
  ) {
    let request: URLRequest
    do {
      request = try Self.makeRequest(
        endpoint: Self.endpoint, apiKey: apiKey, query: query, page: page,
        limit: options.maxResults - results.count, options: options)
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
        let pageResults = responseBody.data?.search ?? []
        var updatedResults = results
        var updatedSeenURLs = seenURLs
        for result in pageResults {
          guard result.type == nil || result.type == 0,
            let rawURL = result.url, let url = URL(string: rawURL),
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
              snippet: normalizeWhitespace(result.snippet ?? ""), source: "kagi"))
          if updatedResults.count >= options.maxResults { break }
        }

        guard page < options.pages, updatedResults.count < options.maxResults,
          !pageResults.isEmpty
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
    limit: Int,
    options: KagiSearchOptions,
    now: Date = Date()
  ) throws -> URLRequest {
    let filters = options.timeFilter.map {
      RequestBody.Filters(after: afterDate(for: $0, now: now))
    }
    let body = RequestBody(
      query: query, timeout: min(max(0.5, options.timeout), 4), page: page,
      limit: min(max(1, limit), 1_024), filters: filters, safeSearch: options.safeSearch)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.httpBody = try encoder.encode(body)
    request.timeoutInterval = options.timeout
    request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
    request.setValue(options.userAgent, forHTTPHeaderField: "User-Agent")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    return request
  }

  private static func afterDate(for timeFilter: WebSearchTimeFilter, now: Date) -> String {
    var calendar = Calendar(identifier: .gregorian)
    calendar.timeZone = TimeZone(secondsFromGMT: 0)!
    let component: Calendar.Component
    let value: Int
    switch timeFilter {
    case .day:
      component = .day
      value = -1
    case .week:
      component = .day
      value = -7
    case .month:
      component = .month
      value = -1
    case .year:
      component = .year
      value = -1
    }
    let date = calendar.date(byAdding: component, value: value, to: now) ?? now
    let formatter = DateFormatter()
    formatter.calendar = calendar
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = calendar.timeZone
    formatter.dateFormat = "yyyy-MM-dd"
    return formatter.string(from: date)
  }
}
