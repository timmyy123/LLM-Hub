import Foundation
import SwiftSoup

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Browser fallback used when DuckDuckGo's HTML endpoint cannot return a search page.
public protocol DuckDuckGoBrowserSearch {
  func search(
    query: String, options: DuckDuckGoSearchOptions,
    completion: @escaping (Result<[SearchResult], Error>) -> Void)
}

/// Searches DuckDuckGo's HTML endpoint, falling back to a browser when available.
public struct DuckDuckGoSearch {
  private let httpTransport: HttpTransport
  private let browserSearch: DuckDuckGoBrowserSearch?
  private static let endpoint = URL(string: "https://html.duckduckgo.com/html/")!

  public static var defaultBrowserSearch: DuckDuckGoBrowserSearch? {
    #if canImport(WebKit)
      return WebKitDuckDuckGoSearch()
    #else
      return nil
    #endif
  }

  /// Pass `nil` for `browserSearch` to use only HTTP (for example on a headless server).
  public init(
    httpTransport: HttpTransport = URLSessionHttpTransport(),
    browserSearch: DuckDuckGoBrowserSearch? = DuckDuckGoSearch.defaultBrowserSearch
  ) {
    self.httpTransport = httpTransport
    self.browserSearch = browserSearch
  }

  /// Searches DuckDuckGo and calls `completion` with normalized, de-duplicated results.
  public func search(
    query: String,
    options: DuckDuckGoSearchOptions = DuckDuckGoSearchOptions(),
    completion: @escaping (Result<[SearchResult], Error>) -> Void
  ) {
    let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !normalizedQuery.isEmpty, options.pages > 0, options.maxResults > 0 else {
      completion(.success([]))
      return
    }

    searchPage(
      query: normalizedQuery, options: options, page: 0, nextParameters: nil, results: [],
      seenURLs: []
    ) { result in
      if case .failure(let error) = result, WebSearchBrowserFallback.shouldUseBrowser(after: error),
        let browserSearch = self.browserSearch
      {
        browserSearch.search(query: normalizedQuery, options: options, completion: completion)
      } else {
        completion(result)
      }
    }
  }

  /// Searches DuckDuckGo with async/await by wrapping the completion-handler API.
  public func search(query: String, options: DuckDuckGoSearchOptions = DuckDuckGoSearchOptions())
    async throws -> [SearchResult]
  {
    try await withCheckedThrowingContinuation { continuation in
      search(query: query, options: options) { result in
        continuation.resume(with: result)
      }
    }
  }

  private func searchPage(
    query: String,
    options: DuckDuckGoSearchOptions,
    page: Int,
    nextParameters: [(String, String)]?,
    results: [SearchResult],
    seenURLs: Set<String>,
    completion: @escaping (Result<[SearchResult], Error>) -> Void
  ) {
    let request: URLRequest
    do {
      if page == 0 {
        request = try Self.makeInitialRequest(
          endpoint: Self.endpoint, query: query, options: options)
      } else if let nextParameters {
        request = Self.makeNextRequest(
          endpoint: Self.endpoint, parameters: nextParameters, options: options)
      } else {
        completion(.success(results))
        return
      }
    } catch {
      completion(.failure(error))
      return
    }

    httpTransport.data(for: request) { result in
      do {
        let (data, response) = try result.get()
        let decodedBody = Self.decodeBody(data)
        if Self.isAccessChallenge(statusCode: response.statusCode, body: decodedBody) {
          throw WebSearchError.searchBlocked(
            "DuckDuckGo returned an access challenge instead of search results.", response.url)
        }
        guard (200..<300).contains(response.statusCode) else {
          throw WebSearchError.httpStatus(
            response.statusCode, response.url, body: decodedBody,
            headers: response.allHeaderFields.reduce(into: [String: String]()) {
              guard let key = $1.key as? String else { return }
              $0[key] = String(describing: $1.value)
            }, byteCount: data.count)
        }
        guard let html = decodedBody else {
          throw WebSearchError.bodyDecodingFailed(response.url)
        }

        let parsed = try DuckDuckGoHTMLParser.parse(html: html, baseURL: Self.endpoint)
        guard !parsed.results.isEmpty || parsed.isEmptyResult else {
          throw WebSearchError.unexpectedSearchResponse(response.url)
        }
        var updatedResults = results
        var updatedSeenURLs = seenURLs
        for result in parsed.results {
          let key = result.url.absoluteString
          guard !updatedSeenURLs.contains(key) else {
            continue
          }
          updatedSeenURLs.insert(key)
          updatedResults.append(
            SearchResult(
              rank: updatedResults.count + 1,
              title: result.title,
              url: result.url,
              displayURL: result.displayURL,
              snippet: result.snippet,
              source: result.source))
          if updatedResults.count >= options.maxResults {
            break
          }
        }

        let nextPage = page + 1
        guard nextPage < options.pages, updatedResults.count < options.maxResults,
          parsed.nextParameters != nil
        else {
          completion(.success(updatedResults))
          return
        }
        self.searchPage(
          query: query, options: options, page: nextPage,
          nextParameters: parsed.nextParameters, results: updatedResults,
          seenURLs: updatedSeenURLs, completion: completion)
      } catch {
        completion(.failure(error))
      }
    }
  }

  static func makeInitialRequest(
    endpoint: URL, query: String, options: DuckDuckGoSearchOptions
  ) throws -> URLRequest {
    guard var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
      throw WebSearchError.invalidURL(endpoint.absoluteString)
    }
    var queryItems = [
      URLQueryItem(name: "q", value: query),
      URLQueryItem(name: "kl", value: options.region),
      URLQueryItem(name: "kp", value: options.safeSearch.duckDuckGoValue),
    ]
    if let timeFilter = options.timeFilter {
      queryItems.append(URLQueryItem(name: "df", value: timeFilter.duckDuckGoValue))
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
    request.setValue(options.userAgent, forHTTPHeaderField: "User-Agent")
    request.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
    return request
  }

  static func makeNextRequest(
    endpoint: URL, parameters: [(String, String)], options: DuckDuckGoSearchOptions
  ) -> URLRequest {
    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.timeoutInterval = options.timeout
    request.setValue(options.userAgent, forHTTPHeaderField: "User-Agent")
    request.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
    request.setValue(
      "application/x-www-form-urlencoded; charset=utf-8", forHTTPHeaderField: "Content-Type")
    request.httpBody = Self.formURLEncodedData(parameters)
    return request
  }

  static func formURLEncodedData(_ parameters: [(String, String)]) -> Data {
    var components = URLComponents()
    components.queryItems = parameters.map { URLQueryItem(name: $0.0, value: $0.1) }
    let encoded =
      components.percentEncodedQuery?
      .replacingOccurrences(of: "+", with: "%2B")
      .replacingOccurrences(of: "%20", with: "+") ?? ""
    return Data(encoded.utf8)
  }

  static func decodeBody(_ data: Data) -> String? {
    String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1)
  }

  static func isAccessChallenge(statusCode: Int, body: String?) -> Bool {
    guard let body, let document = try? SwiftSoup.parse(body) else {
      return false
    }
    if (try? document.select("form#challenge-form, form[action*=anomaly.js], .anomaly-modal")
      .isEmpty()) == false
    {
      return true
    }
    // A query or snippet may quote the challenge wording. Only use the text heuristic
    // when there are no result links, and ignore scripts containing UI string tables.
    if (try? document.select("a.result__a, a[data-testid=result-title-a]").isEmpty()) == false {
      return false
    }
    let lowercased = ((try? document.body()?.text()) ?? "").lowercased()
    if lowercased.contains("unfortunately, bots use duckduckgo too")
      || lowercased.contains("please complete the following challenge")
      || lowercased.contains("select all squares containing a duck")
    {
      return true
    }
    return statusCode == 202 && lowercased.contains("duckduckgo")
      && lowercased.contains("challenge")
  }
}

struct DuckDuckGoParsedPage {
  var results: [SearchResult]
  var nextParameters: [(String, String)]?
  var isEmptyResult: Bool
}

enum DuckDuckGoHTMLParser {
  static func parse(html: String, baseURL: URL) throws -> DuckDuckGoParsedPage {
    let document = try SwiftSoup.parse(html, baseURL.absoluteString)
    let elements = try document.select("div.result.web-result, article[data-testid=result]")
    var results = [SearchResult]()
    for element in elements {
      guard
        let titleElement = try element.select("a.result__a, a[data-testid=result-title-a]").first()
      else {
        continue
      }
      let rawHref = try titleElement.attr("href")
      guard let url = decodeDuckDuckGoURL(rawHref, baseURL: baseURL) else {
        continue
      }
      let title = normalizeWhitespace(try titleElement.text())
      guard !title.isEmpty else {
        continue
      }
      let snippet = normalizeWhitespace(
        try element.select(".result__snippet, [data-result=snippet]").first()?.text() ?? "")
      let displayURL = normalizeWhitespace(
        try element.select("a.result__url, a[data-testid=result-extras-url-link]").first()?.text()
          ?? "")
      results.append(
        SearchResult(
          rank: results.count + 1,
          title: title,
          url: url,
          displayURL: displayURL,
          snippet: snippet,
          source: "duckduckgo"))
    }
    return DuckDuckGoParsedPage(
      results: results, nextParameters: try nextFormParameters(from: document),
      isEmptyResult: try !document.select(".no-results, .no-results__title").isEmpty()
        || document.select("[data-testid=mainline] p").contains(where: {
          try $0.text().hasPrefix("No results found for ")
        }))
  }

  static func decodeDuckDuckGoURL(_ rawHref: String, baseURL: URL) -> URL? {
    let trimmed = rawHref.trimmingCharacters(in: .whitespacesAndNewlines)
    let candidate: URL?
    if trimmed.hasPrefix("//") {
      candidate = URL(string: "https:" + trimmed)
    } else {
      candidate = URL(string: trimmed, relativeTo: baseURL)?.absoluteURL
    }
    guard let url = candidate, ["https", "http"].contains(url.scheme?.lowercased() ?? "") else {
      return nil
    }
    guard
      url.host?.lowercased() == "duckduckgo.com",
      url.path == "/l/" || url.path == "/l",
      let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
      let encoded = components.queryItems?.first(where: { $0.name == "uddg" })?.value,
      let decoded = URL(string: encoded),
      ["https", "http"].contains(decoded.scheme?.lowercased() ?? "")
    else {
      return url
    }
    return decoded
  }

  private static func nextFormParameters(from document: Document) throws -> [(String, String)]? {
    guard let form = try document.select("div.nav-link form").first() else {
      return nil
    }
    let inputs = try form.select("input[type=hidden]")
    var parameters = [(String, String)]()
    for input in inputs {
      let name = try input.attr("name")
      guard !name.isEmpty else {
        continue
      }
      parameters.append((name, try input.attr("value")))
    }
    return parameters.isEmpty ? nil : parameters
  }
}

func normalizeWhitespace(_ value: String) -> String {
  value
    .components(separatedBy: .whitespacesAndNewlines)
    .filter { !$0.isEmpty }
    .joined(separator: " ")
}
