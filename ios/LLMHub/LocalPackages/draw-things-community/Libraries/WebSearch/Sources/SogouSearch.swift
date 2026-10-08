import Foundation
import SwiftSoup

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Browser fallback used when Sogou cannot return a search page.
public protocol SogouBrowserSearch {
  func search(
    query: String, options: SogouSearchOptions,
    completion: @escaping (Result<[SearchResult], Error>) -> Void)
}

/// Searches Sogou's web endpoint, falling back to a browser when available.
public struct SogouSearch {
  private let httpTransport: HttpTransport
  private let browserSearch: SogouBrowserSearch?
  private static let endpoint = URL(string: "https://www.sogou.com/web")!

  public static var defaultBrowserSearch: SogouBrowserSearch? {
    #if canImport(WebKit)
      return WebKitSogouSearch()
    #else
      return nil
    #endif
  }

  /// Pass nil for browserSearch to use only HTTP, including on headless servers.
  public init(
    httpTransport: HttpTransport = URLSessionHttpTransport(),
    browserSearch: SogouBrowserSearch? = SogouSearch.defaultBrowserSearch
  ) {
    self.httpTransport = httpTransport
    self.browserSearch = browserSearch
  }

  /// Searches Sogou and calls `completion` with normalized, de-duplicated results.
  public func search(
    query: String,
    options: SogouSearchOptions = SogouSearchOptions(),
    completion: @escaping (Result<[SearchResult], Error>) -> Void
  ) {
    let normalizedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !normalizedQuery.isEmpty, options.pages > 0, options.maxResults > 0 else {
      completion(.success([]))
      return
    }

    searchPage(
      query: normalizedQuery, options: options, page: 1, results: [], seenURLs: []
    ) { result in
      if case .failure(let error) = result,
        WebSearchBrowserFallback.shouldUseBrowser(after: error),
        let browserSearch = self.browserSearch
      {
        browserSearch.search(query: normalizedQuery, options: options, completion: completion)
      } else {
        completion(result)
      }
    }
  }

  /// Searches Sogou with async/await by wrapping the completion-handler API.
  public func search(query: String, options: SogouSearchOptions = SogouSearchOptions())
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
    options: SogouSearchOptions,
    page: Int,
    results: [SearchResult],
    seenURLs: Set<String>,
    completion: @escaping (Result<[SearchResult], Error>) -> Void
  ) {
    let request: URLRequest
    do {
      request = try Self.makeRequest(
        endpoint: Self.endpoint, query: query, page: page, options: options)
    } catch {
      completion(.failure(error))
      return
    }

    httpTransport.data(for: request) { result in
      do {
        let (data, response) = try result.get()
        let decodedBody = DuckDuckGoSearch.decodeBody(data)
        if Self.isAccessChallenge(statusCode: response.statusCode, body: decodedBody) {
          throw WebSearchError.searchBlocked(
            "Sogou returned an access challenge instead of search results.", response.url)
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

        let parsed = try SogouHTMLParser.parse(html: html, baseURL: Self.endpoint)
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

        guard page < options.pages, updatedResults.count < options.maxResults,
          updatedResults.count > results.count
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
    endpoint: URL, query: String, page: Int, options: SogouSearchOptions
  ) throws -> URLRequest {
    guard var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
      throw WebSearchError.invalidURL(endpoint.absoluteString)
    }
    var queryItems = [URLQueryItem(name: "query", value: query)]
    if page > 1 {
      queryItems.append(URLQueryItem(name: "page", value: String(page)))
      queryItems.append(URLQueryItem(name: "ie", value: "utf8"))
    }
    if let timeFilter = options.timeFilter {
      queryItems.append(URLQueryItem(name: "tsn", value: timeFilter.sogouValue))
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
    request.setValue("zh-CN,zh;q=0.9,en;q=0.8", forHTTPHeaderField: "Accept-Language")
    return request
  }

  static func isAccessChallenge(statusCode: Int, body: String?) -> Bool {
    if statusCode == 403 || statusCode == 429 { return true }
    guard let body, let document = try? SwiftSoup.parse(body) else { return false }
    // The live antispider page uses seccodeForm and does not contain "captcha".
    if (try? document.select("form#seccodeForm, #seccodeInput, .verify-img-panel").isEmpty())
      == false
    {
      return true
    }
    // Ignore challenge words quoted by search results and script string tables.
    if (try? document.select("h3.vr-title a, a.vr-title").isEmpty()) == false {
      return false
    }
    let text = ((try? document.body()?.text()) ?? "").lowercased()
    return (text.contains("captcha") && text.contains("sogou"))
      || text.contains("请输入验证码") || text.contains("此验证码用于确认")
  }
}

struct SogouParsedPage {
  var results: [SearchResult]
  var isEmptyResult: Bool
}

enum SogouHTMLParser {
  static func parse(html: String, baseURL: URL) throws -> SogouParsedPage {
    let document = try SwiftSoup.parse(html, baseURL.absoluteString)
    let elements = try document.select("div.vrwrap")
    var results = [SearchResult]()
    for element in elements.array() {
      guard let titleElement = try element.select("h3.vr-title a, a.vr-title").first() else {
        continue
      }
      guard let url = try resultURL(from: element, titleElement: titleElement, baseURL: baseURL)
      else {
        continue
      }
      let title = normalizeWhitespace(try titleElement.text())
      guard !title.isEmpty else {
        continue
      }
      let snippet = normalizeWhitespace(
        try element.select("div.fz-mid, p.star-wiki").first()?.text() ?? "")
      results.append(
        SearchResult(
          rank: results.count + 1,
          title: title,
          url: url,
          displayURL: try displayURL(from: element),
          snippet: snippet,
          source: "sogou"))
    }
    // Only explicit no-results messages are empty; a loading or verification shell is not.
    let isEmptyResult =
      try !document.select(".vrTips .icon_noRes").isEmpty()
      || document.select("p").contains { element in
        let text = try element.text()
        return text == "未找到相关结果" || text == "没有找到相关结果"
          || text.hasPrefix("抱歉，没有找到与") && text.contains("相关的网页")
      }
    return SogouParsedPage(results: results, isEmptyResult: isEmptyResult)
  }

  private static func resultURL(from element: Element, titleElement: Element, baseURL: URL) throws
    -> URL?
  {
    if let metadataURL = try element.select("div.r-sech[data-url]").first()?.attr("data-url"),
      let url = decodeSogouURL(metadataURL, baseURL: baseURL)
    {
      return url
    }
    return decodeSogouURL(try titleElement.attr("href"), baseURL: baseURL)
  }

  static func decodeSogouURL(_ rawHref: String, baseURL: URL) -> URL? {
    let trimmed = rawHref.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
      return nil
    }
    let candidate =
      trimmed.hasPrefix("//")
      ? URL(string: "https:" + trimmed) : URL(string: trimmed, relativeTo: baseURL)?.absoluteURL
    guard let url = candidate, ["http", "https"].contains(url.scheme?.lowercased() ?? "") else {
      return nil
    }
    return url
  }

  private static func displayURL(from element: Element) throws -> String {
    let citationTexts = try element.select("a.citeLinkClass span").array().map {
      normalizeWhitespace(try $0.text())
    }.filter { !$0.isEmpty }
    if let visibleURL = citationTexts.first(where: { $0.contains(".") || $0.contains("/") }) {
      return visibleURL
    }
    return citationTexts.joined(separator: " ")
  }
}
