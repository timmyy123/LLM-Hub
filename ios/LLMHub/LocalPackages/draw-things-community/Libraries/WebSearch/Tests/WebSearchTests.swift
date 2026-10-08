import Foundation
import SwiftSoup
import XCTest

@testable import WebSearch

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

private struct StubHttpTransport: HttpTransport {
  var handler: (URLRequest) -> Result<(Data, HTTPURLResponse), Error>

  func data(
    for request: URLRequest,
    completion: @escaping (Result<(Data, HTTPURLResponse), Error>) -> Void
  ) {
    completion(handler(request))
  }
}

final class WebSearchTests: XCTestCase {
  func testWebSearchProviderConstruction() {
    XCTAssertEqual(WebSearchProvider.duckDuckGo.identifier, "duckduckgo")
    XCTAssertEqual(WebSearchProvider.sogou.identifier, "sogou")
    XCTAssertEqual(WebSearchProvider.kagi(apiKey: "test-key").identifier, "kagi")
    XCTAssertEqual(WebSearchProvider.brave(apiKey: "test-key").identifier, "brave")
    XCTAssertEqual(WebSearchProvider.disabled.identifier, "disabled")
    XCTAssertEqual(
      WebSearchProvider(identifier: "kagi", apiKey: "test-key"),
      .kagi(apiKey: "test-key"))
    XCTAssertEqual(
      WebSearchProvider(identifier: "brave", apiKey: "test-key"),
      .brave(apiKey: "test-key"))
    XCTAssertTrue(WebSearchProvider.allCases.contains(.brave(apiKey: "")))
    XCTAssertNil(WebSearchProvider(identifier: "unknown", apiKey: "test-key"))
  }

  func testDuckDuckGoRedirectDecoding() {
    let raw =
      "//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fscinfu%2FSwiftSoup&rut=abc"
    let decoded = DuckDuckGoHTMLParser.decodeDuckDuckGoURL(
      raw, baseURL: URL(string: "https://html.duckduckgo.com/html/")!)
    XCTAssertEqual(decoded?.absoluteString, "https://github.com/scinfu/SwiftSoup")
  }

  func testSearchResultParsing() throws {
    let html = """
      <html><body>
        <div class="result results_links results_links_deep web-result">
          <h2 class="result__title">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc"> Example Doc </a>
          </h2>
          <a class="result__url"> example.com/doc </a>
          <a class="result__snippet"> A <b>useful</b> result. </a>
        </div>
        <div class="nav-link">
          <form action="/html/" method="post">
            <input type="hidden" name="q" value="example" />
            <input type="hidden" name="s" value="10" />
          </form>
        </div>
      </body></html>
      """
    let parsed = try DuckDuckGoHTMLParser.parse(
      html: html, baseURL: URL(string: "https://html.duckduckgo.com/html/")!)
    XCTAssertEqual(parsed.results.count, 1)
    XCTAssertEqual(parsed.results[0].title, "Example Doc")
    XCTAssertEqual(parsed.results[0].url.absoluteString, "https://example.com/doc")
    XCTAssertEqual(parsed.results[0].displayURL, "example.com/doc")
    XCTAssertEqual(parsed.results[0].snippet, "A useful result.")
    XCTAssertEqual(parsed.nextParameters?.count, 2)
  }

  func testSearchRequestParameters() throws {
    let request = try DuckDuckGoSearch.makeInitialRequest(
      endpoint: URL(string: "https://html.duckduckgo.com/html/")!,
      query: "swift urlsession",
      options: DuckDuckGoSearchOptions(
        region: "us-en", safeSearch: .off, timeFilter: .week, maxResults: 10, pages: 1))
    let components = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)
    let queryItems = Dictionary(
      uniqueKeysWithValues: components!.queryItems!.map { ($0.name, $0.value ?? "") })
    XCTAssertEqual(queryItems["q"], "swift urlsession")
    XCTAssertEqual(queryItems["kl"], "us-en")
    XCTAssertEqual(queryItems["kp"], "-2")
    XCTAssertEqual(queryItems["df"], "w")
    XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
    XCTAssertTrue(request.value(forHTTPHeaderField: "User-Agent")?.contains("Mozilla/5.0") == true)
    XCTAssertTrue(request.value(forHTTPHeaderField: "User-Agent")?.contains("Chrome/") == true)
  }

  func testDuckDuckGoAccessChallengeDetection() async throws {
    let html = """
      <html><body>
        <h1>DuckDuckGo</h1>
        <p>Unfortunately, bots use DuckDuckGo too.</p>
        <p>Please complete the following challenge to confirm this search was made by a human.</p>
        <p>Select all squares containing a duck:</p>
      </body></html>
      """
    let response = HTTPURLResponse(
      url: URL(string: "https://html.duckduckgo.com/html/")!,
      statusCode: 202,
      httpVersion: nil,
      headerFields: ["Content-Type": "text/html"])!
    let search = DuckDuckGoSearch(
      httpTransport: StubHttpTransport { _ in .success((Data(html.utf8), response)) },
      browserSearch: nil)

    do {
      _ = try await search.search(query: "site:github.com ios_system")
      XCTFail("Expected searchBlocked")
    } catch WebSearchError.searchBlocked(let message, let url) {
      XCTAssertEqual(message, "DuckDuckGo returned an access challenge instead of search results.")
      XCTAssertEqual(url?.absoluteString, "https://html.duckduckgo.com/html/")
    } catch {
      XCTFail("Unexpected error: \(error)")
    }
  }

  func testSogouSearchRequestParameters() throws {
    let request = try SogouSearch.makeRequest(
      endpoint: URL(string: "https://www.sogou.com/web")!,
      query: "SwiftSoup GitHub",
      page: 2,
      options: SogouSearchOptions(
        timeFilter: .week, maxResults: 10, pages: 2, timeout: 12))
    let components = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)
    let queryItems = Dictionary(
      uniqueKeysWithValues: components!.queryItems!.map { ($0.name, $0.value ?? "") })
    XCTAssertEqual(queryItems["query"], "SwiftSoup GitHub")
    XCTAssertEqual(queryItems["page"], "2")
    XCTAssertEqual(queryItems["ie"], "utf8")
    XCTAssertEqual(queryItems["tsn"], "2")
    XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
    XCTAssertEqual(
      request.value(forHTTPHeaderField: "Accept-Language"), "zh-CN,zh;q=0.9,en;q=0.8")
  }

  func testSogouSearchResultParsingPrefersMetadataURL() throws {
    let html = """
      <html><body>
        <div class="vrwrap" id="sogou_vr_30000000_wrap_4">
          <h3 class="vr-title">
            <a target="_blank" href="/link?url=wrapped">GitHub - holzschu/<em>ios_system</em></a>
          </h3>
          <div class="fz-mid space-txt base-ellipsis clamp2">
            Drop-in replacement for system() in iOS.
          </div>
          <a class="citeLinkClass" target="_blank" href="/link?url=wrapped">
            <span>GitHub</span>
            <span>https://github.com/h...</span>
            <span>2024-03-23</span>
          </a>
          <div class="r-sech ext_query" data-url="https://github.com/holzschu/ios_system"></div>
        </div>
      </body></html>
      """
    let parsed = try SogouHTMLParser.parse(
      html: html, baseURL: URL(string: "https://www.sogou.com/web")!)
    XCTAssertEqual(parsed.results.count, 1)
    XCTAssertEqual(parsed.results[0].title, "GitHub - holzschu/ios_system")
    XCTAssertEqual(parsed.results[0].url.absoluteString, "https://github.com/holzschu/ios_system")
    XCTAssertEqual(parsed.results[0].displayURL, "https://github.com/h...")
    XCTAssertEqual(parsed.results[0].snippet, "Drop-in replacement for system() in iOS.")
    XCTAssertEqual(parsed.results[0].source, "sogou")
  }

  func testKagiSearchRequestParameters() throws {
    let request = try KagiSearch.makeRequest(
      endpoint: URL(string: "https://kagi.com/api/v1/search")!, apiKey: "test-key",
      query: "swift urlsession", page: 2, limit: 7,
      options: KagiSearchOptions(
        timeFilter: .week, safeSearch: true, maxResults: 10, pages: 2, timeout: 12),
      now: Date(timeIntervalSince1970: 1_768_521_600))
    XCTAssertEqual(request.httpMethod, "POST")
    XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer test-key")
    XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "application/json")
    XCTAssertEqual(request.value(forHTTPHeaderField: "Content-Type"), "application/json")
    XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
    XCTAssertEqual(request.timeoutInterval, 12)

    let body = try XCTUnwrap(request.httpBody)
    let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
    XCTAssertEqual(json["query"] as? String, "swift urlsession")
    XCTAssertEqual(json["workflow"] as? String, "search")
    XCTAssertEqual(json["format"] as? String, "json")
    XCTAssertEqual(json["page"] as? Int, 2)
    XCTAssertEqual(json["limit"] as? Int, 7)
    XCTAssertEqual(json["timeout"] as? Double, 4)
    XCTAssertEqual(json["safe_search"] as? Bool, true)
    let filters = try XCTUnwrap(json["filters"] as? [String: Any])
    XCTAssertEqual(filters["after"] as? String, "2026-01-09")
  }

  func testKagiSearchCompletionAPIUsesTransportDirectly() {
    let body =
      """
      {
        "data": {
          "search": [
            {
              "url": "https://example.com/doc",
              "title": " Example Doc ",
              "snippet": " A useful result. "
            }
          ]
        }
      }
      """
    let response = HTTPURLResponse(
      url: URL(string: "https://kagi.com/api/v1/search")!, statusCode: 200,
      httpVersion: nil, headerFields: ["Content-Type": "application/json"])!
    let search = KagiSearch(
      apiKey: "test-key",
      httpTransport: StubHttpTransport { request in
        XCTAssertEqual(request.httpMethod, "POST")
        return .success((Data(body.utf8), response))
      })
    var didComplete = false

    search.search(query: "example") { result in
      switch result {
      case .success(let results):
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].rank, 1)
        XCTAssertEqual(results[0].title, "Example Doc")
        XCTAssertEqual(results[0].url.absoluteString, "https://example.com/doc")
        XCTAssertEqual(results[0].displayURL, "example.com")
        XCTAssertEqual(results[0].snippet, "A useful result.")
        XCTAssertEqual(results[0].source, "kagi")
      case .failure(let error):
        XCTFail("Unexpected error: \(error)")
      }
      didComplete = true
    }

    XCTAssertTrue(didComplete)
  }

  func testKagiSearchDoesNotRetryUnauthorizedRequest() {
    var requestCount = 0
    let search = KagiSearch(
      apiKey: "test-key",
      httpTransport: StubHttpTransport { request in
        requestCount += 1
        XCTAssertEqual(request.url?.path, "/api/v1/search")
        return .success(
          (
            Data("{}".utf8),
            HTTPURLResponse(
              url: request.url!, statusCode: 401, httpVersion: nil,
              headerFields: ["Content-Type": "application/json"])!
          ))
      })
    var didComplete = false

    search.search(query: "example") { result in
      if case .failure(WebSearchError.httpStatus(let status, _, _, _, _)) = result {
        XCTAssertEqual(status, 401)
      } else {
        XCTFail("Expected HTTP 401")
      }
      didComplete = true
    }

    XCTAssertTrue(didComplete)
    XCTAssertEqual(requestCount, 1)
  }

  func testKagiSearchRequiresAPIKeyBeforeTransport() {
    var didRequest = false
    let search = KagiSearch(
      apiKey: " ",
      httpTransport: StubHttpTransport { _ in
        didRequest = true
        return .failure(KagiSearchError.missingAPIKey)
      })
    var didComplete = false

    search.search(query: "example") { result in
      if case .failure(KagiSearchError.missingAPIKey) = result {
        didComplete = true
      } else {
        XCTFail("Expected missingAPIKey")
      }
    }

    XCTAssertTrue(didComplete)
    XCTAssertFalse(didRequest)
  }

  func testKagiLiveSearchWhenAPIKeyIsProvided() throws {
    guard
      let apiKey = ProcessInfo.processInfo.environment["KAGI_API_KEY"]?
        .trimmingCharacters(in: .whitespacesAndNewlines),
      !apiKey.isEmpty
    else {
      throw XCTSkip("Set KAGI_API_KEY to run the live Kagi smoke test.")
    }
    let completionExpectation = expectation(description: "Kagi search completion")
    var searchResult: Result<[SearchResult], Swift.Error>?

    KagiSearch(apiKey: apiKey).search(
      query: "Draw Things app",
      options: KagiSearchOptions(maxResults: 1, pages: 1, timeout: 20)
    ) { result in
      searchResult = result
      completionExpectation.fulfill()
    }

    wait(for: [completionExpectation], timeout: 30)
    let results: [SearchResult]
    do {
      results = try XCTUnwrap(searchResult).get()
    } catch WebSearchError.httpStatus(let status, _, let body, _, _) {
      XCTFail("Kagi returned HTTP \(status): \(body ?? "no response body")")
      return
    }
    XCTAssertEqual(results.count, 1)
    XCTAssertFalse(results[0].title.isEmpty)
    XCTAssertFalse(results[0].url.absoluteString.isEmpty)
    XCTAssertEqual(results[0].source, "kagi")
  }

  func testBraveSearchRequestParameters() throws {
    for (filter, freshness) in [
      (WebSearchTimeFilter.day, "pd"), (.week, "pw"), (.month, "pm"), (.year, "py"),
    ] {
      let request = try BraveSearch.makeRequest(
        endpoint: URL(string: "https://api.search.brave.com/res/v1/web/search")!,
        apiKey: "test-key", query: "C++ & Swift 日本語", page: 2,
        options: BraveSearchOptions(
          timeFilter: filter, safeSearch: false, maxResults: 25, pages: 2, timeout: 12))
      let components = try XCTUnwrap(
        URLComponents(url: request.url!, resolvingAgainstBaseURL: false))
      let items = Dictionary(
        uniqueKeysWithValues: components.queryItems!.map { ($0.name, $0.value!) })
      XCTAssertEqual(items["q"], "C++ & Swift 日本語")
      XCTAssertTrue(components.percentEncodedQuery?.contains("C%2B%2B") == true)
      XCTAssertEqual(items["count"], "20")
      XCTAssertEqual(items["offset"], "1")
      XCTAssertEqual(items["safesearch"], "off")
      XCTAssertEqual(items["freshness"], freshness)
      XCTAssertEqual(items["text_decorations"], "false")
      XCTAssertEqual(items["result_filter"], "web")
      XCTAssertEqual(request.httpMethod, "GET")
      XCTAssertNil(request.httpBody)
      XCTAssertFalse(request.url!.absoluteString.contains("test-key"))
      XCTAssertEqual(request.value(forHTTPHeaderField: "X-Subscription-Token"), "test-key")
      XCTAssertEqual(request.value(forHTTPHeaderField: "Accept"), "application/json")
      XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
      XCTAssertEqual(request.timeoutInterval, 12)
    }
    XCTAssertEqual(BraveSearchOptions(maxResults: 1_000).maxResults, 200)
    XCTAssertEqual(BraveSearchOptions(pages: 100).pages, 10)
  }

  func testBraveSearchPaginationDeduplicatesAndKeepsPageSize() throws {
    var offsets = [String]()
    let search = BraveSearch(
      apiKey: " test-key\n",
      httpTransport: StubHttpTransport { request in
        let components = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!
        let items = Dictionary(
          uniqueKeysWithValues: components.queryItems!.map { ($0.name, $0.value!) })
        offsets.append(items["offset"]!)
        XCTAssertEqual(items["count"], "3")
        XCTAssertEqual(items["q"], "example")
        XCTAssertEqual(items["safesearch"], "moderate")
        XCTAssertNil(items["freshness"])
        XCTAssertEqual(request.value(forHTTPHeaderField: "X-Subscription-Token"), "test-key")
        let body =
          offsets.count == 1
          ? """
          {"query":{"more_results_available":true},"web":{"results":[
            {"url":"https://example.com/1","title":" First ","description":" A   result. "},
            {"url":"https://example.com/1","title":"Duplicate"},
            {"url":"https://example.com/blank","title":" "},
            {"title":"Missing URL"}
          ]}}
          """
          : """
          {"query":{"more_results_available":true},"web":{"results":[
            {"url":"https://example.com/1","title":"Duplicate"},
            {"url":"https://example.com/2","title":"Second"},
            {"url":"https://example.com/3","title":"Third"},
            {"url":"https://example.com/4","title":"Beyond limit"}
          ]}}
          """
        return .success(
          (
            Data(body.utf8),
            HTTPURLResponse(
              url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
          ))
      })
    var result: Result<[SearchResult], Error>?
    search.search(query: " example\n", options: BraveSearchOptions(maxResults: 3, pages: 5)) {
      result = $0
    }
    let results = try XCTUnwrap(result).get()
    XCTAssertEqual(offsets, ["0", "1"])
    XCTAssertEqual(results.map(\.rank), [1, 2, 3])
    XCTAssertEqual(results.map(\.title), ["First", "Second", "Third"])
    XCTAssertEqual(results.map(\.snippet), ["A result.", "", ""])
    XCTAssertEqual(Set(results.map(\.url)).count, 3)
  }

  func testBraveSearchStopsAtLastOrEmptyPage() throws {
    for body in [
      """
      {"query":{"more_results_available":false},"web":{"results":[
        {"url":"https://example.com/doc","title":"Doc"}
      ]}}
      """,
      "{\"web\":{\"results\":[]}}", "{\"query\":{\"more_results_available\":false}}",
    ] {
      var requestCount = 0
      let search = BraveSearch(
        apiKey: "test-key",
        httpTransport: StubHttpTransport { request in
          requestCount += 1
          return .success(
            (
              Data(body.utf8),
              HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            ))
        })
      var result: Result<[SearchResult], Error>?
      search.search(query: "example", options: BraveSearchOptions(pages: 10)) { result = $0 }
      _ = try XCTUnwrap(result).get()
      XCTAssertEqual(requestCount, 1)
    }
  }

  func testBraveSearchPropagatesHTTPFailuresWithoutRetry() {
    for status in [401, 403, 422, 429, 500] {
      var requestCount = 0
      let search = BraveSearch(
        apiKey: "test-key",
        httpTransport: StubHttpTransport { request in
          requestCount += 1
          return .success(
            (
              Data("failure".utf8),
              HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil,
                headerFields: ["Retry-After": "1"])!
            ))
        })
      var didComplete = false
      search.search(query: "example", options: BraveSearchOptions(pages: 5)) { result in
        if case .failure(
          WebSearchError.httpStatus(let code, let url, let body, let headers, let bytes)) = result
        {
          XCTAssertEqual(code, status)
          XCTAssertEqual(url?.host, "api.search.brave.com")
          XCTAssertEqual(body, "failure")
          XCTAssertEqual(headers["Retry-After"], "1")
          XCTAssertEqual(bytes, 7)
        } else {
          XCTFail("Expected HTTP failure")
        }
        didComplete = true
      }
      XCTAssertTrue(didComplete)
      XCTAssertEqual(requestCount, 1)
    }
  }

  func testBraveSearchCompletionAPIUsesTransportDirectly() {
    let body =
      """
      {
        "web": {
          "results": [
            {
              "url": "https://example.com/doc",
              "title": " Example Doc ",
              "description": " A useful result. "
            }
          ]
        }
      }
      """
    let response = HTTPURLResponse(
      url: URL(string: "https://api.search.brave.com/res/v1/web/search")!, statusCode: 200,
      httpVersion: nil, headerFields: ["Content-Type": "application/json"])!
    let search = BraveSearch(
      apiKey: "test-key",
      httpTransport: StubHttpTransport { request in
        XCTAssertEqual(request.httpMethod, "GET")
        return .success((Data(body.utf8), response))
      })
    var didComplete = false

    search.search(query: "example") { result in
      switch result {
      case .success(let results):
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].rank, 1)
        XCTAssertEqual(results[0].title, "Example Doc")
        XCTAssertEqual(results[0].url.absoluteString, "https://example.com/doc")
        XCTAssertEqual(results[0].displayURL, "example.com")
        XCTAssertEqual(results[0].snippet, "A useful result.")
        XCTAssertEqual(results[0].source, "brave")
      case .failure(let error):
        XCTFail("Unexpected error: \(error)")
      }
      didComplete = true
    }

    XCTAssertTrue(didComplete)
  }

  func testBraveSearchPropagatesTransportAndDecodingFailures() {
    for malformedJSON in [false, true] {
      let search = BraveSearch(
        apiKey: "test-key",
        httpTransport: StubHttpTransport { request in
          if !malformedJSON { return .failure(URLError(.cancelled)) }
          return .success(
            (
              Data("not JSON".utf8),
              HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            ))
        })
      var didComplete = false
      search.search(query: "example") { result in
        if case .failure(let error) = result {
          if malformedJSON {
            XCTAssertTrue(error is DecodingError)
          } else {
            XCTAssertEqual((error as? URLError)?.code, .cancelled)
          }
        } else {
          XCTFail("Expected failure")
        }
        didComplete = true
      }
      XCTAssertTrue(didComplete)
    }
  }

  func testBraveSearchEmptyQuerySkipsTransport() throws {
    let search = BraveSearch(
      apiKey: "test-key",
      httpTransport: StubHttpTransport { _ in
        XCTFail("Empty query must not send a request")
        return .failure(URLError(.badURL))
      })
    var result: Result<[SearchResult], Error>?
    search.search(query: " \n ") { result = $0 }
    XCTAssertEqual(try XCTUnwrap(result).get(), [])
  }

  func testBraveSearchRequiresAPIKeyBeforeTransport() {
    var didRequest = false
    let search = BraveSearch(
      apiKey: " ",
      httpTransport: StubHttpTransport { _ in
        didRequest = true
        return .failure(BraveSearchError.missingAPIKey)
      })
    var didComplete = false

    search.search(query: "example") { result in
      if case .failure(BraveSearchError.missingAPIKey) = result {
        didComplete = true
      } else {
        XCTFail("Expected missingAPIKey")
      }
    }

    XCTAssertTrue(didComplete)
    XCTAssertFalse(didRequest)
  }

  func testBraveLiveSearchWhenAPIKeyIsProvided() throws {
    guard
      let apiKey = ProcessInfo.processInfo.environment["BRAVE_API_KEY"]?
        .trimmingCharacters(in: .whitespacesAndNewlines),
      !apiKey.isEmpty
    else {
      throw XCTSkip("Set BRAVE_API_KEY to run the live Brave smoke test.")
    }
    let completionExpectation = expectation(description: "Brave search completion")
    var searchResult: Result<[SearchResult], Swift.Error>?

    BraveSearch(apiKey: apiKey).search(
      query: "Draw Things app",
      options: BraveSearchOptions(maxResults: 1, pages: 1, timeout: 20)
    ) { result in
      searchResult = result
      completionExpectation.fulfill()
    }

    wait(for: [completionExpectation], timeout: 30)
    let results: [SearchResult]
    do {
      results = try XCTUnwrap(searchResult).get()
    } catch WebSearchError.httpStatus(let status, _, let body, _, _) {
      XCTFail("Brave returned HTTP \(status): \(body ?? "no response body")")
      return
    }
    XCTAssertEqual(results.count, 1)
    XCTAssertFalse(results[0].title.isEmpty)
    XCTAssertFalse(results[0].url.absoluteString.isEmpty)
    XCTAssertEqual(results[0].source, "brave")
  }

  func testMarkdownConversionPreservesCommonShapes() throws {
    let document = try SwiftSoup.parse(
      """
      <main>
        <h1>Title</h1>
        <p>See <a href="https://example.com">Example</a>.</p>
        <ul><li>One</li><li>Two</li></ul>
        <pre><code>let x = 1</code></pre>
        <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
      </main>
      """, "https://example.com")
    let markdown = try HTMLMarkdownConverter.convert(try document.select("main").first()!)
    XCTAssertTrue(markdown.contains("# Title"))
    XCTAssertTrue(markdown.contains("[Example](https://example.com)"))
    XCTAssertTrue(markdown.contains("- One"))
    XCTAssertTrue(markdown.contains("```"))
    XCTAssertTrue(markdown.contains("| A | B |"))
  }

  func testWebFetchRenderOutputFormatsHTMLLikeOpenCodeFetch() throws {
    let html = """
      <html>
        <head><title>Ignored</title><script>bad()</script></head>
        <body><main><h1>Title</h1><p>Body <a href="/doc">link</a>.</p></main></body>
      </html>
      """
    let baseURL = URL(string: "https://example.com/root/")!

    let markdown = try WebFetch.renderOutput(
      body: html, contentType: "text/html; charset=utf-8", format: .markdown, baseURL: baseURL)
    XCTAssertTrue(markdown.contains("# Title"))
    XCTAssertTrue(markdown.contains("[link](https://example.com/doc)"))
    XCTAssertFalse(markdown.contains("bad()"))

    let text = try WebFetch.renderOutput(
      body: html, contentType: "text/html; charset=utf-8", format: .text, baseURL: baseURL)
    XCTAssertEqual(text, "Title Body link.")

    let rawHTML = try WebFetch.renderOutput(
      body: html, contentType: "text/html; charset=utf-8", format: .html, baseURL: baseURL)
    XCTAssertTrue(rawHTML.contains("<script>bad()</script>"))
  }

  func testWebFetchResultEncodesOpenCodeToolFields() throws {
    let url = URL(string: "https://example.com")!
    let result = WebFetchResult(
      title: "https://example.com (text/html)",
      metadata: WebFetchMetadata(
        url: url,
        finalURL: url,
        contentType: "text/html",
        statusCode: 200,
        elapsedSeconds: 0.1,
        byteCount: 12,
        format: .markdown),
      output: "Body")
    let data = try JSONEncoder().encode(result)
    let json = String(decoding: data, as: UTF8.self)
    XCTAssertTrue(json.contains("\"title\""))
    XCTAssertTrue(json.contains("\"metadata\""))
    XCTAssertTrue(json.contains("\"output\""))
  }

  func testDuckDuckGoSearchCompletionAPIUsesTransportDirectly() {
    let html = """
      <html><body>
        <div class="result results_links results_links_deep web-result">
          <h2 class="result__title">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc"> Example Doc </a>
          </h2>
          <a class="result__url"> example.com/doc </a>
          <a class="result__snippet"> A useful result. </a>
        </div>
      </body></html>
      """
    let response = HTTPURLResponse(
      url: URL(string: "https://html.duckduckgo.com/html/")!,
      statusCode: 200,
      httpVersion: nil,
      headerFields: ["Content-Type": "text/html"])!
    let search = DuckDuckGoSearch(
      httpTransport: StubHttpTransport { _ in .success((Data(html.utf8), response)) },
      browserSearch: nil)
    var didComplete = false

    search.search(query: "example") { result in
      switch result {
      case .success(let results):
        XCTAssertEqual(results.first?.title, "Example Doc")
        XCTAssertEqual(results.first?.url.absoluteString, "https://example.com/doc")
      case .failure(let error):
        XCTFail("Unexpected error: \(error)")
      }
      didComplete = true
    }

    XCTAssertTrue(didComplete)
  }

  func testSogouSearchCompletionAPIUsesTransportDirectly() {
    let html = """
      <html><body>
        <div class="vrwrap">
          <h3 class="vr-title"><a href="https://example.com/doc">Example Doc</a></h3>
          <div class="fz-mid">A useful result.</div>
        </div>
      </body></html>
      """
    let response = HTTPURLResponse(
      url: URL(string: "https://www.sogou.com/web")!,
      statusCode: 200,
      httpVersion: nil,
      headerFields: ["Content-Type": "text/html"])!
    let search = SogouSearch(
      httpTransport: StubHttpTransport { _ in .success((Data(html.utf8), response)) })
    var didComplete = false

    search.search(query: "example", options: SogouSearchOptions(pages: 1)) { result in
      switch result {
      case .success(let results):
        XCTAssertEqual(results.first?.title, "Example Doc")
        XCTAssertEqual(results.first?.url.absoluteString, "https://example.com/doc")
      case .failure(let error):
        XCTFail("Unexpected error: \(error)")
      }
      didComplete = true
    }

    XCTAssertTrue(didComplete)
  }

  func testWebFetchAsyncAPIWrapsCompletionTransport() async throws {
    let html = "<html><body><h1>Title</h1><p>Body</p></body></html>"
    let url = URL(string: "https://example.com")!
    let response = HTTPURLResponse(
      url: url,
      statusCode: 200,
      httpVersion: nil,
      headerFields: ["Content-Type": "text/html"])!
    let fetch = WebFetch(
      httpTransport: StubHttpTransport { request in
        XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
        XCTAssertEqual(
          request.value(forHTTPHeaderField: "Accept"),
          WebFetchFormat.markdown.acceptHeader)
        return .success((Data(html.utf8), response))
      })

    let result = try await fetch.fetch(url: url)

    XCTAssertEqual(result.title, "https://example.com (text/html)")
    XCTAssertTrue(result.output.contains("# Title"))
    XCTAssertEqual(result.metadata.format, .markdown)
  }

  func testWebFetchFallsBackToDefaultUserAgentWhenEmpty() async throws {
    let html = "<html><body><p>Body</p></body></html>"
    let url = URL(string: "https://example.com")!
    let response = HTTPURLResponse(
      url: url,
      statusCode: 200,
      httpVersion: nil,
      headerFields: ["Content-Type": "text/html"])!
    let fetch = WebFetch(
      httpTransport: StubHttpTransport { request in
        XCTAssertEqual(request.value(forHTTPHeaderField: "User-Agent"), WebSearchDefaultUserAgent)
        return .success((Data(html.utf8), response))
      })

    _ = try await fetch.fetch(url: url, options: WebFetchOptions(userAgent: " "))
  }
}
