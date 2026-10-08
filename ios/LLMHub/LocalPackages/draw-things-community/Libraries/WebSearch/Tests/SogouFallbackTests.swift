import Foundation
import XCTest

@testable import WebSearch

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

private struct SogouTransport: HttpTransport {
  var handler: (URLRequest) -> Result<(Data, HTTPURLResponse), Error>
  func data(
    for request: URLRequest, completion: @escaping (Result<(Data, HTTPURLResponse), Error>) -> Void
  ) { completion(handler(request)) }
}

private struct StubSogouBrowser: SogouBrowserSearch {
  var handler: (String, SogouSearchOptions) -> Result<[SearchResult], Error>
  func search(
    query: String, options: SogouSearchOptions,
    completion: @escaping (Result<[SearchResult], Error>) -> Void
  ) { completion(handler(query, options)) }
}

final class SogouFallbackTests: XCTestCase {
  private let endpoint = URL(string: "https://www.sogou.com/web")!
  private let resultHTML = """
    <div class="vrwrap"><h3 class="vr-title"><a href="https://example.com/doc">Example</a></h3><div class="fz-mid">Snippet</div></div>
    """
  // Minimal structure from the observed antispider response, without its identifiers or IP.
  private let challengeHTML = """
    <p>此验证码用于确认这些请求是您的正常行为而不是自动程序发出的，需要您协助验证。</p>
    <form id="seccodeForm"><div id="seccodeInput"></div></form>
    """

  func testRecoverableResponsesUseBrowserOnceAndPreserveOptions() {
    for (status, body) in [
      (200, challengeHTML), (200, "Loading"), (403, "Forbidden"), (429, "Slow down"),
      (503, "Unavailable"),
    ] {
      var browserCalls = 0
      let search = SogouSearch(
        httpTransport: SogouTransport { request in
          .success(
            (
              Data(body.utf8),
              HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
            ))
        },
        browserSearch: StubSogouBrowser { query, options in
          browserCalls += 1
          XCTAssertEqual(query, "C++ 搜索")
          XCTAssertEqual(options.pages, 2)
          XCTAssertEqual(options.maxResults, 12)
          XCTAssertEqual(options.timeout, 9)
          XCTAssertEqual(options.timeFilter, .week)
          return .failure(WebSearchError.searchBlocked("Verification required", nil))
        })
      var completed = 0
      search.search(
        query: " C++ 搜索 ",
        options: SogouSearchOptions(timeFilter: .week, maxResults: 12, pages: 2, timeout: 9)
      ) { result in
        guard case .failure(WebSearchError.searchBlocked) = result else {
          return XCTFail("Expected browser failure")
        }
        completed += 1
      }
      XCTAssertEqual(browserCalls, 1)
      XCTAssertEqual(completed, 1)
    }
  }

  func testSuccessfulAndExplicitEmptyResultsStayOnHTTP() {
    for body in [resultHTML, "<p>抱歉，没有找到与“example”相关的网页。</p>"] {
      let search = SogouSearch(
        httpTransport: SogouTransport { request in
          .success(
            (
              Data(body.utf8),
              HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            ))
        },
        browserSearch: StubSogouBrowser { _, _ in
          XCTFail("Unexpected browser fallback")
          return .success([])
        })
      var completed = false
      search.search(query: "example") { result in
        XCTAssertNotNil(try? result.get())
        completed = true
      }
      XCTAssertTrue(completed)
    }
  }

  func testHTTPOnlyReportsChallengeAndUnknownPageInsteadOfEmptySuccess() {
    for body in [challengeHTML, "<html>Loading</html>"] {
      let search = SogouSearch(
        httpTransport: SogouTransport { request in
          .success(
            (
              Data(body.utf8),
              HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            ))
        }, browserSearch: nil)
      search.search(query: "example") { result in
        switch result {
        case .failure(WebSearchError.searchBlocked): XCTAssertEqual(body, self.challengeHTML)
        case .failure(WebSearchError.unexpectedSearchResponse):
          XCTAssertNotEqual(body, self.challengeHTML)
        default: XCTFail("Expected explicit failure")
        }
      }
    }
  }

  func testTransientErrorsFallbackButOfflineCancellationAndTLSDoNot() {
    for code: URLError.Code in [
      .timedOut, .networkConnectionLost, .cancelled, .notConnectedToInternet,
      .secureConnectionFailed,
    ] {
      var browserCalls = 0
      let search = SogouSearch(
        httpTransport: SogouTransport { _ in .failure(URLError(code)) },
        browserSearch: StubSogouBrowser { _, _ in
          browserCalls += 1
          return .success([])
        })
      search.search(query: "example") { _ in }
      XCTAssertEqual(browserCalls, [.timedOut, .networkConnectionLost].contains(code) ? 1 : 0)
      search.search(query: " ") { result in XCTAssertEqual(try? result.get(), []) }
    }
  }

  func testLaterHTTPChallengeRestartsOriginalQueryInBrowser() {
    var requests = 0
    var browserCalls = 0
    let search = SogouSearch(
      httpTransport: SogouTransport { request in
        requests += 1
        return .success(
          (
            Data((requests == 1 ? self.resultHTML : self.challengeHTML).utf8),
            HTTPURLResponse(
              url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
          ))
      },
      browserSearch: StubSogouBrowser { query, options in
        browserCalls += 1
        XCTAssertEqual(query, "example")
        XCTAssertEqual(options.pages, 3)
        return .success([])
      })
    search.search(query: "example", options: SogouSearchOptions(pages: 3)) { _ in }
    XCTAssertEqual(requests, 2)
    XCTAssertEqual(browserCalls, 1)
  }

  func testDuplicateAndEmptyHTTPPagesPreserveResultsAndStopPagination() {
    for nextHTML in [resultHTML, "<p>没有找到相关结果</p>"] {
      var requests = 0
      let search = SogouSearch(
        httpTransport: SogouTransport { request in
          requests += 1
          return .success(
            (
              Data((requests == 1 ? self.resultHTML : nextHTML).utf8),
              HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            ))
        }, browserSearch: nil)
      search.search(query: "example", options: SogouSearchOptions(pages: 3)) { result in
        XCTAssertEqual(try? result.get().map(\.rank), [1])
      }
      XCTAssertEqual(requests, 2)
    }
  }

  func testChallengeDetectionIgnoresResultAndScriptWording() {
    XCTAssertTrue(SogouSearch.isAccessChallenge(statusCode: 200, body: challengeHTML))
    XCTAssertFalse(
      SogouSearch.isAccessChallenge(
        statusCode: 200, body: resultHTML + "<p>Sogou captcha 请输入验证码</p>"))
    XCTAssertFalse(
      SogouSearch.isAccessChallenge(
        statusCode: 200, body: "<script>const message = 'sogou captcha 请输入验证码';</script>"))
    XCTAssertTrue(SogouSearch.isAccessChallenge(statusCode: 200, body: resultHTML + challengeHTML))
  }

  func testLiteralPlusAndResultURLSchemes() throws {
    let request = try SogouSearch.makeRequest(
      endpoint: endpoint, query: "C++ 搜索", page: 2, options: SogouSearchOptions())
    XCTAssertTrue(request.url!.absoluteString.contains("C%2B%2B"))
    XCTAssertNil(SogouHTMLParser.decodeSogouURL("javascript:alert(1)", baseURL: endpoint))
    XCTAssertNil(SogouHTMLParser.decodeSogouURL("data:text/html,test", baseURL: endpoint))
    XCTAssertEqual(
      SogouHTMLParser.decodeSogouURL("//example.com/doc", baseURL: endpoint)?.absoluteString,
      "https://example.com/doc")
    XCTAssertFalse(
      try SogouHTMLParser.parse(
        html: "<input value='未找到相关结果'><script>没有找到相关结果</script>", baseURL: endpoint
      ).isEmptyResult)
  }

  func testObservedSiteSearchEmptyResultIsNotAChallengeOrLoadingPage() throws {
    let html = """
      <p>相关推荐</p><div class="vrTips"><p class="icon_noRes">
      developer.apple.com站内没有找到能和“urlsession swift”匹配的内容。</p></div>
      """
    let parsed = try SogouHTMLParser.parse(html: html, baseURL: endpoint)
    XCTAssertTrue(parsed.isEmptyResult)
    XCTAssertTrue(parsed.results.isEmpty)
    XCTAssertFalse(SogouSearch.isAccessChallenge(statusCode: 200, body: html))
  }
}
