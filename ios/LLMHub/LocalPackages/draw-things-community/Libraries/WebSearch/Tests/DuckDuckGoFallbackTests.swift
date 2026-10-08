import Foundation
import XCTest

@testable import WebSearch

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

private struct FallbackTransport: HttpTransport {
  var handler: (URLRequest) -> Result<(Data, HTTPURLResponse), Error>
  func data(
    for request: URLRequest, completion: @escaping (Result<(Data, HTTPURLResponse), Error>) -> Void
  ) {
    completion(handler(request))
  }
}

private struct StubBrowserSearch: DuckDuckGoBrowserSearch {
  var handler: (String, DuckDuckGoSearchOptions) -> Result<[SearchResult], Error>
  func search(
    query: String, options: DuckDuckGoSearchOptions,
    completion: @escaping (Result<[SearchResult], Error>) -> Void
  ) {
    completion(handler(query, options))
  }
}

final class DuckDuckGoFallbackTests: XCTestCase {
  private let endpoint = URL(string: "https://html.duckduckgo.com/html/")!
  private let resultHTML = """
    <div class="result web-result"><a class="result__a" href="https://example.com/doc">Example</a></div>
    """

  func testChallengeAndUnexpectedPageUseBrowserExactlyOnce() throws {
    for (status, body) in [
      (202, "<form id='challenge-form'></form>"), (200, "<html>Loading</html>"), (403, "Forbidden"),
      (429, "Slow down"),
    ] {
      var httpCalls = 0
      var browserCalls = 0
      var completed = 0
      let search = DuckDuckGoSearch(
        httpTransport: FallbackTransport { request in
          httpCalls += 1
          return .success(
            (
              Data(body.utf8),
              HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
            ))
        },
        browserSearch: StubBrowserSearch { query, options in
          browserCalls += 1
          XCTAssertEqual(query, "example")
          XCTAssertEqual(options.timeFilter, .week)
          XCTAssertEqual(options.region, "uk-en")
          XCTAssertEqual(options.safeSearch, .strict)
          XCTAssertEqual(options.pages, 2)
          XCTAssertEqual(options.maxResults, 12)
          return .success([
            SearchResult(
              rank: 1, title: "Browser result", url: self.endpoint, displayURL: "example.com",
              snippet: "Snippet", source: "duckduckgo")
          ])
        })
      search.search(
        query: " example ",
        options: DuckDuckGoSearchOptions(
          region: "uk-en", safeSearch: .strict, timeFilter: .week, maxResults: 12, pages: 2)
      ) { result in
        XCTAssertEqual(try? result.get().first?.title, "Browser result")
        completed += 1
      }
      XCTAssertEqual(httpCalls, 1)
      XCTAssertEqual(browserCalls, 1)
      XCTAssertEqual(completed, 1)
    }
  }

  func testSuccessfulAndExplicitEmptyPagesDoNotUseBrowser() throws {
    for body in [resultHTML, "<div class='no-results'><h1>No results found</h1></div>"] {
      let search = DuckDuckGoSearch(
        httpTransport: FallbackTransport { request in
          .success(
            (
              Data(body.utf8),
              HTTPURLResponse(
                url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
            ))
        },
        browserSearch: StubBrowserSearch { _, _ in
          XCTFail("Browser must not run for a valid response")
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

  func testBrowserFailureIsReportedWithoutRetryLoop() {
    var browserCalls = 0
    let search = DuckDuckGoSearch(
      httpTransport: FallbackTransport { _ in .failure(URLError(.timedOut)) },
      browserSearch: StubBrowserSearch { _, _ in
        browserCalls += 1
        return .failure(WebSearchError.searchBlocked("Verification required", nil))
      })
    search.search(query: "example") { result in
      guard case .failure(WebSearchError.searchBlocked) = result else {
        return XCTFail("Expected browser's challenge error")
      }
    }
    XCTAssertEqual(browserCalls, 1)
  }

  func testOfflineCancellationAndInvalidInputDoNotLaunchBrowser() {
    for error in [
      URLError(.cancelled), URLError(.notConnectedToInternet), URLError(.secureConnectionFailed),
    ] {
      let search = DuckDuckGoSearch(
        httpTransport: FallbackTransport { _ in .failure(error) },
        browserSearch: StubBrowserSearch { _, _ in
          XCTFail("Should not launch browser")
          return .success([])
        })
      search.search(query: "example") { result in
        guard case .failure(let received) = result else { return XCTFail("Expected error") }
        XCTAssertEqual((received as? URLError)?.code, error.code)
      }
      search.search(query: "  ") { result in XCTAssertEqual(try? result.get(), []) }
    }
  }

  func testUnrecognizedPageIsNotSuccessfulEmptySearch() {
    let search = DuckDuckGoSearch(
      httpTransport: FallbackTransport { request in
        .success(
          (
            Data("<html>Unexpected page</html>".utf8),
            HTTPURLResponse(
              url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
          ))
      }, browserSearch: nil)
    search.search(query: "example") { result in
      guard case .failure(WebSearchError.unexpectedSearchResponse) = result else {
        return XCTFail("Expected unrecognized search page error")
      }
    }
  }

  func testSecondPageChallengeRestartsWholeQueryInBrowser() {
    var calls = 0
    let html =
      resultHTML
      + "<div class='nav-link'><form><input type='hidden' name='s' value='10'></form></div>"
    let search = DuckDuckGoSearch(
      httpTransport: FallbackTransport { request in
        calls += 1
        let body = calls == 1 ? html : "<form id='challenge-form'></form>"
        return .success(
          (
            Data(body.utf8),
            HTTPURLResponse(
              url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
          ))
      },
      browserSearch: StubBrowserSearch { query, options in
        XCTAssertEqual(query, "example")
        XCTAssertEqual(options.pages, 2)
        return .success([])
      })
    search.search(query: "example", options: DuckDuckGoSearchOptions(pages: 2)) { result in
      XCTAssertEqual(try? result.get(), [])
    }
    XCTAssertEqual(calls, 2)
  }

  func testRenderedBrowserResultsAndNoResults() throws {
    let html = """
      <section data-testid="mainline">
        <article data-testid="result">
          <h2><a data-testid="result-title-a" href="https://example.com/doc">Example Doc</a></h2>
          <a data-testid="result-extras-url-link">example.com › doc</a>
          <div data-result="snippet">Useful <b>documentation</b>.</div>
        </article>
      </section>
      """
    let page = try DuckDuckGoHTMLParser.parse(html: html, baseURL: endpoint)
    XCTAssertEqual(page.results.count, 1)
    XCTAssertEqual(page.results.first?.snippet, "Useful documentation.")
    XCTAssertEqual(page.results.first?.displayURL, "example.com › doc")
    XCTAssertFalse(page.isEmptyResult)
    let empty = try DuckDuckGoHTMLParser.parse(
      html:
        "<section data-testid='mainline'><p>Related searches</p><p>No results found for <b>nonsense</b></p></section>",
      baseURL: endpoint)
    XCTAssertTrue(empty.isEmptyResult)
    let loading = try DuckDuckGoHTMLParser.parse(
      html: "<section data-testid='mainline'>Loading</section>", baseURL: endpoint)
    XCTAssertFalse(loading.isEmptyResult)
  }

  func testChallengeWordingInResultsOrScriptsDoesNotBlockSearch() {
    XCTAssertFalse(
      DuckDuckGoSearch.isAccessChallenge(
        statusCode: 200, body: resultHTML + "<p>Please complete the following challenge</p>"))
    XCTAssertFalse(
      DuckDuckGoSearch.isAccessChallenge(
        statusCode: 200,
        body: "<script>const message = 'Please complete the following challenge';</script>"))
    XCTAssertTrue(
      DuckDuckGoSearch.isAccessChallenge(
        statusCode: 200, body: resultHTML + "<form id='challenge-form'></form>"))
  }

  func testLiteralPlusIsPreservedInQueryAndPagination() throws {
    let request = try DuckDuckGoSearch.makeInitialRequest(
      endpoint: endpoint, query: "C++ & Swift", options: DuckDuckGoSearchOptions())
    XCTAssertTrue(request.url!.absoluteString.contains("C%2B%2B"))
    XCTAssertEqual(
      String(data: DuckDuckGoSearch.formURLEncodedData([("q", "C++ & Swift")]), encoding: .utf8),
      "q=C%2B%2B+%26+Swift")
  }

  func testResultURLsRejectScriptsAndDoNotDecodeLookalikeHosts() {
    XCTAssertNil(DuckDuckGoHTMLParser.decodeDuckDuckGoURL("javascript:alert(1)", baseURL: endpoint))
    let lookalike = "https://notduckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com"
    XCTAssertEqual(
      DuckDuckGoHTMLParser.decodeDuckDuckGoURL(lookalike, baseURL: endpoint)?.host,
      "notduckduckgo.com")
  }
}
