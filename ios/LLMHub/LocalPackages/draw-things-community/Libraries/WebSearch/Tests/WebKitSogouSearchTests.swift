#if canImport(WebKit)
  import Foundation
  import WebKit
  import XCTest

  @testable import WebSearch

  private final class SogouWebView: WKWebView {
    var currentURL: URL?
    var html = "<html>Loading</html>"
    var onLoad: ((URLRequest) -> Void)?
    var onSnapshot: (() -> Void)?
    override var url: URL? { currentURL }

    override func load(_ request: URLRequest) -> WKNavigation? {
      currentURL = request.url
      navigationDelegate?.webView?(self, didStartProvisionalNavigation: nil)
      onLoad?(request)
      navigationDelegate?.webView?(self, didCommit: nil)
      navigationDelegate?.webView?(self, didFinish: nil)
      return nil
    }

    override func evaluateJavaScript(
      _ javaScriptString: String,
      completionHandler: (@MainActor @Sendable (Any?, Error?) -> Void)? = nil
    ) {
      onSnapshot?()
      completionHandler?(html, nil)
    }
  }

  private final class SogouNavigationAction: WKNavigationAction {
    var destination = URLRequest(url: URL(string: "https://www.sogou.com/web")!)
    override var request: URLRequest { destination }
  }

  private struct SogouPolicyInterruption: Error, CustomNSError {
    let url: URL
    static var errorDomain: String { "WebKitErrorDomain" }
    var errorCode: Int { 102 }
    var errorUserInfo: [String: Any] { [NSURLErrorFailingURLErrorKey: url] }
  }

  final class WebKitSogouSearchTests: XCTestCase {
    private let results = """
      <div class="vrwrap"><h3 class="vr-title"><a href="https://example.com/doc">Example</a></h3><div class="fz-mid">Snippet</div></div>
      """
    private let challenge = "<form id='seccodeForm'><div id='seccodeInput'></div></form>"

    func testSeparatePagesDeduplicateRankAndPreserveOptions() {
      let done = expectation(description: "paginated results")
      var pages = [String]()
      let browser = WebKitSogouSearch(makeWebView: { configuration in
        XCTAssertTrue(configuration.websiteDataStore.isPersistent)
        let webView = SogouWebView(frame: .zero, configuration: configuration)
        webView.onLoad = { [weak webView] request in
          let items = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!.queryItems!
          let page = items.first(where: { $0.name == "page" })?.value ?? "1"
          pages.append(page)
          XCTAssertEqual(items.first(where: { $0.name == "query" })?.value, "C++ 搜索")
          XCTAssertEqual(items.first(where: { $0.name == "tsn" })?.value, "2")
          XCTAssertNil(request.value(forHTTPHeaderField: "User-Agent"))
          XCTAssertEqual(request.cachePolicy, .reloadIgnoringLocalCacheData)
          webView?.html = self.results
          if page == "2" {
            webView?.html += self.results.replacingOccurrences(of: "/doc", with: "/second")
          }
        }
        return webView
      })
      browser.search(
        query: " C++ 搜索 ", options: SogouSearchOptions(timeFilter: .week, maxResults: 2, pages: 3)
      ) { result in
        XCTAssertEqual(try? result.get().map(\.rank), [1, 2])
        XCTAssertEqual(try? result.get().map(\.url.path), ["/doc", "/second"])
        done.fulfill()
      }
      wait(for: [done], timeout: 5)
      XCTAssertEqual(pages, ["1", "2"])
    }

    func testDuplicateEmptyAndStalledAdditionalPagesPreserveResults() {
      for nextHTML in [results, "<p>没有找到相关结果</p>", "<html>Loading</html>"] {
        let done = expectation(description: "exhausted page")
        var loads = 0
        let browser = WebKitSogouSearch(makeWebView: { configuration in
          let webView = SogouWebView(frame: .zero, configuration: configuration)
          webView.onLoad = { [weak webView] _ in
            loads += 1
            webView?.html = loads == 1 ? self.results : nextHTML
          }
          return webView
        })
        browser.search(query: "example", options: SogouSearchOptions(pages: 3, timeout: 2)) {
          result in
          XCTAssertEqual(try? result.get().map(\.rank), [1])
          done.fulfill()
        }
        wait(for: [done], timeout: 5)
        XCTAssertEqual(loads, 2)
      }
    }

    func testPaginationChallengeRestartsQueryAndResumesOnSameView() {
      let done = expectation(description: "challenge resumed")
      var pages = [String]()
      var presented = 0
      var dismissed = 0
      var created = 0
      let browser = WebKitSogouSearch(
        challengeHandler: { webView, _ in
          presented += 1
          let webView = webView as! SogouWebView
          webView.currentURL = URL(string: "https://www.sogou.com/antispider/confirmed")!
          webView.html = "<p>验证成功</p>"
          return { dismissed += 1 }
        },
        makeWebView: { configuration in
          created += 1
          let webView = SogouWebView(frame: .zero, configuration: configuration)
          webView.onLoad = { [weak webView] request in
            let page =
              URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!.queryItems!.first(
                where: { $0.name == "page" })?.value ?? "1"
            pages.append(page)
            webView?.html =
              page == "1"
              ? self.results
              : (presented == 0
                ? self.challenge : self.results.replacingOccurrences(of: "/doc", with: "/second"))
          }
          return webView
        })
      browser.search(
        query: "example", options: SogouSearchOptions(maxResults: 10, pages: 2, timeout: 2)
      ) { result in
        XCTAssertEqual(try? result.get().map(\.rank), [1, 2])
        done.fulfill()
      }
      wait(for: [done], timeout: 7)
      XCTAssertEqual(pages, ["1", "2", "1", "2"])
      XCTAssertEqual(created, 1)
      XCTAssertEqual(presented, 1)
      XCTAssertEqual(dismissed, 1)
    }

    func testCancellationCompletesOnceAndReleasesQueueToDuckDuckGo() {
      let cancelled = expectation(description: "cancelled once")
      cancelled.assertForOverFulfill = true
      let next = expectation(description: "next provider")
      var created = 0
      var dismissed = 0
      let browser = WebKitSogouSearch(
        challengeHandler: { _, cancel in
          XCTAssertEqual(created, 1)
          cancel()
          cancel()
          return { dismissed += 1 }
        },
        makeWebView: { configuration in
          created += 1
          let webView = SogouWebView(frame: .zero, configuration: configuration)
          webView.html = self.challenge
          return webView
        })
      let duckDuckGo = WebKitDuckDuckGoSearch(makeWebView: { configuration in
        created += 1
        let webView = SogouWebView(frame: .zero, configuration: configuration)
        webView.html =
          "<article data-testid='result'><a data-testid='result-title-a' href='https://example.com'>Example</a></article>"
        return webView
      })
      browser.search(query: "example", options: SogouSearchOptions()) { result in
        guard case .failure(let error) = result else { return XCTFail("Expected cancellation") }
        XCTAssertEqual((error as? URLError)?.code, .cancelled)
        cancelled.fulfill()
      }
      duckDuckGo.search(query: "next", options: DuckDuckGoSearchOptions()) { result in
        XCTAssertEqual(try? result.get().count, 1)
        next.fulfill()
      }
      wait(for: [cancelled, next], timeout: 5)
      XCTAssertEqual(created, 2)
      XCTAssertEqual(dismissed, 1)
    }

    func testChallengeWithoutPresenterAndChallengeTimeoutFailExplicitly() {
      for hasPresenter in [false, true] {
        let done = expectation(description: "challenge failure")
        var dismissed = 0
        let browser = WebKitSogouSearch(
          challengeTimeout: 1,
          challengeHandler: hasPresenter ? { _, _ in { dismissed += 1 } } : nil,
          makeWebView: { configuration in
            let webView = SogouWebView(frame: .zero, configuration: configuration)
            webView.html = self.challenge
            return webView
          })
        browser.search(query: "example", options: SogouSearchOptions()) { result in
          guard case .failure(WebSearchError.searchBlocked) = result else {
            return XCTFail("Expected challenge failure")
          }
          done.fulfill()
        }
        wait(for: [done], timeout: 4)
        XCTAssertEqual(dismissed, hasPresenter ? 1 : 0)
      }
    }

    func testLoadingAndWrongQueryRetryOnceThenFail() {
      for wrongQuery in [false, true] {
        let done = expectation(description: "no matching results")
        var loads = 0
        let browser = WebKitSogouSearch(makeWebView: { configuration in
          let webView = SogouWebView(frame: .zero, configuration: configuration)
          webView.onLoad = { [weak webView] _ in
            loads += 1
            if wrongQuery {
              webView?.currentURL = URL(string: "https://www.sogou.com/web?query=other")!
              webView?.html = self.results
            }
          }
          return webView
        })
        browser.search(query: "example", options: SogouSearchOptions(timeout: 1)) { result in
          guard case .failure(let error) = result else { return XCTFail("Expected timeout") }
          XCTAssertEqual((error as? URLError)?.code, .timedOut)
          done.fulfill()
        }
        wait(for: [done], timeout: 4)
        XCTAssertEqual(loads, 2)
      }
    }

    func testFormEncodedQueryAndExplicitEmptyResult() {
      let done = expectation(description: "explicit empty result")
      let browser = WebKitSogouSearch(makeWebView: { configuration in
        let webView = SogouWebView(frame: .zero, configuration: configuration)
        webView.onLoad = { [weak webView] _ in
          webView?.currentURL = URL(string: "https://www.sogou.com/web?query=C%2B%2B+reference")!
          webView?.html = "<p>抱歉，没有找到与“C++ reference”相关的网页。</p>"
        }
        return webView
      })
      browser.search(query: "C++ reference", options: SogouSearchOptions()) { result in
        XCTAssertEqual(try? result.get(), [])
        done.fulfill()
      }
      wait(for: [done], timeout: 3)
    }

    func testHTTPChallengeNavigationUpgradesAndExternalNavigationFails() {
      for destination in [
        "http://www.sogou.com/antispider/?from=test", "https://www.sogou.com.example.org/web",
      ] {
        let done = expectation(description: "navigation policy")
        var loads = 0
        let browser = WebKitSogouSearch(makeWebView: { configuration in
          let webView = SogouWebView(frame: .zero, configuration: configuration)
          webView.onLoad = { [weak webView] request in
            loads += 1
            guard let webView else { return }
            if loads == 1 {
              let action = SogouNavigationAction()
              action.destination = URLRequest(url: URL(string: destination)!)
              webView.navigationDelegate?.webView?(webView, decidePolicyFor: action) { policy in
                XCTAssertEqual(policy, .cancel)
                if destination.hasPrefix("http:") {
                  webView.navigationDelegate?.webView?(
                    webView, didFailProvisionalNavigation: nil,
                    withError: SogouPolicyInterruption(url: action.destination.url!))
                }
              }
            } else {
              XCTAssertEqual(request.url!.scheme, "https")
              webView.html = self.challenge
            }
          }
          return webView
        })
        browser.search(query: "example", options: SogouSearchOptions()) { result in
          defer { done.fulfill() }
          if destination.hasPrefix("http:") {
            guard case .failure(WebSearchError.searchBlocked) = result else {
              return XCTFail("Expected reachable challenge")
            }
          } else {
            guard case .failure(WebSearchError.invalidResponse) = result else {
              return XCTFail("Expected rejected host")
            }
          }
        }
        wait(for: [done], timeout: 4)
        XCTAssertEqual(loads, destination.hasPrefix("http:") ? 2 : 1)
      }
    }

    func testUnrelatedPolicyInterruptionIsNotSuppressed() {
      let done = expectation(description: "unexpected policy failure")
      let browser = WebKitSogouSearch(makeWebView: { configuration in
        let webView = SogouWebView(frame: .zero, configuration: configuration)
        webView.onLoad = { [weak webView] request in
          guard let webView else { return }
          webView.navigationDelegate?.webView?(
            webView, didFailProvisionalNavigation: nil,
            withError: SogouPolicyInterruption(url: request.url!))
        }
        return webView
      })
      browser.search(query: "example", options: SogouSearchOptions()) { result in
        defer { done.fulfill() }
        guard case .failure(let error) = result else { return XCTFail("Expected policy failure") }
        XCTAssertEqual((error as NSError).domain, "WebKitErrorDomain")
        XCTAssertEqual((error as NSError).code, 102)
      }
      wait(for: [done], timeout: 3)
    }
  }
#endif
