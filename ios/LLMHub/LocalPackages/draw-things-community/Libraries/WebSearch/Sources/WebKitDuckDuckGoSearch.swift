#if canImport(WebKit)
  import Foundation
  import WebKit

  /// Searches the regular DuckDuckGo site with JavaScript and persistent website data.
  /// Browser operations are serialized on the main run loop; HTTP transport remains thread-agnostic.
  public struct WebKitDuckDuckGoSearch: DuckDuckGoBrowserSearch {
    /// Called on the main thread with the live challenge view and a cancel action.
    /// Return a dismissal action, or nil if presentation is unavailable. Search resumes automatically.
    public typealias ChallengeHandler = (WKWebView, @escaping () -> Void) -> (() -> Void)?

    private let challengeHandler: ChallengeHandler?
    private let challengeTimeout: TimeInterval
    private var makeWebView: (WKWebViewConfiguration) -> WKWebView = {
      WKWebView(frame: CGRect(x: 0, y: 0, width: 1100, height: 850), configuration: $0)
    }

    public init(challengeTimeout: TimeInterval = 120, challengeHandler: ChallengeHandler? = nil) {
      self.challengeTimeout = challengeTimeout
      self.challengeHandler = challengeHandler
    }

    init(
      challengeTimeout: TimeInterval = 120, challengeHandler: ChallengeHandler? = nil,
      makeWebView: @escaping (WKWebViewConfiguration) -> WKWebView
    ) {
      self.init(challengeTimeout: challengeTimeout, challengeHandler: challengeHandler)
      self.makeWebView = makeWebView
    }

    public func search(
      query: String, options: DuckDuckGoSearchOptions,
      completion: @escaping (Result<[SearchResult], Error>) -> Void
    ) {
      // WebKit requires the main thread even when the caller is a background agent or a CLI.
      DispatchQueue.main.async {
        WebKitSearchSession.shared.enqueue(
          Request(
            query: query, options: options, challengeTimeout: self.challengeTimeout,
            challengeHandler: self.challengeHandler, makeWebView: self.makeWebView,
            completion: completion))
      }
    }

    private final class Request: NSObject, WKNavigationDelegate, WebKitSearchRequest {
      let query: String
      let options: DuckDuckGoSearchOptions
      let challengeTimeout: TimeInterval
      let challengeHandler: ChallengeHandler?
      let makeWebView: (WKWebViewConfiguration) -> WKWebView
      let completion: (Result<[SearchResult], Error>) -> Void
      private var webView: WKWebView?
      private var timer: Timer?
      private var deadline = Date.distantPast
      private var didCommit = false
      private var didFinish = false
      private var evaluating = false
      private var finished = false
      private var presentedChallenge = false
      private var awaitingChallenge = false
      private var resumedAfterChallenge = false
      private var reloaded = false
      private var initialRequest: URLRequest?
      private var dismissChallenge: (() -> Void)?
      private var onFinish: (() -> Void)?
      private var page = 1
      private var previousCount = 0
      private var stableCount = 0
      private var countBeforeNextPage = 0
      private var results = [SearchResult]()

      init(
        query: String, options: DuckDuckGoSearchOptions, challengeTimeout: TimeInterval,
        challengeHandler: ChallengeHandler?,
        makeWebView: @escaping (WKWebViewConfiguration) -> WKWebView,
        completion: @escaping (Result<[SearchResult], Error>) -> Void
      ) {
        self.query = query
        self.options = options
        self.challengeTimeout = challengeTimeout
        self.challengeHandler = challengeHandler
        self.makeWebView = makeWebView
        self.completion = completion
      }

      func start(onFinish: @escaping () -> Void) {
        self.onFinish = onFinish
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
          options.maxResults > 0, options.pages > 0
        else {
          finish(.success([]))
          return
        }
        do {
          var request = try DuckDuckGoSearch.makeInitialRequest(
            endpoint: URL(string: "https://duckduckgo.com/")!, query: query, options: options)
          // Use WebKit's native user agent, cookie handling and navigation headers.
          request.setValue(nil, forHTTPHeaderField: "User-Agent")
          request.cachePolicy = .reloadIgnoringLocalCacheData
          initialRequest = request
          let configuration = WKWebViewConfiguration()
          configuration.websiteDataStore = .default()
          let webView = makeWebView(configuration)
          self.webView = webView
          webView.navigationDelegate = self
          deadline = Date().addingTimeInterval(max(1, options.timeout))
          let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in self?.poll() }
          self.timer = timer
          RunLoop.main.add(timer, forMode: .common)
          webView.load(request)
        } catch {
          finish(.failure(error))
        }
      }

      private func poll() {
        guard !finished else { return }
        if Date() >= deadline {
          if awaitingChallenge {
            finish(.failure(blockedError))
          } else if countBeforeNextPage > 0, !results.isEmpty {
            // An additional page may contain no new unique results. Keep the results already
            // collected, but allow the full timeout so a slow page is not mistaken for exhaustion.
            finish(.success(normalizedResults))
          } else if results.isEmpty, !reloaded, let initialRequest, let webView {
            // The page shell can load while its asynchronous results request fails.
            // Allow one ordinary navigation retry, never a loop around a challenge.
            reloaded = true
            deadline = Date().addingTimeInterval(max(1, options.timeout))
            didCommit = false
            webView.load(initialRequest)
          } else {
            finish(.failure(URLError(.timedOut)))
          }
          return
        }
        guard didCommit, !evaluating, let webView else { return }
        evaluating = true
        webView.evaluateJavaScript("document.documentElement.outerHTML") {
          [weak self] value, error in
          guard let self, !self.finished else { return }
          self.evaluating = false
          guard let html = value as? String, let url = webView.url else { return }
          do {
            if DuckDuckGoSearch.isAccessChallenge(statusCode: 200, body: html) {
              self.presentChallenge(in: webView)
              return
            }
            // A challenge can navigate; never mistake another page/query for this request's results.
            guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
              return
            }
            components.percentEncodedQuery = components.percentEncodedQuery?.replacingOccurrences(
              of: "+", with: "%20")
            guard components.queryItems?.first(where: { $0.name == "q" })?.value == self.query
            else {
              // Some verification forms finish on a confirmation URL instead of redirecting
              // back to search. Resume once, using the same browser's updated cookie store.
              if self.presentedChallenge, self.didFinish, !self.resumedAfterChallenge,
                let initialRequest = self.initialRequest
              {
                self.resumedAfterChallenge = true
                self.page = 1
                self.previousCount = 0
                self.stableCount = 0
                self.countBeforeNextPage = 0
                self.results = []
                self.deadline = Date().addingTimeInterval(max(1, self.options.timeout))
                webView.load(initialRequest)
              }
              return
            }
            let parsed = try DuckDuckGoHTMLParser.parse(html: html, baseURL: url)
            if parsed.isEmptyResult && parsed.results.isEmpty {
              self.finish(.success(self.normalizedResults))
              return
            }
            guard !parsed.results.isEmpty else { return }
            self.awaitingChallenge = false
            var seen = Set<String>()
            self.results = parsed.results.filter { seen.insert($0.url.absoluteString).inserted }
            guard self.results.count > self.countBeforeNextPage else { return }
            // React renders incrementally. Require an unchanged count across several snapshots.
            self.stableCount = self.results.count == self.previousCount ? self.stableCount + 1 : 0
            self.previousCount = self.results.count
            guard self.stableCount >= 2 else { return }
            if self.results.count >= self.options.maxResults || self.page >= self.options.pages {
              self.finish(.success(self.normalizedResults))
              return
            }
            self.evaluating = true
            webView.evaluateJavaScript(
              """
              (() => {
                const button = document.querySelector('#more-results');
                if (!button || button.disabled) return false;
                button.click();
                return true;
              })()
              """
            ) { [weak self] clicked, error in
              guard let self, !self.finished else { return }
              self.evaluating = false
              guard clicked as? Bool == true else {
                self.finish(.success(self.normalizedResults))
                return
              }
              self.page += 1
              self.countBeforeNextPage = self.results.count
              self.stableCount = 0
              self.deadline = Date().addingTimeInterval(max(1, self.options.timeout))
            }
          } catch {
            self.finish(.failure(error))
          }
        }
      }

      private var normalizedResults: [SearchResult] {
        results.prefix(options.maxResults).enumerated().map { index, result in
          var result = result
          result.rank = index + 1
          return result
        }
      }

      private var blockedError: WebSearchError {
        .searchBlocked(
          "DuckDuckGo requires verification. Complete the challenge in the search browser and retry if needed.",
          webView?.url)
      }

      private func presentChallenge(in webView: WKWebView) {
        awaitingChallenge = true
        guard !presentedChallenge else { return }
        presentedChallenge = true
        deadline = Date().addingTimeInterval(max(1, challengeTimeout))
        let dismiss = challengeHandler?(webView) { [weak self] in
          self?.finish(.failure(URLError(.cancelled)))
        }
        // A presenter can cancel synchronously (for example if its scene closes).
        guard !finished else {
          dismiss?()
          return
        }
        dismissChallenge = dismiss
        if dismissChallenge == nil {
          finish(.failure(blockedError))
        }
      }

      private func finish(_ result: Result<[SearchResult], Error>) {
        guard !finished else { return }
        finished = true
        timer?.invalidate()
        timer = nil
        webView?.stopLoading()
        webView?.navigationDelegate = nil
        dismissChallenge?()
        dismissChallenge = nil
        webView = nil
        completion(result)
        onFinish?()
        onFinish = nil
      }

      func webView(_ webView: WKWebView, didCommit navigation: WKNavigation!) {
        didCommit = true
      }

      func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
        didCommit = false
        didFinish = false
      }

      func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        didFinish = true
      }

      func webView(
        _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
      ) {
        // This browser is only for search and verification. Do not follow result links or !bangs.
        if navigationAction.targetFrame?.isMainFrame != false {
          let url = navigationAction.request.url
          let host = url?.host?.lowercased() ?? ""
          guard url?.scheme == "https",
            host == "duckduckgo.com" || host.hasSuffix(".duckduckgo.com")
          else {
            decisionHandler(.cancel)
            finish(.failure(WebSearchError.invalidResponse))
            return
          }
        }
        decisionHandler(.allow)
      }

      func webView(
        _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
        withError error: Error
      ) {
        if (error as? URLError)?.code != .cancelled { finish(.failure(error)) }
      }

      func webView(
        _ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error
      ) {
        if (error as? URLError)?.code != .cancelled { finish(.failure(error)) }
      }

      func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        finish(.failure(WebSearchError.invalidResponse))
      }
    }
  }
#endif
