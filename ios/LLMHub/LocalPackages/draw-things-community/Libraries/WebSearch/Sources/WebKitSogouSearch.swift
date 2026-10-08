#if canImport(WebKit)
  import Foundation
  import WebKit

  /// Searches the regular Sogou site with JavaScript and persistent website data.
  /// Browser operations are serialized on the main run loop; HTTP transport remains thread-agnostic.
  public struct WebKitSogouSearch: SogouBrowserSearch {
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
      query: String, options: SogouSearchOptions,
      completion: @escaping (Result<[SearchResult], Error>) -> Void
    ) {
      // WebKit requires the main thread even when the caller is a background agent or a CLI.
      DispatchQueue.main.async {
        WebKitSearchSession.shared.enqueue(
          Request(
            query: query.trimmingCharacters(in: .whitespacesAndNewlines), options: options,
            challengeTimeout: self.challengeTimeout,
            challengeHandler: self.challengeHandler, makeWebView: self.makeWebView,
            completion: completion))
      }
    }

    private final class Request: NSObject, WKNavigationDelegate, WebKitSearchRequest {
      let query: String
      let options: SogouSearchOptions
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
      private var stableCount = 0
      private var previousURLs = [URL]()
      private var results = [SearchResult]()
      private var upgradedNavigationURLs = Set<URL>()

      init(
        query: String, options: SogouSearchOptions, challengeTimeout: TimeInterval,
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
          var request = try SogouSearch.makeRequest(
            endpoint: URL(string: "https://www.sogou.com/web")!, query: query, page: 1,
            options: options)
          // Use WebKit's native user agent, cookie handling and navigation headers.
          request.setValue(nil, forHTTPHeaderField: "User-Agent")
          request.cachePolicy = .reloadIgnoringLocalCacheData
          initialRequest = request
          let configuration = WKWebViewConfiguration()
          configuration.websiteDataStore = .default()
          #if os(iOS)
            configuration.defaultWebpagePreferences.preferredContentMode = .desktop
          #endif
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
          } else if page > 1, !results.isEmpty {
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
            if SogouSearch.isAccessChallenge(statusCode: 200, body: html) {
              self.presentChallenge(in: webView)
              return
            }
            // A challenge can navigate; never mistake another page/query for this request's results.
            guard var components = URLComponents(url: url, resolvingAgainstBaseURL: false) else {
              return
            }
            components.percentEncodedQuery = components.percentEncodedQuery?.replacingOccurrences(
              of: "+", with: "%20")
            let queryItems = components.queryItems ?? []
            let responsePage = queryItems.first(where: { $0.name == "page" })?.value ?? "1"
            guard url.path == "/web",
              queryItems.first(where: { $0.name == "query" })?.value == self.query,
              responsePage == String(self.page)
            else {
              // Some verification forms finish on a confirmation URL instead of redirecting
              // back to search. Resume once, using the same browser's updated cookie store.
              if self.presentedChallenge, self.didFinish, !self.resumedAfterChallenge,
                let initialRequest = self.initialRequest
              {
                self.resumedAfterChallenge = true
                self.page = 1
                self.stableCount = 0
                self.previousURLs = []
                self.results = []
                self.deadline = Date().addingTimeInterval(max(1, self.options.timeout))
                webView.load(initialRequest)
              }
              return
            }
            let parsed = try SogouHTMLParser.parse(html: html, baseURL: url)
            if parsed.isEmptyResult && parsed.results.isEmpty {
              self.finish(.success(self.normalizedResults))
              return
            }
            guard !parsed.results.isEmpty else { return }
            self.awaitingChallenge = false
            // Sogou navigates to a separate document per page, unlike DuckDuckGo's
            // cumulative More results list. Wait for stable URLs before appending this page.
            var seen = Set<URL>()
            let pageResults = parsed.results.filter { seen.insert($0.url).inserted }
            let urls = pageResults.map(\.url)
            self.stableCount = urls == self.previousURLs ? self.stableCount + 1 : 0
            self.previousURLs = urls
            guard self.stableCount >= 2 else { return }
            seen = Set(self.results.map(\.url))
            let newResults = pageResults.filter { seen.insert($0.url).inserted }
            self.results.append(contentsOf: newResults)
            guard !newResults.isEmpty, self.results.count < self.options.maxResults,
              self.page < self.options.pages
            else {
              self.finish(.success(self.normalizedResults))
              return
            }
            self.page += 1
            self.previousURLs = []
            self.stableCount = 0
            var request = try SogouSearch.makeRequest(
              endpoint: URL(string: "https://www.sogou.com/web")!, query: self.query,
              page: self.page, options: self.options)
            request.setValue(nil, forHTTPHeaderField: "User-Agent")
            request.cachePolicy = .reloadIgnoringLocalCacheData
            self.deadline = Date().addingTimeInterval(max(1, self.options.timeout))
            self.didCommit = false
            webView.load(request)
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
          "Sogou requires verification. Complete the challenge in the search browser and retry if needed.",
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
        // Keep main-frame navigation on the provider. Challenge resources may use CDNs.
        if navigationAction.targetFrame?.isMainFrame != false {
          let url = navigationAction.request.url
          let host = url?.host?.lowercased() ?? ""
          guard host == "sogou.com" || host == "www.sogou.com" else {
            decisionHandler(.cancel)
            finish(.failure(WebSearchError.invalidResponse))
            return
          }
          // Live Sogou search redirects to an HTTP antispider URL. Load its HTTPS
          // equivalent instead of rejecting verification or submitting it over HTTP.
          if url?.scheme == "http",
            url?.path == "/antispider" || url?.path.hasPrefix("/antispider/") == true,
            var components = url.flatMap({ URLComponents(url: $0, resolvingAgainstBaseURL: false) })
          {
            components.scheme = "https"
            if let upgradedURL = components.url {
              var request = navigationAction.request
              request.url = upgradedURL
              if let url { upgradedNavigationURLs.insert(url) }
              decisionHandler(.cancel)
              webView.load(request)
              return
            }
          }
          guard url?.scheme == "https" else {
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
        handleNavigationFailure(error)
      }

      func webView(
        _ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error
      ) {
        handleNavigationFailure(error)
      }

      private func handleNavigationFailure(_ error: Error) {
        if (error as? URLError)?.code == .cancelled { return }
        // WebKit reports a policy cancellation as legacy WebKitErrorDomain/102 on both
        // platforms. Ignore only a navigation we deliberately replaced with HTTPS.
        let failure = error as NSError
        let failingURL = failure.userInfo[NSURLErrorFailingURLErrorKey] as? URL
        if failure.domain == "WebKitErrorDomain", failure.code == 102,
          let failingURL, upgradedNavigationURLs.remove(failingURL) != nil
        {
          return
        }
        finish(.failure(error))
      }

      func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        finish(.failure(WebSearchError.invalidResponse))
      }
    }
  }
#endif
