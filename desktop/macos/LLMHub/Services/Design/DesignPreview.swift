import AppKit
import WebKit

/// The live preview: one WKWebView for the app's lifetime (so switching tabs keeps the page), the element picker injected
/// into every page, and clicks reported back to the Design tab as the selected element.
@MainActor
final class DesignPreview: NSObject, WKNavigationDelegate {
    let webView: WKWebView
    /// Called with "h1 · hero-title · "…"" when the user clicks an element in the page.
    var onPick: ((String) -> Void)?
    private(set) var shown: URL?
    private(set) var isLoading = false

    override init() {
        let config = WKWebViewConfiguration()
        let controller = WKUserContentController()
        controller.addUserScript(WKUserScript(source: DesignService.pickerScript("window.webkit.messageHandlers.llmhub.postMessage"),
                                              injectionTime: .atDocumentEnd, forMainFrameOnly: true))
        config.userContentController = controller
        webView = WKWebView(frame: NSRect(x: 0, y: 0, width: 800, height: 600), configuration: config)
        super.init()
        controller.add(MessageProxy(self), name: "llmhub")
        webView.navigationDelegate = self
        webView.setAccessibilityIdentifier("DesignPreview")
        // The page is white while it loads, never the window color.
        webView.underPageBackgroundColor = .white
        showBlank()
    }

    /// Loads a version file, or a blank white page; does nothing when that file is already shown.
    func show(_ file: URL?) {
        guard file != shown else { return }
        shown = file
        guard let file, FileManager.default.fileExists(atPath: file.path) else {
            showBlank()
            return
        }
        isLoading = true
        webView.loadFileURL(file, allowingReadAccessTo: file.deletingLastPathComponent())
    }

    /// Reloads the shown file (after it was rewritten).
    func reload() {
        let file = shown
        shown = nil
        show(file)
    }

    private func showBlank() {
        isLoading = true
        webView.loadHTMLString("<html><body style='background:#fff'></body></html>", baseURL: nil)
    }

    /// Waits until the current navigation finishes (or `timeout` passes).
    func waitForLoad(timeout: Duration = .seconds(20)) async {
        guard isLoading else { return }
        let deadline = ContinuousClock.now + timeout
        while isLoading, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(100))
        }
    }

    func evaluate(_ script: String) async -> String {
        do {
            let value = try await webView.evaluateJavaScript(script)
            return value.map { "\($0)" } ?? "null"
        } catch {
            return "error: \(error.localizedDescription)"
        }
    }

    /// The whole page as a PDF (one tall page, as WebKit lays it out at the preview's width).
    func pdf() async throws -> Data {
        try await webView.pdf(configuration: WKPDFConfiguration())
    }

    // MARK: - WKNavigationDelegate

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { isLoading = false }
    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        isLoading = false
        Log.error("Design preview failed to load", error)
    }
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        isLoading = false
        Log.error("Design preview failed to load", error)
    }

    /// Links in a generated page never navigate the preview away (the picker swallows clicks; this covers scripts).
    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
        if action.navigationType == .linkActivated || (action.targetFrame?.isMainFrame == true && action.request.url?.isFileURL == false
            && action.request.url?.absoluteString != "about:blank") {
            return .cancel
        }
        return .allow
    }

    fileprivate func received(_ body: Any) {
        let picked = DesignDocument.describePicked(body)
        onPick?(picked)
    }

    /// WKUserContentController retains its handlers; the proxy keeps that from retaining the preview.
    private final class MessageProxy: NSObject, WKScriptMessageHandler {
        weak var owner: DesignPreview?
        init(_ owner: DesignPreview) { self.owner = owner }

        func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
            let body = message.body
            MainActor.assumeIsolated { owner?.received(body) }
        }
    }
}
