import SwiftUI
import WebKit

#if os(macOS)
public struct WebPreviewView: NSViewRepresentable {
    public let htmlContent: String

    public init(htmlContent: String) {
        self.htmlContent = htmlContent
    }

    public func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "allowFileAccessFromFileURLs")
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.setValue(false, forKey: "drawsBackground")
        return webView
    }

    public func updateNSView(_ nsView: WKWebView, context: Context) {
        nsView.loadHTMLString(htmlContent, baseURL: URL(string: "http://localhost:8080/"))
    }
}
#endif
