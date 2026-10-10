import Foundation
#if canImport(AppKit) && !canImport(UIKit)
import AppKit

public typealias UIImage = NSImage

extension NSImage {
    public convenience init(cgImage: CGImage) {
        self.init(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }

    public var cgImage: CGImage? {
        cgImage(forProposedRect: nil, context: nil, hints: nil)
    }

    public func jpegData(compressionQuality: CGFloat) -> Data? {
        guard let tiffRepresentation,
              let bitmapImage = NSBitmapImageRep(data: tiffRepresentation) else {
            return nil
        }
        return bitmapImage.representation(using: .jpeg, properties: [.compressionFactor: compressionQuality])
    }

    public func pngData() -> Data? {
        guard let tiffRepresentation,
              let bitmapImage = NSBitmapImageRep(data: tiffRepresentation) else {
            return nil
        }
        return bitmapImage.representation(using: .png, properties: [:])
    }
}

public final class UIApplication: @unchecked Sendable {
    public static let shared = UIApplication()
    private init() {}

    @discardableResult
    public func open(_ url: URL) -> Bool {
        NSWorkspace.shared.open(url)
    }

    public func canOpenURL(_ url: URL) -> Bool {
        return true
    }
}

public final class UIPasteboard: @unchecked Sendable {
    public static let general = UIPasteboard()
    private init() {}

    public var string: String? {
        get {
            NSPasteboard.general.string(forType: .string)
        }
        set {
            NSPasteboard.general.clearContents()
            if let newValue {
                NSPasteboard.general.setString(newValue, forType: .string)
            }
        }
    }
}

import SwiftUI

public struct RenderMessageSegments: View {
    public let displayContent: String
    public init(displayContent: String) {
        self.displayContent = displayContent
    }
    public var body: some View {
        MarkdownTextView(text: displayContent)
    }
}

/// Reads only GGUF header metadata, without mapping or loading model tensors.
enum GGUFLayerLimits {
    static let unknown = 999

    static func read(from url: URL) -> Int? {
        readInteger(from: url, suffix: "block_count", offset: 1)
    }

    static func readContextLength(from url: URL) -> Int? {
        readInteger(from: url, suffix: "context_length", offset: 0)
    }

    private static func readInteger(from url: URL, suffix: String, offset: Int) -> Int? {
        guard let handle = try? FileHandle(forReadingFrom: url),
              let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.uint64Value else {
            return nil
        }
        defer { try? handle.close() }
        do {
            let reader = Reader(handle: handle, size: size)
            guard try reader.bytes(4) == Data("GGUF".utf8),
                  (2...3).contains(try reader.number(4)) else { return nil }
            _ = try reader.number(8) // tensor count
            let keyCount = try reader.number(8)
            guard keyCount <= 1_000_000 else { return nil }
            var architecture: String?
            var metadataValues: [String: UInt64] = [:]
            for _ in 0..<keyCount {
                let key = try reader.string()
                let type = try reader.number(4)
                if key == "general.architecture", type == 8 {
                    architecture = try reader.string()
                } else if key.hasSuffix(".\(suffix)"), type == 4 || type == 10 {
                    metadataValues[key] = try reader.number(type == 4 ? 4 : 8)
                } else {
                    try reader.skipValue(type)
                }
                if let architecture,
                   let count = metadataValues["\(architecture).\(suffix)"],
                   count > 0, count <= UInt64(Int.max - offset) {
                    return Int(count) + offset
                }
            }
            guard let architecture,
                  let count = metadataValues["\(architecture).\(suffix)"],
                  count > 0, count <= UInt64(Int.max - offset) else { return nil }
            return Int(count) + offset
        } catch {
            return nil
        }
    }

    private struct Reader {
        let handle: FileHandle
        let size: UInt64

        func bytes(_ count: Int) throws -> Data {
            let data = try handle.read(upToCount: count) ?? Data()
            guard data.count == count else { throw CocoaError(.fileReadCorruptFile) }
            return data
        }

        func number(_ width: Int) throws -> UInt64 {
            let data = try bytes(width)
            return data.enumerated().reduce(UInt64(0)) { value, byte in
                value | (UInt64(byte.element) << (byte.offset * 8))
            }
        }

        func skip(_ count: UInt64) throws {
            let offset = handle.offsetInFile
            guard offset <= size, count <= size - offset else { throw CocoaError(.fileReadCorruptFile) }
            try handle.seek(toOffset: offset + count)
        }

        func string() throws -> String {
            let length = try number(8)
            guard length <= 1_048_576,
                  let value = String(data: try bytes(Int(length)), encoding: .utf8) else {
                throw CocoaError(.fileReadCorruptFile)
            }
            return value
        }

        func skipValue(_ type: UInt64) throws {
            switch type {
            case 0, 1, 7: try skip(1)
            case 2, 3: try skip(2)
            case 4, 5, 6: try skip(4)
            case 8: try skip(number(8))
            case 10, 11, 12: try skip(8)
            case 9:
                let itemType = try number(4)
                let count = try number(8)
                let width: UInt64
                switch itemType {
                case 0, 1, 7: width = 1
                case 2, 3: width = 2
                case 4, 5, 6: width = 4
                case 10, 11, 12: width = 8
                case 8: width = 0
                default: throw CocoaError(.fileReadCorruptFile)
                }
                if width > 0 {
                    guard count <= (size - handle.offsetInFile) / width else { throw CocoaError(.fileReadCorruptFile) }
                    try skip(count * width)
                } else {
                    guard count <= 1_000_000 else { throw CocoaError(.fileReadCorruptFile) }
                    for _ in 0..<count { try skip(number(8)) }
                }
            default: throw CocoaError(.fileReadCorruptFile)
            }
        }
    }
}

struct WebSearchResult {
    let title: String
    let snippet: String
    let url: String
    let source: String
}

struct URLReaderResult {
    let title: String
    let url: String
    let text: String
    let source: String
    let truncated: Bool
}

actor WebSearchService {
    static let shared = WebSearchService()

    private static let firefoxUA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0"
    private static let maxResponseBytes = 2_000_000
    private static let defaultReaderChars = 8_000
    private static let searchContextChars = 2_500
    private static let cacheLifetime: TimeInterval = 10 * 60

    private struct CachedArticle { let article: URLReaderResult; let savedAt: Date }
    private var articleCache: [String: CachedArticle] = [:]

    private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 12
        config.timeoutIntervalForResource = 20
        config.httpShouldSetCookies = true
        config.httpCookieAcceptPolicy = .always
        return URLSession(configuration: config)
    }()

    func search(query: String, maxResults: Int = 5) async -> [WebSearchResult] {
        let contentResults = await searchWithContent(query: query, maxResults: maxResults)
        if !contentResults.isEmpty { return contentResults }

        if let instantResults = await searchInstantAnswer(query: query), !instantResults.isEmpty {
            return Array(instantResults.prefix(maxResults))
        }

        let htmlResults = await searchHTML(query: query, maxResults: maxResults)
        return htmlResults
    }

    private struct UrlData { let title: String; let url: String }

    private func searchWithContent(query: String, maxResults: Int) async -> [WebSearchResult] {
        if let directURL = extractURL(from: query) {
            if let article = await readURL(directURL, maxChars: Self.defaultReaderChars) {
                return [WebSearchResult(
                    title: article.title,
                    snippet: article.text,
                    url: article.url,
                    source: article.source
                )]
            }
        }

        let urlData = await getSearchURLs(query: query, maxResults: maxResults)
        guard !urlData.isEmpty else { return [] }

        var results: [WebSearchResult] = []
        for item in urlData {
            if results.count >= maxResults { break }
            guard let article = await readURL(item.url, maxChars: Self.searchContextChars) else { continue }
            results.append(WebSearchResult(
                title: article.title.isEmpty ? item.title : article.title,
                snippet: article.text,
                url: article.url,
                source: article.source
            ))
        }
        return results
    }

    private func getSearchURLs(query: String, maxResults: Int) async -> [UrlData] {
        guard let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://duckduckgo.com/html/?q=\(encoded)")
        else { return [] }

        var req = URLRequest(url: url)
        req.setValue(Self.firefoxUA, forHTTPHeaderField: "User-Agent")
        req.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", forHTTPHeaderField: "Accept")
        req.setValue("en-US,en;q=0.5", forHTTPHeaderField: "Accept-Language")

        guard let (data, resp) = try? await session.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let html = String(data: data, encoding: .utf8)
        else { return [] }

        return extractURLsFromHTML(html, maxResults: maxResults)
    }

    private func extractURLsFromHTML(_ html: String, maxResults: Int) -> [UrlData] {
        var results: [UrlData] = []
        let nsHTML = html as NSString
        let range = NSRange(location: 0, length: nsHTML.length)

        if let re = try? NSRegularExpression(
            pattern: #"<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>"#,
            options: .dotMatchesLineSeparators
        ) {
            for m in re.matches(in: html, range: range) {
                if results.count >= maxResults { break }
                let urlStr = nsHTML.substring(with: m.range(at: 1))
                let title  = cleanHTML(nsHTML.substring(with: m.range(at: 2)))
                if isValidContentURL(urlStr), title.count > 5 {
                    results.append(UrlData(title: String(title.prefix(100)), url: urlStr))
                }
            }
        }

        if results.isEmpty,
           let re = try? NSRegularExpression(
               pattern: #"<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>"#,
               options: .dotMatchesLineSeparators
           ) {
            for m in re.matches(in: html, range: range) {
                if results.count >= maxResults { break }
                let urlStr = nsHTML.substring(with: m.range(at: 1))
                let title  = cleanHTML(nsHTML.substring(with: m.range(at: 2)))
                if isValidContentURL(urlStr), title.count > 5 {
                    results.append(UrlData(title: String(title.prefix(100)), url: urlStr))
                }
            }
        }

        return results
    }

    func readURL(_ urlString: String, maxChars: Int = 8_000) async -> URLReaderResult? {
        guard let url = safeWebURL(urlString) else { return nil }
        let cacheKey = url.absoluteString
        if let cached = articleCache[cacheKey], Date().timeIntervalSince(cached.savedAt) < Self.cacheLifetime {
            return URLReaderResult(
                title: cached.article.title, url: cached.article.url,
                text: String(cached.article.text.prefix(maxChars)), source: cached.article.source,
                truncated: cached.article.text.count > maxChars
            )
        }

        var req = URLRequest(url: url)
        req.setValue(Self.firefoxUA, forHTTPHeaderField: "User-Agent")
        req.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")
        req.timeoutInterval = 8

        guard let (data, resp) = try? await session.data(for: req),
              let http = resp as? HTTPURLResponse,
              http.statusCode == 200,
              http.mimeType?.lowercased() == "text/html",
              data.count <= Self.maxResponseBytes,
              let html = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1)
        else { return nil }

        let text = extractTextFromHTML(html, maxChars: Self.defaultReaderChars)
        guard !text.isEmpty else { return nil }
        let finalURL = http.url?.absoluteString ?? url.absoluteString
        let article = URLReaderResult(
            title: extractTitle(from: html).isEmpty ? "Content from \(domain(finalURL))" : extractTitle(from: html),
            url: finalURL, text: text, source: domain(finalURL),
            truncated: text.count >= Self.defaultReaderChars
        )
        articleCache[cacheKey] = CachedArticle(article: article, savedAt: Date())
        return URLReaderResult(title: article.title, url: article.url, text: String(article.text.prefix(maxChars)), source: article.source, truncated: article.text.count > maxChars)
    }

    private func safeWebURL(_ value: String) -> URL? {
        guard let url = URL(string: value), let scheme = url.scheme?.lowercased(),
              ["http", "https"].contains(scheme), let host = url.host?.lowercased(),
              host != "localhost", !host.hasSuffix(".local"), !host.hasPrefix("127."), host != "0.0.0.0", host != "::1"
        else { return nil }
        return url
    }

    private func extractTitle(from html: String) -> String {
        guard let re = try? NSRegularExpression(pattern: #"<title[^>]*>(.*?)</title>"#, options: [.caseInsensitive, .dotMatchesLineSeparators]),
              let match = re.firstMatch(in: html, range: NSRange(html.startIndex..., in: html)),
              let range = Range(match.range(at: 1), in: html)
        else { return "" }
        return String(cleanHTML(String(html[range])).prefix(200))
    }

    private func extractTextFromHTML(_ html: String, maxChars: Int) -> String {
        var s = html
        if let re = try? NSRegularExpression(pattern: "<(script|style)[^>]*>.*?</(script|style)>", options: .dotMatchesLineSeparators) {
            s = re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "")
        }
        if let re = try? NSRegularExpression(pattern: "<(nav|header|footer|aside)[^>]*>.*?</\\1>", options: [.caseInsensitive, .dotMatchesLineSeparators]) {
            s = re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "")
        }
        s = cleanHTML(s)
        let sentences = s.components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { $0.count > 20 && $0.count < 500 && $0.split(separator: " ").count > 4
                   && !$0.lowercased().contains("click")
                   && !$0.lowercased().contains("menu")
                   && !$0.lowercased().contains("navigation") }
        return String(sentences.joined(separator: ". ").prefix(maxChars))
    }

    private func isValidContentURL(_ url: String) -> Bool {
        let lower = url.lowercased()
        return lower.hasPrefix("http")
            && !lower.contains("duckduckgo.com")
            && !lower.contains("javascript:")
            && !lower.contains("#")
            && !lower.contains("privacy")
            && !lower.contains("settings")
            && !lower.contains("ads")
    }

    private func extractURL(from text: String) -> String? {
        let patterns = [
            #"https?://[^\s]+"#,
            #"www\.[^\s]+"#,
        ]
        for pattern in patterns {
            if let re = try? NSRegularExpression(pattern: pattern),
               let m = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) {
                var match = (text as NSString).substring(with: m.range)
                if !match.hasPrefix("http") { match = "https://\(match)" }
                return match
            }
        }
        return nil
    }

    private func searchInstantAnswer(query: String) async -> [WebSearchResult]? {
        guard let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://api.duckduckgo.com/?q=\(encoded)&format=json&no_html=1&skip_disambig=1")
        else { return nil }

        var req = URLRequest(url: url)
        req.setValue("LLM Hub macOS", forHTTPHeaderField: "User-Agent")

        guard let (data, _) = try? await session.data(for: req),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        var results: [WebSearchResult] = []
        if let abstract = json["Abstract"] as? String, !abstract.isEmpty {
            let src = json["AbstractSource"] as? String ?? "DuckDuckGo"
            results.append(WebSearchResult(title: src, snippet: abstract, url: json["AbstractURL"] as? String ?? "", source: src))
        }
        if let def = json["Definition"] as? String, !def.isEmpty {
            let src = json["DefinitionSource"] as? String ?? "DuckDuckGo"
            results.append(WebSearchResult(title: "Definition", snippet: def, url: json["DefinitionURL"] as? String ?? "", source: src))
        }
        if let topics = json["RelatedTopics"] as? [[String: Any]] {
            for topic in topics.prefix(3) {
                if let text = topic["Text"] as? String, !text.isEmpty {
                    results.append(WebSearchResult(title: "Related", snippet: text, url: topic["FirstURL"] as? String ?? "", source: "DuckDuckGo"))
                }
            }
        }
        return results.isEmpty ? nil : results
    }

    private func searchHTML(query: String, maxResults: Int) async -> [WebSearchResult] {
        guard let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://duckduckgo.com/html/?q=\(encoded)")
        else { return [] }

        var req = URLRequest(url: url)
        req.setValue(Self.firefoxUA, forHTTPHeaderField: "User-Agent")
        req.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", forHTTPHeaderField: "Accept")
        req.setValue("en-US,en;q=0.5", forHTTPHeaderField: "Accept-Language")

        guard let (data, resp) = try? await session.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let html = String(data: data, encoding: .utf8)
        else { return [] }

        return parseHTMLResults(html: html, maxResults: maxResults)
    }

    private func parseHTMLResults(html: String, maxResults: Int) -> [WebSearchResult] {
        var results: [WebSearchResult] = []
        let nsHTML = html as NSString
        let range = NSRange(location: 0, length: nsHTML.length)

        let patterns: [(title: String, snippet: String, url: String)] = [
            (#"<a class="result__a"[^>]*>(.*?)</a>"#,
             #"<a class="result__snippet"[^>]*>(.*?)</a>"#,
             #"<a class="result__url"[^>]*href="([^"]*)"[^>]*>"#),
        ]

        for pat in patterns {
            guard results.isEmpty,
                  let tRe = try? NSRegularExpression(pattern: pat.title,   options: .dotMatchesLineSeparators),
                  let sRe = try? NSRegularExpression(pattern: pat.snippet, options: .dotMatchesLineSeparators),
                  let uRe = try? NSRegularExpression(pattern: pat.url,     options: [])
            else { continue }
            let titles   = tRe.matches(in: html, range: range)
            let snippets = sRe.matches(in: html, range: range)
            let urls     = uRe.matches(in: html, range: range)
            let count    = min(titles.count, min(snippets.count, min(urls.count, maxResults)))
            for i in 0..<count {
                let t = cleanHTML(nsHTML.substring(with: titles[i].range(at: 1)))
                let s = cleanHTML(nsHTML.substring(with: snippets[i].range(at: 1)))
                let u = nsHTML.substring(with: urls[i].range(at: 1))
                if !t.isEmpty, !s.isEmpty {
                    results.append(WebSearchResult(title: t, snippet: s, url: u, source: domain(u)))
                }
            }
        }
        return results
    }

    private func cleanHTML(_ raw: String) -> String {
        var s = raw
        if let re = try? NSRegularExpression(pattern: "<[^>]*>") {
            s = re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: " ")
        }
        return s
            .replacingOccurrences(of: "&amp;",  with: "&")
            .replacingOccurrences(of: "&lt;",   with: "<")
            .replacingOccurrences(of: "&gt;",   with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;",  with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func domain(_ urlString: String) -> String {
        URL(string: urlString)?.host?.replacingOccurrences(of: "www.", with: "") ?? ""
    }
}
#endif
