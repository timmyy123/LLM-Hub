import Foundation

/// One search result, cited in the reply as [number]. Same JSON keys as the Windows and Linux apps.
struct WebSource: Codable, Hashable, Identifiable {
    var number: Int
    var title: String
    var url: String
    var snippet: String

    var id: Int { number }
    var host: String { URL(string: url)?.host?.replacingOccurrences(of: "www.", with: "") ?? url }
    var link: URL? { URL(string: url) }

    private enum CodingKeys: String, CodingKey { case number = "Number", title = "Title", url = "Url", snippet = "Snippet" }
}

/// Web search for chat: DuckDuckGo's HTML results (no API key, same approach as the other apps), then the top pages are
/// fetched and reduced to text so the model can answer from them and cite them as [1], [2]. This is the only part of
/// chat that goes online, and only when the user turns it on.
enum WebSearch {
    private static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 12
        // The Android and iOS apps' request: Firefox for Android, which DuckDuckGo's HTML page serves without a bot check
        // far more often than a desktop browser posting the form.
        config.httpAdditionalHeaders = ["User-Agent": "Mozilla/5.0 (Android 10; Mobile; rv:91.0) Gecko/91.0 Firefox/91.0"]
        return URLSession(configuration: config)
    }()

    private static func regex(_ pattern: String, _ options: NSRegularExpression.Options = []) -> NSRegularExpression {
        try! NSRegularExpression(pattern: pattern, options: options)
    }

    private static let resultRegex = regex(#"<a(?<attrs>[^>]*class="result__a"[^>]*)>(?<title>.*?)</a>.*?class="result__snippet"[^>]*>(?<snippet>.*?)</(?:a|div|td)>"#, .dotMatchesLineSeparators)
    private static let hrefRegex = regex(#"href="([^"]+)""#)
    private static let noiseRegex = regex(#"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>"#, [.dotMatchesLineSeparators, .caseInsensitive])
    private static let tagRegex = regex("<[^>]+>")
    private static let whitespaceRegex = regex(#"\s+"#)

    // DuckDuckGo Lite: the fallback when the HTML endpoint is throttled and returns nothing. Links are direct URLs;
    // links and snippets pair up by index.
    private static let liteLinkRegex = regex(#"<a(?=[^>]*class=['"]result-link['"])[^>]*href=['"](?<href>[^'"]+)['"][^>]*>(?<title>.*?)</a>"#, .dotMatchesLineSeparators)
    private static let liteSnippetRegex = regex(#"<td[^>]*class=['"]result-snippet['"][^>]*>(?<snippet>.*?)</td>"#, .dotMatchesLineSeparators)

    static func search(_ query: String, max: Int) async throws -> [WebSource] {
        let html = try await get("https://duckduckgo.com/html/", query)
        var results = parseHtml(html, max: max)
        if results.isEmpty {
            Log.info("Web search \"\(query)\": no results from the HTML endpoint\(isChallenge(html) ? " (bot check)" : ""), trying Lite")
            let lite = try await get("https://lite.duckduckgo.com/lite/", query)
            results = parseLite(lite, max: max)
            // Search engines answer scripted queries with bot checks at times; Wikipedia's API is made for programs.
            if results.isEmpty { results = await searchWikipedia(query, max: max) }
            // Still nothing and DuckDuckGo showed a bot check: say so rather than "0 results".
            if results.isEmpty && isChallenge(html) && isChallenge(lite) {
                Log.info("Web search \"\(query)\": DuckDuckGo is showing a bot check to this network")
                throw NSError(domain: "LLMHub", code: 50, userInfo: [NSLocalizedDescriptionKey: "DuckDuckGo is limiting searches from this network"])
            }
        }
        Log.info("Web search \"\(query)\": \(results.count) results")
        return results
    }

    /// Wikipedia in the app's language (same as the Windows and Linux apps): keywords first, since its search matches every
    /// word, then any keyword (ranked by relevance) if an odd word still blocks a match.
    private static func searchWikipedia(_ query: String, max: Int) async -> [WebSource] {
        let tag = L.override.isEmpty ? (Locale.preferredLanguages.first ?? "en") : L.override
        let language = String(tag.prefix(2)).lowercased()
        let words = keywords(query)
        guard !words.isEmpty else { return [] }
        for search in [words.joined(separator: " "), words.joined(separator: " OR ")] {
            var components = URLComponents(string: "https://\(language).wikipedia.org/w/api.php")!
            components.queryItems = [.init(name: "action", value: "query"), .init(name: "list", value: "search"), .init(name: "format", value: "json"),
                                     .init(name: "srprop", value: "snippet"), .init(name: "srlimit", value: "\(max)"), .init(name: "srsearch", value: search)]
            var request = URLRequest(url: components.url!)
            // Wikimedia asks API clients to identify themselves.
            request.setValue("LLMHub/1.0 (https://llm-hub.app)", forHTTPHeaderField: "User-Agent")
            guard let (data, _) = try? await session.data(for: request),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let hits = (json["query"] as? [String: Any])?["search"] as? [[String: Any]], !hits.isEmpty else { continue }
            Log.info("Web search: answered from \(language).wikipedia.org (\"\(search)\")")
            return hits.prefix(max).enumerated().map { i, hit in
                let title = hit["title"] as? String ?? ""
                let page = "https://\(language).wikipedia.org/wiki/" + (title.replacingOccurrences(of: " ", with: "_").addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? title)
                return WebSource(number: i + 1, title: title, url: page, snippet: clean(hit["snippet"] as? String ?? ""))
            }
        }
        return []
    }

    /// The words of a request that say what it is about, without filler and instructions to the assistant.
    private static func keywords(_ text: String) -> [String] {
        let filler: Set<String> = [
            "search", "web", "internet", "online", "google", "look", "find", "tell", "me", "please", "answer", "reply", "cite", "source", "sources",
            "one", "two", "sentence", "sentences", "word", "words", "short", "briefly", "brief", "explain", "describe", "give", "show", "list",
            "what", "whats", "who", "whom", "which", "when", "where", "why", "how", "is", "are", "was", "were", "be", "been", "do", "does", "did",
            "can", "could", "would", "should", "will", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with", "about", "from",
            "by", "at", "as", "it", "its", "this", "that", "these", "those", "there", "i", "you", "your", "my", "we", "our", "up", "out", "into",
        ]
        var seen = Set<String>()
        return text.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
            .filter { $0.count > 1 && !filler.contains($0) && seen.insert($0).inserted }
            .prefix(8).map { $0 }
    }

    private static func isChallenge(_ html: String) -> Bool { html.contains("anomaly-modal") || html.contains("anomaly.js") }

    private static func get(_ endpoint: String, _ query: String) async throws -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-._~"))
        let q = query.addingPercentEncoding(withAllowedCharacters: allowed) ?? ""
        let (data, response) = try await session.data(from: URL(string: "\(endpoint)?q=\(q)")!)
        try HuggingFaceClient.check(response)
        return String(decoding: data, as: UTF8.self)
    }

    private static func parseHtml(_ html: String, max: Int) -> [WebSource] {
        let ns = html as NSString
        var results: [WebSource] = []
        for m in resultRegex.matches(in: html, range: NSRange(location: 0, length: ns.length)) {
            let attrs = ns.substring(with: m.range(withName: "attrs"))
            guard let h = hrefRegex.firstMatch(in: attrs, range: NSRange(location: 0, length: (attrs as NSString).length)) else { continue }
            let href = decodeEntities((attrs as NSString).substring(with: h.range(at: 1)))
            guard let url = realUrl(href), !url.localizedCaseInsensitiveContains("duckduckgo.com"), !results.contains(where: { $0.url == url }) else { continue }
            results.append(WebSource(number: results.count + 1, title: clean(ns.substring(with: m.range(withName: "title"))), url: url,
                                     snippet: clean(ns.substring(with: m.range(withName: "snippet")))))
            if results.count == max { break }
        }
        return results
    }

    private static func parseLite(_ html: String, max: Int) -> [WebSource] {
        let ns = html as NSString
        let all = NSRange(location: 0, length: ns.length)
        let snippets = liteSnippetRegex.matches(in: html, range: all).map { clean(ns.substring(with: $0.range(withName: "snippet"))) }
        var results: [WebSource] = []
        for (i, m) in liteLinkRegex.matches(in: html, range: all).enumerated() {
            let href = decodeEntities(ns.substring(with: m.range(withName: "href")))
            guard let url = realUrl(href), !url.localizedCaseInsensitiveContains("duckduckgo.com"), !results.contains(where: { $0.url == url }) else { continue }
            results.append(WebSource(number: results.count + 1, title: clean(ns.substring(with: m.range(withName: "title"))), url: url,
                                     snippet: i < snippets.count ? snippets[i] : ""))
            if results.count == max { break }
        }
        return results
    }

    /// Readable text of a page (scripts, styles and tags removed), or "" when it cannot be fetched in time.
    static func fetchText(_ url: String, maxChars: Int) async -> String {
        guard let link = URL(string: url) else { return "" }
        var request = URLRequest(url: link)
        request.timeoutInterval = 8
        guard let (data, response) = try? await session.data(for: request),
              let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
              let type = http.mimeType, type == "text/html" || type == "text/plain" else { return "" }
        let html = String(decoding: data, as: UTF8.self)
        let text = clean(noiseRegex.stringByReplacingMatches(in: html, range: NSRange(location: 0, length: (html as NSString).length), withTemplate: " "))
        return text.count <= maxChars ? text : String(text.prefix(maxChars))
    }

    /// The system note that carries search results into the conversation.
    static func buildContext(_ question: String) async throws -> (context: String, sources: [WebSource]) {
        let query = question.count <= 200 ? question : String(question.prefix(200))
        let sources = try await search(query, max: 5)
        guard !sources.isEmpty else { return ("", sources) }
        let pages = await withTaskGroup(of: (Int, String).self) { group in
            for s in sources.prefix(3) { group.addTask { (s.number, await fetchText(s.url, maxChars: 2500)) } }
            var map: [Int: String] = [:]
            for await (n, text) in group { map[n] = text }
            return map
        }
        try Task.checkCancellation()
        let today = Date().formatted(.iso8601.year().month().day())
        var context = "Web search results for \"\(query)\" (today is \(today)). Use them to answer, cite them inline as [1], [2], and say so if they do not answer the question.\n"
        for s in sources {
            let page = pages[s.number] ?? ""
            context += "\n[\(s.number)] \(s.title)\n\(s.url)\n\(page.count > s.snippet.count ? page : s.snippet)\n"
        }
        return (context, sources)
    }

    /// DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<encoded url>.
    private static func realUrl(_ href: String) -> String? {
        let full = href.hasPrefix("//") ? "https:" + href : href
        guard let components = URLComponents(string: full), let host = components.host else { return nil }
        if host.hasSuffix("duckduckgo.com"), components.path == "/l/" {
            return components.queryItems?.first { $0.name == "uddg" }?.value
        }
        return components.scheme == "http" || components.scheme == "https" ? full : nil
    }

    private static func clean(_ html: String) -> String {
        let noTags = tagRegex.stringByReplacingMatches(in: html, range: NSRange(location: 0, length: (html as NSString).length), withTemplate: " ")
        let decoded = decodeEntities(noTags)
        return whitespaceRegex.stringByReplacingMatches(in: decoded, range: NSRange(location: 0, length: (decoded as NSString).length), withTemplate: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    private static let entityRegex = regex("&(#x?[0-9a-fA-F]+|[a-zA-Z]+);")
    private static let named: [String: String] = ["amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": " ", "#39": "'",
                                                  "mdash": "-", "ndash": "-", "hellip": "…", "rsquo": "'", "lsquo": "'", "rdquo": "\"", "ldquo": "\""]

    private static func decodeEntities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        let ns = text as NSString
        var result = ""
        var last = 0
        for m in entityRegex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            result += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            let body = ns.substring(with: m.range(at: 1))
            if body.hasPrefix("#") {
                let hex = body.dropFirst().lowercased().hasPrefix("x")
                let digits = hex ? String(body.dropFirst(2)) : String(body.dropFirst())
                if let code = UInt32(digits, radix: hex ? 16 : 10), let scalar = Unicode.Scalar(code) {
                    result += String(Character(scalar))
                } else {
                    result += ns.substring(with: m.range)
                }
            } else {
                result += named[body] ?? ns.substring(with: m.range)
            }
            last = m.range.location + m.range.length
        }
        return result + ns.substring(from: last)
    }
}
