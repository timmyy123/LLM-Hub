import AVFoundation
import Foundation
import PhotosUI
import RunAnywhere
import Speech
import SwiftUI
import UIKit
import UniformTypeIdentifiers
import WebKit
#if canImport(FoundationModels)
import FoundationModels
#endif
#if canImport(ImageIO)
import ImageIO
#endif

// MARK: - Web Search Service

private struct WebSearchResult {
    let title: String
    let snippet: String
    let url: String
    let source: String
}

private struct URLReaderResult {
    let title: String
    let url: String
    let text: String
    let source: String
    let truncated: Bool
}

private actor WebSearchService {
    static let shared = WebSearchService()

    // Firefox mobile UA — same as Android build, avoids DuckDuckGo bot blocks
    private static let firefoxUA = "Mozilla/5.0 (Android 10; Mobile; rv:91.0) Gecko/91.0 Firefox/91.0"
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

    // MARK: - Public API

    func search(query: String, maxResults: Int = 5) async -> [WebSearchResult] {
        // 1. Try content-based search (fetches actual page content like Android does)
        let contentResults = await searchWithContent(query: query, maxResults: maxResults)
        if !contentResults.isEmpty { return contentResults }

        // 2. Fallback: DuckDuckGo Instant Answer JSON API
        if let instantResults = await searchInstantAnswer(query: query), !instantResults.isEmpty {
            return Array(instantResults.prefix(maxResults))
        }

        // 3. Fallback: DuckDuckGo HTML scrape
        let htmlResults = await searchHTML(query: query, maxResults: maxResults)
        return htmlResults
    }

    // MARK: - Content-based search (mirrors Android searchWithContent)

    private struct UrlData { let title: String; let url: String }

    private func searchWithContent(query: String, maxResults: Int) async -> [WebSearchResult] {
        // If query contains a URL, fetch that page directly
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

        // Get search result URLs from DuckDuckGo HTML
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

        // Pattern 1: standard DuckDuckGo result__a links
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

        // Pattern 2: any https links with text (generic fallback)
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

    /// Reads and cleans a page on-device. This deliberately does not use Jina or any reader proxy.
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
        // Strip scripts and styles
        if let re = try? NSRegularExpression(pattern: "<(script|style)[^>]*>.*?</(script|style)>", options: .dotMatchesLineSeparators) {
            s = re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "")
        }
        if let re = try? NSRegularExpression(pattern: "<(nav|header|footer|aside)[^>]*>.*?</\\1>", options: [.caseInsensitive, .dotMatchesLineSeparators]) {
            s = re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "")
        }
        s = cleanHTML(s)
        // Keep only meaningful sentences
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

    // MARK: - DuckDuckGo Instant Answer (JSON API)

    private func searchInstantAnswer(query: String) async -> [WebSearchResult]? {
        guard let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://api.duckduckgo.com/?q=\(encoded)&format=json&no_html=1&skip_disambig=1")
        else { return nil }

        var req = URLRequest(url: url)
        req.setValue("LLM Hub iOS", forHTTPHeaderField: "User-Agent")

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

    // MARK: - DuckDuckGo HTML fallback

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
            // Approach 1: DuckDuckGo classic classes
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

    // MARK: - Helpers

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

// MARK: -

private func persistentAttachmentDirectoryURL() -> URL {
    let fileManager = FileManager.default
    let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        ?? fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
        ?? fileManager.temporaryDirectory
    let dir = base.appendingPathComponent("LLMHubAttachments", isDirectory: true)
    try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func resolveStoredAttachmentURL(_ storedPath: String?) -> URL? {
    guard var raw = storedPath?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
        return nil
    }

    if raw.hasPrefix("Optional(\"") && raw.hasSuffix("\")") {
        raw = String(raw.dropFirst("Optional(\"".count).dropLast(2))
    }

    let fm = FileManager.default
    var candidates: [URL] = []

    if let url = URL(string: raw), url.isFileURL {
        candidates.append(url)
    }

    candidates.append(URL(fileURLWithPath: raw))

    for candidate in candidates where fm.fileExists(atPath: candidate.path) {
        return candidate
    }

    let fallbackName = (candidates.last ?? URL(fileURLWithPath: raw)).lastPathComponent
    guard !fallbackName.isEmpty else { return nil }

    let fallbackDirs = [
        persistentAttachmentDirectoryURL(),
        fm.temporaryDirectory.appendingPathComponent("llmhub_attachments", isDirectory: true),
    ]

    for dir in fallbackDirs {
        let candidate = dir.appendingPathComponent(fallbackName)
        if fm.fileExists(atPath: candidate.path) {
            return candidate
        }
    }

    return nil
}

private let chatAppleFoundationModelId = "apple.foundation.system"

@MainActor
private func chatAppleFoundationModelIfAvailable() -> AIModel? {
    #if canImport(FoundationModels)
    if #available(iOS 26.0, *) {
        let model = SystemLanguageModel.default
        guard model.isAvailable else { return nil }

        return AIModel(
            id: chatAppleFoundationModelId,
            name: "Apple Foundation Model",
            description: "On-device Apple Intelligence foundation model.",
            url: "apple://foundation-model",
            category: .text,
            sizeBytes: 0,
            source: "Apple",
            supportsVision: false,
            supportsAudio: false,
            supportsThinking: false,
            supportsGpu: true,
            requirements: ModelRequirements(minRamGB: 8, recommendedRamGB: 8),
            contextWindowSize: max(4096, model.contextSize),
            modelFormat: .platform,
            additionalFiles: []
        )
    }
    #endif

    return nil
}

@MainActor
private func chatModel(named modelName: String) -> AIModel? {
    if let model = ModelData.allModels().first(where: { $0.name == modelName }) {
        return model
    }
    if let appleModel = chatAppleFoundationModelIfAvailable(), appleModel.name == modelName {
        return appleModel
    }
    return nil
}

// MARK: - Chat Mic Transcriber
// Lightweight transcriber for injecting speech into the input field.
@available(iOS 17.0, *)
@MainActor
final class ChatMicTranscriber: NSObject, ObservableObject {
    @Published var liveText: String = ""
    @Published var isRecording: Bool = false
    @Published var isPreparing: Bool = false
    @Published var isTranscribing: Bool = false

    private var baseText: String = ""
    private let engine = SpeechEngine()
    private let speechRecognizer = SFSpeechRecognizer()
    private var silenceTask: Task<Void, Never>?

    func startLive() async {
        guard !isRecording, !isPreparing else { return }
        isPreparing = true
        liveText = ""
        baseText = ""

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            let authStatus = await withCheckedContinuation { c in
                SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0) }
            }
            let micOK = await AVAudioApplication.requestRecordPermission()
            guard authStatus == .authorized && micOK else {
                await MainActor.run { self.isPreparing = false }
                return
            }
            do {
                try await self.engine.start(onResult: { result in
                    let text = result.bestTranscription.formattedString
                    Task { @MainActor [weak self] in
                        guard let self = self else { return }
                        self.liveText = self.baseText + (self.baseText.isEmpty ? "" : " ") + text
                        self.rescheduleSilenceDetection()
                    }
                }, onError: { _ in })
                await MainActor.run {
                    self.isRecording = true
                    self.isPreparing = false
                }
            } catch {
                await MainActor.run { self.isPreparing = false }
            }
        }
    }

    /// Stop live recording and return the final transcript.
    func stopLive() async -> String {
        silenceTask?.cancel()
        silenceTask = nil
        await engine.stop()
        let result = liveText
        await MainActor.run {
            self.isRecording = false
            self.isPreparing = false
            self.liveText = ""
            self.baseText = ""
        }
        return result
    }

    /// Reschedule the silence auto-stop timer. Each new speech result resets the 1.8s countdown.
    private func rescheduleSilenceDetection() {
        guard isRecording else { return }
        silenceTask?.cancel()
        silenceTask = Task { [weak self] in
            do {
                try await Task.sleep(nanoseconds: 1_800_000_000) // 1.8s silence
                guard !Task.isCancelled else { return }
                guard let self = self, self.isRecording else { return }
                _ = await self.stopLive()
            } catch {}
        }
    }

    /// Transcribe an audio file URL and return the resulting text.
    func transcribeFile(_ url: URL) async -> String {
        await MainActor.run { self.isTranscribing = true }
        defer { Task { @MainActor in self.isTranscribing = false } }

        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mic_transcribe_\(UUID().uuidString).\(url.pathExtension.isEmpty ? "m4a" : url.pathExtension)")
        try? FileManager.default.copyItem(at: url, to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        return await withCheckedContinuation { continuation in
            let request = SFSpeechURLRecognitionRequest(url: tempURL)
            request.shouldReportPartialResults = false
            if speechRecognizer?.supportsOnDeviceRecognition == true {
                request.requiresOnDeviceRecognition = true
            }
            speechRecognizer?.recognitionTask(with: request) { result, error in
                if let result, result.isFinal {
                    continuation.resume(returning: result.bestTranscription.formattedString)
                } else if error != nil {
                    continuation.resume(returning: "")
                }
            }
        }
    }
}

// MARK: - Chat ViewModel
@MainActor
class ChatViewModel: ObservableObject {
    private struct ModelGenerationSettings: Codable {
        var maxTokens: Double
        var contextWindow: Double
        var topK: Double
        var topP: Double
        var temperature: Double
        var selectedBackend: String
        var enableVision: Bool
        var enableAudio: Bool
        var enableThinking: Bool
        var enableAgentTools: Bool
        var systemPrompt: String?

        enum CodingKeys: String, CodingKey {
            case maxTokens, contextWindow, topK, topP, temperature, selectedBackend, enableVision, enableAudio, enableThinking, enableAgentTools, systemPrompt
        }

        init(maxTokens: Double, contextWindow: Double, topK: Double, topP: Double, temperature: Double, selectedBackend: String, enableVision: Bool, enableAudio: Bool, enableThinking: Bool, enableAgentTools: Bool, systemPrompt: String?) {
            self.maxTokens = maxTokens
            self.contextWindow = contextWindow
            self.topK = topK
            self.topP = topP
            self.temperature = temperature
            self.selectedBackend = selectedBackend
            self.enableVision = enableVision
            self.enableAudio = enableAudio
            self.enableThinking = enableThinking
            self.enableAgentTools = enableAgentTools
            self.systemPrompt = systemPrompt
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            maxTokens = try container.decode(Double.self, forKey: .maxTokens)
            contextWindow = try container.decode(Double.self, forKey: .contextWindow)
            topK = try container.decode(Double.self, forKey: .topK)
            topP = try container.decode(Double.self, forKey: .topP)
            temperature = try container.decode(Double.self, forKey: .temperature)
            selectedBackend = try container.decode(String.self, forKey: .selectedBackend)
            enableVision = try container.decode(Bool.self, forKey: .enableVision)
            enableAudio = try container.decode(Bool.self, forKey: .enableAudio)
            enableThinking = try container.decodeIfPresent(Bool.self, forKey: .enableThinking) ?? true
            enableAgentTools = try container.decodeIfPresent(Bool.self, forKey: .enableAgentTools) ?? true
            systemPrompt = try container.decodeIfPresent(String.self, forKey: .systemPrompt)
        }
    }

    private enum PersistenceKeys {
        static let selectedModelName = "chat_selected_model_name"
        static let perModelSettings = "chat_model_generation_settings_v1"

        // Legacy global settings keys for migration defaults.
        static let maxTokens = "chat_max_tokens"
        static let contextWindow = "chat_context_window"
        static let topK = "chat_top_k"
        static let topP = "chat_top_p"
        static let temperature = "chat_temperature"
        static let selectedBackend = "chat_selected_backend"
        static let enableVision = "chat_enable_vision"
        static let enableAudio = "chat_enable_audio"
        static let enableThinking = "chat_enable_thinking"
        static let enableAgentTools = "chat_enable_agent_tools"
        static let systemPrompt = "chat_system_prompt"
    }

    // MARK: - RAG/Memory
    @Published var isIndexingDocument: Bool = false
    @Published var ragDocumentCount: Int = 0
    /// Char count of pending attached document (used for context ring when RAG is off).
    @Published var pendingAttachedDocumentChars: Int = 0

    // MARK: - Web Search
    @Published var isWebSearchEnabled: Bool = false
    @Published var isSearching: Bool = false

    var isRagEnabled: Bool {
        AppSettings.shared.selectedEmbeddingModelId != nil
    }
    var isMemoryEnabled: Bool {
        AppSettings.shared.memoryEnabled && isRagEnabled
    }

    private static let defaultGenerationSettings = ModelGenerationSettings(
        maxTokens: 512,
        contextWindow: 2048,
        topK: 64,
        topP: 0.95,
        temperature: 1.0,
        selectedBackend: "GPU",
        enableVision: true,
        enableAudio: true,
        enableThinking: true,
        enableAgentTools: true,
        systemPrompt: ""
    )

    @Published var inputText: String = ""
    /// Set by the view layer to indicate the user has scrolled away from the
    /// generating message. When true, streaming UI updates are throttled to
    /// ~8fps to avoid layout thrashing on off-screen content.
    var viewIsScrolledUp: Bool = false

    private var streamThrottleTask: Task<Void, Never>?
    private var pendingStreamUpdate: (() -> Void)?

    @Published var isGenerating: Bool = false {
        didSet {
            // When generation ends, flush the streamed content to disk. We skip
            // saves during streaming to avoid JSON encode storms that can make
            // SwiftUI's LazyVStack render blank briefly on big chats.
            if oldValue == true && isGenerating == false {
                flushThrottledStreamUpdate()
                chatStore.saveSessions()
            }
        }
    }
    var tokensPerSecond: Double = 0
    var totalTokens: Int = 0
    @Published var selectedModelName: String = AppSettings.shared.localized("no_model_selected") {
        didSet {
            guard selectedModelName != oldValue else { return }
            userDefaults.set(selectedModelName, forKey: PersistenceKeys.selectedModelName)
            loadSettingsForSelectedModel()
        }
    }
    @Published var isBackendLoading: Bool = false
    @Published private(set) var lastModelLoadErrorMessage: String? = nil
    
    // Config Properties (Persisted per model)
    @Published var maxTokens: Double = ChatViewModel.defaultGenerationSettings.maxTokens {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var contextWindow: Double = ChatViewModel.defaultGenerationSettings.contextWindow {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var topK: Double = ChatViewModel.defaultGenerationSettings.topK {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var topP: Double = ChatViewModel.defaultGenerationSettings.topP {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var temperature: Double = ChatViewModel.defaultGenerationSettings.temperature {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var selectedBackend: String = ChatViewModel.defaultGenerationSettings.selectedBackend {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var enableVision: Bool = ChatViewModel.defaultGenerationSettings.enableVision {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var enableAudio: Bool = ChatViewModel.defaultGenerationSettings.enableAudio {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var enableThinking: Bool = ChatViewModel.defaultGenerationSettings.enableThinking {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var enableAgentTools: Bool = ChatViewModel.defaultGenerationSettings.enableAgentTools {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }
    @Published var systemPrompt: String = ChatViewModel.defaultGenerationSettings.systemPrompt ?? "" {
        didSet { persistCurrentModelSettingsIfNeeded() }
    }

    private let chatStore = ChatStore.shared
    private let llmBackend = LLMBackend.shared
    private let userDefaults = UserDefaults.standard
    private let ttsManager = OnDeviceTtsManager.shared
    private var settingsByModelId: [String: ModelGenerationSettings] = [:]
    private var contextResetStartBySessionId: [UUID: Int] = [:]
    private var isApplyingPersistedSettings = false
    @Published var currentSessionId: UUID = UUID()
    private var activeGeneratingMessageId: UUID?
    
    // Compute current title from sessionId
    var currentTitle: String {
        get { chatStore.chatSessions.first(where: { $0.id == currentSessionId })?.title ?? "" }
        set {
            if let index = chatStore.chatSessions.firstIndex(where: { $0.id == currentSessionId }) {
                chatStore.chatSessions[index].title = newValue
                chatStore.saveSessions()
            }
        }
    }

    var chatSessions: [ChatSession] { chatStore.chatSessions }

    var latestUserMessageId: UUID? {
        messages.last(where: { $0.isFromUser })?.id
    }

    var latestAssistantMessageId: UUID? {
        messages.last(where: { !$0.isFromUser && !$0.isGenerating && !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })?.id
    }

    var contextWindowCapForSession: Double {
        let configuredCap = Double(max(1, Int(contextWindow)))
        if let loadedContextWindow = llmBackend.loadedContextWindow {
            return Double(max(1, loadedContextWindow))
        }
        return configuredCap
    }

    var contextBudgetForRing: Double {
        contextWindowCapForSession
    }

    var approximateContextTokensUsed: Double {
        let startIndex = max(0, min(messages.count, contextResetStartBySessionId[currentSessionId] ?? 0))
        let visibleMessages = Array(messages.dropFirst(startIndex))
        // Strip thinking blocks — they are NOT re-sent as history in multi-turn prompts,
        // so they should not count toward the context window budget.
        // During the streaming thinking phase (no answer yet), count 0 for that message.
        let messageChars = visibleMessages.reduce(0) { acc, msg in
            if msg.isFromUser {
                return acc + msg.content.count
            }
            if contentHasThinkingMarkers(msg.content) {
                let answer = getDisplayContentWithoutThinking(msg.content)
                return acc + answer.count     // 0 while still thinking
            }
            return acc + msg.content.count
        }
        let composerChars = inputText.count
        // When RAG is disabled the attached doc is stuffed into system prompt — count it.
        let docChars = isRagEnabled ? 0 : pendingAttachedDocumentChars
        let totalChars = messageChars + composerChars + docChars
        return max(0, Double(totalChars) / 4.0)
    }

    var contextUsageFractionRaw: Double {
        guard contextBudgetForRing > 0 else { return 0 }
        return min(max(approximateContextTokensUsed / contextBudgetForRing, 0), 1)
    }

    var contextUsageFractionDisplay: Double {
        if approximateContextTokensUsed <= 0 {
            return 0
        }
        return min(max(contextUsageFractionRaw, 0.02), 1)
    }

    var contextUsageLabel: String {
        if approximateContextTokensUsed > 0 {
            return "\(max(1, Int((contextUsageFractionRaw * 100).rounded())))%"
        }
        return "0%"
    }

    var isContextBudgetExceededForSession: Bool {
        contextUsageFractionRaw >= 0.995
    }
    
    private func setMessagesSilently(_ newValue: [ChatMessage]) {
        if let index = chatStore.chatSessions.firstIndex(where: { $0.id == currentSessionId }) {
            chatStore.chatSessions[index].messages = newValue
        }
    }

    var messages: [ChatMessage] {
        get {
            chatStore.chatSessions.first(where: { $0.id == currentSessionId })?.messages ?? []
        }
        set {
            if let index = chatStore.chatSessions.firstIndex(where: { $0.id == currentSessionId }) {
                chatStore.chatSessions[index].messages = newValue
                // Skip disk writes during streaming - they fire on every token and
                // the rapid JSON encode + publish storm causes the LazyVStack to
                // briefly render nothing when messages are large. We save once
                // when generation finishes (finishGeneratingMessage / stopGeneration).
                if !isGenerating {
                    chatStore.saveSessions()
                }
                objectWillChange.send()
            }
        }
    }

    init() {
        do {
            try RunAnywhere.initialize(environment: .development)
        } catch {
            // Ignore repeated initialization attempts.
        }

        Task {
            await RunAnywhere.refreshModelRegistry()
        }

        settingsByModelId = Self.loadPerModelSettings(from: userDefaults)

        if let savedModelName = userDefaults.string(forKey: PersistenceKeys.selectedModelName),
           !savedModelName.isEmpty {
            selectedModelName = savedModelName
        }

        loadSettingsForSelectedModel()

        if let empty = chatStore.chatSessions.first(where: { $0.messages.isEmpty }) {
            currentSessionId = empty.id
        } else {
            newChat()
        }
    }

    var loadedModelName: String? { llmBackend.currentlyLoadedModel }

    func loadModelIfNecessary(force: Bool = false) async {
        syncBackendSettings()

        guard selectedModelName != AppSettings.shared.localized("no_model_selected") else { return }
        guard let model = chatModel(named: selectedModelName) else { return }

        let modelMaxContext = max(1, model.contextWindowSize > 0 ? model.contextWindowSize : 2048)
        let desiredContextWindow = min(max(1, Int(contextWindow)), modelMaxContext)

        if !force,
           llmBackend.currentlyLoadedModel == selectedModelName,
           llmBackend.loadedContextWindow == desiredContextWindow {
            return
        }
        
        isBackendLoading = true
        defer { isBackendLoading = false }
        
        do {
            try await llmBackend.loadModel(model)
            lastModelLoadErrorMessage = nil
        } catch {
            print("Failed to load model: \(error)")
            lastModelLoadErrorMessage = error.localizedDescription
        }
    }

    private func syncBackendSettings() {
        llmBackend.maxTokens = Int(maxTokens)
        llmBackend.contextWindow = Int(contextWindow)
        llmBackend.topK = Int(topK)
        llmBackend.topP = Float(topP)
        llmBackend.temperature = Float(temperature)
        llmBackend.enableVision = enableVision
        llmBackend.enableAudio = enableAudio
        llmBackend.enableThinking = enableThinking
        llmBackend.enableAgentTools = enableAgentTools
        llmBackend.selectedBackend = selectedBackend
    }

    private var selectedModelId: String? {
        guard selectedModelName != AppSettings.shared.localized("no_model_selected") else { return nil }
        return chatModel(named: selectedModelName)?.id
    }

    private func loadSettingsForSelectedModel() {
        let settings: ModelGenerationSettings
        let hasPersisted: Bool

        if let modelId = selectedModelId,
           let persisted = settingsByModelId[modelId] {
            settings = clampSettings(persisted, forModelName: selectedModelName)
            hasPersisted = true
        } else {
            settings = clampSettings(Self.legacyDefaults(from: userDefaults), forModelName: selectedModelName)
            hasPersisted = false
        }

        let adjusted = settings

        applySettings(adjusted)

        // Ensure first-time model selections get persisted immediately.
        persistCurrentModelSettingsIfNeeded(force: true)
    }

    private func applySettings(_ settings: ModelGenerationSettings) {
        isApplyingPersistedSettings = true
        maxTokens = settings.maxTokens
        contextWindow = settings.contextWindow
        topK = settings.topK
        topP = settings.topP
        temperature = settings.temperature
        selectedBackend = settings.selectedBackend
        enableVision = settings.enableVision
        enableAudio = settings.enableAudio
        enableThinking = settings.enableThinking
        enableAgentTools = settings.enableAgentTools
        systemPrompt = settings.systemPrompt ?? ""
        isApplyingPersistedSettings = false
    }

    private func currentSettingsSnapshot() -> ModelGenerationSettings {
        ModelGenerationSettings(
            maxTokens: maxTokens,
            contextWindow: contextWindow,
            topK: topK,
            topP: topP,
            temperature: temperature,
            selectedBackend: selectedBackend,
            enableVision: enableVision,
            enableAudio: enableAudio,
            enableThinking: enableThinking,
            enableAgentTools: enableAgentTools,
            systemPrompt: systemPrompt
        )
    }

    private func persistCurrentModelSettingsIfNeeded(force: Bool = false) {
        if isApplyingPersistedSettings && !force { return }
        guard let modelId = selectedModelId else { return }

        let normalized = clampSettings(currentSettingsSnapshot(), forModelName: selectedModelName)
        settingsByModelId[modelId] = normalized

        if !isApplyingPersistedSettings {
            applySettings(normalized)
        }

        Self.savePerModelSettings(settingsByModelId, to: userDefaults)
    }

    private func clampSettings(_ settings: ModelGenerationSettings, forModelName modelName: String) -> ModelGenerationSettings {
        guard let model = chatModel(named: modelName) else {
            var fallback = settings
            fallback.contextWindow = max(1, fallback.contextWindow)
            fallback.maxTokens = min(max(1, fallback.maxTokens), fallback.contextWindow)
            fallback.topK = min(max(1, fallback.topK), 256)
            fallback.topP = min(max(0, fallback.topP), 1)
            fallback.temperature = min(max(0, fallback.temperature), 2)
            return fallback
        }

        let modelMaxContext = Double(max(1, model.contextWindowSize > 0 ? model.contextWindowSize : 2048))
        var clamped = settings
        clamped.contextWindow = min(max(1, clamped.contextWindow), modelMaxContext)
        clamped.maxTokens = min(max(1, clamped.maxTokens), clamped.contextWindow)
        clamped.topK = min(max(1, clamped.topK), 256)
        clamped.topP = min(max(0, clamped.topP), 1)
        clamped.temperature = min(max(0, clamped.temperature), 2)
        return clamped
    }

    private static func loadPerModelSettings(from defaults: UserDefaults) -> [String: ModelGenerationSettings] {
        guard let data = defaults.data(forKey: PersistenceKeys.perModelSettings) else {
            return [:]
        }
        guard let decoded = try? JSONDecoder().decode([String: ModelGenerationSettings].self, from: data) else {
            return [:]
        }
        return decoded
    }

    private static func savePerModelSettings(_ settingsByModelId: [String: ModelGenerationSettings], to defaults: UserDefaults) {
        guard let data = try? JSONEncoder().encode(settingsByModelId) else { return }
        defaults.set(data, forKey: PersistenceKeys.perModelSettings)
    }

    private static func legacyDefaults(from defaults: UserDefaults) -> ModelGenerationSettings {
        var settings = defaultGenerationSettings

        if defaults.object(forKey: PersistenceKeys.maxTokens) != nil {
            settings.maxTokens = defaults.double(forKey: PersistenceKeys.maxTokens)
        }
        if defaults.object(forKey: PersistenceKeys.contextWindow) != nil {
            settings.contextWindow = defaults.double(forKey: PersistenceKeys.contextWindow)
        }
        if defaults.object(forKey: PersistenceKeys.topK) != nil {
            settings.topK = defaults.double(forKey: PersistenceKeys.topK)
        }
        if defaults.object(forKey: PersistenceKeys.topP) != nil {
            settings.topP = defaults.double(forKey: PersistenceKeys.topP)
        }
        if defaults.object(forKey: PersistenceKeys.temperature) != nil {
            settings.temperature = defaults.double(forKey: PersistenceKeys.temperature)
        }
        if let backend = defaults.string(forKey: PersistenceKeys.selectedBackend), !backend.isEmpty {
            settings.selectedBackend = backend
        }
        if defaults.object(forKey: PersistenceKeys.enableVision) != nil {
            settings.enableVision = defaults.bool(forKey: PersistenceKeys.enableVision)
        }
        if defaults.object(forKey: PersistenceKeys.enableAudio) != nil {
            settings.enableAudio = defaults.bool(forKey: PersistenceKeys.enableAudio)
        }
        // Thinking is always enabled for thinking-capable models — no toggle.
        settings.enableThinking = true
        if defaults.object(forKey: PersistenceKeys.enableAgentTools) != nil {
            settings.enableAgentTools = defaults.bool(forKey: PersistenceKeys.enableAgentTools)
        }
        if let sp = defaults.string(forKey: PersistenceKeys.systemPrompt) {
            settings.systemPrompt = sp
        }

        return settings
    }

    func unloadModel() {
        llmBackend.isLoaded = false
        llmBackend.currentlyLoadedModel = nil
    }

    // MARK: - Web-search-enabled send
    @discardableResult
    func sendMessageWithWebSearch(imageURL: URL? = nil, documentURL: URL? = nil, documentName: String? = nil) -> Bool {
        let input = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return false }
        guard !isGenerating else { return false }

        let userMsg = ChatMessage(content: input, isFromUser: true, attachmentDocumentName: documentName)
        messages.append(userMsg)
        inputText = ""
        pendingAttachedDocumentChars = 0

        if currentTitle == AppSettings.shared.localized("drawer_new_chat") {
            currentTitle = String(input.prefix(20))
        }

        let aiMsg = ChatMessage(content: AppSettings.shared.localized("web_searching"), isFromUser: false, isGenerating: true)
        messages.append(aiMsg)
        activeGeneratingMessageId = aiMsg.id
        isGenerating = true
        isSearching = true

        let chatId = currentSessionId
        let capturedPrompt = input
        let capturedDocURL = documentURL
        let capturedDocName = documentName ?? "document"
        let ragEnabled = isRagEnabled

        streamingTask = Task {
            await loadModelIfNecessary()

            if !llmBackend.isLoaded {
                let msg = lastModelLoadErrorMessage ?? AppSettings.shared.localized("please_download_model")
                await updateLastAIMessage(content: msg, isGenerating: false)
                await MainActor.run {
                    self.isGenerating = false
                    self.isSearching = false
                }
                return
            }

            // 1. Perform web search BEFORE calling LLM (mirrors Android InferenceService approach)
            let searchQuery = Self.extractSearchQuery(from: capturedPrompt)
            let searchResults = await WebSearchService.shared.search(query: searchQuery, maxResults: 5)
            let resultCount = searchResults.count

            await MainActor.run {
                self.isSearching = false
                if let idx = self.messages.firstIndex(where: { $0.id == self.activeGeneratingMessageId }) {
                    if resultCount > 0 {
                        self.messages[idx].content = String(
                            format: AppSettings.shared.localized("web_search_found_results"), resultCount
                        )
                    } else {
                        self.messages[idx].content = AppSettings.shared.localized("web_search_no_results") + "\n\n"
                    }
                }
            }

            // 2. Extract document text if attached.
            var inlineDocumentText: String? = nil
            if let docURL = capturedDocURL {
                inlineDocumentText = try? DocumentTextExtractor.extract(from: docURL)
            }

            if capturedDocURL != nil && ragEnabled {
                await MainActor.run { self.isIndexingDocument = true }
                if let text = inlineDocumentText {
                    _ = await RagServiceManager.shared.addDocument(chatId: chatId, text: text, fileName: capturedDocName)
                    let count = await RagServiceManager.shared.documentCount(chatId: chatId)
                    await MainActor.run { self.ragDocumentCount = count }
                }
                await MainActor.run { self.isIndexingDocument = false }
            }

            // 3. Build RAG context prefix (same as sendMessage)
            var ragContextPrefix = ""
            if ragEnabled {
                var contextParts: [String] = []
                if isMemoryEnabled {
                    let memChunks = await RagServiceManager.shared.searchGlobalContext(query: capturedPrompt, maxResults: 3, relaxed: false)
                    for chunk in memChunks {
                        contextParts.append("📝 **\(chunk.fileName)**:\n\(chunk.content.trimmingCharacters(in: .whitespacesAndNewlines))")
                    }
                }
                if await RagServiceManager.shared.hasDocuments(chatId: chatId) {
                    let docChunks = await RagServiceManager.shared.searchRelevantContext(chatId: chatId, query: capturedPrompt, maxResults: 3)
                    for chunk in docChunks {
                        contextParts.append("📄 **\(chunk.fileName)**:\n\(chunk.content.trimmingCharacters(in: .whitespacesAndNewlines))")
                    }
                }
                if !contextParts.isEmpty {
                    ragContextPrefix = "---\n\nUSER MEMORY FACTS AND DOCUMENT CONTEXT:\n\n"
                        + contextParts.joined(separator: "\n\n")
                        + "\n\n---\n\n"
                }
            } else if let docText = inlineDocumentText, !docText.isEmpty {
                ragContextPrefix = "The user has attached the following document (\(capturedDocName)):\n\n\(String(docText.prefix(8000)))\n\n---\n\n"
            }

            // 4. Build search-enhanced system prompt (mirrors Android enhancedPrompt logic)
            let systemPrompt: String
            if searchResults.isEmpty {
                systemPrompt = ragContextPrefix.isEmpty ? "" : ragContextPrefix
            } else {
                let resultsText = searchResults.enumerated().map { i, r in
                    "SOURCE: \(r.source)\nTITLE: \(r.title)\nURL: \(r.url)\nCONTENT: \(r.snippet)\n---"
                }.joined(separator: "\n\n")

                systemPrompt = """
                \(ragContextPrefix)CURRENT WEB SEARCH RESULTS:
                \(resultsText)

                Based on the above current web search results, please answer the user's question: "\(capturedPrompt)"

                IMPORTANT INSTRUCTIONS:
                - Use ONLY the information from the web search results above
                - If the search results contain the answer, provide a clear and specific response
                - If the search results don't contain enough information, say so clearly
                - For dates and events, be specific based on what you find in the results
                - Do not make up information not found in the search results
                - Cite factual claims with the provided source URLs when possible
                """
            }

            // 5. Stream using normal LLM generation (NOT generateWithTools) — same path as sendMessage
            do {
                try await llmBackend.generate(
                    prompt: capturedPrompt,
                    systemPrompt: systemPrompt.isEmpty ? nil : systemPrompt
                ) { [weak self] content, tokens, tps in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.updateLastAIMessageSync(content: content, tokens: tokens, tps: tps)
                    }
                }
                await MainActor.run {
                    self.appendWebSources(searchResults)
                    self.finishGeneratingMessage()
                }
            } catch {
                await updateLastAIMessage(content: "Error: \(error.localizedDescription)", isGenerating: false)
            }

            await MainActor.run {
                self.isGenerating = false
                self.isSearching = false
                self.activeGeneratingMessageId = nil
            }
        }

        return true
    }

    // Strips conversational filler to get a clean search query (mirrors Android SearchIntentDetector.extractSearchQuery)
    private static func extractSearchQuery(from prompt: String) -> String {
        var q = prompt
        let fillers = [
            "search for ", "look up ", "find information about ", "find info about ",
            "what's the latest on ", "tell me about ", "please ", "what's the ",
            "what is the ", "can you search for ", "google ", "search "
        ]
        for filler in fillers {
            if q.lowercased().hasPrefix(filler) {
                q = String(q.dropFirst(filler.count))
                break
            }
        }
        return q.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? prompt : q.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    @discardableResult
    func sendMessage(imageURL: URL? = nil, audioURL: URL? = nil, documentURL: URL? = nil, documentName: String? = nil) -> Bool {
        let selectedModel = chatModel(named: selectedModelName)
        let effectiveImageURL = (enableVision && selectedModel?.supportsVision == true
            && (selectedModel.map { LLMBackend.shared.isVisionProjectorAvailable(for: $0) } ?? false)) ? imageURL : nil
        let effectiveAudioURL = (enableAudio && selectedModel?.supportsAudio == true) ? audioURL : nil

        let input = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasAttachment = effectiveImageURL != nil || effectiveAudioURL != nil || documentURL != nil
        guard !input.isEmpty || hasAttachment else { return false }
        guard !isGenerating else { return false }

        // New turn: cut off any ongoing auto-readout from previous turn
        stopAutoReadout()

        let generationPrompt: String = {
            if !input.isEmpty { return input }
            if effectiveImageURL != nil { return "Describe this image." }
            if effectiveAudioURL != nil { return "Transcribe this audio." }
            if documentURL != nil { return "Summarize the attached document." }
            return ""
        }()

        // Use the reset-aware approximateContextTokensUsed as the baseline so that
        // after a context reset the projected fraction starts from 0 (only new messages),
        // not from all-time total. Without this, projectedFraction stays ≥ 99.5% forever
        // after the first reset, causing every subsequent send to also reset.
        let newPromptTokens = Double(generationPrompt.count) / 4.0
        let projectedTokens = approximateContextTokensUsed + newPromptTokens
        let projectedFraction = contextBudgetForRing > 0 ? (projectedTokens / contextBudgetForRing) : 0
        let shouldResetInferenceContext = (isContextBudgetExceededForSession || projectedFraction >= 0.995) && !messages.isEmpty

        if shouldResetInferenceContext {
            contextResetStartBySessionId[currentSessionId] = messages.count
        }

        let userMsg = ChatMessage(
            content: input,
            isFromUser: true,
            attachmentImagePath: effectiveImageURL?.path,
            attachmentAudioPath: effectiveAudioURL?.path,
            attachmentDocumentName: documentName
        )
        messages.append(userMsg)
        inputText = ""
        pendingAttachedDocumentChars = 0

        // Auto-update title
        if currentTitle == AppSettings.shared.localized("drawer_new_chat") {
            let titleSeed = !input.isEmpty ? input : (effectiveImageURL != nil ? "Image" : (documentURL != nil ? documentName ?? "Document" : "Audio"))
            currentTitle = String(titleSeed.prefix(20))
        }

        let aiMsg = ChatMessage(content: "", isFromUser: false, isGenerating: true)
        messages.append(aiMsg)
        activeGeneratingMessageId = aiMsg.id
        isGenerating = true

        let chatId = currentSessionId
        let ragEnabled = isRagEnabled
        let memoryEnabled = isMemoryEnabled
        let capturedDocURL = documentURL
        let capturedDocName = documentName ?? "document"
        let capturedPrompt = generationPrompt

        streamingTask = Task {
            // 1. Extract document text (always, when a document is attached).
            var inlineDocumentText: String? = nil
            if let docURL = capturedDocURL {
                inlineDocumentText = try? DocumentTextExtractor.extract(from: docURL)
            }

            // 1b. Index document into RAG if RAG is enabled.
            if capturedDocURL != nil && ragEnabled {
                await MainActor.run { self.isIndexingDocument = true }
                if let text = inlineDocumentText {
                    _ = await RagServiceManager.shared.addDocument(chatId: chatId, text: text, fileName: capturedDocName)
                    let count = await RagServiceManager.shared.documentCount(chatId: chatId)
                    await MainActor.run { self.ragDocumentCount = count }
                }
                await MainActor.run { self.isIndexingDocument = false }
            }

            // 2. Build context prefix: RAG chunks + fallback full document text when RAG disabled.
            var ragContextPrefix = ""
            if !capturedPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if ragEnabled {
                var contextParts: [String] = []
                print("🔍 [Chat] Send — RAG enabled, memoryEnabled=\(memoryEnabled)")

                // 2a. Global memory search.
                if memoryEnabled {
                    print("🔍 [Chat] Send — searching global memory for: \"\(capturedPrompt.prefix(50))\"")
                    let memChunks = await RagServiceManager.shared.searchGlobalContext(
                        query: capturedPrompt, maxResults: 3, relaxed: false
                    )
                    print("🔍 [Chat] Send — got \(memChunks.count) memory chunks")
                    for chunk in memChunks {
                        contextParts.append("📝 **\(chunk.fileName)**:\n\(chunk.content.trimmingCharacters(in: .whitespacesAndNewlines))")
                    }
                }

                // 2b. Per-chat document search.
                if await RagServiceManager.shared.hasDocuments(chatId: chatId) {
                    let docChunks = await RagServiceManager.shared.searchRelevantContext(
                        chatId: chatId, query: capturedPrompt, maxResults: 3
                    )
                    for chunk in docChunks {
                        contextParts.append("📄 **\(chunk.fileName)**:\n\(chunk.content.trimmingCharacters(in: .whitespacesAndNewlines))")
                    }
                }

                if !contextParts.isEmpty {
                    print("🔍 [Chat] Send — injecting \(contextParts.count) RAG context parts into system prompt")
                    ragContextPrefix = "---\n\nUSER MEMORY FACTS AND DOCUMENT CONTEXT:\n\nIMPORTANT: The following lines contain relevant information from user documents and memory. Use them to answer accurately.\n\n"
                        + contextParts.joined(separator: "\n\n")
                        + "\n\n---\n\n"
                } else {
                    print("🔍 [Chat] Send — no RAG context found")
                }
            } else if let docText = inlineDocumentText, !docText.isEmpty {
                // RAG disabled but document attached: stuff full text directly into context.
                let truncated = String(docText.prefix(8000))
                ragContextPrefix = "The user has attached the following document (\(capturedDocName)):\n\n\(truncated)\n\n---\n\n"
            }
            } // end: if !capturedPrompt.isEmpty
            await loadModelIfNecessary(force: shouldResetInferenceContext)

            do {
                if !llmBackend.isLoaded {
                    let msg = lastModelLoadErrorMessage ?? AppSettings.shared.localized("please_download_model")
                    await updateLastAIMessage(content: msg, isGenerating: false)
                    await MainActor.run { self.isGenerating = false }
                    return
                }

                let finalSystemPrompt: String? = {
                    let userSP = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                    if userSP.isEmpty {
                        return ragContextPrefix.isEmpty ? nil : ragContextPrefix
                    } else if ragContextPrefix.isEmpty {
                        return userSP
                    } else {
                        return "\(userSP)\n\n\(ragContextPrefix)"
                    }
                }()

                // Build a multi-turn prompt so the model sees the full conversation history.
                // Images/audio attachments skip multi-turn formatting (handled by the VLM path).
                let multiTurnPrompt: String
                if effectiveImageURL != nil || effectiveAudioURL != nil {
                    multiTurnPrompt = capturedPrompt
                } else {
                    multiTurnPrompt = await MainActor.run {
                        self.buildMultiTurnPrompt(currentUserPrompt: capturedPrompt, ragPrefix: finalSystemPrompt)
                    }
                }

                // Determine stop sequences based on the chat template in use.
                let stopSeqs: [String] = await MainActor.run {
                    self.stopSequencesForCurrentModel()
                }

                try await llmBackend.generate(
                    prompt: multiTurnPrompt,
                    imageURL: effectiveImageURL,
                    audioURL: effectiveAudioURL,
                    systemPrompt: finalSystemPrompt,
                    stopSequences: stopSeqs
                ) { [weak self] content, tokens, tps in
                    Task { @MainActor [weak self] in
                        guard let self = self else { return }
                        self.updateLastAIMessageSync(content: content, tokens: tokens, tps: tps)
                    }
                }
                await MainActor.run { self.finishGeneratingMessage() }
            } catch is CancellationError {
                // Task was cancelled (user stop, turn-leak auto-stop) — not an error.
                await MainActor.run { self.finishGeneratingMessage() }
            } catch {
                await updateLastAIMessage(content: "Error: \(error.localizedDescription)", isGenerating: false)
            }

            await MainActor.run {
                self.isGenerating = false
                self.activeGeneratingMessageId = nil
            }
        }

        return true
    }

    private func updateLastAIMessage(content: String, isGenerating: Bool) async {
        await MainActor.run {
            updateLastAIMessageSync(content: content, isGenerating: isGenerating)
        }
    }

    private func updateLastAIMessageSync(content: String, tokens: Int = 0, tps: Double = 0, isGenerating: Bool = true) {
        let targetIndex: Int?
        if let activeId = activeGeneratingMessageId {
            targetIndex = messages.firstIndex(where: { $0.id == activeId })
        } else {
            targetIndex = messages.indices.last
        }

        if let idx = targetIndex, !messages[idx].isFromUser {
            var msgs = self.messages
            let normalizedContent = normalizeStreamText(content)

            // Detect turn leak: if the raw content is significantly longer than
            // the normalized content, the model generated past its turn boundary
            // (e.g. hallucinated "User:" / "Assistant:" exchanges). Auto-stop
            // generation to avoid wasting tokens and leaving the stop button stuck.
            let rawLen = content.trimmingCharacters(in: .whitespacesAndNewlines).count
            let normLen = normalizedContent.count
            if isGenerating && rawLen > normLen + 10 {
                print("⚠️ [TurnLeak] Detected turn leak: raw=\(rawLen) normalized=\(normLen). Auto-stopping generation.")
                msgs[idx].content = normalizedContent
                msgs[idx].isGenerating = false
                if totalTokens > 0 {
                    msgs[idx].tokenCount = totalTokens
                    msgs[idx].tokensPerSecond = tokensPerSecond
                }
                self.messages = msgs
                stopGeneration()
                return
            }

            let parsed = parseThinkingAndAnswer(normalizedContent)
            if contentHasThinkingMarkers(normalizedContent) || !parsed.thinking.isEmpty {
                print(
                    "🧠 [ThinkingDebug][chat] isGenerating=\(isGenerating) rawChars=\(content.count) normalizedChars=\(normalizedContent.count) thinkingChars=\(parsed.thinking.count) answerChars=\(parsed.answer.count) preview=\(String(normalizedContent.prefix(120)))"
                )
            }
            msgs[idx].content = normalizedContent
            msgs[idx].isGenerating = isGenerating
            msgs[idx].tokenCount = tokens > 0 ? tokens : msgs[idx].tokenCount
            msgs[idx].tokensPerSecond = tps > 0 ? tps : msgs[idx].tokensPerSecond

            if isGenerating && viewIsScrolledUp {
                setMessagesSilently(msgs)
                scheduleThrottledStreamUpdate { [weak self] in
                    guard let self else { return }
                    self.totalTokens = tokens
                    self.tokensPerSecond = tps
                    self.objectWillChange.send()
                }
            } else {
                self.totalTokens = tokens
                self.tokensPerSecond = tps
                self.messages = msgs
            }

            // Progressive TTS: speak completed sentences as they stream in
            if isGenerating {
                progressiveTTSUpdate(fullContent: normalizedContent, messageKey: msgs[idx].id.uuidString)
            }
        }
    }

    private func scheduleThrottledStreamUpdate(_ update: @escaping () -> Void) {
        pendingStreamUpdate = update
        guard streamThrottleTask == nil else { return }
        streamThrottleTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 500_000_000) // ~2fps while scrolled away
            guard let self, !Task.isCancelled else { return }
            self.pendingStreamUpdate?()
            self.pendingStreamUpdate = nil
            self.streamThrottleTask = nil
        }
    }

    func flushThrottledStreamUpdate() {
        streamThrottleTask?.cancel()
        streamThrottleTask = nil
        pendingStreamUpdate = nil
        objectWillChange.send()
    }

    private func normalizeStreamText(_ text: String) -> String {
        var normalized = text
            .replacingOccurrences(of: "â€™", with: "'")
            .replacingOccurrences(of: "â€˜", with: "'")
            .replacingOccurrences(of: "â€œ", with: "\"")
            .replacingOccurrences(of: "â€", with: "\"")
            .replacingOccurrences(of: "â€“", with: "-")
            .replacingOccurrences(of: "â€”", with: "-")
            .replacingOccurrences(of: "�", with: "'")

        let leakedPrefixes = [
            "<bos>",
            "<eos>",
            "<|turn>model\n",
            "<|turn>model\r\n",
            "<|turn>model",
            "<|turn>",
            "<turn|>",
            "<start_of_turn>model\n",
            "<start_of_turn>model\r\n",
            "<start_of_turn>model",
            "<start_of_turn>",
            "model\n",
            "model\r\n",
        ]

        var removedPrefix = true
        while removedPrefix {
            removedPrefix = false
            for prefix in leakedPrefixes {
                if normalized.hasPrefix(prefix) {
                    normalized.removeFirst(prefix.count)
                    normalized = normalized.trimmingCharacters(in: .whitespacesAndNewlines)
                    removedPrefix = true
                }
            }
        }

        // Truncate at leaked turn markers — the model should never generate a new
        // "User:" turn. If it does, chop the response there to prevent fake exchanges.
        let turnLeakPatterns = ["\nUser:", "\nuser:", "\nAssistant:", "\nassistant:", "\nHuman:", "\nhuman:"]
        for pattern in turnLeakPatterns {
            if let range = normalized.range(of: pattern) {
                normalized = String(normalized[normalized.startIndex..<range.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
        }

        return normalized
    }

    private func finishGeneratingMessage() {
        let targetIndex: Int?
        if let activeId = activeGeneratingMessageId {
            targetIndex = messages.firstIndex(where: { $0.id == activeId })
        } else {
            targetIndex = messages.indices.last
        }

        if let idx = targetIndex, !messages[idx].isFromUser {
            var msgs = self.messages
            msgs[idx].isGenerating = false
            if totalTokens > 0 {
                msgs[idx].tokenCount = totalTokens
                msgs[idx].tokensPerSecond = tokensPerSecond
            }
            self.messages = msgs

            // Speak any remaining unread text (progressive TTS handles sentence-by-sentence
            // during streaming; this catches the final fragment after the last sentence boundary).
            let finishedMessage = msgs[idx]
            progressiveTTSFinish(fullContent: finishedMessage.content, messageKey: finishedMessage.id.uuidString)
        }
        activeGeneratingMessageId = nil
    }

    private func appendWebSources(_ results: [WebSearchResult]) {
        let unique = Dictionary(grouping: results.filter { URL(string: $0.url)?.scheme != nil }, by: \.url).compactMap { $0.value.first }
        guard !unique.isEmpty else { return }
        let sources = unique.map { "- [\($0.title.isEmpty ? $0.source : $0.title)](\($0.url))" }.joined(separator: "\n")
        guard let idx = messages.indices.last, !messages[idx].isFromUser, !messages[idx].content.contains("\n### Sources\n") else { return }
        messages[idx].content += "\n\n### Sources\n\(sources)"
    }

    func stopGeneration() {
        streamingTask?.cancel()
        streamingTask = nil
        stopAutoReadout()
        if let activeId = activeGeneratingMessageId,
           let idx = messages.firstIndex(where: { $0.id == activeId }),
           !messages[idx].isFromUser {
            messages[idx].isGenerating = false
        } else if let idx = messages.indices.last, !messages[idx].isFromUser {
            messages[idx].isGenerating = false
        }
        activeGeneratingMessageId = nil
        isGenerating = false
    }

    func copyMessage(_ message: ChatMessage) {
        UIPasteboard.general.string = message.content
    }

    /// Stop any ongoing auto-readout and reset the progressive TTS cursor.
    func stopAutoReadout() {
        ttsManager.stop()
        ttsReadCursor = 0
    }

    func newChat() {
        stopAutoReadout()
        let session = ChatSession(title: AppSettings.shared.localized("drawer_new_chat"))
        chatStore.addSession(session)
        currentSessionId = session.id
        contextResetStartBySessionId[session.id] = 0
        ragDocumentCount = 0
        objectWillChange.send()

        // Fire interstitial ad (skipped for premium users, every 4th new chat)
        InterstitialAdManager.shared.onEvent()

        // Populate new chat with global memory so RAG search includes it.
        if isMemoryEnabled {
            Task {
                await RagServiceManager.shared.populateChat(chatId: session.id)
            }
        }
    }

    func deleteSession(_ id: UUID) {
        chatStore.deleteSession(id: id)
        contextResetStartBySessionId.removeValue(forKey: id)
        if currentSessionId == id {
            if let first = chatSessions.first {
                currentSessionId = first.id
            } else {
                newChat()
            }
        }
        objectWillChange.send()
    }

    func regenerateResponse(for assistantMessageId: UUID) {
        guard !isGenerating else { return }
        guard let assistantIndex = messages.firstIndex(where: { $0.id == assistantMessageId && !$0.isFromUser }) else { return }
        guard let userIndex = messages[..<assistantIndex].lastIndex(where: { $0.isFromUser }) else { return }

        let userMessage = messages[userIndex]
        let trimmedPrompt = userMessage.content.trimmingCharacters(in: .whitespacesAndNewlines)
        let imageURL = existingFileURL(atPath: userMessage.attachmentImagePath)
        let audioURL = existingFileURL(atPath: userMessage.attachmentAudioPath)

        let prompt: String = {
            if !trimmedPrompt.isEmpty { return trimmedPrompt }
            if imageURL != nil { return "Describe this image." }
            if audioURL != nil { return "Transcribe this audio." }
            return ""
        }()
        guard !prompt.isEmpty else { return }

        var msgs = messages
        msgs[assistantIndex].content = ""
        msgs[assistantIndex].isGenerating = true
        msgs[assistantIndex].tokenCount = nil
        msgs[assistantIndex].tokensPerSecond = nil
        messages = msgs

        totalTokens = 0
        tokensPerSecond = 0
        activeGeneratingMessageId = assistantMessageId
        isGenerating = true

        streamingTask = Task {
            await loadModelIfNecessary()

            do {
                if !llmBackend.isLoaded {
                    let msg = lastModelLoadErrorMessage ?? AppSettings.shared.localized("please_download_model")
                    await updateLastAIMessage(content: msg, isGenerating: false)
                    await MainActor.run {
                        self.isGenerating = false
                        self.activeGeneratingMessageId = nil
                    }
                    return
                }

                try await llmBackend.generate(prompt: prompt, imageURL: imageURL, audioURL: audioURL) { [weak self] content, tokens, tps in
                    Task { @MainActor [weak self] in
                        guard let self = self else { return }
                        self.updateLastAIMessageSync(content: content, tokens: tokens, tps: tps)
                    }
                }
                await MainActor.run { self.finishGeneratingMessage() }
            } catch {
                await updateLastAIMessage(content: "Error: \(error.localizedDescription)", isGenerating: false)
            }

            await MainActor.run {
                self.isGenerating = false
                self.activeGeneratingMessageId = nil
            }
        }
    }

    func editAssistantMessage(_ messageId: UUID, newText: String) {
        let trimmed = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let idx = messages.firstIndex(where: { $0.id == messageId && !$0.isFromUser }) else { return }
        var msgs = messages
        msgs[idx].content = trimmed
        messages = msgs
    }

    func editUserPrompt(_ messageId: UUID, newText: String) {
        guard !isGenerating else { return }
        let trimmed = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let userIndex = messages.firstIndex(where: { $0.id == messageId && $0.isFromUser }) else { return }

        var msgs = messages
        msgs[userIndex].content = trimmed
        messages = msgs

        let nextIndex = userIndex + 1
        if nextIndex < messages.count,
           let assistantIndex = messages[nextIndex...].firstIndex(where: { !$0.isFromUser }) {
            regenerateResponse(for: messages[assistantIndex].id)
        }
    }

    private var streamingTask: Task<Void, Never>?

    // MARK: - Progressive TTS (sentence-by-sentence readout during streaming)
    /// Tracks how many characters of the current AI message have already been sent to TTS.
    private var ttsReadCursor: Int = 0
    /// The key used for the current auto-readout turn, so we can stop it on new turn/chat switch.
    private var autoReadoutKey: String?

    /// Called from `updateLastAIMessageSync` during streaming to progressively speak completed sentences.
    /// Extracts only the NEW delta text since last call and feeds it token-by-token into the TTS
    /// streaming buffer — mirrors Android's `ttsService.addStreamingText(piece)` pattern.
    private func progressiveTTSUpdate(fullContent: String, messageKey: String) {
        guard AppSettings.shared.autoReadoutEnabled else { return }

        // Strip thinking tokens — only speak the answer portion
        let displayContent: String
        if contentHasThinkingMarkers(fullContent) {
            displayContent = getDisplayContentWithoutThinking(fullContent)
        } else {
            displayContent = fullContent
        }

        guard displayContent.count > ttsReadCursor else { return }

        // Extract only the NEW characters since the last update
        let delta = String(displayContent.dropFirst(ttsReadCursor))
        ttsReadCursor = displayContent.count

        guard !delta.isEmpty else { return }

        autoReadoutKey = messageKey
        // Feed the delta into the streaming buffer; it will speak each sentence as it completes
        ttsManager.addStreamingToken(delta, fallbackLanguage: AppSettings.shared.selectedLanguage, key: messageKey)
    }

    /// Flushes any remaining buffered text after generation finishes.
    private func progressiveTTSFinish(fullContent: String, messageKey: String) {
        guard AppSettings.shared.autoReadoutEnabled else { return }

        // Feed any final delta that may not have been processed yet
        let displayContent: String
        if contentHasThinkingMarkers(fullContent) {
            displayContent = getDisplayContentWithoutThinking(fullContent)
        } else {
            displayContent = fullContent
        }
        if displayContent.count > ttsReadCursor {
            let delta = String(displayContent.dropFirst(ttsReadCursor))
            if !delta.isEmpty {
                ttsManager.addStreamingToken(delta, fallbackLanguage: AppSettings.shared.selectedLanguage, key: messageKey)
            }
        }

        // Flush any leftover text in the buffer (e.g. last sentence without punctuation)
        ttsManager.flushStreamingBuffer(fallbackLanguage: AppSettings.shared.selectedLanguage, key: messageKey)
        ttsReadCursor = 0
    }

    private func existingFileURL(atPath path: String?) -> URL? {
        resolveStoredAttachmentURL(path)
    }

    // MARK: - Multi-turn prompt builder

    /// Formats the full conversation history into the model's native chat template.
    /// Uses the __RAW_PROMPT__ prefix to bypass SDK auto-formatting, ensuring we have 
    /// full control over the multi-turn sequence and context budget.
    @MainActor
    /// Returns stop sequences appropriate for the current model's chat template.
    /// This prevents models from generating past their turn boundary (e.g. hallucinating
    /// fake "User:" / "Assistant:" exchanges).
    private func stopSequencesForCurrentModel() -> [String] {
        let modelName = selectedModelName.lowercased()
        let isGemma  = modelName.contains("gemma")
        let isLlama  = modelName.contains("llama") || modelName.contains("mistral")
        let isLlama3 = isLlama && (modelName.contains("llama-3") || modelName.contains("llama 3") || modelName.contains("llama-3."))
        let isHarmony = modelName.contains("gpt-oss") || modelName.contains("gpt_oss")

        if isGemma {
            return []
        } else if isLlama3 {
            return ["<|eot_id|>"]
        } else if isLlama {
            return ["[INST]"]
        } else if isHarmony {
            return ["<|start|>user"]
        } else {
            return ["\nUser:", "\nuser:"]
        }
    }

    func buildMultiTurnPrompt(currentUserPrompt: String, ragPrefix: String? = nil) -> String {
        let modelName = selectedModelName.lowercased()
        let modelSupportsThinking = chatModel(named: selectedModelName)?.supportsThinking == true
        let isGemma  = modelName.contains("gemma")
        let isGemma4 = isGemma && (modelName.contains("gemma 4") || modelName.contains("gemma-4")) && !modelName.contains("translate")
        let isLlama  = modelName.contains("llama") || modelName.contains("mistral")
        let isLlama3 = isLlama && (modelName.contains("llama-3") || modelName.contains("llama 3") || modelName.contains("llama-3."))
        let isHarmonyModel = modelName.contains("gpt-oss") || modelName.contains("gpt_oss")

        // 1. Identify history (exclude placeholder turns).
        // Respect context-reset boundary: if the context ring was previously reset,
        // only feed messages from the reset point onward so the model truly forgets
        // pre-reset history (matching what the display ring already shows).
        let resetStart = contextResetStartBySessionId[currentSessionId] ?? 0
        let allHistory: [ChatMessage] = messages.count >= 2 ? Array(messages.dropLast(2)) : []
        var history: [ChatMessage] = resetStart > 0 ? Array(allHistory.dropFirst(min(resetStart, allHistory.count))) : allHistory

        // 2. Context Window Management (Sliding Window)
        // Size history budget from the actual loaded context window so prompt + response fit.
        // Reserve response = min(maxTokens, ctx/4) — maxTokens is a CAP, the real
        // reply is almost always shorter, so reserving the full cap starves history
        // and causes the model to "forget" its own prior replies (user sees it only
        // remembering their own messages). Guaranteeing at least ctx/4 for response
        // also prevents the 1-word-reply failure when history grows huge.
        let effectiveCtxTokens = llmBackend.loadedContextWindow ?? max(512, Int(contextWindow))
        let currentPromptTokensEstimate = max(32, currentUserPrompt.count / 3)
        let ragTokensEstimate = (ragPrefix?.count ?? 0) / 3
        let reservedForResponse = max(256, min(Int(maxTokens), effectiveCtxTokens / 4))
        let reservedForCurrent = currentPromptTokensEstimate + ragTokensEstimate + 64
        let reservedSafety = 128
        let availableHistoryTokens = max(128, effectiveCtxTokens - reservedForResponse - reservedForCurrent - reservedSafety)
        let maxHistoryChars = availableHistoryTokens * 3
        var currentChars = 0
        var truncatedHistory: [ChatMessage] = []
        
        // Walk backwards through history to keep most recent turns.
        // If a single message is larger than the full budget, truncate its MIDDLE
        // rather than dropping it entirely — this preserves important context at
        // the start and end of long assistant replies (e.g. stories).
        for msg in history.reversed() {
            // For context budget, count only the answer portion (thinking is stripped)
            let rawContent: String
            if !msg.isFromUser && contentHasThinkingMarkers(msg.content) {
                rawContent = getDisplayContentWithoutThinking(msg.content)
            } else {
                rawContent = msg.content
            }
            let effectiveLen = rawContent.count
            let remaining = maxHistoryChars - currentChars

            if effectiveLen <= remaining {
                // Message fits entirely — include as-is.
                truncatedHistory.insert(msg, at: 0)
                currentChars += effectiveLen
            } else if remaining > 300 {
                // Message exceeds remaining budget but there is still meaningful space.
                // Truncate the MIDDLE so we keep both the opening and closing context
                // (e.g. a long story's intro + conclusion).  This prevents a single long
                // assistant reply from silently wiping out all history on the next turn.
                let half = remaining / 2
                let startSlice = String(rawContent.prefix(half))
                let endSlice   = String(rawContent.suffix(half))
                var elided = msg
                elided.content = startSlice + "\n…\n" + endSlice
                truncatedHistory.insert(elided, at: 0)
                currentChars = maxHistoryChars // budget exhausted
                break
            } else {
                // Too little budget left for even a truncated version — stop here.
                break
            }
        }
        history = truncatedHistory

        // 3. Build the Raw Prompt String
        var parts: [String] = []

        // Prepend the RAW prompt sentinel for the RunAnywhere SDK
        // This prevents the SDK from wrapping our already-formatted string.
        parts.append("__RAW_PROMPT__")

        if isLlama3 {
            parts.append("<|begin_of_text|>")
        }

        if isHarmonyModel {
            var harmonyParts: [String] = []
            let systemContent = ragPrefix?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let effectiveSystem = systemContent.isEmpty ? "You are a helpful assistant." : systemContent
            harmonyParts.append("<|start|>system<|message|>\(effectiveSystem)<|end|>")

            for msg in history {
                let rawContent = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
                let content: String
                if msg.isFromUser {
                    content = rawContent
                } else {
                    let answer = getDisplayContentWithoutThinking(rawContent)
                    content = answer.isEmpty ? rawContent : answer
                }
                guard !content.isEmpty else { continue }

                let role = msg.isFromUser ? "user" : "assistant"
                harmonyParts.append("<|start|>\(role)<|message|>\(content)<|end|>")
            }

            harmonyParts.append("<|start|>user<|message|>\(currentUserPrompt)<|end|>")
            if modelSupportsThinking && enableThinking {
                harmonyParts.append("<|start|>assistant")
            } else {
                harmonyParts.append("<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>")
            }
            parts.append(contentsOf: harmonyParts)
            return parts.joined()
        }

        // Optionally inject RAG context or System Message as an opening turn.
        if let rag = ragPrefix, !rag.isEmpty {
            if isGemma4 {
                parts.append("<|turn>system\n\(rag)<turn|>")
            } else if isGemma {
                parts.append("<start_of_turn>user\n\(rag)<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>")
            } else if isLlama3 {
                parts.append("<|start_header_id|>system<|end_header_id|>\n\n\(rag)<|eot_id|>")
            } else if isLlama {
                parts.append("[INST] \(rag) [/INST]\nUnderstood.")
            } else {
                parts.append("System: \(rag)")
            }
        }

        // Add history turns
        for msg in history {
            // Strip thinking blocks from assistant messages — the reasoning chain should
            // not be re-fed into the context window as "said text".
            let rawContent = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
            let content: String
            if msg.isFromUser {
                content = rawContent
            } else {
                let answer = getDisplayContentWithoutThinking(rawContent)
                content = answer.isEmpty ? rawContent : answer
            }
            guard !content.isEmpty else { continue }

            if isGemma4 {
                let gemmaRole = msg.isFromUser ? "user" : "model"
                parts.append("<|turn>\(gemmaRole)\n\(content)<turn|>")
            } else if isGemma {
                let gemmaRole = msg.isFromUser ? "user" : "model"
                parts.append("<start_of_turn>\(gemmaRole)\n\(content)<end_of_turn>")
            } else if isLlama3 {
                let role = msg.isFromUser ? "user" : "assistant"
                parts.append("<|start_header_id|>\(role)<|end_header_id|>\n\n\(content)<|eot_id|>")
            } else if isLlama {
                if msg.isFromUser {
                    parts.append("[INST] \(content) [/INST]")
                } else {
                    parts.append(content)
                }
            } else {
                let prefix = msg.isFromUser ? "User" : "Assistant"
                parts.append("\(prefix): \(content)")
            }
        }

        // 4. Append the active new user prompt
        if isGemma4 {
            parts.append("<|turn>user\n\(currentUserPrompt)<turn|>")
            parts.append("<|turn>model\n")
        } else if isGemma {
            parts.append("<start_of_turn>user\n\(currentUserPrompt)<end_of_turn>")
            parts.append("<start_of_turn>model\n")
        } else if isLlama3 {
            parts.append("<|start_header_id|>user<|end_header_id|>\n\n\(currentUserPrompt)<|eot_id|>")
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        } else if isLlama {
            parts.append("[INST] \(currentUserPrompt) [/INST]")
        } else {
            parts.append("User: \(currentUserPrompt)")
            parts.append("Assistant:")
        }

        let finalPrompt = parts.joined(separator: "\n")
        #if DEBUG
        let userTurns = truncatedHistory.filter { $0.isFromUser }.count
        let asstTurns = truncatedHistory.filter { !$0.isFromUser }.count
        print("📝 [PromptBuild] ctx=\(effectiveCtxTokens) histBudget=\(availableHistoryTokens)tok historyMsgs=user:\(userTurns)/asst:\(asstTurns) historyChars=\(currentChars) totalPromptChars=\(finalPrompt.count)")
        #endif
        return finalPrompt
    }
}


// MARK: - Message Bubble
struct MessageBubble: View {
    @EnvironmentObject var settings: AppSettings
    let message: ChatMessage
    let preferThinkingWhileStreaming: Bool
    let onCopy: () -> Void
    let onOpenImage: ((String) -> Void)?
    let onEditUserMessage: ((String) -> Void)?
    let onEditAssistantMessage: ((String) -> Void)?
    let onRegenerateResponse: (() -> Void)?
    let onToggleTts: (() -> Void)?
    let isTtsSpeaking: Bool
    @State private var showActions = false
    @State private var isEditing = false
    @State private var editedText = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if message.isFromUser {
                HStack {
                    Spacer(minLength: 40)
                    if isEditing {
                        VStack(alignment: .trailing, spacing: 8) {
                            TextEditor(text: $editedText)
                                .frame(minHeight: 90)
                                .padding(8)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.16), lineWidth: 1)
                                )
                            HStack(spacing: 8) {
                                Button {
                                    isEditing = false
                                    editedText = ""
                                } label: {
                                    Image(systemName: "xmark")
                                }
                                Button {
                                    let trimmed = editedText.trimmingCharacters(in: .whitespacesAndNewlines)
                                    if !trimmed.isEmpty {
                                        onEditUserMessage?(trimmed)
                                        isEditing = false
                                        editedText = ""
                                    }
                                } label: {
                                    Image(systemName: "checkmark")
                                }
                                .disabled(editedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            }
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.68))
                        }
                        .frame(maxWidth: 320)
                    } else {
                        VStack(alignment: .trailing, spacing: 8) {
                            if let imagePath = message.attachmentImagePath,
                                         let uiImage = previewImage(from: imagePath) {
                                Image(uiImage: uiImage)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxWidth: 220)
                                    .clipShape(RoundedRectangle(cornerRadius: 14))
                                    .onTapGesture {
                                        onOpenImage?(imagePath)
                                    }
                            }

                            if let audioPath = message.attachmentAudioPath,
                               let audioURL = resolveStoredAttachmentURL(audioPath) {
                                HStack(spacing: 8) {
                                    AudioPlaybackButton(url: audioURL)
                                    Label(settings.localized("audio"), systemImage: "waveform")
                                        .font(.caption)
                                        .foregroundColor(.white)
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(
                                    RoundedRectangle(cornerRadius: 14)
                                        .fill(LinearGradient(colors: [Color(hex: "5e7bb2").opacity(0.92), Color(hex: "455a7d").opacity(0.94)], startPoint: .topLeading, endPoint: .bottomTrailing))
                                )
                            } else if message.attachmentAudioPath != nil {
                                Label(settings.localized("audio"), systemImage: "waveform")
                                    .font(.caption)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(
                                        RoundedRectangle(cornerRadius: 14)
                                            .fill(LinearGradient(colors: [Color(hex: "5e7bb2").opacity(0.92), Color(hex: "455a7d").opacity(0.94)], startPoint: .topLeading, endPoint: .bottomTrailing))
                                    )
                            }

                            if let docName = message.attachmentDocumentName {
                                Label(docName, systemImage: "doc.text")
                                    .font(.caption)
                                    .foregroundColor(.white)
                                    .lineLimit(1)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(
                                        RoundedRectangle(cornerRadius: 14)
                                            .fill(LinearGradient(colors: [Color(hex: "4a7d5e").opacity(0.92), Color(hex: "2d5e40").opacity(0.94)], startPoint: .topLeading, endPoint: .bottomTrailing))
                                    )
                            }

                            if !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                Text(message.content)
                                    .font(.body)
                                    .foregroundColor(.white)
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 10)
                                    .background(
                                        RoundedRectangle(cornerRadius: 18)
                                            .fill(LinearGradient(colors: [Color(hex: "6f93cd"), Color(hex: "455c82")], startPoint: .topLeading, endPoint: .bottomTrailing))
                                    )
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 18)
                                            .stroke(Color.white.opacity(0.16), lineWidth: 1)
                                    )
                            }
                        }
                        .onLongPressGesture {
                            showActions = true
                        }
                    }
                }
            } else {
                if message.isGenerating && message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    TypingIndicator()
                        .padding(.vertical, 6)
                } else {
                    if isEditing {
                        VStack(alignment: .leading, spacing: 8) {
                            TextEditor(text: $editedText)
                                .frame(minHeight: 100)
                                .padding(8)
                                .background(.ultraThinMaterial)
                                .clipShape(RoundedRectangle(cornerRadius: 12))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.16), lineWidth: 1)
                                )
                            HStack(spacing: 10) {
                                Button {
                                    isEditing = false
                                    editedText = ""
                                } label: {
                                    Image(systemName: "xmark")
                                }
                                Button {
                                    let trimmed = editedText.trimmingCharacters(in: .whitespacesAndNewlines)
                                    if !trimmed.isEmpty {
                                        onEditAssistantMessage?(trimmed)
                                        isEditing = false
                                        editedText = ""
                                    }
                                } label: {
                                    Image(systemName: "checkmark")
                                }
                                .disabled(editedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            }
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.68))
                        }
                    } else {
                        ThinkingAwareResultContent(
                            content: message.content,
                            isGenerating: message.isGenerating,
                            preferThinkingWhileStreaming: preferThinkingWhileStreaming,
                            useChatRenderer: true
                        )
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 4)
                        // No long-press gesture here: `.textSelection(.enabled)`
                        // inside the markdown renderer provides the native
                        // long-press-to-select-partial-text-and-copy UX.
                        // Full-copy/edit/regenerate buttons live in the action
                        // row below the bubble.
                    }
                }
            }

            if !isEditing && (
                !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || message.attachmentImagePath != nil
                || message.attachmentAudioPath != nil
                || message.attachmentDocumentName != nil
            ) {
                HStack(spacing: 8) {
                    if message.isFromUser {
                        Spacer()
                    }

                    Button(action: onCopy) {
                        Image(systemName: "doc.on.doc")
                    }
                    .buttonStyle(.plain)
                    .foregroundColor(.white.opacity(0.68))

                    if !message.isFromUser, let onToggleTts {
                        Button(action: onToggleTts) {
                            Image(systemName: isTtsSpeaking ? "stop.fill" : "speaker.wave.2")
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.white.opacity(0.68))
                    }

                    if message.isFromUser,
                       !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       onEditUserMessage != nil {
                        Button {
                            editedText = message.content
                            isEditing = true
                        } label: {
                            Image(systemName: "pencil")
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.white.opacity(0.68))
                    }

                    if !message.isFromUser, onEditAssistantMessage != nil {
                        Button {
                            editedText = message.content
                            isEditing = true
                        } label: {
                            Image(systemName: "pencil")
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.white.opacity(0.68))
                    }

                    if !message.isFromUser, let onRegenerateResponse {
                        Button(action: onRegenerateResponse) {
                            Image(systemName: "arrow.clockwise")
                        }
                        .buttonStyle(.plain)
                        .foregroundColor(.white.opacity(0.68))
                    }

                    if !message.isFromUser,
                       let tokenCount = message.tokenCount,
                       let tps = message.tokensPerSecond,
                       tokenCount > 0 {
                        // Only show stats once thinking is done and there's an answer.
                        // While still in the thinking phase, hide the token badge entirely
                        // so thinking tokens never appear in the count.
                        let hasMarkers = contentHasThinkingMarkers(message.content)
                        let answer = hasMarkers ? getDisplayContentWithoutThinking(message.content) : message.content
                        if !hasMarkers || !answer.isEmpty {
                            Spacer()
                            // The backend already provides responseTokens (answer-only count)
                            // via result.responseTokens in the final callback, so tokenCount
                            // is already the answer token count — no need to subtract an estimate.
                            Label(String(format: settings.localized("tokens_per_second_format"), tokenCount, tps), systemImage: "bolt.fill")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.63))
                        }
                    }
                }
            }

            Text(message.timestamp, style: .time)
                .font(.caption2)
                .foregroundColor(.white.opacity(0.5))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .confirmationDialog(settings.localized("more_options"), isPresented: $showActions) {
            Button(settings.localized("copy_message")) {
                onCopy()
            }
            Button(settings.localized("cancel"), role: .cancel) {}
        }
    }

    private func previewImage(from path: String) -> UIImage? {
        guard let resolvedURL = resolveStoredAttachmentURL(path) else { return nil }

        #if canImport(ImageIO)
        let sourceOptions = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(resolvedURL as CFURL, sourceOptions) else {
            return UIImage(contentsOfFile: resolvedURL.path)
        }

        let thumbOptions = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: false,
            kCGImageSourceThumbnailMaxPixelSize: 640,
        ] as CFDictionary

        if let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbOptions) {
            return UIImage(cgImage: cgImage)
        }
        #endif
        return UIImage(contentsOfFile: resolvedURL.path)
    }
}

// MARK: - Typing Indicator
struct TypingIndicator: View {
    @State private var phase = 0.0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .fill(Color.white.opacity(0.68))
                    .frame(width: 6, height: 6)
                    .scaleEffect(1.0 + 0.4 * sin(phase + Double(i) * .pi / 1.5))
            }
        }
        .onAppear {
            withAnimation(.linear(duration: 1).repeatForever(autoreverses: false)) {
                phase = .pi * 2
            }
        }
    }
}

private enum ParsedSegment: Hashable {
    case text(String)
    case code(language: String?, content: String)
    case math(content: String, isBlock: Bool)
    case table(String)
}

struct RenderMessageSegments: View {
    let displayContent: String

    var body: some View {
        let segments = parseSegments(normalized(displayContent))
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(segments.enumerated()), id: \.offset) { item in
                let segment = item.element
                switch segment {
                case .text(let text):
                    MarkdownMessageText(text: text)
                case .code(let language, let content):
                    VStack(alignment: .leading, spacing: 0) {
                        HStack {
                            if let language, !language.isEmpty {
                                Text(language)
                                    .font(.caption2.weight(.semibold))
                                    .foregroundColor(.white.opacity(0.65))
                            }
                            Spacer()
                            Button {
                                UIPasteboard.general.string = content
                            } label: {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 11))
                                    .foregroundColor(.white.opacity(0.6))
                                    .padding(6)
                                    .background(Color.white.opacity(0.1))
                                    .clipShape(Circle())
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.white.opacity(0.04))

                        ScrollView(.horizontal, showsIndicators: false) {
                            Text(content.trimmingCharacters(in: .newlines))
                                .font(.system(.body, design: .monospaced))
                                .foregroundColor(.white.opacity(0.92))
                                .padding(10)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .background(Color.white.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.white.opacity(0.14), lineWidth: 1)
                    )
                case .math(let content, let isBlock):
                    if isBlock {
                        MathView(equation: content, isBlock: true)
                            .frame(minHeight: 60)
                            .padding(10)
                            .background(Color.white.opacity(0.05))
                            .cornerRadius(10)
                    } else {
                        MathView(equation: content, isBlock: false)
                            .frame(minWidth: 40, maxWidth: .infinity, minHeight: 28, maxHeight: 44)
                    }
                case .table(let content):
                    MarkdownTableView(rawTable: content)
                }
            }
        }
    }

    private func normalized(_ raw: String) -> String {
        var value = raw
        let markers = ["<end_of_turn>", "<|eot_id|>", "<|endoftext|>", "</s>"]
        for marker in markers {
            value = value.replacingOccurrences(of: marker, with: "")
        }
        return value
    }

    private func parseSegments(_ input: String) -> [ParsedSegment] {
        let codePattern = #"```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```"#
        let mathBlockPattern = #"\$\$([\s\S]*?)\$\$"#
        let mathInlinePattern = #"\$(?!\$)([^\n$]+)\$"#
        let tablePattern = #"(?:\n|^)(\|.*\|[ \t]*\n\|[ \t]*[:-].*\|[ \t]*\n(?:\|.*\|[ \t]*\n?)+)"#

        let patterns = [
            (codePattern, 0),
            (mathBlockPattern, 1),
            (mathInlinePattern, 2),
            (tablePattern, 3)
        ]

        var segments: [ParsedSegment] = []
        var cursor = 0
        let nsInput = input as NSString

        while cursor < nsInput.length {
            var bestMatch: (NSTextCheckingResult, Int)? = nil

            for (pattern, id) in patterns {
                if let regex = try? NSRegularExpression(pattern: pattern, options: []),
                   let match = regex.firstMatch(in: input, options: [], range: NSRange(location: cursor, length: nsInput.length - cursor)) {
                    if bestMatch == nil || match.range.location < bestMatch!.0.range.location {
                        bestMatch = (match, id)
                    }
                }
            }

            guard let (match, id) = bestMatch else {
                let remaining = nsInput.substring(from: cursor)
                if !remaining.isEmpty { segments.append(.text(remaining)) }
                break
            }

            if match.range.location > cursor {
                let textPart = nsInput.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
                if !textPart.isEmpty { segments.append(.text(textPart)) }
            }

            switch id {
            case 0: // Code
                let langRange = match.range(at: 1)
                let language = langRange.location != NSNotFound ? nsInput.substring(with: langRange) : nil
                let content = nsInput.substring(with: match.range(at: 2))
                segments.append(.code(language: language, content: content))
            case 1: // Math Block
                let content = nsInput.substring(with: match.range(at: 1))
                segments.append(.math(content: content, isBlock: true))
            case 2: // Math Inline
                let content = nsInput.substring(with: match.range(at: 1))
                segments.append(.math(content: content, isBlock: false))
            case 3: // Table
                let content = nsInput.substring(with: match.range(at: 1))
                segments.append(.table(content))
            default: break
            }

            cursor = match.range.location + match.range.length
        }

        return segments.isEmpty ? [.text(input)] : segments
    }
}

private struct MathView: UIViewRepresentable {
    let equation: String
    let isBlock: Bool

    func makeUIView(context: Context) -> WKWebView {
        let webView = WKWebView()
        webView.backgroundColor = .clear
        webView.isOpaque = false
        webView.scrollView.isScrollEnabled = false
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {
        let isDark = true // App is mainly dark
        let html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
            <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
            <style>
                body {
                    margin: 0;
                    padding: \(isBlock ? "10px" : "2px 4px");
                    display: flex;
                    justify-content: \(isBlock ? "center" : "flex-start");
                    background: transparent;
                    color: \(isDark ? "white" : "black");
                    font-family: -apple-system;
                    font-size: \(isBlock ? "1.1em" : "1.0em");
                }
                .katex-display { margin: 0; }
            </style>
        </head>
        <body>
            <div id="math"></div>
            <script>
                try {
                    katex.render(\(equation.debugDescription), document.getElementById('math'), {
                        throwOnError: false,
                        displayMode: \(isBlock ? "true" : "false")
                    });
                } catch (e) {
                    document.getElementById('math').textContent = \(equation.debugDescription);
                }
            </script>
        </body>
        </html>
        """
        uiView.loadHTMLString(html, baseURL: nil)
    }
}

private struct MarkdownTableView: View {
    let rawTable: String

    var body: some View {
        let rows = parseTable(rawTable)
        VStack(alignment: .leading, spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(0..<rows.count, id: \.self) { rowIndex in
                        HStack(spacing: 0) {
                            ForEach(0..<rows[rowIndex].count, id: \.self) { colIndex in
                                Text(rows[rowIndex][colIndex])
                                    .font(rowIndex == 0 ? .body.bold() : .body)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .frame(minWidth: 80, alignment: .leading)
                                    .background(rowIndex == 0 ? Color.white.opacity(0.12) : Color.clear)
                                    .border(Color.white.opacity(0.15), width: 0.5)
                            }
                        }
                    }
                }
            }
        }
        .background(Color.white.opacity(0.05))
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.white.opacity(0.2), lineWidth: 1)
        )
        .padding(.vertical, 4)
    }

    private func parseTable(_ raw: String) -> [[String]] {
        let lines = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .newlines)
        var result: [[String]] = []
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.replacingOccurrences(of: "|", with: "").replacingOccurrences(of: "-", with: "").replacingOccurrences(of: ":", with: "").trimmingCharacters(in: .whitespaces).isEmpty {
                continue
            }
            let components = line.components(separatedBy: "|")
            let validCells = components.enumerated().filter { (index, _) in
                index > 0 && index < components.count - 1
            }.map { $0.element.trimmingCharacters(in: .whitespaces) }
            if !validCells.isEmpty { result.append(validCells) }
        }
        return result
    }
}

private struct MarkdownMessageText: View {
    let text: String

    var body: some View {
        let normalizedText = text
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\r\n", with: "\n")

        SelectableMarkdownText(text: normalizedText)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Selectable Markdown UITextView Wrapper

/// A `UITextView`-backed view that renders markdown as `NSAttributedString`.
/// This gives native long-press-to-select with highlight handles and copy menu,
/// exactly like ChatGPT / Gemini iOS apps.
private struct SelectableMarkdownText: UIViewRepresentable {
    let text: String

    func makeUIView(context: Context) -> UITextView {
        let tv = UITextView()
        tv.isEditable = false
        tv.isSelectable = true
        tv.isScrollEnabled = false
        tv.backgroundColor = .clear
        tv.textContainerInset = .zero
        tv.textContainer.lineFragmentPadding = 0
        // Do NOT use dataDetectorTypes — it re-runs detection on every
        // attributedText update during streaming, causing links to blink.
        tv.dataDetectorTypes = []
        tv.linkTextAttributes = [
            .foregroundColor: UIColor(red: 0.54, green: 0.71, blue: 0.97, alpha: 1.0) // #8ab4f8
        ]
        // Prevent the text view from absorbing scroll events
        tv.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        tv.setContentHuggingPriority(.defaultHigh, for: .vertical)
        return tv
    }

    func updateUIView(_ uiView: UITextView, context: Context) {
        let attributed = markdownToAttributedString(text)
        // Only update if content changed to avoid resetting selection
        if uiView.attributedText.string != attributed.string {
            uiView.attributedText = attributed
        }
        uiView.invalidateIntrinsicContentSize()
    }

    @MainActor
    func sizeThatFits(_ proposal: ProposedViewSize, uiView: UITextView, context: Context) -> CGSize? {
        let width = proposal.width ?? UIScreen.main.bounds.width
        let size = uiView.sizeThatFits(CGSize(width: width, height: .greatestFiniteMagnitude))
        return CGSize(width: width, height: size.height)
    }

    // MARK: - Markdown → NSAttributedString

    private func markdownToAttributedString(_ markdown: String) -> NSAttributedString {
        let baseFont = UIFont.preferredFont(forTextStyle: .body)
        let baseFontSize = baseFont.pointSize
        let white = UIColor.white
        let dimWhite = UIColor.white.withAlphaComponent(0.7)

        let paragraphStyle = NSMutableParagraphStyle()
        paragraphStyle.lineSpacing = 4
        paragraphStyle.paragraphSpacing = 8

        let result = NSMutableAttributedString()
        let lines = markdown.components(separatedBy: "\n")
        var i = 0

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            // Heading detection
            if let headingMatch = trimmed.range(of: #"^(#{1,3})\s+"#, options: .regularExpression) {
                let level = trimmed[headingMatch].filter { $0 == "#" }.count
                let headingText = String(trimmed[headingMatch.upperBound...])
                let fontSize: CGFloat = level == 1 ? baseFontSize * 1.4 : level == 2 ? baseFontSize * 1.2 : baseFontSize * 1.1
                let headingFont = UIFont.boldSystemFont(ofSize: fontSize)
                let headingPara = NSMutableParagraphStyle()
                headingPara.paragraphSpacingBefore = 12
                headingPara.paragraphSpacing = 6
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: headingFont,
                    .foregroundColor: white,
                    .paragraphStyle: headingPara
                ]
                if result.length > 0 { result.append(NSAttributedString(string: "\n")) }
                result.append(NSAttributedString(string: headingText, attributes: attrs))
                i += 1
                continue
            }

            // Numbered list item: "1. ", "2. ", etc.
            if let listMatch = trimmed.range(of: #"^(\d+)\.\s+"#, options: .regularExpression) {
                let number = String(trimmed[trimmed.startIndex..<listMatch.upperBound]).trimmingCharacters(in: .whitespaces)
                let itemText = String(trimmed[listMatch.upperBound...])
                let listPara = NSMutableParagraphStyle()
                listPara.lineSpacing = 4
                listPara.paragraphSpacing = 4
                listPara.headIndent = 24
                listPara.firstLineHeadIndent = 4
                let tabStop = NSTextTab(textAlignment: .left, location: 24)
                listPara.tabStops = [tabStop]
                if result.length > 0 { result.append(NSAttributedString(string: "\n")) }
                let prefix = NSAttributedString(string: "\(number)\t", attributes: [
                    .font: baseFont,
                    .foregroundColor: white,
                    .paragraphStyle: listPara
                ])
                result.append(prefix)
                result.append(applyInlineFormatting(itemText, font: baseFont, color: white, paragraphStyle: listPara))
                i += 1
                continue
            }

            // Bullet list item: "- ", "* ", "• "
            if let bulletMatch = trimmed.range(of: #"^[-*•]\s+"#, options: .regularExpression) {
                let itemText = String(trimmed[bulletMatch.upperBound...])
                let listPara = NSMutableParagraphStyle()
                listPara.lineSpacing = 4
                listPara.paragraphSpacing = 4
                listPara.headIndent = 24
                listPara.firstLineHeadIndent = 4
                let tabStop = NSTextTab(textAlignment: .left, location: 24)
                listPara.tabStops = [tabStop]
                if result.length > 0 { result.append(NSAttributedString(string: "\n")) }
                let bullet = NSAttributedString(string: "•\t", attributes: [
                    .font: baseFont,
                    .foregroundColor: white,
                    .paragraphStyle: listPara
                ])
                result.append(bullet)
                result.append(applyInlineFormatting(itemText, font: baseFont, color: white, paragraphStyle: listPara))
                i += 1
                continue
            }

            // Blockquote
            if trimmed.hasPrefix(">") {
                let quoteText = String(trimmed.dropFirst().trimmingCharacters(in: .whitespaces))
                let quotePara = NSMutableParagraphStyle()
                quotePara.lineSpacing = 4
                quotePara.paragraphSpacing = 6
                quotePara.firstLineHeadIndent = 16
                quotePara.headIndent = 16
                let quoteFont = UIFont.italicSystemFont(ofSize: baseFontSize)
                if result.length > 0 { result.append(NSAttributedString(string: "\n")) }
                result.append(NSAttributedString(string: quoteText, attributes: [
                    .font: quoteFont,
                    .foregroundColor: dimWhite,
                    .paragraphStyle: quotePara
                ]))
                i += 1
                continue
            }

            // Empty line = paragraph break
            if trimmed.isEmpty {
                result.append(NSAttributedString(string: "\n", attributes: [
                    .font: baseFont,
                    .paragraphStyle: paragraphStyle
                ]))
                i += 1
                continue
            }

            // Regular paragraph text
            if result.length > 0 {
                // Add newline between paragraphs for non-empty content
                let lastChar = (result.string as NSString).substring(from: max(0, result.length - 1))
                if lastChar != "\n" {
                    result.append(NSAttributedString(string: "\n"))
                }
            }
            result.append(applyInlineFormatting(trimmed, font: baseFont, color: white, paragraphStyle: paragraphStyle))
            i += 1
        }

        return result
    }

    /// Apply bold, italic, inline code, and link formatting within a line.
    private func applyInlineFormatting(_ text: String, font: UIFont, color: UIColor, paragraphStyle: NSParagraphStyle) -> NSAttributedString {
        let result = NSMutableAttributedString(string: text, attributes: [
            .font: font,
            .foregroundColor: color,
            .paragraphStyle: paragraphStyle
        ])
        let nsText = text as NSString

        // Bold: **text** or __text__
        applyRegex(#"\*\*(.+?)\*\*|__(.+?)__"#, to: result, nsText: nsText) { range, match in
            let captureRange = match.range(at: 1).location != NSNotFound ? match.range(at: 1) : match.range(at: 2)
            let boldFont = UIFont.boldSystemFont(ofSize: font.pointSize)
            result.addAttribute(.font, value: boldFont, range: captureRange)
            // Remove markers
            let boldText = nsText.substring(with: captureRange)
            result.replaceCharacters(in: range, with: boldText)
        }

        // Re-fetch string after bold replacements
        let afterBold = result.string as NSString

        // Italic: *text* or _text_ (not inside bold)
        applyRegex(#"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)"#, to: result, nsText: afterBold) { range, match in
            let captureRange = match.range(at: 1).location != NSNotFound ? match.range(at: 1) : match.range(at: 2)
            let italicFont = UIFont.italicSystemFont(ofSize: font.pointSize)
            result.addAttribute(.font, value: italicFont, range: captureRange)
            let italicText = afterBold.substring(with: captureRange)
            result.replaceCharacters(in: range, with: italicText)
        }

        // Inline code: `text`
        let afterItalic = result.string as NSString
        applyRegex(#"`([^`]+)`"#, to: result, nsText: afterItalic) { range, match in
            let codeRange = match.range(at: 1)
            let codeFont = UIFont.monospacedSystemFont(ofSize: font.pointSize * 0.9, weight: .regular)
            let codeColor = UIColor(red: 0.79, green: 0.82, blue: 0.85, alpha: 1.0) // #c9d1d9
            result.addAttribute(.font, value: codeFont, range: codeRange)
            result.addAttribute(.foregroundColor, value: codeColor, range: codeRange)
            result.addAttribute(.backgroundColor, value: UIColor.white.withAlphaComponent(0.1), range: codeRange)
            let codeText = afterItalic.substring(with: codeRange)
            result.replaceCharacters(in: range, with: codeText)
        }

        // URLs: use NSDataDetector (same engine as dataDetectorTypes) to find links
        // and bake them into the attributed string. This avoids the blink caused by
        // dataDetectorTypes re-running detection on every UITextView update.
        let afterCode = result.string as NSString
        if let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue) {
            let linkMatches = detector.matches(in: afterCode as String, options: [], range: NSRange(location: 0, length: afterCode.length))
            for linkMatch in linkMatches {
                if let url = linkMatch.url {
                    result.addAttribute(.link, value: url, range: linkMatch.range)
                }
            }
        }

        return result
    }

    /// Helper: apply regex replacements in reverse order to preserve indices.
    private func applyRegex(_ pattern: String, to attrStr: NSMutableAttributedString, nsText: NSString, handler: (NSRange, NSTextCheckingResult) -> Void) {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return }
        let matches = regex.matches(in: nsText as String, options: [], range: NSRange(location: 0, length: nsText.length))
        // Process in reverse to keep ranges valid
        for match in matches.reversed() {
            handler(match.range, match)
        }
    }
}

// MARK: - Drawer Panel
struct ChatDrawerPanel: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject var vm: ChatViewModel
    let onClose: () -> Void
    let onNavigateBack: () -> Void
    let onNavigateToModels: () -> Void
    let onNavigateToSettings: () -> Void
    @State private var showDeleteAllAlert = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        vm.newChat()
                        onClose()
                    } label: {
                        Label(settings.localized("drawer_new_chat"), systemImage: "plus.bubble.fill")
                            .foregroundColor(ApolloPalette.accentStrong)
                            .fontWeight(.semibold)
                    }
                    if !vm.chatSessions.isEmpty {
                        Button {
                            showDeleteAllAlert = true
                        } label: {
                            Label(settings.localized("drawer_clear_all_chats"), systemImage: "trash.fill")
                                .foregroundColor(ApolloPalette.destructive)
                                .fontWeight(.semibold)
                        }
                    }
                }

                Section(settings.localized("drawer_recent_chats")) {
                    if vm.chatSessions.isEmpty {
                        Text(settings.localized("drawer_no_chats"))
                            .foregroundColor(.secondary)
                            .font(.subheadline)
                    } else {
                        ForEach(vm.chatSessions) { session in
                            Button {
                                vm.stopAutoReadout()
                                vm.currentSessionId = session.id
                                onClose()
                            } label: {
                                HStack {
                                    Image(systemName: "bubble.left.fill")
                                        .foregroundColor(ApolloPalette.accent.opacity(0.9))
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(session.title)
                                            .foregroundColor(.primary)
                                            .lineLimit(1)
                                        Text(session.createdAt, style: .date)
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                    Spacer()
                                    if session.id == vm.currentSessionId {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundColor(ApolloPalette.accentStrong)
                                    }
                                }
                            }
                            .swipeActions(edge: .trailing) {
                                Button(role: .destructive) {
                                    vm.deleteSession(session.id)
                                } label: {
                                    Label(settings.localized("action_delete"), systemImage: "trash")
                                }
                            }
                        }
                    }
                }


            }
            .scrollContentBackground(.hidden)
            .background(ApolloLiquidBackground())
            .navigationTitle(settings.localized("drawer_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    // Back arrow to Home - same as Android drawer's ArrowBack
                    Button {
                        onClose()
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                            onNavigateBack()
                        }
                    } label: {
                        Image(systemName: "arrow.left")
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(settings.localized("done"), action: onClose)
                }
            }
        }
        .alert(settings.localized("dialog_delete_all_chats_title"), isPresented: $showDeleteAllAlert) {
            Button(settings.localized("action_delete_all"), role: .destructive) {
                ChatStore.shared.clearAll()
                vm.newChat()
            }
            Button(settings.localized("action_cancel"), role: .cancel) {}
        } message: {
            Text(settings.localized("dialog_delete_all_chats_message"))
        }
    }
}


// MARK: - ChatScreen
struct ChatScreen: View {
    @EnvironmentObject var settings: AppSettings
    @StateObject private var vm = ChatViewModel()
    @ObservedObject private var ttsManager = OnDeviceTtsManager.shared
    @ObservedObject private var interstitialManager = InterstitialAdManager.shared
    var onNavigateToSettings: () -> Void
    var onNavigateToModels: () -> Void
    var onNavigateBack: () -> Void

    @State private var showPremiumFromAd = false
    @State private var showDrawer = false
    @State private var showSettings = false
    @State private var copiedMessageId: UUID? = nil
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var showPhotosPicker = false
    @State private var attachedImageURL: URL?
    @State private var attachedAudioURL: URL?
    @State private var attachedDocumentURL: URL?
    @State private var attachedDocumentName: String?
    @State private var previewImagePath: String?
    @State private var showDocumentImporter = false
    @State private var showAudioTranscribeImporter = false
    @State private var showAttachMenu = false
    @State private var hasInitializedChatSession = false
    @StateObject private var micTranscriber = ChatMicTranscriber()
    @StateObject private var audioRecorder = AudioRecorder()
    @FocusState private var isComposerFocused: Bool
    @State private var userHasScrolledUp = false
    @State private var isAtBottom = true
    @State private var scrollDetectorCoordinator: ScrollDragDetector.Coordinator?


    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .bottom) {
                ScrollViewReader { proxy in
                    ZStack(alignment: .bottom) {
                    ScrollView {
                        // VStack instead of LazyVStack: the lazy view recycling can
                        // leave the scroll area blank when streaming rapidly mutates
                        // the last message's height near the context window limit.
                        // Chat sessions are capped in length so eager layout is fine.
                        VStack(spacing: 12) {
                            ScrollDragDetector(onDragBegan: {
                                userHasScrolledUp = true
                            }, onAttached: { coordinator in
                                scrollDetectorCoordinator = coordinator
                            })
                            .frame(width: 0, height: 0)
                            if vm.messages.isEmpty {
                                emptyState
                            } else {
                                ForEach(vm.messages) { msg in
                                    messageBubble(for: msg)
                                }
                                
                                // Bottom sentinel to check when user scrolls back to the bottom
                                Color.clear
                                    .frame(height: 1)
                                    .id("bottom_sentinel")
                                    .background(
                                        GeometryReader { sentinelGeo in
                                            let minY = sentinelGeo.frame(in: .named("chat_geo")).minY
                                            let atBottom = minY <= geo.size.height - 130
                                            Color.clear
                                                .preference(key: AtBottomPreferenceKey.self, value: atBottom)
                                        }
                                    )
                            }
                        }
                        .padding(.vertical, 12)
                        .padding(.bottom, 12)
                    }
                    .safeAreaPadding(.bottom, 130)
                    .scrollDismissesKeyboard(.interactively)
                    .onTapGesture {
                        isComposerFocused = false
                    }
                    .onChange(of: userHasScrolledUp) { _, scrolledUp in
                        vm.viewIsScrolledUp = scrolledUp
                        if !scrolledUp {
                            vm.flushThrottledStreamUpdate()
                            if let last = vm.messages.last {
                                withAnimation {
                                    proxy.scrollTo(last.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                    .onChange(of: vm.messages.count) { _, _ in
                        guard !userHasScrolledUp else { return }
                        if let last = vm.messages.last {
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                    }
                    .onChange(of: vm.messages.last?.content) { _, _ in
                        guard vm.isGenerating, !userHasScrolledUp, let last = vm.messages.last else { return }
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                    .onChange(of: vm.currentSessionId) { _, _ in
                        userHasScrolledUp = false
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                            if let last = vm.messages.last {
                                withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                            }
                        }
                    }
                    .onChange(of: isComposerFocused) { _, focused in
                        if focused {
                            userHasScrolledUp = false
                        }
                        if focused, let last = vm.messages.last {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                                withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                            }
                        }
                    }
                    scrollToBottomButton(proxy: proxy)
                    } // ZStack
                }

                // Floating composer panel on top layer
                VStack(spacing: 8) {
                    if let _ = copiedMessageId {
                        Text(settings.localized("message_copied"))
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(.ultraThinMaterial)
                            .clipShape(Capsule())
                            .transition(.scale.combined(with: .opacity))
                    }
                    
                    VStack(spacing: 0) {
                        if attachedImageURL != nil || attachedAudioURL != nil || attachedDocumentURL != nil {
                            HStack(spacing: 8) {
                                if attachedImageURL != nil {
                                    attachmentPill(label: settings.localized("vision"), icon: "photo") {
                                        attachedImageURL = nil
                                        selectedImageItem = nil
                                    }
                                }
                                if attachedAudioURL != nil {
                                    attachmentPill(label: settings.localized("audio"), icon: "waveform") {
                                        attachedAudioURL = nil
                                    }
                                }
                                if let docName = attachedDocumentName {
                                    attachmentPill(label: docName, icon: "doc.text") {
                                        attachedDocumentURL = nil
                                        attachedDocumentName = nil
                                    }
                                }
                                Spacer()
                            }
                            .padding(.horizontal, 16)
                            .padding(.top, 12)
                            .padding(.bottom, 4)
                        }
                        
                        // Text field fills the top of the box
                        ZStack(alignment: .leading) {
                            if micTranscriber.isPreparing && !shouldUseModelAudioInput {
                                Text(settings.localized("preparing_mic"))
                                    .foregroundColor(.white.opacity(0.45))
                                    .font(.body)
                                    .padding(.horizontal, 16)
                                    .padding(.top, 14)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            TextField(settings.localized("type_a_message"), text: $vm.inputText, axis: .vertical)
                                .lineLimit(1...6)
                                .padding(.horizontal, 16)
                                .padding(.top, 14)
                                .padding(.bottom, 8)
                                .focused($isComposerFocused)
                                .foregroundColor(.white)
                                .onSubmit {
                                    if vm.isWebSearchEnabled {
                                        if vm.sendMessageWithWebSearch(documentURL: attachedDocumentURL, documentName: attachedDocumentName) {
                                            clearAttachments()
                                        }
                                    } else {
                                        if vm.sendMessage(imageURL: attachedImageURL, audioURL: attachedAudioURL, documentURL: attachedDocumentURL, documentName: attachedDocumentName) {
                                            clearAttachments()
                                        }
                                    }
                                }
                        }
                        
                        // Action row — all buttons in one row at the bottom of the box
                        HStack(spacing: 6) {
                            // ─── + attach ────────────────────────────────────────────
                            Button {
                                showAttachMenu = true
                            } label: {
                                Image(systemName: "plus")
                                    .font(.system(size: 18, weight: .semibold))
                                    .foregroundColor(
                                        (attachedImageURL != nil || attachedDocumentURL != nil)
                                            ? ApolloPalette.accentStrong : .white.opacity(0.75)
                                    )
                                    .frame(width: 34, height: 34)
                                    .background(Color.white.opacity(0.09))
                                    .clipShape(Circle())
                            }
                            .disabled(vm.isGenerating)
                            .confirmationDialog("", isPresented: $showAttachMenu) {
                                let attachSelectedModel = chatModel(named: vm.selectedModelName)
                                let attachCanVision = vm.enableVision
                                    && (attachSelectedModel?.supportsVision == true)
                                    && (attachSelectedModel.map { LLMBackend.shared.isVisionProjectorAvailable(for: $0) } ?? false)
                                if attachCanVision {
                                    Button(settings.localized("images"))    { showPhotosPicker = true }
                                }
                                Button(settings.localized("documents")) { showDocumentImporter = true }
                                Button(settings.localized("audio_file")){ showAudioTranscribeImporter = true }
                                Button(settings.localized("cancel"), role: .cancel) {}
                            }
                            
                            // ─── 🌐 Web Search toggle ─────────────────────────────────
                            Button {
                                vm.isWebSearchEnabled.toggle()
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: vm.isSearching ? "arrow.triangle.2.circlepath" : "globe")
                                        .font(.system(size: 12, weight: .semibold))
                                        .symbolEffect(.pulse, isActive: vm.isSearching)
                                    Text(settings.localized("web_search"))
                                        .font(.system(size: 12, weight: .semibold))
                                }
                                .foregroundColor(vm.isWebSearchEnabled ? .black : .white.opacity(0.75))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(vm.isWebSearchEnabled ? Color.white : Color.white.opacity(0.09))
                                .clipShape(Capsule())
                                .overlay(
                                    Capsule().stroke(
                                        vm.isWebSearchEnabled ? Color.clear : Color.white.opacity(0.15),
                                        lineWidth: 1
                                    )
                                )
                            }
                            .disabled(vm.isGenerating)
                            .animation(.easeInOut(duration: 0.15), value: vm.isWebSearchEnabled)
                            
                            Spacer()
                            
                            // ─── context usage ring ─────────────────────────────
                            ZStack {
                                Circle()
                                    .inset(by: 0.75)
                                    .stroke(Color.white.opacity(0.18), lineWidth: 1.5)
                                Circle()
                                    .inset(by: 1.0)
                                    .trim(from: 0, to: vm.contextUsageFractionDisplay)
                                    .stroke(
                                        vm.contextUsageFractionRaw < 0.90 ? ApolloPalette.accentStrong : ApolloPalette.warning,
                                        style: StrokeStyle(lineWidth: 2.0, lineCap: .round)
                                    )
                                    .rotationEffect(.degrees(-90))
                                
                                Text(vm.contextUsageFractionRaw < 0.995 ? vm.contextUsageLabel : "!")
                                    .font(.system(size: 8, weight: .bold, design: .rounded))
                            }
                            .frame(width: 32, height: 32)
                            .padding(.trailing, 4)
                            .accessibilityLabel("Context usage \(vm.contextUsageLabel)")
                            
                            // ─── mic ─────────────────────────────────────────────────
                            Button {
                                if shouldUseModelAudioInput {
                                    if audioRecorder.isRecording {
                                        if let url = audioRecorder.stopRecording() {
                                            attachedAudioURL = url
                                        }
                                    } else {
                                        Task { @MainActor in
                                            let destination = attachmentStorageDirectory()
                                                .appendingPathComponent("audio_\(UUID().uuidString)")
                                                .appendingPathExtension("wav")
                                            _ = await audioRecorder.startRecording(outputURL: destination, autoStopAfterSilence: false, isFloat32Wav: true) { url in
                                                Task { @MainActor in
                                                    attachedAudioURL = url
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if micTranscriber.isRecording {
                                        Task {
                                            // Text is already in vm.inputText via live onChange — just stop.
                                            _ = await micTranscriber.stopLive()
                                        }
                                    } else {
                                        Task { await micTranscriber.startLive() }
                                    }
                                }
                            } label: {
                                ZStack {
                                    Circle()
                                        .fill(.ultraThinMaterial)
                                        .frame(width: 32, height: 32)
                                        .overlay(
                                            Circle()
                                                .stroke(Color.white.opacity(0.18), lineWidth: 1)
                                        )
                                    if micTranscriber.isRecording || audioRecorder.isRecording {
                                        Circle()
                                            .fill(Color.red.opacity(0.25))
                                            .frame(width: 32, height: 32)
                                    }
                                    Image(systemName: (micTranscriber.isPreparing || audioRecorder.isPreparing) ? "ellipsis"
                                          : (micTranscriber.isRecording || audioRecorder.isRecording) ? "stop.fill" : "mic.fill")
                                        .font(.system(size: 14, weight: .semibold))
                                        .foregroundColor((micTranscriber.isRecording || audioRecorder.isRecording) ? .red : .white)
                                }
                            }
                            .disabled(vm.isGenerating)
                            
                            // ─── send / stop ──────────────────────────────────────────
                            Button {
                                isComposerFocused = false
                                // Stop mic recording immediately so the spoken text stays in inputText.
                                if micTranscriber.isRecording {
                                    Task { _ = await micTranscriber.stopLive() }
                                }
                                if vm.isGenerating {
                                    vm.stopGeneration()
                                } else if vm.isWebSearchEnabled {
                                    if vm.sendMessageWithWebSearch(documentURL: attachedDocumentURL, documentName: attachedDocumentName) {
                                        clearAttachments()
                                    }
                                } else {
                                    if vm.sendMessage(imageURL: attachedImageURL, audioURL: attachedAudioURL, documentURL: attachedDocumentURL, documentName: attachedDocumentName) {
                                        clearAttachments()
                                    }
                                }
                            } label: {
                                Image(systemName: vm.isGenerating ? "stop.fill" : "arrow.up")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(vm.isGenerating ? .white : .black)
                                    .frame(width: 32, height: 32)
                                    .background(vm.isGenerating ? Color.red.opacity(0.8) : Color.white)
                                    .clipShape(Circle())
                            }
                            .disabled(
                                !vm.isGenerating
                                && vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                && attachedImageURL == nil
                                && attachedAudioURL == nil
                                && attachedDocumentURL == nil
                            )
                        }
                        .padding(.horizontal, 10)
                        .padding(.bottom, 10)
                    }
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 24))
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .stroke(Color.white.opacity(0.12), lineWidth: 1)
                    )
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 12)
                .padding(.top, 4)
                .animation(.easeOut(duration: 0.2), value: isComposerFocused)
            }
        }
        .coordinateSpace(name: "chat_geo")
        .apolloScreenBackground()
        .onPreferenceChange(AtBottomPreferenceKey.self) { atBottom in
            isAtBottom = atBottom
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            BannerAdContainer()
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    showDrawer = true
                } label: {
                    Image(systemName: "line.3.horizontal")
                }
            }
            ToolbarItem(placement: .principal) {
                Button {
                    showSettings = true
                } label: {
                    HStack(spacing: 4) {
                        Text(vm.selectedModelName)
                            .font(.system(size: 14, weight: .bold))
                            .foregroundColor(.white)
                            .lineLimit(1)
                            .truncationMode(.tail)
                        Image(systemName: "chevron.right")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(.white.opacity(0.78))
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(vm.isBackendLoading ? Color.orange.opacity(0.26) : Color.white.opacity(0.12))
                    .clipShape(Capsule())
                    .overlay(
                        Capsule()
                            .stroke(Color.white.opacity(0.18), lineWidth: 1)
                    )
                }
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    vm.newChat()
                } label: {
                    Image(systemName: "square.and.pencil")
                }
            }
        }
        .sheet(isPresented: $showSettings) {
             ChatSettingsSheet(vm: vm)
                .environmentObject(settings)
        }
        // Show premium upsell after every interstitial ad
        .sheet(isPresented: $showPremiumFromAd) {
            PremiumScreen()
                .environmentObject(settings)
        }
        .onChange(of: interstitialManager.showPremiumAfterAd) { _, triggered in
            if triggered {
                showPremiumFromAd = true
                interstitialManager.showPremiumAfterAd = false
            }
        }
        .fullScreenCover(isPresented: Binding(
            get: { previewImagePath != nil },
            set: { isPresented in
                if !isPresented {
                    previewImagePath = nil
                }
            }
        )) {
            FullScreenImagePreview(path: previewImagePath) {
                previewImagePath = nil
            }
        }
        .sheet(isPresented: $showDrawer) {
            ChatDrawerPanel(
                vm: vm,
                onClose: { showDrawer = false },
                onNavigateBack: onNavigateBack,
                onNavigateToModels: onNavigateToModels,
                onNavigateToSettings: onNavigateToSettings
            )
            .environmentObject(settings)
        }
        // Audio transcribe importer stays here as the primary chain modifier.
        .fileImporter(
            isPresented: $showAudioTranscribeImporter,
            allowedContentTypes: [.audio, .mpeg4Audio, .mp3],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let sourceURL = urls.first else { return }
            Task { @MainActor in
                if shouldUseModelAudioInput {
                    if let convertedURL = prepareGemmaAudioInput(
                        from: sourceURL,
                        destinationDirectory: persistentAttachmentDirectoryURL(),
                        filePrefix: "chat_audio"
                    ) {
                        attachedAudioURL = convertedURL
                    }
                } else {
                    let speechURL = prepareGemmaAudioInput(
                        from: sourceURL,
                        destinationDirectory: FileManager.default.temporaryDirectory,
                        filePrefix: "chat_speech"
                    ) ?? sourceURL
                    let transcript = await micTranscriber.transcribeFile(speechURL)
                    if !transcript.isEmpty {
                        vm.inputText += (vm.inputText.isEmpty ? "" : " ") + transcript
                    }
                }
            }
        }
        // Document importer is on a separate hierarchy level to avoid the SwiftUI
        // limitation where only the last .fileImporter in a modifier chain presents.
        .background(
            EmptyView()
                .fileImporter(
                    isPresented: $showDocumentImporter,
                    allowedContentTypes: DocumentTextExtractor.supportedTypes,
                    allowsMultipleSelection: false
                ) { result in
                    guard case .success(let urls) = result, let sourceURL = urls.first else { return }
                    attachedDocumentURL = sourceURL
                    attachedDocumentName = sourceURL.lastPathComponent
                    Task {
                        if let text = try? DocumentTextExtractor.extract(from: sourceURL) {
                            await MainActor.run { vm.pendingAttachedDocumentChars = min(text.count, 8000) }
                        }
                    }
                }
        )
        .photosPicker(isPresented: $showPhotosPicker, selection: $selectedImageItem, matching: .images)
        .onChange(of: selectedImageItem) { _, item in
            guard let item else {
                attachedImageURL = nil
                return
            }

            Task {
                if let sourceURL = try? await item.loadTransferable(type: URL.self),
                   let copiedURL = copyAttachmentToTemp(sourceURL, preferredExtension: sourceURL.pathExtension) {
                    await MainActor.run {
                        attachedImageURL = copiedURL
                    }
                    return
                }

                if let data = try? await item.loadTransferable(type: Data.self) {
                    let preferredExt = item.supportedContentTypes
                        .compactMap { $0.preferredFilenameExtension }
                        .first ?? "bin"

                    await MainActor.run {
                        attachedImageURL = writeAttachmentData(data, preferredExtension: preferredExt)
                    }
                }
            }
        }
        .onChange(of: vm.enableVision) { _, enabled in
            if !enabled {
                attachedImageURL = nil
                selectedImageItem = nil
            }
        }
        .onChange(of: vm.enableAudio) { _, enabled in
            if !enabled {
                attachedAudioURL = nil
            }
        }
        .onChange(of: vm.selectedModelName) { _, _ in
            let selectedModel = ModelData.allModels().first(where: { $0.name == vm.selectedModelName })
            let canAttachVision = (selectedModel?.supportsVision == true) && vm.enableVision
            let canAttachAudio = (selectedModel?.supportsAudio == true) && vm.enableAudio

            if !canAttachVision {
                attachedImageURL = nil
                selectedImageItem = nil
            }
            if !canAttachAudio {
                attachedAudioURL = nil
            }
        }
        .onChange(of: micTranscriber.liveText) { _, newText in
            // Mirror live transcript into the input field in real time.
            guard !shouldUseModelAudioInput, !newText.isEmpty else { return }
            vm.inputText = newText
        }
        .onAppear {
            guard !hasInitializedChatSession else { return }
            hasInitializedChatSession = true
            vm.unloadModel()
            Task {
                await RunAnywhere.refreshModelRegistry()
                await RagServiceManager.shared.initialize(modelId: AppSettings.shared.selectedEmbeddingModelId)
            }
        }
        .onDisappear {
            vm.stopAutoReadout()
            vm.unloadModel()
        }
    }

    private func clearAttachments() {
        attachedImageURL = nil
        attachedAudioURL = nil
        attachedDocumentURL = nil
        attachedDocumentName = nil
        selectedImageItem = nil
        vm.pendingAttachedDocumentChars = 0
    }

    private func attachmentPill(label: String, icon: String, onRemove: @escaping () -> Void) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
            Text(label)
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill")
            }
        }
        .font(.caption)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial)
        .clipShape(Capsule())
        .overlay(
            Capsule()
                .stroke(Color.white.opacity(0.14), lineWidth: 1)
        )
    }

    private func writeAttachmentData(_ data: Data, preferredExtension: String) -> URL? {
        let dir = attachmentStorageDirectory()
        let ext = preferredExtension.isEmpty ? "bin" : preferredExtension
        let url = dir.appendingPathComponent(UUID().uuidString).appendingPathExtension(ext)
        do {
            try data.write(to: url, options: .atomic)
            return url
        } catch {
            return nil
        }
    }

    private func copyAttachmentToTemp(_ sourceURL: URL, preferredExtension: String) -> URL? {
        let dir = attachmentStorageDirectory()
        let ext = preferredExtension.isEmpty ? sourceURL.pathExtension : preferredExtension
        let destinationURL = dir.appendingPathComponent(UUID().uuidString).appendingPathExtension(ext)
        let didStartScopedAccess = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if didStartScopedAccess {
                sourceURL.stopAccessingSecurityScopedResource()
            }
        }
        do {
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            return destinationURL
        } catch {
            return nil
        }
    }

    private func attachmentStorageDirectory() -> URL {
        persistentAttachmentDirectoryURL()
    }

    private var shouldUseModelAudioInput: Bool {
        guard let model = chatModel(named: vm.selectedModelName) else { return false }
        guard model.isGemma4LiteRTLM, vm.enableAudio else { return false }
        return vm.loadedModelName == vm.selectedModelName
    }

    @ViewBuilder
    private func scrollToBottomButton(proxy: ScrollViewProxy) -> some View {
        if userHasScrolledUp {
            HStack {
                Spacer()
                Button {
                    userHasScrolledUp = false
                    if let coordinator = scrollDetectorCoordinator {
                        coordinator.scrollToBottom()
                    } else if let last = vm.messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                } label: {
                    Image(systemName: "chevron.down")
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(.primary.opacity(0.7))
                        .frame(width: 36, height: 36)
                        .background(.ultraThinMaterial, in: Circle())
                        .overlay(Circle().strokeBorder(.white.opacity(0.08), lineWidth: 0.5))
                }
                .buttonStyle(.plain)
                .padding(.trailing, 16)
                .padding(.bottom, 148)
            }
            .transition(.opacity.combined(with: .scale(scale: 0.8, anchor: .bottomTrailing)))
            .animation(.spring(response: 0.3), value: userHasScrolledUp)
        }
    }

    @ViewBuilder
    private func messageBubble(for msg: ChatMessage) -> some View {
        let isLatestAssistant = (msg.id == vm.latestAssistantMessageId)
        let canRegenerate = isLatestAssistant && !vm.isGenerating && !msg.isGenerating
        let canEditUser = msg.isFromUser && msg.id == vm.latestUserMessageId && !vm.isGenerating
        let canEditAssistant = !msg.isFromUser && !vm.isGenerating && !msg.isGenerating
        let modelSupportsThinking = chatModel(named: vm.selectedModelName)?.supportsThinking == true
        let useStreamingThinkingHeuristic = supportsUnmarkedStreamingThinkingHeuristic(forModelNamed: vm.selectedModelName)
        let regenerateAction: (() -> Void)? = canRegenerate ? {
            userHasScrolledUp = false
            vm.regenerateResponse(for: msg.id)
        } : nil
        
        MessageBubble(
            message: msg,
            preferThinkingWhileStreaming: modelSupportsThinking && vm.enableThinking && useStreamingThinkingHeuristic,
            onCopy: {
                vm.copyMessage(msg)
                copiedMessageId = msg.id
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    copiedMessageId = nil
                }
            },
            onOpenImage: { imagePath in
                previewImagePath = imagePath
            },
            onEditUserMessage: { updatedPrompt in
                if canEditUser {
                    vm.editUserPrompt(msg.id, newText: updatedPrompt)
                }
            },
            onEditAssistantMessage: { updatedResponse in
                if canEditAssistant {
                    vm.editAssistantMessage(msg.id, newText: updatedResponse)
                }
            },
            onRegenerateResponse: regenerateAction,
            onToggleTts: !msg.isFromUser && !msg.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? {
                let contentToSpeak = contentHasThinkingMarkers(msg.content) ? getDisplayContentWithoutThinking(msg.content) : msg.content
                ttsManager.toggleSpeaking(
                    contentToSpeak,
                    fallbackLanguage: settings.selectedLanguage,
                    key: msg.id.uuidString
                )
            } : nil,
            isTtsSpeaking: ttsManager.isSpeaking(key: msg.id.uuidString)
        )
        .id(msg.id)
        .padding(.horizontal, 16)
    }

    var emptyState: some View {
        VStack(spacing: 20) {
            Spacer(minLength: 60)
            
            Text(settings.localized("welcome_to_llm_hub"))
                .font(.title2.bold())
                .foregroundColor(.white)
                
            if downloadedModels.isEmpty {
                Text(settings.localized("no_models_downloaded"))
                    .foregroundColor(.white.opacity(0.68))
                Button {
                    onNavigateToModels()
                } label: {
                    Label(settings.localized("download_a_model"), systemImage: "arrow.down.circle")
                }
                .buttonStyle(ApolloIconButtonStyle())
            } else if vm.selectedModelName == settings.localized("no_model_selected") {
                Text(settings.localized("load_model_to_start"))
                    .foregroundColor(.white.opacity(0.68))
            }
        }
        .padding(.horizontal, 32)
    }
    
    private var downloadedModels: [AIModel] {
        let legacyModelsDir: URL? = {
            guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
            return documentsDir.appendingPathComponent("models")
        }()

        var models = ModelData.allModels().filter { model in
            if model.isDependencyOnly { return false }
            if model.name.hasPrefix("Translate Gemma") { return false }
            if model.category == .embedding || model.category == .imageGeneration || model.category == .videoGeneration || model.category == .imageUpscale { return false }

            guard ModelData.isModelFullyAvailableLocally(model) else { return false }
            return true
        }

        if let appleModel = chatAppleFoundationModelIfAvailable(),
           !models.contains(where: { $0.id == appleModel.id }) {
            models.append(appleModel)
        }

        return models
    }
}

private struct FullScreenImagePreview: View {
    let path: String?
    let onDismiss: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let uiImage = loadImage() {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(20)
            } else {
                Text("Image unavailable")
                    .foregroundColor(.white.opacity(0.9))
                    .font(.headline)
            }

            VStack {
                HStack {
                    Spacer()
                    Button(action: onDismiss) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 30))
                            .foregroundColor(.white.opacity(0.95))
                    }
                }
                .padding(.top, 12)
                .padding(.horizontal, 16)

                Spacer()
            }
        }
        .contentShape(Rectangle())
        .onTapGesture {
            onDismiss()
        }
    }

    private func loadImage() -> UIImage? {
        guard let resolvedURL = resolveStoredAttachmentURL(path) else { return nil }
        return UIImage(contentsOfFile: resolvedURL.path)
    }
}

// MARK: - PreferenceKey for Scroll-to-Bottom detection

struct AtBottomPreferenceKey: PreferenceKey {
    static let defaultValue: Bool = true
    static func reduce(value: inout Bool, nextValue: () -> Bool) {
        value = nextValue()
    }
}

// Placed *inside* scroll content so it is a real UIScrollView descendant.
// Attaches a target to panGestureRecognizer — no delegate replacement needed.
// scrollToBottom() stops any active deceleration and jumps to bottom via UIScrollView directly.
private struct ScrollDragDetector: UIViewRepresentable {
    let onDragBegan: @MainActor () -> Void
    let onAttached: @MainActor (Coordinator) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onDragBegan: onDragBegan) }

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        view.isUserInteractionEnabled = false
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        let coordinator = context.coordinator
        let onAttached = onAttached
        Task { @MainActor in
            guard let scrollView = uiView.nearestScrollView() else { return }
            let wasAttached = coordinator.isAttached
            coordinator.attach(to: scrollView)
            if !wasAttached { onAttached(coordinator) }
        }
    }

    @MainActor
    class Coordinator: NSObject {
        let onDragBegan: @MainActor () -> Void
        private weak var attachedScrollView: UIScrollView?

        init(onDragBegan: @escaping @MainActor () -> Void) {
            self.onDragBegan = onDragBegan
        }

        var isAttached: Bool { attachedScrollView != nil }

        func attach(to scrollView: UIScrollView) {
            guard scrollView !== attachedScrollView else { return }
            attachedScrollView?.panGestureRecognizer.removeTarget(self, action: nil)
            attachedScrollView = scrollView
            scrollView.panGestureRecognizer.addTarget(self, action: #selector(handlePan(_:)))
        }

        func scrollToBottom() {
            guard let scrollView = attachedScrollView else { return }
            let maxOffset = scrollView.contentSize.height
                - scrollView.bounds.height
                + scrollView.contentInset.bottom
            guard maxOffset > 0 else { return }
            // setContentOffset stops deceleration immediately, unlike scrollTo
            scrollView.setContentOffset(CGPoint(x: 0, y: maxOffset), animated: true)
        }

        @objc private func handlePan(_ recognizer: UIPanGestureRecognizer) {
            if recognizer.state == .began {
                onDragBegan()
            }
        }
    }
}

private extension UIView {
    func nearestScrollView() -> UIScrollView? {
        if let sv = self as? UIScrollView { return sv }
        return superview?.nearestScrollView()
    }
}
