import Foundation
import SwiftSoup

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Fetches one URL directly and returns an OpenCode-style `title`/`metadata`/`output` result.
public struct WebFetch {
  private let httpTransport: HttpTransport

  /// Creates a direct web fetch tool.
  public init(httpTransport: HttpTransport = URLSessionHttpTransport()) {
    self.httpTransport = httpTransport
  }

  /// Fetches one HTTP(S) URL with async/await by wrapping the completion-handler API.
  public func fetch(url: URL, options: WebFetchOptions = WebFetchOptions()) async throws
    -> WebFetchResult
  {
    try await withCheckedThrowingContinuation { continuation in
      fetch(url: url, options: options) { result in
        continuation.resume(with: result)
      }
    }
  }
  /// Fetches one HTTP(S) URL and calls `completion` with converted output.
  public func fetch(
    url: URL,
    options: WebFetchOptions = WebFetchOptions(),
    completion: @escaping (Result<WebFetchResult, Error>) -> Void
  ) {
    guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
      completion(.failure(WebSearchError.unsupportedScheme(url.scheme ?? "")))
      return
    }

    var request = URLRequest(url: url)
    request.httpMethod = "GET"
    request.timeoutInterval = options.timeout
    let userAgent = options.userAgent.trimmingCharacters(in: .whitespacesAndNewlines)
    request.setValue(
      userAgent.isEmpty ? WebSearchDefaultUserAgent : userAgent, forHTTPHeaderField: "User-Agent")
    request.setValue(options.format.acceptHeader, forHTTPHeaderField: "Accept")

    let start = Date()
    httpTransport.data(for: request) { result in
      do {
        let (data, response) = try result.get()
        let elapsed = Date().timeIntervalSince(start)
        guard (200..<300).contains(response.statusCode) else {
          throw WebSearchError.httpStatus(
            response.statusCode, response.url, body: DuckDuckGoSearch.decodeBody(data),
            headers: response.allHeaderFields.reduce(into: [String: String]()) {
              guard let key = $1.key as? String else { return }
              $0[key] = String(describing: $1.value)
            }, byteCount: data.count)
        }
        guard data.count <= options.maxBytes else {
          throw WebSearchError.responseTooLarge(data.count, options.maxBytes)
        }
        let contentType = response.value(forHTTPHeaderField: "Content-Type")
        guard let body = DuckDuckGoSearch.decodeBody(data) else {
          throw WebSearchError.bodyDecodingFailed(response.url)
        }

        let finalURL = response.url ?? url
        let output = try Self.renderOutput(
          body: body, contentType: contentType, format: options.format, baseURL: finalURL)
        let metadata = WebFetchMetadata(
          url: url,
          finalURL: finalURL,
          contentType: contentType,
          statusCode: response.statusCode,
          elapsedSeconds: elapsed,
          byteCount: data.count,
          format: options.format)
        completion(
          .success(
            WebFetchResult(
              title: "\(finalURL.absoluteString) (\(contentType ?? ""))",
              metadata: metadata,
              output: output)))
      } catch let error {
        completion(.failure(error))
      }
    }
  }

  static func renderOutput(
    body: String, contentType: String?, format: WebFetchFormat, baseURL: URL
  ) throws -> String {
    guard contentType?.lowercased().contains("html") == true else {
      return body
    }
    switch format {
    case .html:
      return body
    case .markdown:
      let document = try cleanedHTMLDocument(body, baseURL: baseURL)
      let element = document.body() ?? document
      return try HTMLMarkdownConverter.convert(element).trimmingCharacters(
        in: .whitespacesAndNewlines)
    case .text:
      let document = try cleanedHTMLDocument(body, baseURL: baseURL)
      return normalizeWhitespace(try (document.body() ?? document).text())
    }
  }

  private static func cleanedHTMLDocument(_ html: String, baseURL: URL) throws -> Document {
    let document = try SwiftSoup.parse(html, baseURL.absoluteString)
    try document.select("script,style,noscript,meta,link").remove()
    return document
  }
}

/// Converts a SwiftSoup element tree into lightweight Markdown.
public enum HTMLMarkdownConverter {
  /// Converts `element` and its descendants into Markdown.
  public static func convert(_ element: Element) throws -> String {
    let visitor = MarkdownNodeVisitor()
    try NodeTraversor(visitor).traverse(element)
    return visitor.output
  }
}

private final class MarkdownNodeVisitor: NodeVisitor {
  var output = ""
  private var orderedListCounters = [Int?]()
  private var preDepth: Int? = nil
  private var skipDepth: Int? = nil

  func head(_ node: Node, _ depth: Int) throws {
    if let skipDepth, depth > skipDepth {
      return
    }
    if let textNode = node as? TextNode {
      appendText(preDepth == nil ? normalizeWhitespace(textNode.text()) : textNode.getWholeText())
      return
    }
    guard let element = node as? Element else {
      return
    }

    let tag = element.tagNameNormal()
    switch tag {
    case "h1", "h2", "h3", "h4", "h5", "h6":
      blockBreak()
      let level = Int(String(tag.dropFirst())) ?? 1
      output += String(repeating: "#", count: max(1, min(6, level))) + " "
    case "p":
      blockBreak()
    case "br":
      output += "\n"
    case "a":
      output += "["
    case "img":
      let alt = normalizeWhitespace(try element.attr("alt"))
      let src = try element.absUrl("src")
      if !src.isEmpty {
        output += "![\(alt)](\(src))"
      }
      skipDepth = depth
    case "ul":
      blockBreak()
      orderedListCounters.append(nil)
    case "ol":
      blockBreak()
      orderedListCounters.append(0)
    case "li":
      output += "\n"
      if let last = orderedListCounters.indices.last, let current = orderedListCounters[last] {
        let next = current + 1
        orderedListCounters[last] = next
        output += "\(next). "
      } else {
        output += "- "
      }
    case "blockquote":
      blockBreak()
      output += "> "
    case "pre":
      blockBreak()
      output += "```\n"
      preDepth = depth
    case "code":
      if preDepth == nil {
        output += "`"
      }
    case "table":
      blockBreak()
      output += try tableMarkdown(element)
      blockBreak()
      skipDepth = depth
    case "tr", "th", "td":
      break
    default:
      if isBlock(tag) {
        blockBreak()
      }
    }
  }

  func tail(_ node: Node, _ depth: Int) throws {
    if let skipDepth, depth == skipDepth {
      self.skipDepth = nil
      return
    }
    if let skipDepth, depth > skipDepth {
      return
    }
    guard let element = node as? Element else {
      return
    }

    let tag = element.tagNameNormal()
    switch tag {
    case "a":
      let href = try element.absUrl("href")
      output += href.isEmpty ? "]" : "](\(href))"
    case "p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote":
      blockBreak()
    case "ul", "ol":
      if !orderedListCounters.isEmpty {
        orderedListCounters.removeLast()
      }
      blockBreak()
    case "pre":
      if !output.hasSuffix("\n") {
        output += "\n"
      }
      output += "```"
      preDepth = nil
      blockBreak()
    case "code":
      if preDepth == nil {
        output += "`"
      }
    default:
      if isBlock(tag) {
        blockBreak()
      }
    }
  }

  private func appendText(_ text: String) {
    guard !text.isEmpty else {
      return
    }
    if preDepth == nil, !output.isEmpty, !output.hasSuffix(" "), !output.hasSuffix("\n"),
      !output.hasSuffix("["),
      !text.hasPrefix(".") && !text.hasPrefix(",")
    {
      output += " "
    }
    output += text
  }

  private func blockBreak() {
    let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
      output = ""
      return
    }
    output = trimmed + "\n\n"
  }

  private func isBlock(_ tag: String) -> Bool {
    [
      "article", "main", "section", "div", "header", "footer", "aside", "figure", "figcaption",
    ].contains(tag)
  }

  private func tableMarkdown(_ element: Element) throws -> String {
    var rows = [[String]]()
    for row in try element.select("tr") {
      var cells = [String]()
      for cell in try row.select("th,td") {
        cells.append(
          normalizeWhitespace(try cell.text()).replacingOccurrences(of: "|", with: "\\|"))
      }
      if !cells.isEmpty {
        rows.append(cells)
      }
    }
    guard let first = rows.first else {
      return normalizeWhitespace(try element.text())
    }
    let columnCount = first.count
    var lines = [markdownTableRow(first, columnCount: columnCount)]
    lines.append(
      markdownTableRow(Array(repeating: "---", count: columnCount), columnCount: columnCount))
    for row in rows.dropFirst() {
      lines.append(markdownTableRow(row, columnCount: columnCount))
    }
    return lines.joined(separator: "\n")
  }

  private func markdownTableRow(_ cells: [String], columnCount: Int) -> String {
    var padded = cells
    if padded.count < columnCount {
      padded.append(contentsOf: Array(repeating: "", count: columnCount - padded.count))
    }
    return "| " + padded.prefix(columnCount).joined(separator: " | ") + " |"
  }
}
