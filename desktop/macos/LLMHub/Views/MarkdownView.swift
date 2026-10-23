import SwiftUI

/// The Markdown subset local models write (headings, lists, quotes, fenced code, bold/italic, inline code, links and
/// [n] citations), parsed the same way as the Windows and Linux apps (core Services/Markdown.cs). Tolerant of
/// half-streamed text: an unclosed code fence is code to the end.
enum Markdown {
    enum Kind { case paragraph, heading, bullet, numbered, code, quote, rule }

    struct Block: Identifiable {
        let id: Int
        let kind: Kind
        let text: String
        /// Heading level, list nesting, or the number of a numbered item.
        var level = 0
        var language = ""
    }

    static func parse(_ text: String) -> [Block] {
        var blocks: [Block] = []
        var paragraph: [String] = []
        let lines = text.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        func add(_ kind: Kind, _ text: String, level: Int = 0, language: String = "") {
            blocks.append(Block(id: blocks.count, kind: kind, text: text, level: level, language: language))
        }
        func flush() {
            if !paragraph.isEmpty { add(.paragraph, paragraph.joined(separator: " ").trimmingCharacters(in: .whitespaces)) }
            paragraph.removeAll()
        }

        var i = 0
        while i < lines.count {
            let line = lines[i]
            let trimmed = line.drop { $0 == " " || $0 == "\t" }
            defer { i += 1 }
            if trimmed.hasPrefix("```") {
                flush()
                let language = trimmed.dropFirst(3).trimmingCharacters(in: .whitespaces)
                var code: [String] = []
                i += 1
                while i < lines.count, !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    code.append(lines[i])
                    i += 1
                }
                add(.code, code.joined(separator: "\n"), language: language)
                continue
            }
            if trimmed.isEmpty {
                flush()
                continue
            }
            let t = String(trimmed)
            if let m = t.firstMatch(of: /^(#{1,6})\s+(.*)$/) {
                flush()
                add(.heading, String(m.2).trimmingCharacters(in: CharacterSet(charactersIn: "# ")), level: m.1.count)
                continue
            }
            if t.wholeMatch(of: /([-*_])(\s*\1){2,}\s*/) != nil {
                flush()
                add(.rule, "")
                continue
            }
            let indent = (line.count - trimmed.count) / 2
            if let m = t.firstMatch(of: /^[-*+]\s+(.*)$/) {
                flush()
                add(.bullet, String(m.1), level: indent)
                continue
            }
            if let m = t.firstMatch(of: /^(\d{1,3})[.)]\s+(.*)$/) {
                flush()
                add(.numbered, String(m.2), level: Int(m.1) ?? 1)
                continue
            }
            if t.hasPrefix(">") {
                flush()
                add(.quote, t.drop { $0 == ">" }.trimmingCharacters(in: .whitespaces))
                continue
            }
            paragraph.append(t)
        }
        flush()
        return blocks
    }

    // Same pattern as the C# parser (Swift regex literals have no lookbehind, so NSRegularExpression).
    private static let inlinePattern = try! NSRegularExpression(pattern:
        #"`(?<code>[^`]+)`|\*\*\*(?<bi>[^*]+)\*\*\*|\*\*(?<b>[^*]+)\*\*|__(?<u>[^_]+)__|(?<![\w*])\*(?<i>[^*\s][^*]*)\*(?!\w)|\[(?<label>[^\]]+)\]\((?<url>https?://[^)\s]+)\)|\[(?<cite>\d{1,2})\]"#)

    /// One line as attributed text: bold, italic, inline code, links, and [n] citations in the accent color.
    static func inline(_ text: String, accent: Color = Theme.accent) -> AttributedString {
        let ns = text as NSString
        var result = AttributedString()
        var last = 0
        func group(_ m: NSTextCheckingResult, _ name: String) -> String? {
            let r = m.range(withName: name)
            return r.location == NSNotFound ? nil : ns.substring(with: r)
        }
        for m in inlinePattern.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            if m.range.location > last { result += AttributedString(ns.substring(with: NSRange(location: last, length: m.range.location - last))) }
            var piece: AttributedString
            if let code = group(m, "code") {
                piece = AttributedString(code)
                piece.font = .system(size: 13.5, design: .monospaced)
                piece.backgroundColor = Theme.surfaceAlt
            } else if let bi = group(m, "bi") {
                piece = AttributedString(bi)
                piece.inlinePresentationIntent = [.stronglyEmphasized, .emphasized]
            } else if let b = group(m, "b") ?? group(m, "u") {
                piece = AttributedString(b)
                piece.inlinePresentationIntent = .stronglyEmphasized
            } else if let it = group(m, "i") {
                piece = AttributedString(it)
                piece.inlinePresentationIntent = .emphasized
            } else if let label = group(m, "label"), let url = group(m, "url") {
                piece = AttributedString(label)
                piece.link = URL(string: url)
                piece.foregroundColor = accent
            } else if let cite = group(m, "cite") {
                piece = AttributedString("[\(cite)]")
                piece.foregroundColor = accent
                piece.font = .system(size: 11, weight: .semibold)
                piece.baselineOffset = 5
            } else {
                piece = AttributedString(ns.substring(with: m.range))
            }
            result += piece
            last = m.range.location + m.range.length
        }
        if last < ns.length { result += AttributedString(ns.substring(from: last)) }
        return result
    }

    /// Plain text for reading aloud and copying, without Markdown symbols or citations.
    static func plainText(_ text: String) -> String {
        parse(text).map { block in
            if block.kind == .code { return block.text }
            return String(inline(block.text).characters).replacing(/\[\d{1,2}\]/, with: "")
        }.joined(separator: "\n")
    }
}

/// A reply rendered from <see cref="Markdown"/>: headings, lists, quotes, code blocks with a monospaced card.
struct MarkdownView: View {
    let text: String
    var fontSize: CGFloat = 15

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Markdown.parse(text)) { block in
                switch block.kind {
                case .heading:
                    Text(Markdown.inline(block.text)).font(.system(size: block.level <= 2 ? fontSize + 4 : fontSize + 1, weight: .semibold))
                case .code:
                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(block.text).font(.system(size: 13, design: .monospaced)).textSelection(.enabled).padding(.horizontal, 14).padding(.vertical, 10)
                    }
                    .background(Theme.surfaceAlt, in: RoundedRectangle(cornerRadius: 10))
                    .overlay(RoundedRectangle(cornerRadius: 10).stroke(Theme.divider))
                case .bullet:
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text("•").foregroundStyle(Theme.textMuted)
                        Text(Markdown.inline(block.text))
                    }
                    .padding(.leading, CGFloat(block.level) * 18)
                case .numbered:
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text("\(block.level).").foregroundStyle(Theme.textMuted).monospacedDigit()
                        Text(Markdown.inline(block.text))
                    }
                case .quote:
                    Text(Markdown.inline(block.text)).italic().padding(.leading, 12)
                        .overlay(alignment: .leading) { Rectangle().fill(Theme.controlBorder).frame(width: 3) }
                case .rule:
                    Divider()
                case .paragraph:
                    Text(Markdown.inline(block.text))
                }
            }
        }
        .font(.system(size: fontSize))
        .lineSpacing(4)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
