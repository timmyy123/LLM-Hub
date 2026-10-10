import SwiftUI

public struct MarkdownTextView: View {
    public let text: String

    public init(text: String) {
        self.text = text
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(parseBlocks(text: text), id: \.id) { block in
                switch block.type {
                case .code(let code, let lang):
                    CodeBlockView(code: code, language: lang)
                case .text(let content):
                    Text(LocalizedStringKey(content))
                        .font(.system(size: 14))
                        .foregroundColor(.white.opacity(0.92))
                        .textSelection(.enabled)
                }
            }
        }
    }

    private struct ParsedBlock: Identifiable {
        let id = UUID()
        enum BlockType {
            case text(String)
            case code(String, String?)
        }
        let type: BlockType
    }

    private func parseBlocks(text: String) -> [ParsedBlock] {
        var blocks: [ParsedBlock] = []
        let pattern = "```([a-zA-Z0-9_-]*)\\n?([\\s\\S]*?)```"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return [ParsedBlock(type: .text(text))]
        }

        let nsString = text as NSString
        var lastIndex = 0
        let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length))

        for match in matches {
            if match.range.location > lastIndex {
                let textPart = nsString.substring(with: NSRange(location: lastIndex, length: match.range.location - lastIndex)).trimmingCharacters(in: .newlines)
                if !textPart.isEmpty {
                    blocks.append(ParsedBlock(type: .text(textPart)))
                }
            }

            let lang = match.numberOfRanges > 1 && match.range(at: 1).location != NSNotFound
                ? nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                : nil
            let code = match.numberOfRanges > 2 && match.range(at: 2).location != NSNotFound
                ? nsString.substring(with: match.range(at: 2))
                : ""

            blocks.append(ParsedBlock(type: .code(code, lang?.isEmpty == true ? nil : lang)))
            lastIndex = match.range.location + match.range.length
        }

        if lastIndex < nsString.length {
            let remaining = nsString.substring(from: lastIndex).trimmingCharacters(in: .newlines)
            if !remaining.isEmpty {
                blocks.append(ParsedBlock(type: .text(remaining)))
            }
        }

        return blocks.isEmpty ? [ParsedBlock(type: .text(text))] : blocks
    }
}
