import Foundation

/// Some models (LFM2.5 Thinking, many fine-tunes) write their reasoning inline as <think>…</think>
/// instead of a separate reasoning field. Feed streamed text in; reasoning and answer come out separately,
/// even when a tag is split across chunks.
struct ThinkSplitter {
    private static let open = "<think>"
    private static let close = "</think>"
    private var pending = ""
    private var inThink = false

    mutating func push(_ chunk: String) -> (answer: String, reasoning: String) {
        var text = pending + chunk
        pending = ""
        var answer = ""
        var reasoning = ""
        while !text.isEmpty {
            let tag = inThink ? Self.close : Self.open
            if let range = text.range(of: tag) {
                if inThink { reasoning += text[..<range.lowerBound] } else { answer += text[..<range.lowerBound] }
                text = String(text[range.upperBound...])
                inThink.toggle()
                continue
            }
            // Hold back a possible partial tag at the end ("<thi") until the next chunk.
            let keep = (1..<tag.count).reversed().first { text.hasSuffix(String(tag.prefix($0))) } ?? 0
            let body = String(text.dropLast(keep))
            if inThink { reasoning += body } else { answer += body }
            pending = String(text.suffix(keep))
            break
        }
        return (answer, reasoning)
    }
}
