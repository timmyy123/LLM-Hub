import Foundation

/// The text side of the Design tab, ported from LLMHub.Core's DesignService: the system prompt, pulling the HTML out of the
/// model's reply, the page title, the element picker injected into the preview, and the HTML → Markdown export.
enum DesignDocument {
    // MARK: - Prompt

    static func systemPrompt(kind: DesignKind, system: DesignSystem) -> String {
        let instructions: String
        switch kind {
        case .slides:
            instructions = """
                Make a slide deck as ONE HTML document: each slide is a <section class="slide"> sized 1280x720 (16:9), stacked
                vertically with 24 px gaps on a neutral page background, so every slide is visible when scrolling. Title slide first,
                5 to 8 slides, one idea per slide, large type (slide titles 48 px or more), real content, no lorem ipsum.
                """
        case .motion:
            instructions = """
                Make an animated piece as ONE HTML document using CSS keyframes (and small inline JavaScript only if needed): a 1280x720
                stage that loops smoothly, with the motion described in the request. No external libraries.
                """
        case .image:
            instructions = """
                Make a single poster or social image as ONE HTML document: a fixed-size canvas (1080x1350 unless the request says
                otherwise) built from HTML and CSS only (shapes, gradients, type), centered on the page.
                """
        case .prototype:
            instructions = """
                Make a high-fidelity web prototype as ONE HTML document: semantic HTML and a <style> block, responsive from 360 px
                phones to wide desktops, real copy written for the request (no lorem ipsum), accessible contrast, hover states.
                """
        }
        return """
            You are an expert product and visual designer who writes production-quality HTML and CSS.
            \(instructions)
            \(system.brief)
            Rules:
            - Output exactly one block: <artifact type="text/html"> then a complete document starting with <!doctype html>, then </artifact>.
            - Everything inline: CSS in <style>, no external JS libraries, images as CSS shapes or gradients (no external image URLs).
            - Add data-od-id="short-name" to major sections and key elements so the user can point at them.
            - Before the artifact, write one or two sentences saying what you made or changed. Nothing after the artifact.
            """
    }

    /// The user turn: the current document (for edits), the element picked in the preview, and the request.
    static func userTurn(request: String, current: String?, selectedElement: String) -> String {
        var user = ""
        if let current {
            user += "Here is the current document. Apply the change and return the whole updated document in one <artifact> block.\n"
            user += "<current>\n\(current)\n</current>\n"
        }
        if !selectedElement.isEmpty {
            user += "The user selected this element in the preview: \(selectedElement). Focus the change there.\n"
        }
        return user + "Request: " + request
    }

    // MARK: - Reply

    /// The document inside <artifact>…</artifact>; some models skip the wrapper, so a fenced html block or a bare document works too.
    static func extractHtml(_ text: String) -> String? {
        if let body = firstGroup(artifact, in: text) { return body.trimmingCharacters(in: .whitespacesAndNewlines) }
        if let fenced = firstGroup(fence, in: text), fenced.range(of: "<html", options: .caseInsensitive) != nil {
            return fenced.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let start = text.range(of: "<!doctype", options: .caseInsensitive),
              let end = text.range(of: "</html>", options: [.caseInsensitive, .backwards]),
              start.lowerBound < end.lowerBound else { return nil }
        return String(text[start.lowerBound..<end.upperBound])
    }

    /// The model's note before the artifact ("I used Editorial Warm…").
    static func summary(_ text: String) -> String {
        let note: String
        if let i = text.range(of: "<artifact", options: .caseInsensitive) {
            note = String(text[..<i.lowerBound])
        } else {
            note = text.components(separatedBy: "```")[0]
        }
        let trimmed = note.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.count > 600 ? String(trimmed.prefix(600)) : trimmed
    }

    static func title(_ html: String) -> String? {
        guard let value = firstGroup(titleTag, in: html)?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else { return nil }
        return decodeEntities(value)
    }

    // MARK: - Preview

    /// Injected into every preview: hovering outlines an element, clicking selects it and tells the app which one
    /// (its data-od-id, tag and text) through `postMessage`, the web view's message function, so the next change can be aimed there.
    static func pickerScript(_ postMessage: String) -> String { pickerTemplate.replacingOccurrences(of: "POST(", with: postMessage + "(") }

    private static let pickerTemplate = """
        (() => {
          if (window.__llmhubPicker) return; window.__llmhubPicker = true;
          const style = document.createElement('style');
          style.textContent = '[data-llmhub-hover]{outline:2px dashed #8ab4ff!important;outline-offset:2px}[data-llmhub-picked]{outline:2px solid #8ab4ff!important;outline-offset:3px}';
          (document.head || document.documentElement).appendChild(style);
          let hovered = null;
          document.addEventListener('mouseover', e => { if (hovered) hovered.removeAttribute('data-llmhub-hover'); hovered = e.target; hovered.setAttribute('data-llmhub-hover', ''); });
          document.addEventListener('click', e => {
            e.preventDefault(); e.stopPropagation();
            document.querySelectorAll('[data-llmhub-picked]').forEach(x => x.removeAttribute('data-llmhub-picked'));
            const el = e.target.closest('[data-od-id]') || e.target;
            el.setAttribute('data-llmhub-picked', '');
            POST({ id: el.getAttribute('data-od-id') || '', tag: el.tagName.toLowerCase(), text: (el.innerText || '').trim().slice(0, 80) });
          }, true);
        })();
        """

    /// "button · cta · "Book a table"" from the picker's message (a dictionary from WebKit, or its JSON).
    static func describePicked(_ body: Any) -> String {
        var fields: [String: Any] = [:]
        if let dict = body as? [String: Any] {
            fields = dict
        } else if let json = body as? String, let data = json.data(using: .utf8),
                  let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            fields = dict
        }
        let text = ((fields["text"] as? String) ?? "").split(whereSeparator: \.isWhitespace).joined(separator: " ")
        return [fields["tag"] as? String, fields["id"] as? String, text.isEmpty ? nil : "\"\(text)\""]
            .compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
    }

    // MARK: - Markdown export

    static func htmlToMarkdown(_ html: String) -> String {
        var body = replace(styleBlocks, in: html, with: "")
        body = replace(headings, in: body) { groups in "\n" + String(repeating: "#", count: Int(groups[1]) ?? 1) + " " + groups[2] + "\n" }
        body = replace(listItems, in: body) { groups in "\n- " + groups[1] }
        body = replace(blocks, in: body, with: "\n\n")
        body = replace(tags, in: body, with: "")
        body = decodeEntities(body)
        body = replace(blankLines, in: body, with: "\n\n")
        body = replace(spaces, in: body, with: " ")
        return body.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
    }

    // MARK: - Helpers

    private static func regex(_ pattern: String, _ options: NSRegularExpression.Options = [.caseInsensitive, .dotMatchesLineSeparators]) -> NSRegularExpression {
        try! NSRegularExpression(pattern: pattern, options: options)
    }

    private static let artifact = regex(#"<artifact[^>]*>(.*?)</artifact>"#)
    private static let fence = regex(#"```(?:html)?\s*(.*?)```"#)
    private static let titleTag = regex(#"<title>(.*?)</title>"#)
    private static let styleBlocks = regex(#"<(style|script|head)[^>]*>.*?</\1>"#)
    private static let headings = regex(#"<h([1-6])[^>]*>(.*?)</h\1>"#)
    private static let listItems = regex(#"<li[^>]*>(.*?)</li>"#)
    private static let blocks = regex(#"</?(p|div|section|header|footer|main|article|br|tr|table)[^>]*>"#, [.caseInsensitive])
    private static let tags = regex(#"<[^>]+>"#, [])
    private static let blankLines = regex(#"\n\s*\n+"#, [])
    private static let spaces = regex(#"[ \t]+"#, [])

    private static func firstGroup(_ regex: NSRegularExpression, in text: String) -> String? {
        let ns = text as NSString
        guard let match = regex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)),
              match.range(at: 1).location != NSNotFound else { return nil }
        return ns.substring(with: match.range(at: 1))
    }

    private static func replace(_ regex: NSRegularExpression, in text: String, with template: String) -> String {
        regex.stringByReplacingMatches(in: text, range: NSRange(location: 0, length: (text as NSString).length),
                                       withTemplate: NSRegularExpression.escapedTemplate(for: template))
    }

    private static func replace(_ regex: NSRegularExpression, in text: String, _ transform: ([String]) -> String) -> String {
        let ns = text as NSString
        var result = ""
        var last = 0
        for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            result += ns.substring(with: NSRange(location: last, length: match.range.location - last))
            let groups = (0..<match.numberOfRanges).map { i -> String in
                let r = match.range(at: i)
                return r.location == NSNotFound ? "" : ns.substring(with: r)
            }
            result += transform(groups)
            last = match.range.location + match.range.length
        }
        return result + ns.substring(from: last)
    }

    private static let namedEntities: [String: String] = [
        "amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": "\u{00A0}", "copy": "©", "reg": "®",
        "mdash": "\u{2014}", "ndash": "\u{2013}", "hellip": "…", "middot": "·", "rsquo": "\u{2019}", "lsquo": "\u{2018}",
        "rdquo": "\u{201D}", "ldquo": "\u{201C}", "times": "×", "euro": "€", "pound": "£", "deg": "°", "bull": "•",
    ]
    private static let entity = regex(#"&(#[0-9]+|#[xX][0-9a-fA-F]+|[a-zA-Z]+);"#, [])

    /// HTML character references (&amp;, &#39;, &#x2019;, the common named ones) to text.
    static func decodeEntities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        return replace(entity, in: text) { groups in
            let name = groups[1]
            if name.hasPrefix("#x") || name.hasPrefix("#X") {
                return UInt32(name.dropFirst(2), radix: 16).flatMap(Unicode.Scalar.init).map { String(Character($0)) } ?? groups[0]
            }
            if name.hasPrefix("#") {
                return UInt32(name.dropFirst()).flatMap(Unicode.Scalar.init).map { String(Character($0)) } ?? groups[0]
            }
            return namedEntities[name.lowercased()] ?? groups[0]
        }
    }
}
