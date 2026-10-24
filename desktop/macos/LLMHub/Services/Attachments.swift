import AppKit
import PDFKit

/// A file attached to a chat message. Images go to the model as pictures; everything else as extracted text.
/// Stored as JSON in the messages table with the same keys as the Windows and Linux apps.
struct Attachment: Codable, Hashable, Identifiable {
    var name: String
    var path: String
    var kind: String
    var mediaType: String
    var detail: String

    var id: String { path }
    var isImage: Bool { kind == "image" }
    var url: URL { URL(fileURLWithPath: path) }

    /// Short label for the file chip: "PDF", "PNG", "TXT".
    var badge: String {
        let ext = (name as NSString).pathExtension.uppercased()
        return (1...4).contains(ext.count) ? ext : "FILE"
    }

    private enum CodingKeys: String, CodingKey {
        case name = "Name", path = "Path", kind = "Kind", mediaType = "MediaType", detail = "Detail"
    }
}

enum Attachments {
    private static let imageTypes: [String: String] = [
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp",
        "heic": "image/heic", "tiff": "image/tiff",
    ]

    /// Text-like files read as UTF-8. Anything not listed (and not an image or PDF) is refused.
    private static let textTypes: Set<String> = [
        "txt", "md", "markdown", "csv", "tsv", "json", "xml", "yaml", "yml", "html", "htm", "css", "js", "ts",
        "tsx", "jsx", "py", "cs", "swift", "kt", "java", "c", "h", "cpp", "hpp", "go", "rs", "rb", "php",
        "sh", "ps1", "sql", "toml", "ini", "log", "tex", "rtf",
    ]

    /// Most text a single document may add to the prompt, so one big file cannot crowd out the conversation.
    static let maxDocumentChars = 24_000

    static var extensions: [String] { Array(imageTypes.keys) + Array(textTypes) + ["pdf"] }

    static func isSupported(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return imageTypes[ext] != nil || textTypes.contains(ext) || ext == "pdf"
    }

    /// Copies the file into the app's data folder (so chats keep working if the original moves) and describes it.
    /// HEIC, TIFF, GIF and BMP pictures are converted to PNG, which every vision model accepts.
    static func importFile(_ source: URL) throws -> Attachment {
        guard isSupported(source) else {
            throw NSError(domain: "LLMHub", code: 30, userInfo: [NSLocalizedDescriptionKey: L.f("AttachUnsupported", source.lastPathComponent)])
        }
        let folder = AppPaths.attachments.appendingPathComponent(String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(12)), isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let name = source.lastPathComponent
        let target = folder.appendingPathComponent(name)
        try FileManager.default.copyItem(at: source, to: target)
        let ext = source.pathExtension.lowercased()
        if let mime = imageTypes[ext] {
            if ["png", "jpg", "jpeg", "webp"].contains(ext) {
                return Attachment(name: name, path: target.path, kind: "image", mediaType: mime, detail: L.s("AttachImage"))
            }
            let png = target.deletingPathExtension().appendingPathExtension("png")
            guard let tiff = NSImage(contentsOf: target)?.tiffRepresentation, let bitmap = NSBitmapImageRep(data: tiff),
                  let data = bitmap.representation(using: .png, properties: [:]) else {
                throw NSError(domain: "LLMHub", code: 31, userInfo: [NSLocalizedDescriptionKey: L.f("AttachUnsupported", name)])
            }
            try data.write(to: png)
            return Attachment(name: name, path: png.path, kind: "image", mediaType: "image/png", detail: L.s("AttachImage"))
        }
        if ext == "pdf" {
            guard let pdf = PDFDocument(url: target) else {
                throw NSError(domain: "LLMHub", code: 32, userInfo: [NSLocalizedDescriptionKey: L.f("AttachUnsupported", name)])
            }
            return Attachment(name: name, path: target.path, kind: "document", mediaType: "application/pdf", detail: L.f("AttachPages", "\(pdf.pageCount)"))
        }
        let size = ((try? FileManager.default.attributesOfItem(atPath: target.path)[.size]) as? NSNumber)?.int64Value ?? 0
        return Attachment(name: name, path: target.path, kind: "document", mediaType: "text/plain", detail: Fmt.bytes(size))
    }

    /// The document's text for the prompt, cut to `maxDocumentChars`.
    static func readText(_ a: Attachment) -> String {
        var text: String
        if a.mediaType == "application/pdf" {
            text = ""
            if let pdf = PDFDocument(url: a.url) {
                for i in 0..<pdf.pageCount {
                    text += "[Page \(i + 1)]\n" + (pdf.page(at: i)?.string ?? "") + "\n"
                    if text.count > maxDocumentChars { break }
                }
            }
        } else if a.path.lowercased().hasSuffix(".rtf"), let rtf = try? NSAttributedString(url: a.url, options: [:], documentAttributes: nil) {
            text = rtf.string
        } else {
            text = (try? String(contentsOf: a.url, encoding: .utf8)) ?? (try? String(contentsOf: a.url, encoding: .isoLatin1)) ?? "[could not read this file]"
        }
        return text.count <= maxDocumentChars ? text : String(text.prefix(maxDocumentChars)) + "\n[…truncated]"
    }

    /// The user's message with attached documents folded in as `<file name="…">` blocks.
    static func withDocuments(_ text: String, _ attachments: [Attachment]) -> String {
        let docs = attachments.filter { !$0.isImage }
        guard !docs.isEmpty else { return text }
        var result = ""
        for doc in docs {
            result += "<file name=\"\(doc.name)\">\n\(readText(doc))\n</file>\n"
        }
        return result + text
    }
}
