import Foundation
import PDFKit
import UniformTypeIdentifiers

// MARK: - DocumentTextExtractor
// Extracts plain text from user-provided documents for RAG ingestion.
// Mirrors Android's FileUtils text extraction (PDF, TXT, MD, JSON, XML, CSV).

enum DocumentExtractionError: Error, LocalizedError {
    case unsupportedType(String)
    case pdfHasNoText
    case readFailed(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedType(let ext): return "Unsupported document type: .\(ext)"
        case .pdfHasNoText: return "PDF has no extractable text (may be image-only)."
        case .readFailed(let msg): return "Failed to read document: \(msg)"
        }
    }
}

struct DocumentTextExtractor {

    /// Supported UTTypes for the document file-picker.
    static var supportedTypes: [UTType] {
        var types: [UTType] = [
            .plainText,
            .pdf,
        ]
        // Additional common types with graceful fallback to UTF-8 read.
        if let md = UTType("net.daringfireball.markdown") { types.append(md) }
        if let csv = UTType(filenameExtension: "csv") { types.append(csv) }
        types.append(.json)
        types.append(.xml)
        return types
    }

    /// Supported file extensions for display in the attachment UI.
    static var supportedExtensions: [String] {
        ["txt", "md", "pdf", "json", "xml", "csv"]
    }

    /// Extract plain text from the given security-scoped URL.
    static func extract(from url: URL) throws -> String {
        let ext = url.pathExtension.lowercased()
        let hasAccess = url.startAccessingSecurityScopedResource()
        defer { if hasAccess { url.stopAccessingSecurityScopedResource() } }

        switch ext {
        case "pdf":
            return try extractPDF(from: url)
        case "json":
            return try extractJSON(from: url)
        default:
            // TXT, MD, XML, CSV — try UTF-8 then Latin-1.
            do {
                return try String(contentsOf: url, encoding: .utf8)
            } catch {
                if let s = try? String(contentsOf: url, encoding: .isoLatin1) { return s }
                throw DocumentExtractionError.readFailed(error.localizedDescription)
            }
        }
    }

    // MARK: - PDF

    private static func extractPDF(from url: URL) throws -> String {
        guard let document = PDFDocument(url: url) else {
            throw DocumentExtractionError.pdfHasNoText
        }
        var pages: [String] = []
        for i in 0..<document.pageCount {
            if let page = document.page(at: i), let text = page.string, !text.isEmpty {
                pages.append(text)
            }
        }
        let result = pages.joined(separator: "\n")
        guard !result.isEmpty else { throw DocumentExtractionError.pdfHasNoText }
        return result
    }

    // MARK: - JSON

    private static func extractJSON(from url: URL) throws -> String {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw DocumentExtractionError.readFailed(error.localizedDescription)
        }
        guard let parsed = try? JSONSerialization.jsonObject(with: data) else {
            return (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        }
        var strings: [String] = []
        extractStrings(from: parsed, into: &strings)
        return strings.joined(separator: "\n")
    }

    private static func extractStrings(from value: Any, into result: inout [String]) {
        if let string = value as? String {
            result.append(string)
        } else if let dict = value as? [String: Any] {
            for (_, v) in dict { extractStrings(from: v, into: &result) }
        } else if let array = value as? [Any] {
            for element in array { extractStrings(from: element, into: &result) }
        }
    }
}
