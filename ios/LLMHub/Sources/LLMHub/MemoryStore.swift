import Foundation

// MARK: - MemoryDocument
// A single persisted global-memory entry (mirrors Android's MemoryDocument).

struct MemoryDocument: Codable, Identifiable, Sendable {
    let id: String
    var fileName: String
    var content: String
    var metadata: String   // "pasted" | "uploaded" | "chat_import" | "image" | "audio"
    let createdAt: Date

    var isMedia: Bool { MemoryMedia.isMediaType(metadata) }

    init(id: String = "mem_\(UUID().uuidString)", fileName: String, content: String, metadata: String, createdAt: Date = Date()) {
        self.id = id
        self.fileName = fileName
        self.content = content
        self.metadata = metadata
        self.createdAt = createdAt
    }
}

// MARK: - MemoryMedia
// Image and audio memories for multimodal embedding models (EmbeddingGemma 2). The media bytes
// live in Documents/memory_media/<docId>; MemoryDocument.content holds a text label plus the
// user's note, which is what gets injected into chat prompts.

enum MemoryMedia {
    static let typeImage = "image"
    static let typeAudio = "audio"

    /// Cross-modal (text query vs. image/audio) cosine scores run lower than text-to-text ones.
    static let similarityThreshold: Float = 0.30

    private static let imageLabel = "[Image memory"
    private static let audioLabel = "[Audio memory"
    private static let maxAudioSeconds = 120
    private static let wavHeaderBytes = 44

    static func isMediaType(_ metadata: String) -> Bool {
        metadata == typeImage || metadata == typeAudio
    }

    static func isMediaContent(_ content: String) -> Bool {
        content.hasPrefix(imageLabel) || content.hasPrefix(audioLabel)
    }

    static func buildContent(type: String, fileName: String, note: String) -> String {
        let label = type == typeImage ? imageLabel : audioLabel
        let trimmed = note.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "\(label): \(fileName)]" : "\(label): \(fileName)]\n\(trimmed)"
    }

    /// The user's note without the generated label line.
    static func note(fromContent content: String) -> String? {
        guard isMediaContent(content), let newline = content.firstIndex(of: "\n") else { return nil }
        let note = content[content.index(after: newline)...].trimmingCharacters(in: .whitespacesAndNewlines)
        return note.isEmpty ? nil : note
    }

    static var directory: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("memory_media", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func fileURL(docId: String) -> URL {
        directory.appendingPathComponent(docId)
    }

    static func deleteMedia(docId: String) {
        try? FileManager.default.removeItem(at: fileURL(docId: docId))
    }

    static func deleteAllMedia() {
        try? FileManager.default.removeItem(at: directory)
    }

    /// Keep the first `maxAudioSeconds` of a 16 kHz mono float32 WAV produced by the app.
    static func trimWav(_ wav: Data) -> Data {
        let maxBytes = wavHeaderBytes + maxAudioSeconds * 16_000 * 4
        guard wav.count > maxBytes,
              String(data: wav.subdata(in: 36..<40), encoding: .ascii) == "data" else { return wav }
        var trimmed = Data(wav.prefix(maxBytes))
        let dataSize = UInt32(maxBytes - wavHeaderBytes)
        withUnsafeBytes(of: (36 + dataSize).littleEndian) { trimmed.replaceSubrange(4..<8, with: $0) }
        withUnsafeBytes(of: dataSize.littleEndian) { trimmed.replaceSubrange(40..<44, with: $0) }
        return trimmed
    }
}

// MARK: - MemoryChunk (legacy struct — kept for migration only)

struct MemoryChunk: Codable, Identifiable, Sendable {
    let id: UUID
    let fileName: String
    let content: String
    let chunkIndex: Int
    let embedding: [Float]?
    let addedAt: Date

    init(fileName: String, content: String, chunkIndex: Int, embedding: [Float]? = nil) {
        self.id = UUID()
        self.fileName = fileName
        self.content = content
        self.chunkIndex = chunkIndex
        self.embedding = embedding
        self.addedAt = Date()
    }
}

// MARK: - MemoryStore
// Persists global memory documents to Documents/llmhub_memory.json.

@MainActor
final class MemoryStore: ObservableObject {

    static let shared = MemoryStore()

    @Published private(set) var documents: [MemoryDocument] = []

    private struct Store: Codable {
        var documents: [MemoryDocument]
    }

    private var fileURL: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return dir.appendingPathComponent("llmhub_memory.json")
    }

    private init() {
        load()
    }

    // MARK: - Document Mutations

    func appendDocument(_ doc: MemoryDocument) {
        documents.append(doc)
        save()
    }

    func removeDocument(id: String) {
        documents.removeAll { $0.id == id }
        MemoryMedia.deleteMedia(docId: id)
        save()
    }

    func updateDocument(_ doc: MemoryDocument) {
        if let idx = documents.firstIndex(where: { $0.id == doc.id }) {
            documents[idx] = doc
            save()
        }
    }

    func clearAllDocuments() {
        documents.removeAll()
        MemoryMedia.deleteAllMedia()
        save()
    }

    // MARK: - Persistence

    private func load() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        guard let rawData = try? Data(contentsOf: fileURL) else { return }

        // Try new Store format first.
        if let decoded = try? JSONDecoder().decode(Store.self, from: rawData) {
            documents = decoded.documents
            return
        }

        // Migrate from legacy [MemoryChunk] format.
        if let oldChunks = try? JSONDecoder().decode([MemoryChunk].self, from: rawData) {
            let fileGroups = Dictionary(grouping: oldChunks, by: { $0.fileName })
            documents = fileGroups.map { fileName, chunks in
                let body = chunks.sorted { $0.chunkIndex < $1.chunkIndex }
                    .map { $0.content }.joined(separator: "\n\n")
                return MemoryDocument(fileName: fileName, content: body, metadata: "uploaded")
            }.sorted { $0.createdAt < $1.createdAt }
            save()
        }
    }

    func save() {
        let store = Store(documents: documents)
        guard let data = try? JSONEncoder().encode(store) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
