import Foundation

// MARK: - MemoryDocument
// A single persisted global-memory entry (mirrors Android's MemoryDocument).

struct MemoryDocument: Codable, Identifiable, Sendable {
    let id: String
    var fileName: String
    var content: String
    var metadata: String   // "pasted" | "uploaded" | "chat_import"
    let createdAt: Date

    init(id: String = "mem_\(UUID().uuidString)", fileName: String, content: String, metadata: String, createdAt: Date = Date()) {
        self.id = id
        self.fileName = fileName
        self.content = content
        self.metadata = metadata
        self.createdAt = createdAt
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
        save()
    }

    // MARK: - Persistence

    private func load() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        guard let rawData = try? Data(contentsOf: fileURL) else { return }

        // Try new Store format first.
        if let decoded = try? JSONDecoder().decode(Store.self, from: rawData) {
            // Memory is text-only: image/audio entries can't be embedded or injected into chats.
            documents = decoded.documents.filter { $0.metadata != "image" && $0.metadata != "audio" }
            if documents.count != decoded.documents.count {
                let mediaDir = fileURL.deletingLastPathComponent().appendingPathComponent("memory_media")
                try? FileManager.default.removeItem(at: mediaDir)
                save()
            }
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
