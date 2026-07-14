import Foundation
import RunAnywhere

// MARK: - RagServiceManager
// Singleton that manages the EmbeddingService lifecycle and RagService.
// Mirrors Android's RagServiceManager design.

@MainActor
final class RagServiceManager: ObservableObject {

    static let shared = RagServiceManager()

    // MARK: - Constants

    private let globalMemoryChatId = "__global_memory__"

    // MARK: - Published State

    @Published private(set) var isReady: Bool = false         // C embedding API loaded successfully
    @Published private(set) var statusMessage: String = ""
    @Published private(set) var isReembedding: Bool = false

    /// RAG is configured when an embedding model is selected (no separate toggle).
    var isConfigured: Bool {
        AppSettings.shared.selectedEmbeddingModelId != nil
    }

    /// Human-readable reason why memory operations are blocked, or nil when ready.
    var embeddingNotReadyReason: String? {
        guard !isReady else { return nil }
        if !isConfigured {
            return "No embedding model selected. Go to Settings → Embedding Model to choose one."
        }
        return statusMessage.isEmpty ? "Embedding model not loaded." : statusMessage
    }

    // MARK: - Private

    private let embeddingService = EmbeddingService()
    private let ragService = RagService()
    private let memoryStore = MemoryStore.shared
    private var initializedModelName: String? = nil
    private var populatedChatIds: Set<UUID> = []

    private init() {}

    // MARK: - Initialization

    /// Call this when the selected embedding model changes (from SettingsScreen or Settings).
    func initialize(modelId: String?) async {
        guard let modelId, !modelId.isEmpty else {
            print("ℹ️ [RAG] initialize — no modelId, cleaning up")
            await embeddingService.cleanup()
            isReady = false
            initializedModelName = nil
            statusMessage = AppSettings.shared.localized("embedding_disabled")
            return
        }

        // Skip if already loaded with the same model.
        if initializedModelName == modelId, await embeddingService.isInitialized {
            print("ℹ️ [RAG] initialize — model \(modelId) already loaded, skipping")
            return
        }

        print("ℹ️ [RAG] initialize — loading model \(modelId)")
        statusMessage = "Loading embedding model…"
        isReady = false

        // Locate the downloaded ONNX file.
        guard let model = ModelData.allModels().first(where: { $0.id == modelId }) else {
            print("❌ [RAG] initialize — model \(modelId) not found in catalog")
            statusMessage = "Embedding model not found in catalog."
            return
        }

        guard let modelDir = try? SimplifiedFileManager.shared.getModelFolderURL(modelId: modelId, framework: model.inferenceFramework) else {
            print("❌ [RAG] initialize — model dir not found for \(modelId)")
            statusMessage = "Embedding model not downloaded."
            return
        }

        let onnxURL: URL
        if let found = (try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil))?.first(where: { $0.pathExtension.lowercased() == "onnx" }) {
            onnxURL = found
        } else {
            print("❌ [RAG] initialize — no .onnx file in \(modelDir.path)")
            statusMessage = "ONNX file not found for embedding model."
            return
        }

        do {
            try ensureTokenizerVocab(in: modelDir)
        } catch {
            print("❌ [RAG] initialize — tokenizer prep failed: \(error.localizedDescription)")
            statusMessage = "Tokenizer files missing for embedding model."
            return
        }

        do {
            print("ℹ️ [RAG] initialize — calling embeddingService.initialize at \(onnxURL.lastPathComponent)")
            try await embeddingService.initialize(modelID: modelId, modelPath: onnxURL.path, modelName: model.name)
            initializedModelName = modelId
            isReady = true
            statusMessage = AppSettings.shared.localized("embedding_enabled")
            print("✅ [RAG] initialize — SUCCESS, isReady=true")

            // Restore global memory chunks into RAG service.
            await restoreGlobalMemory()
        } catch {
            isReady = false
            statusMessage = "Failed to load embedding model: \(error.localizedDescription)"
            print("❌ [RAG] initialize — FAILED: \(error.localizedDescription)")
        }
    }

    // MARK: - Per-chat documents

    /// Add a text document to the per-chat RAG pool, chunk + embed it.
    func addDocument(chatId: UUID, text: String, fileName: String) async -> Bool {
        guard isConfigured else {
            print("⚠️ [RAG] addDocument — not configured, skipping")
            return false
        }
        let idStr = chatId.uuidString
        print("ℹ️ [RAG] addDocument — chatId=\(idStr.prefix(8)) fileName=\(fileName) textLen=\(text.count)")
        await ragService.addRawDocument(chatId: idStr, content: text, fileName: fileName)
        if isReady {
            await ragService.embedAllPending(chatId: idStr, service: embeddingService)
            print("✅ [RAG] addDocument — embedded pending chunks for chatId=\(idStr.prefix(8))")
        } else {
            print("⚠️ [RAG] addDocument — embedding skipped (isReady=false)")
        }
        return true
    }

    /// Inject any global memory chunks into a specific chat (called on chat load).
    func populateChat(chatId: UUID) async {
        guard isReady, !populatedChatIds.contains(chatId) else { return }
        populatedChatIds.insert(chatId)
        await ragService.replicateChunks(fromChatId: globalMemoryChatId, toChatId: chatId.uuidString)
    }

    func clearChat(chatId: UUID) async {
        await ragService.clear(chatId: chatId.uuidString)
        populatedChatIds.remove(chatId)
    }

    func hasDocuments(chatId: UUID) async -> Bool {
        await ragService.hasDocuments(chatId: chatId.uuidString)
    }

    func documentCount(chatId: UUID) async -> Int {
        await ragService.documentCount(chatId: chatId.uuidString)
    }

    // MARK: - Search

    func searchRelevantContext(
        chatId: UUID,
        query: String,
        maxResults: Int = 3
    ) async -> [ContextChunk] {
        let idStr = chatId.uuidString
        let queryEmbedding = isReady ? (try? await embeddingService.embed(query)) : nil
        let results = await ragService.search(chatId: idStr, query: query, queryEmbedding: queryEmbedding, maxResults: maxResults)
        print("🔍 [RAG] searchRelevantContext — chatId=\(idStr.prefix(8)) query=\"\(query.prefix(40))\" hasEmbedding=\(queryEmbedding != nil) results=\(results.count)")
        return results
    }

    func searchGlobalContext(query: String, maxResults: Int = 5, relaxed: Bool = false) async -> [ContextChunk] {
        let queryEmbedding = isReady ? (try? await embeddingService.embed(query)) : nil
        let results = await ragService.search(chatId: globalMemoryChatId, query: query, queryEmbedding: queryEmbedding, maxResults: maxResults, relaxedLexicalFallback: relaxed)
        print("🔍 [Memory] searchGlobalContext — query=\"\(query.prefix(40))\" hasEmbedding=\(queryEmbedding != nil) relaxed=\(relaxed) results=\(results.count)")
        for r in results {
            print("🔍 [Memory]   → sim=\(String(format: "%.3f", r.similarity)) file=\(r.fileName) content=\"\(r.content.prefix(60))...\"")
        }
        return results
    }

    // MARK: - Global Memory

    func addGlobalMemory(text: String, fileName: String, metadata: String = "pasted") async -> Bool {
        guard isReady else {
            let reason = isConfigured ? "embedding model selected but not yet loaded" : "no embedding model selected"
            print("⚠️ [Memory] addGlobalMemory — BLOCKED (\(reason)). Memory requires an active embedding model.")
            return false
        }
        print("ℹ️ [Memory] addGlobalMemory — fileName=\(fileName) metadata=\(metadata) textLen=\(text.count)")
        await ragService.addRawDocument(chatId: globalMemoryChatId, content: text, fileName: fileName)
        await ragService.embedAllPending(chatId: globalMemoryChatId, service: embeddingService)
        print("✅ [Memory] addGlobalMemory — embedded chunks for \(fileName)")
        let doc = MemoryDocument(fileName: fileName, content: text, metadata: metadata)
        memoryStore.appendDocument(doc)
        let totalDocs = memoryStore.documents.count
        print("✅ [Memory] addGlobalMemory — persisted to MemoryStore, total documents=\(totalDocs)")
        return true
    }

    func clearGlobalMemory() async {
        print("ℹ️ [Memory] clearGlobalMemory — clearing all documents")
        await ragService.clear(chatId: globalMemoryChatId)
        populatedChatIds.removeAll()
        memoryStore.clearAllDocuments()
    }

    func removeGlobalDocument(docId: String) async {
        print("ℹ️ [Memory] removeGlobalDocument — docId=\(docId)")
        memoryStore.removeDocument(id: docId)
        // Rebuild ragService global memory from remaining documents.
        await ragService.clear(chatId: globalMemoryChatId)
        populatedChatIds.removeAll()
        await restoreGlobalMemory()
        print("ℹ️ [Memory] removeGlobalDocument — rebuilt with \(memoryStore.documents.count) remaining docs")
    }

    func updateGlobalMemoryDocument(docId: String, newContent: String) async {
        guard var doc = memoryStore.documents.first(where: { $0.id == docId }) else {
            print("⚠️ [Memory] updateGlobalMemoryDocument — docId=\(docId) not found")
            return
        }
        print("ℹ️ [Memory] updateGlobalMemoryDocument — docId=\(docId) newLen=\(newContent.count)")
        doc.content = newContent
        memoryStore.updateDocument(doc)
        // Rebuild ragService global memory with updated content.
        await ragService.clear(chatId: globalMemoryChatId)
        populatedChatIds.removeAll()
        await restoreGlobalMemory()
        if isReady {
            await ragService.embedAllPending(chatId: globalMemoryChatId, service: embeddingService)
        }
        print("✅ [Memory] updateGlobalMemoryDocument — rebuilt and re-embedded")
    }

    func globalDocumentCount() async -> Int {
        await ragService.documentCount(chatId: globalMemoryChatId)
    }

    // MARK: - Re-embedding (when model changes)

    func reembedGlobalMemory(newModelId: String) async {
        print("[Memory] reembedGlobalMemory — starting with newModelId=\(newModelId), \(memoryStore.documents.count) documents to re-embed")
        isReembedding = true
        defer { isReembedding = false }

        await initialize(modelId: newModelId)
        guard isReady else {
            print("❌ [Memory] reembedGlobalMemory — model init failed, aborting")
            return
        }

        await ragService.clear(chatId: globalMemoryChatId)
        populatedChatIds.removeAll()

        for doc in memoryStore.documents {
            print("🔍 [Memory] reembedGlobalMemory — chunking doc: \(doc.fileName) (len=\(doc.content.count))")
            await ragService.addRawDocument(chatId: globalMemoryChatId, content: doc.content, fileName: doc.fileName)
        }
        await ragService.embedAllPending(chatId: globalMemoryChatId, service: embeddingService)
        print("✅ [Memory] reembedGlobalMemory — DONE, all \(memoryStore.documents.count) documents re-embedded")
    }

    // MARK: - Persistence Helpers

    private func restoreGlobalMemory() async {
        let docs = memoryStore.documents
        print("ℹ️ [Memory] restoreGlobalMemory — restoring \(docs.count) documents into RAG")
        for doc in docs {
            await ragService.addRawDocument(chatId: globalMemoryChatId, content: doc.content, fileName: doc.fileName)
        }
        if isReady {
            await ragService.embedAllPending(chatId: globalMemoryChatId, service: embeddingService)
            let chunkCount = await ragService.documentCount(chatId: globalMemoryChatId)
            print("✅ [Memory] restoreGlobalMemory — embedded, total chunks=\(chunkCount)")
        } else {
            print("⚠️ [Memory] restoreGlobalMemory — chunked but NOT embedded (isReady=false)")
        }
    }

    private func ensureTokenizerVocab(in modelDir: URL) throws {
        let fm = FileManager.default
        let vocabURL = modelDir.appendingPathComponent("vocab.txt")
        if fm.fileExists(atPath: vocabURL.path) {
            return
        }

        let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json")
        guard fm.fileExists(atPath: tokenizerURL.path) else {
            throw NSError(domain: "RAG", code: 1, userInfo: [NSLocalizedDescriptionKey: "tokenizer.json not found"])
        }

        let data = try Data(contentsOf: tokenizerURL)
        guard
            let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let model = root["model"] as? [String: Any],
            let vocabAny = model["vocab"] as? [String: Any]
        else {
            throw NSError(domain: "RAG", code: 2, userInfo: [NSLocalizedDescriptionKey: "invalid tokenizer.json format"])
        }

        var pairs: [(String, Int)] = []
        pairs.reserveCapacity(vocabAny.count)
        for (token, rawId) in vocabAny {
            if let id = rawId as? Int {
                pairs.append((token, id))
            } else if let num = rawId as? NSNumber {
                pairs.append((token, num.intValue))
            }
        }

        guard !pairs.isEmpty else {
            throw NSError(domain: "RAG", code: 3, userInfo: [NSLocalizedDescriptionKey: "tokenizer vocab is empty"])
        }

        pairs.sort { $0.1 < $1.1 }
        let maxId = pairs.last?.1 ?? 0
        var vocabLines = Array(repeating: "[UNK]", count: max(0, maxId + 1))
        for (token, id) in pairs where id >= 0 && id < vocabLines.count {
            vocabLines[id] = token
        }

        let text = vocabLines.joined(separator: "\n")
        try text.write(to: vocabURL, atomically: true, encoding: .utf8)
        print("ℹ️ [RAG] initialize — generated vocab.txt from tokenizer.json (\(vocabLines.count) entries)")
    }
}
