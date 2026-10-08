// MARK: - RagService (actor)
// Pure-Swift in-memory RAG store: smart chunking, cosine-similarity
// search over embedded document chunks, and a lexical (Jaccard) fallback.
// Mirrors the Android InMemoryRagService design.

import Foundation
import Darwin

// MARK: - Types

struct ContextChunk: Sendable {
    let content: String
    let fileName: String
    let similarity: Float
    let chunkIndex: Int
}

private struct DocumentChunk: Sendable {
    let content: String
    let fileName: String
    let chunkIndex: Int
    var embedding: [Float]?
}

// MARK: - RagService

actor RagService {

    // MARK: - Similarity thresholds for EmbeddingGemma GGUF
    private static let semanticThreshold: Float = 0.60
    private static let semanticRelaxedThreshold: Float = 0.45
    private static let jaccardThreshold: Float = 0.030

    // MARK: - Storage
    // chatId -> [DocumentChunk]
    private var store: [String: [DocumentChunk]] = [:]

    init() {}

    // MARK: - Ingest

    func addRawDocument(chatId: String, content: String, fileName: String) {
        let chunks = createSmartChunks(content, maxChunkSize: 800, overlapSize: 100)
        var existing = store[chatId] ?? []
        for (i, chunk) in chunks.enumerated() {
            existing.append(DocumentChunk(content: chunk, fileName: fileName, chunkIndex: i, embedding: nil))
        }
        store[chatId] = existing
    }

    func addChunk(chatId: String, content: String, fileName: String, chunkIndex: Int, embedding: [Float]) {
        var existing = store[chatId] ?? []
        existing.append(DocumentChunk(content: content, fileName: fileName, chunkIndex: chunkIndex, embedding: embedding))
        store[chatId] = existing
    }

    func embedAllPending(chatId: String, service: EmbeddingService) async -> Int {
        let chunks = store[chatId] ?? []
        var count = 0
        for i in chunks.indices {
            guard chunks[i].embedding == nil else { continue }
            let text = chunks[i].content
            guard text.count >= 50 || chunks.count == 1 else { continue }
            guard let emb = try? await service.embed(text) else { continue }
            store[chatId]?[i].embedding = emb
            count += 1
        }
        return count
    }

    // MARK: - Search

    func search(
        chatId: String,
        query: String,
        queryEmbedding: [Float]?,
        maxResults: Int = 3,
        relaxedLexicalFallback: Bool = false
    ) -> [ContextChunk] {
        let chunks = store[chatId] ?? []
        guard !chunks.isEmpty else { return [] }

        let queryTokens = tokenize(query)
        var candidates: [(chunk: DocumentChunk, similarity: Float)] = []

        for chunk in chunks {
            var semantic: Float = 0
            if let qEmb = queryEmbedding, let cEmb = chunk.embedding, qEmb.count == cEmb.count {
                semantic = cosineSimilarity(qEmb, cEmb)
            }
            let jaccard = jaccardSimilarity(queryTokens, tokenize(chunk.content))

            let shouldInclude: Bool
            if queryEmbedding != nil && chunk.embedding != nil {
                shouldInclude = semantic > Self.semanticThreshold
                    || (semantic > Self.semanticRelaxedThreshold && jaccard > Self.jaccardThreshold)
            } else {
                shouldInclude = jaccard > 0.05
            }

            if shouldInclude {
                candidates.append((chunk: chunk, similarity: semantic > 0 ? semantic : jaccard))
            }
        }

        if candidates.isEmpty && relaxedLexicalFallback {
            candidates = chunks
                .filter { jaccardSimilarity(queryTokens, tokenize($0.content)) > 0.03 }
                .map { ($0, jaccardSimilarity(queryTokens, tokenize($0.content))) }
        }

        return candidates
            .sorted { $0.similarity > $1.similarity }
            .prefix(maxResults)
            .map { ContextChunk(content: $0.chunk.content, fileName: $0.chunk.fileName, similarity: $0.similarity, chunkIndex: $0.chunk.chunkIndex) }
    }

    // MARK: - Queries

    func hasDocuments(chatId: String) -> Bool {
        !(store[chatId]?.isEmpty ?? true)
    }

    func documentCount(chatId: String) -> Int {
        store[chatId]?.count ?? 0
    }

    func clear(chatId: String) {
        store.removeValue(forKey: chatId)
    }

    func clearAll() {
        store.removeAll()
    }

    func replicateChunks(fromChatId: String, toChatId: String) {
        guard let source = store[fromChatId] else { return }
        var dest = store[toChatId] ?? []
        let existingFiles = Set(dest.map { $0.fileName })
        for chunk in source where !existingFiles.contains(chunk.fileName) {
            dest.append(chunk)
        }
        store[toChatId] = dest
    }

    // MARK: - Chunking

    private func createSmartChunks(_ text: String, maxChunkSize: Int, overlapSize: Int) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        guard trimmed.count > maxChunkSize else { return [trimmed] }

        var chunks: [String] = []
        let paragraphs = trimmed.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var buffer = ""
        for para in paragraphs {
            if buffer.isEmpty {
                buffer = para
            } else if buffer.count + para.count + 2 <= maxChunkSize {
                buffer += "\n\n" + para
            } else {
                if buffer.count > maxChunkSize {
                    chunks.append(contentsOf: splitBySentence(buffer, maxSize: maxChunkSize, overlap: overlapSize))
                } else {
                    chunks.append(buffer)
                }
                buffer = para
            }
        }
        if !buffer.isEmpty {
            if buffer.count > maxChunkSize {
                chunks.append(contentsOf: splitBySentence(buffer, maxSize: maxChunkSize, overlap: overlapSize))
            } else {
                chunks.append(buffer)
            }
        }

        if overlapSize > 0 && chunks.count > 1 {
            var overlapped: [String] = [chunks[0]]
            for i in 1..<chunks.count {
                let suffix = String(chunks[i - 1].suffix(overlapSize))
                overlapped.append(suffix + " " + chunks[i])
            }
            return overlapped
        }
        return chunks
    }

    private func splitBySentence(_ text: String, maxSize: Int, overlap: Int) -> [String] {
        let sentences = text.components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var chunks: [String] = []
        var buffer = ""
        for sentence in sentences {
            if buffer.isEmpty {
                buffer = sentence
            } else if buffer.count + sentence.count + 2 <= maxSize {
                buffer += ". " + sentence
            } else {
                if !buffer.isEmpty { chunks.append(buffer) }
                buffer = sentence
            }
        }
        if !buffer.isEmpty { chunks.append(buffer) }
        return chunks.isEmpty ? [text] : chunks
    }

    // MARK: - Math

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, normA: Float = 0, normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrtf(normA) * sqrtf(normB)
        return denom > 0 ? dot / denom : 0
    }

    private func tokenize(_ text: String) -> Set<String> {
        Set(text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 2 })
    }

    private func jaccardSimilarity(_ a: Set<String>, _ b: Set<String>) -> Float {
        guard !a.isEmpty || !b.isEmpty else { return 0 }
        let intersection = Float(a.intersection(b).count)
        let union = Float(a.union(b).count)
        return union > 0 ? intersection / union : 0
    }
}

