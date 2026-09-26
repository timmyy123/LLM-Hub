import Foundation
import MagentaLiteRTBridge
import SentencePiece

/// EmbeddingGemma .tflite inference using the already-linked LiteRT runtime.
/// One instance owns one compiled graph; calls must be serialized by its owner.
public final class LiteRTEmbeddingModel {
    public enum ModelError: Error, LocalizedError {
        case missingTokenizer
        case loadFailed(Int32)
        case inferenceFailed(Int32)

        public var errorDescription: String? {
            switch self {
            case .missingTokenizer: "sentencepiece.model not found next to the LiteRT model"
            case .loadFailed(let code): "LiteRT embedding model load failed (status \(code))"
            case .inferenceFailed(let code): "LiteRT embedding inference failed (status \(code))"
            }
        }
    }

    private let handle: UnsafeMutableRawPointer
    private let tokenizer: SentencePiece
    public let sequenceLength: Int
    public let dimension: Int

    public init(modelURL: URL) throws {
        let tokenizerURL = modelURL.deletingLastPathComponent().appendingPathComponent("sentencepiece.model")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw ModelError.missingTokenizer
        }
        let tokenizer = SentencePiece(file: tokenizerURL.path)
        guard !tokenizer.encode("test").isEmpty else {
            throw ModelError.missingTokenizer
        }
        var status: Int32 = 0
        guard let handle = modelURL.path.withCString({ LiteRTEmbeddingCreate($0, &status) }) else {
            throw ModelError.loadFailed(status)
        }
        self.handle = handle
        self.tokenizer = tokenizer
        self.sequenceLength = Int(LiteRTEmbeddingSequenceLength(handle))
        self.dimension = Int(LiteRTEmbeddingDimension(handle))
    }

    deinit { LiteRTEmbeddingDestroy(handle) }

    public func embed(_ text: String, isQuery: Bool) throws -> [Float] {
        let prompt = isQuery
            ? "task: search result | query: \(text)"
            : "title: none | text: \(text)"
        // Gemma's BOS/EOS/PAD IDs are 2/1/0. Keep EOS inside the graph's
        // fixed sequence length; padding follows it.
        let pieceIDs = tokenizer.encode(prompt).map(\.id)
        var tokens = [Int32](repeating: 0, count: sequenceLength)
        tokens[0] = 2
        for (index, id) in pieceIDs.prefix(sequenceLength - 2).enumerated() {
            tokens[index + 1] = id
        }
        tokens[min(pieceIDs.count + 1, sequenceLength - 1)] = 1
        var output = [Float](repeating: 0, count: dimension)
        let status = tokens.withUnsafeBufferPointer { input in
            output.withUnsafeMutableBufferPointer { result in
                LiteRTEmbeddingRun(handle, input.baseAddress, input.count,
                                   result.baseAddress, result.count)
            }
        }
        guard status == 0, output.allSatisfy(\.isFinite) else {
            throw ModelError.inferenceFailed(status)
        }
        return output
    }
}
