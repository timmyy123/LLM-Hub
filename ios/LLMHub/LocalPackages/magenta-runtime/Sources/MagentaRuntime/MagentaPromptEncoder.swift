#if canImport(SentencePiece)
import Foundation
import MagentaLiteRTBridge
import SentencePiece

/// MusicCoCa text conditioning used by Magenta RealTime 2.
///
/// The LiteRT C symbols are supplied by the already-linked CLiteRTLM
/// framework. Keeping this wrapper on the C API avoids a second 80 MB runtime.
enum MagentaPromptEncoder {
    private static let float32Type: Int32 = 1
    private static let int32Type: Int32 = 2
    private static let sequenceLength = 128
    private static let embeddingSize = 768
    private static let rvqLevels = 12

    static func encode(prompt: String, resourceDirectory: URL) throws -> [Int32] {
        let vocabularyURL = resourceDirectory.appendingPathComponent("spm.model")
        let encoderURL = resourceDirectory.appendingPathComponent("text_encoder.tflite")
        let mapperURL = resourceDirectory.appendingPathComponent("mapper.tflite")
        let quantizerURL = resourceDirectory.appendingPathComponent("pretrained_vector_quantizer.tflite")
        for file in [vocabularyURL, encoderURL, mapperURL, quantizerURL] where !FileManager.default.fileExists(atPath: file.path) {
            throw PromptError.missingResource(file.lastPathComponent)
        }

        let vocabulary = SentencePiece(file: vocabularyURL.path)
        let labels = vocabulary.encode(prompt.lowercased()).map(\.id)
        NSLog("[LLMHub][MusicGen] prompt=%@ sentencePieceIds=%@", prompt, labels.map(String.init).joined(separator: ","))
        var ids = [Int32](repeating: 0, count: sequenceLength)
        var paddings = [Float](repeating: 1, count: sequenceLength)
        ids[0] = 1
        paddings[0] = 0
        for (offset, id) in labels.prefix(sequenceLength - 1).enumerated() {
            ids[offset + 1] = id
            paddings[offset + 1] = 0
        }

        let embedding = try runTextEncoder(url: encoderURL, ids: ids, paddings: paddings)
        let mappedEmbedding = try runMapper(url: mapperURL, embedding: embedding)
        let tokens = try runQuantizer(url: quantizerURL, embedding: mappedEmbedding)
        NSLog(
            "[LLMHub][MusicGen] musicCoCaRVQ=%@ activeRVQ=%@",
            tokens.map(String.init).joined(separator: ","),
            tokens.map(String.init).joined(separator: ",")
        )
        return tokens
    }

    private static func runTextEncoder(url: URL, ids: [Int32], paddings: [Float]) throws -> [Float] {
        var output = [Float](repeating: 0, count: embeddingSize)
        let status = url.path.withCString { path in
            ids.withUnsafeBufferPointer { idsBuffer in
                paddings.withUnsafeBufferPointer { paddingsBuffer in
                    output.withUnsafeMutableBytes { outputBuffer in
                        MagentaLiteRTRunModel(
                            path,
                            idsBuffer.baseAddress,
                            idsBuffer.count,
                            paddingsBuffer.baseAddress,
                            paddingsBuffer.count,
                            nil,
                            0,
                            outputBuffer.baseAddress,
                            outputBuffer.count,
                            float32Type
                        )
                    }
                }
            }
        }
        guard status == 0 else { throw PromptError.inferenceFailed("text encoder (LiteRT status \(status))") }
        return output
    }

    /// Matches Magenta's `use_mapper=True, seed=0` MusicCoCa path. The mapper
    /// projects the raw language embedding into the music embedding space before
    /// RVQ; omitting it makes prompts weak and biases output toward the checkpoint's
    /// default electronic texture.
    private static func runMapper(url: URL, embedding: [Float]) throws -> [Float] {
        var rng = NumpyRandomState(seed: 0)
        let noise = (0..<embeddingSize).map { _ in Float(rng.nextGaussian()) }
        var output = [Float](repeating: 0, count: embeddingSize)
        let status = url.path.withCString { path in
            embedding.withUnsafeBufferPointer { embeddingBuffer in
                noise.withUnsafeBufferPointer { noiseBuffer in
                    output.withUnsafeMutableBytes { outputBuffer in
                        MagentaLiteRTRunModel(
                            path,
                            nil,
                            0,
                            embeddingBuffer.baseAddress,
                            embeddingBuffer.count,
                            noiseBuffer.baseAddress,
                            noiseBuffer.count,
                            outputBuffer.baseAddress,
                            outputBuffer.count,
                            float32Type
                        )
                    }
                }
            }
        }
        guard status == 0 else { throw PromptError.inferenceFailed("mapper (LiteRT status \(status))") }
        let norm = sqrt(output.reduce(Float.zero) { $0 + $1 * $1 })
        if norm > 0 { output = output.map { $0 / norm } }
        return output
    }

    private static func runQuantizer(url: URL, embedding: [Float]) throws -> [Int32] {
        var output = [Int32](repeating: 0, count: rvqLevels)
        let status = url.path.withCString { path in
            embedding.withUnsafeBufferPointer { embeddingBuffer in
                output.withUnsafeMutableBytes { outputBuffer in
                    MagentaLiteRTRunModel(
                        path,
                        nil,
                        0,
                        embeddingBuffer.baseAddress,
                        embeddingBuffer.count,
                        nil,
                        0,
                        outputBuffer.baseAddress,
                        outputBuffer.count,
                        int32Type
                    )
                }
            }
        }
        guard status == 0 else { throw PromptError.inferenceFailed("vector quantizer (LiteRT status \(status))") }
        return output
    }

    /// NumPy RandomState(0).randn compatibility used by Magenta's mapper.
    private struct NumpyRandomState {
        private var state = [UInt32](repeating: 0, count: 624)
        private var position = 624
        private var cachedGaussian: Double?

        init(seed: UInt32) {
            state[0] = seed
            for index in 1..<624 {
                let previous = state[index - 1]
                state[index] = 1_812_433_253 &* (previous ^ (previous >> 30)) &+ UInt32(index)
            }
        }

        mutating func nextGaussian() -> Double {
            if let cachedGaussian {
                self.cachedGaussian = nil
                return cachedGaussian
            }
            var first = 0.0
            var second = 0.0
            var radius = 0.0
            repeat {
                first = 2 * nextDouble() - 1
                second = 2 * nextDouble() - 1
                radius = first * first + second * second
            } while radius >= 1 || radius == 0
            let factor = sqrt(-2 * log(radius) / radius)
            cachedGaussian = factor * first
            return factor * second
        }

        private mutating func nextDouble() -> Double {
            let first = nextUInt32() >> 5
            let second = nextUInt32() >> 6
            return (Double(first) * 67_108_864 + Double(second)) / 9_007_199_254_740_992
        }

        private mutating func nextUInt32() -> UInt32 {
            if position == 624 { generate() }
            var value = state[position]
            position += 1
            value ^= value >> 11
            value ^= (value << 7) & 0x9d2c5680
            value ^= (value << 15) & 0xefc60000
            value ^= value >> 18
            return value
        }

        private mutating func generate() {
            for index in 0..<624 {
                let value = (state[index] & 0x80000000) | (state[(index + 1) % 624] & 0x7fffffff)
                state[index] = state[(index + 397) % 624] ^ (value >> 1) ^ (value & 1 == 1 ? 0x9908b0df : 0)
            }
            position = 0
        }
    }

    enum PromptError: LocalizedError {
        case missingResource(String)
        case cannotLoad(String)
        case invalidTensor(String)
        case inferenceFailed(String)

        var errorDescription: String? {
            switch self {
            case .missingResource(let name): return "Magenta MusicCoCa resource is missing: \(name)"
            case .cannotLoad(let name): return "Could not load Magenta resource: \(name)"
            case .invalidTensor(let name): return "Magenta has an incompatible \(name) tensor"
            case .inferenceFailed(let stage): return "Magenta \(stage) inference failed"
            }
        }
    }
}
#endif
