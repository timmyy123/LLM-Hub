import Foundation
import MagentaLiteRTBridge
import SentencePiece

/// MusicCoCa text conditioning for Magenta RealTime 2, as in the iOS app's MagentaPromptEncoder (text encoder, mapper
/// with NumPy RandomState(0) noise, RVQ), plus style blending: each style's mapped, normalized embedding is weighted,
/// averaged and normalized again before quantization, which is how Magenta RealTime mixes prompts.
public enum MusicCoCa {
    public static let resourceFiles = ["spm.model", "text_encoder.tflite", "mapper.tflite", "pretrained_vector_quantizer.tflite"]
    private static let float32Type: Int32 = 1
    private static let int32Type: Int32 = 2
    private static let sequenceLength = 128
    private static let embeddingSize = 768
    private static let rvqLevels = 12

    /// 12 MusicCoCa RVQ tokens for a weighted blend of text styles. Weights need not sum to one.
    public static func tokens(styles: [(text: String, weight: Float)], resources: URL) throws -> [Int32] {
        let active = styles.filter { $0.weight > 0 && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard !active.isEmpty else { throw CoCaError.noStyle }
        var blend = [Float](repeating: 0, count: embeddingSize)
        for style in active {
            let embedding = try embed(style.text, resources: resources)
            for i in 0..<embeddingSize { blend[i] += embedding[i] * style.weight }
        }
        let norm = sqrt(blend.reduce(Float.zero) { $0 + $1 * $1 })
        if norm > 0 { blend = blend.map { $0 / norm } }
        return try quantize(blend, url: resources.appendingPathComponent("pretrained_vector_quantizer.tflite"))
    }

    /// The mapped, unit-length music embedding of one text prompt.
    public static func embed(_ text: String, resources: URL) throws -> [Float] {
        for name in resourceFiles where !FileManager.default.fileExists(atPath: resources.appendingPathComponent(name).path) {
            throw CoCaError.missingResource(name)
        }
        let vocabulary = SentencePiece(file: resources.appendingPathComponent("spm.model").path)
        let labels = vocabulary.encode(text.lowercased()).map(\.id)
        var ids = [Int32](repeating: 0, count: sequenceLength)
        var paddings = [Float](repeating: 1, count: sequenceLength)
        ids[0] = 1
        paddings[0] = 0
        for (offset, id) in labels.prefix(sequenceLength - 1).enumerated() {
            ids[offset + 1] = id
            paddings[offset + 1] = 0
        }
        let raw = try run(resources.appendingPathComponent("text_encoder.tflite"), ints: ids, floats: paddings, second: nil,
                          count: embeddingSize, stage: "text encoder")
        var rng = NumpyRandomState(seed: 0)
        let noise = (0..<embeddingSize).map { _ in Float(rng.nextGaussian()) }
        var mapped = try run(resources.appendingPathComponent("mapper.tflite"), ints: nil, floats: raw, second: noise,
                             count: embeddingSize, stage: "mapper")
        let norm = sqrt(mapped.reduce(Float.zero) { $0 + $1 * $1 })
        if norm > 0 { mapped = mapped.map { $0 / norm } }
        return mapped
    }

    private static func quantize(_ embedding: [Float], url: URL) throws -> [Int32] {
        var output = [Int32](repeating: 0, count: rvqLevels)
        let status = url.path.withCString { path in
            embedding.withUnsafeBufferPointer { input in
                output.withUnsafeMutableBytes { out in
                    MagentaLiteRTRunModel(path, nil, 0, input.baseAddress, input.count, nil, 0, out.baseAddress, out.count, int32Type)
                }
            }
        }
        guard status == 0 else { throw CoCaError.inferenceFailed("vector quantizer (LiteRT status \(status))") }
        return output
    }

    private static func run(_ url: URL, ints: [Int32]?, floats: [Float], second: [Float]?, count: Int, stage: String) throws -> [Float] {
        var output = [Float](repeating: 0, count: count)
        let ints = ints ?? []
        let second = second ?? []
        let status = url.path.withCString { path in
            ints.withUnsafeBufferPointer { intBuffer in
                floats.withUnsafeBufferPointer { floatBuffer in
                    second.withUnsafeBufferPointer { secondBuffer in
                        output.withUnsafeMutableBytes { out in
                            MagentaLiteRTRunModel(path,
                                                  ints.isEmpty ? nil : intBuffer.baseAddress, ints.count,
                                                  floatBuffer.baseAddress, floatBuffer.count,
                                                  second.isEmpty ? nil : secondBuffer.baseAddress, second.count,
                                                  out.baseAddress, out.count, float32Type)
                        }
                    }
                }
            }
        }
        guard status == 0 else { throw CoCaError.inferenceFailed("\(stage) (LiteRT status \(status))") }
        return output
    }

    public enum CoCaError: LocalizedError {
        case noStyle
        case missingResource(String)
        case inferenceFailed(String)

        public var errorDescription: String? {
            switch self {
            case .noStyle: return "Add a style to play"
            case .missingResource(let name): return "Magenta MusicCoCa resource is missing: \(name)"
            case .inferenceFailed(let stage): return "Magenta \(stage) inference failed"
            }
        }
    }

    /// NumPy RandomState(seed).randn, which Magenta's mapper uses for its fixed noise input.
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
            var first = 0.0, second = 0.0, radius = 0.0
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
}
