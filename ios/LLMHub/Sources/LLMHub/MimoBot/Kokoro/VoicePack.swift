import Foundation

/// Loads a Kokoro voice file (onnx-community export format).
///
/// `voices/<id>.bin` is a flat little-endian float32 tensor of shape
/// (N, 1, 256), N usually 511. For a token sequence of length L the inference
/// uses `voice[L]` (clamped to the last row) as the 256-d style embedding.
struct VoicePack {
    static let defaultStyleDim = 256

    let rows: Int
    let styleDim: Int
    private let data: [Float]

    static func load(from url: URL) throws -> VoicePack {
        let blob = try Data(contentsOf: url)
        let floats = blob.count / 4
        precondition(floats > 0 && floats % defaultStyleDim == 0,
                     "voice file size \(blob.count) not a multiple of \(defaultStyleDim) floats")
        var arr = [Float](repeating: 0, count: floats)
        _ = arr.withUnsafeMutableBytes { dst in
            blob.copyBytes(to: dst)
        }
        return VoicePack(rows: floats / defaultStyleDim, styleDim: defaultStyleDim, data: arr)
    }

    func style(forTokens numTokens: Int) -> [Float] {
        let row = max(0, min(numTokens, rows - 1))
        let start = row * styleDim
        return Array(data[start..<(start + styleDim)])
    }
}
