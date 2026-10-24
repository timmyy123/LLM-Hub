import Foundation
import MLX

/// SAME latent decoders, ported from Stability's models/defs/same_s_decoder.py (for sa3-sm-*: 6 blocks of 768, chunked
/// differential attention, conv mapping) and same_l_decoder.py (for sa3-medium: 12 blocks of 1536, differential sliding
/// window attention, sin(pi x) gates from block 5). FP32. Latents [B, 256, T] to audio patches [B, 512, T * 16].
final class SameDecoder {
    enum Kind {
        case small, large

        var dim: Int { self == .small ? 768 : 1536 }
        var headCount: Int { self == .small ? 12 : 24 }
        var blocks: Int { self == .small ? 6 : 12 }
        /// sa3_mlx.py's decode_chunked settings (chunk, overlap) per decoder.
        var chunking: (chunk: Int, overlap: Int) { self == .small ? (8, 2) : (128, 8) }
    }

    private static let stride = 16
    private static let subChunk = 17
    private static let scale: Float = pow(64, -0.5)

    let kind: Kind
    private var w: Weights
    /// SAME-L's sliding window: a query group of 17 sees three consecutive key groups (51).
    private let windowMask: MLXArray

    init(url: URL, kind: Kind) throws {
        self.kind = kind
        var arrays = try Npz.load(url)
        if kind == .large, let mapping = arrays["mapping.weight"], mapping.ndim == 3, mapping.dim(-1) == 1 {
            arrays["mapping.weight"] = mapping.reshaped(mapping.dim(0), mapping.dim(1))
        }
        if kind == .small, let mapping = arrays["mapping.weight"], mapping.ndim == 3, mapping.dim(1) == kind.dim, mapping.dim(2) == 3 {
            arrays["mapping.weight"] = mapping.transposed(0, 2, 1)
        }
        w = Weights(arrays: arrays)
        w.cast(.float32)
        let q = MLXArray(0 ..< Self.subChunk)[0..., .newAxis]
        let kv = MLXArray(0 ..< 3 * Self.subChunk)[.newAxis, 0...]
        windowMask = which((kv .>= q) .&& (kv .<= q + 2 * Self.subChunk), MLXArray(Float(0)), MLXArray(Float(-1e9)))
        eval(windowMask)
    }

    // MARK: Attention

    /// Differential attention: SDPA(q, k, v) - SDPA(q_diff, k_diff, v), DyT q/k norms and partial RoPE.
    private func attention(_ x: MLXArray, _ p: String) -> MLXArray {
        let n = kind.headCount
        let parts = split(w.linear(x, p + "to_qkv"), parts: 5, axis: -1)
        func prep(_ t: MLXArray, _ norm: String) -> MLXArray { rope32(w.dyT(heads(t, n), p + norm)) }
        let q1 = prep(parts[0], "q_norm"), k1 = prep(parts[1], "k_norm"), v = heads(parts[2], n)
        let q2 = prep(parts[3], "q_norm"), k2 = prep(parts[4], "k_norm")
        let out: MLXArray
        switch kind {
        case .small:
            out = MLXFast.scaledDotProductAttention(queries: q1, keys: k1, values: v, scale: Self.scale, mask: nil)
                - MLXFast.scaledDotProductAttention(queries: q2, keys: k2, values: v, scale: Self.scale, mask: nil)
        case .large:
            out = x.dim(1) <= Self.subChunk ? differential(q1, k1, v, q2, k2, mask: nil) : slidingWindow(q1, k1, v, q2, k2)
        }
        return w.linear(merge(out), p + "to_out")
    }

    /// SAME-L's batched differential SDPA (both maps in one call).
    private func differential(_ q1: MLXArray, _ k1: MLXArray, _ v: MLXArray, _ q2: MLXArray, _ k2: MLXArray, mask: MLXArray?) -> MLXArray {
        let out = MLXFast.scaledDotProductAttention(queries: concatenated([q1, q2], axis: 1), keys: concatenated([k1, k2], axis: 1),
                                                    values: concatenated([v, v], axis: 1), scale: Self.scale, mask: mask)
        let halves = split(out, parts: 2, axis: 1)
        return halves[0] - halves[1]
    }

    /// Grouped sliding-window attention: each group of 17 queries attends to its own key group and both neighbours,
    /// as `_swa` does with a strided view; here the three neighbouring groups are concatenated instead.
    private func slidingWindow(_ q1: MLXArray, _ k1: MLXArray, _ v: MLXArray, _ q2: MLXArray, _ k2: MLXArray) -> MLXArray {
        let (b, h, t, d) = (q1.dim(0), q1.dim(1), q1.dim(2), q1.dim(3))
        let block = Self.subChunk, groups = t / block, window = 3 * block
        func windows(_ x: MLXArray) -> MLXArray {
            let padded = MLX.padded(x, widths: [0, 0, .init((block, block)), 0])
            let grouped = padded.reshaped(b, h, groups + 2, block, d)
            let stacked = concatenated([grouped[0..., 0..., 0 ..< groups], grouped[0..., 0..., 1 ..< groups + 1], grouped[0..., 0..., 2 ..< groups + 2]], axis: 3)
            return stacked.transposed(0, 2, 1, 3, 4).reshaped(b * groups, h, window, d)
        }
        func queries(_ x: MLXArray) -> MLXArray {
            x.reshaped(b, h, groups, block, d).transposed(0, 2, 1, 3, 4).reshaped(b * groups, h, block, d)
        }
        let g = MLXArray(0 ..< groups)[0..., .newAxis]
        let wIndex = MLXArray(0 ..< window)[.newAxis, 0...]
        let position = g * block + wIndex
        let boundary = which((position .>= block) .&& (position .< t + block), MLXArray(Float(0), dtype: q1.dtype), MLXArray(Float(-1e9), dtype: q1.dtype))[0..., .newAxis, 0...]
        let combined = broadcast((windowMask.asType(q1.dtype) + boundary)[.newAxis], to: [b, groups, block, window]).reshaped(b * groups, 1, block, window)
        let out = differential(queries(q1), windows(k1), windows(v), queries(q2), windows(k2), mask: combined)
        return out.reshaped(b, groups, h, block, d).transposed(0, 2, 1, 3, 4).reshaped(b, h, t, d)
    }

    private func block(_ x: MLXArray, _ i: Int) -> MLXArray {
        let p = "blocks.\(i)."
        var x = x + attention(w.dyT(x, p + "pre_norm"), p + "attn.")
        let parts = split(w.linear(w.dyT(x, p + "ff_norm"), p + "ff.glu_proj"), parts: 2, axis: -1)
        let gated = kind == .large && i >= 5 ? parts[0] * sin(parts[1] * Float.pi) : parts[0] * silu(parts[1])
        x = x + w.linear(gated, p + "ff.proj_out")
        return x
    }

    // MARK: Decode

    /// One window of latents [B, 256, T] to patches [B, 512, T * 16].
    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let b = latents.dim(0), length = latents.dim(2), dim = kind.dim
        var x = latents * w["running_std"]
        x = w.linear(x.transposed(0, 2, 1), "project_in")
        let newTokens = broadcast(w["new_tokens"][.newAxis], to: [b, length, Self.stride, dim])
        x = concatenated([x[0..., 0..., .newAxis, 0...], newTokens], axis: 2).reshaped(b, length * Self.subChunk, dim)
        let internalLength = length * Self.subChunk

        switch kind {
        case .small:
            // Blocks 0-2 in chunks of 34, then blocks 3-5 on chunks shifted by 17.
            let effective = 34, shift = 17
            x = x.reshaped(b * (internalLength / effective), effective, dim)
            for i in 0 ..< 3 { x = block(x, i) }
            x = x.reshaped(b, internalLength, dim)
            x = concatenated([x[0..., ..<shift, 0...], x, x[0..., (internalLength - shift)..., 0...]], axis: 1)
            x = x.reshaped(b * ((internalLength + effective) / effective), effective, dim)
            for i in 3 ..< 6 { x = block(x, i) }
            x = x.reshaped(b, internalLength + effective, dim)[0..., shift ..< (shift + internalLength), 0...]
        case .large:
            for i in 0 ..< kind.blocks { x = block(x, i) }
        }

        x = x.reshaped(b * length, Self.subChunk, dim)[0..., 1..., 0...].reshaped(b, length * Self.stride, dim)
        let out = kind == .small ? w.conv1d(x, "mapping", padding: 1) : w.linear(x, "mapping")
        return out.transposed(0, 2, 1)
    }

    /// decode_chunked: every call sees `chunk + 2 * overlap` real latents (never zero padded) and keeps its middle.
    /// Falls back as sa3_mlx.py does for short clips (SAME-S needs an even window).
    func decode(_ latents: MLXArray, isCancelled: () -> Bool = { false }) throws -> MLXArray {
        let length = latents.dim(2)
        var (chunk, overlap) = kind.chunking
        if length <= chunk + 2 * overlap {
            if kind == .large || length % 2 == 0 { return self(latents) }
            if length > 6 {
                (chunk, overlap) = (2, 2)
            } else {
                let even = concatenated([latents, latents[.ellipsis, (length - 1)...]], axis: -1)
                return self(even)[.ellipsis, ..<(length * Self.stride)]
            }
        }
        let kernel = chunk + 2 * overlap
        var pieces: [MLXArray] = []
        let first = self(latents[.ellipsis, 0 ..< kernel])
        pieces.append(first[.ellipsis, ..<((chunk + overlap) * Self.stride)])
        eval(pieces)
        var i = chunk + overlap
        while i + chunk + overlap <= length {
            if isCancelled() { throw CancellationError() }
            let out = self(latents[.ellipsis, (i - overlap) ..< (i + chunk + overlap)])
            let piece = out[.ellipsis, (overlap * Self.stride) ..< ((overlap + chunk) * Self.stride)]
            eval(piece)
            pieces.append(piece)
            i += chunk
        }
        let remaining = length - i
        if remaining > 0 {
            let last = self(latents[.ellipsis, (length - kernel) ..< length])
            pieces.append(last[.ellipsis, (last.dim(-1) - remaining * Self.stride)...])
        }
        return concatenated(pieces, axis: -1)
    }
}
