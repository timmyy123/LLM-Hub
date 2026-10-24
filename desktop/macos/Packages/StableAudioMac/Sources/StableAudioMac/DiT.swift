import Foundation
import MLX

/// The SA3 diffusion transformer, ported from Stability's models/defs/dit_mlx.py (sa3-sm-music: 1024 wide, 20 layers,
/// standard attention) and dit_mlx_medium.py (sa3-medium: 1536 wide, 24 layers, differential attention). Same weight
/// names as the official .npz files. Channels-first in and out: x [B, 256, T], t [B], cross [B, 257, 768], global [B, 768].
final class DiT {
    struct Config {
        let embed: Int
        let depth: Int
        let headCount: Int
        /// sa3-medium subtracts a second attention map (q_diff, k_diff) in self- and cross-attention.
        let differential: Bool

        static let small = Config(embed: 1024, depth: 20, headCount: 16, differential: false)
        static let medium = Config(embed: 1536, depth: 24, headCount: 24, differential: true)
    }

    static let ioChannels = 256
    static let memoryTokens = 64
    static let localAddCondDim = 257
    private static let normEps: Float = 1e-5
    private static let qkNormEps: Float = 1e-6
    private static let scale: Float = pow(64, -0.5)

    let config: Config
    private var w: Weights
    private let freqs: MLXArray
    /// The conditioner baked into the DiT file under "cond.": prompt padding embedding and the seconds_total embedder.
    let paddingEmbedding: MLXArray
    private let secondsWeight: MLXArray
    private let secondsBias: MLXArray

    init(url: URL, config: Config, dtype: DType = .float16) throws {
        self.config = config
        var arrays = try Npz.load(url)
        guard let pad = arrays.removeValue(forKey: "cond.padding_embedding"),
              let sw = arrays.removeValue(forKey: "cond.seconds_total_weight"),
              let sb = arrays.removeValue(forKey: "cond.seconds_total_bias") else {
            throw Npz.NpzError(message: "\(url.lastPathComponent) has no conditioner")
        }
        paddingEmbedding = pad.asType(.float32)
        secondsWeight = sw.asType(.float32)
        secondsBias = sb.asType(.float32)
        w = Weights(arrays: arrays)
        w.cast(dtype)
        // ExpoFourierFeatures(256, 0.5, 10000) for the timestep, in fp32.
        let ramp = MLXArray.linspace(Float(0), Float(1), count: 128)
        let logMin = log(0.5), logMax = log(10000.0)
        freqs = exp(ramp * Float(logMax - logMin) + Float(logMin)) * 2 * Float.pi
        eval(paddingEmbedding, secondsWeight, secondsBias, freqs)
    }

    // MARK: Conditioning (sa3_pipeline.py)

    /// SecondsTotalEmbedder: expo Fourier features of seconds / 384 through a Linear, [1, 1, 768] fp32.
    func secondsEmbedding(_ seconds: Double) -> MLXArray {
        var s = MLXArray([Float(seconds)])
        s = clip(s, min: 0, max: 384)
        let normalized = (s - 0) / (384 - 0)
        let t = normalized.asType(.float32).reshaped(-1, 1)
        let ramp = MLXArray(0 ..< 128).asType(.float32) / Float(127)
        let f = exp(ramp * Float(log(10000.0) - log(0.5)) + Float(log(0.5)))
        let args = t * f * 2 * Float.pi
        let features = concatenated([cos(args), sin(args)], axis: -1)
        let out = matmul(features, secondsWeight.T) + secondsBias
        return out[0..., .newAxis, 0...]
    }

    /// Replaces padded T5Gemma positions with the learned padding embedding.
    func applyPromptPadding(_ embeds: MLXArray, mask: MLXArray) -> MLXArray {
        let m = mask.asType(embeds.dtype)[.ellipsis, .newAxis]
        let pe = paddingEmbedding.asType(embeds.dtype).reshaped(1, 1, -1)
        return embeds * m + pe * (1 - m)
    }

    // MARK: Forward

    private func selfAttention(_ x: MLXArray, _ p: String) -> MLXArray {
        let n = config.headCount
        let qkv = w.linear(x, p + "to_qkv")
        let parts = split(qkv, parts: config.differential ? 5 : 3, axis: -1)
        var q = rope32(w.rmsNorm(heads(parts[0], n), p + "q_norm", eps: Self.qkNormEps))
        var k = rope32(w.rmsNorm(heads(parts[1], n), p + "k_norm", eps: Self.qkNormEps))
        let v = heads(parts[2], n)
        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: Self.scale, mask: nil)
        if config.differential {
            q = rope32(w.rmsNorm(heads(parts[3], n), p + "q_norm", eps: Self.qkNormEps))
            k = rope32(w.rmsNorm(heads(parts[4], n), p + "k_norm", eps: Self.qkNormEps))
            out = out - MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: Self.scale, mask: nil)
        }
        return w.linear(merge(out), p + "to_out")
    }

    private func crossAttention(_ x: MLXArray, _ context: MLXArray, _ p: String) -> MLXArray {
        let n = config.headCount
        let qs = config.differential ? split(w.linear(x, p + "to_q"), parts: 2, axis: -1) : [w.linear(x, p + "to_q")]
        let kvs = split(w.linear(context, p + "to_kv"), parts: config.differential ? 3 : 2, axis: -1)
        // Standard: k, v. Differential: k, k_diff, v.
        let v = heads(config.differential ? kvs[2] : kvs[1], n)
        let q = w.rmsNorm(heads(qs[0], n), p + "q_norm", eps: Self.qkNormEps)
        let k = w.rmsNorm(heads(kvs[0], n), p + "k_norm", eps: Self.qkNormEps)
        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: Self.scale, mask: nil)
        if config.differential {
            let q2 = w.rmsNorm(heads(qs[1], n), p + "q_norm", eps: Self.qkNormEps)
            let k2 = w.rmsNorm(heads(kvs[1], n), p + "k_norm", eps: Self.qkNormEps)
            out = out - MLXFast.scaledDotProductAttention(queries: q2, keys: k2, values: v, scale: Self.scale, mask: nil)
        }
        return w.linear(merge(out), p + "to_out")
    }

    private func block(_ x: MLXArray, _ context: MLXArray, _ global: MLXArray, _ local: MLXArray, _ i: Int) -> MLXArray {
        let p = "transformer.layers.\(i)."
        let ss = (w[p + "to_scale_shift_gate"] + global)[0..., .newAxis, 0...]
        let m = split(ss, parts: 6, axis: -1)
        var x = x
        var h = w.rmsNorm(x, p + "pre_norm", eps: Self.normEps)
        h = h * (1 + m[0]) + m[1]
        h = selfAttention(h, p + "self_attn.")
        h = h * sigmoid(1 - m[2])
        x = h + x

        x = x + crossAttention(w.rmsNorm(x, p + "cross_attend_norm", eps: Self.normEps), context, p + "cross_attn.")

        // Local add-cond (zeros for text to audio) through this block's MLP, left-padded for the memory tokens.
        var localEmb = w.linear(local, p + "to_local_embed.seq.0")
        localEmb = w.linear(silu(localEmb), p + "to_local_embed.seq.2")
        let pad = MLXArray.zeros([localEmb.dim(0), Self.memoryTokens, config.embed], dtype: localEmb.dtype)
        x = x + concatenated([pad, localEmb], axis: 1)

        h = w.rmsNorm(x, p + "ff_norm", eps: Self.normEps)
        h = h * (1 + m[3]) + m[4]
        let glu = split(w.linear(h, p + "ff.ff.0.proj"), parts: 2, axis: -1)
        h = w.linear(glu[0] * silu(glu[1]), p + "ff.ff.2")
        h = h * sigmoid(1 - m[5])
        return h + x
    }

    /// Velocity v(x, t).
    func callAsFunction(_ x: MLXArray, t: MLXArray, cross: MLXArray, global: MLXArray) -> MLXArray {
        let b = x.dim(0), length = x.dim(-1)
        let context = w.linear(silu(w.linear(cross, "to_cond_embed.0")), "to_cond_embed.2")
        let globalPre = w.linear(silu(w.linear(global, "to_global_embed.0")), "to_global_embed.2")
        let args = t[0..., .newAxis] * freqs
        var tf = concatenated([cos(args), sin(args)], axis: -1)
        tf = w.linear(silu(w.linear(tf, "to_timestep_embed.0")), "to_timestep_embed.2")
        let globalEmbed = globalPre + tf

        let xlc = x.transposed(0, 2, 1)
        let xpp = w.conv1d(xlc, "preprocess_conv") + xlc
        let local = MLXArray.zeros([b, length, Self.localAddCondDim], dtype: .float32)

        // ContinuousTransformer.
        var h = w.linear(xpp, "transformer.project_in")
        let memory = broadcast(w["transformer.memory_tokens"][.newAxis], to: [b, Self.memoryTokens, config.embed])
        h = concatenated([memory, h], axis: 1)
        let g = w.linear(silu(w.linear(globalEmbed, "transformer.global_cond_embedder.0")), "transformer.global_cond_embedder.2")
        for i in 0 ..< config.depth {
            h = block(h, context, g, local, i)
        }
        h = w.linear(h[0..., Self.memoryTokens..., 0...], "transformer.project_out")

        let out = w.conv1d(h, "postprocess_conv") + h
        return out.transposed(0, 2, 1)
    }
}
