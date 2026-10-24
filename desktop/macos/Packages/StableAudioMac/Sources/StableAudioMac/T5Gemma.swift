import Foundation
import MLX
import SentencePiece

/// T5Gemma (google/t5gemma-b-b-ul2) encoder, ported from Stability's models/defs/t5gemma_mlx.py: 12 Gemma 2 style layers
/// (RMSNorm with 1 + w, RoPE, attention logit softcap 50, GeGLU), FP16. The tokenizer ships inside t5gemma_f16.npz.
/// Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms.
final class T5GemmaEncoder {
    static let maxTokens = 256

    private var w: Weights
    private let tokenizer: SentencePiece
    private let layers: Int
    private let headCount: Int
    private let headDim: Int
    private let eps: Float
    private let softcap: Float
    private let scaling: Float
    private let invFreq: MLXArray

    init(url: URL) throws {
        var arrays = try Npz.load(url)
        guard let model = arrays.removeValue(forKey: "TOKENIZER_MODEL"), let meta = arrays.removeValue(forKey: "META") else {
            throw Npz.NpzError(message: "t5gemma_f16.npz has no tokenizer")
        }
        tokenizer = SentencePiece(data: Data(model.asArray(UInt8.self)))
        let config = (try? JSONSerialization.jsonObject(with: Data(meta.asArray(UInt8.self)))) as? [String: Any] ?? [:]
        layers = config["num_hidden_layers"] as? Int ?? 12
        headCount = config["num_attention_heads"] as? Int ?? 12
        let headDim = config["head_dim"] as? Int ?? 64
        self.headDim = headDim
        eps = Float(config["rms_norm_eps"] as? Double ?? 1e-6)
        softcap = Float(config["attn_logit_softcapping"] as? Double ?? 50)
        scaling = pow(Float(config["query_pre_attn_scalar"] as? Int ?? 64), -0.5)
        let theta = Float(config["rope_theta"] as? Double ?? 10000)
        invFreq = arrays.removeValue(forKey: "rope_inv_freq")?.asType(.float32)
            ?? 1 / pow(MLXArray(theta), MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) }))
        w = Weights(arrays: arrays)
        eval(Array(w.arrays.values) + [invFreq])
    }

    /// Token ids of a prompt (no BOS/EOS, as SentencePieceProcessor.Encode), truncated to 256.
    func tokenize(_ prompt: String) -> [Int32] {
        Array(tokenizer.encode(prompt).map(\.id).prefix(Self.maxTokens))
    }

    /// (embeddings [1, 256, 768] fp16, mask [1, 256] int32) for one prompt, padded to 256 tokens with id 0.
    func encode(_ prompt: String) -> (embeds: MLXArray, mask: MLXArray) {
        let tokens = tokenize(prompt)
        var ids = [Int32](repeating: 0, count: Self.maxTokens)
        var mask = [Int32](repeating: 0, count: Self.maxTokens)
        for (i, t) in tokens.enumerated() {
            ids[i] = t
            mask[i] = 1
        }
        // An empty prompt would make every attention row -inf; give the pass one visible position (the mask stays 0).
        var visible = mask
        if tokens.isEmpty { visible[0] = 1 }
        let out = forward(MLXArray(ids, [1, Self.maxTokens]), MLXArray(visible, [1, Self.maxTokens]))
        return (out, MLXArray(mask, [1, Self.maxTokens]))
    }

    /// Gemma RMSNorm: normalize in fp32, scale by (1 + weight), cast back.
    private func norm(_ x: MLXArray, _ key: String) -> MLXArray {
        let dtype = x.dtype
        let x32 = x.asType(.float32)
        let variance = (x32 * x32).mean(axis: -1, keepDims: true)
        let n = x32 * rsqrt(variance + eps)
        return (n * (1 + w[key].asType(.float32))).asType(dtype)
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    private func forward(_ ids: MLXArray, _ attention: MLXArray) -> MLXArray {
        var x = w["embed_tokens.weight"][ids].asType(.float16)
        x = x * MLXArray(Float(w["embed_tokens.weight"].dim(1)).squareRoot(), dtype: x.dtype)
        let length = x.dim(1)

        let positions = MLXArray(0 ..< length).asType(.float32)
        let freqs = outer(positions, invFreq)
        let emb = concatenated([freqs, freqs], axis: -1)
        let cos = MLX.cos(emb)[.newAxis, .newAxis], sin = MLX.sin(emb)[.newAxis, .newAxis]
        let keep = attention.asType(.float32)
        let addMask = ((1.0 - keep) * -1e9)[0..., .newAxis, .newAxis, 0...].asType(x.dtype)

        for i in 0 ..< layers {
            let p = "layers.\(i)."
            var h = norm(x, p + "pre_self_attn_layernorm.weight")
            var q = heads(w.linear(h, p + "self_attn.q_proj"), headCount)
            var k = heads(w.linear(h, p + "self_attn.k_proj"), headCount)
            let v = heads(w.linear(h, p + "self_attn.v_proj"), headCount)
            q = (q * cos.asType(q.dtype)) + (Self.rotateHalf(q) * sin.asType(q.dtype))
            k = (k * cos.asType(k.dtype)) + (Self.rotateHalf(k) * sin.asType(k.dtype))
            var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scaling
            scores = tanh(scores / softcap) * softcap
            scores = scores + addMask
            let probs = softmax(scores.asType(.float32), axis: -1).asType(v.dtype)
            h = w.linear(merge(matmul(probs, v)), p + "self_attn.o_proj")
            h = norm(h, p + "post_self_attn_layernorm.weight")
            x = x + h
            h = norm(x, p + "pre_feedforward_layernorm.weight")
            let gate = w.linear(h, p + "mlp.gate_proj")
            let gelu = 0.5 * gate * (1 + tanh((2 / Float.pi).squareRoot() * (gate + 0.044715 * pow(gate, 3))))
            h = w.linear(gelu * w.linear(h, p + "mlp.up_proj"), p + "mlp.down_proj")
            h = norm(h, p + "post_feedforward_layernorm.weight")
            x = x + h
        }
        return norm(x, "norm.weight")
    }
}
