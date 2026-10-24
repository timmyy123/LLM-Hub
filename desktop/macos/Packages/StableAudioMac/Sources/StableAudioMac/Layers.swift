import Foundation
import MLX

/// The handful of layers the SA3 models use, written as functions over a weight dictionary so the npz keys map 1:1
/// onto Stability's MLX modules. Each mirrors the mlx Python layer it replaces (same ops, so the same dtypes and
/// numerics): nn.Linear, nn.RMSNorm, nn.Conv1d, DyT, SiLU.
struct Weights {
    var arrays: [String: MLXArray]

    subscript(_ key: String) -> MLXArray {
        guard let value = arrays[key] else { fatalError("Stable Audio 3: missing weight \(key)") }
        return value
    }

    func has(_ key: String) -> Bool { arrays[key] != nil }

    /// nn.Linear: `addmm(bias, x, W.T)` with a bias, else `x @ W.T`.
    func linear(_ x: MLXArray, _ prefix: String) -> MLXArray {
        if let bias = arrays[prefix + ".bias"] {
            return addMM(bias, x, self[prefix + ".weight"].T)
        }
        return matmul(x, self[prefix + ".weight"].T)
    }

    /// nn.RMSNorm (mx.fast.rms_norm, weight only).
    func rmsNorm(_ x: MLXArray, _ prefix: String, eps: Float) -> MLXArray {
        MLXFast.rmsNorm(x, weight: self[prefix + ".weight"], eps: eps)
    }

    /// nn.Conv1d on channels-last input, weight [out, k, in].
    func conv1d(_ x: MLXArray, _ prefix: String, padding: Int = 0) -> MLXArray {
        var y = MLX.conv1d(x, self[prefix + ".weight"], stride: 1, padding: padding)
        if let bias = arrays[prefix + ".bias"] { y = y + bias }
        return y
    }

    /// DyT: gamma * tanh(alpha * x) + beta.
    func dyT(_ x: MLXArray, _ prefix: String) -> MLXArray {
        self[prefix + ".gamma"] * tanh(self[prefix + ".alpha"] * x) + self[prefix + ".beta"]
    }

    /// Casts every array to `dtype` (as `load_weights([(k, v.astype(dtype))])` does) and materializes them.
    mutating func cast(_ dtype: DType) {
        for (key, value) in arrays where value.dtype != dtype { arrays[key] = value.asType(dtype) }
        eval(Array(arrays.values))
    }
}

func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// [B, T, H*D] to [B, H, T, D].
func heads(_ x: MLXArray, _ count: Int) -> MLXArray {
    let (b, t, e) = (x.dim(0), x.dim(1), x.dim(2))
    return x.reshaped(b, t, count, e / count).transposed(0, 2, 1, 3)
}

/// [B, H, T, D] back to [B, T, H*D].
func merge(_ x: MLXArray) -> MLXArray {
    let (b, h, t, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    return x.transposed(0, 2, 1, 3).reshaped(b, t, h * d)
}

/// mx.fast.rope over the first 32 of 64 head dims, half-half pairing, base 10000 (every SA3 model).
func rope32(_ x: MLXArray) -> MLXArray {
    MLXFast.RoPE(x, dimensions: 32, traditional: false, base: 10000, scale: 1, offset: 0)
}
