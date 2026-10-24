using System.Numerics.Tensors;

namespace LLMHub.Services.Studio.Npu;

/// <summary>
/// Stable Diffusion 1.x's text encoder (OpenAI CLIP ViT-L/14 text transformer: 12 pre-norm layers, 12 heads, causal
/// attention, quick GELU) on the CPU. Weights come from the checkpoint's own text encoder in safetensors form, either
/// diffusers layout ("text_model.*") or a single-file checkpoint ("cond_stage_model.transformer.text_model.*").
/// It takes well under a second per prompt, so it does not need the NPU.
/// </summary>
public sealed class ClipTextEncoder
{
    private const int Tokens = ClipTokenizer.Length;
    private const int Width = 768;
    private const int Heads = 12;
    private const int HeadDim = Width / Heads;
    private const int Hidden = 3072;
    private readonly Dictionary<string, float[]> _w;
    private readonly int _layers;

    public ClipTextEncoder(string safetensorsPath)
    {
        using (var stream = File.OpenRead(safetensorsPath))
        {
            var (entries, _) = SafeTensors.ReadHeader(stream);
            Prefix = entries.Any(e => e.Name.StartsWith("cond_stage_model.transformer.text_model.", StringComparison.Ordinal))
                ? "cond_stage_model.transformer.text_model."
                : "text_model.";
        }
        _w = SafeTensors.Load(safetensorsPath, Prefix);
        _layers = Enumerable.Range(0, 48).TakeWhile(i => _w.ContainsKey($"encoder.layers.{i}.mlp.fc1.weight")).Count();
        if (_layers == 0 || !_w.ContainsKey("embeddings.token_embedding.weight"))
            throw new InvalidDataException($"No CLIP text encoder in {Path.GetFileName(safetensorsPath)}");
    }

    public string Prefix { get; }

    /// <summary>last_hidden_state [77, 768] for 77 token ids.</summary>
    public float[] Encode(int[] ids)
    {
        var x = new float[Tokens * Width];
        var tok = _w["embeddings.token_embedding.weight"];
        var pos = _w["embeddings.position_embedding.weight"];
        for (var t = 0; t < Tokens; t++)
        {
            for (var c = 0; c < Width; c++)
                x[t * Width + c] = tok[ids[t] * Width + c] + pos[t * Width + c];
        }

        var h = new float[Tokens * Width];
        var q = new float[Tokens * Width];
        var k = new float[Tokens * Width];
        var v = new float[Tokens * Width];
        var attn = new float[Tokens * Width];
        var mlp = new float[Tokens * Hidden];
        var gate = new float[Tokens * Hidden];
        for (var l = 0; l < _layers; l++)
        {
            var p = $"encoder.layers.{l}.";
            LayerNorm(x, h, _w[p + "layer_norm1.weight"], _w[p + "layer_norm1.bias"]);
            Linear(h, q, _w[p + "self_attn.q_proj.weight"], _w[p + "self_attn.q_proj.bias"], Width, Width);
            Linear(h, k, _w[p + "self_attn.k_proj.weight"], _w[p + "self_attn.k_proj.bias"], Width, Width);
            Linear(h, v, _w[p + "self_attn.v_proj.weight"], _w[p + "self_attn.v_proj.bias"], Width, Width);
            Attention(q, k, v, attn);
            Linear(attn, h, _w[p + "self_attn.out_proj.weight"], _w[p + "self_attn.out_proj.bias"], Width, Width);
            Add(x, h);
            LayerNorm(x, h, _w[p + "layer_norm2.weight"], _w[p + "layer_norm2.bias"]);
            Linear(h, mlp, _w[p + "mlp.fc1.weight"], _w[p + "mlp.fc1.bias"], Width, Hidden);
            // quick GELU: x * sigmoid(1.702 x)
            TensorPrimitives.Multiply(mlp, 1.702f, gate);
            TensorPrimitives.Sigmoid(gate, gate);
            TensorPrimitives.Multiply(mlp, gate, mlp);
            Linear(mlp, h, _w[p + "mlp.fc2.weight"], _w[p + "mlp.fc2.bias"], Hidden, Width);
            Add(x, h);
        }
        LayerNorm(x, h, _w["final_layer_norm.weight"], _w["final_layer_norm.bias"]);
        return h;
    }

    private static void Add(float[] x, float[] y) => TensorPrimitives.Add(x, y, x);

    private static void LayerNorm(float[] x, float[] y, float[] gamma, float[] beta)
    {
        for (var t = 0; t < Tokens; t++)
        {
            var row = x.AsSpan(t * Width, Width);
            float mean = 0, var = 0;
            foreach (var value in row)
                mean += value;
            mean /= Width;
            foreach (var value in row)
                var += (value - mean) * (value - mean);
            var inv = 1f / MathF.Sqrt(var / Width + 1e-5f);
            for (var c = 0; c < Width; c++)
                y[t * Width + c] = (row[c] - mean) * inv * gamma[c] + beta[c];
        }
    }

    /// <summary>y[t, o] = x[t, :] · W[o, :] + b[o], parallel over output features.</summary>
    private static void Linear(float[] x, float[] y, float[] w, float[] b, int inFeatures, int outFeatures)
    {
        Parallel.For(0, (outFeatures + 63) / 64, block =>
        {
            for (var o = block * 64; o < Math.Min(outFeatures, block * 64 + 64); o++)
            {
                var row = w.AsSpan(o * inFeatures, inFeatures);
                for (var t = 0; t < Tokens; t++)
                    y[t * outFeatures + o] = Dot(x.AsSpan(t * inFeatures, inFeatures), row) + b[o];
            }
        });
    }

    // TensorPrimitives is SIMD code from an optimized library, so this stays fast in Debug builds of the app too.
    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b) => TensorPrimitives.Dot(a, b);

    private static void Attention(float[] q, float[] k, float[] v, float[] output)
    {
        var scale = 1f / MathF.Sqrt(HeadDim);
        Parallel.For(0, Heads, head =>
        {
            var scores = new float[Tokens];
            var off = head * HeadDim;
            for (var i = 0; i < Tokens; i++)
            {
                var qi = q.AsSpan(i * Width + off, HeadDim);
                var max = float.NegativeInfinity;
                for (var j = 0; j <= i; j++) // causal
                {
                    scores[j] = Dot(qi, k.AsSpan(j * Width + off, HeadDim)) * scale;
                    max = MathF.Max(max, scores[j]);
                }
                float sum = 0;
                for (var j = 0; j <= i; j++)
                {
                    scores[j] = MathF.Exp(scores[j] - max);
                    sum += scores[j];
                }
                var o = output.AsSpan(i * Width + off, HeadDim);
                o.Clear();
                for (var j = 0; j <= i; j++)
                {
                    TensorPrimitives.MultiplyAdd(v.AsSpan(j * Width + off, HeadDim), scores[j] / sum, o, o);
                }
            }
        });
    }
}
