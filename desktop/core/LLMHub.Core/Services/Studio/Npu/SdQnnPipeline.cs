using System.Diagnostics;

namespace LLMHub.Services.Studio.Npu;

/// <summary>One image request for <see cref="SdQnnPipeline"/>. <see cref="InitImage"/> (512 x 512 RGB) makes it image to image.</summary>
public sealed record SdRequest(string Prompt, string Negative, int Steps, float Cfg, long Seed, byte[]? InitImage = null, float Strength = 0.6f);

/// <summary>Timings of one generation, for the log and the Studio status line.</summary>
public sealed record SdTimings(double TextSeconds, double DenoiseSeconds, double DecodeSeconds, int UnetRuns, int Steps)
{
    public double TotalSeconds => TextSeconds + DenoiseSeconds + DecodeSeconds;
    public double StepsPerSecond => DenoiseSeconds > 0 ? Steps / DenoiseSeconds : 0;
    public override string ToString() =>
        $"text {TextSeconds:F2}s, {Steps} steps in {DenoiseSeconds:F2}s ({StepsPerSecond:F2} steps/s, {UnetRuns} UNet runs, {DenoiseSeconds * 1000 / Math.Max(1, UnetRuns):F0} ms each), decode {DecodeSeconds:F2}s, total {TotalSeconds:F2}s";
}

/// <summary>
/// Stable Diffusion 1.5 at 512 x 512 on the Snapdragon NPU, with the Android app's QNN builds (xororz/sd-qnn): the UNet
/// and the VAE decoder/encoder are Hexagon context binaries run through <see cref="QnnGraphSession"/>; the CLIP text
/// encoder runs on the CPU (<see cref="ClipTextEncoder"/>); sampling is DPM++ 2M Karras with classifier-free guidance.
/// </summary>
public sealed class SdQnnPipeline : IDisposable
{
    public const int Size = 512;
    private const int Latent = Size / 8;
    private const int LatentCount = 4 * Latent * Latent;
    private const float VaeScale = 0.18215f;

    private readonly string _dir;
    private readonly QnnGraphSession _unet;
    private readonly QnnGraphSession _decoder;
    private QnnGraphSession? _encoder;
    private readonly ClipTokenizer _tokenizer;
    private readonly ClipTextEncoder _clip;
    private readonly DpmSolver _solver = new();
    private readonly int _sample, _timestep, _text;

    /// <summary>Loads unet.bin, vae_decoder.bin, tokenizer.json and text_encoder.safetensors from <paramref name="dir"/>.</summary>
    public SdQnnPipeline(string dir)
    {
        _dir = dir;
        var clock = Stopwatch.StartNew();
        _tokenizer = new ClipTokenizer(Path.Combine(dir, "tokenizer.json"));
        _clip = new ClipTextEncoder(Path.Combine(dir, "text_encoder.safetensors"));
        var clipSeconds = clock.Elapsed.TotalSeconds;
        _unet = new QnnGraphSession(Path.Combine(dir, "unet.bin"));
        _decoder = new QnnGraphSession(Path.Combine(dir, "vae_decoder.bin"));
        var inputs = _unet.Graph.Inputs;
        _sample = Find(inputs, t => t.Dims.Length == 4);
        _text = Find(inputs, t => t.Dims.Length == 3 && t.Dims[^1] == 768);
        _timestep = Find(inputs, t => t.Count == 1);
        Log.Info($"SD NPU {Path.GetFileName(dir)} loaded in {clock.Elapsed.TotalSeconds:F1}s (CLIP {clipSeconds:F1}s); UNet {_unet.Context.BuildId} v{_unet.Context.DspArch}: {string.Join("; ", inputs)}");
    }

    public bool CanEncode => File.Exists(Path.Combine(_dir, "vae_encoder.bin"));

    private static int Find(IReadOnlyList<QnnTensorSpec> inputs, Func<QnnTensorSpec, bool> match)
    {
        for (var i = 0; i < inputs.Count; i++)
        {
            if (match(inputs[i]))
                return i;
        }
        throw new InvalidDataException("Unexpected UNet inputs: " + string.Join("; ", inputs));
    }

    /// <summary>Generates one 512 x 512 RGB image. <paramref name="progress"/> gets (step, steps).</summary>
    public (byte[] Rgb, SdTimings Timings) Generate(SdRequest request, Action<int, int>? progress, CancellationToken ct)
    {
        var clock = Stopwatch.StartNew();
        var cond = EncodeText(request.Prompt);
        var guided = request.Cfg > 1.0001f;
        var uncond = guided ? EncodeText(request.Negative) : null;
        var textSeconds = clock.Elapsed.TotalSeconds;

        clock.Restart();
        var random = new Gaussian(request.Seed);
        var steps = Math.Clamp(request.Steps, 1, 100);
        var sigmas = _solver.Sigmas(steps);
        var x = new float[LatentCount];
        var start = 0;
        if (request.InitImage is { } image)
        {
            var latent = EncodeImage(image, random);
            start = Math.Clamp(steps - (int)Math.Round(steps * Math.Clamp(request.Strength, 0.05f, 1f)), 0, steps - 1);
            for (var i = 0; i < x.Length; i++)
                x[i] = latent[i] + random.Next() * (float)sigmas[start];
        }
        else
        {
            for (var i = 0; i < x.Length; i++)
                x[i] = random.Next() * (float)sigmas[0];
        }

        var runs = 0;
        var scaled = new float[LatentCount];
        var epsCond = new float[LatentCount];
        var epsUncond = new float[LatentCount];
        var total = steps - start;
        DpmSolver.Sample(x, sigmas, start, (latent, sigma) =>
        {
            ct.ThrowIfCancellationRequested();
            var cIn = (float)(1 / Math.Sqrt(sigma * sigma + 1));
            for (var i = 0; i < scaled.Length; i++)
                scaled[i] = latent[i] * cIn;
            _unet.SetInput(_sample, scaled);
            _unet.SetInput(_timestep, [(float)Math.Round(_solver.Timestep(sigma))]);
            _unet.SetInput(_text, cond);
            _unet.Run();
            _unet.GetOutput(0, epsCond);
            runs++;
            if (uncond is not null)
            {
                _unet.SetInput(_text, uncond);
                _unet.Run();
                _unet.GetOutput(0, epsUncond);
                runs++;
            }
            var denoised = new float[LatentCount];
            for (var i = 0; i < denoised.Length; i++)
            {
                var eps = uncond is null ? epsCond[i] : epsUncond[i] + request.Cfg * (epsCond[i] - epsUncond[i]);
                denoised[i] = latent[i] - (float)sigma * eps;
            }
            return denoised;
        }, step => progress?.Invoke(step, total));
        var denoiseSeconds = clock.Elapsed.TotalSeconds;

        clock.Restart();
        ct.ThrowIfCancellationRequested();
        for (var i = 0; i < x.Length; i++)
            x[i] /= VaeScale;
        _decoder.SetInput(0, x);
        _decoder.Run();
        var pixels = new float[3 * Size * Size];
        _decoder.GetOutput(0, pixels);
        var rgb = new byte[Size * Size * 3];
        const int plane = Size * Size;
        for (var p = 0; p < plane; p++)
        {
            for (var c = 0; c < 3; c++)
                rgb[p * 3 + c] = (byte)Math.Clamp((int)MathF.Round((pixels[c * plane + p] + 1) * 127.5f), 0, 255);
        }
        return (rgb, new SdTimings(textSeconds, denoiseSeconds, clock.Elapsed.TotalSeconds, runs, total));
    }

    /// <summary>CLIP hidden states, remembered for the last few prompts (batches and re-runs reuse them).</summary>
    private float[] EncodeText(string text)
    {
        if (_textCache.TryGetValue(text, out var cached))
            return cached;
        if (_textCache.Count >= 8)
            _textCache.Clear();
        return _textCache[text] = _clip.Encode(_tokenizer.Encode(text));
    }

    private readonly Dictionary<string, float[]> _textCache = [];

    /// <summary>VAE encoder on the NPU: 512 x 512 RGB to a scaled latent sample (mean + std * noise).</summary>
    private float[] EncodeImage(byte[] rgb, Gaussian random)
    {
        if (!CanEncode)
            throw new NotSupportedException("This model has no VAE encoder");
        _encoder ??= new QnnGraphSession(Path.Combine(_dir, "vae_encoder.bin"));
        const int plane = Size * Size;
        var input = new float[3 * plane];
        for (var p = 0; p < plane; p++)
        {
            for (var c = 0; c < 3; c++)
                input[c * plane + p] = rgb[p * 3 + c] / 127.5f - 1;
        }
        _encoder.SetInput(0, input);
        _encoder.Run();
        var mean = new float[LatentCount];
        var std = new float[LatentCount];
        _encoder.GetOutput(_encoder.OutputIndex("mean"), mean);
        _encoder.GetOutput(_encoder.OutputIndex("std"), std);
        var latent = new float[LatentCount];
        for (var i = 0; i < latent.Length; i++)
            latent[i] = (mean[i] + std[i] * random.Next()) * VaeScale;
        return latent;
    }

    public void Dispose()
    {
        _unet.Dispose();
        _decoder.Dispose();
        _encoder?.Dispose();
    }

    /// <summary>Seeded standard normal samples (Box-Muller).</summary>
    private sealed class Gaussian(long seed)
    {
        private readonly Random _random = new((int)(seed ^ (seed >> 32)));
        private float? _spare;

        public float Next()
        {
            if (_spare is { } spare)
            {
                _spare = null;
                return spare;
            }
            double u1;
            do
                u1 = _random.NextDouble();
            while (u1 <= double.Epsilon);
            var u2 = _random.NextDouble();
            var r = Math.Sqrt(-2 * Math.Log(u1));
            _spare = (float)(r * Math.Sin(2 * Math.PI * u2));
            return (float)(r * Math.Cos(2 * Math.PI * u2));
        }
    }
}
