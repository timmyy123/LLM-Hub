using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace LLMHub.Services.Studio;

/// <summary>
/// Stable Audio 3 Small music clips on Snapdragon PCs, where ComfyUI has no backend: Stability AI's ONNX exports run by
/// ONNX Runtime with DirectML on the Adreno GPU (the DiT and decoder; T5Gemma on the CPU), driven by
/// shared/stable-audio/sa3_dml.py with the tokenizer, schedule and sampler of Stability's own LiteRT runtime. The same
/// model family as the Android app's SoundGen HD. onnxruntime-directml has no Windows on Arm wheel, so it runs in a private
/// x64 Python under Windows' x64 emulation (10 s of audio in about 16 s on a Snapdragon X Elite, 36 s on its CPU alone).
/// </summary>
public sealed partial class StableAudioDmlService
{
    private const string Runtime = "https://raw.githubusercontent.com/Stability-AI/stable-audio-3/77e0398c26f7d22d801ec1bb036307e8b231f741/optimized/tflite";
    private const string Weights = "https://huggingface.co/stabilityai/stable-audio-3-optimized/resolve/08b17423649b5714b22eab20f96bf00751f2358a/onnx";
    public const int MaxSeconds = 120;
    private const int Steps = 8;

    /// <summary>The LiteRT runtime's pipeline helpers and tokenizer (relative to optimized/tflite); its LiteRT models are not used.</summary>
    private static readonly string[] RuntimeFiles = ["models/__init__.py", "models/defs/__init__.py", "models/defs/tflite_pipeline.py", "models/tokenizer.model"];

    private static readonly (string Name, string Remote, long Size)[] ModelFiles =
    [
        ("encoder.onnx", "t5gemma/encoder.onnx", 620_393_530),
        ("dit.onnx", "sa3-sm-music/dit_fp16.onnx", 921_133_617),
        ("decoder.onnx", "same-s/dec_bf16.onnx", 218_677_278),
    ];

    private static readonly string[] Packages = ["onnxruntime-directml==1.24.4", "numpy==2.5.4", "sentencepiece==0.2.2"];

    private static string Root => Path.Combine(AppPaths.Engines, "stable-audio-3");
    private static string RuntimeDir => Path.Combine(Root, "tflite");
    private static string ModelDir => Path.Combine(Root, "onnx");
    private static string Marker => Path.Combine(Root, "installed.txt");
    public static string Script => Path.Combine(AppContext.BaseDirectory, "stable-audio", "sa3_dml.py");

    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Windows on Arm, where ComfyUI (Studio's usual music engine) does not run.</summary>
    public static bool IsSupported => OperatingSystem.IsWindows() && Ioc.Hardware.IsArm64 && File.Exists(Script);

    public static long DownloadSize => ModelFiles.Sum(f => f.Size);

    public static bool IsInstalled => File.Exists(Marker) && ModelFiles.All(f => File.Exists(Path.Combine(ModelDir, f.Name)));

    public async Task EnsureInstalledAsync(Action<double, string> progress, CancellationToken ct)
    {
        if (IsInstalled)
            return;
        var setup = Loc.S("StableAudioSetup");
        progress(0, setup);
        await StandalonePython.EnsureAsync(Root, p => progress(p / 100 * 0.05, setup), ct, x64: true);
        await StandalonePython.PipAsync(Root, _ => progress(0.05, setup), ct, ["install", .. Packages]);
        foreach (var file in RuntimeFiles)
            await Fetch.FileAsync($"{Runtime}/{file}", Path.Combine(RuntimeDir, file), null, ct);
        double total = DownloadSize;
        long done = 0;
        foreach (var (name, remote, size) in ModelFiles)
        {
            var path = Path.Combine(ModelDir, name);
            if (!File.Exists(path))
            {
                var before = done;
                await Fetch.FileAsync($"{Weights}/{remote}", path, p =>
                {
                    var share = (before + size * p / 100) / total;
                    progress(0.1 + 0.9 * share, Loc.F("ComfyDownloadingModel", "Stable Audio 3", (share * 100).ToString("F0")));
                }, ct);
            }
            done += size;
        }
        await File.WriteAllTextAsync(Marker, string.Join('\n', Packages), ct);
        Log.Info($"Stable Audio 3 (ONNX Runtime, DirectML) installed in {Root}");
    }

    [GeneratedRegex(@"sampling step (\d+)/(\d+)")]
    private static partial Regex StepLine();

    [GeneratedRegex(@"load (\w+)\.onnx on (\w+)")]
    private static partial Regex LoadLine();

    /// <summary>
    /// One clip of <paramref name="seconds"/> as a WAV in <paramref name="folder"/>. <paramref name="progress"/> gets
    /// (0..1, status text) for setup, the prompt encoder, each of the 8 sampling steps and the decoder.
    /// </summary>
    public async Task<string> GenerateAsync(string prompt, int seconds, long seed, string folder, Action<double, string> progress, CancellationToken ct)
    {
        await _gate.WaitAsync(ct);
        try
        {
            await EnsureInstalledAsync(progress, ct);
            var output = Path.Combine(folder, $"stable-audio-3-small_{DateTime.Now:yyyyMMdd-HHmmss}_{seed}.wav");
            var info = new ProcessStartInfo(StandalonePython.Exe(Root))
            {
                WorkingDirectory = Root,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
            };
            foreach (var arg in new[]
                     {
                         Script, "--prompt", prompt, "--seconds", seconds.ToString(), "--steps", Steps.ToString(),
                         "--seed", (seed & int.MaxValue).ToString(), "--out", output, "--onnx", ModelDir, "--runtime", RuntimeDir,
                     })
                info.ArgumentList.Add(arg);
            info.Environment["PYTHONUTF8"] = "1";
            info.Environment["PYTHONUNBUFFERED"] = "1";
            using var process = Process.Start(info)!;
            process.StandardInput.Close();
            var tail = new StringBuilder();
            process.OutputDataReceived += (_, e) =>
            {
                if (e.Data is not { } line)
                    return;
                if (LoadLine().Match(line) is { Success: true } load)
                    Log.Info($"Stable Audio 3 {load.Groups[1].Value} on {load.Groups[2].Value}");
                if (line.Contains("[1/3]", StringComparison.Ordinal))
                    progress(0.03, Loc.S("MusicEncodingPrompt"));
                else if (StepLine().Match(line) is { Success: true } m)
                    progress(0.08 + 0.8 * int.Parse(m.Groups[1].Value) / int.Parse(m.Groups[2].Value), Loc.F("ComfyStep", m.Groups[1].Value, m.Groups[2].Value));
                else if (line.Contains("[3/3]", StringComparison.Ordinal))
                    progress(0.9, Loc.S("MusicDecoding"));
            };
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data is { } line)
                    lock (tail)
                        tail.AppendLine(line);
            };
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            try
            {
                await process.WaitForExitAsync(ct);
            }
            catch (OperationCanceledException)
            {
                process.Kill(entireProcessTree: true);
                throw;
            }
            if (process.ExitCode != 0 || !File.Exists(output))
                throw new InvalidOperationException($"Stable Audio 3 failed: {tail.ToString()[^Math.Min(tail.Length, 600)..]}");
            return output;
        }
        finally
        {
            _gate.Release();
        }
    }
}
