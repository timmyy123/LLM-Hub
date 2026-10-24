using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services.Studio.Npu;

/// <summary>
/// One Hexagon build of LTX-Video 2B 0.9.8 distilled (the Android app's, from android/tools/ltxv_qnn, QAIRT 2.50): four
/// transformer chunks and the VAE decoder as QNN context binaries for one clip shape.
/// </summary>
public sealed record LtxNpuBuild(string Arch, int Width, int Height, int Frames, IReadOnlyList<(string Name, long Size)> Files);

/// <summary>
/// Studio video on Snapdragon PCs: LTX-Video with the transformer and VAE decoder on the Hexagon NPU, run by
/// stable-diffusion.cpp's sd-cli.exe (built from android/stable-diffusion.cpp by desktop/windows/sdcpp) with the app's
/// QNN runtime. T5 (Q4_0 GGUF) and image-to-video encoding run on the CPU. Frames become an MP4 in the gallery.
/// </summary>
public sealed partial class LtxNpuService : ObservableObject
{
    private const string QnnRepo = "https://huggingface.co/timmyboy26/LTX-Video-2B-0.9.8-distilled-QNN/resolve/fc8e5ed0f1f2877ab58ab1ee1db262e77a5506f6";
    private const string GgufRepo = "https://huggingface.co/calcuis/ltxv-gguf/resolve/0990265881485e3419312141f6a1974ab2f7a4c6";
    private const string Stub = "ltxv_qnn_stub.safetensors";
    private const string T5 = "t5xxl_fp16-q4_0.gguf";
    private const string Vae = "ltxv_0.9.5_vae_fp16.safetensors";

    /// <summary>LTX-Video 0.9.8 distilled's sigma grid ("allowed_inference_steps").</summary>
    private static readonly float[] Distilled = [1.0f, 0.9937f, 0.9875f, 0.9812f, 0.975f, 0.9094f, 0.725f, 0.4219f];
    public const int MaxSteps = 8;

    private static readonly LtxNpuBuild V73 = new("v73", 512, 320, 25,
    [
        ("ltxv_chunk0.bin", 532_620_128), ("ltxv_chunk1.bin", 489_185_184), ("ltxv_chunk2.bin", 489_185_184),
        ("ltxv_chunk3.bin", 490_160_456), ("ltxv_vae_512x320x25.bin", 635_607_096),
    ]);

    private static readonly LtxNpuBuild V81 = new("v81", 768, 512, 25,
    [
        ("ltxv_chunk0.bin", 589_058_912), ("ltxv_chunk1.bin", 545_005_472), ("ltxv_chunk2.bin", 545_009_568),
        ("ltxv_chunk3.bin", 547_479_880), ("ltxv_vae_768x512x25.bin", 712_386_616),
    ]);

    private readonly SemaphoreSlim _gate = new(1, 1);

    public LtxNpuService() => Status = "";

    [ObservableProperty] public partial string Status { get; set; }

    /// <summary>Snapdragon X Elite / X Plus: Hexagon v73 (512 x 320); X2 Elite: v81 (768 x 512).</summary>
    public static LtxNpuBuild Build => QnnRuntime.HexagonArch(Ioc.Hardware.NpuName) >= 81 ? V81 : V73;

    public static string CliPath => Path.Combine(AppContext.BaseDirectory, "sd-cli.exe");

    /// <summary>A Snapdragon NPU and sd-cli.exe next to the app.</summary>
    public static bool IsSupported => SdNpuService.IsSupported && File.Exists(CliPath);

    public static string ModelDir => Path.Combine(SdNpuService.Root, "ltxv-" + Build.Arch);

    private static IEnumerable<(string Name, string Url, long Size)> Files =>
        Build.Files.Select(f => (f.Name, $"{QnnRepo}/{Build.Arch}/{f.Name}", f.Size))
            .Append((Stub, $"{QnnRepo}/{Stub}", 26_911_008L))
            .Append((T5, $"{GgufRepo}/{T5}", 2_896_123_072L))
            .Append((Vae, $"{GgufRepo}/{Vae}", 2_493_868_876L));

    public static long DownloadSize => Files.Sum(f => f.Size);

    public static bool IsInstalled => Files.All(f => File.Exists(Path.Combine(ModelDir, f.Name)));

    /// <summary>Writes the frames as an MP4 (frames, fps, output path); set by the platform app.</summary>
    public static Func<IReadOnlyList<string>, int, string, Task>? VideoEncoder { get; set; }

    /// <summary>A video's poster frame, shown as its gallery thumbnail (not listed as a picture of its own).</summary>
    public static string PosterOf(string video) => video + ".poster.png";

    public async Task EnsureModelAsync(CancellationToken ct)
    {
        if (IsInstalled)
            return;
        double total = DownloadSize;
        long done = 0;
        foreach (var (name, url, size) in Files)
        {
            var path = Path.Combine(ModelDir, name);
            if (!File.Exists(path))
            {
                var before = done;
                await Fetch.FileAsync(url, path, p => SetStatus(Loc.F("ComfyDownloadingModel", "LTX-Video", ((before + size * p / 100) * 100 / total).ToString("F0"))), ct);
            }
            done += size;
        }
        SetStatus("");
        Log.Info($"LTX-Video NPU {Build.Arch} installed in {ModelDir}");
    }

    /// <summary>Sigmas for <paramref name="steps"/> steps (evenly spaced picks of the distilled grid), ending in 0.</summary>
    public static float[] Sigmas(int steps)
    {
        var n = Math.Clamp(steps, 1, Distilled.Length);
        var picked = n == Distilled.Length
            ? Distilled
            : Enumerable.Range(0, n).Select(i => Distilled[(int)(i * (Distilled.Length - 1f) / Math.Max(1, n - 1))]).ToArray();
        return [.. picked, 0f];
    }

    /// <summary>
    /// Generates one clip (text to video, or image to video from <paramref name="initImage"/>) and returns the MP4 in the
    /// gallery. <paramref name="progress"/> gets (0..1, status text).
    /// </summary>
    public async Task<string> GenerateAsync(string prompt, int steps, long seed, string? initImage, Action<double, string> progress, CancellationToken ct)
    {
        await _gate.WaitAsync(ct);
        var work = Directory.CreateDirectory(Path.Combine(Path.GetTempPath(), "llmhub-ltxv-" + Guid.NewGuid().ToString("N")[..8])).FullName;
        try
        {
            await EnsureModelAsync(ct);
            var build = Build;
            string? init = null;
            if (initImage is not null)
            {
                var rgb = await (SdNpuService.ImageLoader ?? throw new NotSupportedException("No image decoder"))(initImage, build.Width, build.Height);
                init = Path.Combine(work, "init.png");
                PngWriter.Write(init, rgb, build.Width, build.Height);
            }
            var sigmas = Sigmas(steps);
            var dir = ModelDir.Replace('\\', '/');
            var args = new List<string>
            {
                "-M", "vid_gen",
                "--diffusion-model", $"{dir}/{Stub}", "--t5xxl", $"{dir}/{T5}", "--vae", $"{dir}/{Vae}",
                "--model-args", $"ltxv_qnn_dir={dir},ltxv_qnn_lib_dir={QnnRuntime.LibraryDir.TrimEnd('\\', '/').Replace('\\', '/')}",
                "--backend", "cpu", "--params-backend", "te=disk,vae=disk", "--mmap",
                "-t", Math.Clamp(Environment.ProcessorCount, 2, 12).ToString(),
                "-p", prompt,
                "-W", build.Width.ToString(), "-H", build.Height.ToString(), "--video-frames", build.Frames.ToString(), "--fps", "24",
                "--steps", (sigmas.Length - 1).ToString(), "--cfg-scale", "1", "--sampling-method", "euler",
                "--sigmas", string.Join(',', sigmas.Select(s => s.ToString(System.Globalization.CultureInfo.InvariantCulture))),
                "-s", seed.ToString(), "--vae-tiling", "--temporal-tiling",
                "-o", Path.Combine(work, "frame_%03d.png").Replace('\\', '/'),
            };
            if (init is not null)
                args.AddRange(["-i", init.Replace('\\', '/')]);

            var clock = Stopwatch.StartNew();
            var tail = await RunAsync(args, sigmas.Length - 1, progress, ct);
            var frames = Directory.EnumerateFiles(work, "frame_*.png").Order(StringComparer.Ordinal).ToList();
            if (frames.Count == 0)
                throw new InvalidOperationException("LTX-Video produced no frames: " + tail);
            var seconds = clock.Elapsed.TotalSeconds;

            progress(1, Loc.S("NpuEncodingVideo"));
            var name = $"{DateTime.Now:yyyyMMdd-HHmmss}-ltxv-npu-{seed}";
            var video = Path.Combine(ComfyService.Gallery, name + ".mp4");
            File.Copy(frames[0], PosterOf(video), true);
            if (VideoEncoder is null)
                throw new NotSupportedException("No video encoder");
            await VideoEncoder(frames, 24, video);
            Log.Info($"LTX-Video NPU {build.Arch} {build.Width}x{build.Height}x{frames.Count} seed {seed}{(init is null ? "" : " image-to-video")}: " +
                     $"{seconds:F1}s in sd-cli ({Timings(tail)}), MP4 in {clock.Elapsed.TotalSeconds - seconds:F1}s -> {video}");
            return video;
        }
        finally
        {
            _gate.Release();
            try
            {
                Directory.Delete(work, true);
            }
            catch (IOException)
            {
            }
        }
    }

    /// <summary>sd-cli's own timing lines (prompt encoding, sampling, decoding), for the log.</summary>
    private static string Timings(string tail) =>
        string.Join("; ", tail.Split('\n').Select(l => l.Trim())
            .Where(l => l.Contains("get_learned_condition completed", StringComparison.Ordinal) || l.Contains("sampling completed", StringComparison.Ordinal)
                        || l.Contains("encode_first_stage completed", StringComparison.Ordinal) || l.Contains("VAE decode on the NPU took", StringComparison.Ordinal)
                        || l.Contains("generate_video completed", StringComparison.Ordinal))
            .Select(l => Regex.Replace(Regex.Replace(l, @"^\[[A-Z]\]\s*", ""), @"\s*---.*$", "")));

    /// <summary>Runs sd-cli, reporting denoising steps ("3/8") as progress; returns the end of its output.</summary>
    private async Task<string> RunAsync(List<string> args, int steps, Action<double, string> progress, CancellationToken ct)
    {
        var info = new ProcessStartInfo(CliPath)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = AppContext.BaseDirectory,
        };
        foreach (var arg in args)
            info.ArgumentList.Add(arg);
        using var process = Process.Start(info) ?? throw new InvalidOperationException("sd-cli did not start");
        Ioc.Engines.Llama.Track(process);
        var tail = new StringBuilder();
        var sampling = false;
        void OnLine(string line)
        {
            lock (tail)
            {
                tail.AppendLine(line);
                if (tail.Length > 16000)
                    tail.Remove(0, tail.Length - 16000);
            }
            if (line.Contains("sampling", StringComparison.OrdinalIgnoreCase))
                sampling = true;
            if (line.Contains("decod", StringComparison.OrdinalIgnoreCase) && line.Contains("vae", StringComparison.OrdinalIgnoreCase))
                progress(0.95, Loc.S("NpuDecodingVideo"));
            var m = StepPattern().Match(line);
            if (sampling && m.Success && int.Parse(m.Groups[2].Value) == steps)
            {
                var step = int.Parse(m.Groups[1].Value);
                progress(0.05 + 0.85 * step / steps, Loc.F("ComfyStep", step.ToString(), steps.ToString()));
            }
            else if (!sampling && line.Contains("ltxv qnn: loaded", StringComparison.Ordinal))
            {
                progress(0.02, Loc.S("NpuLoadingVideo"));
            }
        }
        var readers = new[] { Pump(process.StandardOutput, OnLine), Pump(process.StandardError, OnLine) };
        using (ct.Register(() =>
               {
                   try
                   {
                       process.Kill(true);
                   }
                   catch (InvalidOperationException)
                   {
                   }
               }))
        {
            await process.WaitForExitAsync(CancellationToken.None);
            await Task.WhenAll(readers);
        }
        ct.ThrowIfCancellationRequested();
        string text;
        lock (tail)
            text = tail.ToString();
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"sd-cli exited with {process.ExitCode}: {text[^Math.Min(text.Length, 800)..]}");
        return text;
    }

    /// <summary>Reads a stream and splits it on both \n and \r (sd-cli redraws its progress bar with \r).</summary>
    private static Task Pump(StreamReader reader, Action<string> onLine) => Task.Run(async () =>
    {
        var buffer = new char[4096];
        var line = new StringBuilder();
        int read;
        while ((read = await reader.ReadAsync(buffer)) > 0)
        {
            for (var i = 0; i < read; i++)
            {
                var c = buffer[i];
                if (c is '\n' or '\r')
                {
                    if (line.Length > 0)
                        onLine(line.ToString());
                    line.Clear();
                }
                else
                {
                    line.Append(c);
                }
            }
        }
        if (line.Length > 0)
            onLine(line.ToString());
    });

    [GeneratedRegex(@"\|\s*(\d+)/(\d+)\s")]
    private static partial Regex StepPattern();

    private void SetStatus(string text) => UiThread.Post(() => Status = text);
}
