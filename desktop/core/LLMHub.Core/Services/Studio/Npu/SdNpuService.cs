using System.Diagnostics;
using System.IO.Compression;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json.Nodes;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services.Studio.Npu;

/// <summary>
/// A Stable Diffusion 1.5 checkpoint compiled for the Hexagon NPU: the Android app's QNN package (UNet, VAE and
/// tokenizer, built with QNN 2.28 for Hexagon v73) plus the checkpoint's own CLIP text encoder, taken from its
/// author's Hugging Face repository (the whole diffusers file, or just the text encoder's byte range of a single-file
/// checkpoint). Lykon's models only: no checkpoints from Chinese creators or with Chinese components (the owner's rule).
/// </summary>
public sealed record SdNpuModel(string Id, string Name, string Package, long PackageSize, string TextEncoderUrl, long TextEncoderSize, bool TextEncoderInCheckpoint)
{
    /// <summary>xororz/sd-qnn pinned to a commit; "8gen2" is the Hexagon v73 build the Android app also runs on newer chips.</summary>
    private const string Repo = "https://huggingface.co/xororz/sd-qnn/resolve/64d16039e0c3a71d3c5db1faccc090faae93cb28/";

    public string PackageUrl => Repo + Package;
    public long DownloadSize => PackageSize + TextEncoderSize;

    public static readonly SdNpuModel AbsoluteReality = new("absolutereality", "AbsoluteReality 1.8.1", "AbsoluteReality_qnn2.28_8gen2.zip", 1_054_661_172,
        "https://huggingface.co/Lykon/absolute-reality-1.81/resolve/569d201654dca1426f1e090fca3953acc7db2c30/text_encoder/model.fp16.safetensors", 246_144_152, false);

    public static readonly SdNpuModel DreamShaper = new("dreamshaper8", "DreamShaper 8", "DreamShaperV8_qnn2.28_8gen2.zip", 1_032_290_626,
        "https://huggingface.co/Lykon/dreamshaper-8/resolve/a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917/text_encoder/model.fp16.safetensors", 246_144_152, false);

    public static readonly SdNpuModel NeverEndingDream = new("neverendingdream", "NeverEnding Dream 1.22", "NeverEndingDreamV122_qnn2.28_8gen2.zip", 1_031_689_930,
        "https://huggingface.co/Lykon/NeverEnding-Dream/resolve/239d0482dc703082d1b2b1a7b6051790ecd6d28c/NeverEndingDream_1.22_NoVae_fp16_pruned.safetensors", 246_121_114, true);

    public static IReadOnlyList<SdNpuModel> All { get; } = [AbsoluteReality, DreamShaper, NeverEndingDream];
}

/// <summary>
/// Studio's image engine on Snapdragon (Windows on Arm), where ComfyUI has no GPU backend: Stable Diffusion 1.5 on the
/// Hexagon NPU (<see cref="SdQnnPipeline"/>). Downloads models on first use, keeps one pipeline loaded and saves PNGs to
/// the Studio gallery. Used by Studio and by the Local API's /v1/images/generations.
/// </summary>
public sealed partial class SdNpuService : ObservableObject
{
    private static readonly HttpClient Http = CreateClient();
    private readonly SemaphoreSlim _gate = new(1, 1);
    private SdQnnPipeline? _pipeline;
    private SdNpuModel? _loaded;

    public SdNpuService()
    {
        Status = "";
        StatusLine = IsSupported ? Loc.F("NpuReady", NpuLabel) : "";
    }

    [ObservableProperty] public partial string Status { get; set; }
    [ObservableProperty] public partial string StatusLine { get; set; }
    [ObservableProperty] public partial bool IsLoaded { get; set; }

    /// <summary>Windows on Arm with a Qualcomm Hexagon NPU and the QNN runtime next to the app.</summary>
    public static bool IsSupported => QnnRuntime.IsAvailable && Ioc.Hardware.NpuName.Contains("Hexagon", StringComparison.OrdinalIgnoreCase);

    /// <summary>"Snapdragon NPU (Hexagon v73)".</summary>
    public static string NpuLabel => $"Snapdragon NPU (Hexagon v{QnnRuntime.HexagonArch(Ioc.Hardware.NpuName)})";

    public static string Root => Path.Combine(AppPaths.Models, "studio-npu");
    public static string ModelDir(SdNpuModel model) => Path.Combine(Root, model.Id);
    private static readonly string[] PackageFiles = ["unet.bin", "vae_decoder.bin", "vae_encoder.bin", "tokenizer.json"];

    public static bool IsInstalled(SdNpuModel model) =>
        PackageFiles.Append("text_encoder.safetensors").All(f => File.Exists(Path.Combine(ModelDir(model), f)));

    private static HttpClient CreateClient()
    {
        var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("LLMHub-Desktop/0.1");
        return http;
    }

    // ---------------------------------------------------------------- download

    public async Task EnsureModelAsync(SdNpuModel model, CancellationToken ct)
    {
        if (IsInstalled(model))
            return;
        var dir = Directory.CreateDirectory(ModelDir(model)).FullName;
        double total = model.DownloadSize;
        if (!PackageFiles.All(f => File.Exists(Path.Combine(dir, f))))
        {
            var zip = Path.Combine(dir, "package.zip");
            await Fetch.FileAsync(model.PackageUrl, zip, p => SetStatus(Loc.F("ComfyDownloadingModel", model.Name, (p * model.PackageSize / total).ToString("F0"))), ct);
            SetStatus(Loc.F("NpuUnpacking", model.Name));
            await Task.Run(() => ExtractPackage(zip, dir), ct);
            File.Delete(zip);
        }
        var encoder = Path.Combine(dir, "text_encoder.safetensors");
        if (!File.Exists(encoder))
        {
            void Report(double p) => SetStatus(Loc.F("ComfyDownloadingModel", model.Name, ((model.PackageSize + p * model.TextEncoderSize / 100) * 100 / total).ToString("F0")));
            if (model.TextEncoderInCheckpoint)
                await DownloadTextEncoderAsync(model.TextEncoderUrl, encoder, Report, ct);
            else
                await Fetch.FileAsync(model.TextEncoderUrl, encoder, Report, ct);
        }
        SetStatus("");
        Log.Info($"SD NPU model {model.Id} installed in {dir}");
    }

    /// <summary>Keeps only the files the NPU pipeline uses from the Android package (it also carries an MNN CLIP and resolution patches).</summary>
    private static void ExtractPackage(string zip, string dir)
    {
        using var archive = System.IO.Compression.ZipFile.OpenRead(zip);
        foreach (var name in PackageFiles)
        {
            var entry = archive.Entries.FirstOrDefault(e => e.Name == name) ?? throw new InvalidDataException($"{name} missing from {Path.GetFileName(zip)}");
            entry.ExtractToFile(Path.Combine(dir, name), overwrite: true);
        }
    }

    /// <summary>
    /// Downloads only the text encoder's tensors of a single-file checkpoint (they are stored together) with HTTP range
    /// requests, and writes them as a small safetensors file.
    /// </summary>
    private static async Task DownloadTextEncoderAsync(string url, string target, Action<double> progress, CancellationToken ct)
    {
        var lengthBytes = await RangeAsync(url, 0, 7, ct);
        var headerLength = BitConverter.ToInt64(lengthBytes);
        var header = SafeTensors.ParseHeader(await RangeAsync(url, 8, 8 + headerLength - 1, ct));
        var tensors = header.Where(e => e.Name.StartsWith("cond_stage_model.transformer.", StringComparison.Ordinal)).ToList();
        if (tensors.Count == 0)
            throw new InvalidDataException("No text encoder in the checkpoint");
        var begin = tensors.Min(e => e.Begin);
        var end = tensors.Max(e => e.End);
        var dataStart = 8 + headerLength;

        var part = target + ".data.part";
        var have = File.Exists(part) ? new FileInfo(part).Length : 0;
        if (have < end - begin)
        {
            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            request.Headers.Range = new RangeHeaderValue(dataStart + begin + have, dataStart + end - 1);
            using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
            if (response.StatusCode != System.Net.HttpStatusCode.PartialContent)
                throw new InvalidOperationException($"Range request failed: {(int)response.StatusCode}");
            await using var output = new FileStream(part, FileMode.Append, FileAccess.Write, FileShare.None, 1 << 20);
            await using var input = await response.Content.ReadAsStreamAsync(ct);
            var buffer = new byte[1 << 20];
            int read;
            var last = DateTime.MinValue;
            while ((read = await input.ReadAsync(buffer, ct)) > 0)
            {
                await output.WriteAsync(buffer.AsMemory(0, read), ct);
                have += read;
                if ((DateTime.UtcNow - last).TotalMilliseconds > 250)
                {
                    progress(have * 100.0 / (end - begin));
                    last = DateTime.UtcNow;
                }
            }
        }

        var json = new JsonObject();
        foreach (var e in tensors)
        {
            json[e.Name] = new JsonObject
            {
                ["dtype"] = e.DType,
                ["shape"] = new JsonArray(e.Shape.Select(d => (JsonNode)d).ToArray()),
                ["data_offsets"] = new JsonArray(e.Begin - begin, e.End - begin),
            };
        }
        var headerBytes = Encoding.UTF8.GetBytes(json.ToJsonString());
        var padded = (headerBytes.Length + 7) / 8 * 8;
        var tmp = target + ".tmp";
        await using (var output = File.Create(tmp))
        {
            await output.WriteAsync(BitConverter.GetBytes((long)padded), ct);
            await output.WriteAsync(headerBytes, ct);
            for (var i = headerBytes.Length; i < padded; i++)
                output.WriteByte((byte)' ');
            await using var data = File.OpenRead(part);
            await data.CopyToAsync(output, ct);
        }
        File.Move(tmp, target, overwrite: true);
        File.Delete(part);
        progress(100);
    }

    private static async Task<byte[]> RangeAsync(string url, long from, long to, CancellationToken ct)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.Range = new RangeHeaderValue(from, to);
        using var response = await Http.SendAsync(request, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsByteArrayAsync(ct);
    }

    // ---------------------------------------------------------------- generate

    /// <summary>
    /// Generates <paramref name="count"/> images (seed, seed + 1, ...) and saves them to the gallery. <paramref name="initImage"/>
    /// (a picture file) makes it image to image. <paramref name="progress"/> gets (0..1, status text).
    /// </summary>
    public async Task<List<string>> GenerateAsync(SdNpuModel model, string prompt, string negative, int steps, float cfg, long seed, int count,
        string? initImage, float strength, Action<double, string> progress, CancellationToken ct)
    {
        await _gate.WaitAsync(ct);
        try
        {
            await EnsureModelAsync(model, ct);
            byte[]? init = null;
            if (initImage is not null)
                init = await (ImageLoader ?? throw new NotSupportedException("No image decoder"))(initImage, SdQnnPipeline.Size, SdQnnPipeline.Size);
            return await Task.Factory.StartNew(() =>
            {
                var pipeline = Load(model);
                var saved = new List<string>();
                for (var n = 0; n < count; n++)
                {
                    var request = new SdRequest(prompt, negative, steps, cfg, seed + n, init, strength);
                    var (rgb, timings) = pipeline.Generate(request, (step, total) =>
                        progress((n + step / (double)total) / count, Loc.F("ComfyStep", step.ToString(), total.ToString())), ct);
                    var file = Path.Combine(ComfyService.Gallery, $"{DateTime.Now:yyyyMMdd-HHmmss}-npu-{model.Id}-{seed + n}.png");
                    var parameters = $"{prompt}\nNegative prompt: {negative}\nSteps: {steps}, Sampler: DPM++ 2M Karras, CFG scale: {cfg}, Seed: {seed + n}, Size: 512x512, Model: {model.Name}"
                                     + (init is null ? "" : $", Denoising strength: {strength}");
                    PngWriter.Write(file, rgb, SdQnnPipeline.Size, SdQnnPipeline.Size, parameters);
                    Log.Info($"SD NPU {model.Id} seed {seed + n}{(init is null ? "" : $" img2img {strength}")}: {timings}");
                    saved.Add(file);
                }
                return saved;
            }, ct, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }
        finally
        {
            _gate.Release();
        }
    }

    private SdQnnPipeline Load(SdNpuModel model)
    {
        if (_pipeline is not null && _loaded == model)
            return _pipeline;
        Unload();
        SetStatus(Loc.F("NpuLoading", model.Name));
        var clock = Stopwatch.StartNew();
        _pipeline = new SdQnnPipeline(ModelDir(model));
        _loaded = model;
        SetStatus("");
        UiThread.Post(() =>
        {
            IsLoaded = true;
            StatusLine = Loc.F("NpuLoaded", model.Name, NpuLabel);
        });
        Log.Info($"SD NPU pipeline {model.Id} ready in {clock.Elapsed.TotalSeconds:F1}s");
        return _pipeline;
    }

    /// <summary>Frees the NPU and CPU memory of the loaded model.</summary>
    public void Unload()
    {
        _pipeline?.Dispose();
        _pipeline = null;
        _loaded = null;
        UiThread.Post(() =>
        {
            IsLoaded = false;
            StatusLine = Loc.F("NpuReady", NpuLabel);
        });
    }

    /// <summary>Unloads (if idle) from outside a generation, e.g. Settings > Studio > Stop.</summary>
    public async Task StopAsync()
    {
        await _gate.WaitAsync();
        try
        {
            Unload();
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Decodes a picture file to width x height RGB (scaled to cover, centre crop); set by the platform app.</summary>
    public static Func<string, int, int, Task<byte[]>>? ImageLoader { get; set; }

    private void SetStatus(string text) => UiThread.Post(() => Status = text);
}
