using System.Diagnostics;
using System.Net.Http.Headers;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services.Studio;

/// <summary>
/// Studio's image and video engine: ComfyUI, installed by the app (standalone Python from python-build-standalone,
/// ComfyUI's release source, PyTorch from pytorch.org) so the user never sets up Python, then run hidden on a loopback
/// port and driven through its HTTP and WebSocket API. Outputs are copied to Pictures/LLM Hub.
/// </summary>
public sealed partial class ComfyService : ObservableObject
{
    private const string PythonTag = "3.12.";
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromMinutes(10) };
    private readonly SemaphoreSlim _gate = new(1, 1);
    private Process? _process;
    private string? _url;

    public ComfyService()
    {
        // GitHub's API refuses requests without a User-Agent.
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("LLMHub-Desktop/0.1");
        Status = "";
        StatusLine = Loc.S("ComfyNotInstalled");
    }

    /// <summary>Install or download progress, or the last error.</summary>
    [ObservableProperty] public partial string Status { get; set; }
    /// <summary>"ComfyUI connected · 127.0.0.1:8188" in the Studio header.</summary>
    [ObservableProperty] public partial string StatusLine { get; set; }
    [ObservableProperty] public partial bool IsRunning { get; set; }
    [ObservableProperty] public partial bool IsBusy { get; set; }
    [ObservableProperty] public partial double Progress { get; set; }

    public static string Root => Path.Combine(AppPaths.Engines, "comfyui");
    private static string DataDir => Path.Combine(Root, "data");
    public static string ModelsDir => Path.Combine(DataDir, "models");
    public static string Gallery => Directory.CreateDirectory(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures) is { Length: > 0 } pictures
        ? pictures : AppPaths.Root, "LLM Hub")).FullName;
    private static string PythonExe => OperatingSystem.IsWindows() ? Path.Combine(Root, "python", "python.exe") : Path.Combine(Root, "python", "bin", "python3");
    private static string MarkerFile => Path.Combine(Root, "installed.txt");

    public bool IsInstalled => File.Exists(MarkerFile);
    public string? Url => _url;

    /// <summary>ComfyUI has no GPU backend for Snapdragon (Windows on Arm).</summary>
    public static bool IsSupported => !(OperatingSystem.IsWindows() && Ioc.Hardware.IsArm64);

    // ---------------------------------------------------------------- install

    public async Task EnsureRunningAsync(CancellationToken ct)
    {
        await _gate.WaitAsync(ct);
        try
        {
            if (_process is { HasExited: false } && _url is not null)
                return;
            if (!IsSupported)
                throw new PlatformNotSupportedException(Loc.S("ComfyUnsupported"));
            if (!IsInstalled)
                await InstallAsync(ct);
            await StartAsync(ct);
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task InstallAsync(CancellationToken ct)
    {
        Directory.CreateDirectory(Root);
        var hw = Ioc.Hardware;

        // 1. A standalone Python (relocatable CPython builds that ComfyUI's own desktop app also uses).
        if (!File.Exists(PythonExe))
        {
            var triple = OperatingSystem.IsWindows()
                ? hw.IsArm64 ? "aarch64-pc-windows-msvc" : "x86_64-pc-windows-msvc"
                : hw.IsArm64 ? "aarch64-unknown-linux-gnu" : "x86_64-unknown-linux-gnu";
            var asset = await Fetch.GitHubAssetAsync("astral-sh/python-build-standalone",
                n => n.StartsWith("cpython-" + PythonTag, StringComparison.Ordinal) && n.EndsWith($"-{triple}-install_only.tar.gz", StringComparison.Ordinal), ct, scan: 3)
                ?? throw new InvalidOperationException("No standalone Python for this computer");
            var archive = Path.Combine(Root, "python.tar.gz");
            await Fetch.FileAsync(asset.Url, archive, p => SetStatus(Loc.F("ComfyStepPython", p.ToString("F0"))), ct);
            Fetch.Extract(archive, Root);
            File.Delete(archive);
        }

        // 2. ComfyUI's latest release source.
        var source = Path.Combine(Root, "ComfyUI");
        if (!File.Exists(Path.Combine(source, "main.py")))
        {
            using var doc = JsonDocument.Parse(await _http.GetStringAsync("https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest", ct));
            var tag = doc.RootElement.GetProperty("tag_name").GetString()!;
            var zip = Path.Combine(Root, "comfyui.zip");
            await Fetch.FileAsync($"https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/{tag}.zip", zip, p => SetStatus(Loc.F("ComfyStepSource", p.ToString("F0"))), ct);
            var unpacked = Path.Combine(Root, "src");
            Fetch.Extract(zip, unpacked);
            File.Delete(zip);
            if (Directory.Exists(source))
                Directory.Delete(source, true);
            Directory.Move(Directory.EnumerateDirectories(unpacked).Single(), source);
            Directory.Delete(unpacked);
            Log.Info($"ComfyUI {tag} unpacked");
        }

        // 3. PyTorch for this GPU, then ComfyUI's requirements.
        var index = hw.HasNvidiaGpu ? "https://download.pytorch.org/whl/cu130"
            : hw.HasIntelGpu && !hw.HasAmdGpu ? "https://download.pytorch.org/whl/xpu"
            : OperatingSystem.IsLinux() && hw.HasAmdGpu ? "https://download.pytorch.org/whl/rocm6.4"
            : "https://download.pytorch.org/whl/cpu";
        SetStatus(Loc.S("ComfyStepTorch"));
        await PipAsync(ct, "install", "--upgrade", "pip");
        await PipAsync(ct, "install", "torch", "torchvision", "torchaudio", "--index-url", index);
        SetStatus(Loc.S("ComfyStepRequirements"));
        await PipAsync(ct, "install", "-r", Path.Combine(source, "requirements.txt"));
        await File.WriteAllTextAsync(MarkerFile, index, ct);
        SetStatus("");
        Log.Info($"ComfyUI installed with torch from {index}");
    }

    private async Task PipAsync(CancellationToken ct, params string[] args)
    {
        var info = new ProcessStartInfo(PythonExe) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var arg in new[] { "-m", "pip" }.Concat(args).Append("--disable-pip-version-check").Append("--no-input"))
            info.ArgumentList.Add(arg);
        using var process = Process.Start(info)!;
        var tail = new StringBuilder();
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data is { } line && (line.StartsWith("Collecting", StringComparison.Ordinal) || line.StartsWith("Downloading", StringComparison.Ordinal)))
                SetStatus(line.Trim()[..Math.Min(line.Trim().Length, 90)]);
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is { } line)
                lock (tail)
                    tail.AppendLine(line);
        };
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        await process.WaitForExitAsync(ct);
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"pip {string.Join(' ', args.Take(3))} failed: {tail.ToString()[^Math.Min(tail.Length, 600)..]}");
    }

    private async Task StartAsync(CancellationToken ct)
    {
        // --base-directory expects its folders to exist (ComfyUI scans custom_nodes before creating anything).
        foreach (var folder in new[] { "models", "custom_nodes", "input", "output", "temp", "user" })
            Directory.CreateDirectory(Path.Combine(DataDir, folder));
        var port = FreePort();
        var info = new ProcessStartInfo(PythonExe)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = Path.Combine(Root, "ComfyUI"),
        };
        foreach (var arg in new[] { "-s", "main.py", "--listen", "127.0.0.1", "--port", port.ToString(), "--base-directory", DataDir,
                     "--disable-auto-launch", "--preview-method", "none" })
            info.ArgumentList.Add(arg);
        if (!Ioc.Hardware.HasNvidiaGpu && !Ioc.Hardware.HasIntelGpu && !OperatingSystem.IsMacOS())
            info.ArgumentList.Add("--cpu");
        info.Environment["PYTHONUNBUFFERED"] = "1";
        var tail = new StringBuilder();
        _process = Process.Start(info)!;
        _process.OutputDataReceived += (_, e) => Remember(tail, e.Data);
        _process.ErrorDataReceived += (_, e) => Remember(tail, e.Data);
        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();
        Ioc.Engines.Llama.Track(_process);
        SetStatus(Loc.S("ComfyStarting"));

        var url = $"http://127.0.0.1:{port}";
        var deadline = DateTime.UtcNow.AddMinutes(4);
        while (DateTime.UtcNow < deadline)
        {
            if (_process.HasExited)
                throw new InvalidOperationException($"ComfyUI exited with {_process.ExitCode}: {Tail(tail)}");
            try
            {
                using var response = await _http.GetAsync(url + "/system_stats", ct);
                if (response.IsSuccessStatusCode)
                {
                    _url = url;
                    var stats = JsonNode.Parse(await response.Content.ReadAsStringAsync(ct));
                    var device = stats?["devices"]?[0]?["name"]?.GetValue<string>() ?? "";
                    Log.Info($"ComfyUI ready on {port} ({device})");
                    UiThread.Run(() =>
                    {
                        IsRunning = true;
                        StatusLine = Loc.F("ComfyConnected", $"127.0.0.1:{port}");
                        Status = "";
                    });
                    return;
                }
            }
            catch (HttpRequestException)
            {
            }
            await Task.Delay(500, ct);
        }
        throw new TimeoutException("ComfyUI did not start: " + Tail(tail));
    }

    private static void Remember(StringBuilder tail, string? line)
    {
        if (line is null)
            return;
        lock (tail)
        {
            tail.AppendLine(line);
            if (tail.Length > 8000)
                tail.Remove(0, tail.Length - 8000);
        }
    }

    private static string Tail(StringBuilder tail)
    {
        lock (tail)
            return tail.ToString()[^Math.Min(tail.Length, 1200)..];
    }

    public void Stop()
    {
        if (_process is { HasExited: false })
            _process.Kill(true);
        _process = null;
        _url = null;
        IsRunning = false;
    }

    // ---------------------------------------------------------------- models

    public static bool HasModels(ComfyWorkflow workflow) => workflow.Files.All(f => File.Exists(Path.Combine(ModelsDir, f.Folder, f.Name)));

    public async Task EnsureModelsAsync(ComfyWorkflow workflow, CancellationToken ct)
    {
        var total = (double)workflow.DownloadSize;
        long done = 0;
        foreach (var file in workflow.Files)
        {
            var path = Path.Combine(ModelsDir, file.Folder, file.Name);
            if (!File.Exists(path))
            {
                var before = done;
                await Fetch.FileAsync(file.Url, path, p => SetStatus(Loc.F("ComfyDownloadingModel", file.Name, ((before + file.Size * p / 100) * 100 / total).ToString("F0"))), ct);
            }
            done += file.Size;
        }
        SetStatus("");
    }

    // ---------------------------------------------------------------- run

    /// <summary>Uploads an input image (for video and upscale) and returns the name ComfyUI knows it by.</summary>
    public async Task<string> UploadAsync(string path, CancellationToken ct)
    {
        using var form = new MultipartFormDataContent();
        var content = new ByteArrayContent(await File.ReadAllBytesAsync(path, ct));
        content.Headers.ContentType = new MediaTypeHeaderValue("image/" + Path.GetExtension(path).TrimStart('.').ToLowerInvariant().Replace("jpg", "jpeg"));
        form.Add(content, "image", Path.GetFileName(path));
        form.Add(new StringContent("true"), "overwrite");
        using var response = await _http.PostAsync(_url + "/upload/image", form, ct);
        response.EnsureSuccessStatusCode();
        return JsonNode.Parse(await response.Content.ReadAsStringAsync(ct))!["name"]!.GetValue<string>();
    }

    /// <summary>Queues a graph, reports progress (0..1), and returns the output files copied into the gallery.</summary>
    public async Task<List<string>> RunAsync(JsonObject graph, Action<double, string> progress, CancellationToken ct)
    {
        var clientId = Guid.NewGuid().ToString("N");
        using var socket = new ClientWebSocket();
        await socket.ConnectAsync(new Uri(_url!.Replace("http://", "ws://") + $"/ws?clientId={clientId}"), ct);

        using var queued = await _http.PostAsync(_url + "/prompt",
            new StringContent(new JsonObject { ["prompt"] = graph, ["client_id"] = clientId }.ToJsonString(), Encoding.UTF8, "application/json"), ct);
        var body = await queued.Content.ReadAsStringAsync(ct);
        if (!queued.IsSuccessStatusCode)
            throw new InvalidOperationException(DescribeError(body));
        var promptId = JsonNode.Parse(body)!["prompt_id"]!.GetValue<string>();
        using var registration = ct.Register(() => _ = _http.PostAsync(_url + "/interrupt", null));

        var buffer = new byte[64 * 1024];
        while (socket.State == WebSocketState.Open)
        {
            var message = new StringBuilder();
            WebSocketReceiveResult result;
            do
            {
                result = await socket.ReceiveAsync(buffer, ct);
                if (result.MessageType == WebSocketMessageType.Text)
                    message.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
            }
            while (!result.EndOfMessage);
            if (result.MessageType != WebSocketMessageType.Text)
                continue;
            var e = JsonNode.Parse(message.ToString());
            var data = e?["data"];
            if (data?["prompt_id"]?.GetValue<string>() is { } id && id != promptId)
                continue;
            switch (e?["type"]?.GetValue<string>())
            {
                case "progress":
                    var value = data!["value"]!.GetValue<double>();
                    var max = Math.Max(1, data["max"]!.GetValue<double>());
                    progress(value / max, Loc.F("ComfyStep", value.ToString("F0"), max.ToString("F0")));
                    break;
                case "executing" when (data?["node"]) is null:
                    return await CollectAsync(promptId, ct);
                case "execution_error":
                    throw new InvalidOperationException(data?["exception_message"]?.GetValue<string>() ?? "ComfyUI error");
                case "execution_interrupted":
                    throw new OperationCanceledException();
            }
        }
        return await CollectAsync(promptId, ct);
    }

    private async Task<List<string>> CollectAsync(string promptId, CancellationToken ct)
    {
        var history = JsonNode.Parse(await _http.GetStringAsync($"{_url}/history/{promptId}", ct))?[promptId]?["outputs"] as JsonObject;
        var saved = new List<string>();
        foreach (var (_, output) in history ?? [])
        {
            foreach (var key in new[] { "images", "gifs", "videos" })
            {
                foreach (var file in output?[key]?.AsArray() ?? [])
                {
                    var name = file?["filename"]?.GetValue<string>() ?? "";
                    var subfolder = file?["subfolder"]?.GetValue<string>() ?? "";
                    var type = file?["type"]?.GetValue<string>() ?? "output";
                    if (type != "output")
                        continue;
                    var bytes = await _http.GetByteArrayAsync($"{_url}/view?filename={Uri.EscapeDataString(name)}&subfolder={Uri.EscapeDataString(subfolder)}&type={type}", ct);
                    var target = Path.Combine(Gallery, $"{DateTime.Now:yyyyMMdd-HHmmss}-{name.Replace('/', '_')}");
                    await File.WriteAllBytesAsync(target, bytes, ct);
                    saved.Add(target);
                }
            }
        }
        Log.Info($"ComfyUI {promptId}: {saved.Count} output(s)");
        return saved;
    }

    private static string DescribeError(string body)
    {
        try
        {
            var node = JsonNode.Parse(body);
            var errors = node?["node_errors"] as JsonObject;
            var first = errors?.FirstOrDefault().Value?["errors"]?[0];
            return first?["details"]?.GetValue<string>() is { Length: > 0 } details
                ? $"{first["message"]?.GetValue<string>()}: {details}"
                : node?["error"]?["message"]?.GetValue<string>() ?? body;
        }
        catch (JsonException)
        {
            return body;
        }
    }

    private void SetStatus(string text) => UiThread.Post(() => Status = text);

    private static int FreePort()
    {
        var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        var port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }
}
