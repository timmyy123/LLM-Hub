using System.Diagnostics;
using System.IO.Compression;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services;

public enum EngineState
{
    Idle,
    Installing,
    Loading,
    Ready,
    Error,
}

/// <summary>
/// Runs models with the official prebuilt llama.cpp release (llama-server), downloaded on first use.
/// No CUDA Toolkit is needed: the CUDA build ships with its runtime DLLs and only needs the NVIDIA driver.
/// </summary>
public sealed partial class LlamaEngine : ObservableObject
{
    private const string ReleasesApi = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20";

    private readonly HttpClient _http = new() { Timeout = Timeout.InfiniteTimeSpan };
    private readonly JobObject? _job = OperatingSystem.IsWindows() ? new JobObject() : null;
    private static readonly string ServerExe = OperatingSystem.IsWindows() ? "llama-server.exe" : "llama-server";
    private readonly StringBuilder _stderrTail = new();
    private readonly SemaphoreSlim _gate = new(1, 1);
    private Process? _process;

    private static string PidFile => Path.Combine(EngineRoot, "server.pid");

    public LlamaEngine()
    {
        if (!OperatingSystem.IsWindows())
            KillStaleServer();
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("LLMHub-Desktop/0.1");
        StatusText = Loc.S("EngineIdle");
    }

    [ObservableProperty] public partial EngineState State { get; set; }
    [ObservableProperty] public partial string StatusText { get; set; }
    [ObservableProperty] public partial InstalledModel? LoadedModel { get; set; }

    public string? BaseUrl { get; private set; }
    /// <summary>The llama.cpp build currently running, e.g. win-cuda-13.4-x64.</summary>
    public string? Variant { get; private set; }

    private static string EngineRoot => Path.Combine(AppPaths.Engines, "llama.cpp");
    private static string WorkingFile => Path.Combine(EngineRoot, "working.txt");

    private sealed record InstalledEngine(string Tag, string Variant, string ServerPath);

    private sealed class EngineStartException(string message) : Exception(message);

    /// <summary>
    /// llama.cpp release builds to try on this PC, best first. One ARM64 app covers every Windows on Arm PC:
    /// Snapdragon gets the Adreno OpenCL build, NVIDIA gets CUDA, everything falls back to CPU.
    /// </summary>
    private static List<string> PreferredVariants()
    {
        var hw = Ioc.Hardware;
        var list = new List<string>();
#if DEBUG
        // Testing: force one llama.cpp build, e.g. --engine-variant win-sycl-x64 to try the Intel GPU on a laptop that also has an NVIDIA GPU.
        if (DebugArg("--engine-variant") is { } forced)
            return [forced];
#endif
        if (OperatingSystem.IsLinux())
        {
            // Release builds named llama-<tag>-bin-ubuntu-*.tar.gz.
            var arch = hw.IsArm64 ? "arm64" : "x64";
            if (hw.HasNvidiaGpu)
                list.Add($"ubuntu-cuda-13.4-{arch}");
            if (hw.HasAmdGpu && !hw.IsArm64)
                list.Add("ubuntu-rocm-10.0-x64");
            if (hw.HasIntelGpu && !hw.HasNvidiaGpu && !hw.HasAmdGpu && !hw.IsArm64)
                list.Add("ubuntu-sycl-fp16-x64");
            list.Add($"ubuntu-vulkan-{arch}");
            list.Add($"ubuntu-{arch}");
        }
        else if (hw.IsArm64)
        {
            if (hw.HasAdrenoGpu)
                list.Add("win-opencl-adreno-arm64");
            if (hw.HasNvidiaGpu)
                list.Add("win-cuda-13.4-arm64");
            list.Add("win-cpu-arm64");
            list.Add("win-vulkan-arm64");
        }
        else
        {
            // Vendor builds first (CUDA, ROCm for Radeon, SYCL for Intel Arc/iGPU), then Vulkan, which runs on all three.
            // SYCL measured on a 14th-gen Intel iGPU: 28.6 tok/s vs 23.9 on Vulkan with driver 101.7092; older drivers
            // (101.6129) crash it, and the fallback below then moves on to Vulkan. ROCm is not yet verified on AMD hardware.
            if (hw.HasNvidiaGpu)
                list.Add("win-cuda-13.4-x64");
            if (hw.HasAmdGpu)
                list.Add("win-rocm-10.0-x64");
            if (hw.HasIntelGpu && !hw.HasNvidiaGpu && !hw.HasAmdGpu)
                list.Add("win-sycl-x64");
            list.Add("win-vulkan-x64");
            list.Add("win-cpu-x64");
        }

        // Start from the build that last worked here, so a failed GPU backend is not retried on every load.
        var working = File.Exists(WorkingFile) ? File.ReadAllText(WorkingFile).Trim() : null;
        var index = working is null ? -1 : list.IndexOf(working);
        return index > 0 ? list.Skip(index).ToList() : list;
    }

    /// <summary>Context window of the running server; a request for a larger one reloads the model.</summary>
    public int LoadedContext { get; private set; }

    public static int ContextFor(InstalledModel model, int wanted) => Math.Min(model.ContextLength > 0 ? model.ContextLength : 8192, wanted);

    public async Task LoadAsync(InstalledModel model, CancellationToken ct, int? contextLength = null)
    {
        var context = ContextFor(model, contextLength ?? AppSettings.Current.ContextLength);
        await _gate.WaitAsync(ct);
        try
        {
            if (LoadedModel?.Id == model.Id && _process is { HasExited: false } && LoadedContext >= context)
                return;

            StopProcess();
            Exception? last = null;
            foreach (var variant in PreferredVariants())
            {
                var engine = await EnsureInstalledAsync(variant, ct);
                if (engine is null)
                    continue;
                try
                {
                    SetState(EngineState.Loading, Loc.F("LoadingModel", model.Name));
                    await StartServerAsync(engine, model, context, ct);
                    LoadedContext = context;
                    await File.WriteAllTextAsync(WorkingFile, variant, ct);
                    Variant = variant;
                    UiThread.Run(() => LoadedModel = model);
                    SetState(EngineState.Ready, Loc.F("EngineReady", model.Name));
                    return;
                }
                catch (EngineStartException e)
                {
                    Log.Error($"llama.cpp {variant} could not run {model.Id}: {e.Message}");
                    StopProcess();
                    last = e;
                }
            }
            throw last ?? new InvalidOperationException("No llama.cpp build found for this PC");
        }
        catch (OperationCanceledException)
        {
            StopProcess();
            SetState(EngineState.Idle, Loc.S("EngineIdle"));
            throw;
        }
        catch (Exception e)
        {
            Log.Error("Engine load failed", e);
            StopProcess();
            SetState(EngineState.Error, Loc.F("EngineError", e.Message));
            throw;
        }
        finally
        {
            _gate.Release();
        }
    }

    public void Stop()
    {
        StopProcess();
        SetState(EngineState.Idle, Loc.S("EngineIdle"));
    }

    /// <summary>Without job objects, a crash can leave llama-server running; the PID file lets the next launch stop it.</summary>
    private static void KillStaleServer()
    {
        try
        {
            if (File.Exists(PidFile) && int.TryParse(File.ReadAllText(PidFile), out var pid))
            {
                using var stale = Process.GetProcessById(pid);
                if (stale.ProcessName.StartsWith("llama-server", StringComparison.Ordinal))
                    stale.Kill();
            }
        }
        catch
        {
            // Already gone.
        }
        try { File.Delete(PidFile); } catch (IOException) { }
    }

    /// <summary>
    /// Windows ties llama-server to the app with a job object. Elsewhere a tiny shell loop does it: if the app dies without
    /// cleaning up (crash, force quit), the loop stops the server instead of leaving the model in memory. It exits on its
    /// own when the server stops first, and checks the process name so a reused PID is never touched.
    /// </summary>
    private static void StartWatchdog(int serverPid, string name, string? pidFile)
    {
        var cleanup = pidFile is null ? "" : $" rm -f '{pidFile}';";
        var script = $"while kill -0 {Environment.ProcessId} 2>/dev/null && kill -0 {serverPid} 2>/dev/null; do sleep 2; done; " +
                     $"kill -0 {Environment.ProcessId} 2>/dev/null || {{ ps -p {serverPid} -o comm= | grep -q {name[..Math.Min(name.Length, 15)]} && kill {serverPid};{cleanup} }}";
        try
        {
            using var _ = Process.Start(new ProcessStartInfo("/bin/sh") { ArgumentList = { "-c", script }, UseShellExecute = false });
        }
        catch (Exception e)
        {
            Log.Error("llama-server watchdog did not start", e);
        }
    }

    /// <summary>Ties another helper process (whisper-server, ComfyUI) to the app's lifetime, like llama-server.</summary>
    public void Track(Process process)
    {
        _job?.Add(process);
        if (!OperatingSystem.IsWindows())
            StartWatchdog(process.Id, Path.GetFileName(process.StartInfo.FileName), null);
    }

    private void StopProcess()
    {
        var process = _process;
        _process = null;
        try
        {
            if (process is { HasExited: false })
                process.Kill(entireProcessTree: true);
        }
        catch (Exception e)
        {
            Log.Error("Stopping llama-server failed", e);
        }
        process?.Dispose();
        BaseUrl = null;
        UiThread.Run(() => LoadedModel = null);
    }

    private void SetState(EngineState state, string text) => UiThread.Run(() =>
    {
        State = state;
        StatusText = text;
    });

    /// <summary>Returns the installed build for a variant, downloading the newest release that has it. Null if no release ships it.</summary>
    private async Task<InstalledEngine?> EnsureInstalledAsync(string variant, CancellationToken ct)
    {
        Directory.CreateDirectory(EngineRoot);
        var existing = Directory.EnumerateDirectories(EngineRoot, $"*-{variant}")
            .OrderByDescending(d => d, StringComparer.Ordinal)
            .Select(d => (Dir: d, Server: Directory.EnumerateFiles(d, ServerExe, SearchOption.AllDirectories).FirstOrDefault()))
            .FirstOrDefault(x => x.Server is not null);
        if (existing.Server is not null)
            return new InstalledEngine(Path.GetFileName(existing.Dir).Split('-')[0], variant, existing.Server);

        SetState(EngineState.Installing, Loc.S("InstallingEngine"));
        _releases ??= JsonDocument.Parse(await _http.GetStringAsync(ReleasesApi, ct));

        foreach (var release in _releases.RootElement.EnumerateArray())
        {
            var tag = release.GetProperty("tag_name").GetString() ?? "";
            if (!tag.StartsWith('b'))
                continue; // Build tags look like b11836; skip the vX.Y.Z summary releases.
            var assets = release.GetProperty("assets").EnumerateArray()
                .Select(a => (Name: a.GetProperty("name").GetString() ?? "", Url: a.GetProperty("browser_download_url").GetString() ?? "", Size: a.GetProperty("size").GetInt64()))
                .ToList();
            var main = assets.FirstOrDefault(a => a.Name == $"llama-{tag}-bin-{variant}.zip" || a.Name == $"llama-{tag}-bin-{variant}.tar.gz");
            if (string.IsNullOrEmpty(main.Url))
                continue;

            var downloads = new List<(string Name, string Url, long Size)> { main };
            if (variant.Contains("cuda"))
            {
                // The CUDA runtime ships separately: cudart-llama-bin-win-cuda-*.zip, cudart-llama-<tag>-bin-ubuntu-cuda-*.tar.gz.
                var cudart = assets.FirstOrDefault(a => a.Name.StartsWith("cudart-llama", StringComparison.Ordinal) && a.Name.Contains($"-bin-{variant}."));
                if (!string.IsNullOrEmpty(cudart.Url))
                    downloads.Add(cudart);
            }

            var target = Path.Combine(EngineRoot, $"{tag}-{variant}");
            await DownloadAndExtractAsync(downloads, target, ct);
            var server = Directory.EnumerateFiles(target, ServerExe, SearchOption.AllDirectories).FirstOrDefault()
                         ?? throw new FileNotFoundException($"{ServerExe} not found in release");
            if (!OperatingSystem.IsWindows())
                File.SetUnixFileMode(server, File.GetUnixFileMode(server) | UnixFileMode.UserExecute | UnixFileMode.GroupExecute);
            Log.Info($"Installed llama.cpp {tag} ({variant})");
            return new InstalledEngine(tag, variant, server);
        }

        Log.Info($"No llama.cpp release ships {variant}");
        return null;
    }

    private JsonDocument? _releases;

    private async Task DownloadAndExtractAsync(List<(string Name, string Url, long Size)> assets, string target, CancellationToken ct)
    {
        Directory.CreateDirectory(target);
        var total = assets.Sum(a => a.Size);
        long done = 0;
        foreach (var asset in assets)
        {
            var zip = Path.Combine(EngineRoot, asset.Name);
            using (var response = await _http.GetAsync(asset.Url, HttpCompletionOption.ResponseHeadersRead, ct))
            {
                response.EnsureSuccessStatusCode();
                await using var input = await response.Content.ReadAsStreamAsync(ct);
                await using var output = File.Create(zip);
                var buffer = new byte[1 << 20];
                int read;
                var clock = Stopwatch.StartNew();
                while ((read = await input.ReadAsync(buffer, ct)) > 0)
                {
                    await output.WriteAsync(buffer.AsMemory(0, read), ct);
                    done += read;
                    if (clock.ElapsedMilliseconds > 300)
                    {
                        SetState(EngineState.Installing, Loc.F("DownloadingEngine", (int)(done * 100 / Math.Max(total, 1))));
                        clock.Restart();
                    }
                }
            }
            await Task.Run(() =>
            {
                if (zip.EndsWith(".zip", StringComparison.Ordinal))
                {
                    ZipFile.ExtractToDirectory(zip, target, overwriteFiles: true);
                    return;
                }
                using var file = File.OpenRead(zip);
                using var gz = new GZipStream(file, CompressionMode.Decompress);
                System.Formats.Tar.TarFile.ExtractToDirectory(gz, target, overwriteFiles: true);
            }, ct);
            File.Delete(zip);
        }
    }

    private async Task StartServerAsync(InstalledEngine engine, InstalledModel model, int context, CancellationToken ct)
    {
#if DEBUG
        // Lets the UI tests check that a failing GPU backend falls back to the next build.
        if (Environment.GetEnvironmentVariable("LLMHUB_FAIL_VARIANT") == engine.Variant)
            throw new EngineStartException($"forced failure of {engine.Variant} (LLMHUB_FAIL_VARIANT)");
#endif
        var port = FreePort();
        var args = new List<string>
        {
            "-m", model.PrimaryPath,
            "--host", "127.0.0.1",
            "--port", port.ToString(),
            "--alias", model.Id,
            "-ngl", "999",
            "-c", context.ToString(),
            "--jinja",
        };
#if DEBUG
        // Testing: pick the GPU llama.cpp uses, e.g. --engine-device Vulkan1 (list them with llama-server --list-devices).
        if (DebugArg("--engine-device") is { } device)
        {
            args.Add("--device");
            args.Add(device);
        }
#endif
        if (model.MmprojPath is { } mmproj && File.Exists(mmproj))
        {
            args.Add("--mmproj");
            args.Add(mmproj);
        }

        var info = new ProcessStartInfo(engine.ServerPath)
        {
            WorkingDirectory = Path.GetDirectoryName(engine.ServerPath)!,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
        };
        foreach (var arg in args)
            info.ArgumentList.Add(arg);
        if (OperatingSystem.IsLinux())
        {
            var libDirs = Directory.EnumerateFiles(Path.GetDirectoryName(Path.GetDirectoryName(engine.ServerPath)!)!, "*.so*", SearchOption.AllDirectories)
                .Select(Path.GetDirectoryName).Distinct();
            info.Environment["LD_LIBRARY_PATH"] = string.Join(':', libDirs.Append(Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? ""));
        }

        lock (_stderrTail)
            _stderrTail.Clear();
        var process = new Process { StartInfo = info, EnableRaisingEvents = true };
        process.ErrorDataReceived += (_, e) => Remember(e.Data);
        process.OutputDataReceived += (_, e) => Remember(e.Data);
        process.Exited += (_, _) =>
        {
            if (ReferenceEquals(_process, process))
            {
                Log.Error($"llama-server exited with {process.ExitCode}: {Tail()}");
                SetState(EngineState.Error, Loc.F("EngineError", $"exit {process.ExitCode}"));
                UiThread.Run(() => LoadedModel = null);
            }
        };
        process.Start();
        _job?.Add(process);
        if (!OperatingSystem.IsWindows())
        {
            await File.WriteAllTextAsync(PidFile, process.Id.ToString(), ct);
            StartWatchdog(process.Id, "llama-server", PidFile);
        }
        process.BeginErrorReadLine();
        process.BeginOutputReadLine();
        _process = process;
        Log.Info($"Started llama-server ({engine.Tag} {engine.Variant}) on {port} for {model.Id}");

        var baseUrl = $"http://127.0.0.1:{port}";
        using var health = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
        var deadline = DateTime.UtcNow.AddMinutes(10);
        while (DateTime.UtcNow < deadline)
        {
            ct.ThrowIfCancellationRequested();
            if (process.HasExited)
                throw new EngineStartException(LastLine() ?? $"llama-server exited ({process.ExitCode})");
            try
            {
                using var response = await health.GetAsync($"{baseUrl}/health", ct);
                if (response.StatusCode == HttpStatusCode.OK)
                {
                    BaseUrl = baseUrl;
                    return;
                }
            }
            catch (HttpRequestException)
            {
                // Not listening yet.
            }
            catch (TaskCanceledException) when (!ct.IsCancellationRequested)
            {
                // Health probe timed out while the model loads.
            }
            await Task.Delay(400, ct);
        }
        throw new TimeoutException("Model took too long to load");
    }

    private void Remember(string? line)
    {
        if (line is null)
            return;
        lock (_stderrTail)
        {
            _stderrTail.AppendLine(line);
            if (_stderrTail.Length > 8000)
                _stderrTail.Remove(0, _stderrTail.Length - 8000);
        }
    }

    private string Tail()
    {
        lock (_stderrTail)
            return _stderrTail.ToString();
    }

    private string? LastLine() =>
        Tail().Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).LastOrDefault();

#if DEBUG
    private static string? DebugArg(string name)
    {
        var args = Environment.GetCommandLineArgs();
        var i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }
#endif

    private static int FreePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }
}
