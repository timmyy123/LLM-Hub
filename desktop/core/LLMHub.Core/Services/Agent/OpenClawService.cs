using System.Diagnostics;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services.Agent;

/// <summary>
/// The OpenClaw agent engine for the Agent tab. OpenClaw (MIT, openclaw.ai) is installed by the app: a private Node runtime
/// from nodejs.org (checksum verified), then a pinned openclaw from npm with every extension and skill of Chinese origin
/// deleted. Its Gateway runs warm on a loopback port, pointed only at LLM Hub's Agent bridge, with channels, Bonjour,
/// update checks and other plugins off. Each session talks to it through <c>openclaw acp</c> (the Agent Client Protocol,
/// JSON-RPC over stdio).
/// </summary>
public sealed partial class OpenClawService : ObservableObject
{
    public const string Version = "2026.10.7";
    private const string NodeVersion = "24.22.1";

    /// <summary>
    /// Removed after install. In the npm package: alibaba (DashScope/Wan), minimax, kie (Kling, Hailuo, Seedance, Wan),
    /// senseaudio. The rest are separate @openclaw packages that are never installed, deleted too should a version ship them.
    /// </summary>
    private static readonly string[] StrippedExtensions =
    [
        "alibaba", "minimax", "kie", "senseaudio", "byteplus", "volcengine", "feishu", "deepseek", "kimi-coding", "moonshot", "longcat",
        "qianfan", "qwen", "stepfun", "tencent", "xiaomi", "zai", "pixverse", "fish-audio-speech", "novita", "vydra",
    ];

    /// <summary>sherpa-onnx (k2-fsa, Next-gen Kaldi) text to speech.</summary>
    private static readonly string[] StrippedSkills = ["sherpa-onnx-tts"];

    private readonly SemaphoreSlim _gate = new(1, 1);
    private Process? _gateway;
    private int _port;
    private string _token = "";

    public OpenClawService()
    {
        Status = "";
    }

    [ObservableProperty] public partial string Status { get; set; }
    [ObservableProperty] public partial bool IsInstalling { get; set; }

    public static string Root => Path.Combine(AppPaths.Engines, "openclaw");
    private static string NodeDir => Path.Combine(Root, "node");
    public static string NodeExe => OperatingSystem.IsWindows() ? Path.Combine(NodeDir, "node.exe") : Path.Combine(NodeDir, "bin", "node");
    private static string NpmCli => OperatingSystem.IsWindows()
        ? Path.Combine(NodeDir, "node_modules", "npm", "bin", "npm-cli.js")
        : Path.Combine(NodeDir, "lib", "node_modules", "npm", "bin", "npm-cli.js");
    private static string AppDir => Path.Combine(Root, "app");
    private static string Package => Path.Combine(AppDir, "node_modules", "openclaw");
    public static string Entry => Path.Combine(Package, "openclaw.mjs");
    /// <summary>OPENCLAW_HOME: its config, sessions (SQLite) and workspaces.</summary>
    private static string Home => Path.Combine(Root, "home");
    private static string ConfigFile => Path.Combine(Home, ".openclaw", "openclaw.json");
    private static string TokenFile => Path.Combine(Root, "gateway.token");
    private static string MarkerFile => Path.Combine(Root, "installed.txt");

    public static bool IsInstalled => File.Exists(MarkerFile) && File.Exists(NodeExe) && File.Exists(Entry);

    /// <summary>Node needs win-x64/arm64, linux-x64/arm64 or darwin-arm64; OpenClaw's native modules ship for all of them.</summary>
    public static bool IsSupported => OperatingSystem.IsWindows() || OperatingSystem.IsLinux() || OperatingSystem.IsMacOS();

    // ---------------------------------------------------------------- install

    public async Task InstallAsync(CancellationToken ct)
    {
        if (IsInstalling)
            return;
        IsInstalling = true;
        try
        {
            Directory.CreateDirectory(Root);
            await InstallNodeAsync(ct);
            SetStatus(Loc.S("OpenClawInstallingPackage"));
            Directory.CreateDirectory(AppDir);
            var cache = Path.Combine(Root, "npm-cache");
            var (code, output) = await RunAsync(NodeExe, [NpmCli, "install", "--prefix", AppDir, $"openclaw@{Version}", "--omit=dev", "--no-audit", "--no-fund",
                "--cache", cache, "--loglevel", "error"], AppDir, ct);
            TryDelete(cache);
            if (code != 0 || !File.Exists(Entry))
                throw new InvalidOperationException($"npm install openclaw@{Version} failed ({code}): {output[^Math.Min(output.Length, 600)..]}");
            Strip();
            await File.WriteAllTextAsync(MarkerFile, $"openclaw {Version}, node {NodeVersion}", ct);
            SetStatus("");
            Log.Info($"OpenClaw {Version} installed with Node {NodeVersion}");
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Error("OpenClaw install failed", e);
            SetStatus(Loc.F("ErrorPrefix", e.Message));
            throw;
        }
        finally
        {
            IsInstalling = false;
        }
    }

    private async Task InstallNodeAsync(CancellationToken ct)
    {
        if (File.Exists(NodeExe))
            return;
        var arch = Ioc.Hardware.IsArm64 ? "arm64" : "x64";
        var name = OperatingSystem.IsWindows() ? $"node-v{NodeVersion}-win-{arch}.zip"
            : OperatingSystem.IsMacOS() ? $"node-v{NodeVersion}-darwin-{arch}.tar.gz"
            : $"node-v{NodeVersion}-linux-{arch}.tar.gz";
        var baseUrl = $"https://nodejs.org/dist/v{NodeVersion}/";
        using var http = new HttpClient();
        var sums = await http.GetStringAsync(baseUrl + "SHASUMS256.txt", ct);
        var expected = sums.Split('\n').Select(l => l.Trim().Split("  ")).FirstOrDefault(p => p.Length == 2 && p[1] == name)?[0]
                       ?? throw new InvalidOperationException($"{name} is not listed in Node's checksums");
        var archive = Path.Combine(Root, name);
        await Fetch.FileAsync(baseUrl + name, archive, p => SetStatus(Loc.F("OpenClawDownloadingNode", p.ToString("F0"))), ct);
        string actual;
        await using (var stream = File.OpenRead(archive))
            actual = Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, ct));
        if (actual != expected)
        {
            File.Delete(archive);
            throw new InvalidOperationException($"{name} checksum mismatch");
        }
        var unpacked = Path.Combine(Root, "node-unpacked");
        TryDelete(unpacked);
        Fetch.Extract(archive, unpacked);
        File.Delete(archive);
        TryDelete(NodeDir);
        Directory.Move(Directory.EnumerateDirectories(unpacked).Single(), NodeDir);
        Directory.Delete(unpacked);
    }

    /// <summary>Deletes the Chinese-origin extensions and skills from the installed package.</summary>
    private static void Strip()
    {
        var removed = new List<string>();
        foreach (var root in new[] { Path.Combine(Package, "dist", "extensions"), Path.Combine(Package, "extensions") })
        {
            foreach (var id in StrippedExtensions)
            {
                var dir = Path.Combine(root, id);
                if (Directory.Exists(dir))
                {
                    TryDelete(dir);
                    removed.Add(id);
                }
            }
        }
        foreach (var root in new[] { Path.Combine(Package, "skills"), Path.Combine(Package, "dist", "skills") })
        {
            foreach (var id in StrippedSkills)
            {
                var dir = Path.Combine(root, id);
                if (Directory.Exists(dir))
                {
                    TryDelete(dir);
                    removed.Add(id);
                }
            }
        }
        Log.Info($"OpenClaw: removed {string.Join(", ", removed.Distinct())}");
    }

    public void Remove()
    {
        StopGateway();
        TryDelete(Root);
    }

    // ---------------------------------------------------------------- config and gateway

    /// <summary>
    /// openclaw.json: one provider, LLM Hub's Agent bridge, offering the installed chat models; one agent per session with
    /// the session's folder as its workspace and the session's model. Plugins other than memory are off; commands ask first.
    /// The Gateway reloads this file when it changes.
    /// </summary>
    private void WriteConfig(IEnumerable<AgentSession> sessions)
    {
        var models = new JsonArray();
        foreach (var m in Ioc.Models.ChatModels)
        {
            models.Add(new JsonObject
            {
                ["id"] = LocalApiServer.ApiName(m),
                ["name"] = m.Name,
                ["reasoning"] = m.Tags.Contains("thinking"),
                ["input"] = new JsonArray("text"),
                ["cost"] = new JsonObject { ["input"] = 0, ["output"] = 0, ["cacheRead"] = 0, ["cacheWrite"] = 0 },
                ["contextWindow"] = AppSettings.Current.ApiContextLength,
                ["maxTokens"] = 4096,
            });
        }
        var entries = new JsonObject();
        foreach (var session in sessions)
            entries[AgentId(session)] = new JsonObject { ["workspace"] = session.Folder };
        var primary = Ioc.Models.ChatModels.FirstOrDefault(m => m.Id == sessions.LastOrDefault()?.Info.ModelId) ?? Ioc.Models.ChatModels.FirstOrDefault();
        var config = new JsonObject
        {
            ["gateway"] = new JsonObject
            {
                ["mode"] = "local",
                ["bind"] = "loopback",
                ["port"] = _port,
                ["auth"] = new JsonObject { ["mode"] = "token", ["token"] = _token },
            },
            ["update"] = new JsonObject { ["checkOnStart"] = false },
            ["agents"] = new JsonObject
            {
                ["defaults"] = new JsonObject
                {
                    ["model"] = new JsonObject { ["primary"] = primary is null ? "llmhub/none" : "llmhub/" + LocalApiServer.ApiName(primary) },
                    ["skipBootstrap"] = true,
                    ["workspace"] = Path.Combine(Root, "workspace"),
                },
                ["entries"] = entries,
            },
            ["models"] = new JsonObject
            {
                ["mode"] = "replace",
                ["providers"] = new JsonObject
                {
                    ["llmhub"] = new JsonObject
                    {
                        ["baseUrl"] = Ioc.AgentBridge.BaseUrl + "/v1",
                        ["apiKey"] = Ioc.AgentBridge.Key,
                        ["api"] = "openai-completions",
                        ["timeoutSeconds"] = 600,
                        ["models"] = models,
                    },
                },
            },
            ["tools"] = new JsonObject { ["exec"] = new JsonObject { ["mode"] = "ask" } },
            ["plugins"] = new JsonObject { ["allow"] = new JsonArray("memory-core") },
        };
        Directory.CreateDirectory(Path.GetDirectoryName(ConfigFile)!);
        Directory.CreateDirectory(Path.Combine(Root, "workspace"));
        var json = config.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
        if (!File.Exists(ConfigFile) || File.ReadAllText(ConfigFile) != json)
            File.WriteAllText(ConfigFile, json);
    }

    internal static string AgentId(AgentSession session) => "s-" + session.Id.Replace("-", "")[..12].ToLowerInvariant();

    /// <summary>Starts the Gateway once (it takes 15 to 20 seconds) and keeps it running while the app is open.</summary>
    internal async Task EnsureGatewayAsync(IEnumerable<AgentSession> sessions, CancellationToken ct)
    {
        await _gate.WaitAsync(ct);
        try
        {
            if (!IsInstalled)
                throw new InvalidOperationException(Loc.S("OpenClawNotInstalled"));
            await Ioc.AgentBridge.EnsureBridgeAsync();
            if (_gateway is { HasExited: false })
            {
                WriteConfig(sessions);
                return;
            }
            _port = FreePort();
            _token = Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(24));
            await File.WriteAllTextAsync(TokenFile, _token, ct);
            WriteConfig(sessions);
            SetStatus(Loc.S("OpenClawStarting"));
            var clock = Stopwatch.StartNew();
            for (var attempt = 0; attempt < 2; attempt++)
            {
                var info = Node(["gateway"], Root);
                var tail = new StringBuilder();
                var process = Process.Start(info)!;
                process.OutputDataReceived += (_, e) => Remember(tail, e.Data);
                process.ErrorDataReceived += (_, e) => Remember(tail, e.Data);
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                Ioc.Engines.Llama.Track(process);
                _gateway = process;
                var deadline = DateTime.UtcNow.AddMinutes(2);
                while (DateTime.UtcNow < deadline && !process.HasExited && !await PortOpenAsync(_port, ct))
                    await Task.Delay(300, ct);
                if (!process.HasExited && await PortOpenAsync(_port, ct))
                {
                    SetStatus("");
                    Log.Info($"OpenClaw gateway ready on {_port} in {clock.Elapsed.TotalSeconds:F0}s");
                    return;
                }
                // 78 is a config error: OpenClaw's doctor repairs it, then one more try.
                if (process.HasExited && process.ExitCode == 78 && attempt == 0)
                {
                    Log.Info($"OpenClaw gateway config error, running doctor: {Tail(tail)}");
                    await RunAsync(NodeExe, [Entry, "doctor", "--fix", "--yes", "--non-interactive"], Root, ct, ChildEnvironment());
                    continue;
                }
                if (!process.HasExited)
                    process.Kill(true);
                throw new InvalidOperationException($"OpenClaw gateway did not start: {Tail(tail)}");
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    internal ProcessStartInfo Acp(string folder) =>
        Node(["acp", "--url", $"ws://127.0.0.1:{_port}", "--token-file", TokenFile], folder);

    private ProcessStartInfo Node(IEnumerable<string> args, string workingDirectory)
    {
        var info = new ProcessStartInfo(NodeExe)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            WorkingDirectory = workingDirectory,
        };
        info.ArgumentList.Add(Entry);
        foreach (var arg in args)
            info.ArgumentList.Add(arg);
        foreach (var (key, value) in ChildEnvironment())
            info.Environment[key] = value;
        return info;
    }

    /// <summary>Keeps OpenClaw local and quiet: its own home, no update or telemetry checks, no channels, no Bonjour.</summary>
    private static Dictionary<string, string> ChildEnvironment() => new()
    {
        ["OPENCLAW_HOME"] = Home,
        ["OPENCLAW_CONFIG_PATH"] = ConfigFile,
        ["OPENCLAW_NO_AUTO_UPDATE"] = "1",
        ["OPENCLAW_OFFLINE"] = "1",
        ["OPENCLAW_DISABLE_BONJOUR"] = "1",
        ["OPENCLAW_SKIP_CHANNELS"] = "1",
        ["OPENCLAW_NO_RESPAWN"] = "1",
        ["OPENCLAW_EXEC_SHELL_SNAPSHOT"] = "0",
        ["OPENCLAW_HIDE_BANNER"] = "1",
        ["OPENCLAW_SUPPRESS_NOTES"] = "1",
        ["NO_COLOR"] = "1",
        ["PATH"] = Path.GetDirectoryName(NodeExe) + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH"),
    };

    public void StopGateway()
    {
        if (_gateway is { HasExited: false } gateway)
            gateway.Kill(entireProcessTree: true);
        _gateway = null;
    }

    // ---------------------------------------------------------------- helpers

    private static async Task<(int Code, string Output)> RunAsync(string exe, IEnumerable<string> args, string folder, CancellationToken ct, Dictionary<string, string>? env = null)
    {
        var info = new ProcessStartInfo(exe)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = folder,
        };
        foreach (var arg in args)
            info.ArgumentList.Add(arg);
        info.Environment["PATH"] = Path.GetDirectoryName(NodeExe) + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH");
        foreach (var (key, value) in env ?? [])
            info.Environment[key] = value;
        using var process = Process.Start(info)!;
        var stdout = process.StandardOutput.ReadToEndAsync(ct);
        var stderr = process.StandardError.ReadToEndAsync(ct);
        await process.WaitForExitAsync(ct);
        return (process.ExitCode, await stdout + await stderr);
    }

    private static async Task<bool> PortOpenAsync(int port, CancellationToken ct)
    {
        try
        {
            using var client = new TcpClient();
            await client.ConnectAsync("127.0.0.1", port, ct);
            return true;
        }
        catch (SocketException)
        {
            return false;
        }
    }

    private static int FreePort()
    {
        var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        var port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static void Remember(StringBuilder tail, string? line)
    {
        if (line is null)
            return;
        lock (tail)
        {
            tail.AppendLine(line);
            if (tail.Length > 6000)
                tail.Remove(0, tail.Length - 6000);
        }
    }

    private static string Tail(StringBuilder tail)
    {
        lock (tail)
            return tail.ToString()[^Math.Min(tail.Length, 1200)..];
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (Directory.Exists(path))
                Directory.Delete(path, true);
            else if (File.Exists(path))
                File.Delete(path);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Log.Error($"Could not delete {path}", e);
        }
    }

    private void SetStatus(string text) => UiThread.Post(() => Status = text);
}
