using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Channels;

namespace LLMHub.Services;

/// <summary>
/// Runs .litertlm models in-process through LiteRT-LM's C API (litert-lm.dll), taken from Google's official
/// litert-lm-api wheel on PyPI on first use. GPU first (DirectX 12 on Windows, Vulkan/OpenCL on Linux), CPU fallback.
/// Windows x64 and Linux x64/ARM64; Google ships no Windows ARM64 build.
/// </summary>
public sealed class LiteRtEngine
{
    private const string WheelIndex = "https://pypi.org/pypi/litert-lm-api/json";

    private readonly HttpClient _http = new() { Timeout = Timeout.InfiniteTimeSpan };
    private readonly SemaphoreSlim _gate = new(1, 1);
    private Api? _api;
    private IntPtr _engine;
    private IntPtr _conversation;

    public static bool IsSupported =>
        OperatingSystem.IsLinux() || (OperatingSystem.IsWindows() && RuntimeInformation.ProcessArchitecture == Architecture.X64);

    private static string LibName => OperatingSystem.IsWindows() ? "litert-lm.dll" : "liblitert-lm.so";

    private static string WheelTag => OperatingSystem.IsWindows() ? "win_amd64.whl"
        : RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "manylinux_2_27_aarch64.whl" : "manylinux_2_27_x86_64.whl";

    public InstalledModel? LoadedModel { get; private set; }
    /// <summary>Whether the loaded engine was created with multi-token prediction (speculative decoding).</summary>
    public bool LoadedWithMtp { get; private set; }
    public string? Backend { get; private set; }

    private static string Root => Path.Combine(AppPaths.Engines, "litert-lm");

    public async Task LoadAsync(InstalledModel model, Action<string> status, CancellationToken ct)
    {
        if (!IsSupported)
            throw new PlatformNotSupportedException(Loc.S("LiteRtNoArm"));
        await _gate.WaitAsync(ct);
        try
        {
            if (LoadedModel?.Id == model.Id && _engine != IntPtr.Zero && LoadedWithMtp == model.MtpEnabled)
                return;
            Unload();
            _api ??= Api.Load(await EnsureInstalledAsync(status, ct));
            status(Loc.F("LoadingModel", model.Name));

            var cacheDir = Directory.CreateDirectory(Path.Combine(model.Folder, "cache")).FullName;
            var maxTokens = Math.Min(Math.Max(model.ContextLength, 4096), AppSettings.Current.ContextLength);
            if (model.Format == "litertlm" && model.HasMtpDrafter is null)
            {
                model.HasMtpDrafter = _api.FileHasMtp(model.PrimaryPath);
                Ioc.Models.Save(model);
                Log.Info($"LiteRT-LM {model.Id}: file {(model.HasMtpDrafter == true ? "has" : "has no")} MTP drafter");
            }
            var fileHasMtp = model.MtpEnabled;
            foreach (var backend in Backends())
            {
                // MTP only helps (and is only supported) on the GPU, as in the mobile apps.
                var mtp = fileHasMtp && backend == "gpu";
                if (backend == "gpu" && !await GpuSurvivesAsync(model, cacheDir, maxTokens, mtp, status, ct))
                    continue;
                var engine = await Task.Run(() => _api.CreateEngine(model.PrimaryPath, backend, cacheDir, maxTokens, mtp), ct);
                if (engine == IntPtr.Zero)
                {
                    Log.Error($"LiteRT-LM could not start {model.Id} on {backend}");
                    continue;
                }
                _engine = engine;
                Backend = backend;
                LoadedModel = model;
                LoadedWithMtp = model.MtpEnabled;
                Log.Info($"LiteRT-LM loaded {model.Id} on {backend}{(mtp ? " with MTP" : "")}");
                return;
            }
            // GPU-only files (labelled "LiteRT GPU") have no CPU section, so a failed GPU leaves nothing to fall back to.
            throw new InvalidOperationException(model.Quant.Contains("GPU", StringComparison.Ordinal) ? Loc.S("LiteRtNeedsGpu") : "LiteRT-LM could not load this model");
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>GPU first, then CPU. LLMHUB_LITERT_BACKEND=cpu|gpu pins one (for testing and for machines whose GPU driver crashes).</summary>
    private static string[] Backends() =>
        Environment.GetEnvironmentVariable("LLMHUB_LITERT_BACKEND") is { Length: > 0 } pinned ? [pinned] : ["gpu", "cpu"];

    /// <summary>
    /// On Linux a broken or missing GPU driver makes LiteRT-LM's GPU backend crash the whole process (seen under WSL)
    /// instead of returning an error. So the first GPU load of each model runs in a throwaway child process
    /// (<see cref="RunProbe"/>) and the result is remembered. Windows drivers report errors normally.
    /// </summary>
    private async Task<bool> GpuSurvivesAsync(InstalledModel model, string cacheDir, int maxTokens, bool mtp, Action<string> status, CancellationToken ct)
    {
        if (!OperatingSystem.IsLinux() || Environment.ProcessPath is not { } exe)
            return true;
        var results = Path.Combine(Root, "gpu-probe.txt");
        var key = $"{_api!.LibraryPath}|{model.Id}|{(mtp ? "mtp" : "plain")}=";
        var known = File.Exists(results) ? File.ReadLines(results).LastOrDefault(l => l.StartsWith(key, StringComparison.Ordinal)) : null;
        if (known is not null)
            return known.EndsWith("ok", StringComparison.Ordinal);

        status(Loc.F("LoadingModel", model.Name));
        var info = new System.Diagnostics.ProcessStartInfo(exe) { UseShellExecute = false, RedirectStandardError = true, RedirectStandardOutput = true };
        // Started as "dotnet llmhub.dll" (development) rather than through the app host.
        if (Path.GetFileNameWithoutExtension(exe) == "dotnet" && System.Reflection.Assembly.GetEntryAssembly()?.Location is { Length: > 0 } entry)
            info.ArgumentList.Add(entry);
        foreach (var arg in new[] { "--litert-probe", _api.LibraryPath, model.PrimaryPath, cacheDir, maxTokens.ToString(), mtp ? "1" : "0" })
            info.ArgumentList.Add(arg);
        using var probe = System.Diagnostics.Process.Start(info)!;
        _ = probe.StandardOutput.ReadToEndAsync(ct);
        _ = probe.StandardError.ReadToEndAsync(ct);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromMinutes(5));
        try
        {
            await probe.WaitForExitAsync(timeout.Token);
        }
        catch (OperationCanceledException)
        {
            probe.Kill(true);
            ct.ThrowIfCancellationRequested();
        }
        var ok = probe.HasExited && probe.ExitCode == 0;
        Log.Info($"LiteRT-LM GPU probe for {model.Id}: {(ok ? "ok" : probe.HasExited ? $"exit {probe.ExitCode}" : "timed out")}");
        File.AppendAllText(results, key + (ok ? "ok" : "failed") + "\n");
        return ok;
    }

    /// <summary>
    /// Entry point for the GPU probe child process: <c>--litert-probe lib model cacheDir maxTokens mtp</c>.
    /// Exit 0 when the engine starts and streams a few tokens.
    /// </summary>
    public static int RunProbe(string[] args)
    {
        var probe = new LiteRtEngine { _api = Api.Load(args[0]) };
        probe._engine = probe._api.CreateEngine(args[1], "gpu", args[2], int.Parse(args[3]), args[4] == "1");
        if (probe._engine == IntPtr.Zero)
            return 2;
        return Task.Run(async () =>
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(3));
            var pieces = 0;
            await foreach (var delta in probe.StreamAsync([new ChatTurn("user", "Say hello.")], cts.Token))
                if (delta.Content is { Length: > 0 } && ++pieces >= 3)
                    break;
            probe.Unload();
            return pieces > 0 ? 0 : 3;
        }).GetAwaiter().GetResult();
    }

    public void Unload()
    {
        if (_api is null)
            return;
        if (_conversation != IntPtr.Zero)
            _api.ConversationDelete(_conversation);
        if (_engine != IntPtr.Zero)
            _api.EngineDelete(_engine);
        _conversation = IntPtr.Zero;
        _engine = IntPtr.Zero;
        LoadedModel = null;
        Backend = null;
    }

    /// <summary>Streams a reply. History (all turns but the last) seeds a fresh conversation; the last user turn is sent.</summary>
    public async IAsyncEnumerable<ChatDelta> StreamAsync(IReadOnlyList<ChatTurn> turns, [EnumeratorCancellation] CancellationToken ct)
    {
        if (_api is null || _engine == IntPtr.Zero)
            throw new InvalidOperationException("No LiteRT-LM model loaded");

        if (_conversation != IntPtr.Zero)
            _api.ConversationDelete(_conversation);
        var history = new JsonArray(turns.Take(turns.Count - 1).Select(t => (JsonNode)Message(t.Role, t.Content)).ToArray());
        _conversation = _api.CreateConversation(_engine, history.ToJsonString());
        if (_conversation == IntPtr.Zero)
            throw new InvalidOperationException("LiteRT-LM could not start a conversation");

        var channel = Channel.CreateUnbounded<(string? Text, string? Error, bool Final)>();
        var handle = GCHandle.Alloc(channel.Writer);
        DateTime? firstToken = null;
        var chunks = 0;
        try
        {
            var last = turns[^1];
            var rc = _api.SendStream(_conversation, Message(last.Role, last.Content).ToJsonString(), GCHandle.ToIntPtr(handle));
            if (rc != 0)
                throw new InvalidOperationException("LiteRT-LM send failed");

            using var registration = ct.Register(() => _api.ConversationCancel(_conversation));
            await foreach (var (text, error, final) in channel.Reader.ReadAllAsync(CancellationToken.None))
            {
                if (error is not null)
                {
                    if (error.Contains("CANCELLED") || error.Contains("Max number of tokens"))
                        break;
                    throw new InvalidOperationException(error);
                }
                if (chunks < 3 && text is not null)
                    Log.Info($"LiteRT-LM chunk: {text[..Math.Min(text.Length, 400)]}");
                if (TextOf(text) is { Length: > 0 } piece)
                {
                    firstToken ??= DateTime.UtcNow;
                    chunks++;
                    yield return new ChatDelta(piece, null);
                }
                if (final)
                    break;
            }
            ct.ThrowIfCancellationRequested();
            // What the user feels: decoded tokens over the wall time from first to last token.
            // (The engine's own decode rate counts steps, and with MTP a step yields several tokens.)
            var wall = firstToken is { } t ? (DateTime.UtcNow - t).TotalSeconds : 0;
            var (tokens, engineRate) = _api.LastDecode(_conversation);
            Log.Info($"LiteRT-LM decode: {tokens} tokens, {chunks} chunks, {wall:F2}s wall, engine rate {engineRate:F1} tok/s");
            // A reply that arrives in one or two chunks has no meaningful wall time; use the engine's figure then.
            if (chunks >= 3 && tokens > 1 && wall > 0)
                yield return new ChatDelta(null, null, (tokens - 1) / wall);
            else if (engineRate > 0)
                yield return new ChatDelta(null, null, engineRate);
        }
        finally
        {
            handle.Free();
        }
    }

    private static JsonObject Message(string role, string text) => new()
    {
        ["role"] = role,
        ["content"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = text }),
    };

    /// <summary>Stream chunks are JSON messages; plain text is accepted too.</summary>
    private static string TextOf(string? chunk)
    {
        if (string.IsNullOrEmpty(chunk))
            return "";
        try
        {
            var node = JsonNode.Parse(chunk);
            var content = node?["content"];
            if (content is JsonValue v)
                return v.GetValue<string>();
            return string.Concat((content as JsonArray ?? []).Select(c => c?["text"]?.GetValue<string>() ?? ""));
        }
        catch (JsonException)
        {
            return chunk;
        }
    }

    private async Task<string> EnsureInstalledAsync(Action<string> status, CancellationToken ct)
    {
        var existing = Directory.Exists(Root)
            ? Directory.EnumerateFiles(Root, LibName, SearchOption.AllDirectories).OrderByDescending(p => p, StringComparer.Ordinal).FirstOrDefault()
            : null;
        if (existing is not null)
            return existing;

        status(Loc.S("InstallingLiteRt"));
        using var index = JsonDocument.Parse(await _http.GetStringAsync(WheelIndex, ct));
        var version = index.RootElement.GetProperty("info").GetProperty("version").GetString()!;
        var wheel = index.RootElement.GetProperty("urls").EnumerateArray()
            .First(u => u.GetProperty("filename").GetString()!.EndsWith(WheelTag, StringComparison.Ordinal));
        var target = Path.Combine(Root, version);
        Directory.CreateDirectory(target);
        var file = Path.Combine(Root, wheel.GetProperty("filename").GetString()!);
        await using (var input = await _http.GetStreamAsync(wheel.GetProperty("url").GetString(), ct))
        await using (var output = File.Create(file))
            await input.CopyToAsync(output, ct);

        // A wheel is a zip; keep only the native libraries (the engine, plus DirectX shader compilers on Windows).
        using (var zip = ZipFile.OpenRead(file))
        {
            foreach (var entry in zip.Entries.Where(e => e.FullName.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) || e.FullName.Contains(".so")))
            {
                var dest = Path.Combine(target, entry.FullName.Replace('/', Path.DirectorySeparatorChar));
                Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
                entry.ExtractToFile(dest, overwrite: true);
            }
        }
        File.Delete(file);
        Log.Info($"Installed LiteRT-LM {version}");
        return Directory.EnumerateFiles(target, LibName, SearchOption.AllDirectories).First();
    }

    /// <summary>The subset of LiteRT-LM's C API (litert_lm/c/engine.h) that LLM Hub uses.</summary>
    private sealed unsafe class Api
    {
        public static Api? Current;

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
        private static void OnChunk(IntPtr userData, IntPtr chunk)
        {
            var writer = (ChannelWriter<(string?, string?, bool)>)GCHandle.FromIntPtr(userData).Target!;
            var api = Current!;
            var error = Marshal.PtrToStringUTF8(api.ChunkError(chunk));
            if (error is not null)
            {
                writer.TryWrite((null, error, true));
                writer.TryComplete();
                return;
            }
            var final = api.ChunkIsFinal(chunk) != 0;
            writer.TryWrite((Marshal.PtrToStringUTF8(api.ChunkText(chunk)), null, final));
            if (final)
                writer.TryComplete();
        }

        private readonly IntPtr _lib;

        private Api(IntPtr lib, string libraryPath) => (_lib, LibraryPath) = (lib, libraryPath);

        public string LibraryPath { get; }

        public static Api Load(string dllPath)
        {
            // dxcompiler.dll and dxil.dll sit next to litert-lm.dll and are loaded lazily by name.
            if (OperatingSystem.IsWindows())
                SetDllDirectory(Path.GetDirectoryName(dllPath)!);
            var lib = NativeLibrary.Load(dllPath);
            Current = new Api(lib, dllPath);
            ((delegate* unmanaged[Cdecl]<int, void>)Current.Fn("litert_lm_set_min_log_level"))(3);
            return Current;
        }

        private IntPtr Fn(string name) => NativeLibrary.GetExport(_lib, name);

        public IntPtr CreateEngine(string modelPath, string backend, string cacheDir, int maxTokens, bool speculative)
        {
            var settings = Call4S("litert_lm_engine_settings_create", modelPath, backend, null, null);
            if (settings == IntPtr.Zero)
                return IntPtr.Zero;
            try
            {
                CallPS("litert_lm_engine_settings_set_cache_dir", settings, cacheDir);
                ((delegate* unmanaged[Cdecl]<IntPtr, int, void>)Fn("litert_lm_engine_settings_set_max_num_tokens"))(settings, maxTokens);
                if (speculative)
                    ((delegate* unmanaged[Cdecl]<IntPtr, byte, void>)Fn("litert_lm_engine_settings_set_enable_speculative_decoding"))(settings, 1);
                // Real decode tokens/s for the reply footer (stream chunks can hold several tokens, especially with MTP).
                ((delegate* unmanaged[Cdecl]<IntPtr, void>)Fn("litert_lm_engine_settings_enable_benchmark"))(settings);
                return ((delegate* unmanaged[Cdecl]<IntPtr, IntPtr>)Fn("litert_lm_engine_create"))(settings);
            }
            finally
            {
                ((delegate* unmanaged[Cdecl]<IntPtr, void>)Fn("litert_lm_engine_settings_delete"))(settings);
            }
        }

        /// <summary>Whether the .litertlm file carries an MTP drafter; without one, speculative decoding does nothing.</summary>
        public bool FileHasMtp(string modelPath)
        {
            var path = Utf8(modelPath);
            try
            {
                var file = ((delegate* unmanaged[Cdecl]<IntPtr, IntPtr>)Fn("litert_lm_loaded_file_create"))(path);
                if (file == IntPtr.Zero)
                    return false;
                var has = ((delegate* unmanaged[Cdecl]<IntPtr, byte>)Fn("litert_lm_loaded_file_has_speculative_decoding_support"))(file) != 0;
                ((delegate* unmanaged[Cdecl]<IntPtr, void>)Fn("litert_lm_loaded_file_delete"))(file);
                return has;
            }
            finally
            {
                Marshal.FreeHGlobal(path);
            }
        }

        public void EngineDelete(IntPtr engine) => ((delegate* unmanaged[Cdecl]<IntPtr, void>)Fn("litert_lm_engine_delete"))(engine);

        public IntPtr CreateConversation(IntPtr engine, string historyJson)
        {
            var session = ((delegate* unmanaged[Cdecl]<IntPtr>)Fn("litert_lm_session_config_create"))();
            var config = ((delegate* unmanaged[Cdecl]<IntPtr>)Fn("litert_lm_conversation_config_create"))();
            try
            {
                ((delegate* unmanaged[Cdecl]<IntPtr, IntPtr, void>)Fn("litert_lm_conversation_config_set_session_config"))(config, session);
                if (historyJson != "[]")
                    CallPS("litert_lm_conversation_config_set_messages", config, historyJson);
                return ((delegate* unmanaged[Cdecl]<IntPtr, IntPtr, IntPtr>)Fn("litert_lm_conversation_create"))(engine, config);
            }
            finally
            {
                ((delegate* unmanaged[Cdecl]<IntPtr, void>)Fn("litert_lm_session_config_delete"))(session);
                ((delegate* unmanaged[Cdecl]<IntPtr, void>)Fn("litert_lm_conversation_config_delete"))(config);
            }
        }

        public void ConversationDelete(IntPtr c) => ((delegate* unmanaged[Cdecl]<IntPtr, void>)Fn("litert_lm_conversation_delete"))(c);
        public void ConversationCancel(IntPtr c) => ((delegate* unmanaged[Cdecl]<IntPtr, void>)Fn("litert_lm_conversation_cancel_process"))(c);

        public int SendStream(IntPtr conversation, string messageJson, IntPtr userData)
        {
            var message = Utf8(messageJson);
            var context = Utf8("{}");
            try
            {
                var fn = (delegate* unmanaged[Cdecl]<IntPtr, IntPtr, IntPtr, IntPtr, delegate* unmanaged[Cdecl]<IntPtr, IntPtr, void>, IntPtr, int>)Fn("litert_lm_conversation_send_message_stream");
                return fn(conversation, message, context, IntPtr.Zero, &OnChunk, userData);
            }
            finally
            {
                Marshal.FreeHGlobal(message);
                Marshal.FreeHGlobal(context);
            }
        }

        public IntPtr ChunkText(IntPtr chunk) => ((delegate* unmanaged[Cdecl]<IntPtr, IntPtr>)Fn("litert_lm_stream_chunk_get_text"))(chunk);
        public IntPtr ChunkError(IntPtr chunk) => ((delegate* unmanaged[Cdecl]<IntPtr, IntPtr>)Fn("litert_lm_stream_chunk_get_error"))(chunk);
        /// <summary>Token count and engine-reported decode rate of the last reply (0 when there is no benchmark data).</summary>
        public (int Tokens, double Rate) LastDecode(IntPtr conversation)
        {
            var info = ((delegate* unmanaged[Cdecl]<IntPtr, IntPtr>)Fn("litert_lm_conversation_get_benchmark_info"))(conversation);
            if (info == IntPtr.Zero)
                return (0, 0);
            try
            {
                var turns = ((delegate* unmanaged[Cdecl]<IntPtr, int>)Fn("litert_lm_benchmark_info_get_num_decode_turns"))(info);
                if (turns <= 0)
                    return (0, 0);
                var tokens = ((delegate* unmanaged[Cdecl]<IntPtr, int, int>)Fn("litert_lm_benchmark_info_get_decode_token_count_at"))(info, turns - 1);
                var rate = ((delegate* unmanaged[Cdecl]<IntPtr, int, double>)Fn("litert_lm_benchmark_info_get_decode_tokens_per_sec_at"))(info, turns - 1);
                return (tokens, rate);
            }
            finally
            {
                ((delegate* unmanaged[Cdecl]<IntPtr, void>)Fn("litert_lm_benchmark_info_delete"))(info);
            }
        }

        public byte ChunkIsFinal(IntPtr chunk) => ((delegate* unmanaged[Cdecl]<IntPtr, byte>)Fn("litert_lm_stream_chunk_is_final"))(chunk);

        private IntPtr Call4S(string name, string? a, string? b, string? c, string? d)
        {
            IntPtr pa = Utf8(a), pb = Utf8(b), pc = Utf8(c), pd = Utf8(d);
            try
            {
                return ((delegate* unmanaged[Cdecl]<IntPtr, IntPtr, IntPtr, IntPtr, IntPtr>)Fn(name))(pa, pb, pc, pd);
            }
            finally
            {
                Marshal.FreeHGlobal(pa);
                Marshal.FreeHGlobal(pb);
                Marshal.FreeHGlobal(pc);
                Marshal.FreeHGlobal(pd);
            }
        }

        private void CallPS(string name, IntPtr p, string s)
        {
            var ps = Utf8(s);
            try
            {
                ((delegate* unmanaged[Cdecl]<IntPtr, IntPtr, void>)Fn(name))(p, ps);
            }
            finally
            {
                Marshal.FreeHGlobal(ps);
            }
        }

        private static IntPtr Utf8(string? s)
        {
            if (s is null)
                return IntPtr.Zero;
            var bytes = Encoding.UTF8.GetBytes(s + '\0');
            var ptr = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, ptr, bytes.Length);
            return ptr;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool SetDllDirectory(string path);
    }
}
