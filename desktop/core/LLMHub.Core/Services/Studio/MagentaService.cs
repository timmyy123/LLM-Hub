using System.Buffers.Binary;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using CommunityToolkit.Mvvm.ComponentModel;
using LLMHub.Services.Speech;

namespace LLMHub.Services.Studio;

/// <summary>One style in the live jam: a description and how much of it to blend in.</summary>
public sealed partial class MusicStyle : ObservableObject
{
    [ObservableProperty] public partial string Text { get; set; } = "";
    [ObservableProperty] public partial double Weight { get; set; }
}

/// <summary>
/// Live music with Magenta RealTime 2 on Linux with an NVIDIA GPU, as on the Mac (Music.dc.html): Google's JAX runtime
/// in a private Python, driven by shared/magenta/mrt_live.py. The sidecar streams 48 kHz stereo blocks that are played
/// through PipeWire (pw-play), PulseAudio (pacat) or ALSA (aplay), recorded on request, and kept for Export.
/// </summary>
public sealed partial class MagentaService : ObservableObject
{
    public const int SampleRate = 48000;
    private const string Model = "mrt2_small";
    private const string Repo = "google/magenta-realtime-2";
    private const string Revision = "010aa0dcb0dfd27b24f0ad07b4dad63e8f9521cc";
    private const string Packages = "magenta-rt==2.0.3";
    private const string Jax = "jax[cuda13]==0.12.0.1";
    private const long PackageBytes = 2_600_000_000;

    /// <summary>Files the JAX runtime reads, under MAGENTA_HOME/magenta-rt-v2, with their sizes.</summary>
    private static readonly (string Path, long Bytes)[] Files =
    [
        ("checkpoints/mrt2_small.safetensors", 1_128_840_272),
        ("resources/musiccoca/text_encoder.tflite", 418_674_324),
        ("resources/musiccoca/music_encoder.tflite", 370_935_584),
        ("resources/musiccoca/mapper.tflite", 86_166_664),
        ("resources/musiccoca/pretrained_vector_quantizer.tflite", 72_422_108),
        ("resources/musiccoca/audio_preprocessor.tflite", 8_729_640),
        ("resources/musiccoca/spm.model", 517_448),
        ("resources/spectrostream/decoder.safetensors", 209_853_216),
        ("resources/spectrostream/quantizer.safetensors", 67_108_984),
    ];

    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly object _audioLock = new();
    private readonly float[] _levels = new float[80];
    private Process? _process;
    private Process? _player;
    private TaskCompletionSource<string>? _ready;
    private WavWriter? _session;
    private WavWriter? _recording;
    private Timer? _clock;
    private Timer? _styleDebounce;
    private double _made;
    private Stopwatch _sinceStart = new();
    private Stopwatch _sinceFirstAudio = new();

    public MagentaService()
    {
        Status = "";
        Device = "";
        Styles = [new() { Text = "Lo-fi hip hop", Weight = 0.7 }, new() { Text = "Warm Rhodes piano", Weight = 0.45 },
            new() { Text = "Rain ambience", Weight = 0.2 }];
        foreach (var style in Styles)
            style.PropertyChanged += OnStyleChanged;
        Styles.CollectionChanged += (_, e) =>
        {
            foreach (var style in e.NewItems?.OfType<MusicStyle>() ?? [])
                style.PropertyChanged += OnStyleChanged;
            StylesChanged();
        };
        Temperature = 0.55;
        Guidance = 0.6;
        Variety = 0.4;
    }

    /// <summary>Google's JAX build of Magenta RealTime needs CUDA; on Windows only MLX runs, far slower than real time.</summary>
    public static bool IsSupported => OperatingSystem.IsLinux() && !Ioc.Hardware.IsArm64 && Ioc.Hardware.HasNvidiaGpu;

    public static string Root => Path.Combine(AppPaths.Engines, "magenta");
    private static string Home => Path.Combine(Root, "home");
    private static string DataDir => Path.Combine(Home, "magenta-rt-v2");
    private static string MarkerFile => Path.Combine(Root, "installed.txt");
    public static string Script => Path.Combine(AppContext.BaseDirectory, "magenta", "mrt_live.py");

    public static bool IsInstalled => File.Exists(MarkerFile) && Files.All(f => File.Exists(Path.Combine(DataDir, f.Path)));
    /// <summary>"Magenta RealTime · 4.9 GB to download on first use" before the first jam.</summary>
    public static long DownloadSize => PackageBytes + Files.Sum(f => f.Bytes);

    public ObservableCollection<MusicStyle> Styles { get; }

    [ObservableProperty] public partial string Status { get; set; }
    /// <summary>The GPU the sidecar runs on, from its ready message.</summary>
    [ObservableProperty] public partial string Device { get; set; }
    [ObservableProperty] public partial bool IsBusy { get; set; }
    [ObservableProperty] public partial bool IsPlaying { get; set; }
    [ObservableProperty] public partial bool IsRecording { get; set; }
    /// <summary>Seconds played since Play.</summary>
    [ObservableProperty] public partial double Elapsed { get; set; }
    /// <summary>Audio made per second of wall time; below 1 the music pauses now and then.</summary>
    [ObservableProperty] public partial double Speed { get; set; }
    /// <summary>Sliders 0…1, mapped like the Mac app to temperature 0.5…2, guidance 0…5 and top-k 1…100.</summary>
    [ObservableProperty] public partial double Temperature { get; set; }
    [ObservableProperty] public partial double Guidance { get; set; }
    [ObservableProperty] public partial double Variety { get; set; }

    public int ActiveStyleCount => Styles.Count(s => s.Weight > 0 && s.Text.Trim().Length > 0);

    /// <summary>The last 80 block levels (0…1) for the level bars.</summary>
    public float[] Levels
    {
        get
        {
            lock (_audioLock)
                return (float[])_levels.Clone();
        }
    }

    /// <summary>Raised with each saved WAV (recording or export).</summary>
    public event Action<string>? Saved;

    partial void OnTemperatureChanged(double value) => SendControls();
    partial void OnGuidanceChanged(double value) => SendControls();
    partial void OnVarietyChanged(double value) => SendControls();

    public void AddStyle() => Styles.Add(new MusicStyle { Text = "", Weight = 0.5 });

    public void RemoveStyle(MusicStyle style)
    {
        style.PropertyChanged -= OnStyleChanged;
        Styles.Remove(style);
    }

    // ---------------------------------------------------------------- play and stop

    public async Task ToggleAsync()
    {
        if (IsPlaying)
            Stop();
        else
            await PlayAsync(CancellationToken.None);
    }

    public async Task PlayAsync(CancellationToken ct)
    {
        if (IsPlaying || IsBusy)
            return;
        IsBusy = true;
        try
        {
            await EnsureRunningAsync(ct);
            _player = AudioIO.OpenStream(SampleRate, 2);
            lock (_audioLock)
            {
                _session?.Dispose();
                _session = new WavWriter(Path.Combine(Root, "session.wav"), SampleRate, 2);
                Array.Clear(_levels);
            }
            _made = 0;
            _sinceStart = Stopwatch.StartNew();
            _sinceFirstAudio = new Stopwatch();
            Elapsed = 0;
            Speed = 0;
            SendStyles();
            SendControls();
            Send(new JsonObject { ["cmd"] = "play" });
            IsPlaying = true;
            Status = "";
            _clock = new Timer(_ => UiThread.Post(Tick), null, 100, 100);
            Log.Info($"Magenta live started on {Device}");
            if (IsRecording)
                StartRecording();
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Error("Magenta live failed", e);
            Status = Loc.F("ErrorPrefix", e.Message);
        }
        finally
        {
            IsBusy = false;
        }
    }

    public void Stop()
    {
        if (!IsPlaying)
            return;
        Send(new JsonObject { ["cmd"] = "stop" });
        IsPlaying = false;
        _clock?.Dispose();
        _clock = null;
        if (_player is { HasExited: false })
        {
            try
            {
                _player.Kill();
            }
            catch (InvalidOperationException)
            {
            }
        }
        _player = null;
        FinishRecording();
        Tick();
        Log.Info($"Magenta live stopped after {_made:F1} s of audio at {Speed:F2}x real time");
    }

    public void SetRecording(bool on)
    {
        IsRecording = on;
        if (!IsPlaying)
            return;
        if (on)
            StartRecording();
        else
            FinishRecording();
    }

    /// <summary>Saves everything made since Play as a WAV in the Music folder.</summary>
    public void Export()
    {
        try
        {
            var path = NewRecordingPath();
            lock (_audioLock)
                (_session ?? throw new InvalidOperationException("Nothing played yet")).CopyTo(path);
            Added(path);
        }
        catch (Exception e) when (e is IOException or InvalidOperationException)
        {
            Status = Loc.F("ErrorPrefix", e.Message);
        }
    }

    private void StartRecording()
    {
        lock (_audioLock)
            _recording ??= new WavWriter(NewRecordingPath(), SampleRate, 2);
    }

    private void FinishRecording()
    {
        WavWriter? writer;
        lock (_audioLock)
        {
            writer = _recording;
            _recording = null;
        }
        if (writer is null)
            return;
        writer.Dispose();
        Added(writer.Path);
    }

    private void Added(string path)
    {
        Status = Loc.F("MusicSaved", Path.GetFileName(path));
        Log.Info($"Music saved {path}");
        Saved?.Invoke(path);
    }

    private string NewRecordingPath()
    {
        var top = Styles.Where(s => s.Text.Trim().Length > 0).MaxBy(s => s.Weight)?.Text ?? "music";
        var words = string.Join('-', new string(top.Select(c => char.IsLetterOrDigit(c) ? c : ' ').ToArray()).Split(' ').Where(w => w.Length > 0).Take(4)).ToLowerInvariant();
        return Path.Combine(ViewModels.StudioViewModel.MusicFolder, $"{DateTime.Now:yyyyMMdd-HHmmss}-{(words.Length > 0 ? words : "music")}.wav");
    }

    private void Tick()
    {
        var played = _sinceFirstAudio.IsRunning ? Math.Min(_made, _sinceFirstAudio.Elapsed.TotalSeconds) : 0;
        Elapsed = played;
        Speed = _made / Math.Max(0.1, _sinceStart.Elapsed.TotalSeconds);
        OnPropertyChanged(nameof(Levels));
    }

    // ---------------------------------------------------------------- styles and controls

    private void OnStyleChanged(object? sender, PropertyChangedEventArgs e) => StylesChanged();

    private void StylesChanged()
    {
        OnPropertyChanged(nameof(ActiveStyleCount));
        // Typing in a style box would otherwise embed every keystroke.
        _styleDebounce?.Dispose();
        _styleDebounce = new Timer(_ => UiThread.Post(SendStyles), null, 350, Timeout.Infinite);
    }

    private void SendStyles()
    {
        var styles = new JsonArray();
        foreach (var s in Styles.Where(s => s.Weight > 0 && s.Text.Trim().Length > 0))
            styles.Add(new JsonObject { ["text"] = s.Text.Trim(), ["weight"] = s.Weight });
        Send(new JsonObject { ["cmd"] = "styles", ["styles"] = styles });
    }

    private void SendControls() => Send(new JsonObject
    {
        ["cmd"] = "controls",
        ["temperature"] = 0.5 + Temperature * 1.5,
        ["guidance"] = Guidance * 5,
        ["top_k"] = Math.Max(1, (int)Math.Round(Variety * 100)),
    });

    private void Send(JsonObject command)
    {
        if (_process is not { HasExited: false } process)
            return;
        try
        {
            lock (process)
            {
                process.StandardInput.WriteLine(command.ToJsonString());
                process.StandardInput.Flush();
            }
        }
        catch (IOException e)
        {
            Log.Error("Magenta command failed", e);
        }
    }

    // ---------------------------------------------------------------- install and start

    private async Task EnsureRunningAsync(CancellationToken ct)
    {
        await _gate.WaitAsync(ct);
        try
        {
            if (_process is { HasExited: false })
                return;
            if (!IsSupported)
                throw new PlatformNotSupportedException(Loc.S("MusicNeedsNvidia"));
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
        await StandalonePython.EnsureAsync(Root, p => SetStatus(Loc.F("ComfyStepPython", p.ToString("F0"))), ct);
        if (!File.Exists(MarkerFile))
        {
            SetStatus(Loc.S("MusicStepPackages"));
            await StandalonePython.PipAsync(Root, SetStatus, ct, "install", "--upgrade", "pip");
            await StandalonePython.PipAsync(Root, SetStatus, ct, "install", Packages, Jax);
            await File.WriteAllTextAsync(MarkerFile, $"{Packages} {Jax}", ct);
        }
        var total = (double)Files.Sum(f => f.Bytes);
        var done = 0L;
        foreach (var (file, bytes) in Files)
        {
            var target = Path.Combine(DataDir, file);
            if (!File.Exists(target))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                var before = done;
                await Fetch.FileAsync($"https://huggingface.co/{Repo}/resolve/{Revision}/{file}", target,
                    p => SetStatus(Loc.F("MusicDownloading", ((before + bytes * p / 100) * 100 / total).ToString("F0"))), ct);
            }
            done += bytes;
        }
        SetStatus("");
        Log.Info($"Magenta RealTime installed ({Packages}, {Jax})");
    }

    private async Task StartAsync(CancellationToken ct)
    {
        if (!File.Exists(Script))
            throw new FileNotFoundException("The Magenta script is missing from the app", Script);
        SetStatus(Loc.S("MusicLoading"));
        var info = new ProcessStartInfo(StandalonePython.Exe(Root))
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var arg in new[] { "-s", "-u", Script, Model })
            info.ArgumentList.Add(arg);
        info.Environment["MAGENTA_HOME"] = Home;
        // JAX takes most of the GPU's memory up front by default; leave room for a chat model.
        info.Environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false";
        info.Environment["HF_HUB_OFFLINE"] = "1";
        var tail = new StringBuilder();
        _ready = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        var process = Process.Start(info)!;
        _process = process;
        Ioc.Engines.Llama.Track(process);
        process.ErrorDataReceived += (_, e) => OnSidecarLine(e.Data, tail);
        process.BeginErrorReadLine();
        _ = Task.Run(() => ReadAudio(process));
        _ = process.WaitForExitAsync(CancellationToken.None).ContinueWith(_ =>
        {
            _ready.TrySetException(new InvalidOperationException($"Magenta RealTime exited with {process.ExitCode}: {Tail(tail)}"));
            UiThread.Post(() =>
            {
                if (_process == process && IsPlaying)
                {
                    Status = Loc.F("ErrorPrefix", $"Magenta RealTime exited with {process.ExitCode}");
                    Stop();
                }
            });
        }, TaskScheduler.Default);
        var clock = Stopwatch.StartNew();
        // First start compiles the model for this GPU: about a minute.
        Device = await _ready.Task.WaitAsync(TimeSpan.FromMinutes(6), ct);
        SetStatus("");
        Log.Info($"Magenta RealTime ready on {Device} in {clock.Elapsed.TotalSeconds:F0}s");
    }

    private void OnSidecarLine(string? line, StringBuilder tail)
    {
        if (line is null)
            return;
        if (!line.StartsWith("EVT ", StringComparison.Ordinal))
        {
            lock (tail)
            {
                tail.AppendLine(line);
                if (tail.Length > 6000)
                    tail.Remove(0, tail.Length - 6000);
            }
            return;
        }
        var evt = JsonNode.Parse(line[4..]);
        switch (evt?["type"]?.GetValue<string>())
        {
            case "ready":
                _ready?.TrySetResult(evt["device"]?.GetValue<string>() ?? "GPU");
                break;
            case "error":
                var message = evt["message"]?.GetValue<string>() ?? "error";
                Log.Info($"Magenta RealTime: {message}");
                if (_ready?.TrySetException(new InvalidOperationException(message)) != true)
                    UiThread.Post(() => Status = Loc.F("ErrorPrefix", message));
                break;
        }
    }

    /// <summary>Reads "AUD0" blocks from the sidecar and plays, records and keeps them.</summary>
    private void ReadAudio(Process process)
    {
        var stdout = process.StandardOutput.BaseStream;
        var header = new byte[8];
        var buffer = Array.Empty<byte>();
        try
        {
            while (ReadExactly(stdout, header, 8))
            {
                if (header[0] != 'A' || header[1] != 'U' || header[2] != 'D' || header[3] != '0')
                    throw new InvalidDataException("Unexpected data from Magenta RealTime");
                var length = (int)BinaryPrimitives.ReadUInt32LittleEndian(header.AsSpan(4));
                if (buffer.Length < length)
                    buffer = new byte[length];
                if (!ReadExactly(stdout, buffer, length))
                    break;
                if (!IsPlaying)
                    continue;
                OnAudio(buffer.AsSpan(0, length));
            }
        }
        catch (Exception e) when (e is IOException or InvalidDataException or ObjectDisposedException)
        {
            Log.Error("Magenta audio stream ended", e);
        }
    }

    private void OnAudio(ReadOnlySpan<byte> block)
    {
        var samples = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(block);
        if (_player is { HasExited: false } player)
        {
            try
            {
                player.StandardInput.BaseStream.Write(block);
                player.StandardInput.BaseStream.Flush();
            }
            catch (IOException)
            {
                // The player was stopped.
            }
        }
        if (!_sinceFirstAudio.IsRunning)
            _sinceFirstAudio.Start();
        lock (_audioLock)
        {
            _session?.Write(samples);
            _recording?.Write(samples);
            // One level per 1/25 s (a Magenta frame).
            const int frame = SampleRate / 25 * 2;
            for (var start = 0; start + frame <= samples.Length; start += frame)
            {
                var sum = 0f;
                foreach (var s in samples.Slice(start, frame))
                    sum += s * s;
                Array.Copy(_levels, 1, _levels, 0, _levels.Length - 1);
                _levels[^1] = Math.Min(1f, MathF.Sqrt(sum / frame) * 3.5f);
            }
        }
        _made += samples.Length / 2.0 / SampleRate;
    }

    private static bool ReadExactly(Stream stream, byte[] buffer, int count)
    {
        var read = 0;
        while (read < count)
        {
            var n = stream.Read(buffer, read, count - read);
            if (n == 0)
                return false;
            read += n;
        }
        return true;
    }

    private static string Tail(StringBuilder tail)
    {
        lock (tail)
            return tail.ToString()[^Math.Min(tail.Length, 1200)..];
    }

    private void SetStatus(string text) => UiThread.Post(() => Status = text);

    /// <summary>Stops the music and the sidecar (app exit, or freeing the GPU).</summary>
    public void Shutdown()
    {
        Stop();
        if (_process is { HasExited: false } process)
        {
            Send(new JsonObject { ["cmd"] = "quit" });
            if (!process.WaitForExit(2000))
                process.Kill(true);
        }
        _process = null;
        lock (_audioLock)
        {
            _session?.Dispose();
            _session = null;
        }
    }
}
