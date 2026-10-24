using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using LLMHub.Services.Speech.Supertonic;

namespace LLMHub.Services.Speech;

/// <summary>
/// Read aloud (Supertonic 3, ONNX on the CPU) and dictation (whisper.cpp's whisper-server). Both download on first use
/// into the engines folder; nothing is sent anywhere else. The Local API's /v1/audio endpoints use the same engines.
/// </summary>
public sealed partial class SpeechService : ObservableObject
{
    // Same files as the Android app: Supertone's FP32 release, and a community FP16 conversion (half the download; inputs
    // and outputs stay FP32). Voice styles come from Supertone's repo for both.
    private const string SupertonicRepo = "https://huggingface.co/Supertone/supertonic-3/resolve/3cadd1ee6394adea1bd021217a0e650ede09a323/";
    private const string SupertonicFp16Repo = "https://huggingface.co/Kyumdroid/supertonic-3-quant/resolve/d9bfd0a9384b42a76f9256eafdd3d8a5cc4706ba/fp16/";
    private static readonly string[] ModelFiles =
    [
        "onnx/duration_predictor.onnx", "onnx/text_encoder.onnx", "onnx/vector_estimator.onnx", "onnx/vocoder.onnx",
        "onnx/tts.json", "onnx/unicode_indexer.json",
    ];
    /// <summary>Supertonic 3's ten built-in voices.</summary>
    public static IReadOnlyList<string> BuiltInVoices { get; } = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"];
    /// <summary>The 31 languages Supertonic 3 reads (ISO 639-1).</summary>
    public static IReadOnlyList<string> Languages { get; } =
    [
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl",
        "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
    ];
    private const long MaxVoiceBytes = 2 * 1024 * 1024;

    private const string WhisperModel = "ggml-large-v3-turbo-q5_0.bin";
    private const string WhisperModelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/" + WhisperModel;

    private readonly SemaphoreSlim _ttsGate = new(1, 1);
    private readonly SemaphoreSlim _whisperGate = new(1, 1);
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromMinutes(5) };
    private TextToSpeech? _tts;
    private Process? _whisper;
    private string? _whisperUrl;
    private CancellationTokenSource? _speaking;

    public SpeechService()
    {
        Status = "";
    }

    /// <summary>Id of the message being read aloud, or null.</summary>
    [ObservableProperty] public partial string? SpeakingId { get; set; }
    [ObservableProperty] public partial bool IsRecording { get; set; }
    [ObservableProperty] public partial bool IsTranscribing { get; set; }
    /// <summary>Setup progress or the last error, for the composer and Settings › Speech and voice.</summary>
    [ObservableProperty] public partial string Status { get; set; }

    public static string TtsRoot => Path.Combine(AppPaths.Engines, "supertonic-3");
    public static string TtsFp16Root => Path.Combine(AppPaths.Engines, "supertonic-3-fp16");
    public static string CustomVoicesDir => Directory.CreateDirectory(Path.Combine(AppPaths.Root, "voices")).FullName;
    private static bool UseFp16 => AppSettings.Current.SpeechPrecision == "fp16";
    private static string ModelRoot => UseFp16 ? TtsFp16Root : TtsRoot;
    public static string WhisperRoot => Path.Combine(AppPaths.Engines, "whisper.cpp");
    public static bool IsTtsInstalled => ModelFiles.All(f => File.Exists(Path.Combine(ModelRoot, f)))
                                         && BuiltInVoices.All(v => File.Exists(Path.Combine(TtsRoot, "voice_styles", v + ".json")));
    public static bool IsAsrInstalled => File.Exists(Path.Combine(WhisperRoot, "models", WhisperModel));
    /// <summary>"Supertonic 3 · 4 voices", "Whisper large-v3 turbo (q5_0)" for Settings.</summary>
    public static string TtsName => UseFp16 ? "Supertonic 3 · FP16" : "Supertonic 3";
    public static string AsrName => "Whisper large-v3 turbo · q5_0";

    /// <summary>Downloads both speech engines now (Settings › Speech and voice) instead of on first use.</summary>
    public async Task SetUpAsync(CancellationToken ct)
    {
        await EnsureTtsAsync(ct);
        await EnsureWhisperAsync(ct);
        Status = "";
    }

    /// <summary>Stops the engines and deletes their downloads.</summary>
    public void Remove()
    {
        Stop();
        _tts = null;
        foreach (var dir in new[] { TtsRoot, TtsFp16Root, WhisperRoot })
            if (Directory.Exists(dir))
                Directory.Delete(dir, true);
    }

    /// <summary>The built-in voices, then voices the user imported (named after their files).</summary>
    public static IReadOnlyList<string> Voices =>
        [.. BuiltInVoices, .. Directory.EnumerateFiles(CustomVoicesDir, "*.json").Select(Path.GetFileNameWithoutExtension).OfType<string>().Order()];

    private static string VoicePath(string voice) => BuiltInVoices.Contains(voice)
        ? Path.Combine(TtsRoot, "voice_styles", voice + ".json")
        : Path.Combine(CustomVoicesDir, voice + ".json") is var custom && File.Exists(custom) ? custom : Path.Combine(TtsRoot, "voice_styles", "F1.json");

    /// <summary>
    /// Adds a Supertonic voice style (JSON with style_ttl [1,50,256] and style_dp [1,8,16], at most 2 MB, as the Android app
    /// accepts) and returns its name, or throws with a message for the user.
    /// </summary>
    public static string ImportVoice(string path)
    {
        var info = new FileInfo(path);
        if (!info.Exists || info.Length > MaxVoiceBytes)
            throw new InvalidDataException(Loc.S("VoiceImportInvalid"));
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            static long[] Dims(JsonElement root, string name) => root.GetProperty(name).GetProperty("dims").EnumerateArray().Select(d => d.GetInt64()).ToArray();
            if (!Dims(doc.RootElement, "style_ttl").SequenceEqual([1L, 50, 256]) || !Dims(doc.RootElement, "style_dp").SequenceEqual([1L, 8, 16]))
                throw new InvalidDataException(Loc.S("VoiceImportInvalid"));
        }
        catch (Exception e) when (e is JsonException or KeyNotFoundException or InvalidOperationException)
        {
            throw new InvalidDataException(Loc.S("VoiceImportInvalid"));
        }
        var name = string.Concat(Path.GetFileNameWithoutExtension(path).Where(c => char.IsLetterOrDigit(c) || c is '-' or '_' or ' ')).Trim();
        if (name.Length == 0 || BuiltInVoices.Contains(name, StringComparer.OrdinalIgnoreCase))
            name = "Voice " + name;
        File.Copy(path, Path.Combine(CustomVoicesDir, name + ".json"), overwrite: true);
        Log.Info($"Imported voice {name}");
        return name;
    }

    public static void DeleteVoice(string name)
    {
        if (BuiltInVoices.Contains(name))
            return;
        var file = Path.Combine(CustomVoicesDir, name + ".json");
        if (File.Exists(file))
            File.Delete(file);
        if (AppSettings.Current.Voice == name)
        {
            AppSettings.Current.Voice = "F1";
            AppSettings.Save();
        }
    }

    /// <summary>Called when the FP32/FP16 choice changes: the next request loads the other model.</summary>
    public void ReloadModel()
    {
        _tts = null;
        OnPropertyChanged(nameof(Status));
    }

    // ---------------------------------------------------------------- read aloud

    /// <summary>Reads text aloud; a second call with the same id stops it.</summary>
    public async Task ToggleSpeakAsync(string id, string text)
    {
        if (SpeakingId == id)
        {
            StopSpeaking();
            return;
        }
        StopSpeaking();
        var cts = new CancellationTokenSource();
        _speaking = cts;
        UiThread.Run(() => SpeakingId = id);
        try
        {
            var wav = await SynthesizeAsync(Markdown.ToPlainText(text), AppSettings.Current.Voice, null, cts.Token);
            await AudioIO.PlayAsync(wav.Path, wav.Seconds, cts.Token);
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception e)
        {
            Log.Error("Read aloud failed", e);
            UiThread.Run(() => Status = Loc.F("ErrorPrefix", e.Message));
        }
        finally
        {
            if (_speaking == cts)
                UiThread.Run(() => SpeakingId = null);
        }
    }

    public void StopSpeaking()
    {
        _speaking?.Cancel();
        _speaking = null;
        AudioIO.Stop();
        SpeakingId = null;
    }

    /// <summary>Text to a WAV file (44.1 kHz). The language is guessed from the text unless given.</summary>
    public async Task<(string Path, double Seconds)> SynthesizeAsync(string text, string voice, string? language, CancellationToken ct)
    {
        var tts = await EnsureTtsAsync(ct);
        var style = Helper.LoadVoiceStyle([VoicePath(voice)]);
        var chosen = AppSettings.Current.SpeechLanguage;
        var lang = language ?? (Languages.Contains(chosen) ? chosen : GuessLanguage(text));
        var speed = (float)Math.Clamp(AppSettings.Current.SpeechSpeed, 0.5, 2.0);
        var (wav, duration) = await Task.Run(() => tts.Call(text, lang, style, totalStep: 8, speed: speed), ct);
        var length = Math.Min(wav.Length, (int)(tts.SampleRate * duration[0]));
        var path = Path.Combine(Path.GetTempPath(), $"llmhub-tts-{Guid.NewGuid():N}.wav");
        AudioIO.WriteWav(path, wav.AsSpan(0, length), tts.SampleRate);
        return (path, duration[0]);
    }

    private async Task<TextToSpeech> EnsureTtsAsync(CancellationToken ct)
    {
        await _ttsGate.WaitAsync(ct);
        try
        {
            if (_tts is not null)
                return _tts;
            var downloads = ModelFiles.Select(f => (Url: (UseFp16 ? SupertonicFp16Repo : SupertonicRepo) + f, Target: Path.Combine(ModelRoot, f)))
                .Concat(BuiltInVoices.Select(v => (Url: $"{SupertonicRepo}voice_styles/{v}.json", Target: Path.Combine(TtsRoot, "voice_styles", v + ".json"))))
                .ToList();
            for (var i = 0; i < downloads.Count; i++)
            {
                var n = i;
                await Fetch.FileAsync(downloads[i].Url, downloads[i].Target,
                    p => UiThread.Post(() => Status = Loc.F("DownloadingVoice", ((n * 100 + p) / downloads.Count).ToString("F0"))), ct);
            }
            UiThread.Post(() => Status = "");
            _tts = await Task.Run(() => Helper.LoadTextToSpeech(Path.Combine(ModelRoot, "onnx")), ct);
            Log.Info($"Supertonic 3 loaded ({(UseFp16 ? "FP16" : "FP32")})");
            return _tts;
        }
        finally
        {
            _ttsGate.Release();
        }
    }

    /// <summary>
    /// The writing system first (Hangul, kana, Arabic, Greek, Devanagari, Cyrillic), then the app language if Supertonic
    /// reads it, else English, so Japanese text is read as Japanese in an English UI.
    /// </summary>
    private static string GuessLanguage(string text)
    {
        var tag = AppSettings.Current.Language.Length > 0 ? AppSettings.Current.Language : System.Globalization.CultureInfo.CurrentUICulture.Name;
        var ui = tag.Length >= 2 ? tag[..2].ToLowerInvariant() : "en";
        if (text.Any(c => c is >= '\uAC00' and <= '\uD7A3' or >= '\u1100' and <= '\u11FF'))
            return "ko";
        if (text.Any(c => c is >= '\u3040' and <= '\u30FF'))
            return "ja";
        if (text.Any(c => c is >= '\u0600' and <= '\u06FF'))
            return "ar";
        if (text.Any(c => c is >= '\u0370' and <= '\u03FF'))
            return "el";
        if (text.Any(c => c is >= '\u0900' and <= '\u097F'))
            return "hi";
        if (text.Any(c => c is >= '\u0400' and <= '\u04FF'))
            return ui is "uk" or "bg" ? ui : "ru";
        return Languages.Contains(ui) ? ui : "en";
    }

    // ---------------------------------------------------------------- dictation

    public void StartDictation()
    {
        if (IsRecording)
            return;
        StopSpeaking();
        AudioIO.StartRecording(Path.Combine(Path.GetTempPath(), $"llmhub-dictation-{Guid.NewGuid():N}.wav"));
        IsRecording = true;
        // Warm whisper-server up while the user talks.
        _ = EnsureWhisperAsync(CancellationToken.None).ContinueWith(t => Log.Error("Whisper setup failed", t.Exception), TaskContinuationOptions.OnlyOnFaulted);
    }

    /// <summary>Stops recording and returns the transcript ("" on failure, with <see cref="Status"/> set).</summary>
    public async Task<string> StopDictationAsync()
    {
        if (!IsRecording)
            return "";
        var wav = AudioIO.StopRecording();
        IsRecording = false;
        IsTranscribing = true;
        try
        {
            return await TranscribeAsync(wav, null, CancellationToken.None);
        }
        catch (Exception e)
        {
            Log.Error("Dictation failed", e);
            Status = Loc.F("ErrorPrefix", e.Message);
            return "";
        }
        finally
        {
            IsTranscribing = false;
            TryDelete(wav);
        }
    }

    /// <summary>Speech in a WAV (16 kHz mono works best) to text, through whisper-server.</summary>
    public async Task<string> TranscribeAsync(string wavPath, string? language, CancellationToken ct)
    {
        var url = await EnsureWhisperAsync(ct);
        using var form = new MultipartFormDataContent();
        var audio = new ByteArrayContent(await File.ReadAllBytesAsync(wavPath, ct));
        audio.Headers.ContentType = new MediaTypeHeaderValue("audio/wav");
        form.Add(audio, "file", "audio.wav");
        form.Add(new StringContent("json"), "response_format");
        form.Add(new StringContent(language ?? "auto"), "language");
        form.Add(new StringContent("0.0"), "temperature");
        using var response = await _http.PostAsync(url + "/inference", form, ct);
        var body = await response.Content.ReadAsStringAsync(ct);
        if (!response.IsSuccessStatusCode)
            throw new InvalidOperationException($"whisper-server {(int)response.StatusCode}: {body}");
        using var doc = JsonDocument.Parse(body);
        var text = doc.RootElement.GetProperty("text").GetString()?.Trim() ?? "";
        Log.Info($"Transcribed {Path.GetFileName(wavPath)}: {text.Length} chars");
        return text;
    }

    private async Task<string> EnsureWhisperAsync(CancellationToken ct)
    {
        await _whisperGate.WaitAsync(ct);
        try
        {
            if (_whisper is { HasExited: false } && _whisperUrl is not null)
                return _whisperUrl;
            var server = await EnsureWhisperBinaryAsync(ct);
            var model = Path.Combine(WhisperRoot, "models", WhisperModel);
            await Fetch.FileAsync(WhisperModelUrl, model, p => UiThread.Post(() => Status = Loc.F("DownloadingWhisper", p.ToString("F0"))), ct);
            UiThread.Post(() => Status = "");

            var port = FreePort();
            var info = new ProcessStartInfo(server)
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                WorkingDirectory = Path.GetDirectoryName(server)!,
            };
            foreach (var arg in new[] { "-m", model, "--host", "127.0.0.1", "--port", port.ToString(), "-t", Math.Max(4, Environment.ProcessorCount / 2).ToString() })
                info.ArgumentList.Add(arg);
            if (!OperatingSystem.IsWindows())
                info.Environment["LD_LIBRARY_PATH"] = Path.GetDirectoryName(server) + ":" + Environment.GetEnvironmentVariable("LD_LIBRARY_PATH");
            _whisper = Process.Start(info)!;
            _whisper.ErrorDataReceived += (_, _) => { };
            _whisper.OutputDataReceived += (_, _) => { };
            _whisper.BeginErrorReadLine();
            _whisper.BeginOutputReadLine();
            Ioc.Engines.Llama.Track(_whisper);
            var url = $"http://127.0.0.1:{port}";
            var deadline = DateTime.UtcNow.AddMinutes(2);
            while (DateTime.UtcNow < deadline)
            {
                if (_whisper.HasExited)
                    throw new InvalidOperationException($"whisper-server exited with {_whisper.ExitCode}");
                try
                {
                    using var health = await _http.GetAsync(url + "/health", ct);
                    if (health.IsSuccessStatusCode)
                    {
                        _whisperUrl = url;
                        Log.Info($"whisper-server ready on {port}");
                        return url;
                    }
                }
                catch (HttpRequestException)
                {
                }
                await Task.Delay(300, ct);
            }
            throw new TimeoutException("whisper-server did not start");
        }
        finally
        {
            _whisperGate.Release();
        }
    }

    /// <summary>Official whisper.cpp build for this machine: CUDA on NVIDIA, Adreno OpenCL on Snapdragon, BLAS/CPU otherwise.</summary>
    private static async Task<string> EnsureWhisperBinaryAsync(CancellationToken ct)
    {
        var exe = OperatingSystem.IsWindows() ? "whisper-server.exe" : "whisper-server";
        if (Directory.Exists(WhisperRoot) && Directory.EnumerateFiles(WhisperRoot, exe, SearchOption.AllDirectories).FirstOrDefault() is { } found)
            return found;

        var hw = Ioc.Hardware;
        string[] wanted = OperatingSystem.IsWindows()
            ? hw.IsArm64
                ? hw.HasAdrenoGpu ? ["whisper-bin-win-opencl-adreno-arm64.zip", "whisper-bin-win-cpu-arm64.zip"] : ["whisper-bin-win-cpu-arm64.zip"]
                : hw.HasNvidiaGpu ? ["whisper-bin-win-cuda-12.4.0-x64.zip", "whisper-blas-bin-x64.zip"] : ["whisper-blas-bin-x64.zip", "whisper-bin-x64.zip"]
            : hw.IsArm64 ? ["whisper-bin-ubuntu-arm64.tar.gz"] : ["whisper-bin-ubuntu-x64.tar.gz"];
        foreach (var name in wanted)
        {
            if (await Fetch.GitHubAssetAsync("ggml-org/whisper.cpp", n => n == name, ct) is not { } asset)
                continue;
            var archive = Path.Combine(WhisperRoot, name);
            Log.Info($"Installing whisper.cpp {asset.Tag} ({name})");
            await Fetch.FileAsync(asset.Url, archive, p => UiThread.Post(() => Ioc.Speech.Status = Loc.F("DownloadingWhisper", p.ToString("F0"))), ct);
            var target = Path.Combine(WhisperRoot, asset.Tag);
            Fetch.Extract(archive, target);
            File.Delete(archive);
            if (Directory.EnumerateFiles(target, exe, SearchOption.AllDirectories).FirstOrDefault() is { } server)
            {
                if (!OperatingSystem.IsWindows())
                    File.SetUnixFileMode(server, File.GetUnixFileMode(server) | UnixFileMode.UserExecute);
                return server;
            }
        }
        throw new InvalidOperationException("No whisper.cpp build for this computer");
    }

    public void Stop()
    {
        StopSpeaking();
        if (_whisper is { HasExited: false })
            _whisper.Kill(true);
        _whisper = null;
        _whisperUrl = null;
    }

    private static int FreePort()
    {
        var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
        listener.Start();
        var port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
        }
    }
}
