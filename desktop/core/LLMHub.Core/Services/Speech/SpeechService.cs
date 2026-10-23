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
    private const string SupertonicRepo = "https://huggingface.co/supertone-oss-archive/supertonic-3/resolve/aafc6e32416a594460b32413efc49d7fe4ce6d46/";
    private static readonly string[] SupertonicFiles =
    [
        "onnx/duration_predictor.onnx", "onnx/text_encoder.onnx", "onnx/vector_estimator.onnx", "onnx/vocoder.onnx",
        "onnx/tts.json", "onnx/unicode_indexer.json",
        "voice_styles/F1.json", "voice_styles/F2.json", "voice_styles/M1.json", "voice_styles/M2.json",
    ];
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
    public static string WhisperRoot => Path.Combine(AppPaths.Engines, "whisper.cpp");
    public static bool IsTtsInstalled => SupertonicFiles.All(f => File.Exists(Path.Combine(TtsRoot, f)));
    public static bool IsAsrInstalled => File.Exists(Path.Combine(WhisperRoot, WhisperModel));
    /// <summary>"Supertonic 3 · 4 voices", "Whisper large-v3 turbo (q5_0)" for Settings.</summary>
    public static string TtsName => "Supertonic 3";
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
        foreach (var dir in new[] { TtsRoot, WhisperRoot })
            if (Directory.Exists(dir))
                Directory.Delete(dir, true);
    }

    public static IReadOnlyList<string> Voices => ["F1", "F2", "M1", "M2"];

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
        var style = Helper.LoadVoiceStyle([Path.Combine(TtsRoot, "voice_styles", (Voices.Contains(voice) ? voice : "F1") + ".json")]);
        var lang = language ?? GuessLanguage(text);
        var (wav, duration) = await Task.Run(() => tts.Call(text, lang, style, totalStep: 8, speed: 1.05f), ct);
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
            for (var i = 0; i < SupertonicFiles.Length; i++)
            {
                var file = SupertonicFiles[i];
                var n = i;
                await Fetch.FileAsync(SupertonicRepo + file, Path.Combine(TtsRoot, file),
                    p => UiThread.Post(() => Status = Loc.F("DownloadingVoice", ((n * 100 + p) / SupertonicFiles.Length).ToString("F0"))), ct);
            }
            UiThread.Post(() => Status = "");
            _tts = await Task.Run(() => Helper.LoadTextToSpeech(Path.Combine(TtsRoot, "onnx")), ct);
            Log.Info("Supertonic 3 loaded");
            return _tts;
        }
        finally
        {
            _ttsGate.Release();
        }
    }

    /// <summary>App language first, then a cheap script check, so Russian text is read as Russian in an English UI.</summary>
    private static string GuessLanguage(string text)
    {
        if (text.Any(c => c is >= 'Ѐ' and <= 'ӿ'))
            return "ru";
        var tag = AppSettings.Current.Language.Length > 0 ? AppSettings.Current.Language : System.Globalization.CultureInfo.CurrentUICulture.Name;
        var ui = tag.Length >= 2 ? tag[..2] : "en";
        return ui is "es" or "pt" or "fr" or "de" or "it" or "ru" ? ui : "en";
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
