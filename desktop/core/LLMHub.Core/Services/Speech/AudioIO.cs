using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace LLMHub.Services.Speech;

/// <summary>
/// Microphone recording to a 16 kHz mono WAV and WAV playback, without extra packages: Windows uses the built-in MCI and
/// PlaySound APIs (winmm); Linux uses PipeWire's pw-record/pw-play or ALSA's arecord/aplay, whichever is installed.
/// </summary>
public static partial class AudioIO
{
    private static Process? _recorder;
    private static Process? _player;
    private static string? _recordPath;

    public static bool IsRecording => _recordPath is not null;

    public static void StartRecording(string wavPath)
    {
        StopRecordingInternal(save: false);
        _recordPath = wavPath;
        if (OperatingSystem.IsWindows())
        {
            Mci("open new type waveaudio alias llmhubrec");
            Mci("set llmhubrec bitspersample 16 samplespersec 16000 channels 1");
            Mci("record llmhubrec");
            return;
        }
        var (tool, args) = Which("pw-record") is { } pw
            ? (pw, $"--rate 16000 --channels 1 --format s16 \"{wavPath}\"")
            : (Which("arecord") ?? throw new InvalidOperationException("No recorder found (install pipewire or alsa-utils)"),
               $"-q -f S16_LE -r 16000 -c 1 \"{wavPath}\"");
        _recorder = Process.Start(new ProcessStartInfo(tool, args) { UseShellExecute = false, RedirectStandardError = true });
    }

    /// <summary>Stops recording and returns the WAV path.</summary>
    public static string StopRecording()
    {
        var path = _recordPath ?? throw new InvalidOperationException("Not recording");
        StopRecordingInternal(save: true);
        return path;
    }

    private static void StopRecordingInternal(bool save)
    {
        if (_recordPath is null)
            return;
        if (OperatingSystem.IsWindows())
        {
            Mci("stop llmhubrec");
            if (save)
                Mci($"save llmhubrec \"{_recordPath}\"");
            Mci("close llmhubrec");
        }
        else if (_recorder is { HasExited: false })
        {
            // SIGINT lets the recorder finish the WAV header.
            Kill(_recorder.Id, 2);
            _recorder.WaitForExit(3000);
        }
        _recorder = null;
        _recordPath = null;
    }

    /// <summary>Plays a WAV file and completes when it has finished (or was stopped).</summary>
    public static async Task PlayAsync(string wavPath, double seconds, CancellationToken ct)
    {
        Stop();
        if (OperatingSystem.IsWindows())
        {
            PlaySound(wavPath, IntPtr.Zero, SndFilename | SndAsync | SndNoDefault);
            try
            {
                await Task.Delay(TimeSpan.FromSeconds(seconds + 0.3), ct);
            }
            finally
            {
                PlaySound(null, IntPtr.Zero, 0);
            }
            return;
        }
        var tool = Which("pw-play") ?? Which("paplay") ?? Which("aplay") ?? throw new InvalidOperationException("No audio player found");
        _player = Process.Start(new ProcessStartInfo(tool, $"\"{wavPath}\"") { UseShellExecute = false, RedirectStandardError = true });
        using var registration = ct.Register(Stop);
        await _player!.WaitForExitAsync(CancellationToken.None);
    }

    /// <summary>
    /// A player that reads raw float32 interleaved samples from its standard input (Linux: pw-play, pacat or aplay).
    /// Returns null when LLMHUB_AUDIO_SINK=null, which the self-tests use on machines without a sound server.
    /// </summary>
    public static Process? OpenStream(int sampleRate, int channels)
    {
        if (Environment.GetEnvironmentVariable("LLMHUB_AUDIO_SINK") == "null")
            return null;
        var (tool, args) = Which("pw-play") is { } pw ? (pw, $"--rate {sampleRate} --channels {channels} --format f32 -")
            : Which("pacat") is { } pacat ? (pacat, $"--playback --raw --format=float32le --rate={sampleRate} --channels={channels}")
            : Which("aplay") is { } aplay ? (aplay, $"-q -t raw -f FLOAT_LE -r {sampleRate} -c {channels} -")
            : throw new InvalidOperationException("No audio player found (install pipewire, pulseaudio-utils or alsa-utils)");
        return Process.Start(new ProcessStartInfo(tool, args) { UseShellExecute = false, RedirectStandardInput = true, RedirectStandardError = true });
    }

    public static void Stop()
    {
        if (OperatingSystem.IsWindows())
            PlaySound(null, IntPtr.Zero, 0);
        if (_player is { HasExited: false })
            _player.Kill();
        _player = null;
    }

    /// <summary>Writes 32-bit float samples as a 16-bit PCM WAV.</summary>
    public static void WriteWav(string path, ReadOnlySpan<float> samples, int sampleRate)
    {
        using var writer = new WavWriter(path, sampleRate, 1);
        writer.Write(samples);
    }

    private static string? Which(string tool) =>
        (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator)
            .Select(dir => Path.Combine(dir, tool)).FirstOrDefault(File.Exists);

    private static void Mci(string command)
    {
        var error = mciSendStringW(command, null, 0, IntPtr.Zero);
        if (error != 0)
        {
            var text = new StringBuilder(256);
            mciGetErrorStringW(error, text, text.Capacity);
            throw new InvalidOperationException($"Microphone: {text}");
        }
    }

    private const uint SndAsync = 0x0001, SndNoDefault = 0x0002, SndFilename = 0x00020000;

    [DllImport("winmm.dll", CharSet = CharSet.Unicode)]
    private static extern bool PlaySound(string? sound, IntPtr module, uint flags);

    [DllImport("winmm.dll", CharSet = CharSet.Unicode)]
    private static extern int mciSendStringW(string command, StringBuilder? returnString, int returnLength, IntPtr callback);

    [DllImport("winmm.dll", CharSet = CharSet.Unicode)]
    private static extern bool mciGetErrorStringW(int error, StringBuilder text, int length);

    [DllImport("libc", EntryPoint = "kill")]
    private static extern int Kill(int pid, int signal);
}

/// <summary>A 16-bit PCM WAV written as samples arrive; the header's sizes are filled in on Dispose (and in copies).</summary>
public sealed class WavWriter : IDisposable
{
    private readonly FileStream _stream;
    private readonly int _sampleRate;
    private readonly short _channels;
    private long _dataBytes;

    public WavWriter(string path, int sampleRate, int channels)
    {
        Path = path;
        _sampleRate = sampleRate;
        _channels = (short)channels;
        _stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.Read, 1 << 16);
        _stream.Write(Header(0));
    }

    public string Path { get; }

    public void Write(ReadOnlySpan<float> samples)
    {
        Span<byte> pcm = stackalloc byte[4096];
        for (var i = 0; i < samples.Length; i += pcm.Length / 2)
        {
            var count = Math.Min(pcm.Length / 2, samples.Length - i);
            for (var j = 0; j < count; j++)
                System.Buffers.Binary.BinaryPrimitives.WriteInt16LittleEndian(pcm[(j * 2)..],
                    (short)Math.Clamp(samples[i + j] * 32767f, short.MinValue, short.MaxValue));
            _stream.Write(pcm[..(count * 2)]);
        }
        _dataBytes += samples.Length * 2L;
    }

    /// <summary>Copies what has been written so far to a complete WAV at <paramref name="path"/>.</summary>
    public void CopyTo(string path)
    {
        _stream.Flush();
        using var copy = File.Create(path);
        copy.Write(Header(_dataBytes));
        _stream.Position = 44;
        var buffer = new byte[1 << 16];
        var left = _dataBytes;
        while (left > 0)
        {
            var n = _stream.Read(buffer, 0, (int)Math.Min(buffer.Length, left));
            if (n == 0)
                break;
            copy.Write(buffer, 0, n);
            left -= n;
        }
        _stream.Seek(0, SeekOrigin.End);
    }

    private byte[] Header(long dataBytes)
    {
        var data = (int)Math.Min(dataBytes, int.MaxValue - 36);
        var header = new byte[44];
        using var w = new BinaryWriter(new MemoryStream(header));
        w.Write("RIFF"u8);
        w.Write(36 + data);
        w.Write("WAVEfmt "u8);
        w.Write(16);
        w.Write((short)1);
        w.Write(_channels);
        w.Write(_sampleRate);
        w.Write(_sampleRate * _channels * 2);
        w.Write((short)(_channels * 2));
        w.Write((short)16);
        w.Write("data"u8);
        w.Write(data);
        return header;
    }

    public void Dispose()
    {
        _stream.Position = 0;
        _stream.Write(Header(_dataBytes));
        _stream.Dispose();
    }
}
