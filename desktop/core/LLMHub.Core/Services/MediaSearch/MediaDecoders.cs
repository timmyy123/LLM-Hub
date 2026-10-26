namespace LLMHub.Services.MediaSearch;

/// <summary>
/// Image, audio and video decoding for media search. Each app plugs in its platform's decoders (WinRT imaging, media
/// transcoding and media editing on Windows; GdkPixbuf and GStreamer or ffmpeg on Linux); without them, photos are passed
/// as their own JPEG/PNG bytes, only WAV audio is read and videos cannot be opened.
/// </summary>
public static class MediaDecoders
{
    public const int SampleRate = 16_000;

    /// <summary>A JPEG of the image, no larger than maxEdge on its long side; null if it cannot be read.</summary>
    public static Func<string, int, CancellationToken, Task<byte[]?>> Image { get; set; } = DefaultImageAsync;

    /// <summary>The first maxSeconds of an audio file (or a video's sound) as 16 kHz mono floats; null if it cannot be decoded.</summary>
    public static Func<string, int, CancellationToken, Task<float[]?>> Audio { get; set; } = (path, maxSeconds, ct) => Task.FromResult(ReadWav(path, maxSeconds));

    /// <summary>
    /// Opens a video (path, maxEdge, maxSeconds) for frame grabs, frames no larger than maxEdge and none past maxSeconds;
    /// null if it cannot be read.
    /// </summary>
    public static Func<string, int, int, CancellationToken, Task<VideoFrames?>> Video { get; set; } = (_, _, _, _) => Task.FromResult<VideoFrames?>(null);

    /// <summary>One frame (path, timeMs, maxEdge) as a JPEG for a result thumbnail; null if it cannot be read.</summary>
    public static Func<string, int, int, CancellationToken, Task<byte[]?>> VideoThumbnail { get; set; } = DefaultVideoThumbnailAsync;

    /// <summary>Why no video can be opened on this computer (say, no decoder installed), or null when videos can be read.</summary>
    public static Func<string?> VideoUnavailable { get; set; } = () => null;

    public static readonly string[] ImageExtensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic", ".tif", ".tiff"];
    public static readonly string[] AudioExtensions = [".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif"];
    public static readonly string[] VideoExtensions = [".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".mpg", ".mpeg", ".3gp", ".ogv"];

    private static async Task<byte[]?> DefaultVideoThumbnailAsync(string path, int timeMs, int maxEdge, CancellationToken ct)
    {
        using var video = await Video(path, maxEdge, timeMs / 1000 + 1, ct);
        return video is null ? null : (await video.FramesAsync([timeMs], ct))[0];
    }

    private static async Task<byte[]?> DefaultImageAsync(string path, int maxEdge, CancellationToken ct)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext is ".jpg" or ".jpeg" or ".png" ? await File.ReadAllBytesAsync(path, ct) : null;
    }

    /// <summary>PCM WAV (8/16/24/32-bit integer or 32-bit float, any rate and channel count) to 16 kHz mono.</summary>
    public static float[]? ReadWav(string path, int maxSeconds)
    {
        if (!path.EndsWith(".wav", StringComparison.OrdinalIgnoreCase))
            return null;
        try
        {
            using var reader = new BinaryReader(File.OpenRead(path));
            if (new string(reader.ReadChars(4)) != "RIFF")
                return null;
            reader.ReadInt32();
            if (new string(reader.ReadChars(4)) != "WAVE")
                return null;
            int format = 1, channels = 1, rate = SampleRate, bits = 16;
            while (reader.BaseStream.Position + 8 <= reader.BaseStream.Length)
            {
                var id = new string(reader.ReadChars(4));
                var size = reader.ReadInt32();
                if (id == "fmt ")
                {
                    format = reader.ReadInt16();
                    channels = reader.ReadInt16();
                    rate = reader.ReadInt32();
                    reader.ReadInt32();
                    reader.ReadInt16();
                    bits = reader.ReadInt16();
                    reader.BaseStream.Seek(size - 16, SeekOrigin.Current);
                }
                else if (id == "data")
                {
                    var bytesPerSample = bits / 8;
                    var frames = Math.Min(size / (bytesPerSample * channels), (long)rate * maxSeconds);
                    var mono = new float[frames];
                    for (var i = 0; i < frames; i++)
                    {
                        float sum = 0;
                        for (var c = 0; c < channels; c++)
                            sum += Sample(reader, format, bits);
                        mono[i] = sum / channels;
                    }
                    return Resample(mono, rate);
                }
                else
                    reader.BaseStream.Seek(size + (size & 1), SeekOrigin.Current);
            }
        }
        catch (Exception e) when (e is IOException or EndOfStreamException or ArgumentException)
        {
            Log.Error($"WAV {path} unreadable", e);
        }
        return null;
    }

    private static float Sample(BinaryReader r, int format, int bits) => (format, bits) switch
    {
        (3, 32) => r.ReadSingle(),
        (_, 8) => (r.ReadByte() - 128) / 128f,
        (_, 16) => r.ReadInt16() / 32768f,
        (_, 24) => ((r.ReadByte() | (r.ReadByte() << 8) | (r.ReadSByte() << 16))) / 8388608f,
        (_, 32) => r.ReadInt32() / 2147483648f,
        _ => throw new InvalidDataException($"Unsupported WAV format {format}/{bits}"),
    };

    /// <summary>Linear resampling to 16 kHz; plenty for embeddings.</summary>
    public static float[] Resample(float[] samples, int rate)
    {
        if (rate == SampleRate)
            return samples;
        var length = (int)((long)samples.Length * SampleRate / rate);
        var output = new float[length];
        var step = (double)rate / SampleRate;
        for (var i = 0; i < length; i++)
        {
            var pos = i * step;
            var index = (int)pos;
            var frac = (float)(pos - index);
            var a = samples[Math.Min(index, samples.Length - 1)];
            var b = samples[Math.Min(index + 1, samples.Length - 1)];
            output[i] = a + (b - a) * frac;
        }
        return output;
    }

    /// <summary>A 16 kHz mono PCM16 WAV of samples[from, to).</summary>
    public static byte[] Wav(float[] samples, int from, int to)
    {
        var count = Math.Max(0, to - from);
        using var stream = new MemoryStream(44 + count * 2);
        using var w = new BinaryWriter(stream);
        w.Write("RIFF"u8);
        w.Write(36 + count * 2);
        w.Write("WAVEfmt "u8);
        w.Write(16);
        w.Write((short)1);
        w.Write((short)1);
        w.Write(SampleRate);
        w.Write(SampleRate * 2);
        w.Write((short)2);
        w.Write((short)16);
        w.Write("data"u8);
        w.Write(count * 2);
        for (var i = from; i < to; i++)
            w.Write((short)(Math.Clamp(samples[i], -1f, 1f) * 32767f));
        w.Flush();
        return stream.ToArray();
    }

    public static float Rms(float[] samples, int from, int to)
    {
        if (to <= from)
            return 0;
        double sum = 0;
        for (var i = from; i < to; i++)
            sum += samples[i] * samples[i];
        return (float)Math.Sqrt(sum / (to - from));
    }
}

/// <summary>An open video: its length and JPEG frames at chosen times, from the platform's decoder.</summary>
public abstract class VideoFrames : IDisposable
{
    /// <summary>The video's length in milliseconds (0 when unknown).</summary>
    public abstract int DurationMs { get; }

    /// <summary>A JPEG of the frame at each time (ms), in order; null where a frame cannot be read.</summary>
    public abstract Task<byte[]?[]> FramesAsync(IReadOnlyList<int> timesMs, CancellationToken ct);

    public virtual void Dispose()
    {
    }
}
