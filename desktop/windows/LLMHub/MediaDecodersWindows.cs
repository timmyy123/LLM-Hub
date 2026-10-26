using System.Runtime.InteropServices.WindowsRuntime;
using LLMHub.Services;
using LLMHub.Services.MediaSearch;
using Windows.Graphics.Imaging;
using Windows.Media.MediaProperties;
using Windows.Media.Transcoding;
using Windows.Storage;
using Windows.Storage.Streams;

namespace LLMHub;

/// <summary>
/// Studio › Search decoding with Windows' own codecs (Media Foundation): any photo format Windows reads, MP3/AAC/FLAC/WMA
/// audio, and videos' frames and sound through Media Foundation's Source Reader.
/// </summary>
internal static class MediaDecodersWindows
{
    public static void Register()
    {
        MediaDecoders.Image = ImageAsync;
        MediaDecoders.Audio = AudioAsync;
        MediaDecoders.Video = VideoAsync;
    }

    private static Task<VideoFrames?> VideoAsync(string path, int maxEdge, int maxSeconds, CancellationToken ct) => Task.Run(() =>
    {
        try
        {
            if (MediaFoundationReader.Open(path, video: true) is { } reader)
                return (VideoFrames?)new WindowsVideo(reader, maxEdge, path);
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Info($"Media search: could not open video {path}: {e.Message}");
            return null;
        }
        Log.Info($"Media search: Windows cannot decode the video {path}");
        return null;
    }, ct);

    /// <summary>
    /// Frames read in order: each request takes the decoded frame nearest its time, reading on from the last one (or
    /// seeking to the key frame before it when it is behind or far ahead).
    /// </summary>
    private sealed class WindowsVideo(MediaFoundationReader reader, int maxEdge, string path) : VideoFrames
    {
        private readonly SemaphoreSlim _gate = new(1, 1);
        private (IntPtr Sample, long Time)? _previous;
        private (IntPtr Sample, long Time)? _next;
        private bool _ended;
        private (long Time, byte[] Jpeg)? _last;

        public override int DurationMs { get; } = reader.DurationMs;

        public override async Task<byte[]?[]> FramesAsync(IReadOnlyList<int> timesMs, CancellationToken ct)
        {
            var frames = new byte[]?[timesMs.Count];
            await _gate.WaitAsync(ct);
            try
            {
                for (var i = 0; i < frames.Length; i++)
                {
                    ct.ThrowIfCancellationRequested();
                    if (await Task.Run(() => Nearest(timesMs[i] * 10_000L), ct) is not { } frame)
                        continue;
                    if (_last is { } last && last.Time == frame.Time)
                    {
                        frames[i] = last.Jpeg;
                        continue;
                    }
                    if (reader.Pixels(frame.Sample) is not { } pixels)
                        continue;
                    frames[i] = await JpegAsync(pixels, reader.Width, reader.Height, maxEdge);
                    _last = (frame.Time, frames[i]!);
                }
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                Log.Info($"Media search: could not read frames of {path}: {e.Message}");
            }
            finally
            {
                _gate.Release();
            }
            return frames;
        }

        private (IntPtr Sample, long Time)? Nearest(long time)
        {
            // Behind what was read, or more than 3 s ahead: seek instead of decoding everything in between.
            var position = _next?.Time ?? _previous?.Time ?? 0;
            if (time < (_previous?.Time ?? 0) || time > position + 30_000_000)
            {
                Clear();
                reader.Seek((int)(time / 10_000));
            }
            if (_previous is null && _next is null)
                _next = Advance();
            while (!_ended && _next is { } next && next.Time < time)
            {
                Drop(_previous);
                _previous = next;
                _next = Advance();
            }
            return (_previous, _next) switch
            {
                ({ } before, { } after) => time - before.Time <= after.Time - time ? before : after,
                ({ } before, null) => before,
                (null, { } after) => after,
                _ => null,
            };
        }

        private (IntPtr, long)? Advance()
        {
            var sample = reader.Read();
            _ended = sample is null;
            return sample;
        }

        private void Clear()
        {
            Drop(_previous);
            Drop(_next);
            _previous = _next = null;
            _ended = false;
        }

        private static void Drop((IntPtr Sample, long Time)? frame)
        {
            if (frame is { } f)
                MediaFoundationReader.Release(f.Sample);
        }

        public override void Dispose()
        {
            Clear();
            reader.Dispose();
        }
    }

    /// <summary>BGRA pixels scaled to fit maxEdge, as a JPEG of quality 80 (as on Android).</summary>
    private static async Task<byte[]> JpegAsync(byte[] pixels, int width, int height, int maxEdge)
    {
        var scale = Math.Min(1.0, (double)maxEdge / Math.Max(width, height));
        using var output = new InMemoryRandomAccessStream();
        var quality = new BitmapPropertySet { ["ImageQuality"] = new BitmapTypedValue(0.8, Windows.Foundation.PropertyType.Single) };
        var encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.JpegEncoderId, output, quality);
        encoder.SetPixelData(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore, (uint)width, (uint)height, 96, 96, pixels);
        encoder.BitmapTransform.ScaledWidth = (uint)Math.Max(1, Math.Round(width * scale));
        encoder.BitmapTransform.ScaledHeight = (uint)Math.Max(1, Math.Round(height * scale));
        encoder.BitmapTransform.InterpolationMode = BitmapInterpolationMode.Fant;
        await encoder.FlushAsync();
        var bytes = new byte[output.Size];
        output.Seek(0);
        await output.ReadAsync(bytes.AsBuffer(), (uint)bytes.Length, InputStreamOptions.None);
        return bytes;
    }

    /// <summary>A video's sound, read with Media Foundation as float PCM, downmixed and resampled to 16 kHz mono.</summary>
    private static float[]? VideoSound(string path, int maxSeconds)
    {
        using var reader = MediaFoundationReader.Open(path, video: false);
        if (reader is null)
            return null;
        var limit = (long)maxSeconds * reader.SampleRate;
        var mono = new List<float>(reader.SampleRate * 60);
        while (mono.Count < limit && reader.Read() is { } sample)
        {
            var floats = reader.Floats(sample.Sample);
            MediaFoundationReader.Release(sample.Sample);
            if (floats is null)
                continue;
            for (var i = 0; i + reader.Channels <= floats.Length && mono.Count < limit; i += reader.Channels)
            {
                float sum = 0;
                for (var c = 0; c < reader.Channels; c++)
                    sum += floats[i + c];
                mono.Add(sum / reader.Channels);
            }
        }
        return mono.Count == 0 ? null : MediaDecoders.Resample([.. mono], reader.SampleRate);
    }

    /// <summary>The image scaled to fit maxEdge, as a JPEG.</summary>
    private static async Task<byte[]?> ImageAsync(string path, int maxEdge, CancellationToken ct)
    {
        try
        {
            var file = await StorageFile.GetFileFromPathAsync(path);
            using var source = await file.OpenReadAsync();
            var decoder = await BitmapDecoder.CreateAsync(source);
            var scale = Math.Min(1.0, (double)maxEdge / Math.Max(decoder.OrientedPixelWidth, decoder.OrientedPixelHeight));
            var transform = new BitmapTransform
            {
                ScaledWidth = (uint)Math.Max(1, decoder.PixelWidth * scale),
                ScaledHeight = (uint)Math.Max(1, decoder.PixelHeight * scale),
                InterpolationMode = BitmapInterpolationMode.Fant,
            };
            var pixels = await decoder.GetPixelDataAsync(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore, transform,
                ExifOrientationMode.RespectExifOrientation, ColorManagementMode.ColorManageToSRgb);
            var width = (uint)Math.Max(1, decoder.OrientedPixelWidth * scale);
            var height = (uint)Math.Max(1, decoder.OrientedPixelHeight * scale);
            using var output = new InMemoryRandomAccessStream();
            var encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.JpegEncoderId, output);
            encoder.SetPixelData(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore, width, height, 96, 96, pixels.DetachPixelData());
            await encoder.FlushAsync();
            var bytes = new byte[output.Size];
            output.Seek(0);
            await output.ReadAsync(bytes.AsBuffer(), (uint)bytes.Length, InputStreamOptions.None);
            return bytes;
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Info($"Media search: could not read image {path}: {e.Message}");
            return null;
        }
    }

    /// <summary>
    /// The first maxSeconds transcoded to a 16 kHz mono WAV with Media Foundation, then read as samples. A video's sound
    /// is read with the Source Reader that also gives its frames, without a temporary file.
    /// </summary>
    private static async Task<float[]?> AudioAsync(string path, int maxSeconds, CancellationToken ct)
    {
        if (MediaDecoders.ReadWav(path, maxSeconds) is { } wav)
            return wav;
        if (MediaDecoders.VideoExtensions.Contains(Path.GetExtension(path).ToLowerInvariant()))
        {
            try
            {
                return await Task.Run(() => VideoSound(path, maxSeconds), ct);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                Log.Info($"Media search: could not decode the sound of {path}: {e.Message}");
                return null;
            }
        }
        var temp = Path.Combine(Path.GetTempPath(), $"llmhub-audio-{Guid.NewGuid():N}.wav");
        try
        {
            var source = await StorageFile.GetFileFromPathAsync(path);
            var folder = await StorageFolder.GetFolderFromPathAsync(Path.GetTempPath());
            var target = await folder.CreateFileAsync(Path.GetFileName(temp), CreationCollisionOption.ReplaceExisting);
            var profile = MediaEncodingProfile.CreateWav(AudioEncodingQuality.Low);
            profile.Audio = AudioEncodingProperties.CreatePcm(MediaDecoders.SampleRate, 1, 16);
            var transcoder = new MediaTranscoder { TrimStopTime = TimeSpan.FromSeconds(maxSeconds) };
            var prepared = await transcoder.PrepareFileTranscodeAsync(source, target, profile);
            if (!prepared.CanTranscode)
            {
                Log.Info($"Media search: cannot decode {path}: {prepared.FailureReason}");
                return null;
            }
            await prepared.TranscodeAsync().AsTask(ct);
            return MediaDecoders.ReadWav(temp, maxSeconds);
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Info($"Media search: could not decode audio {path}: {e.Message}");
            return null;
        }
        finally
        {
            try
            {
                File.Delete(temp);
            }
            catch (IOException)
            {
            }
        }
    }
}
