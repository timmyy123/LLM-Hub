using System.Runtime.InteropServices.WindowsRuntime;
using LLMHub.Services;
using LLMHub.Services.MediaSearch;
using Windows.Graphics.Imaging;
using Windows.Media.MediaProperties;
using Windows.Media.Transcoding;
using Windows.Storage;
using Windows.Storage.Streams;

namespace LLMHub;

/// <summary>Studio › Search decoding with Windows' own codecs: any photo format Windows reads, and MP3/AAC/FLAC/WMA audio.</summary>
internal static class MediaDecodersWindows
{
    public static void Register()
    {
        MediaDecoders.Image = ImageAsync;
        MediaDecoders.Audio = AudioAsync;
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

    /// <summary>The first maxSeconds transcoded to a 16 kHz mono WAV with Media Foundation, then read as samples.</summary>
    private static async Task<float[]?> AudioAsync(string path, int maxSeconds, CancellationToken ct)
    {
        if (MediaDecoders.ReadWav(path, maxSeconds) is { } wav)
            return wav;
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
