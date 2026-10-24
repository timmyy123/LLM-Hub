using Windows.Graphics.Imaging;
using Windows.Media.Editing;
using Windows.Media.MediaProperties;
using Windows.Storage;

namespace LLMHub;

/// <summary>
/// Windows imaging for Studio's Snapdragon NPU engines: reference pictures in any format Windows decodes, scaled to
/// cover and centre-cropped to the model's size, and LTX-Video frames written as an H.264 MP4 (Media Foundation).
/// </summary>
internal static class NpuImages
{
    public static async Task<byte[]> LoadRgbAsync(string path, int width, int height)
    {
        var file = await StorageFile.GetFileFromPathAsync(path);
        using var stream = await file.OpenReadAsync();
        var decoder = await BitmapDecoder.CreateAsync(stream);
        var (w, h) = (decoder.OrientedPixelWidth, decoder.OrientedPixelHeight);
        var scale = Math.Max((double)width / w, (double)height / h);
        var scaledW = (uint)Math.Max(width, Math.Round(w * scale));
        var scaledH = (uint)Math.Max(height, Math.Round(h * scale));
        var transform = new BitmapTransform
        {
            ScaledWidth = scaledW,
            ScaledHeight = scaledH,
            InterpolationMode = BitmapInterpolationMode.Fant,
            Bounds = new BitmapBounds { X = (scaledW - (uint)width) / 2, Y = (scaledH - (uint)height) / 2, Width = (uint)width, Height = (uint)height },
        };
        var pixels = await decoder.GetPixelDataAsync(BitmapPixelFormat.Rgba8, BitmapAlphaMode.Straight, transform,
            ExifOrientationMode.RespectExifOrientation, ColorManagementMode.ColorManageToSRgb);
        var rgba = pixels.DetachPixelData();
        var rgb = new byte[width * height * 3];
        for (var i = 0; i < width * height; i++)
        {
            rgb[i * 3] = rgba[i * 4];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
        }
        return rgb;
    }

    /// <summary>One clip per frame for 1/fps seconds, rendered to H.264 at the frames' own size.</summary>
    public static async Task EncodeMp4Async(IReadOnlyList<string> frames, int fps, string output)
    {
        var composition = new MediaComposition();
        var duration = TimeSpan.FromTicks(TimeSpan.TicksPerSecond / fps);
        uint width = 0, height = 0;
        foreach (var path in frames)
        {
            var file = await StorageFile.GetFileFromPathAsync(path);
            if (width == 0)
            {
                using var stream = await file.OpenReadAsync();
                var decoder = await BitmapDecoder.CreateAsync(stream);
                (width, height) = (decoder.PixelWidth, decoder.PixelHeight);
            }
            composition.Clips.Add(await MediaClip.CreateFromImageFileAsync(file, duration));
        }
        var folder = await StorageFolder.GetFolderFromPathAsync(Path.GetDirectoryName(output)!);
        var target = await folder.CreateFileAsync(Path.GetFileName(output), CreationCollisionOption.ReplaceExisting);
        var profile = MediaEncodingProfile.CreateMp4(VideoEncodingQuality.HD720p);
        profile.Video.Width = width;
        profile.Video.Height = height;
        profile.Video.FrameRate.Numerator = (uint)fps;
        profile.Video.FrameRate.Denominator = 1;
        profile.Video.Bitrate = 6_000_000;
        profile.Audio = null;
        var result = await composition.RenderToFileAsync(target, MediaTrimmingPreference.Precise, profile);
        if (result != Windows.Media.Transcoding.TranscodeFailureReason.None)
            throw new InvalidOperationException($"MP4 encoding failed: {result}");
    }
}
