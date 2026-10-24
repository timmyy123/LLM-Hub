using Windows.Graphics.Imaging;
using Windows.Storage;

namespace LLMHub;

/// <summary>Reference pictures for Studio's NPU image to image: any format Windows decodes, scaled and centre-cropped to a square.</summary>
internal static class NpuImages
{
    public static async Task<byte[]> LoadSquareRgbAsync(string path, int size)
    {
        var file = await StorageFile.GetFileFromPathAsync(path);
        using var stream = await file.OpenReadAsync();
        var decoder = await BitmapDecoder.CreateAsync(stream);
        var (w, h) = (decoder.OrientedPixelWidth, decoder.OrientedPixelHeight);
        var scale = (double)size / Math.Min(w, h);
        var scaledW = (uint)Math.Max(size, Math.Round(w * scale));
        var scaledH = (uint)Math.Max(size, Math.Round(h * scale));
        var transform = new BitmapTransform
        {
            ScaledWidth = scaledW,
            ScaledHeight = scaledH,
            InterpolationMode = BitmapInterpolationMode.Fant,
            Bounds = new BitmapBounds { X = (scaledW - (uint)size) / 2, Y = (scaledH - (uint)size) / 2, Width = (uint)size, Height = (uint)size },
        };
        var pixels = await decoder.GetPixelDataAsync(BitmapPixelFormat.Rgba8, BitmapAlphaMode.Straight, transform,
            ExifOrientationMode.RespectExifOrientation, ColorManagementMode.ColorManageToSRgb);
        var rgba = pixels.DetachPixelData();
        var rgb = new byte[size * size * 3];
        for (var i = 0; i < size * size; i++)
        {
            rgb[i * 3] = rgba[i * 4];
            rgb[i * 3 + 1] = rgba[i * 4 + 1];
            rgb[i * 3 + 2] = rgba[i * 4 + 2];
        }
        return rgb;
    }
}
