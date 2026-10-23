#if DEBUG
using System.Runtime.InteropServices.WindowsRuntime;
using LLMHub.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media.Imaging;
using Windows.Graphics.Imaging;

namespace LLMHub;

/// <summary>
/// Debug builds only: `LLMHub.exe [--selftest catalogId] --capture out.png [--page settings|models|chat] [--exit]`
/// optionally downloads a catalog model and sends a chat message through the real UI, then renders
/// the window's own XAML to a PNG so changes can be checked from a build script. The Agent, Design and
/// Studio end-to-end runs live in <see cref="DevTests"/>.
/// </summary>
internal static class DevCapture
{
    public static async void RunIfRequested(MainWindow window)
    {
        var args = Environment.GetCommandLineArgs();
        var path = ValueOf(args, "--capture");
        if (path is null)
            return;

        if (ValueOf(args, "--selftest") is { } catalogId)
            await SelfTestAsync(catalogId, ValueOf(args, "--format") ?? "gguf", ValueOf(args, "--quant"), ValueOf(args, "--prompt"));
        if (ValueOf(args, "--studiotest") is { } workflowId)
            await DevTests.StudioAsync(workflowId, ValueOf(args, "--prompt") ?? "a cozy cafe interior in morning light, film photo", ValueOf(args, "--image"));
        if (ValueOf(args, "--designtest") is { } request)
        {
            await DevTests.DesignAsync(request, ValueOf(args, "--kind"), ValueOf(args, "--model"), ValueOf(args, "--followup"));
            window.ShowSection("design");
            await Task.Delay(4000);
        }
        if (ValueOf(args, "--agenttest") is { } folder)
            await DevTests.AgentAsync(folder, ValueOf(args, "--prompt") ?? "Describe this project in one sentence.", ValueOf(args, "--model"));

        try
        {
            switch (ValueOf(args, "--page"))
            {
                case "settings":
                    window.ShowSettings("general");
                    break;
                case "models":
                case "hf":
                case "installed":
                    window.ShowSettings("models");
                    break;
                case "api":
                    window.ShowSettings("api");
                    break;
                case "agent":
                case "design":
                case "studio":
                    window.ShowSection(ValueOf(args, "--page")!);
                    break;
            }
            await Task.Delay(int.TryParse(ValueOf(args, "--wait"), out var ms) ? ms : 2500);

            var root = (UIElement)window.Content;
            var bitmap = new RenderTargetBitmap();
            await bitmap.RenderAsync(root);
            var pixels = (await bitmap.GetPixelsAsync()).ToArray();
            var dpi = 96 * root.XamlRoot.RasterizationScale;

            await using (var file = File.Create(path))
            {
                var encoder = await BitmapEncoder.CreateAsync(BitmapEncoder.PngEncoderId, file.AsRandomAccessStream());
                encoder.SetPixelData(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied,
                    (uint)bitmap.PixelWidth, (uint)bitmap.PixelHeight, dpi, dpi, pixels);
                await encoder.FlushAsync();
            }
            Log.Info($"Captured {path}");
        }
        catch (Exception e)
        {
            Log.Error("Capture failed", e);
        }

        if (args.Contains("--exit"))
            window.Close();
    }

    private static async Task SelfTestAsync(string catalogId, string format, string? quant, string? prompt)
    {
        try
        {
            var model = Ioc.Catalog.Models.First(m => m.Id == catalogId);
            var source = model.Sources.First(s => s.Format == format);
            var variants = format == "litertlm"
                ? await Ioc.HuggingFace.GetLiteRtVariantsAsync(source.Repo, CancellationToken.None)
                : await Ioc.HuggingFace.GetGgufVariantsAsync(source.Repo, CancellationToken.None);
            Log.Info($"SELFTEST variants: {string.Join(", ", variants.Select(v => $"{v.Quant} {Fmt.Bytes(v.TotalBytes)} {string.Join("+", v.Files.Select(f => f.FileName))}"))}");
            var variant = variants.FirstOrDefault(v => v.Quant == quant) ?? variants.FirstOrDefault(v => v.Quant == source.PreferredQuant) ?? variants[0];
            var id = InstalledModel.MakeId(variant.Repo, variant.Quant);
            Log.Info($"SELFTEST {id}: {variant.Files.Count} file(s), mmproj={variant.Mmproj?.FileName}, {Fmt.Bytes(variant.TotalBytes)}");

            if (Ioc.Models.Find(id) is null)
            {
                var job = Ioc.Downloads.Start(variant, model.Name, model.Id, model.Tags, model.ContextLength);
                var clock = System.Diagnostics.Stopwatch.StartNew();
                while (Ioc.Downloads.Find(id) is not null)
                {
                    await Task.Delay(1000);
                    if (clock.ElapsedMilliseconds > 15000)
                    {
                        Log.Info($"SELFTEST download {job.Percent:F1}% {job.Status}");
                        clock.Restart();
                    }
                }
                if (!string.IsNullOrEmpty(job.Status) && !job.Status.Contains('/'))
                    Log.Info($"SELFTEST download ended: {job.Status}");
            }
            var installed = Ioc.Models.Find(id) ?? throw new InvalidOperationException("model not installed after download");

            var chat = Ioc.Chat;
            chat.NewChatCommand.Execute(null);
            chat.SelectedModel = installed;
            chat.Draft = prompt ?? "In two sentences, what makes Lisbon worth visiting?";
            var watch = System.Diagnostics.Stopwatch.StartNew();
            await chat.SendCommand.ExecuteAsync(null);
            var reply = chat.Messages.LastOrDefault();
            Log.Info($"SELFTEST reply in {watch.Elapsed.TotalSeconds:F1}s (incl. load) status='{reply?.Status}': {reply?.Content}");
        }
        catch (Exception e)
        {
            Log.Error("SELFTEST failed", e);
        }
    }

    private static string? ValueOf(string[] args, string name)
    {
        var i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }
}
#endif
