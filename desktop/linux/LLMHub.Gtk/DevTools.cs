#if DEBUG
using LLMHub.Services;

namespace LLMHub.Linux;

/// <summary>
/// Debug builds only, same flags as Windows and macOS:
/// <c>llmhub [--selftest catalogId [--format litertlm] [--quant label] [--prompt text]] [--capture out.png] [--page chat|settings|models|api] [--wait ms] [--theme dark|light] [--exit]</c>,
/// or <c>llmhub --uitest [filter]</c> for the UI test suite (<see cref="UiTests"/>).
/// Capture renders the window itself to PNG (GTK snapshot), so it works without screen-capture permissions.
/// </summary>
internal static class DevTools
{
    public static async void RunIfRequested(MainWindow window, string[] args)
    {
        if (Value(args, "--theme") is { } theme)
            Adw.StyleManager.GetDefault().SetColorScheme(theme == "dark" ? Adw.ColorScheme.ForceDark : Adw.ColorScheme.ForceLight);
        if (Value(args, "--speechtest") is { } sentence)
        {
            await DevTests.SpeechAsync(sentence);
            if (args.Contains("--exit"))
                window.Window.Close();
            return;
        }
        if (args.Contains("--speechmatrix"))
        {
            await DevTests.SpeechMatrixAsync();
            if (args.Contains("--exit"))
                window.Window.Close();
            return;
        }
        if (Value(args, "--searchtest") is { } searchMode)
        {
            await DevTests.MediaSearchAsync(searchMode, Value(args, "--folder")!, Value(args, "--model") ?? "embeddinggemma-2-740m",
                (Value(args, "--queries") ?? "").Split('|', StringSplitOptions.RemoveEmptyEntries));
            if (args.Contains("--exit"))
                window.Window.Close();
            return;
        }
        if (Value(args, "--exportdesign") is { } exportFormat)
        {
            await DevTests.DesignExportAsync(exportFormat, Value(args, "--out") ?? $"/tmp/design.{exportFormat}", Value(args, "--kind"));
            if (args.Contains("--exit"))
                window.Window.Close();
            return;
        }
        if (Value(args, "--magentatest") is { } liveSeconds)
        {
            await DevTests.MagentaLiveAsync(int.Parse(liveSeconds));
            if (args.Contains("--exit"))
                window.Window.Close();
            return;
        }
        if (Value(args, "--musictest") is { } clipModel)
        {
            await DevTests.MusicAsync(clipModel, Value(args, "--prompt") ?? "warm lo-fi hip hop with rhodes piano, 85 bpm",
                int.TryParse(Value(args, "--seconds"), out var clipSeconds) ? clipSeconds : 30);
            if (args.Contains("--exit"))
                window.Window.Close();
            return;
        }
        if (args.Contains("--uitest"))
        {
            var filter = Value(args, "--uitest");
            UiTests.Run(window, filter is null || filter.StartsWith("--") ? null : filter);
            return;
        }
        if (!args.Contains("--capture") && !args.Contains("--selftest"))
            return;

        if (Value(args, "--selftest") is { } id)
            await SelfTest(window, id, Value(args, "--format") ?? "gguf", Value(args, "--quant"), Value(args, "--prompt"));
        if (Value(args, "--studiotest") is { } workflow)
            await DevTests.StudioAsync(workflow, Value(args, "--prompt") ?? "a cozy cafe interior in morning light, film photo", Value(args, "--image"));
        if (Value(args, "--designtest") is { } request)
            await DevTests.DesignAsync(request, Value(args, "--kind"), Value(args, "--model"), Value(args, "--followup"));
        if (Value(args, "--agenttest") is { } folder)
            await DevTests.AgentAsync(folder, Value(args, "--prompt") ?? "Describe this project in one sentence.", Value(args, "--model"), Value(args, "--engine") ?? "claude", Value(args, "--followup"));

        switch (Value(args, "--page"))
        {
            case "settings": window.ShowSettings("general"); break;
            case "models": window.ShowSettings("models"); break;
            case "api": window.ShowSettings("api"); break;
            case "engines" or "speech" or "studiosettings" or "agentsettings" or "websearch" or "storage" or "general":
                window.ShowSettings(Value(args, "--page")!.Replace("settings", ""));
                break;
            case "agent" or "design" or "studio": window.ShowSection(Value(args, "--page")!); break;
            case "voice" or "music" or "video":
                Ioc.StudioVm.Mode = Value(args, "--page")!;
                window.ShowSection("studio");
                break;
            default: window.ShowChat(); break;
        }
        await Task.Delay(int.TryParse(Value(args, "--wait"), out var ms) ? ms : 2500);
        if (Value(args, "--capture") is { } path)
        {
            Capture(window.Window, path);
            if (Value(args, "--page") == "design")
                await window.Design.ProbePreviewAsync();
        }
        if (args.Contains("--exit"))
            window.Window.Close();
    }

    public static void Capture(Gtk.Window window, string path)
    {
        var width = window.GetWidth();
        var height = window.GetHeight();
        var paintable = Gtk.WidgetPaintable.New(window);
        var snapshot = Gtk.Snapshot.New();
        paintable.Snapshot(snapshot, width, height);
        var node = snapshot.ToNode();
        if (node is null || window.GetRenderer() is not { } renderer)
        {
            Log.Error("Capture: nothing to render");
            return;
        }
        var texture = renderer.RenderTexture(node, null);
        texture.SaveToPng(path);
        Log.Info($"Captured {path}");
    }

    private static async Task SelfTest(MainWindow window, string catalogId, string format, string? quant, string? prompt)
    {
        try
        {
            var model = Ioc.Catalog.Models.First(m => m.Id == catalogId);
            var source = model.Sources.First(s => s.Format == format);
            var variants = format == "litertlm"
                ? await Ioc.HuggingFace.GetLiteRtVariantsAsync(source.Repo, CancellationToken.None)
                : await Ioc.HuggingFace.GetGgufVariantsAsync(source.Repo, CancellationToken.None);
            var variant = variants.FirstOrDefault(v => v.Quant == quant) ?? variants.FirstOrDefault(v => v.Quant == source.PreferredQuant) ?? variants[0];
            var id = InstalledModel.MakeId(variant.Repo, variant.Quant);
            Log.Info($"SELFTEST {id}: {Fmt.Bytes(variant.TotalBytes)}");
            if (Ioc.Models.Find(id) is null)
            {
                var job = Ioc.Downloads.Start(variant, model.Name, model.Id, model.Tags, model.ContextLength);
                var last = DateTime.UtcNow;
                while (Ioc.Downloads.Find(id) is not null)
                {
                    await Task.Delay(1000);
                    if ((DateTime.UtcNow - last).TotalSeconds > 15)
                    {
                        Log.Info($"SELFTEST download {job.Percent:F1}% {job.Status}");
                        last = DateTime.UtcNow;
                    }
                }
            }
            var installed = Ioc.Models.Find(id) ?? throw new InvalidOperationException("model not installed after download");

            window.ShowChat();
            Ioc.Chat.NewChatCommand.Execute(null);
            Ioc.Chat.SelectedModel = installed;
            window.Chat.InputText = prompt ?? "In two sentences, what makes Lisbon worth visiting?";
            var started = DateTime.UtcNow;
            window.Chat.Send();
            while (Ioc.Chat.IsGenerating)
                await Task.Delay(250);
            var reply = Ioc.Chat.Messages.LastOrDefault();
            Log.Info($"SELFTEST reply in {(DateTime.UtcNow - started).TotalSeconds:F1}s (incl. load) status='{reply?.Status}': {reply?.Content}");
        }
        catch (Exception e)
        {
            Log.Error("SELFTEST failed", e);
        }
    }

    /// <summary>Round trip: Supertonic reads the sentence into a WAV, Whisper writes it back. Logs both timings.</summary>
    private static string? Value(string[] args, string name)
    {
        var i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }
}
#endif
