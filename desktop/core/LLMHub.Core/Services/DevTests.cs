namespace LLMHub.Services;

/// <summary>
/// End-to-end checks the debug builds run from the command line (--agenttest, --designtest) on real hardware:
/// a real model, real Claude Code session or design generation, results in the log.
/// </summary>
public static class DevTests
{
    /// <summary>
    /// Runs one Agent turn in "ask before edits" mode, approving each permission request (and logging it), so the whole
    /// Claude Code → bridge → MCP permission → model path is exercised.
    /// </summary>
    public static async Task AgentAsync(string folder, string prompt, string? modelId, string engine = "claude", string? followUp = null)
    {
        try
        {
            var agent = Ioc.Agent;
            if (engine == "openclaw" && !Agent.OpenClawService.IsInstalled)
            {
                var install = System.Diagnostics.Stopwatch.StartNew();
                Ioc.OpenClaw.PropertyChanged += (_, e) =>
                {
                    if (e.PropertyName == nameof(Agent.OpenClawService.Status) && Ioc.OpenClaw.Status.Length > 0)
                        Log.Info($"AGENTTEST {Ioc.OpenClaw.Status}");
                };
                await Ioc.OpenClaw.InstallAsync(CancellationToken.None);
                Log.Info($"AGENTTEST OpenClaw installed in {install.Elapsed.TotalSeconds:F0}s");
            }
            else if (engine != "openclaw" && !agent.IsClaudeInstalled)
            {
                await agent.InstallClaudeAsync();
                Log.Info($"AGENTTEST install: {agent.InstallStatus} installed={agent.IsClaudeInstalled}");
            }
            var model = Ioc.Models.Installed.FirstOrDefault(m => m.Id == modelId || m.CatalogId == modelId) ?? Ioc.Models.Installed.First(m => m.Format == "gguf");
            var session = agent.CreateSession(folder, model.Id, "default", engine);
            UiThread.Run(() => Ioc.AgentVm.Selected = session);
            session.PropertyChanged += (_, e) =>
            {
                if (e.PropertyName == nameof(Agent.AgentSession.Pending) && session.Pending is { } p)
                {
                    Log.Info($"AGENTTEST permission: {p.Question} {p.Summary}");
                    agent.Answer(session, allow: true);
                }
            };
            var clock = System.Diagnostics.Stopwatch.StartNew();
            await agent.SendAsync(session, prompt);
            if (followUp is not null)
            {
                Log.Info($"AGENTTEST first turn in {clock.Elapsed.TotalSeconds:F1}s status={session.Status}");
                await agent.SendAsync(session, followUp);
            }
            foreach (var item in session.Items)
                Log.Info($"AGENTTEST {item.Kind} {item.Detail} {item.Title[..Math.Min(item.Title.Length, 160)].Replace('\n', ' ')} {item.Meta}");
            foreach (var task in session.Tasks)
                Log.Info($"AGENTTEST task [{task.Status}] {task.Text}");
            foreach (var change in session.Changes)
                Log.Info($"AGENTTEST change {change.Kind} {change.Path} {change.Stats}");
            Log.Info($"AGENTTEST done in {clock.Elapsed.TotalSeconds:F1}s status={session.Status}");
        }
        catch (Exception e)
        {
            Log.Error("AGENTTEST failed", e);
        }
    }

    /// <summary>Creates a design project, generates version 1 (and optionally a follow-up change as version 2), and logs the result.</summary>
    public static async Task DesignAsync(string request, string? kind, string? modelId, string? followUp)
    {
        var design = Ioc.DesignVm;
        if (modelId is not null && Ioc.Models.Installed.FirstOrDefault(m => m.Id == modelId || m.CatalogId == modelId) is { } model)
            design.SelectedModel = model;
        design.Kind = design.Kinds.First(k => k.Kind.ToString().Equals(kind ?? "Prototype", StringComparison.OrdinalIgnoreCase));
        design.NewProjectCommand.Execute(null);
        foreach (var text in new[] { request, followUp }.Where(t => t is not null))
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            design.Draft = text!;
            await design.SendCommand.ExecuteAsync(null);
            var file = design.PreviewFile;
            Log.Info($"DESIGNTEST v{design.Version} in {clock.Elapsed.TotalSeconds:F0}s: {(file is null ? "no file" : $"{new FileInfo(file).Length} bytes {file}")} status='{design.Status}' name='{design.Selected?.Name}' steps='{design.Messages.LastOrDefault()?.StepsText.Replace('\n', ';')}'");
        }
    }

    /// <summary>
    /// Generates with one Studio workflow through the real view model (installing ComfyUI and models if needed). The
    /// Snapdragon NPU workflows ("npu-absolutereality", ...) do image to image when <paramref name="image"/> is given.
    /// </summary>
    public static async Task StudioAsync(string workflowId, string prompt, string? image, string? seed = null)
    {
        var studio = Ioc.StudioVm;
        studio.Workflow = Studio.ComfyWorkflow.All.Concat(Studio.ComfyWorkflow.NpuWorkflows).First(w => w.Id == workflowId);
        studio.Mode = studio.Workflow.Kind;
        studio.Prompt = prompt;
        if (image is not null)
            studio.ReferenceImage = image;
        if (seed is not null)
            studio.Seed = seed;
        var last = DateTime.UtcNow;
        void Report(string status)
        {
            if (status.Length > 0 && (DateTime.UtcNow - last).TotalSeconds > 20)
            {
                Log.Info($"STUDIOTEST {status}");
                last = DateTime.UtcNow;
            }
        }
        Ioc.Comfy.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(Studio.ComfyService.Status))
                Report(Ioc.Comfy.Status);
        };
        Ioc.SdNpu.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(Studio.Npu.SdNpuService.Status))
                Report(Ioc.SdNpu.Status);
        };
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await studio.GenerateCommand.ExecuteAsync(null);
        Log.Info($"STUDIOTEST {workflowId} in {clock.Elapsed.TotalSeconds:F0}s: {studio.Status}; newest {studio.Gallery.FirstOrDefault()?.Path}");
    }

    /// <summary>Speaks a sentence with Supertonic, then transcribes the audio with whisper.cpp (both downloaded on first use).</summary>
    public static async Task SpeechAsync(string sentence)
    {
        try
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var (wav, seconds) = await Ioc.Speech.SynthesizeAsync(sentence, "F1", null, CancellationToken.None);
            Log.Info($"SPEECHTEST tts {seconds:F1}s of audio in {clock.Elapsed.TotalSeconds:F1}s -> {wav}");
            clock.Restart();
            var text = await Ioc.Speech.TranscribeAsync(wav, null, CancellationToken.None);
            Log.Info($"SPEECHTEST asr in {clock.Elapsed.TotalSeconds:F1}s: {text}");
        }
        catch (Exception e)
        {
            Log.Error("SPEECHTEST failed", e);
        }
    }

    /// <summary>
    /// Supertonic parity with Android: new voices and languages on the FP32 and FP16 models, and an imported voice.
    /// English results go back through Whisper so a garbled voice shows up as a wrong transcript.
    /// </summary>
    public static async Task SpeechMatrixAsync()
    {
        var saved = (AppSettings.Current.SpeechPrecision, AppSettings.Current.SpeechLanguage);
        try
        {
            foreach (var precision in new[] { "fp32", "fp16" })
            {
                AppSettings.Current.SpeechPrecision = precision;
                Ioc.Speech.ReloadModel();
                foreach (var (voice, lang, text) in new[]
                         {
                             ("F3", "en", "The quick brown fox jumps over the lazy dog."),
                             ("M4", "ja", "今日はとても良い天気ですね。"),
                             ("F5", "de", "Guten Morgen, wie geht es dir heute?"),
                         })
                {
                    AppSettings.Current.SpeechLanguage = lang;
                    var clock = System.Diagnostics.Stopwatch.StartNew();
                    var (wav, seconds) = await Ioc.Speech.SynthesizeAsync(text, voice, null, CancellationToken.None);
                    var line = $"SPEECHMATRIX {precision} {voice} {lang}: {seconds:F1}s audio in {clock.Elapsed.TotalSeconds:F1}s";
                    if (lang == "en")
                        line += $" -> \"{await Ioc.Speech.TranscribeAsync(wav, "en", CancellationToken.None)}\"";
                    Log.Info(line);
                }
            }
            AppSettings.Current.SpeechLanguage = "";
            var copy = Path.Combine(Path.GetTempPath(), "Narrator.json");
            File.Copy(Path.Combine(Speech.SpeechService.TtsRoot, "voice_styles", "M2.json"), copy, true);
            var name = Speech.SpeechService.ImportVoice(copy);
            var (custom, len) = await Ioc.Speech.SynthesizeAsync("This is an imported voice.", name, null, CancellationToken.None);
            Log.Info($"SPEECHMATRIX imported '{name}': {len:F1}s -> \"{await Ioc.Speech.TranscribeAsync(custom, "en", CancellationToken.None)}\"; voices: {string.Join(",", Speech.SpeechService.Voices)}");
            Speech.SpeechService.DeleteVoice(name);
            try
            {
                Speech.SpeechService.ImportVoice(Path.Combine(Speech.SpeechService.TtsRoot, "onnx", "tts.json"));
                Log.Info("SPEECHMATRIX invalid voice accepted (wrong)");
            }
            catch (InvalidDataException)
            {
                Log.Info("SPEECHMATRIX invalid voice rejected (right)");
            }
        }
        catch (Exception e)
        {
            Log.Error("SPEECHMATRIX failed", e);
        }
        finally
        {
            (AppSettings.Current.SpeechPrecision, AppSettings.Current.SpeechLanguage) = saved;
            Ioc.Speech.ReloadModel();
        }
    }

    /// <summary>Generates a music clip through the real Studio view model (installing ComfyUI and the model if needed).</summary>
    public static async Task MusicAsync(string workflowId, string prompt, int seconds)
    {
        var studio = Ioc.StudioVm;
        studio.MusicWorkflow = Studio.ComfyWorkflow.Music.First(w => w.Id == workflowId);
        studio.MusicPrompt = prompt;
        studio.MusicSeconds = seconds;
        studio.Mode = "music";
        var last = DateTime.UtcNow;
        Ioc.Comfy.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(Studio.ComfyService.Status) && (DateTime.UtcNow - last).TotalSeconds > 20)
            {
                Log.Info($"MUSICTEST {Ioc.Comfy.Status}");
                last = DateTime.UtcNow;
            }
        };
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await studio.GenerateMusicCommand.ExecuteAsync(null);
        var clip = studio.MusicClips.FirstOrDefault()?.Path;
        Log.Info($"MUSICTEST {workflowId} {seconds}s in {clock.Elapsed.TotalSeconds:F0}s: {studio.MusicStatus}; {clip} ({(clip is null ? 0 : new FileInfo(clip).Length)} bytes)");
    }

    /// <summary>
    /// Studio › Search: installs an EmbeddingGemma 2 build from the catalog if needed, indexes a folder of photos, audio or
    /// videos, then logs the best matches for each query (text, or a photo path for photo-to-photo). A query can name its
    /// right answer after "=>": a file name, or for videos "file.mp4@20-30", where the best moment must start between
    /// those seconds. Each such query logs PASS or FAIL, then a total: <c>SEARCHTEST RESULT 4/4 passed</c>.
    /// </summary>
    public static async Task MediaSearchAsync(string mode, string folder, string catalogId, IReadOnlyList<string> queries)
    {
        try
        {
            var model = Ioc.Catalog.Models.First(m => m.Id == catalogId);
            var source = model.Sources.First(s => s.Format == "litertlm");
            // An installed build is used as it is, without asking the Hub (so a flaky network does not stop the test).
            var id = Ioc.Models.Installed.FirstOrDefault(m => m.CatalogId == catalogId && m.Format == "litertlm")?.Id;
            if (id is null)
            {
                var variant = (await Ioc.HuggingFace.GetLiteRtVariantsAsync(source.Repo, CancellationToken.None))[0];
                id = InstalledModel.MakeId(variant.Repo, variant.Quant);
                var clock = System.Diagnostics.Stopwatch.StartNew();
                Ioc.Downloads.Start(variant, model.Name, model.Id, model.Tags, model.ContextLength);
                while (Ioc.Downloads.Find(id) is not null)
                    await Task.Delay(1000);
                Log.Info($"SEARCHTEST downloaded {id} ({Fmt.Bytes(variant.TotalBytes)}) in {clock.Elapsed.TotalSeconds:F0}s");
            }
            await Task.Delay(500);
            var search = Ioc.MediaSearchVm;
            search.Mode = mode;
            search.SelectedModel = search.Models.First(m => m.Id == id);
            search.SetFolder(folder);
            var watch = System.Diagnostics.Stopwatch.StartNew();
            await search.IndexCommand.ExecuteAsync(null);
            Log.Info($"SEARCHTEST index {mode} with {search.SelectedModel?.Id} on {Ioc.MediaEmbedder.Loaded?.Backend} ({Ioc.MediaEmbedder.Loaded?.MaxInput} tokens) " +
                     $"in {watch.Elapsed.TotalSeconds:F1}s: {search.Status} ({search.IndexedText})");
            int passed = 0, checkedCount = 0;
            foreach (var entry in queries)
            {
                var arrow = entry.IndexOf("=>", StringComparison.Ordinal);
                var query = arrow < 0 ? entry : entry[..arrow].Trim();
                var expected = arrow < 0 ? null : entry[(arrow + 2)..].Trim();
                watch.Restart();
                if (File.Exists(query))
                    await search.SearchByPhotoAsync(query);
                else
                {
                    search.Query = query;
                    await search.SearchCommand.ExecuteAsync(null);
                }
                var top = mode switch
                {
                    "photos" => search.Photos.Take(4).Select(p => $"{p.Name} {p.ScoreText}"),
                    "audio" => search.Audio.Take(4).Select(a => $"{a.Name} {a.ScoreText} @{string.Join("/", a.Moments.Select(m => m.Label))}"),
                    _ => search.Videos.Select(v => $"{v.Name}@{v.Label} {v.ScoreText}"),
                };
                Log.Info($"SEARCHTEST \"{Path.GetFileName(query)}\" in {watch.Elapsed.TotalSeconds:F2}s: {string.Join(" | ", top)}");
                if (expected is null)
                    continue;
                checkedCount++;
                var (ok, got) = mode switch
                {
                    "photos" => (search.Photos.FirstOrDefault()?.Name == expected, search.Photos.FirstOrDefault()?.Name),
                    "audio" => (search.Audio.FirstOrDefault()?.Name == expected, search.Audio.FirstOrDefault()?.Name),
                    _ => VideoMatches(search.Videos.FirstOrDefault(), expected),
                };
                if (ok)
                    passed++;
                Log.Info($"SEARCHTEST {(ok ? "PASS" : "FAIL")} \"{query}\": expected {expected}, got {got ?? "nothing"}");
            }
            if (checkedCount > 0)
                Log.Info($"SEARCHTEST RESULT {passed}/{checkedCount} passed");
            if (mode == "videos" && queries.Count > 0 && Ioc.MediaEmbedder.Loaded is { } longer)
                await CompareInputLengthsAsync(longer, queries[0].Split("=>")[0].Trim());
        }
        catch (Exception e)
        {
            Log.Error("SEARCHTEST failed", e);
        }
    }

    /// <summary>"name@from-to" (seconds): the top moment is in that file and starts inside the range.</summary>
    private static (bool Ok, string? Got) VideoMatches(ViewModels.VideoMomentMatch? top, string expected)
    {
        if (top is null)
            return (false, null);
        var got = $"{top.Name}@{top.StartMs / 1000.0:0.#}s ({top.ScoreText})";
        var at = expected.LastIndexOf('@');
        var name = at < 0 ? expected : expected[..at];
        if (!string.Equals(top.Name, name, StringComparison.OrdinalIgnoreCase))
            return (false, got);
        if (at < 0)
            return (true, got);
        var range = expected[(at + 1)..].Split('-');
        var from = double.Parse(range[0], System.Globalization.CultureInfo.InvariantCulture);
        var to = range.Length > 1 ? double.Parse(range[1], System.Globalization.CultureInfo.InvariantCulture) : from + 2;
        return (top.StartMs >= from * 1000 && top.StartMs < to * 1000, got);
    }

    /// <summary>
    /// The shared engine made for video moments (1024 input tokens) also answers photo, audio and API requests; checks it
    /// embeds a text like the usual 256-token engine does.
    /// </summary>
    private static async Task CompareInputLengthsAsync(LiteRtEngine.LiteRtEmbedder longer, string text)
    {
        if (longer.MaxInput == LiteRtEngine.LiteRtEmbedder.MaxInputTokens)
            return;
        using var shorter = await Ioc.LiteRt.OpenEmbedderAsync(longer.Model, _ => { }, CancellationToken.None);
        var a = await Task.Run(() => longer.EmbedText(text, query: true));
        var b = await Task.Run(() => shorter.EmbedText(text, query: true));
        Log.Info($"SEARCHTEST {longer.MaxInput} vs {shorter.MaxInput} input tokens on {longer.Backend}/{shorter.Backend}: cosine {(a is null || b is null ? -1 : MediaSearch.MediaIndex.Dot(a, b)):F4}");
    }

    /// <summary>Exports the most recent design project (or the one with <paramref name="kind"/>) to <paramref name="path"/>.</summary>
    public static async Task DesignExportAsync(string format, string path, string? kind)
    {
        var project = Ioc.Design.Projects.Where(p => p.Versions > 0 && (kind is null || p.Kind.ToString().Equals(kind, StringComparison.OrdinalIgnoreCase)))
            .OrderByDescending(p => p.UpdatedAt).FirstOrDefault();
        if (project is null)
        {
            Log.Info("EXPORTTEST no design project");
            return;
        }
        await Design.DesignService.ExportAsync(project, format, path);
        var slides = Design.DesignService.Slides(project, await File.ReadAllTextAsync(project.CurrentFile!));
        Log.Info($"EXPORTTEST {project.Name} ({project.Kind}) -> {path}: {new FileInfo(path).Length} bytes, {slides.Count} slides: {string.Join(" | ", slides.Select(s => s.Title))}");
    }

    /// <summary>
    /// Live Magenta RealTime: plays <paramref name="seconds"/> while recording, swaps the style halfway, then checks speed
    /// and that the recording holds music. Run with LLMHUB_AUDIO_SINK=null where there is no sound server.
    /// </summary>
    public static async Task MagentaLiveAsync(int seconds)
    {
        var music = Ioc.Magenta;
        var last = "";
        music.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(Studio.MagentaService.Status) && music.Status != last)
                Log.Info($"MAGENTATEST {last = music.Status}");
        };
        string? saved = null;
        music.Saved += path => saved = path;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        music.SetRecording(true);
        await music.PlayAsync(CancellationToken.None);
        if (!music.IsPlaying)
        {
            Log.Info($"MAGENTATEST failed to start: {music.Status}");
            return;
        }
        var start = clock.Elapsed.TotalSeconds;
        Log.Info($"MAGENTATEST playing on {music.Device} after {start:F0}s");
        await Task.Delay(TimeSpan.FromSeconds(seconds / 2.0));
        music.Styles[0].Weight = 0;
        music.Styles[1].Text = "Upbeat disco funk with slap bass";
        music.Styles[1].Weight = 1;
        music.Temperature = 0.7;
        await Task.Delay(TimeSpan.FromSeconds(seconds / 2.0));
        music.Stop();
        music.Export();
        var played = clock.Elapsed.TotalSeconds - start;
        Log.Info($"MAGENTATEST {music.Elapsed:F1}s played in {played:F1}s, speed {music.Speed:F2}x real time, levels max {music.Levels.Max():F2}");
        if (saved is null)
            return;
        var bytes = File.ReadAllBytes(saved);
        var samples = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, short>(bytes.AsSpan(44)).ToArray();
        var halves = new[] { samples[..(samples.Length / 2)], samples[(samples.Length / 2)..] }
            .Select(h => Math.Sqrt(h.Sum(s => (double)s * s) / Math.Max(1, h.Length))).ToArray();
        var recordings = Directory.GetFiles(ViewModels.StudioViewModel.MusicFolder, "*.wav").OrderByDescending(File.GetLastWriteTimeUtc).Take(2);
        Log.Info($"MAGENTATEST export {saved}: {bytes.Length} bytes, {(bytes.Length - 44) / 4.0 / Studio.MagentaService.SampleRate:F1}s, RMS {halves[0]:F0} / {halves[1]:F0}; files {string.Join(", ", recordings.Select(Path.GetFileName))}");
    }
}
