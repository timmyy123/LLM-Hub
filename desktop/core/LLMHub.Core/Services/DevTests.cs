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
    public static async Task AgentAsync(string folder, string prompt, string? modelId)
    {
        try
        {
            var agent = Ioc.Agent;
            if (!agent.IsClaudeInstalled)
            {
                await agent.InstallClaudeAsync();
                Log.Info($"AGENTTEST install: {agent.InstallStatus} installed={agent.IsClaudeInstalled}");
            }
            var model = Ioc.Models.Installed.FirstOrDefault(m => m.Id == modelId || m.CatalogId == modelId) ?? Ioc.Models.Installed.First(m => m.Format == "gguf");
            var session = agent.CreateSession(folder, model.Id, "default");
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
            Log.Info($"DESIGNTEST v{design.Version} in {clock.Elapsed.TotalSeconds:F0}s: {(file is null ? "no file" : $"{new FileInfo(file).Length} bytes {file}")} status='{design.Status}' name='{design.Selected?.Name}'");
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
}
