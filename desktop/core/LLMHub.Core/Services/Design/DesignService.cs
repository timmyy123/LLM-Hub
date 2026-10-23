using System.Collections.ObjectModel;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services.Design;

public sealed record DesignMessage(string Role, string Text);

/// <summary>A design project on disk: design/&lt;id&gt;/v1.html, v2.html, … plus project.json.</summary>
public sealed partial class DesignProject : ObservableObject
{
    public DesignProject() => Name = "";

    public string Id { get; init; } = Guid.NewGuid().ToString("N")[..12];
    [ObservableProperty] public partial string Name { get; set; }
    public DesignKind Kind { get; set; } = DesignKind.Prototype;
    public string SystemId { get; set; } = DesignSystem.All[0].Id;
    public int Versions { get; set; }
    public DateTimeOffset UpdatedAt { get; set; } = DateTimeOffset.Now;
    public List<DesignMessage> History { get; set; } = [];

    public string Folder => Directory.CreateDirectory(Path.Combine(AppPaths.Root, "design", Id)).FullName;
    public string VersionFile(int n) => Path.Combine(Folder, $"v{n}.html");
    public string? CurrentFile => Versions > 0 ? VersionFile(Versions) : null;
    public DesignSystem System => DesignSystem.All.FirstOrDefault(s => s.Id == SystemId) ?? DesignSystem.All[0];
}

/// <summary>
/// The Design tab's engine, after open-design's approach: the local model writes one complete, self-contained HTML document
/// inside an &lt;artifact&gt; block, shown in a sandboxed preview. Each result is saved as a new version; edits send the current
/// document back with the request (and the element the user clicked, when there is one).
/// </summary>
public sealed partial class DesignService : ObservableObject
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    public DesignService()
    {
        Status = "";
        foreach (var dir in Directory.Exists(Root) ? Directory.EnumerateDirectories(Root) : [])
        {
            try
            {
                if (JsonSerializer.Deserialize<DesignProject>(File.ReadAllText(Path.Combine(dir, "project.json"))) is { } p)
                    Projects.Add(p);
            }
            catch (Exception)
            {
            }
        }
        var sorted = Projects.OrderByDescending(p => p.UpdatedAt).ToList();
        Projects.Clear();
        foreach (var p in sorted)
            Projects.Add(p);
    }

    private static string Root => Path.Combine(AppPaths.Root, "design");

    public ObservableCollection<DesignProject> Projects { get; } = [];
    [ObservableProperty] public partial string Status { get; set; }

    public DesignProject Create(DesignKind kind, string systemId)
    {
        var project = new DesignProject { Kind = kind, SystemId = systemId, Name = Loc.S("DesignUntitled") };
        Projects.Insert(0, project);
        Save(project);
        return project;
    }

    public void Delete(DesignProject project)
    {
        Projects.Remove(project);
        try
        {
            Directory.Delete(project.Folder, true);
        }
        catch (IOException)
        {
        }
    }

    public void Save(DesignProject project)
    {
        project.UpdatedAt = DateTimeOffset.Now;
        File.WriteAllText(Path.Combine(project.Folder, "project.json"), JsonSerializer.Serialize(project, Json));
    }

    /// <summary>Generates a new version from <paramref name="request"/>. Streams progress through <paramref name="onText"/>.</summary>
    public async Task<int> GenerateAsync(DesignProject project, string request, string? selectedElement, InstalledModel model, Action<string> onText, CancellationToken ct)
    {
        if (!Ioc.Engines.IsReady(model, 32768))
        {
            Status = Loc.F("LoadingModel", model.Name);
            await Ioc.Engines.LoadAsync(model, ct, 32768);
        }
        Status = Loc.S("DesignWorking");
        var turns = new List<ChatTurn> { new("system", SystemPrompt(project)) };
        foreach (var m in project.History.TakeLast(6))
            turns.Add(new ChatTurn(m.Role, m.Text));
        var user = new StringBuilder();
        if (project.CurrentFile is { } current && File.Exists(current))
        {
            user.AppendLine("Here is the current document. Apply the change and return the whole updated document in one <artifact> block.");
            user.AppendLine("<current>").AppendLine(await File.ReadAllTextAsync(current, ct)).AppendLine("</current>");
        }
        if (selectedElement is { Length: > 0 })
            user.AppendLine($"The user selected this element in the preview: {selectedElement}. Focus the change there.");
        user.Append("Request: ").Append(request);
        turns.Add(new ChatTurn("user", user.ToString()));

        var text = new StringBuilder();
        var splitter = new ThinkSplitter();
        await foreach (var delta in Ioc.Engines.StreamAsync(model, turns, ct))
        {
            if (delta.Content is { } c)
            {
                var (answer, _) = splitter.Push(c);
                text.Append(answer);
                onText(text.ToString());
            }
        }
        var html = ExtractHtml(text.ToString()) ?? throw new InvalidOperationException(Loc.S("DesignNoHtml"));
        project.Versions++;
        await File.WriteAllTextAsync(project.VersionFile(project.Versions), html, ct);
        var note = Summary(text.ToString());
        project.History.Add(new DesignMessage("user", request));
        project.History.Add(new DesignMessage("assistant", note.Length > 0 ? note : "Updated the design."));
        if (project.Versions == 1)
            project.Name = Title(html) ?? (request.Length <= 40 ? request : request[..37] + "…");
        Save(project);
        Status = "";
        return project.Versions;
    }

    private static string SystemPrompt(DesignProject project)
    {
        var kind = project.Kind switch
        {
            DesignKind.Slides => """
                Make a slide deck as ONE HTML document: each slide is a <section class="slide"> sized 1280x720 (16:9), stacked
                vertically with 24 px gaps on a neutral page background, so every slide is visible when scrolling. Title slide first,
                5 to 8 slides, one idea per slide, large type (slide titles 48 px or more), real content, no lorem ipsum.
                """,
            DesignKind.Motion => """
                Make an animated piece as ONE HTML document using CSS keyframes (and small inline JavaScript only if needed): a 1280x720
                stage that loops smoothly, with the motion described in the request. No external libraries.
                """,
            DesignKind.Image => """
                Make a single poster or social image as ONE HTML document: a fixed-size canvas (1080x1350 unless the request says
                otherwise) built from HTML and CSS only (shapes, gradients, type), centered on the page.
                """,
            _ => """
                Make a high-fidelity web prototype as ONE HTML document: semantic HTML and a <style> block, responsive from 360 px
                phones to wide desktops, real copy written for the request (no lorem ipsum), accessible contrast, hover states.
                """,
        };
        return $"""
            You are an expert product and visual designer who writes production-quality HTML and CSS.
            {kind}
            {project.System.Brief}
            Rules:
            - Output exactly one block: <artifact type="text/html"> then a complete document starting with <!doctype html>, then </artifact>.
            - Everything inline: CSS in <style>, no external JS libraries, images as CSS shapes or gradients (no external image URLs).
            - Add data-od-id="short-name" to major sections and key elements so the user can point at them.
            - Before the artifact, write one or two sentences saying what you made or changed. Nothing after the artifact.
            """;
    }

    private static string? ExtractHtml(string text)
    {
        var m = ArtifactRegex().Match(text);
        if (m.Success)
            return m.Groups[1].Value.Trim();
        // Some models skip the wrapper: accept a bare document or a fenced html block.
        var fence = FenceRegex().Match(text);
        if (fence.Success && fence.Groups[1].Value.Contains("<html", StringComparison.OrdinalIgnoreCase))
            return fence.Groups[1].Value.Trim();
        var start = text.IndexOf("<!doctype", StringComparison.OrdinalIgnoreCase);
        var end = text.LastIndexOf("</html>", StringComparison.OrdinalIgnoreCase);
        return start >= 0 && end > start ? text[start..(end + 7)] : null;
    }

    /// <summary>The model's note before the artifact ("I used Editorial Warm…").</summary>
    private static string Summary(string text)
    {
        var i = text.IndexOf("<artifact", StringComparison.OrdinalIgnoreCase);
        var note = (i >= 0 ? text[..i] : text.Split("```")[0]).Trim();
        return note.Length > 600 ? note[..600] : note;
    }

    private static string? Title(string html)
    {
        var m = TitleRegex().Match(html);
        return m.Success && m.Groups[1].Value.Trim().Length > 0 ? System.Net.WebUtility.HtmlDecode(m.Groups[1].Value.Trim()) : null;
    }

    // ---------------------------------------------------------------- preview

    /// <summary>
    /// Injected into every preview: hovering outlines an element, clicking selects it and tells the app which one
    /// (its data-od-id, tag and text) through <paramref name="postMessage"/>, the web view's message function,
    /// so the next change can be aimed there.
    /// </summary>
    public static string PickerScript(string postMessage) => PickerTemplate.Replace("POST(", postMessage + "(");

    private const string PickerTemplate = """
        (() => {
          if (window.__llmhubPicker) return; window.__llmhubPicker = true;
          const style = document.createElement('style');
          style.textContent = '[data-llmhub-hover]{outline:2px dashed #8ab4ff!important;outline-offset:2px}[data-llmhub-picked]{outline:2px solid #8ab4ff!important;outline-offset:3px}';
          document.head.appendChild(style);
          let hovered = null;
          document.addEventListener('mouseover', e => { if (hovered) hovered.removeAttribute('data-llmhub-hover'); hovered = e.target; hovered.setAttribute('data-llmhub-hover', ''); });
          document.addEventListener('click', e => {
            e.preventDefault(); e.stopPropagation();
            document.querySelectorAll('[data-llmhub-picked]').forEach(x => x.removeAttribute('data-llmhub-picked'));
            const el = e.target.closest('[data-od-id]') || e.target;
            el.setAttribute('data-llmhub-picked', '');
            POST({ id: el.getAttribute('data-od-id') || '', tag: el.tagName.toLowerCase(), text: (el.innerText || '').trim().slice(0, 80) });
          }, true);
        })();
        """;

    /// <summary>"button · cta · "Book a table"" from the picker's message, for <see cref="ViewModels.DesignViewModel.SelectedElement"/>.</summary>
    public static string DescribePicked(string json)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        var root = doc.RootElement;
        string? Get(string name) => root.TryGetProperty(name, out var v) ? v.GetString() : null;
        var text = Get("text");
        return string.Join(" · ", new[] { Get("tag"), Get("id"), text is { Length: > 0 } ? $"\"{text}\"" : null }.Where(s => !string.IsNullOrEmpty(s)));
    }

    // ---------------------------------------------------------------- export

    /// <summary>The current version as one HTML file, a ZIP of every version, or Markdown text of the page.</summary>
    public static async Task ExportAsync(DesignProject project, string format, string target)
    {
        var current = project.CurrentFile ?? throw new InvalidOperationException(Loc.S("DesignNothingYet"));
        switch (format)
        {
            case "html":
                File.Copy(current, target, true);
                break;
            case "zip":
                if (File.Exists(target))
                    File.Delete(target);
                using (var zip = ZipFile.Open(target, ZipArchiveMode.Create))
                {
                    for (var n = 1; n <= project.Versions; n++)
                        zip.CreateEntryFromFile(project.VersionFile(n), n == project.Versions ? "index.html" : $"versions/v{n}.html");
                }
                break;
            case "md":
                await File.WriteAllTextAsync(target, HtmlToMarkdown(await File.ReadAllTextAsync(current)));
                break;
            default:
                throw new ArgumentException(format);
        }
    }

    private static string HtmlToMarkdown(string html)
    {
        var body = StyleRegex().Replace(html, "");
        body = HeadingRegex().Replace(body, m => "\n" + new string('#', int.Parse(m.Groups[1].Value)) + " " + m.Groups[2].Value + "\n");
        body = ListItemRegex().Replace(body, m => "\n- " + m.Groups[1].Value);
        body = BlockRegex().Replace(body, "\n\n");
        body = TagRegex().Replace(body, "");
        body = System.Net.WebUtility.HtmlDecode(body);
        return SpacesRegex().Replace(BlankLinesRegex().Replace(body, "\n\n"), " ").Trim() + "\n";
    }

    [GeneratedRegex(@"<artifact[^>]*>(.*?)</artifact>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex ArtifactRegex();
    [GeneratedRegex(@"```(?:html)?\s*(.*?)```", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex FenceRegex();
    [GeneratedRegex(@"<title>(.*?)</title>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex TitleRegex();
    [GeneratedRegex(@"<(style|script|head)[^>]*>.*?</\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex StyleRegex();
    [GeneratedRegex(@"<h([1-6])[^>]*>(.*?)</h\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex HeadingRegex();
    [GeneratedRegex(@"<li[^>]*>(.*?)</li>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex ListItemRegex();
    [GeneratedRegex(@"</?(p|div|section|header|footer|main|article|br|tr|table)[^>]*>", RegexOptions.IgnoreCase)]
    private static partial Regex BlockRegex();
    [GeneratedRegex("<[^>]+>")]
    private static partial Regex TagRegex();
    [GeneratedRegex(@"\n\s*\n+")]
    private static partial Regex BlankLinesRegex();
    [GeneratedRegex(@"[ \t]+")]
    private static partial Regex SpacesRegex();
}
