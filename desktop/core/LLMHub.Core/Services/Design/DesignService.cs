using System.Collections.ObjectModel;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services.Design;

/// <summary>A turn in a design project; the assistant's carries the steps it took ("Wrote index.html", "Checked layout at phone width").</summary>
public sealed record DesignMessage(string Role, string Text, List<string>? Steps = null)
{
    [System.Text.Json.Serialization.JsonIgnore] public string StepsText => Steps is { Count: > 0 } s ? string.Join("\n", s.Select(x => "✓ " + x)) : "";
}

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
        var steps = new List<string> { Loc.S(project.Versions == 0 ? "DesignStepWrote" : "DesignStepUpdated") };
        html = await RenderImagesAsync(project, html, project.Versions + 1, steps, ct);
        project.Versions++;
        await File.WriteAllTextAsync(project.VersionFile(project.Versions), html, ct);
        var note = Summary(text.ToString());
        project.History.Add(new DesignMessage("user", request));
        project.History.Add(new DesignMessage("assistant", note.Length > 0 ? note : "Updated the design.", steps));
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
            - Add data-od-id="short-name" to major sections and key elements so the user can point at them.{(project.Kind is DesignKind.Prototype or DesignKind.Slides ? """

            - For a real photo (one hero photo, two at most), write <img data-generate="a vivid one-sentence description of the photo" alt="..."> with CSS
              that sizes it (for example width:100%;height:100%;object-fit:cover). LLM Hub makes the photo on this computer. Never use other image URLs.
            """ : "")}
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

    /// <summary>
    /// Turns &lt;img data-generate="…"&gt; into photos made with ComfyUI when Studio is set up (an image model is on disk), saved
    /// next to the version as v3-image1.png; otherwise a soft gradient in the design system's colors stands in.
    /// </summary>
    private async Task<string> RenderImagesAsync(DesignProject project, string html, int version, List<string> steps, CancellationToken ct)
    {
        var matches = GenerateImageRegex().Matches(html).Take(2).ToList();
        if (matches.Count == 0)
            return html;
        var workflow = new[] { Studio.ComfyWorkflow.Sdxl, Studio.ComfyWorkflow.FluxSchnell }.FirstOrDefault(Studio.ComfyService.HasModels);
        var canGenerate = Studio.ComfyService.IsSupported && workflow is not null;
        var made = 0;
        for (var i = matches.Count - 1; i >= 0; i--)
        {
            var m = matches[i];
            var prompt = System.Net.WebUtility.HtmlDecode(m.Groups["prompt"].Value);
            string src;
            if (canGenerate)
            {
                try
                {
                    Status = Loc.S("DesignMakingImage");
                    await Ioc.Comfy.EnsureRunningAsync(ct);
                    var graph = workflow!.Build(prompt + ", " + project.System.Name.ToLowerInvariant() + " style, photograph", 1344, 768, 1, workflow.DefaultSteps,
                        Random.Shared.NextInt64(1, 1L << 48), null);
                    var files = await Ioc.Comfy.RunAsync(graph, (_, _) => { }, ct);
                    var name = $"v{version}-image{i + 1}{Path.GetExtension(files[0])}";
                    File.Copy(files[0], Path.Combine(project.Folder, name), true);
                    src = name;
                    made++;
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    Log.Error("Design image failed", e);
                    src = Placeholder(project.System);
                }
            }
            else
                src = Placeholder(project.System);
            var tag = SrcRegex().Replace(m.Value, "");
            html = html[..m.Index] + tag.Insert(4, $" src=\"{src}\"") + html[(m.Index + m.Length)..];
        }
        Status = Loc.S("DesignWorking");
        steps.Add(made > 0 ? Loc.S("DesignStepImage") : Loc.S("DesignStepPlaceholder"));
        return html;
    }

    private static string Placeholder(DesignSystem system)
    {
        var from = system.Colors.Length > 1 ? system.Colors[1] : system.Colors[0];
        var to = system.Colors.Length > 3 ? system.Colors[3] : system.Colors[^1];
        var svg = $"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1344 768'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='{from}'/><stop offset='1' stop-color='{to}'/></linearGradient></defs><rect width='1344' height='768' fill='url(#g)'/></svg>";
        return "data:image/svg+xml," + Uri.EscapeDataString(svg);
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
            const cs = getComputedStyle(el);
            POST({ type: 'pick', id: el.getAttribute('data-od-id') || '', tag: el.tagName.toLowerCase(), text: (el.innerText || '').trim().slice(0, 80),
                   fill: cs.backgroundColor, ink: cs.color, radius: cs.borderTopLeftRadius, font: cs.fontFamily.split(',')[0].replace(/["']/g, ''), size: cs.fontSize });
          }, true);
        })();
        """;

    /// <summary>
    /// Renders the page at phone width (390 px) in a hidden frame and posts how wide its content really is, so the app can
    /// say whether the layout fits a phone.
    /// </summary>
    public static string PhoneCheckScript(string postMessage) => PhoneCheckTemplate.Replace("POST(", postMessage + "(");

    private const string PhoneCheckTemplate = """
        (() => {
          const f = document.createElement('iframe');
          f.style.cssText = 'position:fixed;left:-10000px;top:0;width:390px;height:844px;border:0;visibility:hidden';
          f.srcdoc = '<!doctype html>' + document.documentElement.outerHTML;
          f.onload = () => setTimeout(() => {
            const d = f.contentDocument.documentElement;
            POST({ type: 'phone', width: Math.max(d.scrollWidth, f.contentDocument.body ? f.contentDocument.body.scrollWidth : 0) });
            f.remove();
          }, 400);
          document.body.appendChild(f);
        })();
        """;

    /// <summary>"Fill: #b4532a", "Radius: 10 px", "Font: Fraunces · 32 px" for the selected element panel.</summary>
    public static IReadOnlyList<string> PickedProperties(string json)
    {
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        var root = doc.RootElement;
        string Get(string name) => root.TryGetProperty(name, out var v) ? v.GetString() ?? "" : "";
        var list = new List<string>();
        if (CssColor(Get("fill")) is { } fill)
            list.Add(Loc.F("DesignPropFill", fill));
        if (CssColor(Get("ink")) is { } ink)
            list.Add(Loc.F("DesignPropText", ink));
        if (Get("radius") is { Length: > 0 } radius && radius != "0px")
            list.Add(Loc.F("DesignPropRadius", radius.Replace("px", " px")));
        if (Get("font") is { Length: > 0 } font)
            list.Add(Loc.F("DesignPropFont", $"{font} · {Get("size").Replace("px", " px")}"));
        return list;
    }

    /// <summary>"rgb(180, 83, 42)" as "#b4532a"; null for transparent.</summary>
    private static string? CssColor(string css)
    {
        var m = RgbRegex().Match(css);
        if (!m.Success || (m.Groups[4].Success && double.TryParse(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture, out var alpha) && alpha == 0))
            return null;
        return $"#{int.Parse(m.Groups[1].Value):x2}{int.Parse(m.Groups[2].Value):x2}{int.Parse(m.Groups[3].Value):x2}";
    }

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

    /// <summary>
    /// The current version as one self-contained HTML file, a ZIP of every version with its images, Markdown text of the
    /// page, or a PowerPoint deck (one slide per slide or section, with the generated photos).
    /// </summary>
    public static async Task ExportAsync(DesignProject project, string format, string target)
    {
        var current = project.CurrentFile ?? throw new InvalidOperationException(Loc.S("DesignNothingYet"));
        switch (format)
        {
            case "html":
                await File.WriteAllTextAsync(target, InlineImages(project, await File.ReadAllTextAsync(current)));
                break;
            case "zip":
                if (File.Exists(target))
                    File.Delete(target);
                using (var zip = ZipFile.Open(target, ZipArchiveMode.Create))
                {
                    for (var n = 1; n <= project.Versions; n++)
                    {
                        var prefix = n == project.Versions ? "" : "versions/";
                        zip.CreateEntryFromFile(project.VersionFile(n), n == project.Versions ? "index.html" : $"versions/v{n}.html");
                        foreach (var image in Directory.EnumerateFiles(project.Folder, $"v{n}-image*"))
                            zip.CreateEntryFromFile(image, prefix + Path.GetFileName(image));
                    }
                }
                break;
            case "pptx":
                PptxWriter.Write(target, project.Name, Slides(project, await File.ReadAllTextAsync(current)), project.System);
                break;
            case "md":
                await File.WriteAllTextAsync(target, HtmlToMarkdown(await File.ReadAllTextAsync(current)));
                break;
            default:
                throw new ArgumentException(format);
        }
    }

    /// <summary>Generated photos as data: URIs, so the exported file opens anywhere on its own.</summary>
    private static string InlineImages(DesignProject project, string html) => LocalImageRegex().Replace(html, m =>
    {
        var file = Path.Combine(project.Folder, m.Groups["file"].Value);
        if (!File.Exists(file))
            return m.Value;
        var mime = file.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ? "image/png" : "image/jpeg";
        return $"src=\"data:{mime};base64,{Convert.ToBase64String(File.ReadAllBytes(file))}\"";
    });

    /// <summary>Slides from the document: each &lt;section class="slide"&gt; for decks, otherwise each top-level section.</summary>
    public static IReadOnlyList<PptxSlide> Slides(DesignProject project, string html)
    {
        var body = StyleRegex().Replace(html, "");
        var parts = SlideSectionRegex().Matches(body).Select(m => m.Groups[1].Value).ToList();
        if (parts.Count == 0)
            parts = SectionRegex().Matches(body).Select(m => m.Groups[1].Value).ToList();
        if (parts.Count == 0)
            parts = [body];
        var slides = new List<PptxSlide>();
        foreach (var part in parts.Take(40))
        {
            var heading = FirstHeadingRegex().Match(part);
            var title = heading.Success ? Clean(heading.Groups[2].Value) : project.Name;
            var rest = heading.Success ? part.Remove(heading.Index, heading.Length) : part;
            var lines = LineRegex().Matches(rest).Select(m => Clean(m.Groups[2].Value)).Where(l => l.Length > 0 && !l.StartsWith('[')).Take(9)
                .Select(l => l.Length > 220 ? l[..217] + "…" : l).ToList();
            var image = LocalImageRegex().Match(part) is { Success: true } img && Path.Combine(project.Folder, img.Groups["file"].Value) is var path && File.Exists(path) ? path : null;
            if (title.Length > 0 || lines.Count > 0)
                slides.Add(new PptxSlide(title.Length > 0 ? title : project.Name, lines, image));
        }
        return slides;
    }

    private static string Clean(string html) => WhitespaceRegex().Replace(System.Net.WebUtility.HtmlDecode(TagRegex().Replace(html, " ")), " ").Trim();

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
    [GeneratedRegex(@"\s+")]
    private static partial Regex WhitespaceRegex();
    [GeneratedRegex(@"<img\b(?=[^>]*\bdata-generate=""(?<prompt>[^""]+)"")[^>]*>", RegexOptions.IgnoreCase)]
    private static partial Regex GenerateImageRegex();
    [GeneratedRegex(@"\ssrc=""[^""]*""", RegexOptions.IgnoreCase)]
    private static partial Regex SrcRegex();
    [GeneratedRegex(@"src=""(?<file>v\d+-image\d+\.(?:png|jpe?g|webp))""", RegexOptions.IgnoreCase)]
    private static partial Regex LocalImageRegex();
    [GeneratedRegex(@"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)")]
    private static partial Regex RgbRegex();
    [GeneratedRegex(@"<section\b[^>]*class=""[^""]*\bslide\b[^""]*""[^>]*>(.*?)</section>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex SlideSectionRegex();
    [GeneratedRegex(@"<(?:section|header|footer)\b[^>]*>(.*?)</(?:section|header|footer)>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex SectionRegex();
    [GeneratedRegex(@"<h([1-3])[^>]*>(.*?)</h\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex FirstHeadingRegex();
    [GeneratedRegex(@"<(p|li|h[4-6]|blockquote|figcaption|dt|dd|td)\b[^>]*>(.*?)</\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex LineRegex();
}
