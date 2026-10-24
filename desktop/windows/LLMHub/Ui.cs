using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;

namespace LLMHub;

/// <summary>A design-system colour for the swatch row.</summary>
public sealed record Swatch(string Hex, Brush Fill);

/// <summary>Small helpers for x:Bind function bindings.</summary>
public static class Ui
{
    public static Visibility Show(bool value) => value ? Visibility.Visible : Visibility.Collapsed;
    public static Visibility Hide(bool value) => value ? Visibility.Collapsed : Visibility.Visible;
    public static Visibility ShowText(string? value) => string.IsNullOrEmpty(value) ? Visibility.Collapsed : Visibility.Visible;
    public static Visibility ShowAll(bool a, bool b) => a && b ? Visibility.Visible : Visibility.Collapsed;
    public static bool Not(bool value) => !value;
    /// <summary>Looks up a brush from Themes/Theme.xaml for the element's current light/dark theme.</summary>
    public static Brush? ThemeBrush(FrameworkElement element, string key)
    {
        var theme = element.ActualTheme == ElementTheme.Light ? "Light" : "Default";
        foreach (var dictionary in Application.Current.Resources.MergedDictionaries)
        {
            if (dictionary.ThemeDictionaries.TryGetValue(theme, out var themed)
                && themed is ResourceDictionary resources
                && resources.TryGetValue(key, out var value))
                return value as Brush;
        }
        return null;
    }

    public static bool HasText(string? value) => !string.IsNullOrEmpty(value);
    public static Visibility ShowChecked(bool? value) => value == true ? Visibility.Visible : Visibility.Collapsed;
    public static string Numbered(int n) => $"{n}.";
    /// <summary>"Thinking…" while the reply streams, then the design's "Thought for a few seconds" pill label.</summary>
    public static string ThinkingLabel(bool done) => Services.Loc.S(done ? "ThoughtLabel" : "Thinking");

    /// <summary>File chip badge colours: blue for images, terracotta for documents (from the design).</summary>
    public static Brush? BadgeBrush(bool isImage, string part)
    {
        var key = (isImage ? "ImageBadge" : "PdfBadge") + (part == "fg" ? "TextBrush" : "Brush");
        return App.Window?.Content is FrameworkElement root ? ThemeBrush(root, key) : null;
    }

    public static Visibility ShowObject(object? value) => value is null ? Visibility.Collapsed : Visibility.Visible;
    public static Visibility ShowCount(int count) => count > 0 ? Visibility.Visible : Visibility.Collapsed;
    public static Visibility HideCount(int count) => count > 0 ? Visibility.Collapsed : Visibility.Visible;

    // Agent: diff lines, task checklist, changed-file badges (colours from the design's diff block).
    private static Brush Solid(byte r, byte g, byte b, byte a = 255) => new SolidColorBrush(Windows.UI.Color.FromArgb(a, r, g, b));
    public static Brush DiffBackground(char kind) => kind switch
    {
        '+' => Solid(0x1E, 0x33, 0x26),
        '-' => Solid(0x3A, 0x22, 0x22),
        '@' => Solid(0x26, 0x2C, 0x38),
        _ => Solid(0, 0, 0, 0),
    };
    public static Brush DiffForeground(char kind) => kind switch
    {
        '+' => Solid(0xC5, 0xF0, 0xD3),
        '-' => Solid(0xFF, 0xB4, 0xB4),
        '@' => Solid(0xB8, 0xCD, 0xF5),
        _ => Solid(0xA5, 0xA4, 0x9E),
    };
    public static string DiffText(char kind, string text) => kind == '@' ? text : $"{kind} {text}";
    public static string TaskGlyph(string status) => status switch { "completed" => "\uE73E", "in_progress" => "\uEA3A", _ => "\uEA3A" };
    public static Brush TaskBrush(string status) => status switch
    {
        "completed" => Solid(0x5F, 0xD3, 0x8D),
        "in_progress" => Solid(0x8A, 0xB4, 0xFF),
        _ => Solid(0x8F, 0x8E, 0x88),
    };
    public static Brush TaskTextBrush(string status) => status == "completed" ? Solid(0x8F, 0x8E, 0x88) : Solid(0xEC, 0xEB, 0xE7);
    public static Windows.UI.Text.TextDecorations TaskDecoration(string status) =>
        status == "completed" ? Windows.UI.Text.TextDecorations.Strikethrough : Windows.UI.Text.TextDecorations.None;
    public static string ChangeBadge(char kind) => kind.ToString();
    public static Brush ChangeBrush(char kind) => kind switch
    {
        'A' => Solid(0x5F, 0xD3, 0x8D),
        'D' => Solid(0xFF, 0x9A, 0x9A),
        _ => Solid(0xE8, 0xB0, 0x4A),
    };

    public static Brush AgentDot(Services.Agent.AgentStatus status) => status switch
    {
        Services.Agent.AgentStatus.Working => Solid(0x8A, 0xB4, 0xFF),
        Services.Agent.AgentStatus.NeedsYou => Solid(0xE8, 0xB0, 0x4A),
        Services.Agent.AgentStatus.Error => Solid(0xFF, 0x9A, 0x9A),
        _ => Solid(0x5A, 0x59, 0x55),
    };

    // Studio
    public static Brush RunningBrush(bool running) => running ? Solid(0x5F, 0xD3, 0x8D) : Solid(0x8F, 0x8E, 0x88);
    /// <summary>"llama.cpp", "LiteRT-LM" or "llama.cpp · embeddings" for the Local API's model list.</summary>
    public static string EngineLabel(Services.InstalledModel m) => Services.LocalApiServer.EngineLabel(m);
    public static string Total(string size) => size.Length > 0 ? Services.Loc.F("SettingsTotal", size) : "";
    public static object Box(int value) => value;
    public static double ToDouble(int value) => value;
    public static string ReferenceLabel(string path) => string.IsNullOrEmpty(path) ? Services.Loc.S("StudioReference") : System.IO.Path.GetFileName(path);
    public static string MusicSavedTo() => Services.Loc.F("StudioSavedTo", ViewModels.StudioViewModel.MusicFolder);
    public static string SavedTo() => Services.Loc.F("StudioSavedTo", Services.Studio.ComfyService.Gallery);
    public static ImageSource? Thumbnail(string path)
    {
        try
        {
            return new Microsoft.UI.Xaml.Media.Imaging.BitmapImage(new Uri(path)) { DecodePixelWidth = 480 };
        }
        catch (Exception)
        {
            return null;
        }
    }

    // Design
    public static Visibility IsRole(string role, string wanted) => role == wanted ? Visibility.Visible : Visibility.Collapsed;
    public static string DesignPlaceholder(bool hasPreview) => Services.Loc.S(hasPreview ? "DesignPlaceholder" : "DesignStartPlaceholder");
    public static string SelectedLabel(string element) => string.IsNullOrEmpty(element) ? Services.Loc.S("DesignSelectHint") : element;
    /// <summary>"Editing the hours section" while a change runs, or "Designing… (2.1k characters)".</summary>
    public static string DesignProgress(string element, string live) =>
        Services.Loc.S("DesignWorking") + (live.Length > 0 ? $" ({live.Length / 1000.0:F1}k)" : "") + (string.IsNullOrEmpty(element) ? "" : " · " + element);
    public static Brush Hex(string hex)
    {
        var h = hex.TrimStart('#');
        return Solid(Convert.ToByte(h[..2], 16), Convert.ToByte(h[2..4], 16), Convert.ToByte(h[4..6], 16));
    }

    public static List<Swatch> Swatches(Services.Design.DesignSystem system) => system.Colors.Select(c => new Swatch(c, Hex(c))).ToList();

    public static string StarLabel(bool starred) => Services.Loc.S(starred ? "Unstar" : "Star");
    public static string StarGlyph(bool starred) => starred ? "\uE735" : "\uE734";
    public static string ReadGlyph(bool reading) => reading ? "\uE71A" : "\uE767";

    public static HorizontalAlignment MessageAlign(bool isUser) => isUser ? HorizontalAlignment.Right : HorizontalAlignment.Stretch;
}
