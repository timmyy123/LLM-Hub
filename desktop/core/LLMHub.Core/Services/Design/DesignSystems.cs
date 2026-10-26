namespace LLMHub.Services.Design;

/// <summary>A named visual direction the model designs with: colors, type, and short guidance (our own, not brand copies).</summary>
public sealed record DesignSystem(string Id, string Name, string[] Colors, string HeadingFont, string BodyFont, string Guidance)
{
    public override string ToString() => Name;

    /// <summary>The DESIGN.md-style brief given to the model.</summary>
    public string Brief => $"""
        Design system: {Name}
        Palette (use these hex values; first is the page background, last is the darkest ink): {string.Join(", ", Colors)}
        Headings: {HeadingFont}. Body text: {BodyFont}. Load fonts from Google Fonts with <link> tags.
        {Guidance}
        """;

    public static IReadOnlyList<DesignSystem> All { get; } =
    [
        new("editorial-warm", "Editorial Warm", ["#f6f1e7", "#efe6d6", "#d9c6ae", "#b4532a", "#2a211b"], "Fraunces", "Inter",
            "Warm cream backgrounds, serif headings, generous whitespace, one terracotta accent used sparingly for calls to action."),
        new("clean-modern", "Clean Modern", ["#ffffff", "#f3f5f8", "#d7dde6", "#2f6fe4", "#121826"], "Inter", "Inter",
            "Crisp white surfaces, strong grid, 8 px spacing scale, rounded 12 px cards, one blue accent, high-contrast text."),
        new("midnight", "Midnight", ["#0f1117", "#171a23", "#2a2f3d", "#8ab4ff", "#eef1f7"], "Space Grotesk", "Inter",
            "Dark interface, subtle borders instead of shadows, luminous blue accent, tight technical typography."),
        new("bold-contrast", "Bold Contrast", ["#fffdf5", "#ffe45c", "#ff5a36", "#1d1d1d", "#000000"], "Archivo Black", "Archivo",
            "Poster-like: oversized headlines, thick black rules, flat saturated color blocks, no gradients."),
        new("soft-pastel", "Soft Pastel", ["#fbf8ff", "#efe7ff", "#ffd8e4", "#7c5cff", "#2b2440"], "DM Serif Display", "DM Sans",
            "Gentle pastel fields, rounded 20 px shapes, friendly illustrations described with CSS shapes, airy layouts."),
        new("minimal-mono", "Minimal Mono", ["#fafafa", "#eeeeee", "#cccccc", "#333333", "#111111"], "IBM Plex Sans", "IBM Plex Sans",
            "Black and white only, typography-led, thin hairlines, lots of space, no color accents."),
    ];
}

/// <summary>What the Design tab makes; each kind gets its own instructions.</summary>
public enum DesignKind { Prototype, Slides, Image, Motion }
