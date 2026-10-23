using System.Globalization;
using System.Reflection;
using System.Text.Json;

namespace LLMHub.Services;

/// <summary>
/// Localized strings from the embedded strings.json (generated from the same table as the Mac string catalog).
/// Usable from WinUI x:Bind as Loc.S('Key').
/// </summary>
public static class Loc
{
    public static readonly (string Tag, string Name)[] Languages =
    [
        ("en-US", "English"),
        ("es-ES", "Español"),
        ("pt-BR", "Português"),
        ("fr-FR", "Français"),
        ("de-DE", "Deutsch"),
        ("it-IT", "Italiano"),
        ("ru-RU", "Русский"),
    ];

    private static readonly Dictionary<string, Dictionary<string, string>> Tables = LoadTables();
    private static Dictionary<string, string> _current = Pick(CultureInfo.CurrentUICulture.Name);

    /// <summary>Applies a language tag such as "de-DE"; empty follows the system language.</summary>
    public static void ApplyLanguage(string tag) =>
        _current = Pick(string.IsNullOrEmpty(tag) ? CultureInfo.CurrentUICulture.Name : tag);

    public static string S(string key) =>
        _current.TryGetValue(key, out var value) || Tables["en-US"].TryGetValue(key, out value) ? value : key;

    public static string F(string key, params object[] args) => string.Format(S(key), args);

    private static Dictionary<string, string> Pick(string culture)
    {
        if (Tables.TryGetValue(culture, out var exact))
            return exact;
        var language = culture.Split('-')[0];
        return Tables.FirstOrDefault(t => t.Key.StartsWith(language + "-", StringComparison.OrdinalIgnoreCase)).Value ?? Tables["en-US"];
    }

    private static Dictionary<string, Dictionary<string, string>> LoadTables()
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("LLMHub.strings.json")!;
        return JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(stream)!;
    }
}
