namespace LLMHub.Services;

public static class AppPaths
{
    public static string Root { get; } = Ensure(Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LLMHub"));

    public static string Models => Ensure(Path.Combine(Root, "models"));
    public static string Engines => Ensure(Path.Combine(Root, "engines"));
    public static string Logs => Ensure(Path.Combine(Root, "logs"));
    /// <summary>Copies of files attached to chat messages.</summary>
    public static string Attachments => Ensure(Path.Combine(Root, "attachments"));
    public static string Database => Path.Combine(Root, "llmhub.db");
    public static string SettingsFile => Path.Combine(Root, "settings.json");
    public static string BundledCatalog => Path.Combine(AppContext.BaseDirectory, "Assets", "catalog.json");

    private static string Ensure(string path)
    {
        Directory.CreateDirectory(path);
        return path;
    }
}
