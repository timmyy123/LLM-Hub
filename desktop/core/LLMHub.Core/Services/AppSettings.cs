using System.Text.Json;

namespace LLMHub.Services;

public sealed class AppSettings
{
    public static AppSettings Current { get; private set; } = new();

    /// <summary>BCP-47 tag such as "de-DE", or empty to follow Windows.</summary>
    public string Language { get; set; } = "";
    /// <summary>"dark" (the designed look, default), "light", or "system" to follow the OS.</summary>
    public string Appearance { get; set; } = "dark";
    /// <summary>Composer switches, remembered between chats.</summary>
    public bool WebSearch { get; set; }
    public bool Thinking { get; set; }
    /// <summary>Supertonic voice for Read aloud: F1, F2, M1 or M2.</summary>
    public string Voice { get; set; } = "F1";
    /// <summary>Supertonic model: fp32 (Supertone's release) or fp16 (half the download).</summary>
    public string SpeechPrecision { get; set; } = "fp32";
    /// <summary>Language Supertonic reads in ("" detects it from the text).</summary>
    public string SpeechLanguage { get; set; } = "";
    public double SpeechSpeed { get; set; } = 1.05;
    /// <summary>Model new Agent sessions use (defaults to the catalog's agent pick, Gemma 4 26B-A4B).</summary>
    public string? AgentModelId { get; set; }
    /// <summary>Studio › Search: the folders indexed for photos, audio and videos, and the EmbeddingGemma 2 build used.</summary>
    public string MediaPhotoFolder { get; set; } = "";
    public string MediaAudioFolder { get; set; } = "";
    public string MediaVideoFolder { get; set; } = "";
    public string? MediaModelId { get; set; }
    /// <summary>"claude" or "openclaw", for new Agent sessions.</summary>
    public string AgentEngine { get; set; } = "claude";
    public string? DesignModelId { get; set; }
    public string HuggingFaceToken { get; set; } = "";
    public string? LastChatModelId { get; set; }
    public int ContextLength { get; set; } = 16384;
    /// <summary>Context window for Local API and Agent requests (capped by the model). Claude Code alone sends about 18k tokens.</summary>
    public int ApiContextLength { get; set; } = 65536;
    /// <summary>Local API: free the model's memory after this many idle minutes; 0 keeps it loaded.</summary>
    public int ApiUnloadAfterMinutes { get; set; } = 10;
    /// <summary>Models where the user turned multi-token prediction off (it is on by default where supported).</summary>
    public List<string> MtpOff { get; set; } = [];

    // Local API
    public bool ApiEnabled { get; set; }
    public int ApiPort { get; set; } = 8642;
    public bool ApiAllowLan { get; set; }
    public bool ApiRequireKey { get; set; } = true;
    public string ApiKey { get; set; } = "";

    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    public static void Load()
    {
        try
        {
            if (File.Exists(AppPaths.SettingsFile))
                Current = JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(AppPaths.SettingsFile), Options) ?? new();
            // Hand-edited files can carry "MtpOff": null.
            Current.MtpOff ??= [];
        }
        catch (Exception e)
        {
            Log.Error("Settings load failed", e);
        }
    }

    public static void Save()
    {
        try
        {
            File.WriteAllText(AppPaths.SettingsFile, JsonSerializer.Serialize(Current, Options));
        }
        catch (Exception e)
        {
            Log.Error("Settings save failed", e);
        }
    }
}
