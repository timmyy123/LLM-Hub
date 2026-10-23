using LLMHub.ViewModels;

namespace LLMHub.Services;

/// <summary>App-wide singletons. Order matters: later services use earlier ones.</summary>
public static class Ioc
{
    public static HardwareInfo Hardware { get; } = HardwareInfo.Detect();
    public static CatalogService Catalog { get; } = new();
    public static HuggingFaceClient HuggingFace { get; } = new();
    public static ModelStore Models { get; } = new();
    public static DownloadManager Downloads { get; } = new(HuggingFace, Models);
    public static LlamaEngine Llama { get; } = new();
    public static LiteRtEngine LiteRt { get; } = new();
    public static EngineHub Engines { get; } = new(Llama, LiteRt);
    public static ChatStore Chats { get; } = new();
    public static LocalApiServer Api { get; } = new();
    public static Speech.SpeechService Speech { get; } = new();
    /// <summary>Private API instance the Agent's Claude Code talks to (see <see cref="LocalApiServer(bool)"/>).</summary>
    public static LocalApiServer AgentBridge { get; } = new(agentBridge: true);
    public static Agent.AgentService Agent { get; } = new();
    public static Studio.ComfyService Comfy { get; } = new();
    public static Design.DesignService Design { get; } = new();

    private static ChatViewModel? _chat;
    private static ModelsViewModel? _modelsVm;

    public static ChatViewModel Chat => _chat ??= new ChatViewModel();
    public static ModelsViewModel ModelsVm => _modelsVm ??= new ModelsViewModel();

    private static AgentViewModel? _agentVm;
    public static AgentViewModel AgentVm => _agentVm ??= new AgentViewModel();

    private static StudioViewModel? _studioVm;
    public static StudioViewModel StudioVm => _studioVm ??= new StudioViewModel();

    private static SettingsViewModel? _settingsVm;
    public static SettingsViewModel SettingsVm => _settingsVm ??= new SettingsViewModel();

    private static DesignViewModel? _designVm;
    public static DesignViewModel DesignVm => _designVm ??= new DesignViewModel();
}
