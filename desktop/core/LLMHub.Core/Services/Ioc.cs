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
    /// <summary>EmbeddingGemma for /v1/embeddings, on its own small llama-server next to the chat model.</summary>
    public static LlamaEngine Embeddings { get; } = new(embeddings: true);
    public static LiteRtEngine LiteRt { get; } = new();
    public static EngineHub Engines { get; } = new(Llama, LiteRt);
    public static ChatStore Chats { get; } = new();
    public static LocalApiServer Api { get; } = new();
    public static Speech.SpeechService Speech { get; } = new();
    /// <summary>Private API instance the Agent's Claude Code talks to (see <see cref="LocalApiServer(bool)"/>).</summary>
    public static LocalApiServer AgentBridge { get; } = new(agentBridge: true);
    public static Agent.AgentService Agent { get; } = new();
    public static Studio.ComfyService Comfy { get; } = new();
    /// <summary>Studio images on the Snapdragon NPU (Windows on Arm), where ComfyUI has no GPU backend.</summary>
    public static Studio.Npu.SdNpuService SdNpu { get; } = new();
    /// <summary>Studio video on the Snapdragon NPU (LTX-Video through sd-cli.exe).</summary>
    public static Studio.Npu.LtxNpuService LtxNpu { get; } = new();
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
