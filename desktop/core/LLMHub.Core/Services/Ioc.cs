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

    private static ChatViewModel? _chat;
    private static ModelsViewModel? _modelsVm;

    public static ChatViewModel Chat => _chat ??= new ChatViewModel();
    public static ModelsViewModel ModelsVm => _modelsVm ??= new ModelsViewModel();
}
