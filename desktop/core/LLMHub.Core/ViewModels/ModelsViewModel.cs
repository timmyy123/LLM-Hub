using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LLMHub.Services;

namespace LLMHub.ViewModels;

/// <summary>A downloadable quantization shown in a picker, with how well it fits this PC.</summary>
public sealed class VariantOption(GgufVariant variant)
{
    public GgufVariant Variant { get; } = variant;
    public ModelFit Fit { get; } = Ioc.Hardware.Fit(variant.WeightBytes);
    public string Label => $"{Variant.Quant} · {Fmt.Bytes(Variant.TotalBytes)}";
    public bool IsLiteRt => Variant.Format == "litertlm";
    public string FitText => Fit switch
    {
        ModelFit.Gpu => Loc.S("FitGpu"),
        ModelFit.Partial => Loc.S("FitPartial"),
        _ => Loc.S("FitTooLarge"),
    };

    /// Screen readers and UI Automation read list items through ToString.
    public override string ToString() => $"{Label} · {FitText}";
}

/// <summary>Shared download/installed state for one repo + quant, used by the catalog and the Hugging Face tab.</summary>
public abstract partial class DownloadableViewModel : ObservableObject
{
    protected DownloadableViewModel()
    {
        Status = "";
        Ioc.Models.Installed.CollectionChanged += (_, _) => Refresh();
        Ioc.Downloads.Jobs.CollectionChanged += (_, _) => Refresh();
        Ioc.Downloads.Finished += OnDownloadFinished;
    }

    private void OnDownloadFinished(string modelId, string? error)
    {
        if (SelectedVariant is not null && modelId == InstalledModel.MakeId(SelectedVariant.Variant.Repo, SelectedVariant.Variant.Quant))
            Status = error ?? "";
    }

    public ObservableCollection<VariantOption> Variants { get; } = [];

    [ObservableProperty] public partial VariantOption? SelectedVariant { get; set; }
    [ObservableProperty] public partial bool IsLoadingVariants { get; set; }
    [ObservableProperty] public partial string Status { get; set; }
    [ObservableProperty] public partial DownloadJob? Job { get; set; }
    [ObservableProperty] public partial InstalledModel? Installed { get; set; }

    public bool CanPick => Variants.Count > 0;
    public bool CanDownload => SelectedVariant is not null && Job is null && Installed is null;
    public bool IsDownloading => Job is not null;
    public bool IsInstalled => Installed is not null && Job is null;
    public string FitText => SelectedVariant?.FitText ?? "";

    protected abstract string DisplayName { get; }
    protected abstract string? CatalogId { get; }
    protected abstract string[] Tags { get; }
    protected abstract int ContextLength { get; }

    partial void OnSelectedVariantChanged(VariantOption? value) => Refresh();

    protected void Refresh()
    {
        var id = SelectedVariant is null ? null : InstalledModel.MakeId(SelectedVariant.Variant.Repo, SelectedVariant.Variant.Quant);
        Job = id is null ? null : Ioc.Downloads.Find(id);
        Installed = id is null ? null : Ioc.Models.Find(id);
        OnPropertyChanged(nameof(CanDownload));
        OnPropertyChanged(nameof(IsDownloading));
        OnPropertyChanged(nameof(IsInstalled));
        OnPropertyChanged(nameof(FitText));
    }

    [RelayCommand]
    private void Download()
    {
        if (SelectedVariant is null)
            return;
        Status = "";
        Ioc.Downloads.Start(SelectedVariant.Variant, DisplayName, CatalogId, Tags, ContextLength);
        Refresh();
    }

    [RelayCommand]
    private void Cancel()
    {
        if (Job is not null)
            Ioc.Downloads.Cancel(Job);
    }

    [RelayCommand]
    private void Delete()
    {
        if (Installed is not null)
            Ioc.Models.Delete(Installed);
    }

    protected void SetVariants(IEnumerable<GgufVariant> variants, string? preferredQuant)
    {
        Variants.Clear();
        foreach (var option in variants.Select(v => new VariantOption(v)))
        {
            var installed = Ioc.Models.Find(InstalledModel.MakeId(option.Variant.Repo, option.Variant.Quant)) is not null;
            if (installed || Ioc.Hardware.ShouldOffer(option.Fit))
                Variants.Add(option);
        }
        SelectedVariant =
            Variants.FirstOrDefault(v => Ioc.Models.Find(InstalledModel.MakeId(v.Variant.Repo, v.Variant.Quant)) is not null)
            ?? Variants.FirstOrDefault(v => string.Equals(v.Variant.Quant, preferredQuant, StringComparison.OrdinalIgnoreCase))
            ?? Variants.FirstOrDefault(v => v.Variant.Quant == "Q4_K_M")
            ?? Variants.LastOrDefault(v => v.Variant.Quant.Contains("Q4"))
            ?? Variants.LastOrDefault();
        OnPropertyChanged(nameof(CanPick));
    }
}

public sealed partial class CatalogItemViewModel(CatalogModel model) : DownloadableViewModel
{
    public CatalogModel Model { get; } = model;
    public string Name => Model.Name;
    public string Subtitle => string.Join(" · ", new[] { Model.Maker }.Concat(Model.Tags.Select(TagLabel)));
    public string EngineLabel => (GgufSource, LiteRtSource) switch
    {
        (not null, not null) when LiteRtEngine.IsSupported => "llama.cpp · LiteRT-LM",
        (not null, _) => "llama.cpp",
        _ => "LiteRT-LM",
    };
    public bool NeedsLiteRt => GgufSource is null && !LiteRtEngine.IsSupported;
    public bool NothingFits { get; private set; }

    private CatalogSource? GgufSource => Model.Sources.FirstOrDefault(s => s.Format == "gguf");
    private CatalogSource? LiteRtSource => Model.Sources.FirstOrDefault(s => s.Format == "litertlm");

    protected override string DisplayName => Model.Name;
    protected override string? CatalogId => Model.Id;
    protected override string[] Tags => Model.Tags;
    protected override int ContextLength => Model.ContextLength;

    public async Task LoadVariantsAsync(CancellationToken ct)
    {
        if (NeedsLiteRt)
        {
            Status = Loc.S("LiteRtNoArm");
            return;
        }
        IsLoadingVariants = true;
        Status = Loc.S("LoadingVariants");
        var source = GgufSource ?? LiteRtSource!;
        try
        {
            var variants = new List<GgufVariant>();
            if (GgufSource is { } gguf)
                variants.AddRange(await Ioc.HuggingFace.GetGgufVariantsAsync(gguf.Repo, ct));
            if (LiteRtSource is { } litert && LiteRtEngine.IsSupported)
                variants.AddRange(await Ioc.HuggingFace.GetLiteRtVariantsAsync(litert.Repo, ct));
            SetVariants(variants, GgufSource?.PreferredQuant);
            NothingFits = Variants.Count == 0;
            Status = NothingFits ? Loc.S("NothingFits") : "";
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Error($"Variants for {source.Repo}", e);
            Status = Loc.S("VariantsError");
        }
        finally
        {
            IsLoadingVariants = false;
        }
    }

    private static string TagLabel(string tag) => tag switch
    {
        "vision" => Loc.S("TagVision"),
        "audio" => Loc.S("TagAudio"),
        "thinking" => Loc.S("TagThinking"),
        "embedding" => Loc.S("TagEmbedding"),
        _ => tag,
    };
}

public sealed partial class HfRepoViewModel(HfRepo repo) : DownloadableViewModel
{
    public HfRepo Repo { get; } = repo;
    public string Id => Repo.Id;
    public string Subtitle => Loc.F("RepoStats", Repo.Downloads.ToString("N0"), Repo.Likes.ToString("N0"));

    protected override string DisplayName => Repo.Id.Split('/').Last().Replace("-GGUF", "", StringComparison.OrdinalIgnoreCase);
    protected override string? CatalogId => null;
    protected override string[] Tags => [];
    protected override int ContextLength => 0;

    public async Task LoadVariantsAsync(CancellationToken ct)
    {
        IsLoadingVariants = true;
        Status = Loc.S("LoadingVariants");
        try
        {
            var variants = await Ioc.HuggingFace.GetGgufVariantsAsync(Repo.Id, ct);
            SetVariants(variants, "Q4_K_M");
            Status = variants.Count == 0 ? Loc.S("HfNoGguf") : Variants.Count == 0 ? Loc.S("NothingFits") : "";
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            Log.Error($"Variants for {Repo.Id}", e);
            Status = Loc.S("VariantsError");
        }
        finally
        {
            IsLoadingVariants = false;
        }
    }
}

public sealed partial class ModelsViewModel : ObservableObject
{
    private CancellationTokenSource? _search;
    private bool _catalogLoaded;

    public ModelsViewModel()
    {
        Query = "";
        foreach (var model in Ioc.Catalog.Models)
            Catalog.Add(new CatalogItemViewModel(model));
        Installed.CollectionChanged += (_, _) => OnPropertyChanged(nameof(HasInstalled));
    }

    public ObservableCollection<CatalogItemViewModel> Catalog { get; } = [];
    public ObservableCollection<HfRepoViewModel> SearchResults { get; } = [];
    public ObservableCollection<InstalledModel> Installed => Ioc.Models.Installed;
    public bool HasInstalled => Installed.Count > 0;
    public bool HasSelectedRepo => SelectedRepo is not null;

    public string HardwareText => Ioc.Hardware.Summary(withRam: true);

    [ObservableProperty] public partial string Query { get; set; }
    [ObservableProperty] public partial bool IsSearching { get; set; }
    [ObservableProperty] public partial HfRepoViewModel? SelectedRepo { get; set; }

    public async Task LoadCatalogAsync()
    {
        if (_catalogLoaded)
            return;
        _catalogLoaded = true;
        using var throttle = new SemaphoreSlim(4);
        await Task.WhenAll(Catalog.ToList().Select(async item =>
        {
            await throttle.WaitAsync();
            try
            {
                await item.LoadVariantsAsync(CancellationToken.None);
            }
            finally
            {
                throttle.Release();
            }
        }));
        // Models with no quant that fits this computer are not offered at all.
        foreach (var item in Catalog.Where(i => i.NothingFits).ToList())
            Catalog.Remove(item);
    }

    partial void OnSelectedRepoChanged(HfRepoViewModel? value)
    {
        OnPropertyChanged(nameof(HasSelectedRepo));
        if (value is not null && value.Variants.Count == 0)
            _ = value.LoadVariantsAsync(CancellationToken.None);
    }

    [RelayCommand]
    private async Task SearchAsync()
    {
        _search?.Cancel();
        _search = new CancellationTokenSource();
        var ct = _search.Token;
        IsSearching = true;
        try
        {
            var results = await Ioc.HuggingFace.SearchAsync(Query.Trim(), "gguf", ct);
            SearchResults.Clear();
            foreach (var repo in results)
                SearchResults.Add(new HfRepoViewModel(repo));
            SelectedRepo = SearchResults.FirstOrDefault();
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception e)
        {
            Log.Error("HF search failed", e);
        }
        finally
        {
            IsSearching = false;
        }
    }

    [RelayCommand]
    private void DeleteInstalled(InstalledModel? model)
    {
        if (model is not null)
            Ioc.Models.Delete(model);
    }
}
