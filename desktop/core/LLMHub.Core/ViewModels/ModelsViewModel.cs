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
    /// <summary>"ok" (runs on the GPU), "warn" (partly on the CPU) or "bad", for the fit label's color.</summary>
    public string FitLevel => SelectedVariant?.Fit switch { ModelFit.Gpu => "ok", ModelFit.Partial => "warn", null => "", _ => "bad" };

    protected abstract string DisplayName { get; }
    protected abstract string? CatalogId { get; }
    protected abstract string[] Tags { get; }
    protected abstract int ContextLength { get; }

    private VariantOption? _lastSelected;

    partial void OnSelectedVariantChanged(VariantOption? value)
    {
        // A list row being recycled (when a filter chip moves rows) unbinds its drop-down, which writes null back; keep the pick.
        if (value is null && _lastSelected is not null && Variants.Contains(_lastSelected))
        {
            SelectedVariant = _lastSelected;
            return;
        }
        _lastSelected = value;
        Refresh();
    }

    protected void Refresh()
    {
        var id = SelectedVariant is null ? null : InstalledModel.MakeId(SelectedVariant.Variant.Repo, SelectedVariant.Variant.Quant);
        Job = id is null ? null : Ioc.Downloads.Find(id);
        Installed = id is null ? null : Ioc.Models.Find(id);
        OnPropertyChanged(nameof(CanDownload));
        OnPropertyChanged(nameof(IsDownloading));
        OnPropertyChanged(nameof(IsInstalled));
        OnPropertyChanged(nameof(FitText));
        OnPropertyChanged(nameof(FitLevel));
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
            // The pickers are bound to Variants: change it on the UI thread whatever thread the lookup finished on.
            await UiThread.RunAsync(() =>
            {
                SetVariants(variants, GgufSource?.PreferredQuant);
                NothingFits = Variants.Count == 0;
                Status = NothingFits ? Loc.S("NothingFits") : "";
            });
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

    /// <summary>The Models page filter chips: all, chat, vision, agent, embedding, or fits (runs fully on this GPU).</summary>
    public bool Matches(string filter) => filter switch
    {
        "chat" => !Model.Tags.Contains("embedding"),
        "vision" => Model.Tags.Contains("vision"),
        "agent" => Model.Tags.Contains("agent"),
        "embedding" => Model.Tags.Contains("embedding"),
        "fits" => Variants.Any(v => v.Fit == ModelFit.Gpu),
        _ => true,
    };

    public static string TagLabel(string tag) => tag switch
    {
        "agent" => Loc.S("TagAgent"),
        "vision" => Loc.S("TagVision"),
        "audio" => Loc.S("TagAudio"),
        "thinking" => Loc.S("TagThinking"),
        "embedding" => Loc.S("TagEmbedding"),
        _ => tag,
    };
}

/// <summary>One file (or split set) in a Hugging Face repo's table, with how it fits this computer.</summary>
public sealed partial class HfFileRow(HfRepoViewModel owner, VariantOption option, string fitText, string fitLevel, bool isBest)
{
    public VariantOption Option { get; } = option;
    public string File => Option.Variant.Files[0].FileName;
    public string Quant => Option.Variant.Quant;
    public string Size => Fmt.Bytes(Option.Variant.TotalBytes);
    public string QuantAndSize => $"{Quant} · {Size}";
    /// <summary>"Best fit", "Fits", "Fits, lower quality", "Fits, slower" or "Too large".</summary>
    public string FitText { get; } = fitText;
    /// <summary>"ok", "warn" or "bad", for the label's color.</summary>
    public string FitLevel { get; } = fitLevel;
    public bool IsBest { get; } = isBest;
    public bool IsInstalled => Ioc.Models.Find(InstalledModel.MakeId(Option.Variant.Repo, Option.Variant.Quant)) is not null;
    public bool CanDownload => FitLevel != "bad" && !IsInstalled && Ioc.Downloads.Find(InstalledModel.MakeId(Option.Variant.Repo, Option.Variant.Quant)) is null;

    [RelayCommand]
    private void Download() => owner.DownloadRow(this);
}

public sealed partial class HfRepoViewModel(HfRepo repo) : DownloadableViewModel
{
    public HfRepo Repo { get; } = repo;
    public string Id => Repo.Id;
    public string Subtitle => Loc.F("RepoStats", Repo.Downloads.ToString("N0"), Repo.Likes.ToString("N0"));
    public string FormatLabel => Repo.Format == "litertlm" ? "LiteRT-LM" : "GGUF";
    public string Url => $"{HuggingFaceClient.Host}/{Repo.Id}";
    /// <summary>The catalog model that downloads from this repo, if any.</summary>
    public CatalogModel? CatalogModel { get; } = Ioc.Catalog.Models.FirstOrDefault(m => m.Sources.Any(s => s.Repo.Equals(repo.Id, StringComparison.OrdinalIgnoreCase)));
    /// <summary>"In LLM Hub catalog" or "Community quants" under the repo name.</summary>
    public string Meta => Loc.S(CatalogModel is null ? "HfCommunity" : "HfInCatalog");
    /// <summary>"Google Gemma 4 12B · Vision · runs on llama.cpp" above the file table.</summary>
    public string Detail => string.Join(" · ", (CatalogModel is { } m ? new[] { $"{m.Maker} {m.Name}" }.Concat(m.Tags.Select(CatalogItemViewModel.TagLabel)) : [Subtitle])
        .Append(Loc.F("HfRunsOn", Repo.Format == "litertlm" ? "LiteRT-LM" : "llama.cpp")));
    /// <summary>Gated repos need the license accepted on the Hub, and a token from that account to download.</summary>
    public bool IsGated => Repo.Gated;
    public bool NeedsSignIn => Repo.Gated && string.IsNullOrEmpty(AppSettings.Current.HuggingFaceToken);
    public ObservableCollection<HfFileRow> Rows { get; } = [];
    private bool _watching;

    public void DownloadRow(HfFileRow row)
    {
        if (!Variants.Contains(row.Option))
            Variants.Add(row.Option);
        SelectedVariant = row.Option;
        DownloadCommand.Execute(null);
        RefreshRows();
    }

    private IReadOnlyList<GgufVariant> _all = [];

    private void RefreshRows()
    {
        // Rows show Download, a disabled button while fetching, or Installed: redraw when downloads or installs change.
        if (!_watching)
        {
            _watching = true;
            Ioc.Models.Installed.CollectionChanged += (_, _) => UiThread.Post(RefreshRows);
            Ioc.Downloads.Jobs.CollectionChanged += (_, _) => UiThread.Post(RefreshRows);
        }
        Rows.Clear();
        var best = SelectedVariant?.Variant;
        var bestSize = best?.WeightBytes ?? 0;
        foreach (var v in _all.OrderBy(v => v.WeightBytes))
        {
            var option = Variants.FirstOrDefault(o => o.Variant == v) ?? new VariantOption(v);
            var (text, level) = option.Fit switch
            {
                ModelFit.TooLarge => (Loc.S("FitTooLarge"), "bad"),
                ModelFit.Partial => (Loc.S("FitSlower"), "warn"),
                _ when v == best => (Loc.S("FitBest"), "ok"),
                _ when v.WeightBytes < bestSize => (Loc.S("FitLower"), "ok"),
                _ => (Loc.S("FitOk"), "ok"),
            };
            Rows.Add(new HfFileRow(this, option, text, level, v == best));
        }
    }

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
            var variants = Repo.Format == "litertlm"
                ? await Ioc.HuggingFace.GetLiteRtVariantsAsync(Repo.Id, ct)
                : await Ioc.HuggingFace.GetGgufVariantsAsync(Repo.Id, ct);
            await UiThread.RunAsync(() =>
            {
                _all = variants;
                SetVariants(variants, "Q4_K_M");
                RefreshRows();
                Status = variants.Count == 0 ? Loc.S("HfNoGguf") : Variants.Count == 0 ? Loc.S("NothingFits") : "";
            });
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
    private readonly bool _initialized;

    public ModelsViewModel()
    {
        Query = "";
        Filter = "all";
        HfFormat = "all";
        HfSort = "trendingScore";
        _initialized = true;
        Ioc.Engines.PropertyChanged += (_, _) => UiThread.Post(() =>
        {
            OnPropertyChanged(nameof(LlamaEngineLine));
            OnPropertyChanged(nameof(LiteRtEngineLine));
            OnPropertyChanged(nameof(LlamaRunning));
            OnPropertyChanged(nameof(LiteRtRunning));
        });
        foreach (var model in Ioc.Catalog.Models)
            Catalog.Add(new CatalogItemViewModel(model));
        ApplyFilter();
        Installed.CollectionChanged += (_, _) => OnPropertyChanged(nameof(HasInstalled));
    }

    public ObservableCollection<CatalogItemViewModel> Catalog { get; } = [];
    /// <summary>The catalog cards that pass the selected filter chip.</summary>
    public ObservableCollection<CatalogItemViewModel> VisibleCatalog { get; } = [];

    public static IReadOnlyList<(string Key, string Label)> Filters { get; } =
    [
        ("all", Loc.S("FilterAll")), ("chat", Loc.S("FilterChat")), ("vision", Loc.S("TagVision")), ("agent", Loc.S("TagAgent")),
        ("embedding", Loc.S("TagEmbedding")), ("fits", Loc.S("FilterFits")),
    ];

    public static IReadOnlyList<(string Key, string Label)> HfFormats { get; } =
        [("all", Loc.S("HfAllRunnable")), ("gguf", "GGUF"), ("litertlm", "LiteRT-LM")];

    public static IReadOnlyList<(string Key, string Label)> HfSorts { get; } =
        [("trendingScore", Loc.S("HfSortTrending")), ("downloads", Loc.S("HfSortDownloads")), ("lastModified", Loc.S("HfSortUpdated"))];

    [ObservableProperty] public partial string Filter { get; set; }
    [ObservableProperty] public partial string HfFormat { get; set; }
    [ObservableProperty] public partial string HfSort { get; set; }

    partial void OnFilterChanged(string value) => ApplyFilter();
    partial void OnHfFormatChanged(string value)
    {
        if (_initialized)
            SearchCommand.Execute(null);
    }

    partial void OnHfSortChanged(string value)
    {
        if (_initialized)
            SearchCommand.Execute(null);
    }

    /// <summary>Updates <see cref="VisibleCatalog"/> in place, so rows that stay are not rebuilt under the pointer.</summary>
    public void ApplyFilter()
    {
        var wanted = Catalog.Where(i => i.Matches(Filter)).ToList();
        for (var i = VisibleCatalog.Count - 1; i >= 0; i--)
            if (!wanted.Contains(VisibleCatalog[i]))
                VisibleCatalog.RemoveAt(i);
        for (var i = 0; i < wanted.Count; i++)
        {
            if (i < VisibleCatalog.Count && VisibleCatalog[i] == wanted[i])
                continue;
            VisibleCatalog.Remove(wanted[i]);
            VisibleCatalog.Insert(i, wanted[i]);
        }
    }

    // Engine cards: "GGUF · CUDA · running" and ".litertlm · GPU · ready".
    public bool LlamaRunning => Ioc.Engines.LoadedModel is { Format: "gguf" };
    public bool LiteRtRunning => Ioc.Engines.LoadedModel is { Format: "litertlm" };

    public string LlamaEngineLine => $"GGUF · {LlamaBackend(Ioc.Llama.Variant ?? LlamaEngine.LastWorkingVariant)} · " +
        Loc.S(LlamaRunning ? "EngineCardRunning" : (Ioc.Llama.Variant ?? LlamaEngine.LastWorkingVariant) is null ? "EngineCardOnDemand" : "EngineCardReady");

    public string LiteRtEngineLine => !LiteRtEngine.IsSupported ? $".litertlm · {Loc.S("EngineCardUnavailable")}"
        : $".litertlm · {Ioc.LiteRt.Backend?.ToUpperInvariant() ?? "GPU"} · " + Loc.S(LiteRtRunning ? "EngineCardRunning" : "EngineCardReady");

    private static string LlamaBackend(string? variant)
    {
        var hw = Ioc.Hardware;
        var v = variant ?? (hw.HasNvidiaGpu ? "cuda" : hw.HasAdrenoGpu ? "opencl-adreno" : OperatingSystem.IsMacOS() ? "metal" : hw.GpuName.Length > 0 ? "vulkan" : "cpu");
        return v.Contains("cuda") ? "CUDA" : v.Contains("adreno") ? "OpenCL (Adreno)" : v.Contains("vulkan") ? "Vulkan" : v.Contains("metal") ? "Metal"
            : v.Contains("hip") || v.Contains("rocm") ? "ROCm" : v.Contains("sycl") ? "SYCL" : "CPU";
    }

    /// <summary>Adds a .gguf or .litertlm from disk; returns the message to show.</summary>
    public string AddLocalFile(string path)
    {
        try
        {
            var model = Ioc.Models.AddLocalFile(path);
            return Loc.F("ModelsAdded", model.DisplayName);
        }
        catch (Exception e) when (e is IOException or InvalidOperationException or UnauthorizedAccessException)
        {
            Log.Error($"Add local model {path}", e);
            return Loc.F("ModelsAddFailed", Path.GetFileName(path), e.Message);
        }
    }
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
        ApplyFilter();
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
            var formats = HfFormat == "all" ? new[] { "gguf", "litert-lm" } : [HfFormat == "litertlm" ? "litert-lm" : "gguf"];
            var lists = await Task.WhenAll(formats.Select(f => Ioc.HuggingFace.SearchAsync(Query.Trim(), f, ct, HfSort)));
            IEnumerable<HfRepo> merged = lists.SelectMany(l => l);
            if (lists.Length > 1)
                merged = HfSort switch
                {
                    "downloads" => merged.OrderByDescending(r => r.Downloads),
                    "lastModified" => merged.OrderByDescending(r => r.LastModified ?? DateTimeOffset.MinValue),
                    _ => merged.OrderByDescending(r => r.Trending).ThenByDescending(r => r.Downloads),
                };
            var results = merged.Take(40).ToList();
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
