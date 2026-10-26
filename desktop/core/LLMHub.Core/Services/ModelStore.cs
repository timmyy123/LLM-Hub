using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LLMHub.Services;

public sealed class InstalledModel
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string? CatalogId { get; set; }
    public string Repo { get; set; } = "";
    public string Format { get; set; } = "gguf";
    public string Quant { get; set; } = "";
    public List<string> Files { get; set; } = [];
    public string? Mmproj { get; set; }
    public long SizeBytes { get; set; }
    public int ContextLength { get; set; }
    public string[] Tags { get; set; } = [];
    public DateTimeOffset InstalledAt { get; set; }
    /// <summary>Whether the .litertlm file carries an MTP drafter; null until the engine has looked (on first load).</summary>
    public bool? HasMtpDrafter { get; set; }

    [JsonIgnore] public string Folder { get; set; } = "";
    /// <summary>Downloads live in <see cref="Folder"/>; a local file added from disk keeps its full path and stays where it is.</summary>
    [JsonIgnore] public string PrimaryPath => Path.IsPathRooted(Files[0]) ? Files[0] : Path.Combine(Folder, Path.GetFileName(Files[0]));
    [JsonIgnore] public string? MmprojPath => Mmproj is null ? null : Path.IsPathRooted(Mmproj) ? Mmproj : Path.Combine(Folder, Path.GetFileName(Mmproj));
    [JsonIgnore] public bool IsLocalFile => Repo == "local";
    [JsonIgnore] public string DisplayName => $"{Name} · {Quant}";
    [JsonIgnore] public string SizeText => Fmt.Bytes(SizeBytes);
    /// <summary>EmbeddingGemma and the like: served on /v1/embeddings, never offered for chat.</summary>
    [JsonIgnore] public bool IsEmbedding => Tags.Contains("embedding");

    /// <summary>
    /// LiteRT-LM model whose catalog entry has an MTP drafter (looked up live so older downloads get it too), unless the
    /// downloaded file turned out to have none: Google's GPU-only builds leave it out.
    /// </summary>
    [JsonIgnore]
    public bool SupportsMtp => Format == "litertlm" && HasMtpDrafter != false && CatalogId is not null &&
        Ioc.Catalog.Models.FirstOrDefault(m => m.Id == CatalogId)?.Sources.Any(s => s.Format == "litertlm" && s.Mtp) == true;

    [JsonIgnore]
    public bool MtpEnabled => SupportsMtp && !AppSettings.Current.MtpOff.Contains(Id);

    public override string ToString() => DisplayName;

    public static string MakeId(string repo, string quant) =>
        $"{repo.Replace('/', '_')}__{quant.Replace(' ', '-')}".ToLowerInvariant();
}

/// <summary>Models on disk. Each lives in models/&lt;id&gt;/ with a manifest.json written after a verified download.</summary>
public sealed class ModelStore
{
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    public ObservableCollection<InstalledModel> Installed { get; } = [];

    /// <summary>Installed models that can chat (everything but embedding models), for the Chat, Agent and Design pickers.</summary>
    public ObservableCollection<InstalledModel> ChatModels { get; } = [];

    public ModelStore()
    {
        Installed.CollectionChanged += (_, _) =>
        {
            ChatModels.Clear();
            foreach (var m in Installed.Where(m => !m.IsEmbedding))
                ChatModels.Add(m);
        };
        Reload();
    }

    public void Reload()
    {
        Installed.Clear();
        foreach (var dir in Directory.EnumerateDirectories(AppPaths.Models))
        {
            var manifest = Path.Combine(dir, "manifest.json");
            if (!File.Exists(manifest))
                continue;
            try
            {
                var model = JsonSerializer.Deserialize<InstalledModel>(File.ReadAllText(manifest), Options);
                if (model is null || model.Files.Count == 0)
                    continue;
                model.Folder = dir;
                if (File.Exists(model.PrimaryPath))
                    Installed.Add(model);
            }
            catch (Exception e)
            {
                Log.Error($"Bad manifest {manifest}", e);
            }
        }
    }

    public string FolderFor(string id) => Path.Combine(AppPaths.Models, id);

    public InstalledModel? Find(string id) => Installed.FirstOrDefault(m => m.Id == id);

    /// <summary>Rewrites a model's manifest after a property such as <see cref="InstalledModel.HasMtpDrafter"/> is learned.</summary>
    public void Save(InstalledModel model) =>
        File.WriteAllText(Path.Combine(model.Folder, "manifest.json"), JsonSerializer.Serialize(model, Options));

    public void Add(InstalledModel model)
    {
        model.Folder = FolderFor(model.Id);
        Save(model);
        UiThread.Post(() =>
        {
            var existing = Find(model.Id);
            if (existing is not null)
                Installed.Remove(existing);
            Installed.Add(model);
        });
    }

    /// <summary>
    /// Adds a .gguf or .litertlm the user already has, in place (nothing is copied). A vision projector (mmproj*.gguf) in
    /// the same folder is picked up too. Deleting it later removes only LLM Hub's record, never the file.
    /// </summary>
    public InstalledModel AddLocalFile(string path)
    {
        var info = new FileInfo(path);
        if (!info.Exists)
            throw new FileNotFoundException(path);
        var format = info.Extension.ToLowerInvariant() switch
        {
            ".gguf" => "gguf",
            ".litertlm" => "litertlm",
            _ => throw new InvalidOperationException(Loc.S("ModelsLocalUnsupported")),
        };
        if (format == "gguf" && info.Name.StartsWith("mmproj", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(Loc.S("ModelsLocalProjector"));
        var stem = Path.GetFileNameWithoutExtension(info.Name);
        var quant = System.Text.RegularExpressions.Regex.Match(stem, @"(?:UD-)?(?:IQ\d_[A-Z0-9]+|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)$",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase).Value;
        var name = (quant.Length > 0 ? stem[..^quant.Length] : stem).TrimEnd('-', '_', '.', ' ');
        var mmproj = format == "gguf"
            ? Directory.EnumerateFiles(info.DirectoryName!, "mmproj*.gguf").OrderBy(f => f.Length).FirstOrDefault()
            : null;
        var tags = new List<string>();
        if (mmproj is not null)
            tags.Add("vision");
        if (name.Contains("embedding", StringComparison.OrdinalIgnoreCase))
            tags.Add("embedding");
        var model = new InstalledModel
        {
            Id = InstalledModel.MakeId("local/" + stem, quant.Length > 0 ? quant : format),
            Name = name.Length > 0 ? name : stem,
            Repo = "local",
            Format = format,
            Quant = quant.Length > 0 ? quant.ToUpperInvariant() : format == "gguf" ? "GGUF" : "LiteRT",
            Files = [info.FullName],
            Mmproj = mmproj,
            SizeBytes = info.Length + (mmproj is null ? 0 : new FileInfo(mmproj).Length),
            // The file's own limit is not read; the Settings context length still caps what is used.
            ContextLength = 32768,
            Tags = [.. tags],
            InstalledAt = DateTimeOffset.Now,
        };
        Add(model);
        Log.Info($"Added local model {info.FullName}{(mmproj is null ? "" : $" with {Path.GetFileName(mmproj)}")}");
        return model;
    }

    public void Delete(InstalledModel model)
    {
        if (Ioc.Engines.LoadedModel?.Id == model.Id)
            Ioc.Engines.Stop();
        try
        {
            Directory.Delete(model.Folder, recursive: true);
        }
        catch (Exception e)
        {
            Log.Error($"Delete {model.Folder} failed", e);
        }
        Installed.Remove(model);
    }
}

public static class Fmt
{
    public static string Bytes(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / 1073741824.0:F1} GB",
        >= 1L << 20 => $"{bytes / 1048576.0:F0} MB",
        _ => $"{bytes / 1024.0:F0} KB",
    };

    /// <summary>Token counts the way the context meter shows them: 950, 11k, 128k.</summary>
    public static string Tokens(int tokens) => tokens < 1000 ? tokens.ToString() : $"{Math.Round(tokens / 1000.0):F0}k";
}
