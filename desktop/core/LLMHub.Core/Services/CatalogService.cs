using System.Text.Json;

namespace LLMHub.Services;

/// <param name="Mtp">LiteRT-LM builds that ship a multi-token-prediction drafter (faster decoding on GPU).</param>
public sealed record CatalogSource(string Repo, string Format, string? PreferredQuant, bool Mtp = false);

public sealed record CatalogModel(
    string Id,
    string Name,
    string Maker,
    string[] Tags,
    int ContextLength,
    int? MinRamGB,
    CatalogSource[] Sources);

public sealed record CatalogDocument(int Version, Dictionary<string, string> Defaults, CatalogModel[] Models);

/// <summary>The curated model list shared with the mobile apps (desktop/shared/catalog.json).
/// It names Hugging Face repos only; actual files are resolved live from the Hub.</summary>
public sealed class CatalogService
{
    private CatalogDocument? _doc;

    public CatalogDocument Document => _doc ??= Load();

    public IReadOnlyList<CatalogModel> Models => Document.Models;

    public string? DefaultFor(string role) => Document.Defaults.GetValueOrDefault(role);

    private static CatalogDocument Load()
    {
        try
        {
            var json = File.ReadAllText(AppPaths.BundledCatalog);
            return JsonSerializer.Deserialize<CatalogDocument>(json, new JsonSerializerOptions(JsonSerializerDefaults.Web))
                   ?? new CatalogDocument(1, [], []);
        }
        catch (Exception e)
        {
            Log.Error("Catalog load failed", e);
            return new CatalogDocument(1, [], []);
        }
    }
}
