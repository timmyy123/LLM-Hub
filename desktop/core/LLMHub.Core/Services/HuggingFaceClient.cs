using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LLMHub.Services;

public sealed record HfFile(string Path, long Size, string? Sha256)
{
    public string FileName => System.IO.Path.GetFileName(Path);
}

public sealed record HfRepo(string Id, long Downloads, long Likes, DateTimeOffset? LastModified, bool Gated);

/// <summary>One downloadable quantization of a GGUF repo: its weight files (possibly split) plus the vision projector if any.</summary>
public sealed record GgufVariant(string Repo, string Quant, IReadOnlyList<HfFile> Files, HfFile? Mmproj, string Format = "gguf")
{
    public long WeightBytes => Files.Sum(f => f.Size);
    public long TotalBytes => WeightBytes + (Mmproj?.Size ?? 0);
}

public sealed partial class HuggingFaceClient
{
    public const string Host = "https://huggingface.co";

    private readonly HttpClient _http;
    private readonly Dictionary<string, IReadOnlyList<HfFile>> _treeCache = new();

    public HuggingFaceClient()
    {
        _http = new HttpClient(new SocketsHttpHandler { AutomaticDecompression = System.Net.DecompressionMethods.All })
        {
            Timeout = TimeSpan.FromSeconds(30),
        };
        _http.DefaultRequestHeaders.UserAgent.ParseAdd("LLMHub-Desktop/0.1");
    }

    public HttpRequestMessage Request(HttpMethod method, string url)
    {
        var request = new HttpRequestMessage(method, url);
        var token = AppSettings.Current.HuggingFaceToken;
        if (!string.IsNullOrWhiteSpace(token) && url.StartsWith(Host, StringComparison.Ordinal))
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token.Trim());
        return request;
    }

    public static string ResolveUrl(string repo, string path) =>
        $"{Host}/{repo}/resolve/main/{string.Join('/', path.Split('/').Select(Uri.EscapeDataString))}";

    public async Task<IReadOnlyList<HfRepo>> SearchAsync(string query, string? libraryFilter, CancellationToken ct)
    {
        var url = $"{Host}/api/models?search={Uri.EscapeDataString(query)}&sort=trendingScore&direction=-1&limit=40";
        if (!string.IsNullOrEmpty(libraryFilter))
            url += $"&filter={Uri.EscapeDataString(libraryFilter)}";
        using var request = Request(HttpMethod.Get, url);
        using var response = await _http.SendAsync(request, ct);
        response.EnsureSuccessStatusCode();
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
        var list = new List<HfRepo>();
        foreach (var item in doc.RootElement.EnumerateArray())
        {
            var id = item.GetProperty("id").GetString() ?? "";
            var downloads = item.TryGetProperty("downloads", out var d) && d.TryGetInt64(out var dl) ? dl : 0;
            var likes = item.TryGetProperty("likes", out var l) && l.TryGetInt64(out var lk) ? lk : 0;
            DateTimeOffset? modified = item.TryGetProperty("lastModified", out var m) && m.TryGetDateTimeOffset(out var dt) ? dt : null;
            var gated = item.TryGetProperty("gated", out var g) && g.ValueKind is JsonValueKind.String or JsonValueKind.True;
            list.Add(new HfRepo(id, downloads, likes, modified, gated));
        }
        return list;
    }

    public async Task<IReadOnlyList<HfFile>> ListFilesAsync(string repo, CancellationToken ct)
    {
        lock (_treeCache)
            if (_treeCache.TryGetValue(repo, out var cached))
                return cached;

        using var request = Request(HttpMethod.Get, $"{Host}/api/models/{repo}/tree/main?recursive=true");
        using var response = await _http.SendAsync(request, ct);
        response.EnsureSuccessStatusCode();
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
        var files = new List<HfFile>();
        foreach (var item in doc.RootElement.EnumerateArray())
        {
            if (item.GetProperty("type").GetString() != "file")
                continue;
            var path = item.GetProperty("path").GetString() ?? "";
            var size = item.TryGetProperty("size", out var s) ? s.GetInt64() : 0;
            string? sha = null;
            if (item.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object)
            {
                sha = lfs.TryGetProperty("oid", out var oid) ? oid.GetString() : null;
                if (lfs.TryGetProperty("size", out var ls))
                    size = ls.GetInt64();
            }
            files.Add(new HfFile(path, size, sha));
        }
        lock (_treeCache)
            _treeCache[repo] = files;
        return files;
    }

    public async Task<IReadOnlyList<GgufVariant>> GetGgufVariantsAsync(string repo, CancellationToken ct) =>
        GroupGguf(repo, await ListFilesAsync(repo, ct));

    /// <summary>Generic .litertlm files in a repo. Builds made for one phone chip (Qualcomm, MediaTek, Tensor NPUs) or the web are skipped.</summary>
    public async Task<IReadOnlyList<GgufVariant>> GetLiteRtVariantsAsync(string repo, CancellationToken ct)
    {
        string[] skip = ["qualcomm", "mediatek", "intel", "sm8", "mt6", "tensor", "npu", "web", "ios"];
        return (await ListFilesAsync(repo, ct))
            .Where(f => f.Path.EndsWith(".litertlm", StringComparison.OrdinalIgnoreCase)
                        && !skip.Any(s => f.FileName.Contains(s, StringComparison.OrdinalIgnoreCase)))
            .Select(f => new GgufVariant(repo, LiteRtQuant(f.FileName), [f], null, "litertlm"))
            .OrderBy(v => v.WeightBytes)
            .ToList();
    }

    /// <summary>Distinct label per file ("LiteRT GPU", "LiteRT int4"), so each file gets its own model id.</summary>
    private static string LiteRtQuant(string fileName)
    {
        var name = Path.GetFileNameWithoutExtension(fileName).ToLowerInvariant();
        var tags = new[] { "int4", "int8", "q4", "q8", "f16", "fp16", "f32", "gpu", "cpu" }.Where(name.Contains).ToList();
        return tags.Count == 0 ? "LiteRT" : "LiteRT " + string.Join(" ", tags.Select(t => t is "gpu" or "cpu" ? t.ToUpperInvariant() : t));
    }

    [GeneratedRegex(@"(?<q>(?:UD-)?(?:IQ\d_[A-Z0-9]+|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))(?:-\d{5}-of-\d{5})?\.gguf$", RegexOptions.IgnoreCase)]
    private static partial Regex QuantRegex();

    public static IReadOnlyList<GgufVariant> GroupGguf(string repo, IEnumerable<HfFile> files)
    {
        // "mtp-*.gguf" files are multi-token-prediction draft heads that ship beside the weights; they are not models.
        var all = files.Where(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) &&
                                   !f.FileName.StartsWith("mtp-", StringComparison.OrdinalIgnoreCase)).ToList();
        var projectors = all.Where(f => f.FileName.Contains("mmproj", StringComparison.OrdinalIgnoreCase)).ToList();
        var mmproj = projectors.FirstOrDefault(f => f.FileName.Contains("F16", StringComparison.OrdinalIgnoreCase) && !f.FileName.Contains("BF16", StringComparison.OrdinalIgnoreCase))
                     ?? projectors.FirstOrDefault(f => f.FileName.Contains("BF16", StringComparison.OrdinalIgnoreCase))
                     ?? projectors.FirstOrDefault();

        return all.Except(projectors)
            .GroupBy(f =>
            {
                var match = QuantRegex().Match(f.FileName);
                return match.Success ? match.Groups["q"].Value.ToUpperInvariant() : "GGUF";
            })
            .Select(g => new GgufVariant(repo, g.Key, g.OrderBy(f => f.Path, StringComparer.Ordinal).ToList(), mmproj))
            .OrderBy(v => v.WeightBytes)
            .ToList();
    }
}
