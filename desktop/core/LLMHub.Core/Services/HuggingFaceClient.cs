using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LLMHub.Services;

public sealed record HfFile(string Path, long Size, string? Sha256)
{
    public string FileName => System.IO.Path.GetFileName(Path);
}

/// <summary>A search result. <paramref name="Format"/> is "gguf" or "litertlm"; <paramref name="Trending"/> is the Hub's trending score.</summary>
public sealed record HfRepo(string Id, long Downloads, long Likes, DateTimeOffset? LastModified, bool Gated, string Format = "gguf", double Trending = 0);

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
        _http = new HttpClient(new RetryHandler(new SocketsHttpHandler { AutomaticDecompression = System.Net.DecompressionMethods.All }))
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

    /// <summary>
    /// Repos matching <paramref name="query"/> with the "gguf" or "litert-lm" library tag, sorted by "trendingScore",
    /// "downloads" or "lastModified" (newest first).
    /// </summary>
    public async Task<IReadOnlyList<HfRepo>> SearchAsync(string query, string? libraryFilter, CancellationToken ct, string sort = "trendingScore")
    {
        var url = $"{Host}/api/models?search={Uri.EscapeDataString(query)}&sort={Uri.EscapeDataString(sort)}&direction=-1&limit=40&expand[]=downloads&expand[]=likes"
                  + "&expand[]=lastModified&expand[]=gated&expand[]=trendingScore";
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
            var trending = item.TryGetProperty("trendingScore", out var t) && t.TryGetDouble(out var ts) ? ts : 0;
            if (IsChineseOrigin(id))
                continue;
            list.Add(new HfRepo(id, downloads, likes, modified, gated, libraryFilter == "litert-lm" ? "litertlm" : "gguf", trending));
        }
        return list;
    }

    /// <summary>
    /// LLM Hub offers no models of Chinese origin, nor quants and fine-tunes of them; search results drop repos whose owner
    /// or name points to one of those model families.
    /// </summary>
    public static bool IsChineseOrigin(string repoId) => ChineseOriginRegex().IsMatch(repoId);

    [GeneratedRegex(@"(^|[/_\-. ])(qwen|qwq|deepseek|glm|chatglm|codegeex|thudm|zai-org|01-ai|yi|baichuan|internlm|internvl|intern-s1|minimax|kimi|moonshot|hunyuan|tencent|ernie|baidu|wan|wan-ai|wan2|step|stepfun|step3|seed-oss|bytedance|doubao|cogvideo|cogview|minicpm|openbmb|telechat|skywork|xverse|aquila|baai|bailing|ling|ring|inclusionai|longcat|meituan|mimo|xiaomi|xiaomimimo|kwai|kuaishou|keye|ovis|seedream|seedance|opengvlab|iflytek|spark|sensenova|pangu|hy|mistral-chinese)(?=$|[/_\-.0-9 ])", RegexOptions.IgnoreCase)]
    private static partial Regex ChineseOriginRegex();

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

        // One variant per model file, with the parts of a split model ("-00001-of-00003") kept together. Two different
        // files can carry the same quant (LFM2.5's Q4_0 and QAD-Q4_0): those get the words that tell them apart.
        var groups = all.Except(projectors)
            .GroupBy(f => SplitSuffixRegex().Replace(f.Path, ".gguf"), StringComparer.OrdinalIgnoreCase)
            .Select(g =>
            {
                var match = QuantRegex().Match(g.First().FileName);
                return (Quant: match.Success ? match.Groups["q"].Value.ToUpperInvariant() : "GGUF",
                    Stem: Path.GetFileNameWithoutExtension(g.Key), Files: g.OrderBy(f => f.Path, StringComparer.Ordinal).ToList());
            })
            .ToList();
        return groups
            .GroupBy(v => v.Quant)
            .SelectMany(same => same.Count() == 1
                ? same.Select(v => (Label: v.Quant, v.Files))
                : same.Select(v => (Label: Distinct(v.Stem, same.Select(o => o.Stem).ToList()), v.Files)))
            .Select(v => new GgufVariant(repo, v.Label, v.Files, mmproj))
            .OrderBy(v => v.WeightBytes)
            .ToList();
    }

    /// <summary>The words of <paramref name="stem"/> after those every file in <paramref name="stems"/> shares ("QAD-Q4_0").</summary>
    private static string Distinct(string stem, IReadOnlyList<string> stems)
    {
        var words = stem.Split('-', '_', '.');
        var common = 0;
        while (common < words.Length - 1 && stems.All(s => s.Split('-', '_', '.') is var w && w.Length > common && w[common].Equals(words[common], StringComparison.OrdinalIgnoreCase)))
            common++;
        var rest = stem;
        for (var i = 0; i < common; i++)
            rest = rest[(rest.IndexOf(words[i], StringComparison.Ordinal) + words[i].Length)..].TrimStart('-', '_', '.');
        return rest.Length > 0 ? rest.ToUpperInvariant() : stem;
    }

    [GeneratedRegex(@"-\d{5}-of-\d{5}\.gguf$", RegexOptions.IgnoreCase)]
    private static partial Regex SplitSuffixRegex();
}
