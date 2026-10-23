using System.Net;
using System.Text;
using System.Text.RegularExpressions;

namespace LLMHub.Services;

public sealed record WebSource(int Number, string Title, string Url, string Snippet)
{
    public string Host => Uri.TryCreate(Url, UriKind.Absolute, out var u) ? u.Host.Replace("www.", "") : Url;
    [System.Text.Json.Serialization.JsonIgnore]
    public Uri? Link => Uri.TryCreate(Url, UriKind.Absolute, out var u) ? u : null;
}

/// <summary>
/// Web search for chat: DuckDuckGo's HTML results (no API key, same approach as the mobile apps), then the top pages are
/// fetched and reduced to text so the model can answer from them and cite them as [1], [2]. This is the only part of
/// chat that goes online, and only when the user turns it on.
/// </summary>
public static partial class WebSearch
{
    private static readonly HttpClient Http = CreateClient();

    private static HttpClient CreateClient()
    {
        var http = new HttpClient(new SocketsHttpHandler { AutomaticDecompression = DecompressionMethods.All, AllowAutoRedirect = true })
        {
            Timeout = TimeSpan.FromSeconds(12),
        };
        // A browser user agent; DuckDuckGo's HTML endpoint refuses obvious bots.
        http.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0");
        http.DefaultRequestHeaders.AcceptLanguage.ParseAdd("en-US,en;q=0.8");
        return http;
    }

    public static async Task<List<WebSource>> SearchAsync(string query, int max, CancellationToken ct)
    {
        using var form = new FormUrlEncodedContent([new("q", query), new("kl", "wt-wt")]);
        using var response = await Http.PostAsync("https://html.duckduckgo.com/html/", form, ct);
        response.EnsureSuccessStatusCode();
        var html = await response.Content.ReadAsStringAsync(ct);

        var results = new List<WebSource>();
        foreach (Match m in ResultRegex().Matches(html))
        {
            var href = HrefRegex().Match(m.Groups["attrs"].Value).Groups[1].Value;
            var url = RealUrl(WebUtility.HtmlDecode(href));
            if (url is null || url.Contains("duckduckgo.com", StringComparison.OrdinalIgnoreCase) || results.Any(r => r.Url == url))
                continue;
            var title = Clean(m.Groups["title"].Value);
            var snippet = Clean(m.Groups["snippet"].Value);
            results.Add(new WebSource(results.Count + 1, title, url, snippet));
            if (results.Count == max)
                break;
        }
        Log.Info($"Web search \"{query}\": {results.Count} results");
        return results;
    }

    /// <summary>Readable text of a page (scripts, styles and tags removed), or "" when it cannot be fetched in time.</summary>
    public static async Task<string> FetchTextAsync(string url, int maxChars, CancellationToken ct)
    {
        try
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeout.CancelAfter(TimeSpan.FromSeconds(8));
            using var response = await Http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
            if (!response.IsSuccessStatusCode || response.Content.Headers.ContentType?.MediaType is not ("text/html" or "text/plain"))
                return "";
            var html = await response.Content.ReadAsStringAsync(timeout.Token);
            var text = Clean(NoiseRegex().Replace(html, " "));
            return text.Length <= maxChars ? text : text[..maxChars];
        }
        catch (Exception e) when (e is HttpRequestException or TaskCanceledException or InvalidOperationException)
        {
            return "";
        }
    }

    /// <summary>The system note that carries search results into the conversation.</summary>
    public static async Task<(string Context, List<WebSource> Sources)> BuildContextAsync(string question, CancellationToken ct)
    {
        var query = question.Length <= 200 ? question : question[..200];
        var sources = await SearchAsync(query, 5, ct);
        if (sources.Count == 0)
            return ("", sources);
        var pages = await Task.WhenAll(sources.Take(3).Select(s => FetchTextAsync(s.Url, 2500, ct)));
        var sb = new StringBuilder();
        sb.AppendLine($"Web search results for \"{query}\" (today is {DateTime.Now:yyyy-MM-dd}). Use them to answer, cite them inline as [1], [2], and say so if they do not answer the question.");
        foreach (var s in sources)
        {
            sb.AppendLine().Append('[').Append(s.Number).Append("] ").AppendLine(s.Title).AppendLine(s.Url);
            var page = s.Number <= pages.Length ? pages[s.Number - 1] : "";
            sb.AppendLine(page.Length > s.Snippet.Length ? page : s.Snippet);
        }
        return (sb.ToString(), sources);
    }

    /// <summary>DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=&lt;encoded url&gt;.</summary>
    private static string? RealUrl(string href)
    {
        if (href.StartsWith("//", StringComparison.Ordinal))
            href = "https:" + href;
        if (!Uri.TryCreate(href, UriKind.Absolute, out var uri))
            return null;
        if (uri.Host.EndsWith("duckduckgo.com", StringComparison.OrdinalIgnoreCase) && uri.AbsolutePath == "/l/")
        {
            var query = System.Web.HttpUtility.ParseQueryString(uri.Query);
            return query["uddg"];
        }
        return uri.Scheme is "http" or "https" ? uri.ToString() : null;
    }

    private static string Clean(string html) =>
        WhitespaceRegex().Replace(WebUtility.HtmlDecode(TagRegex().Replace(html, " ")), " ").Trim();

    [GeneratedRegex(@"<a(?<attrs>[^>]*class=""result__a""[^>]*)>(?<title>.*?)</a>.*?class=""result__snippet""[^>]*>(?<snippet>.*?)</(?:a|div|td)>", RegexOptions.Singleline)]
    private static partial Regex ResultRegex();

    [GeneratedRegex(@"href=""([^""]+)""")]
    private static partial Regex HrefRegex();

    [GeneratedRegex(@"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex NoiseRegex();

    [GeneratedRegex("<[^>]+>")]
    private static partial Regex TagRegex();

    [GeneratedRegex(@"\s+")]
    private static partial Regex WhitespaceRegex();
}
