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
    private const string MobileUserAgent = "Mozilla/5.0 (Android 10; Mobile; rv:91.0) Gecko/91.0 Firefox/91.0";
    private static readonly HttpClient Http = CreateClient();

    private static HttpClient CreateClient()
    {
        var http = new HttpClient(new RetryHandler(new SocketsHttpHandler { AutomaticDecompression = DecompressionMethods.All, AllowAutoRedirect = true }))
        {
            Timeout = TimeSpan.FromSeconds(12),
        };
        // The Android and iOS apps' request: Firefox for Android, which DuckDuckGo's HTML page serves without a bot check
        // far more often than a desktop browser posting the form.
        http.DefaultRequestHeaders.UserAgent.ParseAdd(MobileUserAgent);
        return http;
    }

    public static async Task<List<WebSource>> SearchAsync(string query, int max, CancellationToken ct)
    {
        var results = await TryAsync("DuckDuckGo", () => SearchHtmlAsync(query, max, ct), ct);
        // DuckDuckGo throttles repeated queries from one address with an empty or challenge page; its lite page often still answers.
        if (results.Count == 0)
            results = await TryAsync("DuckDuckGo lite", () => SearchLiteAsync(query, max, ct), ct);
        // Search engines answer scripted queries with bot challenges at times; Wikipedia's API is made for programs.
        if (results.Count == 0)
            results = await TryAsync("Wikipedia", () => SearchWikipediaAsync(query, max, ct), ct);
        Log.Info($"Web search \"{query}\": {results.Count} results");
        return results;
    }

    /// <summary>A source that can't be reached or times out (HttpClient throws TaskCanceledException) hands over to the next one.</summary>
    private static async Task<List<WebSource>> TryAsync(string source, Func<Task<List<WebSource>>> search, CancellationToken ct)
    {
        try
        {
            return await search();
        }
        catch (Exception e) when (!ct.IsCancellationRequested && e is HttpRequestException or TaskCanceledException or IOException)
        {
            Log.Info($"Web search: {source} failed ({e.Message})");
            return [];
        }
    }

    private static async Task<List<WebSource>> SearchHtmlAsync(string query, int max, CancellationToken ct)
    {
        using var response = await Http.GetAsync($"https://duckduckgo.com/html/?q={Uri.EscapeDataString(query)}", ct);
        var html = await response.Content.ReadAsStringAsync(ct);
        var results = new List<WebSource>();
        foreach (Match m in ResultRegex().Matches(html))
        {
            Add(results, HrefRegex().Match(m.Groups["attrs"].Value).Groups[1].Value, m.Groups["title"].Value, m.Groups["snippet"].Value);
            if (results.Count == max)
                break;
        }
        if (results.Count == 0)
            Log.Info($"Web search: DuckDuckGo HTML gave no results (HTTP {(int)response.StatusCode}{(html.Contains("anomaly", StringComparison.OrdinalIgnoreCase) ? ", challenge page" : "")})");
        return results;
    }

    private static async Task<List<WebSource>> SearchLiteAsync(string query, int max, CancellationToken ct)
    {
        try
        {
            using var response = await Http.GetAsync($"https://lite.duckduckgo.com/lite/?q={Uri.EscapeDataString(query)}", ct);
            var html = await response.Content.ReadAsStringAsync(ct);
            var links = LiteLinkRegex().Matches(html);
            var snippets = LiteSnippetRegex().Matches(html);
            var results = new List<WebSource>();
            for (var i = 0; i < links.Count && results.Count < max; i++)
                Add(results, links[i].Groups["href"].Value, links[i].Groups["title"].Value, i < snippets.Count ? snippets[i].Groups["snippet"].Value : "");
            if (results.Count == 0)
                Log.Info($"Web search: DuckDuckGo lite gave no results either (HTTP {(int)response.StatusCode})");
            return results;
        }
        catch (HttpRequestException e)
        {
            Log.Error("Web search (lite) failed", e);
            return [];
        }
    }

    /// <summary>Wikipedia in the app's language: narrower than the web, but it never blocks a well-behaved client.</summary>
    private static async Task<List<WebSource>> SearchWikipediaAsync(string query, int max, CancellationToken ct)
    {
        var tag = AppSettings.Current.Language is { Length: >= 2 } chosen ? chosen : System.Globalization.CultureInfo.CurrentUICulture.Name;
        var language = tag.Length >= 2 && char.IsLetter(tag[0]) && char.IsLetter(tag[1]) ? tag[..2].ToLowerInvariant() : "en";
        // Wikipedia matches every word, so send keywords, not the request ("search the web", "one sentence"…);
        // if an odd word still blocks a match, any keyword may match (ranked by relevance).
        var keywords = Keywords(query);
        if (keywords.Count == 0)
            return [];
        try
        {
            foreach (var search in new[] { string.Join(' ', keywords), string.Join(" OR ", keywords) })
            {
                var results = await WikipediaQueryAsync(language, search, max, ct);
                if (results.Count > 0)
                {
                    Log.Info($"Web search: answered from {language}.wikipedia.org (\"{search}\")");
                    return results;
                }
            }
            return [];
        }
        catch (Exception e) when (e is HttpRequestException or System.Text.Json.JsonException or KeyNotFoundException)
        {
            Log.Error("Web search (Wikipedia) failed", e);
            return [];
        }
    }

    private static async Task<List<WebSource>> WikipediaQueryAsync(string language, string search, int max, CancellationToken ct)
    {
        var url = $"https://{language}.wikipedia.org/w/api.php?action=query&list=search&format=json&srprop=snippet&srlimit={max}&srsearch={Uri.EscapeDataString(search)}";
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        // Wikimedia asks API clients to identify themselves.
        request.Headers.UserAgent.Clear();
        request.Headers.UserAgent.ParseAdd("LLMHub/1.0 (https://llm-hub.app)");
        using var response = await Http.SendAsync(request, ct);
        response.EnsureSuccessStatusCode();
        using var doc = System.Text.Json.JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
        var results = new List<WebSource>();
        foreach (var hit in doc.RootElement.GetProperty("query").GetProperty("search").EnumerateArray())
        {
            var title = hit.GetProperty("title").GetString() ?? "";
            Add(results, $"https://{language}.wikipedia.org/wiki/{Uri.EscapeDataString(title.Replace(' ', '_'))}", title, hit.GetProperty("snippet").GetString() ?? "");
        }
        return results;
    }

    /// <summary>The words of a request that say what it is about, without filler and instructions to the assistant.</summary>
    private static List<string> Keywords(string text)
    {
        string[] filler =
        [
            "search", "web", "internet", "online", "google", "look", "find", "tell", "me", "please", "answer", "reply", "cite", "source", "sources",
            "one", "two", "sentence", "sentences", "word", "words", "short", "briefly", "brief", "explain", "describe", "give", "show", "list",
            "what", "whats", "who", "whom", "which", "when", "where", "why", "how", "is", "are", "was", "were", "be", "been", "do", "does", "did",
            "can", "could", "would", "should", "will", "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with", "about", "from",
            "by", "at", "as", "it", "its", "this", "that", "these", "those", "there", "i", "you", "your", "my", "we", "our", "up", "out", "into",
        ];
        return WordRegex().Matches(text.ToLowerInvariant()).Select(m => m.Value)
            .Where(w => w.Length > 1 && !filler.Contains(w)).Distinct().Take(8).ToList();
    }

    private static void Add(List<WebSource> results, string href, string title, string snippet)
    {
        var url = RealUrl(WebUtility.HtmlDecode(href));
        if (url is null || url.Contains("duckduckgo.com", StringComparison.OrdinalIgnoreCase) || results.Any(r => r.Url == url))
            return;
        results.Add(new WebSource(results.Count + 1, Clean(title), url, Clean(snippet)));
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

    [GeneratedRegex(@"<a(?=[^>]*class=['""]result-link['""])[^>]*href=['""](?<href>[^'""]+)['""][^>]*>(?<title>.*?)</a>", RegexOptions.Singleline)]
    private static partial Regex LiteLinkRegex();

    [GeneratedRegex(@"<td[^>]*class=['""]result-snippet['""][^>]*>(?<snippet>.*?)</td>", RegexOptions.Singleline)]
    private static partial Regex LiteSnippetRegex();

    [GeneratedRegex(@"[\p{L}\p{N}]+")]
    private static partial Regex WordRegex();
}
