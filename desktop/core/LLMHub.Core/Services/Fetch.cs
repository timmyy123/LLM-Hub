using System.IO.Compression;
using System.Text.Json;

namespace LLMHub.Services;

/// <summary>Downloads for engines and support files (speech, Studio): resumable, with progress, plus archive extraction.</summary>
public static class Fetch
{
    private static readonly HttpClient Http = CreateClient();

    private static HttpClient CreateClient()
    {
        var http = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("LLMHub-Desktop/0.1");
        return http;
    }

    /// <summary>Downloads to <paramref name="path"/> via a .part file that resumes after an interruption.</summary>
    public static async Task FileAsync(string url, string path, Action<double>? progress, CancellationToken ct)
    {
        if (File.Exists(path))
            return;
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var part = path + ".part";
        var existing = File.Exists(part) ? new FileInfo(part).Length : 0;
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (existing > 0)
            request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(existing, null);
        using var response = await Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        if (response.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            File.Move(part, path, overwrite: true);
            return;
        }
        response.EnsureSuccessStatusCode();
        if (response.StatusCode != System.Net.HttpStatusCode.PartialContent)
            existing = 0;
        var total = existing + (response.Content.Headers.ContentLength ?? 0);
        await using (var output = new FileStream(part, existing > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
        await using (var input = await response.Content.ReadAsStreamAsync(ct))
        {
            var buffer = new byte[1 << 20];
            var done = existing;
            var last = DateTime.MinValue;
            int read;
            while ((read = await input.ReadAsync(buffer, ct)) > 0)
            {
                await output.WriteAsync(buffer.AsMemory(0, read), ct);
                done += read;
                if (total > 0 && (DateTime.UtcNow - last).TotalMilliseconds > 250)
                {
                    progress?.Invoke(done * 100.0 / total);
                    last = DateTime.UtcNow;
                }
            }
        }
        File.Move(part, path, overwrite: true);
        progress?.Invoke(100);
    }

    /// <summary>Unpacks .zip or .tar.gz into <paramref name="target"/>.</summary>
    public static void Extract(string archive, string target)
    {
        Directory.CreateDirectory(target);
        if (archive.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
        {
            ZipFile.ExtractToDirectory(archive, target, overwriteFiles: true);
            return;
        }
        using var file = File.OpenRead(archive);
        using var gz = new GZipStream(file, CompressionMode.Decompress);
        System.Formats.Tar.TarFile.ExtractToDirectory(gz, target, overwriteFiles: true);
    }

    /// <summary>The newest GitHub release (of the first <paramref name="scan"/>) that has an asset named <paramref name="asset"/>.</summary>
    public static async Task<(string Tag, string Url)?> GitHubAssetAsync(string repo, Func<string, bool> asset, CancellationToken ct, int scan = 20)
    {
        using var doc = JsonDocument.Parse(await Http.GetStringAsync($"https://api.github.com/repos/{repo}/releases?per_page={scan}", ct));
        foreach (var release in doc.RootElement.EnumerateArray())
        {
            foreach (var a in release.GetProperty("assets").EnumerateArray())
            {
                if (asset(a.GetProperty("name").GetString()!))
                    return (release.GetProperty("tag_name").GetString()!, a.GetProperty("browser_download_url").GetString()!);
            }
        }
        return null;
    }
}
