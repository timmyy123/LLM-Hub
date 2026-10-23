using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Net;
using System.Security.Cryptography;
using CommunityToolkit.Mvvm.ComponentModel;

namespace LLMHub.Services;

public sealed partial class DownloadJob : ObservableObject
{
    public DownloadJob(string id, string title)
    {
        Id = id;
        Title = title;
        Status = "";
    }

    public string Id { get; }
    public string Title { get; }
    public CancellationTokenSource Cancellation { get; } = new();

    [ObservableProperty] public partial double Percent { get; set; }
    [ObservableProperty] public partial string Status { get; set; }
    [ObservableProperty] public partial bool IsRunning { get; set; }
}

/// <summary>Resumable, hash-checked downloads from Hugging Face. Partial files survive restarts as *.part.</summary>
public sealed class DownloadManager(HuggingFaceClient hf, ModelStore store)
{
    private readonly HttpClient _http = new(new SocketsHttpHandler
    {
        AllowAutoRedirect = true,
        MaxAutomaticRedirections = 10,
        PooledConnectionLifetime = TimeSpan.FromMinutes(5),
    })
    {
        Timeout = Timeout.InfiniteTimeSpan,
    };

    public ObservableCollection<DownloadJob> Jobs { get; } = [];

    /// <summary>Raised on the UI thread when a job ends: model id and a user-facing error, or null on success/cancel.</summary>
    public event Action<string, string?>? Finished;

    public DownloadJob? Find(string modelId) => Jobs.FirstOrDefault(j => j.Id == modelId);

    public DownloadJob Start(GgufVariant variant, string name, string? catalogId, string[] tags, int contextLength)
    {
        var id = InstalledModel.MakeId(variant.Repo, variant.Quant);
        if (Find(id) is { } running)
            return running;

        var job = new DownloadJob(id, $"{name} · {variant.Quant}") { IsRunning = true };
        Jobs.Add(job);
        _ = RunAsync(job, variant, name, catalogId, tags, contextLength);
        return job;
    }

    public void Cancel(DownloadJob job) => job.Cancellation.Cancel();

    private async Task RunAsync(DownloadJob job, GgufVariant variant, string name, string? catalogId, string[] tags, int contextLength)
    {
        var folder = store.FolderFor(job.Id);
        Directory.CreateDirectory(folder);
        var files = variant.Files.ToList();
        if (variant.Mmproj is not null)
            files.Add(variant.Mmproj);
        var total = files.Sum(f => f.Size);
        long completedBefore = 0;
        string? error = null;

        try
        {
            foreach (var file in files)
            {
                var target = Path.Combine(folder, file.FileName);
                if (!File.Exists(target))
                    await DownloadFileAsync(variant.Repo, file, target, done => Report(job, completedBefore + done, total), job.Cancellation.Token);
                completedBefore += file.Size;
            }

            store.Add(new InstalledModel
            {
                Id = job.Id,
                Name = name,
                CatalogId = catalogId,
                Repo = variant.Repo,
                Format = variant.Format,
                Quant = variant.Quant,
                Files = variant.Files.Select(f => f.FileName).ToList(),
                Mmproj = variant.Mmproj?.FileName,
                SizeBytes = total,
                ContextLength = contextLength,
                Tags = tags,
                InstalledAt = DateTimeOffset.Now,
            });
            Log.Info($"Installed {job.Id}");
        }
        catch (OperationCanceledException)
        {
            Log.Info($"Download {job.Id} paused");
        }
        catch (HttpRequestException e) when (e.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
        {
            Log.Error($"Download {job.Id} needs access", e);
            error = Loc.S("NeedsToken");
        }
        catch (Exception e)
        {
            Log.Error($"Download {job.Id} failed", e);
            error = Loc.F("DownloadFailed", e.Message);
        }
        finally
        {
            // A download refused before any data arrived (gated, no access) would otherwise leave an empty folder behind.
            if (error is not null && Directory.Exists(folder) && !Directory.EnumerateFileSystemEntries(folder).Any())
                Directory.Delete(folder);
            UiThread.Run(() =>
            {
                job.IsRunning = false;
                Jobs.Remove(job);
                Finished?.Invoke(job.Id, error);
            });
        }
    }

    private async Task DownloadFileAsync(string repo, HfFile file, string target, Action<long> progress, CancellationToken ct)
    {
        var part = target + ".part";
        var existing = File.Exists(part) ? new FileInfo(part).Length : 0;
        if (existing > file.Size && file.Size > 0)
        {
            File.Delete(part);
            existing = 0;
        }

        using var request = hf.Request(HttpMethod.Get, HuggingFaceClient.ResolveUrl(repo, file.Path));
        if (existing > 0)
            request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(existing, null);

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        if (response.StatusCode == HttpStatusCode.RequestedRangeNotSatisfiable)
            existing = file.Size;
        else
        {
            response.EnsureSuccessStatusCode();
            if (existing > 0 && response.StatusCode != HttpStatusCode.PartialContent)
                existing = 0; // Server ignored the range; start over.

            await using var input = await response.Content.ReadAsStreamAsync(ct);
            await using var output = new FileStream(part, existing > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20, useAsync: true);
            var buffer = new byte[1 << 20];
            var done = existing;
            var clock = Stopwatch.StartNew();
            int read;
            while ((read = await input.ReadAsync(buffer, ct)) > 0)
            {
                await output.WriteAsync(buffer.AsMemory(0, read), ct);
                done += read;
                if (clock.ElapsedMilliseconds > 250)
                {
                    progress(done);
                    clock.Restart();
                }
            }
        }

        if (file.Sha256 is { Length: 64 } expected)
        {
            await using var stream = new FileStream(part, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, useAsync: true);
            var actual = Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, ct));
            if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
            {
                File.Delete(part);
                throw new InvalidDataException($"Checksum mismatch for {file.FileName}");
            }
        }

        File.Move(part, target, overwrite: true);
        progress(file.Size);
    }

    private static void Report(DownloadJob job, long done, long total) => UiThread.Run(() =>
    {
        job.Percent = total > 0 ? done * 100.0 / total : 0;
        job.Status = $"{Fmt.Bytes(done)} / {Fmt.Bytes(total)}";
    });
}
