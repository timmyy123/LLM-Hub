using System.Diagnostics;
using System.Text;

namespace LLMHub.Services.Studio;

/// <summary>
/// A private CPython for engines that need one (ComfyUI, Magenta RealTime): relocatable builds from python-build-standalone
/// unpacked into the engine's folder, so the user never installs or configures Python.
/// </summary>
public static class StandalonePython
{
    private const string Tag = "3.12.";

    public static string Exe(string root) =>
        OperatingSystem.IsWindows() ? Path.Combine(root, "python", "python.exe") : Path.Combine(root, "python", "bin", "python3");

    public static async Task EnsureAsync(string root, Action<double> progress, CancellationToken ct)
    {
        if (File.Exists(Exe(root)))
            return;
        Directory.CreateDirectory(root);
        var arm = Ioc.Hardware.IsArm64;
        var triple = OperatingSystem.IsWindows()
            ? arm ? "aarch64-pc-windows-msvc" : "x86_64-pc-windows-msvc"
            : arm ? "aarch64-unknown-linux-gnu" : "x86_64-unknown-linux-gnu";
        var asset = await Fetch.GitHubAssetAsync("astral-sh/python-build-standalone",
            n => n.StartsWith("cpython-" + Tag, StringComparison.Ordinal) && n.EndsWith($"-{triple}-install_only.tar.gz", StringComparison.Ordinal), ct, scan: 3)
            ?? throw new InvalidOperationException("No standalone Python for this computer");
        var archive = Path.Combine(root, "python.tar.gz");
        await Fetch.FileAsync(asset.Url, archive, progress, ct);
        Fetch.Extract(archive, root);
        File.Delete(archive);
    }

    /// <summary>Runs pip, showing "Collecting …" and "Downloading …" lines through <paramref name="status"/>.</summary>
    public static async Task PipAsync(string root, Action<string> status, CancellationToken ct, params string[] args)
    {
        var info = new ProcessStartInfo(Exe(root)) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var arg in new[] { "-m", "pip" }.Concat(args).Append("--disable-pip-version-check").Append("--no-input"))
            info.ArgumentList.Add(arg);
        using var process = Process.Start(info)!;
        var tail = new StringBuilder();
        process.OutputDataReceived += (_, e) =>
        {
            if (e.Data is { } line && (line.StartsWith("Collecting", StringComparison.Ordinal) || line.StartsWith("Downloading", StringComparison.Ordinal)))
                status(line.Trim()[..Math.Min(line.Trim().Length, 90)]);
        };
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is { } line)
                lock (tail)
                    tail.AppendLine(line);
        };
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        await process.WaitForExitAsync(ct);
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"pip {string.Join(' ', args.Take(3))} failed: {tail.ToString()[^Math.Min(tail.Length, 600)..]}");
    }
}
