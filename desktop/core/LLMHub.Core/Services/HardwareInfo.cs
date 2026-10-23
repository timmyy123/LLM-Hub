using System.Runtime.InteropServices;
using Microsoft.Win32;

namespace LLMHub.Services;

public sealed partial class HardwareInfo
{
    public ulong TotalRamBytes { get; private set; }
    public ulong GpuVramBytes { get; private set; }
    public string GpuName { get; private set; } = "";
    public bool HasNvidiaGpu { get; private set; }
    public bool HasAdrenoGpu { get; private set; }
    public bool HasAmdGpu { get; private set; }
    public bool HasIntelGpu { get; private set; }
    /// <summary>NPUs show up in the ComputeAccelerator device class (Intel AI Boost, AMD Ryzen AI, Qualcomm Hexagon).</summary>
    public string NpuName { get; private set; } = "";
    /// <summary>An integrated GPU that uses system memory (Snapdragon Adreno).</summary>
    public string SharedGpuName { get; private set; } = "";
    public bool IsArm64 => RuntimeInformation.OSArchitecture == Architecture.Arm64;

    public double RamGB => TotalRamBytes / 1073741824.0;
    public double VramGB => GpuVramBytes / 1073741824.0;

    public static HardwareInfo Detect()
    {
        var info = new HardwareInfo();
        if (OperatingSystem.IsWindows())
            info.DetectWindows();
        else if (OperatingSystem.IsLinux())
            info.DetectLinux();
        Log.Info($"Hardware: RAM {info.RamGB:F1} GB, GPU '{info.GpuName}' {info.VramGB:F1} GB, NVIDIA={info.HasNvidiaGpu}, AMD={info.HasAmdGpu}, Intel={info.HasIntelGpu}, Adreno={info.HasAdrenoGpu}, NPU='{info.NpuName}', ARM64={info.IsArm64}");
        return info;
    }

    [System.Runtime.Versioning.SupportedOSPlatform("linux")]
    private void DetectLinux()
    {
        // RAM: MemTotal in kB.
        var mem = File.ReadLines("/proc/meminfo").FirstOrDefault(l => l.StartsWith("MemTotal:"));
        if (mem is not null && ulong.TryParse(mem.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1], out var kb))
            TotalRamBytes = kb * 1024;

        // NVIDIA (also inside WSL, where /sys has no DRM devices but nvidia-smi works).
        try
        {
            var smi = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo("nvidia-smi", "--query-gpu=name,memory.total --format=csv,noheader,nounits")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
            });
            var line = smi?.StandardOutput.ReadLine();
            smi?.WaitForExit(5000);
            if (line?.Split(',') is [var name, var mib] && ulong.TryParse(mib.Trim(), out var mb))
            {
                HasNvidiaGpu = true;
                GpuName = name.Trim();
                GpuVramBytes = mb * 1048576;
            }
        }
        catch
        {
            // No NVIDIA driver.
        }

        // AMD and Intel GPUs, and NPUs (amdxdna, intel_vpu), from sysfs.
        foreach (var card in Directory.Exists("/sys/class/drm") ? Directory.GetDirectories("/sys/class/drm", "card?") : [])
        {
            var vendor = Read(Path.Combine(card, "device/vendor"));
            if (vendor == "0x1002")
            {
                HasAmdGpu = true;
                if (ulong.TryParse(Read(Path.Combine(card, "device/mem_info_vram_total")), out var vram) && vram > GpuVramBytes)
                {
                    GpuVramBytes = vram;
                    GpuName = "AMD Radeon";
                }
            }
            else if (vendor == "0x8086")
                HasIntelGpu = true;
        }
        foreach (var accel in Directory.Exists("/sys/class/accel") ? Directory.GetDirectories("/sys/class/accel") : [])
        {
            var driver = Path.GetFileName(Read(Path.Combine(accel, "device/uevent")).Split('\n').FirstOrDefault(l => l.StartsWith("DRIVER="))?[7..] ?? "");
            NpuName = driver switch { "amdxdna" => "AMD Ryzen AI NPU", "intel_vpu" => "Intel AI Boost NPU", _ => driver };
        }

        static string Read(string path) => File.Exists(path) ? File.ReadAllText(path).Trim() : "";
    }

    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    private void DetectWindows()
    {
        var info = this;
        var status = new MemoryStatusEx { Length = (uint)Marshal.SizeOf<MemoryStatusEx>() };
        if (GlobalMemoryStatusEx(ref status))
            info.TotalRamBytes = status.TotalPhys;

        // Display adapter class key: HardwareInformation.qwMemorySize is the dedicated VRAM as a 64-bit value.
        try
        {
            using var adapters = Registry.LocalMachine.OpenSubKey(
                @"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}");
            foreach (var name in adapters?.GetSubKeyNames() ?? [])
            {
                if (!int.TryParse(name, out _))
                    continue;
                try
                {
                    using var key = adapters!.OpenSubKey(name);
                    var desc = key?.GetValue("DriverDesc") as string ?? "";
                    var vram = key?.GetValue("HardwareInformation.qwMemorySize") switch
                    {
                        long l => (ulong)l,
                        byte[] b when b.Length >= 8 => BitConverter.ToUInt64(b, 0),
                        _ => 0UL,
                    };
                    if (desc.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase))
                        info.HasNvidiaGpu = true;
                    if (desc.Contains("Radeon", StringComparison.OrdinalIgnoreCase) || desc.StartsWith("AMD", StringComparison.OrdinalIgnoreCase))
                        info.HasAmdGpu = true;
                    if (desc.Contains("Intel", StringComparison.OrdinalIgnoreCase) && (desc.Contains("Arc", StringComparison.OrdinalIgnoreCase) || desc.Contains("Graphics", StringComparison.OrdinalIgnoreCase)))
                        info.HasIntelGpu = true;
                    if (desc.Contains("Adreno", StringComparison.OrdinalIgnoreCase) || desc.Contains("Qualcomm", StringComparison.OrdinalIgnoreCase))
                    {
                        info.HasAdrenoGpu = true;
                        info.SharedGpuName = desc;
                    }
                    if (vram > info.GpuVramBytes)
                    {
                        info.GpuVramBytes = vram;
                        info.GpuName = desc;
                    }
                }
                catch
                {
                    // Some adapter subkeys deny read access; skip them.
                }
            }
        }
        catch (Exception e)
        {
            Log.Error("GPU detection failed", e);
        }

        try
        {
            using var accelerators = Registry.LocalMachine.OpenSubKey(
                @"SYSTEM\CurrentControlSet\Control\Class\{f01a9d53-3ff6-48d2-9f97-c8a7004be10c}");
            foreach (var name in accelerators?.GetSubKeyNames() ?? [])
            {
                using var key = accelerators!.OpenSubKey(name);
                if (key?.GetValue("DriverDesc") is string desc && desc.Length > 0)
                    info.NpuName = desc;
            }
        }
        catch
        {
            // No accelerator class on this PC.
        }

    }

    /// <summary>A GPU with its own memory worth offloading to (not a small shared iGPU allocation).</summary>
    public bool HasDedicatedGpu => GpuVramBytes >= 4L * 1073741824;

    /// <summary>"RTX 4090 · 16 GB VRAM · 32 GB RAM", "Adreno X2-90 · 31 GB shared memory", or RAM only.</summary>
    public string Summary(bool withRam) =>
        HasDedicatedGpu ? (withRam ? Loc.F("HardwareSummary", GpuName, Math.Round(VramGB), Math.Round(RamGB)) : $"{GpuName} · {Math.Round(VramGB)} GB")
        : SharedGpuName.Length > 0 ? Loc.F("HardwareSummaryShared", SharedGpuName, Math.Round(RamGB))
        : Loc.F("HardwareSummaryCpu", Math.Round(RamGB));

    /// <summary>Only offer quants that run fully on the GPU; without a dedicated GPU, those that fit in RAM.</summary>
    public bool ShouldOffer(ModelFit fit) => HasDedicatedGpu ? fit == ModelFit.Gpu : fit != ModelFit.TooLarge;

    public ModelFit Fit(long modelBytes)
    {
        // Weights plus room for the KV cache and runtime buffers.
        var needed = modelBytes * 1.2;
        if (GpuVramBytes > 0 && needed <= GpuVramBytes * 0.92)
            return ModelFit.Gpu;
        if (needed <= GpuVramBytes * 0.92 + TotalRamBytes * 0.55)
            return ModelFit.Partial;
        return ModelFit.TooLarge;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatusEx
    {
        public uint Length;
        public uint MemoryLoad;
        public ulong TotalPhys;
        public ulong AvailPhys;
        public ulong TotalPageFile;
        public ulong AvailPageFile;
        public ulong TotalVirtual;
        public ulong AvailVirtual;
        public ulong AvailExtendedVirtual;
    }

    [LibraryImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GlobalMemoryStatusEx(ref MemoryStatusEx buffer);
}

public enum ModelFit
{
    Gpu,
    Partial,
    TooLarge,
}
