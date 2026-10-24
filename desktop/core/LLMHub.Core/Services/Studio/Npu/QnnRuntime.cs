using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace LLMHub.Services.Studio.Npu;

/// <summary>
/// The Snapdragon NPU runtime: Qualcomm's QNN execution provider for ONNX Runtime (NuGet Qualcomm.ML.OnnxRuntime.QNN,
/// which ships QnnHtp.dll, QnnSystem.dll and the signed Hexagon v73 / v81 skeletons) registered as an ORT plugin EP.
/// Precompiled QNN context binaries run through it as EPContext models (see <see cref="QnnGraphSession"/>).
/// </summary>
public static class QnnRuntime
{
    private const string EpName = "QNNExecutionProvider";
    private static readonly object Gate = new();
    private static IReadOnlyList<OrtEpDevice>? _devices;

    /// <summary>Folder that holds the QNN DLLs (next to the app, where NuGet puts runtimes/win-arm64/native).</summary>
    public static string LibraryDir { get; set; } = AppContext.BaseDirectory;

    public static string HtpDll => Path.Combine(LibraryDir, "QnnHtp.dll");
    public static string SystemDll => Path.Combine(LibraryDir, "QnnSystem.dll");
    private static string EpDll => Path.Combine(LibraryDir, "onnxruntime_providers_qnn.dll");

    /// <summary>Windows on Arm with the QNN runtime next to the app (the NPU itself is checked when a model loads).</summary>
    public static bool IsAvailable =>
        OperatingSystem.IsWindows() && RuntimeInformation.OSArchitecture == Architecture.Arm64 && File.Exists(EpDll) && File.Exists(HtpDll);

    /// <summary>
    /// Hexagon generation of this PC's NPU, from its name: Snapdragon X Elite / X Plus (X1) are v73, X2 Elite is v81.
    /// The QNN builds of the models are chosen with it.
    /// </summary>
    public static int HexagonArch(string npuName) =>
        npuName.Contains("X2", StringComparison.OrdinalIgnoreCase) ? 81 : 73;

    /// <summary>Registers the plugin EP once and returns the NPU device(s) it exposes.</summary>
    public static IReadOnlyList<OrtEpDevice> NpuDevices()
    {
        lock (Gate)
        {
            if (_devices is not null)
                return _devices;
            var env = OrtEnv.Instance();
            env.RegisterExecutionProviderLibrary(EpName, EpDll);
            var all = env.GetEpDevices();
            var npu = all.Where(d => d.EpName == EpName && d.HardwareDevice.Type == OrtHardwareDeviceType.NPU).ToList();
            Log.Info($"QNN EP registered: {string.Join(", ", all.Select(d => $"{d.EpName}/{d.HardwareDevice.Type}/{d.HardwareDevice.Vendor}"))}");
            if (npu.Count == 0)
                npu = all.Where(d => d.EpName == EpName).ToList();
            if (npu.Count == 0)
                throw new InvalidOperationException("The QNN execution provider found no Hexagon NPU");
            _devices = npu;
            return _devices;
        }
    }

    /// <summary>Session options that put the whole graph on the HTP in burst mode, with no CPU fallback.</summary>
    public static SessionOptions CreateSessionOptions()
    {
        var options = new SessionOptions();
        options.AppendExecutionProvider(OrtEnv.Instance(), NpuDevices(), new Dictionary<string, string>
        {
            ["backend_path"] = HtpDll,
            ["htp_performance_mode"] = "burst",
        });
        options.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");
        return options;
    }
}
