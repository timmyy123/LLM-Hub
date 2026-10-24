using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;

namespace LLMHub.Services.Studio.Npu;

/// <summary>QNN tensor element types the Studio models use (values from QnnTypes.h).</summary>
public enum QnnDataType : uint
{
    Int8 = 0x0008, Int16 = 0x0016, Int32 = 0x0032, Int64 = 0x0064,
    UInt8 = 0x0108, UInt16 = 0x0116, UInt32 = 0x0132,
    Float16 = 0x0216, Float32 = 0x0232,
    SFixed8 = 0x0308, SFixed16 = 0x0316,
    UFixed8 = 0x0408, UFixed16 = 0x0416,
    Bool8 = 0x0508,
}

/// <summary>One graph input or output: name, shape, element type and (for fixed point) real = scale * (q + offset).</summary>
public sealed record QnnTensorSpec(string Name, int[] Dims, QnnDataType Type, float Scale, int Offset)
{
    public long Count => Dims.Aggregate(1L, (a, d) => a * d);
    public bool IsQuantized => Type is QnnDataType.UFixed16 or QnnDataType.UFixed8 or QnnDataType.SFixed16 or QnnDataType.SFixed8;
    public override string ToString() => $"{Name} {string.Join('x', Dims)} {Type}" + (IsQuantized ? $" scale={Scale:G6} offset={Offset}" : "");
}

public sealed record QnnGraphSpec(string Name, IReadOnlyList<QnnTensorSpec> Inputs, IReadOnlyList<QnnTensorSpec> Outputs);

/// <summary>What a QNN context binary contains: the SDK that built it, the Hexagon arch and its graphs.</summary>
public sealed record QnnContextSpec(string BuildId, int DspArch, IReadOnlyList<QnnGraphSpec> Graphs);

/// <summary>
/// Reads a context binary's graphs and tensor metadata through QnnSystem.dll (QnnSystemContext_getBinaryInfo), so the
/// app can wrap any context binary for ONNX Runtime and quantize its inputs without hard-coding names or encodings.
/// Struct offsets follow QnnSystemContext.h / QnnTypes.h for 64-bit targets.
/// </summary>
public static unsafe class QnnContextInfo
{
    private const int TensorSize = 144;      // sizeof(Qnn_Tensor_t): version + Qnn_TensorV2_t
    private const int GraphInfoSize = 88;    // sizeof(QnnSystemContext_GraphInfo_t): version + GraphInfoV3

    public static QnnContextSpec Read(string binaryPath, string qnnSystemDll)
    {
        var lib = NativeLibrary.Load(qnnSystemDll);
        // QnnSystemInterface_getProviders(const QnnSystemInterface_t*** list, uint32_t* count)
        var getProviders = (delegate* unmanaged<IntPtr*, uint*, ulong>)NativeLibrary.GetExport(lib, "QnnSystemInterface_getProviders");
        IntPtr list;
        uint count;
        Check(getProviders(&list, &count), "QnnSystemInterface_getProviders");
        if (count == 0)
            throw new InvalidOperationException("QnnSystem has no providers");
        var iface = *(byte**)list; // first provider
        // QnnSystemInterface_t: backendId, providerName, systemApiVersion, then the function table at offset 32.
        var create = (delegate* unmanaged<IntPtr*, ulong>)*(IntPtr*)(iface + 32);
        var getBinaryInfo = (delegate* unmanaged<IntPtr, void*, ulong, byte**, ulong*, ulong>)*(IntPtr*)(iface + 40);
        var free = (delegate* unmanaged<IntPtr, ulong>)*(IntPtr*)(iface + 56);

        using var file = MemoryMappedFile.CreateFromFile(binaryPath, FileMode.Open, null, 0, MemoryMappedFileAccess.CopyOnWrite);
        using var view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.CopyOnWrite);
        byte* data = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref data);
        IntPtr ctx;
        Check(create(&ctx), "QnnSystemContext_create");
        try
        {
            byte* info;
            ulong infoSize;
            Check(getBinaryInfo(ctx, data + view.PointerOffset, (ulong)new FileInfo(binaryPath).Length, &info, &infoSize), "QnnSystemContext_getBinaryInfo");
            return Parse(info);
        }
        finally
        {
            free(ctx);
            view.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    private static QnnContextSpec Parse(byte* info)
    {
        var version = *(uint*)info;
        var u = info + 8;
        var buildId = Str(*(IntPtr*)(u + 8));
        uint numGraphs;
        byte* graphs;
        var arch = 0;
        switch (version)
        {
            case 1 or 2:
                numGraphs = *(uint*)(u + 112);
                graphs = *(byte**)(u + 120);
                break;
            case 3:
                numGraphs = *(uint*)(u + 88);
                graphs = *(byte**)(u + 96);
                // contextMetadata (HTP): { version, { dspArch, ... } }
                if (*(uint*)(u + 112) >= 8 && *(IntPtr*)(u + 120) != IntPtr.Zero)
                    arch = (int)*(uint*)(*(byte**)(u + 120) + 4);
                break;
            default:
                throw new NotSupportedException($"Context binary info version {version}");
        }
        var list = new List<QnnGraphSpec>();
        for (var g = 0; g < numGraphs; g++)
        {
            var graph = graphs + g * GraphInfoSize + 8; // skip the version; V1/V2/V3 share the leading fields
            list.Add(new QnnGraphSpec(Str(*(IntPtr*)graph),
                Tensors(*(byte**)(graph + 16), *(uint*)(graph + 8)),
                Tensors(*(byte**)(graph + 32), *(uint*)(graph + 24))));
        }
        return new QnnContextSpec(buildId, arch, list);
    }

    private static List<QnnTensorSpec> Tensors(byte* tensors, uint count)
    {
        var list = new List<QnnTensorSpec>();
        for (var i = 0; i < count; i++)
        {
            // Qnn_Tensor_t { version; union { V1, V2 } } with V1 and V2 sharing the fields read here.
            var t = tensors + i * TensorSize + 8;
            var name = Str(*(IntPtr*)(t + 8));
            var type = (QnnDataType)(*(uint*)(t + 24));
            var encodingDefined = *(uint*)(t + 32) == 1;
            var encoding = *(uint*)(t + 36);
            float scale = 1;
            var offset = 0;
            if (encodingDefined && encoding == 0) // QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
            {
                scale = *(float*)(t + 40);
                offset = *(int*)(t + 44);
            }
            var rank = *(uint*)(t + 72);
            var dimsPtr = *(uint**)(t + 80);
            var dims = new int[rank];
            for (var d = 0; d < rank; d++)
                dims[d] = (int)dimsPtr[d];
            list.Add(new QnnTensorSpec(name, dims, type, scale, offset));
        }
        return list;
    }

    private static string Str(IntPtr p) => p == IntPtr.Zero ? "" : Marshal.PtrToStringUTF8(p) ?? "";

    private static void Check(ulong error, string what)
    {
        if (error != 0)
            throw new InvalidOperationException($"{what} failed with QNN error {error & 0xFFFF}");
    }
}
