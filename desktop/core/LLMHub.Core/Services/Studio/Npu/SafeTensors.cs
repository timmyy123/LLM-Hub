using System.Text.Json;

namespace LLMHub.Services.Studio.Npu;

/// <summary>One tensor's entry in a .safetensors header.</summary>
public sealed record SafeTensorEntry(string Name, string DType, long[] Shape, long Begin, long End);

/// <summary>Reads .safetensors files (F32, F16 and BF16 tensors, converted to float).</summary>
public static class SafeTensors
{
    /// <summary>Header entries and the byte offset where tensor data starts.</summary>
    public static (List<SafeTensorEntry> Entries, long DataStart) ReadHeader(Stream stream)
    {
        var lengthBytes = new byte[8];
        stream.ReadExactly(lengthBytes);
        var length = BitConverter.ToInt64(lengthBytes);
        if (length is <= 0 or > 100_000_000)
            throw new InvalidDataException("Not a safetensors file");
        var json = new byte[length];
        stream.ReadExactly(json);
        return (ParseHeader(json), 8 + length);
    }

    public static List<SafeTensorEntry> ParseHeader(byte[] json)
    {
        using var doc = JsonDocument.Parse(json);
        var list = new List<SafeTensorEntry>();
        foreach (var p in doc.RootElement.EnumerateObject())
        {
            if (p.Name == "__metadata__")
                continue;
            var offsets = p.Value.GetProperty("data_offsets");
            list.Add(new SafeTensorEntry(p.Name, p.Value.GetProperty("dtype").GetString()!,
                p.Value.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()).ToArray(), offsets[0].GetInt64(), offsets[1].GetInt64()));
        }
        return list;
    }

    /// <summary>All tensors whose names start with <paramref name="prefix"/>, keyed without the prefix, as float arrays.</summary>
    public static Dictionary<string, float[]> Load(string path, string prefix)
    {
        using var stream = File.OpenRead(path);
        var (entries, start) = ReadHeader(stream);
        var result = new Dictionary<string, float[]>();
        // Integer tensors (position_ids) are skipped.
        foreach (var e in entries.Where(e => e.Name.StartsWith(prefix, StringComparison.Ordinal) && e.DType is "F32" or "F16" or "BF16").OrderBy(e => e.Begin))
        {
            var bytes = new byte[e.End - e.Begin];
            stream.Position = start + e.Begin;
            stream.ReadExactly(bytes);
            result[e.Name[prefix.Length..]] = ToFloat(bytes, e.DType);
        }
        return result;
    }

    public static float[] ToFloat(byte[] bytes, string dtype)
    {
        switch (dtype)
        {
            case "F32":
            {
                var f = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
                return f;
            }
            case "F16":
            {
                var f = new float[bytes.Length / 2];
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, Half>(bytes), f);
                return f;
            }
            case "BF16":
            {
                var f = new float[bytes.Length / 2];
                for (var i = 0; i < f.Length; i++)
                    f[i] = BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(bytes, i * 2) << 16);
                return f;
            }
            default:
                throw new NotSupportedException($"safetensors dtype {dtype}");
        }
    }
}
