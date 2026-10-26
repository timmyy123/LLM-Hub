namespace LLMHub.Services.MediaSearch;

/// <summary>One embedded item: a photo, or an audio moment covering StartMs..EndMs of a file.</summary>
public sealed record MediaVector(string Id, int StartMs, int EndMs, float[] Vector);

/// <summary>
/// Vectors for one search feature, in the Android app's on-disk format ("LHM1": count, then id, start, end and the
/// floats of each record, big-endian). Also keeps each file's modified time, so changed files are embedded again.
/// </summary>
public sealed class MediaIndex
{
    private const int Magic = 0x4C484D31;
    private readonly string _file;
    private readonly Dictionary<string, MediaVector> _records = [];
    private readonly Dictionary<string, long> _stamps = [];
    private bool _dirty;

    public MediaIndex(string file) => _file = file;

    public IReadOnlyCollection<MediaVector> All => _records.Values;
    public int FileCount => _stamps.Count;
    /// <summary>The model the vectors came from; vectors from another model are not comparable.</summary>
    public string ModelId { get; private set; } = "";

    public void Load()
    {
        _records.Clear();
        _stamps.Clear();
        try
        {
            if (File.Exists(_file))
            {
                using var input = new BinaryReader(File.OpenRead(_file));
                if (BigEndian(input.ReadInt32()) == Magic)
                {
                    var count = BigEndian(input.ReadInt32());
                    for (var i = 0; i < count; i++)
                    {
                        var id = ReadUtf(input);
                        var start = BigEndian(input.ReadInt32());
                        var end = BigEndian(input.ReadInt32());
                        var vector = new float[BigEndian(input.ReadInt32())];
                        for (var j = 0; j < vector.Length; j++)
                            vector[j] = BitConverter.Int32BitsToSingle(BigEndian(input.ReadInt32()));
                        _records[Key(id, start)] = new MediaVector(id, start, end, vector);
                    }
                }
            }
            var stamps = _file + ".files";
            if (File.Exists(stamps))
            {
                var lines = File.ReadAllLines(stamps);
                ModelId = lines.FirstOrDefault() ?? "";
                foreach (var line in lines.Skip(1))
                {
                    var tab = line.IndexOf('\t');
                    if (tab > 0 && long.TryParse(line[..tab], out var stamp))
                        _stamps[line[(tab + 1)..]] = stamp;
                }
            }
        }
        catch (Exception e) when (e is IOException or EndOfStreamException or InvalidDataException)
        {
            Log.Error($"Media index {_file} unreadable, starting fresh", e);
            _records.Clear();
            _stamps.Clear();
        }
    }

    public bool IsCurrent(string path, long stamp) => _stamps.TryGetValue(path, out var known) && known == stamp;

    /// <summary>Replaces a file's vectors (one for a photo, one per moment for audio).</summary>
    public void Put(string path, long stamp, IEnumerable<MediaVector> vectors)
    {
        Remove(path);
        foreach (var v in vectors)
            _records[Key(v.Id, v.StartMs)] = v;
        _stamps[path] = stamp;
        _dirty = true;
    }

    public void Remove(string path)
    {
        foreach (var key in _records.Where(r => r.Value.Id == path).Select(r => r.Key).ToList())
            _records.Remove(key);
        _dirty |= _stamps.Remove(path);
    }

    public IEnumerable<string> Files => _stamps.Keys;

    public void Reset(string modelId)
    {
        _records.Clear();
        _stamps.Clear();
        ModelId = modelId;
        _dirty = true;
    }

    public void SaveIfDirty()
    {
        if (!_dirty)
            return;
        Directory.CreateDirectory(Path.GetDirectoryName(_file)!);
        var tmp = _file + ".tmp";
        using (var output = new BinaryWriter(File.Create(tmp)))
        {
            output.Write(BigEndian(Magic));
            output.Write(BigEndian(_records.Count));
            foreach (var r in _records.Values)
            {
                WriteUtf(output, r.Id);
                output.Write(BigEndian(r.StartMs));
                output.Write(BigEndian(r.EndMs));
                output.Write(BigEndian(r.Vector.Length));
                foreach (var f in r.Vector)
                    output.Write(BigEndian(BitConverter.SingleToInt32Bits(f)));
            }
        }
        File.Move(tmp, _file, overwrite: true);
        File.WriteAllLines(_file + ".files", new[] { ModelId }.Concat(_stamps.Select(s => $"{s.Value}\t{s.Key}")));
        _dirty = false;
    }

    /// <summary>Vectors are L2-normalized, so the dot product is the cosine similarity.</summary>
    public static float Dot(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            return -1f;
        return System.Numerics.Tensors.TensorPrimitives.Dot(a, b);
    }

    private static string Key(string id, int start) => $"{id}#{start}";
    private static int BigEndian(int value) => BitConverter.IsLittleEndian ? System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(value) : value;

    // Java's DataOutputStream.writeUTF: a 2-byte big-endian length, then (modified) UTF-8.
    private static string ReadUtf(BinaryReader input)
    {
        var length = (input.ReadByte() << 8) | input.ReadByte();
        return System.Text.Encoding.UTF8.GetString(input.ReadBytes(length));
    }

    private static void WriteUtf(BinaryWriter output, string text)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(text);
        output.Write((byte)(bytes.Length >> 8));
        output.Write((byte)bytes.Length);
        output.Write(bytes);
    }
}
