using System.Text;
using Microsoft.ML.OnnxRuntime;

namespace LLMHub.Services.Studio.Npu;

/// <summary>
/// One QNN context binary (a precompiled Hexagon graph, as published for Android phones) running on the NPU through
/// ONNX Runtime's QNN EP. The binary is wrapped in a one-node EPContext ONNX model written next to it, with the graph's
/// own input and output names and element types; inputs are quantized and outputs dequantized here.
/// </summary>
public sealed class QnnGraphSession : IDisposable
{
    private readonly InferenceSession _session;
    private readonly OrtValue[] _inputValues;
    private readonly OrtValue[] _outputValues;
    private readonly Array[] _inputs;
    private readonly Array[] _outputs;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly RunOptions _run = new();

    public QnnGraphSession(string binaryPath)
    {
        var spec = QnnContextInfo.Read(binaryPath, QnnRuntime.SystemDll);
        if (spec.Graphs.Count != 1)
            throw new NotSupportedException($"{Path.GetFileName(binaryPath)} has {spec.Graphs.Count} graphs");
        Graph = spec.Graphs[0];
        Context = spec;
        var wrapper = Path.ChangeExtension(binaryPath, ".epctx.onnx");
        File.WriteAllBytes(wrapper, EpContextModel(Graph, Path.GetFileName(binaryPath), spec.BuildId));
        using var options = QnnRuntime.CreateSessionOptions();
        _session = new InferenceSession(wrapper, options);

        _inputNames = Graph.Inputs.Select(t => t.Name).ToArray();
        _outputNames = Graph.Outputs.Select(t => t.Name).ToArray();
        _inputs = Graph.Inputs.Select(Allocate).ToArray();
        _outputs = Graph.Outputs.Select(Allocate).ToArray();
        _inputValues = Graph.Inputs.Select((t, i) => Wrap(t, _inputs[i])).ToArray();
        _outputValues = Graph.Outputs.Select((t, i) => Wrap(t, _outputs[i])).ToArray();
    }

    public QnnGraphSpec Graph { get; }
    public QnnContextSpec Context { get; }

    public int InputIndex(string name) => Array.IndexOf(_inputNames, name) is var i and >= 0 ? i : throw new KeyNotFoundException(name);
    public int OutputIndex(string name) => Array.IndexOf(_outputNames, name) is var i and >= 0 ? i : throw new KeyNotFoundException(name);

    /// <summary>Converts real values into input <paramref name="index"/>'s element type (quantizing fixed-point inputs).</summary>
    public void SetInput(int index, ReadOnlySpan<float> values)
    {
        var spec = Graph.Inputs[index];
        switch (_inputs[index])
        {
            case ushort[] q:
            {
                var inv = 1f / spec.Scale;
                for (var i = 0; i < q.Length; i++)
                    q[i] = (ushort)Math.Clamp(MathF.Round(values[i] * inv) - spec.Offset, 0, 65535);
                break;
            }
            case byte[] q:
            {
                var inv = 1f / spec.Scale;
                for (var i = 0; i < q.Length; i++)
                    q[i] = (byte)Math.Clamp(MathF.Round(values[i] * inv) - spec.Offset, 0, 255);
                break;
            }
            case float[] f:
                values[..f.Length].CopyTo(f);
                break;
            case Float16[] h:
                for (var i = 0; i < h.Length; i++)
                    h[i] = (Float16)values[i];
                break;
            case int[] n:
                for (var i = 0; i < n.Length; i++)
                    n[i] = (int)MathF.Round(values[i]);
                break;
            default:
                throw new NotSupportedException($"Input {spec}");
        }
    }

    public void Run() => _session.Run(_run, _inputNames, _inputValues, _outputNames, _outputValues);

    /// <summary>Output <paramref name="index"/> as real values (dequantized).</summary>
    public void GetOutput(int index, Span<float> values)
    {
        var spec = Graph.Outputs[index];
        switch (_outputs[index])
        {
            case ushort[] q:
                for (var i = 0; i < q.Length; i++)
                    values[i] = spec.Scale * (q[i] + spec.Offset);
                break;
            case byte[] q:
                for (var i = 0; i < q.Length; i++)
                    values[i] = spec.Scale * (q[i] + spec.Offset);
                break;
            case float[] f:
                f.CopyTo(values);
                break;
            case Float16[] h:
                for (var i = 0; i < h.Length; i++)
                    values[i] = (float)h[i];
                break;
            default:
                throw new NotSupportedException($"Output {spec}");
        }
    }

    public void Dispose()
    {
        foreach (var value in _inputValues.Concat(_outputValues))
            value.Dispose();
        _session.Dispose();
        _run.Dispose();
    }

    private static Array Allocate(QnnTensorSpec t) => t.Type switch
    {
        QnnDataType.UFixed16 or QnnDataType.UInt16 => new ushort[t.Count],
        QnnDataType.UFixed8 or QnnDataType.UInt8 => new byte[t.Count],
        QnnDataType.Float32 => new float[t.Count],
        QnnDataType.Float16 => new Float16[t.Count],
        QnnDataType.Int32 => new int[t.Count],
        _ => throw new NotSupportedException($"QNN tensor {t}"),
    };

    private static OrtValue Wrap(QnnTensorSpec t, Array data)
    {
        var shape = t.Dims.Select(d => (long)d).ToArray();
        return data switch
        {
            ushort[] a => OrtValue.CreateTensorValueFromMemory(a, shape),
            byte[] a => OrtValue.CreateTensorValueFromMemory(a, shape),
            float[] a => OrtValue.CreateTensorValueFromMemory(a, shape),
            Float16[] a => OrtValue.CreateTensorValueFromMemory(a, shape),
            int[] a => OrtValue.CreateTensorValueFromMemory(a, shape),
            _ => throw new NotSupportedException(),
        };
    }

    // ---------------------------------------------------------------- EPContext wrapper (ONNX protobuf, written by hand)

    private static int OnnxType(QnnDataType type) => type switch
    {
        QnnDataType.Float32 => 1,
        QnnDataType.UFixed8 or QnnDataType.UInt8 => 2,
        QnnDataType.SFixed8 or QnnDataType.Int8 => 3,
        QnnDataType.UFixed16 or QnnDataType.UInt16 => 4,
        QnnDataType.SFixed16 or QnnDataType.Int16 => 5,
        QnnDataType.Int32 => 6,
        QnnDataType.Int64 => 7,
        QnnDataType.Bool8 => 9,
        QnnDataType.Float16 => 10,
        QnnDataType.UInt32 => 12,
        _ => throw new NotSupportedException(type.ToString()),
    };

    /// <summary>
    /// ModelProto { graph { node: EPContext(embed_mode=0, ep_cache_context=&lt;bin&gt;, main_context=1, source=QNN), inputs, outputs } }.
    /// The node is named after the QNN graph, which is how the QNN EP finds the graph inside the binary.
    /// </summary>
    public static byte[] EpContextModel(QnnGraphSpec graph, string binaryFileName, string sdkVersion)
    {
        static void Attr(Proto node, string name, long? i = null, string? s = null) => node.Message(5, a =>
        {
            a.String(1, name);
            if (i is { } value)
            {
                a.Varint(20, 2);
                a.Varint(3, (ulong)value);
            }
            else
            {
                a.Varint(20, 3);
                a.String(4, s!);
            }
        });

        static void ValueInfo(Proto g, int field, QnnTensorSpec t) => g.Message(field, v =>
        {
            v.String(1, t.Name);
            v.Message(2, type => type.Message(1, tensor =>
            {
                tensor.Varint(1, (ulong)OnnxType(t.Type));
                tensor.Message(2, shape =>
                {
                    foreach (var d in t.Dims)
                        shape.Message(1, dim => dim.Varint(1, (ulong)d));
                });
            }));
        });

        var model = new Proto();
        model.Varint(1, 10); // IR version
        model.String(2, "LLMHub");
        model.Message(7, g =>
        {
            g.Message(1, node =>
            {
                foreach (var t in graph.Inputs)
                    node.String(1, t.Name);
                foreach (var t in graph.Outputs)
                    node.String(2, t.Name);
                node.String(3, graph.Name);
                node.String(4, "EPContext");
                node.String(7, "com.microsoft");
                Attr(node, "embed_mode", 0);
                Attr(node, "ep_cache_context", s: binaryFileName);
                Attr(node, "main_context", 1);
                Attr(node, "source", s: "QNN");
                Attr(node, "ep_sdk_version", s: sdkVersion);
            });
            g.String(2, "llmhub_" + graph.Name);
            foreach (var t in graph.Inputs)
                ValueInfo(g, 11, t);
            foreach (var t in graph.Outputs)
                ValueInfo(g, 12, t);
        });
        model.Message(8, op =>
        {
            op.String(1, "");
            op.Varint(2, 17);
        });
        model.Message(8, op =>
        {
            op.String(1, "com.microsoft");
            op.Varint(2, 1);
        });
        return model.ToArray();
    }

    /// <summary>Minimal protobuf writer: varints and length-delimited fields.</summary>
    private sealed class Proto
    {
        private readonly MemoryStream _out = new();

        public void Varint(int field, ulong value)
        {
            Raw(((ulong)field << 3) | 0);
            Raw(value);
        }

        public void String(int field, string value) => Bytes(field, Encoding.UTF8.GetBytes(value));

        public void Message(int field, Action<Proto> build)
        {
            var inner = new Proto();
            build(inner);
            Bytes(field, inner.ToArray());
        }

        private void Bytes(int field, byte[] value)
        {
            Raw(((ulong)field << 3) | 2);
            Raw((ulong)value.Length);
            _out.Write(value);
        }

        private void Raw(ulong value)
        {
            while (value >= 0x80)
            {
                _out.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }
            _out.WriteByte((byte)value);
        }

        public byte[] ToArray() => _out.ToArray();
    }
}
