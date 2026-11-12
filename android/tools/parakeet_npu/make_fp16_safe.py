"""Make Parakeet's encoder safe for float16 NPUs.

LayerNorm(x) == LayerNorm(s * x) for any s > 0 (up to epsilon), so each LayerNorm composite gets a
MUL by a power of two in front of it that brings its input down to about unit RMS. The squares that
LayerNorm sums then stay far below the float16 maximum (65504); without this they reach 1e9.

Google's own NPU builds of this model (litert-community/parakeet-tdt-0.6b-v3) skip this, and on NPUs
that compute in float16 the encoder then returns NaN (Snapdragon 8 Elite, 8 Elite Gen 5) or the same
output for any audio (Dimensity 9400, 9500). Compiled from the patched model, those chips transcribe.

Usage: make_fp16_safe.py SOURCE.tflite OUTPUT.tflite
SOURCE is parakeet_tdt_0.6b_v3_5s_f32_stateful.tflite from that repo; compile OUTPUT with compile_npu.py.
"""
import sys, math, collections
import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType

src, dst = sys.argv[1], sys.argv[2]

# 1. Calibrate: the largest row RMS each LayerNorm sees over a spread of inputs.
interp = Interpreter(model_path=src, experimental_preserve_all_tensors=True, num_threads=16,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
runner = interp.get_signature_runner("encode")
name = list(runner.get_input_details())[0]
shape = runner.get_input_details()[name]["shape"]
rng = np.random.default_rng(7)
ops = interp._get_ops_details()
comps = [o for o in ops if o["op_name"] == "STABLEHLO_COMPOSITE"]
t = np.arange(shape[-1], dtype=np.float32)
band = np.arange(shape[-2], dtype=np.float32)[:, None]
calibration = {
    "noise": rng.standard_normal(shape).astype(np.float32),
    "zeros": np.zeros(shape, np.float32),
    "loud noise": 3 * rng.standard_normal(shape).astype(np.float32),
    "quiet noise": 0.2 * rng.standard_normal(shape).astype(np.float32),
    # Speech-like: slow modulation across time, tilt across bands.
    "modulated": (np.sin(t / 7.0) * np.cos(band / 9.0) + 0.5 * rng.standard_normal(shape)).astype(np.float32).reshape(shape),
    "half silent": np.concatenate([rng.standard_normal(shape)[..., : shape[-1] // 2], np.full(shape, -1.5)[..., shape[-1] // 2:]], axis=-1).astype(np.float32),
}
worst = collections.defaultdict(float)   # op index -> max row rms
least = collections.defaultdict(lambda: 1e30)
reference = {}
for label, x in calibration.items():
    reference[label] = list(runner(**{name: x}).values())[0].copy()
    for o in comps:
        v = interp.get_tensor(o["inputs"][0]).astype(np.float64)
        rms = np.sqrt((v ** 2).mean(axis=-1))
        worst[o["index"]] = max(worst[o["index"]], float(rms.max()))
        least[o["index"]] = min(least[o["index"]], float(rms.min()))
del interp, runner

# 2. Edit the graph.
data = open(src, "rb").read()
model = schema.ModelT.InitFromPackedBuf(data, 0)
sig = next(s for s in model.signatureDefs if (s.signatureKey.decode() if isinstance(s.signatureKey, bytes) else s.signatureKey) == "encode")
graph = model.subgraphs[sig.subgraphIndex]
codes = model.operatorCodes
def code(i): return max(codes[i].builtinCode, codes[i].deprecatedBuiltinCode)
mul_code = next(i for i, c in enumerate(codes) if code(i) == schema.BuiltinOperator.MUL)
comp_code = schema.BuiltinOperator.STABLEHLO_COMPOSITE
external = [b for b in model.buffers if b.offset and b.offset > 1]
print(f"encode subgraph {sig.subgraphIndex}: {len(graph.operators)} ops, {len(graph.tensors)} tensors; "
      f"{len(model.buffers)} buffers, {len(external)} stored outside the flatbuffer", flush=True)
assert len(graph.operators) == len(ops), (len(graph.operators), len(ops))

new_ops, scales = [], []
for index, op in enumerate(graph.operators):
    if code(op.opcodeIndex) == comp_code:
        rms = worst[index]
        scale = 2.0 ** -max(0, math.ceil(math.log2(max(rms, 1e-6))))
        scales.append((index, rms, least[index], scale))
        if scale < 1.0:
            source = graph.tensors[op.inputs[0]]
            buffer = schema.BufferT(); buffer.data = np.frombuffer(np.float32(scale).tobytes(), dtype=np.uint8)
            model.buffers.append(buffer)
            const = schema.TensorT(); const.shape = np.array([1], np.int32); const.type = schema.TensorType.FLOAT32
            const.buffer = len(model.buffers) - 1; const.name = f"fp16_scale_{index}"
            scaled = schema.TensorT(); scaled.shape = source.shape; scaled.shapeSignature = source.shapeSignature
            scaled.type = source.type; scaled.buffer = 0; scaled.name = f"fp16_scaled_{index}"
            graph.tensors += [const, scaled]
            mul = schema.OperatorT(); mul.opcodeIndex = mul_code
            mul.inputs = np.array([op.inputs[0], len(graph.tensors) - 2], np.int32)
            mul.outputs = np.array([len(graph.tensors) - 1], np.int32)
            mul.builtinOptionsType = schema.BuiltinOptions.MulOptions; mul.builtinOptions = schema.MulOptionsT()
            new_ops.append(mul)
            inputs = np.array(op.inputs, np.int32); inputs[0] = len(graph.tensors) - 1; op.inputs = inputs
    new_ops.append(op)
graph.operators = new_ops
print(f"LayerNorms: {len(scales)}, scaled: {sum(1 for s in scales if s[3] < 1)}; scales used: {collections.Counter(s[3] for s in scales).most_common(8)}")
print("smallest scaled row RMS seen in calibration (should stay well above sqrt(eps)=0.003):",
      min(s[2] * s[3] for s in scales))

# 3. Write it back: the flatbuffer grew, so the tensor data stored after it moves by the same amount.
def pack():
    builder = flatbuffers.Builder(64 * 1024 * 1024)
    builder.Finish(model.Pack(builder), b"TFL3")
    return bytes(builder.Output())
if external:
    start = min(b.offset for b in external)
    end = max(b.offset + b.size for b in external)
    size = len(pack())
    new_start = (size + 63) // 64 * 64
    for b in external: b.offset += new_start - start
    large = [o for g in model.subgraphs for o in g.operators if getattr(o, "largeCustomOptionsOffset", 0)]
    for o in large: o.largeCustomOptionsOffset += new_start - start
    packed = pack()
    assert len(packed) == size, (len(packed), size)
    with open(dst, "wb") as f:
        f.write(packed); f.write(b"\0" * (new_start - size)); f.write(data[start:])
    print(f"flatbuffer {start} -> {size} bytes, data moved by {new_start - start}; tail {len(data) - start} bytes (data end {end}, file {len(data)})")
else:
    open(dst, "wb").write(pack())

# 4. Check: same answers as the original on the CPU.
interp = Interpreter(model_path=dst, num_threads=16)
runner = interp.get_signature_runner("encode")
for label, x in calibration.items():
    y = list(runner(**{name: x}).values())[0]
    ref = reference[label]
    print(f"  {label:12s} max |diff| {np.abs(y - ref).max():.2e}  (output range {ref.min():.3f}..{ref.max():.3f}, absSum {np.abs(ref).sum():.1f} vs {np.abs(y).sum():.1f})")
print("signatures:", list(interp.get_signature_list()))
print("PATCH_DONE")
