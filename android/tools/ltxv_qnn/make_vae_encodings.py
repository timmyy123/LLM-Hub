"""Full W8A16 encodings for the VAE decoder, calibrated with onnxruntime (multi-threaded).

qairt-quantizer calibrates on a single CPU thread (~25+ min per sample for this decoder). Instead,
ONNX Runtime's MinMax calibrator records the range of every activation in seconds per sample; the
ranges become 16-bit asymmetric activation encodings and conv/linear weights get symmetric
per-output-channel 8-bit encodings. The result is passed to qairt-converter as
--quantization_overrides, then `qairt-quantizer --enable_float_fallback` applies it without running
its own calibration.

Usage: python make_vae_encodings.py onnx/ltxv_vae_512x320x25.onnx calib_512/input_list.txt act.json
       python make_overrides.py onnx/ltxv_vae_512x320x25.onnx params.json
       python make_vae_encodings.py --merge act.json params.json out.json model.onnx [--complete]
"""

import json
import os
import sys
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, create_calibrator


class OneSample(CalibrationDataReader):
    """Yields a single sample, loaded from the raw files of one input-list line."""

    def __init__(self, line, shapes):
        entries = dict(part.split(":=") for part in line.split())
        self.item = {name: np.fromfile(path, dtype=np.float32).reshape(shapes[name]) for name, path in entries.items()}

    def get_next(self):
        item, self.item = self.item, None
        return item


def main():
    model_path, input_list, out_path = sys.argv[1:4]
    model = onnx.load(model_path, load_external_data=False)
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input}

    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(out_path))) as tmp:
        calibrator = create_calibrator(model_path, calibrate_method=CalibrationMethod.MinMax,
                                       augmented_model_path=os.path.join(tmp, "augmented.onnx"),
                                       use_external_data_format=True,
                                       extra_options={"symmetric": False})
        # One sample per collect_data call: each call merges its min/max into the running ranges, so
        # only one sample's intermediate tensors (several GB for a transformer chunk) are held at a time.
        lines = [l for l in open(input_list) if l.strip()]
        for n, line in enumerate(lines):
            calibrator.collect_data(OneSample(line, shapes))
            print(f"calibrated {n + 1}/{len(lines)}", flush=True)
        ranges = calibrator.calibrate_tensors_range

    activation = {}
    for name, rng in ranges.items():
        lo, hi = (float(v) for v in (rng.range_value if hasattr(rng, "range_value") else rng))
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        if hi - lo < 1e-8:
            hi = lo + 1e-8
        scale = (hi - lo) / 65535.0
        offset = int(round(lo / scale))
        activation[name] = [{"bitwidth": 16, "dtype": "int", "is_symmetric": "False",
                             "min": offset * scale, "max": (offset + 65535) * scale,
                             "offset": offset, "scale": scale}]

    # Weight encodings come from make_overrides.py in a separate process (onnxruntime still holds its
    # memory here); merge_encodings() combines the two files.
    with open(out_path, "w") as fh:
        json.dump({"activation_encodings": activation, "param_encodings": {}}, fh)
    print(f"{out_path}: {len(activation)} activation encodings")


PASSTHROUGH_OPS = {"Gather", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Slice", "Split", "Flatten",
                   "Expand", "Tile", "Identity", "Pad", "DepthToSpace", "SpaceToDepth"}


def tie_passthrough_encodings(model_path, act):
    """Data-movement ops must keep their input encoding on the HTP: copy it to their outputs, and give
    every Concat input and its output one shared range."""
    graph = onnx.load(model_path, load_external_data=False).graph

    def encode(lo, hi):
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        scale = max(hi - lo, 1e-8) / 65535.0
        offset = int(round(lo / scale))
        return [{"bitwidth": 16, "dtype": "int", "is_symmetric": "False",
                 "min": offset * scale, "max": (offset + 65535) * scale, "offset": offset, "scale": scale}]

    for _ in range(4):  # concat unions can change upstream ranges; iterate to a fixed point
        for node in graph.node:  # ONNX nodes are topologically sorted
            if node.op_type in PASSTHROUGH_OPS and node.input and node.input[0] in act:
                for out in node.output:
                    act[out] = act[node.input[0]]
            elif node.op_type == "Concat":
                names = [n for n in list(node.input) + list(node.output) if n in act]
                if names:
                    shared = encode(min(act[n][0]["min"] for n in names), max(act[n][0]["max"] for n in names))
                    for n in names:
                        act[n] = shared
    return symmetrize_activation_matmuls(graph, act)


def symmetrize_activation_matmuls(graph, act):
    """The HTP's 16-bit activation x activation MatMul (attention Q.K^T and P.V) needs symmetric inputs
    (uint16 offset -32768). Give those inputs, and every tensor tied to them through data-movement ops
    or Concat, one symmetric range."""
    params = {t.name for t in graph.initializer} | {o for n in graph.node if n.op_type == "Constant" for o in n.output}
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for node in graph.node:
        if node.op_type in PASSTHROUGH_OPS and node.input:
            for out in node.output:
                parent[find(out)] = find(node.input[0])
        elif node.op_type == "Concat":
            for name in list(node.input[1:]) + list(node.output):
                parent[find(name)] = find(node.input[0])
    needs = {name for node in graph.node if node.op_type == "MatMul" and not set(node.input) & params
             for name in node.input}
    roots = {find(n) for n in needs}
    groups = {}
    for name in act:
        if find(name) in roots:
            groups.setdefault(find(name), []).append(name)
    for names in groups.values():
        m = max(max(abs(act[n][0]["min"]), abs(act[n][0]["max"])) for n in names)
        scale = max(m, 1e-8) / 32767.0
        enc = [{"bitwidth": 16, "dtype": "int", "is_symmetric": "True",
                "min": -32768 * scale, "max": 32767 * scale, "offset": -32768, "scale": scale}]
        for n in names:
            act[n] = enc
    print(f"symmetric encodings for {sum(len(v) for v in groups.values())} tensors feeding activation MatMuls")
    return act


def complete_param_encodings(model_path, act, params):
    """Encodings for everything --enable_float_fallback would otherwise leave in FP16, for HTPs
    without FP16 (8s Gen 4 and other v73 parts): remaining float initializers (e.g. the VAE's
    scale_shift_table) get 16-bit ranges from their values, Cast outputs share their input's range,
    and the weight/bias the converter generates when it fuses x / sqrt(mean(x^2) + eps) into
    RmsNorm (no learned scale: ones and zeros) get exact encodings."""
    from onnx import numpy_helper
    model = onnx.load(model_path, load_external_data=True)
    graph = model.graph
    consts = {o for n in graph.node if n.op_type == "Constant" for o in n.output}
    for _ in range(4):
        for node in graph.node:
            if node.op_type == "Cast" and node.input[0] in act:
                act[node.output[0]] = act[node.input[0]]
            # Mul by a constant inserted by unfuse_rms_norm.py (x * 1.0001): same range as its input.
            if node.op_type == "Mul" and node.output[0] not in act and node.input[1] in consts and node.input[0] in act:
                act[node.output[0]] = act[node.input[0]]
    # Biases of Conv/Gemm/MatMul stay with the quantizer (32-bit, --bias_bitwidth).
    biases = {n.input[2] for n in graph.node if n.op_type in ("Conv", "Gemm", "ConvTranspose") and len(n.input) > 2}
    for init in graph.initializer:
        if init.name in params or init.name in biases or init.data_type != onnx.TensorProto.FLOAT:
            continue
        v = numpy_helper.to_array(init)
        lo, hi = min(float(v.min()), 0.0), max(float(v.max()), 0.0)
        scale = max(hi - lo, 1e-8) / 65535.0
        offset = int(round(lo / scale))
        params[init.name] = [{"bitwidth": 16, "dtype": "int", "is_symmetric": "False",
                              "min": offset * scale, "max": (offset + 65535) * scale, "offset": offset, "scale": scale}]
    for node in graph.node:
        if node.op_type == "Div" and "/norm" in node.name:
            prefix = node.name.rsplit("/", 1)[0]
            params[f"{prefix}/Div_weight"] = [{"bitwidth": 8, "dtype": "int", "is_symmetric": "True",
                                               "min": -128 / 127.0, "max": 1.0, "offset": -128, "scale": 1 / 127.0}]
            s = 1e-6
            params[f"{prefix}/Div_weight_bias"] = [{"bitwidth": 32, "dtype": "int", "is_symmetric": "True",
                                                    "min": -2147483648 * s, "max": 2147483647 * s, "offset": -2147483648, "scale": s}]
    return act, params


def merge_encodings(activation_path, params_path, out_path, model_path=None, complete=None):
    act = json.load(open(activation_path))["activation_encodings"]
    if model_path:
        act = tie_passthrough_encodings(model_path, act)
    params = json.load(open(params_path))["param_encodings"]
    if model_path and complete == "--complete":
        act, params = complete_param_encodings(model_path, act, params)
    with open(out_path, "w") as fh:
        json.dump({"activation_encodings": act, "param_encodings": params}, fh)
    print(f"{out_path}: {len(act)} activation + {len(params)} weight encodings")



if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge_encodings(*sys.argv[2:7])
    else:
        main()
    # The output is complete here; onnx/onnxruntime can crash while tearing down at interpreter exit.
    sys.stdout.flush()
    os._exit(0)
