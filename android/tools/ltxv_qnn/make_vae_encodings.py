"""Full W8A16 encodings for the VAE decoder, calibrated with onnxruntime (multi-threaded).

qairt-quantizer calibrates on a single CPU thread (~25+ min per sample for this decoder). Instead,
ONNX Runtime's MinMax calibrator records the range of every activation in seconds per sample; the
ranges become 16-bit asymmetric activation encodings and conv/linear weights get symmetric
per-output-channel 8-bit encodings. The result is passed to qairt-converter as
--quantization_overrides, then `qairt-quantizer --enable_float_fallback` applies it without running
its own calibration.

Usage: python make_vae_encodings.py onnx/ltxv_vae_512x320x25.onnx calib_512/input_list.txt act.json
       python make_overrides.py onnx/ltxv_vae_512x320x25.onnx params.json
       python make_vae_encodings.py --merge act.json params.json out.json model.onnx
"""

import json
import os
import sys
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, create_calibrator


class RawReader(CalibrationDataReader):
    def __init__(self, input_list, shapes):
        self.items = []
        for line in open(input_list):
            entries = dict(part.split(":=") for part in line.split())
            self.items.append({name: np.fromfile(path, dtype=np.float32).reshape(shapes[name]) for name, path in entries.items()})
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.items):
            return None
        self.pos += 1
        return self.items[self.pos - 1]


def main():
    model_path, input_list, out_path = sys.argv[1:4]
    model = onnx.load(model_path, load_external_data=False)
    shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input}

    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(out_path))) as tmp:
        calibrator = create_calibrator(model_path, calibrate_method=CalibrationMethod.MinMax,
                                       augmented_model_path=os.path.join(tmp, "augmented.onnx"),
                                       use_external_data_format=True,
                                       extra_options={"symmetric": False})
        calibrator.collect_data(RawReader(input_list, shapes))
        ranges = calibrator.compute_data()

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
    return act


def merge_encodings(activation_path, params_path, out_path, model_path=None):
    act = json.load(open(activation_path))["activation_encodings"]
    if model_path:
        act = tie_passthrough_encodings(model_path, act)
    params = json.load(open(params_path))["param_encodings"]
    with open(out_path, "w") as fh:
        json.dump({"activation_encodings": act, "param_encodings": params}, fh)
    print(f"{out_path}: {len(act)} activation + {len(params)} weight encodings")



if __name__ == "__main__":
    if sys.argv[1] == "--merge":
        merge_encodings(*sys.argv[2:6])
    else:
        main()
