"""AIMET-format quantization overrides for weight-only INT8 (activations stay FP16).

Every linear weight [out, in] and conv weight [out, in, k...] gets symmetric per-output-channel
int8 encodings computed from the
weights themselves, so no calibration data is needed. Everything else (activations, norms, biases)
has no encoding and falls back to FP16 with `qairt-quantizer --enable_float_fallback`.

Usage: python make_overrides.py onnx/ltxv_chunk1.onnx overrides/ltxv_chunk1.json
"""

import json
import sys

import numpy as np
import onnx
from onnx import numpy_helper


def main():
    src, dst = sys.argv[1], sys.argv[2]
    model = onnx.load(src, load_external_data=True)
    gemm_weights = {n.input[1] for n in model.graph.node if n.op_type in ("Gemm", "Conv")}
    params = {}
    for init in model.graph.initializer:
        if init.name not in gemm_weights:
            continue
        w = numpy_helper.to_array(init).astype(np.float32)
        w = w.reshape(w.shape[0], -1)
        scales = np.maximum(np.abs(w).max(axis=1), 1e-8) / 127.0
        params[init.name] = [
            {"bitwidth": 8, "dtype": "int", "is_symmetric": "True",
             "min": float(-128.0 * s), "max": float(127.0 * s), "offset": -128, "scale": float(s)}
            for s in scales
        ]
    with open(dst, "w") as fh:
        json.dump({"activation_encodings": {}, "param_encodings": params}, fh)
    print(f"{dst}: {len(params)} int8 weights")


if __name__ == "__main__":
    main()
