"""Keep the VAE's RMS norms as plain element-wise ops for integer-only (no FP16) HTP builds.

qairt-converter fuses x / sqrt(mean(x^2) + eps) into RmsNorm and generates its scale and bias
tensors itself; those get no quantization encoding and stay FP16, which NPUs without FP16 (Snapdragon
8s Gen 4 and other v73 parts) reject. Scaling mean(x^2) by 1.0001 before eps is added (a 0.005% change
of the normalization) keeps the pattern from matching, so every op uses the calibrated activation
encodings; make_vae_encodings.py --complete gives the inserted Mul its input's encoding.

Usage: python unfuse_rms_norm.py onnx/ltxv_vae_512x320x25.onnx onnx/ltxv_vae_512x320x25_int.onnx
The output references the same external weight file, so it must be written next to the input.
"""

import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    src, dst = sys.argv[1], sys.argv[2]
    assert os.path.dirname(os.path.abspath(src)) == os.path.dirname(os.path.abspath(dst)), "write next to the source"
    model = onnx.load(src, load_external_data=False)
    graph = model.graph
    nodes = list(graph.node)
    out = []
    replaced = 0
    for node in nodes:
        out.append(node)
        if node.op_type != "ReduceMean" or "/norm" not in node.name:
            continue
        prefix = node.name.rsplit("/", 1)[0]
        scaled = f"{prefix}/rms_mean_scaled"
        const = helper.make_node("Constant", [], [f"{prefix}/rms_mean_scale"], name=f"{prefix}/rms_mean_scale_const",
                                 value=numpy_helper.from_array(np.array(1.0001, dtype=np.float32)))
        out += [const, helper.make_node("Mul", [node.output[0], const.output[0]], [scaled], name=f"{prefix}/rms_mean_scale_mul")]
        for consumer in nodes:
            for k, name in enumerate(consumer.input):
                if name == node.output[0] and consumer is not node:
                    consumer.input[k] = scaled
        replaced += 1
    del graph.node[:]
    graph.node.extend(out)
    onnx.save_model(model, dst)
    print(f"{dst}: scaled {replaced} RMS norm means", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
