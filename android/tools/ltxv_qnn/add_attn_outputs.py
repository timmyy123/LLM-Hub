"""Expose the input of every attention output projection (attn*/to_out.0) as an extra graph output.

With 16-bit integer activations, QAIRT 2.50's HTP kernels crash when the attention's
MatMul -> Transpose -> Reshape -> to_out Gemm chain stays fused inside one graph (on the device and in
the x86 emulator). Making the Reshape result a graph output materializes it and avoids the crash; the
runtime ignores the extra outputs.

Usage: python add_attn_outputs.py onnx/ltxv_chunk1.onnx onnx/ltxv_chunk1_w8.onnx
The output references the same external weight file, so it must be written next to the input.
"""

import os
import sys

import onnx
from onnx import TensorProto, helper


def main():
    src, dst = sys.argv[1], sys.argv[2]
    assert os.path.dirname(os.path.abspath(src)) == os.path.dirname(os.path.abspath(dst)), "write next to the source"
    model = onnx.load(src, load_external_data=False)
    graph = model.graph
    existing = {o.name for o in graph.output}
    tokens = graph.input[0].type.tensor_type.shape.dim[0].dim_value
    added = []
    for node in graph.node:
        if node.op_type == "Gemm" and "/to_out.0/" in node.name and node.input[0] not in existing:
            dim = next(t.dims[1] if node.attribute and any(a.name == "transB" and a.i for a in node.attribute) else t.dims[0]
                       for t in graph.initializer if t.name == node.input[1])
            graph.output.append(helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, [tokens, dim]))
            added.append(node.input[0])
    onnx.save_model(model, dst)
    print(f"{dst}: {len(added)} extra attention outputs", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
