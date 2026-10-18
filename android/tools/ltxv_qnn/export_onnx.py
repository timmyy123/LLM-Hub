"""Export LTX-Video transformer chunks to static-shape ONNX graphs for QAIRT.

Usage: python export_onnx.py --ckpt ltxv-2b-0.9.8-distilled.safetensors --out onnx/ [--tokens 640]

Each chunk is exported in fp32 with external weights (QAIRT converts to fp16). Shapes are fixed:
tokens = latent W*H*T (640 = 16x10x4 for 512x320x25), 128 T5 tokens.
"""

import argparse
import json
import os

import onnx
import torch
from safetensors.torch import load_file

from ltxv_model import Chunk, config_from_state_dict

CHUNKS = [(0, 7), (7, 14), (14, 21), (21, 28)]
TEXT_TOKENS = 128


def chunk_io(index: int, count: int, tokens: int, cfg: dict):
    dim = cfg["dim"]
    shared = {
        "mask": (1, TEXT_TOKENS),
        "cos": (tokens, dim // 2),
        "sin": (tokens, dim // 2),
    }
    if index == 0:
        inputs = {"latent": (tokens, cfg["in_channels"]), "t_freq": (tokens, 256), "caption": (TEXT_TOKENS, cfg["caption_channels"])}
        inputs.update(shared)
        outputs = ["x", "tmod", "emb", "context"]
    elif index == count - 1:
        inputs = {"x": (tokens, dim), "tmod": (tokens, 6 * dim), "emb": (tokens, dim), "context": (TEXT_TOKENS, dim)}
        inputs.update(shared)
        outputs = ["out"]
    else:
        inputs = {"x": (tokens, dim), "tmod": (tokens, 6 * dim), "context": (TEXT_TOKENS, dim)}
        inputs.update(shared)
        outputs = ["x_out"]
    return inputs, outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=640)
    ap.add_argument("--only", type=int, default=-1, help="export a single chunk index")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sd = load_file(args.ckpt)
    cfg = config_from_state_dict(sd)
    manifest = {"config": cfg, "tokens": args.tokens, "text_tokens": TEXT_TOKENS, "chunks": []}

    for i, (start, end) in enumerate(CHUNKS):
        inputs, outputs = chunk_io(i, len(CHUNKS), args.tokens, cfg)
        manifest["chunks"].append({"name": f"ltxv_chunk{i}", "blocks": [start, end],
                                   "inputs": {k: list(v) for k, v in inputs.items()}, "outputs": outputs})
        if args.only >= 0 and i != args.only:
            continue
        model = Chunk(cfg, start, end, with_prelude=(i == 0), with_head=(i == len(CHUNKS) - 1))
        model.load_from(sd)
        model.eval()
        dummy = tuple(torch.randn(*shape) * 0.1 for shape in inputs.values())
        path = os.path.join(args.out, f"ltxv_chunk{i}.onnx")
        print(f"exporting chunk {i} blocks {start}-{end} -> {path}")
        with torch.no_grad():
            torch.onnx.export(model, dummy, path, input_names=list(inputs.keys()), output_names=outputs,
                              opset_version=17, do_constant_folding=True, dynamo=False)
        # Re-save with external weights so every chunk stays under the 2 GB protobuf limit.
        m = onnx.load(path, load_external_data=True)
        for f in os.listdir(args.out):
            if f.startswith(f"ltxv_chunk{i}.onnx") and f != f"ltxv_chunk{i}.onnx":
                os.remove(os.path.join(args.out, f))
        onnx.save_model(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=f"ltxv_chunk{i}.onnx.data", size_threshold=1024)
        del model, m

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("done")


if __name__ == "__main__":
    main()
