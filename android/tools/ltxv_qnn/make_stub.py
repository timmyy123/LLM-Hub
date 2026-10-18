"""Write ltxv_qnn_stub.safetensors: the few transformer tensors stable-diffusion.cpp needs to detect
LTX-Video 0.9 and read its config when the transformer itself runs from QNN context binaries.

Usage: python make_stub.py --ckpt ltxv-2b-0.9.8-distilled.safetensors --out ltxv_qnn_stub.safetensors
"""

import argparse

from safetensors.torch import load_file, save_file

PREFIX = "model.diffusion_model."
KEEP = [
    "adaln_single.emb.timestep_embedder.linear_1.bias",
    "patchify_proj.weight",
    "proj_out.weight",
    "caption_projection.linear_1.weight",
    "caption_projection.linear_2.weight",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sd = load_file(args.ckpt)
    stub = {PREFIX + k: sd[PREFIX + k].half().contiguous() for k in KEEP}
    # One small tensor per block so the block count is detected.
    blocks = sorted({int(k[len(PREFIX):].split(".")[1]) for k in sd if k.startswith(PREFIX + "transformer_blocks.")})
    for b in blocks:
        name = f"{PREFIX}transformer_blocks.{b}.scale_shift_table"
        stub[name] = sd[name].half().contiguous()
    save_file(stub, args.out, metadata={"format": "pt", "note": "LTX-Video 0.9.8 2B distilled QNN stub"})
    print("wrote", args.out, len(stub), "tensors")


if __name__ == "__main__":
    main()
