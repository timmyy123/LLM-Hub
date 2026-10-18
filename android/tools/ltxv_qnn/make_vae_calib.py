"""Calibration inputs for the W8A16 NPU VAE decoder: real LTX latents -> decoder inputs (z, temb).

Latents are per-channel normalized, either .npy [tokens, 128] from validate_torch.py or .raw fp32 in
ggml order [W, H, T, C] from SD_DUMP_LATENT_PATH. Each is un-normalized and noised like the pipeline.
Usage: python make_vae_calib.py --ckpt ckpt.safetensors --out calib --shape 16 10 4 lat1.npy lat2.raw ...
"""

import argparse
import os

import numpy as np
import torch
from diffusers import AutoencoderKLLTXVideo

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shape", type=int, nargs=3, required=True, help="latent W H T")
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("latents", nargs="+")
args = ap.parse_args()
w, h, t = args.shape
vae = AutoencoderKLLTXVideo.from_single_file(args.ckpt, torch_dtype=torch.float32)
mean = vae.latents_mean.view(1, -1, 1, 1, 1)
std = vae.latents_std.view(1, -1, 1, 1, 1)
os.makedirs(args.out, exist_ok=True)
np.array([0.05], dtype=np.float32).tofile(os.path.join(args.out, "temb.raw"))
lines = []
for i, path in enumerate(args.latents):
    if path.endswith(".npy"):
        lat = torch.from_numpy(np.load(path)).reshape(t, h, w, 128).permute(3, 0, 1, 2).unsqueeze(0)
    else:
        lat = torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(1, 128, t, h, w))
    z0 = lat * std + mean
    for s in range(args.seeds):
        g = torch.Generator().manual_seed(1000 * i + s)
        z = z0 * (1 - 0.025) + torch.randn(z0.shape, generator=g) * 0.025
        name = os.path.join(args.out, f"z_{i}_{s}.raw")
        z.numpy().astype(np.float32).tofile(name)
        lines.append(f"z:={os.path.abspath(name)} temb:={os.path.abspath(os.path.join(args.out, 'temb.raw'))}")
with open(os.path.join(args.out, "input_list.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(len(lines), "calibration samples")
