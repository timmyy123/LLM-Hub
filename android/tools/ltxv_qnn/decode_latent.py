"""Decode a latent dumped by stable-diffusion.cpp (SD_DUMP_LATENT_PATH) with the reference diffusers VAE.

Usage: python decode_latent.py latent.raw W H T out_dir --ckpt ltxv-2b-0.9.8-distilled.safetensors
The raw file is fp32 in ggml order [W, H, T, C] (W fastest), still per-channel normalized.
"""

import argparse
import os

import numpy as np
import torch
from diffusers import AutoencoderKLLTXVideo
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("latent")
ap.add_argument("w", type=int)
ap.add_argument("h", type=int)
ap.add_argument("t", type=int)
ap.add_argument("out")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

lat = np.fromfile(args.latent, dtype=np.float32).reshape(128, args.t, args.h, args.w)
vae = AutoencoderKLLTXVideo.from_single_file(args.ckpt, torch_dtype=torch.float32).cuda().eval()
z = torch.from_numpy(lat).unsqueeze(0)
mean = vae.latents_mean.view(1, -1, 1, 1, 1).cpu()
std = vae.latents_std.view(1, -1, 1, 1, 1).cpu()
z = z * std / vae.config.scaling_factor + mean
g = torch.Generator().manual_seed(args.seed)
z = z * (1 - 0.025) + torch.randn(z.shape, generator=g) * 0.025
with torch.no_grad():
    video = vae.decode(z.cuda(), temb=torch.tensor([0.05], device="cuda"), return_dict=False)[0]
video = ((video.clamp(-1, 1) + 1) * 127.5).round().byte().cpu()[0]
os.makedirs(args.out, exist_ok=True)
for f in range(video.shape[1]):
    Image.fromarray(video[:, f].permute(1, 2, 0).numpy()).save(os.path.join(args.out, f"frame_{f:03d}.png"))
print("decoded", video.shape)
