"""End-to-end check of the export module on a CUDA GPU before any QNN conversion.

T5 (transformers) -> chunked LTXV transformer (ltxv_model.py) -> official LTX VAE (diffusers).
Writes PNG frames so the result can be compared with the stable-diffusion.cpp output.
"""

import argparse
import gc
import os

import numpy as np
import torch
from safetensors.torch import load_file

from ltxv_model import Chunk, config_from_state_dict, rope_tables, timestep_freq, MASK_NEG

SIGMAS = [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219, 0.0]
CHUNKS = [(0, 7), (7, 14), (14, 21), (21, 28)]


def encode_prompt(repo_dir: str, prompt: str, max_len: int, device: str):
    from transformers import T5EncoderModel, T5TokenizerFast

    tok = T5TokenizerFast.from_pretrained(os.path.join(repo_dir, "tokenizer"))
    enc = T5EncoderModel.from_pretrained(os.path.join(repo_dir, "text_encoder"), torch_dtype=torch.bfloat16).to(device)
    ids = tok(prompt, padding="max_length", max_length=max_len, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = enc(ids.input_ids.to(device), attention_mask=ids.attention_mask.to(device)).last_hidden_state
    mask = ids.attention_mask[0].float()
    del enc
    gc.collect()
    torch.cuda.empty_cache()
    return hidden[0].float().cpu(), mask


def build_chunks(sd: dict, cfg: dict, dtype, device):
    chunks = []
    for i, (s, e) in enumerate(CHUNKS):
        c = Chunk(cfg, s, e, with_prelude=(i == 0), with_head=(i == len(CHUNKS) - 1))
        c.load_from(sd)
        chunks.append(c.to(device=device, dtype=dtype).eval())
    return chunks


@torch.no_grad()
def run_transformer(chunks, latent, t_freq, caption, mask, cos, sin):
    x, tmod, emb, context = chunks[0](latent, t_freq, caption, mask, cos, sin)
    for c in chunks[1:-1]:
        x = c(x, tmod, context, mask, cos, sin)
    return chunks[-1](x, tmod, emb, context, mask, cos, sin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True, help="dir with tokenizer/ and text_encoder/ of Lightricks/LTX-Video")
    ap.add_argument("--prompt", default="A golden retriever puppy running across a sunny green meadow, slow motion, cinematic lighting")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--out", default="val_out")
    ap.add_argument("--save-io", action="store_true", help="save one step of chunk inputs for QNN accuracy checks")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    os.makedirs(args.out, exist_ok=True)

    caption, attn = encode_prompt(args.repo, args.prompt, 128, device)
    mask = ((1.0 - attn) * MASK_NEG).reshape(1, -1)

    sd = load_file(args.ckpt)
    cfg = config_from_state_dict(sd)
    print("config", cfg)
    chunks = build_chunks(sd, cfg, dtype, device)
    vae_sd = {k[len("vae."):]: v for k, v in sd.items() if k.startswith("vae.")}
    del sd
    gc.collect()

    lw, lh, lt = args.width // 32, args.height // 32, (args.frames - 1) // 8 + 1
    tokens = lw * lh * lt
    cos, sin = rope_tables(lw, lh, lt, cfg["dim"], args.fps)
    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(tokens, cfg["in_channels"], generator=g)

    to = lambda t: t.to(device=device, dtype=dtype)
    for i in range(len(SIGMAS) - 1):
        s, s_next = SIGMAS[i], SIGMAS[i + 1]
        t_freq = timestep_freq(torch.full((tokens,), s))
        inputs = (to(x), to(t_freq), to(caption), to(mask), to(cos), to(sin))
        if args.save_io and i == 0:
            torch.save({k: v.cpu() for k, v in zip(["latent", "t_freq", "caption", "mask", "cos", "sin"], inputs)},
                       os.path.join(args.out, "step0_inputs.pt"))
        v = run_transformer(chunks, *inputs).float().cpu()
        x = x + (s_next - s) * v
        print(f"step {i + 1}/{len(SIGMAS) - 1} sigma {s:.4f} |v|={v.abs().mean():.4f} finite={torch.isfinite(v).all().item()}")

    del chunks
    gc.collect()
    torch.cuda.empty_cache()
    np.save(os.path.join(args.out, "latent.npy"), x.numpy())

    del vae_sd
    from diffusers import AutoencoderKLLTXVideo

    # from_single_file detects the 0.9.5+ VAE layout (timestep-conditioned decoder) from the keys.
    vae = AutoencoderKLLTXVideo.from_single_file(args.ckpt, torch_dtype=torch.float32).to(device).eval()
    print("vae timestep_conditioning", vae.config.timestep_conditioning)

    # [T_tokens, C] -> [1, C, T, H, W], then undo the per-channel latent normalisation.
    lat = x.reshape(lt, lh, lw, -1).permute(3, 0, 1, 2).unsqueeze(0)
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).cpu()
    std = vae.latents_std.view(1, -1, 1, 1, 1).cpu()
    lat = lat * std + mean
    noise = torch.randn(lat.shape, generator=g)
    lat = lat * (1 - 0.025) + noise * 0.025
    with torch.no_grad():
        video = vae.decode(lat.to(device), temb=torch.tensor([0.05], device=device), return_dict=False)[0]
    video = ((video.clamp(-1, 1) + 1) * 127.5).round().byte().cpu()[0]  # [3, F, H, W]

    from PIL import Image
    for f in range(video.shape[1]):
        Image.fromarray(video[:, f].permute(1, 2, 0).numpy()).save(os.path.join(args.out, f"frame_{f:03d}.png"))
    print("wrote", video.shape[1], "frames to", args.out)


if __name__ == "__main__":
    main()
