"""Calibration inputs for fully integer (W8A16) transformer chunks.

Runs the reference sampler (validate_torch.py's chunks, fp16 on CUDA) for several prompts and saves the
inputs of every chunk at every denoising step as raw fp32 files, plus one onnxruntime/QAIRT input list
per chunk. Activation ranges move with the timestep, so all 8 steps are captured.

Usage: python make_dit_calib.py --ckpt models/ltxv-2b-0.9.8-distilled.safetensors --repo repo \
           --width 512 --height 320 --out calib_dit_512
"""

import argparse
import gc
import os

import numpy as np
import torch
from safetensors.torch import load_file

from ltxv_model import config_from_state_dict, rope_tables, timestep_freq, MASK_NEG
from validate_torch import CHUNKS, SIGMAS, build_chunks, encode_prompt

PROMPTS = [
    "A fluffy golden retriever puppy runs across a sunny green meadow toward the camera, ears bouncing, low-angle tracking shot, cinematic lighting.",
    "A busy neon-lit city street at night in the rain, cars passing with glowing headlights, reflections on the wet asphalt, people with umbrellas, handheld camera.",
    "A red vintage sports car drives along a winding coastal road at sunset, waves crashing on the rocks below, the camera tracks alongside the car.",
    "Close-up of a woman with curly hair laughing in a cozy cafe, warm window light, shallow depth of field, she turns her head toward the window.",
    "Aerial drone shot over snowy mountain peaks at sunrise, clouds drifting through the valleys, slow forward camera movement, golden light on the snow.",
    "A chef tosses vegetables in a flaming wok in a busy restaurant kitchen, steam and sparks, fast handheld close-up, dramatic orange lighting.",
    "Underwater coral reef with colorful tropical fish swimming past, sunbeams filtering through clear blue water, slow gliding camera.",
    "An astronaut walks across a dusty red Mars landscape, footprints in the sand, Earth visible in the dark sky, wide cinematic shot.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    captions = [encode_prompt(args.repo, p, 128, device) for p in PROMPTS]  # T5 loads once per prompt, then frees

    sd = load_file(args.ckpt)
    cfg = config_from_state_dict(sd)
    chunks = build_chunks(sd, cfg, torch.float16, device)
    del sd
    gc.collect()

    lw, lh, lt = args.width // 32, args.height // 32, (args.frames - 1) // 8 + 1
    tokens = lw * lh * lt
    cos, sin = rope_tables(lw, lh, lt, cfg["dim"], args.fps)
    to = lambda t: t.to(device=device, dtype=torch.float16)
    lists = [[] for _ in CHUNKS]

    def save(chunk, sample, named):
        d = os.path.join(args.out, f"chunk{chunk}", sample)
        os.makedirs(d, exist_ok=True)
        parts = []
        for name, t in named.items():
            path = os.path.join(d, f"{name}.raw")
            t.detach().float().cpu().numpy().astype(np.float32).tofile(path)
            parts.append(f"{name}:={os.path.abspath(path)}")
        lists[chunk].append(" ".join(parts))

    for p, (caption, attn) in enumerate(captions):
        mask = ((1.0 - attn) * MASK_NEG).reshape(1, -1)
        x = torch.randn(tokens, cfg["in_channels"], generator=torch.Generator().manual_seed(100 + p))
        for i in range(len(SIGMAS) - 1):
            s, s_next = SIGMAS[i], SIGMAS[i + 1]
            sample = f"p{p}_s{i}"
            t_freq = timestep_freq(torch.full((tokens,), s))
            common = {"mask": to(mask), "cos": to(cos), "sin": to(sin)}
            with torch.no_grad():
                inp0 = {"latent": to(x), "t_freq": to(t_freq), "caption": to(caption), **common}
                save(0, sample, inp0)
                h, tmod, emb, context = chunks[0](inp0["latent"], inp0["t_freq"], inp0["caption"], *common.values())
                for c in range(1, len(CHUNKS) - 1):
                    save(c, sample, {"x": h, "tmod": tmod, "context": context, **common})
                    h = chunks[c](h, tmod, context, *common.values())
                save(len(CHUNKS) - 1, sample, {"x": h, "tmod": tmod, "emb": emb, "context": context, **common})
                v = chunks[-1](h, tmod, emb, context, *common.values()).float().cpu()
            x = x + (s_next - s) * v
        print(f"prompt {p + 1}/{len(PROMPTS)} done, |x|={x.abs().mean():.4f}", flush=True)

    for c, lines in enumerate(lists):
        with open(os.path.join(args.out, f"chunk{c}", "input_list.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
    print("CALIB_DONE", sum(len(l) for l in lists), "samples")


if __name__ == "__main__":
    main()
