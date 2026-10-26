"""Stable Audio 3 Small (music) on the GPU with ONNX Runtime + DirectML, for Snapdragon PCs.

Stability AI's ONNX exports (stable-audio-3-optimized/onnx: T5Gemma encoder, the sm-music DiT in fp16 and the SAME-S
decoder with PCM output) driven by the tokenizer, pingpong schedule, noise and sampler of Stability's own LiteRT
runtime (optimized/tflite/models/defs/tflite_pipeline.py), so a seed means the same thing on both paths.
Prints "[1/3]", "sampling step i/n" and "[3/3]" lines that the app turns into progress.

Measured on a Snapdragon X Elite (Adreno X1-85), 10 s of audio: DiT 0.2 s per step on DirectML vs 1.7 s for the int8
LiteRT DiT on the CPU; the clip matches the CPU-only ONNX run for the same seed (spectrogram correlation 0.95).
T5Gemma stays on the CPU: on DirectML its output is wrong (the clip comes out near-silent, correlation 0.14).
"""
import argparse
import math
import os
import sys
import time
import wave

import numpy as np

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", required=True, help="Folder with encoder.onnx, dit.onnx and decoder.onnx")
    ap.add_argument("--runtime", required=True, help="Stability's optimized/tflite folder (pipeline helpers, tokenizer)")
    ap.add_argument("--cpu", default="encoder", help="Comma-separated stages to run on the CPU (encoder, dit, decoder; all)")
    args = ap.parse_args()

    sys.path.insert(0, args.runtime)
    from models.defs import tflite_pipeline as P
    import onnxruntime as ort

    on_cpu = set(args.cpu.split(","))
    if "DmlExecutionProvider" not in ort.get_available_providers():
        on_cpu.add("all")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    def session(name):
        t = time.time()
        cpu = "all" in on_cpu or name.split(".")[0] in on_cpu
        providers = ["CPUExecutionProvider"] if cpu else ["DmlExecutionProvider", "CPUExecutionProvider"]
        s = ort.InferenceSession(os.path.join(args.onnx, name), opts, providers=providers)
        print(f"        load {name} on {s.get_providers()[0]} {time.time() - t:.1f}s", flush=True)
        return s

    total = time.time()
    latents = max(1, int(math.ceil(args.seconds * SAMPLE_RATE / SAMPLES_PER_LATENT)))
    print(f"  [1/3] T5Gemma (tokenize + encode)", flush=True)
    ids, mask = P.Tokenizer(os.path.join(args.runtime, "models", "tokenizer.model"))(args.prompt)
    encoder = session("encoder.onnx")
    hidden = encoder.run(None, {"input_ids": ids.astype(np.int64), "attention_mask": mask.astype(np.int64)})[0].astype(np.float32)
    del encoder

    print(f"  [2/3] DiT (sm-music, {args.steps} steps, {latents} latents)", flush=True)
    dit = session("dit.onnx")
    feeds = {
        "t5_hidden": hidden,
        "t5_mask": mask.astype(np.float32),
        "seconds_total": np.array([args.seconds], np.float32),
        "local_add_cond": np.zeros((1, 257, latents), np.float32),
    }
    clock = [time.time()]

    def forward(x, t, _cross, _gcond):
        return dit.run(["velocity"], {**feeds, "x": x.astype(np.float32), "t": np.array([t], np.float32)})[0]

    def on_step(i, n):
        print(f"        sampling step {i}/{n}  {(time.time() - clock[0]) * 1000:.0f} ms", flush=True)
        clock[0] = time.time()

    x0, noise = P.make_noise(latents, args.steps, args.seed)
    latent = P.sample(forward, x0, noise, P.build_pingpong_schedule(args.steps), None, None, on_step=on_step)
    del dit

    print(f"  [3/3] Decoder (same-s) + WAV", flush=True)
    decoder = session("decoder.onnx")
    padded = latent if latents % 2 == 0 else np.concatenate([latent, latent[..., -1:]], axis=-1)  # SAME-S wants an even length
    t = time.time()
    pcm = decoder.run(None, {"latent": padded.astype(np.float32)})[0][0]  # (samples, 2) int32, scaled and limited in-graph
    print(f"        decode {(time.time() - t) * 1000:.0f} ms", flush=True)
    samples = int(round(args.seconds * SAMPLE_RATE))
    pcm = np.clip(pcm[:samples], -32767, 32767).astype(np.int16)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    audio = pcm.astype(np.float32) / 32767
    print(f"  done {time.time() - total:.1f}s  peak {np.abs(audio).max():.3f} rms {np.sqrt((audio ** 2).mean()):.3f}  {args.out}", flush=True)


if __name__ == "__main__":
    main()
