# LTX-Video on the Snapdragon NPU (QNN)

Scripts that turn Lightricks **LTX-Video 2B 0.9.8 distilled** into QNN context binaries for the
Hexagon HTP. The app runs them through `LTXVQnnRunner` / `LTXVideoQnnVAE` in
`android/stable-diffusion.cpp/src/model/diffusion/ltxv_qnn.hpp`. T5 and image-to-video encoding
stay on the CPU (GGUF files from `calcuis/ltxv-gguf`).

## Host setup (x86-64 Linux / WSL2)

- QAIRT SDK **2.50.0.260828** (must match the runtime the app ships through
  `litert-npu-runtime-qualcomm`). Download:
  `https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.50.0.260828/v2.50.0.260828.zip`
- Python 3.10 venv with `torch`, `diffusers`, `transformers`, `safetensors`, `onnx==1.16.2`,
  `onnxruntime`, `numpy<2`, plus the QAIRT converter dependencies.
- `libc++1`, `libc++abi1`, `libunwind` for the QAIRT binaries (can be unpacked without root with
  `apt download` + `dpkg -x`), and `libpython3.10.so` on `LD_LIBRARY_PATH`.
- ~32 GB RAM (+ swap) and ~150 GB disk.
- Checkpoint: `Lightricks/LTX-Video` @ `8984fa25007f376c1a299016d0957a37a2f797bb`,
  file `ltxv-2b-0.9.8-distilled.safetensors`.

## Files

| Script | Purpose |
|---|---|
| `ltxv_model.py` | Static-shape PyTorch rewrite of the LTX transformer (mirrors stable-diffusion.cpp's `LTXVRunner`), host-side timestep/RoPE helpers |
| `export_onnx.py` | Exports the 28 blocks as 4 chunks (prelude+0-6, 7-13, 14-20, 21-27+head), fixed `--tokens` |
| `make_overrides.py` | Weight-only INT8 encodings (per output channel), activations stay FP16 |
| `export_vae_onnx.py` | VAE decoder export with rank ≤ 5 reshapes/norms/broadcasts (HTP limits), verified bit-exact against diffusers |
| `make_vae_calib.py` | Calibration inputs (real latents) for the W8A16 VAE decoder |
| `build_shape.sh` | Export → convert → compile all graphs for one `<arch> <soc_model> <W> <H> <frames>` |
| `make_stub.py` | `ltxv_qnn_stub.safetensors`: the transformer shape tensors stable-diffusion.cpp needs for detection (the weights live in the context binaries) |
| `validate_torch.py`, `decode_latent.py` | Reference pipeline / reference VAE decode on a CUDA GPU for quality checks |

## Precision choices (measured on device)

- **Transformer:** INT8 weights + FP16 activations. FP16 weights (3.9 GB) do not fit the
  Hexagon's ~3.5 GB near-memory VA space; INT8 halves it and is ~3x faster. Error vs FP32:
  ~3% per 7-block chunk, visually indistinguishable.
- **VAE decoder:** Conv3d has no INT8-weight/FP16-activation kernel. Plain FP16 on the HTP loses
  precision in long convolution accumulations (late frames blur); W8A16 (calibrated) uses exact
  INT32 accumulation.
- Chunks are loaded as one group sharing a single spill-fill buffer
  (`QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS`). If the VAE does not fit next to the
  transformer, the runner frees the transformer, decodes, and reloads it lazily.

## Shapes per chip

| Chip | Hexagon | soc_model | Shape |
|---|---|---|---|
| Snapdragon 8 Gen 2 | v73 | 43 | 512x320, 25 frames (768x512 does not fit the VA space) |
| Snapdragon 8 Gen 3 | v75 | 57 | 768x512, 25 frames |
| Snapdragon 8 Elite | v79 | 69 | 768x512, 25 frames |
| Snapdragon 8 Elite Gen 5 | v81 | 87 | 768x512, 25 frames |

## Hugging Face layout expected by the app

```
<repo>/
  ltxv_qnn_stub.safetensors
  v73/ltxv_chunk0.bin … ltxv_chunk3.bin, ltxv_vae_512x320x25.bin
  v75/ltxv_chunk0.bin … ltxv_chunk3.bin, ltxv_vae_768x512x25.bin
  v79/…
  v81/…
```

Published at [timmyboy26/LTX-Video-2B-0.9.8-distilled-QNN](https://huggingface.co/timmyboy26/LTX-Video-2B-0.9.8-distilled-QNN).
After uploading a new build, point `LTXV_QNN_REPO` in
`app/src/main/java/com/llmhub/llmhub/data/ModelData.kt` at the new commit
(`https://huggingface.co/<repo>/resolve/<commit>`, never `main`) and update the sizes in
`ltxvNpuBuilds` from `curl -sIL <url> | grep -i x-linked-size`.
