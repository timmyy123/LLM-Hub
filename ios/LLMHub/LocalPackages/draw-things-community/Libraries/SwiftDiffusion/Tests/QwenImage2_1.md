# Qwen Image 2.1 integration reference

The runtime implements text-to-image, reference-image editing, and native RGBA through `FirstStage`'s existing `.transparent` option. `QwenImage2_1Fixed` computes the invariant prefix once; `QwenImage2_1` only evaluates target-image tokens during denoising. Tiled diffusion slices the full-image target RoPE while sharing the prefix cache across tiles.

## Reference sources

- [Official Qwen repository](https://github.com/QwenLM/Qwen-Image-2.1)
- [Official Diffusers integration PR](https://github.com/huggingface/diffusers/pull/14804)
- Pinned Diffusers revision: `6256aa7666cedd47443adc8f82da9a10e110b09c`:
  - [Transformer](https://github.com/huggingface/diffusers/blob/6256aa7666cedd47443adc8f82da9a10e110b09c/src/diffusers/models/transformers/transformer_qwenimage21.py)
  - [Pipeline](https://github.com/huggingface/diffusers/blob/6256aa7666cedd47443adc8f82da9a10e110b09c/src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py)
  - [VAE](https://github.com/huggingface/diffusers/blob/6256aa7666cedd47443adc8f82da9a10e110b09c/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py)
- [Checkpoint/configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/main)

Use `qwen_image_2.1_f16.ckpt`, `qwen_3_vl_8b_instruct_f16.ckpt`, and `qwen_image_2.1_vae_f16.ckpt`. The local Qwen3-VL q8p checkpoint contains only `text_model`; editing requires `vision_model` from the f16 checkpoint.

## Contracts worth preserving

- For query `q` and key `k`, attention is allowed when `keyValid[k] && (k <= q || (imageID[q] >= 0 && imageID[q] == imageID[k]))`. Adjacent images have distinct IDs. Text is causal; each image sees its own whole block and everything before it. The target cannot influence prefix states.
- Prefix tokens always use the timestep-zero modulation row. Only target tokens use the current denoising timestep. Modulation is shared across the 32 layers.
- DiT RoPE axes are `[16, 56, 56]`. Text advances all axes; an image uses a fixed sequence coordinate and centered spatial coordinates. The cursor advances by `max(height, width)` after an image.
- Qwen3-VL uses the exact raw system/user/assistant template, DeepStack features, and final hidden states before final RMS normalization. Drop the system message. Each VLM image slot is replaced by four VAE latent tokens; surrounding text stays in sequence order.
- Vision and VAE use the same reference geometry, rounded to multiples of 32. Only the vision copy composites alpha over white. App tensors store alpha first in `[0, 1]` and RGB in `[-1, 1]`; the VAE uses RGBA in `[-1, 1]`.
- Latents have 64 channels and 16x spatial compression. Use the checkpoint's 64-channel mean/std. The old Qwen tiny decoder and preview projection are incompatible.
- The default CLI schedule is deterministic DDIM trailing / flow Euler, 40 steps, CFG 1. Shift is `exp(0.5 + (imageTokens - 256) * 0.4 / (8192 - 256))`; stretch the last nonterminal sigma to `0.02` before appending zero.

## Validation

`bazel test //Libraries/SwiftDiffusion:QwenImage2_1ContractTests` checks block boundaries against the independent image-ID predicate, rotary positions, and the shifted/stretched schedule, including preservation of existing schedules without `shiftTerminal`.

Run actual generation with a fixed seed and inspect the resulting image. For transparency, inspect alpha values as well as PNG metadata. For editing, use an RGBA reference and verify the requested edit while preserving alpha. Same seeds across PyTorch and NNC do not establish numerical parity: a parity harness must supply identical initial tensors. End-to-end PyTorch numerical parity is not claimed by these tests.

```sh
bazel run //Apps:DrawThingsCLI -- generate \
  --model qwen_image_2.1_f16.ckpt \
  --prompt 'a red apple on a wooden table, studio photograph' \
  --steps 40 --cfg 1 --seed 42 --width 512 --height 512 \
  --offline --no-download-missing --output /tmp/qwen21.png
```

Build the full app with `bazel build //Apps/DrawThings:DrawThings --ios_multi_cpus=arm64`.

Additional runtime checks use `--negative-prompt 'blue, blurry' --cfg 2 --config-json '{"batchSize":2}'` for CFG plus batching, and two repeated `--image` arguments for multiple references. Keep image-slot metadata on the GPU along with the other text-encoder outputs: the shared batch-expansion path copies those tensors on-device.

Initial full-sequence validation on 2026-09-21:

- Four Qwen Image 2.1 contract tests and ten existing compute-estimate tests pass.
- Diffusion library, CLI, full arm64 iOS app, ModelConverter, ModelQuantizer, and LoRAConverter build.
- Strict checkpoint loading passed during bring-up for DiT, text, vision, VAE encoder, and VAE decoder; those temporary strict checks were removed.
- 40-step text-to-image and transparent generation were visually inspected. The transparent 512x512 sample has alpha spanning 0–255, with 137,003 pixels below 16 and 122,434 above 239.
- A 40-step RGBA reference edit changed a red apple to green while preserving its leaf, outline, and transparency (136,093 pixels below alpha 16).
- Two-image batching with CFG 2 and unequal prompt lengths produced two distinct PNGs.
- A two-reference one-step run completed, exercising both vision and VAE packing. This is a runtime smoke test, not a multi-reference image-quality benchmark.

GPU timings are not acceptance criteria because the device is shared. The frozen full-sequence model lives in `QwenImage2_1Unsplit.swift` in the test target only.

## Fixed-prefix execution

- The fixed graph projects the text and reference latents, interleaves them, and uses timestep zero through all prefix blocks. With Flash Attention, attention is segmented: each image attends to its own entire block and preceding tokens; each text segment uses `isCausal: true`, aligned to the end of its key sequence. No segment can see later blocks, and neither text nor image segments require an explicit mask. The explicit-matmul fallback retains the full prefix causal mask and follows QwenImage's serial per-head execution when batch size times head count is at most 256.
- Cache post-normalization, post-RoPE keys and unmodified values for every transformer layer. The final prefix layer stops after K/V projection: its query, attention output, and MLP are unused.
- The target graph concatenates each cached K/V pair with that layer's target K/V. Target queries attend bidirectionally to the concatenated prefix and target tokens without a mask. `UNetProtocol` evaluates CFG branches separately and trims cached K/V to the actual prefix length before calling the model.
- Trailing text padding never affects earlier prefix tokens. Trim its cached keys and values before target attention and advance target RoPE using the actual branch length. Distinct CFG branches have separate prefix caches; generated-image batches share the computation and receive repeated cache tensors.
- Fixed outputs are **five timestep tables** (attention scale/gate, MLP scale/gate, final scale), then **32 K/V pairs**, all in the model dtype. Tables have `[steps, 4096]`; K/V have `[CFG branches, prefix length, 32, 128]`. The timestep MLP input adds a final zero-time row for prefix modulation.
- Runtime conditions are target RoPE, the five tables, then the 32 K/V pairs (70 tensors). `UNetExtractConditions` slices only the five timestep tables. After slicing, `QwenImage2_1` takes target latent followed by these conditions.
- Both graphs load their subset of the existing `dit` weights with unchanged native names. No checkpoint conversion is required. Fixed/main instruction counts reflect the split, including skipping the final prefix residual/MLP.

The checkpoint-backed parity test uses identical deterministic context/reference/target tensors against the frozen 32-layer baseline. It covers text-only and two adjacent reference blocks, unequal CFG lengths, two timesteps reusing one prefix cache, and all four attention levels. INT8 attention uses a separate smoke-test error budget (10% relative RMS versus 1% for FP16); the current quantized cases differ from the baseline by up to 6.7%. It requires the external-data codec used by the app loader; strict name checking alone does not decode external tensor storage.

VAE parity covers encoder and decoder in FP32 and FP16 with Flash Attention enabled and disabled. FirstStage enables VAE Flash Attention when supported and the actual attention grid, accounting for tiling and the encoder's 16x downsampling, reaches 256 x 176 tokens.

FirstStage also honors the existing high-precision retry when compiling either Qwen VAE graph. The retry compiles with FP32 inputs so checkpoint parameters match FP32 execution. `testDecoderHighPrecisionFallback` forces a failed first decode and checks that the checkpoint-backed retry returns finite RGBA pixels.

The VAE retains input precision until the decoder's final upsample convolution. That upsample and the final residual stage run in FP32, returning to input precision after the final normalization. On the captured two-step reference-edit input, the correct final-upsample output reached 107,039, beyond FP16 range; the original decoder produced 2,240 nonfinite output values in the second batch image. Encoder, attention, and earlier residual-block normalization are unchanged. Both captured images decode with finite values using either attention backend without the full-decoder retry. Relative RMS error against full FP32 is 0.17–1.28%, with maximum absolute error up to 2.54; this is a finite-output smoke check, not strict per-pixel parity.

`testDecoderFloat16Range` checks finite output and reports relative RMS and maximum absolute error against full FP32 as diagnostics. It uses deterministic unscaled latents in the observed input range by default. To replay a captured input, set `QWEN_IMAGE_2_1_VAE_LATENT` to a tensor store containing an NHWC tensor named `latent`, already unscaled for the VAE.

`testTiledDiffusionMatchesExplicitTiles` exercises the public UNet runtime on a rectangular canvas with two images per CFG branch, unequal prompt lengths, and cached reference conditioning. It compares all four edge-anchored tiles and their overlap blending against explicit tile calls with independently gathered global RoPE coordinates. Target tokens use one latent pixel each, so tile coordinates are not divided by two. Cached K/V and timestep modulation are shared across tiles. Batched K/V concatenation disables the same optimization as Flux2 to keep RoPE outputs contiguous.

Tiled runtime validation on 2026-09-21: the explicit-tile comparison passed with maximum absolute error 0 and finite outputs; the forced VAE precision retry also passed. Two-step CLI smoke runs completed for 512 x 384 text-to-image and 384 x 256 reference editing with CFG 2 and a batch of two, using 256 x 256 tiles and 64-pixel overlap. Both edit outputs are RGBA with alpha spanning 0–255. These low-step images exercise execution and PNG output only; their coarse artifacts do not establish image quality or edit fidelity.

```sh
bazel test //Libraries/SwiftDiffusion:QwenImage2_1Tests \
  --test_env=QWEN_IMAGE_2_1_CHECKPOINT=/path/to/qwen_image_2.1_f16.ckpt \
  --test_output=all
```

Fixed-prefix validation on 2026-09-21:

- All three test targets pass: `QwenImage2_1Tests`, `QwenImage2_1ContractTests`, and `ComputeUnitsTests` (including shared prefix work across generated-image batches).
- The eight checkpoint comparisons (two attention backends × text/reference prefix × two timesteps) have relative RMS error between 0.103% and 0.155%, with maximum absolute element error 0.0105. All outputs are finite. Fixed/main mapper keys together exactly cover the frozen baseline, with identical native parameter names.
- The explicit-matmul fallback intentionally uses the full prefix mask. Segmenting that path produced approximately 1.35% relative RMS error in the synthetic reference case; retaining its masked-softmax layout restored the above parity without reintroducing per-step prefix work.
- The cleaned CLI and full arm64 iOS app build successfully. Strict loading remains in the checkpoint test only.
- The same seed-42, 40-step 512×512 RGBA generation and red-to-green reference edit were re-run and visually inspected. RGB channel mean absolute errors versus the unsplit images were below 0.10 and 0.083 respectively (8-bit scale); alpha mean absolute errors were 0.0239 and 0.0081. Both outputs span alpha 0–255 and retain transparent backgrounds.
- Two-step runtime checks passed for CFG 2 with unequal prompt lengths and an image batch of two (distinct output PNGs), and for two reference images. These exercise cache reuse across steps; they are smoke checks, not low-step image-quality benchmarks.

## Model-style and NHWC refactor validation

- DiT builders compose file-private transformer blocks with explicit weight mappers. The shared `timeEmbedding` helper replaces the model-specific helper, and Q/K use ordinary `RMSNorm`.
- Flash Attention uses `isCausal: true` for text segments and unmasked attention for each image segment. The target model is unmasked; `UNetProtocol` splits CFG branches and trims cached K/V to their valid lengths.
- Separate VAE encoder/decoder builders keep activations NHWC, including attention and resampling shortcuts. The downsampling shortcut follows WanVAE's explicit permutation/copy before channel grouping.
- Eight checkpoint-backed DiT comparisons against the frozen full-sequence implementation pass with relative RMS error 0.107–0.221%, maximum absolute error 0.01465, and unchanged checkpoint parameter names.
- The NHWC encoder and decoder match the frozen NCHW implementation on rectangular inputs with relative RMS errors 0.00359% and 0.00712%, respectively. Maximum absolute errors are 0.001606 and 0.000419. Both load the existing VAE checkpoint strictly.

To include VAE parity in the checkpoint test, add:

```sh
--test_env=QWEN_IMAGE_2_1_VAE_CHECKPOINT=/path/to/qwen_image_2.1_vae_f16.ckpt
```

## LoRA import and inference

`LoRAQwenImage2_1` and `LoRAQwenImage2_1Fixed` retain the original builders' parameter names and attention paths. Both use the same transformer-layer indices for attention and MLP projections. The fixed encoder applies adapters before computing the prefix K/V and timestep tables. It supports merged weights and separate LoRA execution using the same selection and loading policy as Flux2.

The importer recognizes 4096-wide layer-31 Qwen 2.1 attention / MLP adapters and uses the existing PEFT, Diffusers, and flattened-name normalization. Partial adapters that lack the identification layers can use `--version qwen_image_2.1`. Packed `modulation.1` adapters split into four projections with the existing alpha/rank scaling; shared fixed/main projections retain one checkpoint key.

Published adapters used for validation on 2026-09-21:

- [Anime consistency](https://huggingface.co/WarmBloodAban/Qwen-Image-2.1-LoRAs/tree/90abd1e7f590fc8ba052e0d2e6de9dc01730e260): `Qwen2.1_Anime_consistency.safetensors`, rank 32, 448 BF16 tensors covering all 32 layers' attention and MLP projections. SHA-256: `0c171eb802ea8051b511f2d93c1743eeadd030316255aa98fa65d40809366752`.
- [Stop-motion consistency](https://huggingface.co/ProCreations/stopmotion-consistency-v1-lora/tree/700ec417de23c1ca75a1dc96ffb4cc3999632b16): `stopmotion-consistency-v1.safetensors`, rank 16, 256 FP32 tensors covering all 32 layers' attention projections. SHA-256: `d68479ed2c246bc7ebea8a62019b10da903772d30854385302b2d66de99df51a`.

Both use `.lora_A.default.weight` / `.lora_B.default.weight` and import without a forced version. No verified Qwen Image 2.1 Turbo / Lightning adapter was found during this search; older Qwen Image / 2512 distilled adapters are not compatible with this architecture.

```sh
bazel test //Libraries/ModelOp:QwenImage2_1LoRATests \
  --test_env=QWEN_IMAGE_2_1_CHECKPOINT=/path/to/qwen_image_2.1_f16.ckpt \
  --test_env=QWEN_IMAGE_2_1_LORA=/path/to/Qwen2.1_Anime_consistency.safetensors \
  --test_output=all
```

The synthetic import test verifies shared input/output and timestep projections, non-unit alpha/rank scaling, and packed modulation splitting. The checkpoint test checks that every published adapter projection is imported, compares merged versus separate execution for first/last prefix keys and the final denoiser output, checks finite outputs, and confirms that the adapter changes the base result.

With the anime adapter at weight 0.75, all 224 projection pairs imported. Merged versus separate FP16 inference had relative RMS errors of 0.058% for the first prefix keys, 1.179% for the final prefix keys, and 0.287% for the denoiser output, within the test's 2% budget. Both tests passed, as did the LoRAConverter, optimized CLI, and full arm64 iOS app builds.

The normal CLI also completed a 40-step, seed-42, 512x512 reference edit with the anime adapter at weight 0.6. The red apple became green while retaining its leaf, outline, and transparency; the RGBA output's alpha spans 0–255. A preceding attempt at weight 0.75 to turn the apple into an anime character mostly retained the original apple, so that run establishes execution rather than style-transfer quality. These are integration checks, not a quality benchmark for this experimental character-consistency adapter.

For an imported adapter outside `custom_lora.json`, pass its version with the CLI override:

```sh
--config-json '{"loras":[{"file":"qwen_image_2.1_anime_consistency_lora_f16.ckpt","weight":0.6,"version":"qwen_image_2.1"}]}'
```
