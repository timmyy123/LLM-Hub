---
name: train-lora
description: Validate Draw Things LoRA training end to end with draw-things-cli, including tiny-dataset training, loss and scaler checks, checkpoint sanity, and base-versus-LoRA generation comparison.
---

# Train LoRA Skill

Use this workflow to validate Draw Things LoRA training end to end with `draw-things-cli`.

## Goal

Train on a tiny local dataset, watch loss and scaler health, then verify visually with a reference/base/LoRA comparison.

## Build Once

Build the optimized CLI first:

```sh
bazel build --compilation_mode=opt //Apps:DrawThingsCLI
```

Use the built binary for every run:

```sh
bazel-bin/Apps/DrawThingsCLI
```

Do not switch between `bazel run` and `bazel-bin/...` during one validation cycle unless you need to. Reuse the same binary so compile/runtime behavior stays comparable and permission prompts stay predictable.

## Runtime Notes

- Graph compile can be quiet for a long time. With `--compilation_mode=opt`, 5 to 30 minutes is possible on heavy trainers. Do not assume a hang too early.
- Training checkpoints are written into the models directory.
- If the environment requires command approvals, ask once for a stable command shape and stable output/log names, then rename artifacts afterward.
- For local unregistered LoRAs, prefer passing explicit `loras[].version` during generation instead of depending on `custom_lora.json`.

## Dataset Setup

For a single-image reconstruction check:

- Create a local dataset directory.
- Put the image in that directory.
- Add a matching `.txt` caption file beside it.
- Keep the caption minimal for trigger-only tests, for example:

```text
zimgdogref
```

## Validation Ladder

Use this order:

1. Run a 1-step smoke test to confirm the graph compiles, the loss is finite, and a checkpoint is written.
2. Run a 20-step probe to confirm `scale` stays healthy and loss is not obviously blowing up.
3. Run a 100-step check to see whether like-for-like timestep bands decline.
4. Run a 500-step run before declaring the trainer healthy.
5. If you are validating a new attention backend, run the 500-step check on that intended backend, not only on a fallback path.
6. Generate a base image and a LoRA image with the same prompt, seed, and settings.
7. Compose reference/base/LoRA into one image for visual review.

## Baseline Train Command

Use this as the generic `512x512` single-image baseline:

```sh
bazel-bin/Apps/DrawThingsCLI train lora \
  --models-dir /Users/liu/Library/Containers/com.liuliu.draw-things/Data/Documents/Models \
  --model MODEL.ckpt \
  --dataset /tmp/single_image_dataset \
  --output RUN_NAME \
  --name RUN_NAME \
  --steps 500 \
  --rank 32 \
  --scale 1 \
  --learning-rate 0:4e-4 \
  --gradient-accumulation 4 \
  --warmup-steps 20 \
  --save-every 100 \
  --width 512 \
  --height 512 \
  --seed 7 \
  --config-json '{"steps_between_restarts":200}' \
  --no-download-missing \
  --offline
```

## Model Baselines

### Scaler Rules

- Healthy scale is architecture-dependent; choose it from the model's numeric contract.
- Do not set `scale` lower than `1` to make a run stable. That hides overflow and can prevent useful learning.
- If the model does not apply internal scaling that shrinks gradients, start from `32768.0`.
- If the model has explicit internal downscaling or projection compensation, use the validated lower scale for that model family and record why.

### FLUX.1

- Base model: `flux_1_dev_q8p.ckpt`
- Validated guidance settings:
  - `guidanceScale = 3.5`
  - `guidanceEmbed = 3.5`
  - `shift = 2`
  - `resolutionDependentShift = false`
- Healthy `scale` is typically `32768.0`

### Z Image Turbo

- Base model: `z_image_turbo_1.0_i8x.ckpt`
- The validated training baseline is the generic command above.
- The validated trainer scale is `1024.0`
- For generation validation, use `cfg = 1`

### Z Image Base

- Base model: `z_image_1.0_q8p.ckpt`
- The validated training baseline is the generic command above.
- The validated trainer scale is `1024.0`
- For generation validation, keep the model’s recommended Base path:

```json
{"sampler":17,"shift":1.8776105999999999,"resolutionDependentShift":true}
```

- A validated comparison used `cfg = 4`

### Qwen Image BF16

- Base model: `qwen_image_2512_bf16_i8x.ckpt`
- The validated training baseline is the generic command above.
- Healthy `scale` is `32768.0`
- Use the exact training caption first before trying richer prompts

## What To Watch During Training

- Raw loss is noisy because each step samples a different timestep. Do not expect monotonic decline step by step.
- Compare like-for-like timestep bands instead.
- Mid/high timestep bands should usually improve first.
- Low timestep spikes can happen, but they should stay bounded.
- For flow-style objectives, low timestep loss is not always the easiest band. If the target includes a full noise or velocity term that is weakly visible in the low-timestep input, low timestep bins can be intrinsically harder.
- If `scale` steadily collapses, something is seriously wrong.
- If `scale` collapses only on a new backend, compare against the known-stable backend before changing learning rate or dataset settings.
- If `scale` collapses on a model with rotary applied through `cmul`, check whether trainer rotary constants are expanded to the real query/key head count.
- Before blaming the optimizer, confirm the checkpoint is real:
  - nontrivial file size
  - `lora_up` tensors are not all zero

## Generation Validation

Always compare base and LoRA with the exact same:

- prompt
- seed
- width
- height
- steps
- CFG
- model-specific sampler/shift settings

Use the exact training caption first. If that fails, richer prompts are not useful for debugging.

For non-distilled base models, do not under-sample the generation validation. Use the model's real baseline settings, including enough steps, the correct CFG behavior, and the correct sampler family.

For local LoRAs, pass explicit version metadata:

```json
"loras": [
  {
    "file": "RUN_NAME_500_lora_f32.ckpt",
    "version": "MODEL_VERSION",
    "weight": 1.0
  }
]
```

If the model also needs LoRA mode metadata, pass `mode` too.

## Example Generate Commands

### Z Image Turbo

```sh
bazel-bin/Apps/DrawThingsCLI generate \
  --models-dir /Users/liu/Library/Containers/com.liuliu.draw-things/Data/Documents/Models \
  --model z_image_turbo_1.0_i8x.ckpt \
  --prompt zimgdogref \
  --width 512 \
  --height 512 \
  --steps 15 \
  --cfg 1 \
  --seed 7 \
  --config-json '{"loras":[{"file":"RUN_NAME_500_lora_f32.ckpt","version":"z_image","weight":1.0}]}' \
  --offline \
  --no-download-missing \
  --output /tmp/zimg_lora.png
```

### Z Image Base

```sh
bazel-bin/Apps/DrawThingsCLI generate \
  --models-dir /Users/liu/Library/Containers/com.liuliu.draw-things/Data/Documents/Models \
  --model z_image_1.0_q8p.ckpt \
  --prompt zimgdogref \
  --width 512 \
  --height 512 \
  --steps 20 \
  --cfg 4 \
  --seed 7 \
  --config-json '{"sampler":17,"shift":1.8776105999999999,"resolutionDependentShift":true,"loras":[{"file":"RUN_NAME_500_lora_f32.ckpt","version":"z_image","weight":1.0}]}' \
  --offline \
  --no-download-missing \
  --output /tmp/zimg_base_lora.png
```

## Compose A Review Image

Use `ffmpeg` to compose reference, base, and LoRA side by side:

```sh
ffmpeg -y \
  -i /tmp/single_image_dataset/dog.png \
  -i /tmp/base.png \
  -i /tmp/lora.png \
  -filter_complex hstack=inputs=3 \
  -frames:v 1 \
  /tmp/compare.png
```

## Expected Outcomes

- Base should usually look generic or unrelated to the exact training identity.
- A healthy LoRA should pull noticeably toward the training subject by 100 to 500 steps.
- For single-image dog tests, the correct check is not “perfect reconstruction”; it is whether the LoRA image is materially closer to the reference than the base image.
