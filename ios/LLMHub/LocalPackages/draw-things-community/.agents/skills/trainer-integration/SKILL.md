---
name: trainer-integration
description: Add or tighten Draw Things LoRA trainer support for generative models available in the Draw Things app / CLI, covering LoRA builders, trainer dispatch, tokenizer and fixed-encoder wiring, checkpoint export, numerical debugging, and validation.
---

# Trainer Integration Skill

Use this workflow when adding LoRA training support for generative models available in the Draw Things app / CLI or when tightening an existing trainer path. Follow `Flux1` first, then copy only the model-specific pieces you actually need.

## Goal

Add a new trainer path that:

- compiles with `DrawThingsCLI`
- writes real LoRA checkpoints
- keeps loss finite
- survives a real 100 to 500 step run
- reproduces an obvious visual shift during generation

## Primary Files

- `Libraries/Trainer/Sources/LoRATrainer.swift`
- `Libraries/Trainer/Sources/LoRATrainerCheckpoint.swift`
- `Libraries/SwiftDiffusion/Sources/Models/<Model>.swift`
- `Apps/DrawThingsCLI/DrawThingsCLI.swift`
- `Apps/DrawThings/Sources/LoRA/LoRATrainingWorkflow.swift`

Use these references first:

- `trainFlux1(...)` in `LoRATrainer.swift`
- `LoRAFlux1` and `LoRAFlux1Fixed` in `Flux1.swift`

## Integration Checklist

### 1. Add the LoRA model builders

- Add `LoRA<Model>` and, if needed, `LoRA<Model>Fixed`.
- Keep changes local to the LoRA path whenever possible.
- Do not rewrite the base runtime path unless training forces it.
- Do not assume trainer-side `LoRANetworkConfiguration` is enough. The LoRA builders themselves may need changes to:
  - the outer `Model(..., trainable:)` setting
  - inner `Model(..., trainable:)` boundaries
  - gradient-checkpoint wiring on the relevant block builders
- Follow `Flux1` exactly here:
  - keep the outer LoRA model `trainable: false`
  - thread gradient-checkpoint flags through the LoRA-specific builders, not just the trainer call site
  - if a nested `Model` wrapper needs an explicit `trainable:` value to preserve the intended trainable surface, do that in the LoRA path instead of changing the base model path
- Remember the rule: if a parent `Model` is `trainable: false`, submodules are effectively non-trainable unless the graph structure explicitly reintroduces the LoRA trainable surface the same way the working reference path does.
- If the top-level runtime model is also wrapped by a LoRA builder, keep that wrapper `trainable: false` and rely on the individual LoRA layers for trainability. Accidentally making the parent trainable turns the path into a full fine-tune.
- Mirror Flux1-style `LoRANetworkConfiguration` usage and checkpointing flags.

### 2. Add trainer dispatch

- Add a `train<Model>(...)` entry in `LoRATrainer.swift`.
- Dispatch to it from the main trainer switch.
- Add the matching `version` handling in CLI and app workflow code.

### 3. Wire trainable keys in both entry points

- Add trainable-key helpers in `DrawThingsCLI.swift`.
- Mirror the same version handling in `LoRATrainingWorkflow.swift`.
- Do not stop after CLI only; the app workflow needs the same version-specific key selection.

### 4. Wire tokenizers cleanly

- Extend `LoRATrainingDependency` if the model needs a tokenizer that is not already injected.
- Keep the dependency factory-based, like the existing tokenizer fields.
- Mirror the version-specific tokenizer stack in both CLI and app workflow code.

Small but important example:

- `Qwen Image` should use the Qwen 2.5 tokenizer, not Qwen 3.
- `Z Image` uses Qwen 3.

### 5. Mirror the fixed encoder path, but keep it trainer-friendly

- Start from the runtime fixed path, but do not copy disk-cache shortcuts from `UNetFixedEncoder.swift`.
- Treat `UNetFixedEncoder` as a legacy-named integration boundary for the main diffusion model / DiT path, not as a literal UNet requirement.
- Run the fixed model directly in the trainer.
- Prefer batched fixed inference over per-sample loops when the fixed builder supports it.
- Feed fixed outputs back into the main trainable graph as `graph.constant(...)`.
- Avoid per-sample `toCPU()/toGPU()` rebuilds when a batched constant path works.
- Flush partial batches. If fixed inference batches at a fixed size, make the target batch size shrink near the end of training so the tail samples are actually trained.
- Preserve output dtypes exactly:
  - context stays in the model float type
  - AdaLN chunks stay `Float`
  - shift/scale stays in the model float type
  - do not force every fixed condition to `Float` just because one condition needs it

### 6. Match weight loading exactly

- Use the same model key and codec list that the runtime path needs.
- Mapping is not used when loading trainer weights; mapping matters for import, not for `read(...)`.
- If the runtime model needs `.i8x`, the trainer needs `.i8x` too.

This was the difference between immediate `nan` and a finite first step for Qwen Image.

### 7. Handle rotary the training-safe way

- If `cmul` backward cannot reduce broadcast semantics, fully expand rotary on the training path.
- Keep the expanded rotary parity-preserving.
- Do not trust `[1, seq, 1, dim]` rotary broadcasting into `[batch, seq, heads, dim]` just because forward compiles. The backward path can still be wrong or unstable.
- If memory matters, cache compact one-head rotary constants, then expand them to the actual query/key head count before entering the trainable graph.
- Prefer slicing query rotary from the shared `rot` tensor instead of inventing a separate query-rotary input when the slice is enough.
- Do not let training-only rotary plumbing leak into the normal inference interface if you can avoid it.

### 8. Choose scaler and attention backend intentionally

- Pick the initial `GradScaler` from the model's numeric contract, not by trial-and-error lowering.
- If the model has no internal residual/projection scaling that already shrinks gradients, start from the high healthy scale used by the stable trainers, usually `32_768`.
- Never accept a scaler below `1` as a fix. That usually masks overflow and can make the trainer learn too slowly or not at all.
- When validating attention backends, first prove a known-stable fallback can train, then run the intended backend for the full validation ladder.
- Do not accidentally force the fallback path in trainer code. Use the model-appropriate default such as `valueOr(.scale1)` when the configured backend is supposed to participate in training.

### 9. Keep the fixed conditions simple

- Use `graph.constant(...)` for precomputed fixed conditions.
- Use `.copied()` when slicing batched fixed outputs back into per-sample constants.
- Do a dry-run forward before the first real step to allocate the largest graph state up front:

```swift
let _ = dit((width: latentsWidth, height: latentsHeight), inputs: latents, cArr)
```

### 10. Update checkpoint export

- Add the new version branch in `LoRATrainerCheckpoint.swift`.
- Point it at the correct model key, usually `dit`.
- Confirm the saved LoRA file is real:
  - nontrivial size
  - `__up__` tensors present and nonempty

## Model-Specific Lessons

### Z Image

- Follow the Flux1 trainer pattern closely.
- Keep changes concentrated in `LoRAZImage`, `LoRAZImageFixed`, and `trainZImage(...)`.
- Do not add broad changes to base `ZImage` / `ZImageFixed` unless training truly needs them.
- Use the shared `rot` path; slice from it instead of carrying a second query-rotary input.
- Training currently uses fully expanded rotary because backward needs it.
- Keep `x_pad_token` on the GPU if you cache it for trainer constants.
- Current healthy trainer scale is `1024`.

### Qwen Image

- Use the Qwen 2.5 tokenizer.
- Use the real training token length, not a padded constant everywhere.
- Batched `encodeQwenFixed(...)` is better than the old per-sample CPU/GPU rebuild path.
- Feed fixed outputs back as constants.
- Keep `.i8x` in both fixed and main trainer reads when the checkpoint needs it.
- `isBF16` on the fixed side is mostly about scaling math; do not assume the whole fixed contract becomes BF16.
- The main LoRA model still owns the explicit BF16 conversion path.

## First Debug Pass

If a new trainer is broken, check these in order:

1. 1-step run: finite loss?
2. checkpoint written?
3. checkpoint nonempty?
4. `lora_up` tensors nonzero?
5. fixed conditions fed as constants, not variables?
6. tokenizer stack correct?
7. runtime and trainer codec lists match?
8. rotary fully expanded only where backward needs it?
9. scaler chosen from the model's numeric contract, not lowered below `1` to hide instability?
10. optional attention backend selected intentionally, not accidentally through a global default?

## Failure Patterns

### Immediate `nan` at step 0

- Wrong codec list, especially missing `.i8x`
- Wrong tokenizer or token length
- Broken fixed-condition path
- Wrong dtype assumptions on BF16 models

### `lora_up` stays zero almost everywhere

- Gradient is being cut before most LoRA layers
- Check the trainable surface first
- Then check the fixed-condition boundary and backward-only branches

### Dynamic scale steadily collapses

- Treat this as a numerical stability issue first, especially when validating a new attention backend.
- Do not lower the initial scale below `1`; fix the overflow source instead.
- Compare against the known-stable attention mode before changing optimizer hyperparameters.
- If only the experimental attention mode collapses, suspect backward precision, scaling, or gradient staging rather than the dataset.
- If collapse disappears after expanding rotary to full heads, the root cause was shape/broadcast semantics in backward, not optimizer settings.

### Low timestep loss is higher than mid/high timestep loss

- This is not automatically a bug for flow-style objectives.
- If the training target includes a full noise or velocity term that is only weakly present in the low-timestep input, low timestep bins can have higher irreducible error.
- Compare like-for-like timestep bins over time instead of expecting low timesteps to be easiest.

### Generation ignores the LoRA

- The LoRA may be loading with the wrong version
- Use explicit `loras[].version` in CLI validation

## Validation

After code integration, validate with the `$train-lora` workflow:

1. `bazel build --compilation_mode=opt //Apps:DrawThingsCLI`
2. 1-step smoke test
3. 20-step stability probe
4. 100-step run
5. 500-step run on the intended attention backend
6. base vs LoRA generation comparison

Do not call the trainer integrated until it survives the full ladder.
