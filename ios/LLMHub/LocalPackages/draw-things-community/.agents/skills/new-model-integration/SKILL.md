---
name: new-model-integration
description: Add a new image or video generative model to the Draw Things app / CLI with a compile-first, end-to-end workflow across SwiftDiffusion, tokenizer plumbing, text encoder, fixed encoder, UNet / DiT runtime, VAE, converter and quantizer tooling, and CLI validation.
---

# Model Integration Skill

Use this when adding a new image or video generative model, or a new model version, to the Draw Things app / CLI, especially when most of the surrounding components already exist and the task is mainly integration.

This skill is optimized for the current SwiftDiffusion / Draw Things layout:

- architecture builder in `Libraries/SwiftDiffusion/Sources/Models`
- weight-loading logic returned as `ModelWeightMapper` from model builders and helper blocks
- text path in `Libraries/SwiftDiffusion/Sources/TextEncoder.swift`
- fixed encoder path in `Libraries/SwiftDiffusion/Sources/UNetFixedEncoder.swift`
- runtime UNet / DiT path in `Libraries/SwiftDiffusion/Sources/Models/UNetProtocol.swift`
- VAE path in `Libraries/SwiftDiffusion/Sources/FirstStage.swift`

In this repo, `UNet` in names such as `UNetProtocol`, `UNetFixedEncoder`, and `UNetExtractConditions` is the legacy name for the main diffusion model integration boundary. The underlying architecture may be a DiT or another non-UNet model.

## Default Approach

- Get the new generative model compiling end-to-end before optimizing or slicing.
- Prefer minimal, explicit integration over generic abstractions.
- If a new generative model reuses an existing tokenizer, text encoder, or VAE, hook that up first instead of creating a new variant.
- For compile sweeps, it is acceptable to add `case .newModel: fatalError()` placeholders first, then fill them in.
- For bring-up, temporarily prefer strict loading such as `read(model: "model", strict: true, ...)` to surface missing or mismatched keys early. Do not ship that debugging behavior unintentionally.
- If text-conditioning boundary placement is ambiguous, choose the boundary that avoids passing large intermediate tensors across module boundaries.
- Prefer the repo's usual `UNetFixedEncoder` / `UNetProtocol` integration split as the long-term structure, but do not force that split into the first implementation if it makes bring-up materially harder.
- Keep the known-good unsplit / unsliced path as the release baseline until any later fixed split or cache path proves output parity.
- Do not introduce a model-specific config bag just to build one model graph; follow nearby builders and pass the needed parameters explicitly.
- If a reference implementation uses a different tensor layout from the app runtime, adapt layout at the model boundary first before debugging higher-level plumbing.
- Prefer the partner runtime path over converter or export harnesses when they disagree about runtime behavior.
- Treat text-conditioning contract bugs as first-class integration bugs. Wrong padding, masking, unconditional handling, prompt templates, or adapter boundary placement can preserve tensor shapes while destroying prompt adherence.
- Do not assume all runtime side inputs have the same rank. CFG splitting and extracted-condition logic must respect the actual tensor ranks.
- If a model removes an external mask or padding input, only synthesize an internal zero mask when the architecture really allows it.
- If a model checkpoint needs to flow through an existing asset slot to reach the right subsystem, prefer the existing pass-through pattern over inventing a new file-plumbing path.
- If partner implementations exist, prefer the one that matches the shipped model behavior over a loosely related upstream base repo.
- If a model supports multiple SDPA scaling modes, keep that as an enum-like runtime choice end to end instead of collapsing it to a boolean.
- If a tiled path uses spatial rotary embeddings, generate the full-image rotary tensor once and slice tiles from it. Do not regenerate tile-local rotary tensors unless the partner runtime does that explicitly.
- If a fixed split is introduced later, keep self-attention and cross-attention weight naming distinct and assert fixed-output count and ordering so silent misloads fail early.
- A first successful CLI run is not enough validation. Run a real sample, inspect the image, and pin `--seed` when comparing semantic fixes.
- Temporary debug prints, env toggles, and strict-load hooks are bring-up tools only. Remove them before handoff and validate the cleaned tree again.

## Workflow

### 1. Add the main model builder and weight mapping

Add the new diffusion builder in `Libraries/SwiftDiffusion/Sources/Models/<Model>.swift`.

Follow the existing structure used by nearby large-model integrations such as:

- `Libraries/SwiftDiffusion/Sources/Models/Flux2.swift`
- `Libraries/SwiftDiffusion/Sources/Models/QwenImage.swift`

For weight loading:

- build the model and return `ModelWeightMapper`
- keep mapper construction close to the builder/helper that owns the weights
- compose sub-mappers the same way neighboring model files do
- do not invent a separate mapping abstraction if a plain `ModelWeightMapper` closure is sufficient
- do not leave placeholder mapper closures once the integration starts running end to end
- if the export layout packs weights, map them explicitly with the same packing order the reference exporter uses

Compile goal for this step:

- the model builder exists
- the model can be instantiated
- the returned `ModelWeightMapper` can resolve checkpoint keys deterministically

### 2. Introduce the new `ModelVersion`

Add the enum case in `Libraries/SwiftDiffusion/Sources/Samplers/Sampler.swift`, for example:

```swift
case newModel = "new.model"
```

Then sweep the repo for switches over `ModelVersion`.

Preferred bring-up tactic:

- first add explicit `case .newModel: fatalError()` where behavior is not decided yet
- keep the compile surface honest
- replace placeholders only after the full switch surface is visible

High-value sweep targets:

- `Libraries/SwiftDiffusion/Sources/TextEncoder.swift`
- `Libraries/SwiftDiffusion/Sources/UNetFixedEncoder.swift`
- `Libraries/SwiftDiffusion/Sources/Models/UNetProtocol.swift`
- `Libraries/SwiftDiffusion/Sources/FirstStage.swift`
- `Libraries/ModelZoo/Sources/ModelZoo.swift`
- `Libraries/ModelZoo/Sources/ComputeUnits.swift`
- `Libraries/ModelOp/Sources/ModelImporter.swift`
- `Apps/ModelConverter/Converter.swift`
- `Apps/LoRAConverter/Converter.swift`
- `Apps/ModelQuantizer/Quantizer.swift`
- `Apps/DrawThings/Sources/Model/*`
- `Apps/DrawThings/Sources/Edit/EditWorkflow.swift`
- any tests or compatibility gates matching nearby model families

Useful search pattern:

```sh
rg -n "case \\.ltx2|case \\.flux2_4b|case \\.qwenImage|switch version|switch modelVersion" Libraries Apps -g '*.swift'
```

### 3. Hook tokenizer plumbing

Do not assume a single tokenizer stream.

If the model uses more than one text stream:

- keep each stream explicit instead of forcing it into one tokenizer abstraction
- verify which stream feeds which subsystem
- trim each stream to its real non-pad length before compiling dependent models when possible
- pad back only where the downstream model contract actually requires a fixed length
- do not assume converter-time fixed lengths are the runtime contract

Validate the runtime contract, not only the submodule math:

- compare real prompt-conditioned tensors, not only random-tensor parity harnesses
- confirm prompt, negative prompt, unconditional, and empty-prompt behavior at the actual runtime boundary
- verify masking and padding semantics before and after any adapter or projection stage
- if prompt adherence is broken but the model runs, inspect conditioning tensors before changing samplers or DiT math

Check and update the places that vend or select tokenizers:

- `Apps/DrawThings/Sources/Edit/Tokenizers.swift`
- `Apps/DrawThings/Sources/Edit/EditWorkflow.swift`
- `Apps/DrawThingsCLI/DrawThingsCLI.swift`
- `Apps/gRPCServerCLI/gRPCServerCLI.swift`
- `Libraries/DrawThingsSDK/Sources/DrawThingsSDK.swift`
- `Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift`

Goal:

- selecting the new model version produces the tokenizer behavior the model actually expects, in every runtime entry point

### 4. Hook text encoding and decide the `llm_adapter` boundary

Decision rule:

- put text projections / adapters in `TextEncoder.swift` if that avoids returning large hidden-state tensors across module boundaries
- put them in `UNetFixedEncoder.swift` only when the boundary is cleaner and tensor traffic remains reasonable
- make the decision based on tensor movement and reuse, not style

Implementation target:

- `Libraries/SwiftDiffusion/Sources/TextEncoder.swift`

Bring-up advice:

- temporarily `debugPrint` relevant tensors around the new path
- confirm shapes are what the diffusion model expects
- confirm the outputs are finite and there is no NaN contamination
- if a partner runtime exists, compare the actual conditioning tensor contract for one real prompt rather than relying only on export-script parity
- remove noisy debugging once the path is validated

### 5. Integrate `UNetFixedEncoder` and `UNetProtocol`

After text encoding is stable, hook the new generative model into:

- `Libraries/SwiftDiffusion/Sources/UNetFixedEncoder.swift`
- `Libraries/SwiftDiffusion/Sources/Models/UNetProtocol.swift`

Default rule for a first integration:

- prefer the eventual `UNetFixedEncoder` / `UNetProtocol` integration split structurally
- do not slice the model yet
- do not extract adaln or KV-precompute paths yet
- usually keep the fixed part unextracted and wire the full model through so end-to-end generation can run first
- only pull work into `UNetFixedEncoder` early if it is already simple, obvious, and low-risk

Only add slicing later if profiling or architecture constraints require it.

- make the unsliced path work first
- if a later fixed / sliced split changes semantic output, keep the unsplit path as the release baseline until parity is proven
- do not leave a speculative fixed split half-wired into the normal runtime path just because it compiles
- confirm the runtime model input order matches the actual `UNetProtocol` call contract
- if CFG splitting is enabled, do not assume every side input is rank-3
- if `UNetExtractConditions` is used, only slice tensors that are actually timestep-major extracted conditions
- do not slice or index batched text context by sampler step unless the data is explicitly laid out that way
- if technical execution succeeds but samples remain noise, compare the attention and residual-dtype path against the partner implementation before changing higher-level app plumbing
- if a partner implementation keeps the residual stream in higher precision than attention / FFN, mirror that split if the lower-precision path produces NaNs or semantic collapse
- if a split fixed path is added, keep the unsplit path as the output-parity baseline until the split path is proven on real images
- if the split path precomputes cross-attention KV or modulation terms, assert the expected number of returned tensors and keep their ordering explicit
- if the model has repeated self-attention and cross-attention blocks, do not let their checkpoint keys collide by sharing a naming pattern that only differs by enumeration position
- if tiled diffusion is supported, feed full-image rotary into the shared slice path instead of rebuilding tile-local rotary tensors

### 6. Sweep converter and quantizer tools

If the model is user-facing or importable, extend the non-runtime tools too:

- `Apps/ModelConverter/Converter.swift`
- `Apps/LoRAConverter/Converter.swift`
- `Apps/ModelQuantizer/Quantizer.swift`

Common integration miss:

- runtime code builds, but converter or quantizer switches are non-exhaustive
- tool help text omits the new model version
- quantization policy silently uses the wrong fallback because the new model family is unhandled
- checkpoint key remap shims are kept after the converted checkpoint has been regenerated with correct names

### 7. Hook VAE through the existing first-stage path

If the new model reuses an existing latent contract:

- reuse the existing first-stage path in `Libraries/SwiftDiffusion/Sources/FirstStage.swift`
- avoid creating a new VAE branch unless the latent contract actually proves incompatible

Goal:

- encode/decode works by routing through the minimal existing first-stage behavior the model is compatible with

### 8. Validate end-to-end before cleanup

Preferred validation order:

1. compile the affected diffusion library target
2. compile the app or CLI path that exercises the model
3. run an end-to-end generation through `bazel run //Apps:DrawThingsCLI -- ...`

For this class of task, start with:

```sh
bazel build //Libraries/SwiftDiffusion:SwiftDiffusion
bazel build //Apps:DrawThingsCLI
bazel run //Apps:DrawThingsCLI -- --help
```

If the new model is already selectable from the CLI, run an actual generation with the model assets present.

Suggested validation sequence:

```sh
bazel build //Libraries/SwiftDiffusion:SwiftDiffusion
bazel build //Apps:DrawThingsCLI
bazel build //Apps/DrawThings:DrawThings --ios_multi_cpus=arm64
bazel run //Apps:DrawThingsCLI -- generate \
  --models-dir <models-dir> \
  -m <model-file> \
  -p "<prompt>" \
  --steps <steps> \
  --cfg <cfg> \
  --seed <seed> \
  --config-json '<json-if-needed>' \
  --width <width> \
  --height <height> \
  --no-download-missing \
  -o /tmp/model-generate.png
```

Illustrative example using Anima:

- treat this as an example pattern, not the required command shape for every model
- change model file, prompt, image size, sampler knobs, and `--config-json` to match the model family you are integrating

```sh
bazel run //Apps:DrawThingsCLI -- generate \
  --models-dir <models-dir> \
  -m anima_preview_3_f16.ckpt \
  -p "a red apple on a wooden table, studio photograph" \
  --steps 20 \
  --cfg 4 \
  --seed 7 \
  --config-json '{"shift":3}' \
  --width 1024 \
  --height 1024 \
  --no-download-missing \
  -o /tmp/drawthings-generate.png
```

Concrete smoke-test example:

```sh
bazel run //Apps:DrawThingsCLI -- generate \
  --models-dir <models-dir> \
  -m <model-file> \
  -p "<simple prompt>" \
  --steps 1 \
  --cfg 1 \
  --width 512 \
  --height 512 \
  --no-download-missing \
  -o /tmp/drawthings-generate.png
```

Note:

- end-to-end CLI execution may require permission to use the GPU
- prefer `bazel run //Apps:DrawThingsCLI -- ...` directly over `swift run` in this repo
- if the CLI does not expose a runtime knob directly, pass the value through `--config-json`
- if the model is flow-matching or uses a non-default objective/discretization, verify those `ModelZoo` values explicitly instead of assuming the nearest existing model is correct
- when GPU approval is needed for repeated generation comparisons, ask once for a fixed command shape with:
  - a fixed output image path
  - a fixed log path
- after each run completes, move those generic files to a run-specific name yourself to preserve progress history
- this keeps the approved command prefix stable across iterations and avoids re-asking for every output filename change
- after the model is working, remove temporary model-specific env toggles and debug prints before handoff, then rerun:
  - `bazel build //Libraries/SwiftDiffusion:SwiftDiffusion`
  - `bazel build //Apps/DrawThings:DrawThings --ios_multi_cpus=arm64`
  - one real `bazel run //Apps:DrawThingsCLI -- generate ...` sample
- visually inspect the generated image itself; successful execution is necessary but not sufficient

If end-to-end execution is blocked, stop at the highest verified layer and record the blocker precisely.

## Bring-Up Heuristics

- Compile-first beats perfect-first.
- Reuse neighboring model patterns instead of inventing a new framework.
- Add the smallest amount of model-specific code that gets the path working.
- Keep new type handling explicit; do not silently fall through to another model family unless the tensor contract is truly identical.
- Prefer short-lived debugging hooks over speculative refactors.

## Common Failure Modes

- missing `ModelVersion` switch cases outside SwiftDiffusion
- converter / quantizer / LoRAConverter builds break because the new model version was not added there
- tokenizer selected correctly in one runtime but not others
- text encoder outputs the right dtype but wrong sequence length
- text encoder or adapter parity passes on synthetic tensors, but the real runtime conditioning contract is still wrong
- prompt padding or masking semantics copied from the wrong reference path
- empty or unconditional prompt handling differs from the partner runtime
- successful runtime execution but semantically broken image output
- wrong runtime input order between `UNetProtocol` and the model builder
- reference layout copied directly even though app runtime uses a different latent layout
- CFG or extracted-condition code assuming all side inputs have the same rank or timestep layout
- sampler objective / discretization / shift mismatched with the model family
- model-specific attention scaling modes collapsed to a boolean, changing SDPA semantics
- residual precision too low around attention / FFN, causing NaNs or semantic collapse
- `llm_adapter` placed on the wrong side of the boundary, causing oversized tensor traffic
- checkpoint key mismatches hidden by non-strict loading
- checkpoint key collisions between self-attention and cross-attention submodules
- outdated converted checkpoints patched with ad-hoc key remaps instead of regenerating the checkpoint with the right names
- VAE branch accidentally copied instead of reusing an existing compatible first-stage path
- temporary model-specific env toggles or debug files left in the shipping path
- optimized fixed split compiles but changes outputs relative to the unsplit baseline
- tiled diffusion regenerates tile-local rotary instead of slicing from the full-image rotary tensor

## Checklist

- the model builder file exists and builds
- the main model path returns `ModelWeightMapper` in the same style as nearby model integrations
- the new `ModelVersion` exists
- compile sweeps for switches are complete
- converter / quantizer / LoRAConverter switches are covered if the model uses those tools
- tokenizer selection matches the model contract in every runtime entry point
- text encoder path is hooked and tensor outputs are finite
- text-conditioning behavior is validated on at least one real prompt, not only synthetic parity tensors
- `UNetFixedEncoder` and `UNetProtocol` run without slicing
- tiled rotary uses full-image generation plus tile slicing when the architecture needs spatial RoPE
- `FirstStage` reuses an existing compatible VAE path when possible
- CLI or app path can at least reach model construction
- strict loading debug path has been removed or intentionally gated before shipping
- temporary model-specific env toggles / debug files are removed before handoff
- if a fixed split was explored, the shipping path is still the known-good unsplit baseline unless parity is proven
- if a fixed split ships, fixed-output count/order and weight-name disambiguation are asserted explicitly
