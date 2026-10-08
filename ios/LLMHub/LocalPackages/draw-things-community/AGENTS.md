# Session Learnings

- Do not use app-name prefixes such as `LocalCode` for ordinary Swift source filenames, types, or related code abstractions. Preserve intentional CLI entry-point names (including `LocalCodeCLI.swift` and `LocalCodeMain.swift`), app/build/module names, and configuration filenames (schemes, entitlements, and provisioning profiles).

## Linear Main Branch History
- Keep `main` strictly linear. Never create a merge commit on `main`; only fast-forward updates are allowed.
- While on `main`, do not run an ordinary `git merge` or a merge-producing `git pull`. Use `git pull --rebase`, or fetch and then rebase / cherry-pick local commits onto the latest `origin/main`.
- If local `main` and `origin/main` have diverged, stop and report the divergence. Do not resolve it by merging `origin/main` into `main`.
- Do not fast-forward a feature branch into `main` when that branch contains merge commits. Rebase it or cherry-pick the required non-merge commits onto the latest `origin/main` first.
- Before pushing `main`, fetch `origin/main`, verify that `git merge-base --is-ancestor origin/main HEAD` succeeds, and verify that `git rev-list --merges --count origin/main..HEAD` prints `0`.
- Push `main` without force. Rewriting or force-pushing `main` requires explicit user authorization.

## Downloader Naming + Compatibility Policy
- In this repo, prefer one canonical name for a behavior or mode instead of compatibility aliases.
- For the downloader split, use `SegmentedResumableDownloaderBackend` vs `URLSessionDownloadTaskResumableDownloaderBackend`.
- For `ResumableDownloader`, select backend via an explicit constructor parameter with a default instead of process-global environment variables.
- Do not keep compatibility flags or alternate spellings such as `legacy` when renaming internal APIs or test harness flags unless persisted data or an external contract requires it.
- If compatibility is required, constrain it to persisted data / externally consumed formats rather than duplicating internal source-of-truth names in code paths.

## FrameCompression (VideoToolbox + s4nnc)
- Prefer a synchronous public API for single-frame artifact injection:
  - `FrameCompression.applyCompressionArtifacts(to:codec:quality:) throws -> Tensor<FloatType>`
  - Keep codec selection as an enum parameter (`h264`, `h265`, `jpeg`) instead of separate functions.
- For VideoToolbox single-frame encode/decode, avoid semaphores:
  - Encode with callback capture, then block using `VTCompressionSessionCompleteFrames(...)`.
  - Decode with callback capture, then block using `VTDecompressionSessionWaitForAsynchronousFrames(...)`.
- Validate tensor contract strictly before compression:
  - `.CPU`, `.NHWC(1, height, width, 3)`, quality in `0...100`, and even width/height for H264/H265.
- Keep tensor/pixel conversions explicit and stable:
  - Tensor range `[-1, 1]` <-> `CVPixelBuffer` BGRA8.

## Integration + Verification
- Keep artifact injection opportunistic at app call-sites:
  - use `try?` and fall back to original tensor if compression fails.
- Verify behavior with Bazel CLI tests using measurable assertions:
  - shape preserved, output range remains in `[-1, 1]`, non-zero artifacts, lower quality increases error.
- After API refactors, validate both:
  - focused target (`bazel run //Libraries/FrameCompression:FrameCompressionTestCLI`)
  - integration target (`bazel build //Apps/DrawThings:DrawThings`).

## LTX/PyAV Parameter Parity Notes
- To better match LTX `media_io.py` behavior (x264 CRF flow), prefer constant-quality style over explicit bitrate limits in VideoToolbox:
  - do not set `AverageBitRate` / `DataRateLimits`.
  - set `ExpectedFrameRate = 1`, `MaxKeyFrameInterval = 1`, `AllowFrameReordering = false`.
  - set `RealTime = false` and, when supported, `PrioritizeEncodingSpeedOverQuality = true` to approximate `preset=veryfast`.
- Quality mapping now follows CRF semantics:
  - `crf = round((100 - quality) / 100 * 51)`.
  - inverse: `quality ≈ 100 - (crf / 51 * 100)`.
  - example: `crf = 29` corresponds to quality around `43` to `44`.

## LTX2 Video Decoder Refactor Notes
- When aligning `LTX2VideoDecoderCausal3D` with the encoder / LTX-2.3 construction API, switch the public constructor to `layers: [(channels: Int, numRepeat: Int, stride: (Int, Int, Int))]`, but keep the decoder implementation structurally close to the existing code path.
- For decoder stride handling, mirror the encoder branching style instead of introducing generic stride-product helper logic:
  - first branch on `layer.stride.0 == 1`
  - then `layer.stride.1 == 1 && layer.stride.2 == 1`
  - otherwise handle the mixed temporal+spatial upsample path
- In this decoder path, if residual replication needs `Concat`, use the explicit op instance form so memory-saving flags are preserved:
  - `let concat = Concat(axis: n)`
  - `concat.flags = [.disableOpt]`
  - `residual = concat(Array(repeating: residual, count: k))`
  - do not assume `Concat(axis: n, flags: ...)` is available here.
- During signature refactors, prefer writing the parameter list on one line first and let the formatter decide the final line wrapping later.
- Validate decoder/API refactors with:
  - `bazel build //Libraries/SwiftDiffusion:SwiftDiffusion`

## Utility Function Style
- Prefer synchronous utility functions whenever possible.
- If an operation is inherently callback-driven and cannot be made truly synchronous with native APIs, expose an async completion-handler API directly.
- Do not introduce `DispatchQueue`, `DispatchGroup`, or `DispatchSemaphore` in utility functions to simulate sync/async behavior unless absolutely necessary.
- Keep utility functions thread-agnostic and leave threading/queue decisions to upper-level call sites.

## Inline Logic vs Helpers
- Prefer in-place logic for small, local behavior changes, especially in UI workflows and short persistence paths.
- Do not introduce one-off private helpers for a few lines of straightforward code unless the logic is reused or the extracted name captures a real domain concept.
- Good helper candidates are shared policy, non-trivial data transformation, or logic reused across multiple call sites.
- Poor helper candidates are short project-dictionary reads/writes, one-off alert assembly, or tiny control-flow wrappers added during opportunistic cleanup.

## Swift Error Style
- Do not introduce new `NSError` values in Swift-first code paths.
- Prefer dedicated `enum Error` / `LocalizedError` types, including small file-local enums for one-off errors.

## Instruction Count Utilities (s4nnc Models)
- For model compute-estimation helpers, prefer exact, boring names derived from the original builder:
  - use `{ORIGINAL_FUNC}InstructionCount` (for example `Flux1InstructionCount`, `Flux1FixedInstructionCount`).
  - avoid creative wrapper names when the goal is to mirror existing model-builder functions.
- Put shared counting primitives in a separate reusable file so multiple models can reuse them:
  - `DenseInstructionCount(...)`
  - `ConvolutionInstructionCount(...)`
  - `ScaledDotProductAttentionInstructionCount(...)` (use full name, not `SDPA` abbreviation in the function name).
- Keep per-model instruction count functions separate by builder path (for example main denoiser vs fixed encoder), instead of forcing a combined API.
- For the simplified estimator, return a single total `Int` unless a detailed breakdown is explicitly requested.
- Validate Swift syntax/type correctness with Bazel on the affected library target after refactors:
  - `bazel build //Libraries/SwiftDiffusion:SwiftDiffusion`

## Flux2 LoRA Layer Indexing
- In `Flux2` LoRA single blocks, do not assume one shared LoRA layer index for all projections if the exported layer naming hierarchy differs.
- `LoRASingleTransformerBlock` needs separate indices for:
  - attention/output projections (`x_q`, `x_k`, `x_v`, `x_o`) using the global single-block layer index (`i + layers.0`)
  - MLP projections (`x_w1`, `x_w2`, `x_w3`) using the single-block-local MLP index (`i`)
- Reason: the single-block MLP layer names start from `0`, so reusing the global index causes LoRA weight lookup mismatches.
- When reviewing LoRA parity, check not only operator replacement (`LoRADense`/`LoRAConvolution`) but also index mapping semantics against the target naming scheme.

## LTX2 Stateful TAE Integration (ImageConverter)
- `ImageConverter.imagesWithTAESD(fromLatent:version:)` for `.ltx2` should use the **stateful Core ML API** (`prediction(from:)`) rather than the batch API used by image TAEs.
- LTX2 stateful decoder activation tensors are fixed and known in app integration:
  - inputs: `act_0 ... act_8`
  - outputs: `act_0_out ... act_8_out`
  - no parsing/sorting of activation names is needed.
- Keep the app-side LTX2 TAE path **FP16-only**:
  - require FP16 `latent`, `image`, and `act_*` tensors; return `[]` on mismatch instead of adding FP32 handling.
- Preferred latent chunk construction style (match image path ergonomics):
  - build a padded CPU tensor with explicit zero init using `Array(repeating: 0, ...)`
  - copy source frames via tensor slicing/subscript assignment
  - reshape to rank-5 `NTHWC` (`[1, chunkT, H, W, C]`) before `MLShapedArray`
  - convert with `MLMultiArray(MLShapedArray(...))`
- LTX2 stateful output semantics:
  - outputs are raw chunk frames; apply a **global trim** after concatenation
  - trim count is `tUpscale - 1` (preview `dtu000` naturally trims `0`)
- Verify compileability of the integration path with:
  - `bazel build Libraries/LocalImageGenerator --ios_multi_cpus=arm64`

## WAN2.1 Stateful TAE + INSDIFF (ImageConverter)
- `wan_2.1_tae.zip` packaging is intentionally deduped against `qwenimage_tae.zip`:
  - ship `wan_2.1_tae/weight.bin.insdiff` instead of full `weight.bin`
  - reconstruct runtime `weight.bin` from qwen base + insdiff patch.
- Keep `InsdiffPatcher` CoreML-independent and data-centric:
  - API should operate on `Data` + `Data` -> `Data`
  - no coupling to model loading or CoreML feature logic.
- For WAN2.1 unzip flow in `ImageConverter`:
  - unzip `wan_2.1_tae.zip` once (cannot selectively skip entries with `FileManager.unzipItem`)
  - read `weight.bin.insdiff` from the unzipped folder
  - extract qwen `weight.bin` from `qwenimage_tae.zip` via `Archive`
  - apply patch and write reconstructed `wan_2.1_tae/weight.bin`
  - remove `weight.bin.insdiff` after successful reconstruction.
- Current WAN2.1 tiny decoder package targets are:
  - `512`, `768`, `1024`, `1280` only.
- WAN2.1 TAESD decode path should use the stateful branch (`isVideo = true`):
  - models expose fixed activation tensors `act_0 ... act_8` and `act_0_out ... act_8_out`
  - output is `NTHWC` (`image` has temporal dim), matching chunk/state decode flow.

## SwiftPM Support (gRPCServerCLI + Generated Sources)
- The `swift-package-build` workflow validates the SwiftPM path by:
  - running `./Scripts/swift-package/generate_binary_resources.sh`
  - running `./Scripts/swift-package/generate_datamodels.sh`
  - building `swift build --target gRPCServerCLI`
- Keep `Package.swift` in sync with Bazel module splits for `Libraries/SwiftDiffusion`:
  - when `Libraries/SwiftDiffusion/BUILD` splits out `Sources/Mappings/**/*.swift` into module `DiffusionMappings`, mirror that in SPM by adding a `DiffusionMappings` target in `Package.swift`
  - make `Diffusion` depend on `DiffusionMappings`
  - exclude `Mappings` from the `Diffusion` target sources to avoid duplicate compilation
- Symptom of missing SPM mirror for the mappings split:
  - `swift build --target gRPCServerCLI` fails with `no such module 'DiffusionMappings'` from `Libraries/SwiftDiffusion/Sources/...`
- `Libraries/DataModels/PreGeneratedSPM` output can differ from committed files if generated code is copied without formatting:
  - `Scripts/swift-package/generate_datamodels.sh` should run `swift-format` on generated `*.swift` files after copying
  - use repo config `.swift-format.json`
  - use the repo's Bazel formatter target `@SwiftFormat//:swift-format` (same formatting path as the pre-commit hook)
- `Libraries/BinaryResources/GeneratedC` is SPM-generated too, but may already be byte-for-byte up to date when regenerated; check diffs separately from `PreGeneratedSPM`.

## LTX2 LoRA Complementary Models + Integration
- When adding LoRA support for an LTX2 model builder in `Libraries/SwiftDiffusion/Sources/Models/LTX2.swift`, mirror the `Flux2.swift` pattern exactly:
  - public APIs should be `LoRA{Original}` (`LoRALTX2`, `LoRALTX2Fixed`, `LoRAEmbedding1DConnector`)
  - private helpers should also use `LoRA{Helper}` names (for example `LoRALTX2TransformerBlock`, `LoRABasicTransformerBlock1D`)
- Preserve original weight mapping keys and behavior; the LoRA version should primarily swap `Dense` / `Convolution` with `LoRADense` / `LoRAConvolution`.
- For repeated LTX2 transformer blocks, pass explicit LoRA `layerIndex` into helper functions so exported LoRA keys are stable and non-colliding across layers.
- LTX2 LoRA integration is not just the main denoiser:
  - `UNetProtocol.swift` `.ltx2` branch needs separate-LoRA gating and should switch builder between `LTX2` and `LoRALTX2`
  - `UNetFixedEncoder.swift` `.ltx2` branch must cover **both** connectors (`videoConnector` / `audioConnector`) and `LTX2Fixed`
  - `TextEncoder.encodeLTX2(...)` also needs LoRA support for `text_feature_extractor` (the post-text-model projection), not just the Gemma text model
- In `UNetFixedEncoder.swift`, canonicalize/dedupe `lora` once near the top of `encode(...)` and reuse that list in every branch:
  - do not re-merge/re-dedupe LoRA files inside individual model branches (including `.ltx2`)
- For LTX2 separate-LoRA loading in `UNetFixedEncoder.swift`, use per-submodel identity layer mappings that match the component depth:
  - connectors (`Embedding1DConnector`): `0..<2`
  - `LTX2Fixed` transformer blocks: `0..<48`
- When adding separate-LoRA load paths, keep weights-cache logic aligned with existing patterns:
  - use `concatenateLoRA(...)` only for separate execution path
  - use `mergeLoRA(...)` for merged path
  - preserve existing non-LoRA `store.read(...)` behavior and cache attach/detach flow
- Validate all LTX2 LoRA integration changes with:
  - `bazel build //Libraries/SwiftDiffusion:SwiftDiffusion`

## SamplerType Additions (TCDTrailing)
- For FlatBuffers sampler enums, append new values **at the end** in both schemas for backward compatibility:
  - `Libraries/DataModels/Sources/config.fbs`
  - `Libraries/History/Sources/tensor_history.fbs`
- Do not hand-edit `Libraries/DataModels/PreGeneratedSPM/*`; regenerate via:
  - `./Scripts/swift-package/generate_datamodels.sh`
- Generated Swift enum case naming may not preserve acronym casing exactly:
  - `TCDTrailing` in `.fbs` becomes `.tCDTrailing` in generated Swift (similar to `.dDIMTrailing`)
  - use the generated case spelling everywhere in Swift switch/mapping logic.
- Adding a sampler variant requires cross-layer updates, not just enum/schema:
  - DataModels parser/display (`Defaults.swift`)
  - History/DataModels bridging (`ImageHistoryManager.swift`)
  - CLI/display (`Libraries/Invocation/Sources/Parameters.swift`)
  - scripting constants (`Libraries/Scripting/Sources/SharedScript.swift`)
  - runtime sampler construction (`Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift`)
  - compatibility gates (`Libraries/ModelZoo/Sources/ModelZoo.swift`, `SettingsWorkflow.swift`)
  - UI/metadata fields gated on sampler type (for TCD gamma): settings sections, edit summary, serialization/export paths.
- For `TCDTrailing`, behavior should mirror other trailing samplers:
  - use `TCDSampler` with `Denoiser.LinearDiscretization(..., timestepSpacing: .trailing)`.
  - keep regular `.TCD` path unchanged.
- Validation checklist for sampler additions:
  - `bazel build //Libraries/LocalImageGenerator --ios_multi_cpus=arm64`
  - `bazel build //Apps/DrawThings:DrawThings`

## TextHistoryManager Edge Cases (NSRange + Lineage)
- `NSRange` must be treated as UTF-16 indexed when used with `NSString` APIs:
  - for full-text replacements, use `.utf16.count` (not `.count`) at call-sites.
  - affected app-side text history call-sites are in `EditWorkflow.swift`.
- Keep a shared range validator in `TextHistoryManager`:
  - use `private static func normalizedRange(...) -> NSRange?` for bounds checking only.
  - do not include migration/rewriting behavior in the validator; validation only.
- In `setTextHistory`, if a persisted modification range is invalid:
  - skip applying that modification and continue replay safely (avoid crash from invalid ranges).
- In `pushChange`, validate range before any fork/lineage/history state mutation:
  - use explicit `switch textType` validation branches against the current text.
- For non-sacred seeks, persist the resolved lineage:
  - set `project.dictionary["text_seek_to_lineage"]` from `self.lineage` (resolved target), not requested alias lineage.
- During lineage remap/fork caching:
  - key temporary `nodeLineageCache` entries by the remapped lineage (`newLineage`) consistently for insert/cleanup.
- Focused test strategy for these edge cases:
  - use a dedicated `swift_test` target (`//Libraries/History:TextHistoryManagerTests`) backed by a text-only library target (`TextHistory`) to avoid unrelated UIKit dependencies from `History`.
  - include regressions for UTF-16 replacement behavior, invalid persisted range replay safety, segmentation sanity, and unsacred-lineage seek persistence.
- Verification checklist after refactors:
  - `bazel test //Libraries/History:TextHistoryManagerTests`
  - `bazel build //Apps/DrawThings:DrawThings`

## ImageHistoryManager Edge Cases (Lineage Remap + Seek Persistence)
- In non-sacred fork remap paths, cache keys must use the exact remapped lineage per node:
  - compute `updatedLineage = isAncestor ? newLineage + 1 : newLineage`
  - use that value consistently for `nodeLineageCache`, `imageDataCache`, and `shuffleDataCache` insert/cleanup keys.
- In async cleanup, recompute remapped lineage with the same condition used during remap:
  - avoid cleaning with only `newLineage`, which can miss `newLineage + 1` ancestor entries.
- After deletion snapshot refresh, record lineage max time with the snapshot lineage:
  - `maxLogicalTimeForLineage[imageHistory.lineage] = maxLogicalTime`
  - avoid writing with stale `self.lineage` before seek/set updates.
- For explicit lineage seeks, persist the resolved lineage:
  - when a lineage is requested, write `image_seek_to_lineage` from `self.lineage` after seek resolution.
  - use explicit `if let` branches and document why resolved lineage is persisted.
- Verification checklist after refactors:
  - `bazel test //Libraries/History:TextHistoryManagerTests`
  - `bazel build //Apps/DrawThings:DrawThings`

## History Swift Testability (UIKit Gating)
- `Libraries/History/Sources/ImageHistoryManager.swift` should not hard-require UIKit for core history logic:
  - gate UI image import with `#if canImport(UIKit)`.
  - in non-UIKit builds, provide a minimal fallback `UIImage` type so core history APIs remain buildable.
- Keep non-UIKit builds functional by gating preview thumbnail encode/decode paths:
  - only call `jpegData`, `UIImage(data:)`, and bitmap downsampling when UIKit is available.
  - return `nil` from preview fetch helpers on non-UIKit builds.
- When removing UIKit as an implicit dependency, add explicit imports needed by core logic:
  - `Foundation` for `Data`, `Date`, `JSONEncoder`/`JSONDecoder`.
  - `Dispatch` for `DispatchQueue` and `dispatchPrecondition`.
- Use a dedicated Image history test target once History is non-UIKit buildable:
  - `//Libraries/History:ImageHistoryManagerTests` (separate from text-only tests).
  - include regression for unsacred seek persistence (`image_seek_to_lineage` resolved lineage behavior).

## Tokenizers Lazy Init + LoRA Training Dependency
- Keep `Tokenizers` in `Apps/DrawThings/Sources` (app-only dependency), not in `LocalImageGenerator`.
- `Tokenizers` should be a `struct` with static tokenizer storage; rely on Swift static lazy once-semantics for base tokenizer initialization (`v1` / `v2` / `xl`) instead of locking.
- Do not keep `InitializationPolicy`; remove that indirection and keep initialization behavior explicit at call-sites.
- Use `os_unfair_lock` only for textual inversion state:
  - hold lock when constructing/modifying textual inversion arrays.
  - hold lock when returning textual-inversion-aware tokenizer vars so `textualInversions` assignment is consistent.
- Keep `gRPCServerCLI` tokenizer initialization call-sites unchanged (still eager by direct access); defer usage primarily in `EditWorkflow`.
- For `LoRATrainingDependency`, borrow the `LocalImageGenerator` pattern:
  - store tokenizer providers as private closure-backed factories.
  - expose computed tokenizer properties that call the factories.
  - accept tokenizer inputs via `@autoclosure @escaping` initializer parameters.
- At `EditWorkflow` call-sites, snapshot tokenizer values into local variables before passing into escaping autoclosure-based dependency initializers to avoid escaping-capture issues around `self`.

## DrawThingsCLI Migration (ArgumentParser + Legacy Cleanup)
- The canonical local-inference CLI now lives in `Apps/DrawThingsCLI/DrawThingsCLI.swift` and is built as:
  - Bazel: `//Apps:DrawThingsCLI`
  - SwiftPM product: `draw-things-cli` (legacy product name `DrawThingsCLI` removed).
- `generate` model resolution expectations:
  - `--model` is required; if missing/unresolved, print model help/list.
  - model input supports file id, human-readable model name, `hf://owner/repo`, and Hugging Face URL/repo style via `ModelZoo`.
- `--models-dir` is optional and should resolve in this order:
  - explicit `--models-dir`
  - `DRAWTHINGS_MODELS_DIR`
  - executable-adjacent `Models/` (if exists)
  - fallback `~/Documents/Models` (auto-create directory if missing).
- `generate` img2img syntax support:
  - `--image` with aliases `--init-image` and `--input-image` (conflict if multiple values differ).
  - `--strength` in `[0, 1]`.
  - decode input image with the same training-tensor loading path for consistent normalization.
- LoRA train config policy:
  - use `LoRATrainingConfiguration` JSON directly via `--config-json`.
  - removed legacy `--config` / `--train-config`.
  - merge JSON onto `LoRATrainingConfiguration.default` using snake_case decoding semantics.
  - parse `memory_saver` and `weights_memory_management` explicitly from raw JSON dictionary because they are not fields on `LoRATrainingConfiguration`.
- Keep train CLI surface intentionally small; advanced dataset filtering/repeat knobs were removed from CLI and should be handled by preprocessed datasets or JSON workflows.
  - removed flags include `--default-caption`, `--repeats`, `--include`, `--exclude`, `--minimum-mp`, `--maximum-mp`, `--trigger-word`, `--unet-lr-lower`, `--unet-lr-upper`, `--text-learning-rate`, `--max-text-length`.
  - `--learning-rate` accepts either single value (`1e-4`) or range (`[5e-5,1e-4]`, `5e-5,1e-4`, or `5e-5:1e-4`).
- Legacy cleanup completed:
  - removed `Apps/DrawThings/Sources/CLI/*`.
  - removed `Apps/DrawThings/Sources/Trainer/main.swift`.
  - removed `CLILib` target and related legacy references from `Apps/DrawThings/BUILD`.
- Important boundary: do **not** remove `Libraries/Invocation/Sources/Parameters.swift` as part of legacy CLI cleanup alone.
  - `Parameters` is still required by active HTTP API flow in:
    - `Libraries/HTTPAPIServer/Sources/HTTPAPIServer.swift`
    - `Libraries/HTTPAPIServer/Sources/ServerModels.swift`
    - `Apps/DrawThings/Sources/Edit/EditWorkflow.swift` delegate implementation.
- Verification checklist for CLI migration / rebases:
  - `swift run draw-things-cli --help`
  - `bazel build //Apps:DrawThingsCLI`
  - `bazel build //Apps/DrawThings:DrawThings`

## Cloudflare user-accounts Deploy Tags
- For `Cloudflare/user-accounts` deployment workflow, tag format, and other worker-specific instructions, use `Cloudflare/user-accounts/AGENTS.md` as the source of truth.

## Edit Video UI (Playback Slider + Audio Mute)
- For the clip audio button in both `EditViewController.swift` and `EditViewControllerForiPad.swift`:
  - keep the button styled like `clipButton` and anchored to `zoomButton` with trailing `-16`.
  - keep clip slider trailing fallback to `zoomButton` at lower priority, and use one stored `clipSliderTrailingToAudioConstraint` for the `slider -> audio button` path (8pt spacing).
  - toggle only `clipSliderTrailingToAudioConstraint?.isActive` based on audio-button visibility; avoid managing two stored constraints.
- Prefer explicit inline visibility updates for this UI path:
  - update audio-button visibility/constraint activation directly in `isVideoAudioEnabled.didSet` and `isVideo` setter.
  - set explicit initial `clipAudioButton.isHidden` after constraints setup.
- Mute state source of truth:
  - expose `var isAudioMuted: Bool { get set }` on `EditViewControllable`.
  - in both view controllers, map `isAudioMuted` to `clipAudioButton.isSelected`.
  - do not toggle `clipAudioButton.isSelected` in button tap handlers; route taps to delegate (`didTapMuteVideo` / `didTapUnmuteVideo`) based on current `isAudioMuted`.
  - avoid a second mute flag in `EditWorkflow`; use `viewController.isAudioMuted` as the single source.
- In `EditWorkflow`, keep audio-play helper name as `playAudioForVideo(...)`:
  - guard playback with `!viewController.isAudioMuted`.
  - keep a note that although helper takes `clipData`, `imageHistoryManager.popAudioClip` reads from current `imageHistoryManager.clipData`.
  - on mute: set `viewController.isAudioMuted = true` and stop current audio.
  - on unmute: set `viewController.isAudioMuted = false`; if video is playing, resume audio from current animator frame.
- Verification checklist after this UI/audio wiring:
  - `bazel build //Apps/DrawThings:DrawThings --ios_multi_cpus=arm64`

## ChatTools iPad {HIDDEN} Button
- `draftButton` in `ChatToolsController` should be always added to the hierarchy; show/hide via `is{HIDDEN}Available`.
- Keep two trailing paths for each bottom tools view (`edit/selection/eraser/paint`):
  - fallback: `trailing = generateView.leading` (lower priority),
  - draft path: `trailing = draftButton.leading - 16` (stored constraint, toggled active when available).
- For spacing, use `draftButton.trailing = generateView.leading + 10`:
  - because `generateButtonView.leading = generateView.leading + 16`, this yields effective `6pt` spacing between `draftButton` and the visible generate button.
- Constraint ordering matters:
  - if `draftButton` constraints reference `generateButtonView` before `generateButtonView` is in a view hierarchy with a common ancestor, Auto Layout can fail at runtime.
  - add `generateButtonView` before creating `draftButton` constraints that reference it.
- For enabled/selected visual state, use stored state on controller:
  - `is{HIDDEN}Enabled` is the source of truth and updates `draftButton.isSelected`.
  - button action should only toggle `is{HIDDEN}Enabled`.
- Expansion animation pitfall:
  - driving `configuration.title` inside `configurationUpdateHandler` can prevent the enable (expand) animation from animating as expected.
  - set button title outside the update handler (from `is{HIDDEN}Enabled`), then animate `view.layoutIfNeeded()`.

## {HIDDEN} Interactive Generation UI + State
- For adjacent optional controls in this edit UI, prefer keeping views in the hierarchy and switching visibility / operability over inserting and removing subviews dynamically.
  - this keeps Auto Layout stable and makes future transitions much easier.
- For alternate layout paths, prefer one real stored constraint plus a fallback constraint at lower priority.
  - use priority to let Auto Layout choose the right path instead of storing and toggling two competing constraints when only one needs explicit ownership.
- Constraint creation order matters when a new view depends on an existing sibling:
  - add the referenced sibling to a common ancestor first, then create constraints that point to it.
  - otherwise runtime Auto Layout failures are easy to introduce.
- For stateful buttons whose width/title animate on selection:
  - keep one stored property as the source of truth (`is{HIDDEN}Enabled`, `isAudioMuted`, etc.).
  - update `UIButton` visual state from that property.
  - avoid mutating the button state directly inside tap handlers when the state is meant to come from controller/workflow logic.
- If a button title participates in layout animation, do not rely on `configurationUpdateHandler` to change that title.
  - set the title directly from the stored state, then animate `view.layoutIfNeeded()`.
- When a custom mode needs a substantially different presentation, a dedicated view is cleaner than overloading an existing generation view with many hidden subviews.
  - for `{HIDDEN}`, the dedicated interactive preview view made later cross-fades much simpler than trying to keep mutating `configurationFlyer` / `inGenerationController`.
- If a view needs to fade, use `alpha`, not `isHidden`.
  - `isHidden` is still fine as the end state after a transition, but alpha must drive the animation.
- For sibling view swaps, use `UIView.transition(from:to:..., .showHideTransitionViews + .transitionCrossDissolve)` when both views share the same superview.
  - do this immediately when the mode changes, not in some later deferred cleanup block.
- For the interactive preview itself:
  - make the preview image the source of truth for what is shown.
  - do not fall back to `underImage`, `previousImage`, or other unrelated view-controller image state.
  - if generation feedback has no new image, keep the current preview instead of clearing the canvas.
- Keep the latest interactive preview state local to `InteractiveGenerationController`, not on `EditWorkflow`.
  - all preview-session state is easier to reason about when owned by the controller that manages the interactive session.
- If preview aspect ratio is tied to generation configuration and that configuration does not change during reruns, it is acceptable to only rebuild the aspect-ratio constraint on first preview image.
  - document that assumption explicitly in code.
- For busy / waiting visuals in mode-specific UI, copy the existing animation behavior instead of inventing a new one.
  - the busy-canvas pie animation was best duplicated into `InteractiveGenerationController` and dismissed on the first real preview frame.
- For settings / toolbar / project-bar “disabled” visuals, package the alpha/overlay ownership with the UI controller that owns the view.
  - `EditWorkflow` should toggle high-level state, but the overlay view and opacity animation should live in `SettingsWorkflow`, `ToolsBarController`, `ProjectBarController`, etc.
- If an overlay is only visual, keep it always present and animate `alpha`.
  - no need to toggle `isHidden` if the view can simply stay at `alpha = 0`.
- For interactive generation state, the most robust model was:
  - `runningInteractionIdentifier`: the run that actually exists right now.
  - `activeInteractionIdentifier`: the latest desired interaction.
  - `active > running` means a rerun is queued.
  - `active == running` means finish the current run and do not rerun.
- This identifier model is enough to express:
  - typing during a run: queue rerun by advancing `activeInteractionIdentifier`.
  - typing back to the current text: collapse `activeInteractionIdentifier` back to `runningInteractionIdentifier`.
  - exit custom mode while running: collapse `activeInteractionIdentifier` to `runningInteractionIdentifier` so the current run finishes but no rerun happens.
  - stop after promoting the run to regular generation UI: advance `activeInteractionIdentifier` and cancel, so completion naturally launches the rerun.
- Avoid introducing another persistent mode flag if existing identifiers already encode the behavior correctly.
  - for `{HIDDEN}`, generate/stop semantics were expressible by manipulating interaction identifiers plus the existing UI visibility.
- Keep the logic that owns interactive-session state in `InteractiveGenerationController`, and keep `EditWorkflow` focused on:
  - existing generation orchestration,
  - view transitions,
  - and app-wide UI enable/disable state.
- For the final-step hold, `block(interactionIdentifier:)` should return both:
  - whether sampling should continue,
  - and how long the hold lasted.
  - returning the hold duration directly is cleaner than storing transient timing state elsewhere.
- Generation profiling around interactive hold should subtract hold time explicitly.
  - `GenerationProfileBuilder.skip(duration:)` is the clean place to do that so total duration and later timing segments stay correct.
- Verify these UI/state changes with the full app target, not only local typechecks:
  - `bazel build //Apps/DrawThings:DrawThings --ios_multi_cpus=arm64`

## LTX Hires Fix + Latent Upscaler
- `ModelZoo` latent-upscaler plumbing has to cover all download-status surfaces, not just `filesToDownload(...)`:
  - include `latentsUpscalers` in `isModelDownloaded(_ specification:)`
  - include them in `availableFiles(excluding:)`
  - include them in app-side manual model dependency lists that do not already go through `ModelZoo.filesToDownload(...)`
- In `LocalImageGenerator.generateTextOnly(...)`, keep the LTX hires-fix latent-upscaler integration narrowly scoped:
  - change only `generateTextOnly`
  - preserve the non-LTX and non-hires-fix paths unchanged
- For the LTX hires-fix shortcut, compute second-pass latent geometry once before branching:
  - `startWidth`
  - `startHeight`
  - `startScaleFactor`
  - `audioHeight`
  - this keeps the latent-upscaler condition and the fallback decode/encode path aligned
- Only consider the latent-upscaler path when `hiresFixEnabled` is true for `.ltx2` / `.ltx2_3`.
- The LTX latent-upscaler mode selection in `generateTextOnly(...)` is geometry-based:
  - `x2` when `firstPassStartWidth * 2 == startWidth` and `firstPassStartHeight * 2 == startHeight`
  - `x1.5` when `firstPassStartWidth * 3 == startWidth * 2` and `firstPassStartHeight * 3 == startHeight * 2`
- LTX latent-upscaler checkpoints are stored under the top-level model name `spatial_upsampler`.
- For the current LTX spatial-upscaler runtime path, prefer the known architecture over runtime tensor introspection:
  - `inChannels = 128`
  - `midChannels = 1024`
  - `numBlocks = 4`
  - load with `store.read("spatial_upsampler", model: spatialUpscaler, codec: [.jit, .externalData])`
- In the LTX hires-fix shortcut, keep the existing audio-latent retention path intact:
  - use the existing `audioHeight` / retained-audio logic in `generateTextOnly(...)`
  - only replace the video latent upsample part
  - do not run the spatial upscaler over the packed audio rows
- When bypassing the first-stage decode for a narrow optimization, it is acceptable to preserve the old control-flow shape with a placeholder value like `firstStageResult = (x, nil)` if that keeps the change local and avoids a wider refactor.
- Validation checklist for this path:
  - `bazel build //Libraries/LocalImageGenerator:LocalImageGenerator --ios_multi_cpus=arm64`
  - `bazel build //Apps/DrawThings:DrawThings --ios_multi_cpus=arm64` when the surrounding app integration changes
