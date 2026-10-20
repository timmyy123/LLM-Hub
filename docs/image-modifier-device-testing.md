# Image modifier device tests

Use physical devices for generation coverage. A passing build or settings screenshot does not establish that an imported weight affects generation.

Image files are imported through the existing model import panel on both platforms. On Android, open AI Models → Import External Model and choose a base checkpoint, LoRA, CLIP-L, CLIP-G, text encoder, VAE, textual inversion, tag list, definitions, or translation list in Model Format. On iOS, the sheet's Image tab has the same Model Format menu: Base checkpoint keeps the Draw Things catalog search, and the other types pick an installed target model. The image generator's settings sheet only adjusts installed items: negative prompt, guidance, CLIP skip, enable toggles, LoRA weights, and removal. Image imports therefore share the import panel's premium gate.

## iOS

`ImageExtrasDeviceTests` is an app-hosted XCTest target. Run its shared scheme on a connected iPhone. The generation test explicitly skips simulators and requires a downloaded compatible Draw Things SD1.5 model. The current fixture setup uses Counterfeit v3.0 plus its CLIP-L encoder and VAE.

Stage these temporary files in the app's Documents directory with `devicectl device copy to`:

- `llmhub-test-lora.ckpt`: Draw Things EpiNoiseOffset v2, from `https://static.libnnc.org/epi_noiseoffset_v2_lora_f16.ckpt`.
- `llmhub-test-portable-lora.safetensors`: `pytorch_lora_weights.safetensors` from `latent-consistency/lcm-lora-sdv1-5` on Hugging Face.
- `llmhub-test-embedding.safetensors`: a valid F16 `[1,768]` embedding tensor named `*`. A deterministic synthetic tensor tests import and conditioning; it does not test a learned artistic style.

```sh
xcodebuild -project ios/LLMHub/LLMHub.xcodeproj -scheme ImageExtrasDeviceTests \
  -destination 'id=PHYSICAL_IPHONE_UDID' -allowProvisioningUpdates test
```

The tests check prompt resources and persistence, fixed-seed guidance/negative-prompt/CLIP-skip differences, LoRA weight changes and disabling, embedding conditioning for the same prompt, encoder/VAE overrides, SafeTensors LoRA conversion, and image-to-image strength. Options are restored and imported weight copies removed. The final cleanup test removes the explicitly named staged fixtures and stale test registry records.

CLIP-G and T5 generation need suitable models and weights; SD1.5 coverage does not establish those paths work.

## Android

Build and install both debug APKs. Stage the following files under the debug app's private `files/modifier-test-fixtures/` directory:

- `sd15.safetensors`: the official `stable-diffusion-v1-5/stable-diffusion-v1-5` EMA checkpoint, converted to F16 without changing tensor names or shapes.
- `clip.safetensors` and `vae.safetensors`: its CLIP and VAE tensor subsets in standalone component format. Strip `cond_stage_model.transformer.` from CLIP keys and `first_stage_model.` from VAE keys; the native loader supplies the component prefixes.
- `lora.safetensors`: the LCM SD1.5 LoRA above.
- `embedding.safetensors`: the valid embedding fixture above.

```sh
adb shell am instrument -w \
  -e class com.llmhub.llmhub.imagegen.ImageExtrasInstrumentedTest,com.llmhub.llmhub.imagegen.ImageModifierGenerationTest \
  com.llmhub.llmhub.test/androidx.test.runner.AndroidJUnitRunner
```

Add `-e modifierSteps 2` for two-step functional checks across the complete generation suite. For a focused component rerun, add `-e modifierPhase components`; it uses two denoising steps. GPU image generation keeps text encoding and VAE on CPU to avoid mobile Vulkan embedding-table and VAE encoder failures. Diffusion remains GPU accelerated. `-e modifierPhase strengths` isolates image-to-image strength checks with imported components.

Imported checkpoints run on the CPU on every Snapdragon device, as video GGUF models do. `SdCppImageRuntime.usesGpu` uses `DeviceInfo.isSnapdragon()`, which is also the default for GGUF Vulkan. Converted Local Dream models are unaffected and still use the QNN NPU. On a Galaxy S24 Ultra, the Adreno 750 driver could not create `matmul_quant_f32_aligned_1` or its f16-accumulator variant, even with `GGML_VK_DISABLE_DOT2`, `GGML_VK_DISABLE_COOPMAT`, or `GGML_VK_DISABLE_F16`. The device tests call the same policy, so they exercise the production backend. Tab S9+ results below predate this policy and used Vulkan diffusion.

When a float `.safetensors` checkpoint plus its imported encoders exceeds available RAM less 3 GiB, the native loader converts weights to Q8_0 at load and applies LoRAs at runtime. stable-diffusion.cpp's automatic LoRA mode reads the file's float types, and merging into converted Q8_0 weights aborts. Phone GPUs share RAM with the CPU while Adreno reports a Vulkan heap larger than physical memory, so fp16 SDXL (6.6 GB of weights) otherwise exhausts memory on a 12 GB phone.

The first class covers UI placement, persistence, portable resources, invalid imports, slider controls, and legacy-backend restrictions. The second performs native inference and compares settings and embedding effects. Preferences are restored before native inference; successful completion removes imported weight copies. Remove the staged fixture directory after testing. A native abort bypasses JUnit cleanup, so inspect and remove orphaned test imports after a crash.

## Implementation references reviewed

- Local Dream: `app/src/main/java/io/github/xororz/localdream/ui/screens/ModelListScreen.kt`, `app/src/main/cpp/src/main.cpp` (`runConvertMode`), `RequestParser.hpp`, and `dit/DitEngine.cpp`.
- Draw Things: `Libraries/MediaGenerationKit/Sources/LoRAImporter.swift`, `MediaGenerationExecutionUtilities.swift`, `Libraries/ModelOp/Sources/EmbeddingImporter.swift`, `Libraries/Tokenizer/Sources/TextualInversion.swift`, `Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift`, and the model registries.

Local Dream's converted MNN path bakes LoRA weights and CLIP skip into conversion. Its behavior differs from runtime modifier application on a raw checkpoint. Draw Things uses registered converted weights and embedding-aware tokenization.

## Verification on 18 October 2026

- Physical iPhone 15 Pro Max: generation/import/persistence tests passed, including LoRA weights and disabling, negative prompt, guidance, CLIP skip, CLIP-L/VAE and generic encoder overrides, SafeTensors LoRA conversion, embedding conditioning, and image strength. A separate uppercase embedding regression and SDK encoder-slot metadata checks passed. Settings-sheet and Photos selection UI checks passed on the same phone.
- Physical Samsung Tab S9+ (SM-X816B): the final build passed all seven instrumentation tests with `modifierSteps=2`. This includes six UI/import/persistence checks plus inference comparisons for guidance, CLIP skip, negative prompt, LoRA weights 0.2/0.8 and disabling, imported CLIP-L/VAE, same-prompt embedding conditioning, and image strength 0.3/0.8.
- AWS Device Farm Galaxy S24 Ultra (SM-S928U1, SM8650, Android 14), CPU backend: `SdxlGenerationDeviceTest` passed all 12 renders with SDXL 1.0 loaded as Q8_0. The identical external CLIP-G and the disabled LoRA reproduced the baseline bytes. The zero-projection CLIP-G, LoRA 0.8/0.2, guidance, negative prompt, CLIP skip 3, and image strength 0.3/0.8 each changed the output. Two-step 512×512 renders took 75–137 s at 5.0–5.7 GB PSS; the eight-step preview (a red teapot) took 321 s.
- Same device: `Sd3GenerationDeviceTest` passed all 11 renders with SD3.5 Medium Q4_0 and imported CLIP-L, CLIP-G, and Q4_0 T5-XXL. Repeated and reloaded renders were identical. Output changed without T5, with the negated-norm T5 (also distinct from no T5), without CLIP-G, and with guidance, negative prompt, and image strength 0.3. At two steps, strength 0.8 starts from full noise and matched the text-to-image baseline. Two-step renders took 95–181 s at 4.5–5.7 GB PSS; the 12-step preview took 649 s.
- Same device: all seven `ImageExtrasInstrumentedTest` UI tests passed, including image types in the Import External Model dialog.
- Galaxy S25 Ultra (SM8750, Android 15) and Galaxy S26 Ultra (SM8850, Android 16), CPU backend: both suites passed on both phones, and baseline hashes matched the S24 Ultra. SDXL two-step renders took 73–112 s on the S25 Ultra and 49–99 s on the S26 Ultra, with eight-step previews at 247 s and 216 s. SD3.5 two-step renders took 101–183 s and 78–133 s, with 12-step previews at 640 s and 474 s. Peak PSS stayed at or below 5.9 GB.
- Other chips, on the backend each one uses (Snapdragon never uses Vulkan, and Tensor G5's GPU produces noise, so both use the CPU):
  - Galaxy Tab S11 (MediaTek MT6991, Android 16), GPU: both suites passed. SDXL two-step renders took 89–149 s (eight-step preview 454 s); SD3.5 took 103–204 s (12-step preview 906 s).
  - Galaxy A56 (Exynos 1580, 8 GB, Android 15), GPU: SDXL passed with 174–265 s two-step renders. VAE decoding is tiled on devices under 10 GB of RAM. SD3.5 runs out of memory on this phone.
  - Pixel 9 Pro (Tensor G4, Android 15), GPU: SDXL passed with 174–375 s two-step renders (eight-step preview 1,103 s).
  - Pixel 10 Pro (Tensor G5, Android 16), CPU: both suites passed. SDXL took 52–92 s (preview 225 s); SD3.5 took 61–129 s (preview 454 s).
  - Mali GPUs and the fp16-to-Q8_0 switch on reload change pixels slightly. Reload checks therefore require the reloaded image to be much closer to the baseline than the altered encoder's image, instead of byte-identical.
- iOS CLIP-G/T5 generation remains unverified end to end: Draw Things SDXL/SD3 models were not available on the test iPhone. iOS metadata-slot assignment is tested.
- All 16 modifier localization keys exist in all 22 iOS and 24 Android locale files; iOS string-file and Android XML validation passed.

Device tests found and corrected missing native CLIP vocabulary, missing iOS embedding-token registration and lowercase keywords, incorrect multi-encoder CLIP slot assignment, and Vulkan shader-compilation claims left pending after failure. Mobile Vulkan embedding/VAE encoding failures are avoided by CPU text/VAE execution with GPU diffusion.

Device Farm runs found and corrected:
- out-of-memory crashes with fp16 SDXL, now loaded as Q8_0 when RAM is short;
- a scheduler abort when LoRAs were merged into converted Q8_0 weights, now applied at runtime;
- a rank mismatch when SD3 ran without CLIP-G (stable-diffusion.cpp built rank-2 zero pooled outputs);
- ggml-vulkan matmul pipeline failures on Adreno 750, now avoided by the existing Snapdragon Vulkan blocklist.

ggml abort messages and failed Vulkan pipeline names now reach logcat and tombstones. One limitation remains on GPUs that run imported SD3.5 checkpoints through Vulkan: the ~900 MB F32 `pos_embed` table exceeds Adreno's storage-buffer range, so it aborts in the scheduler. Blocked Snapdragon SoCs use the CPU and are unaffected. Other devices with small storage-buffer limits may hit this.

## SDXL and SD3.5 on AWS Device Farm

`SdxlGenerationDeviceTest` (CLIP-G) and `Sd3GenerationDeviceTest` (CLIP-L, CLIP-G and T5-XXL) are opt-in Android physical-device tests. They fail if their fixtures are missing. Build the debug app and instrumentation APK, then generate a custom AWS instrumentation specification per suite:

```sh
python3 android/tools/imagegen_devicefarm/make_test_spec.py sdxl /tmp/llmhub-sdxl-spec.yml
python3 android/tools/imagegen_devicefarm/make_test_spec.py sd3 /tmp/llmhub-sd3-spec.yml
```

Optional extra arguments, such as `GGML_VK_DISABLE_DOT2=1`, run each `vulkanProbe` (one load and render) in its own process with that ggml-vulkan setting before the full test. They are for driver diagnosis on devices that use Vulkan.

Upload the app APK, test APK, and generated spec to a Device Farm project. Select a Galaxy S24 Ultra (Snapdragon 8 Gen 3) or a newer supported Snapdragon device, and set a bounded job timeout. Device Farm runs each spec command in its own wrapper, so the generator keeps every command on one line and embeds `prepare_fixtures.py` as base64 rather than a heredoc. The spec checks the device's reported SoC, downloads pinned weights, and verifies their SHA-256 hashes. No model weights are added to this repository.

### SD3.5 Medium (T5-XXL)

The SD3 suite uses the ungated `second-state/stable-diffusion-3.5-medium-GGUF` conversion: the Q4_0 diffusion model (with VAE, no text encoders), Q4_0 T5-XXL, and CLIP-L/CLIP-G safetensors whose hashes match Stability's official files. Every encoder is therefore supplied through the app's import path. The host derives a test-only T5 copy with the sign of `encoder.final_layer_norm.weight` flipped (each Q4_0 block scale), which negates every T5 output.

The test requires repeated and reloaded generations to be identical. It also requires guidance and negative prompts to change output, and the following to differ from the baseline: generation without T5, generation with the altered T5, and generation without CLIP-G. The altered T5 must also differ from a missing T5. Image-to-image strength is compared and a 12-step preview is saved. Comparisons use 512×512 and two denoising steps.

### SDXL (CLIP-G)

The host derives a standalone CLIP-G encoder from the checkpoint with the native loader's canonical tensor names. A second test-only copy has its pooled projection zeroed. Generation compares the built-in and identical external encoder, then requires the altered encoder to change output for the same prompt and seed. Further comparisons cover guidance, negative prompts, CLIP skip, LoRA weights and disabling, and image-to-image strength. Functional comparisons use 512×512 images with two denoising steps; an eight-step preview is also saved. These checks establish functional effects, not final image quality or production performance at 1024×1024.

The custom spec collects output PNGs, SHA-256 image hashes, timings, process memory snapshots, device properties, fixture hashes, instrumentation output, and native logs. It requires a completed `OK (1 test)` result and rejects native process crashes. Test imports are deleted after execution and Device Farm handles device cleanup. SDXL tests do not cover T5; the SD3 suite does.
