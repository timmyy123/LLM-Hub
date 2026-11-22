# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b12631`
- Commit: `e300919cc704ddde3a8a8346a8714f57eeeee71b`
- Source archive SHA-256: `bea5a657605d90dc2104ec7f3134cbec90e5fdbe1a2fdd50e66782cbedaff56b`
- Snapdragon release: `llama-b12631-bin-android-arm64-snapdragon.tar.gz`, SHA-256
  `fb4ad7a08bbc99b5f64570d955c0bbd02eba77a877e37a6594c8adbc98e78310`
- License: MIT (see `LICENSE`)

## What is built from where

- This source: `libllmhub_llama_cpu.so` (CPU) and `libllmhub_llama_vulkan.so` (Vulkan) JNI libraries over shared
  `libllmhub_llama.so`, `libllmhub_mtmd.so`, `libllmhub_ggml.so` and `libllmhub_ggml-base.so`, built with
  `GGML_BACKEND_DL` and `GGML_CPU_ALL_VARIANTS`: one `libllmhub_ggml-cpu-android_<arch>.so` per ARM feature level
  (ARMv8.0 up to ARMv9.2 with SVE2/SME), of which the JNI library loads the highest `ggml_backend_score()`, and
  the `libllmhub_ggml-vulkan.so` module (built with the NDK's glslc and the vendored Vulkan-Headers), loaded only
  by the Vulkan runtime. Everything is renamed `libllmhub_*` so the Snapdragon runtime's
  `ggml_backend_load_all_from_path()` scan of the native library directory never loads it. Also built from this
  source: `libggml-opencl.so`, and ggml inside `libllmhub_sdcpp.so`.
- The official Snapdragon release: headers in `app/src/main/cpp/llama-prebuilt`, stripped runtime libraries in
  `app/src/main/jniLibs/arm64-v8a`, Hexagon skels (unstripped) in `app/src/main/assets/llama_htp`. They are
  linked into `libllmhub_llama_snapdragon.so`, which is loaded for OpenCL or Hexagon. Two of those libraries are
  built from this source instead, with the release preset flags (`android/tools/llama_opencl/build.sh`):
  `libggml-opencl.so` (OpenCL patches) and `libllama.so` (the Aion architecture and chat template).

## Local patches (marked `LLMHub:`)

`ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
- `vkGetPhysicalDeviceFeatures2` goes through the Vulkan dispatcher (Android API 27).
- A failed pipeline compile releases its claim, so waiting threads fail instead of hanging, and is logged
  with `GGML_LOG_ERROR` (reaches logcat).
- `ggml_vk_device_is_supported` rejects drivers below Vulkan 1.2, also in the first-non-CPU-device fallback
  (Mali-G57 reports 1.1 and its 1.2 entry points such as `vkGetBufferDeviceAddress` are null).

`ggml/src/ggml-opencl/ggml-opencl.cpp`:
- The tiled Q6_K/Q4_K GEMV layout is chosen only when its kernel accepts the 256 work-item groups it is
  dispatched with (Adreno 830 allows 192 and aborted with `CL_INVALID_WORK_GROUP_SIZE`).
- Flash-attention "split" kernels are registered only when their dispatch work-group size fits.
- A failed kernel launch logs the kernel name and global/local sizes before aborting.
- `kernel_mul_mat_f16_f16`, `_f32_f32` and `_f16_f32` are split into dispatches of at most 2^28
  multiply-adds, and `kernel_cpy_f32_f32_pack` into dispatches of at most 2^20 elements, so long video
  generation kernels do not make Android report the app as not responding (issue #133).
  `kernel_mul_mv_f32_f32_r2` is not split.

`ggml/src/ggml-opencl/kernels/cpy.cl`:
- The copy kernels locate the start of each source row in dst once, with 32-bit division when the index fits,
  instead of four 64-bit divisions and modulos per element (b12477 moved the unflattening into the element loop).
  Only a source row that spans dst rows is located per element. On Adreno the per-element version made the
  per-token V transpose (flash attention auto, resolved off on OpenCL) slow enough to drop LFM3 1.7B decode on
  the Galaxy Tab S9+ from 18-19 to 7.7 tok/s.

`src/models/aion.cpp`, `conversion/aion.py` and their registrations (`llama-arch`, `llama-model`, `models.h`,
`gguf-py/gguf/constants.py`, `conversion/__init__.py`, `tests/test-llama-archs.cpp`):
- The `aion` architecture for microsoft/Aion-1.0-Instruct-2.3B (Qwen3-style block, per-layer sliding-window and
  full attention, YaRN on the full-attention layers only), from Xylolabs' `aion-llama.cpp.patch`
  (huggingface.co/Xylolabs/Aion-1.0-Instruct-2.3B-GGUF). Upstream has no Aion support as of b12631. Drop this
  patch once upstream adds the architecture, and check that its GGUF keys match before reusing our GGUF.

`src/llama-chat.cpp`, `src/llama-chat.h`:
- An `aion` built-in chat template, detected from the GGUF's template ("You are Aion Instruct"). Without it the
  template is taken for Phi-3, which puts a newline after every role marker and drops Aion's default system turn
  (identity, knowledge cutoff, current date). Tools are not formatted.

## Upgrading

1. Unpack the new source archive over this tree, keeping this file. `diff -ru` the old tree against its own
   release archive to find the local patches and re-apply them; check whether upstream made any unnecessary.
2. Update the version lines above. Both CMake builds (app and `sdcpp`) pass the version to every compile
   command, so a new version recompiles all of llama.cpp/ggml. This is required: upstream archives keep old
   file dates, and Ninja otherwise keeps stale objects.
3. From the matching Snapdragon release (verify its digest with the GitHub releases API): copy the headers,
   `llvm-strip --strip-unneeded` `libggml`, `libggml-base`, `libggml-cpu`, `libggml-hexagon`, `libggml-rpc`
   and `libmtmd` into `jniLibs` (`libllama` comes from step 4), copy every `libggml-htp-v*.so`, and update the `llama_htp_<version>`
   cache name and skel list in `LlamaCppInferenceService.kt`.
4. Rebuild `libggml-opencl.so` and `libllama.so` with `bash android/tools/llama_opencl/build.sh` (release preset flags with
   `-DGGML_HEXAGON=OFF -DBUILD_SHARED_LIBS=ON`); their exports must match the release libraries' (plus the `llama_model_aion` symbols).
