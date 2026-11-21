# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b12613`
- Commit: `362ac929749d56e433878585a8506f24968698b4`
- Source archive SHA-256: `539f5b7a306c9392e0308094f9c39d1c793041bf18a2c5da5e93f79050be2eb7`
- Snapdragon release: `llama-b12613-bin-android-arm64-snapdragon.tar.gz`, SHA-256
  `734c4fc7845d9df9847bec5f18f9df55ce9e8aa118669ae9af974a974ca58816`
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
  linked into `libllmhub_llama_snapdragon.so`, which is loaded for OpenCL or Hexagon.

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

## Upgrading

1. Unpack the new source archive over this tree, keeping this file. `diff -ru` the old tree against its own
   release archive to find the local patches and re-apply them; check whether upstream made any unnecessary.
2. Update the version lines above. Both CMake builds (app and `sdcpp`) pass the version to every compile
   command, so a new version recompiles all of llama.cpp/ggml. This is required: upstream archives keep old
   file dates, and Ninja otherwise keeps stale objects.
3. From the matching Snapdragon release (verify its digest with the GitHub releases API): copy the headers,
   `llvm-strip --strip-unneeded` `libggml`, `libggml-base`, `libggml-cpu`, `libggml-hexagon`, `libggml-rpc`,
   `libllama` and `libmtmd` into `jniLibs`, copy every `libggml-htp-v*.so`, and update the `llama_htp_<version>`
   cache name and skel list in `LlamaCppInferenceService.kt`.
4. Rebuild `libggml-opencl.so` with `bash android/tools/llama_opencl/build.sh` (release preset flags with
   `-DGGML_HEXAGON=OFF -DBUILD_SHARED_LIBS=ON`); its exports must match the release library's.
