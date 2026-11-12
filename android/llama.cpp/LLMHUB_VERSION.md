# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b12317`
- Commit: `e0949d09b130707b3fa93d2cff616024231cf939`
- Source archive SHA-256: `d3607be1788451ae71a138479ea78e4b4b408b82bfae99c01f42b2a3d9357a46`
- Snapdragon release: `llama-b12317-bin-android-arm64-snapdragon.tar.gz`, SHA-256
  `8a9568a8b2d9f8afc7fcc3bb330f48df4283b6234c48af98ef9f1193f75f166e`
- License: MIT (see `LICENSE`)

## What is built from where

- This source: `libllmhub_llama_cpu.so` (ARMv8 CPU), `libllmhub_llama_vulkan.so` (Vulkan, built with the
  NDK's glslc and the vendored Vulkan-Headers), `libggml-opencl.so`, and ggml inside `libllmhub_sdcpp.so`.
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
