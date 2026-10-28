# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b11946`
- Commit: `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`
- Source archive SHA-256: `76ad58014a4ec1093ffe1fd8e23e263a1fe76fd1aa44e3db788ca0b6023a2b71`
- License: MIT (see `LICENSE`)

This source builds the baseline ARMv8 CPU runtime (`libllmhub_llama_cpu.so`) and
the Vulkan GPU runtime (`libllmhub_llama_vulkan.so`) for arm64 devices.
Android Studio builds Vulkan with NDK shader tools and vendored Vulkan-Headers.
Local patches in `ggml/src/ggml-opencl/ggml-opencl.cpp` (below) and `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: the Android API 27 compatibility
fix uses the Vulkan dispatcher for `vkGetPhysicalDeviceFeatures2`; a failed pipeline compile
releases its claim so waiting threads fail instead of hanging; pipeline creation
failures are logged with `GGML_LOG_ERROR`, which reaches logcat; and `ggml_vk_device_is_supported`
rejects devices whose driver is below Vulkan 1.2 (Mali-G57 reports 1.1 under a 1.2 instance, and
its 1.2 entry points such as `vkGetBufferDeviceAddress` are null), including in the
first-non-CPU-device fallback.
Since b11946 upstream's buffer allocator (`ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp`) uses
`posix_memalign` on every non-Windows platform, so the old API 27 `aligned_alloc` patch is gone.

The Android app also vendors the official **b11946 Snapdragon** Android release
(`llama-b11946-bin-android-arm64-snapdragon.tar.gz`, SHA-256
`15dc2e7cb188fd40c119272bb8315786bace7ce1e042c55427700e79a0d95963`).
Its headers and stripped arm64 runtime libraries are under `app/src/main/cpp/llama-prebuilt`
and `app/src/main/jniLibs/arm64-v8a`; Hexagon v73/v75/v79/v81/v85 skels are under
`app/src/main/assets/llama_htp` (copied unstripped, as released). Android Studio links these into the separate
`libllmhub_llama_snapdragon.so` JNI bridge. The Snapdragon bridge is loaded for
OpenCL or Hexagon selection; Vulkan uses the source-built runtime instead.

`libggml-opencl.so` is the one Snapdragon library built from this source instead of taken from
the release, with `LLMHub:` patches in `ggml/src/ggml-opencl/ggml-opencl.cpp`:
- The tiled Q6_K/Q4_K GEMV layout for large lm_head/embedding weights is only chosen by default
  when its kernel accepts the 64x4 = 256 work-item groups it is dispatched with. Adreno 830
  (Snapdragon 8 Elite, Galaxy S25) allows 192 for `kernel_gemv_noshuffle_q6_K_f32_tiled`, so the
  launch failed with `CL_INVALID_WORK_GROUP_SIZE` (-54) and ggml aborted on the first decoded token.
- Flash-attention "split" kernels are registered only when their dispatch work-group size fits,
  as upstream already does for its other flash-attention kernels.
- A failed kernel launch logs the kernel name and global/local sizes before aborting.
Build it with the release preset's flags (`docs/backend/snapdragon/CMakeUserPresets.json`,
`arm64-android-snapdragon`) but `-DGGML_HEXAGON=OFF -DBUILD_SHARED_LIBS=ON`, against the Khronos
OpenCL headers and an NDK build of the OpenCL ICD loader, then strip it with
`llvm-strip --strip-unneeded`. Its exports match the release library's.

To upgrade the accelerator runtime, obtain the matching official Android Snapdragon
release archive (verify its digest with the GitHub releases API), replace its headers and
libraries together, strip the arm64 libraries with `llvm-strip --strip-unneeded` (`libggml`,
`libggml-base`, `libggml-cpu`, `libggml-hexagon`, `libggml-rpc`, `libllama`, `libmtmd`), copy every
`libggml-htp-v*.so` skel the release ships, and update the `llama_htp_<version>` asset-cache name
(and the skel list) in `LlamaCppInferenceService.kt`. Keep source and Snapdragon package versions aligned.

`libggml-opencl.so` is rebuilt with `bash android/tools/llama_opencl/build.sh` after the source is updated.

When replacing this source tree, update the `- Version:` line above: both CMake builds (app and
`sdcpp`) pass it to every compile command, so a new version recompiles all of llama.cpp/ggml.
Upstream archives keep their original file dates, which can be older than existing objects; in
4.5.2/4.5.3 Ninja kept a stale `llama-model.cpp.o` from before the b11738 bump, and every CPU model
load crashed in `llama_vocab::n_tokens()`. Local patches: `diff -ru` this tree against the release
archive and re-apply them to the new one.
