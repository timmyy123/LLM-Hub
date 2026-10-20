# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b11738`
- Commit: `f174553e1621c6a22a64e7bfcbb58d3a5c49e6fb`
- Source archive SHA-256: `79200a2770f86f8490299c598a2e0d4521574b2027802780b4985fc9b5928f9b`
- License: MIT (see `LICENSE`)

This source builds the baseline ARMv8 CPU runtime (`libllmhub_llama_cpu.so`) and
the Vulkan GPU runtime (`libllmhub_llama_vulkan.so`) for arm64 devices.
Android Studio builds Vulkan with NDK shader tools and vendored Vulkan-Headers.
Local patches in `ggml/src/ggml-opencl/ggml-opencl.cpp` (below) and `ggml/src/ggml-vulkan/ggml-vulkan.cpp`: the Android API 27 compatibility
fix uses the Vulkan dispatcher for `vkGetPhysicalDeviceFeatures2`; a failed pipeline compile
releases its claim so waiting threads fail instead of hanging; and pipeline creation
failures are logged with `GGML_LOG_ERROR`, which reaches logcat.
The buffer allocator in `ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp` uses
`posix_memalign` below Android API 28, where `aligned_alloc` is unavailable.

The Android app also vendors the official **b11738 Snapdragon** Android release
(`llama-b11738-bin-android-arm64-snapdragon.tar.gz`, SHA-256
`2e3953725fa6a78103d83777ed5edb9dc5194e920e47364bdfc2e4968fc8e8e3`).
Its headers and stripped arm64 runtime libraries are under `app/src/main/cpp/llama-prebuilt`
and `app/src/main/jniLibs/arm64-v8a`; Hexagon v73/v75/v79/v81 skels are under
`app/src/main/assets/llama_htp`. Android Studio links these into the separate
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
release archive, replace its headers and libraries together, strip debug symbols from
the arm64 libraries, and update the `llama_htp_b11738` asset-cache name in
`LlamaCppInferenceService.kt`. Keep source and Snapdragon package versions aligned.
