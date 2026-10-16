# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b11683`
- Commit: `a8d9a7e1e55f5b613f297504d9e159807ec3a99b`
- Source archive SHA-256: `f6dc7d2650fc75ca3c3bcee71a0e03243ec5339d7f727a854fd9d35da15ca7b7`
- License: MIT (see `LICENSE`)

This source builds the baseline ARMv8 CPU runtime (`libllmhub_llama_cpu.so`) and
the Vulkan GPU runtime (`libllmhub_llama_vulkan.so`) for arm64 devices.
Android Studio builds Vulkan with NDK shader tools and vendored Vulkan-Headers.
The Android API 27 compatibility patch in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
uses the Vulkan dispatcher for `vkGetPhysicalDeviceFeatures2`.
The buffer allocator in `ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp` uses
`posix_memalign` below Android API 28, where `aligned_alloc` is unavailable.

The Android app also vendors the official **b11683 Snapdragon** Android release
(`llama-b11683-bin-android-arm64-snapdragon.tar.gz`, SHA-256
`8cd14e096f99ca66f94418dc1c68dac3f7a9f4ac72b7344e77e79e3dc68be2bd`).
Its headers and stripped arm64 runtime libraries are under `app/src/main/cpp/llama-prebuilt`
and `app/src/main/jniLibs/arm64-v8a`; Hexagon v73/v75/v79/v81 skels are under
`app/src/main/assets/llama_htp`. Android Studio links these into the separate
`libllmhub_llama_snapdragon.so` JNI bridge. The Snapdragon bridge is loaded for
OpenCL or Hexagon selection; Vulkan uses the source-built runtime instead.

To upgrade the accelerator runtime, obtain the matching official Android Snapdragon
release archive, replace its headers and libraries together, strip debug symbols from
the arm64 libraries, and update the `llama_htp_b11683` asset-cache name in
`LlamaCppInferenceService.kt`. Keep source and Snapdragon package versions aligned.
