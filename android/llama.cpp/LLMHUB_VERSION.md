# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b11491`
- Commit: `9b4ed0ca572be213f4440a490f341f810f7aa2cf`
- Source archive SHA-256: `2de22c2dda0b9cb12e303fcf6902e1b2d15e2b7e5e8a8fcfc6c7f5952805ffe1`
- License: MIT (see `LICENSE`)

This source builds the baseline ARMv8 CPU runtime (`libllmhub_llama_cpu.so`) and
the Vulkan GPU runtime (`libllmhub_llama_vulkan.so`) for arm64 devices.
Android Studio builds Vulkan with NDK shader tools and vendored Vulkan-Headers.
The Android API 27 compatibility patch in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
uses the Vulkan dispatcher for `vkGetPhysicalDeviceFeatures2`.

The Android app also vendors the official **b11491 Snapdragon** Android release
(`llama-b11491-bin-android-arm64-snapdragon.tar.gz`, SHA-256
`4383ed7f2d5ad436b27a63c3ecfa9ed2779a12cc52a2e7170c000b8f781a4013`).
Its headers and stripped arm64 runtime libraries are under `app/src/main/cpp/llama-prebuilt`
and `app/src/main/jniLibs/arm64-v8a`; Hexagon v73/v75/v79/v81 skels are under
`app/src/main/assets/llama_htp`. Android Studio links these into the separate
`libllmhub_llama_snapdragon.so` JNI bridge. The Snapdragon bridge is loaded for
OpenCL or Hexagon selection; Vulkan uses the source-built runtime instead.

To upgrade the accelerator runtime, obtain the matching official Android Snapdragon
release archive, replace its headers and libraries together, strip debug symbols from
the arm64 libraries, and update the `llama_htp_b11491` asset-cache name in
`LlamaCppInferenceService.kt`. Keep source and Snapdragon package versions aligned.
