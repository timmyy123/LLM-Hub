# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b11552`
- Commit: `23b0202a189c44a54625aadcb37a946dd1d6278d`
- Source archive SHA-256: `1e4265e90b3c7450914a631925497b70d8a27b8cc29a1bda4380367e82e9870e`
- License: MIT (see `LICENSE`)

This source builds the baseline ARMv8 CPU runtime (`libllmhub_llama_cpu.so`) and
the Vulkan GPU runtime (`libllmhub_llama_vulkan.so`) for arm64 devices.
Android Studio builds Vulkan with NDK shader tools and vendored Vulkan-Headers.
The Android API 27 compatibility patch in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
uses the Vulkan dispatcher for `vkGetPhysicalDeviceFeatures2`.

The Android app also vendors the official **b11552 Snapdragon** Android release
(`llama-b11552-bin-android-arm64-snapdragon.tar.gz`, SHA-256
`c057c5d8f6632b83bbb25ad8f60f567bf59e3e761c5f6e1a56b96ac4938d596d`).
Its headers and stripped arm64 runtime libraries are under `app/src/main/cpp/llama-prebuilt`
and `app/src/main/jniLibs/arm64-v8a`; Hexagon v73/v75/v79/v81 skels are under
`app/src/main/assets/llama_htp`. Android Studio links these into the separate
`libllmhub_llama_snapdragon.so` JNI bridge. The Snapdragon bridge is loaded for
OpenCL or Hexagon selection; Vulkan uses the source-built runtime instead.

To upgrade the accelerator runtime, obtain the matching official Android Snapdragon
release archive, replace its headers and libraries together, strip debug symbols from
the arm64 libraries, and update the `llama_htp_b11552` asset-cache name in
`LlamaCppInferenceService.kt`. Keep source and Snapdragon package versions aligned.
