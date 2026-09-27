# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b11218`
- Commit: `33c923db1b3fec0d20818e32facb4914534c415c`
- Source archive SHA-256: `82b243797d5888011b4ac09825fee867c6679d6985d9fd2f94d49e624c2ead19`
- License: MIT (see `LICENSE`)

This source builds the baseline ARMv8 CPU runtime (`libllmhub_llama_cpu.so`) and
the Vulkan GPU runtime (`libllmhub_llama_vulkan.so`) for arm64 devices.
Android Studio builds Vulkan with NDK shader tools and vendored Vulkan-Headers.
The Android API 27 compatibility patch in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
uses the Vulkan dispatcher for `vkGetPhysicalDeviceFeatures2`.

The Android app also vendors the official **b11218 Snapdragon** Android release
(`llama-b11218-bin-android-arm64-snapdragon.tar.gz`, SHA-256
`c647a1b61742cc1eea81473418175e20aee463acc4ca5177f0638923729de11b`).
Its headers and stripped arm64 runtime libraries are under `app/src/main/cpp/llama-prebuilt`
and `app/src/main/jniLibs/arm64-v8a`; Hexagon v73/v75/v79/v81 skels are under
`app/src/main/assets/llama_htp`. Android Studio links these into the separate
`libllmhub_llama_snapdragon.so` JNI bridge. The Snapdragon bridge is loaded for
OpenCL or Hexagon selection; Vulkan uses the source-built runtime instead.

To upgrade the accelerator runtime, obtain the matching official Android Snapdragon
release archive, replace its headers and libraries together, strip debug symbols from
the arm64 libraries, and update the `llama_htp_b11218` asset-cache name in
`LlamaCppInferenceService.kt`. Keep source and Snapdragon package versions aligned.
