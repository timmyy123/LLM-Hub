# Vendored llama.cpp

- Upstream: https://github.com/ggml-org/llama.cpp
- Version: `b11048`
- Commit: `59fc5a1ca3842241dd53617ae2ae030c1a015061`
- Source archive SHA-256: `fc0a53035c43453d0ce8d0cb8aa26a046617dce1098138b6416276264ec38f67`
- License: MIT (see `LICENSE`)

This source builds the baseline ARMv8 CPU runtime (`libllmhub_llama_cpu.so`) for all
arm64 devices, including those without Snapdragon accelerators.

The Android app also vendors the official **b11179 Snapdragon** Android release
(`llama-b11179-bin-android-arm64-snapdragon.tar.gz`, SHA-256
`6e9dd527cd3d5164e06e1b4543630a61da740c67fd271b0a1b9d5db9a6748a74`).
Its headers and stripped arm64 runtime libraries are under `app/src/main/cpp/llama-prebuilt`
and `app/src/main/jniLibs/arm64-v8a`; Hexagon v73/v75/v79/v81 skels are under
`app/src/main/assets/llama_htp`. Android Studio links these into the separate
`libllmhub_llama_snapdragon.so` JNI bridge. The Snapdragon bridge is loaded only when
Adreno GPU or Hexagon NPU is selected; CPU never loads it.

To upgrade the accelerator runtime, obtain the matching official Android Snapdragon
release archive, replace its headers and libraries together, strip debug symbols from
the arm64 libraries, and update the `llama_htp_b11179` asset-cache name in
`LlamaCppInferenceService.kt`. The CPU source is a separate pinned build and must be
updated independently.
