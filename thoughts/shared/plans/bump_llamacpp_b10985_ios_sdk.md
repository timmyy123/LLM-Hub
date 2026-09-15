# Bump llama.cpp to b10985 and Adapt Upstream GGUF LLM Performance Improvements

- Preserve clean baseline and verify no Android or third-party wrapper churn.
- Update the canonical llama.cpp pin in `ios/runanywhere-sdks-latest/sdk/runanywhere-commons/VERSIONS` from `b10819` to upstream release `b10985` (commit `7609846557c50f9d984719a9e1e8c5f3d02f807b`).
- Adapt upstream commit `70f01af` in `engines/llamacpp/llamacpp_backend.cpp` (`std::optional<int> user_gpu_layers`) so explicit requests to offload all layers to GPU (`gpu_layers = -1` via `ACCELERATOR_POLICY_GPU`) are not dropped by `>= 0` guards.
- Adapt upstream commit `8ad9ad3` in `engines/llamacpp/rac_llamacpp_vlm_ops.cpp` so VLM creation honours `accelerator_policy`, `use_gpu`, and `context_length`.
- Rebuild and package `RABackendLLAMACPP.xcframework` for arm64 iOS device, arm64 iOS simulator, and arm64 macOS under macOS 27 / Xcode 27.
- Adjust `sdk/runanywhere-swift/scripts/build-core-xcframework.sh` archive normalization for b10985's uncollided `llama.o` and support Xcode 27 build output paths.
- Update `Package.swift` MLX distribution search path to use absolute package directory under Swift 6.4 explicit module compilation.
- Verify symbols (`_rac_llm_llamacpp_create`, `hash_sha256_hex`), run Swift package builds (`RunAnywhere`, `RunAnywhereLlamaCPP`, `RunAnywhereONNX`), and validate consumer app compilation via `LLMHub.xcodeproj`.

## Completion notes

- Canonical llama.cpp pin is updated to `b10985`.
- Upstream GPU offloading fix (`std::optional<int> user_gpu_layers`) and VLM load option parser (`context_length`, `accelerator_policy`, `use_gpu`) were successfully adapted.
- `RABackendLLAMACPP.xcframework` was rebuilt with 10 parallel jobs, producing clean arm64 slices for iOS device, iOS simulator, and native macOS.
- All three slices contain the required `_rac_llm_llamacpp_create` entrypoint and `hash_sha256_hex` symbols.
- Swift package local-natives builds (`RunAnywhere`, `RunAnywhereLlamaCPP`, `RunAnywhereONNX`) passed with exit code 0.
- Consumer validation build `xcodebuild -project LLMHub.xcodeproj -scheme LLMHub -configuration Debug -destination 'generic/platform=iOS' -jobs 10 CODE_SIGNING_ALLOWED=NO build` finished with `** BUILD SUCCEEDED **`.
- Binary churn from unrelated regenerated frameworks was cleanly reverted; `git diff --check` passed with no issues.
