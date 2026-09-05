# Bump llama.cpp to b10819 in the Apple RunAnywhere SDK

- Preserve the clean baseline and audit the final diff to ensure no Android path or Android binary changes.
- Update the canonical llama.cpp pin in `ios/runanywhere-sdks-latest/sdk/runanywhere-commons/VERSIONS` from `b10585` to upstream release `b10819` (short commit `6a1a922`; resolve and record the full commit during configure).
- Reconfigure and build the existing Apple-only SDK workflow with full local CPU capacity, retaining the b10585 `vendor::hash` dependency and archive-packaging fix unless upstream proves it unnecessary.
- Rebuild `RABackendLLAMACPP.xcframework` for arm64 iOS device, arm64 iOS simulator, and arm64 macOS. Do not add Mac Catalyst.
- Verify all three slices and required symbols, run the Swift package local-native build, then build the real `LLMHub` Debug iPhone scheme with signing disabled to catch consumer-link failures.
- Remove any unrelated deterministic-build churn and confirm `git diff --check`, no React Native/Flutter tracked changes, and no Android changes.

## Completion notes

- Implementation approved. Baseline is clean and the canonical pin is now `b10819`.
- The first configure resolved b10819 to `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae` and discovered 44 mtmd implementations. It exposed a sticky CMake-cache bug: normal MLX-on builds did not reset a previously cached `RAC_BACKEND_MLX=OFF`. The Apple build script now passes the requested ON state explicitly.
- b10819 added an `mtmd_helper_init_opt` argument to `mtmd_helper_bitmap_init_from_file`; the RunAnywhere VLM adapter now supplies upstream's default options.
- The full Apple SDK build completed successfully with `RAC_BUILD_JOBS=10`. `RABackendLLAMACPP.xcframework` contains exactly three arm64 slices: iOS device, iOS simulator, and native macOS; no Mac Catalyst slice was added.
- All three configured source trees resolve to the exact b10819 commit above. Each packaged llama.cpp archive contains both the `hash_sha256_hex` reference and implementation, preserving the prior vendor-hash linker fix.
- `RUNANYWHERE_USE_LOCAL_NATIVES=1 swift build --jobs 10` completed successfully. The real consumer validation, `xcodebuild -project LLMHub.xcodeproj -scheme LLMHub -configuration Debug -destination 'generic/platform=iOS' -jobs 10 CODE_SIGNING_ALLOWED=NO build`, finished with `BUILD SUCCEEDED`.
- The build script's React Native and Flutter sync destinations byte-match the canonical llama.cpp XCFramework, but neither wrapper has tracked changes. Unrelated regenerated MLX runtime binary churn was reverted.
- Final scope and hygiene checks pass: `git diff --check` is clean, the only binary changes are the three llama.cpp archives, and there are no Android, RunAnywhere Kotlin, React Native, or Flutter paths in the tracked diff.
