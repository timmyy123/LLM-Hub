# Bump llama.cpp to b10585 in the iOS RunAnywhere SDK

- Preserve a clean baseline and snapshot Android-related paths so the final diff can prove no Android source or binary changed.
- In `ios/runanywhere-sdks-latest/sdk/runanywhere-commons/VERSIONS`, update only the canonical llama.cpp pin and its release note from `b10360` to upstream release `b10585` (commit `d9f918d`).
- Run the Apple-only `sdk/runanywhere-swift/scripts/build-core-xcframework.sh` workflow with the host's available CPU capacity. This rebuilds the iOS device, iOS simulator, and macOS slices and synchronizes only the React Native and Flutter iOS XCFramework copies.
- Verify the rebuilt `RABackendLLAMACPP.xcframework` slices/architectures, confirm the compiled backend reports `b10585`, run the Swift SDK's local-native build validation, and run `git diff --check`.
- Audit the final diff against the baseline and fail the task if any Android path or Android artifact was modified.

## Mac Catalyst scope decision

- Do not add Mac Catalyst in this bump. CMake can select Catalyst with the single platform override `-DCMAKE_OSX_SYSROOT=macosx`, but the complete SDK cannot be added with only that build flag: the packaging script needs another build/staging/XCFramework slice, and the pinned ONNX Runtime and Sherpa-ONNX XCFrameworks contain no `ios`/`maccatalyst` variant. Adding genuine SDK support would therefore require dependency and packaging work beyond the requested one-flag limit.

## Completion notes

- Implementation approved. Android-related paths were clean before the pin change and remained untouched in the final diff.
- Updated the canonical pin to `b10585`; all three CMake fetches resolved the exact upstream commit `d9f918d2d06079b4336688e819eee821c8a9cd9e`.
- Updated the `common_fit_params` call for b10585's new optional extra-model parameter by passing `nullptr`; RunAnywhere fits no secondary model in this path.
- The full Apple-only SDK build completed successfully with 10 parallel jobs and synchronized the rebuilt llama.cpp XCFramework into the Swift, React Native, and Flutter iOS packages.
- Verified `RABackendLLAMACPP.xcframework` contains arm64 iOS device, arm64 iOS simulator, and arm64 macOS slices only. No Mac Catalyst slice was added.
- `RUNANYWHERE_USE_LOCAL_NATIVES=1 swift build --jobs 10` completed successfully, and `git diff --check` passed.
- Removed unrelated byte-only MLX runtime churn produced by the full rebuild, leaving the committed binary changes scoped to the three llama.cpp archives.
- Follow-up app-link validation found b10585's `mtmd-helper.cpp` now depends on upstream `vendor::hash`. Added that explicit CMake dependency and merged `libvendor-hash.a` into each Apple llama.cpp XCFramework slice so consuming Xcode apps resolve `hash_sha256_hex`.
- Verified every packaged llama.cpp slice now contains both the `hash_sha256_hex` reference and definition. A Debug arm64 iPhone build of the real `LLMHub` Xcode scheme completed with `** BUILD SUCCEEDED **`, confirming the reported linker failure is resolved.
