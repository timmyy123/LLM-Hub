# llama.cpp iOS runtime

- Upstream: https://github.com/ggml-org/llama.cpp/releases/tag/b12020
- Version: `b12020`
- Commit: `0ec4027a2a0183de87c007069adddc06e665f7de`
- Artifact: `llama-b12020-xcframework.zip`
- SHA-256: `78fdf6ec32fad1f0e55a3fd78401841ba0a7b34f26b30078c6dc8d7951d81ae5`

The local XCFramework provides llama.cpp and mtmd with Metal GPU support.
b11946 added the `lfm3` architecture and LFM3-VL (`lfm3vl` projector) in mtmd
(ggml-org/llama.cpp#31252). b12020 keeps every C function the adapter calls unchanged;
it adds the F8_E4M3 tensor type, batched `ggml_backend_tensor_{copy,set,get}_n`,
LFM3 video prompt templates (`mtmd_get_video_config`, `mtmd_bitmap_set_is_video`) and a
`no_soundtrack` field in `mtmd_helper_video_init_params` (the adapter only takes the defaults).
The official b12020 archive still contains only iOS device and macOS slices, not iOS Simulator.
Its unchanged device/macOS frameworks are combined with a simulator framework built
from the same verified b12020 source using upstream `build-xcframework.sh ios-sim`.
The simulator slice supports arm64 and x86_64. Only the matching slice is used by Xcode.

Source archive SHA-256: `5496c675a36bcba53660a922f26cff75a8d24222d0554d337b450b552a30b502`.
Run `bash rebuild-xcframework.sh` on a Mac with Xcode and CMake to reproduce the artifact.
The script builds the native dependency only, not the LLMHub app. Build files and dSYMs
remain in the temporary directory printed by the script; dSYMs are not bundled in the app.

The app-owned Swift adapter is in `Sources/LlamaCppRuntime`; preserve it when upgrading.
When upgrading, update the package references and script's version/checksums together,
and verify the artifact includes iOS device, iOS Simulator and macOS libraries.
The references are this package's `Package.swift` name, `ios/LLMHub/Package.swift`, and
the `XCLocalSwiftPackageReference` in `ios/LLMHub/LLMHub.xcodeproj/project.pbxproj`.
