# llama.cpp iOS runtime

- Upstream: https://github.com/ggml-org/llama.cpp/releases/tag/b11946
- Version: `b11946`
- Commit: `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`
- Artifact: `llama-b11946-xcframework.zip`
- SHA-256: `3d126a914f8527838764902af8fabe7a48a20833918349b57df5f1b889097db6`

The local XCFramework provides llama.cpp and mtmd with Metal GPU support.
b11946 adds the `lfm3` architecture and LFM3-VL (`lfm3vl` projector) in mtmd
(ggml-org/llama.cpp#31252).
The official b11946 archive still contains only iOS device and macOS slices, not iOS Simulator.
Its unchanged device/macOS frameworks are combined with a simulator framework built
from the same verified b11946 source using upstream `build-xcframework.sh ios-sim`.
The simulator slice supports arm64 and x86_64. Only the matching slice is used by Xcode.

Source archive SHA-256: `76ad58014a4ec1093ffe1fd8e23e263a1fe76fd1aa44e3db788ca0b6023a2b71`.
Run `bash rebuild-xcframework.sh` on a Mac with Xcode and CMake to reproduce the artifact.
The script builds the native dependency only, not the LLMHub app. Build files and dSYMs
remain in the temporary directory printed by the script; dSYMs are not bundled in the app.

The app-owned Swift adapter is in `Sources/LlamaCppRuntime`; preserve it when upgrading.
When upgrading, update the package references and script's version/checksums together,
and verify the artifact includes iOS device, iOS Simulator and macOS libraries.
The references are this package's `Package.swift` name, `ios/LLMHub/Package.swift`, and
the `XCLocalSwiftPackageReference` in `ios/LLMHub/LLMHub.xcodeproj/project.pbxproj`.
