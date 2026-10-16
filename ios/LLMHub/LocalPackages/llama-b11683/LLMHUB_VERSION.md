# llama.cpp iOS runtime

- Upstream: https://github.com/ggml-org/llama.cpp/releases/tag/b11683
- Version: `b11683`
- Commit: `a8d9a7e1e55f5b613f297504d9e159807ec3a99b`
- Artifact: `llama-b11683-xcframework.zip`
- SHA-256: `efdd0dabe6013653ed57a80b316d1600a7a349d86f8545beca3cd8a4804795bd`

The local XCFramework provides llama.cpp and mtmd with Metal GPU support.
The official b11683 archive contains only iOS device and macOS slices, not iOS Simulator.
Its unchanged device/macOS frameworks are combined with a simulator framework built
from the same verified b11683 source using upstream `build-xcframework.sh ios-sim`.
The simulator slice supports arm64 and x86_64. Only the matching slice is used by Xcode.

Source archive SHA-256: `f6dc7d2650fc75ca3c3bcee71a0e03243ec5339d7f727a854fd9d35da15ca7b7`.
Run `bash rebuild-xcframework.sh` on a Mac with Xcode and CMake to reproduce the artifact.
The script builds the native dependency only, not the LLMHub app. Build files and dSYMs
remain in the temporary directory printed by the script; dSYMs are not bundled in the app.

The app-owned Swift adapter is in `Sources/LlamaCppRuntime`; preserve it when upgrading.
When upgrading, update the package references and script's version/checksums together,
and verify the artifact includes iOS device, iOS Simulator and macOS libraries.
