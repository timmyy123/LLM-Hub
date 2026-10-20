# llama.cpp iOS runtime

- Upstream: https://github.com/ggml-org/llama.cpp/releases/tag/b11738
- Version: `b11738`
- Commit: `f174553e1621c6a22a64e7bfcbb58d3a5c49e6fb`
- Artifact: `llama-b11738-xcframework.zip`
- SHA-256: `a5e259da0415bfd84cdbe9145deeffd3c5029a8d2e2fd3c06c3807d6aaf47502`

The local XCFramework provides llama.cpp and mtmd with Metal GPU support.
The official b11738 archive contains only iOS device and macOS slices, not iOS Simulator.
Its unchanged device/macOS frameworks are combined with a simulator framework built
from the same verified b11738 source using upstream `build-xcframework.sh ios-sim`.
The simulator slice supports arm64 and x86_64. Only the matching slice is used by Xcode.

Source archive SHA-256: `79200a2770f86f8490299c598a2e0d4521574b2027802780b4985fc9b5928f9b`.
Run `bash rebuild-xcframework.sh` on a Mac with Xcode and CMake to reproduce the artifact.
The script builds the native dependency only, not the LLMHub app. Build files and dSYMs
remain in the temporary directory printed by the script; dSYMs are not bundled in the app.

The app-owned Swift adapter is in `Sources/LlamaCppRuntime`; preserve it when upgrading.
When upgrading, update the package references and script's version/checksums together,
and verify the artifact includes iOS device, iOS Simulator and macOS libraries.
