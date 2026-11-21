# llama.cpp iOS runtime

- Upstream: https://github.com/ggml-org/llama.cpp/releases/tag/b12613
- Version: `b12613`
- Commit: `362ac929749d56e433878585a8506f24968698b4`
- Artifact: `llama-b12613-xcframework.zip`, SHA-256 `7075378ab937650128d1356d6952600ae2e65dbcf9528a07907b337290528a68`
- Source archive SHA-256: `539f5b7a306c9392e0308094f9c39d1c793041bf18a2c5da5e93f79050be2eb7`
- License: MIT

The local XCFramework provides llama.cpp and mtmd with Metal GPU support. The official archive has only
iOS device and macOS slices, so `rebuild-xcframework.sh` combines them, unchanged, with an iOS Simulator
slice (arm64 and x86_64) built from the same verified source with upstream `build-xcframework.sh ios-sim`.
Run `bash rebuild-xcframework.sh` on a Mac with Xcode and CMake to reproduce it; build files and dSYMs stay
in the temporary directory it prints and are not bundled in the app.

The app-owned Swift adapter is in `Sources/LlamaCppRuntime`; keep it when upgrading. In b12613
`llama_decode`, `llama_batch_init` and `llama_batch_free` are deprecated in favour of `llama_batch_ext`;
they still work and the adapter still uses them.

## Upgrading

Rename this folder to the new version and update, together: the folder name, this package's `Package.swift`
name, `ios/LLMHub/Package.swift`, the `XCLocalSwiftPackageReference` in
`ios/LLMHub/LLMHub.xcodeproj/project.pbxproj`, and the version and checksums in `rebuild-xcframework.sh`.
Check that the result has iOS device, iOS Simulator and macOS libraries.
