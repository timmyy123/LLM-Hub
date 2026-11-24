# llama.cpp iOS runtime

- Upstream: https://github.com/ggml-org/llama.cpp/releases/tag/b12631
- Version: `b12631`
- Commit: `e300919cc704ddde3a8a8346a8714f57eeeee71b`
- Source archive SHA-256: `bea5a657605d90dc2104ec7f3134cbec90e5fdbe1a2fdd50e66782cbedaff56b`
- License: MIT

The local XCFramework provides llama.cpp and mtmd with Metal GPU support. `rebuild-xcframework.sh` builds all
three slices (iOS device, iOS Simulator arm64/x86_64, macOS) from the verified source with upstream
`build-xcframework.sh ios-device ios-sim macos`, after applying `llmhub-aion.patch`. The official binary
xcframework is no longer used: it cannot load Aion. Run `bash rebuild-xcframework.sh` on a Mac with Xcode and
CMake to reproduce it; build files and dSYMs stay in the temporary directory it prints and are not bundled in
the app. The public llama, ggml and mtmd symbols are the same as the official framework's.

## Flash streaming (LLMHubBmoe.xcframework)

`LLMHubBmoe.xcframework` is a static library for flash streaming of MoE GGUF models: BigMoeOnEdge's engine
(`bigmoeonedge`, see its `LLMHUB_VERSION.md`), llama.cpp's common helpers it needs, and the C bridge in
`bmoe/` (`include/llmhub_bmoe.h`). It is compiled against the same verified, patched llama.cpp source as
`llama.xcframework` and links to that framework's llama, ggml and mtmd, so it carries no second copy of them.
`DirectLlamaCppBackend.load(..., flashStreaming: true)` uses it: the routed experts stay in the file and are read as
tokens need them, so a MoE model larger than the device's memory runs (CPU decode). Rebuild it with
`bash bmoe/rebuild-xcframework.sh` whenever `llama.xcframework` or the vendored engine changes.

## Local patch

`llmhub-aion.patch` is the same change as in `llama.cpp` (see its `LLMHUB_VERSION.md`): the `aion`
architecture for microsoft/Aion-1.0-Instruct-2.3B (from Xylolabs' `aion-llama.cpp.patch`) and an `aion` built-in
chat template. Upstream has no Aion support as of b12631; drop the patch once it does. Recreate it from the Android
tree with `git diff --relative=llama.cpp/` over the Aion commit (without `LLMHUB_VERSION.md`).

The app-owned Swift adapter is in `Sources/LlamaCppRuntime`; keep it when upgrading. In b12631
`llama_decode`, `llama_batch_init` and `llama_batch_free` are deprecated in favour of `llama_batch_ext`;
they still work and the adapter still uses them.

`llmhub-metal-nvfp4.patch` adds NVFP4 to the Metal backend: upstream PR ggml-org/llama.cpp#32825 ("metal : add
NVFP4 support", open as of 2026-11-24) ported to b12631 (its `mul_mm` template lines rewritten for b12631's longer
`kernel_mul_mm` parameter list; its `mul_mm_tensor.metal` and `prec_bf16` hunks dropped, b12631 has neither).
Unsloth's Inkling 2 Mini quants keep `ffn_down_exps` in NVFP4, and without the patch an iPhone cannot read the
prompt on the GPU with them (flash streaming refuses the load). Drop it once a llama.cpp release carries the PR.

## Upgrading

Rename this folder to the new version and update, together: the folder name, this package's `Package.swift`
name, `ios/LLMHub/Package.swift`, the `XCLocalSwiftPackageReference` in
`ios/LLMHub/LLMHub.xcodeproj/project.pbxproj`, the version and checksum in `rebuild-xcframework.sh`, and `llmhub-aion.patch` (re-apply it to the new source).
Check that the result has iOS device, iOS Simulator and macOS libraries.
