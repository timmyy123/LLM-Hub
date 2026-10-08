# llama.cpp iOS runtime

- Upstream: https://github.com/ggml-org/llama.cpp/releases/tag/b11491
- Version: `b11491`
- Commit: `9b4ed0ca572be213f4440a490f341f810f7aa2cf`
- Artifact: `llama-b11491-xcframework.zip`
- SHA-256: `641362e8f05f12dd82b41ea37555e96e86e8305480f1cb48165a2f68db1410d7`

The official XCFramework provides llama.cpp and mtmd with Metal GPU support.
Xcode downloads and verifies this pinned artifact through `Package.swift`.
The app-owned Swift adapter is in `Sources/LlamaCppRuntime`; preserve it when upgrading.
Update the package directory, package references, artifact URL and checksum together.
