# llama.cpp iOS runtime

- Upstream: https://github.com/ggml-org/llama.cpp/releases/tag/b11552
- Version: `b11552`
- Commit: `23b0202a189c44a54625aadcb37a946dd1d6278d`
- Artifact: `llama-b11552-xcframework.zip`
- SHA-256: `ac195b4693da8e250ed778bc66a137912b912c0a6733f9c3e52153d7258172b8`

The official XCFramework provides llama.cpp and mtmd with Metal GPU support.
Xcode downloads and verifies this pinned artifact through `Package.swift`.
The app-owned Swift adapter is in `Sources/LlamaCppRuntime`; preserve it when upgrading.
Update the package directory, package references, artifact URL and checksum together.
