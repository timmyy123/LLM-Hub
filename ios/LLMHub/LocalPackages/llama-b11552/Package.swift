// swift-tools-version: 6.2
import PackageDescription

// Official upstream llama.cpp XCFramework, including the Metal and mtmd backends.
let package = Package(
    name: "llama-b11552",
    platforms: [.iOS("17.5"), .macOS(.v14)],
    products: [.library(name: "LlamaCppBinary", targets: ["LlamaCppRuntime"])],
    targets: [
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11552/llama-b11552-xcframework.zip",
            checksum: "ac195b4693da8e250ed778bc66a137912b912c0a6733f9c3e52153d7258172b8"
        ),
        .target(name: "LlamaCppRuntime", dependencies: ["llama"]),
    ]
)
