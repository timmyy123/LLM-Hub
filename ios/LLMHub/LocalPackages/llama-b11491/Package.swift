// swift-tools-version: 6.2
import PackageDescription

// Official upstream llama.cpp XCFramework, including the Metal and mtmd backends.
let package = Package(
    name: "llama-b11491",
    platforms: [.iOS("17.5"), .macOS(.v14)],
    products: [.library(name: "LlamaCppBinary", targets: ["LlamaCppRuntime"])],
    targets: [
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11491/llama-b11491-xcframework.zip",
            checksum: "641362e8f05f12dd82b41ea37555e96e86e8305480f1cb48165a2f68db1410d7"
        ),
        .target(name: "LlamaCppRuntime", dependencies: ["llama"]),
    ]
)
