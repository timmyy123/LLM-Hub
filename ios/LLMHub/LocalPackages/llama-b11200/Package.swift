// swift-tools-version: 6.2
import PackageDescription

// Official upstream llama.cpp XCFramework, including the Metal and mtmd backends.
let package = Package(
    name: "llama-b11200",
    platforms: [.iOS("17.5"), .macOS(.v14)],
    products: [.library(name: "LlamaCppBinary", targets: ["LlamaCppRuntime"])],
    targets: [
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b11200/llama-b11200-xcframework.zip",
            checksum: "c62cae37316b12938cde3224493e008d121635bf759ddd08fbcdd587b3b8808c"
        ),
        .target(name: "LlamaCppRuntime", dependencies: ["llama"]),
    ]
)
