// swift-tools-version: 6.2
import PackageDescription

// Upstream device/Metal binaries plus the matching source-built simulator slice.
let package = Package(
    name: "llama-b11683",
    platforms: [.iOS("17.5"), .macOS(.v14)],
    products: [.library(name: "LlamaCppBinary", targets: ["LlamaCppRuntime"])],
    targets: [
        .binaryTarget(
            name: "llama",
            path: "llama.xcframework"
        ),
        .target(name: "LlamaCppRuntime", dependencies: ["llama"]),
    ]
)
