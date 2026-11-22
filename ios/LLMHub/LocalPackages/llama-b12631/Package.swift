// swift-tools-version: 6.2
import PackageDescription

// llama.cpp built from source (llama.xcframework) plus the flash streaming engine for MoE GGUF models
// (LLMHubBmoe.xcframework, BigMoeOnEdge built against the same llama.cpp; see bmoe/).
let package = Package(
    name: "llama-b12631",
    platforms: [.iOS("17.5"), .macOS(.v14)],
    products: [.library(name: "LlamaCppBinary", targets: ["LlamaCppRuntime"])],
    targets: [
        .binaryTarget(
            name: "llama",
            path: "llama.xcframework"
        ),
        .binaryTarget(
            name: "LLMHubBmoe",
            path: "LLMHubBmoe.xcframework"
        ),
        .target(
            name: "LlamaCppRuntime",
            dependencies: ["llama", "LLMHubBmoe"],
            linkerSettings: [.linkedLibrary("c++")]
        ),
    ]
)
