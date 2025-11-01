// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LLMHub",
    platforms: [
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "LLMHub",
            targets: ["LLMHub"]),
    ],
    dependencies: [
        // MediaPipe Tasks for LLM inference
        .package(url: "https://github.com/google/mediapipe", from: "0.10.0")
    ],
    targets: [
        .target(
            name: "LLMHub",
            dependencies: [
                .product(name: "MediaPipeTasksGenAI", package: "mediapipe")
            ])
    ]
)
