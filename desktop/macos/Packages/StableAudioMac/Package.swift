// swift-tools-version: 6.2

import PackageDescription

/// Stable Audio 3 (Small-Music and Medium) for Studio › Music: a Swift port of Stability AI's official MLX runtime
/// (https://github.com/Stability-AI/stable-audio-3, optimized/mlx, MIT License) on the same mlx-swift build the app
/// already uses for Magenta RealTime. It reads the official `stabilityai/stable-audio-3-optimized` MLX weights (.npz).
let package = Package(
    name: "StableAudioMac",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "StableAudioMac", targets: ["StableAudioMac"]),
    ],
    dependencies: [
        .package(path: "../../../../ios/LLMHub/LocalPackages/mlx-swift"),
        // T5Gemma's SentencePiece tokenizer (Google's C++ library); MagentaMac pins the same revision.
        .package(url: "https://github.com/liuliu/swift-sentencepiece", revision: "8d17bf2e017c97563e8805545d676be9739b6c0e"),
    ],
    targets: [
        .target(
            name: "StableAudioMac",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "SentencePiece", package: "swift-sentencepiece"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
