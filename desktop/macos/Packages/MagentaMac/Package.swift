// swift-tools-version: 6.2

import PackageDescription

/// Magenta RealTime 2 for the Mac app. Reuses the iOS app's runtime (the LiteRT bridge for MusicCoCa and the lock-free
/// audio ring are symlinks into ios/LLMHub/LocalPackages/magenta-runtime) and adds a live driver whose style blend and
/// sampling settings can change while it plays. The LiteRT C symbols come from LiteRT-LM's macOS dylib, which the app
/// already links for its LiteRT-LM chat engine.
let package = Package(
    name: "MagentaMac",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "MagentaMac", targets: ["MagentaMac"]),
    ],
    dependencies: [
        .package(path: "../../../../ios/LLMHub/LocalPackages/mlx-swift"),
        .package(url: "https://github.com/liuliu/swift-sentencepiece", revision: "8d17bf2e017c97563e8805545d676be9739b6c0e"),
    ],
    targets: [
        .target(
            name: "MagentaMac",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "SentencePiece", package: "swift-sentencepiece"),
                "MagentaLiteRTBridge",
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MagentaLiteRTBridge",
            publicHeadersPath: "include"
        ),
    ]
)
