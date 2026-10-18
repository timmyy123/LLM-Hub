// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MagentaRuntime",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "MagentaRuntime", type: .dynamic, targets: ["MagentaRuntime"]),
    ],
    dependencies: [
        .package(path: "../mlx-swift"),
        .package(url: "https://github.com/liuliu/swift-sentencepiece", revision: "8d17bf2e017c97563e8805545d676be9739b6c0e"),
        .package(path: "../LiteRT-LM"),
    ],
    targets: [
        .target(
            name: "MagentaRuntime",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "SentencePiece", package: "swift-sentencepiece"),
                "MagentaLiteRTBridge",
            ]
        ),
        .target(
            name: "MagentaLiteRTBridge",
            dependencies: [
                .product(name: "CLiteRTLMRuntime", package: "LiteRT-LM"),
            ],
            publicHeadersPath: "include"
        ),
    ]
)
