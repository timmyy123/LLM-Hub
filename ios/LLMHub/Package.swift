// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "LLMHub",
    defaultLocalization: "en",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "LLMHub",
            targets: ["LLMHub"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.6")
    ],
    targets: [
        .target(
            name: "LLMHub",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift")
            ],
            resources: [
                .process("Icon.png"),
                .process("en.lproj"),
                .process("ar.lproj"),
                .process("de.lproj"),
                .process("es.lproj"),
                .process("fa.lproj"),
                .process("fr.lproj"),
                .process("he.lproj"),
                .process("id.lproj"),
                .process("it.lproj"),
                .process("ja.lproj"),
                .process("ko.lproj"),
                .process("pl.lproj"),
                .process("pt.lproj"),
                .process("ru.lproj"),
                .process("tr.lproj"),
                .process("uk.lproj")
            ]
        ),
    ]
)
