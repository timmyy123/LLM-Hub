// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "LLMHub",
    defaultLocalization: "en",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "LLMHubMac",
            targets: ["LLMHub"]
        ),
    ],
    dependencies: [
        .package(path: "../../ios/LLMHub/LocalPackages/magenta-runtime"),
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.20"),
        .package(path: "../../ios/LLMHub/LocalPackages/media-generation-kit"),
        .package(path: "../../ios/LLMHub/LocalPackages/LiteRT-LM"),
        .package(path: "../../ios/LLMHub/LocalPackages/llama-b11491"),
        .package(path: "../../ios/LLMHub/LocalPackages/whisper-wrapper"),
    ],
    targets: [
        .executableTarget(
            name: "LLMHub",
            dependencies: [
                .product(name: "LlamaCppBinary", package: "llama-b11491"),
                .product(name: "MagentaRuntime", package: "magenta-runtime"),
                .product(name: "ZIPFoundation", package: "ZIPFoundation"),
                .product(name: "LiteRTLM", package: "LiteRT-LM"),
                .product(name: "MediaGenerationKit", package: "media-generation-kit"),
                .product(name: "WhisperWrapper", package: "whisper-wrapper"),
            ],
            path: "Sources/LLMHub",
            resources: [
                .copy("Resources/models.json"),
                .copy("Resources/configs.json"),
                .process("Resources/Icon.png"),
                .process("Resources/en.lproj"),
                .process("Resources/ar.lproj"),
                .process("Resources/da.lproj"),
                .process("Resources/de.lproj"),
                .process("Resources/es.lproj"),
                .process("Resources/fa.lproj"),
                .process("Resources/fr.lproj"),
                .process("Resources/he.lproj"),
                .process("Resources/hi.lproj"),
                .process("Resources/id.lproj"),
                .process("Resources/it.lproj"),
                .process("Resources/ja.lproj"),
                .process("Resources/ko.lproj"),
                .process("Resources/nl.lproj"),
                .process("Resources/pl.lproj"),
                .process("Resources/pt.lproj"),
                .process("Resources/ru.lproj"),
                .process("Resources/th.lproj"),
                .process("Resources/tr.lproj"),
                .process("Resources/uk.lproj"),
                .process("Resources/vi.lproj"),
                .process("Resources/zh-TW.lproj")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        ),
    ]
)
