// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "LLMHub",
    defaultLocalization: "en",
    platforms: [
        .iOS("17.5"),
        .macOS(.v14),
    ],
    products: [
        .library(
            name: "LLMHub",
            targets: ["LLMHub"]
        ),
    ],
    dependencies: [
        .package(path: "LocalPackages/magenta-runtime"),
        .package(url: "https://github.com/apple/ml-stable-diffusion", from: "1.1.1"),
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.20"),
        .package(path: "LocalPackages/media-generation-kit"),
        .package(path: "LocalPackages/LiteRT-LM"),
        .package(path: "LocalPackages/llama-b11200"),
        .package(path: "LocalPackages/whisper-wrapper"),
    ],
    targets: [
        .target(
            name: "LLMHub",
            dependencies: [
                .product(name: "LlamaCppBinary", package: "llama-b11200"),
                .product(name: "MagentaRuntime", package: "magenta-runtime"),
                .product(name: "StableDiffusion", package: "ml-stable-diffusion"),
                .product(name: "ZIPFoundation", package: "ZIPFoundation"),
                .product(name: "LiteRTLM", package: "LiteRT-LM"),
                .product(name: "MediaGenerationKit", package: "media-generation-kit"),
                .product(name: "WhisperWrapper", package: "whisper-wrapper"),
            ],
            exclude: [
                "check_strings.py"
            ],
            resources: [
                .process("Icon.png"),
                .process("en.lproj"),
                .process("ar.lproj"),
                .process("da.lproj"),
                .process("de.lproj"),
                .process("es.lproj"),
                .process("fa.lproj"),
                .process("fr.lproj"),
                .process("he.lproj"),
                .process("hi.lproj"),
                .process("id.lproj"),
                .process("it.lproj"),
                .process("ja.lproj"),
                .process("ko.lproj"),
                .process("nl.lproj"),
                .process("pl.lproj"),
                .process("pt.lproj"),
                .process("ru.lproj"),
                .process("th.lproj"),
                .process("tr.lproj"),
                .process("uk.lproj"),
                .process("vi.lproj"),
                .process("zh-TW.lproj")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        ),
    ]
)
