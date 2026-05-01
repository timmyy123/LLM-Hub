// swift-tools-version: 6.2

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
        .package(path: "../runanywhere-sdks-latest"),
        .package(url: "https://github.com/apple/ml-stable-diffusion", from: "1.1.1"),
        .package(url: "https://github.com/weichsel/ZIPFoundation", from: "0.9.20"),
        // Raw ONNX Runtime Objective-C API for the Kokoro TTS path. RunAnywhereONNX
        // wraps ORT for LLM inference but doesn't expose generic ORT graph APIs.
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.20.0")
    ],
    targets: [
        .target(
            name: "LLMHub",
            dependencies: [
                .product(name: "RunAnywhere", package: "runanywhere-sdks-latest"),
                .product(name: "RunAnywhereLlamaCPP", package: "runanywhere-sdks-latest"),
                .product(name: "RunAnywhereONNX", package: "runanywhere-sdks-latest"),
                .product(name: "StableDiffusion", package: "ml-stable-diffusion"),
                .product(name: "ZIPFoundation", package: "ZIPFoundation"),
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
            ],
            exclude: [
                "check_strings.py"
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
                .process("uk.lproj"),
                .process("zh.lproj")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        ),
    ]
)
