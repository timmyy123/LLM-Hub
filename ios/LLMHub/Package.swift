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
        .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.20"),
        .package(path: "LocalPackages/media-generation-kit"),
        .package(
            url: "https://github.com/googleads/swift-package-manager-google-mobile-ads",
            from: "11.0.0"
        ),
        .package(
            url: "https://github.com/googleads/swift-package-manager-google-user-messaging-platform",
            from: "2.0.0"
        ),
        .package(url: "https://github.com/google-ai-edge/LiteRT-LM", exact: "0.14.0"),
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
                .product(name: "GoogleMobileAds", package: "swift-package-manager-google-mobile-ads"),
                .product(name: "GoogleUserMessagingPlatform", package: "swift-package-manager-google-user-messaging-platform"),
                .product(name: "LiteRTLM", package: "LiteRT-LM"),
                .product(name: "MediaGenerationKit", package: "media-generation-kit"),
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
