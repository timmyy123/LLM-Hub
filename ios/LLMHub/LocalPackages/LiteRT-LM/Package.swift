// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "LiteRTLM",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
    ],
    products: [
        .library(
            name: "LiteRTLM",
            targets: ["LiteRTLM"]
        ),
        .library(
            name: "CLiteRTLMRuntime",
            targets: ["CLiteRTLM"]
        ),
    ],
    targets: [
        // The Prebuilt Binary Target for iOS
        .binaryTarget(
            name: "CLiteRTLM",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.16.0/CLiteRTLM.xcframework.zip",
            checksum: "4e0f683da07566ee79c143d2d58d387f77052b0e6a41562c969e5d2728fc9f4b"
        ),
        // The Prebuilt Binary Target for Mac
        .binaryTarget(
            name: "CLiteRTLM_mac",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.16.0/CLiteRTLM_mac.xcframework.zip",
            checksum: "3ae6c876abd74614b1869bfc40cb4d0b892981363564740268b1f8ac5cf895a4"
        ),
        // The Swift Wrapper Target
        .target(
            name: "LiteRTLM",
            dependencies: [
                .target(name: "CLiteRTLM", condition: .when(platforms: [.iOS])),
                .target(name: "CLiteRTLM_mac", condition: .when(platforms: [.macOS]))
            ],
            path: "swift",
            exclude: [
                "apple_fm",
                "CapabilitiesTests.swift",
                "EngineTests.swift",
                "ConversationTests.swift",
                "ToolTests.swift",
                "MessageTests.swift",
                "BUILD",
                "Info.plist",
            ]
        ),
    ]
)
