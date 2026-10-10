// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "LiteRTLM",
    platforms: [
        .iOS(.v15),
        .macOS(.v14),
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
        .library(
            name: "CLiteRTLMRuntime_mac",
            targets: ["CLiteRTLM_mac"]
        ),
    ],
    targets: [
        // The Prebuilt Binary Target for iOS
        .binaryTarget(
            name: "CLiteRTLM",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.18.0/CLiteRTLM.xcframework.zip",
            checksum: "d765b99592d4ec3d0c9e2bd69469454af06c834861340672da1891c0c121c347"
        ),
        // The Prebuilt Binary Target for Mac
        .binaryTarget(
            name: "CLiteRTLM_mac",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.18.0/CLiteRTLM_mac.xcframework.zip",
            checksum: "5f6ee68d95eeccb084c6e66d5ee47255e3020fa0fb29696dd0301ae26d6cfb4f"
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
                "ModelInfoTests.swift",
                "EngineTests.swift",
                "EmbeddingEngineTests.swift",
                "ConversationTests.swift",
                "ToolTests.swift",
                "MessageTests.swift",
                "BUILD",
                "Info.plist",
            ]
        ),
    ]
)
