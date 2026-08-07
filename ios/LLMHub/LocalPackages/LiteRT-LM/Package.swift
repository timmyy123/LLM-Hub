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
    ],
    targets: [
        // The Prebuilt Binary Target for iOS
        .binaryTarget(
            name: "CLiteRTLM",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.15.0/CLiteRTLM.xcframework.zip",
            checksum: "d6ccf6b54362d894ff71a7580c7e446d36767dab908aecfbb16ffca0fa0bc59b"
        ),
        // The Prebuilt Binary Target for Mac
        .binaryTarget(
            name: "CLiteRTLM_mac",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.15.0/CLiteRTLM_mac.xcframework.zip",
            checksum: "d23cf189ce8f6bb2556c0a023805e245d1ec862434e501eb60f353488033c1b5"
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
