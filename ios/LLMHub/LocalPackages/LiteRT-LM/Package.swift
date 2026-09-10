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
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.17.0/CLiteRTLM.xcframework.zip",
            checksum: "c94fc12aa0403cb47208e419cc3bfe258214ea17035f7a63c16de536869f2186"
        ),
        // The Prebuilt Binary Target for Mac
        .binaryTarget(
            name: "CLiteRTLM_mac",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.17.0/CLiteRTLM_mac.xcframework.zip",
            checksum: "83efd536485c9d58fcd7fb7d4556ddb16ca46bb775b0449d08d9825c6836c1a4"
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
