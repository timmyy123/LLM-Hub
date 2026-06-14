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
        .binaryTarget(
            name: "CLiteRTLM",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.13.1/CLiteRTLM.xcframework.zip",
            checksum: "7ff01c42106b754748b5dd3036a4a57161b25ebf523e705bebc1219061852362"
        ),
        .target(
            name: "LiteRTLM",
            dependencies: ["CLiteRTLM"],
            path: "swift",
            exclude: [
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
