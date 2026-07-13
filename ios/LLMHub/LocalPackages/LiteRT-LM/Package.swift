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
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.14.0/CLiteRTLM.xcframework.zip",
            checksum: "dddac2f6713ed65eaf01c18e115d9fec22184adf575cc7856a21387e8ba937e1"
        ),
        // The Prebuilt Binary Target for Mac
        .binaryTarget(
            name: "CLiteRTLM_mac",
            url: "https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.14.0/CLiteRTLM_mac.xcframework.zip",
            checksum: "450615483509aaa6d34b321fdc6862e41a224b674468ab10aff64ebe113d21b7"
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
