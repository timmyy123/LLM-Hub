// swift-tools-version:5.9
import PackageDescription

// whisper.cpp's official Apple build (Metal, Accelerate), from https://github.com/ggml-org/whisper.cpp/releases.
// Dictation and the speech test on macOS run Whisper large-v3 turbo through it, in process.
let package = Package(
    name: "WhisperCpp",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "whisper", targets: ["whisper"]),
    ],
    targets: [
        .binaryTarget(
            name: "whisper",
            url: "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/whisper-b5454-xcframework.zip",
            checksum: "e57f8c48933000acabc13bb913fbe82805d483a2deff692cc6738836bd92393b"
        ),
    ]
)
