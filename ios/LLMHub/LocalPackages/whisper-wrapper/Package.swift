// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "whisper-wrapper",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "WhisperWrapper", targets: ["WhisperWrapper"]),
    ],
    dependencies: [
        .package(path: "../whisper.spm"),
    ],
    targets: [
        .target(
            name: "WhisperWrapper",
            dependencies: [
                .product(name: "whisper", package: "whisper.spm"),
            ]
        ),
    ]
)
