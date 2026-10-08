// swift-tools-version:5.9
import PackageDescription

let package = Package(
  name: "media-generation-kit",
  platforms: [.macOS(.v13), .iOS(.v16), .tvOS(.v16), .visionOS(.v1)],
  products: [
    .library(name: "MediaGenerationKit", targets: ["MediaGenerationKit"]),
    .executable(name: "media-generation-kit-cli", targets: ["MediaGenerationKitCLI"]),
  ],
  dependencies: [
    .package(path: "../draw-things-community"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.1"),
    .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.4.5"),
  ],
  targets: [
    .target(
      name: "MediaGenerationKit",
      dependencies: [
        .product(name: "_MediaGenerationKit", package: "draw-things-community")
      ]
    ),
    .executableTarget(
      name: "MediaGenerationKitCLI",
      dependencies: [
        "MediaGenerationKit",
        .product(name: "CLICloudAuth", package: "draw-things-community"),
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
  ]
)
