// swift-tools-version:5.9
import Foundation
import PackageDescription

let package = Package(
  name: "DrawThings",
  platforms: [.macOS(.v13), .iOS(.v16), .tvOS(.v16), .visionOS(.v1)],
  products: [
    .executable(name: "gRPCServerCLI", targets: ["gRPCServerCLI"]),
    .executable(name: "draw-things-cli", targets: ["DrawThingsCLI"]),
    .library(name: "_MediaGenerationKit", targets: ["_MediaGenerationKit"]),
    .library(name: "CLICloudAuth", targets: ["CLICloudAuth"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/liuliu/ccv.git", revision: "6a611be1aab6470ae279115ae4eaf7e01bc87135"
    ),
    .package(
      url: "https://github.com/liuliu/s4nnc.git",
      revision: "e2926609354a093395a4c3c30bcb627b9c1787aa"),
    .package(
      url: "https://github.com/liuliu/dflat.git",
      revision: "73925e51e4f44add842177a229f9990cb13711ff"),
    .package(
      url: "https://github.com/liuliu/swift-fickling.git",
      revision: "5c982bf479c4cdf8c7f72002cd79ec88b553ab34"),
    .package(
      url: "https://github.com/liuliu/swift-sentencepiece",
      revision: "8d17bf2e017c97563e8805545d676be9739b6c0e"),
    .package(url: "https://github.com/apple/swift-log.git", from: "1.4.4"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.1"),
    .package(url: "https://github.com/apple/swift-crypto.git", from: "3.7.1"),
    .package(url: "https://github.com/apple/swift-atomics.git", from: "1.2.0"),
    .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.4.5"),
    .package(url: "https://github.com/apple/swift-nio-ssl.git", from: "2.23.1"),
    .package(url: "https://github.com/jagreenwood/swift-log-datadog.git", from: "0.3.0"),

    .package(url: "https://github.com/grpc/grpc-swift.git", from: "1.16.0"),
    .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.27.0"),
    .package(url: "https://github.com/apple/swift-collections.git", from: "1.1.3"),
    .package(url: "https://github.com/apple/swift-algorithms.git", from: "1.1.0"),
    .package(url: "https://github.com/apple/swift-numerics.git", from: "1.0.0"),
    .package(
      url: "https://github.com/kelvin13/swift-png",
      revision: "075dfb248ae327822635370e9d4f94a5d3fe93b2"),
    .package(url: "https://github.com/weichsel/ZIPFoundation.git", from: "0.9.0"),
    .package(name: "SwiftSoup", path: "Vendors/SwiftSoup"),
    .package(
      url: "https://github.com/jpsim/Yams.git",
      revision: "948991e19e795cdd7bd310756a97b5fbda559535"),
  ],
  targets: [
    .target(
      name: "Utils",
      path: "Libraries/Utils/Sources"
    ),
    .target(
      name: "Tokenizer",
      dependencies: [
        .product(name: "SentencePiece", package: "swift-sentencepiece")
      ],
      path: "Libraries/Tokenizer/Sources"
    ),
    .target(
      name: "WeightsCache",
      dependencies: [
        .product(name: "Collections", package: "swift-collections"),
        .product(name: "Numerics", package: "swift-numerics"),
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/WeightsCache/Sources"
    ),
    .target(
      name: "SFMT",
      dependencies: [
        .product(name: "sfmt", package: "ccv")
      ],
      path: "Libraries/SFMT/Sources"
    ),
    .target(
      name: "C_Resources",
      path: "Libraries/BinaryResources/GeneratedC",
      publicHeadersPath: "."
    ),
    .target(
      name: "BinaryResources",
      dependencies: ["C_Resources"],
      path: "Libraries/BinaryResources",
      exclude: [
        "BUILD",
        "Resources",
        "GeneratedC",
      ],
      sources: ["Sources/BinaryResources.swift"]
    ),

    .target(
      name: "DiffusionMappings",
      path: "Libraries/SwiftDiffusion/Sources/Mappings"
    ),
    .target(
      name: "LLM",
      dependencies: [
        "Tokenizer",
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/SwiftLLM/Sources"
    ),
    .target(
      name: "PromptJSONExpansion",
      dependencies: [
        "BinaryResources",
        "LLM",
        "Tokenizer",
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/PromptJSONExpansion/Sources"
    ),
    .target(
      name: "Diffusion",
      dependencies: [
        "DiffusionMappings",
        "LLM",
        "Tokenizer",
        "WeightsCache",
        .product(name: "ZIPFoundation", package: "ZIPFoundation"),
        .product(name: "Numerics", package: "swift-numerics"),
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "Fickling", package: "swift-fickling"),
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/SwiftDiffusion/Sources",
      exclude: [
        "CoreML",
        "Preprocessors",
        "CoreMLModelManager",
        "UNetWrapper",
        "Mappings",
      ]
    ),
    .target(
      name: "DiffusionPreprocessors",
      dependencies: [
        "Diffusion",
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "NNCCoreMLConversion", package: "s4nnc"),
      ],
      path: "Libraries/SwiftDiffusion/Sources/Preprocessors"
    ),
    .target(
      name: "DiffusionCoreMLModelManager",
      dependencies: [
        "DataModels",
        .product(name: "Algorithms", package: "swift-algorithms"),
        .product(name: "Atomics", package: "swift-atomics"),
      ],
      path: "Libraries/SwiftDiffusion/Sources/CoreMLModelManager"
    ),
    .target(
      name: "DiffusionCoreML",
      dependencies: [
        "Diffusion",
        "DiffusionCoreMLModelManager",
        "WeightsCache",
        .product(name: "ZIPFoundation", package: "ZIPFoundation"),
        .product(name: "Algorithms", package: "swift-algorithms"),
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "NNCCoreMLConversion", package: "s4nnc"),
      ],
      path: "Libraries/SwiftDiffusion/Sources/CoreML"
    ),
    .target(
      name: "DiffusionUNetWrapper",
      dependencies: [
        "Diffusion",
        "DiffusionCoreML",
        "DataModels",
        .product(name: "Algorithms", package: "swift-algorithms"),
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/SwiftDiffusion/Sources/UNetWrapper"
    ),

    .target(
      name: "DataModels",
      dependencies: [
        "Diffusion",
        "Utils",
        .product(name: "SQLiteDflat", package: "dflat"),
      ],
      path: "Libraries/DataModels",
      exclude: [
        "BUILD",
        "Sources/config.fbs",
        "Sources/estimation.fbs",
        "Sources/mixing.fbs",
        "Sources/lora.fbs",
        "Sources/dataset.fbs",
        "Sources/paint_color.fbs",
        "Sources/peer_connection_id.fbs",
      ],
      sources: ["Sources", "PreGeneratedSPM"]
    ),
    .target(
      name: "ScriptDataModels",
      dependencies: [
        "DataModels",
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/Scripting",
      exclude: [
        "BUILD",
        "Sources/ScriptExecutor.swift",
        "Sources/ScriptZoo.swift",
        "Sources/SharedScript.swift",
      ],
      sources: ["Sources/ScriptModels.swift"]
    ),
    .target(
      name: "Scripting",
      dependencies: [
        "ScriptDataModels",
        "DataModels",
        "Diffusion",
        "ImageSegmentation",
        "Utils",
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/Scripting",
      exclude: [
        "BUILD",
        "Sources/ScriptModels.swift",
      ],
      sources: [
        "Sources/ScriptExecutor.swift",
        "Sources/ScriptZoo.swift",
        "Sources/SharedScript.swift",
      ]
    ),

    .target(
      name: "Upscaler",
      dependencies: [
        .product(name: "NNC", package: "s4nnc")
      ],
      path: "Libraries/Upscaler/Sources"
    ),
    .target(
      name: "ImageSegmentation",
      dependencies: [
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/ImageSegmentation/Sources"
    ),
    .target(
      name: "FaceRestorer",
      dependencies: [
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "ccv", package: "ccv"),
      ],
      path: "Libraries/FaceRestorer/Sources"
    ),

    .target(
      name: "ModelZoo",
      dependencies: [
        "DataModels",
        "Diffusion",
        "LLM",
        "Upscaler",
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/ModelZoo/Sources"
    ),
    .target(
      name: "Downloader",
      dependencies: [
        "ModelZoo"
      ],
      path: "Libraries/Downloader/Sources"
    ),
    .target(
      name: "Trainer",
      dependencies: [
        "DataModels",
        "Diffusion",
        "ModelZoo",
        "Tokenizer",
        "WeightsCache",
        "SFMT",
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "TensorBoard", package: "s4nnc"),
        .product(name: "SQLiteDflat", package: "dflat"),
      ],
      path: "Libraries/Trainer/Sources"
    ),
    .target(
      name: "ModelOp",
      dependencies: [
        "DataModels",
        "Diffusion",
        "ModelZoo",
        "Upscaler",
        "WeightsCache",
        .product(name: "ZIPFoundation", package: "ZIPFoundation"),
        .product(name: "Fickling", package: "swift-fickling"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/ModelOp/Sources"
    ),
    .target(
      name: "ConfigurationZoo",
      dependencies: [
        "DataModels",
        "ModelZoo",
        "ScriptDataModels",
      ],
      path: "Libraries/ConfigurationZoo/Sources"
    ),

    .target(
      name: "AudioConverter",
      dependencies: [
        "Diffusion",
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/AudioConverter/Sources"
    ),
    .target(
      name: "ImageGenerator",
      dependencies: [
        "DataModels",
        "ModelZoo",
        "Diffusion",
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/ImageGenerator/Sources"
    ),
    .target(
      name: "LocalImageGenerator",
      dependencies: [
        "AudioConverter",
        "DataModels",
        "ImageGenerator",
        "ModelZoo",
        "ScriptDataModels",
        "Diffusion",
        "DiffusionCoreMLModelManager",
        "DiffusionPreprocessors",
        "DiffusionUNetWrapper",
        "Upscaler",
        "FaceRestorer",
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "ccv", package: "ccv"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/LocalImageGenerator/Sources"
    ),
    .target(
      name: "RemoteImageGenerator",
      dependencies: [
        "DataModels",
        "ImageGenerator",
        "ModelZoo",
        "GRPCImageServiceModels",
        "GRPCServer",
        "Diffusion",
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "GRPC", package: "grpc-swift"),
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "Collections", package: "swift-collections"),
      ],
      path: "Libraries/RemoteImageGenerator/Sources"
    ),
    .target(
      name: "CLICloudAuth",
      path: "Libraries/CLICloudAuth/Sources"
    ),
    .target(
      name: "GRPCImageServiceModels",
      dependencies: [
        .product(name: "GRPC", package: "grpc-swift")
      ],
      path: "Libraries/GRPC/Models/Sources/imageService",
      exclude: ["imageService.proto"]
    ),
    .target(
      name: "GRPCControlPanelModels",
      dependencies: [
        .product(name: "GRPC", package: "grpc-swift")
      ],
      path: "Libraries/GRPC/Models/Sources/controlPanel",
      exclude: ["controlPanel.proto"]
    ),
    .target(
      name: "ServerConfigurationRewriter",
      dependencies: [
        "DataModels",
        .product(name: "GRPC", package: "grpc-swift"),
      ],
      path: "Libraries/GRPC/ServerConfigurationRewriter/Sources"
    ),
    .target(
      name: "GRPCServer",
      dependencies: [
        "GRPCImageServiceModels",
        "ServerConfigurationRewriter",
        "BinaryResources",
        "DataModels",
        "ImageGenerator",
        "ModelZoo",
        "ScriptDataModels",
        "Diffusion",
        "Utils",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "NNC", package: "s4nnc"),
      ],
      path: "Libraries/GRPC/Server/Sources",
      sources: [
        "GRPCFileUploader.swift",
        "GRPCHostnameUtils.swift",
        "GRPCServerAdvertiser.swift",
        "ImageGenerationClientWrapper.swift",
        "ImageGenerationServiceImpl.swift",
        "ProtectedValue.swift",
        "GRPCServiceBrowser.swift",
      ]
    ),
    .target(
      name: "ProxyControlClient",
      dependencies: [
        "GRPCControlPanelModels",
        "GRPCImageServiceModels",
        "BinaryResources",
        "DataModels",
        "ModelZoo",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "Logging", package: "swift-log"),
      ],
      path: "Libraries/GRPC/ProxyControlClient/Sources"
    ),
    .target(
      name: "ServerLoRALoader",
      dependencies: [
        "ServerConfigurationRewriter",
        "DataModels",
        "ModelZoo",
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "GRPC", package: "grpc-swift"),
      ],
      path: "Libraries/GRPC/ServerLoRALoader/Sources"
    ),

    .executableTarget(
      name: "gRPCServerCLI",
      dependencies: [
        "BinaryResources",
        "DataModels",
        "GRPCControlPanelModels",
        "GRPCImageServiceModels",
        "GRPCServer",
        "ProxyControlClient",
        "ServerLoRALoader",
        "ImageGenerator",
        "LocalImageGenerator",
        "Diffusion",
        "Utils",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "GRPC", package: "grpc-swift"),
        .product(name: "DataDogLog", package: "swift-log-datadog"),
      ],
      path: "Apps/gRPCServerCLI",
      exclude: ["SupportingFiles"],
      sources: ["gRPCServerCLI.swift"]
    ),
    .target(
      name: "DrawThingsCLILib",
      dependencies: [
        "AudioConverter",
        "BinaryResources",
        "ConfigurationZoo",
        "DataModels",
        "Downloader",
        "CLICloudAuth",
        "GRPCServer",
        "ImageGenerator",
        "LocalImageGenerator",
        "ModelOp",
        "ModelZoo",
        "RemoteImageGenerator",
        "ScriptDataModels",
        "Diffusion",
        "DiffusionCoreMLModelManager",
        "Trainer",
        "Tokenizer",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "PNG", package: "swift-png"),
      ],
      path: "Apps/DrawThingsCLI",
      exclude: ["Executable", "IOSSystem", "Tests"],
      sources: ["DrawThingsCLI.swift", "DrawThingsCLIContext.swift"]
    ),
    .executableTarget(
      name: "DrawThingsCLI",
      dependencies: ["DrawThingsCLILib"],
      path: "Apps/DrawThingsCLI/Executable"
    ),
    .target(
      name: "DeviceAttestation",
      path: "Libraries/DeviceAttestation/Sources"
    ),
    .target(
      name: "Localization",
      path: "Libraries/Localization/Sources"
    ),
    .target(
      name: "TextHistory",
      dependencies: [
        .product(name: "Dflat", package: "dflat"),
        .product(name: "SQLiteDflat", package: "dflat"),
      ],
      path: "Libraries/History",
      exclude: [
        "BUILD",
        "Sources/clip.fbs",
        "Sources/script_session.fbs",
        "Sources/tensor_data.fbs",
        "Sources/tensor_history.fbs",
        "Sources/tensor_moodboard_data.fbs",
        "Sources/text_history.fbs",
        "Sources/text_lineage.fbs",
        "Sources/thumbnail_history.fbs",
        "Sources/thumbnail_history_half.fbs",
        "Tests",
      ],
      sources: ["Sources/TextHistoryManager.swift", "PreGeneratedSPM"]
    ),
    .target(
      name: "WebSearch",
      dependencies: [
        .product(name: "SwiftSoup", package: "SwiftSoup")
      ],
      path: "Libraries/WebSearch/Sources"
    ),
    .target(
      name: "_MediaGenerationKit",
      dependencies: [
        "BinaryResources",
        "ConfigurationZoo",
        "DataModels",
        "Diffusion",
        "Downloader",
        "GRPCImageServiceModels",
        "ImageGenerator",
        "LocalImageGenerator",
        "ModelOp",
        "RemoteImageGenerator",
        "ModelZoo",
        "ScriptDataModels",
        "Tokenizer",
        "GRPCServer",
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "Dflat", package: "dflat"),
        .product(name: "GRPC", package: "grpc-swift"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "SQLiteDflat", package: "dflat"),
        .product(name: "Crypto", package: "swift-crypto"),
      ],
      path: "Libraries/MediaGenerationKit/Sources"
    ),
  ]
)

// LocalCode sources and their app-only dependencies are not included in the community sync.
// SwiftPM validates every declared path, even when building only a CLI product.
let localCodePath = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
  .appendingPathComponent("Apps/LocalCode").path
if FileManager.default.fileExists(atPath: localCodePath) {
  package.products.append(.library(name: "LocalCodeApp", targets: ["LocalCodeApp"]))
  package.dependencies += [
    .package(
      url: "https://github.com/airbnb/lottie-ios.git",
      revision: "a004050748dc197c56256a14dca49a035d74726c"),
    .package(name: "HighlighterSwift", path: "Vendors/HighlighterSwift"),
    .package(name: "ExceptionCatcher", path: "Vendors/ExceptionCatcher"),
    .package(name: "Nantes", path: "Vendors/Nantes"),
    .package(name: "SnapKit", path: "Vendors/SnapKit"),
    .package(name: "SwiftMath", path: "Vendors/SwiftMath"),
  ]
  package.targets += [
    .target(
      name: "PrivacyPassClient",
      dependencies: [
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "_CryptoExtras", package: "swift-crypto"),
      ],
      path: "Libraries/PrivacyPassClient/Sources/PrivacyPassClient"
    ),
    .target(
      name: "LinkTimeFlags",
      path: "Apps/LocalCodeCatalyst/Support/LinkTimeFlags",
      publicHeadersPath: "include"
    ),
    .target(
      name: "Features",
      dependencies: ["LinkTimeFlags"],
      path: "Libraries/Features/Sources",
      exclude: ["LinkTimeFlags.h"]
    ),
    .target(
      name: "Style",
      path: "Libraries/Style/Sources"
    ),
    .target(
      name: "MultiSlider",
      path: "Vendors/MultiSlider/Sources"
    ),
    .target(
      name: "Nuke",
      path: "Vendors/Nuke/Sources/Nuke"
    ),
    .target(
      name: "NukeUI",
      dependencies: ["Nuke"],
      path: "Vendors/Nuke/Sources/NukeUI"
    ),
    .target(
      name: "FBShimmer",
      path: "Vendors/FBShimmer/Sources"
    ),
    .target(
      name: "Components",
      dependencies: [
        "DataModels",
        "Features",
        "Localization",
        "MultiSlider",
        "NukeUI",
        "Style",
        .product(name: "Dflat", package: "dflat"),
        .product(name: "ExceptionCatcher", package: "ExceptionCatcher"),
        .product(name: "Nantes", package: "Nantes"),
        .product(name: "SnapKit", package: "SnapKit"),
        .product(name: "SQLiteDflat", package: "dflat"),
      ],
      path: "Libraries/Components/Sources"
    ),
    .target(
      name: "Advance",
      path: "Vendors/Advance/Sources/Advance"
    ),
    .target(
      name: "ProjectHistoryManager",
      dependencies: [
        .product(name: "Algorithms", package: "swift-algorithms"),
        .product(name: "Dflat", package: "dflat"),
        .product(name: "SQLiteDflat", package: "dflat"),
      ],
      path: "Libraries/ProjectHistoryManager",
      exclude: [
        "BUILD",
        "Sources/project_history.fbs",
        "Sources/project_queued_item.fbs",
        "Sources/project_thread.fbs",
        "Tests",
      ],
      sources: [
        "Sources/ProjectHistoryManager.swift", "Sources/EmbedSpan.swift",
        "Sources/EmbedsFileStore.swift", "PreGeneratedSPM",
      ]
    ),
    .target(
      name: "UserAccount",
      dependencies: [
        "DeviceAttestation",
        "Features",
        "Localization",
        "PrivacyPassClient",
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "Crypto", package: "swift-crypto"),
        .product(name: "Logging", package: "swift-log"),
        .product(name: "SQLiteDflat", package: "dflat"),
      ],
      path: "Libraries/UserAccount",
      exclude: [
        "BUILD",
        "Sources/account.fbs",
        "Sources/local_consumable_metadata.fbs",
        "Sources/privacy_pass.fbs",
        "Tests",
      ],
      sources: ["Sources", "PreGeneratedSPM"]
    ),
    .target(
      name: "RemoteAPI",
      path: "Libraries/RemoteAPI/Sources"
    ),
    .target(
      name: "MarkdownEngine",
      path: "Vendors/SwiftMarkdownEngine",
      exclude: ["Package.swift", "Tests", "Sources/MarkdownEngine/MarkdownEngine.docc"],
      sources: [
        "Sources/MarkdownEngine/Configuration",
        "Sources/MarkdownEngine/Diagnostics",
        "Sources/MarkdownEngine/Extensions",
        "Sources/MarkdownEngine/Parser",
        "Sources/MarkdownEngine/Platform",
        "Sources/MarkdownEngine/Services",
        "Sources/MarkdownEngine/UIKit",
        "Sources/MarkdownEngine/TextView/CodeBlockButton.swift",
        "Sources/MarkdownEngine/TextView/NativeTextViewSelectionTypes.swift",
      ]
    ),
    .target(
      name: "MarkdownEngineCodeBlocks",
      dependencies: [
        "MarkdownEngine",
        .product(name: "Highlighter", package: "HighlighterSwift"),
      ],
      path: "Vendors/SwiftMarkdownEngine/Sources/MarkdownEngineCodeBlocks"
    ),
    .target(
      name: "MarkdownEngineLatex",
      dependencies: [
        "MarkdownEngine",
        .product(name: "SwiftMath", package: "SwiftMath"),
      ],
      path: "Vendors/SwiftMarkdownEngine/Sources/MarkdownEngineLatex"
    ),
    .target(
      name: "Ripgrep",
      path: "Apps/LocalCodeCatalyst/Support/Ripgrep"
    ),
    .target(
      name: "ios_system",
      path: "Apps/LocalCodeCatalyst/Support/ios_system"
    ),
    .target(
      name: "BashToolContext",
      dependencies: ["CLICloudAuth", "Downloader", "ImageGenerator", "ios_system"],
      path: "Libraries/BashToolContext/Sources"
    ),
    .target(
      name: "LocalCodeApp",
      dependencies: [
        "Advance",
        "BashToolContext",
        "BinaryResources",
        "CLICloudAuth",
        "Components",
        "DataModels",
        "Diffusion",
        "Downloader",
        "ImageGenerator",
        .product(name: "ExceptionCatcher", package: "ExceptionCatcher"),
        "FBShimmer",
        "LLM",
        "LocalImageGenerator",
        "Localization",
        "MarkdownEngine",
        "MarkdownEngineCodeBlocks",
        "MarkdownEngineLatex",
        "ModelZoo",
        "Nuke",
        "ProjectHistoryManager",
        "RemoteAPI",
        "Ripgrep",
        "SFMT",
        "Style",
        "TextHistory",
        "Tokenizer",
        "UserAccount",
        "WebSearch",
        "ZIPFoundation",
        "ios_system",
        .product(name: "Atomics", package: "swift-atomics"),
        .product(name: "Dflat", package: "dflat"),
        .product(name: "Lottie", package: "lottie-ios"),
        .product(name: "NNC", package: "s4nnc"),
        .product(name: "SnapKit", package: "SnapKit"),
        .product(name: "SQLiteDflat", package: "dflat"),
        .product(name: "Yams", package: "Yams"),
      ],
      path: "Apps/LocalCode",
      exclude: [
        "BUILD",
        "CLI",
        "ContainerizedCommandExecution.md",
        "DDDoc",
        "Debug",
        "DeepSeek4Eval",
        "DirectoryEnvironment.md",
        "Eval",
        "JailedDirectoryStructure.md",
        "JavaScriptProjectSupport.md",
        "NodeMobileSupportBundle.md",
        "ProvisioningFiles",
        "Resources",
        "Sources/Features",
        "StaticInlineNode",
        "StaticInlinePython",
        "SupportingFiles",
      ],
      sources: ["Sources"],
      swiftSettings: [.define("LOCALCODE_CATALYST_ASAN")]
    ),
  ]
}
