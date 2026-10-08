load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

swift_library(
    name = "MediaGenerationKit",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "MediaGenerationKit",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/BinaryResources",
        "//Libraries/ConfigurationZoo",
        "//Libraries/DataModels",
        "//Libraries/DeviceAttestation",
        "//Libraries/Downloader",
        "//Libraries/GRPC:GRPCServer",
        "//Libraries/ImageGenerator",
        "//Libraries/LocalImageGenerator",
        "//Libraries/ModelOp",
        "//Libraries/ModelZoo",
        "//Libraries/RemoteImageGenerator",
        "//Libraries/Scripting:ScriptDataModels",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Tokenizer",
        "@SwiftCrypto//:Crypto",
        "@SwiftLog//:Logging",
        "@dflat//:SQLiteDflat",
        "@s4nnc//nnc",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_test(
    name = "MediaGenerationKitTests",
    srcs = [
        "Tests/MediaGenerationKitTests.swift",
        "Tests/MediaGenerationKitExampleCompileTests.swift",
    ],
    deps = [
        ":MediaGenerationKit",
        "//Libraries/ImageGenerator",
        "@SwiftLog//:Logging",
    ],
)
