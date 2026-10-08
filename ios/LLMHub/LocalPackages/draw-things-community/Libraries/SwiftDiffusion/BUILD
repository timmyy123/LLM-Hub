load("@bazel_skylib//lib:selects.bzl", "selects")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

config_setting(
    name = "macos_build",
    constraint_values = [
        "@platforms//os:osx",
    ],
)

config_setting(
    name = "ios_build",
    constraint_values = [
        "@platforms//os:ios",
    ],
)

selects.config_setting_group(
    name = "ios_or_macos_build",
    match_any = [
        ":macos_build",
        ":ios_build",
    ],
)

swift_library(
    name = "SwiftDiffusion",
    srcs = glob(
        ["Sources/**/*.swift"],
        exclude = [
            "Sources/CoreML/**/*.swift",
            "Sources/Preprocessors/**/*.swift",
            "Sources/CoreMLModelManager/**/*.swift",
            "Sources/UNetWrapper/**/*.swift",
            "Sources/Mappings/**/*.swift",
        ],
    ),
    module_name = "Diffusion",
    visibility = ["//visibility:public"],
    deps = [
        ":Mappings",
        "//Libraries/SwiftLLM",
        "//Libraries/Tokenizer",
        "//Libraries/WeightsCache",
        "//Vendors/ZIPFoundation",
        "@SwiftNumerics//:Numerics",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
        "@swift-atomics//:SwiftAtomics",
        "@swift-fickling//:fickling",
    ],
)

swift_library(
    name = "Mappings",
    srcs = glob(["Sources/Mappings/**/*.swift"]),
    module_name = "DiffusionMappings",
    visibility = ["//visibility:public"],
)

swift_test(
    name = "FirstStageTests",
    srcs = ["Tests/FirstStageTests.swift"],
    tags = [
        "manual",
        "requires-gpu",
    ],
    deps = [
        ":SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "MiniMaxH3Tests",
    srcs = ["Tests/MiniMaxH3Tests.swift"],
    tags = [
        "manual",
        "requires-gpu",
    ],
    deps = [
        ":SwiftDiffusion",
        "//Libraries/BinaryResources",
        "//Libraries/Tokenizer",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "TeaCacheTests",
    srcs = ["Tests/TeaCacheTests.swift"],
    tags = [
        "manual",
        "requires-gpu",
    ],
    deps = [
        ":SwiftDiffusion",
        "//Libraries/WeightsCache",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "QwenImage2_1ContractTests",
    srcs = ["Tests/QwenImage2_1ContractTests.swift"],
    deps = [
        ":SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "SamplerTests",
    srcs = ["Tests/SamplerTests.swift"],
    tags = [
        "manual",
        "requires-gpu",
    ],
    deps = [
        ":SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

swift_library(
    name = "SwiftDiffusionCoreML",
    srcs = glob(["Sources/CoreML/**/*.swift"]),
    module_name = "DiffusionCoreML",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiffusion",
        ":SwiftDiffusionCoreMLModelManager",
        "@SwiftAlgorithms//:Algorithms",
        "@s4nnc//nnc:nnc_coreml",
    ],
)

swift_library(
    name = "SwiftDiffusionCoreMLModelManager",
    srcs = glob(["Sources/CoreMLModelManager/CoreMLModelManager.swift"]),
    module_name = "DiffusionCoreMLModelManager",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "@SwiftAlgorithms//:Algorithms",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "SwiftDiffusionWrapper",
    srcs = glob(["Sources/UNetWrapper/UNetWrapper.swift"]),
    module_name = "DiffusionUNetWrapper",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiffusion",
        "//Libraries/DataModels",
        "@SwiftAlgorithms//:Algorithms",
        "@s4nnc//nnc:nnc_coreml",
        "@swift-atomics//:SwiftAtomics",
    ] + select({
        ":ios_or_macos_build": [
            ":SwiftDiffusionCoreML",
        ],
        "//conditions:default": [],
    }),
)

swift_library(
    name = "SwiftDiffusionPreprocessors",
    srcs = glob(
        ["Sources/Preprocessors/**/*.swift"],
    ),
    module_name = "DiffusionPreprocessors",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiffusion",
        "@s4nnc//nnc:nnc_coreml",
    ],
)

swift_test(
    name = "QwenImage2_1Tests",
    srcs = [
        "Tests/QwenImage2_1Tests.swift",
        "Tests/QwenImage2_1Unsplit.swift",
        "Tests/QwenImage2_1VAEReference.swift",
    ],
    tags = [
        "manual",
        "requires-gpu",
    ],
    deps = [
        ":Mappings",
        ":SwiftDiffusion",
        "//Libraries/WeightsCache",
        "@s4nnc//nnc",
    ],
)
