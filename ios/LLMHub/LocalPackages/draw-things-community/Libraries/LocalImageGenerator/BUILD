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
    name = "LocalImageGenerator",
    srcs = glob(
               ["Sources/**/*.swift"],
               exclude = [
                   "Sources/ImageConverter.swift",
               ],
           ) +
           select({
               ":ios_or_macos_build": ["Sources/ImageConverter.swift"],
               "//conditions:default": [],
           }),
    module_name = "LocalImageGenerator",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/AudioConverter",
        "//Libraries/DataModels",
        "//Libraries/ImageGenerator",
        "//Libraries/ModelZoo",
        "//Libraries/Scripting:ScriptDataModels",
        "//Libraries/SwiftDiffusion",
        "//Libraries/SwiftDiffusion:SwiftDiffusionCoreMLModelManager",
        "//Libraries/SwiftDiffusion:SwiftDiffusionPreprocessors",
        "//Libraries/SwiftDiffusion:SwiftDiffusionWrapper",
        "//Libraries/Upscaler",
        "@SwiftCrypto//:Crypto",
        "@SwiftLog//:Logging",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
        "@swift-atomics//:SwiftAtomics",
    ] + select({
        ":ios_or_macos_build": [
            "//Libraries/FaceRestorer",
            "//Libraries/SwiftDiffusion:SwiftDiffusionCoreML",
        ],
        "//conditions:default": [],
    }),
)

swift_test(
    name = "ModelPreloaderTests",
    srcs = ["Tests/ModelPreloaderTests.swift"],
    deps = [
        ":LocalImageGenerator",
        "//Libraries/DataModels",
        "//Libraries/WeightsCache",
        "@dflat//:Dflat",
        "@dflat//:SQLiteDflat",
    ],
)

swift_test(
    name = "LongCatAvatarTests",
    srcs = ["Tests/LongCatAvatarTests.swift"],
    deps = [
        ":LocalImageGenerator",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "ColorCalibrationTests",
    srcs = [
        "Tests/ColorCalibrationTests.swift",
    ],
    deps = [
        ":LocalImageGenerator",
        "//Libraries/DataModels",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)
