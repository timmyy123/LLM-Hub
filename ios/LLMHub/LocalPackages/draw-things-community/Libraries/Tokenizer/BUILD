load("@bazel_skylib//lib:selects.bzl", "selects")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

swift_library(
    name = "Tokenizer",
    srcs = glob(
        ["Sources/**/*.swift"],
        exclude = [
            "Sources/CoreML/**/*.swift",
            "Sources/Preprocessors/**/*.swift",
            "Sources/CoreMLModelManager/**/*.swift",
            "Sources/UNetWrapper/**/*.swift",
        ],
    ),
    module_name = "Tokenizer",
    visibility = ["//visibility:public"],
    deps = [
        "@swift-sentencepiece//:swift-sentencepiece",
    ],
)

swift_test(
    name = "TiktokenIncrementalDecoderTests",
    srcs = [
        "Tests/TiktokenIncrementalDecoderTests.swift",
    ],
    deps = [
        "//Libraries/BinaryResources",
        ":Tokenizer",
    ],
)
