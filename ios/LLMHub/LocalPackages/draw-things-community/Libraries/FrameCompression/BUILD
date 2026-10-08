load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")

swift_library(
    name = "FrameCompression",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "FrameCompression",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

swift_binary(
    name = "FrameCompressionTestCLI",
    srcs = glob(["TestProgram/*.swift"]),
    visibility = ["//visibility:public"],
    deps = [
        ":FrameCompression",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)
