load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

swift_library(
    name = "AudioConverter",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "AudioConverter",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "AudioConverterTests",
    srcs = glob(["Tests/**/*.swift"]),
    deps = [
        ":AudioConverter",
        "@s4nnc//nnc",
    ],
)
