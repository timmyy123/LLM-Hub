load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Exporter",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "Exporter",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/SwiftDiffusion",
        "@ccv//lib:ccv",
        "@s4nnc//nnc",
    ],
)
