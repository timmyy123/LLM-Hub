load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "FaceRestorer",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "FaceRestorer",
    visibility = ["//visibility:public"],
    deps = [
        "@s4nnc//nnc",
        "@ccv//lib:ccv",
    ],
)
