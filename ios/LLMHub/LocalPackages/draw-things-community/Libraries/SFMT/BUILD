load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "SFMT",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "SFMT",
    visibility = ["//visibility:public"],
    deps = [
        "@s4nnc//nnc",
    ],
)
