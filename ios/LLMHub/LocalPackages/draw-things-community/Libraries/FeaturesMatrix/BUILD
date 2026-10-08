load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "FeaturesMatrix",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "FeaturesMatrix",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/SwiftDiffusion",
    ],
)
