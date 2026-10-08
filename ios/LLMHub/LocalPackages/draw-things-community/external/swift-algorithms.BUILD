load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Algorithms",
    srcs = glob([
        "Sources/Algorithms/**/*.swift",
    ]),
    module_name = "Algorithms",
    visibility = ["//visibility:public"],
    deps = [
        "@SwiftNumerics//:RealModule",
    ],
)
