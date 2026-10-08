load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)

swift_library(
    name = "DataDogLog",
    srcs = glob([
        "Sources/**/*.swift",
    ]),
    module_name = "DataDogLog",
    visibility = ["//visibility:public"],
    deps = [
        "@SwiftLog//:Logging",
    ],
)
