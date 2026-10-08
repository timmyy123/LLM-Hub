load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Guaka",
    srcs = glob([
        "Sources/Guaka/**/*.swift",
    ]),
    module_name = "Guaka",
    visibility = ["//visibility:public"],
    deps = [
        "@SwiftStringScanner//:StringScanner",
    ],
)
