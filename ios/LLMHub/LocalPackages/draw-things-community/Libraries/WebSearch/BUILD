load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

swift_library(
    name = "WebSearch",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "WebSearch",
    visibility = ["//visibility:public"],
    deps = [
        "//Vendors/SwiftSoup",
    ],
)

swift_test(
    name = "WebSearchTests",
    srcs = glob(["Tests/**/*.swift"]),
    deps = [
        ":WebSearch",
        "//Vendors/SwiftSoup",
    ],
)
