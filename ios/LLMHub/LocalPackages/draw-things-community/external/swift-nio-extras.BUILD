load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)

swift_library(
    name = "NIOExtras",
    srcs = glob([
        "Sources/NIOExtras/**/*.swift",
    ]),
    module_name = "NIOExtras",
    visibility = ["//visibility:public"],
    deps = [
        "@SwiftNIO//:NIO",
        "@SwiftNIO//:NIOCore",
        "@SwiftNIO//:NIOHTTP1",
    ],
)
