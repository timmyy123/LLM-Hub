load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)

swift_library(
    name = "NIOHPACK",
    srcs = glob([
        "Sources/NIOHPACK/**/*.swift",
    ]),
    module_name = "NIOHPACK",
    visibility = ["//visibility:public"],
    deps = [
        "@SwiftNIO//:NIO",
        "@SwiftNIO//:NIOConcurrencyHelpers",
        "@SwiftNIO//:NIOCore",
        "@SwiftNIO//:NIOHTTP1",
    ],
)

swift_library(
    name = "NIOHTTP2",
    srcs = glob([
        "Sources/NIOHTTP2/**/*.swift",
    ]),
    module_name = "NIOHTTP2",
    visibility = ["//visibility:public"],
    deps = [
        ":NIOHPACK",
        "@SwiftCollections//:Collections",
        "@SwiftNIO//:NIO",
        "@SwiftNIO//:NIOConcurrencyHelpers",
        "@SwiftNIO//:NIOCore",
        "@SwiftNIO//:NIOHTTP1",
        "@SwiftNIO//:NIOTLS",
        "@swift-atomics//:SwiftAtomics",
    ],
)
