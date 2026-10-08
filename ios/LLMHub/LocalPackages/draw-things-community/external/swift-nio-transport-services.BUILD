load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_library",
)

swift_library(
    name = "NIOTransportServices",
    srcs = glob([
        "Sources/NIOTransportServices/**/*.swift",
    ]),
    module_name = "NIOTransportServices",
    visibility = ["//visibility:public"],
    deps = [
        "@SwiftNIO//:NIO",
        "@SwiftNIO//:NIOCore",
        "@SwiftNIO//:NIOFoundationCompat",
        "@SwiftNIO//:NIOTLS",
        "@swift-atomics//:SwiftAtomics",
    ],
)
