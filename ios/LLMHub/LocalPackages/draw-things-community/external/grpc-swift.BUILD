load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_binary",
    "swift_interop_hint",
    "swift_library",
)

cc_library(
    name = "CGRPCZlib",
    srcs = glob([
        "Sources/CGRPCZlib/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CGRPCZlib/**/*.h",
    ]),
    aspect_hints = [":CGRPCZlib_swift_interop"],
    includes = ["Sources/CGRPCZlib/include"],
    linkopts = ["-lz"],
    tags = ["swift_module=CGRPCZlib"],
)

swift_interop_hint(
    name = "CGRPCZlib_swift_interop",
    module_name = "CGRPCZlib",
)

swift_library(
    name = "GRPC",
    srcs = glob([
        "Sources/GRPC/**/*.swift",
    ]),
    defines = ["SWIFT_PACKAGE"],  # activates CgRPC imports
    module_name = "GRPC",
    visibility = ["//visibility:public"],
    deps = [
        ":CGRPCZlib",
        "@SwiftLog//:Logging",
        "@SwiftNIO//:NIO",
        "@SwiftNIO//:NIOCore",
        "@SwiftNIO//:NIOEmbedded",
        "@SwiftNIO//:NIOFoundationCompat",
        "@SwiftNIO//:NIOHTTP1",
        "@SwiftNIO//:NIOPosix",
        "@SwiftNIO//:NIOTLS",
        "@SwiftNIOExtras//:NIOExtras",
        "@SwiftNIOHTTP2//:NIOHTTP2",
        "@SwiftNIOSSL//:NIOSSL",
        "@SwiftNIOTransportService//:NIOTransportServices",
        "@SwiftProtobuf",
    ],
)

swift_binary(
    name = "protoc-gen-grpc-swift",
    srcs = glob([
        "Sources/protoc-gen-grpc-swift/*.swift",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_apple_swift_protobuf//:SwiftProtobuf",
        "@com_github_apple_swift_protobuf//:SwiftProtobufPluginLibrary",
    ],
)
