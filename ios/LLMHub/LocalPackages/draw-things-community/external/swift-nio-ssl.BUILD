load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_interop_hint",
    "swift_library",
)

swift_library(
    name = "NIOSSL",
    srcs = glob([
        "Sources/NIOSSL/**/*.swift",
    ]),
    module_name = "NIOSSL",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOBoringSSL",
        ":CNIOBoringSSLShims",
        "@SwiftNIO//:NIO",
        "@SwiftNIO//:NIOConcurrencyHelpers",
        "@SwiftNIO//:NIOCore",
        "@SwiftNIO//:NIOTLS",
    ],
)

cc_library(
    name = "CNIOBoringSSLShims",
    srcs = glob([
        "Sources/CNIOBoringSSLShims/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOBoringSSLShims/include/**/*.h",
    ]),
    aspect_hints = [":CNIOBoringSSLShims_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOBoringSSLShims/include"],
    local_defines = [
        "_GNU_SOURCE",
    ],
    tags = ["swift_module=CNIOBoringSSLShims"],
    visibility = ["//visibility:public"],
    deps = [":CNIOBoringSSL"],
)

swift_interop_hint(
    name = "CNIOBoringSSLShims_swift_interop",
    module_name = "CNIOBoringSSLShims",
)

cc_library(
    name = "CNIOBoringSSL",
    srcs = glob([
        "Sources/CNIOBoringSSL/**/*.h",
        "Sources/CNIOBoringSSL/**/*.c",
        "Sources/CNIOBoringSSL/**/*.cc",
        "Sources/CNIOBoringSSL/**/*.S",
    ]),
    hdrs = glob([
        "Sources/CNIOBoringSSL/include/**/*.h",
        "Sources/CNIOBoringSSL/**/*.inc",
    ]),
    aspect_hints = [":CNIOBoringSSL_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOBoringSSL/include"],
    local_defines = [
        "_GNU_SOURCE",
        "_POSIX_C_SOURCE=200112L",
        "_DARWIN_C_SOURCE",
    ],
    tags = ["swift_module=CNIOBoringSSL"],
    visibility = ["//visibility:public"],
    deps = [],
)

swift_interop_hint(
    name = "CNIOBoringSSL_swift_interop",
    module_name = "CNIOBoringSSL",
)
