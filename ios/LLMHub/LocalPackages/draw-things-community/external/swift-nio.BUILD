load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_interop_hint",
    "swift_library",
)

cc_library(
    name = "CNIOAtomics",
    srcs = glob([
        "Sources/CNIOAtomics/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOAtomics/**/*.h",
    ]),
    aspect_hints = [":CNIOAtomics_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOAtomics/include"],
    tags = ["swift_module=CNIOAtomics"],
)

swift_interop_hint(
    name = "CNIOAtomics_swift_interop",
    module_name = "CNIOAtomics",
)

cc_library(
    name = "CNIODarwin",
    srcs = glob([
        "Sources/CNIODarwin/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIODarwin/**/*.h",
    ]),
    aspect_hints = [":CNIODarwin_swift_interop"],
    defines = [
        "__APPLE_USE_RFC_3542",
    ],
    includes = ["Sources/CNIODarwin/include"],
    tags = ["swift_module=CNIODarwin"],
)

swift_interop_hint(
    name = "CNIODarwin_swift_interop",
    module_name = "CNIODarwin",
)

cc_library(
    name = "CNIOLLHTTP",
    srcs = glob([
        "Sources/CNIOLLHTTP/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOLLHTTP/**/*.h",
    ]),
    aspect_hints = [":CNIOLLHTTP_swift_interop"],
    copts = [],
    defines = [
        "LLHTTP_STRICT_MODE",
    ],
    includes = ["Sources/CNIOLLHTTP/include"],
    tags = ["swift_module=CNIOLLHTTP"],
)

swift_interop_hint(
    name = "CNIOLLHTTP_swift_interop",
    module_name = "CNIOLLHTTP",
)

cc_library(
    name = "CNIOLinux",
    srcs = glob([
        "Sources/CNIOLinux/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOLinux/**/*.h",
    ]),
    aspect_hints = [":CNIOLinux_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOLinux/include"],
    tags = ["swift_module=CNIOLinux"],
)

swift_interop_hint(
    name = "CNIOLinux_swift_interop",
    module_name = "CNIOLinux",
)

cc_library(
    name = "CNIOSHA1",
    srcs = glob([
        "Sources/CNIOSHA1/**/*.c",
    ]),
    hdrs = [
        "Sources/CNIOSHA1/**/*.h",
    ],
    aspect_hints = [":CNIOSHA1_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOSHA1/include"],
    tags = ["swift_module=CNIOSHA1"],
)

swift_interop_hint(
    name = "CNIOSHA1_swift_interop",
    module_name = "CNIOSHA1",
)

cc_library(
    name = "CNIOWindows",
    srcs = glob([
        "Sources/CNIOWindows/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CNIOWindows/**/*.h",
    ]),
    aspect_hints = [":CNIOWindows_swift_interop"],
    copts = [],
    includes = ["Sources/CNIOWindows/include"],
    tags = ["swift_module=CNIOWindows"],
)

swift_interop_hint(
    name = "CNIOWindows_swift_interop",
    module_name = "CNIOWindows",
)

swift_library(
    name = "NIO",
    srcs = glob([
        "Sources/NIO/*.swift",
    ]),
    module_name = "NIO",
    visibility = ["//visibility:public"],
    deps = [
        ":NIOCore",
        ":NIOEmbedded",
        ":NIOPosix",
    ],
)

swift_library(
    name = "NIOConcurrencyHelpers",
    srcs = glob([
        "Sources/NIOConcurrencyHelpers/*.swift",
    ]),
    module_name = "NIOConcurrencyHelpers",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOAtomics",
    ],
)

swift_library(
    name = "NIOCore",
    srcs = glob([
        "Sources/NIOCore/**/*.swift",
    ]),
    copts = [],
    module_name = "NIOCore",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOLinux",
        ":CNIOWindows",
        ":NIOConcurrencyHelpers",
        "@SwiftCollections//:Collections",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIOEmbedded",
    srcs = glob([
        "Sources/NIOEmbedded/*.swift",
    ]),
    copts = [],
    module_name = "NIOEmbedded",
    visibility = ["//visibility:public"],
    deps = [
        ":NIOConcurrencyHelpers",
        ":NIOCore",
        ":_NIODataStructures",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIOFoundationCompat",
    srcs = glob([
        "Sources/NIOFoundationCompat/*.swift",
    ]),
    module_name = "NIOFoundationCompat",
    visibility = ["//visibility:public"],
    deps = [
        ":NIO",
        ":NIOCore",
    ],
)

swift_library(
    name = "NIOHTTP1",
    srcs = glob([
        "Sources/NIOHTTP1/*.swift",
    ]),
    module_name = "NIOHTTP1",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIOLLHTTP",
        ":NIO",
        ":NIOConcurrencyHelpers",
        ":NIOCore",
    ],
)

swift_library(
    name = "NIOPosix",
    srcs = glob([
        "Sources/NIOPosix/*.swift",
    ]),
    copts = [],
    module_name = "NIOPosix",
    visibility = ["//visibility:public"],
    deps = [
        ":CNIODarwin",
        ":CNIOLinux",
        ":CNIOWindows",
        ":NIOConcurrencyHelpers",
        ":NIOCore",
        ":_NIODataStructures",
        "@swift-atomics//:SwiftAtomics",
    ],
)

swift_library(
    name = "NIOTLS",
    srcs = glob([
        "Sources/NIOTLS/*.swift",
    ]),
    module_name = "NIOTLS",
    visibility = ["//visibility:public"],
    deps = [
        ":NIO",
        ":NIOCore",
        "@SwiftCollections//:Collections",
    ],
)

swift_library(
    name = "_NIODataStructures",
    srcs = glob([
        "Sources/_NIODataStructures/*.swift",
    ]),
    module_name = "_NIODataStructures",
    visibility = ["//visibility:public"],
    deps = [],
)
