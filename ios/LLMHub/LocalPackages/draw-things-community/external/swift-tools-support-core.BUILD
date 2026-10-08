load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint", "swift_library")

cc_library(
    name = "TSCclibc",
    srcs = glob(["Sources/TSCclibc/*.c"]),
    hdrs = glob([
        "Sources/TSCclibc/include/*.h",
    ]),
    aspect_hints = [":TSCclibc_swift_interop"],
    includes = [
        "Sources/TSCclibc/include/",
    ],
    tags = ["swift_module=TSCclibc"],
)

swift_interop_hint(
    name = "TSCclibc_swift_interop",
    module_name = "TSCclibc",
)

swift_library(
    name = "TSCLibc",
    srcs = glob([
        "Sources/TSCLibc/**/*.swift",
    ]),
    module_name = "TSCLibc",
    deps = [],
)

swift_library(
    name = "TSCBasic",
    srcs = glob([
        "Sources/TSCBasic/**/*.swift",
    ]),
    module_name = "TSCBasic",
    visibility = ["//visibility:public"],
    deps = [
        ":TSCLibc",
        ":TSCclibc",
        "@SwiftSystem//:SystemPackage",
    ],
)

swift_library(
    name = "TSCUtility",
    srcs = glob([
        "Sources/TSCUtility/**/*.swift",
    ]),
    module_name = "TSCUtility",
    visibility = ["//visibility:public"],
    deps = [
        ":TSCBasic",
        ":TSCLibc",
        ":TSCclibc",
        "@SwiftSystem//:SystemPackage",
    ],
)
