load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint", "swift_library")

cc_library(
    name = "CAtomic",
    srcs = glob(["Sources/CAtomic/*.c"]),
    hdrs = glob([
        "Sources/CAtomic/include/*.h",
    ]),
    aspect_hints = [":CAtomic_swift_interop"],
    includes = [
        "Sources/CAtomic/include/",
    ],
    tags = ["swift_module=CAtomic"],
)

swift_interop_hint(
    name = "CAtomic_swift_interop",
    module_name = "CAtomic",
)

swift_library(
    name = "Markdown",
    srcs = glob([
        "Sources/Markdown/**/*.swift",
    ]),
    module_name = "Markdown",
    visibility = ["//visibility:public"],
    deps = [
        ":CAtomic",
        "@SwiftCMark//:cmark-gfm",
        "@SwiftCMark//:cmark-gfm-extensions",
    ],
)
