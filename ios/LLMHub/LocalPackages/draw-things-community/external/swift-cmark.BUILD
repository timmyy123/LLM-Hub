load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_interop_hint",
)

cc_library(
    name = "cmark-gfm",
    srcs = glob([
        "src/*.c",
        "src/*.h",
        "src/*.inc",
    ]),
    hdrs = glob([
        "src/include/*.h",
    ]),
    aspect_hints = [":cmark_gfm_swift_interop"],
    includes = [
        "src/include/",
    ],
    tags = ["swift_module=cmark_gfm"],
    visibility = ["//visibility:public"],
)

swift_interop_hint(
    name = "cmark_gfm_swift_interop",
    module_name = "cmark_gfm",
)

cc_library(
    name = "cmark-gfm-extensions",
    srcs = glob([
        "extensions/*.c",
        "extensions/*.h",
    ]),
    hdrs = glob([
        "extensions/include/*.h",
    ]),
    aspect_hints = [":cmark_gfm_extensions_swift_interop"],
    includes = [
        "extensions/include/",
    ],
    tags = ["swift_module=cmark_gfm_extensions"],
    visibility = ["//visibility:public"],
    deps = [":cmark-gfm"],
)

swift_interop_hint(
    name = "cmark_gfm_extensions_swift_interop",
    module_name = "cmark_gfm_extensions",
)
