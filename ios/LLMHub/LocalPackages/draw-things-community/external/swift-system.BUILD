load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint", "swift_library")
load("@bazel_skylib//lib:selects.bzl", "selects")

cc_library(
    name = "CSystem",
    srcs = ["Sources/CSystem/shims.c"],
    hdrs = glob([
        "Sources/CSystem/include/*.h",
    ]),
    aspect_hints = [":CSystem_swift_interop"],
    includes = [
        "Sources/CSystem/include/",
    ],
    tags = ["swift_module=CSystem"],
)

swift_interop_hint(
    name = "CSystem_swift_interop",
    module_name = "CSystem",
)

config_setting(
    name = "macos_build",
    constraint_values = [
        "@platforms//os:osx",
    ],
)

config_setting(
    name = "ios_build",
    constraint_values = [
        "@platforms//os:ios",
    ],
)

selects.config_setting_group(
    name = "ios_or_macos_build",
    match_any = [
        ":macos_build",
        ":ios_build",
    ],
)

swift_library(
    name = "SystemPackage",
    srcs = glob([
        "Sources/System/**/*.swift",
    ]),
    defines = [
        "_CRT_SECURE_NO_WARNINGS",
        "SYSTEM_PACKAGE",
    ] + select({
        ":ios_or_macos_build": ["SYSTEM_PACKAGE_DARWIN"],
        "//conditions:default": [],
    }),
    module_name = "SystemPackage",
    visibility = ["//visibility:public"],
    deps = [
        ":CSystem",
    ],
)
