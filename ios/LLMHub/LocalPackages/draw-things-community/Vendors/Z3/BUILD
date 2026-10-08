load("@build_bazel_rules_apple//apple:macos.bzl", "macos_command_line_application")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "embedded_support",
    srcs = ["Z3Embedded.cpp"],
    hdrs = ["Z3Embedded.h"],
    copts = ["-std=c++20"],
    includes = ["."],
)

cc_library(
    name = "Z3Shell",
    srcs = ["Z3Shell.cpp"],
    hdrs = ["Z3Shell.h"],
    copts = ["-std=c++20"],
    deps = ["@z3//:shell"],
)

objc_library(
    name = "Z3Command",
    srcs = ["Z3Command.mm"],
    hdrs = ["Z3Command.h"],
    copts = ["-std=c++20"],
    module_name = "Z3Command",
    deps = [
        ":Z3Shell",
        "//Vendors/ios_system:api",
    ],
)

objc_library(
    name = "Z3Bridge",
    srcs = ["Z3Bridge.cpp"],
    hdrs = ["Z3Bridge.h"],
    copts = ["-std=c++20"],
    module_name = "Z3Bridge",
    deps = ["@z3"],
)

cc_test(
    name = "Z3BridgeTests",
    srcs = ["Z3BridgeTests.cpp"],
    deps = [":Z3Bridge"],
)

cc_test(
    name = "Z3ShellTests",
    srcs = ["Z3ShellTests.cpp"],
    copts = ["-std=c++20"],
    deps = [
        ":Z3Bridge",
        ":Z3Shell",
    ],
)

objc_library(
    name = "command_validation_main",
    srcs = ["Z3CommandTests.m"],
    sdk_frameworks = ["Foundation"],
    deps = [
        ":Z3Bridge",
        ":Z3Command",
        "//Vendors/ios_system",
    ],
)

macos_command_line_application(
    name = "command_validation",
    bundle_id = "com.liuliu.z3-command-validation",
    minimum_os_version = "13.5",
    deps = [
        ":command_validation_main",
        "//Vendors/NodeJS:NodeMobile",
        "//Vendors/PythonIOS:PythonRuntime",
    ],
)

sh_test(
    name = "Z3CommandTests",
    srcs = ["run-command-tests.sh"],
    args = ["$(location :command_validation)"],
    data = [":command_validation"],
)

exports_files([
    "embed.py",
    "generate.py",
    "repositories.bzl",
    "z3.BUILD.bazel",
])
