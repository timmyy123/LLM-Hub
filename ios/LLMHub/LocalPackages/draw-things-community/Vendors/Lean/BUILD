load("@build_bazel_rules_apple//apple:macos.bzl", "macos_command_line_application")
load("@build_bazel_rules_apple//apple:resources.bzl", "apple_resource_group")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint", "swift_library", "swift_test")
load(":encapsulated_archive.bzl", "lean_encapsulated_archive")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "PackageRuntime",
    srcs = ["package_runtime.py"],
    main = "package_runtime.py",
)

genrule(
    name = "LeanRuntimeArchive",
    srcs = [
        "ExportModuleDependencies.lean",
        "@lean4_toolchain//:LeanRuntimeArtifacts",
        "@lean4_toolchain//:lib/lean/Init.olean",
    ],
    outs = [
        "Lean_Init_Std.zip",
        "Lean_Init_Std.json",
    ],
    cmd = "$(execpath :PackageRuntime) " +
          "--lean $(execpath @lean4_toolchain//:bin/lean) " +
          "--exporter $(execpath ExportModuleDependencies.lean) " +
          "--root $$(dirname $(execpath @lean4_toolchain//:lib/lean/Init.olean)) " +
          "--archive $(location Lean_Init_Std.zip) --manifest $(location Lean_Init_Std.json)",
    tools = [
        ":PackageRuntime",
        "@lean4_toolchain//:LeanCompilerRuntime",
        "@lean4_toolchain//:bin/lean",
    ],
)

apple_resource_group(
    name = "LeanRuntimeResources",
    resources = [":LeanRuntimeArchive"],
)

swift_library(
    name = "LeanResources",
    srcs = ["LeanRuntimeResources.swift"],
    generates_header = True,
    module_name = "LeanResources",
    deps = [
        ":LeanBridge",
        "//Vendors/ZIPFoundation",
    ],
)

swift_test(
    name = "LeanResourcesTests",
    srcs = glob(["Tests/*.swift"]),
    deps = [
        ":LeanResources",
        "//Vendors/ZIPFoundation",
    ],
)

cc_library(
    name = "LeanConfiguration",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
)

genrule(
    name = "CheckLeanBridgeGenerated",
    srcs = ["CheckLeanBridge.lean"],
    outs = ["CheckLeanBridgeGenerated.c"],
    cmd = "mkdir -p $(@D)/lean-src && " +
          "cp $(execpath CheckLeanBridge.lean) $(@D)/lean-src/CheckLeanBridge.lean && " +
          "$(execpath @lean4_toolchain//:bin/lean) -R $(@D)/lean-src " +
          "-c $@ $(@D)/lean-src/CheckLeanBridge.lean",
    tools = [
        "@lean4_toolchain//:LeanCompilerRuntime",
        "@lean4_toolchain//:bin/lean",
    ],
)

cc_library(
    name = "CheckLeanBridgeObject",
    srcs = [":CheckLeanBridgeGenerated"],
    copts = [
        "-w",
    ],
    deps = [
        ":LeanConfiguration",
        "@lean4//:LeanNative",
    ],
)

cc_library(
    name = "LeanBridge",
    hdrs = ["LeanBridge.h"],
    additional_linker_inputs = ["lean-symbol-roots.rsp"],
    aspect_hints = [":LeanBridgeSwiftInterop"],
    linkopts = ["-Wl,@$(location lean-symbol-roots.rsp)"],
    deps = [":LeanRuntime"],
)

cc_import(
    name = "LeanRuntime",
    static_library = ":LeanEncapsulatedArchive",
)

lean_encapsulated_archive(
    name = "LeanEncapsulatedArchive",
    roots = "lean-symbol-roots.rsp",
    symbols = "lean-symbols.txt",
    deps = [":LeanBridgeInternal"],
)

cc_library(
    name = "LeanBridgeInternal",
    srcs = ["LeanBridge.cpp"],
    hdrs = [
        "LeanBridge.h",
        ":LeanShellOptions",
    ],
    copts = ["-std=c++20"],
    deps = [
        ":CheckLeanBridgeObject",
        ":LeanConfiguration",
        ":LeanPlatformStubs",
        "@lean4//:LeanGenerated",
    ],
)

cc_library(
    name = "LeanPlatformStubs",
    srcs = ["LeanPlatformStubs.cpp"],
    deps = ["@lean4//:LeanNative"],
)

py_binary(
    name = "GenerateSymbolRoots",
    srcs = ["generate_symbol_roots.py"],
    main = "generate_symbol_roots.py",
)

py_test(
    name = "LeanSymbolRootsTests",
    srcs = [
        "Tests/test_symbol_roots.py",
        "generate_symbol_roots.py",
    ],
    imports = ["."],
    main = "Tests/test_symbol_roots.py",
)

genrule(
    name = "LeanSymbolRoots",
    srcs = [
        "exported_symbols.txt",
        ":LeanBridgeInternal",
        ":LeanPlatformStubs",
        "@lean4//:LeanGenerated",
        "@lean4//:LeanNative",
    ],
    outs = [
        "lean-symbols.txt",
        "lean-symbol-roots.rsp",
    ],
    cmd = "$(execpath :GenerateSymbolRoots) --allowlist $(location exported_symbols.txt) " +
          "--symbols $(location lean-symbols.txt) --response $(location lean-symbol-roots.rsp) " +
          "$(locations :LeanBridgeInternal) $(locations :LeanPlatformStubs) " +
          "$(locations @lean4//:LeanGenerated) $(locations @lean4//:LeanNative)",
    tools = [":GenerateSymbolRoots"],
)

py_binary(
    name = "GenerateShellOptions",
    srcs = ["generate_shell_options.py"],
    main = "generate_shell_options.py",
)

objc_library(
    name = "LeanCommand",
    srcs = ["LeanCommand.mm"],
    hdrs = ["LeanCommand.h"],
    copts = ["-std=c++20"],
    module_name = "LeanCommand",
    deps = [
        ":LeanBridge",
        ":LeanResources",
        "//Vendors/ios_system:api",
    ],
)

objc_library(
    name = "LeanCommandValidationMain",
    srcs = ["Tests/LeanCommandTests.m"],
    deps = [
        ":LeanBridge",
        ":LeanCommand",
        "//Vendors/ios_system",
    ],
)

macos_command_line_application(
    name = "LeanCommandValidation",
    bundle_id = "com.liuliu.lean-command-validation",
    minimum_os_version = "13.5",
    deps = [
        ":LeanCommandValidationMain",
        "//Vendors/NodeJS:NodeMobile",
        "//Vendors/PythonIOS:PythonRuntime",
    ],
)

sh_test(
    name = "LeanEncapsulationTests",
    srcs = ["Tests/encapsulate-lean-test.sh"],
    args = ["$(location encapsulate_lean.sh)"],
    data = glob(["Tests/Encapsulation/*"]) + [
        "encapsulate_lean.sh",
        "include/LeanAllocatorPolicy.h",
    ],
    tags = ["requires-darwin"],
)

sh_test(
    name = "LeanCommandTests",
    timeout = "long",
    srcs = ["Tests/run-command-tests.sh"],
    args = [
        "$(location :LeanCommandValidation)",
        "$(location Lean_Init_Std.zip)",
        "$(location lean-symbols.txt)",
    ],
    data = [
        "Lean_Init_Std.json",
        "Lean_Init_Std.zip",
        "lean-symbols.txt",
        ":LeanCommandValidation",
    ],
)

genrule(
    name = "LeanShellOptions",
    srcs = ["@lean4//:src/util/shell.cpp"],
    outs = ["LeanShellOptions.inc"],
    cmd = "$(execpath :GenerateShellOptions) $(location @lean4//:src/util/shell.cpp) $@",
    tools = [":GenerateShellOptions"],
)

swift_interop_hint(
    name = "LeanBridgeSwiftInterop",
    module_name = "LeanBridge",
)

exports_files([
    "Tests/run-command-tests.sh",
    "include/LeanAllocatorPolicy.h",
    "apple.patch",
    "lean4.BUILD.bazel",
    "lean4_toolchain.BUILD.bazel",
    "gmp.BUILD.bazel",
    "libuv.BUILD.bazel",
    "mimalloc.BUILD.bazel",
    "repositories.bzl",
])
