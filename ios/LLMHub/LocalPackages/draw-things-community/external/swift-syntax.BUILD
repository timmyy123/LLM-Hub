load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "SwiftSyntax",
    srcs = glob(
        [
            "Sources/SwiftSyntax/**/*.swift",
        ],
        exclude = ["Sources/SwiftSyntax/Documentation.docc/**/*.swift"],
    ),
    module_name = "SwiftSyntax",
    visibility = ["//visibility:public"],
    deps = [],
)

swift_library(
    name = "SwiftBasicFormat",
    srcs = glob([
        "Sources/SwiftBasicFormat/**/*.swift",
    ]),
    module_name = "SwiftBasicFormat",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftSyntax",
    ],
)

swift_library(
    name = "SwiftDiagnostics",
    srcs = glob([
        "Sources/SwiftDiagnostics/**/*.swift",
    ]),
    module_name = "SwiftDiagnostics",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftSyntax",
    ],
)

swift_library(
    name = "SwiftParser",
    srcs = glob([
        "Sources/SwiftParser/**/*.swift",
    ]),
    module_name = "SwiftParser",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiagnostics",
        ":SwiftSyntax",
    ],
)

swift_library(
    name = "SwiftParserDiagnostics",
    srcs = glob([
        "Sources/SwiftParserDiagnostics/**/*.swift",
    ]),
    module_name = "SwiftParserDiagnostics",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftBasicFormat",
        ":SwiftDiagnostics",
        ":SwiftParser",
        ":SwiftSyntax",
    ],
)

swift_library(
    name = "SwiftSyntaxBuilder",
    srcs = glob([
        "Sources/SwiftSyntaxBuilder/**/*.swift",
    ]),
    module_name = "SwiftSyntaxBuilder",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftBasicFormat",
        ":SwiftParser",
        ":SwiftParserDiagnostics",
        ":SwiftSyntax",
    ],
)

swift_library(
    name = "SwiftSyntaxMacros",
    srcs = glob([
        "Sources/SwiftSyntaxMacros/**/*.swift",
    ]),
    module_name = "SwiftSyntaxMacros",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiagnostics",
        ":SwiftParser",
        ":SwiftSyntax",
        ":SwiftSyntaxBuilder",
    ],
)

swift_library(
    name = "SwiftOperators",
    srcs = glob([
        "Sources/SwiftOperators/**/*.swift",
    ]),
    module_name = "SwiftOperators",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiagnostics",
        ":SwiftParser",
        ":SwiftSyntax",
    ],
)

swift_library(
    name = "SwiftSyntaxMacroExpansion",
    srcs = glob([
        "Sources/SwiftSyntaxMacroExpansion/**/*.swift",
    ]),
    module_name = "SwiftSyntaxMacroExpansion",
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftDiagnostics",
        ":SwiftOperators",
        ":SwiftSyntax",
        ":SwiftSyntaxBuilder",
        ":SwiftSyntaxMacros",
    ],
)
