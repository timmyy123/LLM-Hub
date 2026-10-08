load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")

swift_library(
    name = "SwiftFormat",
    srcs = glob([
        "Sources/SwiftFormat/**/*.swift",
    ]),
    module_name = "SwiftFormat",
    deps = [
        "@SwiftMarkdown//:Markdown",
        "@SwiftSyntax//:SwiftOperators",
        "@SwiftSyntax//:SwiftParser",
        "@SwiftSyntax//:SwiftParserDiagnostics",
    ],
)

swift_binary(
    name = "swift-format",
    srcs = glob([
        "Sources/swift-format/**/*.swift",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftFormat",
        "@SwiftArgumentParser//:ArgumentParser",
        "@SwiftSyntax",
        "@SwiftSyntax//:SwiftParser",
        "@SwiftToolsSupportCore//:TSCBasic",
    ],
)
