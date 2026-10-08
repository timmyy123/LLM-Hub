load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")

swift_library(
    name = "SwiftProtobuf",
    srcs = glob([
        "Sources/SwiftProtobuf/**/*.swift",
    ]),
    module_name = "SwiftProtobuf",
    visibility = ["//visibility:public"],
    deps = [],
)

swift_library(
    name = "SwiftProtobufPluginLibrary",
    srcs = glob([
        "Sources/SwiftProtobufPluginLibrary/**/*.swift",
    ]),
    module_name = "SwiftProtobufPluginLibrary",
    visibility = ["//visibility:public"],
    deps = [":SwiftProtobuf"],
)

swift_binary(
    name = "protoc-gen-swift",
    srcs = glob([
        "Sources/protoc-gen-swift/**/*.swift",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":SwiftProtobuf",
        ":SwiftProtobufPluginLibrary",
    ],
)
