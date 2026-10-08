load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

swift_library(
    name = "PromptJSONExpansion",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "PromptJSONExpansion",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/BinaryResources",
        "//Libraries/SwiftLLM",
        "//Libraries/Tokenizer",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "PromptJSONExpansionTests",
    srcs = glob(["Tests/**/*.swift"]),
    deps = [
        ":PromptJSONExpansion",
        "//Libraries/SwiftLLM",
        "@s4nnc//nnc",
    ],
)
