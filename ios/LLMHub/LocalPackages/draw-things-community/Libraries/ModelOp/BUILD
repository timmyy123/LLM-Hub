load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

swift_library(
    name = "ModelOp",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "ModelOp",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "QwenImage2_1LoRATests",
    srcs = ["Tests/QwenImage2_1LoRATests.swift"],
    tags = [
        "manual",
        "requires-gpu",
    ],
    deps = [
        ":ModelOp",
        "//Libraries/ModelZoo",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "MiniMaxH3LoRATests",
    srcs = ["Tests/MiniMaxH3LoRATests.swift"],
    tags = [
        "manual",
        "requires-gpu",
    ],
    deps = [
        ":ModelOp",
        "//Libraries/ModelZoo",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)
