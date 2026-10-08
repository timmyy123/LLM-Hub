load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

swift_library(
    name = "Trainer",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "Trainer",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "//Libraries/SFMT",
        "//Libraries/SwiftDiffusion",
        "@dflat//:SQLiteDflat",
        "@s4nnc//nnc",
        "@s4nnc//tensorboard",
    ],
)

swift_test(
    name = "LoRATrainerCheckpointTests",
    srcs = ["Tests/LoRATrainerCheckpointTests.swift"],
    deps = [
        ":Trainer",
        "//Libraries/SwiftDiffusion",
        "@s4nnc//nnc",
    ],
)
