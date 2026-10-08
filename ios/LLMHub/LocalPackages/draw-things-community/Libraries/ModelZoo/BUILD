load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library", "swift_test")

swift_library(
    name = "ModelZoo",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "ModelZoo",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/SwiftDiffusion",
        "//Libraries/SwiftLLM",
        "//Libraries/Upscaler",
        "@s4nnc//nnc",
    ],
)

swift_test(
    name = "ModelZooTests",
    srcs = [
        "Tests/ModelZooTests.swift",
    ],
    deps = [
        ":ModelZoo",
    ],
)

swift_test(
    name = "ComputeUnitsTests",
    srcs = [
        "Tests/ComputeUnitsTests.swift",
    ],
    deps = [
        ":ModelZoo",
        "//Libraries/DataModels",
        "//Libraries/SwiftDiffusion",
    ],
)
