load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Scripting",
    srcs = glob(
        [
            "Sources/ScriptExecutor.swift",
            "Sources/SharedScript.swift",
            "Sources/ScriptZoo.swift",
        ],
    ),
    module_name = "Scripting",
    visibility = ["//visibility:public"],
    deps = [
        ":ScriptDataModels",
        "//Libraries/DataModels",
        "//Libraries/ImageSegmentation",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Utils",
        "@s4nnc//nnc",
    ],
)

swift_library(
    name = "ScriptDataModels",
    srcs = [
        "Sources/ScriptModels.swift",
    ],
    module_name = "ScriptDataModels",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "@s4nnc//nnc",
    ],
)
