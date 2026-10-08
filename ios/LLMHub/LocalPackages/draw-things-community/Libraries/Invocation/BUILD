load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Invocation",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "Invocation",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/LocalImageGenerator",
        "//Libraries/Localization",
        "//Libraries/ModelZoo",
        "//Libraries/Scripting:ScriptDataModels",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Utils",
        "@SwiftGuaka//:Guaka",
        "@s4nnc//nnc",
    ],
)
