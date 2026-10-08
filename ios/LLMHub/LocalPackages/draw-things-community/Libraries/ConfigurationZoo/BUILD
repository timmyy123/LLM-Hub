load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "ConfigurationZoo",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "ConfigurationZoo",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "//Libraries/Scripting:ScriptDataModels",
    ],
)
