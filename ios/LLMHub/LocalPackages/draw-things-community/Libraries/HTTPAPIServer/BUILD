load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "HTTPAPIServer",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "HTTPAPIServer",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "//Libraries/HTTPServer",
        "//Libraries/ImageGenerator",
        "//Libraries/Invocation",
        "//Libraries/ModelZoo",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Utils",
        "@SwiftArgumentParser//:ArgumentParser",
        "@s4nnc//nnc",
    ],
)
