load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "RemoteImageGenerator",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "RemoteImageGenerator",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/GRPC:GRPCImageServiceModels",
        "//Libraries/GRPC:GRPCServer",
        "//Libraries/ImageGenerator",
        "@grpc-swift//:GRPC",
    ],
)
