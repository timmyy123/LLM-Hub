load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library", "swift_test")

swift_library(
    name = "Downloader",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "Downloader",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/ModelZoo",
    ],
)

swift_test(
    name = "ResumableDownloaderTests",
    srcs = [
        "Tests/ResumableDownloaderTests.swift",
    ],
    deps = [
        ":Downloader",
    ],
)

swift_binary(
    name = "DownloaderTestCLI",
    srcs = glob(["TestProgram/*.swift"]),
    visibility = ["//visibility:public"],
    deps = [
        ":Downloader",
    ],
)
