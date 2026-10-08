load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

exports_files([
    "BUILD.oils",
    "detected-cpp-config.h",
    "osh-ios.patch",
])

cc_library(
    name = "api",
    hdrs = ["osh_ios.h"],
    strip_include_prefix = "/",
)

cc_test(
    name = "OSHConformanceTests",
    srcs = ["OSHConformanceTests.cc"],
    deps = ["@oils_for_unix//:osh_ios"],
)
