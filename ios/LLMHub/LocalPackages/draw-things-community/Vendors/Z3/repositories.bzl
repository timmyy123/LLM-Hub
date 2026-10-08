load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


def z3_repositories():
    http_archive(
        name = "z3",
        build_file = "//Vendors/Z3:z3.BUILD.bazel",
        sha256 = "dae526252cb0585c8c863292ebec84cace4901a014b190a73f14087dd08d252b",
        strip_prefix = "z3-z3-4.15.4",
        urls = ["https://github.com/Z3Prover/z3/archive/refs/tags/z3-4.15.4.tar.gz"],
    )
