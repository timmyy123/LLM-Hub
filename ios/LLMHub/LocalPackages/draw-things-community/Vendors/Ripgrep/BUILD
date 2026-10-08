load("@rules_rust//rust:defs.bzl", "rust_static_library", "rust_test")

package(default_visibility = ["//visibility:public"])

objc_library(
    name = "api",
    hdrs = [
        "rg.h",
        "search.h",
    ],
    includes = ["."],
    module_name = "Ripgrep",
)

rust_static_library(
    name = "search",
    srcs = ["search.rs"],
    crate_name = "local_code_ripgrep_search",
    crate_root = "search.rs",
    edition = "2024",
    deps = [
        "@ripgrep_crates//:grep",
        "@ripgrep_crates//:ignore",
    ],
)

rust_test(
    name = "search_tests",
    crate = ":search",
)

exports_files([
    "Cargo.lock",
    "Cargo.toml",
    "cargo-bazel-lock.json",
    "fake.rs",
    "ios-system.patch",
    "repositories.bzl",
    "ripgrep.BUILD.bazel",
])
