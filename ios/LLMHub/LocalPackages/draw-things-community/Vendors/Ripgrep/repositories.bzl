load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@rules_rust//crate_universe:defs.bzl", "crates_repository")
load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")
load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

_RIPGREP_TARGET_TRIPLES = [
    "aarch64-apple-darwin",
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
    "x86_64-apple-darwin",
    "x86_64-apple-ios",
]

_RUST_EXTRA_TARGET_TRIPLES = [
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
    "x86_64-apple-ios",
]

def ripgrep_repositories():
    rules_rust_dependencies()
    crate_universe_dependencies(rust_version = "1.89.0")

    rust_register_toolchains(
        edition = "2024",
        extra_target_triples = _RUST_EXTRA_TARGET_TRIPLES,
        versions = ["1.89.0"],
    )

    crates_repository(
        name = "ripgrep_crates",
        cargo_lockfile = "//Vendors/Ripgrep:Cargo.lock",
        lockfile = "//Vendors/Ripgrep:cargo-bazel-lock.json",
        manifests = ["//Vendors/Ripgrep:Cargo.toml"],
        rust_version = "1.89.0",
        supported_platform_triples = _RIPGREP_TARGET_TRIPLES,
    )

    http_archive(
        name = "ripgrep",
        build_file = "//Vendors/Ripgrep:ripgrep.BUILD.bazel",
        patch_args = ["-p1"],
        patches = ["//Vendors/Ripgrep:ios-system.patch"],
        sha256 = "4dad02a2f9c8c3c8d89434e47337aa654cb0e2aa50e806589132f186bf5c2b66",
        strip_prefix = "ripgrep-14.1.1",
        urls = ["https://github.com/BurntSushi/ripgrep/archive/refs/tags/14.1.1.tar.gz"],
    )
