load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_library")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_push", "oci_tarball")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "gRPCServerCLI_layer",
    srcs = [
        "//Apps:gRPCServerCLI",
    ],
    package_dir = "/usr/local/bin",
)

pkg_tar(
    name = "launch_script_layer",
    srcs = [
        "//linux-script:launch_gpu_servers.sh",
    ],
    mode = "0755",
    package_dir = "/usr/local/bin",
)

oci_image(
    name = "gRPCServerCLI_image",
    base = "@cuda12.8.1_swift6.0.3_ubuntu22.04",
    tars = [
        ":gRPCServerCLI_layer",
    ],
)

oci_tarball(
    name = "gRPCServerCLI_tar",
    image = ":gRPCServerCLI_image",
    repo_tags = ["draw-things-grpc-server-cli:latest"],
)

oci_push(
    name = "push_gRPCServerCLI_image",
    image = ":gRPCServerCLI_image",
    remote_tags = ["latest"],
    repository = "docker.io/drawthingsai/draw-things-grpc-server-cli",
)

pkg_tar(
    name = "ProxyServiceCLI_layer",
    srcs = [
        "//Apps:ProxyServiceCLI",
    ],
    package_dir = "/usr/local/bin",
)

pkg_tar(
    name = "ProxyServerControlPanelCLI_layer",
    srcs = [
        "//Apps:ProxyServerControlPanelCLI",
    ],
    package_dir = "/usr/local/bin",
)

oci_image(
    name = "ProxyServiceCLI_image",
    base = "@dt_swift6.0.3_ubuntu22.04",
    tars = [
        ":ProxyServiceCLI_layer",
        ":ProxyServerControlPanelCLI_layer",
    ],
)

oci_tarball(
    name = "ProxyServiceCLI_tar",
    image = ":ProxyServiceCLI_image",
    repo_tags = ["draw-things-proxy-server-cli:latest"],
)

oci_push(
    name = "push_ProxyServiceCLI_image",
    image = ":ProxyServiceCLI_image",
    remote_tags = [
        "latest",
    ],
    repository = "docker.io/drawthingsai/draw-things-proxy-server-cli",
)
