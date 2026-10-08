load("@build_bazel_rules_swift//swift:swift.bzl", "swift_binary", "swift_interop_hint", "swift_library", "swift_test")
load("@bazel_skylib//lib:selects.bzl", "selects")

config_setting(
    name = "macos_build",
    constraint_values = [
        "@platforms//os:osx",
    ],
)

config_setting(
    name = "ios_build",
    constraint_values = [
        "@platforms//os:ios",
    ],
)

selects.config_setting_group(
    name = "ios_or_macos_build",
    match_any = [
        ":macos_build",
        ":ios_build",
    ],
)

cc_library(
    name = "C_DNSAdvertiser",
    srcs = ["Advertiser/Sources/dnssd_advertiser.c"],
    hdrs = ["Advertiser/Sources/dnssd_advertiser.h"],
    aspect_hints = [":C_DNSAdvertiser_swift_interop"],
    tags = ["swift_module=C_DNSAdvertiser"],
)

swift_interop_hint(
    name = "C_DNSAdvertiser_swift_interop",
    module_name = "C_DNSAdvertiser",
)

swift_library(
    name = "GRPCServer",
    srcs = [
        "Server/Sources/GRPCFileUploader.swift",
        "Server/Sources/GRPCHostnameUtils.swift",
        "Server/Sources/GRPCServerAdvertiser.swift",
        "Server/Sources/ImageGenerationClientWrapper.swift",
        "Server/Sources/ImageGenerationServiceImpl.swift",
        "Server/Sources/ProtectedValue.swift",
    ] + select({
        ":ios_or_macos_build": [
            "Server/Sources/GRPCServiceBrowser.swift",
        ],
        "//conditions:default": [
        ],
    }),
    module_name = "GRPCServer",
    visibility = ["//visibility:public"],
    deps = [
        ":GRPCImageServiceModels",
        ":ServerConfigurationRewriter",
        "//Libraries/BinaryResources",
        "//Libraries/DataModels",
        "//Libraries/ImageGenerator",
        "//Libraries/ModelZoo",
        "//Libraries/Scripting:ScriptDataModels",
        "//Libraries/SwiftDiffusion",
        "//Libraries/Utils",
        "@SwiftArgumentParser//:ArgumentParser",
        "@SwiftCrypto//:CCryptoBoringSSL",
        "@SwiftCrypto//:CCryptoBoringSSLShims",
        "@SwiftCrypto//:Crypto",
        "@SwiftCrypto//:CryptoBoringWrapper",
        "@SwiftLog//:Logging",
        "@s4nnc//nnc",
    ] + select({
        ":ios_or_macos_build": [
        ],
        "//conditions:default": [
            ":C_DNSAdvertiser",
        ],
    }),
)

swift_library(
    name = "ProxyControlClient",
    srcs = glob([
        "ProxyControlClient/Sources/*.swift",
    ]),
    module_name = "ProxyControlClient",
    visibility = ["//visibility:public"],
    deps = [
        ":GRPCControlPanelModels",
        ":GRPCImageServiceModels",
        "//Libraries/BinaryResources",
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "@SwiftArgumentParser//:ArgumentParser",
        "@SwiftCrypto//:CCryptoBoringSSL",
        "@SwiftCrypto//:CCryptoBoringSSLShims",
        "@SwiftCrypto//:Crypto",
        "@SwiftCrypto//:CryptoBoringWrapper",
        "@SwiftLog//:Logging",
    ],
)

swift_library(
    name = "ProxyServer",
    srcs = glob([
        "ProxyServer/Sources/*.swift",
    ]),
    module_name = "ProxyServer",
    visibility = ["//visibility:public"],
    deps = [
        ":GRPCControlPanelModels",
        ":GRPCImageServiceModels",
        "//Libraries/BinaryResources",
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "@SwiftArgumentParser//:ArgumentParser",
        "@SwiftCrypto//:CCryptoBoringSSL",
        "@SwiftCrypto//:CCryptoBoringSSLShims",
        "@SwiftCrypto//:Crypto",
        "@SwiftCrypto//:CryptoBoringWrapper",
        "@SwiftLog//:Logging",
    ],
)

swift_library(
    name = "GRPCControlPanelModels",
    srcs = glob(["Models/Sources/controlPanel/*.swift"]),
    module_name = "GRPCControlPanelModels",
    visibility = ["//visibility:public"],
    deps = [
        "@grpc-swift//:GRPC",
    ],
)

swift_library(
    name = "ServerConfigurationRewriter",
    srcs = glob(["ServerConfigurationRewriter/Sources/*.swift"]),
    module_name = "ServerConfigurationRewriter",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/DataModels",
        "@grpc-swift//:GRPC",
    ],
)

swift_library(
    name = "ServerLoRALoader",
    srcs = glob(["ServerLoRALoader/Sources/*.swift"]),
    module_name = "ServerLoRALoader",
    visibility = ["//visibility:public"],
    deps = [
        ":ServerConfigurationRewriter",
        "//Libraries/DataModels",
        "//Libraries/ModelZoo",
        "@SwiftCrypto//:Crypto",
        "@SwiftCrypto//:CryptoBoringWrapper",
        "@SwiftLog//:Logging",
        "@grpc-swift//:GRPC",
    ],
)

swift_library(
    name = "GRPCImageServiceModels",
    srcs = glob(["Models/Sources/imageService/*.swift"]),
    module_name = "GRPCImageServiceModels",
    visibility = ["//visibility:public"],
    deps = [
        "@grpc-swift//:GRPC",
    ],
)
