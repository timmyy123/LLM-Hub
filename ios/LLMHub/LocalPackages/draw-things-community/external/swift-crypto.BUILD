load(
    "@build_bazel_rules_swift//swift:swift.bzl",
    "swift_interop_hint",
    "swift_library",
)

cc_library(
    name = "CCryptoBoringSSL",
    srcs = glob([
        "Sources/CCryptoBoringSSL/**/*.h",
        "Sources/CCryptoBoringSSL/**/*.c",
        "Sources/CCryptoBoringSSL/**/*.cc",
        "Sources/CCryptoBoringSSL/**/*.S",
        "Sources/CCryptoBoringSSL/**/*.inc",
    ]),
    hdrs = glob([
        "Sources/CCryptoBoringSSL/include/**/*.h",
    ]),
    aspect_hints = [":CCryptoBoringSSL_swift_interop"],
    # This pinned SwiftCrypto revision omitted these two socket BIO exports
    # from its generated prefix list. Keep them isolated from other libcrypto
    # implementations that may be linked into the same app.
    copts = [
        "-DBIO_new_socket=CCryptoBoringSSL_BIO_new_socket",
        "-DBIO_s_socket=CCryptoBoringSSL_BIO_s_socket",
    ],
    includes = ["Sources/CCryptoBoringSSL/include"],
    tags = ["swift_module=CCryptoBoringSSL"],
    visibility = ["//visibility:public"],
    deps = [],
)

swift_interop_hint(
    name = "CCryptoBoringSSL_swift_interop",
    module_name = "CCryptoBoringSSL",
)

cc_library(
    name = "CCryptoBoringSSLShims",
    srcs = glob([
        "Sources/CCryptoBoringSSLShims/**/*.c",
    ]),
    hdrs = glob([
        "Sources/CCryptoBoringSSLShims/include/**/*.h",
    ]),
    aspect_hints = [":CCryptoBoringSSLShims_swift_interop"],
    copts = [],
    includes = ["Sources/CCryptoBoringSSLShims/include"],
    tags = ["swift_module=CCryptoBoringSSLShims"],
    visibility = ["//visibility:public"],
    deps = [":CCryptoBoringSSL"],
)

swift_interop_hint(
    name = "CCryptoBoringSSLShims_swift_interop",
    module_name = "CCryptoBoringSSLShims",
)

swift_library(
    name = "CryptoBoringWrapper",
    srcs = glob(
        [
            "Sources/CryptoBoringWrapper/**/*.swift",
        ],
        exclude = [
            "Sources/CryptoBoringWrapper/CMakeLists.txt",
            "Sources/CryptoBoringWrapper/PrivacyInfo.xcprivacy",
        ],
    ),
    defines = [
        "CRYPTO_IN_SWIFTPM",
        "CRYPTO_IN_SWIFTPM_FORCE_BUILD_API",
    ],
    module_name = "CryptoBoringWrapper",
    visibility = ["//visibility:public"],
    deps = [
        ":CCryptoBoringSSL",
        ":CCryptoBoringSSLShims",
    ],
)

swift_library(
    name = "Crypto",
    srcs = glob(
        [
            "Sources/Crypto/**/*.swift",
        ],
        exclude = [
            "Sources/Crypto/CMakeLists.txt",
            "Sources/Crypto/AEADs/Nonces.swift.gyb",
            "Sources/Crypto/Digests/Digests.swift.gyb",
            "Sources/Crypto/Key Agreement/ECDH.swift.gyb",
            "Sources/Crypto/Signatures/ECDSA.swift.gyb",
            "Sources/Crypto/PrivacyInfo.xcprivacy",
        ],
    ),
    defines = [
        "MODULE_IS_CRYPTO",
    ],
    module_name = "Crypto",
    visibility = ["//visibility:public"],
    deps = [
        ":CCryptoBoringSSL",
        ":CCryptoBoringSSLShims",
        ":CryptoBoringWrapper",
    ],
)

swift_library(
    name = "_CryptoExtras",
    srcs = glob(
        [
            "Sources/Crypto/CryptoKitErrors.swift",
            "Sources/_CryptoExtras/**/*.swift",
        ],
        exclude = [
            "Sources/_CryptoExtras/CMakeLists.txt",
        ],
    ),
    defines = [
        "MODULE_IS_CRYPTO",
    ],
    module_name = "_CryptoExtras",
    visibility = ["//visibility:public"],
    deps = [
        ":CCryptoBoringSSL",
        ":CCryptoBoringSSLShims",
        ":Crypto",
        ":CryptoBoringWrapper",
    ],
)
