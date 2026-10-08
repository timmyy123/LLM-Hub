package(default_visibility = ["//visibility:public"])

exports_files(["ios.patch"])

cc_library(
    name = "boringssl_headers",
    hdrs = glob(["BoringSSL/openssl/*.h"]),
    includes = ["BoringSSL"],
    deps = ["@SwiftCrypto//:CCryptoBoringSSL"],
)

cc_library(
    name = "config",
    hdrs = [
        "Config/config.h",
        "Config/ios_compat_config.h",
    ],
    includes = ["Config"],
)

cc_library(
    name = "compat_headers",
    textual_hdrs = [
        "CompatInclude/OpenSSHPort.h",
        "CompatInclude/endian.h",
        "CompatInclude/glob.h",
        "CompatInclude/netgroup.h",
        "CompatInclude/nlist.h",
        "CompatInclude/readpassphrase.h",
        "CompatInclude/sys/queue.h",
        "CompatInclude/sys/tree.h",
    ],
    includes = ["CompatInclude"],
    deps = ["//Vendors/ios_system:api"],
)
