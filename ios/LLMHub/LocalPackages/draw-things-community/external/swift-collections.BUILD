load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "InternalCollectionsUtilities",
    srcs = glob([
        "Sources/InternalCollectionsUtilities/**/*.swift",
    ]),
    module_name = "InternalCollectionsUtilities",
)

swift_library(
    name = "BitCollections",
    srcs = glob([
        "Sources/BitCollections/**/*.swift",
    ]),
    module_name = "BitCollections",
    deps = [
        ":InternalCollectionsUtilities",
    ],
)

swift_library(
    name = "DequeModule",
    srcs = glob([
        "Sources/DequeModule/**/*.swift",
    ]),
    module_name = "DequeModule",
    deps = [
        ":InternalCollectionsUtilities",
    ],
)

swift_library(
    name = "HeapModule",
    srcs = glob([
        "Sources/HeapModule/**/*.swift",
    ]),
    module_name = "HeapModule",
    deps = [
        ":InternalCollectionsUtilities",
    ],
)

swift_library(
    name = "OrderedCollections",
    srcs = glob([
        "Sources/OrderedCollections/**/*.swift",
    ]),
    module_name = "OrderedCollections",
    deps = [
        ":InternalCollectionsUtilities",
    ],
)

swift_library(
    name = "HashTreeCollections",
    srcs = glob([
        "Sources/HashTreeCollections/**/*.swift",
    ]),
    module_name = "HashTreeCollections",
    deps = [
        ":InternalCollectionsUtilities",
    ],
)

swift_library(
    name = "Collections",
    srcs = glob([
        "Sources/Collections/**/*.swift",
    ]),
    module_name = "Collections",
    visibility = ["//visibility:public"],
    deps = [
        ":BitCollections",
        ":DequeModule",
        ":HashTreeCollections",
        ":HeapModule",
        ":OrderedCollections",
    ],
)
