load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("@dflat//:dflat.bzl", "dflatc")

dflatc(
    name = "config_schema",
    srcs = ["Sources/config.fbs"],
)

dflatc(
    name = "estimation_schema",
    srcs = ["Sources/estimation.fbs"],
)

dflatc(
    name = "mixing_schema",
    srcs = ["Sources/mixing.fbs"],
)

dflatc(
    name = "lora_trainer_schema",
    srcs = ["Sources/lora.fbs"],
)

dflatc(
    name = "dataset_schema",
    srcs = ["Sources/dataset.fbs"],
)

dflatc(
    name = "paint_color_schema",
    srcs = ["Sources/paint_color.fbs"],
)

dflatc(
    name = "peer_connection_id_schema",
    srcs = ["Sources/peer_connection_id.fbs"],
)

swift_library(
    name = "DataModels",
    srcs = glob(["Sources/**/*.swift"]) + [
        ":config_schema",
        ":estimation_schema",
        ":mixing_schema",
        ":lora_trainer_schema",
        ":dataset_schema",
        ":paint_color_schema",
        ":peer_connection_id_schema",
    ],
    module_name = "DataModels",
    visibility = ["//visibility:public"],
    deps = [
        "//Libraries/SwiftDiffusion",
        "//Libraries/Utils",
        "@dflat//:SQLiteDflat",
    ],
)
