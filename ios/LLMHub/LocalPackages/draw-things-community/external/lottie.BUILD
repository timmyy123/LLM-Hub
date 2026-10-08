load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

swift_library(
    name = "Lottie",
    srcs = glob(["Sources/**/*.swift"], exclude = ["Sources/Private/EmbeddedLibraries/EpoxyCore/SwiftUI/**/*.swift", "Sources/Public/Controls/LottieButton.swift", "Sources/Public/Controls/LottieSwitch.swift", "Sources/Public/Animation/LottieView.swift"]),
    module_name = "Lottie",
    visibility = ["//visibility:public"],
)
