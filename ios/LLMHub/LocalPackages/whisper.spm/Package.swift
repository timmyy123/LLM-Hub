// swift-tools-version:5.3
import PackageDescription

// old target
//.target(name: "whisper", dependencies:[], cSettings: [
//    .unsafeFlags([
//        "-O3",
//        "-fno-objc-arc",
//        "-DGGML_USE_ACCELERATE",
//        "-DGGML_USE_METAL",
//        "-DWHISPER_USE_COREML",
//        "-DWHISPER_COREML_ALLOW_FALLBACK"])
//]),

#if arch(arm) || arch(arm64)
let platforms: [SupportedPlatform]? = [
    .macOS(.v11),
    .iOS(.v14),
    .watchOS(.v4),
    .tvOS(.v14)
]

// TODO: make Metal work - I can't figure out how to build ggml-metal.m
//       it keeps giving an error that NSString is unknown type name
//       in llama.cpp it somehow works ..
//       if someone can figure this out, please let me know or open a PR
let exclude: [String] = ["Sources/whisper/ggml-metal.m", "Sources/whisper/ggml-metal.metal"]
let additionalSources: [String] = []
let additionalSettings: [CSetting] = []

//let exclude: [String] = []
//let additionalSources: [String] = ["Sources/whisper/ggml-metal.m"]
//let additionalSettings: [CSetting] = [
//    .unsafeFlags(["-fno-objc-arc"]),
//    .define("GGML_SWIFT"),
//    .define("GGML_USE_METAL")
//]
#else
let platforms: [SupportedPlatform]? = nil
let exclude: [String] = ["Sources/whisper/ggml-metal.m", "Sources/whisper/ggml-metal.metal"]
let additionalSources: [String] = []
let additionalSettings: [CSetting] = []
#endif

let package = Package(
    name: "whisper.spm",
    platforms: platforms,
    products: [
        .library(
            name: "whisper",
            targets: ["whisper"])
    ],
    targets: [
        .target(
            name: "whisper",
            path: ".",
            exclude: exclude,
            sources: [
                "Sources/whisper/ggml.c",
                "Sources/whisper/ggml-alloc.c",
                "Sources/whisper/ggml-backend.c",
                "Sources/whisper/ggml-metal.m",
                "Sources/whisper/ggml-quants.c",
                "Sources/whisper/coreml/whisper-encoder-impl.m",
                "Sources/whisper/coreml/whisper-encoder.mm",
                "Sources/whisper/whisper.cpp",
            ] + additionalSources,
            publicHeadersPath: "Sources/whisper/include",
            cSettings: [
                .unsafeFlags(["-Wno-shorten-64-to-32", "-fvisibility=hidden"]),
                .define("GGML_USE_ACCELERATE"),
                .define("WHISPER_SHARED"),
                .define("WHISPER_USE_COREML"),
                .define("WHISPER_COREML_ALLOW_FALLBACK")
            ] + additionalSettings,
            cxxSettings: [
                .unsafeFlags(["-fvisibility=hidden"]),
                .define("WHISPER_SHARED")
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        ),
        .target(name: "test-objc",  dependencies:["whisper"]),
        .target(name: "test-swift", dependencies:["whisper"])
    ],
    cxxLanguageStandard: CXXLanguageStandard.cxx11
)
