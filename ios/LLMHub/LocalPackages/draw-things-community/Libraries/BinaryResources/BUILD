load("@build_bazel_rules_swift//swift:swift.bzl", "swift_interop_hint", "swift_library")
load("//Tools:Packager/packager.bzl", "resource_library")

resource_library(
    name = "Resources",
    srcs = [
        "Resources/bpe_simple_vocab_16e6.txt",
        "Resources/chatglm3.spiece.model",
        "Resources/gemma3.spiece.model",
        "Resources/isrgrootx1.pem",
        "Resources/merges.txt",
        "Resources/merges_deepseek4.txt",
        "Resources/merges_llama3.txt",
        "Resources/merges_mistral3.txt",
        "Resources/merges_qwen2.5.txt",
        "Resources/merges_qwen3.5.txt",
        "Resources/merges_qwen3.txt",
        "Resources/pile_t5.spiece.model",
        "Resources/root_ca.crt",
        "Resources/server_crt.crt",
        "Resources/server_key.key",
        "Resources/t5.spiece.model",
        "Resources/umt5.spiece.model",
        "Resources/vocab.json",
        "Resources/vocab_16e6.json",
        "Resources/vocab_deepseek4.json",
        "Resources/vocab_llama3.json",
        "Resources/vocab_mistral3.json",
        "Resources/vocab_qwen2.5.json",
        "Resources/vocab_qwen3.5.json",
        "Resources/vocab_qwen3.json",
        "Resources/xlmroberta.bpe.model",
    ],
)

cc_library(
    name = "CResources",
    hdrs = ["Sources/Module.h"],
    aspect_hints = [":C_Resources_swift_interop"],
    tags = ["swift_module=C_Resources"],
    deps = [":Resources"],
)

swift_interop_hint(
    name = "C_Resources_swift_interop",
    module_name = "C_Resources",
)

swift_library(
    name = "BinaryResources",
    srcs = glob(["Sources/**/*.swift"]),
    module_name = "BinaryResources",
    visibility = ["//visibility:public"],
    deps = [":CResources"],
)
