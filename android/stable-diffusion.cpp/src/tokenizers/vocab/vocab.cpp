#include "vocab.h"
#include "t5.hpp"

// LLM Hub only ships LTX-Video (T5-XXL tokenizer); the other embedded vocabularies (~120 MB of
// source) were removed from this vendored copy. Their loaders return empty strings.

std::string load_clip_merges() {
    return {};
}

std::string load_qwen2_merges() {
    return {};
}

std::string load_mistral_merges() {
    return {};
}

std::string load_mistral_vocab_json() {
    return {};
}

std::string load_t5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(t5_tokenizer_json_str), sizeof(t5_tokenizer_json_str));
    return json_str;
}

std::string load_umt5_tokenizer_json() {
    return {};
}

std::string load_gemma_merges() {
    return {};
}

std::string load_gemma_vocab_json() {
    return {};
}
