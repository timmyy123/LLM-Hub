#include "vocab.h"
#include "t5.hpp"
#include "clip_merges.hpp"

// CLIP supports SD image checkpoints; T5 supports LTX-Video. The other embedded
// vocabularies are omitted from this vendored copy.

std::string load_clip_merges() {
    return std::string(reinterpret_cast<const char*>(clip_merges_utf8_c_str), sizeof(clip_merges_utf8_c_str));
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
