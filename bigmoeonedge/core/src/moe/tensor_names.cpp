#include "tensor_names.h"

namespace bmoe::tensor_names {

namespace {

constexpr std::string_view k_block_prefix = "blk.";
constexpr std::string_view k_weight_tail = ".weight";
// How a block's tensor names say which part of it they belong to, and the marks that place a tensor
// in a multi-token-prediction block or an embedding table wherever they occur in its name.
constexpr std::string_view k_attention_head = "attn";
constexpr std::string_view k_ffn_head = "ffn";
constexpr std::string_view k_mtp_mark = "nextn";
constexpr std::string_view k_embedding_mark = "token_embd";

// What the fitter's per-layer expert pattern looks like around its index, as common/fit.cpp spells
// it. Both halves are regex source text: an escaped dot each, and the alternation that covers every
// expert projection layout the loader knows.
constexpr std::string_view k_override_head = "blk\\.";
constexpr std::string_view k_override_tail = "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
// What every per-layer FFN pattern of the fitter's starts with after the index: the whole FFN, a
// share of it, or the experts alone.
constexpr std::string_view k_ffn_tail = "\\.ffn_";

// A layer index past this is a runaway digit run, not a layer.
constexpr int k_layer_max = 1 << 20;

// The digits at `p` up to the first non-digit, as a layer index: -1 when there are none or too
// many. `end` is set past the last digit.
int leading_layer(const char * p, const char ** end) {
    if (*p < '0' || *p > '9') return -1;
    int il = 0;
    for (; *p >= '0' && *p <= '9'; ++p) {
        il = il * 10 + (*p - '0');
        if (il > k_layer_max) return -1;
    }
    *end = p;
    return il;
}

// As above, with nothing allowed after the digits.
int whole_layer(const char * p) {
    const char * end = nullptr;
    const int il = leading_layer(p, &end);
    return il >= 0 && *end == '\0' ? il : -1;
}

bool starts_with(const char * s, std::string_view prefix) {
    return std::strncmp(s, prefix.data(), prefix.size()) == 0;
}

} // namespace

int block_layer(const char * name, const char ** rest) {
    if (!starts_with(name, k_block_prefix)) return -1;
    const char * end = nullptr;
    const int il = leading_layer(name + k_block_prefix.size(), &end);
    if (il < 0 || *end != '.') return -1;
    if (rest) *rest = end + 1;
    return il;
}

std::string_view weight_suffix(const char * name, int & il_out) {
    const char * rest = nullptr;
    const int il = block_layer(name, &rest);
    if (il < 0) return {};
    const size_t len = std::strlen(rest);
    if (len <= k_weight_tail.size() || k_weight_tail != rest + (len - k_weight_tail.size())) return {};
    il_out = il;
    return std::string_view(rest, len - k_weight_tail.size());
}

bool is_weight(const char * name) {
    const size_t len = std::strlen(name);
    return len >= k_weight_tail.size() && k_weight_tail == name + (len - k_weight_tail.size());
}

BlockPart block_part(const char * name) {
    const char * rest = nullptr;
    if (block_layer(name, &rest) < 0) return BlockPart::Other;
    if (starts_with(rest, k_attention_head)) return BlockPart::Attention;
    if (starts_with(rest, k_ffn_head)) return BlockPart::Ffn;
    return BlockPart::Other;
}

bool is_mtp_tensor(const char * name) {
    return std::strstr(name, k_mtp_mark.data()) != nullptr;
}

bool is_embedding_table(const char * name) {
    return std::strstr(name, k_embedding_mark.data()) != nullptr;
}

int node_layer(const char * name, std::string_view prefix) {
    return starts_with(name, prefix) ? whole_layer(name + prefix.size()) : -1;
}

int any_node_layer(const char * name) {
    const char * dash = std::strrchr(name, '-');
    return dash ? whole_layer(dash + 1) : -1;
}

int weight_node(const char * name, int & il_out) {
    for (int v = 0; v < k_weight_node_count; ++v) {
        const std::string_view n = k_weight_nodes[v];
        if (!starts_with(name, n) || name[n.size()] != '-') continue;
        const int il = whole_layer(name + n.size() + 1);
        if (il < 0) continue;
        il_out = il;
        return v;
    }
    return -1;
}

std::string expert_override_pattern(int il) {
    return std::string(k_override_head) + std::to_string(il) + std::string(k_override_tail);
}

int expert_override_layer(const char * pattern) {
    // Found anywhere, not only at the start: what is asked is whether the pattern names that
    // layer's experts, and the fitter's form can be one part of a longer pattern.
    for (const char * at = std::strstr(pattern, k_override_head.data()); at;
         at = std::strstr(at + 1, k_override_head.data())) {
        const char * end = nullptr;
        const int il = leading_layer(at + k_override_head.size(), &end);
        if (il >= 0 && starts_with(end, k_override_tail)) return il;
    }
    return -1;
}

int ffn_override_layer(const char * pattern) {
    for (const char * at = std::strstr(pattern, k_override_head.data()); at;
         at = std::strstr(at + 1, k_override_head.data())) {
        const char * end = nullptr;
        const int il = leading_layer(at + k_override_head.size(), &end);
        if (il >= 0 && starts_with(end, k_ffn_tail)) return il;
    }
    return -1;
}

} // namespace bmoe::tensor_names
