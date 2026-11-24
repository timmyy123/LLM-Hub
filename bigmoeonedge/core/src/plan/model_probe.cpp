// The model half of the planner's input: everything read from the gguf's own metadata and tensor
// shapes, with no tensor data touched and the model never loaded.
//
// Which tensors are experts is read from the file itself (bmoe/expert_layout.h: a block's 3-D
// weights indexed by the declared expert count), the same discovery the streamer runs at load, so
// there is no table of architectures to fall out of date. A MoE whose layout cannot be read that way
// gets an error, and the planner declines rather than guessing at a layout. Every other name this
// file matches (the token table, the output head, a block's attention and feed-forward) is the gguf
// naming convention itself, shared by every architecture llama.cpp converts: a property of the
// FORMAT, not of any model, and spelled in tensor_names with every other name the engine reads. A
// name matching nothing lands in Other and is priced as read-whole, which is the conservative
// direction — an unrecognised tensor can make a plan cautious, never reckless.

#include "bmoe/probe.h"

#include "../io/platform_io.h"
#include "../moe/gguf_offsets.h"
#include "../moe/tensor_names.h"
#include "bmoe/expert_layout.h"

#include <algorithm>
#include <cstdio>
#include <vector>
#include <map>
#include <set>
#include <string>

namespace bmoe {

GgufHeaderFacts probe_gguf_header(const char * model_path) {
    GgufHeaderFacts h;
    if (!model_path || !*model_path) return h;
    const GgufModelInfo info = read_gguf_model_info(model_path);
    h.ok = info.ok;
    h.arch = info.arch;
    h.n_expert = info.n_expert;
    h.n_expert_used = info.n_expert_used;
    h.type = info.type;
    h.n_block = info.n_block;
    h.n_nextn = info.n_nextn;
    h.reads_target = info.reads_target;
    h.name = info.name;
    return h;
}

ModelProfile probe_model(const char * model_path, uint32_t engine_layers) {
    ModelProfile m;
    if (!model_path || !*model_path) {
        m.error = "no model path";
        return m;
    }

    const GgufMeta meta = read_gguf_meta(model_path);
    if (!meta.ok) {
        m.error = std::string("cannot read gguf: ") + model_path;
        return m;
    }

    m.arch = meta.info.arch;
    m.n_expert = (uint32_t) std::max(0, meta.info.n_expert);
    m.n_expert_used = (uint32_t) std::max(0, meta.info.n_expert_used);
    m.n_embd = (uint32_t) std::max(0, meta.info.n_embd);
    m.n_swa = (uint32_t) std::max(0, meta.info.n_swa);

    // Which tensors are the routed experts, read off the file's shapes, and the same refusal the
    // streamer makes: a file whose experts cannot be told apart gets no plan, since a plan that
    // guessed at them would be worse than none. A dense model is not an error here - it has a plan,
    // with streaming off - so only a declared expert count that fails the reading is one.
    const ExpertLayout layout = discover_expert_layout(meta.offsets.extents, (int) m.n_expert);
    if (m.n_expert > 1 && !layout.ok) {
        m.error = "not a MoE this engine streams: " + layout.error;
        return m;
    }
    m.ok = true;

    // Per MoE layer: the sum over its expert tensors of ONE expert's bytes. Accumulated from the
    // tensors the file actually carries rather than multiplied out, because the per-projection
    // stride is not uniform — a fused gate_up is twice a split one — and an architecture with
    // leading dense blocks has layers that contribute nothing.
    std::map<int, uint64_t> per_layer_one_expert;
    std::map<int, uint64_t> per_layer_expert_bytes;
    std::map<int, uint64_t> per_file_expert_bytes; // by index into the file's shard list
    std::map<int, uint32_t> projections_per_layer;
    std::map<int, uint64_t> per_layer_dense_bytes;       // a block's non-expert tensors
    std::map<int, uint64_t> expert_bytes_by_type;        // by ggml type id
    std::set<int> blocks;                                // every block index the file carries, MTP excluded
    std::set<int> mtp_layers;                            // blocks that carry a multi-token-prediction head
    uint64_t block_bytes[(int) WeightGroup::count] = {}; // each group's bytes that sit in a block

    bool saw_output_weight = false;
    bool saw_token_embd = false;
    uint64_t embd_row_bytes = 0; // bytes one token gathers from the embedding tables
    bool embd_rows_known = true;

    // First pass: which blocks carry a multi-token-prediction head. llama.cpp does not load that
    // block at all unless asked, experts AND dense alike, so nothing of it may count toward what
    // the run holds. (Measured: counting its dense tensors overstated the pinned set by ~850 MiB.)
    for (const auto & kv : meta.offsets.size_by_name) {
        if (!tensor_names::is_mtp_tensor(kv.first.c_str())) continue;
        m.has_mtp = true;
        const int layer = tensor_names::block_layer(kv.first.c_str());
        if (layer >= 0) mtp_layers.insert(layer);
    }

    for (const auto & kv : meta.offsets.size_by_name) {
        const std::string & name = kv.first;
        const uint64_t size = kv.second;

        // Classification and the MTP exclusion see every tensor of a block, scales, biases and
        // probs included: those are bytes the file carries either way.
        const int layer = tensor_names::block_layer(name.c_str());
        const bool in_block = layer >= 0;
        if (in_block && mtp_layers.count(layer)) continue; // never loaded
        if (in_block) blocks.insert(layer);

        m.file_bytes += size;

        if (tensor_names::is_output_head(name)) saw_output_weight = true;
        if (tensor_names::is_token_table(name)) saw_token_embd = true;

        // Only the layout's expert weights: a scale, a bias or a probs tensor is not an expert
        // weight however much its name resembles one, and the layout told them apart by rank.
        int elayer = 0;
        const bool is_expert = layout.slot_of(name.c_str(), elayer) >= 0;

        // An embedding table is gathered by row: the token table, and the per-layer n-gram table some
        // architectures add (gemma4, qwen4exp), which on one released quant is 28.8 GB — 43% of the
        // file. Pricing that as read-whole would put a demand on every token that no token has, and
        // it would dominate every other term in the plan.
        WeightGroup g = WeightGroup::Other;
        const tensor_names::BlockPart part = tensor_names::block_part(name.c_str());
        if (is_expert) {
            g = WeightGroup::Experts;
        } else if (!in_block && tensor_names::is_embedding_table(name.c_str())) {
            g = WeightGroup::Embedding;
            m.embedding_tensors.push_back(name);
            // A table whose row count the file does not state cannot be turned into a per-token
            // demand, and guessing one is how a plan claims a saving it cannot collect.
            const auto r = meta.offsets.rows_by_name.find(name);
            if (r != meta.offsets.rows_by_name.end() && r->second > 0)
                embd_row_bytes += size / r->second;
            else
                embd_rows_known = false;
        } else if (tensor_names::is_output_head(name)) {
            g = WeightGroup::Output;
        } else if (part == tensor_names::BlockPart::Attention) {
            g = WeightGroup::Attention;
        } else if (part == tensor_names::BlockPart::Ffn) {
            g = WeightGroup::DenseFfn;
        }
        m.group(g).bytes += size;
        if (in_block) block_bytes[(int) g] += size;

        if (is_expert) {
            const auto f = meta.offsets.file_by_name.find(name);
            if (f != meta.offsets.file_by_name.end()) per_file_expert_bytes[f->second] += size;
            const uint64_t one = size / m.n_expert;
            per_layer_expert_bytes[elayer] += size;
            per_layer_one_expert[elayer] += one;
            projections_per_layer[elayer]++;
            const auto t = meta.offsets.type_by_name.find(name);
            if (t != meta.offsets.type_by_name.end()) expert_bytes_by_type[t->second] += size;
        } else {
            m.largest_dense_tensor = std::max(m.largest_dense_tensor, size);
            if (in_block) per_layer_dense_bytes[layer] += size;
        }
    }

    // The type that carries the most expert bytes is the one a token mostly reads, so it is the one
    // the bandwidth probe measures; ties go to the lower id so the answer is the same on every
    // standard library. It used to be the first expert tensor met while walking an unordered map,
    // so on a mixed quant (a 2-bit gate and up beside a 3-bit down) the format probed was decided
    // by hash order, and on Linux it was one ggml cannot encode without importance weights.
    uint64_t most_type_bytes = 0;
    for (const auto & kv : expert_bytes_by_type) {
        m.expert_type_ids.push_back(kv.first);
        if (kv.second > most_type_bytes) {
            most_type_bytes = kv.second;
            m.expert_type_id = kv.first;
        }
    }

    uint64_t most_expert_bytes = 0;
    for (const auto & kv : per_file_expert_bytes) {
        if (kv.second <= most_expert_bytes || kv.first < 0) continue;
        if ((size_t) kv.first >= meta.offsets.shard_paths.size()) continue;
        most_expert_bytes = kv.second;
        m.expert_file = meta.offsets.shard_paths[(size_t) kv.first];
    }

    for (const auto & kv : per_layer_expert_bytes) {
        m.expert_bytes += kv.second;
        m.largest_expert_layer_bytes = std::max(m.largest_expert_layer_bytes, kv.second);
    }
    // The engine's layers against the file's blocks. They differ on a model whose graph runs some
    // blocks more than once per token: the engine numbers a layer for every run, and the later ones
    // carry no weights of their own. Which blocks those are is the graph's business and no key of
    // the format says it, so a block's demand is counted `runs` times as an average over all of
    // them: exact for the token's total wherever the blocks are alike, which is every model so far.
    m.n_block = (uint32_t) blocks.size();
    m.n_layer = std::max(m.n_block, engine_layers);
    const auto over_runs = [&m](uint64_t bytes) {
        return m.n_block ? (uint64_t) ((double) bytes * (double) m.n_layer / (double) m.n_block) : bytes;
    };
    // By llama.cpp's own layer index: every block the file carries, an MTP block not among them,
    // and nothing for a layer that runs another block's weights.
    m.layer_dense_bytes.assign(m.n_layer, 0);
    for (const auto & kv : per_layer_dense_bytes) {
        m.largest_layer_dense_bytes = std::max(m.largest_layer_dense_bytes, kv.second);
        if (kv.first >= 0 && (size_t) kv.first < m.n_layer) m.layer_dense_bytes[(size_t) kv.first] = kv.second;
    }
    for (const auto & kv : per_layer_one_expert) {
        // The granule the streamer issues as a read is one expert of one projection; the LARGEST
        // of them is what should be looked up on the storage rate curve, because that is the
        // request the engine will actually make.
        const uint32_t projs = projections_per_layer[kv.first];
        if (projs) m.expert_slice_bytes = std::max(m.expert_slice_bytes, kv.second / projs);
        m.token_cycle_bytes += kv.second * m.n_expert_used;
    }
    m.token_cycle_bytes = over_runs(m.token_cycle_bytes);

    m.dense_bytes = m.file_bytes > m.expert_bytes ? m.file_bytes - m.expert_bytes : 0;
    m.is_moe = m.expert_bytes > 0 && m.n_expert > 0 && m.n_expert_used > 0;
    m.n_moe_layer = (uint32_t) per_layer_one_expert.size(); // the ones that stream
    for (const auto & kv : per_layer_one_expert)
        m.moe_layers.push_back((uint32_t) kv.first);

    uint32_t projections_seen = 0;
    for (const auto & kv : projections_per_layer)
        projections_seen = std::max(projections_seen, kv.second);
    m.n_expert_projections = projections_seen;

    // llama.cpp builds the output head from the embedding table when the file carries no head of
    // its own, leaving two tensor objects over one range. Deciding it the same way the loader does
    // keeps the two from disagreeing.
    m.tied_output_head = saw_token_embd && !saw_output_weight;

    // ── demand per token ────────────────────────────────────────────────────────────
    // Read whole, unless said otherwise below, and a block's part once per run of the block.
    // Nothing here consults a size or an architecture.
    for (int i = 0; i < (int) WeightGroup::count; ++i)
        m.groups[i].bytes_per_token = m.groups[i].bytes - block_bytes[i] + over_runs(block_bytes[i]);

    // The embedding tables are gathered by row — unless the head is tied to one of them, in which
    // case that table is projected against in FULL every token. The presence of a separate output
    // tensor decides it, not a size and not a name. When the head is tied the whole group is
    // priced read-whole, which overstates a per-layer table's demand and is the safe direction.
    GroupDemand & emb = m.group(WeightGroup::Embedding);
    if (emb.bytes > 0 && !m.tied_output_head && embd_rows_known && embd_row_bytes > 0) {
        emb.row_gatherable = true;
        emb.streamable = true;
        emb.bytes_per_token = embd_row_bytes;
    }

    // Experts are read whole but SELECTIVELY: only the routed top-k of each layer. This is the
    // same quantity as the cache floor, and it is deliberately the same number.
    GroupDemand & exp = m.group(WeightGroup::Experts);
    if (exp.bytes > 0 && m.is_moe) {
        exp.bytes_per_token = m.token_cycle_bytes;
        exp.streamable = true;
    }

    return m;
}

std::string model_fingerprint(const char * model_path) {
    // Where the samples are taken, as fractions of the file: the start carries the header and the
    // metadata, the rest the tensor bytes that tell two models of one shape apart.
    constexpr uint64_t k_head_bytes = 1ull << 20;
    constexpr uint64_t k_window_bytes = 64ull << 10;
    constexpr int k_windows = 4;

    const pio::fd_t fd = pio::open_read(model_path, false);
    if (!pio::fd_ok(fd)) return "";
    const uint64_t size = pio::file_size(fd);
    uint64_t h = 1469598103934665603ull; // FNV-1a, 64-bit
    auto mix = [&h](const unsigned char * p, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ull;
        }
    };
    unsigned char size_le[8];
    for (int i = 0; i < 8; ++i)
        size_le[i] = (unsigned char) (size >> (8 * i));
    mix(size_le, sizeof(size_le));

    std::vector<unsigned char> buf(k_head_bytes);
    bool ok = size > 0;
    auto sample = [&](uint64_t off, uint64_t len) {
        if (!ok || off >= size) return;
        len = std::min(len, size - off);
        const long long got = pio::pread_at(fd, buf.data(), (size_t) len, off);
        if (got != (long long) len) {
            ok = false;
            return;
        }
        mix(buf.data(), (size_t) len);
    };
    sample(0, k_head_bytes);
    for (int i = 1; i <= k_windows; ++i)
        sample(size > k_window_bytes ? (size - k_window_bytes) / k_windows * i : 0, k_window_bytes);
    pio::close_fd(fd);
    if (!ok) return "";
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) h);
    return hex;
}

} // namespace bmoe
