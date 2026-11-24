#include "gguf_offsets.h"
#include "bmoe/log.h"

#include "gguf.h"

#include <cstdio>
#include <string>

namespace bmoe {

namespace {

// Read a gguf metadata value as an int, accepting any of the integer scalar types the
// writer might have used for a count (llama.cpp writes expert counts as UINT32, but be
// permissive). Returns `dflt` if the key is absent or not an integer scalar.
int meta_int(const gguf_context * ctx, const std::string & key, int dflt) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return dflt;
    switch (gguf_get_kv_type(ctx, id)) {
    case GGUF_TYPE_UINT8:
        return (int) gguf_get_val_u8(ctx, id);
    case GGUF_TYPE_INT8:
        return (int) gguf_get_val_i8(ctx, id);
    case GGUF_TYPE_UINT16:
        return (int) gguf_get_val_u16(ctx, id);
    case GGUF_TYPE_INT16:
        return (int) gguf_get_val_i16(ctx, id);
    case GGUF_TYPE_UINT32:
        return (int) gguf_get_val_u32(ctx, id);
    case GGUF_TYPE_INT32:
        return (int) gguf_get_val_i32(ctx, id);
    case GGUF_TYPE_UINT64:
        return (int) gguf_get_val_u64(ctx, id);
    case GGUF_TYPE_INT64:
        return (int) gguf_get_val_i64(ctx, id);
    default:
        return dflt;
    }
}

// Single parse; each extraction below reads from the already-parsed context.
gguf_context * open_meta(const char * path) {
    gguf_init_params gp{};
    gp.no_alloc = true; // metadata + offsets only; no tensor bytes touched
    gp.ctx = nullptr;
    return gguf_init_from_file(path, gp);
}

void fill_offsets(const gguf_context * gctx, int file_idx, GgufOffsets & out) {
    const uint64_t data_off = (uint64_t) gguf_get_data_offset(gctx);
    const int64_t n = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        out.off_by_name[name] = data_off + (uint64_t) gguf_get_tensor_offset(gctx, i);
        out.size_by_name[name] = (uint64_t) gguf_get_tensor_size(gctx, i);
        out.type_by_name[name] = (int) gguf_get_tensor_type(gctx, i);
        out.file_by_name[name] = file_idx;
        // Row count, for the one consumer that needs a shape rather than a size: turning a table's
        // total bytes into the bytes a single token gathers from it. Taken here because this parse
        // has already walked the whole KV section, which is exactly what this file's header says a
        // caller must not be made to pay for twice.
        const int64_t * ne = gguf_get_tensor_ne(gctx, i);
        out.rows_by_name[name] = ne ? (uint64_t) (ne[1] > 0 ? ne[1] : 0) : 0;
        // Exactly three dimensions: gguf pads ne to GGML_MAX_DIMS with ones, so a 3-D tensor is one
        // whose fourth extent is 1 and whose third is not.
        out.extents.push_back({name, ne && ne[3] == 1 && ne[2] > 1 ? (uint64_t) ne[2] : 0});
    }
}

void fill_model_info(const gguf_context * gctx, GgufModelInfo & out) {
    const int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(gctx, arch_id) == GGUF_TYPE_STRING) {
        out.arch = gguf_get_val_str(gctx, arch_id);
    }
    const int64_t type_id = gguf_find_key(gctx, "general.type");
    if (type_id >= 0 && gguf_get_kv_type(gctx, type_id) == GGUF_TYPE_STRING) out.type = gguf_get_val_str(gctx, type_id);
    const int64_t name_id = gguf_find_key(gctx, "general.name");
    if (name_id >= 0 && gguf_get_kv_type(gctx, name_id) == GGUF_TYPE_STRING) out.name = gguf_get_val_str(gctx, name_id);
    if (!out.arch.empty()) {
        // Arch-prefixed keys, exactly as llama.cpp names them (LLM_KV_EXPERT_COUNT /
        // LLM_KV_EXPERT_USED_COUNT expand "%s" to the architecture).
        out.n_expert = meta_int(gctx, out.arch + ".expert_count", 0);
        out.n_expert_used = meta_int(gctx, out.arch + ".expert_used_count", 0);
        // The hidden state's width. It is what a tensor crossing a host/device boundary actually
        // carries, so the split probe needs it to measure a crossing at the size this model pays.
        out.n_embd = meta_int(gctx, out.arch + ".embedding_length", 0);
        out.n_swa = meta_int(gctx, out.arch + ".attention.sliding_window", 0);
        out.n_block = meta_int(gctx, out.arch + ".block_count", 0);
        out.n_nextn = meta_int(gctx, out.arch + ".nextn_predict_layers", 0);
        out.reads_target = gguf_find_key(gctx, (out.arch + ".target_layers").c_str()) >= 0;
    }
    out.ok = true;
}

// The llama.cpp split filename convention: `<prefix>-%05d-of-%05d.gguf`, 1-based. If
// `path` matches, returns the shard count and sets `prefix` (everything before the
// shard number); the caller rebuilds each sibling with make_shard_path. Returns 0 for
// a plain single-file path. The FIRST shard must be the one asked for — the same rule
// llama.cpp applies — so downstream shard indices line up with llama.cpp's own mmaps.
int split_count_from_path(const std::string & path, std::string & prefix) {
    // <prefix>-DDDDD-of-DDDDD.gguf → 19 chars after the separating dash
    static const size_t tail_len = 5 + 4 + 5 + 5; // "DDDDD" "-of-" "DDDDD" ".gguf"
    if (path.size() < tail_len + 1) return 0;
    const size_t tail = path.size() - tail_len;
    auto five_digits = [&](size_t at) {
        for (size_t i = at; i < at + 5; ++i)
            if (path[i] < '0' || path[i] > '9') return false;
        return true;
    };
    if (path[tail - 1] != '-' || !five_digits(tail) || path.compare(tail + 5, 4, "-of-") != 0 ||
        !five_digits(tail + 9) || path.compare(tail + 14, 5, ".gguf") != 0)
        return 0;
    const int no = std::stoi(path.substr(tail, 5));
    const int count = std::stoi(path.substr(tail + 9, 5));
    if (count <= 1) return 0; // a -00001-of-00001 file is just a single file
    if (no != 1) {
        log_printf(LogLevel::Error, "%s is shard %d of a split model — pass the first shard (-00001-of-)", path.c_str(),
                   no);
        return -1;
    }
    prefix = path.substr(0, tail - 1);
    return count;
}

std::string make_shard_path(const std::string & prefix, int no, int count) {
    char tail[32];
    std::snprintf(tail, sizeof(tail), "-%05d-of-%05d.gguf", no, count);
    return prefix + tail;
}

// Offsets for `path` and, if it is the first file of a split set, for every sibling
// shard. Any missing or unparsable shard fails the whole read: a partial map would
// surface later as "no gguf offset for tensor X", which points at the wrong culprit.
bool fill_all_offsets(const char * path, GgufOffsets & out, gguf_context * first) {
    std::string prefix;
    const int count = split_count_from_path(path, prefix);
    if (count < 0) return false;

    fill_offsets(first, 0, out);
    out.shard_paths.push_back(path);

    for (int no = 2; no <= count; ++no) {
        const std::string sp = make_shard_path(prefix, no, count);
        gguf_context * gctx = open_meta(sp.c_str());
        if (!gctx) {
            log_printf(LogLevel::Error, "split model is missing shard %s", sp.c_str());
            out = GgufOffsets{};
            return false;
        }
        fill_offsets(gctx, no - 1, out);
        out.shard_paths.push_back(sp);
        gguf_free(gctx);
    }
    out.ok = true;
    return true;
}

} // namespace

GgufOffsets read_gguf_offsets(const char * path) {
    GgufOffsets out;
    if (gguf_context * gctx = open_meta(path)) {
        fill_all_offsets(path, out, gctx);
        gguf_free(gctx);
    }
    return out;
}

GgufModelInfo read_gguf_model_info(const char * path) {
    GgufModelInfo out;
    if (gguf_context * gctx = open_meta(path)) {
        fill_model_info(gctx, out);
        gguf_free(gctx);
    }
    return out;
}

GgufMeta read_gguf_meta(const char * path) {
    GgufMeta out;
    if (gguf_context * gctx = open_meta(path)) {
        const bool offs_ok = fill_all_offsets(path, out.offsets, gctx);
        fill_model_info(gctx, out.info);
        gguf_free(gctx);
        out.ok = offs_ok;
    }
    return out;
}

} // namespace bmoe
