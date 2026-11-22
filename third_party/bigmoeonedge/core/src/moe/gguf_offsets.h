// Read every tensor's absolute byte offset in a gguf file, using only the public gguf
// API. The expert streamer needs these offsets to pread individual expert slices; it
// gets the tensor pointers (to rebind) from the graph, and the file layout from here.
// The two are matched by tensor name.
//
// A model may ship as several shard files (`model-00001-of-00003.gguf`, the llama.cpp
// split convention — Hugging Face rejects single files above 50 GB, so every large model
// arrives this way). llama.cpp loads the whole set when handed the first shard; here the
// same first-shard path fans out to one parse per shard, and every tensor resolves to
// (shard index, offset in that shard) instead of an offset in one file.
#pragma once

#include "bmoe/expert_layout.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace bmoe {

struct GgufOffsets {
    // tensor name -> byte offset of its data (data_offset + per-tensor offset) WITHIN ITS SHARD
    std::unordered_map<std::string, uint64_t> off_by_name;
    // tensor name -> its data size in bytes. The streamer does not need this (an expert slice
    // is strided by nb2, read from the graph's tensor), but the route trace does: it is the
    // only way to say how many bytes of a layer are dense, i.e. left mmap-resident.
    std::unordered_map<std::string, uint64_t> size_by_name;
    // tensor name -> its ggml type id. Carried as an int so consumers that only need to hand it
    // back to a backend (asking whether that backend can execute this layout) need no ggml header.
    std::unordered_map<std::string, int> type_by_name;
    // tensor name -> its row count (ne[1]), 0 for a 1-D tensor. The only SHAPE this map carries,
    // and it exists for one question: how many bytes of a table a single token gathers, which is
    // what turns an embedding table from a residency cost into a per-token demand. Filled from the
    // same parse as everything else rather than from a second one.
    std::unordered_map<std::string, uint64_t> rows_by_name;
    // Every tensor's name with the third extent of a 3-D one (0 for any other rank): the two facts
    // that tell a routed expert weight from everything else in the file (bmoe/expert_layout.h),
    // read off the same parse.
    std::vector<TensorExtent> extents;
    // tensor name -> index into shard_paths. Always filled; 0 for every tensor of a
    // single-file model, so consumers index shard_paths unconditionally.
    std::unordered_map<std::string, int> file_by_name;
    // The files, in shard order. A single-file model is the degenerate case: one entry,
    // the path that was asked for. Never empty when ok.
    std::vector<std::string> shard_paths;
    // Per shard, the file's total data size is not knowable from the gguf header alone;
    // consumers that need it (dense byte ranges) take it from the opened file instead.
    bool ok = false;
};

// Parse `path` with no_alloc (metadata only, no tensor data read into RAM) and collect
// the offset of every tensor — across every shard when `path` is the first file of a
// split set. Returns ok=false if any file cannot be opened as gguf or a shard is missing.
GgufOffsets read_gguf_offsets(const char * path);

// The handful of model metadata needed BEFORE the model is loaded: the architecture (to
// build arch-prefixed metadata keys) and the MoE expert counts. Read via the public gguf
// API, so no per-architecture constants leak into the engine — the arch string drives the
// key names. n_expert/n_expert_used are 0 for a non-MoE model or a missing key.
struct GgufModelInfo {
    std::string arch;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_embd = 0; // hidden state width: what a tensor crossing a backend boundary carries
    // `<arch>.attention.sliding_window`: how many positions a windowed layer attends to, 0 for a
    // model without one. A context checkpoint of such a layer copies that many cells.
    int n_swa = 0;
    // What kind of file this is, by the format's own keys rather than its name: `general.type`
    // ("model", "mmproj", "adapter"; "" when absent), the transformer blocks it declares
    // (`<arch>.block_count`), how many of them are next-token prediction heads
    // (`<arch>.nextn_predict_layers`), and whether it names layers of another model it reads
    // hidden states from (`<arch>.target_layers`), which only a draft for that model does.
    std::string type;
    int n_block = 0;
    int n_nextn = 0;
    bool reads_target = false;
    // `general.name`: the base model the conversion wrote this file from. A model and the companion
    // files of the same conversion (its projector) carry the same one, which is how they are paired.
    std::string name;
    bool ok = false;
};

// Peek `path`'s metadata (no_alloc, no tensor bytes) for the fields above. Returns ok=false
// if the file cannot be opened as gguf; a present file with a missing key leaves that field 0.
// Metadata lives in the first shard, so this never needs the rest of a split set.
GgufModelInfo read_gguf_model_info(const char * path);

// Both of the above from ONE parse of the first file. gguf_init_from_file walks the whole
// KV section even with no_alloc, so a caller that needs offsets and model info should not
// pay that walk twice. Further shards of a split set are parsed once each for offsets.
struct GgufMeta {
    GgufOffsets offsets;
    GgufModelInfo info;
    bool ok = false; // both parses' ok, from the single underlying open
};
GgufMeta read_gguf_meta(const char * path);

} // namespace bmoe
