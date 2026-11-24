// The dense (non-expert) model weights, and the policy for keeping them resident.
//
// The streamer rebinds only the expert tensors; everything else in the gguf — header/metadata,
// embeddings, attention, norms, lm_head — is "dense", and how it is kept in RAM is a policy of its
// own (DenseWeightsMode): left mmap'd, mmap'd-and-warmed, or read once into anonymous buffers via
// O_DIRECT and rebound. This module owns that policy, the buffers it may allocate, and the residency
// sensor that reports how much of the dense set the kernel still has — the half the expert cache's
// own residency sensor is blind to.
//
// It reads through its OWN FileReaders, so its O_DIRECT choice is independent of the expert stream's:
// the dense set may be pulled cache-bypassing while the experts are not, or the reverse. There is one
// definition of "which bytes are dense" here (byte_ranges), shared by the warm sweep, the anonymous
// read, and the sensor — the three used to compute it separately.
//
// A split model hands in one path (and one set of dense ranges) per shard file; every consumer walks
// the set. A single-file model is the one-entry case of the same shape, not a separate path.
#pragma once

#include "../io/file_reader.h"
#include "../io/platform_io.h"
#include "bmoe/config.h" // DenseWeightsMode
#include "bmoe/row_source.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct ggml_tensor;

namespace bmoe {

class RowStream;

// One dense weight tensor: where its bytes live in the gguf, and the tensor whose ->data we rebind.
// Read whole and contiguous (`size` bytes from `file_off` in shard `file_idx`), unlike an expert slice.
struct DenseTensorRef {
    ggml_tensor * tensor = nullptr;
    uint64_t file_off = 0;
    uint64_t size = 0;
    int file_idx = 0; // which shard file holds the bytes (0 for a single-file model)
    // Other tensor objects over the SAME file bytes, which a residency policy must rebind together
    // with `tensor` or leave together where they are. A tied output head is the case that produces
    // them: llama.cpp builds the head from the token embedding table, so the model holds two
    // ggml_tensors, identically named, over one range. Rebinding one and not the other leaves the
    // graph reading the model's mmap every token, and makes the range unsafe to release. Empty for
    // every ordinary weight, so the common path allocates nothing.
    std::vector<ggml_tensor *> aliases;
};

class DenseWeights {
public:
    // Both defined in the .cpp: the row policy is an incomplete type here, and a defaulted
    // constructor in the header would have to instantiate its deleter.
    DenseWeights();
    ~DenseWeights();

    DenseWeights(const DenseWeights &) = delete;
    DenseWeights & operator=(const DenseWeights &) = delete;

    // The dense byte ranges OF ONE FILE: the complement of `expert_ranges` within [0, file_size).
    // Sorts the input; the result is the gaps between expert ranges plus the trailing tail. The
    // single source of truth every consumer here shares; called once per shard.
    static std::vector<std::pair<uint64_t, uint64_t>>
    byte_ranges(std::vector<std::pair<uint64_t, uint64_t>> expert_ranges, uint64_t file_size);

    // Apply `mode` for the model files `paths` (per-shard dense `ranges` precomputed via byte_ranges;
    // `tensors` is what Anonymous/Pinned read, and what every mode checks for a tensor too large to
    // be resident — see hold_back_oversized). The readers it opens ask each file for its own read
    // alignment. Runs once at load, before the streamer's workers start (Anonymous rebinds
    // tensor->data on the caller's thread).
    // Returns false only on a hard Anonymous failure (alloc/read); Mmap and Warmed cannot fail.
    bool init(DenseWeightsMode mode,
              const std::vector<std::string> & paths,
              std::vector<std::vector<std::pair<uint64_t, uint64_t>>> ranges,
              std::vector<DenseTensorRef> tensors);

    // Tables the graph only ever ROW-GATHERS (discovered by RouterHook, never named here): instead
    // of being made resident, they are bound to reserved address space and served from flash a row
    // at a time, inside `budget_bytes` of window. Call BEFORE init. A table taken over leaves the
    // resident dense set entirely — not read, not rebound, not warmed, not sampled — exactly as an
    // oversized one does, because a sensor that counted it would report a set that is resident by
    // design only in part. If the takeover fails, the tables silently keep the mode the run asked
    // for: this is a residency optimisation, never a reason for a run not to start.
    void set_row_gathered(std::vector<DenseTensorRef> tables, uint64_t budget_bytes);

    // Tensors the plan leaves mapped, by name (MoeStreamConfig::dense_mapped): held back from the
    // resident set exactly as an oversized tensor is, under every mode. Call BEFORE init.
    void set_mapped(std::vector<std::string> names) { mapped_names_ = std::move(names); }

    // The row policy, once init has run: null when no table qualified or the takeover failed. The
    // engine's graph adapter needs it to make rows present before a gather node runs.
    IRowSource * row_source() const;

    // Whether any tensor this module was given still reads through the model file's mapping once
    // init has run: every tensor under Mmap/Warmed, and under Anonymous/Pinned the ones held back
    // as oversized. Row-gathered tables are served from our own reader, not the mapping.
    bool file_mapping_in_use() const;

    // Reopen the readers still in service after init: the row policy's, if any (the dense copy's
    // own readers have done their one job). See FileReader::reopen for why this exists.
    bool reopen_readers();
    RowSourceStats row_stats() const;

    // Sample how much of the dense set the kernel still has in RAM (mincore), setting resident_frac().
    // Anonymous samples the anon buffers; Mmap/Warmed sample the model's mapping at the addresses
    // llama.cpp gave each tensor, which every platform answers the same way.
    // The caller throttles (calls this only when it wants a fresh sample); `page` is the OS page size.
    void sample_residency(size_t page);
    double resident_frac() const { return resident_frac_; }

    // Bytes of the dense set that asked for the reclaim-exempt store and were refused it, and so
    // are held as ordinary memory. 0 when the store granted everything, or was never asked.
    uint64_t pin_refused_bytes() const { return pin_refused_bytes_; }

    void shutdown();

    static constexpr int sample_pages = 256; // stratified probe points across the dense bytes

private:
    // A dense tensor larger than the memory the kernel says is available cannot be resident under
    // any mode, and one the plan named stays mapped by decision. Moves such tensors from `tensors_`
    // to `mapped_`, carves their bytes out of `ranges_` (so neither the warm sweep nor the sensor
    // treats them as dense the run could hold), and says so.
    void hold_back_oversized();
    std::vector<std::string> mapped_names_;
    // Hand the qualifying tables to the row policy and take them out of the resident dense set.
    void take_row_gathered();
    // Random-access advice on the mmap of every held-back tensor: a fault brings in one page, not a
    // readahead window the gather will never touch.
    void advise_random_mapped();
    bool read_anonymous();
    void warm();
    void drop_mmap_copies(size_t page);
    void sample_anon(size_t page);
    void sample_mmap(size_t page);
    // Where each tensor's file bytes are mapped, taken from the tensors themselves before anything
    // is rebound: llama.cpp loads memory-mapped, so a dense weight's data IS its address in the
    // mapping. No listing of the process's regions is needed, which some platforms do not offer.
    void index_mapping();
    const char * addr_of(int file_idx, uint64_t off) const;

    DenseWeightsMode mode_ = DenseWeightsMode::Warmed;
    std::vector<std::string> paths_; // one per shard, in shard order

    // Anonymous/Pinned: our own readers (one per shard; FileReader is not movable, hence the
    // unique_ptr), the tensors we read, and the buffers backing them. `bases_` is what the sensor
    // probes and is filled by both modes; `bufs_` and `pinned_` are the two release lists. Both
    // are populated only on a run whose pinned store ran out part way.
    std::vector<std::unique_ptr<FileReader>> readers_;
    std::vector<DenseTensorRef> tensors_;
    std::vector<DenseTensorRef> mapped_; // held back by hold_back_oversized: mmap'd under every mode
    // Row-gathered tables: what was asked for, the window budget, and the policy that owns them.
    std::vector<DenseTensorRef> row_pending_;
    uint64_t row_budget_ = 0;
    std::unique_ptr<RowStream> rows_;
    std::vector<void *> bases_;
    std::vector<void *> bufs_;
    std::vector<pio::PinnedAlloc> pinned_;
    std::vector<size_t> buf_sz_;

    // The dense byte ranges, and where the mapping holds each tensor's bytes, per shard (llama.cpp
    // maps every shard of a split model). Spans are sorted by file offset.
    struct MappedSpan {
        uint64_t file_off = 0;
        uint64_t size = 0;
        const char * addr = nullptr;
    };
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> ranges_;
    std::vector<std::vector<MappedSpan>> spans_;

    double resident_frac_ = -1.0; // last sampled dense residency; -1 = never/unmeasured
    uint64_t pin_refused_bytes_ = 0;
};

} // namespace bmoe
