// Row-gathered dense tables, served from flash instead of RAM.
//
// The default dense policy makes every non-expert weight resident, because that is what a weight
// that is multiplied is: read whole, every token. A table the graph only GATHERS ROWS from is not
// that. A token embedding is the extreme case — one row per decoded token out of a vocabulary of
// hundreds of thousands — and on a big model it is hundreds of MiB of residency bought for a
// kilobyte of use per token. On a phone that RAM is the whole budget: it is what the expert cache
// does not get, and what the kernel reclaims to zram when it runs out.
//
// So: reserve the table's address space, bind the tensor to it, and commit only the pages the
// graph is about to read, filled by an O_DIRECT read of exactly those bytes. The reservation
// mirrors the tensor's own byte layout, so ggml's address arithmetic (`data + i*nb1`) is
// unchanged and the gather kernel is untouched — the same trick the expert cache plays on
// `mul_mat_id`, at row granularity instead of expert granularity.
//
// WHICH tables get this is not a list in this file and not a name pattern: it is what the graph
// says (see RouterHook's capture pass). A table qualifies only when EVERY reference to it in the
// captured graph is a row gather, and the index it is gathered by is materialized before the node
// runs. Anything else — a weight that is also multiplied, tied embeddings used as the output head,
// an index computed inside the graph — is not served here and keeps the dense policy it had.
// materialize() is the belt to that braces: a graph shape we never saw still gets its bytes.
//
// The residency window is bounded (`budget_bytes`) and evicted LRU, in page-aligned SLABS rather
// than single rows: a row is ~1 KiB, a read below the device's request floor costs the same as one
// at it, and adjacent vocabulary rows recur. The budget is a ceiling, not a target — a run that
// gathers few distinct rows simply never fills it.
#pragma once

#include "../io/file_reader.h"
#include "../io/platform_io.h"
#include "bmoe/row_source.h"
#include "dense_weights.h" // DenseTensorRef

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

namespace bmoe {

class RowStream final : public IRowSource {
public:
    RowStream() = default;
    ~RowStream() override;

    RowStream(const RowStream &) = delete;
    RowStream & operator=(const RowStream &) = delete;

    // Take over `tensors` (already filtered by the caller to tables the graph only row-gathers),
    // reserving address space for each and rebinding its ggml tensor onto it. `paths` are the shard
    // files in shard order (each reader asks its file for the read alignment), `budget_bytes` the
    // ceiling on resident slabs across all tables (0 = a built-in default). Returns false only if a
    // table could not be taken over at all, in which case NOTHING has been rebound and the caller
    // keeps its own policy: this module never leaves a tensor half-owned.
    bool init(std::vector<DenseTensorRef> tensors, const std::vector<std::string> & paths, uint64_t budget_bytes);

    bool empty() const { return tables_.empty(); }

    // ── IRowSource ───────────────────────────────────────────────────────────────────
    bool serves(const ggml_tensor * table) const override;
    bool gather(const ggml_tensor * table, const int32_t * idx, int n) override;
    bool materialize(const ggml_tensor * table) override;
    RowSourceStats stats() const override;

    void shutdown();
    // Reopen the per-shard readers (see FileReader::reopen). No gather may be in flight.
    bool reopen_readers();

    // A row is far below any device's request floor, so a slab is the unit actually read: large
    // enough that the read costs what the floor costs anyway, small enough that a scattered
    // vocabulary does not drag megabytes of neighbours in behind it. Not a per-model number —
    // it is a property of the storage, and the same value serves every table. The floor, that is:
    // the slab in use is this rounded up to whole pages (see slab_), because a slab is committed and
    // evicted as pages, and evicting one that shares a page with a resident neighbour would hand
    // that neighbour back as zeros.
    static constexpr uint64_t slab_min_bytes = 16ull << 10;
    static constexpr uint64_t default_budget_bytes = 64ull << 20;

private:
    struct Table {
        DenseTensorRef ref;
        char * base = nullptr;  // reserved span the tensor is bound to
        uint64_t span = 0;      // reserved bytes (ref.size rounded up to a page)
        uint64_t row_bytes = 0; // nb[1]
        uint64_t n_rows = 0;    // ne[1]
        uint64_t n_slabs = 0;
        std::vector<uint8_t> resident; // one flag per slab
        std::vector<uint32_t> stamp;   // last gather generation that needed the slab (eviction guard)
        // LRU over resident slabs: the list holds (table, slab) oldest-first, `spot` the position of
        // a resident slab in it (end() when not resident).
        std::vector<std::list<std::pair<size_t, uint64_t>>::iterator> spot;
        void * orig_data = nullptr; // what the tensor pointed at before we bound it, restored on release
        bool whole = false;         // materialize() has pulled everything; gather() is then a no-op
    };

    // Make slab `s` of `t` present, reading it from the shard. No-op when already resident.
    bool fetch_slab(Table & t, uint64_t s);
    // Hand back oldest slabs until `headroom` more bytes fit under the budget.
    void evict_to_budget(uint64_t headroom);
    void touch(Table & t, uint64_t s);
    Table * find(const ggml_tensor * table);
    const Table * find(const ggml_tensor * table) const;
    void release();

    std::vector<Table> tables_;
    std::vector<std::unique_ptr<FileReader>> readers_; // one per shard; FileReader is not movable
    std::list<std::pair<size_t, uint64_t>> lru_;       // (table index, slab index), oldest first
    uint64_t budget_bytes_ = default_budget_bytes;
    uint64_t table_bytes_ = 0; // total size of the tables taken over
    uint64_t page_ = pio::vm_page();
    uint64_t slab_ = slab_min_bytes; // slab_min_bytes in whole pages; settled at init
    uint32_t gen_ = 0;               // gather generation, stamped on the slabs the current node needs
    std::vector<uint64_t> need_;     // scratch: the slabs of the gather in flight (eval thread only)

    std::atomic<uint64_t> resident_bytes_{0};
    // Counters are written on the eval thread and read by the metrics thread; relaxed atomics keep
    // that honest without pretending the pair is a consistent snapshot.
    std::atomic<uint64_t> gathers_{0};
    std::atomic<uint64_t> rows_{0};
    std::atomic<uint64_t> slab_reads_{0};
    std::atomic<uint64_t> bytes_read_{0};
    std::atomic<uint64_t> evictions_{0};
    std::atomic<uint64_t> materialized_{0};
    std::atomic<uint64_t> io_errors_{0};
};

} // namespace bmoe
