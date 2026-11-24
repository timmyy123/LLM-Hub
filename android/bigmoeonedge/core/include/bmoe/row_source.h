// The row-gathered residency port.
//
// Some dense (non-expert) model weights are not walked whole on a token: the graph gathers a
// handful of ROWS out of them (`ggml_get_rows`) and never touches the rest. A token embedding
// table is the pure case — one row per decoded token out of a vocabulary of hundreds of
// thousands — and it is also, on a big model, hundreds of MiB of RAM that is resident so that
// 0.0004 % of it can be read.
//
// An IRowSource owns such a table's residency at ROW granularity: it holds the address space the
// tensor is bound to, and makes only the rows the graph is about to gather physically present,
// reading them from flash. The engine's router hook calls gather() in the eval callback's ask
// pass, before the gather node runs, with the indices the graph is about to use.
//
// This is a port for the same reason IExpertSource is one: "which rows are resident" is a policy,
// and the flash-backed sliding window is one implementation of it. A test fake, an all-resident
// implementation, or a future per-row cache are others.
#pragma once

#include <cstdint>

struct ggml_tensor;

namespace bmoe {

// What a row source is holding and what it has done, for telemetry and for the one line a run
// prints about it. `table_bytes` is what the tables would occupy under the ordinary dense policy
// and `resident_bytes` what they occupy now: the pair the policy is judged on.
struct RowSourceStats {
    uint64_t table_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t gathers = 0;      // gather() calls (one per row-gather node executed)
    uint64_t rows = 0;         // row indices asked for, duplicates included
    uint64_t slab_reads = 0;   // reads actually issued to flash (the misses)
    uint64_t bytes_read = 0;   // bytes those reads moved
    uint64_t evictions = 0;    // slabs handed back to stay inside the budget
    uint64_t materialized = 0; // whole-table fallbacks taken (see materialize())
    uint64_t io_errors = 0;    // failed fetches; any non-zero value means a decode read garbage
};

class IRowSource {
public:
    virtual ~IRowSource() = default;

    // Does this source own `table`'s residency? A cheap identity test on the bound tensor pointer;
    // the hook calls it for every gather node in the graph, so it must not allocate or lock.
    virtual bool serves(const ggml_tensor * table) const = 0;

    // Make rows `idx[0..n)` of `table` present before the gather reads them. Indices are the raw
    // i32 contents of the gather's index tensor: duplicates are expected, and an out-of-range index
    // is ignored rather than treated as an error (the graph, not this source, defines validity).
    // Called on the eval thread only, from the ask pass of the node that is about to run.
    // Returns false on an I/O failure, which the caller must treat as fatal for the decode: the
    // rows would otherwise be read as whatever the address space holds.
    virtual bool gather(const ggml_tensor * table, const int32_t * idx, int n) = 0;

    // Make the WHOLE table present. The safety net for a graph shape that was not seen when the
    // policy was chosen: residency was granted to a table on the evidence that every reference to
    // it in the captured graph was a row gather, and if some later graph reads it any other way,
    // the bytes must all be there before that node runs. Idempotent; returns false on I/O failure.
    virtual bool materialize(const ggml_tensor * table) = 0;

    // Cumulative accounting since the source was built. Read off the eval thread, so the counters
    // are individually current but not a consistent snapshot.
    virtual RowSourceStats stats() const = 0;
};

} // namespace bmoe
