// Decode traces: what a token's time is actually made of.
//
// The per-token metrics answer *how long*; the route trace answers *what the router asked for*.
// These two answer *where the time went*, and they exist because the headline number they
// decompose is not measured at all: `compute_ms` is a residual (wall − io − mgmt), so every cost
// the engine does not itself clock — page faults, scheduler stalls, the matmuls themselves — is
// silently pooled into it. A residual cannot tell you which of those it is.
//
//   Compute trace — the eval callback, asked to isolate nodes, yields real per-node wall time:
//   ggml computes exactly up to an isolated node, synchronizes, then calls back, so the delta
//   between consecutive boundaries is that node's compute. Sampling major faults across the same
//   boundaries attributes the >RAM residency stall to the node that paid it, which is the whole
//   point: on a >RAM model most of "compute" is faults, and no residual can show that.
//
//   I/O trace — one row per flash read: latency, size, alignment waste and the (layer, expert,
//   projection) it served. The aggregate read bandwidth is far below the drive's sequential
//   ceiling because routed slices are scattered; this says by how much, and whether the cause is
//   per-read latency, request size, or lanes idling.
//
// Both are diagnostics, not telemetry, and both perturb what they measure — isolating nodes
// forbids ggml the operator coalescing it would otherwise do, and the I/O rows take a lock on the
// read path. A traced run is NOT a benchmark run: read the proportions, not the absolutes.
//
//   Layer granularity — the compute trace can instead isolate only the FIRST node of each layer
//   (IComputeTraceSink::layers_only()). ~n_layer barriers per token instead of ~3000 preserves
//   operator coalescing and, crucially, the async expert prefetch: the io lanes keep streaming
//   across a boundary, so the numbers stay close to an untraced run. Rows carry op "LAYER" and
//   aggregate everything since the previous boundary: name "blk.<il>" is layer il's segment,
//   "pre" is the embedding lookup before layer 0, and "post" (emitted when the batch closes) is
//   the last layer's tail plus the final norm and LM head — per-op detail inside a segment is
//   what this mode trades away.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bmoe {

// One isolated graph node's compute. Emitted only while the compute trace is on.
struct ComputeTraceRow {
    int turn = 0;  // session-mode turn (0 for a one-shot run)
    int phase = 0; // 0 = prefill, 1 = decode
    int step = 0;  // absolute context position of the token being computed
    int seq = 0;   // node's position in the graph this decode (0-based), i.e. execution order
    int layer = -1;
    // ggml's op name (ggml_op_name) and the node's own name. Deliberately raw: which node belongs
    // to attention vs the dense FFN vs the expert matmul is naming policy that varies by
    // architecture, so the engine reports what the graph said and the analysis script classifies.
    std::string op;
    std::string name;
    uint64_t wall_ns = 0; // time to compute THIS node (delta between isolation boundaries)
    uint64_t majflt = 0;  // major page faults charged to this node — flash re-reads inside compute
};

// One flash read. Emitted only while the I/O trace is on.
struct IoTraceRow {
    int turn = 0;
    int phase = 0;
    int step = 0;
    int layer = -1;
    int32_t expert = -1;
    int8_t proj = -1;        // projection slot within the layer (expert layout order)
    int8_t lane = -1;        // which read lane served it
    uint8_t spec = 0;        // 1 if issued speculatively by prefetch
    uint64_t offset = 0;     // absolute file offset requested
    uint64_t req_bytes = 0;  // bytes the caller wanted
    uint64_t read_bytes = 0; // bytes actually read (aligned window; ≥ req_bytes with O_DIRECT)
    uint64_t latency_ns = 0; // wall time in the pread loop
};

// Run-level facts, emitted once before any row.
struct DecodeTraceStatic {
    std::string model;
    std::string arch;
    int n_layer = 0;
    int n_threads = 0;
    int io_threads = 0;
    bool o_direct = false;
    bool overlap = false;
};

// Optional sinks. The engine calls on_static once at open, then on_rows once per decode with that
// decode's rows. Rows are buffered in RAM while the graph runs — the callback and the read path
// must not do I/O — and drained after llama_decode returns.
class IComputeTraceSink {
public:
    virtual ~IComputeTraceSink() = default;
    virtual void on_static(const DecodeTraceStatic &) = 0;
    virtual void on_rows(const ComputeTraceRow * rows, size_t n) = 0;
    // The granularity the sink wants, read once at open. false: a barrier per graph node, exact
    // per-op attribution that serializes the graph against the expert stream. true: a barrier per
    // layer boundary only (see "Layer granularity" above). A property of the trace, not of the run:
    // an untraced session has no granularity to choose.
    virtual bool layers_only() const { return false; }
};

class IIoTraceSink {
public:
    virtual ~IIoTraceSink() = default;
    virtual void on_static(const DecodeTraceStatic &) = 0;
    virtual void on_rows(const IoTraceRow * rows, size_t n) = 0;
};

// Sinks writing `path` as long-format CSV: a `#` preamble carrying the static block, then one row
// per node / per read. `layers` picks the compute trace's granularity (layers_only()). Return nullptr
// if the file cannot be opened.
std::unique_ptr<IComputeTraceSink> make_csv_compute_trace_sink(const std::string & path, bool layers = false);
std::unique_ptr<IIoTraceSink> make_csv_io_trace_sink(const std::string & path);

} // namespace bmoe
