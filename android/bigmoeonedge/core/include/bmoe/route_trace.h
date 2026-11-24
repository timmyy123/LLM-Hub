// Per-step, per-layer MoE routing trace: the analysis instrument behind --route-trace.
//
// The per-token metrics (metrics.h) say how long a token took and how many expert bytes it
// pulled. They cannot say WHICH experts each layer routed, how strongly the router weighted
// them, or whether a routed expert was already resident — and that is what decides the
// streaming cost. This port exposes exactly that: one row per (step, layer, slot), i.e. the
// cells of a step x layer matrix whose payload is the routed expert ids.
//
// It is a diagnostic, not a product feature. Capturing it asks the compute graph for extra
// nodes (one more barrier per MoE layer) and writes a row per routed expert, so a traced run
// is not a benchmark run. See docs/telemetry.md.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bmoe {

// How a routed expert stood against the expert cache at the instant it was routed — i.e. what
// that routing cost. Ordered by cost: miss pays a flash read, the other two do not.
enum RouteResidency : uint8_t {
    route_miss = 0,     // not resident: this routing reads it from flash
    route_hit = 1,      // resident from an earlier demanded read
    route_hit_spec = 2, // resident because a speculative prefetch guessed it (first touch)
};

// One cell entry: at `step`, layer `layer` routed `expert` in rank slot `slot`. A cell of the
// step x layer matrix is the n_expert_used rows sharing (turn, phase, step, layer).
struct RouteTraceRow {
    int turn = 0;  // session-mode turn (0 for a one-shot run)
    int phase = 0; // 0 = prefill, 1 = decode
    int step = 0;  // absolute context position of the token being routed
    int layer = 0;
    int slot = 0;              // 0..n_expert_used-1, router rank order (descending weight)
    int32_t expert = 0;        // the routed expert id — the matrix cell's payload
    float weight = 0.0f;       // final applied routing weight; NaN if the graph exposed none
    uint8_t residency = 0;     // RouteResidency
    uint64_t expert_bytes = 0; // flash bytes this routing reads; 0 unless residency == route_miss
    // Cache-aware dropping (MoeStreamConfig::drop_cold_frac) discarded this routing: the expert was
    // a miss weighted below the threshold, so it was never read and its weight was zeroed. `weight`
    // and `residency` stay as the ROUTER produced them — the trace records the routing that was
    // chosen, and this flag records what the policy then did with it — but expert_bytes is 0,
    // because a dropped expert costs no read. Always 0 when dropping is off.
    uint8_t dropped = 0;
};

// Run-level facts the rows cannot carry. Emitted once, before any row.
struct RouteTraceStatic {
    std::string model;
    std::string arch;
    int n_layer = 0;
    int n_expert = 0;
    int n_expert_used = 0; // effective top-k, after any override
    // Indexed by layer. expert_bytes is one expert's slices across projections (what a miss
    // costs); dense_bytes is the layer's non-streamed, mmap-resident bytes. Both 0 for a layer
    // with no streamed experts.
    std::vector<uint64_t> expert_bytes_per_layer;
    std::vector<uint64_t> dense_bytes_per_layer;
};

// Optional trace sink. The engine calls on_static once at open, then on_rows once per decode
// carrying that decode's rows (a prefill chunk emits a whole batch's worth at once).
class IRouteTraceSink {
public:
    virtual ~IRouteTraceSink() = default;
    virtual void on_static(const RouteTraceStatic &) = 0;
    virtual void on_rows(const RouteTraceRow * rows, size_t n) = 0;
};

// A sink that writes `path` as long-format CSV: a `#` preamble carrying the static block, then
// one row per routed expert. Returns nullptr if the file cannot be opened.
std::unique_ptr<IRouteTraceSink> make_csv_route_trace_sink(const std::string & path);

} // namespace bmoe
