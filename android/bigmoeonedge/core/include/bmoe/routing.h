// A model's own routing, recorded, and what the planner reads from it: the hit-rate curve of an LRU
// expert cache at every size, and which experts are hottest.
//
// The plan used to credit the cache linearly in the fraction of the expert set it holds, which is
// conservative by about 5x on the one measurement on record and wrong in shape everywhere. Routing
// is what decides a cache's hit rate, and routing is a property of the model and of the kind of
// prompt, not of the machine. So a session records the experts each MoE layer asked for during its
// first tokens (RunConfig::moe.routing_record_tokens), hands the record back as a value, and the
// front-end keeps it where it keeps its other state; the next plan of the same file replays it
// through the engine's own cache policy at every size the ledger could allow and prices the hit the
// model actually delivers. See docs/measured-planning.md (Step 1).
//
// Pure: no llama.cpp, no I/O, no clock. The text format is a string in and a string out; where it
// is stored is the front-end's decision.
#pragma once

#include "bmoe/model_profile.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bmoe {

// One expert-cache request: what one MoE layer routed in one decode call, token-major, exactly as
// the streamer is asked to load it. A prompt chunk is one batch per layer holding every token of the
// chunk; a decoded token is one batch per layer. It is the unit the cache accounts in: inside a
// batch each expert is looked up once and eviction runs after it, never taking the batch's own.
struct RoutingBatch {
    uint32_t layer = 0;
    bool decode = false;
    std::vector<int32_t> experts;
};

struct RoutingTrace {
    uint32_t n_layer = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0; // the top-k the routing was recorded at
    // One expert's bytes per block index, what a miss costs and what the cache charges: 0 for a
    // block with no streamed experts.
    std::vector<uint64_t> expert_bytes;
    std::vector<RoutingBatch> batches; // in the order the engine issued them
    uint32_t prefill_tokens = 0;
    uint32_t decode_tokens = 0;

    bool empty() const { return batches.empty(); }
    // Every expert of every block the trace routed through: the size of the set its cache caches.
    uint64_t expert_set_bytes() const;
};

// The trace as text, and back. `parse_routing_trace` also reads a `--route-trace` CSV
// (docs/telemetry.md), rebuilding its batches the way scripts/route-replay.py does - one batch per
// layer for the whole prompt, one per layer per decoded token - so a trace recorded for analysis
// prices a plan too. False with `error` set when the text is neither.
std::string format_routing_trace(const RoutingTrace & trace);
bool parse_routing_trace(const std::string & text, RoutingTrace & out, std::string & error);

// The engine's expert cache replayed over a trace: a global cache over (layer, expert) entries budgeted
// in bytes, one lookup per expert per batch (a repeat in the same batch only promotes), a miss made
// resident at its layer's expert bytes, eviction after the batch that never takes an entry the batch
// touched. `half_life` is the engine's eviction policy (MoeStreamConfig::cache_freq_half_life): 0
// evicts the least recently used, above 0 the least decayed frequency with ties to the least recently
// used, a pass (one token, or one prompt) being a batch whose layer does not follow the previous
// one's - the same arithmetic as the engine, so the two agree entry for entry. A prompt is admitted
// hot: the trace keeps it as one batch per layer, where the engine prefills it in pieces whose last,
// narrow one promotes the experts the cold admission (cache_prefill_cold) put at the drop end, so
// what decode starts from is nearly the same. The LRU row is scripts/route-replay.py's, which
// reproduces the hit rates runs recorded.
struct ReplayCounts {
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t decode_lookups = 0;
    uint64_t decode_hits = 0;
    uint64_t read_bytes = 0;
    uint64_t evictions = 0; // 0: the budget never bound, so every larger budget replays the same
    double hit() const { return lookups ? (double) hits / (double) lookups : 0.0; }
    double decode_hit() const { return decode_lookups ? (double) decode_hits / (double) decode_lookups : 0.0; }
};
ReplayCounts replay_cache(const RoutingTrace & trace, uint64_t budget_bytes, int half_life);
inline ReplayCounts replay_lru(const RoutingTrace & trace, uint64_t budget_bytes) {
    return replay_cache(trace, budget_bytes, 0);
}

// The decode hit share at `points` evenly spaced sizes from nothing to the whole set the trace routed
// through, made non-decreasing (a replay can dip by a lookup where a larger budget evicts in a
// different order; a curve the ranking walks must not). `measured` is set when the trace decoded.
// `half_life` as replay_cache: the policy the run's cache will evict by.
HitCurve hit_curve(const RoutingTrace & trace, uint32_t points, int half_life);

// The experts the trace used most, by block and expert id, every routing counted (a token routing an
// expert is one use, prompt and decode alike). Ranked by uses, ties by block then id, so two calls
// agree. `limit` 0 returns every expert used at all.
struct HotExpert {
    uint32_t layer = 0;
    int32_t expert = 0;
    uint64_t uses = 0;
};
std::vector<HotExpert> hottest_experts(const RoutingTrace & trace, size_t limit = 0);

// A trace shorter than this many decoded tokens is not a curve: the first tokens fill an empty cache,
// and a replay of only those would describe the cold start.
inline constexpr uint32_t k_routing_min_decode_tokens = 32;
// Sizes the curve is replayed at. A replay is a few milliseconds for a record of a few hundred
// tokens (measured: 3-4 ms for 128 decoded tokens over 48 blocks, on a desktop core), and the curve
// stops replaying at the first size nothing was evicted at: every larger one gives the same counts.
inline constexpr uint32_t k_hit_curve_points = 64;
// How far the trace's expert set may differ from the file's before it is called another model's: the
// streamer's per-expert bytes and the probe's per-tensor sum are the same bytes counted two ways.
inline constexpr double k_routing_bytes_tolerance = 0.01;

// The hit-rate curve a model is credited before its own routing was recorded: a curve in the cache's
// size counted in TOKEN CYCLES (the bytes one token routes through), the unit in which the measured
// curves of different models agree. In the share of the expert set they do not: a cache holding a
// tenth of the set served 23% of OLMoE's lookups (64 experts, 8 routed) and 62% of Qwen3.6's (256, 8),
// and at three token cycles every one of the five served 57-74%. Fitted on the replayed curves of five records (Gemma 4
// 26B-A4B, OLMoE 1B-7B, Qwen3.6 35B-A3B at two quantizations, DeepSeek V4) and checked one model left
// out at a time against the decode hits thirteen first loads measured: within 0.04 on average, where
// the share of the set held was off by 0.45. Two constants, not a table, so a model none of those is
// gets the shape they share rather than one of theirs. Never below the share held, which is what a
// cache holding all of the set approaches.
inline constexpr double k_prior_hit_ceiling = 0.92; // what the long-run lookups stay under: cold misses
inline constexpr double k_prior_hit_cycles = 2.25;  // the cycles a cache needs for 63% of that
HitCurve prior_hit_curve(uint64_t expert_bytes, uint64_t token_cycle_bytes, uint32_t points = k_hit_curve_points);

// Check that `trace` describes this model routed at `n_expert_used` (the run's effective top-k), and
// set `model.hit_curve` from it. A trace that does not fit leaves the curve absent with the reason in
// `model.hit_curve.rejected`, so the plan is the one it would be without a trace.
// `half_life`: the eviction policy the run will use (replay_cache).
void attach_routing(ModelProfile & model, const RoutingTrace & trace, uint32_t n_expert_used, int half_life);

} // namespace bmoe
