// Engine configuration.
//
// All tunables flow through these structs: the CLI parses flags into a RunConfig, the
// engine consumes it. The library never reads environment variables — env overrides,
// if any, are resolved in the CLI before this struct is built (see cli/main.cpp), so
// the engine's behaviour is fully determined by its inputs and is trivially testable.
//
// This header is pure policy (no llama.cpp dependency); validate() is unit-tested
// without the native backend.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace bmoe {

// Token sampling. temp <= 0 (the default) selects greedy argmax — the deterministic path the
// byte-identity gates depend on, and today's behaviour for any caller that sets nothing. temp > 0
// builds the standard chain top_k -> top_p -> temp -> dist. Opt-in by construction: sampling never
// perturbs a run that did not ask for it, so the gates stay meaningful.
struct SamplingConfig {
    float temp = 0.0f;           // <= 0: greedy (argmax). > 0: stochastic sampling.
    int top_k = 40;              // 0 disables the top-k stage (llama.cpp convention)
    float top_p = 0.95f;         // nucleus cutoff, in (0, top_p_max]
    uint32_t seed = seed_random; // == LLAMA_DEFAULT_SEED, static_assert'd in session.cpp

    static constexpr uint32_t seed_random = 0xFFFFFFFFu; // the seed that draws a fresh one per run
    static constexpr int top_k_max = 1000;   // the widest cutoff a form offers; validate() does not enforce it
    static constexpr float temp_max = 2.0f;  // the hottest a form offers; validate() does not enforce it
    static constexpr float top_p_max = 1.0f; // a probability mass: the whole distribution
};

// How the dense (non-expert) model weights are kept resident. See MoeStreamConfig::dense_weights and
// core/src/moe/dense_weights.h. Ordered cheapest-effort first.
enum class DenseWeightsMode {
    Mmap,      // leave mmap'd, no help (the A/B baseline)
    Warmed,    // mmap'd, but page-cached once at load
    Anonymous, // read via O_DIRECT into our own anon buffers and rebind (swaps to zram, not flash)
    Pinned,    // as Anonymous, but into reclaim-exempt memory the kernel may not take back
               // (pio::pinned_alloc: a dma-buf on Android, wired memory on Darwin). Init fails where
               // the platform has no such store, so an A/B against Anonymous can never silently
               // compare a mode to itself; a store that runs out part way falls back per tensor
               // and says how much, loudly.
};

// Stable lowercase spelling ("mmap", "warm", "anon", "pinned"), shared by the flag, the plan and the
// telemetry.
const char * dense_weights_mode_name(DenseWeightsMode m);

// Whether route-ahead's prediction scores against an int8 mirror of the router gates or their own
// float weights (MoeStreamConfig::gate_mirror). Both rank the same experts to within an int8 step;
// they differ in cost, and which is cheaper is a property of the machine, so neither is a default.
enum class GateMirror {
    Auto, // timed at load (or by --auto's probe) on the model's own gate shape; the faster path wins
    On,   // the int8 mirror: a quarter of the bytes, plus quantizing the activation per call
    Off,  // the gate's own weights
};

// MoE expert-selective streaming knobs.
struct MoeStreamConfig {
    bool enabled = false; // turn streaming on; init fails fast if the model is not MoE

    // LRU expert cache budget, in MiB. 0 disables the cache (experts are re-read from
    // flash every token via three shared slots). Measured pathology: a budget below one
    // token's working set thrashes (evict + re-read, zero hits) and is SLOWER than off,
    // so validate() rejects the 1..cache_min_mb-1 band unless force_cache is set.
    int cache_mb = 0;

    // Size the cache from the device instead of a fixed cache_mb: once at init the budget is set to
    // (available RAM − cache_floor_mb), clamped to [cache_min_mb, total expert bytes], and then held
    // for the whole run. One shot, not a control loop — it spares the caller a hand-tuned number on a
    // device whose free RAM it cannot know in advance; it does not chase memory pressure afterwards.
    // Mutually exclusive with an explicit cache_mb > 0. See docs/cache-sizing.md.
    bool cache_auto = false;

    // Experts COMPUTED on a device, from a cache of them in its memory, in MiB; 0 is off. The
    // layers go to the device and their experts stay in this streamer's host memory, and llama.cpp's
    // device-side expert cache (llama_context_params::moe_cache_size) uploads a routed expert into
    // its slots from where the streamer put it. On a device that shares the host's memory the upload
    // is a copy within one pool; elsewhere it crosses the device's link. See docs/backends.md.
    int device_cache_mb = 0;
    // Where that cache's misses are uploaded FROM, for experts resident on the host: a copy of them in
    // a buffer of the device's own host buffer type (ggml_backend_dev_host_buffer_type) instead of the
    // file's mapping, the same bytes in the same layout, the expert tensors pointed at it as the
    // streamer points them at its slots. The backend's host memory reaches its device faster where it
    // can be locked; a device with no such type keeps the mapping. The device's name (as
    // ggml_backend_dev_name gives it), empty for off: the planner names the device it measured the
    // gain on, where the host holds the copy.
    std::string expert_host_buffer;
    // The share of that cache's misses COMPUTED ON THE HOST, from the host's copy of the expert, instead
    // of uploaded to the device, once the cache is full (llama_context_params::moe_cache_cpu_share). 0
    // uploads every miss. Above it the device and the host each compute part of a layer's experts and
    // llama.cpp adds the two halves, so the link carries fewer experts while the cores, idle otherwise
    // with every expert on the device, compute the rest; below 1, the misses still uploaded keep the
    // cache learning which experts are hot. The planner prices it from the measured rates
    // (docs/hardware-planning.md). Needs device_cache_mb.
    float device_cache_cpu_share = 0.0f;
    int cache_floor_mb = 1536; // RAM to leave free for the rest of the system when auto-sizing
    int cache_ceil_mb = 0;     // upper bound on the auto budget in MiB (0 = cap only at the full
                               // expert-set size); useful to keep the cache from taking all the
                               // headroom when the marginal hit-rate gain no longer justifies the RAM

    // Parallel expert-slice read lanes (incl. the calling thread). 1 = serial baseline.
    // Clamped to [1, io_threads_max]. 4 is the measured sweet spot on UFS4 phones.
    int io_threads = 4;

    bool o_direct = true;     // bypass the page cache (O_DIRECT / FILE_FLAG_NO_BUFFERING)
    bool load_all = false;    // debug/A-B: load ALL experts each token (full-sweep baseline)
    bool force_cache = false; // allow a cache_mb in the pathological band (tests/experiments)

    // Keep cached experts in RAM where the platform lets a process say so (pio::vm_pin). Quantized
    // weights barely compress, so a kernel that answers pressure by compressing anonymous memory
    // spends RAM holding the cache badly and every hit then pays a decompression: on a 16 GB
    // unified-memory desktop that was 2 GB/s of compression during decode and a third of the
    // decode rate. Where pinning is refused the cache is ordinary memory, as it always was.
    bool cache_pin = true;

    // What a lock refused PART WAY does. Off, pinning ends there and the cache keeps its budget as
    // ordinary memory - right wherever losing a page is cheap to recover, and the only safe
    // default, since the engine cannot tell what a reclaim costs here. On, the budget steps down
    // to what the platform grants and pinning continues: right where reclaim compresses, because
    // there a cache that turns over replaces its locked slices with ones the machine squeezes
    // (0.27 tok/s against 2.48 on one 16 GB machine, same request). The planner arms it exactly
    // when it sized the cache to the lockable total; a hand-set budget is never shrunk unasked.
    bool cache_pin_fit = false;

    // Let the system DROP cached experts under memory pressure instead of writing anything to the
    // drive. Anonymous memory that pressure reaches is swapped (a write to flash), and a pinned
    // cache makes the kernel swap every other process instead: measured on a 16 GB machine, ~9 GB
    // pinned cost 3.3 GB of swap writes in 7 minutes of decode, against 0.13 GB idle. An entry no
    // layer used in the last pass is offered to the kernel (purgeable / MADV_FREE / Offer), which
    // reclaims it before it compresses or swaps anything; a hit on a dropped entry reads it again,
    // from flash, which reads only. Supersedes cache_pin where the platform supports it: a locked
    // page cannot be dropped, and locking is what pushed everything else out.
    bool cache_discard = true;
    // Debug: drop an offered entry just before a lookup takes it back, as if pressure had taken it,
    // so every such lookup pays the re-read on zeroed memory. What the byte-identity gate runs.
    bool cache_discard_drop = false;

    // Halve the expert cache when the system takes it back mid-run: a token whose compute stalls
    // against this machine's own reference, confirmed by a sample of the cache's pages that are no
    // longer in RAM. Shrinks only, never under one token's cycle, and never grows back within the run
    // (docs/pressure.md is why). Inert wherever nothing presses on memory.
    bool cache_brake = true;

    // Which cached expert the budget evicts. 0: the least recently used (LRU). Above 0: the one
    // with the least decayed frequency, every use counting half as much after this many tokens
    // (ties go to the least recently used). Below one token's working set (cache_cycle) LRU is a
    // cyclic scan that evicts exactly what the next token needs and hits 0%; a short half-life keeps
    // the experts the recent tokens shared instead. Replayed on recorded routing of Qwen3.6-35B-A3B
    // and OLMoE (docs/cache-sizing.md), half-life 8 read 25-36% fewer bytes than LRU below the cycle,
    // 1-11% fewer above it, and 2-4% more only at exactly one cycle; 32 and longer lost to LRU above
    // the cycle on Qwen. Measured on a 16 GB Mac (Qwen3.6-35B-A3B Q4_K_M, 192 greedy tokens): 440 MiB,
    // 0% -> 27% hit and 6.1 -> 8.5 tok/s; 1170 and 2900 MiB, 4-5% fewer bytes. Lossless: only what
    // stays in RAM changes. The planner's replay (bmoe/routing.h) uses the same policy.
    static constexpr int cache_freq_half_life_default = 8;
    int cache_freq_half_life = cache_freq_half_life_default;

    // Admit a prefill's misses at the cold end of the cache instead of the hot one. A wide graph
    // touches nearly every expert of a layer, so admitting its misses hot is a cyclic scan: each
    // ubatch evicts what the next one needs (prefill hit 1-5%). Entered cold, a miss is still
    // resident for its own layer but is the first victim, and what the cache held before the prompt
    // survives to serve the next ubatch. Measured on a 16 GB Mac, four 3.6-4.3k-token documents
    // prefilled on the CPU at ubatch 512 with an 8000 MiB cache: 423 -> 211 GiB read, prefill 17%
    // faster, decode unchanged. A graph is wide from `prefill_cold_min_tokens` tokens; a speculative
    // verify window is far narrower and admits hot. Lossless: only what stays in RAM changes.
    bool cache_prefill_cold = true;
    static constexpr int prefill_cold_min_tokens = 32;

    // Overlap async expert reads with FFN compute instead of blocking on them: load_layer()
    // publishes the reads and returns immediately, and the CPU mul_mat_id kernel blocks per
    // expert (via the fork's expert-ready hook) only if that expert's slice is not yet in.
    // Requires the Helldez/llama.cpp fork submodule (the hook); run() fails fast otherwise.
    bool overlap = false;

    // Two-wave batch publish (#118). A layer's read batch normally becomes visible to the I/O
    // lanes only after ALL of its staging finished — up to three page-commit syscalls per cold
    // expert sitting in front of the first byte of I/O, on the latency-to-first-slice path the
    // sidecar refutation identified as the binding constraint. With this on, the jobs of the
    // first projection (the one mul_mat_id blocks on first) are committed and published
    // immediately, and the remaining projections are committed and appended while the lanes
    // already read. Overlap + LRU cache only. Default off pending the on-device A/B.
    bool io_two_wave = false;

    // Temporal prefetch: while a token computes layer l, speculatively read on the idle I/O
    // lanes the experts the PREVIOUS token routed at layers l+1..l+prefetch_layers, betting on
    // the strong temporal locality of routing. A correct guess turns the next layer's read into
    // a cache hit; a wrong guess only wastes a read. 0 disables it. Needs the LRU cache on —
    // a fixed cache_mb or cache_auto (speculative slices land in the per-layer cache buffers),
    // so validate() rejects it when both are off. See docs/prefetch.md.
    int prefetch_layers = 0;

    // How many tokens of this model's routing the session keeps for the planner: the experts each
    // MoE layer asks the cache for, for the first N decoded tokens and the last N of the prompt
    // before them (Session::routing_record). A front-end that keeps the record hands it to the next
    // --auto plan of the same file, which then prices the cache on the hit rate this model's routing
    // actually gets at every size instead of a linear credit (bmoe/routing.h). It asks the graph for
    // nothing - the ids are gathered on every routing anyway - and stops by itself; 0 records nothing.
    // Lossless: it changes no output.
    int routing_record_tokens = 256;
    // Record the routing on a run that does not stream as well. With no streamer the routing nodes
    // are not otherwise observed, so recording them costs one barrier per MoE layer per token until
    // the record is full, and nothing after. The planner sets it where a choice it has to make is
    // priced by that record and none exists yet (the device tier); off, an unstreamed run records
    // nothing, as before.
    bool record_unstreamed = false;

    // How the dense (non-expert) weights are treated. The streamer only rebinds experts; the rest —
    // gguf header/metadata, embeddings, attention, norms, lm_head — is handled by one of three
    // policies (see core/src/moe/dense_weights.h):
    //   Mmap      leave them mmap'd; the kernel serves and reclaims them (a >RAM model then demand-
    //             faults them 4 KiB at a time inside the first decodes — the slow-start this exists
    //             to address). The A/B baseline.
    //   Warmed    leave them mmap'd, but page-cache them once at load with a sequential sweep, so the
    //             first decodes do not fault them in. Moves the cost into load_seconds. Best when the
    //             model fits in RAM.
    //   Anonymous (default) read them once via O_DIRECT into our own anon buffers and rebind the
    //             tensors, so a reclaim swaps them to zram (fast) instead of dropping them to a 4 KiB
    //             flash refault. The measured win on a model actively losing its dense set to reclaim,
    //             and the policy the Android app ships by default — the CLI matches it here.
    DenseWeightsMode dense_weights = DenseWeightsMode::Anonymous;

    // ── row-gathered dense tables served from flash (see bmoe/row_source.h) ──────────
    // A dense weight the graph only ever GATHERS ROWS from — a token embedding table is the pure
    // case — is resident for nothing: one row of it is read per decoded token and the rest of the
    // hundreds of MiB sits in RAM the expert cache is competing for. With this on, such a table is
    // bound to reserved address space and only the rows the graph asks for are pulled from flash,
    // inside a bounded window.
    //
    // WHICH tables qualify is decided by the graph at capture time and by nothing else: every
    // reference to the tensor must be a row gather with a readable index. No architecture, tensor
    // name or size threshold appears in the rule, so a model whose embeddings are also used as the
    // output head simply never qualifies, on any architecture, without a special case for it.
    //
    // Off by default: it trades residency for a read per gather miss, and which way that trade goes
    // is a measurement per device class, not a foregone conclusion.
    bool row_stream = false;
    int row_stream_mb = 64; // resident window across all row-streamed tables, in MiB

    // ── tables the plan leaves mapped ─────────────────────────────────────────────────
    // Tensor names the hardware planner priced as left mapped: the graph gathers rows from them,
    // so a resident copy buys bytes nothing reads. Under Anonymous/Pinned they stay mapped, as a
    // tensor too large to hold does; every other mode is unchanged. The names are read off the
    // file by the planner's own grouping, never written here. Measured why it matters: a 28.8 GB
    // per-layer table that "fits" by the kernel's count was copied beside a 25.8 GB expert cache
    // on a 42.6 GiB control group and the run was killed at 43 GB.
    std::vector<std::string> dense_mapped;

    // ── release the model file's mapping after load (Windows) ─────────────────────────
    // llama.cpp keeps the gguf mapped for the model's lifetime, and on Windows a live section of
    // the file serialises the streamer's concurrent unbuffered reads: four lanes deliver one
    // lane's throughput (measured; see core/src/io/mapping_release.h). With this on, once nothing
    // reads through the mapping any more (dense weights copied out under anon/pinned, every file
    // tensor either streamed, copied or row-served) the engine unmaps the file and closes its
    // section. On POSIX the mapping is munmap'd the same way; Android was measured not to serialise
    // reads, so there the gain is smaller and of a different kind (see docs/moe-streaming.md).
    //
    // Opt-in, and it must stay opt-in until the safety question has a positive answer rather than a
    // blacklist. The release is correct only while nothing still dereferences the mapping, and
    // llama.cpp exposes no way to enumerate a loaded model's tensors and prove that. The session
    // declines on every shape where a surviving pointer is KNOWN — a dense policy that keeps the
    // weights mmap'd, a tied output head, an unowned tensor under the MTP draft — but a future
    // architecture can invent another one. See Session::open.
    bool release_mmap = false;

    // ── cache-aware expert dropping (lossy; opt-in) ──────────────────────────────────
    // Skip a routed expert when it is a cache MISS *and* the router weighted it below
    // drop_cold_frac × (1 / n_expert_used) — i.e. below that fraction of the uniform share a
    // top-k routing would give each expert. 0 (the default) disables it and the engine is
    // bit-exact as before.
    //
    // The asymmetry is the whole idea: an expert already resident costs no flash read, so it
    // always runs however small its weight. Quality is spent only where it buys I/O. Because
    // the largest weight in a routing is always >= the uniform share, a frac of 1.0 can never
    // empty a routing; validate() rejects anything above it, and the implementation additionally pins the
    // top-weighted expert so no cell is ever left with nothing to compute.
    //
    // This changes the output — it is a quality/throughput trade like n_expert_used, not an
    // optimisation. Unlike n_expert_used it is *state-dependent*: the same prompt can decode
    // differently depending on what the cache happened to hold, so a run is no longer
    // reproducible token-for-token. See docs/expert-dropping.md.
    float drop_cold_frac = 0.0f;

    // Rescale the surviving weights so the routing still sums to what it did before the drop.
    // Without it the layer's expert output is systematically scaled down by the discarded mass
    // (~10% at frac 1.0), which perturbs the residual stream more than the missing expert does.
    bool drop_renorm = true;

    // Apply dropping during prefill too. Off by default and deliberately so: the cache is cold
    // there, so nearly every expert is a miss and the same threshold discards ~4x the weight
    // mass it does in decode (measured; see docs/expert-dropping.md). Prefill is also
    // compute-bound, so there is little to win.
    bool drop_prefill = false;

    // ── cache-aware substitution (lossy; opt-in) ─────────────────────────────────────
    // Dropping decides whether to PAY for a missing expert. This decides whether to NEED one:
    // before a routing is committed, every expert's router score is raised by
    // substitute_lambda × (this token's score range) if that expert is already resident, and the
    // top-k is taken again. A resident expert therefore wins a slot only when it was within that
    // margin of the one it displaces — a confident routing is untouched, a near-tie resolves
    // toward RAM.
    //
    // The margin is a fraction of the RANGE, so one value means the same thing whatever scale a
    // model's router scores live on: no per-model constant, and no calibration state. The weights
    // are not touched: the graph applies whatever the router itself gives the selected experts.
    //
    // It runs the same NUMBER of experts, just cheaper ones. It is still lossy — they are not the
    // experts the router asked for — and, like dropping, state-dependent. 0 (the default) disables
    // it and the engine is bit-exact as before. Decode only; needs a live cache to report
    // residency. See docs/cache-aware-substitution.md.
    float substitute_lambda = 0.0f;

    // Diagnostics: measure how predictable the routing is, without acting on it. For every decoded
    // token the engine ranks each layer's experts a layer early — running the NEXT layer's gate
    // matrix on the CURRENT layer's gate input, which the residual stream keeps nearly unchanged —
    // and reports what fraction of the routing that would have had in flight. Scored alongside it:
    // the previous-token bet --prefetch already makes, and a zero-staleness control that says how
    // much of any gap is the ranking's own approximation rather than the staleness.
    //
    // Nothing it computes reaches the loading path: a probed run reads exactly the bytes an
    // unprobed one does. It is not free, though — one isolated node and one gate GEMV per MoE layer
    // per token, on the eval thread — so a probed run is not a benchmark run. Requires streaming.
    // See docs/expert-prediction.md.
    bool predict_log = false;

    // Predictive prefetch: act on the stale-gate prediction instead of just logging it. While
    // layer l computes, the NEXT layer's router matrix is run on l's gate input (the same one
    // GEMV predict_log measures) and the predicted experts are handed to the same speculative
    // read path --prefetch uses — same cache buffers, same accounting, same settle. Two things
    // separate it from the temporal prefetch it replaces:
    //   - the prediction is about THIS token (measured ~89% of routed slots on a 128-expert
    //     model, vs ~43% for the previous-token bet), so far fewer speculated bytes are wasted;
    //   - it is drop-aware: with drop_cold_frac armed, predicted experts whose predicted routing
    //     weight falls below the drop threshold are not prefetched — if they miss they would be
    //     dropped unread anyway, so reading them ahead would spend the exact I/O the drop policy
    //     exists to save.
    // Decode only. Needs the LRU cache (speculative slices land in the per-layer cache buffers)
    // and is mutually exclusive with prefetch_layers — one speculative predictor at a time, or
    // the lanes fill with two guesses about the same future. Costs one gate GEMV per MoE layer
    // per token on the eval thread. Byte-identity: like --prefetch, it only warms the cache, so
    // output is unchanged — except under drop_cold_frac, where residency is an input to the
    // routing policy and a correct guess un-drops an expert (same caveat as --prefetch, see
    // docs/expert-dropping.md).
    bool predict_prefetch = false;

    // How many predicted MISSES per layer the predictive prefetch may speculate. The rest of the
    // prediction still works at any value: predicted experts already resident are retained
    // (LRU-protected) whatever this says, because that costs zero bytes. 0 is the retention-only
    // mode — the prediction spends no flash at all and only protects; the measured sweet spot for
    // speculation, if any, is small (the stall a prefetch can remove is head-of-line only).
    int predict_spec_max = 2;

    // ── route-ahead: commit to the prediction (lossy; opt-in; experimental) ───────────
    // With N > 0, decode routing at MoE layer L is REPLACED by the ranking layer L's own gate
    // matrix produced on the hidden state N layers earlier in the same forward pass — the
    // stale-gate prediction predict_log measures, acted on as the routing itself instead of as a
    // prefetch hint. The router still computes: its logits give the substituted experts their
    // true renormalized weights, and each layer's gate input feeds the prediction for layer L+N.
    // Layers 0..N-1, prefill, the first decode token and any unreadable layer route normally.
    //
    // Why anyone would do this: a prediction that IS the routing cannot miss, and it is known a
    // full N layers of compute early — the perfect-prefetch regime no honest predictor reaches
    // (measured here: one layer of staleness costs ~3.5-4pp of slot agreement, and the union of
    // four stale horizons still covers only 81-89% of misses). With the LRU cache on, the engine
    // acts on that: the committed selection of layer L+N is handed to the speculative read path
    // the moment it is fixed (at layer L's load), uncapped — those reads can never be wasted —
    // so by the time layer L+N runs its experts are resident or already in flight. Without the
    // cache the selection still commits but nothing is read early. The price is a quality
    // perturbation: ~15-20% of slots route to a different expert than the router chose at N=1,
    // and the error compounds through the residual stream. LOSSY by construction; changes the
    // output. See docs/route-ahead.md.
    //
    // Mutually exclusive with the probe and both prefetchers: the probe would score a predictor
    // against a routing this policy rewrote from that same predictor (a tautology), and a
    // speculative prefetch would bet lanes on a future this policy has already fixed. Also
    // mutually exclusive with self-speculation, for a measured reason rather than a conceptual
    // one: a verify decode is several positions wide and this policy declines to commit on every
    // one of them, so the pair pays the prediction and the early reads and commits nothing.
    int route_ahead = 0;

    // What route-ahead's prediction GEMV reads (see GateMirror). Inert unless route_ahead > 0. The
    // committed selection is a ranking, and the int8 mirror changes a score by far less than the
    // margins between experts, so the choice is about cost: measured on device the mirror took the
    // prediction from 19 to 6 ms/token at 74.5% agreement with the router against 74.7% for the
    // exact weights, while on an x86 host with bandwidth to spare the float path was faster. Like
    // route-ahead itself the choice can change which expert a near-tie commits to; it is lossy only
    // where route-ahead already is.
    GateMirror gate_mirror = GateMirror::Auto;

    // Test/debug only: complete each prefetch's speculative reads synchronously, on the eval
    // thread, before returning. This defeats the latency-hiding purpose (the reads no longer
    // overlap compute) but makes speculative integration deterministic, so the byte-identity
    // gates can exercise the integrate-then-hit path that a timing race otherwise seldom reaches
    // on a fast host. Serial mode only. Never set in production.
    bool prefetch_sync = false;

    // Top-k at or below which drop_cold_frac gets a warning. This is an EVIDENCE boundary, not a
    // physical one: the policy is measured at top-k 8, where the uniform share is 12.5% and the
    // threshold trims a long tail. At top-k 4 that share is 25% and at top-k 2 it is 50%, so the
    // same fraction removes a far larger part of the routing — a different regime, unmeasured.
    // Nothing in the streaming path reads this; it only decides whether the caller is told.
    static constexpr int drop_low_topk_warn = 4;

    static constexpr int cache_min_mb = 1500; // smallest non-pathological cache (see above)
    static constexpr int io_threads_max = 8;
    static constexpr int prefetch_layers_max = 8;
    static constexpr int cache_freq_half_life_max = 4096; // tokens; far beyond it is plain frequency

    // TEST ONLY: the width from which a graph's misses are admitted cold, in tokens (0 keeps
    // prefill_cold_min_tokens). Lets a gate reach the cold path whatever its prompt's length.
    int test_prefill_cold_min_tokens = 0;
    // A record's memory is a few ints per layer per token; this keeps the longest one in single MiB.
    static constexpr int routing_record_tokens_max = 4096;
    static constexpr int route_ahead_max = 8; // beyond this the staleness has no measured meaning
    static constexpr int predict_spec_max_limit = 8;
    static constexpr float drop_cold_frac_max = 1.0f;         // the uniform share: never empties a routing
    static constexpr float substitute_lambda_max = 1.0f;      // the whole score range: residency always wins
    static constexpr float device_cache_cpu_share_max = 1.0f; // every miss on the host: the cache stops learning
};

// Where the draft tokens of a self-speculative step come from. The verify half of the loop is
// identical for both sources; only the producer differs, and with it what a draft costs.
enum class DraftSource {
    none,  // no speculation: one token per decode
    mtp,   // the model's own trained multi-token-prediction head (needs the nextn block)
    ngram, // prompt-lookup over prompt + generated tokens (needs nothing from the model)
};

// Self-speculative decoding: draft draft_max continuation tokens, then verify all of them in ONE
// decode and accept the longest prefix whose argmax agrees with the draft. Nothing is approximated
// and no weight is skipped, so this is a latency optimisation, not a quality trade.
//
// It is NOT byte-identical the way overlap and prefetch are, and must not be used in a byte-identity
// gate. Verification evaluates 1 + draft_max positions in one batch, and a batched matmul is not
// bit-identical to that many single-token ones; on a near-tie the argmax can flip. Measured on
// Qwen3.6-35B-A3B-MXFP4 over 128 greedy tokens: one token differs from an unspeculated run, and
// every draft width agrees with every other. The caveat is a property of the batched verify, so it
// applies to every draft source. See docs/mtp.md.
//
// Why it can win, and why it can lose. Verifying N positions in one decode reads the dense weights
// and each routed expert ONCE for N tokens instead of N times, which is the whole prize: on a
// DRAM-bandwidth-bound host it amortises the measured bottleneck directly, and on a flash-streamed
// device it amortises latency-to-ready. The counterweight is that the N verify positions route
// INDEPENDENTLY, so a layer's read set widens toward N*k experts instead of k. Measured at draft 3
// on the host, that widening is 97.1% of the extra bytes a speculated run streams — it is what the
// choice of source cannot change. Default off until the A/B says otherwise on the target device.
struct SpecConfig {
    DraftSource source = DraftSource::none;

    // MTP only. A GGUF holding the model's trained MTP head on its own, for a model file that ships
    // without it: llama.cpp's converter writes one (convert_hf_to_gguf.py --mtp). Empty uses the head
    // inside the model file. The head is opened as a model of its own and drafts in a context that
    // reads the target's hidden states, as llama.cpp's own speculative driver loads a draft head.
    std::string mtp_head;
    // MTP only, with a head file: the head's weights on the devices the target's layers are on
    // (true) or on the host (false). The planner sets it where a device computes this model's
    // experts faster than the host by the margin - the head's block is one more expert block - and
    // the head fits in the bytes the capacity fitter gave the experts, which it takes from the
    // device expert cache; the target's own placement is untouched. Measured why: on a 12 GB card
    // with 8 slow cores, drafting on the host cost 35 ms of a 65 ms token and lost to no head at all.
    bool mtp_head_on_devices = false;

    // Tokens drafted per verify batch. The verify decode is 1 + draft_max positions wide, so this
    // sets both the ceiling on tokens-per-decode and the width of the read-set widening above.
    // Past the source's acceptance horizon the extra drafts are rejected and paid for anyway, which
    // is why the useful range is small; 3 is upstream's default for a single trained head.
    int draft_max = 3;
    // The narrowest ubatch an MTP draft context is reserved at: it computes 1 + draft_max positions
    // at decode, a prompt chunk in pieces this wide at prefill (Session::open, and the placement
    // probe that fits it beside the target).
    static constexpr int mtp_draft_ubatch = 32;

    // MTP only. Confidence floor for continuing to draft, as the head's own probability for the
    // token it is proposing. At 0 the head is asked for draft_max tokens unconditionally, however
    // unsure it is; above 0 it stops as soon as its best candidate falls below this, making the
    // draft width adaptive per step at no cost.
    //
    // This is the cheap half of expert-cost-aware drafting, and on a streamed device it pays twice:
    // a draft not made is a pass through the MTP block (with its own MoE FFN) that does not happen,
    // AND one fewer position in the verify batch, so one fewer independent routing widening the
    // layer's read set. A rejected draft costs both of those and buys nothing.
    //
    // 0 by default — it is the width the host measurement was taken at, and the useful value is a
    // property of the device's balance between drafting cost and acceptance, so it is a knob to
    // measure rather than a constant to guess.
    float draft_p_min = 0.0f;

    // N-gram only. Shortest suffix match that is allowed to draft. This is the whole economics of
    // the n-gram source: below it the step drafts NOTHING and costs exactly a plain decode, so the
    // floor of the feature is the baseline rather than a loss. Raising it buys precision (fewer,
    // better drafts) at the cost of coverage. 3 because free prose rarely repeats a trigram, which
    // is what keeps prose runs on the floor while copy-heavy segments still match.
    int ngram_min_match = 3;

    // N-gram only. Longest suffix considered when looking for a match. A cap, not a target: it
    // bounds the per-step scan (O(corpus * ngram_max_match)) and stops a long verbatim quote from
    // making every step scan its full length for no additional selectivity.
    int ngram_max_match = 12;

    static constexpr int draft_max_limit = 8;
    static constexpr int ngram_match_limit = 64;
    static constexpr float draft_p_min_max = 1.0f; // a probability: only drafts the head is sure of

    bool enabled() const { return source != DraftSource::none; }
    bool is_mtp() const { return source == DraftSource::mtp; }
    bool is_ngram() const { return source == DraftSource::ngram; }
};

// Prefill on an accelerator, decode on the CPU. Off unless `device` names a ggml device.
//
// Prefill and decode want opposite hardware. A prefill graph is hundreds of tokens wide, so a
// matrix engine (the Hexagon HMX, a GPU) runs it at many times the CPU's rate; a decode graph is
// one token wide, and on unified memory the per-graph boundary cost of an accelerator eats what it
// saves there. So the weights are moved per GRAPH: before a wide prefill graph the layer weights
// are rebound onto the device's own buffer, and afterwards back onto the CPU mapping. The llama.cpp
// scheduler assigns every op to the backend holding its weight, and it re-decides that on every
// graph it builds, so the rebind is all it takes — no llama.cpp change.
//
// The one hazard is graph REUSE: llama.cpp skips rebuilding (and re-scheduling) a graph of the same
// shape as the previous one. The invariant that makes the rebind safe is therefore about widths:
// every graph run on the device is at least `min_tokens` wide and every graph run on the CPU is
// narrower, so two consecutive graphs on different placements never share a shape.
struct PrefillDeviceConfig {
    // ggml device name as the backend registry reports it (e.g. "HTP0" for the Hexagon NPU).
    // Empty = off. A local device only: no front-end registers a remote (RPC) one.
    std::string device;

    // What a device that was found but fails to OPEN its prefill path does to the load. Off, the
    // load fails and says why: a caller who named a device wants to know it is not being used. On,
    // the session falls back to the CPU and runs, which is what a plan that armed the device on
    // its own must ask for - a speed-up the planner chose is never a reason for a model not to load.
    bool best_effort = false;

    // Confirm the device on THIS model before the session keeps it: one prefill of a fixed text
    // computed on the host and on the device, compared row by row. A device that does not pick the
    // host's next token in nearly every row is handed back like one that failed to open
    // (best_effort decides how). The planner's test is one synthetic matmul, which cannot see a
    // fault only the model's own activations trigger: a phone GPU's half-precision prefill turned
    // a model into noise. Whether the device is FASTER is not judged here - at this width cold
    // expert reads dominate - and stays the planner's call. The cost is three short prefills at
    // load; false skips it.
    bool verify = true;
    static constexpr int verify_tokens = 64; // the probe prefill's width, under max_outputs
    // Positions compared, spread across that width. Each is a row of the output head, the model's
    // largest matrix, and from 32 rows up ggml's BLAS backend unpacks a quantized matrix whole into
    // float32 and keeps that scratch: 1940 MiB on a 248k vocabulary at load, outside every budget.
    // As many rows as stay under that, for the most statistics a narrow product allows.
    static constexpr int verify_rows = 31;
    static constexpr float verify_min_agree = 0.9f; // rows whose top token must match the host's

    // How many layers of streamed experts the device holds at once (with moe.enabled). Two reads a
    // layer while the one before it computes; one reads each layer at its own routing node, with
    // the device waiting, and costs a layer less of memory. Same result either way.
    int slots = 2;
    static constexpr int slots_min = 1;
    static constexpr int slots_max = 2;

    // Hold the expert slots only while a prefill runs on the device, and give their memory to the
    // expert cache the rest of the time (with moe.enabled). The two are never busy at once: a
    // device prefill reads through the slots and not the cache, a decode through the cache and not
    // the slots. Off, the slots are held for the session and the cache is sized around them. On,
    // the cache keeps its full budget for decode; each prefill evicts as much of it as the slots
    // take and builds them, which is paid in re-reads and in a setup per prompt.
    bool slots_on_demand = false;

    // Narrowest prefill graph sent to the device. A prompt is fed in ubatch-wide pieces; pieces at
    // least this wide run on the device and a shorter tail runs on the CPU. It must exceed every
    // graph width the CPU path uses, which is one token for plain decode — speculation's wider
    // verify batch is excluded by validation for now.
    int min_tokens = 32;

    // Loader threads of the streamed-expert arena (with moe.enabled). Separate from moe.io_threads,
    // which sets the decode's read lanes: a prefill graph has the whole CPU idle while the device
    // computes, and each loader both reads and repacks (a K-quant repack is CPU-heavy), while decode
    // wants few lanes. Measured on a Q4_K_M: 4 loaders left the device waiting 10.3 s of a 15.1 s
    // prefill, 8 left it 5.0 of 10.0.
    int load_threads = 8;
    static constexpr int load_threads_max = 16;

    // TEST ONLY: sleep this long before the first upload of each layer in the arena. On a tiny
    // model the loaders always beat the graph to the next layer, so a missing barrier would go
    // unnoticed; slowed down, the barrier is the only thing keeping compute behind the loads, and the
    // gate proves it. Never set by a front-end.
    int test_load_delay_us = 0;

    // Read only the experts each device graph routes to, predicted from the previous graph's routing
    // and completed at each routing node (DeviceExpertArena::set_routed); the result is bit for bit
    // the same as reading whole layers, which `false` restores. Only the expert arena of a streamed
    // model reads experts, so elsewhere it has no effect. `routed_full_frac`: a layer routing more
    // than this fraction of its experts gets the next layer read whole.
    bool routed = true;
    float routed_full_frac = 0.85f;

    // Fill the arena from the expert cache where it can: an expert the decode cache holds is copied
    // from RAM instead of read from flash. The arena never admits into the cache and a device graph
    // never evicts from it, so what the cache held before the prompt is what serves it; on a follow-up
    // turn that is most of what the prompt routes to. The bytes are the same either way.
    bool from_cache = true;
    static constexpr float routed_full_frac_min = 0.0f; // exclusive: 0 would read every layer whole
    static constexpr float routed_full_frac_max = 1.0f; // inclusive: 1 never falls back
    // TEST ONLY: in routed mode, skip the reads at the routing node, so the graph computes on whatever
    // the slot held for the experts the prediction missed. The gate uses it to prove it would see that.
    bool test_routed_skip_demand = false;

    static constexpr int min_tokens_floor = 2; // decode is one token wide; this keeps the shapes apart

    // Logit rows a graph may produce, on every run and not only with the device on. llama.cpp
    // reserves compute memory for as many as the ubatch is wide, and a row is the whole vocabulary:
    // 2048 rows of a 248k-token vocabulary were 2 GB of reservation (measured on a 35B-A3B), which
    // starved decode into thrashing. A prompt needs one row; a perplexity pass is fed in pieces this
    // wide instead.
    static constexpr int max_outputs = 128;

    bool enabled() const { return !device.empty(); }
};

// Whether Session::decide() keeps the model state after a request's prefix and restores it when the
// next request's prefix extends it (bmoe/decide.h).
enum class PrefixCacheMode {
    // On where splitting a prompt at the prefix costs only the tokens on each side — the CPU, where
    // prefill time scales with the tokens fed. Off where every prefill graph costs about the same at
    // any width, because there a second graph for the few tokens after the prefix costs as much as
    // prefilling the whole prompt again.
    Auto,
    On,
    Off,
};

// Stable lowercase spelling ("auto", "on", "off"), shared by the CLI flag and the telemetry.
const char * prefix_cache_mode_name(PrefixCacheMode m);
// The inverse of prefix_cache_mode_name. False, and `out` untouched, on any other spelling.
bool parse_prefix_cache_mode(const std::string & s, PrefixCacheMode & out);

// Session::decide() policy. Fixed for the session.
struct DecideConfig {
    // Off by default: decide() then refuses every request and nothing is allocated for it, so a
    // session that does not use decisions is exactly what it was before the feature existed.
    bool enabled = false;
    PrefixCacheMode prefix_cache = PrefixCacheMode::Auto;
    // Experimental diagnostic: append one JSONL line per decision to this file, with the experts each
    // layer routed and the answer read at every layer's exit (see core/src/moe/decide_probe.h).
    // Empty = off. Needs the eval callback, i.e. --moe-stream or a prefill device.
    std::string probe_path;
};

// Which compute devices llama.cpp is handed at load.
enum class DeviceUse {
    Auto,    // every device when a layer is placed on one, the CPU alone otherwise
    All,     // every device llama.cpp finds, placed on or not
    CpuOnly, // the CPU and nothing else
};

// The element type of the K and V caches, by llama.cpp's own names (its --cache-type-k and
// --cache-type-v). F16 is llama.cpp's default. Every other type changes the output, the quantized
// ones by storing the cache in fewer bits (llama.cpp rotates a quantized cache's heads first, which
// is most of what keeps 4 bits usable). The planner never picks one; the default below is the one
// lossy choice the engine makes for its caller.
enum class KvCacheType { F16, BF16, F32, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0, IQ4_NL };

// Q8_0 by default, where llama.cpp keeps F16: on this engine the KV is memory the expert cache does
// not get, and a long agent chat (18 turns to 60k on Qwen3.6-35B-A3B, 16 GB) decoded in 281-303 s
// at Q8_0 against 442-490 s at F16. F16 is the lossless choice, one flag away.
inline constexpr KvCacheType k_default_kv_cache_type = KvCacheType::Q8_0;
// Every value, in declaration order, and its llama.cpp name: the one list both the parameter table
// and the telemetry read.
inline constexpr KvCacheType k_kv_cache_types[] = {KvCacheType::F16,  KvCacheType::BF16, KvCacheType::F32,
                                                   KvCacheType::Q8_0, KvCacheType::Q5_1, KvCacheType::Q5_0,
                                                   KvCacheType::Q4_1, KvCacheType::Q4_0, KvCacheType::IQ4_NL};
const char * kv_cache_type_name(KvCacheType t);
// Lossless knobs measured on the first real request (docs/measured-planning.md, Step 2). Empty by
// default, and then nothing is measured: the plan arms it when the front-end keeps the result
// (bmoe/measured_knobs.h), or a person does by hand. Once the cache's hit rate has settled, decode
// windows alternate between the incumbent and each candidate, A B A B, every window decoding the
// user's own answer; a candidate replaces the incumbent only if it decodes faster by the planner's
// device margin in the median of its rounds. The winner stays for the rest of the session and is
// returned (Session::measured_knobs). Changes no output: every knob here is lossless.
struct KnobTrialConfig {
    // Decode thread counts tried against n_threads, at most threads_max_candidates. Switched live
    // (llama_set_n_threads); prefill keeps n_threads_batch.
    std::vector<int> threads;
    // Also try the other setting of moe.overlap. Switched live between two decodes, by giving the
    // streamer's first read lane a worker of its own or taking it back; needs streaming, the
    // expert-ready hook, and no layer on a device (overlap changes how a context offloads, which is
    // fixed when it is created).
    bool overlap = false;
    // Tokens per timed window. Small values are for tests: a window of a token or two times the
    // noise of one token.
    int window_tokens = 8;
    // The device tier's side this load measures, decided across loads (MeasuredKnobs::device_cache_tps):
    // 1 times the decode with a device expert cache armed, 0 without one, -1 times neither. Nothing
    // switches: the load runs the side the plan gave it, and its decode rate is measured once the
    // knobs above have their verdicts, so their switches are not in its windows.
    int device_tier = -1;
    // The graph width's side this load measures, decided across loads (MeasuredKnobs::width_narrow):
    // 1 the plan's own width, 0 the narrower `width_narrow`; -1 neither. The load times its decode as
    // above, and its prefill over the prompts longer than `width_narrow`, the only ones the width
    // changes, with the requests it served, which the plan weighs the two sides by.
    int width_side = -1;
    int width_narrow = 0;
    // The device cache's CPU share, decided across loads (MeasuredKnobs::cpu_share_tps): 1 times the
    // decode with part of the cache's misses computed on the host (moe.device_cache_cpu_share, as the
    // plan priced it), 0 with every miss uploaded; -1 neither. A share is fixed when a context is
    // created, so a side is timed by a load that serves it, as the device tier's are.
    int cpu_share_side = -1;
    // Where a resident model runs on a machine whose devices read the host's own memory, decided across
    // loads (MeasuredKnobs::shared_device_tps): 1 times the decode with every layer on the device the
    // plan named (device_split), 0 with every layer on the cores; -1 neither. Which engine decodes the
    // model is fixed when it is loaded, so a side is timed by a load that serves it.
    int shared_device_side = -1;
    // The device whose answers for this model the load compares with the cores' before it serves
    // (ggml's name for it, the key the verdict is kept under: MeasuredKnobs::device_answers); empty,
    // nothing is compared. A plan that moves a whole model onto a device by its own choice asks for
    // it, because the probes that let the device through each compute a part of a model in isolation
    // and none of them is this model's output: on a 16 GB Apple-silicon laptop one small vision model
    // wrote one repeated character with every layer on the GPU and sane text on the cores, where two
    // other models were correct on the same GPU and every probe passed for all three. The load decodes
    // a short fixed sequence on the placement as opened and on a host-only instance of the same file,
    // and a placement that does not pick the cores' next token is not served from: the session reopens
    // on the cores (engine/device_answer.h). Not a timed side: it measures no rate and switches nothing.
    std::string answer_device;
    // Which side then serves, decided by that same load on this model: the comparison decodes the
    // model on the device and on the cores anyway, so it times both (a token decoded alone, and a
    // graph as wide as a prompt) and the session keeps the side a request is shorter on, reopening on
    // the cores where they are faster by the trial's margin. This is the prompt tokens a request
    // brings per decoded token, the weight of the wide timing against the narrow one; below 0 nothing
    // is weighed and a device whose answers match keeps the model, as before. A plan asks for it where
    // no load has measured both sides on this machine: the probes of a part of a model that let the
    // device through do not say which side decodes the model faster (MeasuredKnobs::shared_device_tps
    // has the measurements), and the model itself does, at the first load.
    float answer_prompt_per_token = -1.0f;
    static constexpr float answer_prompt_per_token_max = 1e6f;
    // TEST ONLY: the comparison sees the device's rows shifted by one token, so a gate reaches the
    // refusal on a device that computes correctly. Never set by a front-end.
    bool test_answer_differs = false;

    bool enabled() const {
        return !threads.empty() || overlap || device_tier >= 0 || width_side >= 0 || cpu_share_side >= 0 ||
               shared_device_side >= 0;
    }

    static constexpr int threads_max_candidates = 2;
    static constexpr int window_tokens_max = 256;
};

// A full run: model, prompt, decoding, streaming, telemetry.
struct RunConfig {
    std::string model_path;
    // The multimodal projector (an `mmproj` GGUF) that gives the model image input, loaded beside it.
    // Empty: text only. Any model llama.cpp's projector library knows, any quantization of either
    // file: the projector is read by its own header, nothing here names a family.
    std::string mmproj;
    // The most tokens one image may take, for a projector whose token count follows the image's
    // size (0 = the projector's own cap, thousands for a photo). Each is prefilled as a prompt token,
    // so the cap is what an image costs: a host that prefills image tokens slowly wants few.
    int image_max_tokens = 0;
    std::string prompt = "The capital of Japan is";
    // Image files the prompt places with image_marker() (bmoe/chat.h), in order: the one-shot run's
    // input beside `prompt`, read by the front-end and handed to the session as bytes.
    std::vector<std::string> images;
    int n_predict = 128;
    int n_threads = 4;
    // Threads for prefill, which is a different question from threads for decode: prefill is
    // compute-bound and scales with cores, while a streamed decode spends most of its time waiting
    // on flash and gains nothing from more of them. 0 means "same as n_threads", the historical
    // behaviour and what every caller that never heard of this gets.
    int n_threads_batch = 0;
    int n_ctx = 2048;

    // Largest batch computed in one graph, i.e. the prefill chunk size. 0 (the default) means
    // "follow n_ctx", which prefills any fitting prompt in a single pass.
    //
    // It is worth exposing because it does not only cost time, it costs RESIDENT MEMORY: the
    // scheduler reserves compute buffers for the worst-case graph, which is a full-width prefill,
    // and on this engine every MiB reserved is a MiB the expert cache and the dense weights do not
    // get. Measured at n_ctx 2048: 320 MiB of compute buffer, falling to 80 MiB at 512 — the
    // reservation scales with the context, which is exactly the coupling this knob breaks. It was
    // found while chasing a fault storm that turned out to be the reservation itself, not the
    // workload.
    //
    // Decode is unaffected: a decode graph is one token wide whatever this says. The cost is
    // prefill throughput, which processes a long prompt in more, smaller passes.
    int n_ubatch = 0;

    // The widest graph a run computes at once when n_ubatch is 0: as wide as the context up to here.
    // Past it the buffers stop following the context alone. The attention mask is a row of the
    // whole context per token of width, so at width = context the host buffer grows with its square.
    // Measured on a CPU host, Qwen3.6-35B-A3B Q4_K_M, flash attention on (#566): 2945 MiB at a 16384
    // context, 4801 MiB refused at 24576, and at 32768 an allocation inside llama.cpp failing before
    // the reservation could report it, which aborts the process. At this width the same 32768
    // context reserves about 1.7 GB, and a prompt longer than it takes more passes; under it nothing
    // changes, so the single pass that is the cheapest prefill for flash reads stays for every prompt
    // that fits in one. This is the width of a run nothing planned, and the width --auto's probe
    // measures at: a plan prices any other width from that measurement and may choose a wider graph
    // where the memory takes it (choose_graph_width). The same figure the planner bounds a device at
    // (PlannerPolicy::ubatch_max_device), for a different reason.
    static constexpr int ubatch_auto_max = 8192;
    // The narrowest graph an image allows: 0 without a projector or without a cap on an image's
    // tokens. Some projectors' image tokens attend to each other both ways, and llama.cpp then takes
    // the whole image in one pass of the graph, so an image wider than the graph cannot be decoded.
    // The cap (image_max_tokens) is the most one image takes; neither the default width nor the
    // planner goes under it. A width set by hand is the caller's, and the session says so when an
    // image does not fit it.
    int graph_width_floor() const {
        if (mmproj.empty() || image_max_tokens <= 0) return 0;
        return image_max_tokens < n_ctx ? image_max_tokens : n_ctx;
    }
    // The width this run's graphs are reserved for: n_ubatch when set, otherwise as wide as the
    // context up to ubatch_auto_max (or up to graph_width_floor where an image needs more). The one
    // place that turns 0 into a width: the session opens with it and the planner's probe measures
    // at it.
    int graph_width() const {
        int w = n_ubatch > 0 ? n_ubatch : ubatch_auto_max;
        if (n_ubatch <= 0 && w < graph_width_floor()) w = graph_width_floor();
        return w < n_ctx ? w : n_ctx;
    }

    // Context checkpoints, as llama.cpp's server keeps them (--ctx-checkpoints and
    // --checkpoint-min-step): copies of the part of the context's state that
    // cannot be cut at an arbitrary position, so a chat turn whose prompt diverges from the cached
    // one inside a recurrent state or a sliding window rolls back to the newest one before the
    // divergence instead of prefilling the whole conversation again. One is taken at the end of
    // every prompt, and inside a prompt prefilled in pieces every min_step tokens. A memory that can
    // be cut anywhere has no such state, and keeps none. Lossless: a restored state is the state the
    // same tokens computed.
    //
    // Fewer than the server's 32 by default: here every MiB they hold is a MiB the expert cache does
    // not get, and Qwen3.6-35B-A3B's state is 62.8 MiB. An agent's request diverges at the end of the
    // last prompt nearly always, which the newest checkpoint covers.
    int ctx_checkpoints = 8;
    int checkpoint_min_step = 8192;

    // The K and V caches' element type (llama.cpp's --cache-type-k / --cache-type-v). A quantized
    // cache holds a long context in a half or a quarter of the memory, which on this engine goes to
    // the expert cache, at a cost in output quality: see KvCacheType. A quantized V cache needs flash
    // attention, which llama.cpp turns on where the backend has it; where it has not, a quantized V
    // falls back to F16 rather than failing the run.
    KvCacheType cache_type_k = k_default_kv_cache_type;
    KvCacheType cache_type_v = k_default_kv_cache_type;
    // Wrap the prompt in chat turns, rendered with the chat template the gguf carries (llama.cpp's
    // common_chat_templates_init): the model's own template, not one picked by architecture. A file
    // whose template cannot be read falls back to the raw prompt, with a warning.
    bool chatml = false;

    // Render the chat template with reasoning enabled. Passed to the template as the
    // `enable_thinking` kwarg, so a reasoning model (Qwen3, thinking Gemma, …) only emits
    // its thinking channel when true. Off suppresses reasoning at the source rather than
    // relying on the display-time parser, which cannot strip a format it does not know.
    // Only meaningful with chatml; the raw-prompt path ignores it. Off by default: on device a
    // reasoning model can spend a whole turn's budget thinking before it answers (#331).
    bool think = false;

    // Override the number of active MoE experts per token (top-k routing). 0 = use the
    // model's own <arch>.expert_used_count from the gguf. A lower value cuts per-token
    // compute (and, with moe.enabled, flash I/O) proportionally, at a quality cost — it
    // changes the output. Applied at load via a llama.cpp kv_override on the arch-prefixed
    // expert_used_count metadata key; must stay in [1, n_expert]. Independent of streaming.
    int n_expert_used = 0;

    // ── what the first stage (llama.cpp's capacity fitter) decided ─────────────────
    // Layers stored on devices, counted from the top the way llama.cpp fills them, and the
    // per-tensor buffer-type override patterns the fitter wrote. 0 and empty — the defaults — mean
    // everything on the host, which is what the engine did before any planner existed. The
    // patterns are llama.cpp's own regexes over tensor names; the session maps each to the CPU
    // buffer type, since the streamer can only serve experts that live in host memory.
    int n_gpu_layers = 0;
    std::vector<std::string> buft_overrides;
    // How many of those layers each device takes, by name, in the fitter's device order (see
    // Placement::device_split). Empty leaves the split to llama.cpp, which divides by free memory.
    std::vector<std::pair<std::string, int>> device_split;
    // The devices llama.cpp may use, by name, where the planner chose among backends that drive the
    // same card (bmoe/backend_choice.h). Empty leaves the choice to llama.cpp, which keeps the first
    // backend registered for a card.
    std::vector<std::string> device_list;

    // Put the DENSE weights on an accelerator that shares this host's memory, leaving the routed
    // experts on the host for the streamer. On such a machine the move frees no memory - it is the
    // same pool - so it is worth making only where that device consumes this model's weights faster
    // than the CPU does, which the bandwidth probe measures.
    //
    // Off by default, armed by the caller, and currently REFUSED by the planner - because it was
    // measured three times in July on a device of exactly this class and it lost by 27%: the dense
    // and expert halves interleave, so a two-device split crosses the boundary twice per layer and
    // the boundary tax eats the CPU time the device frees. See the OpenCL row of
    // docs/backends.md. Kept as a named lever so the next person to have this idea finds the
    // verdict instead of the idea.
    bool dense_on_device = false;

    // Hand llama.cpp the CPU and nothing else. Set when this plan has decided no weight goes on a
    // device: leaving one registered is not free, because the scheduler gives it every node it can
    // execute - norms, softmax, the weightless ones - purely because it is there, and each of those
    // is a boundary the graph crosses twice. Measured with nothing placed on the device at all:
    // 61 graph splits per token, and a build with the backend compiled in losing to one without it.
    // A device we are not using should not be in the room.
    //
    // So the default decides by what was placed: with a layer on a device, llama.cpp gets every
    // device it finds; with none, the CPU alone. A build that merely carries a GPU backend then runs
    // like one that does not, to the bit. The prefill device is asked for by name and is outside
    // this choice: it joins the graph whatever this says.
    DeviceUse device_use = DeviceUse::Auto;

    SamplingConfig sampling; // greedy by default (temp <= 0); opt-in stochastic decoding
    MoeStreamConfig moe;
    SpecConfig spec;             // self-speculative decoding (MTP head or n-gram lookup); off by default
    PrefillDeviceConfig prefill; // prefill graphs on an accelerator; off by default
    DecideConfig decide;         // decide()'s prefix-state policy; off by default
    KnobTrialConfig trial;       // lossless knobs measured on the first real request; off by default

    // The configuration the session opens instead, where this one names a device whose answers the
    // load compares with the cores' (trial.answer_device) and the device does not keep the model:
    // its answers differed, could not be compared, the cores were faster, or it could not be opened.
    // Empty, that is this same configuration with no layer on a device (the cores hold the model as
    // it is held here). A plan sets it where the two differ in more than where they compute: a model
    // it would stream, tried whole on a device that reads the host's memory, falls back to the
    // streaming plan and not to the same model held whole on the cores. Not a tunable: one plan in
    // two parts, which no flag spells, so it is not in the parameter table.
    std::shared_ptr<const RunConfig> fallback;

    // Let llama.cpp's and ggml's DEBUG log lines through to stderr. Off, they are dropped: about
    // 1200 lines of graph bookkeeping per model load, which buried the engine's own output. For
    // debugging llama.cpp itself, not the engine. Process-wide, as llama.cpp's log is.
    bool llama_debug_log = false;

    // The upper ends of the ranges a settings form offers. They keep a slider finite and a typo
    // out; validate() does not enforce them, and none is a property of any model or machine.
    static constexpr int tokens_max = 1 << 20; // n_predict, n_ctx and n_ubatch alike
    static constexpr int threads_max = 256;    // n_threads and n_threads_batch
    static constexpr int n_expert_used_max = 256;
    static constexpr int n_gpu_layers_max = 1024;
    static constexpr int ctx_checkpoints_max = 256;

    // The context a client of a server needs, below which a front-end warns before the first
    // request fails rather than after. Measured: opencode's first request, its system prompt and
    // tool schemas alone, is 9911 tokens, so a coding agent is refused with context_full at any
    // context under ~10k before it has done anything. This is the next power of two above that,
    // which leaves room for the first answer and a few steps; a long agent session wants more.
    // Advice, not a limit: validate() accepts any context, and a short one is right for a plain chat
    // on a machine whose expert cache needs the RAM.
    static constexpr int ctx_for_clients = 16384;
};

// Validation result: ok plus a human-readable reason when not.
struct ValidationResult {
    bool ok = true;
    std::string error;
    explicit operator bool() const { return ok; }
};

// Check a RunConfig for internal consistency. Enforces, among others: MoE streaming
// requires a model path; cache_mb is 0 or >= cache_min_mb (unless force_cache);
// io_threads in range; n_predict/n_threads positive. Pure function — no I/O.
ValidationResult validate(const RunConfig & cfg);

// Whether a run lets its GPU backends take a wide graph's matmuls over host-resident weights (llama.cpp's
// op offload: a prompt's expert matmuls computed on the device the layers are on, the weights copied
// over for the pass). The one rule the session loads by and a front-end says where a prompt runs by,
// so the two cannot disagree. Pure. The facts it reads: the expert streamer's config, the layers placed,
// and whether a prefill device takes the prompt.
bool offloads_wide_matmuls(const MoeStreamConfig & moe, int n_gpu_layers, bool prefill_device);
inline bool offloads_wide_matmuls(const RunConfig & cfg) {
    return offloads_wide_matmuls(cfg.moe, cfg.n_gpu_layers, cfg.prefill.enabled());
}

} // namespace bmoe
