// Per-token and end-of-run metrics.
//
// The engine reports one TokenMetrics per generated token and a RunSummary at the end.
// How they are surfaced is a policy choice: the CLI turns TokenMetrics into machine
// telemetry lines (docs/telemetry.md) or an inline stream, and a sink can persist them
// as CSV for benchmarking. The engine itself makes no formatting decisions.
#pragma once

#include "bmoe/chat.h"
#include "bmoe/predict_stats.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bmoe {

// One candidate at a token's position: its id, its text (special tokens spelled out) and its
// natural log-probability.
struct TokenLogprob {
    int32_t id = 0;
    std::string piece;
    float logprob = 0.0f;
};

struct TokenMetrics {
    int step = 0;            // 1-based index of this token
    int steps = 0;           // n_predict target
    double wall_ms = 0.0;    // total wall time for this token
    double io_ms = 0.0;      // flash read time this token (serial: subset of wall; overlap: lane-busy sum)
    double mgmt_ms = 0.0;    // cache-management time (vm commit + evict + LRU bookkeeping) this token
    double compute_ms = 0.0; // residual: serial wall - io - mgmt; overlap wall - stall - mgmt
    double stall_ms = 0.0;   // overlap only: wall time the FFN kernel blocked on flash (0 when serial)
    // The flash term that is a share of wall_ms, already chosen for the run's mode: io_ms in serial
    // (the blocking read), stall_ms under overlap (where io_ms is summed lane time and can exceed the
    // wall). The term the compute residual subtracts, so a consumer never has to know the mode to
    // pick it, and never recovers it by inverting the clamped residual. 0 when streaming is off.
    double flash_wait_ms = 0.0;
    // Named eval-thread costs that otherwise hide inside a residual (all 0 when the feature that
    // pays them is off). drain_ms sits inside compute_ms: the async load waiting out the previous
    // layer's batch before reusing its flags. adopt_ms sits inside mgmt_ms: the route-ahead load
    // waiting for its own committed speculative reads. ra_issue_ms and ra_wd_ms sit inside
    // compute_ms: issuing the committed early reads, and the sampled fresh-gate watchdog. They
    // exist because every wrong theory about this feature was a cost deduced from a residual.
    double drain_ms = 0.0;
    double adopt_ms = 0.0;
    double ra_issue_ms = 0.0;
    double ra_wd_ms = 0.0;
    // Wall time between the PREVIOUS token's decode and this one's: sampling, detokenization,
    // rendering the answer for a UI, this struct, and the sinks. None of it is inside wall_ms, so
    // none of it reaches the reported tok/s — which is exactly why it is worth a column. A change
    // that moves only this number is invisible in s/tok while being paid on every token.
    // On the first token it is the gap from the end of prefill to the first decode.
    double loop_overhead_ms = 0.0;
    uint64_t read_bytes = 0;    // expert bytes pulled from flash this token
    double cache_hit_pct = 0.0; // cumulative cache hit rate (-1 if no cache)
    // Compute-decomposition counters, measured around llama_decode (0 if the platform can't report
    // them). They tell WHY a token's compute residual is large: major faults = dense weight re-read
    // from flash inside the decode; cpu_ms vs wall_ms×threads = how CPU-bound the decode really was
    // (low occupancy ⇒ throttled/preempted, not heavy math). See docs/telemetry.md.
    uint64_t majflt = 0; // major page faults during this decode (backing-store reads)
    double cpu_ms = 0.0; // CPU time summed across all threads during this decode
    // Fraction of the DENSE (non-expert) weights the kernel still had in RAM at the last sample, or -1
    // when unmeasured (throttled, streaming off, or the platform can't report). Under the anon policy
    // this samples our own buffers (is zram holding them?); under mmap/warm the mmap ranges (is the
    // kernel dropping the model?). A diagnostic, read alongside `majflt` and the rss split — nothing
    // acts on it. See docs/pressure.md.
    double dense_resident_frac = -1.0;
    // What those faults actually moved, in MiB (majflt × page size). The same fact as `majflt`, in
    // the unit the rest of this struct is in: 47447 faults is unreadable, 194 MiB re-faulted in one
    // token is immediately comparable to `read_bytes` — the reads we chose against the reads the
    // kernel forced on us. 0 when faults are unmeasured.
    double majflt_mib = 0.0;

    // ── where memory is, per token (0 when the platform cannot report) ──
    // The split is the point: the expert cache is anonymous, the model's weights are file-backed,
    // and they are reclaimed differently — anon is compressed into zram, file pages are just
    // dropped. rss_anon_mib falling while the budget stays put IS the kernel taking the cache.
    double rss_mib = 0.0;
    double rss_anon_mib = 0.0;
    double rss_file_mib = 0.0;
    double swap_mib = 0.0;
    // What the device claims about itself, recorded next to what we measured ourselves — the gap
    // between mem_available_mib and our own residency is the reason this engine trusts neither.
    double mem_available_mib = 0.0;
    double mem_free_mib = 0.0;
    double swap_free_mib = 0.0;

    // Kernel accounting for this process, distinct from resident bytes and system availability.
    // -1 means unavailable; zero is a measured value. Sampled after decode, not a peak.
    double process_footprint_mib = -1.0;
    double process_headroom_mib = -1.0;

    // The cache budget in effect for this token. Fixed for the run now (an explicit --cache-mb, or
    // what --cache-mb auto sized to once at load) — the runtime governor that moved it is gone.
    double cache_budget_mib = 0.0;
    int turn = 0; // session turn this token belongs to (0 for a one-shot run)

    // How many tokens the decode that produced this one confirmed (1 without speculation). Under
    // --mtp a verify decode confirms a whole group, and the group's entire cost — wall, faults,
    // CPU, flash bytes — is charged to its FIRST row; the rest carry zeros. Without this field those
    // zeros read as free tokens. Divide the first row's wall_ms by mtp_batch for the per-token cost.
    int mtp_batch = 1;

    // Time this group spent drafting and catching the draft context up — everything speculation
    // adds OUTSIDE the target decode. A SLICE of loop_overhead_ms, not an addition: both measure
    // the gap between decodes. Charged to the group's first row like every other group cost, and 0
    // without --mtp. Without it the drafting cost can only be inferred by differencing two runs.
    double mtp_draft_ms = 0.0;

    std::string piece; // text of just this token (delta, for inline streaming)
    std::string text;  // full generated answer so far, reasoning stripped (for UI streaming)
    // The reasoning span so far, when the model is thinking and the chat parser separated it from
    // the answer. Empty with chat off, on a non-reasoning model, or on the harmony no-think path.
    // Display-only, and kept apart from `text` on purpose: the UI shows it as a distinct thinking
    // block rather than letting it leak into the answer. See docs/telemetry.md.
    std::string reasoning;
    // The calls the answer holds so far, as the same per-token parse reads them: the model's own
    // template recognises them, so no format, model or tool is named here. A call's name is known
    // once its arguments begin; its id may stay empty until the turn ends; the last call's
    // arguments are the JSON written so far, and grow by appending (the parser's contract, which a
    // client that concatenates fragments relies on). Display-only: what the turn decided is
    // RunResult::tool_calls. Empty with chat off or when the answer has no call.
    std::vector<ChatToolCall> tool_calls;
    // Only when GenerateRequest::n_probs asks: this token's log-probability and the most likely
    // tokens at its position, most likely first, from the model's own distribution before any
    // sampling stage (what llama.cpp's server and the OpenAI protocol report as logprobs).
    bool has_logprobs = false;
    float logprob = 0.0f;
    std::vector<TokenLogprob> top_logprobs;
};

struct RunSummary {
    // The loaded model's architecture, as gguf reports it. Carried here so a caller can tell what
    // the run was capable of and not only what it did: a MoE arch that ran without streaming is the
    // baseline, not a measurement of this engine (see the CLI's mode line).
    std::string arch;
    // The file's routed expert count, 0 for a dense model: with arch, what the run could have streamed.
    int n_expert = 0;

    int n_generated = 0;
    // The answer's budget this turn: the request's n_predict, or less when the context had less
    // room left after the prompt. A run that generated this many tokens stopped on its budget.
    int n_predict = 0;
    double gen_seconds = 0.0;
    double s_per_token = 0.0;
    double tokens_per_second = 0.0;

    // Startup phase (additive telemetry): model load + streaming setup, and prefill.
    // TTFT ~= load_seconds + prefill_seconds; prefill tok/s = n_prompt / prefill_seconds.
    // In a multi-turn chat n_prompt is the tokens actually prefilled THIS turn (the suffix
    // after the reused KV prefix), and n_past is the total context length after the turn.
    int n_prompt = 0;
    int n_past = 0;
    double load_seconds = 0.0;
    double prefill_seconds = 0.0;

    // Prefill-phase attribution (#173): the same wall-additive quantities the decode phase
    // reports, as deltas of the streamer's cumulative counters across THIS turn's prefill
    // chunks (plus process CPU). Before #169-style attribution existed prefill was a single
    // bare wall number, so the phase a >RAM prompt actually waits on had no compute/flash
    // split at all. Zero when streaming is off (prefill_cpu_seconds still reported: CPU is
    // measured regardless). Same reading rules as the decode fields: io is the summed lane
    // busy time under overlap, stall is the union of stalled intervals, and cpu covers the
    // whole process — an upper bound on compute-thread CPU-equivalent time.
    double prefill_cpu_seconds = 0.0;
    double prefill_read_mib = 0.0;
    double prefill_io_seconds = 0.0;
    double prefill_stall_seconds = 0.0;
    double prefill_mgmt_seconds = 0.0;
    // Of n_prompt, the tokens whose prefill graph ran on the prefill device (RunConfig::prefill).
    // Zero when that is off, and for a prompt too short to reach it. What proves the device ran.
    int prefill_device_tokens = 0;
    // Graph nodes a device backend computed during this turn's prefill: the proof the device RAN,
    // where prefill_device_tokens only says the weights were moved there. Zero when that is off.
    long long prefill_device_nodes = 0;
    // With streaming, the experts the device prefill read through its arena, and the wall time the
    // graph waited at a layer for them: the flash half of a device prefill, which the streamer's
    // own prefill_read_mib/prefill_stall_seconds above do not see. Zero without an arena.
    double prefill_device_read_mib = 0.0;
    double prefill_device_stall_seconds = 0.0;
    // What the arena copied from the expert cache instead of reading it (PrefillDeviceConfig::
    // from_cache): the experts a device prefill did NOT read from flash. Zero without an arena.
    double prefill_device_cache_mib = 0.0;

    // MoE streaming totals (zero when streaming is off)
    double moe_read_mib = 0.0;
    double moe_io_seconds = 0.0;
    double moe_compute_s_per_token = 0.0;
    double moe_io_s_per_token = 0.0;
    double moe_mgmt_s_per_token = 0.0;  // cache-management time per token (commit + evict + LRU)
    double moe_stall_s_per_token = 0.0; // overlap only: per-token wall the kernel waited on flash
    // Per token, the wall-additive flash term (TokenMetrics::flash_wait_ms): io in serial, stall
    // under overlap.
    double moe_flash_wait_s_per_token = 0.0;
    double cache_hit_pct = -1.0; // -1 when no cache
    // The same, over this generation's decode alone: the figure a plan's credited hit rate prices
    // (a decoded token), where cache_hit_pct also counts the prompt and every earlier turn. -1 when
    // no cache or nothing was decoded.
    double cache_hit_decode_pct = -1.0;

    // Compute decomposition, generation phase only (measured whether or not streaming is on, since
    // dense-weight faults appear in the mmap baseline too). majflt_per_token ≫ 0 flags a residency
    // stall hiding in "compute"; cpu_util = cpu_s/token ÷ (s/token × threads) near 1 is compute-bound,
    // well below 1 is a throttled/preempted core. cpu_s 0 and majflt -1 when the platform can't
    // measure them: a fault count of 0 is a measurement, and one never taken must not read as one.
    double majflt_per_token = 0.0;
    double cpu_s_per_token = 0.0;
    // Everything between the decodes, per token: the region gen_seconds (and so tok/s) excludes.
    // Includes the tail after the last token, which no row can carry. Read next to s_per_token: the
    // two together are what a caller actually waits through.
    double loop_overhead_s_per_token = 0.0;
    double cache_resident_mib = 0.0;
    double cache_budget_mib = 0.0; // the fixed cache budget the run used (explicit, or auto-sized at load)
    long long cache_resizes =
        0; // runtime budget changes: an app's explicit set_cache_budget, or a pin refused part way
    // Cache churn (see IExpertSource::Stats): entries the budget forced out, and reads that went
    // to an entry the cache had held before. Any prefetch that raises the byte count while
    // claiming its reads are useful is doing it here.
    long long cache_evictions = 0;
    long long cache_rereads = 0;
    long long cache_cold_admits = 0;      // misses a prompt admitted cold (MoeStreamConfig::cache_prefill_cold)
    long long cache_discarded = 0;        // cached experts the system dropped under memory pressure
    long long cache_discarded_on_hit = 0; // of those, the ones a lookup found dropped and read again

    // Row-gathered dense tables served from flash (MoeStreamConfig::row_stream). The pair that says
    // whether the policy paid: row_table_mib is what those tables would have occupied resident,
    // row_resident_mib what they occupy now, and row_read_mib what buying that cost in flash. Zero
    // throughout when the policy is off or no table qualified.
    double row_table_mib = 0.0;
    double row_resident_mib = 0.0;
    double row_read_mib = 0.0;
    long long row_rows = 0;       // row indices gathered, duplicates included
    long long row_slab_reads = 0; // gather misses that went to flash
    long long row_evictions = 0;
    long long row_io_errors = 0; // non-zero means a decode read bytes that were never fetched
    // The named eval-thread waits (see TokenMetrics): the async load's previous-batch drain (part
    // of the compute residual) and the route-ahead adoption wait (part of mgmt), per token.
    double moe_drain_s_per_token = 0.0;
    double moe_adopt_s_per_token = 0.0;

    // What one token actually demands of the cache, measured: the distinct expert bytes routed per
    // token. A cache below this can hold nothing between tokens; a cache far above it is buying
    // hits from inter-token routing correlation only. Reading it against cache_budget_mib is how a
    // budget stops being a guess. 0 when streaming is off or nothing was routed.
    double token_demand_mib = 0.0;
    // The widest layer's routed bytes: the mechanical floor a cache must be able to stage.
    double layer_demand_mib = 0.0;
    // Both measure what reached the streamer. Under MoeStreamConfig::drop_cold_frac that is what a
    // token STAGES, not what it routed — a dropped expert is never handed over — so the "floor a
    // cache must clear" reading stops being mechanical there: the floor shrinks because the cache
    // was small. Size the cache with dropping off, then turn it on.

    // Cache-aware expert dropping (zero when --drop-cold-experts is off). Routed counts what the
    // router selected across the generation, dropped how much of it the policy declined to read;
    // their ratio is the lever's actual bite, which depends on the cache and so cannot be read off
    // the flag. Both cover generation only — prefill drops nothing unless armed for it.
    long long experts_routed = 0;
    long long experts_dropped = 0;
    // Cache-aware substitution (both zero when --expert-substitute is off): reranked is every slot
    // the policy examined — its own denominator, since experts_routed above is the drop policy's —
    // and substituted the slots whose expert it changed for a resident one. Same caveat as dropped:
    // the flag sets a margin, not a rate, and how often a resident stand-in falls inside it depends
    // on the cache.
    long long experts_reranked = 0;
    long long experts_substituted = 0;

    // Temporal prefetch (zero when --prefetch is off): speculative bytes read during generation,
    // experts successfully prefetched, and how many of those a later routing actually used.
    double moe_spec_read_mib = 0.0;
    long long moe_spec_experts = 0;
    long long moe_spec_useful = 0;

    // Self-speculation (zero when SpecConfig::source is none). These are the LOOP's counters, not a
    // source's: whichever drafted, `mtp_drafted` counts tokens proposed and `mtp_accepted` how many
    // the target's argmax confirmed. Their ratio is the acceptance on this prompt, the one number
    // that decides whether the feature can pay. Tokens per verify decode (n_generated / mtp_decodes)
    // is the amortisation actually achieved — it is what tok/s is bought with, and it is always
    // below 1 + draft_max. The keys keep the mtp_ prefix they were published under; renaming them
    // would break every CSV and dashboard already holding measurements.
    long long mtp_drafted = 0;
    long long mtp_accepted = 0;
    long long mtp_decodes = 0; // verify decodes issued; equals n_generated when speculation is off
    // Steps that drafted anything. Against mtp_decodes this is the n-gram source's match rate: the
    // fraction of the run where it had evidence, widened the verify batch and could win — the rest
    // ran as plain decodes at exactly the unspeculated cost. It is the number that says whether a
    // result is about the source's precision or about how rarely it fired. For the head it is every
    // step, unless draft_p_min stopped it.
    long long drafted_steps = 0;
    // Drafting + catch-up seconds per generated token: the price of speculation, measured rather
    // than inferred. tok/s is computed from decode time alone, so this is time the caller waits
    // that the headline rate does not show — compare it against s_per_token before believing a
    // speculated run is faster.
    double mtp_draft_s_per_token = 0.0;
    // Flash MiB the drafting passes streamed, as a subset of moe_read_mib. The MTP block is a MoE
    // layer of its own, so the head has an I/O cost and not just a compute one. Subtracting this
    // from the run's total is what splits the growth in bytes/token under speculation into its two
    // causes — the widened verify union on the trunk, and the head's own routing — which need
    // completely different fixes. The route trace cannot see it: its framing brackets the target
    // decode, and the head only ever runs in the draft context.
    double mtp_draft_read_mib = 0.0;

    // Expert-prediction accuracy (all zero unless MoeStreamConfig::predict_log). `predict_stale` is
    // the next layer's routing ranked a layer early, `predict_prev` the previous token's routing —
    // the bet --prefetch already makes — and `predict_self` a zero-staleness control that bounds
    // what the ranking could show at best on this architecture. The *_by_layer vectors carry the
    // same statistic per layer, which is where the interesting shape is: the first layers of a
    // model route far less predictably than the rest, so an aggregate alone hides the case a
    // prefetch would most need to get right.
    //
    // Unlike the drop counters these are NOT per-generation deltas: they accumulate over the whole
    // session, because an accuracy estimate wants every sample it can get.
    // predict_stale2 is the same predictor two layers early — the staleness the ASYNC prefetch
    // actually runs at — reported in aggregate only.
    PredictorStats predict_stale, predict_stale2, predict_prev, predict_self;
    std::vector<PredictorStats> predict_stale_by_layer, predict_prev_by_layer, predict_self_by_layer;
    long long predict_unscored = 0; // routings the stale-gate probe could not rank (see RouterHook)

    // Route-ahead (all zero unless MoeStreamConfig::route_ahead > 0): how many decode routings had
    // their selection replaced by the N-layers-early prediction, how many eligible ones had no
    // prediction to commit and kept the router's choice, and — over the overridden ones — how many
    // slots the committed selection shared with what the router would have picked. hits/slots is
    // the routing perturbation the run actually generated under; like the probe's counters these
    // are session totals, not per-generation deltas.
    long long route_ahead_overridden = 0;
    long long route_ahead_passthrough = 0;
    long long route_ahead_slots = 0;
    long long route_ahead_hits = 0;
    // The prediction's own CPU bill: worker nanoseconds spent on the gate GEMV, and how many
    // rankings it produced. Reported per token because it is the one cost that hides from wall
    // time wherever a spare core exists — and stops hiding on a phone, where the same core and
    // the same DRAM serve the decode this is meant to accelerate.
    long long route_ahead_gemv_ns = 0;
    long long route_ahead_gemv_jobs = 0;
    long long route_ahead_issue_ns = 0; // eval-thread cost of issuing the early reads
    long long route_ahead_wd_ns = 0;    // eval-thread cost of the sampled fresh-gate watchdog
};

// What this run IS: the model and the configuration every row below it was produced under.
//
// Rows without it are not evidence. Two CSVs put side by side answer "which is faster" only if
// something says what differed between them — and by the time a file is read, the argv that made
// it is long gone. The engine states it once, in the file, next to the numbers it explains.
struct RunInfo {
    std::string engine_version; // the build that produced the rows; see bmoe/version.h
    std::string model;          // file name, not the full path: the path is the reader's machine, not the run's
    std::string arch;
    int n_layer = 0;
    int n_expert = 0;
    int n_expert_used = 0; // effective top-k, after any override
    int n_threads = 0;
    int n_ctx = 0;
    int n_ubatch = 0; // widest graph computed at once, as the context opened it. Sets the compute-buffer
                      // reservation, so it moves the very memory columns these rows record.
    std::string cache_type_k = "f16"; // the K and V caches' element types: a quantized one is lossy
    std::string cache_type_v = "f16";
    bool chatml = false; // a chat-templated prompt is not the prompt that was typed
    // Deliberately absent: `think`. It is a property of a REQUEST, not of the session, so a session
    // preamble stating one value would be wrong for every turn that asked for the other. The `turn`
    // column is where per-turn facts belong.

    // The streaming configuration, as resolved (not as typed): cache_mb is what the engine settled
    // on, which under auto-sizing is a number no flag mentioned.
    bool moe_stream = false;
    int cache_mb = 0;
    bool cache_auto = false;
    int cache_floor_mb = 0; // the RAM auto-sizing was told to leave free — the input behind cache_mb
    int cache_ceil_mb = 0;
    // One token's WORST-CASE routed bytes, priced at load from the model's shape and the effective
    // top-k. A cache_mb under this cannot hold a token cycle, so the run's hit rate is near zero by
    // construction — and that is invisible in a committed CSV unless the cycle is recorded next to
    // the budget, because the cliff is a property of the MODEL, not of the number that was typed.
    // 0 when streaming is off.
    int cache_cycle_mb = 0;
    bool force_cache = false;
    bool load_all = false; // whole-expert-set baseline: reads everything, so its bytes mean something else
    int io_threads = 0;
    bool o_direct = false;
    bool overlap = false;
    bool io_two_wave = false; // two-wave batch publish (#118): first projection published early
    int prefetch_layers = 0;
    int route_ahead = 0;                // decode routing committed to the N-layers-early prediction (LOSSY)
    bool predict_prefetch = false;      // stale-gate predictive prefetch (see MoeStreamConfig)
    bool predict_log = false;           // the accuracy probe: costs a barrier and two GEMVs per layer
    int predict_spec_max = 0;           // how many predicted misses a layer may speculate (0 = retention only)
    bool prefetch_sync = false;         // test path: speculative reads served inline, no latency hiding
    std::string dense_weights = "anon"; // dense (non-expert) policy: "mmap" | "warm" | "anon" | "pinned"
    float drop_cold_frac = 0.0f;        // cache-aware expert dropping threshold (0 = off)
    bool drop_renorm = true;            // survivors rescaled to keep the routing's total mass
    bool drop_prefill = false;          // dropping armed during prefill too, where it discards far more
    float substitute_lambda = 0.0f;     // cache-aware substitution margin, a fraction of the score range (0 = off)

    // Sampling. Greedy (temp <= 0) is the default and the only deterministic one; the byte-identity
    // gates depend on it. A stochastic run and a greedy one are not comparable, and until these were
    // recorded the file could not tell them apart.
    float temp = 0.0f;
    int top_k = 40;
    float top_p = 0.95f;
    uint32_t seed = 0xFFFFFFFFu;

    // Self-speculation, and which source drafted ("off", "mtp", "ngram"). It changes how many tokens
    // a decode confirms, so every per-token row in the file was produced under a different
    // accounting than an unspeculated run — see TokenMetrics::mtp_batch. Recorded so the two are
    // never averaged together by accident, and so two speculated files are not compared across
    // sources without noticing.
    std::string spec = "off";
    int spec_draft_max = 0;
    float mtp_p_min = 0.0f;  // mtp: the head's confidence floor for drafting (0 = no floor)
    int ngram_min_match = 0; // ngram: shortest suffix match allowed to draft

    // Diagnostics that perturb what they measure. A traced run is not a benchmark run — recorded so
    // a file cannot be mistaken for one.
    int compute_trace_layers = 0;
};

// Optional per-token sink (e.g. CSV for benchmarks). The engine calls on_run_info once before the
// first token, then on_token for each token and on_summary at the end of every generation.
class IMetricsSink {
public:
    virtual ~IMetricsSink() = default;
    // Default no-op: a sink that only wants numbers is not obliged to care what produced them.
    virtual void on_run_info(const RunInfo &) {}
    virtual void on_token(const TokenMetrics &) = 0;
    virtual void on_summary(const RunSummary &) = 0;
};

// A sink that appends one CSV row per token to `path` (header written on open).
// Returns nullptr if the file cannot be opened.
std::unique_ptr<IMetricsSink> make_csv_metrics_sink(const std::string & path);

} // namespace bmoe
