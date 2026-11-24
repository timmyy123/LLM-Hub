// Flash-backed expert streaming with an optional LRU cache.
//
// Ported from the original research streamer. One layer computes at a time, so with the
// cache off the expert tensors share up to three heap slots (full n_expert size) that
// every layer's tensors are rebound onto: only the routed slices are ever filled,
// re-read fresh each token. With the cache on, each (layer, projection) gets its own
// reserved-but-uncommitted address range; a routed expert already resident is a hit (no
// read), a miss is read once and kept, and the coldest entries are evicted (pages
// physically released) to hold the budget. Reads are drained across an I/O lane pool.
//
// Correctness rests on the eval-callback single-node barrier: the routing node is
// computed and synchronized before load_layer() runs, and the layer's expert matmul
// runs only after it returns — so the routed slices are valid exactly when read, and the
// next layer cannot overwrite them until this layer's matmul has been synchronized.
#pragma once

#include "bmoe/expert_source.h"
#include "bmoe/config.h"
#include "bmoe/decode_trace.h"
#include "stall_union.h"
#include "bmoe/expert_layout.h"
#include "../io/platform_io.h"
#include "../io/file_reader.h"
#include "dense_weights.h" // DenseWeights + DenseTensorRef
#include "discardable_entries.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct ggml_tensor;

namespace bmoe {

// One expert weight tensor to rebind, with where its data lives in the gguf.
struct ExpertTensorRef {
    ggml_tensor * tensor = nullptr; // persistent weight tensor whose ->data we rebind
    uint64_t file_off = 0;          // byte offset of the tensor's data within its shard
    uint64_t nb2 = 0;               // bytes per expert (== tensor->nb[2])
    int file_idx = 0;               // which shard file holds the bytes (0 for a single-file model)
};

// The expert weight tensors of one MoE layer, one per layout slot (bmoe/expert_layout.h). The
// split layout fills all three; a fused or gate-less layout fills two and leaves the tail slot
// empty (tensor == nullptr). Unbound layers (dense, or non-MoE) stay false.
struct LayerExperts {
    bool bound = false;
    // Some of this layer's experts are on a device. Beside a bound layer that is a split the
    // streamer cannot serve (the projections it would load with this half are elsewhere), and the
    // session refuses it by name rather than tripping over the missing tensor.
    bool on_device = false;
    ExpertTensorRef proj[ExpertLayout::max_slots];
};

class ExpertStreamSource final : public IExpertSource {
public:
    ExpertStreamSource() = default;
    ~ExpertStreamSource() override;

    ExpertStreamSource(const ExpertStreamSource &) = delete;
    ExpertStreamSource & operator=(const ExpertStreamSource &) = delete;

    // Build buffers, rebind the bound layers' expert tensors onto them, and start the
    // I/O pool. `layers` is indexed by layer id (unbound entries are skipped); each tensor's
    // file_idx indexes `shard_paths` (a single-file model passes one path). Returns false on
    // any allocation/open failure or an inconsistent tensor.
    bool init(const std::vector<std::string> & shard_paths,
              int n_expert,
              std::vector<LayerExperts> layers,
              const MoeStreamConfig & cfg);

    // Supply the dense (non-expert) weight tensors the DenseWeights policy may need (only the
    // Anonymous mode reads+rebinds them). Call BEFORE init, which hands them to the dense module.
    // The runtime builds the list from the captured weight leaves and the gguf offsets.
    void set_dense_tensors(std::vector<DenseTensorRef> dense) { dense_tensors_ = std::move(dense); }

    // Supply the subset of those weights the graph only ROW-GATHERS, which the dense policy serves
    // from flash inside a `budget_bytes` window instead of making resident (see bmoe/row_source.h).
    // Call BEFORE init, like set_dense_tensors; an empty list turns the policy off.
    void set_row_tensors(std::vector<DenseTensorRef> rows, uint64_t budget_bytes) {
        row_tensors_ = std::move(rows);
        row_budget_ = budget_bytes;
    }

    // The row policy after init, for the graph adapter that must make rows present before a gather
    // node runs; null when nothing qualified. Its accounting, for telemetry, is row_stats().
    IRowSource * row_source() const { return dense_.row_source(); }
    bool dense_file_mapping_in_use() const { return dense_.file_mapping_in_use(); }
    // Reopen every reader that will serve decode (expert lanes and row tables). Only legal between
    // init and the first decode, when the workers are idle and nothing is queued; checked under the
    // queue lock. See FileReader::reopen.
    bool reopen_readers();
    RowSourceStats row_stats() const { return dense_.row_stats(); }

    // IExpertSource
    bool load_layer(int il, const int32_t * ids, int n_ids) override;
    void prefetch(int il, const int32_t * ids, int n_ids) override;
    void retain(int il, const int32_t * ids, int n_ids) override;
    void settle_spec() override;
    void query_residency(int il, const int32_t * ids, int n_ids, uint8_t * out) const override;
    uint64_t expert_bytes(int il) const override;
    Stats stats() const override;

    // Register the process-global expert-ready hook so the CPU matmul blocks per expert
    // until its slice is resident. Only meaningful in overlap mode. The hook is the fork's and
    // lives in the CPU backend, reached through the backend registry at run time (the CPU
    // backend may be a module loaded at start-up); false when the backend registered here does
    // not expose it. Paired with shutdown(), which unregisters it.
    bool enable_overlap_hook();

    // Switch between serial and overlapped reads with the session open (the knob trial's overlap
    // arm, docs/measured-planning.md). Between two decodes only, from the eval thread. Turning it
    // on registers the hook if it is not, and fails, changing nothing, when the CPU backend has
    // none. Output is identical either way: only who waits for a read, and where, changes.
    bool set_overlap(bool on);
    bool overlap() const { return overlap_.load(std::memory_order_relaxed); }

    // True once an async read has failed or the source is shutting down. Wired to
    // llama_set_abort_callback so a mid-decode I/O failure aborts the graph cleanly
    // instead of computing on a half-read expert.
    bool fatal() const { return fatal_.load(std::memory_order_acquire); }
    void mark_fatal() override { fatal_.store(true, std::memory_order_release); }

    // Explicitly set the cache budget in bytes and evict down to it immediately (clamped to the
    // full expert-set size). PRECONDITION: no decode in flight — the caller must not be inside a
    // load_layer/generate. Intended for an app's memory-pressure callback (Android onTrimMemory)
    // and exercised by the shrink gate. The owner's budget: the brake and a device prefill's loan
    // move the live budget under it (below) without replacing it.
    void set_cache_budget(size_t bytes);
    size_t cache_budget() const { return cache_max_; }
    // The cache brake's resize. An ordinary cache: the new budget, as set_cache_budget. A
    // discardable one: the budget comes down and the budget its owner set stays the ceiling, so it
    // climbs back once the system settles (DropFollowingBudget::braked). Same precondition.
    void brake_cache_budget(size_t bytes);
    // A device prefill's slots borrow `bytes` from the cache for one prompt and give them back
    // (PrefillPath::place). The loan never becomes the cache's budget: what the owner set, and with
    // a discardable cache where the kernel let it climb, is what return_cache restores. Same
    // precondition.
    void lend_cache(size_t bytes);
    void return_cache();
    // The run's effective top-k: a batch of ids is that many per token, which is how load_layer tells
    // a wide graph (cache_prefill_cold) from a decode or a verify window. Before the first decode.
    void set_top_k(int k) { top_k_ = std::max(0, k); }

    // Where projection p of expert e of layer il sits in the cache, or null when it is not resident
    // (or there is no cache). `bytes` is the slice's size. For readers on other threads while a
    // device graph runs: the eval thread is inside that graph, which reads its experts from the
    // arena and never stages or evicts here, and every entry valid at that point was fully read by
    // the graph before it. Nothing may call it while a host graph could be evicting. An entry
    // offered to the kernel is not resident here: its bytes may be gone, and taking it back is the
    // eval thread's.
    const void * resident_slice(int il, int e, int p, uint64_t & bytes) const;

    // The share of the cache's resident pages the system has taken back, from one page of each of up
    // to `n` entries spread over the LRU. What the cache wrote and the kernel then compressed, swapped
    // or trimmed is gone from RAM by definition, and every platform answers that about a process's own
    // pages (pio::vm_resident_sample). -1 when nothing could be sampled. Same precondition as
    // set_cache_budget: no decode in flight.
    double cache_lost_frac(int n);
    // The budget a brake may not go under: one token's worst-case cycle (set_pin_floor).
    size_t cache_floor() const { return pin_floor_; }

    // The cliff from cache-sizing.md, computed at init from model shape alone: the bytes one
    // token's pass over the layer stack demands in the worst case, i.e. every bound layer's top_k
    // most expensive experts. A global-LRU budget below this thrashes to exactly 0% hits — it
    // reads as much as no cache while still paying management and RAM — so the runtime warns
    // when it sees one. Worst case, not the measured demand: the router picks at runtime, and
    // the measured token_demand_ needs a decode before it exists.
    size_t worst_cycle_bytes(int top_k) const;

    // The floor a pin refusal may not shrink the cache under: one token's worst-case cycle, which
    // only the caller knows because the routing width is its to resolve. Unset, a part-way refusal
    // ends pinning and leaves the budget alone.
    void set_pin_floor(size_t bytes) {
        pin_floor_ = bytes;
        follow_.set_floor(bytes);
    }

    // ── I/O trace (diagnostics; see bmoe/decode_trace.h) ────────────────────────────
    // When on, every read_slice records one row. Rows are appended under a dedicated leaf mutex
    // (reads happen on N lanes at once), so this costs a lock per read and is off by default.
    // take_io_trace_rows moves the buffer out; the caller stamps the frame it belongs to.
    void set_io_trace(bool on);
    void take_io_trace_rows(std::vector<IoTraceRow> & out);

    void shutdown();

private:
    struct IoJob {
        void * dst = nullptr;
        uint64_t off = 0;
        uint64_t nbytes = 0;
        int32_t flag = -1; // overlap: index into ready_ to publish on completion; -1 = serial
        // Which shard reader serves this read. Wide enough for the whole filename format
        // (-%05d-of-%05d): an int8_t would wrap a 200-shard model into a negative index that the
        // init-time bounds check, which sees the untruncated value, could never catch.
        int16_t file = 0;
        // Which (layer, expert, projection) this read serves. Known at every enqueue site and
        // otherwise thrown away; carried so the I/O trace can attribute a read without the read
        // path having to guess. Inert unless the trace is on.
        int32_t expert = -1;
        int16_t layer = -1;
        int8_t proj = -1;
        uint8_t spec = 0; // 1 if enqueued speculatively by prefetch
    };

    // One readiness cell per (projection, expert). A cell is "ready for the layer in flight"
    // exactly when gen == async_gen_; padded to a cache line so the compute threads polling it
    // do not false-share the LRU/job bookkeeping the load thread mutates alongside.
    struct alignas(pio::false_sharing_span) ReadyFlag {
        std::atomic<uint32_t> gen{0};
    };

    // `j` carries the read AND (for the trace) what it serves; `lane` is who is doing it.
    bool read_slice(int lane, const IoJob & j);
    void io_drain(int lane, uint64_t my_gen);
    void io_worker(int lane);

    // Speculative prefetch (temporal): drain queued spec reads on an idle lane, and integrate /
    // discard them on the eval thread at the next real load. See docs/prefetch.md.
    void drain_spec(int lane, uint64_t worker_seen);
    // adopt_il >= 0 (route-ahead only): the layer being staged COMMITTED to its speculated ids, so
    // its queued spec reads are the demand reads — finish them instead of cancelling them, and
    // leave every other layer's committed reads queued rather than destroying them.
    void quiesce_spec(int adopt_il = -1);
    void spec_integrate_done();       // integrate completed spec entries; cancel nothing (route-ahead)
    void drain_adopted(int adopt_il); // serial: lane 0 reads the loading layer's adopted jobs inline
    void release_entry_pages(int32_t id);

    // Accumulate one token's routed working set. Eval-thread only, called per layer load.
    void account_demand(int il, int n_unique);

    // Diagnostic: throttled dense-residency sample (dense_.sample_residency), cost timed into
    // mgmt_ns_. Pure telemetry — it feeds nothing, the governor that once consumed it is gone.
    void maybe_sample_dense();

    bool load_layer_async(int il, const int32_t * ids, int n_ids); // overlap path
    void build_ready_map();                                        // overlap's readiness cells and tensor map

    static void c_expert_ready(const ggml_tensor * src0, int expert, void * user_data);
    void on_expert_ready(const ggml_tensor * src0, int expert);

    // One expert's cache accounting for the layer being staged: the lookup, the hit/miss decision,
    // and the residency bookkeeping that follows. Both load paths need exactly this and used to spell
    // it out separately — what a hit means, what a miss commits, how the LRU order is kept — so a fix
    // to any of it had to be made twice or not at all. What they do NOT share is how they schedule the
    // reads (serial emits expert-major and blocks; overlap emits projection-major and publishes), so
    // that stays with each caller. `hit` tells the caller whether this expert still needs reads.
    // Returns false only if committing the pages failed; the caller decides how fatal that is.
    // LRU mode only (cache_max_ > 0) — the shared-slot path has no entries to account for.
    // `promote` = false skips the hit-path LRU move for callers that re-order every touched id
    // afterwards anyway (the overlap path's token-major promote loop); a miss is always linked.
    // `commit_only_proj` >= 0 commits only that projection's pages on a miss (two-wave publish:
    // the caller contracts to commit the rest via commit_proj_pages before emitting their jobs).
    bool touch_entry(int il, int e, bool & hit, bool promote = true, int commit_only_proj = -1);

    // Commit the pages of one (layer, expert, projection) cache slice so a read can land in it.
    bool commit_proj_pages(int il, int e, int p);
    bool commit_slice(int il, int e, int p);
    void pin_pages(void * p, size_t sz);

    // LRU helpers (active only when cache_max_ > 0)
    void lru_unlink(int32_t id);
    void lru_push_front(int32_t id);
    void lru_push_back(int32_t id);
    void admit_entry(int32_t id); // a miss enters the LRU (cold end in a cold batch) and the offer window
    size_t entry_bytes(int il) const;
    void evict_tail();
    void evict_entry(int32_t id);
    // Evict to budget under the configured policy. `spare_current` keeps every entry the layer in
    // flight staged (cstamp_ == cgen_): its reads may still be landing, and its matmul needs it.
    void evict_to_budget(bool spare_current);

    // Decayed-frequency eviction (MoeStreamConfig::cache_freq_half_life > 0). A use at pass t adds
    // growth^t to the entry's score instead of decaying every score each pass: the order is the
    // same as a decayed count, and a pass costs one multiply instead of a sweep. A pass is one
    // graph's walk over the layers (one token in decode, one ubatch in prefill), detected as a
    // non-increasing layer index. Scores are rescaled before the weight could overflow.
    void freq_touch(int32_t id);
    void freq_pass_check(int il);
    int freq_half_life_ = 0;
    // cache_prefill_cold: this batch is a wide graph whose misses enter cold; cold_gen_ marks the
    // entries entered cold in the current generation so no promotion of the same batch warms them.
    bool prefill_cold_ = false;
    bool cold_batch_ = false;
    int cold_min_tokens_ = MoeStreamConfig::prefill_cold_min_tokens;
    int top_k_ = 0; // routed experts per token (set_top_k); 0 leaves the cold rule off
    std::vector<uint64_t> cold_gen_;
    int freq_last_il_ = -1;
    double freq_growth_ = 1.0;
    double freq_weight_ = 1.0;
    size_t min_entry_bytes_ = 0;
    std::vector<double> cfreq_;
    struct EvictCand {
        double score;
        int32_t rank; // position cold to hot: the LRU order that breaks a tie
        int32_t id;
    };
    std::vector<EvictCand> evict_cand_;

    // I/O trace buffer. Its own leaf mutex, never held across a read: the lanes append
    // concurrently, and the eval thread swaps the buffer out between decodes.
    bool io_trace_on_ = false;
    std::mutex io_trace_mtx_;
    std::vector<IoTraceRow> io_trace_rows_;

    bool active_ = false;
    bool load_all_ = false;
    bool pin_ = false; // pin committed cache pages (MoeStreamConfig::cache_pin); cleared on refusal
    bool pin_said_ = false;
    bool pin_fit_ = false;        // step the budget down on a part-way refusal (MoeStreamConfig::cache_pin_fit)
    size_t pin_floor_ = 0;        // see set_pin_floor
    size_t pin_refused_at_ = 0;   // cache bytes resident when the platform first refused a pin; 0 = never
    uint64_t pin_shrunk_gen_ = 0; // the load generation the budget last stepped down in
    // Written by the eval thread only (init, set_overlap); read by a lane on a failed read, hence atomic.
    std::atomic<bool> overlap_{false};
    bool two_wave_ = false;                  // publish the first projection's jobs before committing the rest (#118)
    bool prefetch_sync_ = false;             // test only: drain prefetch reads synchronously (serial mode)
    bool spec_adopt_ = false;                // route-ahead: staged layers adopt committed spec reads (see quiesce_spec)
    std::vector<int32_t> spec_adopt_ids_;    // scratch: the loading layer's pending spec entries
    std::vector<int32_t> spec_done_scratch_; // scratch: completed entries being integrated
    std::vector<int32_t> spec_cand_;         // scratch: prefetch candidates surviving the residency filter
    // How many completed entries are waiting to be integrated. Mirrors spec_done_.size(), but
    // readable without the lanes' mutex, so the route-ahead settle — which runs before every issue,
    // 40 times a token — can skip the lock entirely on the common empty case.
    std::atomic<long long> spec_done_pending_{0};
    // Churn detector: which entries have ever been evicted, and how many reads went to an entry
    // that had been resident before. A prefetch cannot reduce the bytes a routing needs — the
    // ideal is the same bytes, earlier — so any read of something the cache already had once is
    // the cache paying twice, and it is the only way a "100% useful" prefetch can still raise
    // the byte count. Reported per run; cheap enough (one byte per entry, one branch per read).
    std::vector<uint8_t> ever_evicted_;
    long long rereads_ = 0, evictions_ = 0, cold_admits_ = 0;
    // Eval-thread wait counters (see IExpertSource::Stats): the async load's previous-batch drain
    // (lives in the compute residual) and the route-ahead adoption wait (lives inside mgmt).
    // Eval-thread only, like the LRU itself — two clock reads per wait, nothing on the lanes.
    long long drain_wait_ns_ = 0, adopt_wait_ns_ = 0;
    int n_layer_ = 0;
    int n_expert_ = 0;
    // What shared slots are aligned to: the coarsest read granule over the shards' readers, never
    // less than a page. Settled at init, once the readers have asked the files.
    size_t buf_align_ = 0;

    // The positioned readers that own the fd pools, bounces and O_DIRECT decision — one per shard
    // file, in shard order (a single-file model has exactly one). Expert slices are read through
    // them; the dense-weights loader constructs its own, so their O_DIRECT choices are independent
    // (see docs/architecture.md). FileReader is not movable, hence the unique_ptr.
    std::vector<std::unique_ptr<FileReader>> readers_;
    // Whether every shard reader actually got cache bypass (see FileReader::direct) — the AND over
    // shards, settled at init before any worker thread exists and reported verbatim by stats(). The
    // run-level o_direct fact: what the platform served, not what the flag asked for.
    bool effective_direct_ = false;

    std::vector<LayerExperts> layers_;

    // shared-slot mode (cache off): one full-size slot per projection, reused by layers
    void * slot_[ExpertLayout::max_slots] = {nullptr, nullptr, nullptr};

    // LRU mode (cache on): per-(layer, projection) reserved buffers
    size_t cache_max_ = 0; // live budget in bytes: set_cache_budget(), and with a discardable cache follow_
    size_t page_ = pio::vm_page();

    // Every expert of every bound layer resident: the ceiling any budget is clamped to, and (under
    // cache_auto) the cap the init-time sizing starts from. Auto sizing derives cache_max_ once at
    // init and keeps nothing else: its floor and ceiling are locals there, not state.
    size_t total_expert_bytes_ = 0;
    long long cache_resizes_ = 0;

    // The dense (non-expert) weights: their residency policy (mmap / warm / anon) and the buffers it
    // may hold. Set before init via set_dense_tensors, then handed to dense_.init(). The module also
    // exposes a residency sensor (dense_.resident_frac()), sampled here on a throttle for telemetry.
    static constexpr unsigned dense_probe_every = 128; // load_layer calls between dense samples (~2-3 tokens)
    unsigned dense_probe_tick_ = 0;
    std::vector<DenseTensorRef> dense_tensors_; // pending, set before init; moved into dense_
    std::vector<DenseTensorRef> row_tensors_;   // pending row-gathered subset; moved into dense_
    uint64_t row_budget_ = 0;
    DenseWeights dense_;

    // Two measured demands. Both are pure telemetry now that the governor is gone — nothing here
    // sizes the cache — but they are what any future sizing policy would have to reason about, and
    // they explain a run's hit rate after the fact.
    //
    // token_demand_: the bytes routed between two visits to the same layer — one token's working
    // set. Below it the cache stops holding anything BETWEEN tokens, so its hit rate collapses to
    // inter-token correlation only. It says where hits start, and it is not something a device has
    // to concede: measured on gpt-oss it is 1815 MiB, more than the phone gives up — which is why
    // cache-off is the ceiling there.
    //
    // layer_demand_: the largest single layer's routed bytes — the mechanical minimum, since the
    // cache has to hold the layer it is staging right now.
    size_t demand_accum_ = 0;
    size_t token_demand_ = 0;
    size_t layer_demand_accum_ = 0;
    size_t layer_demand_ = 0;
    int last_il_ = -1;
    std::vector<void *> lbuf_[ExpertLayout::max_slots];
    std::vector<size_t> lbuf_sz_[ExpertLayout::max_slots];
    std::vector<uint8_t> cvalid_;
    std::vector<int32_t> cprev_, cnext_;
    std::vector<uint32_t> cstamp_;
    std::vector<uint8_t> cspec_; // 1 if this resident entry was filled speculatively, until first hit

    // Discardable cache (MoeStreamConfig::cache_discard): the policy lives in DiscardableEntries;
    // this source owns the entries, their LRU and the budget, and says which spans an entry owns.
    DiscardableEntries disc_;
    bool discard_wanted_ = false;
    bool discard_drop_ = false;  // debug: drop an offer just before a lookup takes it back
    uint32_t discard_after_ = 0; // generations route-ahead stages ahead of compute
    long long swept_ = 0;        // dropped offers retired from the LRU tail
    uint32_t offer_window_ = 0;  // generations an entry stays ordinary after its last use
    DropFollowingBudget follow_; // the live budget while the kernel may drop from the cache
    size_t lent_from_ = 0;       // an ordinary cache's budget before a loan; 0 = nothing lent
    void move_budget(size_t bytes);
    void settle_offers();
    void retire_dropped_tail();
    void shrink_for_drop(size_t bytes);
    void regrow_for_load();
    bool entry_body(int32_t id, int p, uintptr_t & b0, uintptr_t & b1) const;
    int32_t chead_ = -1, ctail_ = -1;
    uint32_t cgen_ = 0;
    size_t cresident_ = 0;
    long long chits_ = 0, clookups_ = 0;

    // ── speculative prefetch queue (all fields guarded by io_mtx_) ──
    // prefetch() (eval thread) commits pages and enqueues per-projection reads tagged with the
    // entry id; workers drain them on idle lanes; quiesce_spec() (eval thread, at the next real
    // load) integrates fully-read entries into the cache and releases the rest. spec_gen_ bumps to
    // cancel a round. All LRU mutation stays on the eval thread — workers only read bytes.
    std::vector<IoJob> spec_jobs_;
    size_t spec_next_ = 0;
    uint64_t spec_gen_ = 0;
    size_t spec_inflight_ = 0;
    std::vector<int32_t> spec_done_;      // entry ids whose every projection read completed
    std::vector<int32_t> spec_touched_;   // entry ids queued this round (for cleanup at quiesce)
    std::vector<int32_t> spec_remaining_; // per entry id: projection reads still pending (0 = none)
    // Scratch for prefetch(): an expert's jobs are built here and published only once all of its
    // projections have been committed, so a commit that fails part-way can never leave jobs queued
    // against accounting that was never set up. Members rather than locals so the path stays
    // allocation-free after the first call, like the other per-load scratch.
    std::vector<IoJob> spec_stage_;
    std::vector<int32_t> spec_stage_ids_;
    std::vector<int> spec_stage_counts_;
    std::atomic<long long> spec_read_bytes_{0};
    std::atomic<long long> spec_experts_{0};
    std::atomic<long long> spec_useful_{0};

    std::vector<uint8_t> seen_; // per-load dedup scratch [n_expert]

    // I/O lane pool
    int io_threads_ = 1;
    std::vector<std::thread> io_pool_; // lanes 1..N-1
    // Lane 0's worker, which runs only under overlap: serially lane 0 is the eval thread itself.
    // lane0_stop_ (under io_mtx_) hands the lane back without stopping the others.
    std::thread lane0_worker_;
    bool lane0_stop_ = false;
    std::vector<IoJob> jobs_;
    std::mutex io_mtx_;
    std::condition_variable io_cv_, io_cv_done_;
    uint64_t batch_gen_ = 0;
    size_t batch_njobs_ = 0;
    size_t next_idx_ = 0, done_cnt_ = 0;
    bool io_stop_ = false;
    std::atomic<bool> io_err_{false};
    uint32_t batch_flag_gen_ = 0; // async_gen_ of the batch in flight; snapshot under io_mtx_ at publish

    std::atomic<long long> read_ns_{0};
    std::atomic<long long> mgmt_ns_{0}; // staging-section time: vm commit + evict + LRU bookkeeping

    // ── overlap mode: one layer in flight at a time (guaranteed by graph order) ──
    // A ReadyFlag is ready iff its gen == async_gen_; the load thread bumps async_gen_ per
    // layer, marks cache hits ready immediately, and workers mark misses ready after the read.
    // The compute threads look a captured expert tensor up in texp_, then spin/wait on its flag.
    std::vector<ReadyFlag> ready_; // size max_exps * n_expert_; idx = p*n_expert_+e
    std::atomic<uint32_t> async_gen_{0};
    std::atomic<int> cur_il_{-1}; // layer whose experts the hook may wait on
    std::atomic<bool> fatal_{false};
    std::mutex ready_mtx_;
    std::condition_variable ready_cv_;
    // How many compute threads are currently registered to block on a readiness flag. A completing
    // read consults this before taking ready_mtx_: with no waiter there is nothing to wake, and
    // notifying anyway woke every blocked compute thread on every slice completion so each could
    // re-check a predicate that was almost never its own. Registration and publication are both
    // seq_cst so the two cannot miss each other — see on_expert_ready.
    std::atomic<int> ready_waiters_{0};
    // Overlap stall as the UNION of stalled-thread wall intervals (see stall_union.h),
    // not the sum of per-thread waits: one blocked thread already means the graph is not
    // progressing, and sum/n_threads understates whenever a minority of threads waits.
    StallUnion stall_union_;
    // expert tensor* -> (il<<8)|p. Sorted by pointer and static after init, and probed by every
    // compute thread for every routed expert — a flat binary search beats hashing the pointer.
    std::vector<std::pair<const void *, uint32_t>> texp_;
    std::vector<int> staged_; // per-load sorted unique expert scratch
    bool hook_registered_ = false;
    // The fork's setter, resolved from the CPU backend's registry when the hook is enabled.
    void (*set_hook_)(void (*)(const ggml_tensor *, int, void *), void *) = nullptr;
};

} // namespace bmoe
