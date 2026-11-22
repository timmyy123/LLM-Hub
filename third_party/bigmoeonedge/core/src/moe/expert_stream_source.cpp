#include "expert_stream_source.h"
#include "bmoe/log.h"

#include "row_stream.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <utility>

namespace bmoe {

using clock_t_ = std::chrono::steady_clock;

namespace {
// The pages a resident slice is committed over: its own, rounded out, and through the slack after
// it, because a kernel may read past its last block, and where the platform faults on a reserved
// page that was never committed (Windows) that read must land on one that was.
void slice_pages(const char * dst, uint64_t slice, size_t page, uintptr_t & a0, uintptr_t & a1) {
    a0 = (uintptr_t) dst & ~(uintptr_t) (page - 1);
    a1 = ((uintptr_t) dst + slice + pio::tensor_tail_slack() + page - 1) & ~(uintptr_t) (page - 1);
}
} // namespace

ExpertStreamSource::~ExpertStreamSource() {
    shutdown();
}

// ── init: allocate buffers, rebind expert tensors, start the read pool ──────────────
bool ExpertStreamSource::init(const std::vector<std::string> & shard_paths,
                              int n_expert,
                              std::vector<LayerExperts> layers,
                              const MoeStreamConfig & cfg) {
    if (active_) return false;
    if (n_expert <= 0) {
        log_printf(LogLevel::Error, "expert streaming needs a MoE model (n_expert=%d)", n_expert);
        return false;
    }
    if (shard_paths.empty()) {
        log_printf(LogLevel::Error, "expert streaming got no model file");
        return false;
    }

    n_expert_ = n_expert;
    layers_ = std::move(layers);
    n_layer_ = (int) layers_.size();
    load_all_ = cfg.load_all;
    pin_ = cfg.cache_pin;
    pin_fit_ = cfg.cache_pin_fit;
    // A pinned page cannot be dropped, and pinning is what made the kernel swap everyone else, so a
    // cache that can be offered is never pinned. One pass of the stack, plus the layers route-ahead
    // stages before their compute, is the newest an entry can be and still be in a kernel's hands.
    // Offered entries are dropped, never written; a locked page cannot be dropped, and locking is
    // what made the kernel swap everyone else, so a cache that can be offered is never pinned.
    discard_wanted_ = cfg.cache_discard;
    discard_drop_ = cfg.cache_discard_drop;
    discard_after_ = (uint32_t) std::max(0, cfg.route_ahead);
    if (discard_wanted_ && platform_span_ops().supported()) pin_ = false;
    overlap_ = cfg.overlap;
    two_wave_ = cfg.io_two_wave;
    prefetch_sync_ = cfg.prefetch_sync && !cfg.overlap; // serial only: overlap lane 0 is a worker
    // Under route-ahead the speculated ids of a layer ARE the ids its topk will commit, so the
    // demand load adopts that layer's in-flight speculation instead of discarding and re-reading
    // it. Off for every guessing predictor: adopting a wrong guess would make the load wait on
    // reads it does not need.
    spec_adopt_ = cfg.route_ahead > 0;
    freq_half_life_ = std::max(0, cfg.cache_freq_half_life);
    prefill_cold_ = cfg.cache_prefill_cold;
    cold_min_tokens_ = cfg.test_prefill_cold_min_tokens > 0 ? cfg.test_prefill_cold_min_tokens
                                                            : MoeStreamConfig::prefill_cold_min_tokens;
    freq_growth_ = freq_half_life_ > 0 ? std::exp2(1.0 / freq_half_life_) : 1.0;
    cache_max_ = (size_t) std::max(0, cfg.cache_mb) * 1024ull * 1024ull;
    io_threads_ = std::max(1, std::min(MoeStreamConfig::io_threads_max, cfg.io_threads));
    page_ = pio::vm_page(); // the real OS page size, for the dense-residency probe in any cache mode

    // Largest full-tensor byte size per projection, over all bound layers → shared-slot
    // and bounce sizing. Absent projection slots (a fused layout uses fewer than max_exps
    // expert tensors) keep nb2 == 0, so their max_full stays 0 and they are skipped below.
    size_t max_full[ExpertLayout::max_slots] = {0, 0, 0};
    for (const LayerExperts & L : layers_) {
        if (!L.bound) continue;
        for (int p = 0; p < ExpertLayout::max_slots; ++p) {
            const size_t full = (size_t) L.proj[p].nb2 * (size_t) n_expert_;
            max_full[p] = std::max(max_full[p], full);
            total_expert_bytes_ += full; // every expert of every bound layer resident = the full set
        }
    }
    size_t max_full_any = 0;
    for (int p = 0; p < ExpertLayout::max_slots; ++p)
        max_full_any = std::max(max_full_any, max_full[p]);
    if (max_full_any == 0) {
        log_printf(LogLevel::Error, "no MoE layers were bound");
        return false;
    }

    // Read pool: one reader per shard file, each owning a private fd + bounce per lane so concurrent
    // preads never contend, and the O_DIRECT request + its verify/fallback (per file — a shard set
    // could in principle straddle storage with different O_DIRECT behaviour). The dense-weights
    // loader opens its own readers, so its cache-bypass choice is independent of this one. Opened
    // before any expert buffer is allocated, because the buffers are aligned to what the readers
    // learn about the files: the coarsest granule over the shards, and never less than a page.
    const size_t max_slice = max_full_any / (size_t) n_expert_;
    buf_align_ = page_;
    for (const std::string & sp : shard_paths) {
        readers_.push_back(std::unique_ptr<FileReader>(new FileReader()));
        if (!readers_.back()->open(sp, io_threads_, cfg.o_direct, max_slice)) return false;
        buf_align_ = std::max(buf_align_, readers_.back()->buffer_align());
    }

    // Auto budget: size the cache to the device once, now that the full expert-set size is known.
    // (validate() guarantees cache_mb == 0 here, so this is the sole source of cache_max_ when auto.)
    // One shot — the floor and the cap decide the budget here and are not kept: nothing re-sizes it
    // afterwards except an explicit set_cache_budget().
    if (cfg.cache_auto) {
        const size_t floor = (size_t) std::max(0, cfg.cache_floor_mb) * 1024ull * 1024ull; // RAM to leave free
        // Hard cap: the whole expert set, further capped by the user's ceiling when set.
        const size_t ceil = (size_t) std::max(0, cfg.cache_ceil_mb) * 1024ull * 1024ull;
        const size_t hard_cap = ceil > 0 ? std::min(ceil, total_expert_bytes_) : total_expert_bytes_;
        const size_t min_budget =
            std::min<size_t>((size_t) MoeStreamConfig::cache_min_mb * 1024ull * 1024ull, hard_cap);
        uint64_t avail = pio::mem_available_bytes();
        // Under anon/pinned the dense weights are ABOUT to move from reclaimable page cache into
        // buffers the kernel cannot take back (dense_.init runs below). At this moment the kernel
        // still counts those pages as available, so a budget sized from the raw number plans the
        // dense set twice and overcommits by its whole size — enough to take the device down on a
        // model whose dense set is large (DeepSeek V4's is ~6.5 GiB). Budget as if the conversion
        // had already happened.
        if (cfg.dense_weights == DenseWeightsMode::Anonymous || cfg.dense_weights == DenseWeightsMode::Pinned) {
            uint64_t dense_pending = 0;
            // The same rule dense_.init applies below: a tensor larger than what is available is
            // never converted, it stays mmap'd (qwen4exp's ~28.8 GB n-gram table). Counting it here
            // would reserve the whole of RAM for a conversion that will not happen and size the
            // cache to nothing.
            for (const DenseTensorRef & d : dense_tensors_) {
                if (!d.tensor || d.size > avail) continue;
                // Nor is a table the plan left mapped (dense_mapped): it is never converted.
                if (std::find(cfg.dense_mapped.begin(), cfg.dense_mapped.end(), d.tensor->name) !=
                    cfg.dense_mapped.end())
                    continue;
                // Nor is a row-gathered table converted: it costs its window, not its size, and
                // reserving the difference would hand the cache back exactly what this policy freed.
                bool row = false;
                for (const DenseTensorRef & r : row_tensors_)
                    if (r.tensor == d.tensor) {
                        row = true;
                        break;
                    }
                dense_pending += row ? 0 : d.size;
            }
            if (!row_tensors_.empty()) dense_pending += row_budget_ ? row_budget_ : RowStream::default_budget_bytes;
            const uint64_t deduct = std::min(avail, dense_pending);
            if (deduct > 0) {
                avail -= deduct;
                log_printf(LogLevel::Info, "cache auto — reserving %llu MiB for the dense-weight conversion",
                           (unsigned long long) (deduct / (1024 * 1024)));
            }
        }
        if (avail == 0) {
            cache_max_ = min_budget;
            log_printf(LogLevel::Warn, "cache auto — available memory unknown, using the %zu MiB floor",
                       min_budget / (1024 * 1024));
        } else {
            size_t budget = avail > floor ? (size_t) (avail - floor) : 0;
            budget = std::max(budget, min_budget);
            budget = std::min(budget, hard_cap);
            cache_max_ = budget;
            log_printf(LogLevel::Info,
                       "cache auto — %llu MiB available, leaving %zu MiB free → %zu MiB budget"
                       " (cap %zu MiB)",
                       (unsigned long long) (avail / (1024 * 1024)), floor / (1024 * 1024), cache_max_ / (1024 * 1024),
                       hard_cap / (1024 * 1024));
        }
    }

    if (cache_max_ == 0) {
        // One shared slot per present projection, reused across layers (one layer computes
        // at a time). Rebind every bound layer's expert tensors onto them; only routed
        // slices are ever valid. Absent slots (max_full[p] == 0) get no buffer.
        for (int p = 0; p < ExpertLayout::max_slots; ++p) {
            if (max_full[p] == 0) continue;
            slot_[p] = pio::alloc_aligned(buf_align_, max_full[p] + pio::tensor_tail_slack());
            if (!slot_[p]) {
                log_printf(LogLevel::Error, "slot alloc %zu failed", max_full[p]);
                return false;
            }
        }
        for (LayerExperts & L : layers_) {
            if (!L.bound) continue;
            for (int p = 0; p < ExpertLayout::max_slots; ++p)
                if (L.proj[p].tensor) L.proj[p].tensor->data = slot_[p];
        }
    } else {
        // LRU cache: one reserved (address-only) buffer per (layer, projection). Physical
        // pages appear on the first miss and are released on eviction. mul_mat_id needs
        // each expert at its canonical offset e*nb2 inside tensor->data, so the buffers
        // live at fixed per-layer addresses; lazy commit keeps that affordable. (page_ is already
        // set above: the dense-residency probe needs it in every cache mode, not just this one.)
        for (int p = 0; p < ExpertLayout::max_slots; ++p) {
            lbuf_[p].assign(n_layer_, nullptr);
            lbuf_sz_[p].assign(n_layer_, 0);
        }
        for (int il = 0; il < n_layer_; ++il) {
            LayerExperts & L = layers_[il];
            if (!L.bound) continue;
            for (int p = 0; p < ExpertLayout::max_slots; ++p) {
                if (!L.proj[p].tensor) continue; // absent slot in a fused layout
                // The last expert's slice is followed by the slack a kernel may read past it.
                const size_t full = (size_t) L.proj[p].nb2 * (size_t) n_expert_ + pio::tensor_tail_slack();
                lbuf_[p][il] = pio::vm_reserve(full);
                if (!lbuf_[p][il]) {
                    log_printf(LogLevel::Error, "vm_reserve %zu failed (layer %d)", full, il);
                    return false;
                }
                lbuf_sz_[p][il] = full;
                L.proj[p].tensor->data = lbuf_[p][il];
            }
        }
        const size_t n_entry = (size_t) n_layer_ * n_expert_;
        cvalid_.assign(n_entry, 0);
        cstamp_.assign(n_entry, 0);
        // One pass of the stack, plus the layers route-ahead stages before their compute, is the
        // newest an entry can be and still be in a kernel's or an I/O lane's hands.
        offer_window_ = (uint32_t) n_layer_ + discard_after_ + 1;
        follow_.reset(cache_max_, pin_floor_, offer_window_);
        disc_.init(n_entry, offer_window_, discard_drop_, discard_wanted_ ? &platform_span_ops() : nullptr,
                   [this](int32_t id, const std::function<void(Span)> & fn) {
                       for (int p = 0; p < ExpertLayout::max_slots; ++p) {
                           uintptr_t b0 = 0, b1 = 0;
                           if (entry_body(id, p, b0, b1)) fn(Span{(void *) b0, (size_t) (b1 - b0)});
                       }
                   });
        cprev_.assign(n_entry, -1);
        cnext_.assign(n_entry, -1);
        cspec_.assign(n_entry, 0);
        spec_remaining_.assign(n_entry, 0);
        ever_evicted_.assign(n_entry, 0);
        cold_gen_.assign(n_entry, 0);
        if (freq_half_life_ > 0) {
            cfreq_.assign(n_entry, 0.0);
            freq_weight_ = 1.0;
            freq_last_il_ = -1;
            min_entry_bytes_ = 0;
            for (int il = 0; il < n_layer_; ++il) {
                if (!layers_[il].bound) continue;
                const size_t b = entry_bytes(il);
                if (b && (min_entry_bytes_ == 0 || b < min_entry_bytes_)) min_entry_bytes_ = b;
            }
            log_printf(LogLevel::Info, "expert cache evicts by decayed frequency (half-life %d tokens)",
                       freq_half_life_);
        }
        chead_ = ctail_ = -1;
        cresident_ = 0;
        cgen_ = 0;
        chits_ = 0;
        clookups_ = 0;
    }

    // Each shard resolved direct for itself at open — the platform's answer plus the open-time
    // verify — so the run-level fact is the weakest shard: a mixed run must not be advertised as
    // fully direct, and a metadata-only first shard that never carried an expert read must not
    // flatter the shards that do. Settled here, before any worker thread exists; stats() reads it
    // cross-thread for the rest of the run.
    effective_direct_ = true;
    for (const auto & r : readers_)
        effective_direct_ = effective_direct_ && r->direct();
    for (const LayerExperts & L : layers_) {
        if (!L.bound) continue;
        for (int p = 0; p < ExpertLayout::max_slots; ++p)
            if (L.proj[p].nb2 && (L.proj[p].file_idx < 0 || L.proj[p].file_idx >= (int) readers_.size())) {
                log_printf(LogLevel::Error, "expert tensor points at shard %d of %zu", L.proj[p].file_idx,
                           readers_.size());
                return false;
            }
    }

    seen_.assign(n_expert_, 0);
    jobs_.reserve((size_t) n_expert_ * ExpertLayout::max_slots);
    batch_gen_ = 0;
    next_idx_ = 0;
    done_cnt_ = 0;
    io_stop_ = false;
    io_err_.store(false);

    if (overlap_) build_ready_map();

    // Hand the dense (non-expert) weights to their policy module: it warms them into the page cache,
    // or reads them into anon buffers and rebinds, and owns the residency sensor — all on the caller's
    // thread before the workers start (the Anonymous mode rebinds tensor->data). The dense byte ranges
    // are the complement of the expert ranges in the file, computed once here and shared by the warm
    // sweep and the sensor.
    {
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> exp(readers_.size());
        for (const LayerExperts & L : layers_) {
            if (!L.bound) continue;
            for (int p = 0; p < ExpertLayout::max_slots; ++p) {
                const uint64_t sz = (uint64_t) L.proj[p].nb2 * (uint64_t) n_expert_;
                if (sz) exp[L.proj[p].file_idx].push_back({L.proj[p].file_off, L.proj[p].file_off + sz});
            }
        }
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> ranges(readers_.size());
        for (size_t s = 0; s < readers_.size(); ++s)
            ranges[s] = DenseWeights::byte_ranges(std::move(exp[s]), readers_[s]->file_size());
        dense_.set_row_gathered(std::move(row_tensors_), row_budget_);
        dense_.set_mapped(cfg.dense_mapped);
        if (!dense_.init(cfg.dense_weights, shard_paths, std::move(ranges), std::move(dense_tensors_))) {
            log_printf(LogLevel::Error, "dense-weights init failed");
            return false;
        }
    }

    active_ = true;
    // Serial: lane 0 is the calling thread (it drains inline), workers own lanes 1..N-1.
    // Overlap: the caller never drains — every lane 0..N-1 gets a worker so reads proceed
    // while the compute threads run the FFN and block per expert on the readiness flags.
    // Lane 0's worker is kept apart from the others, since a switch of mode starts or stops it alone.
    for (int lane = 1; lane < io_threads_; ++lane)
        io_pool_.emplace_back(&ExpertStreamSource::io_worker, this, lane);
    if (overlap_) lane0_worker_ = std::thread(&ExpertStreamSource::io_worker, this, 0);

    // effective_direct_ (set where the readers opened) is the run's o_direct fact — see above.
    log_printf(LogLevel::Info, "expert streaming ON  n_expert=%d o_direct=%d io_threads=%d cache=%zu MiB shards=%zu",
               n_expert_, (int) effective_direct_, io_threads_, cache_max_ >> 20, readers_.size());
    return true;
}

// One readiness cell per (projection, expert), and a map from each bound expert tensor (the
// persistent model weight, stable across decodes) back to its (layer, projection) so the hook can
// find the cell to wait on. Built after the rebind, at init under overlap or at the first switch to
// it (set_overlap), and never again: the tensors it maps do not move.
void ExpertStreamSource::build_ready_map() {
    ready_ = std::vector<ReadyFlag>((size_t) ExpertLayout::max_slots * (size_t) n_expert_);
    async_gen_.store(0);
    cur_il_.store(-1);
    fatal_.store(false);
    stall_union_.reset();
    batch_flag_gen_ = 0;
    staged_.reserve(n_expert_);
    texp_.clear();
    texp_.reserve((size_t) n_layer_ * ExpertLayout::max_slots);
    for (int il = 0; il < n_layer_; ++il) {
        const LayerExperts & L = layers_[il];
        if (!L.bound) continue;
        for (int p = 0; p < ExpertLayout::max_slots; ++p)
            if (L.proj[p].tensor)
                texp_.emplace_back((const void *) L.proj[p].tensor, ((uint32_t) il << 8) | (uint32_t) p);
    }
    std::sort(texp_.begin(), texp_.end());
}

// Serial and overlap differ in one thread. Serially the eval thread is lane 0 and reads its share
// of a batch itself; under overlap it never reads, so lane 0 gets a worker of its own like the
// others. Everything else - the readers, the cache, the jobs - is the same in both, so switching is
// giving lane 0 its worker or taking it back, plus the hook the overlapped matmul waits through.
// Between two decodes only: no layer is in flight, so the last batch is the only thing to wait for.
bool ExpertStreamSource::set_overlap(bool on) {
    if (!active_) return false;
    if (on == overlap_) return true;
    if (on) {
        if (!hook_registered_ && !enable_overlap_hook()) return false;
        if (ready_.empty()) build_ready_map();
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            lane0_stop_ = false;
        }
        overlap_ = true;
        prefetch_sync_ = false;
        lane0_worker_ = std::thread(&ExpertStreamSource::io_worker, this, 0);
        return true;
    }
    {
        // The last overlapped batch: its reads may outlast the token whose last layer issued them
        // only for load_all, whose surplus no matmul waits on. They are waited for, as the next
        // overlapped load would have.
        std::unique_lock<std::mutex> lk(io_mtx_);
        io_cv_done_.wait(lk, [&] { return done_cnt_ == batch_njobs_ || io_stop_; });
        lane0_stop_ = true;
    }
    io_cv_.notify_all();
    if (lane0_worker_.joinable()) lane0_worker_.join();
    if (hook_registered_ && set_hook_) {
        set_hook_(nullptr, nullptr);
        hook_registered_ = false;
    }
    overlap_ = false;
    return true;
}

bool ExpertStreamSource::reopen_readers() {
    std::lock_guard<std::mutex> lk(io_mtx_);
    if (!jobs_.empty() || !spec_jobs_.empty()) {
        log_printf(LogLevel::Error, "reopen_readers refused: reads are queued");
        return false;
    }
    bool ok = true;
    for (auto & r : readers_)
        ok = r->reopen() && ok;
    return dense_.reopen_readers() && ok;
}

// ── one aligned slice read on a lane ────────────────────────────────────────────────
// The bytes come from the reader; this wraps it with the domain the reader must not know about — the
// per-read I/O trace that attributes a read to its (layer, expert, projection). Latency is timed here
// so the row carries this read's own cost, not the reader's running total.
bool ExpertStreamSource::read_slice(int lane, const IoJob & j) {
    if (j.nbytes == 0) return true;
    const auto t0 = clock_t_::now();
    const long long window = readers_[(size_t) j.file]->read(lane, j.dst, j.off, j.nbytes);
    if (window < 0) return false;

    if (io_trace_on_) {
        const uint64_t lat_ns =
            (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - t0).count();
        IoTraceRow r;
        r.layer = j.layer;
        r.expert = j.expert;
        r.proj = (int8_t) j.proj;
        r.lane = (int8_t) lane;
        r.spec = j.spec;
        r.offset = j.off;
        r.req_bytes = j.nbytes;
        r.read_bytes = (uint64_t) window; // the aligned window pulled — what bandwidth is judged against
        r.latency_ns = lat_ns;
        std::lock_guard<std::mutex> lk(io_trace_mtx_);
        io_trace_rows_.push_back(r);
    }
    return true;
}

void ExpertStreamSource::set_io_trace(bool on) {
    std::lock_guard<std::mutex> lk(io_trace_mtx_);
    io_trace_on_ = on;
    io_trace_rows_.clear();
}

void ExpertStreamSource::take_io_trace_rows(std::vector<IoTraceRow> & out) {
    std::lock_guard<std::mutex> lk(io_trace_mtx_);
    out.swap(io_trace_rows_);
    io_trace_rows_.clear();
}

void ExpertStreamSource::io_drain(int lane, uint64_t my_gen) {
    for (;;) {
        IoJob j;
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            if (batch_gen_ != my_gen || next_idx_ >= batch_njobs_) return;
            // Copy under the lock: a two-wave publish appends to jobs_ mid-batch, and a vector
            // that reallocates would leave a reference taken here dangling.
            j = jobs_[next_idx_++];
        }
        if (!read_slice(lane, j)) {
            io_err_.store(true);
            if (overlap_) fatal_.store(true, std::memory_order_release);
        }
        // Overlap: publish this expert's readiness (batch_flag_gen_ is the async_gen_ of the
        // in-flight batch, fixed until it fully drains) and wake any compute thread blocked on
        // it. Notify even on failure so a straggler wakes and observes fatal_ instead of hanging.
        if (j.flag >= 0) {
            // Publish first, then look for waiters. on_expert_ready registers itself before its own
            // last look at the flag, and both operations are seq_cst, so the two cannot miss each
            // other: either the compute thread sees the flag and never sleeps, or it is counted here
            // and gets woken. With nobody waiting — the common case, since a slice usually lands
            // inside the spin — this costs one atomic load instead of a mutex plus a notify_all that
            // woke EVERY compute thread blocked on ANY expert to re-check its own predicate.
            ready_[(size_t) j.flag].gen.store(batch_flag_gen_, std::memory_order_seq_cst);
            if (ready_waiters_.load(std::memory_order_seq_cst) != 0) {
                std::lock_guard<std::mutex> lk(ready_mtx_);
                ready_cv_.notify_all();
            }
        }
        std::lock_guard<std::mutex> lk(io_mtx_);
        if (++done_cnt_ == batch_njobs_) io_cv_done_.notify_all();
    }
}

void ExpertStreamSource::io_worker(int lane) {
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(io_mtx_);
            // `next_idx_ < batch_njobs_` admits a batch that GREW in the same generation (a
            // two-wave publish): a worker that drained wave one and left has seen == batch_gen_,
            // so the gen comparison alone would never bring it back for wave two.
            io_cv_.wait(lk, [&] {
                return io_stop_ || (lane == 0 && lane0_stop_) || batch_gen_ > seen || next_idx_ < batch_njobs_ ||
                       spec_next_ < spec_jobs_.size();
            });
            if (io_stop_ || (lane == 0 && lane0_stop_)) return;
        }
        uint64_t g;
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            g = batch_gen_;
        }
        if (g > seen) seen = g;
        // Real batch first — it is latency-critical. With nothing (left) to drain this returns
        // on its first lock, so calling it on a spec-only wake costs one mutex round.
        io_drain(lane, seen);
        // Then use spare capacity on queued speculative reads, yielding the moment a real batch
        // arrives (drain_spec bails when batch_gen_ advances past this worker's `seen`).
        drain_spec(lane, seen);
    }
}

// Prefetch: on the eval thread, commit pages and enqueue speculative per-projection reads for the
// given experts of layer il. LRU-safe (same thread as load_layer); workers only read the bytes.
void ExpertStreamSource::prefetch(int il, const int32_t * ids, int n_ids) {
    if (!active_ || cache_max_ == 0 || il < 0 || il >= n_layer_ || !layers_[il].bound || !ids || n_ids <= 0) return;
    const LayerExperts & L = layers_[il];
    bool any = false;

    // Stage each expert's jobs locally and publish them only once ALL of its projections have been
    // committed. Pushing them as they were built meant a commit failure mid-expert could leave jobs
    // queued for an entry whose spec_remaining_ was never set: a worker would then decrement it from
    // zero, the entry could never reach zero to complete, the quiesce would never see it in
    // spec_touched_ to release, and the `!= 0` guard above would skip that expert's speculation for
    // the rest of the run. One failure, permanent damage — in exactly the low-memory situation this
    // path exists to degrade gracefully in.
    //
    // Committing before taking io_mtx_ matters for the same call: prefetch runs on the eval thread
    // right after a real batch was published, so holding the mutex across a syscall per projection
    // stalls the lanes trying to pull real read indices out of it. The pages are the eval thread's
    // to commit — no worker touches them until the jobs below exist.
    std::vector<IoJob> & staged = spec_stage_;
    std::vector<int32_t> & staged_ids = spec_stage_ids_;
    std::vector<int> & staged_counts = spec_stage_counts_;
    staged.clear();
    staged_ids.clear();
    staged_counts.clear();

    // Filter under ONE lock, not one per expert. This runs on the eval thread while every I/O lane
    // is taking the same mutex to pull its next read index, and a blocked eval thread is a
    // descheduled compute thread: measured on a phone, the per-expert acquisition was part of ~120
    // acquisitions per token that cost ~600 ms of wall (the same run with a single lane, hence no
    // contention, spent a quarter of the compute time on identical work).
    std::vector<int32_t> & cand = spec_cand_;
    cand.clear();
    {
        std::lock_guard<std::mutex> lk(io_mtx_);
        for (int i = 0; i < n_ids; ++i) {
            const int e = ids[i];
            if (e < 0 || e >= n_expert_) continue;
            const int32_t id = il * n_expert_ + e;
            if (cvalid_[id] || spec_remaining_[id] != 0) continue; // already resident or already queued
            cand.push_back(e);
        }
    }
    for (int e : cand) {
        const int32_t id = il * n_expert_ + e;
        const size_t mark = staged.size();
        int njobs = 0;
        bool ok = true;
        for (int p = 0; p < ExpertLayout::max_slots; ++p) {
            const uint64_t slice = L.proj[p].nb2;
            if (slice == 0) continue;
            char * dst = (char *) lbuf_[p][il] + (uint64_t) e * slice;
            if (!commit_slice(il, e, p)) { // low on memory — stop quietly
                ok = false;
                break;
            }
            staged.push_back({dst, L.proj[p].file_off + (uint64_t) e * slice, slice, id, (int16_t) L.proj[p].file_idx,
                              e, (int16_t) il, (int8_t) p, 1});
            ++njobs;
        }
        if (!ok) {
            // Hand back what this expert already committed and drop its jobs: an entry is queued
            // whole or not at all.
            staged.resize(mark);
            release_entry_pages(id);
            break;
        }
        if (njobs == 0) continue;
        if (ever_evicted_[id]) ++rereads_; // same accounting on the speculative path
        staged_ids.push_back(id);
        staged_counts.push_back(njobs);
        any = true;
    }
    if (!any) return;

    std::lock_guard<std::mutex> lk(io_mtx_);
    for (size_t s = 0; s < staged_ids.size(); ++s) {
        spec_remaining_[staged_ids[s]] = staged_counts[s];
        spec_touched_.push_back(staged_ids[s]);
    }
    spec_jobs_.insert(spec_jobs_.end(), staged.begin(), staged.end());
    if (prefetch_sync_) {
        // Test path: read the queued slices now on lane 0 (free on the eval thread in serial mode),
        // so the next quiesce integrates them deterministically. Mirrors drain_spec's accounting.
        for (;;) {
            IoJob j;
            uint64_t g;
            if (spec_next_ >= spec_jobs_.size()) break;
            g = spec_gen_;
            j = spec_jobs_[spec_next_++];
            ++spec_inflight_;
            const bool ok = read_slice(0, j);
            if (ok && g == spec_gen_) {
                spec_read_bytes_.fetch_add((long long) j.nbytes);
                if (--spec_remaining_[j.flag] == 0) {
                    spec_done_.push_back(j.flag);
                    spec_done_pending_.fetch_add(1, std::memory_order_relaxed);
                }
            }
            --spec_inflight_;
        }
        return;
    }
    io_cv_.notify_all();
}

// Drain queued speculative reads on an idle lane. Bails as soon as a real batch this worker has
// not served appears, so speculation never delays real work. A completed entry (all projections
// read under the current spec generation) is handed to the eval thread via spec_done_.
void ExpertStreamSource::drain_spec(int lane, uint64_t worker_seen) {
    for (;;) {
        IoJob j;
        uint64_t g;
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            // Shutting down, real work waiting, or lane 0 handed back to the eval thread.
            if (io_stop_ || batch_gen_ > worker_seen || (lane == 0 && lane0_stop_)) return;
            if (spec_next_ >= spec_jobs_.size()) return;
            g = spec_gen_;
            j = spec_jobs_[spec_next_++];
            ++spec_inflight_;
        }
        const bool ok = read_slice(lane, j);
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            if (ok && g == spec_gen_) { // ignore reads from a cancelled round
                spec_read_bytes_.fetch_add((long long) j.nbytes);
                if (spec_remaining_[j.flag] > 0 && --spec_remaining_[j.flag] == 0) {
                    spec_done_.push_back(j.flag);
                    spec_done_pending_.fetch_add(1, std::memory_order_relaxed);
                    // The adoption wait watches per-entry counters, not just the in-flight count.
                    io_cv_done_.notify_all();
                }
            } else if (!ok && spec_adopt_) {
                // Forget-on-failure: zeroing the counter keeps the adoption wait from parking
                // forever, and !cvalid means a later demand read re-buys the entry.
                spec_remaining_[j.flag] = 0;
                io_cv_done_.notify_all();
            }
            if (--spec_inflight_ == 0) io_cv_done_.notify_all();
        }
    }
}

// Quiesce speculation before real staging: cancel queued reads, wait out in-flight ones, then on
// this (eval) thread integrate every fully-read entry into the cache and release the rest. All LRU
// mutation happens here, single-threaded, so it never races the real staging that follows.
void ExpertStreamSource::quiesce_spec(int adopt_il) {
    // Adoption (route-ahead only): the layer about to stage COMMITTED to the ids its speculation
    // was issued for, so its queued spec reads are not bets to discard — they are the layer's own
    // demand reads, already staged and possibly partly done. Keep exactly those, drop the rest of
    // the queue, and FINISH them here (this thread drains alongside any idle lane) before the
    // normal quiesce below integrates them as resident entries — which the staging then sees as
    // hits, so no byte is read twice. Without this, a committed layer's in-flight speculation was
    // released at its own load and re-read on demand: measured on the host at depth 2, every
    // early read was wasted that way (0% useful, double flash per token). The generation is NOT
    // bumped before the drain — bumping is what disowns reads, and these are being adopted.
    // Adoption (route-ahead only): under a committed routing, EVERY job in the spec queue is a
    // read some layer will demand verbatim — the loading layer's own jobs are its demand reads
    // already staged, and the other layers' jobs are demand reads a few callbacks early. So
    // nothing is cancelled: the loading layer's jobs move to the front and are finished now,
    // completed entries integrate, and everything else stays queued for the idle lanes. The
    // destructive quiesce below (cancel, disown, release) remains the right treatment for the
    // guessing predictors, whose queue really is bets. Measured before this branch existed: at
    // depth 2 every early read was destroyed — by the intervening load's quiesce or by the
    // settle preceding each issue — for 0% useful and double flash per token.
    if (spec_adopt_ && adopt_il >= 0) {
        std::vector<int32_t> & adopt = spec_adopt_ids_;
        adopt.clear();
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            std::stable_partition(spec_jobs_.begin() + (ptrdiff_t) spec_next_, spec_jobs_.end(),
                                  [&](const IoJob & j) { return j.layer == (int16_t) adopt_il; });
            for (int32_t id : spec_touched_)
                if (id / n_expert_ == adopt_il && spec_remaining_[id] != 0) adopt.push_back(id);
            // Without the full quiesce ever running, spec_touched_ would grow for the whole
            // session; entries fully settled need no release sweep, so drop them here.
            spec_touched_.erase(std::remove_if(spec_touched_.begin(), spec_touched_.end(),
                                               [&](int32_t id) { return spec_remaining_[id] == 0; }),
                                spec_touched_.end());
        }
        if (!adopt.empty()) {
            io_cv_.notify_all(); // idle worker lanes take the queue
            // Serially, lane 0 belongs to this thread, so it reads its own layer's jobs alongside
            // the workers; under overlap every lane is a worker and the eval thread must not
            // touch a reader — doing so raced worker 0 on one reader and fed the matmul
            // corrupted slices (first symptom: the routing agreeing BETTER with its own
            // prediction, garbage states drifting toward whatever the perturbed layers produce).
            //
            // Then WAIT for the loading layer's own entries, in both modes — measured, not
            // assumed: a no-wait variant that skipped these entries at staging and published
            // per-expert readiness from the completing reads was built and benchmarked, and it
            // LOST (5.8 vs 6.6 tok/s on the host at depth 2). Late integration enters the
            // entries cold, the eviction churns them, and the per-expert stalls it trades this
            // wait for cost more than the wait — which is short by construction, because these
            // jobs started a layer or more ago.
            if (!overlap_) drain_adopted(adopt_il);
            const auto tw0 = clock_t_::now();
            {
                std::unique_lock<std::mutex> lk(io_mtx_);
                io_cv_done_.wait(lk, [&] {
                    if (io_stop_) return true;
                    for (int32_t id : adopt)
                        if (spec_remaining_[id] != 0) return false;
                    return true;
                });
            }
            adopt_wait_ns_ +=
                (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - tw0).count();
        }
        spec_integrate_done();
        return;
    }
    std::vector<int32_t> done, touched;
    {
        std::unique_lock<std::mutex> lk(io_mtx_);
        ++spec_gen_; // cancel queued-but-unstarted reads and disown in-flight ones on completion
        spec_jobs_.clear();
        spec_next_ = 0;
        io_cv_done_.wait(lk, [&] { return spec_inflight_ == 0 || io_stop_; });
        done.swap(spec_done_);
        spec_done_pending_.store(0, std::memory_order_relaxed);
        touched.swap(spec_touched_);
    }
    // Integrate completed entries (all projections resident) into the LRU cache.
    for (int32_t id : done) {
        if (spec_remaining_[id] != 0 || cvalid_[id]) { // incomplete, or a real read already took it
            spec_remaining_[id] = 0;
            continue;
        }
        cvalid_[id] = 1;
        cspec_[id] = 1;  // speculative until a real lookup hits it (then counted useful)
        cstamp_[id] = 0; // not used this generation → evictable if the budget is tight
        cresident_ += entry_bytes(id / n_expert_);
        disc_.used(id, cgen_);
        lru_push_back(id); // cold end: a mispredicted expert is reclaimed before any demanded one
        spec_experts_.fetch_add(1);
    }
    // Release pages of entries that never finished (a cancelled or failed read).
    for (int32_t id : touched) {
        if (spec_remaining_[id] == 0) continue; // completed above (or already integrated)
        release_entry_pages(id);
        spec_remaining_[id] = 0;
    }
}

// A token's pass over the layer stack is monotonic in il, so a non-increasing layer index means the
// previous token's pass just ended and the bytes it demanded are known. Prefill measures the
// batch's union (larger); the first decode token overwrites it with the decode value.
void ExpertStreamSource::account_demand(int il, int n_unique) {
    if (il <= last_il_) {
        token_demand_ = demand_accum_;
        layer_demand_ = layer_demand_accum_;
        demand_accum_ = 0;
        layer_demand_accum_ = 0;
    }
    last_il_ = il;
    const size_t bytes = (size_t) n_unique * entry_bytes(il);
    demand_accum_ += bytes;
    layer_demand_accum_ = std::max(layer_demand_accum_, bytes); // the widest layer of this pass
}

// Diagnostic telemetry (dense_resident_frac): how much of the dense set the kernel still has in RAM —
// the anon buffers under --dense-weights anon, the mmap ranges otherwise. This is the direct signal
// for whether a reclaim is dropping the model's own weights (and, under anon, whether zram is holding
// them). It feeds nothing (the governor is gone); it is measured only to be read. Throttled and timed
// into mgmt_ns_ so its cost is visible, not hidden in the compute residual. Eval-thread only.
void ExpertStreamSource::maybe_sample_dense() {
    if (++dense_probe_tick_ % dense_probe_every != 0) return;
    const auto t0 = clock_t_::now();
    dense_.sample_residency(page_);
    mgmt_ns_.fetch_add((long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - t0).count());
}

// Explicit budget change from outside a decode (an app's memory-pressure callback, or the shrink
// gate): the owner's new budget. The brake and a prefill's loan move the live budget below it
// (brake_cache_budget, lend_cache) without replacing it.
void ExpertStreamSource::set_cache_budget(size_t bytes) {
    if (cache_max_ == 0) return; // initialised off (shared-slot mode); the LRU buffers do not exist
    if (bytes > cache_max_) {
        // Growing evicts nothing, so it needs no quiesce — and must not do one: cancelling the
        // speculative reads in flight on every grow would quietly defeat --prefetch.
        cache_max_ = std::min(bytes, total_expert_bytes_);
        follow_.reset(cache_max_, pin_floor_, offer_window_);
        ++cache_resizes_;
        return;
    }
    if (bytes == cache_max_) return;
    quiesce_spec(); // cancel/drain spec reads so no worker is mid-write to an evicted page
    // Keep the budget strictly positive. The shared-slot buffers were never allocated (LRU mode was
    // chosen at init), and load_layer branches on cache_max_ == 0 to pick shared-slot vs LRU — so a
    // runtime zero would route into buffers that do not exist. This shrinks toward, never to, zero.
    cache_max_ = std::max<size_t>(1, bytes);
    follow_.reset(cache_max_, pin_floor_, offer_window_); // an owner's explicit resize is the new ceiling
    ++cache_resizes_;
    // No cstamp guard: with no decode in flight, nothing is staged for the current generation, so
    // every resident entry (coldest first) is a valid eviction target.
    evict_to_budget(/*spare_current=*/false);
}

// The live budget to `bytes`, the ceiling untouched: what brake_cache_budget and a loan move.
void ExpertStreamSource::move_budget(size_t bytes) {
    bytes = std::max<size_t>(1, std::min(bytes, total_expert_bytes_));
    if (bytes == cache_max_) return;
    const bool shrink = bytes < cache_max_;
    if (shrink) quiesce_spec(); // as set_cache_budget: no worker mid-write to an evicted page
    cache_max_ = bytes;
    ++cache_resizes_;
    if (shrink) evict_to_budget(/*spare_current=*/false);
}

void ExpertStreamSource::brake_cache_budget(size_t bytes) {
    if (cache_max_ == 0) return;
    if (!disc_.enabled()) return set_cache_budget(bytes);
    disc_.pressure();
    follow_.braked(bytes, cgen_);
    move_budget(follow_.current());
}

void ExpertStreamSource::lend_cache(size_t bytes) {
    if (cache_max_ == 0 || bytes == 0) return;
    if (!disc_.enabled()) {
        if (lent_from_ == 0) lent_from_ = cache_max_;
        return set_cache_budget(cache_max_ > bytes ? cache_max_ - bytes : 1);
    }
    follow_.lend(bytes);
    move_budget(follow_.current());
}

void ExpertStreamSource::return_cache() {
    if (cache_max_ == 0) return;
    if (!disc_.enabled()) {
        if (lent_from_ > 0) set_cache_budget(lent_from_);
        lent_from_ = 0;
        return;
    }
    follow_.end_loan();
    move_budget(follow_.current());
}

double ExpertStreamSource::cache_lost_frac(int n) {
    if (cache_max_ == 0 || chead_ == -1 || n <= 0) return -1.0;
    // Spread over the whole list, hot to cold: what a system takes first is the cold end, and a
    // sample of only the hot end would answer "nothing yet" until the damage reached the head.
    size_t entries = 0;
    for (int32_t id = chead_; id != -1; id = cnext_[id])
        ++entries;
    const size_t stride = std::max<size_t>(1, entries / (size_t) n);
    size_t sampled = 0, resident = 0, i = 0;
    for (int32_t id = chead_; id != -1; id = cnext_[id], ++i) {
        // An offered entry the kernel dropped is the discardable cache doing its job - a later miss
        // reads it again, nothing was written - not the reclaim this sensor watches for, which is
        // the system taking pages the cache still counts on (and paying a write for them).
        if (i % stride != 0 || !cvalid_[id] || disc_.offered(id)) continue;
        const int il = id / n_expert_, e = id % n_expert_;
        for (int p = 0; p < ExpertLayout::max_slots; ++p) {
            const uint64_t slice = layers_[il].proj[p].nb2;
            if (slice < 2 * page_) continue; // no page of its own to ask about
            // A page wholly inside the slice: the boundary pages are shared with the neighbours.
            const uintptr_t s = (uintptr_t) lbuf_[p][il] + (uint64_t) e * slice;
            const uintptr_t pg = (s + page_ - 1) & ~(uintptr_t) (page_ - 1);
            pio::vm_resident_sample((void *) pg, page_, &sampled, &resident);
            break; // one page per entry: the entry was written whole, so it is lost whole or not at all
        }
    }
    if (sampled == 0) return -1.0;
    return 1.0 - (double) resident / (double) sampled;
}

size_t ExpertStreamSource::worst_cycle_bytes(int top_k) const {
    // entry_bytes prices a layer's experts uniformly (one entry = every projection of one expert
    // of that layer), so the worst token demands top_k entries from every bound layer. Clamped
    // at n_expert_: a top_k wider than the bank asks for entries that do not exist.
    if (top_k <= 0) return 0;
    size_t cycle = 0;
    for (int il = 0; il < (int) layers_.size(); ++il)
        if (layers_[il].bound) cycle += entry_bytes(il) * (size_t) std::min(top_k, n_expert_);
    return cycle;
}

// ── LRU plumbing ────────────────────────────────────────────────────────────────────
void ExpertStreamSource::lru_unlink(int32_t id) {
    int32_t pv = cprev_[id], nx = cnext_[id];
    if (pv != -1)
        cnext_[pv] = nx;
    else
        chead_ = nx;
    if (nx != -1)
        cprev_[nx] = pv;
    else
        ctail_ = pv;
    cprev_[id] = cnext_[id] = -1;
}
void ExpertStreamSource::lru_push_front(int32_t id) {
    cprev_[id] = -1;
    cnext_[id] = chead_;
    if (chead_ != -1)
        cprev_[chead_] = id;
    else
        ctail_ = id;
    chead_ = id;
}
// Insert at the LRU (cold) end. Speculative entries enter here so a wrong prediction is the first
// thing evicted and can never displace a demanded expert; a real lookup promotes it to the front.
void ExpertStreamSource::lru_push_back(int32_t id) {
    cnext_[id] = -1;
    cprev_[id] = ctail_;
    if (ctail_ != -1)
        cnext_[ctail_] = id;
    else
        chead_ = id;
    ctail_ = id;
}
size_t ExpertStreamSource::entry_bytes(int il) const {
    const LayerExperts & L = layers_[il];
    size_t bytes = 0;
    for (int p = 0; p < ExpertLayout::max_slots; ++p)
        bytes += (size_t) L.proj[p].nb2; // absent slots contribute 0
    return bytes;
}
// Release the physical pages of an entry's slices that no other entry still needs. A slice is
// committed over its own pages and through the slack after it (slice_pages), so the pages at a
// slice boundary belong to two entries: the one that ends there, whose kernels may read past the
// end into them, and the one that starts there. They are released with the last of the two to go.
// Releasing only the pages between the two boundaries, as this once did, left the head page of
// every slice whose left neighbour had ever been resident committed until the buffer was freed:
// one page per expert and projection, outside the ledger `cresident_` keeps. Shared by eviction
// and by discarding an incomplete spec entry; eval thread only, like every LRU mutation.
void ExpertStreamSource::release_entry_pages(int32_t id) {
    disc_.forget(id); // the spans are remapped below; whatever was offered there is gone
    const int il = id / n_expert_, e = id % n_expert_;
    // A neighbour's pages must stay while it is resident or while a speculative read of it is in
    // flight; the workers that finish those reads write the counters under io_mtx_, and only this
    // thread ever raises one, so a value read here can only fall afterwards, which is safe.
    bool left = false, right = false;
    {
        std::lock_guard<std::mutex> lk(io_mtx_);
        const auto live = [&](int x) {
            if (x < 0 || x >= n_expert_) return false;
            const int32_t xid = il * n_expert_ + x;
            return cvalid_[xid] != 0 || spec_remaining_[xid] != 0;
        };
        left = live(e - 1);
        right = live(e + 1);
    }
    for (int p = 0; p < ExpertLayout::max_slots; ++p) {
        const uint64_t slice = layers_[il].proj[p].nb2;
        if (slice == 0) continue; // absent slot in a fused layout
        char * s = (char *) lbuf_[p][il] + (uint64_t) e * slice;
        // head: the pages the left neighbour's slack reaches into; body: ours alone; tail: the
        // pages our slack reaches into, which are the right neighbour's head.
        const uintptr_t head0 = (uintptr_t) s & ~(uintptr_t) (page_ - 1);
        const uintptr_t body0 = ((uintptr_t) s + pio::tensor_tail_slack() + page_ - 1) & ~(uintptr_t) (page_ - 1);
        const uintptr_t body1 = ((uintptr_t) s + slice) & ~(uintptr_t) (page_ - 1);
        const uintptr_t tail1 =
            ((uintptr_t) s + slice + pio::tensor_tail_slack() + page_ - 1) & ~(uintptr_t) (page_ - 1);
        const uintptr_t a0 = left ? body0 : head0;
        const uintptr_t a1 = right ? body1 : tail1;
        if (a1 > a0) pio::vm_evict((void *) a0, (size_t) (a1 - a0));
    }
}
const void * ExpertStreamSource::resident_slice(int il, int e, int p, uint64_t & bytes) const {
    bytes = 0;
    if (cache_max_ == 0 || il < 0 || il >= n_layer_ || e < 0 || e >= n_expert_ || p < 0 ||
        p >= ExpertLayout::max_slots || !layers_[il].bound)
        return nullptr;
    const int32_t id = il * n_expert_ + e;
    if ((size_t) id >= cvalid_.size() || !cvalid_[id] || !lbuf_[p][il]) return nullptr;
    // An entry offered to the kernel (cache_discard) may already be gone, and only the eval thread
    // may take one back: for a reader on another thread it is not resident, and a flash read serves.
    if (disc_.offered(id)) return nullptr;
    bytes = layers_[il].proj[p].nb2;
    if (bytes == 0) return nullptr;
    return (const char *) lbuf_[p][il] + (uint64_t) e * bytes;
}

void ExpertStreamSource::evict_tail() {
    evict_entry(ctail_);
}

void ExpertStreamSource::freq_touch(int32_t id) {
    if (freq_half_life_ > 0) cfreq_[id] += freq_weight_;
}

void ExpertStreamSource::freq_pass_check(int il) {
    if (freq_half_life_ <= 0) return;
    if (il <= freq_last_il_) {
        freq_weight_ *= freq_growth_;
        // Rescale long before a double could overflow: every score and the weight by one factor,
        // which leaves the order (all eviction reads) unchanged.
        if (freq_weight_ > 1e200) {
            const double s = 1.0 / freq_weight_;
            for (double & f : cfreq_)
                f *= s;
            freq_weight_ = 1.0;
        }
    }
    freq_last_il_ = il;
}

void ExpertStreamSource::evict_to_budget(bool spare_current) {
    if (cresident_ <= cache_max_) return;
    if (freq_half_life_ <= 0) {
        // LRU: the staged entries sit at the hot end, so the first spared tail ends the sweep.
        if (!(spare_current && cold_batch_)) {
            while (cresident_ > cache_max_ && ctail_ != -1 && !(spare_current && cstamp_[ctail_] == cgen_))
                evict_tail();
            return;
        }
        // A cold batch put its own misses at the tail: step over them (its matmul needs them) to
        // the older entries behind, and stop at the batch's hot block like the plain sweep does.
        for (int32_t id = ctail_; id != -1 && cresident_ > cache_max_;) {
            const int32_t prev = cprev_[id];
            if (cstamp_[id] != cgen_)
                evict_entry(id);
            else if (cold_gen_[id] != cgen_)
                break;
            id = prev;
        }
        return;
    }
    // Decayed frequency: the lowest scores go first. A tie (entries not used since a rescale took
    // their scores to zero, a prefetched entry not yet hit) falls back to LRU order through the
    // rank. Only as many as the overshoot can need are ordered.
    std::vector<EvictCand> & cand = evict_cand_;
    cand.clear();
    int32_t rank = 0;
    for (int32_t id = ctail_; id != -1; id = cprev_[id], ++rank)
        if (!(spare_current && cstamp_[id] == cgen_)) cand.push_back({cfreq_[id], rank, id});
    if (cand.empty()) return;
    const auto colder = [](const EvictCand & a, const EvictCand & b) {
        return a.score < b.score || (a.score == b.score && a.rank < b.rank);
    };
    const size_t over = cresident_ - cache_max_;
    const size_t need = std::min(cand.size(), over / std::max<size_t>(1, min_entry_bytes_) + 1);
    std::partial_sort(cand.begin(), cand.begin() + (ptrdiff_t) need, cand.end(), colder);
    for (size_t i = 0; i < cand.size() && cresident_ > cache_max_; ++i) {
        if (i == need) // the estimate fell short (entries of unequal size): order the rest too
            std::sort(cand.begin() + (ptrdiff_t) i, cand.end(), colder);
        evict_entry(cand[i].id);
    }
}

void ExpertStreamSource::evict_entry(int32_t id) {
    ++evictions_;
    ever_evicted_[id] = 1; // the next read of this entry is the cache paying twice
    release_entry_pages(id);
    cvalid_[id] = 0;
    cspec_[id] = 0;
    cresident_ -= entry_bytes(id / n_expert_);
    lru_unlink(id);
}

// ── route-trace support: describe what a routing costs, without changing it ─────────
// Move already-resident predicted entries to the MRU end so eviction takes something else first.
// Deliberately NOT a cache hit: chits_ counts lookups the routing actually served, and a
// prediction is not a routing — inflating the hit rate from here would corrupt the one metric
// every cache decision in this project is argued from. Eval-thread only (LRU mutation).
void ExpertStreamSource::retain(int il, const int32_t * ids, int n_ids) {
    if (!active_ || cache_max_ == 0 || il < 0 || il >= n_layer_ || !layers_[il].bound || !ids) return;
    for (int i = 0; i < n_ids; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        const int32_t id = il * n_expert_ + e;
        if (!cvalid_[id]) continue; // absent: keeping it is prefetch's job, not retention's
        lru_unlink(id);
        lru_push_front(id);
        freq_touch(id);
    }
}

void ExpertStreamSource::settle_spec() {
    // Only the LRU path speculates, so mirror load_layer's own guard. Running the quiesce here
    // makes the one inside the load_layer that follows a cheap no-op: nothing left queued, and
    // nothing in flight. The cost is that its time lands outside the mgmt_ns_ window — which is
    // why a traced run is not a benchmark run.
    //
    // Under route-ahead the destructive settle would be a saboteur, not a bookkeeper: the hook
    // settles right before every issue, so the full quiesce here cancelled the future layers'
    // committed reads one callback before their load could adopt them. Integrate-only instead.
    if (!active_ || cache_max_ == 0) return;
    if (spec_adopt_) {
        // Cheap exit without touching the lanes' mutex: this runs before every route-ahead issue
        // (40 times a token) and usually has nothing to integrate. spec_done_pending_ is the
        // publisher's own count, so a relaxed read either sees work — and we take the lock — or
        // races a completion we will pick up at the next issue microseconds later.
        if (spec_done_pending_.load(std::memory_order_relaxed) != 0) spec_integrate_done();
    } else {
        quiesce_spec();
    }
}

// Integrate every completed speculative entry into the LRU cache, cancelling nothing — the
// route-ahead settle. Eval-thread only (LRU mutation), like the quiesce that subsumes it.
void ExpertStreamSource::spec_integrate_done() {
    std::vector<int32_t> & done = spec_done_scratch_;
    done.clear();
    {
        // BOTH eligibility guards run under io_mtx_, exactly as the destructive integrate runs
        // its own, and the remaining check is load-bearing: a completion can land in spec_done_
        // concurrently with (and just before) an issue that re-stages the same entry, and
        // integrating that stale completion would push an entry with LIVE queued jobs into the
        // LRU — the eviction then decommits pages a lane is about to write, which was this
        // feature's one segfault. The stale id is simply dropped; the new round's own completion
        // re-pushes it. After this snapshot the check cannot rot: re-staging happens only on
        // this same (eval) thread.
        std::lock_guard<std::mutex> lk(io_mtx_);
        for (int32_t id : spec_done_)
            if (!cvalid_[id] && spec_remaining_[id] == 0) done.push_back(id);
        spec_done_.clear();
        spec_done_pending_.store(0, std::memory_order_relaxed);
    }
    for (int32_t id : done) {
        cvalid_[id] = 1;
        cspec_[id] = 1;  // speculative until a real lookup hits it (then counted useful)
        cstamp_[id] = 0; // not used this generation → evictable if the budget is tight
        cresident_ += entry_bytes(id / n_expert_);
        disc_.used(id, cgen_);
        // Cold end for a GUESS (a mispredicted expert should be the first thing reclaimed) — but
        // under route-ahead there are no guesses: every integrated entry is a read some layer
        // COMMITTED to, at most N layers from being routed. Entering it cold handed it to the
        // very evictions of the intervening layers' loads, and on a device whose cache runs at
        // capacity that destroyed the entries before use and re-bought them on demand — a
        // perfect prefetch paying its bytes twice (measured: +34% read/token, drop off). A
        // committed read enters HOT; its own routing arrives to promote it within N layers
        // anyway.
        if (spec_adopt_) {
            lru_push_front(id);
            freq_touch(id); // committed: as hot as the use it was read for
        } else {
            lru_push_back(id);
        }
        spec_experts_.fetch_add(1);
    }
}

// Serial-mode helper for adoption: this thread owns lane 0, so it reads the loading layer's own
// adopted jobs — which sit at the front after the partition — and stops at the first job that
// belongs to a future layer, which stays queued for the worker lanes' idle time.
void ExpertStreamSource::drain_adopted(int adopt_il) {
    for (;;) {
        IoJob j;
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            if (spec_next_ >= spec_jobs_.size() || spec_jobs_[spec_next_].layer != (int16_t) adopt_il) return;
            j = spec_jobs_[spec_next_++];
            ++spec_inflight_;
        }
        const bool ok = read_slice(0, j);
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            if (ok) {
                spec_read_bytes_.fetch_add((long long) j.nbytes);
                if (spec_remaining_[j.flag] > 0 && --spec_remaining_[j.flag] == 0) {
                    spec_done_.push_back(j.flag);
                    spec_done_pending_.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                // A failed read forgets the entry: zeroing its counter keeps the adoption wait
                // from hanging, and !cvalid means the demand path simply re-reads it.
                spec_remaining_[j.flag] = 0;
            }
            if (--spec_inflight_ == 0) io_cv_done_.notify_all();
        }
    }
}

void ExpertStreamSource::query_residency(int il, const int32_t * ids, int n_ids, uint8_t * out) const {
    for (int i = 0; i < n_ids; ++i)
        out[i] = 0;
    if (!active_ || il < 0 || il >= n_layer_ || !layers_[il].bound || !ids) return;
    if (cache_max_ == 0) return; // shared-slot mode: nothing is kept, so every routing re-reads
    for (int i = 0; i < n_ids; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= n_expert_) continue;
        const int32_t id = il * n_expert_ + e;
        if (!cvalid_[id]) continue;  // miss: this routing pays a read
        out[i] = cspec_[id] ? 2 : 1; // resident; 2 = a speculative prefetch guessed it right
    }
}

uint64_t ExpertStreamSource::expert_bytes(int il) const {
    if (!active_ || il < 0 || il >= n_layer_ || !layers_[il].bound) return 0;
    return (uint64_t) entry_bytes(il);
}

// The cache accounting both load paths share. See the header for why this is the part that is shared
// and the job emission is not. Eval-thread only, like every other LRU mutation.
bool ExpertStreamSource::commit_proj_pages(int il, int e, int p) {
    if (!commit_slice(il, e, p)) {
        log_printf(LogLevel::Error, "commit failed");
        return false;
    }
    return true;
}

// Commit one projection slice of an entry so reads can fill it: its pages and the slack after it,
// then, with a discardable cache, the span only this entry owns made into something vm_offer can
// hand over later. The span shared with a neighbour stays ordinary memory: dropping it would take
// the neighbour's bytes too.
bool ExpertStreamSource::commit_slice(int il, int e, int p) {
    const uint64_t slice = layers_[il].proj[p].nb2;
    if (slice == 0) return true; // absent slot in a fused layout
    char * dst = (char *) lbuf_[p][il] + (uint64_t) e * slice;
    uintptr_t a0 = 0, a1 = 0;
    slice_pages(dst, slice, page_, a0, a1);
    if (!pio::vm_commit((void *) a0, (size_t) (a1 - a0))) return false;
    uintptr_t b0 = 0, b1 = 0;
    if (disc_.enabled() && entry_body(il * n_expert_ + e, p, b0, b1) && !disc_.prepare(Span{(void *) b0, b1 - b0}))
        return false;
    pin_pages((void *) a0, (size_t) (a1 - a0));
    return true;
}

// The pages of one projection slice that hold this entry's bytes and nobody else's: past the
// slack the left neighbour may read into, and short of the page the right neighbour starts in
// (release_entry_pages draws the same lines). Empty for a slice that does not fill a page of its own.
bool ExpertStreamSource::entry_body(int32_t id, int p, uintptr_t & b0, uintptr_t & b1) const {
    const int il = id / n_expert_, e = id % n_expert_;
    const uint64_t slice = layers_[il].proj[p].nb2;
    if (slice == 0) return false;
    const uintptr_t s = (uintptr_t) lbuf_[p][il] + (uint64_t) e * slice;
    b0 = (s + pio::tensor_tail_slack() + page_ - 1) & ~(uintptr_t) (page_ - 1);
    b1 = (s + slice) & ~(uintptr_t) (page_ - 1);
    return b1 > b0;
}

// Retire every dropped offer at the LRU tail, up to the first one still held. A dropped offer that
// nothing looks up again would otherwise stay in the accounting - resident bytes holding no RAM, and
// budget kept from entries that would - until the LRU reached it. Kernels reclaim offers oldest
// first, so a drop shows at the tail; the walk stops at the first held entry, so its cost is the
// drops it finds plus one sample. Correctness never rests on it: a lookup takes an entry back first.
void ExpertStreamSource::retire_dropped_tail() {
    while (cache_max_ != 0 && ctail_ != -1 && disc_.dropped(ctail_)) {
        const int32_t id = ctail_;
        const size_t bytes = entry_bytes(id / n_expert_);
        ++swept_;
        shrink_for_drop(bytes);
        ever_evicted_[id] = 1; // a later read of it is the cache paying twice
        release_entry_pages(id);
        cvalid_[id] = 0;
        cspec_[id] = 0;
        cresident_ -= bytes;
        lru_unlink(id);
    }
}

// The budget follows what the kernel lets the cache keep (DropFollowingBudget): down by each drop,
// never under one token's cycle, and back up to the budget last set once the kernel stops taking.
void ExpertStreamSource::shrink_for_drop(size_t bytes) {
    follow_.dropped(bytes, cgen_);
    cache_max_ = follow_.current();
}

// Before a load evicts to budget: room for what it staged, where the kernel has stopped dropping.
void ExpertStreamSource::regrow_for_load() {
    if (disc_.enabled() && cresident_ > cache_max_) cache_max_ = follow_.make_room(cresident_ - cache_max_, cgen_);
}

// One load's discard bookkeeping: offer what left the window, retire what the kernel took.
void ExpertStreamSource::settle_offers() {
    if (!disc_.enabled() || cache_max_ == 0) return;
    disc_.offer_cold(
        cgen_, [this](int32_t id) { return cvalid_[id] != 0; }, ctail_ != -1 ? disc_.last_use(ctail_) : cgen_);
    retire_dropped_tail();
}

// Whether a platform grants this is only learnt by asking, and there are two different refusals.
// One refused from the very first slice grants no lock at all: the attempts end and the cache is
// ordinary memory, as all of it was before pinning existed.
//
// One refused part way, with at least a token cycle already granted, has a total worth having, and
// it has run out. Giving up there is what was measured to
// collapse a run: every slice committed afterwards is ordinary memory, the cache keeps turning
// over, and within a few hundred tokens the locked slices have all been replaced by ones the
// machine is free to compress - 0.27 tok/s where the same cache, never over-asked, runs at 3.0. So
// the BUDGET comes down to what was granted and the pinning goes on: evictions release locked
// pages, and the next commit finds room again. Never under one token cycle, where a cache returns
// no hits; a refusal that arrives with the budget already at that floor ends the attempts like the
// first kind. Only the budget moves here - the eviction is the per-layer one that skips what the current
// token staged. Eval-thread only, like the commits it follows.
void ExpertStreamSource::pin_pages(void * p, size_t sz) {
    if (!pin_ || cache_max_ == 0) return; // no cache: a slice lives one token
    if (pio::vm_pin(p, sz)) {
        if (!pin_said_) log_printf(LogLevel::Info, "expert cache pinned in RAM");
        pin_said_ = true;
        return;
    }
    if (!pin_said_) {
        pin_ = false;
        log_printf(LogLevel::Warn, "cache pinning refused by the platform; the cache is ordinary memory");
        return;
    }
    const bool first = pin_refused_at_ == 0;
    // A total too small to hold one token cycle is the first kind of refusal arriving late: a
    // per-process limit of a few MiB grants the first slices and then nothing. Shrinking to it
    // would trade a working cache for a locked one that returns no hits, so the cache keeps its
    // budget and is ordinary memory, as it is where nothing is granted at all.
    if (first && (pin_floor_ == 0 || cresident_ < pin_floor_)) {
        pin_ = false;
        log_printf(LogLevel::Warn,
                   "cache pinning refused after %zu MiB, less than one token cycle; the cache is ordinary "
                   "memory",
                   cresident_ >> 20);
        return;
    }
    // Stepping the budget down is a trade the engine cannot price: it is right where what is not
    // locked gets compressed, and a plain loss where it would only have been left alone. So it is
    // done when asked (MoeStreamConfig::cache_pin_fit) and otherwise the budget is the caller's.
    if (!pin_fit_) {
        pin_ = false;
        pin_refused_at_ = std::max<size_t>(1, cresident_);
        log_printf(LogLevel::Warn,
                   "cache pinning refused after %zu MiB; the rest of the cache is ordinary memory "
                   "(--cache-pin-fit shrinks it to what was pinned instead)",
                   cresident_ >> 20);
        return;
    }
    if (first) pin_refused_at_ = std::max<size_t>(1, cresident_);
    // One step per load: the eviction that makes room runs after the reads, so several commits of
    // the same load can be refused before any of it is released, and they are one event.
    if (pin_shrunk_gen_ == cgen_ && !first) return;
    pin_shrunk_gen_ = cgen_;
    // Below what is resident, by the room a load needs: a layer's slices are committed before the
    // previous ones are evicted, so a budget equal to what was granted is over it by that much on
    // every load, and the refusal would simply repeat.
    const size_t slack = std::max({layer_demand_, sz, cache_max_ / 64});
    // From the smaller of the budget and what is resident: a load can run past the budget before
    // its eviction, and a refusal there says the budget is still too high, not that it was reached.
    const size_t from = std::min(cresident_, cache_max_);
    const size_t keep = std::max(from > slack ? from - slack : 0, std::max<size_t>(1, pin_floor_));
    if (keep < cache_max_) {
        if (first)
            log_printf(LogLevel::Warn,
                       "cache pinning refused after %zu MiB; the cache budget shrinks from %zu to %zu MiB",
                       cresident_ >> 20, cache_max_ >> 20, keep >> 20);
        cache_max_ = keep;
        ++cache_resizes_;
        return;
    }
    pin_ = false;
    log_printf(LogLevel::Warn, "cache pinning refused with the cache at %zu MiB; the rest of it is ordinary memory",
               cresident_ >> 20);
}

// Where a miss enters the LRU: a cold batch (a prompt) admits at the cold end, so one long prompt
// cannot flush what decode keeps reusing; everything else at the hot end. Either way it was just
// used, so it is not offered to the kernel before it has sat out the offer window.
void ExpertStreamSource::admit_entry(int32_t id) {
    if (cold_batch_) {
        lru_push_back(id);
        cold_gen_[id] = cgen_;
        ++cold_admits_;
    } else {
        lru_push_front(id);
    }
    disc_.used(id, cgen_);
}

bool ExpertStreamSource::touch_entry(int il, int e, bool & hit, bool promote, int commit_only_proj) {
    const int32_t id = il * n_expert_ + e;
    clookups_++;
    cstamp_[id] = cgen_;
    // An offer the kernel dropped under pressure is still an entry of the cache, resident in the
    // accounting, but its bytes are gone: a miss the caller reads whole again, so it is scored and
    // placed like one (a cold batch admits it at the cold end, as any other miss of that batch).
    const bool lost = cvalid_[id] && !disc_.take_back(id);
    if ((cvalid_[id] && !lost) || !cold_batch_) freq_touch(id);
    if (lost) {
        shrink_for_drop(entry_bytes(il));
        cspec_[id] = 0;
        if (promote) {
            lru_unlink(id);
            admit_entry(id);
        } else {
            disc_.used(id, cgen_);
        }
        hit = false;
        return true;
    }
    if (cvalid_[id]) {
        chits_++;
        if (cspec_[id]) { // this hit was served by a speculative prefetch — count it useful once
            spec_useful_.fetch_add(1);
            cspec_[id] = 0;
        }
        if (promote) {
            lru_unlink(id);
            lru_push_front(id);
        }
        disc_.used(id, cgen_);
        hit = true;
        return true;
    }
    // Miss: commit the pages the caller's reads will fill, then enter the expert as resident. The
    // entry is valid from here on because the caller is contracted to schedule those reads before
    // anything can consume the slices — the barrier that makes this safe is the eval-callback's.
    // (With commit_only_proj, the caller also contracts to commit the other projections before it
    // emits their jobs; the residency accounting below still covers the whole entry, because the
    // entry lives and is evicted as a unit either way.)
    for (int p = 0; p < ExpertLayout::max_slots; ++p) {
        if (commit_only_proj >= 0 && p != commit_only_proj) continue;
        if (!commit_proj_pages(il, e, p)) return false;
    }
    if (ever_evicted_[id]) ++rereads_; // this entry was resident once; the cache is buying it again
    cvalid_[id] = 1;
    cspec_[id] = 0; // a real read, not speculative
    cresident_ += entry_bytes(il);
    admit_entry(id);
    hit = false;
    return true;
}

// ── load: stage routed experts, read the batch, evict cold entries to budget ────────
bool ExpertStreamSource::load_layer(int il, const int32_t * ids, int n_ids) {
    if (!active_ || il < 0 || il >= n_layer_ || !layers_[il].bound || !ids || n_ids <= 0) return false;
    freq_pass_check(il);
    if (overlap_) return load_layer_async(il, ids, n_ids);
    // A load that failed is the end of the run, as on the overlap path: the entries it staged are
    // already counted resident (touch_entry) and their slices were never read, so a later token
    // would be served them as hits. The session's abort callback reads the flag and stops the graph
    // this layer's matmul belongs to; nothing downstream reads a slot nobody filled.
    if (fatal_.load(std::memory_order_acquire)) return false;
    const auto failed = [&] {
        fatal_.store(true, std::memory_order_release);
        return false;
    };
    LayerExperts & L = layers_[il];
    cgen_++;
    // A wide graph is a prefill; a decode or a speculative verify window is far narrower.
    cold_batch_ = prefill_cold_ && cache_max_ && top_k_ > 0 && n_ids >= top_k_ * cold_min_tokens_;
    jobs_.clear();

    // Cache-management work (vm commit + LRU bookkeeping, then eviction below) is timed into
    // mgmt_ns_ so the metrics can split it out of the compute residual instead of hiding it there.
    const auto tm0 = clock_t_::now();

    settle_offers();

    // Integrate / discard any speculative prefetch before this layer's real staging touches the
    // cache. After this returns no spec read is in flight and completed ones are resident hits. The
    // budget is fixed for the run (auto sizes it once at init), so there is nothing to resize here.
    if (cache_max_) quiesce_spec(spec_adopt_ ? il : -1);

    auto stage = [&](int e) -> bool {
        if (cache_max_ == 0) {
            for (int p = 0; p < ExpertLayout::max_slots; ++p) {
                const uint64_t slice = L.proj[p].nb2;
                if (slice == 0) continue; // absent slot in a fused layout
                jobs_.push_back({(char *) slot_[p] + (uint64_t) e * slice, L.proj[p].file_off + (uint64_t) e * slice,
                                 slice, -1, (int16_t) L.proj[p].file_idx, e, (int16_t) il, (int8_t) p, 0});
            }
            return true;
        }
        bool hit = false;
        if (!touch_entry(il, e, hit)) return false;
        if (hit) return true;
        // A miss: emit this expert's reads, expert-major. The drain below waits on them all, so the
        // order only has to be complete, not clever — unlike the overlap path, where it feeds a
        // kernel that is already blocking.
        for (int p = 0; p < ExpertLayout::max_slots; ++p) {
            const uint64_t slice = L.proj[p].nb2;
            if (slice == 0) continue; // absent slot in a fused layout
            jobs_.push_back({(char *) lbuf_[p][il] + (uint64_t) e * slice, L.proj[p].file_off + (uint64_t) e * slice,
                             slice, -1, (int16_t) L.proj[p].file_idx, e, (int16_t) il, (int8_t) p, 0});
        }
        return true;
    };

    std::fill(seen_.begin(), seen_.end(), (uint8_t) 0);
    int n_unique = 0;
    for (int i = 0; i < n_ids; ++i) {
        const int e = load_all_ ? (i < n_expert_ ? i : -1) : ids[i];
        if (e < 0 || e >= n_expert_) continue;
        if (seen_[e]) {
            // Already staged in this batch: still promote so the LRU order reflects the LAST
            // token that used this expert (ids arrive token-major), not its first touch — this
            // keeps the prompt tail's experts hot across prefill. Reads are scheduled only once
            // (the seen_ guard below), so this is bookkeeping only. The router's own top-k ids are
            // distinct, so in decode (n=1) this used to be unreachable — but cache-aware dropping
            // repoints a dropped slot's id at the routing's top expert, which makes duplicates the
            // normal case there. Promoting the same entry twice is idempotent, so it stays correct.
            if (cache_max_) {
                const int32_t id = il * n_expert_ + e;
                if (cold_gen_[id] != cgen_) {
                    lru_unlink(id);
                    lru_push_front(id);
                }
            }
            continue;
        }
        seen_[e] = 1;
        ++n_unique;
        if (!stage(e)) return failed();
    }
    if (load_all_) {
        for (int e = 0; e < n_expert_; ++e) {
            if (seen_[e]) continue;
            seen_[e] = 1;
            ++n_unique;
            if (!stage(e)) return failed();
        }
    }
    account_demand(il, n_unique);
    maybe_sample_dense();

    const size_t njobs = jobs_.size();
    mgmt_ns_.fetch_add((long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - tm0).count());
    const auto t0 = clock_t_::now();
    if (io_threads_ <= 1 || njobs <= 1) {
        for (size_t i = 0; i < njobs; ++i) {
            const IoJob & j = jobs_[i];
            if (!read_slice(0, j)) return failed();
        }
    } else {
        uint64_t my_gen;
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            batch_njobs_ = njobs;
            next_idx_ = 0;
            done_cnt_ = 0;
            io_err_.store(false);
            my_gen = ++batch_gen_;
        }
        io_cv_.notify_all();
        io_drain(0, my_gen);
        {
            std::unique_lock<std::mutex> lk(io_mtx_);
            io_cv_done_.wait(lk, [&] { return done_cnt_ == njobs || io_stop_; });
        }
        if (io_err_.load()) return failed();
    }
    read_ns_.fetch_add((long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - t0).count());

    if (cache_max_) {
        const auto te0 = clock_t_::now();
        regrow_for_load();
        evict_to_budget(/*spare_current=*/true);
        mgmt_ns_.fetch_add(
            (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - te0).count());
    }
    return true;
}

// ── overlap load: publish the routed reads, mark readiness, return without waiting ──────
//
// The bookkeeping is identical to the serial path but split in two: cache/LRU accounting is
// per-EXPERT (a hit resolves all of an expert's projections at once), while the jobs are
// emitted PROJECTION-MAJOR (all gate slices, then all up, then all down). mul_mat_id consumes
// the projections in that order, so projection-major reads feed the kernel roughly in the
// order it blocks on them, minimising stalls. Correctness does not depend on the order — the
// readiness flags gate each expert regardless — only latency does.
bool ExpertStreamSource::load_layer_async(int il, const int32_t * ids, int n_ids) {
    LayerExperts & L = layers_[il];

    // 1. Drain the previous batch fully before reusing jobs_/the flags. This is the eviction
    //    safety guarantee (no worker still reading an about-to-be-evicted page) and it covers
    //    load_all, whose surplus jobs no hook ever waits on. Serial's epilogue waited here too.
    {
        const auto tw0 = clock_t_::now();
        std::unique_lock<std::mutex> lk(io_mtx_);
        io_cv_done_.wait(lk, [&] { return done_cnt_ == batch_njobs_ || io_stop_; });
        drain_wait_ns_ +=
            (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - tw0).count();
    }
    if (fatal_.load(std::memory_order_acquire)) return false;

    // Cache-management work below (dedup/sort, vm commit, LRU bookkeeping and the eviction at
    // the end) runs on the eval thread and would otherwise hide inside the compute residual. Time
    // it into mgmt_ns_ so the metrics can surface it. The drain wait above is I/O, not mgmt.
    const auto tm0 = clock_t_::now();

    // Integrate / discard speculative prefetch before this layer's real staging (the previous
    // batch is already drained above, so this only waits on in-flight spec reads). The budget is
    // fixed for the run, so there is nothing to resize here.
    if (cache_max_) quiesce_spec(spec_adopt_ ? il : -1);

    // 2. New generation for this layer. A flag counts as ready only once its gen matches.
    const uint32_t gen = async_gen_.fetch_add(1, std::memory_order_relaxed) + 1;
    cur_il_.store(il, std::memory_order_relaxed);
    cgen_++;
    // A wide graph is a prefill; a decode or a speculative verify window is far narrower.
    cold_batch_ = prefill_cold_ && cache_max_ && top_k_ > 0 && n_ids >= top_k_ * cold_min_tokens_;
    jobs_.clear();
    settle_offers();

    auto mark_ready = [&](int p, int e) {
        ready_[(size_t) p * (size_t) n_expert_ + (size_t) e].gen.store(gen, std::memory_order_release);
    };

    // 3a. Dedup + sort ascending (matches the hook's ascending expert visitation order).
    std::fill(seen_.begin(), seen_.end(), (uint8_t) 0);
    staged_.clear();
    for (int i = 0; i < n_ids; ++i) {
        const int e = load_all_ ? (i < n_expert_ ? i : -1) : ids[i];
        if (e < 0 || e >= n_expert_ || seen_[e]) continue;
        seen_[e] = 1;
        staged_.push_back(e);
    }
    if (load_all_) {
        for (int e = 0; e < n_expert_; ++e) {
            if (seen_[e]) continue;
            seen_[e] = 1;
            staged_.push_back(e);
        }
    }
    std::sort(staged_.begin(), staged_.end());
    account_demand(il, (int) staged_.size());
    maybe_sample_dense();

    bool published = false; // a two-wave batch publishes inside the staging branch

    if (cache_max_ == 0) {
        // Cache off: every (projection, expert) is a fresh read into the shared full-size slot
        // at its canonical offset e*slice. Emit projection-major.
        for (int p = 0; p < ExpertLayout::max_slots; ++p) {
            const uint64_t slice = L.proj[p].nb2;
            if (slice == 0) continue; // absent slot in a fused layout
            for (int e : staged_) {
                const int32_t flag = (int32_t) ((size_t) p * (size_t) n_expert_ + (size_t) e);
                jobs_.push_back({(char *) slot_[p] + (uint64_t) e * slice, L.proj[p].file_off + (uint64_t) e * slice,
                                 slice, flag, (int16_t) L.proj[p].file_idx, e, (int16_t) il, (int8_t) p, 0});
            }
        }
    } else {
        // Cache on: per-expert LRU bookkeeping (hit → mark all projections ready now; miss →
        // commit the pages and remember it). Then emit the misses' jobs projection-major.
        //
        // Two-wave publish (#118): normally no lane can start reading until every miss took up
        // to three page-commit syscalls — bookkeeping sitting in front of the first byte of I/O,
        // on the latency-to-first-slice path. With two_wave_, only the FIRST present projection
        // (the one mul_mat_id blocks on first) is committed up front; its jobs publish
        // immediately, and the remaining projections are committed and appended while the lanes
        // already read. io_drain copies jobs under the lock and the worker wait predicate admits
        // a grown batch, so appending mid-generation is safe.
        const int p0 = [&] {
            for (int p = 0; p < ExpertLayout::max_slots; ++p)
                if (L.proj[p].nb2) return p;
            return -1;
        }();
        const bool two_wave = two_wave_ && !load_all_ && p0 >= 0;
        seen_.assign(seen_.size(), 0); // reuse as a per-staged miss marker keyed by expert
        for (int e : staged_) {
            bool hit = false;
            // promote=false: the token-major loop below re-orders every touched id anyway, so a
            // hit-path LRU move here was k wasted pointer operations per layer per token.
            if (!touch_entry(il, e, hit, /*promote=*/false, two_wave ? p0 : -1)) {
                // Nothing waits on a flag we will never publish: abort the graph instead.
                fatal_.store(true, std::memory_order_release);
                return false;
            }
            if (hit) { // resolved without a read — release every projection's waiter now
                for (int p = 0; p < ExpertLayout::max_slots; ++p)
                    if (L.proj[p].nb2) mark_ready(p, e);
                continue;
            }
            seen_[e] = 1; // this expert missed → its projections need reads
        }
        auto emit_proj = [&](int p) {
            const uint64_t slice = L.proj[p].nb2;
            if (slice == 0) return;
            for (int e : staged_) {
                if (!seen_[e]) continue; // cache hit, already marked ready
                const int32_t flag = (int32_t) ((size_t) p * (size_t) n_expert_ + (size_t) e);
                jobs_.push_back({(char *) lbuf_[p][il] + (uint64_t) e * slice,
                                 L.proj[p].file_off + (uint64_t) e * slice, slice, flag, (int16_t) L.proj[p].file_idx,
                                 e, (int16_t) il, (int8_t) p, 0});
            }
        };
        if (!two_wave) {
            for (int p = 0; p < ExpertLayout::max_slots; ++p)
                emit_proj(p);
        } else {
            // Wave one: the first projection's jobs, published before anything else is committed.
            emit_proj(p0);
            {
                std::lock_guard<std::mutex> lk(io_mtx_);
                batch_njobs_ = jobs_.size();
                next_idx_ = 0;
                done_cnt_ = 0;
                io_err_.store(false);
                batch_flag_gen_ = gen;
                ++batch_gen_;
            }
            io_cv_.notify_all();
            published = true;
            // Wave two: commit the remaining projections' pages, then append their jobs. A commit
            // failure here is after wave one published, so waiters may already be blocked on
            // flags this batch will now never flip — go fatal and wake them to observe it.
            for (int e : staged_) {
                if (!seen_[e]) continue;
                for (int p = 0; p < ExpertLayout::max_slots; ++p) {
                    if (p == p0) continue;
                    if (!commit_proj_pages(il, e, p)) {
                        fatal_.store(true, std::memory_order_release);
                        {
                            std::lock_guard<std::mutex> lk(ready_mtx_);
                            ready_cv_.notify_all();
                        }
                        return false;
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lk(io_mtx_);
                for (int p = 0; p < ExpertLayout::max_slots; ++p)
                    if (p != p0) emit_proj(p);
                batch_njobs_ = jobs_.size();
            }
            io_cv_.notify_all();
        }

        // Promote every touched expert in raw id order (token-major) so the LRU order reflects
        // the LAST token that used it, not the sorted/first-touch order staged_ imposes — this
        // keeps the prompt tail's experts hot across prefill. This loop is the ONLY promotion on
        // this path (touch_entry above runs with promote=false).
        // Skipped in load_all (everything is resident, so LRU order is meaningless).
        //
        // The cvalid guard is load-bearing, not defensive: an ADOPTED-PENDING expert (route-ahead)
        // was deliberately never staged, so it is not in the LRU — and lru_unlink on a node whose
        // prev/next are both -1 rewrites chead_/ctail_ to -1, orphaning every linked entry in one
        // call. That was this feature's second segfault: the next push_front made the pending
        // entry the LIST, eviction took it (decommitting pages its reads were still filling), and
        // the orphaned entries could never be evicted again. A pending expert needs no promotion:
        // it enters the LRU at integration, and its first real hit promotes it then.
        if (!load_all_) {
            for (int i = 0; i < n_ids; ++i) {
                const int e = ids[i];
                if (e < 0 || e >= n_expert_) continue;
                const int32_t id = il * n_expert_ + e;
                if (!cvalid_[id]) continue;
                if (cold_gen_[id] == cgen_) continue; // entered cold by this prefill batch
                lru_unlink(id);
                lru_push_front(id);
            }
        }
    }

    // 4. Publish the batch and return immediately — no drain. Workers fill the slices and
    //    flip the flags as they go; the compute threads block on those flags via the hook.
    //    (A two-wave batch already published both waves inside the staging branch.)
    if (!published) {
        {
            std::lock_guard<std::mutex> lk(io_mtx_);
            batch_njobs_ = jobs_.size();
            next_idx_ = 0;
            done_cnt_ = 0;
            io_err_.store(false);
            batch_flag_gen_ = gen;
            ++batch_gen_;
        }
        io_cv_.notify_all();
    }

    // 5. Evict cold entries to budget. Safe to run concurrently with this batch's reads: step 1
    //    guaranteed no stale in-flight jobs, and current-gen entries (cstamp_ == cgen_) — the
    //    ones just staged — are never chosen, so eviction only releases pages nobody is reading.
    if (cache_max_) {
        regrow_for_load();
        evict_to_budget(/*spare_current=*/true);
    }
    mgmt_ns_.fetch_add((long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clock_t_::now() - tm0).count());
    return true;
}

// Static trampoline for the process-global fork hook → member.
void ExpertStreamSource::c_expert_ready(const ggml_tensor * src0, int expert, void * user_data) {
    static_cast<ExpertStreamSource *>(user_data)->on_expert_ready(src0, expert);
}

// Called from EVERY compute thread, for each routed expert, before it reads that expert's rows.
// Blocks until the expert's slice for the layer in flight is resident (or the run goes fatal).
void ExpertStreamSource::on_expert_ready(const ggml_tensor * src0, int expert) {
    const void * key = (const void *) src0;
    auto it = std::lower_bound(texp_.begin(), texp_.end(), key,
                               [](const std::pair<const void *, uint32_t> & a, const void * k) { return a.first < k; });
    if (it == texp_.end() || it->first != key) return; // not one of our streamed expert tensors
    const uint32_t packed = it->second;
    const int il = (int) (packed >> 8);
    const int p = (int) (packed & 0xff);
    // Graph order guarantees exactly one layer's experts are staged at a time. A mismatch means
    // the kernel is asking for a layer we did not stage — fail fast rather than wait forever.
    if (il != cur_il_.load(std::memory_order_relaxed) || expert < 0 || expert >= n_expert_) {
        fatal_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(ready_mtx_);
            ready_cv_.notify_all();
        }
        return;
    }
    const size_t idx = (size_t) p * (size_t) n_expert_ + (size_t) expert;
    const uint32_t want = async_gen_.load(std::memory_order_relaxed);
    if (ready_[idx].gen.load(std::memory_order_acquire) == want) return; // already resident

    // The stall interval opens the moment the need is unmet — before the spin, since the spin is
    // already waiting — and closes on whichever exit this thread takes. Union accounting, not a
    // per-thread sum: see StallUnion.
    stall_union_.enter();
    // Short spin first: a slice usually lands within microseconds, cheaper than a syscall. The
    // beat is a pause instruction, not yield() — 2048 yields burnt up to a millisecond of
    // sched_yield churn per genuinely slow slice, stealing CPU from the I/O lanes and the
    // sibling compute threads that would have finished the slice sooner.
    for (int s = 0; s < 256; ++s) {
        if (ready_[idx].gen.load(std::memory_order_acquire) == want || fatal_.load(std::memory_order_acquire)) {
            stall_union_.exit();
            return;
        }
        pio::cpu_relax(); // a pause instruction, not the scheduler: see the loop's comment
    }
    // Register as a waiter BEFORE the last look at the flag. The publisher sets the flag before it
    // reads this count, and both sides are seq_cst, so one of the two must observe the other: either
    // the flag is already set here and this thread never sleeps, or the publisher sees a non-zero
    // count and notifies. That is what lets io_drain skip the mutex and the notify_all when no
    // compute thread is blocked, which is the usual case.
    ready_waiters_.fetch_add(1, std::memory_order_seq_cst);
    if (ready_[idx].gen.load(std::memory_order_seq_cst) != want && !fatal_.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lk(ready_mtx_);
        ready_cv_.wait(lk, [&] {
            return ready_[idx].gen.load(std::memory_order_acquire) == want || fatal_.load(std::memory_order_acquire);
        });
    }
    ready_waiters_.fetch_sub(1, std::memory_order_seq_cst);
    stall_union_.exit();
}

// The fork's `ggml_cpu_set_expert_ready_hook`, asked of the CPU backend's registry the way ggml
// exposes every other backend-specific function (threads, NUMA, the abort callback). Linking the
// symbol would bind the engine to a CPU backend linked in; through the registry it is reached in
// a build where that backend is a module loaded at start-up, and a build on stock upstream, whose
// CPU backend has no such entry, answers null here instead of failing to link.
namespace {
using set_expert_ready_hook_t = void (*)(void (*)(const ggml_tensor *, int, void *), void *);

set_expert_ready_hook_t resolve_expert_ready_hook() {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) return nullptr;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu);
    if (!reg) return nullptr;
    return (set_expert_ready_hook_t) ggml_backend_reg_get_proc_address(reg, "ggml_cpu_set_expert_ready_hook");
}
} // namespace

bool expert_ready_hook_available() {
    return resolve_expert_ready_hook() != nullptr;
}

bool ExpertStreamSource::enable_overlap_hook() {
    set_hook_ = resolve_expert_ready_hook();
    if (!set_hook_) return false;
    set_hook_(&ExpertStreamSource::c_expert_ready, this);
    hook_registered_ = true;
    return true;
}

IExpertSource::Stats ExpertStreamSource::stats() const {
    Stats s;
    long long rd_bytes = 0, rd_syscall_ns = 0;
    for (const auto & r : readers_) {
        rd_bytes += r->read_bytes();
        rd_syscall_ns += r->syscall_ns();
    }
    s.read_bytes = (uint64_t) rd_bytes;
    // Serial: read_ns_ is the wall time the caller was blocked in the read phase. Overlap: the
    // caller never blocks, so "I/O time" is instead the summed lane-busy time (the readers' syscall
    // ns), which the runtime reports as lane-busy per token rather than as a slice of wall time.
    s.read_seconds = (overlap_ ? rd_syscall_ns : read_ns_.load()) / 1e9;
    s.mgmt_seconds = mgmt_ns_.load() / 1e9;
    s.spec_read_bytes = (uint64_t) spec_read_bytes_.load();
    s.spec_experts = spec_experts_.load();
    s.spec_useful = spec_useful_.load();
    s.cache_hits = chits_;
    s.cache_lookups = clookups_;
    s.cache_resident_bytes = (uint64_t) cresident_;
    s.stall_seconds = stall_union_.total_ns() / 1e9;
    s.cache_budget_bytes = (uint64_t) cache_max_;
    s.cache_resizes = cache_resizes_;
    s.cache_pin_refused_bytes = (uint64_t) pin_refused_at_;
    s.discarded = disc_.lost_on_lookup() + swept_;
    s.discarded_on_hit = disc_.lost_on_lookup();
    s.dense_pin_refused_bytes = dense_.pin_refused_bytes();
    s.o_direct = effective_direct_;
    s.evictions = evictions_;
    s.rereads = rereads_;
    s.cold_admits = cold_admits_;
    s.drain_wait_seconds = drain_wait_ns_ / 1e9;
    s.adopt_wait_seconds = adopt_wait_ns_ / 1e9;
    s.dense_resident_frac = dense_.resident_frac();
    s.token_demand_bytes = (uint64_t) token_demand_;
    s.layer_demand_bytes = (uint64_t) layer_demand_;
    return s;
}

void ExpertStreamSource::shutdown() {
    if (!active_) return;
    // Unregister the process-global hook FIRST: after this no compute thread can enter
    // on_expert_ready and touch members we are about to tear down.
    if (hook_registered_ && set_hook_) {
        set_hook_(nullptr, nullptr);
        hook_registered_ = false;
    }
    // Wake any straggler blocked on a readiness flag so it observes fatal_ and unwinds.
    fatal_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(ready_mtx_);
        ready_cv_.notify_all();
    }
    {
        std::lock_guard<std::mutex> lk(io_mtx_);
        io_stop_ = true;
        ++spec_gen_; // disown any in-flight speculative reads
        spec_jobs_.clear();
        spec_next_ = 0;
    }
    io_cv_.notify_all();
    io_cv_done_.notify_all();
    for (auto & t : io_pool_)
        if (t.joinable()) t.join();
    io_pool_.clear();
    if (lane0_worker_.joinable()) lane0_worker_.join();
    spec_done_.clear();
    spec_done_pending_.store(0, std::memory_order_relaxed);
    spec_touched_.clear();

    for (int p = 0; p < ExpertLayout::max_slots; ++p)
        if (slot_[p]) {
            pio::aligned_free(slot_[p]);
            slot_[p] = nullptr;
        }
    for (int p = 0; p < ExpertLayout::max_slots; ++p) {
        for (int il = 0; il < (int) lbuf_[p].size(); ++il)
            if (lbuf_[p][il]) pio::vm_release(lbuf_[p][il], lbuf_sz_[p][il]);
        lbuf_[p].clear();
        lbuf_sz_[p].clear();
    }
    // The dense-weights module frees its own anon buffers here. Safe: the context (whose rebound
    // dense tensors point at them) is torn down after the source, and no decode is in flight — same
    // contract as the slot and LRU buffers freed above.
    dense_.shutdown();
    dense_tensors_.clear();
    row_tensors_.clear();
    readers_.clear(); // closes every shard's lane fds and frees the bounces
    jobs_.clear();
    layers_.clear();
    active_ = false;
}

} // namespace bmoe
