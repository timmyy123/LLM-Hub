#include "dense_weights.h"
#include "bmoe/log.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "row_stream.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace bmoe {

using clock_t_ = std::chrono::steady_clock;

DenseWeights::DenseWeights() = default;

DenseWeights::~DenseWeights() {
    shutdown();
}

std::vector<std::pair<uint64_t, uint64_t>>
DenseWeights::byte_ranges(std::vector<std::pair<uint64_t, uint64_t>> expert_ranges, uint64_t file_size) {
    std::sort(expert_ranges.begin(), expert_ranges.end());
    std::vector<std::pair<uint64_t, uint64_t>> dense;
    uint64_t pos = 0;
    for (const auto & r : expert_ranges) {
        if (r.first > pos) dense.push_back({pos, r.first}); // the gap before this expert range is dense
        pos = std::max(pos, r.second);
    }
    if (pos < file_size) dense.push_back({pos, file_size}); // the trailing dense tail (lm_head et al.)
    return dense;
}

bool DenseWeights::init(DenseWeightsMode mode,
                        const std::vector<std::string> & paths,
                        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> ranges,
                        std::vector<DenseTensorRef> tensors) {
    mode_ = mode;
    paths_ = paths;
    ranges_ = std::move(ranges);
    tensors_ = std::move(tensors);
    index_mapping();

    hold_back_oversized();
    take_row_gathered();

    if (mode_ == DenseWeightsMode::Anonymous || mode_ == DenseWeightsMode::Pinned) {
        if (tensors_.empty()) { // nothing (left) to rebind — behave as Mmap
            advise_random_mapped();
            return true;
        }
        if (mode_ == DenseWeightsMode::Pinned && pio::pinned_max_bytes() == 0) {
            log_printf(LogLevel::Error, "--dense-weights pinned needs reclaim-exempt memory, which this "
                                        "platform does not provide");
            return false;
        }
        // Single-lane readers (one per shard) with a bounce large enough for our chunk; O_DIRECT
        // independent of the expert stream. Sized to the largest tensor is unnecessary — we read in
        // bounded chunks.
        const size_t chunk = 8ull << 20;
        for (const std::string & p : paths_) {
            readers_.push_back(std::unique_ptr<FileReader>(new FileReader()));
            if (!readers_.back()->open(p, 1, /*direct=*/true, chunk)) return false;
        }
        if (!read_anonymous()) return false;
        // The tensors are copied and rebound; nothing reads through these again. Their fds and
        // per-lane bounce buffers would otherwise sit allocated for the whole session, next to
        // the expert cache that is counting every MiB.
        readers_.clear();
        drop_mmap_copies(pio::vm_page());
    } else if (mode_ == DenseWeightsMode::Warmed) {
        warm();
    }
    advise_random_mapped();
    return true;
}

// Remove [a, b) from a sorted list of disjoint [first, second) ranges, splitting the one it falls in.
static void subtract_range(std::vector<std::pair<uint64_t, uint64_t>> & ranges, uint64_t a, uint64_t b) {
    std::vector<std::pair<uint64_t, uint64_t>> out;
    out.reserve(ranges.size() + 1);
    for (const auto & r : ranges) {
        if (b <= r.first || a >= r.second) {
            out.push_back(r); // disjoint
            continue;
        }
        if (r.first < a) out.push_back({r.first, a});
        if (b < r.second) out.push_back({b, r.second});
    }
    ranges.swap(out);
}

// ── Oversized dense tensors: mmap'd under every mode ─────────────────────────────────
//
// A dense tensor bigger than the memory the kernel says is available cannot be resident under any
// mode: Anonymous/Pinned would ask for an allocation that fails or takes the process with it, and
// Warmed would sweep it through a page cache it cannot fit in, evicting itself as it goes. Until
// qwen4exp this could not happen — the largest dense tensor was an embedding or lm_head — but its
// n-gram table (`per_layer_token_embd`, ~28.8 GB at IQ4_NL) exceeds any phone's RAM while the graph
// gathers a few rows from it per token, so mmap is the only residency it can have. Such a tensor
// leaves the set entirely: not read, not rebound, not dropped, not warmed, and not sampled — a
// sensor that counted it would report a dense set that can never be resident. The rest of the
// dense weights still get the mode the run asked for.
//
// The bound is the kernel's own MemAvailable rather than a constant, because it is the one number
// that already accounts for what this device can still hand out; a tensor that FITS keeps its
// mode even when it is row-gathered, since demand-faulting a table that fits costs more in prefill
// than reading it once sequentially at load (measured: token_embd left mmap'd lost 16 % decode).
void DenseWeights::hold_back_oversized() {
    mapped_.clear();
    const uint64_t avail = pio::mem_available_bytes(); // 0 where unmeasured: nothing is oversized
    const auto named = [this](const DenseTensorRef & d) {
        if (!d.tensor) return false;
        for (const std::string & n : mapped_names_)
            if (n == d.tensor->name) return true;
        return false;
    };
    std::vector<DenseTensorRef> keep;
    keep.reserve(tensors_.size());
    uint64_t oversized = 0, by_plan = 0;
    size_t n_oversized = 0, n_by_plan = 0;
    for (const DenseTensorRef & d : tensors_) {
        const bool too_large = avail && d.size > avail;
        const bool by_name = named(d);
        if (!too_large && !by_name) {
            keep.push_back(d);
            continue;
        }
        mapped_.push_back(d);
        (too_large ? oversized : by_plan) += d.size;
        (too_large ? n_oversized : n_by_plan) += 1;
        if (d.file_idx >= 0 && (size_t) d.file_idx < ranges_.size())
            subtract_range(ranges_[(size_t) d.file_idx], d.file_off, d.file_off + d.size);
    }
    tensors_ = std::move(keep);
    if (oversized)
        log_printf(LogLevel::Warn,
                   "dense-weights=%s — %llu MiB in %zu tensor(s) exceeds the %llu MiB available: left "
                   "mmap'd with random-access advice, outside the warm sweep and the residency sensor",
                   dense_weights_mode_name(mode_), (unsigned long long) (oversized >> 20), n_oversized,
                   (unsigned long long) (avail >> 20));
    if (by_plan)
        log_printf(LogLevel::Info,
                   "dense-weights=%s — %llu MiB in %zu table(s) left mmap'd as the plan priced them: the graph "
                   "gathers rows from them, outside the warm sweep and the residency sensor",
                   dense_weights_mode_name(mode_), (unsigned long long) (by_plan >> 20), n_by_plan);
}

void DenseWeights::set_row_gathered(std::vector<DenseTensorRef> tables, uint64_t budget_bytes) {
    row_pending_ = std::move(tables);
    row_budget_ = budget_bytes;
}

bool DenseWeights::reopen_readers() {
    return rows_ ? rows_->reopen_readers() : true;
}

bool DenseWeights::file_mapping_in_use() const {
    if (!mapped_.empty()) return true;
    return (mode_ == DenseWeightsMode::Mmap || mode_ == DenseWeightsMode::Warmed) && !tensors_.empty();
}

IRowSource * DenseWeights::row_source() const {
    return rows_ && !rows_->empty() ? rows_.get() : nullptr;
}

RowSourceStats DenseWeights::row_stats() const {
    return rows_ ? rows_->stats() : RowSourceStats{};
}

// ── Row-gathered tables: bound to reserved space, served from flash ──────────────────
//
// Runs after hold_back_oversized, which is what restricts this to tables that COULD have been
// resident. That is deliberate rather than incidental: materialize() — the fallback for a graph
// shape the capture pass never saw — has to be able to pull the whole table in, and a table larger
// than memory could not honour it. An oversized row-gathered table keeps the mmap policy that was
// measured for it.
//
// A table only leaves the resident set once the takeover has actually succeeded, so a failed
// reservation or a reader that will not open costs nothing but the log line.
void DenseWeights::take_row_gathered() {
    if (row_pending_.empty()) return;
    // Intersect with what is still resident-eligible, by tensor identity: the caller's list was
    // built from the same capture, but hold_back_oversized may have taken some of it already.
    std::vector<DenseTensorRef> take;
    for (const DenseTensorRef & want : row_pending_)
        for (const DenseTensorRef & have : tensors_)
            if (have.tensor == want.tensor) {
                take.push_back(have);
                break;
            }
    row_pending_.clear();
    if (take.empty()) return;

    rows_.reset(new RowStream());
    if (!rows_->init(take, paths_, row_budget_) || rows_->empty()) {
        rows_.reset();
        return; // the tables keep the mode the run asked for
    }

    std::vector<DenseTensorRef> keep;
    keep.reserve(tensors_.size());
    uint64_t freed = 0;
    for (const DenseTensorRef & d : tensors_) {
        bool taken = false;
        for (const DenseTensorRef & t : take)
            if (t.tensor == d.tensor) {
                taken = true;
                break;
            }
        if (!taken) {
            keep.push_back(d);
            continue;
        }
        freed += d.size;
        if (d.file_idx >= 0 && (size_t) d.file_idx < ranges_.size())
            subtract_range(ranges_[(size_t) d.file_idx], d.file_off, d.file_off + d.size);
    }
    tensors_ = std::move(keep);
    log_printf(LogLevel::Info, "dense-weights=%s — %llu MiB of row-gathered table(s) left out of the resident set",
               dense_weights_mode_name(mode_), (unsigned long long) (freed >> 20));
}

void DenseWeights::advise_random_mapped() {
    if (mapped_.empty()) return;
    const size_t page = pio::vm_page();
    for (const DenseTensorRef & d : mapped_) {
        const char * a = addr_of(d.file_idx, d.file_off);
        if (!a) continue;
        // Inward to whole pages, as drop_mmap_copies does: the edge pages are shared with neighbours
        // that keep their own access pattern, and on a table this size two pages are nothing.
        uintptr_t a0 = ((uintptr_t) a + page - 1) & ~(uintptr_t) (page - 1);
        uintptr_t a1 = ((uintptr_t) a + d.size) & ~(uintptr_t) (page - 1);
        if (a1 > a0) pio::vm_advise_random((void *) a0, (size_t) (a1 - a0));
    }
}

// ── Anonymous / Pinned: read each dense tensor whole into our own buffer and rebind onto it ──
//
// The two modes differ ONLY in where the buffer comes from: ordinary anonymous memory, which the
// kernel may reclaim to zram, or a dma-buf it may not touch at all. Everything else — the O_DIRECT
// read, the rebind, handing back the now-unreferenced mmap pages — is identical, which is what makes
// the A/B between them a single-variable experiment rather than two code paths being compared.
//
// Pinned allocates PER TENSOR, so the 2047 MiB ceiling on a single dma-buf is not a constraint in
// practice: the largest dense tensor here is an embedding or lm_head, far below it. A tensor that
// exceeds it, or that arrives after the store has run out, takes an ordinary buffer - and the run
// says how many MiB did, on stderr and in pin_refused_bytes(), because a SILENT mix would make an
// A/B against anon meaningless in exactly the direction that flatters the feature.
bool DenseWeights::read_anonymous() {
    // Not const: the store can run out part way. A plan sizes the pinned set against a total that
    // is read once and then moves, so a refusal here is something that happens rather than a bug,
    // and a run that stopped loading for it would be worse than one a little slower. From the
    // first refusal on, the remaining tensors are ordinary buffers - the mode every other
    // platform already runs in - and the amount is kept for the telemetry.
    bool pinned = mode_ == DenseWeightsMode::Pinned;
    const uint64_t chunk = 8ull << 20;
    uint64_t total = 0;
    bufs_.reserve(tensors_.size());
    buf_sz_.reserve(tensors_.size());
    if (pinned) pinned_.reserve(tensors_.size());
    for (const DenseTensorRef & d : tensors_) {
        if (!d.tensor || d.size == 0) continue;
        if (d.file_idx < 0 || d.file_idx >= (int) readers_.size()) {
            log_printf(LogLevel::Error, "dense tensor points at shard %d of %zu", d.file_idx, readers_.size());
            return false;
        }
        void * buf = nullptr;
        pio::PinnedAlloc pa;
        const size_t held = (size_t) d.size + pio::tensor_tail_slack(); // readable past the last block
        // A tensor larger than one buffer may be was never going to fit and says nothing about
        // the store's total: it takes an ordinary buffer alone, and the tensors after it still pin.
        const bool too_big = pinned && held > pio::pinned_max_bytes();
        bool pin_this = pinned && !too_big;
        if (pin_this && !pio::pinned_alloc(held, &pa)) {
            log_printf(LogLevel::Warn,
                       "pinned dense buffer %llu MiB refused after %llu MiB (per-buffer ceiling %llu MiB); "
                       "the rest of the dense set is ordinary memory",
                       (unsigned long long) (d.size >> 20), (unsigned long long) (total >> 20),
                       (unsigned long long) (pio::pinned_max_bytes() >> 20));
            pinned = false;
            pin_this = false;
        }
        if (pin_this) {
            pinned_.push_back(pa); // tracked for shutdown even if a chunk read below fails
            buf = pa.base;
        } else {
            if (mode_ == DenseWeightsMode::Pinned) pin_refused_bytes_ += d.size;
            // Aligned for the reader that fills it, and to a page for the sensor and the drop that
            // later clip this buffer to pages.
            buf = pio::alloc_aligned(readers_[(size_t) d.file_idx]->buffer_align(), held);
            if (!buf) {
                log_printf(LogLevel::Error, "dense buffer alloc %llu failed", (unsigned long long) d.size);
                return false;
            }
            bufs_.push_back(buf); // tracked for shutdown even if a chunk read below fails
        }
        bases_.push_back(buf); // what the sensor probes, whichever allocator produced it
        buf_sz_.push_back((size_t) d.size);
        for (uint64_t done = 0; done < d.size;) {
            const uint64_t n = std::min<uint64_t>(chunk, d.size - done);
            if (readers_[(size_t) d.file_idx]->read(0, (char *) buf + done, d.file_off + done, n) < 0) return false;
            done += n;
        }
        d.tensor->data = buf; // rebind the model weight onto its private copy
        for (ggml_tensor * alias : d.aliases)
            alias->data = buf; // and every twin over the same bytes, onto the same copy
        total += d.size;
    }
    if (pin_refused_bytes_)
        log_printf(LogLevel::Info, "dense-weights=pinned — %llu MiB in %zu buffers, %llu MiB of it pinned",
                   (unsigned long long) (total >> 20), buf_sz_.size(),
                   (unsigned long long) ((total - pin_refused_bytes_) >> 20));
    else
        log_printf(LogLevel::Info, "dense-weights=%s — %llu MiB in %zu %s buffers",
                   dense_weights_mode_name(pinned ? DenseWeightsMode::Pinned : DenseWeightsMode::Anonymous),
                   (unsigned long long) (total >> 20), buf_sz_.size(), pinned ? "pinned" : "anon");
    return true;
}

// Every consumer that must turn (file_idx, offset) into an address asks here. It used to ask the
// process's own region list (/proc/self/maps, or a walk of the address space), which an iOS app is
// not given, so the sensor read -1 there and a dense set being refaulted every token went unseen.
// The tensors already know: before anything is rebound, a weight's data is where its bytes are
// mapped. A tensor in a buffer the host cannot address (a device's) has no such address.
void DenseWeights::index_mapping() {
    spans_.assign(paths_.size(), {});
    for (const DenseTensorRef & d : tensors_) {
        const ggml_tensor * t = d.tensor;
        if (!t || !t->data || !t->buffer || !ggml_backend_buffer_is_host(t->buffer) || d.size == 0) continue;
        if (d.file_idx < 0 || (size_t) d.file_idx >= spans_.size()) continue;
        spans_[(size_t) d.file_idx].push_back({d.file_off, d.size, (const char *) t->data});
    }
    for (auto & shard : spans_)
        std::sort(shard.begin(), shard.end(),
                  [](const MappedSpan & a, const MappedSpan & b) { return a.file_off < b.file_off; });
}

const char * DenseWeights::addr_of(int file_idx, uint64_t off) const {
    if (file_idx < 0 || (size_t) file_idx >= spans_.size()) return nullptr;
    const std::vector<MappedSpan> & shard = spans_[(size_t) file_idx];
    // The last span starting at or before `off`; the bytes between tensors map to nothing here.
    auto it = std::upper_bound(shard.begin(), shard.end(), off,
                               [](uint64_t o, const MappedSpan & sp) { return o < sp.file_off; });
    if (it == shard.begin()) return nullptr;
    --it;
    return off < it->file_off + it->size ? it->addr + (off - it->file_off) : nullptr;
}

// Hand the mmap copies back. The capture warm-up decode faulted these dense pages in mmap-resident,
// and read_anonymous has just copied them into anon buffers and rebound every tensor — so the file-
// backed pages are referenced by nobody. Left alone they sit resident until reclaim, doubling the
// dense footprint at the worst moment (right before prefill). Drop them with MADV_DONTNEED: a clean
// read-only mapping, so nothing is lost and nothing will refault the range. Best-effort: a tensor
// whose mapped address was not known (index_mapping) keeps its pages until reclaim.
void DenseWeights::drop_mmap_copies(size_t page) {
    uint64_t dropped = 0;
    for (const DenseTensorRef & d : tensors_) {
        if (!d.tensor || d.size == 0) continue;
        const char * a = addr_of(d.file_idx, d.file_off);
        if (!a) continue;
        // Align INWARD to whole pages (start up, end down), so a page shared with an adjacent tensor
        // that stays mmap-resident — an expert slice, or the next dense tensor — is never dropped.
        uintptr_t a0 = ((uintptr_t) a + page - 1) & ~(uintptr_t) (page - 1);
        uintptr_t a1 = ((uintptr_t) a + d.size) & ~(uintptr_t) (page - 1);
        if (a1 > a0) {
            pio::vm_drop_file_pages((void *) a0, (size_t) (a1 - a0));
            dropped += a1 - a0;
        }
    }
    if (dropped)
        log_printf(LogLevel::Info, "dense-weights=%s — dropped %llu MiB of now-unused mmap pages",
                   dense_weights_mode_name(mode_), (unsigned long long) (dropped >> 20));
}

// ── Warmed: one sequential buffered sweep over the dense ranges to populate the page cache ──
void DenseWeights::warm() {
    const size_t chunk = 8ull << 20;
    void * buf = pio::alloc_aligned(pio::vm_page(), chunk); // buffered reads: any alignment, a page is plenty
    if (!buf) return;
    const auto t0 = clock_t_::now();
    uint64_t warmed = 0;
    bool all_ok = true; // sticky, for the report only: a failed shard must not abort the others
    for (size_t s = 0; s < paths_.size() && s < ranges_.size(); ++s) {
        pio::fd_t fd = pio::open_read(paths_[s].c_str(), false);
        if (!pio::fd_ok(fd)) {
            all_ok = false; // best-effort: that shard's pages just stay cold
            continue;
        }
        bool shard_ok = true;
        for (const auto & r : ranges_[s]) {
            for (uint64_t a = r.first; a < r.second && shard_ok;) {
                const long long got = pio::pread_at(fd, buf, (size_t) std::min<uint64_t>(chunk, r.second - a), a);
                if (got <= 0) {
                    shard_ok = all_ok = false;
                    break;
                }
                a += (uint64_t) got;
                warmed += (uint64_t) got;
            }
        }
        pio::close_fd(fd);
    }
    const bool ok = all_ok;
    pio::aligned_free(buf);
    const double s = std::chrono::duration<double>(clock_t_::now() - t0).count();
    log_printf(LogLevel::Info, "dense warm-up — %llu MiB in %.1f s%s", (unsigned long long) (warmed >> 20), s,
               ok ? "" : " (partial)");
}

// ── residency sensor ─────────────────────────────────────────────────────────────────
void DenseWeights::sample_residency(size_t page) {
    if (mode_ == DenseWeightsMode::Anonymous || mode_ == DenseWeightsMode::Pinned)
        sample_anon(page);
    else
        sample_mmap(page);
}

// Anonymous/Pinned: mincore our own buffers directly. Anon memory is reclaimed to zram, and mincore
// reports resident anon pages just as it does file pages, so resident_frac keeps its meaning under
// the flag. Under Pinned the fraction is the EXPERIMENT rather than a diagnostic: dma-buf pages are
// supposed to be reclaim-exempt, so anything below 1.0 falsifies the premise. Should mincore not
// report on a dma-buf mapping at all, vm_resident_sample leaves the counters alone and the fraction
// stays -1 — unmeasured, which must not be read as "nothing is resident".
void DenseWeights::sample_anon(size_t page) {
    if (bases_.empty()) return;
    uint64_t total = 0;
    for (size_t sz : buf_sz_)
        total += sz;
    if (total == 0) return;
    size_t sampled = 0, resident = 0;
    for (int k = 0; k < sample_pages; ++k) {
        uint64_t target = (total * (uint64_t) k) / (uint64_t) sample_pages;
        for (size_t i = 0; i < bases_.size(); ++i) {
            if (target < buf_sz_[i]) {
                const char * a = (const char *) bases_[i] + target;
                const char * pg = (const char *) ((uintptr_t) a & ~(uintptr_t) (page - 1));
                pio::vm_resident_sample(pg, page, &sampled, &resident);
                break;
            }
            target -= buf_sz_[i];
        }
    }
    resident_frac_ = sampled ? (double) resident / (double) sampled : -1.0;
}

// Mmap/Warmed: the weights are llama.cpp's mmaps of the gguf shards; a dense (shard, offset) becomes
// an address through the tensor that holds it (index_mapping), on every platform alike.
// Probe sample_pages points spread evenly across the dense bytes of ALL shards, one page each, so the
// fraction keeps meaning "of the whole dense set" whatever the shard layout.
void DenseWeights::sample_mmap(size_t page) {
    if (ranges_.empty()) return;

    uint64_t total = 0;
    for (const auto & shard : ranges_)
        for (const auto & r : shard)
            total += r.second - r.first;
    if (total == 0) return;

    size_t sampled = 0, resident = 0;
    for (int k = 0; k < sample_pages; ++k) {
        uint64_t target = (total * (uint64_t) k) / (uint64_t) sample_pages;
        int fi = -1; // found shard; a dense offset of 0 is valid, so the sentinel is the index
        uint64_t off = 0;
        for (size_t s = 0; s < ranges_.size() && fi < 0; ++s) {
            for (const auto & r : ranges_[s]) {
                const uint64_t len = r.second - r.first;
                if (target < len) {
                    fi = (int) s;
                    off = r.first + target;
                    break;
                }
                target -= len;
            }
        }
        // Align the probe DOWN to its page: vm_resident_sample counts only pages fully inside the
        // range, so a single page handed in at an arbitrary offset would clip to nothing.
        if (const char * a = fi >= 0 ? addr_of(fi, off) : nullptr) {
            const char * pg = (const char *) ((uintptr_t) a & ~(uintptr_t) (page - 1));
            pio::vm_resident_sample(pg, page, &sampled, &resident);
        }
    }
    resident_frac_ = sampled ? (double) resident / (double) sampled : -1.0;
}

void DenseWeights::shutdown() {
    if (rows_) rows_->shutdown();
    rows_.reset();
    for (void * b : bufs_)
        if (b) pio::aligned_free(b);
    for (pio::PinnedAlloc & p : pinned_)
        pio::pinned_free(&p);
    bufs_.clear();
    pinned_.clear();
    bases_.clear();
    buf_sz_.clear();
    tensors_.clear();
    readers_.clear();
}

} // namespace bmoe
