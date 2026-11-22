#include "row_stream.h"
#include "bmoe/log.h"

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace bmoe {

RowStream::~RowStream() {
    shutdown();
}

static uint64_t round_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

bool RowStream::init(std::vector<DenseTensorRef> tensors,
                     const std::vector<std::string> & paths,
                     uint64_t budget_bytes) {
    if (tensors.empty()) return true;
    page_ = pio::vm_page();
    slab_ = round_up(slab_min_bytes, page_);
    budget_bytes_ = budget_bytes ? budget_bytes : default_budget_bytes;
    // A budget below one slab would make every gather evict what it just read. Round it up rather
    // than reject the run: the caller asked for "as little as possible", not for a stall.
    budget_bytes_ = std::max<uint64_t>(budget_bytes_, slab_);

    // One single-lane reader per shard, sized for a slab window. Independent of both the expert
    // stream's readers and the dense loader's, for the same reason theirs are independent of each
    // other: the O_DIRECT decision belongs to the consumer, not to the file.
    for (const std::string & p : paths) {
        readers_.push_back(std::unique_ptr<FileReader>(new FileReader()));
        if (!readers_.back()->open(p, 1, /*direct=*/true, (size_t) slab_)) {
            log_printf(LogLevel::Warn, "row-stream could not open %s — tables keep the dense policy", p.c_str());
            readers_.clear();
            return false;
        }
    }

    tables_.reserve(tensors.size());
    for (const DenseTensorRef & d : tensors) {
        if (!d.tensor || d.size == 0) continue;
        if (d.file_idx < 0 || (size_t) d.file_idx >= readers_.size()) {
            log_printf(LogLevel::Error, "row-stream tensor points at shard %d of %zu", d.file_idx, readers_.size());
            release();
            return false;
        }
        Table t;
        t.ref = d;
        t.row_bytes = (uint64_t) d.tensor->nb[1];
        t.n_rows = (uint64_t) d.tensor->ne[1];
        if (t.row_bytes == 0 || t.n_rows == 0) continue;             // not a row-shaped weight; leave it alone
        t.span = round_up(d.size + pio::tensor_tail_slack(), page_); // readable past the last row
        t.base = (char *) pio::vm_reserve((size_t) t.span);
        if (!t.base) {
            log_printf(LogLevel::Error, "row-stream could not reserve %llu MiB for %s",
                       (unsigned long long) (t.span >> 20), d.tensor->name);
            release();
            return false;
        }
        t.n_slabs = round_up(d.size, slab_) / slab_;
        t.resident.assign((size_t) t.n_slabs, 0);
        t.stamp.assign((size_t) t.n_slabs, 0);
        t.spot.assign((size_t) t.n_slabs, lru_.end());
        t.orig_data = d.tensor->data;
        d.tensor->data = t.base; // the reservation mirrors the file layout, so nb/ne still address it
        table_bytes_ += d.size;
        tables_.push_back(std::move(t));
    }

    if (tables_.empty()) { // nothing qualified after the shape check
        readers_.clear();
        return true;
    }

    log_printf(LogLevel::Info, "row-stream — %llu MiB in %zu row-gathered table(s) served from flash, %llu MiB window",
               (unsigned long long) (table_bytes_ >> 20), tables_.size(), (unsigned long long) (budget_bytes_ >> 20));
    return true;
}

RowStream::Table * RowStream::find(const ggml_tensor * table) {
    for (Table & t : tables_)
        if (t.ref.tensor == table) return &t;
    return nullptr;
}

const RowStream::Table * RowStream::find(const ggml_tensor * table) const {
    for (const Table & t : tables_)
        if (t.ref.tensor == table) return &t;
    return nullptr;
}

bool RowStream::serves(const ggml_tensor * table) const {
    return find(table) != nullptr;
}

void RowStream::touch(Table & t, uint64_t s) {
    const size_t ti = (size_t) (&t - tables_.data());
    if (t.spot[(size_t) s] != lru_.end()) lru_.erase(t.spot[(size_t) s]);
    lru_.push_back({ti, s});
    t.spot[(size_t) s] = std::prev(lru_.end());
}

// Bytes a slab holds: the last one is short by whatever the table's size is not a multiple of.
static uint64_t slab_len(const uint64_t size, const uint64_t s, const uint64_t slab) {
    const uint64_t off = s * slab;
    return std::min<uint64_t>(slab, size - off);
}

// Bytes a slab is committed over: whole pages, since the slab is a multiple of the page and the
// reservation is page-aligned. The last slab also owns the slack after the table, which no other
// slab shares, so it is committed and released with it.
static uint64_t
slab_commit(const uint64_t size, const uint64_t span, const uint64_t s, const uint64_t slab, const uint64_t page) {
    const uint64_t off = s * slab;
    if (off + slab >= size) return span - off;
    return std::min<uint64_t>(round_up(slab_len(size, s, slab), page), span - off);
}

bool RowStream::fetch_slab(Table & t, uint64_t s) {
    if (t.resident[(size_t) s]) {
        touch(t, s);
        return true;
    }
    const uint64_t off = s * slab_;
    const uint64_t len = slab_len(t.ref.size, s, slab_);
    const uint64_t commit = slab_commit(t.ref.size, t.span, s, slab_, page_);
    if (!pio::vm_commit(t.base + off, (size_t) commit)) {
        log_printf(LogLevel::Error, "row-stream commit failed at slab %llu of %s", (unsigned long long) s,
                   t.ref.tensor->name);
        io_errors_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const long long got = readers_[(size_t) t.ref.file_idx]->read(0, t.base + off, t.ref.file_off + off, len);
    if (got < 0) {
        log_printf(LogLevel::Error, "row-stream read failed at slab %llu of %s", (unsigned long long) s,
                   t.ref.tensor->name);
        io_errors_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    t.resident[(size_t) s] = 1;
    touch(t, s);
    resident_bytes_.fetch_add(commit, std::memory_order_relaxed);
    slab_reads_.fetch_add(1, std::memory_order_relaxed);
    bytes_read_.fetch_add((uint64_t) got, std::memory_order_relaxed);
    return true;
}

// Hand back the oldest slabs until `headroom` bytes fit under the budget. Slabs stamped for the
// gather in flight are skipped: evicting a row the node about to run is going to read would be a
// correctness bug wearing a performance decision's clothes.
void RowStream::evict_to_budget(uint64_t headroom) {
    if (resident_bytes_.load(std::memory_order_relaxed) + headroom <= budget_bytes_) return;
    for (auto it = lru_.begin(); it != lru_.end();) {
        if (resident_bytes_.load(std::memory_order_relaxed) + headroom <= budget_bytes_) break;
        Table & t = tables_[it->first];
        const uint64_t s = it->second;
        if (t.stamp[(size_t) s] == gen_) { // needed by the gather in flight
            ++it;
            continue;
        }
        const uint64_t off = s * slab_;
        const uint64_t commit = slab_commit(t.ref.size, t.span, s, slab_, page_);
        pio::vm_evict(t.base + off, (size_t) commit);
        t.resident[(size_t) s] = 0;
        t.spot[(size_t) s] = lru_.end();
        resident_bytes_.fetch_sub(commit, std::memory_order_relaxed);
        evictions_.fetch_add(1, std::memory_order_relaxed);
        it = lru_.erase(it);
    }
}

bool RowStream::gather(const ggml_tensor * table, const int32_t * idx, int n) {
    Table * t = find(table);
    if (!t) return true; // not ours: the caller checked serves(), so this is only belt-and-braces
    gathers_.fetch_add(1, std::memory_order_relaxed);
    if (t->whole || n <= 0 || !idx) return true;
    rows_.fetch_add((uint64_t) n, std::memory_order_relaxed);

    // Which slabs the rows land in, deduplicated. A row can straddle a slab boundary, so a row
    // contributes the whole span it covers, not just the slab its first byte is in.
    ++gen_;
    if (gen_ == 0) { // wrapped: no stamp may claim to be from the current gather
        for (Table & tt : tables_)
            std::fill(tt.stamp.begin(), tt.stamp.end(), 0);
        gen_ = 1;
    }
    need_.clear();
    uint64_t missing = 0;
    for (int i = 0; i < n; ++i) {
        const int32_t r = idx[i];
        if (r < 0 || (uint64_t) r >= t->n_rows) continue; // the graph defines validity, not us
        const uint64_t a = (uint64_t) r * t->row_bytes;
        const uint64_t b = std::min<uint64_t>(a + t->row_bytes, t->ref.size);
        for (uint64_t s = a / slab_; s * slab_ < b; ++s) {
            if (s >= t->n_slabs) break;
            if (t->stamp[(size_t) s] == gen_) continue;
            t->stamp[(size_t) s] = gen_;
            need_.push_back(s);
            if (!t->resident[(size_t) s]) missing += slab_;
        }
    }
    if (need_.empty()) return true;

    // Make room first, then read: an eviction pass that ran afterwards could take back a slab this
    // very gather just paid for. A gather that alone exceeds the budget (a long prefill on a small
    // window) simply overshoots it for one node rather than thrashing — the ceiling is there to
    // bound the run, and no bound is worth a read per row.
    evict_to_budget(missing);
    std::sort(need_.begin(), need_.end()); // ascending file offsets: the drive's preferred order
    for (uint64_t s : need_)
        if (!fetch_slab(*t, s)) return false;
    return true;
}

bool RowStream::materialize(const ggml_tensor * table) {
    Table * t = find(table);
    if (!t) return true;
    if (t->whole) return true;
    log_printf(LogLevel::Warn,
               "row-stream — %s is read by a node that is not a row gather; pulling the whole "
               "%llu MiB table in",
               t->ref.tensor->name, (unsigned long long) (t->ref.size >> 20));
    for (uint64_t s = 0; s < t->n_slabs; ++s)
        if (!fetch_slab(*t, s)) return false;
    // A materialized table leaves the LRU: it is resident for the rest of the run, and an eviction
    // that took one of its slabs would put us right back at the fault this call exists to avoid.
    for (uint64_t s = 0; s < t->n_slabs; ++s) {
        if (t->spot[(size_t) s] != lru_.end()) {
            lru_.erase(t->spot[(size_t) s]);
            t->spot[(size_t) s] = lru_.end();
        }
    }
    t->whole = true;
    materialized_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

RowSourceStats RowStream::stats() const {
    RowSourceStats s;
    s.table_bytes = table_bytes_;
    s.resident_bytes = resident_bytes_.load(std::memory_order_relaxed);
    s.gathers = gathers_.load(std::memory_order_relaxed);
    s.rows = rows_.load(std::memory_order_relaxed);
    s.slab_reads = slab_reads_.load(std::memory_order_relaxed);
    s.bytes_read = bytes_read_.load(std::memory_order_relaxed);
    s.evictions = evictions_.load(std::memory_order_relaxed);
    s.materialized = materialized_.load(std::memory_order_relaxed);
    s.io_errors = io_errors_.load(std::memory_order_relaxed);
    return s;
}

void RowStream::release() {
    for (Table & t : tables_) {
        if (t.ref.tensor && t.orig_data) t.ref.tensor->data = t.orig_data; // back to the mmap
        if (t.base) pio::vm_release(t.base, (size_t) t.span);
        t.base = nullptr;
    }
    tables_.clear();
    lru_.clear();
    readers_.clear();
    resident_bytes_.store(0, std::memory_order_relaxed);
    table_bytes_ = 0;
}

bool RowStream::reopen_readers() {
    bool ok = true;
    for (auto & r : readers_)
        ok = r->reopen() && ok;
    return ok;
}

void RowStream::shutdown() {
    release();
}

} // namespace bmoe
