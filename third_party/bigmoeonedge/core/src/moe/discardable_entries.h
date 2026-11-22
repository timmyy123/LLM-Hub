// Discardable cache entries: which cached entries are handed to the kernel to DROP under memory
// pressure, and what became of them.
//
// Memory the kernel may only reclaim by writing it somewhere (swap, or swapping everything else
// around memory that is locked) costs the drive a write. Memory it may drop costs nothing to reclaim
// and one read to restore, and a cache of bytes that already sit in the model file is exactly that.
// This class is the policy; it names no platform. The mechanism is a SpanOps port, implemented over
// pio's offer/reclaim calls (platform_span_ops) and by a fake in the tests. The owner of the cache
// keeps everything else - the entries, their LRU, the budget - and says which spans an entry owns.
//
// The rule is the cache's own and needs no tuning: an entry whose last use is the window's age is
// offered; a lookup takes an offered entry back before anything reads it, and an entry the kernel
// dropped is a miss. The owner picks offer_after, the narrowest the window may be: the oldest an
// entry can be while a kernel or an I/O lane may still be using it.
//
// The window follows the kernel. An offer costs a call, and a hit on an offered entry costs taking
// it back (one write per page where the kernel cancels a free that way): paid on every hit when the
// window is one token, it was 28 ms of a 239 ms token on a phone, and it buys nothing while the
// kernel takes nothing. So the window doubles each time it passes whole without a drop, and returns
// to offer_after at the first drop. It never covers more than the younger half of the ages the
// cache holds: the older half is always offered, so pressure always has something to take, and
// the drop that narrows the window can always happen.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <utility>
#include <vector>

namespace bmoe {

// One page-aligned span of memory that no other entry shares a page with.
struct Span {
    void * p = nullptr;
    size_t size = 0;
};

// The mechanism, per span. See pio::vm_offer for what each platform does underneath.
class SpanOps {
public:
    virtual ~SpanOps() = default;
    virtual bool supported() const = 0;
    // Make a just-committed span offerable later (on some platforms it must become its own object).
    virtual bool prepare(Span s) = 0;
    virtual bool offer(Span s) = 0;
    // Before anything reads an offered span: true when every byte is still what was offered.
    virtual bool reclaim(Span s) = 0;
    // Whether an offered span is still held: 1 held, 0 dropped, -1 cannot tell on this platform.
    virtual int held(Span s) = 0;
    // Drop a span as pressure would (the gates' stand-in for pressure).
    virtual void drop(Span s) = 0;
};

// pio's implementation of the port.
SpanOps & platform_span_ops();

class DiscardableEntries {
public:
    // Calls `fn` for every span entry `id` owns. Called on the owner's thread only.
    using SpansOf = std::function<void(int32_t id, const std::function<void(Span)> & fn)>;

    // `drop_before_reclaim` (debug): drop an offered entry just before a lookup takes it back, so
    // every such lookup finds it gone and the owner's re-read path runs on zeroed memory.
    void init(size_t n_entry, uint32_t offer_after, bool drop_before_reclaim, SpanOps * ops, SpansOf spans);
    bool enabled() const { return ops_ != nullptr; }

    bool prepare(Span s) { return !ops_ || ops_->prepare(s); }

    // The entry was used (looked up, filled, or prefetched) in generation `gen`.
    void used(int32_t id, uint32_t gen);

    // Offer every entry whose last use is the window's age behind `gen`. `valid(id)` says whether
    // the owner still holds the entry; one it no longer does is skipped. `oldest_use` is the last use
    // of the oldest entry the owner holds (its LRU tail), which bounds how wide the window may grow;
    // left at `gen`, the window stays at offer_after.
    void offer_cold(uint32_t gen, const std::function<bool(int32_t)> & valid, uint32_t oldest_use);
    void offer_cold(uint32_t gen, const std::function<bool(int32_t)> & valid) { offer_cold(gen, valid, gen); }
    // The window in force, in generations.
    uint32_t window() const { return after_; }
    // Pressure the owner measured on what it still holds (the brake): the window narrows to
    // offer_after at once, as at a drop, since what the window kept out of the kernel's reach is what
    // the system just compressed or swapped.
    void pressure() {
        if (enabled()) kernel_dropped();
    }
    // An entry's last use, in generations.
    uint32_t last_use(int32_t id) const { return enabled() ? last_use_[id] : 0; }

    bool offered(int32_t id) const { return enabled() && offered_[id] != 0; }

    // Take an offered entry back. False when the kernel dropped any of it: the entry's spans are
    // ordinary memory again and the owner must fill them before anything reads them.
    bool take_back(int32_t id);

    // Whether an offered entry is measurably gone (false when held, or when the platform cannot tell).
    bool dropped(int32_t id);

    // The owner released the entry's memory: whatever was offered there is gone with it.
    void forget(int32_t id) {
        if (enabled()) offered_[id] = 0;
    }

    long long lost_on_lookup() const { return lost_on_lookup_; }

private:
    SpanOps * ops_ = nullptr;
    SpansOf spans_;
    uint32_t offer_after_ = 0;
    uint32_t after_ = 0;   // the window in force: offer_after_ up to half the cache's age span
    uint32_t changed_ = 0; // the generation the window last moved, or the kernel last dropped
    bool restart_ = true;  // a drop (or init) since the last offer_cold: changed_ restarts there
    void kernel_dropped();
    bool drop_before_reclaim_ = false;
    std::vector<uint8_t> offered_;
    std::vector<uint32_t> last_use_;
    // Every use in order, (id, generation): an entry is offered when its LAST use leaves the window,
    // and an older pair for an entry used since is stale and skipped. Amortized O(1) per use.
    std::deque<std::pair<int32_t, uint32_t>> uses_;
    long long lost_on_lookup_ = 0;
};

// The budget of a cache the kernel may drop from. Down by what the kernel takes, so a cache larger
// than the machine lets it keep stops refilling only to be dropped again; back up - never past the
// budget its owner set - once a whole window passes without a drop, and only as fast as misses need
// room. Probing upward is safe here as it is not for ordinary memory: growing into pressure costs
// the kernel a drop and the cache a re-read, never a write.
class DropFollowingBudget {
public:
    // The owner set the budget (at load, or an app's explicit resize): the new ceiling, and where
    // the budget starts. `window` is how many generations without a drop count as the kernel having
    // stopped taking.
    void reset(size_t budget, size_t floor, uint32_t window);
    void set_floor(size_t floor);
    size_t current() const { return current_; }
    size_t ceiling() const { return ceiling_; }

    // The kernel dropped `bytes` in generation `gen`.
    void dropped(size_t bytes, uint32_t gen);

    // The owner measured pressure the kernel's drops do not show (the brake: a stalled token, and
    // pages of the entries still held gone from RAM). The budget comes down to `budget` at once; the
    // ceiling stays, so the cache can come back once the system settles - a brake that set a new
    // ceiling kept a 16 GB laptop at half its cache for the rest of a session. Each brake doubles
    // the quiet stretch growth waits for, so a cache braked again on the way up gives the system
    // ever longer, and never oscillates at a fixed period.
    void braked(size_t budget, uint32_t gen);

    // A neighbour borrows `bytes` for a while (a device prefill's slots) and gives them back. Not
    // pressure: neither moves the ceiling nor delays growth, and nothing grows into the loan.
    void lend(size_t bytes);
    void end_loan();

    // A load needs `need` bytes beyond the budget in generation `gen`: the budget grows by up to
    // that when the kernel has been quiet for a window, and not past the ceiling.
    size_t make_room(size_t need, uint32_t gen);

private:
    size_t ceiling_ = 0, floor_ = 1, current_ = 0, lent_ = 0;
    uint32_t window_ = 0, last_drop_ = 0, brakes_ = 0;
    bool dropped_ever_ = false;
};

} // namespace bmoe
