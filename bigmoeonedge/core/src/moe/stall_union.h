#pragma once

#include <chrono>
#include <mutex>

namespace bmoe {

// The interval-union state machine: the cumulative wall time during which at least one thread was
// stalled, rather than the sum of per-thread stall times. enter/exit bracket each thread's stall;
// the clock starts on the 0→1 transition and the interval accumulates on the 1→0 transition, so
// overlapping waits count once and their union is what a wall-additive "the graph was blocked"
// term needs (summed thread stall divided by thread count is a mean, and understates whenever a
// minority of threads does the waiting — see docs/telemetry.md).
//
// Pure arithmetic over injected timestamps — no locking, no clock — which is what makes the union
// rules deterministically testable: overlap, nesting, separation, the 0→1/1→0 boundaries and the
// open-interval snapshot (tests/stall_union_test.cpp). Concurrency lives in the wrapper below.
class StallUnionState {
public:
    // A thread begins a stall at `now_ns`. The 0→1 transition opens the interval.
    void enter_at(long long now_ns) {
        if (stalled_++ == 0) open_ns_ = now_ns;
    }

    // A thread ends its stall. The 1→0 transition closes the interval and accumulates it.
    void exit_at(long long now_ns) {
        // An exit with nothing stalled cannot arise from a correct caller, but refusing it costs
        // nothing and keeps a stray exit from driving stalled_ to -1 — which would make the next
        // 0→1 look like a close and accumulate garbage.
        if (stalled_ == 0) return;
        if (--stalled_ == 0) {
            total_ns_ += now_ns - open_ns_;
            open_ns_ = 0;
        }
    }

    // Cumulative union through `now_ns`, including an interval still open at the snapshot: a
    // stats() read taken mid-stall must not silently lose the stall in progress, or a token-level
    // delta spanning it undercounts by the part already elapsed.
    long long total_at(long long now_ns) const { return total_ns_ + (stalled_ > 0 ? now_ns - open_ns_ : 0); }

    void reset() {
        stalled_ = 0;
        open_ns_ = 0;
        total_ns_ = 0;
    }

private:
    int stalled_ = 0;       // threads currently inside a stall (0 = no interval open)
    long long open_ns_ = 0; // when the open interval started; meaningful only while stalled_ > 0
    long long total_ns_ = 0;
};

// The production wrapper: mutex + monotonic clock around [StallUnionState]. Deliberately a mutex,
// not lock-free atomics — the obvious lock-free shape (atomic waiter count + open-interval
// timestamp) needs the 0→1 opener's timestamp store to be visible to whichever thread performs the
// 1→0 close; that store is sequenced AFTER the opener's counter RMW, so no acquire/release edge on
// the counter covers it, and a stale read silently drops the interval. A mutex makes the invariant
// local and auditable.
//
// The timestamp is taken INSIDE the critical section, on purpose: mutex acquisition order is the
// transition order, and a timestamp captured before acquiring can be older than a transition that
// acquired first — the last thread to leave could close the interval with a time earlier than a
// thread that already exited, undercounting the union (the same applies to a snapshot racing an
// open). Stall tracking uses this short critical section containing only the union-state
// transition and the timestamp capture; it does not hold the readiness mutex or perform I/O.
class StallUnion {
public:
    void enter() {
        const std::lock_guard<std::mutex> lk(mtx_);
        state_.enter_at(steady_ns());
    }

    void exit() {
        const std::lock_guard<std::mutex> lk(mtx_);
        state_.exit_at(steady_ns());
    }

    long long total_ns() const {
        const std::lock_guard<std::mutex> lk(mtx_);
        return state_.total_at(steady_ns());
    }

    void reset() {
        const std::lock_guard<std::mutex> lk(mtx_);
        state_.reset();
    }

private:
    static long long steady_ns() {
        return (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    mutable std::mutex mtx_;
    StallUnionState state_;
};

} // namespace bmoe
