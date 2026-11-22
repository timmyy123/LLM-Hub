#pragma once

// The expert cache's brake: when the system is taking the cache's own memory back while the run is
// using it, the cache gives half of it up instead of fighting for it.
//
// No figure a platform reports says how much a process can keep resident: a phone granted a 6 GB
// share compressed 2.5 GB of an engine holding 4.6 GB and ended it, and another phone compressed
// 2 GB of the same engine with 5 GB "available". So the question is answered while the run goes,
// from two things every machine shows a process about itself: how long its own work takes, and
// whether its own pages are still in RAM. A token whose compute stalls far past this machine's own
// reference says something went wrong; a sample of the cache's pages that comes back partly gone
// says it was memory and not the cores. Only both together move the budget.
//
// It only ever shrinks. The governor this engine once had grew the cache back whenever things were
// calm, and on a machine that could not concede the budget that turned a reclaim into a standing
// war (docs/pressure.md). A brake that holds what it gave up has no war to start: the budget a run
// loses is what the machine showed it could not keep, and the next run is planned afresh.
//
// Pure policy: the session feeds it the compute time of each decoded token and asks it what a
// braked budget is. The residency sample is the streamer's, the budget change the session's.

#include <cstddef>
#include <vector>

namespace bmoe {

// Not measured constants: a reference wide enough that one slow token does not move it, and a
// stall factor wide enough that ordinary variance is not one. A false stall costs one residency
// sample, never a budget change, because the sample has to agree.
struct CacheBrakePolicy {
    int warmup_tokens = 8;     // the first tokens fill the cache and are not this machine's steady state
    int window = 16;           // tokens the reference is the median of
    double stall_factor = 2.0; // a token this many times slower than the reference is a stall
};

class CacheBrake {
public:
    explicit CacheBrake(CacheBrakePolicy pol = {}) : pol_(pol) {}

    // One decoded token's compute time. True when it stalled against the reference this machine
    // set in its own earlier tokens; a stalled token does not enter the reference, so a run that
    // keeps stalling is still compared with what it did before it started to.
    bool stalled(double compute_s);

    // The reference a stall is measured against, 0 until the warm-up is over.
    double reference() const;

    // The budget after a brake: half, never under the floor (one token's worst-case cycle, under
    // which an LRU cache thrashes to no hits at all). Equal to `budget` when it is already there.
    static size_t braked(size_t budget, size_t floor);

private:
    CacheBrakePolicy pol_;
    int seen_ = 0;
    std::vector<double> recent_; // the last `window` steady tokens, oldest first
};

} // namespace bmoe
