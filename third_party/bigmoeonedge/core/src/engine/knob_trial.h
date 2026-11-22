#pragma once

// The A/B of one lossless knob, run on the first real request (docs/measured-planning.md, Step 2).
//
// A knob that changes no output and can change with the session open - the decode thread count,
// whether reads overlap compute - is not a question for a rule: the run can answer it by trying, on
// the user's own answer, while it is decoded. What makes the answer worth keeping is that the
// comparison is matched. A day of A/B on a phone was once invalidated by throttling, the baseline
// itself falling from 6.54 to 3.93 tok/s as the device heated (docs/expert-prediction.md). So the
// windows are interleaved, every candidate window sits between two windows of the incumbent, and a
// candidate is compared with the mean of the two around it: a drift that is linear over the three
// windows cancels exactly, and what is left of a slower one is shared by both arms of the round.
// The verdict is the median of those per-round ratios, so one window that met a stall elsewhere in
// the machine does not decide it, and a candidate wins only past the margin the device rule
// already uses: a gain inside the measurement's own repeatability is not one.
//
// The trial starts once the cache's hit rate has settled. The first tokens of a session fill an
// empty cache, and a window then is a window of cold reads, whatever knob it ran under.
//
// Pure policy: the session feeds it one decode step at a time - its wall seconds, the tokens it
// produced, the cache lookups and hits it made - and asks which arm the next step runs under. It
// switches nothing itself.

#include "bmoe/planner.h"

#include <cstdint>
#include <vector>

namespace bmoe {

// Not measured constants: a window long enough to hold a few tokens of a decode, few enough
// windows that the trial ends inside the first answer, and a settling test that only asks whether
// two windows in a row hit the cache alike.
struct KnobTrialPolicy {
    int window_tokens = 8;          // tokens a timed window holds (RunConfig::trial.window_tokens)
    int warm_tokens = 1;            // tokens after a switch that are not timed: the switch's own cost
    int rounds = 3;                 // timed windows per candidate, each between two incumbent windows
    double settle_hit_delta = 0.02; // two incumbent windows this close in hit rate: the cache has settled
    int settle_windows_max = 8;     // a cache that never settles is measured anyway after this many
    // How much faster a candidate must be than the incumbent before it replaces it, as a fraction:
    // the planner's own margin for "a device is faster than the host", for the same reason.
    double min_win = PlannerPolicy::defaults().min_backend_win;
};

// The seconds per token past which a step is apart from a reference whose steps ran between `lo`
// and `hi`: its slowest step, times its own range (slowest over fastest) or the margin, whichever is
// wider. The bound is what the reference measured: the margin on a steady machine, its spread on a
// noisy one. 0 while the reference has no step. KnobTrial judges a candidate against its incumbent
// by it.
double apart_bar(double lo, double hi, double min_win);

class KnobTrial {
public:
    // Arm 0 is the incumbent; arms 1..n_candidates the values tried against it. `settle`: wait for
    // the cache's hit rate to settle first (the first knob of a session; a later one starts at once).
    KnobTrial(int n_candidates, bool settle, KnobTrialPolicy pol = {});

    // The arm the next decode step must run under.
    int arm() const;
    bool done() const { return phase_ == Phase::Done; }
    bool settling() const { return phase_ == Phase::Settle; }

    // One decode step, run under arm(): its wall seconds, the tokens it produced, and the cache
    // lookups and hits it made (0 and 0 without a cache, which counts as settled).
    void record(double seconds, int tokens, uint64_t lookups, uint64_t hits);

    // The step stream was broken (the request ended): the window in progress is dropped, and the
    // next step is a warm-up again, since it follows a prefill rather than a token.
    void interrupt();

    // Once done(): the arm that won, 0 when the incumbent held.
    int winner() const { return winner_; }
    // How much faster arm `cand` (1..n) decoded than the incumbent around it, as a speed ratio (1.10
    // is 10% faster); 0 while it has no complete round.
    double gain(int cand) const;
    // Complete rounds measured for arm `cand`.
    int rounds(int cand) const;
    // Whether arm `cand` stopped being measured, for one of two reasons, neither with a constant of
    // its own; its windows still to run leave the schedule, and what it measured stays on record.
    //  - The rounds left could not make it a winner: the median of `rounds` ratios is past the margin
    //    once more than half of them are, so a candidate with that many losing rounds has its verdict.
    //    Measured on a phone: eight decode threads against four ran every window at ten times the
    //    seconds per token, and the third such round would only have repeated what two had settled.
    //  - One window ran apart from the incumbent: its fastest step was slower than the incumbent's
    //    slowest step in this trial by more than the incumbent's own range (slowest over fastest),
    //    and by more than the margin. The median is there for a stall elsewhere in the machine, and a
    //    stall slows some steps of a window, not every one; a gap wider than any swing the incumbent
    //    has shown is not one a swing can bridge. The bound is what the trial measured: the margin on
    //    a steady machine, the incumbent's spread on a noisy one. Measured on a phone: eight decode
    //    threads against four ran every token at about twenty times the incumbent's, and the second
    //    such window, required by the median alone, was another 18 s of stutter that decided nothing.
    //    The window need not run to its end for that: once its seconds exceed the bar times the
    //    window's tokens and more than half of its tokens are past the bar, its mean and its median
    //    are past it whatever the tokens left take, and it is cut there. Measured on a phone: the
    //    one window left still ran nine steps at 2.4-3.0 s each against a bar near 0.6 s.
    bool abandoned(int cand) const { return cand > 0 && cand <= n_cand_ && dead_[(size_t) cand]; }
    // The arm the last record() gave up, 0 when none: for the session's log line.
    int last_abandoned() const { return last_abandoned_; }
    // Whether it was given up on one window apart from the incumbent, rather than on the median.
    bool last_abandoned_apart() const { return last_apart_; }
    // Whether that window was cut before its end: the last window closed is then the candidate's,
    // and no incumbent window has closed after it yet.
    bool last_abandoned_cut() const { return last_cut_; }
    // The seconds per token a candidate's window must exceed to be apart, 0 before it is known.
    double apart_bar() const;

    // The windows in the order they are run, and their seconds per token so far: for a log line.
    const std::vector<int> & schedule() const { return schedule_; }
    const std::vector<double> & window_seconds_per_token() const { return spt_; }

private:
    enum class Phase { Settle, Measure, Done };
    void close_window();
    void abandon(int cand);
    bool apart(size_t window) const;
    void decide();
    std::vector<double> ratios(int cand) const;

    KnobTrialPolicy pol_;
    int n_cand_ = 0;
    Phase phase_ = Phase::Settle;
    std::vector<int> schedule_; // the arm of each timed window: 0 c1 0 c2 0 c1 0 ... 0
    std::vector<double> spt_;   // seconds per token of each window closed, in schedule order
    std::vector<double> lo_;    // its fastest step's seconds per token, in the same order
    int warm_left_ = 0;
    int tokens_ = 0;
    double seconds_ = 0.0;
    double step_lo_ = 0.0, step_hi_ = 0.0; // of the window in progress
    double inc_lo_ = 0.0, inc_hi_ = 0.0;   // of every incumbent step timed in this trial
    uint64_t lookups_ = 0, hits_ = 0;
    double last_hit_ = -1.0;
    int settle_windows_ = 0;
    int winner_ = 0;
    std::vector<char> dead_; // per arm, 1 once given up
    int last_abandoned_ = 0;
    bool last_apart_ = false;
    bool last_cut_ = false;
    int past_ = 0;     // tokens of the window in progress past the bar
    bool cut_ = false; // the window in progress is decided before its end
};

// The decode rate of the configuration a load runs, timed on the user's own answer as the trial
// times its incumbent: once the cache has settled, as many windows as one knob's trial times
// (2 * rounds + 1), and their median, so a window that met a stall elsewhere does not decide it. It
// is what a placement decided across loads is judged by (MeasuredKnobs::device_cache_tps and
// no_device_cache_tps): a load times the side it serves, and a plan made after it reads that side at
// its measurement. Pure policy, fed one decode step at a time like KnobTrial.
class RateProbe {
public:
    // `settle`: wait for the cache to settle first. `hits_visible`: false where the cache that warms
    // up is one whose lookups the session does not count (a device's expert cache, filled inside
    // llama.cpp). The wait is then the longest the trial makes, settle_windows_max windows, rather
    // than none: a window of a cold device cache is a window of uploads, whatever it is meant to time.
    RateProbe(bool settle, bool hits_visible, KnobTrialPolicy pol = {});

    void record(double seconds, int tokens, uint64_t lookups, uint64_t hits);
    void interrupt();

    bool settling() const { return phase_ == Phase::Settle; }
    bool done() const { return phase_ == Phase::Done; }
    // The median window's tokens per second once done(), 0 before.
    double tokens_per_second() const;
    // The cache's hit rate over the timed windows, -1 where they made no lookup.
    double hit_rate() const { return timed_lookups_ ? (double) timed_hits_ / (double) timed_lookups_ : -1.0; }
    int windows() const { return (int) spt_.size(); }
    int windows_wanted() const { return 2 * pol_.rounds + 1; }

private:
    enum class Phase { Settle, Measure, Done };
    void close_window();

    KnobTrialPolicy pol_;
    bool hits_visible_ = true;
    Phase phase_ = Phase::Settle;
    std::vector<double> spt_;
    int warm_left_ = 0;
    int tokens_ = 0;
    double seconds_ = 0.0;
    uint64_t lookups_ = 0, hits_ = 0;
    double last_hit_ = -1.0;
    int settle_windows_ = 0;
    uint64_t timed_lookups_ = 0, timed_hits_ = 0;
};

} // namespace bmoe
