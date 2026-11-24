#include "knob_trial.h"

#include <algorithm>
#include <cmath>

namespace bmoe {

KnobTrial::KnobTrial(int n_candidates, bool settle, KnobTrialPolicy pol) : pol_(pol), n_cand_(n_candidates) {
    pol_.window_tokens = std::max(1, pol_.window_tokens);
    pol_.warm_tokens = std::max(0, pol_.warm_tokens);
    pol_.rounds = std::max(1, pol_.rounds);
    dead_.assign((size_t) std::max(0, n_candidates) + 1, 0);
    // Every candidate window is bracketed by incumbent ones, so the schedule starts and ends on the
    // incumbent and alternates in between: 0 c1 0 c2 0 c1 0 c2 0 for two candidates over two rounds.
    schedule_.push_back(0);
    for (int r = 0; r < pol_.rounds; ++r)
        for (int c = 1; c <= n_cand_; ++c) {
            schedule_.push_back(c);
            schedule_.push_back(0);
        }
    if (n_cand_ <= 0)
        phase_ = Phase::Done;
    else if (!settle)
        phase_ = Phase::Measure;
    // The first step of a trial follows whatever ran before it, a prefill or another knob's switch.
    warm_left_ = pol_.warm_tokens;
}

int KnobTrial::arm() const {
    if (phase_ != Phase::Measure) return phase_ == Phase::Done ? winner_ : 0;
    return schedule_[spt_.size()];
}

void KnobTrial::interrupt() {
    tokens_ = 0;
    past_ = 0;
    seconds_ = 0.0;
    step_lo_ = step_hi_ = 0.0;
    lookups_ = hits_ = 0;
    warm_left_ = pol_.warm_tokens;
}

void KnobTrial::record(double seconds, int tokens, uint64_t lookups, uint64_t hits) {
    last_abandoned_ = 0;
    last_apart_ = false;
    last_cut_ = false;
    if (phase_ == Phase::Done || tokens <= 0 || !(seconds > 0.0)) return;
    if (warm_left_ > 0) {
        warm_left_ -= tokens; // a step is timed whole or not at all
        return;
    }
    const double step = seconds / (double) tokens;
    step_lo_ = tokens_ == 0 ? step : std::min(step_lo_, step);
    step_hi_ = tokens_ == 0 ? step : std::max(step_hi_, step);
    tokens_ += tokens;
    seconds_ += seconds;
    lookups_ += lookups;
    hits_ += hits;
    // A candidate window decided before its end. Two things, both already true whatever its
    // remaining tokens take: its seconds exceed what the whole window may spend and still stand
    // inside the bar, so its mean is past it (no token takes less than nothing); and more than half
    // of its tokens are past the bar, so its median token is too. The second is what keeps a stall
    // out: a stall elsewhere in the machine slows some tokens of a window, not most of them, and the
    // median of the rounds is there for it.
    const double bar = apart_bar();
    if (arm() > 0 && bar > 0.0 && seconds / (double) tokens > bar) past_ += tokens;
    cut_ = arm() > 0 && tokens_ < pol_.window_tokens && bar > 0.0 && seconds_ > bar * (double) pol_.window_tokens &&
           2 * past_ > pol_.window_tokens;
    if (tokens_ >= pol_.window_tokens || cut_) close_window();
}

void KnobTrial::close_window() {
    const double spt = seconds_ / (double) tokens_;
    const double hit = lookups_ ? (double) hits_ / (double) lookups_ : -1.0;
    const double lo = step_lo_, hi = step_hi_;
    const int prev_arm = arm();
    tokens_ = 0;
    past_ = 0;
    seconds_ = 0.0;
    step_lo_ = step_hi_ = 0.0;
    lookups_ = hits_ = 0;

    if (phase_ == Phase::Settle) {
        ++settle_windows_;
        // Without a cache there is nothing to settle; otherwise two windows in a row that hit alike.
        const bool settled = hit < 0.0 || (last_hit_ >= 0.0 && std::fabs(hit - last_hit_) <= pol_.settle_hit_delta) ||
                             settle_windows_ >= pol_.settle_windows_max;
        last_hit_ = hit;
        if (settled) phase_ = Phase::Measure; // still the incumbent: no switch, no warm-up
        return;
    }

    spt_.push_back(spt);
    lo_.push_back(lo);
    if (cut_) {
        // Decided before its end (record()): the candidate is settled now, its incumbent window
        // still runs, and the round's ratio is completed by it like any other.
        cut_ = false;
        abandon(prev_arm);
        last_apart_ = true;
        last_cut_ = true;
        if (arm() != prev_arm) warm_left_ = pol_.warm_tokens;
        return;
    }
    // The incumbent's range is taken over every step it ran in this trial, settling left out: the
    // windows of a cache still filling say nothing about the noise of a settled one.
    if (prev_arm == 0) {
        const bool first = !(inc_hi_ > 0.0);
        inc_lo_ = first ? lo : std::min(inc_lo_, lo);
        inc_hi_ = first ? hi : std::max(inc_hi_, hi);
    }
    // An incumbent window closing completes the ratio of the candidate window before it. A candidate
    // whose window ran apart from the incumbent, or whose losing rounds already outnumber what the
    // median can overturn, is settled: the rounds left leave the schedule, and what was measured
    // stays in the record.
    if (schedule_[spt_.size() - 1] == 0) {
        if (spt_.size() >= 3) {
            const size_t k = spt_.size() - 2;
            const int c = schedule_[k];
            if (c > 0 && !dead_[(size_t) c] && apart(k)) {
                abandon(c);
                last_apart_ = true;
            }
        }
        const double losing = 1.0 / (1.0 + pol_.min_win); // a ratio at or past this is not a win
        const int overturnable = (pol_.rounds - 1) / 2;   // losing rounds a median of `rounds` can still absorb
        for (int c = 1; c <= n_cand_; ++c) {
            if (dead_[(size_t) c]) continue;
            int lost = 0;
            for (const double r : ratios(c))
                lost += r >= losing;
            if (lost > overturnable) abandon(c);
        }
    }
    if (spt_.size() == schedule_.size()) {
        decide();
        return;
    }
    if (arm() != prev_arm) warm_left_ = pol_.warm_tokens;
}

void KnobTrial::abandon(int cand) {
    dead_[(size_t) cand] = 1;
    last_abandoned_ = cand;
    last_apart_ = false; // the caller says so when it was
    // The windows still to run, without this arm's, and without an incumbent window that would then
    // follow another.
    const size_t k = spt_.size() - 1;
    std::vector<int> rest;
    for (size_t j = k + 1; j < schedule_.size(); ++j) {
        if (schedule_[j] == cand) continue;
        // An incumbent window after another incumbent one (the one just closed included) brackets nothing.
        if (schedule_[j] == 0 && (rest.empty() ? schedule_[k] == 0 : rest.back() == 0)) continue;
        rest.push_back(schedule_[j]);
    }
    schedule_.resize(k + 1);
    schedule_.insert(schedule_.end(), rest.begin(), rest.end());
}

// Candidate window k against every incumbent step so far, the window just closed after it included,
// so a drift that slows the incumbent as well raises the bar with it.
bool KnobTrial::apart(size_t k) const {
    const double bar = apart_bar();
    return bar > 0.0 && lo_[k] > bar;
}

// The seconds per token past which a candidate is apart from the incumbent: its slowest step in
// this trial, times its own range (slowest over fastest) or the margin, whichever is wider. 0 before
// an incumbent window has been timed.
double KnobTrial::apart_bar() const {
    return bmoe::apart_bar(inc_lo_, inc_hi_, pol_.min_win);
}

double apart_bar(double lo, double hi, double min_win) {
    if (!(lo > 0.0) || !(hi > 0.0)) return 0.0;
    return hi * std::max(hi / lo, 1.0 + min_win);
}

std::vector<double> KnobTrial::ratios(int cand) const {
    std::vector<double> r;
    // A candidate window counts once the incumbent window after it has closed too.
    for (size_t k = 1; k + 1 < spt_.size(); ++k) {
        if (schedule_[k] != cand) continue;
        const double around = 0.5 * (spt_[k - 1] + spt_[k + 1]);
        if (around > 0.0) r.push_back(spt_[k] / around);
    }
    return r;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double KnobTrial::gain(int cand) const {
    const double m = median(ratios(cand));
    return m > 0.0 ? 1.0 / m : 0.0;
}

int KnobTrial::rounds(int cand) const {
    return (int) ratios(cand).size();
}

void KnobTrial::decide() {
    phase_ = Phase::Done;
    winner_ = 0;
    double best = 1.0 + pol_.min_win;
    for (int c = 1; c <= n_cand_; ++c) {
        const double g = gain(c);
        if (g >= best) {
            best = g;
            winner_ = c;
        }
    }
}

RateProbe::RateProbe(bool settle, bool hits_visible, KnobTrialPolicy pol) : pol_(pol), hits_visible_(hits_visible) {
    pol_.window_tokens = std::max(1, pol_.window_tokens);
    pol_.warm_tokens = std::max(0, pol_.warm_tokens);
    pol_.rounds = std::max(1, pol_.rounds);
    pol_.settle_windows_max = std::max(1, pol_.settle_windows_max);
    if (!settle) phase_ = Phase::Measure;
    warm_left_ = pol_.warm_tokens;
}

double RateProbe::tokens_per_second() const {
    if (phase_ != Phase::Done || spt_.empty()) return 0.0;
    std::vector<double> v = spt_;
    std::nth_element(v.begin(), v.begin() + (long) (v.size() / 2), v.end());
    const double spt = v[v.size() / 2];
    return spt > 0.0 ? 1.0 / spt : 0.0;
}

void RateProbe::interrupt() {
    tokens_ = 0;
    seconds_ = 0.0;
    lookups_ = hits_ = 0;
    warm_left_ = pol_.warm_tokens;
}

void RateProbe::record(double seconds, int tokens, uint64_t lookups, uint64_t hits) {
    if (phase_ == Phase::Done || tokens <= 0 || !(seconds > 0.0)) return;
    if (warm_left_ > 0) {
        warm_left_ -= tokens;
        return;
    }
    tokens_ += tokens;
    seconds_ += seconds;
    lookups_ += lookups;
    hits_ += hits;
    if (tokens_ >= pol_.window_tokens) close_window();
}

void RateProbe::close_window() {
    const double spt = seconds_ / (double) tokens_;
    const double hit = lookups_ ? (double) hits_ / (double) lookups_ : -1.0;
    const uint64_t lookups = lookups_, hits = hits_;
    tokens_ = 0;
    seconds_ = 0.0;
    lookups_ = hits_ = 0;
    if (phase_ == Phase::Settle) {
        ++settle_windows_;
        // As the trial settles where the cache is in view; where it is not, the longest wait.
        const bool settled =
            hits_visible_ ? hit < 0.0 || (last_hit_ >= 0.0 && std::fabs(hit - last_hit_) <= pol_.settle_hit_delta) ||
                                settle_windows_ >= pol_.settle_windows_max
                          : settle_windows_ >= pol_.settle_windows_max;
        last_hit_ = hit;
        if (settled) phase_ = Phase::Measure;
        return;
    }
    spt_.push_back(spt);
    timed_lookups_ += lookups;
    timed_hits_ += hits;
    if ((int) spt_.size() >= windows_wanted()) phase_ = Phase::Done;
}

} // namespace bmoe
