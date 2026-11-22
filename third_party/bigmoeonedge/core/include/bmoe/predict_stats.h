// Expert-prediction accuracy counters.
//
// A prefetch is only worth its bandwidth if it guesses right, and on a device whose flash is
// already saturated a wrong guess is not free — it displaces a read that was needed. So before
// any predictor is wired into the loading path, it gets measured here: how much of a routing it
// would have had in flight, on this model, with these prompts.
//
// The unit is the SLOT, not the routing: a top-8 routing offers eight chances to be right, and a
// predictor that gets six of them turns six reads into hits. `exact` (the whole set predicted)
// is reported alongside because the literature splits on which one it quotes — set-overlap
// ratios and exact-match rates are not comparable, and a paper's headline number is usually the
// first while a system's usable number is closer to the second.
#pragma once

namespace bmoe {

struct PredictorStats {
    long long rows = 0;  // routings scored (one per MoE layer per decoded token)
    long long slots = 0; // sum of top-k over those routings: the denominator for `hits`
    long long hits = 0;  // predicted experts the router did select
    long long exact = 0; // routings whose ENTIRE top-k set was predicted

    // Fraction of routed experts the predictor would have had in flight, in [0, 1].
    double hit_frac() const { return slots > 0 ? (double) hits / (double) slots : 0.0; }
    // Fraction of routings needing no on-demand read at all, in [0, 1].
    double exact_frac() const { return rows > 0 ? (double) exact / (double) rows : 0.0; }
};

} // namespace bmoe
