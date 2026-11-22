// How closely two computations of the same model agree, read off their logits.
//
// Every check that asks "does this device compute this model" ends in the same comparison: the same
// tokens decoded on the cores and on the device, and the next-token rows of both side by side. The
// prefill device's confirmation and the whole-model answer check (device_answer.h) both judge by this
// one, so a device is never held to two different standards depending on which path asked.
//
// Pure: no llama.cpp, so it is unit-testable on its own.
#pragma once

#include "bmoe/config.h"

namespace bmoe::detail {

struct LogitAgreement {
    int rows = 0;         // rows compared
    int same = 0;         // rows whose top token is the reference's
    double kld = 0.0;     // mean KL divergence of the candidate's distribution from the reference's
    double kld_max = 0.0; // the worst row's
    double err_sq = 0.0;  // sum of squared logit differences, every row
    double ref_sq = 0.0;  // sum of squared reference logits, every row

    // One row of `n_vocab` logits from each side. A candidate row holding a value that is not a number
    // agrees with nothing: comparisons with it are all false, and a top token picked from such a row
    // is an accident of the scan, not the model's.
    void add(const float * reference, const float * candidate, int n_vocab);

    double agree() const { return rows > 0 ? (double) same / rows : 0.0; }
    double nmse() const { return ref_sq > 0.0 ? err_sq / ref_sq : 0.0; }
    // The verdict: the candidate picks the reference's next token in nearly every row. The distance
    // figures are logged beside it and do not decide: a near tie that flips moves them little, and a
    // backend that rounds differently moves them without changing an answer.
    bool matches() const { return rows > 0 && agree() >= (double) PrefillDeviceConfig::verify_min_agree; }
};

} // namespace bmoe::detail
