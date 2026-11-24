#include "logit_agreement.h"

#include "logits.h"

#include <algorithm>
#include <cmath>

namespace bmoe::detail {

namespace {
// The index of the row's largest logit, and whether every value in the row is a number.
int top_of(const float * x, int n, bool & finite) {
    int best = 0;
    finite = std::isfinite(x[0]);
    for (int v = 1; v < n; ++v) {
        if (!std::isfinite(x[v])) finite = false;
        if (x[v] > x[best]) best = v;
    }
    return best;
}
} // namespace

void LogitAgreement::add(const float * reference, const float * candidate, int n_vocab) {
    if (n_vocab < 1) return;
    bool ref_finite = true, cand_finite = true;
    const int top_ref = top_of(reference, n_vocab, ref_finite);
    const int top_cand = top_of(candidate, n_vocab, cand_finite);
    ++rows;
    if (cand_finite && top_ref == top_cand) ++same;

    // The KL divergence of the candidate's next-token distribution from the reference's: how far the
    // whole distribution moved, where agreeing on the top token only says whether a near tie flipped.
    // Measured on a 12 GB phone's NPU: Qwen3.6-35B-A3B Q4_K_M a mean of 0.004 (rounding, its two
    // differing rows near ties), Gemma 4 26B-A4B Q4_K_M 0.60 with one row at 13.5 where the host was
    // 0.999 sure (the backend's expert matmul past the fp16 range, ggml-org/llama.cpp#32198).
    const LogNorm nr = log_norm(reference, n_vocab);
    const LogNorm nc = log_norm(candidate, n_vocab);
    double kl = 0.0;
    for (int v = 0; v < n_vocab; ++v) {
        const double e = (double) candidate[v] - (double) reference[v];
        err_sq += e * e;
        ref_sq += (double) reference[v] * (double) reference[v];
        const double lr = nr.logp(reference[v]);
        kl += std::exp(lr) * (lr - nc.logp(candidate[v]));
    }
    // The mean over the rows so far, kept as a running value so a reader needs no finishing call.
    kld += (kl - kld) / rows;
    kld_max = rows == 1 ? kl : std::max(kld_max, kl);
}

} // namespace bmoe::detail
