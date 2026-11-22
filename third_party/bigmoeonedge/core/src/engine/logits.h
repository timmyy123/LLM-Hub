#pragma once

// Numerics over one row of logits, shared by every path that READS a distribution instead of
// sampling from it (perplexity, decide). Pure: no llama.cpp, so it is unit-testable on its own.

namespace bmoe::detail {

// The log-softmax normaliser of a logit row: log p(v) = (x[v] - max) - log_sum. Kept as the pair
// rather than one log-sum-exp value so every reader subtracts in the same, numerically safe order
// and two paths scoring the same row agree to the last bit.
struct LogNorm {
    float max = 0.0f;
    double log_sum = 0.0;
    double logp(float x) const { return (double) (x - max) - log_sum; }
};

// Requires n >= 1.
LogNorm log_norm(const float * x, int n);

} // namespace bmoe::detail
