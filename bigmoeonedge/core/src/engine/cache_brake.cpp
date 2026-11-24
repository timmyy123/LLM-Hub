#include "cache_brake.h"

#include <algorithm>

namespace bmoe {

double CacheBrake::reference() const {
    if (seen_ < pol_.warmup_tokens || recent_.empty()) return 0.0;
    std::vector<double> v = recent_;
    std::nth_element(v.begin(), v.begin() + (std::ptrdiff_t)(v.size() / 2), v.end());
    return v[v.size() / 2];
}

bool CacheBrake::stalled(double compute_s) {
    ++seen_;
    const double ref = reference();
    const bool stall = ref > 0.0 && compute_s >= pol_.stall_factor * ref;
    // Warm-up tokens are not kept: while the cache fills, compute is not what it will settle to.
    if (!stall && seen_ > pol_.warmup_tokens / 2) {
        recent_.push_back(compute_s);
        if ((int) recent_.size() > pol_.window) recent_.erase(recent_.begin());
    }
    return stall;
}

size_t CacheBrake::braked(size_t budget, size_t floor) {
    if (budget <= floor) return budget;
    return std::max(budget / 2, floor);
}

} // namespace bmoe
