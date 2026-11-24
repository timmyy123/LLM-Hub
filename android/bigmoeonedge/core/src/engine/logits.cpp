#include "logits.h"

#include <cmath>

namespace bmoe::detail {

LogNorm log_norm(const float * x, int n) {
    LogNorm z;
    z.max = x[0];
    for (int v = 1; v < n; ++v)
        if (x[v] > z.max) z.max = x[v];
    double sum = 0.0;
    for (int v = 0; v < n; ++v)
        sum += std::exp((double) (x[v] - z.max));
    z.log_sum = std::log(sum);
    return z;
}

} // namespace bmoe::detail
