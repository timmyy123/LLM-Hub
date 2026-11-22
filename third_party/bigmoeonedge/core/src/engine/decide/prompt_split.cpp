#include "prompt_split.h"

#include <algorithm>

namespace bmoe::detail {

int prefix_split(const std::vector<Token> & full, const std::vector<Token> & prefix_alone) {
    const int n = (int) full.size();
    if (n < 1) return 0;
    const int limit = std::min((int) prefix_alone.size(), n);
    int shared = 0;
    while (shared < limit && prefix_alone[shared] == full[shared])
        ++shared;
    return std::max(0, std::min(shared - 1, n - 1));
}

} // namespace bmoe::detail
