#include "bmoe/ngram_draft.h"

#include <algorithm>
#include <cstddef>

namespace bmoe {

int ngram_draft(const std::vector<int32_t> & ctx,
                int32_t last,
                int n_max,
                int min_match,
                int max_match,
                std::vector<int32_t> & out) {
    out.clear();

    if (n_max <= 0) return 0;
    if (min_match < 1) min_match = 1;
    if (max_match < min_match) return 0;

    // The corpus is ctx ++ [last], addressed without materialising it: `last` is the final position
    // and belongs to the pattern, because the token just confirmed is the strongest evidence there
    // is about what comes next.
    const size_t L = ctx.size() + 1;
    const auto at = [&](size_t i) -> int32_t { return i < ctx.size() ? ctx[i] : last; };

    // A match needs min_match tokens ending strictly before the corpus end.
    if (L < (size_t) min_match + 1) return 0;

    const size_t end = L - 1;                    // the pattern's last position
    const size_t j_min = (size_t) min_match - 1; // below this a candidate cannot reach min_match

    int best_m = 0;
    size_t best_j = 0;

    // Scan backwards from the most recent candidate. Because a longer match only ever replaces a
    // shorter one (strict >), the most recent position wins among equal lengths for free.
    for (size_t j = end - 1;; --j) {
        if (at(j) == at(end)) {
            // Longest common suffix of C[..j] and C[..end], capped by max_match and by how much
            // corpus lies at or before j. Overlap with the pattern is intentional.
            const int lim = std::min(max_match, (int) (j + 1));
            int m = 1;
            while (m < lim && at(j - m) == at(end - m)) {
                ++m;
            }
            if (m > best_m) {
                best_m = m;
                best_j = j;
                if (best_m == max_match) break; // capped: nothing earlier can beat it
            }
        }
        if (j <= j_min) break;
    }

    // The confidence gate, and the whole economics of this source: no confident match means no
    // draft, no widened verify batch, and a step that costs exactly a plain decode.
    if (best_m < min_match) return 0;

    const size_t avail = end - best_j; // tokens that followed the match, up to the corpus end
    const size_t n = std::min((size_t) n_max, avail);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(at(best_j + 1 + i));
    }
    return (int) n;
}

} // namespace bmoe
