// Memory copies spread across threads (parallel_copy.h).

#include "parallel_copy.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <system_error>
#include <thread>

namespace bmoe::pio {

void copy_spans(const std::vector<CopySpan> & spans, int threads) {
    std::vector<CopySpan> pieces;
    for (const CopySpan & s : spans)
        for (size_t off = 0; off < s.bytes; off += k_copy_piece_bytes)
            pieces.push_back(
                {(uint8_t *) s.dst + off, (const uint8_t *) s.src + off, std::min(k_copy_piece_bytes, s.bytes - off)});
    std::atomic<size_t> next{0};
    const auto work = [&] {
        for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < pieces.size();)
            std::memcpy(pieces[i].dst, pieces[i].src, pieces[i].bytes);
    };
    const size_t n = std::min(pieces.size(), (size_t) std::max(1, threads));
    std::vector<std::thread> helpers;
    // A thread the system will not start leaves its share to the others: the copy is the same.
    try {
        for (size_t t = 1; t < n; ++t)
            helpers.emplace_back(work);
    } catch (const std::system_error &) {
    }
    work();
    for (std::thread & h : helpers)
        h.join();
}

} // namespace bmoe::pio
