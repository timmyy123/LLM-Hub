#pragma once
// Memory copies spread across threads.
//
// One thread copying from a file's mapping into another buffer is bound by its own page faults and
// a single core's copy rate, not by the memory: on a rented card's host, copying a 126 GB model's
// experts into the device backend's host memory (MoeStreamConfig::expert_host_buffer) took most of
// the two minutes a load with a device cache spent opening, against 15 s for a load without one.
// Spread over the cores the plan computes on, the faults and the copy run
// side by side. The bytes land exactly where one memcpy would put them.
#include <cstddef>
#include <vector>

namespace bmoe::pio {

// One copy: `bytes` from `src` to `dst`, which do not overlap.
struct CopySpan {
    void * dst = nullptr;
    const void * src = nullptr;
    size_t bytes = 0;
};

// The size the spans are cut into for the threads to share: large enough that taking the next piece
// costs nothing against copying it, small enough that one large tensor still spreads over every thread.
inline constexpr size_t k_copy_piece_bytes = size_t(64) << 20;

// Copy every span, on up to `threads` threads (at least one; the caller's is one of them).
void copy_spans(const std::vector<CopySpan> & spans, int threads);

} // namespace bmoe::pio
