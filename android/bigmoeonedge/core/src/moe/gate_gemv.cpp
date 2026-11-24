#include "gate_gemv.h"

#include <algorithm>
#include <chrono>
#include <new>
#include <vector>

namespace bmoe::gate_gemv {

void scores_f32(const float * w, size_t row_bytes, int64_t nd, int64_t ne, const float * h, float * out) {
    const int64_t nd4 = nd & ~(int64_t) 3;
    for (int64_t e = 0; e < ne; ++e) {
        const float * p = (const float *) ((const char *) w + (size_t) e * row_bytes);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int64_t d = 0; d < nd4; d += 4) {
            a0 += p[d + 0] * h[d + 0];
            a1 += p[d + 1] * h[d + 1];
            a2 += p[d + 2] * h[d + 2];
            a3 += p[d + 3] * h[d + 3];
        }
        for (int64_t d = nd4; d < nd; ++d)
            a0 += p[d] * h[d];
        out[e] = (a0 + a1) + (a2 + a3);
    }
}

float quantize_row(const float * src, int64_t nd, int8_t * dst) {
    float amax = 0.0f;
    for (int64_t d = 0; d < nd; ++d) {
        const float a = src[d] < 0.0f ? -src[d] : src[d];
        if (a > amax) amax = a;
    }
    const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int64_t d = 0; d < nd; ++d) {
        const float t = src[d] * inv;
        const int v = (int) (t < 0.0f ? t - 0.5f : t + 0.5f);
        dst[d] = (int8_t) (v < -127 ? -127 : (v > 127 ? 127 : v));
    }
    return scale;
}

void scores_q8(
    const int8_t * q, const float * row_scale, int64_t nd, int64_t ne, const int8_t * qh, float hs, float * out) {
    const int64_t nd4 = nd & ~(int64_t) 3;
    for (int64_t e = 0; e < ne; ++e) {
        const int8_t * p = q + (size_t) e * (size_t) nd;
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int64_t d = 0; d < nd4; d += 4) {
            a0 += (int32_t) p[d + 0] * (int32_t) qh[d + 0];
            a1 += (int32_t) p[d + 1] * (int32_t) qh[d + 1];
            a2 += (int32_t) p[d + 2] * (int32_t) qh[d + 2];
            a3 += (int32_t) p[d + 3] * (int32_t) qh[d + 3];
        }
        for (int64_t d = nd4; d < nd; ++d)
            a0 += (int32_t) p[d] * (int32_t) qh[d];
        out[e] = (float) ((a0 + a1) + (a2 + a3)) * row_scale[e] * hs;
    }
}

namespace {

// The float working set is the model's REAL gate set, every layer, so the timing sees the cache
// regime the run will: gates that fit a large cache are timed from it, and gates that do not are
// timed from memory. A fixed smaller set would be cache-resident on a machine with a large last
// cache while the run's own gates stream from DRAM, and would hand the float path a win it does not
// have. This ceiling only bounds the transient memory for the largest models, and a gate set that
// reaches it is already larger than any cache, so capping it does not change the regime.
constexpr size_t k_max_bytes = (size_t) 512 << 20;

// How long the timing runs, across both paths. Long enough that one sample is not scheduling
// noise, short enough to be invisible next to a model load.
constexpr double k_sample_seconds = 0.12;
constexpr int k_rounds_min = 3;
constexpr int k_rounds_max = 64;

} // namespace

Timing measure(uint32_t n_embd, uint32_t n_expert, uint32_t n_layer) {
    Timing t;
    if (n_embd == 0 || n_expert == 0 || n_layer == 0) return t;
    const int64_t nd = n_embd, ne = n_expert;
    const size_t layer_floats = (size_t) nd * (size_t) ne;
    const size_t fit = std::max<size_t>(1, k_max_bytes / (layer_floats * sizeof(float)));
    const uint32_t layers = (uint32_t) std::min<size_t>(n_layer, fit);

    std::vector<float> w, h, out, scale;
    std::vector<int8_t> q, qh;
    try {
        w.resize(layer_floats * layers);
        q.resize(layer_floats * layers);
        scale.resize((size_t) ne * layers);
        h.resize((size_t) nd);
        qh.resize((size_t) nd);
        out.resize((size_t) ne);
    } catch (const std::bad_alloc &) {
        return t;
    }
    // Values only need to be non-degenerate; a fixed generator keeps the work identical run to run.
    uint32_t x = 2463534242u;
    const auto next = [&x] {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return (float) (int32_t) x * (1.0f / 2147483648.0f);
    };
    for (float & v : w)
        v = next();
    for (float & v : h)
        v = next();
    for (size_t r = 0; r < (size_t) ne * layers; ++r)
        scale[r] = quantize_row(w.data() + r * (size_t) nd, nd, q.data() + r * (size_t) nd);

    using clock = std::chrono::steady_clock;
    volatile float sink = 0.0f;
    const auto pass_f32 = [&] {
        for (uint32_t l = 0; l < layers; ++l) {
            scores_f32(w.data() + l * layer_floats, (size_t) nd * sizeof(float), nd, ne, h.data(), out.data());
            sink = sink + out[0];
        }
    };
    const auto pass_q8 = [&] {
        for (uint32_t l = 0; l < layers; ++l) {
            // As the run does it: the activation is quantized on every call.
            const float hs = quantize_row(h.data(), nd, qh.data());
            scores_q8(q.data() + l * layer_floats, scale.data() + (size_t) l * (size_t) ne, nd, ne, qh.data(), hs,
                      out.data());
            sink = sink + out[0];
        }
    };
    const auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    // One untimed pass each first: page faults on fresh memory are not what the run pays.
    pass_f32();
    pass_q8();
    // Alternated, so a frequency step or a neighbour's burst lands on both paths alike; the best
    // round of each is the figure, the one least disturbed by anything that is not the kernel.
    double best_f = 0.0, best_q = 0.0, spent = 0.0;
    for (int r = 0; r < k_rounds_max && (r < k_rounds_min || spent < k_sample_seconds); ++r) {
        const auto t0 = clock::now();
        pass_f32();
        const auto t1 = clock::now();
        pass_q8();
        const auto t2 = clock::now();
        const double f = secs(t0, t1), qd = secs(t1, t2);
        best_f = r == 0 ? f : std::min(best_f, f);
        best_q = r == 0 ? qd : std::min(best_q, qd);
        spent += f + qd;
    }
    t.f32_us = best_f * 1e6 / layers;
    t.int8_us = best_q * 1e6 / layers;
    t.layers = layers;
    return t;
}

} // namespace bmoe::gate_gemv
