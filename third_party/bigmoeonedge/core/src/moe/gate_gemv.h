// The router-gate GEMV the expert prediction runs, in its two forms: over the gate's own float
// weights, and over an int8 mirror of them. Which one a run uses is a choice between bytes and
// arithmetic - the mirror moves a quarter of the bytes and pays a quantization of the activation
// per call - and where that trade turns over is a property of the machine, not of its instruction
// set: measured, the float path won on an x86 host with bandwidth to spare and lost on a phone whose
// decode already saturates the bus. So the kernels live here, where the router hook runs them and
// measure_gate_gemv() times them, and the choice is read off that timing (MoeStreamConfig::
// gate_mirror) rather than off the compiler's target.
//
// Both kernels are portable C shaped for the vectorizer (independent partial sums, integer
// accumulation): no intrinsics, so every target has both paths and the measurement alone decides.
#pragma once

#include <cstddef>
#include <cstdint>

namespace bmoe::gate_gemv {

// One score per expert: the dot product of each of `ne` rows of `nd` floats (rows `row_bytes`
// apart) with `h`. The four partial sums are what let the compiler vectorize a float reduction at
// all; they change the summation order, so the last bits may differ from the graph's own gate.
void scores_f32(const float * w, size_t row_bytes, int64_t nd, int64_t ne, const float * h, float * out);

// Quantize one row of `nd` floats to int8 with one symmetric scale, written to `dst`. Returns the
// scale that dequantizes it (0 for an all-zero row).
float quantize_row(const float * src, int64_t nd, int8_t * dst);

// The int8 mirror's scores: `ne` int8 rows of `nd` (contiguous), `row_scale[e]` dequantizing row e,
// against an activation `qh` already quantized with scale `hs`. The ORDER of the scores is what the
// ranking reads, and both scales are positive, so they are applied once per expert at the end.
void scores_q8(
    const int8_t * q, const float * row_scale, int64_t nd, int64_t ne, const int8_t * qh, float hs, float * out);

// The two paths, timed on a synthetic gate of the model's shape: `n_layer` matrices of `n_expert`
// rows by `n_embd`, walked in turn the way a decode walks its layers, so the bytes come from where
// the run's own would: the working set is the model's whole gate set, so it sits in a cache exactly
// when the run's gates would (bounded for the largest models at a size no cache reaches).
// The int8 figure includes quantizing the activation, which the run pays per call. Microseconds
// per one layer's GEMV, single-threaded as the prediction worker is; both 0 when the shape is empty
// or the buffers cannot be had.
struct Timing {
    double f32_us = 0.0;
    double int8_us = 0.0;
    uint32_t layers = 0; // how many gate matrices the working set held
    bool measured() const { return f32_us > 0.0 && int8_us > 0.0; }
};
Timing measure(uint32_t n_embd, uint32_t n_expert, uint32_t n_layer);

} // namespace bmoe::gate_gemv
