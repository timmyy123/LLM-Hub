// Which form of the router-gate GEMV this machine runs faster, settled by running both.
//
// Route-ahead's prediction re-reads a whole gate matrix per layer per token to rank the experts. An
// int8 mirror of it moves a quarter of the bytes and costs a quantization of the activation per
// call; whether that trade pays is how scarce bytes are against arithmetic, which machines answer
// differently (measured: an x86 host with bandwidth to spare ran the float path faster, a phone
// whose decode already saturates its bus ran the mirror three times faster). The decision used to
// be read off the compiler's target, which is the one thing that predicts neither result.
//
// The instrument is the run's own kernels (gate_gemv.h) on a gate of the model's own shape, walked
// layer by layer the way a decode walks it. It runs alone, so it sees the bus quieter than a decode
// leaves it: that only flatters the float path, whose cost is the bytes, so a mirror that wins here
// wins at least as clearly in the run.

#include "bmoe/probe.h"

#include "../moe/gate_gemv.h"

namespace bmoe {

void probe_gate_gemv(HardwareProfile & hw, const ModelProfile & model) {
    hw.gate_gemv_f32_us = hw.gate_gemv_int8_us = 0.0;
    if (!model.is_moe) return;
    const uint32_t layers = model.n_moe_layer > 0 ? model.moe_layer_runs() : model.n_layer;
    const gate_gemv::Timing t = gate_gemv::measure(model.n_embd, model.n_expert, layers);
    if (!t.measured()) return;
    hw.gate_gemv_f32_us = t.f32_us;
    hw.gate_gemv_int8_us = t.int8_us;
}

} // namespace bmoe
