// The compute probes' shared pieces: what counts as two backends agreeing, and the real-layer probe
// the bandwidth probe hands the model's own layer to. Internal to core/src/plan.
#pragma once

#include "bmoe/hardware_profile.h"
#include "bmoe/model_profile.h"

#include <cstdint>
#include <vector>

namespace bmoe::detail {

// How many tokens the wide instrument is: several times the narrowest graph a prefill device is
// ever handed, so the sample sits inside the regime it speaks for. Shared by the real layer and the
// synthetic fallback, which report it as HardwareProfile::wide_batch.
constexpr int64_t k_wide_batch = 128;

// How long one sample must last, and the cap that keeps a slow device from turning a probe into a
// benchmark. Sized by DURATION rather than by a repetition count: a fixed count makes the sample
// shorter as the machine gets faster, and a two-millisecond sample on a general-purpose OS reports
// scheduling noise.
constexpr double k_sample_seconds = 0.15;
constexpr int k_reps_max = 512;

// Agreement, not bit-equality: llama.cpp's own normalized mean squared error, sum((a-b)^2) /
// sum(a^2), against the 5e-4 tests/test-backend-ops.cpp allows a MUL_MAT. See bandwidth_probe.cpp
// for the measurements behind taking upstream's bound rather than a stricter private one.
constexpr double k_max_nmse = 5e-4;
// A device judged against the exact result rather than against the host (probe_real_layer): twice
// upstream's per-op bound. Upstream's 5e-4 is a backend against the CPU on random data, and it relaxes
// it where a backend quantizes the activations its own way (2e-2 for MXFP4 activations); a wrong kernel
// is off by 1e-1 to 1. Measured on RTX 3090 cards: GLM-4.5-Air's Q5_0/Q5_1 experts at 5.3e-4 from the
// exact result, the host at 1.1e-5, every other layout measured at 6e-6 or less. The host must still be
// within k_max_nmse of the exact result to be the reference.
constexpr double k_max_nmse_device = 1e-3;
// The same judgment at prefill width, where a device computes with another kernel: a matmul over many
// tokens, which a backend may run in half precision where it runs one token exactly. The bound leans
// towards the device: ten times the one at one token, half of what upstream allows a backend that
// quantizes its activations (2e-2), and still ten times under a wrong kernel's 1e-1. Measured on an
// Apple M5 (Metal), the dense half at width against the exact result, beside what the whole model
// then lost with its prompt computed there (KL of the next-token distribution against the host's, 31
// positions; perplexity over 5.9k tokens): OLMoE Q8_0 1.6e-7 (KL 0.0010), Qwen3.6-35B Q4_K_M 2.3e-4
// (0.0030), Granite-3.1-3B Q4_K_M 9.7e-4 (0.0073, perplexity 33.71 on the host and 33.55 there),
// OLMoE Q4_K_M 2.4e-3 (0.0040, 61.00 and 60.85). The last was refused at 1e-3 and loses no more than
// the ones that passed: one layer's distance does not order what the model loses, so the bound only
// has to separate rounding from a wrong result. A prefill device is confirmed again on the whole model
// at load (PrefillDeviceConfig::verify); decode width keeps k_max_nmse_device. It is the bound of every
// comparison at width, whichever the reference: the exact result, the host where that cannot be
// computed, and the synthetic matmul a model with no readable layer is measured with.
constexpr double k_max_nmse_device_wide = 1e-2;
bool agrees(const std::vector<float> & a, const std::vector<float> & b, double bound = k_max_nmse);

// One real layer of the model, built from the file's own tensors and bytes, run on the host (at
// hw.best_threads) and on every device probed here, at decode width and at prefill width: the
// expert block and the dense matrices apart. Fills ComputeDevice::expert_gibs / dense_gibs /
// wide_matmul_gibs / wide_identity_ok, the host's counterparts, and hw.wide_batch, and refuses a
// device whose result disagrees with the host's at decode width (identity_ok = No). Returns false,
// touching nothing, where no layer could be read (no file, not a MoE, an unreadable tensor): the
// caller then keeps the synthetic matmul for what it filled before.
bool probe_real_layer(HardwareProfile & hw, const ModelProfile & model);

} // namespace bmoe::detail
