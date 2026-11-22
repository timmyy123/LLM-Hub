// Which backend drives each card, where more than one can.
//
// Every backend registers the devices it can drive, so one card can appear once per backend that
// runs on it: CUDA and Vulkan on an NVIDIA board, HIP and Vulkan on an AMD one, SYCL and Vulkan on an
// Intel one. llama.cpp recognises them as one card by the id the backend reports and keeps the first
// registered, which is an order, not a measurement. Here the backends of one card are compared on
// the same measured rate, and one replaces the first only where it is faster by the planner's
// margin. A device whose identity is unknown is never grouped with another: two devices that may be
// two cards are left as llama.cpp would leave them.
//
// Pure: no llama.cpp, no I/O. The facts are the probes' (bmoe/hardware_profile.h).
#pragma once

#include "bmoe/hardware_profile.h"

#include <string>
#include <vector>

namespace bmoe {

struct BackendChoice {
    // The devices to hand llama.cpp, in the order the backends registered them, every card once;
    // empty where nothing was chosen (no card had two backends, or the first one held).
    std::vector<std::string> keep;
    // The devices left out, and one line per card on why: the rates compared.
    std::vector<std::string> left_out;
    std::vector<std::string> why;
};

// The rate a backend is compared on: its rate on this model's own expert block where the real-layer
// probe measured it, otherwise the synthetic GEMV's. Every backend of one card is compared on the
// same kind of figure; a card where they differ in kind keeps the first.
BackendChoice choose_backends(const HardwareProfile & hw, double min_win);

// `hw` without the devices `choice` left out.
HardwareProfile without_left_out(const HardwareProfile & hw, const BackendChoice & choice);

// The devices of `listed` the capacity fitter is asked of. A device whose memory is the host's adds
// no capacity, yet the fitter counts its free memory as a pool of its own and puts layers there, and
// a placement that touches one is set aside whole, the discrete card's layers with it. So where a
// listed device has memory of its own, those are the list, as llama.cpp's own default list leaves an
// integrated GPU out beside a discrete one. Where none has, `listed` is returned as it is.
std::vector<std::string> capacity_devices(const HardwareProfile & hw, const std::vector<std::string> & listed);

// The devices of `listed` that survived their probe. An empty `listed` stands for every device
// registered (llama.cpp's own default list), so where one of them did not survive, the others are
// named instead: left empty, the list would hand the failed device straight back. Where every device
// survived, `listed` is returned as it is and `dropped` (when given) is false.
std::vector<std::string>
surviving_devices(const HardwareProfile & hw, const std::vector<std::string> & listed, bool * dropped = nullptr);

} // namespace bmoe
