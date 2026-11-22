// The first stage: what llama.cpp's own capacity fitter decided for this model on this machine.
//
// llama.cpp ships a fitter (`common_fit_params`) that loads the model with no_alloc, measures the
// memory it would take per device, and if it does not fit first shrinks the context and then moves
// weights from device memory into system memory — per class inside each layer, attention first
// and the sparse expert tensors last. It sees every backend through ggml and knows nothing about
// flash: the comment above it reads "assumes system memory is unlimited". That line is where the
// second stage, the expert streamer, begins.
//
// This header is what the second stage reads from the first: which layers keep their experts on
// the host, and how much host memory the placed model will take. It is pure policy — no llama.cpp
// types — so the planner can be tested against a synthetic placement.
#pragma once

#include "bmoe/model_profile.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace bmoe {

// llama.cpp's assignment of a model's layers to devices, the one rule every figure of a placement is
// read against. `n_gpu_layers` counts slots from the top: the output first, then the blocks downwards,
// and -1 is every slot. The input layer, which holds the embedding tables, never leaves the host.
struct LayerSlots {
    uint32_t n_layer = 0;   // the blocks; the slots are these and the output
    uint32_t on_device = 0; // the slots a device holds, the output first: 0 to n_layer + 1
    uint32_t first_device_slot() const { return n_layer + 1 - on_device; }
    uint32_t host_blocks() const { return first_device_slot() < n_layer ? first_device_slot() : n_layer; }
    bool output_on_device() const { return on_device > 0; }
};
LayerSlots layer_slots(int32_t n_gpu_layers, uint32_t n_layer);

// The model's non-expert bytes the host holds under that assignment, read from the file's own parts:
// the blocks below the first device slot (each block's own bytes where the profile has them, a share
// by count where it does not), the embedding tables whole, the output where it was not moved, and
// whatever else sits outside the blocks, which stays on the host.
struct HostDense {
    uint64_t bytes = 0;
    uint64_t tables = 0; // of those, the embedding tables
};
HostDense host_dense_of(const ModelProfile & m, const LayerSlots & slots);

struct Placement {
    // Whether the fitter ran and produced this. False means the caller planned without it, and
    // every field below is what the engine does on its own: no layers on any device.
    bool fitted = false;
    std::string outcome; // the fitter's own one-line verdict, for the rationale

    int32_t n_gpu_layers = 0;
    uint32_t n_ctx = 0; // the context the fitter settled on (it shrinks this first)

    // Layers whose routed experts the fitter left on the host. These are the layers the streamer
    // can serve; experts the fitter placed on a device are not ours and are never read from flash.
    std::vector<uint32_t> host_expert_layers;

    // Host memory the placed model is projected to occupy, excluding the expert tensors of
    // `host_expert_layers` (which the streamer takes off the residency books) — the dense set that
    // stayed on the host, plus KV and compute buffers that live in host memory. This is what the
    // second stage subtracts from the residency budget before sizing its cache.
    uint64_t host_resident_bytes = 0;
    // The embedding tables' bytes inside `host_resident_bytes` (HostDense::tables). A table is
    // gathered by row, so a run may leave it in the mapping instead of holding it: what the host has
    // to hold is the figure above less this. Zero where the model has no such table.
    uint64_t host_table_bytes = 0;

    // Device-local memory the placement uses, summed over devices with memory of their own.
    uint64_t device_bytes = 0;
    // The projector's bytes the fitter set aside on a device before placing any layer, when the
    // run has one and a device with memory of its own to use it on: llama.cpp's library loads it
    // there whole, on top of whatever the fitter placed, so the fitter is told to leave that room
    // and the layer count already accounts for it. Counted in `device_bytes` and taken off that
    // device's `device_spare`. Zero without a projector, or where no device has memory of its own.
    uint64_t device_projector_bytes = 0;

    // True when the fitter had a device to place on, but one whose memory is the host's - so its
    // capacity arithmetic counted the same pool twice and its placement was NOT applied. Carried so
    // the plan can say that out loud instead of reporting "no device" on a machine that has one.
    bool shared_memory_placement = false;

    // The fitter's own host breakdown, raw, so the derived figures above can be checked against
    // it: what it projects for the model's host buffers, the context (KV), and the compute buffers.
    uint64_t raw_host_model_bytes = 0;
    uint64_t raw_host_context_bytes = 0;
    uint64_t raw_host_compute_bytes = 0;

    // The compute buffers the fitter projects on the devices, summed. Kept apart from
    // `device_bytes` because it is the part a device costs for being COMPUTED on, whatever weights
    // are or are not placed there - which is what a device prefill pays.
    uint64_t device_compute_bytes = 0;
    uint64_t device_context_bytes = 0; // the context the fitter put on the devices, summed

    // The part of the context, on every device and the host together, that does not grow with its
    // length: a recurrent state, a sliding window. It is what one context checkpoint copies into host
    // memory, read off two context lengths rather than off the architecture. Zero for a context that
    // is all attention, or when not measured.
    uint64_t context_fixed_bytes = 0;

    // A sliding-window cache holds its window and one ubatch of cells, up to the context's length
    // (window_cells), so a narrower graph holds less of it. What one of those cells costs, on the host
    // and on the devices, read off the same two widths as the compute buffers' slope, and the window
    // from the file; zero where the model has none.
    uint32_t n_swa = 0;
    uint64_t host_window_cell_bytes = 0;
    uint64_t device_window_cell_bytes = 0;
    // The cells a windowed cache holds at graph width `width`.
    uint64_t window_cells(uint32_t width) const {
        if (n_swa == 0) return 0;
        const uint64_t want = (uint64_t) n_swa + width;
        return n_ctx && want > n_ctx ? n_ctx : want;
    }

    // Which devices those compute buffers were projected for, and each one's part, at the width
    // they were projected at. A device the fitter did not project for (one it never places layers
    // on) has no part: the sum above describes other devices' graphs, not its own.
    std::vector<std::string> projected_devices;
    std::vector<uint64_t> projected_compute_bytes;
    // The part of `device_compute_bytes` that is the named device's, at this placement's width.
    uint64_t device_compute_of(const std::string & device) const;

    // The compute buffers above are those of one graph width - the widest graph the run computes at
    // once (RunConfig::n_ubatch) - and they grow with it: an activation is a row per token. The
    // width they were projected at, and what each further token of width costs, so the second stage
    // can price a narrower graph without asking again. A zero means "not measured" and the width is
    // then not the planner's to choose.
    uint32_t compute_width = 0;
    uint64_t host_compute_per_token = 0;
    uint64_t device_compute_per_token = 0;

    // This placement as it would be with the widest graph `width` tokens wide: the same placement,
    // with the compute buffers and every sum that contains them moved along the measured slope.
    // Unchanged when the slope is not known or `width` is the width it was projected at.
    Placement at_width(uint32_t width) const;

    // The override patterns the fitter wrote, kept verbatim so the session can reapply them at load
    // (they are what routes the placed experts to a device buffer type).
    std::vector<std::string> override_patterns;
    // The pattern that keeps one layer's routed experts on the host, for every layer that carries
    // experts, written by the adapter (which owns tensor naming) for a plan that moves every
    // layer's experts to the host itself: the device tier keeps the layers on the device and makes
    // the bytes the fitter gave their experts a cache instead.
    std::vector<std::string> all_expert_patterns;

    // How many of `n_gpu_layers` each device takes, by device name, in the order the fitter listed
    // the devices. The fitter decides this per device and projects its memory with it; loaded
    // without it, llama.cpp splits the layers by free memory instead, which on two cards put the
    // layers the fitter had spread over both onto whichever had more room (#321). Empty when nothing
    // is placed.
    std::vector<std::pair<std::string, uint32_t>> device_split;

    // What each device in `device_split` has free beyond what the fitter projects for it and the
    // margin it leaves, by device name: the remainder no further whole layer fitted in. The fitter
    // fills a card in whole layers, so this is memory it never uses; a device expert cache can, since
    // it fills by the expert rather than by the layer. Read off the same free figure the fitter
    // planned against, in the same call. Empty when nothing is placed.
    std::vector<std::pair<std::string, uint64_t>> device_spare;
};

} // namespace bmoe
