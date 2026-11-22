// The prefill device as the session sees it: which device, what was moved there, and where the next
// graph runs.
//
// PrefillDevice (layer weights and model state) and DeviceExpertArena (streamed experts) are the
// mechanisms; this is their composition for one session. It picks the weights to move from the
// capture graph, sizes the arena from what the streamer was given, redoes the context's compute
// reservation with the weights in place, and then offers the session four things: place the next
// graph, run a decode there, clear the moved state, and read the arena's counters. The session keeps
// only the decision of WHICH graphs go to the device (PrefillDeviceConfig::min_tokens).
#pragma once

#include "prefill_device.h"
#include "../moe/device_arena.h"
#include "../moe/expert_stream_source.h"

#include "bmoe/config.h"

#include "ggml-backend.h"
#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace bmoe {

class RouterHook;

namespace detail {

class PrefillPath {
public:
    // Looks `name` up in the backend registry and opens it once. Null, with `err` saying why, when it
    // is absent (the build lacks the backend, or the backend found no hardware: a phone without the
    // accelerator's driver registers nothing) or registered but unable to open. The session then
    // runs on the CPU: the device is a speed-up, never a condition for the model to load.
    static ggml_backend_dev_t find_device(const std::string & name, std::string & err);

    // The auto-selected devices minus those that could only be dead weight in a run with no prefill
    // device, or null when none is dropped (llama.cpp's own choice then stands). See the definition.
    static const ggml_backend_dev_t * devices_without_prefill(std::vector<ggml_backend_dev_t> & keep);

    explicit PrefillPath(ggml_backend_dev_t dev) : devs_{dev, nullptr} {}
    PrefillPath(const PrefillPath &) = delete;
    PrefillPath & operator=(const PrefillPath &) = delete;

    // The null-terminated list for llama_model_params::devices. It must outlive the model load.
    ggml_backend_dev_t * model_devices() { return devs_; }

    // With streaming, the arena is sized from exactly what the streamer is given, so the session hands
    // a copy over before the streamer takes it.
    void keep_stream_layout(const std::vector<LayerExperts> & layers, const std::vector<std::string> & shards) {
        arena_layers_ = layers;
        arena_shards_ = shards;
    }

    // Everything after load: learn the layer weights from one graph (streaming already ran that
    // capture; without it, run it here), give them a device copy or an arena, move the model state,
    // and redo the compute reservation. `ctx`'s memory is left empty.
    bool open(llama_context * ctx,
              const llama_vocab * vocab,
              RouterHook & hook,
              const PrefillDeviceConfig & cfg,
              const MoeStreamConfig & moe,
              int n_layer_streamed,
              std::string & err);

    // Placement of the NEXT graph: the layer weights, the streamed experts and the hook move together.
    // Returns where it actually went: with slots on demand a device that has no room right now
    // leaves the graph on the host. `hold_slots` keeps slots taken on demand while the weights are
    // home: a graph of another context in the middle of a prefill (the draft's walk of the same chunk)
    // reads the host bindings, and the next chunk goes back out without asking for room again.
    bool place(bool on_device, bool hold_slots = false);
    // The expert cache the slots take turns with (PrefillDeviceConfig::slots_on_demand). Call before
    // open(); null leaves the cache alone.
    void share_memory_with(ExpertStreamSource * cache) { cache_ = cache; }
    // The expert cache the arena may fill from (PrefillDeviceConfig::from_cache), whether or not it
    // also shares memory with the slots. Call before open().
    void read_experts_from(ExpertStreamSource * cache) { cache_read_ = cache; }
    // One llama_decode, on whichever side place() last chose. On the device the arena is started
    // before and drained after, and a failed expert read fails the decode: the graph would have
    // computed on a slot that never filled.
    int decode(llama_context * ctx, const llama_batch & b);
    // The same bracket around a batch of embeddings (an image's tokens, media_input.h).
    int decode_ext(llama_context * ctx, llama_batch_ext * b);
    // For every llama_memory_clear(data) of the target context: llama.cpp clears the buffers it
    // allocated, which the moved model state no longer lives in.
    void clear_state();

    uint64_t arena_read_bytes() const { return arena_ ? arena_->read_bytes() : 0; }
    uint64_t arena_cache_bytes() const { return arena_ ? arena_->cache_bytes() : 0; }
    double arena_stall_seconds() const { return arena_ ? arena_->stall_seconds() : 0.0; }
    uint64_t arena_routed_used() const { return arena_ ? arena_->routed_used() : 0; }
    uint64_t arena_routed_demand() const { return arena_ ? arena_->routed_demand() : 0; }

private:
    bool open_arena(const PrefillDeviceConfig & cfg,
                    const MoeStreamConfig & moe,
                    const std::vector<std::vector<ggml_tensor *>> & dense_per_layer,
                    std::string & err);
    bool reserve_on_device(llama_context * ctx, const llama_vocab * vocab, int n_tokens);

    ggml_backend_dev_t devs_[2];
    RouterHook * hook_ = nullptr;
    llama_context * ctx_ = nullptr;
    ExpertStreamSource * cache_ = nullptr;
    ExpertStreamSource * cache_read_ = nullptr;
    bool lent_ = false; // the slots hold memory lent by the cache, returned when they are released
    bool said_no_room_ = false;
    std::vector<LayerExperts> arena_layers_;
    std::vector<std::string> arena_shards_;
    // Declared before the arena so it is destroyed after it: the arena hands the experts back to the
    // streamer's binding first, then the device copy goes.
    std::unique_ptr<PrefillDevice> dev_;
    std::unique_ptr<DeviceExpertArena> arena_;
};

} // namespace detail
} // namespace bmoe
