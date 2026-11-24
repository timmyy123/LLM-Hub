// Prefill on an accelerator, decode on the CPU, by rebinding weights between graphs.
//
// The llama.cpp scheduler assigns each op to the backend that holds its weight, and it does so on
// every graph it builds (ggml_backend_sched_backend_from_buffer). A weight's location is nothing but
// three public fields of its ggml_tensor — buffer, data, extra — so swapping them between two
// graphs moves every op that reads the weight from one backend to the other, with no llama.cpp
// change. This class owns the device-side copy and does the swap; the session decides WHEN, which
// is where the one real hazard lives (graph reuse, see PrefillDeviceConfig).
//
// The device copy is written through ggml_backend_tensor_set rather than memcpy, so a backend that
// keeps weights in its own layout (Hexagon repacks Q4_0 into HMX tiles on set_tensor) gets them the
// way its kernels expect. The host binding is left untouched: decode keeps reading the mapping.
#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

namespace bmoe::detail {

class PrefillDevice {
public:
    PrefillDevice() = default;
    ~PrefillDevice();
    PrefillDevice(const PrefillDevice &) = delete;
    PrefillDevice & operator=(const PrefillDevice &) = delete;

    // Allocates `weights` in `dev`'s buffer type and copies them in. Every weight must be host
    // readable (the mmap) and contiguous; anything else is an error, not a skip, because a
    // silently skipped weight would pin its op to the CPU and cost a split per graph. `non_matrix`
    // (RouterHook::non_matrix_weights) go to plain device memory instead of the WEIGHTS buffer, as
    // DeviceExpertArena::init_dense explains.
    bool init(ggml_backend_dev_t dev,
              const std::vector<ggml_tensor *> & weights,
              const std::unordered_set<const ggml_tensor *> & non_matrix,
              std::string & err);

    // Moves the memory module's state (KV cache, recurrent state) into the device's HOST buffer type
    // for good: memory the CPU reads directly and the device addresses too, so decode on the CPU and
    // prefill on the device share one cache without a copy per turn. Left on the CPU, the attention of
    // every device graph would fall back to the CPU, which at prefill widths is most of the time. A
    // device without a host buffer type (RPC) gets a plain CPU buffer: nothing gained, same behaviour.
    // Returns the buffer type used through `where`.
    //
    // llama.cpp keeps the buffers it allocated the state in, and nothing reads them after the move,
    // so their pages are handed back (drop_host_state): left resident they would hold a second copy of
    // the whole cache, which for a model with a large KV is more memory than the device path saves.
    bool init_state(ggml_backend_dev_t dev,
                    const std::vector<ggml_tensor *> & states,
                    std::string & where,
                    std::string & err);
    // Zero the moved state, for every llama_memory_clear(data=true) of the session: that call clears
    // the buffers llama.cpp allocated, which these tensors no longer live in. Its memset faults those
    // pages back in, so they are handed back again.
    void clear_state();
    size_t state_bytes() const { return state_bytes_; }

    // Placement for the NEXT graph. Idempotent; cheap (three stores per tensor).
    void place(bool on_device);
    void to_host() { place(false); }
    bool on_device() const { return on_device_; }

    size_t n_tensors() const { return entries_.size(); }
    size_t bytes() const { return bytes_; }

private:
    struct Binding {
        ggml_backend_buffer_t buffer = nullptr;
        void * data = nullptr;
        void * extra = nullptr;
    };
    struct Entry {
        ggml_tensor * t = nullptr;
        Binding host;
        Binding dev;
    };
    static void apply(ggml_tensor * t, const Binding & b);
    // Release the resident pages of the state's original (host) copies; see init_state.
    void drop_host_state() const;

    std::vector<Entry> entries_;
    ggml_context * ctx_ = nullptr;             // metadata of the device-side twins
    ggml_backend_buffer_t buf_ = nullptr;      // may be a multi-buffer when a backend caps buffer size
    ggml_context * data_ctx_ = nullptr;        // the non-matrix twins
    ggml_backend_buffer_t data_buf_ = nullptr; // ... in plain device memory
    size_t bytes_ = 0;
    bool on_device_ = false;

    std::vector<Entry> states_; // .dev is the moved binding, in force from init_state on
    ggml_context * state_ctx_ = nullptr;
    ggml_backend_buffer_t state_buf_ = nullptr;
    size_t state_bytes_ = 0;
};

} // namespace bmoe::detail
