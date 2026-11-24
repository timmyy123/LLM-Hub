#include "prefill_path.h"

#include "compute_reservation.h"
#include "bmoe/log.h"

#include "../moe/router_hook.h"
#include "../moe/tensor_names.h"

#include <cstdio>
#include <cstring>
#include <exception>

namespace bmoe::detail {

namespace {

// Any valid token serves a graph whose output nobody reads.
llama_token filler_token(const llama_vocab * vocab) {
    const llama_token t = llama_vocab_bos(vocab);
    return t < 0 ? 0 : t;
}

} // namespace

ggml_backend_dev_t PrefillPath::find_device(const std::string & name, std::string & err) {
    if (ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str())) {
        // A registered device can still fail to open: Hexagon registers on any phone with the fastrpc
        // driver, and only opening a session finds a DSP its kernels were not built for. Open it here,
        // before the load, where a failure can still leave the run on the CPU; the backend keeps the
        // session it opened, so the context's own init reuses it.
        std::string what = "no backend";
        ggml_backend_t probe = nullptr;
        try {
            probe = ggml_backend_dev_init(dev, nullptr);
        } catch (const std::exception & e) {
            what = e.what();
        }
        if (probe) {
            ggml_backend_free(probe);
            return dev;
        }
        err = "prefill device '" + name + "' did not open (" + what + ")";
        return nullptr;
    }
    std::string names;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (!names.empty()) names += ", ";
        names += ggml_backend_dev_name(ggml_backend_dev_get(i));
    }
    err = "prefill device '" + name + "' is not available here (devices found: " + names + ")";
    return nullptr;
}

// With no devices given, llama.cpp lists every GPU it finds, and the context opens a backend on each.
// This engine never assigns such a device a layer, so all it can do in the run is pick up ops on the
// host-resident weights, which takes a device that reaches host memory (a host buffer type or buffers
// over host pointers). One with neither, like the Hexagon NPU, is only a session to open, and one that
// may fail to open on a DSP its kernels do not cover: it stays out unless named as the prefill device.
const ggml_backend_dev_t * PrefillPath::devices_without_prefill(std::vector<ggml_backend_dev_t> & keep) {
    std::vector<ggml_backend_dev_t> gpus, igpus;
    bool dropped = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.host_buffer && !props.caps.buffer_from_host_ptr) {
            dropped = true;
            continue;
        }
        // llama.cpp's rule for integrated devices: those of the first backend seen only.
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpus.push_back(dev);
        } else if (igpus.empty() || ggml_backend_dev_backend_reg(dev) == ggml_backend_dev_backend_reg(igpus.back())) {
            igpus.push_back(dev);
        }
    }
    if (!dropped) return nullptr;
    // And integrated devices only when no discrete one is left.
    keep = gpus.empty() ? igpus : gpus;
    keep.push_back(nullptr);
    return keep.data();
}

bool PrefillPath::open(llama_context * ctx,
                       const llama_vocab * vocab,
                       RouterHook & hook,
                       const PrefillDeviceConfig & cfg,
                       const MoeStreamConfig & moe,
                       int n_layer_streamed,
                       std::string & err) {
    hook_ = &hook;
    ctx_ = ctx;
    auto fail = [&](const std::string & what) {
        err = "prefill device " + cfg.device + what;
        return false;
    };

    if (!moe.enabled) {
        hook.begin_capture();
        llama_token tok = filler_token(vocab);
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) return fail(": capture decode failed");
        hook.end_capture();
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    // Layer weights only ("blk.N." is llama.cpp's naming for every architecture). The token table is
    // a row gather and the output head computes the last position alone, so both stay with the CPU:
    // moving them would buy nothing and cost their size in device memory. The usage test drops the
    // graph inputs and KV views the capture records alongside.
    std::vector<ggml_tensor *> layer_weights;
    for (ggml_tensor * t : hook.captured_weight_objects()) {
        if (tensor_names::block_layer(t->name) < 0) continue;
        if (!t->buffer || ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) continue;
        layer_weights.push_back(t);
    }
    // Streaming means the model does not fit, and then the layer weights cannot be resident on the
    // device either: on unified memory a resident copy is RAM taken from the expert cache twice over,
    // and a device session has a ceiling of its own. They go through the arena, two layers at a time,
    // paced at each layer's last node — which the capture must have found for every layer, or the
    // pacing has nowhere to wait and the copy stays resident.
    std::vector<std::vector<ggml_tensor *>> dense_per_layer;
    if (moe.enabled && hook.learned_layer_ends()) {
        dense_per_layer.resize((size_t) n_layer_streamed);
        for (ggml_tensor * t : layer_weights) {
            const int il = tensor_names::block_layer(t->name);
            if (il >= 0 && il < (int) dense_per_layer.size()) dense_per_layer[(size_t) il].push_back(t);
        }
        layer_weights.clear();
    } else if (moe.enabled) {
        log_printf(LogLevel::Warn, "prefill-device: layer ends not found in the graph; layer weights stay "
                                   "resident on the device");
    }

    dev_ = std::make_unique<PrefillDevice>();
    std::string perr;
    if (!dev_->init(devs_[0], layer_weights, hook.non_matrix_weights(), perr)) return fail(": " + perr);
    std::string state_where;
    if (!dev_->init_state(devs_[0], hook.captured_state_objects(), state_where, perr)) return fail(": " + perr);
    log_printf(LogLevel::Info, "prefill-device model state: %.1f MiB in %s",
               (double) dev_->state_bytes() / (1024.0 * 1024.0), state_where.c_str());
    if (moe.enabled && !open_arena(cfg, moe, dense_per_layer, err)) return false;
    hook.count_device_nodes(true);

    if (!reserve_on_device(ctx, vocab, cfg.min_tokens)) return fail(": the device-placed reservation decode failed");
    log_printf(LogLevel::Info, "prefill-device %s: %zu layer tensors, %.1f MiB copied; graphs >= %d tokens run there",
               cfg.device.c_str(), dev_->n_tensors(), (double) dev_->bytes() / (1024.0 * 1024.0), cfg.min_tokens);
    return true;
}

bool PrefillPath::open_arena(const PrefillDeviceConfig & cfg,
                             const MoeStreamConfig & moe,
                             const std::vector<std::vector<ggml_tensor *>> & dense_per_layer,
                             std::string & err) {
    arena_ = std::make_unique<DeviceExpertArena>();
    std::string perr;
    if (!arena_->init(devs_[0], arena_shards_, arena_layers_, cfg.load_threads, moe.o_direct, cfg.slots,
                      cfg.slots_on_demand, perr)) {
        err = "prefill device " + cfg.device + " expert arena: " + perr;
        return false;
    }
    arena_layers_.clear();
    arena_->set_test_delay_us(cfg.test_load_delay_us);
    arena_->set_routed(cfg.routed, cfg.routed_full_frac);
    arena_->set_cache(cfg.from_cache ? cache_read_ : nullptr);
    arena_->set_test_skip_demand(cfg.test_routed_skip_demand);
    if (!arena_->init_dense(devs_[0], dense_per_layer, hook_->non_matrix_weights(), perr)) {
        err = "prefill device " + cfg.device + " dense arena: " + perr;
        return false;
    }
    // The slots follow the graph's own order of layers, which the capture read: a model that runs a
    // block more than once comes back to it.
    arena_->set_graph_order(hook_->layer_order(), hook_->layer_blocks());
    if (arena_->has_dense())
        log_printf(LogLevel::Info,
                   "prefill-device dense arena: layer matrices through 2 slots, the rest held, %.1f MiB (%.1f "
                   "held); %d converted to a type the device takes (%.1f MiB)",
                   (double) arena_->dense_slot_bytes() / (1024.0 * 1024.0),
                   (double) arena_->dense_resident_bytes() / (1024.0 * 1024.0), arena_->dense_converted(),
                   (double) arena_->dense_converted_bytes() / (1024.0 * 1024.0));
    log_printf(LogLevel::Info, "prefill-device expert arena: %d layers through %d slot%s of %.1f MiB, %d loaders%s",
               arena_->n_layers(), arena_->n_slots(), arena_->n_slots() == 1 ? "" : "s",
               (double) arena_->slot_bytes() / (double) arena_->n_slots() / (1024.0 * 1024.0), cfg.load_threads,
               arena_->on_demand() ? ", held only during a prefill" : "");
    return true;
}

// The context reserved its compute buffers at creation, for the widest graph with every weight on the
// CPU — a prefill this session will never run there. At a wide ubatch that reservation is gigabytes
// of RAM the expert cache does not get (measured 2.2 GB at 2048 on a 35B-A3B, while the device's own
// graph needed 136 MB at 1024). So reserve again, with the weights where a wide graph will actually
// find them: request_compute_reservation has llama.cpp redo its reservation at the next decode, and
// one device-placed decode is that decode. The CPU keeps only what its own graphs (decode, a short
// tail) need, grown on demand from there.
bool PrefillPath::reserve_on_device(llama_context * ctx, const llama_vocab * vocab, int n_tokens) {
    request_compute_reservation(ctx);
    std::vector<llama_token> toks((size_t) n_tokens, filler_token(vocab));
    place(true);
    const int rc = decode(ctx, llama_batch_get_one(toks.data(), n_tokens));
    place(false);
    llama_memory_clear(llama_get_memory(ctx), true);
    clear_state();
    return rc == 0;
}

// Memory in turns. A device prefill does not go through the expert cache - the arena reads from
// flash into its own slots - and a decode does not touch the slots, so the two are never busy at
// once. With slots on demand they are not both HELD at once either: before the first device graph
// of a prompt the cache gives up as much as the slots take (its coldest entries), the slots are
// built, and after the last device graph they are given back and the cache has its budget again.
// What it costs is re-reading what was evicted, and building the slots once per prompt.
//
// The device can be out of room at that moment. Then the graph runs on the host, and the context is
// made to schedule it afresh: a host graph of the same width as the device graph before it would
// otherwise be reused as it was placed.
bool PrefillPath::place(bool on_device, bool hold_slots) {
    if (!dev_) return false;
    const bool turns = arena_ && arena_->on_demand();
    if (on_device && turns && !arena_->slots_held()) {
        if (cache_) cache_->lend_cache(arena_->slot_bytes());
        std::string err;
        if (!arena_->acquire(err)) {
            if (cache_) cache_->return_cache();
            if (!said_no_room_)
                log_printf(LogLevel::Warn, "prefill-device: %s; this prefill runs on the host", err.c_str());
            said_no_room_ = true;
            if (ctx_) {
                llama_set_causal_attn(ctx_, false);
                llama_set_causal_attn(ctx_, true);
            }
            on_device = false;
        } else {
            lent_ = cache_ != nullptr;
        }
    }
    dev_->place(on_device);
    if (arena_) {
        arena_->place(on_device);
        hook_->set_device_arena(on_device ? arena_.get() : nullptr);
    }
    if (!on_device && !hold_slots && turns && arena_->slots_held()) {
        arena_->release();
        if (lent_) cache_->return_cache();
        lent_ = false;
    }
    return on_device;
}

int PrefillPath::decode(llama_context * ctx, const llama_batch & b) {
    const bool dev = arena_ && dev_ && dev_->on_device();
    if (dev) arena_->begin_graph();
    int rc = llama_decode(ctx, b);
    if (dev) {
        arena_->end_graph();
        if (rc == 0 && arena_->failed()) rc = -1;
    }
    return rc;
}

int PrefillPath::decode_ext(llama_context * ctx, llama_batch_ext * b) {
    const bool dev = arena_ && dev_ && dev_->on_device();
    if (dev) arena_->begin_graph();
    int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b);
    if (dev) {
        arena_->end_graph();
        if (rc == 0 && arena_->failed()) rc = -1;
    }
    return rc;
}

void PrefillPath::clear_state() {
    if (dev_) dev_->clear_state();
}

} // namespace bmoe::detail
