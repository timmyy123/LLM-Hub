#include "prefill_device.h"

#include "../io/platform_io.h"

#include "ggml-alloc.h"
#include "ggml.h"

#include <cstdint>

namespace bmoe::detail {

PrefillDevice::~PrefillDevice() {
    // The model outlives nothing it cannot read: hand every weight back to its mapping before the
    // device copy it may still point at is freed.
    to_host();
    for (const Entry & e : states_)
        apply(e.t, e.host);
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    if (data_buf_) ggml_backend_buffer_free(data_buf_);
    if (data_ctx_) ggml_free(data_ctx_);
    if (state_buf_) ggml_backend_buffer_free(state_buf_);
    if (state_ctx_) ggml_free(state_ctx_);
}

bool PrefillDevice::init_state(ggml_backend_dev_t dev,
                               const std::vector<ggml_tensor *> & states,
                               std::string & where,
                               std::string & err) {
    if (states.empty()) {
        where = "none";
        return true;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
    if (!buft) buft = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
    where = ggml_backend_buft_name(buft);

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (states.size() + 1);
    ip.no_alloc = true;
    state_ctx_ = ggml_init(ip);
    if (!state_ctx_) {
        err = "ggml_init failed";
        return false;
    }
    std::vector<ggml_tensor *> twins;
    for (ggml_tensor * t : states) {
        if (!t->buffer || !t->data || t->view_src || !ggml_is_contiguous(t)) {
            err = std::string("state tensor ") + t->name + " is not an allocated contiguous leaf";
            return false;
        }
        ggml_tensor * s = ggml_dup_tensor(state_ctx_, t);
        ggml_set_name(s, t->name);
        twins.push_back(s);
    }
    state_buf_ = ggml_backend_alloc_ctx_tensors_from_buft(state_ctx_, buft);
    if (!state_buf_) {
        err = std::string("cannot allocate the model state in ") + where;
        return false;
    }
    ggml_backend_buffer_set_usage(state_buf_, ggml_backend_buffer_get_usage(states[0]->buffer));
    std::vector<uint8_t> tmp;
    for (size_t i = 0; i < states.size(); ++i) {
        ggml_tensor * t = states[i];
        ggml_tensor * s = twins[i];
        tmp.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
        ggml_backend_tensor_set(s, tmp.data(), 0, tmp.size());
        Entry e;
        e.t = t;
        e.host = {t->buffer, t->data, t->extra};
        e.dev = {s->buffer, s->data, s->extra};
        apply(t, e.dev);
        states_.push_back(e);
        state_bytes_ += ggml_nbytes(t);
    }
    drop_host_state();
    return true;
}

void PrefillDevice::clear_state() {
    if (state_buf_) ggml_backend_buffer_clear(state_buf_, 0);
    drop_host_state();
}

void PrefillDevice::drop_host_state() const {
    const uintptr_t page = (uintptr_t) pio::vm_page();
    for (const Entry & e : states_) {
        if (!e.host.buffer || !ggml_backend_buffer_is_host(e.host.buffer)) continue;
        // Only the pages wholly inside the tensor: its neighbours in the same buffer may still be live.
        const uintptr_t lo = ((uintptr_t) e.host.data + page - 1) & ~(page - 1);
        const uintptr_t hi = ((uintptr_t) e.host.data + ggml_nbytes(e.t)) & ~(page - 1);
        if (hi > lo) pio::vm_drop_anon_pages((void *) lo, (size_t) (hi - lo));
    }
}

void PrefillDevice::apply(ggml_tensor * t, const Binding & b) {
    t->buffer = b.buffer;
    t->data = b.data;
    t->extra = b.extra;
}

void PrefillDevice::place(bool on_device) {
    if (on_device == on_device_) return;
    for (const Entry & e : entries_)
        apply(e.t, on_device ? e.dev : e.host);
    on_device_ = on_device;
}

bool PrefillDevice::init(ggml_backend_dev_t dev,
                         const std::vector<ggml_tensor *> & weights,
                         const std::unordered_set<const ggml_tensor *> & non_matrix,
                         std::string & err) {
    if (!dev) {
        err = "no device";
        return false;
    }
    if (weights.empty()) return true; // everything goes through an arena, or the model has no layer weights
    for (ggml_tensor * w : weights) {
        if (!w->buffer || !ggml_backend_buffer_is_host(w->buffer) || !w->data) {
            err = std::string("weight ") + w->name + " is not host readable";
            return false;
        }
        if (!ggml_is_contiguous(w) || w->view_src) {
            err = std::string("weight ") + w->name + " is not a contiguous leaf";
            return false;
        }
    }

    // Two groups, two buffers: the matmul matrices in a WEIGHTS buffer (a backend may choose the layout
    // from the usage — Hexagon repacks WEIGHTS only — and the scheduler's "run the op where the weight
    // is" rule looks at WEIGHTS buffers), every other weight in plain device memory, which a backend
    // that maps WEIGHTS for its matmul alone still lets every kernel address (see init_dense).
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    for (int d = 0; d < 2; ++d) {
        std::vector<ggml_tensor *> group;
        for (ggml_tensor * w : weights)
            if ((non_matrix.count(w) != 0) == (d == 1)) group.push_back(w);
        if (group.empty()) continue;

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (group.size() + 1);
        ip.no_alloc = true;
        ggml_context *& ctx = d ? data_ctx_ : ctx_;
        ctx = ggml_init(ip);
        if (!ctx) {
            err = "ggml_init failed";
            return false;
        }
        std::vector<ggml_tensor *> twins;
        twins.reserve(group.size());
        for (ggml_tensor * w : group) {
            ggml_tensor * s = ggml_dup_tensor(ctx, w);
            ggml_set_name(s, w->name);
            twins.push_back(s);
        }
        ggml_backend_buffer_t & buf = d ? data_buf_ : buf_;
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            err = std::string("cannot allocate the layer weights on ") + ggml_backend_dev_name(dev);
            return false;
        }
        // Before the copy, which is where a backend lays the bytes out.
        ggml_backend_buffer_set_usage(buf, d ? GGML_BACKEND_BUFFER_USAGE_ANY : GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t i = 0; i < group.size(); ++i) {
            ggml_tensor * w = group[i];
            ggml_tensor * s = twins[i];
            ggml_backend_tensor_set(s, w->data, 0, ggml_nbytes(w));
            Entry e;
            e.t = w;
            e.host = {w->buffer, w->data, w->extra};
            e.dev = {s->buffer, s->data, s->extra};
            entries_.push_back(e);
            bytes_ += ggml_nbytes(w);
        }
    }
    return true;
}

} // namespace bmoe::detail
