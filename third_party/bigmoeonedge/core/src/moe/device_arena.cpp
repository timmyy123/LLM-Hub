#include "device_arena.h"

#include "../io/platform_io.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace bmoe {

namespace {
// Whether the device runs a matmul whose weight is `type`, shaped like `w`, from its own buffer. Asked
// of the device itself rather than read off a list, so any backend answers for its own kernels. The
// probe weight is allocated: a backend may remember an unallocated weight (Hexagon notes it for
// repacking) and that note would outlive the probe.
bool device_matmul_takes(ggml_backend_dev_t dev, ggml_type type, const ggml_tensor * w) {
    if (ggml_is_quantized(type) && w->ne[0] % ggml_blck_size(type) != 0) return false;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4;
    ip.no_alloc = true;
    ggml_context * c = ggml_init(ip);
    if (!c) return false;
    ggml_tensor * p = ggml_new_tensor_2d(c, type, w->ne[0], std::min<int64_t>(w->ne[1], 256));
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, ggml_backend_dev_buffer_type(dev));
    bool ok = false;
    if (b) {
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, w->ne[0], 256);
        ok = ggml_backend_dev_supports_op(dev, ggml_mul_mat(c, p, x));
        ggml_backend_buffer_free(b);
    }
    ggml_free(c);
    return ok;
}

// The type a 2-D weight should have on the device: its own if the device takes it, else the closest
// one it takes — Q8_0 for a quantised weight (every 4-6 bit grid re-quantises onto it with an error
// well under one of its own steps), F16 then F32 for a float one. Its own type when nothing fits, so
// the op stays on the CPU as it would have.
ggml_type device_type_for(ggml_backend_dev_t dev, const ggml_tensor * w) {
    if (ggml_n_dims(w) != 2 || device_matmul_takes(dev, w->type, w)) return w->type;
    if (ggml_is_quantized(w->type)) {
        if (device_matmul_takes(dev, GGML_TYPE_Q8_0, w)) return GGML_TYPE_Q8_0;
    } else {
        if (device_matmul_takes(dev, GGML_TYPE_F16, w)) return GGML_TYPE_F16;
        if (device_matmul_takes(dev, GGML_TYPE_F32, w)) return GGML_TYPE_F32;
    }
    return w->type;
}

// `w`'s bytes re-encoded as `type`, a row at a time through f32.
std::vector<uint8_t> convert_weight(const ggml_tensor * w, ggml_type type) {
    const int64_t n_per_row = w->ne[0];
    const int64_t nrows = ggml_nrows(w);
    std::vector<uint8_t> out(ggml_row_size(type, n_per_row) * (size_t) nrows);
    std::vector<float> row((size_t) n_per_row);
    const ggml_type_traits * from = ggml_get_type_traits(w->type);
    for (int64_t r = 0; r < nrows; ++r) {
        const char * src = (const char *) w->data + (size_t) r * w->nb[1];
        if (w->type == GGML_TYPE_F32)
            std::memcpy(row.data(), src, (size_t) n_per_row * sizeof(float));
        else
            from->to_float(src, row.data(), n_per_row);
        ggml_quantize_chunk(type, row.data(), out.data() + (size_t) r * ggml_row_size(type, n_per_row), 0, 1, n_per_row,
                            nullptr);
    }
    return out;
}
} // namespace

DeviceExpertArena::~DeviceExpertArena() {
    place(false);
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        queue_.clear();
    }
    cv_work_.notify_all();
    for (std::thread & t : threads_)
        if (t.joinable()) t.join();
    for (void * s : staging_)
        if (s) pio::aligned_free(s);
    free_slots();
    for (auto & slots : dense_slot_)
        for (ggml_backend_buffer_t b : slots)
            if (b) ggml_backend_buffer_free(b);
    if (dense_ctx_) ggml_free(dense_ctx_);
}

bool DeviceExpertArena::init(ggml_backend_dev_t dev,
                             const std::vector<std::string> & shard_paths,
                             const std::vector<LayerExperts> & layers,
                             int threads,
                             bool direct,
                             int n_slots,
                             bool on_demand,
                             std::string & err) {
    if (!dev || threads < 1) {
        err = "no device or no loader thread";
        return false;
    }
    if (n_slots < 1 || n_slots > max_slots) {
        err = "the arena has one or two expert slots";
        return false;
    }
    n_slots_ = n_slots;

    // The bound layers, in graph order, and the distinct (type, shape) variants of each projection.
    std::vector<const ggml_tensor *> variants[ExpertLayout::max_slots];
    k_of_layer_.assign(layers.size(), -1);
    uint64_t max_nb2 = 0;
    for (size_t il = 0; il < layers.size(); ++il) {
        const LayerExperts & L = layers[il];
        if (!L.bound) continue;
        Layer a;
        a.il = (int) il;
        int np = 0;
        for (int p = 0; p < ExpertLayout::max_slots; ++p) {
            ggml_tensor * t = L.proj[p].tensor;
            if (!t) break; // the layout fills its slots from the front
            if (n_expert_ == 0) n_expert_ = (int) t->ne[2];
            if (t->ne[2] != n_expert_) {
                err = std::string("expert tensor ") + t->name + " has a different expert count";
                return false;
            }
            int v = 0;
            while (v < (int) variants[p].size() &&
                   (variants[p][(size_t) v]->type != t->type || !ggml_are_same_shape(variants[p][(size_t) v], t)))
                ++v;
            if (v == (int) variants[p].size()) variants[p].push_back(t);
            a.variant[p] = v;
            a.t[p] = t;
            a.proj[p].file_off = L.proj[p].file_off;
            a.proj[p].nb2 = (uint64_t) t->nb[2];
            a.proj[p].file_idx = L.proj[p].file_idx;
            if (a.proj[p].file_idx < 0 || a.proj[p].file_idx >= (int) shard_paths.size()) {
                err = std::string("expert tensor ") + t->name + " names a shard that does not exist";
                return false;
            }
            max_nb2 = std::max(max_nb2, a.proj[p].nb2);
            ++np;
        }
        if (n_proj_ == 0) n_proj_ = np;
        if (np != n_proj_ || np == 0) {
            err = "MoE layers disagree on how many expert tensors they have";
            return false;
        }
        k_of_layer_[il] = (int) order_.size();
        order_.push_back(a);
    }
    if (order_.empty()) {
        err = "no MoE layer to stream";
        return false;
    }

    // Each projection gets a region of the slot sized for its largest variant; every variant's slot
    // tensor sits at the start of that region. On the device an expert need not occupy the bytes it
    // does in the file: a backend that re-lays weights out (Hexagon tiles them, and holds a K-quant in
    // a wider tile format) puts expert e at e * (allocated size / n_expert). That is the stride used
    // below, so it must divide evenly.
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t region_off[ExpertLayout::max_slots] = {};
    size_t slot_size = 0;
    size_t n_tensors = 0;
    for (int p = 0; p < n_proj_; ++p) {
        size_t biggest = 0;
        for (const ggml_tensor * v : variants[p]) {
            const size_t dev = ggml_backend_buft_get_alloc_size(buft, v);
            if (dev % (size_t) n_expert_ != 0 || dev / (size_t) n_expert_ < (size_t) v->nb[2]) {
                err = std::string("the device lays out ") + v->name +
                      " in a way the arena cannot address one expert at a time";
                return false;
            }
            // A single expert is placed at e * (dev / n_expert) and must fit inside that share. A
            // backend that pads every tensor by a fixed slack (OpenCL reserves carve room on each
            // quantized tensor) makes one expert's allocation larger than its share, and the last
            // expert would reach past the slot; ggml aborts on that rather than returning, so it is
            // refused here, where the load can still fall back to the host.
            {
                ggml_init_params sp{};
                sp.mem_size = ggml_tensor_overhead() * 2;
                sp.no_alloc = true;
                ggml_context * c = ggml_init(sp);
                ggml_tensor * one = ggml_new_tensor_3d(c, v->type, v->ne[0], v->ne[1], 1);
                const size_t one_size = ggml_backend_buft_get_alloc_size(buft, one);
                ggml_free(c);
                if (one_size > dev / (size_t) n_expert_) {
                    err = std::string("the device pads a single expert of ") + v->name +
                          " past its share of the whole tensor (" + std::to_string(one_size) + " against " +
                          std::to_string(dev / (size_t) n_expert_) +
                          " bytes), so the arena cannot address one expert at a time";
                    return false;
                }
            }
            biggest = std::max(biggest, dev);
            n_tensors += 1 + (size_t) n_expert_;
        }
        region_off[p] = slot_size;
        slot_size += GGML_PAD(biggest, align);
    }
    slot_bytes_ = (size_t) n_slots_ * slot_size;

    for (int p = 0; p < ExpertLayout::max_slots; ++p) {
        variants_[p] = variants[p];
        region_off_[p] = region_off[p];
    }
    slot_size_ = slot_size;
    n_slot_tensors_ = n_tensors;
    buft_ = buft;
    align_ = align;
    dev_ = dev;
    on_demand_ = on_demand;
    // Held for the session, or built for each prefill and given back after it (see acquire()).
    if (!on_demand_ && !build_slots(err)) return false;

    // Staging is filled by every shard's reader, so it is aligned for the coarsest of them.
    size_t stage_align = pio::vm_page();
    for (const std::string & sp : shard_paths) {
        readers_.push_back(std::unique_ptr<FileReader>(new FileReader()));
        if (!readers_.back()->open(sp, threads, direct, (size_t) max_nb2)) {
            err = "cannot open " + sp;
            return false;
        }
        stage_align = std::max(stage_align, readers_.back()->buffer_align());
    }
    staging_.assign((size_t) threads, nullptr);
    for (int i = 0; i < threads; ++i) {
        staging_[(size_t) i] = pio::alloc_aligned(stage_align, (size_t) max_nb2);
        if (!staging_[(size_t) i]) {
            err = "cannot allocate loader staging";
            return false;
        }
    }
    remaining_.assign(order_.size(), 0);
    sched_.assign(order_.size(), std::vector<uint8_t>((size_t) n_expert_, 0));
    prev_.assign(order_.size(), std::vector<uint8_t>((size_t) n_expert_, 0));
    have_prev_.assign(order_.size(), false);
    for (int i = 0; i < threads; ++i)
        threads_.emplace_back(&DeviceExpertArena::worker, this, i);
    return true;
}

bool DeviceExpertArena::init_dense(ggml_backend_dev_t dev,
                                   const std::vector<std::vector<ggml_tensor *>> & per_layer,
                                   const std::unordered_set<const ggml_tensor *> & non_matrix,
                                   std::string & err) {
    size_t n_tensors = 0;
    for (const auto & v : per_layer)
        n_tensors += v.size();
    if (n_tensors == 0) return true;
    for (const auto & v : per_layer)
        for (ggml_tensor * t : v)
            if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer) || !t->data || t->view_src ||
                !ggml_is_contiguous(t)) {
                err = std::string("layer weight ") + t->name + " is not a host-resident contiguous leaf";
                return false;
            }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const size_t align = ggml_backend_buft_get_alignment(buft);
    // Each slot must hold the largest layer of its parity: layers differ (a hybrid model alternates
    // attention and recurrent blocks), and each layer is laid out from the start of its slot.
    // Sized for each weight as the device will hold it, which is not always the file's type: a matrix
    // may be converted for the device's matmul, a non-matrix weight keeps its own type.
    // [data][parity]; data = 1 for the non-matrix weights, which are not streamed: they stay on the
    // device for every layer, in one buffer ([1][0]), written once here (header, init_dense).
    size_t need[2][2] = {{0, 0}, {0, 0}};
    std::vector<std::vector<ggml_type>> dtype(per_layer.size());
    {
        ggml_init_params sp{};
        sp.mem_size = ggml_tensor_overhead() * 2;
        sp.no_alloc = true;
        for (size_t il = 0; il < per_layer.size(); ++il) {
            size_t sz[2] = {0, 0};
            for (ggml_tensor * t : per_layer[il]) {
                const bool data = non_matrix.count(t) != 0;
                const ggml_type dt = data ? t->type : device_type_for(dev, t);
                dtype[il].push_back(dt);
                ggml_context * c = ggml_init(sp);
                ggml_tensor * shaped = ggml_new_tensor(c, dt, GGML_MAX_DIMS, t->ne);
                sz[data] += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, shaped), align);
                ggml_free(c);
            }
            need[0][il % 2] = std::max(need[0][il % 2], sz[0]);
            need[1][0] += sz[1];
        }
    }
    for (int d = 0; d < 2; ++d) {
        ggml_backend_buffer_t * bufs = dense_slot_[d];
        for (int s = 0; s < 2; ++s) {
            if (need[d][s] == 0) continue;
            bufs[s] = ggml_backend_buft_alloc_buffer(buft, need[d][s] + align);
            if (!bufs[s]) {
                err = std::string("cannot allocate a dense layer slot on ") + ggml_backend_dev_name(dev);
                return false;
            }
            // Before any tensor is set: a backend may choose the layout (and how the device maps the
            // buffer) from the usage.
            ggml_backend_buffer_set_usage(bufs[s],
                                          d ? GGML_BACKEND_BUFFER_USAGE_ANY : GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            dense_slot_bytes_ += need[d][s];
            if (d) dense_resident_bytes_ += need[d][s];
        }
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (n_tensors + 8);
    ip.no_alloc = true;
    dense_ctx_ = ggml_init(ip);
    if (!dense_ctx_) {
        err = "ggml_init failed";
        return false;
    }
    dense_.assign(per_layer.size(), DenseLayer{});
    // The non-matrix weights of every layer, one after the other in their one buffer.
    ggml_tallocr resident = {};
    if (dense_slot_[1][0]) resident = ggml_tallocr_new(dense_slot_[1][0]);
    for (size_t il = 0; il < per_layer.size(); ++il) {
        if (per_layer[il].empty()) continue;
        // A fresh allocator per layer for the matrices: every layer of a parity starts at the same
        // offset of its slot, on purpose. (Gate G17, nemotron_h_moe, once aborted under ASan+UBSan
        // with a recurrent layer in the wrong parity's slot; the cause was never found, so treat a
        // return of that failure as an open fault in the arena, not a regression of this layout.)
        ggml_tallocr matrices = {};
        if (dense_slot_[0][il % 2]) matrices = ggml_tallocr_new(dense_slot_[0][il % 2]);
        DenseLayer & D = dense_[il];
        for (size_t j = 0; j < per_layer[il].size(); ++j) {
            ggml_tensor * t = per_layer[il][j];
            const ggml_type dt = dtype[il][j];
            ggml_tensor * w = ggml_new_tensor(dense_ctx_, dt, GGML_MAX_DIMS, t->ne);
            ggml_set_name(w, t->name);
            const bool data = non_matrix.count(t) != 0;
            if (ggml_tallocr_alloc(data ? &resident : &matrices, w) != GGML_STATUS_SUCCESS) {
                err = std::string("cannot place ") + t->name + " in its dense slot";
                return false;
            }
            if (data) {
                // Written here, on the loading thread, before any graph runs. A backend may make a
                // write to plain device memory wait for its queue (Hexagon flushes it), which is not
                // safe from the loaders while the graph's thread drives that queue.
                ggml_backend_tensor_set(w, t->data, 0, ggml_nbytes(w));
            } else {
                D.streamed.push_back((int) D.t.size());
            }
            D.t.push_back(t);
            D.twin.push_back(w);
            if (dt != t->type) {
                D.converted.push_back(convert_weight(t, dt));
                D.src.push_back(D.converted.back().data());
                ++dense_converted_;
                dense_converted_bytes_ += D.converted.back().size();
            } else {
                D.converted.emplace_back();
                D.src.push_back(t->data);
            }
        }
        D.host_type.assign(D.t.size(), GGML_TYPE_F32);
        D.host_nb.assign(D.t.size(), {});
        D.host_buffer.assign(D.t.size(), nullptr);
        D.host_data.assign(D.t.size(), nullptr);
        D.host_extra.assign(D.t.size(), nullptr);
    }
    std::lock_guard<std::mutex> lk(mu_);
    dense_remaining_.assign(dense_.size(), 0);
    dense_scheduled_.assign(dense_.size(), false);
    dense_waited_.assign(dense_.size(), false);
    return true;
}

// The slots themselves: device buffers, a tensor per (slot, projection, variant) and one per expert
// inside it. Everything here is rebuilt from scratch each time, the tensors included: a backend
// that lays weights out its own way keeps state per tensor, and fresh tensors in a fresh buffer are
// exactly what init gave it.
bool DeviceExpertArena::build_slots(std::string & err) {
    auto fail_slots = [&] {
        free_slots();
        return false;
    };
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (2 * n_slot_tensors_ + 8);
    ip.no_alloc = true;
    ctx_ = ggml_init(ip);
    if (!ctx_) {
        err = "ggml_init failed";
        return fail_slots();
    }
    size_t zmax = 0;
    for (int s = 0; s < n_slots_; ++s) {
        slot_buf_[s] = ggml_backend_buft_alloc_buffer(buft_, slot_size_ + align_);
        if (!slot_buf_[s]) {
            err = std::string("cannot allocate ") + (n_slots_ == 1 ? "an expert slot" : "two expert slots") + " on " +
                  ggml_backend_dev_name(dev_);
            return fail_slots();
        }
        ggml_backend_buffer_set_usage(slot_buf_[s], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        char * base = (char *) ggml_backend_buffer_get_base(slot_buf_[s]);
        base += (align_ - ((uintptr_t) base % align_)) % align_;
        for (int p = 0; p < n_proj_; ++p) {
            for (size_t v = 0; v < variants_[p].size(); ++v) {
                Twin w;
                w.t = ggml_dup_tensor(ctx_, variants_[p][v]);
                ggml_format_name(w.t, "arena.s%d.p%d.v%d", s, p, (int) v);
                if (ggml_backend_tensor_alloc(slot_buf_[s], w.t, base + region_off_[p]) != GGML_STATUS_SUCCESS) {
                    err = std::string("cannot place an expert slot tensor for ") + variants_[p][v]->name;
                    return fail_slots();
                }
                // One single-expert tensor per expert, placed where the device keeps that expert. Not
                // views: a view may not reach past the file-sized extent of its source, and on a
                // backend whose per-expert stride is wider than the file's the last experts would.
                const size_t stride = ggml_backend_buft_get_alloc_size(buft_, w.t) / (size_t) n_expert_;
                for (int e = 0; e < n_expert_; ++e) {
                    ggml_tensor * x = ggml_new_tensor_3d(ctx_, w.t->type, w.t->ne[0], w.t->ne[1], 1);
                    if (ggml_backend_tensor_alloc(slot_buf_[s], x, base + region_off_[p] + (size_t) e * stride) !=
                        GGML_STATUS_SUCCESS) {
                        err = "cannot place an expert inside its slot";
                        return fail_slots();
                    }
                    w.views.push_back(x);
                }
                zmax = std::max(zmax, ggml_nbytes(w.t));
                twins_[s][p].push_back(std::move(w));
            }
        }
    }

    // One whole-tensor write per slot tensor, before any view write. A backend records how a tensor
    // is laid out when it is written (Hexagon flags it repacked), and the op reads the slot tensor,
    // not the views the loaders write through — so each slot tensor must have been written once.
    {
        std::vector<uint8_t> zeros(zmax, 0);
        for (int s = 0; s < n_slots_; ++s)
            for (int p = 0; p < n_proj_; ++p)
                for (Twin & w : twins_[s][p])
                    ggml_backend_tensor_set(w.t, zeros.data(), 0, ggml_nbytes(w.t));
    }

    return true;
}

void DeviceExpertArena::free_slots() {
    for (int s = 0; s < max_slots; ++s) {
        if (slot_buf_[s]) ggml_backend_buffer_free(slot_buf_[s]);
        slot_buf_[s] = nullptr;
        for (int p = 0; p < ExpertLayout::max_slots; ++p)
            twins_[s][p].clear();
    }
    if (ctx_) ggml_free(ctx_);
    ctx_ = nullptr;
}

bool DeviceExpertArena::acquire(std::string & err) {
    if (slots_held()) return true;
    return build_slots(err);
}

void DeviceExpertArena::release() {
    if (!on_demand_ || on_device_ || !slots_held()) return;
    std::unique_lock<std::mutex> lk(mu_);
    cv_done_.wait(lk, [&] { return in_flight_ == 0; }); // no loader may still be writing into a slot
    lk.unlock();
    free_slots();
}

void DeviceExpertArena::place(bool on_device) {
    if (on_device == on_device_) return;
    if (on_device && !slots_held()) return; // nothing to bind to: the caller did not acquire()
    for (size_t k = 0; k < order_.size(); ++k) {
        Layer & L = order_[k];
        for (int p = 0; p < n_proj_; ++p) {
            ggml_tensor * t = L.t[p];
            if (on_device) {
                L.host_buffer[p] = t->buffer;
                L.host_data[p] = t->data;
                L.host_extra[p] = t->extra;
                const ggml_tensor * s = twins_[k % (size_t) n_slots_][p][(size_t) L.variant[p]].t;
                t->buffer = s->buffer;
                t->data = s->data;
                t->extra = s->extra;
            } else {
                t->buffer = L.host_buffer[p];
                t->data = L.host_data[p];
                t->extra = L.host_extra[p];
            }
        }
    }
    for (DenseLayer & D : dense_) {
        for (size_t i = 0; i < D.t.size(); ++i) {
            ggml_tensor * t = D.t[i];
            if (on_device) {
                D.host_buffer[i] = t->buffer;
                D.host_data[i] = t->data;
                D.host_extra[i] = t->extra;
                D.host_type[i] = t->type;
                for (int d = 0; d < GGML_MAX_DIMS; ++d)
                    D.host_nb[i][(size_t) d] = t->nb[d];
                t->buffer = D.twin[i]->buffer;
                t->data = D.twin[i]->data;
                t->extra = D.twin[i]->extra;
                t->type = D.twin[i]->type;
                for (int d = 0; d < GGML_MAX_DIMS; ++d)
                    t->nb[d] = D.twin[i]->nb[d];
            } else {
                t->buffer = D.host_buffer[i];
                t->data = D.host_data[i];
                t->extra = D.host_extra[i];
                t->type = D.host_type[i];
                for (int d = 0; d < GGML_MAX_DIMS; ++d)
                    t->nb[d] = D.host_nb[i][(size_t) d];
            }
        }
    }
    on_device_ = on_device;
}

void DeviceExpertArena::rebuild_order() {
    const int n = (int) std::max(k_of_layer_.size(), dense_.size());
    run_block_.clear();
    pos_of_layer_.assign((size_t) std::max(n, 0), -1);
    auto add = [&](int layer, int block) {
        if (layer < 0 || block < 0 || block >= n) return;
        if (layer >= (int) pos_of_layer_.size()) pos_of_layer_.resize((size_t) layer + 1, -1);
        if (pos_of_layer_[(size_t) layer] >= 0) return; // a graph runs a layer of its own once
        pos_of_layer_[(size_t) layer] = (int) run_block_.size();
        run_block_.push_back(block);
    };
    if (given_layers_.empty()) {
        for (int il = 0; il < n; ++il)
            add(il, il);
        return;
    }
    for (size_t i = 0; i < given_layers_.size() && i < given_blocks_.size(); ++i)
        add(given_layers_[i], given_blocks_[i]);
}

void DeviceExpertArena::set_graph_order(const std::vector<int> & layers, const std::vector<int> & blocks) {
    std::lock_guard<std::mutex> lk(mu_);
    given_layers_ = layers;
    given_blocks_ = blocks;
    rebuild_order();
}

int DeviceExpertArena::block_at(int il, int ahead) const {
    if (il < 0 || il >= (int) pos_of_layer_.size() || pos_of_layer_[(size_t) il] < 0) return -1;
    const int p = pos_of_layer_[(size_t) il] + ahead;
    return p >= 0 && p < (int) run_block_.size() ? run_block_[(size_t) p] : -1;
}

int DeviceExpertArena::next_k(int il) const {
    for (int ahead = 1;; ++ahead) {
        const int b = block_at(il, ahead);
        if (b < 0) return -1;
        if (k_of(b) >= 0) return k_of(b);
    }
}

void DeviceExpertArena::schedule(int k, const std::vector<uint8_t> * want, bool front) {
    if (k < 0 || k >= (int) order_.size()) return;
    // The slot may hold another block's experts by now: that block starts over if the graph comes
    // back to it, and this one starts from nothing.
    int & owner = expert_owner_[k % n_slots_];
    if (owner != k) {
        if (owner >= 0) std::fill(sched_[(size_t) owner].begin(), sched_[(size_t) owner].end(), (uint8_t) 0);
        std::fill(sched_[(size_t) k].begin(), sched_[(size_t) k].end(), (uint8_t) 0);
        owner = k;
    }
    std::vector<uint8_t> & s = sched_[(size_t) k];
    std::vector<int> es;
    for (int e = 0; e < n_expert_; ++e)
        if (!s[(size_t) e] && (!want || (*want)[(size_t) e])) {
            s[(size_t) e] = 1;
            es.push_back(e);
        }
    if (es.empty()) return;
    // Projection-major, the order the layer's matmuls consume them in.
    std::vector<Task> add;
    add.reserve((size_t) n_proj_ * es.size());
    for (int p = 0; p < n_proj_; ++p)
        for (int e : es)
            add.push_back(Task{false, k, p, e});
    remaining_[(size_t) k] += (int) add.size();
    in_flight_ += (int) add.size();
    if (front)
        queue_.insert(queue_.begin(), add.begin(), add.end());
    else
        queue_.insert(queue_.end(), add.begin(), add.end());
    cv_work_.notify_all();
}

void DeviceExpertArena::schedule_dense(int il) {
    if (il < 0 || il >= (int) dense_.size()) return;
    // As for the experts: a slot handed to another block forgets the one it held.
    int & owner = dense_owner_[il % 2];
    if (owner != il) {
        if (owner >= 0) dense_scheduled_[(size_t) owner] = dense_waited_[(size_t) owner] = false;
        owner = il;
    }
    if (dense_scheduled_[(size_t) il]) return;
    dense_scheduled_[(size_t) il] = true;
    const std::vector<int> & streamed = dense_[(size_t) il].streamed;
    const int n = (int) streamed.size();
    dense_remaining_[(size_t) il] = n;
    in_flight_ += n;
    for (int i : streamed) {
        Task t;
        t.dense = true;
        t.k = il;
        t.p = i;
        queue_.push_back(t);
    }
    if (n) cv_work_.notify_all();
}

void DeviceExpertArena::begin_graph() {
    std::unique_lock<std::mutex> lk(mu_);
    // A failure fails the graph it happened in, not the session: end_graph() drained the last one, so
    // nothing still running can set it again, and a transient read error must not turn every later
    // device prefill into a decode failure.
    failed_ = false;
    for (std::vector<uint8_t> & s : sched_)
        std::fill(s.begin(), s.end(), (uint8_t) 0);
    std::fill(dense_scheduled_.begin(), dense_scheduled_.end(), false);
    std::fill(dense_waited_.begin(), dense_waited_.end(), false);
    std::fill(std::begin(expert_owner_), std::end(expert_owner_), -1);
    std::fill(std::begin(dense_owner_), std::end(dense_owner_), -1);
    if (run_block_.empty()) rebuild_order();
    // Dense first: the first layer needs them before anything, and there is no node before it to
    // wait at, so this waits for them here. The second rides along where it has a slot of its own.
    const int d0 = run_block_.empty() ? -1 : run_block_[0];
    const int d1 = run_block_.size() > 1 ? run_block_[1] : -1;
    schedule_dense(d0);
    if (d1 >= 0 && d1 % 2 != d0 % 2) schedule_dense(d1);
    // As many expert layers ahead as there are slots to put them in, in the order the graph runs
    // them, stopping at the first that would land in a slot already given.
    int given = 0;
    for (const int b : run_block_) {
        const int k = k_of(b);
        if (k < 0) continue;
        if (given == n_slots_ || expert_owner_[k % n_slots_] >= 0) break;
        schedule(k, routed_ && have_prev_[(size_t) k] ? &prev_[(size_t) k] : nullptr);
        ++given;
    }
    if (!dense_.empty() && d0 >= 0 && d0 < (int) dense_.size()) {
        cv_done_.wait(lk, [&] { return dense_remaining_[(size_t) d0] == 0; });
        dense_waited_[(size_t) d0] = true;
    }
}

void DeviceExpertArena::dense_barrier(int il) {
    if (dense_.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(mu_);
    const int next = block_at(il, 1), after = block_at(il, 2);
    if (next < 0 || next >= (int) dense_.size()) return;
    // This layer is done with its slot, which is the one the layer after the next goes into. Where
    // the graph's order puts that one in the next layer's own slot instead, it waits its turn: it is
    // queued at the next layer's end, and read while nothing else computes.
    schedule_dense(next);
    if (after >= 0 && after % 2 != next % 2) schedule_dense(after);
    cv_done_.wait(lk, [&] { return dense_remaining_[(size_t) next] == 0; });
    dense_waited_[(size_t) next] = true;
    lk.unlock();
    stall_ns_ +=
        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

void DeviceExpertArena::barrier(int il, const ggml_tensor * topk) {
    // The dense slot of this layer must have been waited for before the layer began: if the graph
    // never passed the node that paces it, its attention already ran on whatever the slot held.
    const int block = block_at(il, 0);
    if (block >= 0 && block < (int) dense_.size() && !dense_[(size_t) block].t.empty()) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!dense_waited_[(size_t) block]) failed_ = true;
    }
    // A routing node of a layer the order does not name: nothing says which block it computes
    // from, so nothing was put in a slot for it. The graph fails rather than compute on what is there.
    if (block < 0) {
        failed_ = true;
        return;
    }
    const int k = k_of(block);
    if (k < 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    // Routed mode: the ids this layer's matmul will read, as a set. Read before taking the lock: the
    // loaders are busy with the prediction meanwhile.
    std::vector<uint8_t> used;
    int n_used = 0;
    if (routed_ && topk && topk->type == GGML_TYPE_I32 && topk->ne[1] > 0) {
        const int nu = (int) topk->ne[0], nt = (int) topk->ne[1];
        // A view of the full argsort: rows are nb[1] apart.
        const size_t stride = (size_t) topk->nb[1] / sizeof(int32_t);
        std::vector<int32_t> rows(stride * (size_t) (nt - 1) + (size_t) nu);
        ggml_backend_tensor_get(topk, rows.data(), 0, rows.size() * sizeof(int32_t));
        used.assign((size_t) n_expert_, 0);
        for (int j = 0; j < nt; ++j)
            for (int q = 0; q < nu; ++q) {
                const int32_t e = rows[(size_t) j * stride + (size_t) q];
                if (e >= 0 && e < n_expert_ && !used[(size_t) e]) {
                    used[(size_t) e] = 1;
                    ++n_used;
                }
            }
    }
    std::unique_lock<std::mutex> lk(mu_);
    // The graph has finished everything before this layer's routing, so layer k-1 is done with the
    // slot k+1 shares with it. With ONE slot that is the slot this layer itself goes into: nothing can
    // be loaded ahead, because the only place to put it is still being computed on, so layer k is read
    // here - with its routing already known - and the device waits for it. What that gives up is the
    // overlap of a layer's read with the previous layer's compute; what it buys is a layer of memory.
    // The same holds for the layer the graph runs next, whichever block that is: it may be read
    // ahead only into a slot this layer is not computing from.
    const int k1 = next_k(il);
    const bool ahead = n_slots_ > 1 && k1 >= 0 && k1 % n_slots_ != k % n_slots_;
    if (!used.empty()) {
        int missed = 0;
        for (int e = 0; e < n_expert_; ++e)
            missed += used[(size_t) e] && !sched_[(size_t) k][(size_t) e];
        if (!test_skip_demand_) schedule(k, &used, true);
        routed_used_ += (uint64_t) n_used;
        routed_demand_ += (uint64_t) missed;
        // Layer k's prediction was consumed when k was queued; this routing is the next graph's.
        prev_[(size_t) k] = used;
        have_prev_[(size_t) k] = true;
        const bool whole = (float) n_used > routed_full_frac_ * (float) n_expert_;
        if (ahead) schedule(k1, !whole && have_prev_[(size_t) k1] ? &prev_[(size_t) k1] : nullptr);
    } else {
        schedule(k);
        if (ahead) schedule(k1);
    }
    cv_done_.wait(lk, [&] { return remaining_[(size_t) k] == 0; });
    lk.unlock();
    stall_ns_ +=
        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}

void DeviceExpertArena::end_graph() {
    std::unique_lock<std::mutex> lk(mu_);
    // A graph that stopped early leaves queued reads nobody will wait for: drop them, and wait only
    // for the ones a loader already holds.
    for (const Task & t : queue_) {
        if (t.dense)
            --dense_remaining_[(size_t) t.k];
        else
            --remaining_[(size_t) t.k];
        --in_flight_;
    }
    queue_.clear();
    cv_done_.wait(lk, [&] { return in_flight_ == 0; });
}

void DeviceExpertArena::worker(int lane) {
    for (;;) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_work_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            task = queue_.front();
            queue_.pop_front();
        }
        // Once per layer (its first upload): with one loader that holds the whole layer back, and it
        // stays cheap where the sleep granularity is coarse (Windows rounds up to ~15 ms).
        if (task.dense) {
            if (test_delay_us_ > 0 && task.p == 0)
                std::this_thread::sleep_for(std::chrono::microseconds(test_delay_us_));
            // Host-resident bytes: a copy (and the backend's repack), no flash read.
            const DenseLayer & D = dense_[(size_t) task.k];
            ggml_backend_tensor_set(D.twin[(size_t) task.p], D.src[(size_t) task.p], 0,
                                    ggml_nbytes(D.t[(size_t) task.p]));
            {
                std::lock_guard<std::mutex> lk(mu_);
                --dense_remaining_[(size_t) task.k];
                --in_flight_;
            }
            cv_done_.notify_all();
            continue;
        }
        if (test_delay_us_ > 0 && task.p == 0 && task.e == 0)
            std::this_thread::sleep_for(std::chrono::microseconds(test_delay_us_));
        const Layer & L = order_[(size_t) task.k];
        const Proj & pr = L.proj[task.p];
        const Twin & cw = twins_[task.k % n_slots_][task.p][(size_t) L.variant[task.p]];
        // Held by the expert cache: copy it from RAM. The slice is the file's bytes, laid out as the
        // file lays them out, so the device receives exactly what a flash read would have given it.
        if (cache_) {
            uint64_t cb = 0;
            const void * src = cache_->resident_slice(L.il, task.e, task.p, cb);
            if (src && cb == pr.nb2) {
                ggml_backend_tensor_set(cw.views[(size_t) task.e], src, 0, (size_t) pr.nb2);
                cache_bytes_ += pr.nb2;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    --remaining_[(size_t) task.k];
                    --in_flight_;
                }
                cv_done_.notify_all();
                continue;
            }
        }
        void * stage = staging_[(size_t) lane];
        const long long got =
            readers_[(size_t) pr.file_idx]->read(lane, stage, pr.file_off + (uint64_t) task.e * pr.nb2, pr.nb2);
        if (got < 0) {
            failed_ = true;
        } else {
            const Twin & w = twins_[task.k % n_slots_][task.p][(size_t) L.variant[task.p]];
            ggml_backend_tensor_set(w.views[(size_t) task.e], stage, 0, (size_t) pr.nb2);
            read_bytes_ += pr.nb2;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            --remaining_[(size_t) task.k];
            --in_flight_;
        }
        cv_done_.notify_all();
    }
}

} // namespace bmoe
