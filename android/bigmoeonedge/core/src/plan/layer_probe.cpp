// What a device costs on ONE REAL LAYER of this model (docs/measured-planning.md, Step 3).
//
// The synthetic GEMV of the bandwidth probe is one large matrix in the model's quant format. A
// decode is not that: it is, per layer, a top-k `mul_mat_id` over expert matrices of the model's own
// shapes, and a handful of plain matmuls whose shapes are the attention projections, the router and
// the shared experts. Small shapes are dominated by per-op cost, a threading pattern that scales on
// one big matrix does not scale on eight small ones, and on one 48-core host the GEMV read 77 GiB/s
// while the decode's experts ran at about 12. So this probe builds the layer itself, from the
// file's tensors and bytes, with ggml's public API, and times its two halves apart:
//
//   - the EXPERT BLOCK: every expert tensor of the layer (whatever the file fuses: gate/up/down,
//     gate_up/down, up/down), each as `mul_mat_id` over the experts it holds, k of them per token;
//   - the DENSE MATRICES: every other 2-D weight of the same block, as `mul_mat`.
//
// Each matrix gets an input of its own width. The probe does not chain them into the layer's
// dataflow - that would take per-architecture code (attention, norms, activations), which the
// streaming path must not know - and it does not need to: what is being priced is the matmuls, on
// their real shapes, kernels and bytes. Attention's own arithmetic is not probed (its cost comes
// from the run), which is the limit the design document states.
//
// At two widths. At decode width a token routes to k experts and reads each once, so a pass holds
// enough experts, and enough copies of the dense matrices, to outrun the last-level cache: the
// decode reads every layer's weights from memory, and a probe that loops over one layer's bytes
// would otherwise measure a cache. At prefill width a prompt of k_wide_batch tokens gives each
// expert about k_wide_batch * k / n_expert of them; the probe holds a subset of the experts and
// routes just enough tokens to give each that same batch, so its kernels run at the prompt's own
// arithmetic intensity without holding the whole layer.
//
// Correctness before speed: every backend computes the same bytes and inputs, and a device whose
// results disagree with the host's is refused (upstream's own bound, see layer_probe.h). A node a
// device has no kernel for is left out of its graph and priced at the host's rate for that half,
// which is where the scheduler would run it.
//
// Bounded by construction: the bytes held are sized by the model and capped, samples last a fixed
// duration, and everything is released on every path. Where the file yields no layer, nothing is
// touched and the synthetic probe stands.

#include "layer_probe.h"

#include "../io/platform_io.h"
#include "../moe/tensor_names.h"

#include "bmoe/log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <exception>
#include <string>
#include <vector>

namespace bmoe::detail {

bool agrees(const std::vector<float> & a, const std::vector<float> & b, double bound) {
    if (a.size() != b.size() || a.empty()) return false;
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) a[i] * (double) a[i];
    }
    if (den <= 0) return false;
    return num / den < bound;
}

namespace {

// The bytes a decode-width pass must read to come from memory rather than from a cache: past the
// last-level cache of the machines this runs on, as the synthetic GEMV's matrix is (see its sizing).
constexpr uint64_t k_outrun_bytes = 64ull << 20;
// The most expert bytes, and the most dense bytes, the probe holds of the layer. A layer of a large
// model is gigabytes; the probe needs a working set, not the layer.
constexpr uint64_t k_hold_max_bytes = 256ull << 20;
// A graph's size bounds its nodes AND its leafs together. Every node here is one matmul, whose weight,
// input and (for the experts) selection are leafs of their own, so a graph of n of them holds up to
// 4n entries: sized for nodes alone, a model whose layer needs many copies to outrun the last-level
// cache (a 512-expert block measured on a 24 GB card) overflowed it and aborted the plan.
size_t graph_size(size_t n_nodes) {
    return n_nodes * 4 + 8;
}

// The most copies of a layer's matrices a pass reads, for a model so small that one copy sits in a
// cache. Bounds the graph as well as the memory.
constexpr int k_replicas_max = 16;

using clock_t_ = std::chrono::steady_clock;

struct Matrix {
    std::string name;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne0 = 0, ne1 = 0;
    int64_t n_held = 1;         // experts held (expert matrices); 1 for a dense matrix
    uint64_t expert_bytes = 0;  // one expert's bytes (expert matrices)
    std::vector<uint8_t> bytes; // what the probe holds: n_held experts, or the whole matrix
};

struct Layer {
    int il = -1;
    int64_t n_expert = 0;
    int64_t k_true = 0;      // the model's top-k
    int64_t k = 0;           // experts per token the probe routes: top-k, unless that alone exceeds the cap
    int64_t groups = 0;      // decode: disjoint sets of k experts a pass reads
    int64_t wide_tokens = 0; // prefill: tokens routed through the held experts
    int replicas_e = 1, replicas_d = 1;
    std::vector<Matrix> experts;
    std::vector<Matrix> dense;
    uint64_t held_bytes() const {
        uint64_t b = 0;
        for (const Matrix & m : experts)
            b += m.bytes.size() * (uint64_t) replicas_e;
        for (const Matrix & m : dense)
            b += m.bytes.size() * (uint64_t) replicas_d;
        return b;
    }
};

// Read the layer off the file: the MoE block with the most expert bytes in the file the experts live
// in, its expert matrices and its other 2-D weights, and the bytes the probe will hold of them.
bool read_layer(const ModelProfile & model, Layer & L) {
    if (!model.is_moe || model.expert_file.empty() || model.n_expert < 2 || model.n_expert_used == 0) return false;
    ggml_context * meta = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = &meta;
    gguf_context * g = gguf_init_from_file(model.expert_file.c_str(), gp);
    if (!g || !meta) {
        if (g) gguf_free(g);
        if (meta) ggml_free(meta);
        return false;
    }
    struct Info {
        std::string name;
        ggml_tensor * t;
        uint64_t off;
        int il;
        bool expert;
    };
    std::vector<Info> infos;
    std::vector<int> skip_blocks; // multi-token-prediction blocks: not loaded by a run
    const uint64_t data_off = gguf_get_data_offset(g);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const std::string name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        const int il = tensor_names::block_layer(name.c_str());
        if (!t || il < 0) continue;
        if (tensor_names::is_mtp_tensor(name.c_str())) skip_blocks.push_back(il);
        if (!tensor_names::is_weight(name.c_str())) continue;
        if (ggml_blck_size(t->type) <= 0) continue;
        // A matmul reads a weight as floats: F32 itself, or a type ggml converts (its to_float). A
        // block can also hold an integer table named like a weight (a token-to-expert index map,
        // 2-D and ".weight" in one file): the graph gathers rows from it and no kernel multiplies
        // it, and the CPU backend answers supports_op for such a matmul and then calls no kernel.
        const bool floats = t->type == GGML_TYPE_F32 || ggml_get_type_traits(t->type)->to_float != nullptr;
        if (!floats) continue;
        // A routed expert weight is a block's 3-D weight indexed by the declared expert count
        // (bmoe/expert_layout.h); every other 2-D weight of the block is one of its dense matrices.
        const bool expert = ggml_n_dims(t) == 3 && t->ne[2] == (int64_t) model.n_expert;
        const bool matrix = ggml_n_dims(t) == 2 && t->ne[1] > 1;
        if (!expert && !matrix) continue;
        infos.push_back({name, t, data_off + gguf_get_tensor_offset(g, i), il, expert});
    }

    // The block with the most expert bytes. On a tie, the one with the most dense bytes in this file:
    // a split model's shard can begin or end part way through a block, and a block whose attention
    // sits in the next shard would price the dense half on a router alone. Then the lowest index,
    // so every process picks the same.
    int best = -1;
    uint64_t best_bytes = 0, best_dense = 0;
    for (const Info & a : infos) {
        if (!a.expert || std::find(skip_blocks.begin(), skip_blocks.end(), a.il) != skip_blocks.end()) continue;
        uint64_t sum = 0, dense = 0;
        for (const Info & b : infos)
            if (b.il == a.il) (b.expert ? sum : dense) += ggml_nbytes(b.t);
        const bool better = sum > best_bytes || (sum == best_bytes && dense > best_dense) ||
                            (sum == best_bytes && dense == best_dense && a.il < best);
        if (better) {
            best_bytes = sum;
            best_dense = dense;
            best = a.il;
        }
    }
    bool ok = best >= 0;
    L.il = best;
    L.n_expert = (int64_t) model.n_expert;
    L.k_true = std::min<int64_t>((int64_t) model.n_expert_used, L.n_expert);

    uint64_t per_expert = 0, dense_bytes = 0;
    for (const Info & a : infos) {
        if (!ok || a.il != best) continue;
        Matrix m;
        m.name = a.name;
        m.type = a.t->type;
        m.ne0 = a.t->ne[0];
        m.ne1 = a.t->ne[1];
        if (a.expert) {
            m.expert_bytes = (uint64_t) a.t->nb[2];
            per_expert += m.expert_bytes;
            L.experts.push_back(std::move(m));
        } else {
            // A matrix past the cap alone is left out rather than letting one tensor set the size.
            if (dense_bytes + ggml_nbytes(a.t) > k_hold_max_bytes) continue;
            dense_bytes += ggml_nbytes(a.t);
            L.dense.push_back(std::move(m));
        }
    }
    ok = ok && !L.experts.empty() && per_expert > 0;

    if (ok) {
        // Decode: k experts per token, and enough disjoint sets of them that a pass outruns a cache.
        L.k = std::max<int64_t>(1, std::min<int64_t>(L.k_true, (int64_t) (k_hold_max_bytes / per_expert)));
        const int64_t cap_held = std::max<int64_t>(L.k, (int64_t) (k_hold_max_bytes / per_expert));
        int64_t held = (int64_t) ((k_outrun_bytes + per_expert - 1) / per_expert);
        held = std::clamp<int64_t>(held, L.k, std::min<int64_t>(L.n_expert, cap_held));
        L.groups = std::max<int64_t>(1, held / L.k);
        held = L.groups * L.k;
        const uint64_t expert_set = (uint64_t) held * per_expert;
        L.replicas_e = (int) std::clamp<uint64_t>((k_outrun_bytes + expert_set - 1) / expert_set, 1, k_replicas_max);
        L.replicas_d = dense_bytes ? (int) std::clamp<uint64_t>((k_outrun_bytes + dense_bytes - 1) / dense_bytes, 1,
                                                                k_replicas_max)
                                   : 1;
        // Prefill: the per-expert batch a prompt of k_wide_batch tokens gives, over the held experts.
        const int64_t per_expert_batch = std::max<int64_t>(1, (k_wide_batch * L.k_true + L.n_expert / 2) / L.n_expert);
        L.wide_tokens = std::max<int64_t>(1, (per_expert_batch * held + L.k - 1) / L.k);
        for (Matrix & m : L.experts)
            m.n_held = held;
    }

    // The bytes, from the file. One read per matrix: the held experts are the first ones, contiguous.
    const pio::fd_t fd = ok ? pio::open_read(model.expert_file.c_str(), false) : pio::fd_t{};
    if (ok && !pio::fd_ok(fd)) ok = false;
    for (const Info & a : infos) {
        if (!ok || a.il != best) continue;
        Matrix * m = nullptr;
        for (Matrix & e : L.experts)
            if (e.name == a.name) m = &e;
        for (Matrix & d : L.dense)
            if (d.name == a.name) m = &d;
        if (!m) continue;
        const uint64_t len = a.expert ? (uint64_t) m->n_held * m->expert_bytes : ggml_nbytes(a.t);
        m->bytes.resize(len + pio::tensor_tail_slack());
        if (pio::pread_at(fd, m->bytes.data(), (size_t) len, a.off) != (long long) len) ok = false;
        m->bytes.resize(len);
    }
    if (pio::fd_ok(fd)) pio::close_fd(fd);
    gguf_free(g);
    ggml_free(meta);
    return ok;
}

// Deterministic inputs, different per element and per token, so a kernel that repeats one token
// cannot agree with the reference.
void fill_input(std::vector<float> & x, size_t n, uint32_t salt) {
    x.resize(n);
    for (size_t i = 0; i < n; ++i)
        x[i] = std::cos((float) (i % 509) * 0.013f + (float) salt * 0.71f + (float) (i / 509) * 0.37f) * 0.5f;
}

// One half of the layer at one width, as it ran on one backend.
struct Part {
    double applied = 0.0;                // weight bytes applied per pass, by the nodes this backend ran
    double unsupported = 0.0;            // the same for nodes it has no kernel for
    double seconds = 0.0;                // per pass, 0 when it did not run
    std::vector<std::vector<float>> out; // per node of the first copy; empty where it did not run
    double rate() const { return seconds > 0.0 ? applied / seconds : 0.0; }
};

struct LayerRun {
    bool ok = false;
    Part dec_e, dec_d, wide_e, wide_d;
};

double seconds_since(clock_t_::time_point t0) {
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

// Seconds per pass of `gf`, sized by duration. 0 when the backend refuses to compute it.
//
// Each pass ends with a few bytes of its last result read back. A backend that queues its work and
// whose synchronize does not wait for the far side (a device behind a network link is one) would
// otherwise be timed on how fast it accepts work rather than on how fast it does it - measured at
// fifteen times the host's rate on the host's own kernels - and a read is a wait every backend
// honours. It is also what a decoded token does: the host reads the logits before the next one.
double time_graph(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_tensor * last = ggml_graph_node(gf, -1);
    float probe = 0.0f;
    const auto pass = [&] {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(last, &probe, 0, sizeof(probe));
        return true;
    };
    if (!last || !pass()) return 0.0; // setup, untimed
    const auto t0 = clock_t_::now();
    int reps = 0;
    while (reps < k_reps_max) {
        if (!pass()) return 0.0;
        ++reps;
        if (seconds_since(t0) >= k_sample_seconds) break;
    }
    const double secs = seconds_since(t0);
    return reps > 0 && secs > 0.0 ? secs / reps : 0.0;
}

LayerRun run_layer(ggml_backend_dev_t dev, const Layer & L, int n_threads) {
    LayerRun r;
    ggml_backend_t backend = nullptr;
    try {
        backend = ggml_backend_dev_init(dev, nullptr);
    } catch (const std::exception &) {
        backend = nullptr;
    }
    if (!backend) return r;
    if (n_threads > 0) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto set_threads =
            reg ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")
                : nullptr;
        if (set_threads) set_threads(backend, n_threads);
    }

    const size_t n_ex = L.experts.size(), n_de = L.dense.size();
    const size_t n_nodes =
        (size_t) L.replicas_e * n_ex * (size_t) L.groups + n_ex + (size_t) L.replicas_d * n_de + n_de;
    const size_t n_tensors =
        n_nodes * 2 + (size_t) L.replicas_e * n_ex + (size_t) L.replicas_d * n_de + (size_t) L.groups + 8;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(graph_size(n_nodes), false) * 4;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        ggml_backend_free(backend);
        return r;
    }

    struct Node {
        ggml_tensor * y;
        bool first; // of the first copy: its output is compared
        double applied;
    };
    std::vector<Node> dec_e, dec_d, wide_e, wide_d;
    std::vector<std::pair<ggml_tensor *, const Matrix *>> weights;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> inputs;
    std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> id_sets;

    // The routing: decode group g reads experts [g*k, g*k+k); a prefill token t reads k consecutive
    // experts from t*k on, round the held set, so every held expert gets the same batch.
    std::vector<ggml_tensor *> dec_ids;
    for (int64_t gi = 0; gi < L.groups; ++gi) {
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, L.k, 1);
        std::vector<int32_t> v((size_t) L.k);
        for (int64_t j = 0; j < L.k; ++j)
            v[(size_t) j] = (int32_t) (gi * L.k + j);
        id_sets.emplace_back(ids, std::move(v));
        dec_ids.push_back(ids);
    }
    const int64_t held = L.experts.empty() ? 0 : L.experts.front().n_held;
    ggml_tensor * wide_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, L.k, L.wide_tokens);
    {
        std::vector<int32_t> v((size_t) (L.k * L.wide_tokens));
        for (int64_t t = 0; t < L.wide_tokens; ++t)
            for (int64_t j = 0; j < L.k; ++j)
                v[(size_t) (t * L.k + j)] = (int32_t) ((t * L.k + j) % held);
        id_sets.emplace_back(wide_ids, std::move(v));
    }

    uint32_t salt = 0;
    for (size_t e = 0; e < n_ex; ++e) {
        const Matrix & m = L.experts[e];
        ggml_tensor * xd = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, m.ne0, L.k, 1);
        ggml_tensor * xw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, m.ne0, L.k, L.wide_tokens);
        std::vector<float> vd, vw;
        fill_input(vd, (size_t) (m.ne0 * L.k), ++salt);
        fill_input(vw, (size_t) (m.ne0 * L.k * L.wide_tokens), ++salt);
        inputs.emplace_back(xd, std::move(vd));
        inputs.emplace_back(xw, std::move(vw));
        for (int rep = 0; rep < L.replicas_e; ++rep) {
            ggml_tensor * w = ggml_new_tensor_3d(ctx, m.type, m.ne0, m.ne1, m.n_held);
            weights.emplace_back(w, &m);
            for (int64_t gi = 0; gi < L.groups; ++gi)
                dec_e.push_back({ggml_mul_mat_id(ctx, w, xd, dec_ids[(size_t) gi]), rep == 0,
                                 (double) m.expert_bytes * (double) L.k});
            if (rep == 0)
                wide_e.push_back({ggml_mul_mat_id(ctx, w, xw, wide_ids), true,
                                  (double) m.expert_bytes * (double) (L.k * L.wide_tokens)});
        }
    }
    for (size_t di = 0; di < n_de; ++di) {
        const Matrix & m = L.dense[di];
        ggml_tensor * xd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.ne0, 1);
        ggml_tensor * xw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.ne0, k_wide_batch);
        std::vector<float> vd, vw;
        fill_input(vd, (size_t) m.ne0, ++salt);
        fill_input(vw, (size_t) (m.ne0 * k_wide_batch), ++salt);
        inputs.emplace_back(xd, std::move(vd));
        inputs.emplace_back(xw, std::move(vw));
        for (int rep = 0; rep < L.replicas_d; ++rep) {
            ggml_tensor * w = ggml_new_tensor_2d(ctx, m.type, m.ne0, m.ne1);
            weights.emplace_back(w, &m);
            dec_d.push_back({ggml_mul_mat(ctx, w, xd), rep == 0, (double) m.bytes.size()});
            if (rep == 0) wide_d.push_back({ggml_mul_mat(ctx, w, xw), true, (double) m.bytes.size() * k_wide_batch});
        }
    }

    // A graph of the nodes this backend has a kernel for; the rest are counted, not run.
    const auto build = [&](std::vector<Node> & nodes, Part & part) -> ggml_cgraph * {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size(nodes.size()), false);
        bool any = false;
        part.out.assign(nodes.size(), {});
        for (Node & n : nodes) {
            if (!n.y || !ggml_backend_dev_supports_op(dev, n.y)) {
                part.unsupported += n.applied;
                n.applied = -n.applied; // marks it as not run
                continue;
            }
            ggml_build_forward_expand(gf, n.y);
            part.applied += n.applied;
            any = true;
        }
        return any ? gf : nullptr;
    };
    ggml_cgraph * g_dec_e = build(dec_e, r.dec_e);
    ggml_cgraph * g_dec_d = n_de ? build(dec_d, r.dec_d) : nullptr;
    ggml_cgraph * g_wide_e = build(wide_e, r.wide_e);
    ggml_cgraph * g_wide_d = n_de ? build(wide_d, r.wide_d) : nullptr;

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf) {
        for (auto & w : weights)
            ggml_backend_tensor_set(w.first, w.second->bytes.data(), 0, ggml_nbytes(w.first));
        for (auto & x : inputs)
            ggml_backend_tensor_set(x.first, x.second.data(), 0, ggml_nbytes(x.first));
        for (auto & i : id_sets)
            ggml_backend_tensor_set(i.first, i.second.data(), 0, ggml_nbytes(i.first));

        const auto run = [&](ggml_cgraph * gf, std::vector<Node> & nodes, Part & part) {
            if (!gf) return;
            part.seconds = time_graph(backend, gf);
            if (part.seconds <= 0.0) return;
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (!nodes[i].first || nodes[i].applied < 0) continue;
                part.out[i].resize((size_t) ggml_nelements(nodes[i].y));
                ggml_backend_tensor_get(nodes[i].y, part.out[i].data(), 0, ggml_nbytes(nodes[i].y));
            }
        };
        run(g_dec_e, dec_e, r.dec_e);
        run(g_dec_d, dec_d, r.dec_d);
        run(g_wide_e, wide_e, r.wide_e);
        run(g_wide_d, wide_d, r.wide_d);
        r.ok = r.dec_e.seconds > 0.0 || r.dec_d.seconds > 0.0 || r.wide_e.seconds > 0.0 || r.wide_d.seconds > 0.0;
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(backend);
    return r;
}

// The tokens of a node whose exact result is computed: enough rows to catch a kernel that differs per
// token, few enough that a double-precision product over a real layer stays a fraction of the probe.
constexpr int64_t k_exact_tokens = 4;

// The exact result of the nodes whose outputs are compared (the first copy's, and of the decode groups
// only the first), from the weights dequantized and multiplied in double: what both backends compute
// within their own rounding. A node's vector is a prefix of its output, the first k_exact_tokens
// tokens of it, since outputs are token-major. Empty where a node is not compared or a type has no
// conversion to float, and then that node falls back to the host as the reference.
struct Exact {
    std::vector<std::vector<double>> dec_e, dec_d, wide_e, wide_d;
};

bool dequant_row(const Matrix & m, int64_t expert, int64_t row, std::vector<float> & out) {
    const size_t rs = ggml_row_size(m.type, m.ne0);
    const size_t off = (size_t) expert * (size_t) m.expert_bytes + (size_t) row * rs;
    if (off + rs > m.bytes.size()) return false;
    out.resize((size_t) m.ne0);
    const uint8_t * p = m.bytes.data() + off;
    if (m.type == GGML_TYPE_F32) {
        std::memcpy(out.data(), p, rs);
        return true;
    }
    const ggml_type_traits * tt = ggml_get_type_traits(m.type);
    if (!tt || !tt->to_float) return false;
    tt->to_float(p, out.data(), m.ne0);
    return true;
}

// out[t][j][r] for tokens t < n_tok: expert ids(t, j)'s row r against input slot (t, j).
bool exact_expert(const Matrix & m,
                  const std::vector<float> & x,
                  int64_t k,
                  int64_t n_tok,
                  const std::function<int64_t(int64_t, int64_t)> & id,
                  std::vector<double> & out) {
    out.assign((size_t) (n_tok * k * m.ne1), 0.0);
    std::vector<float> w;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t e = id(t, j);
            const float * xi = x.data() + (size_t) ((t * k + j) * m.ne0);
            for (int64_t r = 0; r < m.ne1; ++r) {
                if (!dequant_row(m, e, r, w)) return false;
                double acc = 0.0;
                for (int64_t c = 0; c < m.ne0; ++c)
                    acc += (double) w[(size_t) c] * (double) xi[c];
                out[(size_t) ((t * k + j) * m.ne1 + r)] = acc;
            }
        }
    return true;
}

bool exact_dense(const Matrix & m, const std::vector<float> & x, int64_t n_tok, std::vector<double> & out) {
    out.assign((size_t) (n_tok * m.ne1), 0.0);
    std::vector<float> w;
    for (int64_t r = 0; r < m.ne1; ++r) {
        if (!dequant_row(m, 0, r, w)) return false;
        for (int64_t t = 0; t < n_tok; ++t) {
            const float * xi = x.data() + (size_t) (t * m.ne0);
            double acc = 0.0;
            for (int64_t c = 0; c < m.ne0; ++c)
                acc += (double) w[(size_t) c] * (double) xi[c];
            out[(size_t) (t * m.ne1 + r)] = acc;
        }
    }
    return true;
}

// The same inputs and routing run_layer builds, in the same order, so node i here is node i there.
Exact exact_layer(const Layer & L) {
    Exact ex;
    const size_t n_ex = L.experts.size(), n_de = L.dense.size();
    ex.dec_e.resize(n_ex * (size_t) L.replicas_e * (size_t) L.groups);
    ex.wide_e.resize(n_ex);
    ex.dec_d.resize(n_de * (size_t) L.replicas_d);
    ex.wide_d.resize(n_de);
    const int64_t held = L.experts.empty() ? 0 : L.experts.front().n_held;
    uint32_t salt = 0;
    for (size_t e = 0; e < n_ex; ++e) {
        const Matrix & m = L.experts[e];
        std::vector<float> vd, vw;
        fill_input(vd, (size_t) (m.ne0 * L.k), ++salt);
        fill_input(vw, (size_t) (m.ne0 * L.k * L.wide_tokens), ++salt);
        // Decode, group 0: experts [0, k).
        if (!exact_expert(
                m, vd, L.k, 1, [](int64_t, int64_t j) { return j; },
                ex.dec_e[e * (size_t) L.replicas_e * (size_t) L.groups]))
            ex.dec_e[e * (size_t) L.replicas_e * (size_t) L.groups].clear();
        const int64_t n_tok = std::min<int64_t>(k_exact_tokens, L.wide_tokens);
        const int64_t k = L.k;
        if (held <= 0 ||
            !exact_expert(
                m, vw, L.k, n_tok, [k, held](int64_t t, int64_t j) { return (t * k + j) % held; }, ex.wide_e[e]))
            ex.wide_e[e].clear();
    }
    for (size_t di = 0; di < n_de; ++di) {
        const Matrix & m = L.dense[di];
        std::vector<float> vd, vw;
        fill_input(vd, (size_t) m.ne0, ++salt);
        fill_input(vw, (size_t) (m.ne0 * k_wide_batch), ++salt);
        if (!exact_dense(m, vd, 1, ex.dec_d[di * (size_t) L.replicas_d])) ex.dec_d[di * (size_t) L.replicas_d].clear();
        if (!exact_dense(m, vw, std::min<int64_t>(k_exact_tokens, k_wide_batch), ex.wide_d[di])) ex.wide_d[di].clear();
    }
    return ex;
}

// How far `run`'s outputs are from the exact result, as llama.cpp's normalized mean squared error
// over every node that has both; a negative value where none has.
double nmse_exact(const Part & run, const std::vector<std::vector<double>> & exact) {
    double num = 0.0, den = 0.0;
    bool any = false;
    for (size_t i = 0; i < run.out.size() && i < exact.size(); ++i) {
        const std::vector<double> & x = exact[i];
        if (x.empty() || run.out[i].size() < x.size()) continue;
        for (size_t j = 0; j < x.size(); ++j) {
            const double d = (double) run.out[i][j] - x[j];
            num += d * d;
            den += x[j] * x[j];
        }
        any = true;
    }
    return any && den > 0.0 ? num / den : -1.0;
}

// Whether `dev` reproduced the host on every node of the first copy both ran; Unknown where none was.
Tri agreement(const Part & dev, const Part & ref, double bound) {
    std::vector<float> a, b;
    for (size_t i = 0; i < dev.out.size() && i < ref.out.size(); ++i) {
        if (dev.out[i].empty() || ref.out[i].empty()) continue;
        a.insert(a.end(), dev.out[i].begin(), dev.out[i].end());
        b.insert(b.end(), ref.out[i].begin(), ref.out[i].end());
    }
    if (a.empty()) return Tri::Unknown;
    return agrees(a, b, bound) ? Tri::Yes : Tri::No;
}

// A device's rate on a half, with the nodes it has no kernel for at the host's rate for that half -
// where the scheduler would run them. 0 where it ran nothing of it.
double blended(const Part & dev, double host_rate) {
    if (dev.rate() <= 0.0) return 0.0;
    if (dev.unsupported <= 0.0) return dev.rate();
    if (host_rate <= 0.0) return 0.0;
    return (dev.applied + dev.unsupported) / (dev.applied / dev.rate() + dev.unsupported / host_rate);
}

constexpr double k_gib = 1024.0 * 1024.0 * 1024.0;

// The layer at prefill width, as one rate: the expert half priced for a prompt of k_wide_batch
// tokens over the whole expert set (the per-expert batch is the prompt's, so the rate carries over),
// the dense half at that width as measured.
double wide_rate(const Layer & L, double expert_rate, double dense_rate) {
    uint64_t per_expert = 0, dense = 0;
    for (const Matrix & m : L.experts)
        per_expert += m.expert_bytes;
    for (const Matrix & m : L.dense)
        dense += m.bytes.size();
    const double ae = (double) per_expert * (double) (k_wide_batch * L.k_true);
    const double ad = (double) dense * (double) k_wide_batch;
    if (expert_rate <= 0.0 || (ad > 0.0 && dense_rate <= 0.0)) return 0.0;
    return (ae + ad) / (ae / expert_rate + (ad > 0.0 ? ad / dense_rate : 0.0));
}

} // namespace

bool probe_real_layer(HardwareProfile & hw, const ModelProfile & model) {
    Layer L;
    if (!read_layer(model, L)) return false;

    ComputeDevice * host = nullptr;
    for (ComputeDevice & d : hw.devices)
        if (d.is_cpu) host = &d;
    ggml_backend_dev_t host_dev = host ? ggml_backend_dev_by_name(host->name.c_str()) : nullptr;
    if (!host_dev) return false;
    const LayerRun ref = run_layer(host_dev, L, (int) hw.best_threads);
    if (!ref.ok || ref.dec_e.rate() <= 0.0 || ref.wide_e.rate() <= 0.0) return false;
    if (ref.dec_e.unsupported > 0.0 || ref.wide_e.unsupported > 0.0) return false; // the reference runs all

    const double host_e = ref.dec_e.rate(), host_d = ref.dec_d.rate();
    const double host_we = ref.wide_e.rate(), host_wd = ref.wide_d.rate();
    hw.host_expert_gibs = host_e / k_gib;
    hw.host_dense_gibs = host_d / k_gib;
    hw.host_wide_matmul_gibs = wide_rate(L, host_we, host_wd) / k_gib;
    hw.wide_batch = (uint32_t) k_wide_batch;
    host->expert_gibs = hw.host_expert_gibs;
    host->dense_gibs = hw.host_dense_gibs;
    host->wide_matmul_gibs = hw.host_wide_matmul_gibs;
    host->wide_identity_ok = Tri::Yes; // the reference

    // The exact result of the compared nodes, computed once: the reference a device is judged by.
    const Exact ex = exact_layer(L);
    const auto judged = [](const Part & dev, const Part & host, const std::vector<std::vector<double>> & exact,
                           const char * name, const char * half, bool wide = false) {
        // One bound at width, whichever the reference; at one token the host's is upstream's own.
        const double bound = wide ? k_max_nmse_device_wide : k_max_nmse_device;
        const double e_host = nmse_exact(host, exact), e_dev = nmse_exact(dev, exact);
        if (e_host < 0.0 || e_host >= k_max_nmse || e_dev < 0.0)
            return agreement(dev, host, wide ? k_max_nmse_device_wide : k_max_nmse);
        log_printf(LogLevel::Info, "real layer: %s on the %s: %.2e from the exact result, the host %.2e (bound %.0e)",
                   name, half, e_dev, e_host, bound);
        return e_dev < bound ? Tri::Yes : Tri::No;
    };
    const uint64_t needed = L.held_bytes();
    for (ComputeDevice & d : hw.devices) {
        if (d.is_cpu || !d.probe_here()) continue;
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(d.name.c_str());
        if (!dev) continue;
        // As the other probes: do not take memory a device may be short of to learn its speed.
        if (d.has_own_memory() && d.memory_free && d.memory_free < needed * 4) continue;
        const LayerRun run = run_layer(dev, L, 0);
        if (!run.ok) continue;
        d.expert_gibs = blended(run.dec_e, host_e) / k_gib;
        d.dense_gibs = blended(run.dec_d, host_d) / k_gib;
        // Correctness first: at decode width a device that computes something else is refused for
        // good, whatever the synthetic probe concluded on its one matrix. Judged against the exact
        // result where the host itself meets the bound on it: two backends round differently (a device
        // quantizes the activations its own way), and comparing one with the other refused a device
        // whose distance from the exact result was the host's own order. Against the host only where
        // the exact result could not be computed, or the host is not within the bound of it either.
        const Tri dec_e = judged(run.dec_e, ref.dec_e, ex.dec_e, d.name.c_str(), "experts");
        const Tri dec_d = judged(run.dec_d, ref.dec_d, ex.dec_d, d.name.c_str(), "dense");
        if (dec_e == Tri::No || dec_d == Tri::No)
            d.identity_ok = Tri::No;
        else if (d.identity_ok == Tri::Unknown && dec_e == Tri::Yes)
            d.identity_ok = Tri::Yes;
        // The layer at width: a half the device ran nothing of is priced where the scheduler runs it.
        const double we = run.wide_e.rate() > 0.0 ? blended(run.wide_e, host_we) : host_we;
        const double wd = run.wide_d.rate() > 0.0 ? blended(run.wide_d, host_wd) : host_wd;
        if (run.wide_e.rate() > 0.0 || run.wide_d.rate() > 0.0) {
            d.wide_matmul_gibs = wide_rate(L, we, wd) / k_gib;
            const Tri ae = judged(run.wide_e, ref.wide_e, ex.wide_e, d.name.c_str(), "experts, wide", /*wide*/ true);
            const Tri ad = judged(run.wide_d, ref.wide_d, ex.wide_d, d.name.c_str(), "dense, wide", /*wide*/ true);
            d.wide_identity_ok = (ae == Tri::No || ad == Tri::No)     ? Tri::No
                                 : (ae == Tri::Yes || ad == Tri::Yes) ? Tri::Yes
                                                                      : Tri::Unknown;
        }
    }
    return true;
}

} // namespace bmoe::detail
