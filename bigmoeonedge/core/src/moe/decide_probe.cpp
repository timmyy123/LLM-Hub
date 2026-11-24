#include "decide_probe.h"

#include "gguf_offsets.h"
#include "tensor_names.h"

#include "../io/file_reader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstring>

namespace bmoe {

// A few KiB read once per load or per choice: buffered, one lane, through the engine's own
// positioned reader, so a 64-bit offset needs no platform branch here.
static bool read_at(const std::string & path, uint64_t off, void * dst, size_t n) {
    FileReader rd;
    if (!rd.open(path, 1, /*direct=*/false, n)) return false;
    return rd.read(0, dst, off, n) == (long long) n;
}

namespace tn = tensor_names;

// The layer of a router-weight node, or -1. Any node of the chain counts: the last one a layer
// offers is what the expert matmul applies (tensor_names.h), and observe() keeps the latest.
static int weight_layer(const char * name) {
    int il = -1;
    return tn::weight_node(name, il) >= 0 ? il : -1;
}

DecideProbe::DecideProbe(
    const std::string & model_path, int n_layer, int n_expert, int n_embd, const std::string & out_path)
    : n_layer_(n_layer), n_expert_(n_expert), n_embd_(n_embd) {
    out_ = std::fopen(out_path.c_str(), "ab");
    if (!out_) {
        error_ = "cannot open " + out_path;
        return;
    }
    if (!load_head(model_path)) return;
    ok_ = true;
}

DecideProbe::~DecideProbe() {
    if (out_) std::fclose(out_);
}

bool DecideProbe::load_head(const std::string & model_path) {
    const GgufOffsets offs = read_gguf_offsets(model_path.c_str());
    if (!offs.ok) {
        error_ = "cannot read gguf offsets";
        return false;
    }
    // Untied models carry output.weight; a tied one multiplies by the embedding table.
    const char * head_name =
        offs.off_by_name.count(tensor_names::k_output_head) ? tensor_names::k_output_head : tensor_names::k_token_table;
    const char * norm_name = tensor_names::k_output_norm;
    bool have_head = false, have_norm = false;
    for (size_t s = 0; s < offs.shard_paths.size() && !(have_head && have_norm); ++s) {
        gguf_init_params gp{};
        gp.no_alloc = true;
        gp.ctx = nullptr;
        gguf_context * g = gguf_init_from_file(offs.shard_paths[s].c_str(), gp);
        if (!g) continue;
        if (s == 0) {
            const int64_t ak = gguf_find_key(g, "general.architecture");
            if (ak >= 0) {
                const std::string key = std::string(gguf_get_val_str(g, ak)) + ".attention.layer_norm_rms_epsilon";
                const int64_t ek = gguf_find_key(g, key.c_str());
                if (ek >= 0) eps_ = gguf_get_val_f32(g, ek);
            }
        }
        const size_t data0 = gguf_get_data_offset(g);
        const int64_t hi = gguf_find_tensor(g, head_name);
        if (hi >= 0 && !have_head) {
            head_path_ = offs.shard_paths[s];
            head_off_ = data0 + gguf_get_tensor_offset(g, hi);
            head_type_ = (int) gguf_get_tensor_type(g, hi);
            head_row_bytes_ = ggml_row_size((ggml_type) head_type_, n_embd_);
            have_head = true;
        }
        const int64_t ni = gguf_find_tensor(g, norm_name);
        if (ni >= 0 && !have_norm) {
            const ggml_type t = gguf_get_tensor_type(g, ni);
            const size_t bytes = ggml_row_size(t, n_embd_);
            std::vector<uint8_t> raw(bytes);
            if (read_at(offs.shard_paths[s], data0 + gguf_get_tensor_offset(g, ni), raw.data(), bytes)) {
                out_norm_.resize(n_embd_);
                if (t == GGML_TYPE_F32)
                    std::memcpy(out_norm_.data(), raw.data(), bytes);
                else
                    ggml_get_type_traits(t)->to_float(raw.data(), out_norm_.data(), n_embd_);
                have_norm = true;
            }
        }
        gguf_free(g);
    }
    if (!have_head || !have_norm) {
        error_ = "lm_head or output_norm not found in the gguf";
        return false;
    }
    return true;
}

bool DecideProbe::row(int32_t tok, std::vector<float> & out) {
    std::vector<uint8_t> raw(head_row_bytes_);
    if (!read_at(head_path_, head_off_ + (uint64_t) tok * head_row_bytes_, raw.data(), raw.size())) return false;
    out.resize(n_embd_);
    const ggml_type t = (ggml_type) head_type_;
    if (t == GGML_TYPE_F32)
        std::memcpy(out.data(), raw.data(), raw.size());
    else
        ggml_get_type_traits(t)->to_float(raw.data(), out.data(), n_embd_);
    return true;
}

void DecideProbe::begin(const std::vector<int32_t> & choice_ids) {
    choice_ids_ = choice_ids;
    choice_rows_.assign(choice_ids.size(), {});
    for (size_t i = 0; i < choice_ids.size(); ++i)
        if (choice_ids[i] >= 0) row(choice_ids[i], choice_rows_[i]);
    ids_.assign(n_layer_, {});
    w_.assign(n_layer_, {});
    nu_.assign(n_layer_, 0);
    cnt_.assign(n_layer_, std::vector<uint32_t>(n_expert_, 0));
    wsum_.assign(n_layer_, std::vector<float>(n_expert_, 0.0f));
    hidden_.assign(n_layer_, {});
    armed_ = true;
}

void DecideProbe::end() {
    for (int il = 0; il < n_layer_; ++il)
        flush_layer(il);
    armed_ = false;
}

bool DecideProbe::wants(const ggml_tensor * t) const {
    const char * n = t->name;
    if (tn::is_moe_node(n)) return tn::node_layer(n, tn::k_topk_node) >= 0 || weight_layer(n) >= 0;
    return tn::node_layer(n, tn::k_layer_out_node) >= 0;
}

void DecideProbe::flush_layer(int il) {
    std::vector<int32_t> & ids = ids_[il];
    if (ids.empty()) return;
    const std::vector<float> & w = w_[il];
    const bool have_w = w.size() == ids.size();
    for (size_t k = 0; k < ids.size(); ++k) {
        const int32_t e = ids[k];
        if (e < 0 || e >= n_expert_) continue;
        cnt_[il][e] += 1;
        if (have_w) wsum_[il][e] += w[k];
    }
    ids.clear();
    w_[il].clear();
}

void DecideProbe::observe(ggml_tensor * t) {
    if (!t->buffer && !(t->view_src && t->view_src->buffer)) return;
    const char * n = t->name;
    int il = tn::node_layer(n, tn::k_topk_node);
    if (il >= 0 && il < n_layer_ && t->type == GGML_TYPE_I32) {
        flush_layer(il); // an earlier graph's routing of this layer
        const int nu = (int) t->ne[0], nt = (int) t->ne[1];
        // A view of the full argsort: rows are nb[1] apart, not nu * 4.
        std::vector<int32_t> rowbuf((size_t) t->nb[1] / sizeof(int32_t) * (size_t) nt);
        ggml_backend_tensor_get(t, rowbuf.data(), 0, (size_t) t->nb[1] * (size_t) (nt - 1) + (size_t) nu * 4);
        std::vector<int32_t> & ids = ids_[il];
        ids.resize((size_t) nu * nt);
        const size_t stride = (size_t) t->nb[1] / sizeof(int32_t);
        for (int j = 0; j < nt; ++j)
            for (int k = 0; k < nu; ++k)
                ids[(size_t) j * nu + k] = rowbuf[(size_t) j * stride + k];
        nu_[il] = nu;
        return;
    }
    il = weight_layer(n);
    if (il >= 0 && il < n_layer_) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return;
        if ((size_t) ggml_nelements(t) != ids_[il].size()) return;
        w_[il].resize(ids_[il].size());
        ggml_backend_tensor_get(t, w_[il].data(), 0, w_[il].size() * sizeof(float));
        return;
    }
    il = tn::node_layer(n, tn::k_layer_out_node);
    if (il >= 0 && il < n_layer_ && t->type == GGML_TYPE_F32 && t->ne[0] == n_embd_) {
        const int nt = (int) t->ne[1];
        hidden_[il].resize(n_embd_);
        ggml_backend_tensor_get(t, hidden_[il].data(), (size_t) t->nb[1] * (size_t) (nt - 1),
                                (size_t) n_embd_ * sizeof(float));
    }
}

void DecideProbe::write(int seq,
                        int n_tokens,
                        double prefill_s,
                        const std::vector<std::string> & choices,
                        const std::vector<double> & logp,
                        int best) {
    if (!out_) return;
    std::string s;
    char buf[64];
    s += "{\"seq\":" + std::to_string(seq) + ",\"n_tokens\":" + std::to_string(n_tokens);
    std::snprintf(buf, sizeof buf, ",\"prefill_s\":%.3f", prefill_s);
    s += buf;
    s += ",\"n_layer\":" + std::to_string(n_layer_) + ",\"n_expert\":" + std::to_string(n_expert_);
    s += ",\"choices\":[";
    for (size_t i = 0; i < choices.size(); ++i) {
        s += i ? ",\"" : "\"";
        for (char c : choices[i])
            if (c == '"' || c == '\\') {
                s += '\\';
                s += c;
            } else if ((unsigned char) c >= 0x20)
                s += c;
        s += '"';
    }
    s += "],\"best\":" + std::to_string(best) + ",\"logp\":[";
    for (size_t i = 0; i < logp.size(); ++i) {
        std::snprintf(buf, sizeof buf, "%s%.5f", i ? "," : "", std::isfinite(logp[i]) ? logp[i] : -1e30);
        s += buf;
    }
    // Lens: per layer, the choice logits read from that layer's last-token state. A layer the graph
    // never offered (none on a normal run) is written as null.
    s += "],\"lens\":[";
    for (int il = 0; il < n_layer_; ++il) {
        s += il ? "," : "";
        const std::vector<float> & h = hidden_[il];
        if (h.empty()) {
            s += "null";
            continue;
        }
        double ss = 0.0;
        for (float v : h)
            ss += (double) v * v;
        const double inv = 1.0 / std::sqrt(ss / n_embd_ + (double) eps_);
        s += "[";
        for (size_t c = 0; c < choice_rows_.size(); ++c) {
            const std::vector<float> & r = choice_rows_[c];
            double acc = 0.0;
            if (r.size() == (size_t) n_embd_)
                for (int d = 0; d < n_embd_; ++d)
                    acc += (double) h[d] * inv * out_norm_[d] * r[d];
            std::snprintf(buf, sizeof buf, "%s%.4f", c ? "," : "", acc);
            s += buf;
        }
        s += "]";
    }
    // Usage: per layer, routings per expert and their summed router weight.
    s += "],\"nu\":[";
    for (int il = 0; il < n_layer_; ++il)
        s += (il ? "," : "") + std::to_string(nu_[il]);
    s += "],\"cnt\":[";
    for (int il = 0; il < n_layer_; ++il) {
        s += il ? ",[" : "[";
        for (int e = 0; e < n_expert_; ++e)
            s += (e ? "," : "") + std::to_string(cnt_[il][e]);
        s += "]";
    }
    s += "],\"wsum\":[";
    for (int il = 0; il < n_layer_; ++il) {
        s += il ? ",[" : "[";
        for (int e = 0; e < n_expert_; ++e) {
            std::snprintf(buf, sizeof buf, "%s%.3f", e ? "," : "", wsum_[il][e]);
            s += buf;
        }
        s += "]";
    }
    s += "]}\n";
    std::fwrite(s.data(), 1, s.size(), out_);
    std::fflush(out_);
}

} // namespace bmoe
