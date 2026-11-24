// Diagnostic probe for decide(): what one decision's prefill routed, and what it would have answered
// had the model stopped at an earlier layer. Experimental, off unless --decide-probe names a file.
//
// Two questions, both about making a decision's prefill cheaper:
//
//   Expert usage: per layer, how many (token, slot) routings each expert took and the router weight
//   they carried. Says how concentrated a task's routing is, i.e. how much of it a resident set sized
//   to RAM could serve.
//
//   Layer exit ("logit lens"): the last token's hidden state at the end of every layer, put through
//   the model's final norm and the lm_head rows of the choice tokens. Causality makes this exact for
//   the question asked: the last token's state after layer L depends on layers 0..L only, so the
//   answer read there is the answer a prefill cut after layer L would give. One full prefill thus
//   prices every cut.
//
// It reads graph nodes through the router hook (the only eval callback) with ggml_backend_tensor_get,
// so it works whether the graph runs on the host or on a prefill device. The lm_head rows and the
// final norm are read from the gguf file itself, never from the live tensors a prefill device may
// have moved or converted. It observes only: nothing the graph computes is changed.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct ggml_tensor;

namespace bmoe {

class DecideProbe {
public:
    // `n_layer` counts the layers the main pass runs (MTP blocks excluded). Opens `out_path` for
    // appending; ok() is false when the file or the model's head cannot be read.
    DecideProbe(const std::string & model_path, int n_layer, int n_expert, int n_embd, const std::string & out_path);
    ~DecideProbe();
    bool ok() const { return ok_; }
    const std::string & error() const { return error_; }

    // Around one decision's prefill. `choice_ids` are the scored tokens, in the request's order.
    void begin(const std::vector<int32_t> & choice_ids);
    void end();
    bool armed() const { return armed_; }

    // From the eval callback. wants(): the ask pass, by name. observe(): the compute pass.
    bool wants(const ggml_tensor * t) const;
    void observe(ggml_tensor * t);

    // One JSONL line for the decision just probed. `logp`/`best` are decide()'s own answer, written
    // beside the lens so the last layer's reading can be checked against it.
    void write(int seq,
               int n_tokens,
               double prefill_s,
               const std::vector<std::string> & choices,
               const std::vector<double> & logp,
               int best);

private:
    bool load_head(const std::string & model_path);
    bool row(int32_t tok, std::vector<float> & out);

    bool ok_ = false;
    std::string error_;
    bool armed_ = false;
    int n_layer_ = 0, n_expert_ = 0, n_embd_ = 0;
    std::FILE * out_ = nullptr;

    // lm_head, as the file stores it
    std::string head_path_;
    uint64_t head_off_ = 0;
    int head_type_ = 0;
    size_t head_row_bytes_ = 0;
    std::vector<float> out_norm_;
    float eps_ = 1e-6f;

    // per decision
    std::vector<int32_t> choice_ids_;
    std::vector<std::vector<float>> choice_rows_;
    std::vector<std::vector<int32_t>> ids_;  // [layer] the last graph's topk, flattened [nt][nu]
    std::vector<std::vector<float>> w_;      // [layer] the last-offered weight node, [nt][nu]
    std::vector<std::vector<uint32_t>> cnt_; // [layer][expert] routings, summed over graphs
    std::vector<std::vector<float>> wsum_;   // [layer][expert] router weight, summed
    std::vector<int> nu_;                    // [layer]
    std::vector<std::vector<float>> hidden_; // [layer] last token's l_out
    void flush_layer(int il);
};

} // namespace bmoe
