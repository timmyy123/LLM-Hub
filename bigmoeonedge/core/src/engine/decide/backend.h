#pragma once

// The port between decide()'s policy and the model it runs on.
//
// Everything decide() DECIDES — where the prompt splits, whether a kept state is usable, how the
// choices are scored, what is reported — is policy over token ids and a logit row, and lives in
// this directory with no llama.cpp include. Everything it DOES to the model goes through this
// interface. The session implements it over its live context (llama_backend.h); the unit tests
// implement it with a scripted fake, so the policy is tested without a model.

#include "bmoe/decide.h"

#include <cstdint>
#include <string>
#include <vector>

namespace bmoe::detail {

using Token = int32_t; // == llama_token, static_assert'd where the two meet

class IDecideBackend {
public:
    virtual ~IDecideBackend() = default;

    // The prompt tokens of `content` rendered as one user turn, exactly as generate() renders the
    // first turn of a conversation, with reasoning off (see DecideRequest). False, with `error` set,
    // when the template cannot be applied.
    virtual bool render(const std::string & content, std::vector<Token> & out, std::string & error) = 0;
    // A choice as plain text: no template, no special tokens, no BOS.
    virtual std::vector<Token> tokenize_plain(const std::string & text) = 0;

    virtual int n_ctx() const = 0;
    virtual int n_vocab() const = 0;

    // Whether prefilling a prompt in two pieces costs about what the pieces cost together — true on
    // the CPU, where prefill time scales with the tokens fed. A backend on which every graph costs
    // the same at any width answers false, and there splitting at the prefix is a whole extra pass.
    virtual bool prefill_cost_scales_with_tokens() const = 0;

    // Empty the sequence state.
    virtual void clear() = 0;
    // Feed tokens[from, to) at positions from..to-1 of an empty-or-restored sequence. Logits are
    // computed only for the prompt's last position, and only when `to` is the end of `tokens`.
    virtual bool prefill(const std::vector<Token> & tokens, int from, int to) = 0;
    // The logit row after a prefill that reached the end of its prompt; null if there is none.
    virtual const float * last_logits() = 0;

    // The whole sequence state (every layer's memory, recurrent state included), as bytes that
    // load_state() puts back. save_state returns false when the backend cannot serialise it.
    virtual bool save_state(std::vector<uint8_t> & out) = 0;
    virtual bool load_state(const std::vector<uint8_t> & in) = 0;

    // Bracket the prefill work of one decision, for DecideResult::prefill.
    virtual void begin_prefill_measure() = 0;
    virtual void end_prefill_measure(PrefillStats & out) = 0;
};

} // namespace bmoe::detail
