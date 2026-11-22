#pragma once

#include "backend.h"

#include <string>
#include <vector>

namespace bmoe::detail {

// The token each choice is scored by: its first. Checked BEFORE the prompt is prefilled, so a
// request that cannot be answered costs nothing. Refused, with `error` naming the culprits, when a
// choice has no tokens, when a token falls outside the vocabulary, or when two choices share a first
// token — a tie by construction, which no amount of model quality can break.
bool choice_first_tokens(const std::vector<std::string> & choices,
                         const std::vector<std::vector<Token>> & tokenized,
                         int n_vocab,
                         std::vector<Token> & out,
                         std::string & error);

// log p(ids[i]) under the softmax of `logits` (n_vocab wide), and the index of the highest. The
// normaliser is the whole vocabulary, not the choices: see DecideResult::choice_logp.
void score_choices(
    const float * logits, int n_vocab, const std::vector<Token> & ids, std::vector<double> & logp, int & best);

} // namespace bmoe::detail
