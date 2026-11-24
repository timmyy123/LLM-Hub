#pragma once

#include "backend.h"

#include <vector>

namespace bmoe::detail {

// How many leading tokens of `full` (the rendered prefix + suffix) belong to the prefix, given
// `prefix_alone` (the same template rendered over the prefix by itself).
//
// Found on tokens, never on text: where the prefix ends is decided by the template's own markup
// and by the tokenizer's merges, and neither is spelled out by the engine. The two renderings agree
// up to the join and then diverge (the prefix-alone one closes the turn there). One matching token
// is given back, because a merge across the join can make the last shared-looking token a
// different token in context — a state kept up to it would then not be the state `full` needs.
//
// Clamped to [0, full.size() - 1]: at least one token is always left to prefill, since the
// decision is read from the logits of the prompt's last position.
int prefix_split(const std::vector<Token> & full, const std::vector<Token> & prefix_alone);

} // namespace bmoe::detail
